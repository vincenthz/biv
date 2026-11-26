//! The GGUF metadata as nested structs: each `.` of a key is a field access,
//! so `qwen35.attention.head_count` is `metadata.qwen35.attention.head_count`.
//!
//! Every leaf is an `Option`, `None` when the file doesn't have the key. A key
//! with no field here (`general.base_model.0.name`), or whose value has another
//! type than its field, is kept by [`Gguf`](super::Gguf) in `extra`.

use super::Value;
use crate::prelude::*;

/// A node of the key tree: a struct of fields, or a leaf value
pub trait Field {
    /// Store `v` under `key`, the rest of the dotted path below this node
    /// (empty for a leaf itself). Gives `v` back when no field has that path
    /// or the value doesn't convert to the field's type.
    fn set(&mut self, key: &str, v: Value) -> Result<(), Value>;
    /// Whether a value is stored under `key`
    fn contains(&self, key: &str) -> bool;
    /// The value stored under `key`, converted back to a GGUF value
    fn get(&self, key: &str) -> Option<Value>;
    /// Call `f` with the full key and the value of every stored leaf below
    /// this node, whose own key is `path`
    fn visit(&self, path: &str, f: &mut dyn FnMut(&str, Value));
}

/// A leaf type: the conversion from and to a GGUF value
pub trait Leaf: Sized {
    fn from_value(v: &Value) -> Option<Self>;
    fn to_value(&self) -> Value;
}

impl Leaf for u64 {
    fn from_value(v: &Value) -> Option<Self> {
        match *v {
            Value::F32(_) | Value::F64(_) | Value::Bool(_) => None,
            Value::I8(_) | Value::I16(_) | Value::I32(_) | Value::I64(_) => {
                v.as_u64().filter(|&x| (x as i64) >= 0)
            }
            _ => v.as_u64(),
        }
    }
    fn to_value(&self) -> Value {
        Value::U64(*self)
    }
}

impl Leaf for u32 {
    fn from_value(v: &Value) -> Option<Self> {
        u64::from_value(v).and_then(|x| x.try_into().ok())
    }
    fn to_value(&self) -> Value {
        Value::U32(*self)
    }
}

impl Leaf for i32 {
    fn from_value(v: &Value) -> Option<Self> {
        match *v {
            Value::I8(x) => Some(x.into()),
            Value::I16(x) => Some(x.into()),
            Value::I32(x) => Some(x),
            Value::I64(x) => x.try_into().ok(),
            Value::U8(x) => Some(x.into()),
            Value::U16(x) => Some(x.into()),
            Value::U32(x) => x.try_into().ok(),
            Value::U64(x) => x.try_into().ok(),
            _ => None,
        }
    }
    fn to_value(&self) -> Value {
        Value::I32(*self)
    }
}

/// floats, and integers as floats
impl Leaf for f32 {
    fn from_value(v: &Value) -> Option<Self> {
        match v {
            Value::Bool(_) => None,
            _ => v.as_f32(),
        }
    }
    fn to_value(&self) -> Value {
        Value::F32(*self)
    }
}

/// booleans, and the integers 0 and 1
impl Leaf for bool {
    fn from_value(v: &Value) -> Option<Self> {
        match *v {
            Value::Bool(b) => Some(b),
            _ => match u64::from_value(v)? {
                0 => Some(false),
                1 => Some(true),
                _ => None,
            },
        }
    }
    fn to_value(&self) -> Value {
        Value::Bool(*self)
    }
}

impl Leaf for String {
    fn from_value(v: &Value) -> Option<Self> {
        v.as_str().map(String::from)
    }
    fn to_value(&self) -> Value {
        Value::Str(self.clone())
    }
}

impl<T: Leaf> Leaf for Vec<T> {
    fn from_value(v: &Value) -> Option<Self> {
        v.as_array()?.iter().map(T::from_value).collect()
    }
    fn to_value(&self) -> Value {
        Value::Array(self.iter().map(T::to_value).collect())
    }
}

impl<T: Leaf> Field for Option<T> {
    fn set(&mut self, key: &str, v: Value) -> Result<(), Value> {
        if !key.is_empty() {
            return Err(v);
        }
        *self = Some(T::from_value(&v).ok_or(v)?);
        Ok(())
    }
    fn contains(&self, key: &str) -> bool {
        key.is_empty() && self.is_some()
    }
    fn get(&self, key: &str) -> Option<Value> {
        self.as_ref().filter(|_| key.is_empty()).map(T::to_value)
    }
    fn visit(&self, path: &str, f: &mut dyn FnMut(&str, Value)) {
        if let Some(v) = self {
            f(path, v.to_value());
        }
    }
}

/// A struct whose fields are the next component of the key, each one a
/// [`Field`]: a leaf `Option` or another such struct.
macro_rules! node {
    ($(
        $(#[$m:meta])*
        pub struct $name:ident {
            $($(#[$fm:meta])* $field:ident: $ty:ty,)*
        }
    )*) => {$(
        $(#[$m])*
        #[derive(Debug, Clone, Default)]
        pub struct $name {
            $($(#[$fm])* pub $field: $ty,)*
        }

        impl Field for $name {
            fn set(&mut self, key: &str, v: Value) -> Result<(), Value> {
                let (head, rest) = key.split_once('.').unwrap_or((key, ""));
                match head {
                    $(stringify!($field) => self.$field.set(rest, v),)*
                    _ => Err(v),
                }
            }

            fn contains(&self, key: &str) -> bool {
                let (head, rest) = key.split_once('.').unwrap_or((key, ""));
                match head {
                    $(stringify!($field) => self.$field.contains(rest),)*
                    _ => false,
                }
            }

            fn get(&self, key: &str) -> Option<Value> {
                let (head, rest) = key.split_once('.').unwrap_or((key, ""));
                match head {
                    $(stringify!($field) => self.$field.get(rest),)*
                    _ => None,
                }
            }

            fn visit(&self, path: &str, f: &mut dyn FnMut(&str, Value)) {
                $(
                    let key = stringify!($field);
                    let key = if path.is_empty() {
                        String::from(key)
                    } else {
                        format!("{path}.{key}")
                    };
                    self.$field.visit(&key, f);
                )*
            }
        }
    )*};
}

node! {
    /// The root of the key tree
    pub struct Metadata {
        general: General,
        /// the hyperparameters of the `qwen35` architecture
        qwen35: Hparams,
        /// the hyperparameters of the `qwen35moe` architecture: `qwen35`'s,
        /// with mixture of experts feed-forwards
        qwen35moe: Hparams,
        /// the hyperparameters of the `flint` architecture
        flint: Hparams,
        /// the hyperparameters of the `inkling2_loop` architecture. Its
        /// per-layer head counts are arrays, which the scalar fields don't
        /// take, and its `loop.*` keys have no field (`loop` is a keyword):
        /// both are in `extra`, read with [`Gguf::get`](super::Gguf::get)
        inkling2_loop: Hparams,
        tokenizer: Tokenizer,
        split: Split,
    }

    pub struct General {
        architecture: Option<String>,
        name: Option<String>,
        basename: Option<String>,
        size_label: Option<String>,
        description: Option<String>,
        license: Option<String>,
        /// the data alignment of the file, 32 by default
        alignment: Option<u32>,
        file_type: Option<u32>,
        quantization_version: Option<u32>,
        quantized_by: Option<String>,
        repo_url: Option<String>,
        tags: Option<Vec<String>>,
        /// sampling settings recommended by the model's publisher
        sampling: Sampling,
    }

    pub struct Sampling {
        temp: Option<f32>,
        top_k: Option<u32>,
        top_p: Option<f32>,
        min_p: Option<f32>,
    }

    /// The hyperparameters of an architecture, under its name: the keys of
    /// every supported one, each having those it uses
    pub struct Hparams {
        /// layers of the trunk, then the next-n (MTP) layers
        block_count: Option<u64>,
        nextn_predict_layers: Option<u64>,
        context_length: Option<u64>,
        embedding_length: Option<u64>,
        feed_forward_length: Option<u64>,
        /// every this many layers, one is full attention, the others delta nets
        full_attention_interval: Option<u64>,
        attention: Attention,
        rope: Rope,
        ssm: Ssm,

        /// the mixture of experts of `qwen35moe`: experts per layer, and
        /// used per token
        expert_count: Option<u64>,
        expert_used_count: Option<u64>,
        /// the width of an expert's feed-forward, and the shared expert's
        expert_feed_forward_length: Option<u64>,
        expert_shared_feed_forward_length: Option<u64>,
        /// the shared expert's width in experts' widths (`flint`)
        expert_shared_count: Option<u64>,
        /// how the router's logits become probabilities: 1 softmax, 2
        /// sigmoid (`flint`; softmax when absent)
        expert_gating_func: Option<u32>,
        /// whether the chosen experts' weights are divided by their sum, and
        /// the factor they are then scaled by
        expert_weights_norm: Option<bool>,
        expert_weights_scale: Option<f32>,
        /// the first layers, which have a dense feed-forward instead of
        /// experts (`flint`)
        leading_dense_block_count: Option<u64>,

        /// rows of the embeddings and output head, and the tokens of the
        /// vocabulary, fewer (`inkling2_loop`)
        vocab_size: Option<u64>,
        vocab_size_unpadded: Option<u64>,
        /// `cap * tanh(logit / cap)`, none when 0 (`inkling2_loop`)
        final_logit_softcapping: Option<f32>,
        attn_res: AttnRes,
        activation: Activation,
        shortconv: Shortconv,
    }

    /// the depth mixes of block outputs (`inkling2_loop`)
    pub struct AttnRes {
        /// layers of a block: the first of each starts a new residual stream
        block_size: Option<u64>,
    }

    /// the SITU activation of the experts (`inkling2_loop`)
    pub struct Activation {
        situ_beta: Option<f32>,
        situ_linear_beta: Option<f32>,
    }

    /// the short causal convolutions of the keys, values and layer outputs
    /// (`inkling2_loop`)
    pub struct Shortconv {
        /// taps, the current token's included
        l_cache: Option<u64>,
    }

    pub struct Attention {
        head_count: Option<u64>,
        head_count_kv: Option<u64>,
        key_length: Option<u64>,
        value_length: Option<u64>,
        layer_norm_rms_epsilon: Option<f32>,
        /// per layer, whether it is a delta net (instead of `full_attention_interval`)
        recurrent_layers: Option<Vec<bool>>,
        /// positions a sliding window layer sees, its token's included
        sliding_window: Option<u64>,
        /// per layer, whether it attends over the sliding window only
        sliding_window_pattern: Option<Vec<bool>>,
        /// per layer, whether it has rotary embeddings (all when absent)
        rope_pattern: Option<Vec<bool>>,

        /// head sizes of the sliding window layers (`inkling2_loop`)
        key_length_swa: Option<u64>,
        value_length_swa: Option<u64>,
        /// the relative position bias: channels per head, and the distances
        /// of the full attention layers' table (`inkling2_loop`)
        relative_length: Option<u64>,
        relative_extent: Option<u64>,
        /// the L2 norm the queries and keys, and the relative position
        /// channels, are clipped to (`inkling2_loop`)
        qk_clip: Option<f32>,
        relative_clip: Option<f32>,
        /// the full attention layers' query scale `1 + alpha * ln(max(1,
        /// (pos + 1) / floor))` (`inkling2_loop`)
        log_scale_floor: Option<u64>,
        log_scale_alpha: Option<f32>,
    }

    pub struct Rope {
        dimension_count: Option<u64>,
        dimension_sections: Option<Vec<i32>>,
        freq_base: Option<f32>,
    }

    /// the delta net layers
    pub struct Ssm {
        conv_kernel: Option<u64>,
        /// key heads
        group_count: Option<u64>,
        inner_size: Option<u64>,
        /// key and value head size
        state_size: Option<u64>,
        /// value heads
        time_step_rank: Option<u64>,
    }

    pub struct Tokenizer {
        ggml: Ggml,
        chat_template: Option<String>,
    }

    pub struct Ggml {
        model: Option<String>,
        pre: Option<String>,
        tokens: Option<Vec<String>>,
        token_type: Option<Vec<i32>>,
        merges: Option<Vec<String>>,
        bos_token_id: Option<u32>,
        eos_token_id: Option<u32>,
        /// every token that ends a generation, the eos token's included
        eog_token_ids: Option<Vec<u32>>,
        padding_token_id: Option<u32>,
        add_bos_token: Option<bool>,
    }

    /// the shards of a split model
    pub struct Split {
        count: Option<u32>,
        no: Option<u32>,
        tensors: SplitTensors,
    }

    pub struct SplitTensors {
        count: Option<u32>,
    }
}

/// The value of a required leaf, by reference, or [`Error::MissingKey`] with
/// its dotted key: `need!(meta, qwen35.block_count)?`. For a section whose
/// key prefix is only known at run time, `need!(section @ prefix, block_count)?`.
macro_rules! need {
    ($m:expr, @ $prefix:expr, $($p:ident).+) => {
        $m.$($p).+.as_ref().ok_or_else(|| {
            $crate::Error::MissingKey(::alloc::format!(
                "{}.{}",
                $prefix,
                [$(stringify!($p)),+].join(".")
            ))
        })
    };
    ($m:expr, $($p:ident).+) => {
        $m.$($p).+.as_ref().ok_or_else(|| {
            $crate::Error::MissingKey([$(stringify!($p)),+].join("."))
        })
    };
}
pub(crate) use need;

/// A per-layer value stored as one value for every layer or as an array of
/// `n` (llama.cpp's `get_key_or_arr`); None when it is neither, or negative
pub fn per_layer(v: &Value, n: usize) -> Option<Vec<u64>> {
    match v.as_array() {
        Some(a) if a.len() == n => a.iter().map(u64::from_value).collect(),
        Some(_) => None,
        None => u64::from_value(v).map(|x| vec![x; n]),
    }
}
