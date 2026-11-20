//! The tensor table as nested structs: each `.` of a tensor name is a field
//! access, and a number an index, so `blk.3.attn_q.weight` is
//! `tensors.blk[3].attn_q.weight`.
//!
//! Every tensor is an `Option`, `None` when the file doesn't have it. A tensor
//! with no field here is kept in [`Tensors::extra`].

use core::ops::Index;

use super::TensorInfo;
use crate::prelude::*;

/// blocks past this index go to `extra`, not to a `Vec` of that length
const MAX_BLOCKS: usize = 1 << 16;

/// A node of the name tree: a struct of fields, a list of blocks, or a tensor
pub trait TensorField {
    /// Store `t` under `name`, the rest of its dotted name below this node
    /// (empty for a tensor itself). Gives `t` back when no field has that
    /// name, or the field already holds a tensor.
    fn set(&mut self, name: &str, t: TensorInfo) -> Result<(), TensorInfo>;
    /// The tensor stored under `name`
    fn get(&self, name: &str) -> Option<&TensorInfo>;
}

fn split(name: &str) -> (&str, &str) {
    name.split_once('.').unwrap_or((name, ""))
}

impl TensorField for Option<TensorInfo> {
    fn set(&mut self, name: &str, t: TensorInfo) -> Result<(), TensorInfo> {
        if !name.is_empty() || self.is_some() {
            return Err(t);
        }
        *self = Some(t);
        Ok(())
    }
    fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.as_ref().filter(|_| name.is_empty())
    }
}

/// `blk.<i>`: the number is the index, written without leading zeros
fn index(s: &str) -> Option<usize> {
    let canonical =
        !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()) && (s == "0" || !s.starts_with('0'));
    s.parse().ok().filter(|_| canonical)
}

impl<T: TensorField + Default> TensorField for Vec<T> {
    fn set(&mut self, name: &str, t: TensorInfo) -> Result<(), TensorInfo> {
        let (head, rest) = split(name);
        let Some(i) = index(head).filter(|&i| i < MAX_BLOCKS) else {
            return Err(t);
        };
        if i >= self.len() {
            self.resize_with(i + 1, T::default);
        }
        self[i].set(rest, t)
    }
    fn get(&self, name: &str) -> Option<&TensorInfo> {
        let (head, rest) = split(name);
        self.as_slice().get(index(head)?)?.get(rest)
    }
}

/// A struct whose fields are the next component of the name, each one a
/// [`TensorField`]
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

        impl TensorField for $name {
            fn set(&mut self, name: &str, t: TensorInfo) -> Result<(), TensorInfo> {
                let (head, rest) = split(name);
                match head {
                    $(stringify!($field) => self.$field.set(rest, t),)*
                    _ => Err(t),
                }
            }

            fn get(&self, name: &str) -> Option<&TensorInfo> {
                let (head, rest) = split(name);
                match head {
                    $(stringify!($field) => self.$field.get(rest),)*
                    _ => None,
                }
            }
        }
    )*};
}

type Tensor = Option<TensorInfo>;

node! {
    /// A linear map or a norm: `<name>.weight`, and `<name>.bias`
    pub struct Param {
        weight: Tensor,
        bias: Tensor,
    }

    /// The tensors of `blk.<i>`: an attention or a delta net layer
    pub struct Block {
        attn_norm: Param,
        /// the norm before the feed-forward: `post_attention_norm` in `qwen35`,
        /// `ffn_norm` in `flint`
        post_attention_norm: Param,
        ffn_norm: Param,
        ffn_gate: Param,
        ffn_up: Param,
        ffn_down: Param,

        /// mixture of experts: the router, `[n_embd, n_expert]`
        ffn_gate_inp: Param,
        /// a bias of the router's probabilities for the choice of the
        /// experts only, `exp_probs_b.bias`
        exp_probs_b: Param,
        /// the experts' matrices, `[cols, rows, n_expert]`; gate and up may
        /// be one, gate rows then up rows
        ffn_gate_exps: Param,
        ffn_up_exps: Param,
        ffn_gate_up_exps: Param,
        ffn_down_exps: Param,
        /// the shared expert, and the vector of its sigmoid gate
        ffn_gate_shexp: Param,
        ffn_up_shexp: Param,
        ffn_down_shexp: Param,
        ffn_gate_inp_shexp: Param,

        /// the queries; in `qwen35`, query and output gate, interleaved per head
        attn_q: Param,
        attn_k: Param,
        attn_v: Param,
        attn_output: Param,
        attn_q_norm: Param,
        attn_k_norm: Param,
        /// a logit per head added to the softmax of its scores, which no
        /// value goes with
        attn_sinks: Param,

        /// q, k, v projection of the delta net
        attn_qkv: Param,
        /// z: output gate of the delta net
        attn_gate: Param,
        ssm_conv1d: Param,
        ssm_dt: Param,
        /// `A_log`
        ssm_a: Tensor,
        ssm_beta: Param,
        ssm_alpha: Param,
        ssm_norm: Param,
        ssm_out: Param,

        /// the MTP head on top of this layer
        nextn: Nextn,
    }

    pub struct Nextn {
        enorm: Param,
        hnorm: Param,
        eh_proj: Param,
        shared_head_norm: Param,
    }
}

/// The root of the name tree
#[derive(Debug, Clone, Default)]
pub struct Tensors {
    pub token_embd: Param,
    pub output_norm: Param,
    pub output: Param,
    pub blk: Vec<Block>,
    /// tensors with no field, or a name already taken, in file order
    pub extra: Vec<TensorInfo>,
}

impl Tensors {
    /// Store `t` under its name, or in `extra`
    pub(crate) fn insert(&mut self, t: TensorInfo) {
        let name = t.name.clone();
        let (head, rest) = split(&name);
        let placed = match head {
            "token_embd" => self.token_embd.set(rest, t),
            "output_norm" => self.output_norm.set(rest, t),
            "output" => self.output.set(rest, t),
            "blk" => self.blk.set(rest, t),
            _ => Err(t),
        };
        if let Err(t) = placed {
            self.extra.push(t);
        }
    }

    /// The tensor named `name`, from its field or `extra`
    pub fn get(&self, name: &str) -> Option<&TensorInfo> {
        let (head, rest) = split(name);
        let field = match head {
            "token_embd" => self.token_embd.get(rest),
            "output_norm" => self.output_norm.get(rest),
            "output" => self.output.get(rest),
            "blk" => TensorField::get(&self.blk, rest),
            _ => None,
        };
        field.or_else(|| self.extra.iter().find(|t| t.name == name))
    }
}

/// `tensors[name]`, panicking when there's no such tensor
impl<K: AsRef<str> + ?Sized> Index<&K> for Tensors {
    type Output = TensorInfo;

    fn index(&self, name: &K) -> &TensorInfo {
        let name = name.as_ref();
        self.get(name).unwrap_or_else(|| panic!("no tensor {name}"))
    }
}

/// The tensor at a field path, or [`Error::MissingTensor`](crate::Error) with
/// its name: `tensor!(t, token_embd.weight)?`, `tensor!(t, blk[i].attn_q.weight)?`
macro_rules! tensor {
    ($t:expr, blk[$i:expr] $(. $p:ident)+) => {{
        let i: usize = $i;
        $t.blk
            .as_slice()
            .get(i)
            .and_then(|b| b $(. $p)+ .as_ref())
            .ok_or_else(|| {
                $crate::Error::MissingTensor(format!(
                    "blk.{i}{}",
                    concat!($(".", stringify!($p)),+)
                ))
            })
    }};
    ($t:expr, $($p:ident).+) => {
        $t.$($p).+.as_ref().ok_or_else(|| {
            $crate::Error::MissingTensor([$(stringify!($p)),+].join("."))
        })
    };
}
pub(crate) use tensor;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::GgmlType;

    fn info(name: &str) -> TensorInfo {
        TensorInfo {
            name: name.into(),
            dims: vec![4],
            ty: GgmlType::F32,
            shard: 0,
            offset: 0,
        }
    }

    #[test]
    fn names_to_fields() {
        let mut t = Tensors::default();
        for n in [
            "blk.2.attn_q.weight",
            "blk.0.ssm_a",
            "blk.1.nextn.eh_proj.weight",
            "output.weight",
            // no field, a non-canonical index, a duplicate
            "blk.0.attn_q.scale",
            "blk.01.attn_q.weight",
            "blk.2.attn_q.weight",
            "rope_freqs.weight",
        ] {
            t.insert(info(n));
        }
        assert_eq!(t.blk.len(), 3);
        assert!(t.blk[2].attn_q.weight.is_some());
        assert!(t.blk[0].ssm_a.is_some());
        assert!(t.blk[1].nextn.eh_proj.weight.is_some());
        assert!(t.output.weight.is_some() && t.token_embd.weight.is_none());
        let extra: Vec<_> = t.extra.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            extra,
            [
                "blk.0.attn_q.scale",
                "blk.01.attn_q.weight",
                "blk.2.attn_q.weight",
                "rope_freqs.weight"
            ]
        );
        assert_eq!(
            t["blk.1.nextn.eh_proj.weight"].name,
            "blk.1.nextn.eh_proj.weight"
        );
        assert_eq!(t["rope_freqs.weight"].name, "rope_freqs.weight");
        assert!(t.get("blk.5.attn_q.weight").is_none());
    }

    #[test]
    fn missing_tensor_names() {
        let t = Tensors::default();
        let name = |r: Result<&TensorInfo, crate::Error>| match r {
            Err(crate::Error::MissingTensor(n)) => n,
            _ => panic!(),
        };
        assert_eq!(name(tensor!(t, token_embd.weight)), "token_embd.weight");
        assert_eq!(name(tensor!(t, blk[7].ssm_a)), "blk.7.ssm_a");
        assert_eq!(
            name(tensor!(t, blk[3].nextn.eh_proj.weight)),
            "blk.3.nextn.eh_proj.weight"
        );
    }
}
