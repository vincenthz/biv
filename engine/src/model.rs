//! The decoders of the llama.cpp archs `qwen35` and `qwen35moe` (Qwen3.5 /
//! Qwen3.8), `flint`, and `inkling2_loop` (Inkling-2, in `model/inkling.rs`).
//!
//! Ported from llama.cpp `src/models/qwen35.cpp`, `qwen35moe.cpp`,
//! `delta-net-base.cpp` and `flint.cpp`.
//!
//! `qwen35`: every `full_attention_interval`-th layer is gated full attention
//! (GQA, partial NeoX rope, sigmoid output gate); the others are Gated
//! DeltaNet linear attention with a short causal conv and a recurrent state.
//! The feed-forward is dense, or a mixture of experts: a router picks the
//! top-k experts of each token, whose SwiGLU outputs are summed with their
//! weights, plus a shared expert behind a sigmoid gate.
//!
//! `flint`: every layer is attention (GQA, QK norm), over a sliding window
//! of the last positions with NeoX rope, or over all of them without
//! positional encoding, as the GGUF's patterns say; each head has a learned
//! sink, a logit in its softmax that no value goes with. The first layers'
//! feed-forwards are dense, the others a mixture of experts routed by
//! sigmoid, plus a shared expert without a gate.
//!
//! The MTP (next-n) block stored after the trunk drafts tokens for
//! speculative decoding: [`Model::draft`] proposes, [`Model::verify`] runs
//! the proposals through the trunk and [`Model::keep`] drops the rejected ones.

use crate::Error;
use crate::engine::{DraftLogits, Pass};
use crate::gguf::{Gguf, TensorInfo, need, tensor};
#[cfg(not(any(feature = "std", test)))]
use crate::math::Float;
use crate::memory::{ModelMemory, SessionMemory, Shape};
use crate::ops::{self, Input, Weight, matmul_new};
use crate::par::*;
use crate::prelude::*;
use crate::quant::{self, GgmlType};
use crate::tokenizer::TokenId;
use alloc::collections::VecDeque;

pub(crate) mod inkling;
pub use inkling::InklingConfig;

/// The llama.cpp architecture of a model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    Qwen35,
    Qwen35Moe,
    Flint,
    Inkling2Loop,
}

impl Arch {
    pub fn name(self) -> &'static str {
        match self {
            Arch::Qwen35 => "qwen35",
            Arch::Qwen35Moe => "qwen35moe",
            Arch::Flint => "flint",
            Arch::Inkling2Loop => "inkling2_loop",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub arch: Arch,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_vocab: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_rot: usize,
    pub rope_base: f32,
    pub eps: f32,
    pub context_length: usize,
    /// per layer: gated delta net (true) or attention (false)
    pub recurrent: Vec<bool>,
    /// per layer: whether its attention sees the last `sliding_window`
    /// positions only (false for the delta nets)
    pub swa: Vec<bool>,
    /// positions a sliding window layer sees, its token's included
    pub sliding_window: usize,
    /// per layer: whether its attention has rotary embeddings
    pub rope: Vec<bool>,
    /// whether the query projection has a sigmoid output gate per head,
    /// interleaved with the queries (`qwen35`)
    pub attn_gate: bool,
    /// whether each attention head has a sink logit (`flint`)
    pub attn_sinks: bool,
    pub conv_kernel: usize,
    /// key/value head size of the delta net
    pub ssm_state: usize,
    pub ssm_k_heads: usize,
    pub ssm_v_heads: usize,
    /// the mixture of experts feed-forwards, instead of dense ones but for
    /// the first `n_dense_lead` layers
    pub moe: Option<MoeConfig>,
    pub n_dense_lead: usize,
    /// the hyperparameters of `inkling2_loop`, whose layers are run by its
    /// own code (`model/inkling.rs`) instead
    pub inkling: Option<InklingConfig>,
}

/// How the router's logits become the experts' probabilities
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gating {
    Softmax,
    Sigmoid,
}

/// A mixture of experts: each token goes through the `n_expert_used` experts
/// of highest router probability (plus a bias, when the layers have one),
/// weighted by these probabilities, over their sum when `norm_weights`, times
/// `weights_scale`; and through the shared expert, gated by a sigmoid when
/// `shared_gate`
#[derive(Debug, Clone)]
pub struct MoeConfig {
    pub n_expert: usize,
    /// experts per token
    pub n_expert_used: usize,
    /// width of an expert's feed-forward
    pub n_ff_exp: usize,
    /// width of the shared expert's, 0 without one
    pub n_ff_shexp: usize,
    pub gating: Gating,
    pub norm_weights: bool,
    pub weights_scale: f32,
    pub shared_gate: bool,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self, Error> {
        let m = &g.metadata;
        let name = need!(m, general.architecture)?.as_str();
        let (arch, q) = match name {
            "qwen35" => (Arch::Qwen35, &m.qwen35),
            "qwen35moe" => (Arch::Qwen35Moe, &m.qwen35moe),
            "flint" => (Arch::Flint, &m.flint),
            "inkling2_loop" => return Self::inkling(g),
            _ => return Err(Error::UnsupportedArchitecture(name.into())),
        };
        let arch_name = name;
        let u = |v: Option<u64>| v.map(|v| v as usize);
        let need = |v: Result<&u64, Error>| v.map(|&v| v as usize);

        let block_count = need(need!(q, @arch_name, block_count))?;
        let nextn = u(q.nextn_predict_layers).unwrap_or(0);
        let n_layer = block_count - nextn;
        let n_embd = need(need!(q, @arch_name, embedding_length))?;
        let n_head = need(need!(q, @arch_name, attention.head_count))?;
        let head_dim = u(q.attention.key_length).unwrap_or(n_embd / n_head);
        if u(q.attention.value_length).unwrap_or(head_dim) != head_dim {
            return Err(Error::UnsupportedModel("key_length != value_length"));
        }
        let n_vocab = tensor!(g.tensors, token_embd.weight)?.dims[1] as usize;
        // a per-layer pattern, of at least `n_layer` layers
        let pattern = |p: &Option<Vec<bool>>, key: &str| -> Result<Vec<bool>, Error> {
            match p {
                Some(a) if a.len() >= n_layer => Ok(a[..n_layer].to_vec()),
                Some(_) => Err(Error::UnsupportedModel(match key {
                    "rope" => "rope_pattern shorter than block_count",
                    _ => "sliding_window_pattern shorter than block_count",
                })),
                None => Err(Error::MissingKey(format!(
                    "{arch_name}.attention.{key}_pattern"
                ))),
            }
        };

        let mut cfg = Config {
            arch,
            n_layer,
            n_embd,
            n_ff: 0,
            n_vocab,
            n_head,
            n_head_kv: need(need!(q, @arch_name, attention.head_count_kv))?,
            head_dim,
            n_rot: u(q.rope.dimension_count).unwrap_or(head_dim),
            rope_base: q.rope.freq_base.unwrap_or(10_000.0),
            eps: *need!(q, @arch_name, attention.layer_norm_rms_epsilon)?,
            context_length: u(q.context_length).unwrap_or(4096),
            recurrent: vec![false; n_layer],
            swa: vec![false; n_layer],
            sliding_window: 0,
            rope: vec![true; n_layer],
            attn_gate: arch != Arch::Flint,
            attn_sinks: arch == Arch::Flint,
            conv_kernel: 0,
            ssm_state: 0,
            ssm_k_heads: 0,
            ssm_v_heads: 0,
            moe: None,
            n_dense_lead: 0,
            inkling: None,
        };

        if arch == Arch::Flint {
            cfg.swa = pattern(&q.attention.sliding_window_pattern, "sliding_window")?;
            cfg.rope = pattern(&q.attention.rope_pattern, "rope")?;
            cfg.sliding_window = need(need!(q, @arch_name, attention.sliding_window))?;
            if cfg.sliding_window == 0 && cfg.swa.contains(&true) {
                return Err(Error::UnsupportedModel("sliding window of 0 positions"));
            }
            cfg.n_dense_lead = u(q.leading_dense_block_count).unwrap_or(0).min(n_layer);
        } else {
            cfg.recurrent = match &q.attention.recurrent_layers {
                Some(a) => a.iter().take(n_layer).copied().collect(),
                None => {
                    let interval = u(q.full_attention_interval).unwrap_or(4);
                    (0..n_layer).map(|i| (i + 1) % interval != 0).collect()
                }
            };
            cfg.ssm_state = need(need!(q, @arch_name, ssm.state_size))?;
            cfg.ssm_v_heads = need(need!(q, @arch_name, ssm.time_step_rank))?;
            if need(need!(q, @arch_name, ssm.inner_size))? != cfg.ssm_state * cfg.ssm_v_heads {
                return Err(Error::UnsupportedModel(
                    "ssm.inner_size != state_size * time_step_rank",
                ));
            }
            cfg.conv_kernel = need(need!(q, @arch_name, ssm.conv_kernel))?;
            cfg.ssm_k_heads = need(need!(q, @arch_name, ssm.group_count))?;
        }

        if arch != Arch::Qwen35 {
            let n_expert = need(need!(q, @arch_name, expert_count))?;
            let n_expert_used = need(need!(q, @arch_name, expert_used_count))?;
            if n_expert_used == 0 || n_expert_used > n_expert {
                return Err(Error::UnsupportedModel(
                    "expert_used_count not in 1..=expert_count",
                ));
            }
            // qwen35moe: softmax, the weights normalized
            let gating = match q.expert_gating_func {
                None | Some(1) => Gating::Softmax,
                Some(2) => Gating::Sigmoid,
                Some(_) => return Err(Error::UnsupportedModel("expert_gating_func not 1 or 2")),
            };
            cfg.moe = Some(MoeConfig {
                n_expert,
                n_expert_used,
                n_ff_exp: need(need!(q, @arch_name, expert_feed_forward_length))?,
                n_ff_shexp: u(q.expert_shared_feed_forward_length).unwrap_or(0),
                gating,
                norm_weights: q.expert_weights_norm.unwrap_or(arch == Arch::Qwen35Moe),
                weights_scale: q.expert_weights_scale.filter(|&s| s != 0.0).unwrap_or(1.0),
                shared_gate: arch == Arch::Qwen35Moe,
            });
        }
        // a mixture of experts has no dense feed-forward, but in its leading layers
        cfg.n_ff = match cfg.moe {
            Some(_) if cfg.n_dense_lead == 0 => u(q.feed_forward_length).unwrap_or(0),
            _ => need(need!(q, @arch_name, feed_forward_length))?,
        };
        Ok(cfg)
    }

    /// Whether layer `i` (the MTP head's, past the trunk) attends over the
    /// sliding window only
    pub fn is_swa(&self, i: usize) -> bool {
        self.swa.get(i).copied().unwrap_or(false)
    }

    /// Whether layer `i` (the MTP head's, past the trunk) has rotary embeddings
    pub fn has_rope(&self, i: usize) -> bool {
        self.rope.get(i).copied().unwrap_or(true)
    }

    /// Whether layer `i` (the MTP head's, past the trunk) has a dense
    /// feed-forward
    pub fn is_dense(&self, i: usize) -> bool {
        self.moe.is_none() || i < self.n_dense_lead
    }

    /// Floats per position of a kv cache, keys or values
    pub fn kv_dim(&self) -> usize {
        self.n_head_kv * self.head_dim
    }

    /// Positions the CPU caches of a sliding window layer hold, for a
    /// session of `context` tokens whose verify passes take `draft` tokens
    /// after the chosen one: the window but the next token's own position,
    /// and the positions a verify pass writes, the rejected ones too. A pass
    /// writes its rows after its scores, which read its own rows elsewhere.
    pub fn cpu_ring(&self, context: usize, draft: usize) -> usize {
        self.sliding_window
            .saturating_add(draft)
            .min(context)
            .max(1)
    }

    /// width of the conv / qkv projection of the delta net: q, k, v
    pub fn conv_dim(&self) -> usize {
        self.ssm_state * (2 * self.ssm_k_heads + self.ssm_v_heads)
    }

    pub fn d_inner(&self) -> usize {
        self.ssm_state * self.ssm_v_heads
    }
}

pub(crate) struct Attention<'a> {
    /// the queries, or with [`Config::attn_gate`] query and output gate
    /// interleaved per head: `[q(head_dim), gate(head_dim)] * n_head`
    pub(crate) wq: Weight<'a>,
    pub(crate) wk: Weight<'a>,
    pub(crate) wv: Weight<'a>,
    pub(crate) wo: Weight<'a>,
    pub(crate) q_norm: Vec<f32>,
    pub(crate) k_norm: Vec<f32>,
    /// a logit per head in the softmax of its scores, no value with it
    pub(crate) sinks: Option<Vec<f32>>,
    /// rotary embeddings of the queries and keys
    pub(crate) rope: bool,
    /// positions a token sees, its own included, when not all before it
    pub(crate) window: Option<usize>,
}

pub(crate) struct DeltaNet<'a> {
    /// q, k, v projection (input of the conv)
    pub(crate) wqkv: Weight<'a>,
    /// z: output gate
    pub(crate) wgate: Weight<'a>,
    /// `[conv_dim][conv_kernel]`
    pub(crate) conv1d: Vec<f32>,
    pub(crate) dt_bias: Vec<f32>,
    /// `-exp(A_log)`
    pub(crate) a: Vec<f32>,
    pub(crate) wbeta: Weight<'a>,
    pub(crate) walpha: Weight<'a>,
    pub(crate) norm: Vec<f32>,
    pub(crate) wout: Weight<'a>,
}

pub(crate) enum Mixer<'a> {
    Attention(Attention<'a>),
    DeltaNet(DeltaNet<'a>),
}

/// A stack of weight matrices of one shape, one per expert, in one tensor
#[derive(Clone, Copy)]
pub(crate) struct Experts<'a> {
    pub(crate) ty: GgmlType,
    pub(crate) cols: usize,
    pub(crate) rows: usize,
    pub(crate) n: usize,
    /// the tensor's data
    pub(crate) data: &'a [u8],
    /// bytes from an expert's matrix to the next
    pub(crate) stride: usize,
    /// bytes from the start of an expert's matrix to these rows: the up half
    /// of a fused gate and up
    pub(crate) offset: usize,
}

impl<'a> Experts<'a> {
    pub(crate) fn row_bytes(&self) -> usize {
        self.ty.row_size(self.cols)
    }

    /// The matrix of expert `e`
    pub(crate) fn expert(&self, e: usize) -> Weight<'a> {
        let start = e * self.stride + self.offset;
        let data = &self.data[start..start + self.rows * self.row_bytes()];
        Weight::new(self.ty, self.cols, self.rows, data)
    }
}

/// A feed-forward every token goes through, its output scaled by
/// `sigmoid(gate_inp · x)` when it has a gate
pub(crate) struct SharedExpert<'a> {
    pub(crate) gate: Weight<'a>,
    pub(crate) up: Weight<'a>,
    pub(crate) down: Weight<'a>,
    /// one row
    pub(crate) gate_inp: Option<Weight<'a>>,
}

/// A mixture of experts feed-forward, see [`MoeConfig`]
pub(crate) struct Moe<'a> {
    /// the experts' logits of a token
    pub(crate) router: Weight<'a>,
    /// added to the probabilities to choose the experts, not to weigh them
    pub(crate) bias: Option<Vec<f32>>,
    pub(crate) gate: Experts<'a>,
    pub(crate) up: Experts<'a>,
    pub(crate) down: Experts<'a>,
    pub(crate) shared: Option<SharedExpert<'a>>,
}

pub(crate) enum Ffn<'a> {
    /// `down(silu(gate x) * up x)`
    Dense {
        gate: Weight<'a>,
        up: Weight<'a>,
        down: Weight<'a>,
    },
    Moe(Box<Moe<'a>>),
}

pub(crate) struct Layer<'a> {
    pub(crate) attn_norm: Vec<f32>,
    pub(crate) post_norm: Vec<f32>,
    pub(crate) mixer: Mixer<'a>,
    pub(crate) ffn: Ffn<'a>,
}

/// The `n_expert_used` experts of highest probability (plus `bias`), from a
/// token's router logits, and their weights: the probabilities, over their
/// sum when [`MoeConfig::norm_weights`], scaled. In order of probability plus
/// bias, the lowest index first among equals.
pub(crate) fn route(logits: &[f32], m: &MoeConfig, bias: Option<&[f32]>) -> (Vec<usize>, Vec<f32>) {
    let mut p = logits.to_vec();
    match m.gating {
        Gating::Softmax => ops::softmax(&mut p),
        Gating::Sigmoid => p.iter_mut().for_each(|x| *x = ops::sigmoid(*x)),
    }
    let score = |i: usize| p[i] + bias.map_or(0.0, |b| b[i]);
    let mut ids: Vec<usize> = (0..p.len()).collect();
    ids.sort_by(|&a, &b| score(b).total_cmp(&score(a)).then(a.cmp(&b)));
    ids.truncate(m.n_expert_used);
    let sum = if m.norm_weights {
        ids.iter().map(|&i| p[i]).sum::<f32>().max(1.0 / 16384.0)
    } else {
        1.0
    };
    let scale = m.weights_scale;
    let w = ids
        .iter()
        .map(|&i| {
            if scale == 1.0 {
                p[i] / sum
            } else {
                p[i] / sum * scale
            }
        })
        .collect();
    (ids, w)
}

/// The MTP head: row `p` combines the normalized hidden state of token `p`
/// with the embedding of token `p + 1`, runs one gated attention layer over
/// its own history of such rows, and predicts token `p + 2` through the
/// trunk's output head. Chained drafts feed back its normalized output as the
/// hidden state.
pub(crate) struct Mtp<'a> {
    pub(crate) enorm: Vec<f32>,
    pub(crate) hnorm: Vec<f32>,
    /// `[enorm(embedding) ‖ hnorm(hidden)]` to the model width
    pub(crate) eh_proj: Weight<'a>,
    pub(crate) layer: Layer<'a>,
    pub(crate) head_norm: Vec<f32>,
}

pub struct Model<'a> {
    pub cfg: Config,
    pub(crate) tok_embd: Weight<'a>,
    pub(crate) output_norm: Vec<f32>,
    pub(crate) output: Weight<'a>,
    pub(crate) layers: Vec<Layer<'a>>,
    pub(crate) mtp: Option<Mtp<'a>>,
    pub(crate) inv_freq: Vec<f32>,
    pub(crate) memory: ModelMemory,
    /// the layers of `inkling2_loop` (then `layers` is empty)
    pub(crate) inkling: Option<Box<inkling::Inkling<'a>>>,
}

struct Loader<'g> {
    g: &'g Gguf,
}

impl<'g> Loader<'g> {
    fn weight(
        &self,
        info: Result<&TensorInfo, Error>,
        cols: usize,
        rows: usize,
    ) -> Result<Weight<'g>, Error> {
        let info = info?;
        let expected = [cols as u64, rows as u64];
        if info.dims != expected {
            return Err(Error::TensorShape {
                tensor: info.name.clone(),
                dims: info.dims.clone(),
                expected: expected.into(),
            });
        }
        if !info.ty.has_dot() {
            return Err(Error::NoKernel {
                tensor: info.name.clone(),
                ty: info.ty,
            });
        }
        Ok(Weight::new(info.ty, cols, rows, self.g.data(info)))
    }

    /// A stack of `n` matrices of `rows` rows of `cols`, one per expert
    fn experts(
        &self,
        info: Result<&TensorInfo, Error>,
        cols: usize,
        rows: usize,
        n: usize,
    ) -> Result<Experts<'g>, Error> {
        let info = info?;
        let expected = [cols as u64, rows as u64, n as u64];
        if info.dims != expected {
            return Err(Error::TensorShape {
                tensor: info.name.clone(),
                dims: info.dims.clone(),
                expected: expected.into(),
            });
        }
        if !info.ty.has_dot() {
            return Err(Error::NoKernel {
                tensor: info.name.clone(),
                ty: info.ty,
            });
        }
        Ok(Experts {
            ty: info.ty,
            cols,
            rows,
            n,
            data: self.g.data(info),
            stride: rows * info.ty.row_size(cols),
            offset: 0,
        })
    }

    /// A vector of `n` as a weight of one row
    fn row(&self, info: Result<&TensorInfo, Error>, n: usize) -> Result<Weight<'g>, Error> {
        let info = info?;
        if info.dims != [n as u64] {
            return Err(Error::TensorShape {
                tensor: info.name.clone(),
                dims: info.dims.clone(),
                expected: vec![n as u64],
            });
        }
        if !info.ty.has_dot() {
            return Err(Error::NoKernel {
                tensor: info.name.clone(),
                ty: info.ty,
            });
        }
        Ok(Weight::new(info.ty, n, 1, self.g.data(info)))
    }

    fn ffn(&self, cfg: &Config, i: usize) -> Result<Ffn<'g>, Error> {
        let l = self;
        let e = cfg.n_embd;
        let t = &self.g.tensors;
        let Some(moe) = cfg.moe.as_ref().filter(|_| !cfg.is_dense(i)) else {
            return Ok(Ffn::Dense {
                gate: l.weight(tensor!(t, blk[i].ffn_gate.weight), e, cfg.n_ff)?,
                up: l.weight(tensor!(t, blk[i].ffn_up.weight), e, cfg.n_ff)?,
                down: l.weight(tensor!(t, blk[i].ffn_down.weight), cfg.n_ff, e)?,
            });
        };
        let (n, ff) = (moe.n_expert, moe.n_ff_exp);
        let (gate, up) = match tensor!(t, blk[i].ffn_gate_up_exps.weight) {
            // gate rows then up rows, per expert
            Ok(info) => {
                let both = l.experts(Ok(info), e, 2 * ff, n)?;
                let half = Experts { rows: ff, ..both };
                let up = Experts {
                    offset: ff * both.row_bytes(),
                    ..half
                };
                (half, up)
            }
            Err(_) => (
                l.experts(tensor!(t, blk[i].ffn_gate_exps.weight), e, ff, n)?,
                l.experts(tensor!(t, blk[i].ffn_up_exps.weight), e, ff, n)?,
            ),
        };
        let shared = match moe.n_ff_shexp {
            0 => None,
            sff => Some(SharedExpert {
                gate: l.weight(tensor!(t, blk[i].ffn_gate_shexp.weight), e, sff)?,
                up: l.weight(tensor!(t, blk[i].ffn_up_shexp.weight), e, sff)?,
                down: l.weight(tensor!(t, blk[i].ffn_down_shexp.weight), sff, e)?,
                gate_inp: if moe.shared_gate {
                    Some(l.row(tensor!(t, blk[i].ffn_gate_inp_shexp.weight), e)?)
                } else {
                    None
                },
            }),
        };
        let bias = match tensor!(t, blk[i].exp_probs_b.bias) {
            Ok(info) => Some(l.vec(Ok(info), n)?),
            Err(_) => None,
        };
        Ok(Ffn::Moe(Box::new(Moe {
            router: l.weight(tensor!(t, blk[i].ffn_gate_inp.weight), e, n)?,
            bias,
            gate,
            up,
            down: l.experts(tensor!(t, blk[i].ffn_down_exps.weight), ff, e, n)?,
            shared,
        })))
    }

    fn vec(&self, info: Result<&TensorInfo, Error>, n: usize) -> Result<Vec<f32>, Error> {
        let info = info?;
        if info.n_elements() as usize != n {
            return Err(Error::TensorShape {
                tensor: info.name.clone(),
                dims: info.dims.clone(),
                expected: vec![n as u64],
            });
        }
        let mut out = vec![0.0; n];
        quant::dequantize_row(info.ty, self.g.data(info), &mut out);
        Ok(out)
    }

    fn layer(&self, cfg: &Config, i: usize, recurrent: bool) -> Result<Layer<'g>, Error> {
        let l = self;
        let (e, hd) = (cfg.n_embd, cfg.head_dim);
        let t = &self.g.tensors;
        let mixer = if recurrent {
            let conv_dim = cfg.conv_dim();
            let d_inner = cfg.d_inner();
            let hv = cfg.ssm_v_heads;
            if !cfg.ssm_v_heads.is_multiple_of(cfg.ssm_k_heads) {
                return Err(Error::UnsupportedModel(
                    "ssm value heads not a multiple of key heads",
                ));
            }
            Mixer::DeltaNet(DeltaNet {
                wqkv: l.weight(tensor!(t, blk[i].attn_qkv.weight), e, conv_dim)?,
                wgate: l.weight(tensor!(t, blk[i].attn_gate.weight), e, d_inner)?,
                conv1d: l.vec(
                    tensor!(t, blk[i].ssm_conv1d.weight),
                    cfg.conv_kernel * conv_dim,
                )?,
                dt_bias: l.vec(tensor!(t, blk[i].ssm_dt.bias), hv)?,
                a: l.vec(tensor!(t, blk[i].ssm_a), hv)?,
                wbeta: l.weight(tensor!(t, blk[i].ssm_beta.weight), e, hv)?,
                walpha: l.weight(tensor!(t, blk[i].ssm_alpha.weight), e, hv)?,
                norm: l.vec(tensor!(t, blk[i].ssm_norm.weight), cfg.ssm_state)?,
                wout: l.weight(tensor!(t, blk[i].ssm_out.weight), d_inner, e)?,
            })
        } else {
            let q_rows = if cfg.attn_gate { 2 } else { 1 } * hd * cfg.n_head;
            Mixer::Attention(Attention {
                wq: l.weight(tensor!(t, blk[i].attn_q.weight), e, q_rows)?,
                wk: l.weight(tensor!(t, blk[i].attn_k.weight), e, hd * cfg.n_head_kv)?,
                wv: l.weight(tensor!(t, blk[i].attn_v.weight), e, hd * cfg.n_head_kv)?,
                wo: l.weight(tensor!(t, blk[i].attn_output.weight), hd * cfg.n_head, e)?,
                q_norm: l.vec(tensor!(t, blk[i].attn_q_norm.weight), hd)?,
                k_norm: l.vec(tensor!(t, blk[i].attn_k_norm.weight), hd)?,
                sinks: if cfg.attn_sinks {
                    Some(l.vec(tensor!(t, blk[i].attn_sinks.weight), cfg.n_head)?)
                } else {
                    None
                },
                rope: cfg.has_rope(i),
                window: cfg.is_swa(i).then_some(cfg.sliding_window),
            })
        };
        // the norm before the feed-forward, as each architecture names it
        let post_norm = match cfg.arch {
            Arch::Flint => tensor!(t, blk[i].ffn_norm.weight),
            _ => tensor!(t, blk[i].post_attention_norm.weight),
        };
        Ok(Layer {
            attn_norm: l.vec(tensor!(t, blk[i].attn_norm.weight), e)?,
            post_norm: l.vec(post_norm, e)?,
            mixer,
            ffn: l.ffn(cfg, i)?,
        })
    }
}

impl<'a> Model<'a> {
    pub fn load(g: &'a Gguf) -> Result<Self, Error> {
        let cfg = Config::from_gguf(g)?;
        if cfg.inkling.is_some() {
            return Self::load_inkling(g, cfg);
        }
        let l = Loader { g };
        let e = cfg.n_embd;

        let t = &g.tensors;
        let tok_embd = l.weight(tensor!(t, token_embd.weight), e, cfg.n_vocab)?;
        let output = if t.output.weight.is_some() {
            l.weight(tensor!(t, output.weight), e, cfg.n_vocab)?
        } else {
            tok_embd
        };

        let layers = (0..cfg.n_layer)
            .map(|i| l.layer(&cfg, i, cfg.recurrent[i]))
            .collect::<Result<_, _>>()?;

        // the first next-n block, when the file has one
        let m = cfg.n_layer;
        let mtp = if tensor!(t, blk[m].nextn.eh_proj.weight).is_ok() {
            Some(Mtp {
                enorm: l.vec(tensor!(t, blk[m].nextn.enorm.weight), e)?,
                hnorm: l.vec(tensor!(t, blk[m].nextn.hnorm.weight), e)?,
                eh_proj: l.weight(tensor!(t, blk[m].nextn.eh_proj.weight), 2 * e, e)?,
                layer: l.layer(&cfg, m, false)?,
                head_norm: l.vec(tensor!(t, blk[m].nextn.shared_head_norm.weight), e)?,
            })
        } else {
            None
        };

        let inv_freq = (0..cfg.n_rot / 2)
            .map(|i| cfg.rope_base.powf(-2.0 * i as f32 / cfg.n_rot as f32))
            .collect();

        Ok(Model {
            output_norm: l.vec(tensor!(t, output_norm.weight), e)?,
            memory: ModelMemory::from_gguf(g, &cfg, mtp.is_some()),
            cfg,
            tok_embd,
            output,
            layers,
            mtp,
            inv_freq,
            inkling: None,
        })
    }

    /// Whether the model can draft tokens for speculative decoding
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    pub(crate) fn shape(&self) -> Shape<'_> {
        Shape {
            cfg: &self.cfg,
            has_mtp: self.has_mtp(),
            head_row_bytes: self.output.data.len() / self.output.rows,
        }
    }

    /// An empty sequence able to hold `context` tokens, its memory allocated
    /// up front, whose verify passes take up to `draft` tokens after the
    /// chosen one; with `draft` > 0, the MTP head follows it to draft tokens
    pub fn new_state(&self, context: usize, draft: usize) -> Result<State, Error> {
        if self.inkling.is_some() {
            return self.inkling_state(context, draft);
        }
        let c = &self.cfg;
        let kv_dim = c.kv_dim();
        let kv = || -> Result<_, Error> {
            // an absurd context fails to allocate rather than wrap around
            let n = context.saturating_mul(kv_dim);
            Ok(LayerState::Attention {
                k: reserved(n)?,
                v: reserved(n)?,
            })
        };
        let ring = c.cpu_ring(context, draft);
        let mut layers = Vec::with_capacity(c.n_layer);
        for (i, &r) in c.recurrent.iter().enumerate() {
            layers.push(if r {
                LayerState::DeltaNet {
                    conv: zeroed((c.conv_kernel - 1) * c.conv_dim())?,
                    s: zeroed(c.ssm_v_heads * c.ssm_state * c.ssm_state)?,
                    saved: Vec::new(),
                }
            } else if c.is_swa(i) {
                LayerState::Window {
                    k: zeroed(ring * kv_dim)?,
                    v: zeroed(ring * kv_dim)?,
                    ring,
                }
            } else {
                kv()?
            });
        }
        let mtp = if draft > 0 && self.has_mtp() {
            let head = match DraftVocab::new(c.n_vocab) {
                Some(vocab) => {
                    let rb = self.output.data.len() / self.output.rows;
                    let mut rows = Vec::new();
                    rows.try_reserve_exact(DRAFT_CAP * rb)
                        .map_err(|_| Error::OutOfMemory {
                            bytes: DRAFT_CAP * rb,
                        })?;
                    rows.resize(vocab.len() * rb, 0);
                    copy_head_rows(&self.output, vocab.tokens(0..vocab.len()), &mut rows);
                    Some((vocab, rows))
                }
                None => None,
            };
            Some(MtpState {
                kv: kv()?,
                rows: 0,
                last: zeroed(c.n_embd)?,
                head,
            })
        } else {
            None
        };
        let state = State {
            layers,
            n_past: 0,
            context,
            draft,
            kv_dim,
            window: c.sliding_window,
            mtp,
            verify: None,
        };
        debug_assert_eq!(
            state.memory().allocated(),
            self.shape().cpu_session(context, draft, 1).allocated()
        );
        Ok(state)
    }

    /// Run `tokens` through the model, appending them to `state`.
    /// Returns the logits of the last token.
    pub fn forward(&self, state: &mut State, tokens: &[TokenId]) -> Vec<f32> {
        self.run(state, tokens, Pass::Forward).pop().unwrap()
    }

    /// Like [`Model::forward`], skipping the output head
    pub fn prefill(&self, state: &mut State, tokens: &[TokenId]) {
        self.run(state, tokens, Pass::Prefill);
    }

    /// Up to `n` tokens the MTP head predicts after `next`, the token chosen to
    /// follow the sequence. Empty when `state` was made without the head.
    pub fn draft(
        &self,
        state: &mut State,
        next: TokenId,
        n: usize,
        pick: &mut dyn FnMut(&DraftLogits<'_>) -> TokenId,
    ) -> Vec<TokenId> {
        let (Some(mtp), Some(m)) = (&self.mtp, &mut state.mtp) else {
            return Vec::new();
        };
        let c = &self.cfg;
        // the last token's row waits for its successor, `next`
        assert_eq!(
            m.rows + 1,
            state.n_past,
            "draft must follow a forward pass or keep"
        );
        let base = m.rows;
        let mut drafts = Vec::with_capacity(n);
        let (mut tok, mut hidden) = (next, m.last.clone());
        for i in 0..n.min(state.context - base) {
            let row = self.mtp_input(mtp, &[tok], &hidden);
            // the first row is history, the chained ones are speculative and
            // overwritten by the next rows entered
            hidden = self.mtp_block(mtp, &mut m.kv, row, 1, base + i);
            ops::rms_norm(&mut hidden, &mtp.head_norm, c.eps);
            let x = Input::new(&hidden, 1);
            tok = match &m.head {
                Some((vocab, rows)) if !vocab.full() => {
                    let w = Weight::new(self.output.ty, self.output.cols, vocab.len(), rows);
                    pick(&DraftLogits {
                        logits: &matmul_new(&w, &x),
                        tokens: Some(vocab.tokens(0..vocab.len())),
                    })
                }
                _ => pick(&DraftLogits {
                    logits: &matmul_new(&self.output, &x),
                    tokens: None,
                }),
            };
            drafts.push(tok);
        }
        m.rows = base + 1;
        drafts
    }

    /// Append `tokens` provisionally and return the logits after each one.
    /// [`Model::keep`] must follow.
    pub fn verify(&self, state: &mut State, tokens: &[TokenId]) -> Vec<Vec<f32>> {
        self.run(state, tokens, Pass::Verify)
    }

    /// Keep the first `n` tokens of the last [`Model::verify`], dropping the others
    pub fn keep(&self, state: &mut State, n: usize) {
        let v = state.verify.take().expect("keep without verify");
        let verified = state.n_past - v.pos0;
        assert!(
            n >= 1 && n <= verified,
            "keep {n} of {verified} verified tokens"
        );
        state.n_past = v.pos0 + n;
        // attention caches are truncated by the next pass that appends to them
        for ls in &mut state.layers {
            if let LayerState::DeltaNet { conv, s, saved } = ls {
                if n < verified {
                    (*conv, *s) = saved.swap_remove(n - 1);
                }
                saved.clear();
            } else if let LayerState::Inkling { conv, saved, .. } = ls {
                if n < verified {
                    inkling::keep_conv(conv, saved, n, verified);
                }
                *saved = Vec::new();
            }
        }
        if let Some(m) = &mut state.mtp {
            let e = self.cfg.n_embd;
            m.rows = m.rows.min(state.n_past - 1);
            m.last.copy_from_slice(&v.hidden[(n - 1) * e..n * e]);
        }
    }

    fn run(&self, state: &mut State, tokens: &[TokenId], pass: Pass) -> Vec<Vec<f32>> {
        if let Some(ink) = &self.inkling {
            return self.run_inkling(ink, state, tokens, pass);
        }
        let c = &self.cfg;
        let n = tokens.len();
        let e = c.n_embd;
        assert!(n > 0);
        let pos0 = state.n_past;
        assert!(
            pos0 + n <= state.context,
            "context overflow ({pos0} + {n} > {})",
            state.context
        );
        // the rings of the sliding windows hold the rejected tokens' rows too
        assert!(
            pass != Pass::Verify || n <= state.draft + 1,
            "verify of {n} tokens, {} at most",
            state.draft + 1
        );

        if let Some((vocab, rows)) = state.mtp.as_mut().and_then(|m| m.head.as_mut()) {
            let added = vocab.observe(tokens, if pass == Pass::Verify { 1 } else { n });
            let rb = self.output.data.len() / self.output.rows;
            rows.resize(vocab.len() * rb, 0);
            let dst = &mut rows[added.start * rb..];
            copy_head_rows(&self.output, vocab.tokens(added), dst);
        }

        let mut x = vec![0.0f32; n * e];
        for (t, &tok) in tokens.iter().enumerate() {
            self.tok_embd
                .dequantize_row(tok as usize, &mut x[t * e..(t + 1) * e]);
        }
        let save = pass == Pass::Verify;
        for (layer, ls) in self.layers.iter().zip(state.layers.iter_mut()) {
            self.layer(layer, ls, &mut x, n, pos0, save);
        }
        state.n_past += n;

        // normalized hidden states: every row for the MTP head and verify, else the last
        let all = state.mtp.is_some() || pass == Pass::Verify;
        if pass == Pass::Prefill && !all {
            return Vec::new();
        }
        let mut hidden = if all { x } else { x.split_off((n - 1) * e) };
        ops::rms_norm(&mut hidden, &self.output_norm, c.eps);
        if let (Some(mtp), Some(m)) = (&self.mtp, &mut state.mtp) {
            self.mtp_enter(mtp, m, tokens, pos0, &hidden);
        }

        let logits = match pass {
            Pass::Prefill => Vec::new(),
            Pass::Forward => vec![matmul_new(
                &self.output,
                &Input::new(&hidden[hidden.len() - e..], 1),
            )],
            Pass::Verify => {
                let l = matmul_new(&self.output, &Input::new(&hidden, n));
                l.chunks_exact(c.n_vocab).map(<[f32]>::to_vec).collect()
            }
        };
        if pass == Pass::Verify {
            state.verify = Some(Verify { pos0, hidden });
        }
        logits
    }

    /// One decoder layer over the residual stream `x` of `n` tokens at `pos0`;
    /// with `save`, a delta net keeps its state after each token but the last
    fn layer(
        &self,
        layer: &Layer<'_>,
        ls: &mut LayerState,
        x: &mut [f32],
        n: usize,
        pos0: usize,
        save: bool,
    ) {
        let c = &self.cfg;
        let mut h = vec![0.0f32; x.len()];
        ops::rms_norm_into(x, &layer.attn_norm, c.eps, &mut h);
        let mixed = match (&layer.mixer, ls) {
            (Mixer::Attention(a), LayerState::Attention { k, v }) => {
                // drop positions left over from rejected tokens
                let kv_dim = c.kv_dim();
                k.truncate(pos0 * kv_dim);
                v.truncate(pos0 * kv_dim);
                self.attention(a, &h, n, pos0, Cache::Full { k, v })
            }
            (Mixer::Attention(a), LayerState::Window { k, v, ring }) => {
                self.attention(a, &h, n, pos0, Cache::Ring { k, v, ring: *ring })
            }
            (Mixer::DeltaNet(d), LayerState::DeltaNet { conv, s, saved }) => {
                self.delta_net(d, &h, n, conv, s, save.then_some(saved))
            }
            _ => unreachable!("layer / state kind mismatch"),
        };
        ops::add_inplace(x, &mixed);

        ops::rms_norm_into(x, &layer.post_norm, c.eps, &mut h);
        let ffn = self.ffn(layer, &h, n);
        ops::add_inplace(x, &ffn);
    }

    /// The MTP input rows for `tokens` and the hidden states before them
    fn mtp_input(&self, mtp: &Mtp<'_>, tokens: &[TokenId], hidden: &[f32]) -> Vec<f32> {
        let e = self.cfg.n_embd;
        let mut rows = vec![0.0f32; tokens.len() * 2 * e];
        for ((row, &tok), h) in rows
            .chunks_exact_mut(2 * e)
            .zip(tokens)
            .zip(hidden.chunks_exact(e))
        {
            let (emb, hid) = row.split_at_mut(e);
            self.tok_embd.dequantize_row(tok as usize, emb);
            ops::rms_norm(emb, &mtp.enorm, self.cfg.eps);
            hid.copy_from_slice(h);
            ops::rms_norm(hid, &mtp.hnorm, self.cfg.eps);
        }
        rows
    }

    /// The MTP layer over `n` input rows from position `pos0`: its output before the head norm
    fn mtp_block(
        &self,
        mtp: &Mtp<'_>,
        kv: &mut LayerState,
        input: Vec<f32>,
        n: usize,
        pos0: usize,
    ) -> Vec<f32> {
        let mut x = matmul_new(&mtp.eh_proj, &Input::new(&input, n));
        self.layer(&mtp.layer, kv, &mut x, n, pos0, false);
        x
    }

    /// Enter the MTP rows a pass completes: each row whose successor token is
    /// now known, from the normalized hidden states of the pass's `tokens`
    fn mtp_enter(
        &self,
        mtp: &Mtp<'_>,
        m: &mut MtpState,
        tokens: &[TokenId],
        pos0: usize,
        hidden: &[f32],
    ) {
        let e = self.cfg.n_embd;
        let end = pos0 + tokens.len() - 1;
        // the first row is the previous pass's last token, unless a draft entered it
        let first = m.rows;
        debug_assert!(first + 1 >= pos0 && first <= pos0);
        if first < end {
            let mut h = Vec::with_capacity((end - first) * e);
            if first < pos0 {
                h.extend_from_slice(&m.last);
            }
            h.extend_from_slice(&hidden[(first.max(pos0) - pos0) * e..(end - pos0) * e]);
            let input = self.mtp_input(mtp, &tokens[first + 1 - pos0..], &h);
            self.mtp_block(mtp, &mut m.kv, input, end - first, first);
            m.rows = end;
        }
        m.last.copy_from_slice(&hidden[hidden.len() - e..]);
    }

    fn ffn(&self, layer: &Layer<'_>, h: &[f32], n: usize) -> Vec<f32> {
        match &layer.ffn {
            Ffn::Dense { gate, up, down } => dense_ffn(gate, up, down, &Input::new(h, n)),
            Ffn::Moe(m) => self.moe(m, h, n),
        }
    }

    /// Each token through its experts, each expert over its tokens at once
    fn moe(&self, m: &Moe<'_>, h: &[f32], n: usize) -> Vec<f32> {
        let e = self.cfg.n_embd;
        let mc = self.cfg.moe.as_ref().expect("a moe config");
        let input = Input::new(h, n);
        let logits = matmul_new(&m.router, &input);
        // the (token, weight) pairs of each expert
        let mut routed: Vec<Vec<(usize, f32)>> = vec![Vec::new(); m.gate.n];
        for (t, l) in logits.chunks_exact(m.gate.n).enumerate() {
            let (ids, w) = route(l, mc, m.bias.as_deref());
            for (i, w) in ids.into_iter().zip(w) {
                routed[i].push((t, w));
            }
        }
        let mut out = vec![0.0f32; n * e];
        for (i, pairs) in routed.iter().enumerate().filter(|(_, p)| !p.is_empty()) {
            let mut x = Vec::with_capacity(pairs.len() * e);
            for &(t, _) in pairs {
                x.extend_from_slice(&h[t * e..(t + 1) * e]);
            }
            let y = dense_ffn(
                &m.gate.expert(i),
                &m.up.expert(i),
                &m.down.expert(i),
                &Input::new(&x, pairs.len()),
            );
            for (&(t, w), y) in pairs.iter().zip(y.chunks_exact(e)) {
                quant::axpy(&mut out[t * e..(t + 1) * e], w, y);
            }
        }
        if let Some(s) = &m.shared {
            let y = dense_ffn(&s.gate, &s.up, &s.down, &input);
            match &s.gate_inp {
                Some(gate_inp) => {
                    let g = matmul_new(gate_inp, &input);
                    for ((o, y), g) in out.chunks_exact_mut(e).zip(y.chunks_exact(e)).zip(g) {
                        quant::axpy(o, ops::sigmoid(g), y);
                    }
                }
                None => ops::add_inplace(&mut out, &y),
            }
        }
        out
    }

    fn attention(
        &self,
        a: &Attention<'_>,
        h: &[f32],
        n: usize,
        n_past: usize,
        cache: Cache<'_>,
    ) -> Vec<f32> {
        let c = &self.cfg;
        let hd = c.head_dim;
        let kv_dim = c.kv_dim();
        let input = Input::new(h, n);
        let qg = matmul_new(&a.wq, &input);
        let mut k = matmul_new(&a.wk, &input);
        let v = matmul_new(&a.wv, &input);

        // split the joint projection into query and gate, per head
        let (mut q, gate) = if c.attn_gate {
            let mut q = vec![0.0f32; n * c.n_head * hd];
            let mut gate = vec![0.0f32; n * c.n_head * hd];
            for (i, pair) in qg.chunks_exact(2 * hd).enumerate() {
                q[i * hd..(i + 1) * hd].copy_from_slice(&pair[..hd]);
                gate[i * hd..(i + 1) * hd].copy_from_slice(&pair[hd..]);
            }
            (q, Some(gate))
        } else {
            (qg, None)
        };

        ops::rms_norm(&mut q, &a.q_norm, c.eps);
        ops::rms_norm(&mut k, &a.k_norm, c.eps);
        if a.rope {
            for t in 0..n {
                let pos = n_past + t;
                ops::rope_neox(
                    &mut q[t * c.n_head * hd..(t + 1) * c.n_head * hd],
                    hd,
                    pos,
                    &self.inv_freq,
                );
                ops::rope_neox(
                    &mut k[t * kv_dim..(t + 1) * kv_dim],
                    hd,
                    pos,
                    &self.inv_freq,
                );
            }
        }

        // the rows of positions before the pass, and of the pass's: `k`, `v`
        let (k_hist, v_hist, ring) = match &cache {
            Cache::Full { k, v } => (&k[..], &v[..], None),
            Cache::Ring { k, v, ring } => (&k[..], &v[..], Some(*ring)),
        };
        let hist_row = |p: usize| ring.map_or(p, |r| p % r) * kv_dim;
        let k_at = |p: usize| -> &[f32] {
            if p >= n_past {
                &k[(p - n_past) * kv_dim..]
            } else {
                &k_hist[hist_row(p)..]
            }
        };
        let v_at = |p: usize| -> &[f32] {
            if p >= n_past {
                &v[(p - n_past) * kv_dim..]
            } else {
                &v_hist[hist_row(p)..]
            }
        };

        let scale = 1.0 / (hd as f32).sqrt();
        let group = c.n_head / c.n_head_kv;
        let mut out = vec![0.0f32; n * c.n_head * hd];
        out.par_chunks_mut(hd).enumerate().for_each(|(i, o)| {
            let (t, head) = (i / c.n_head, i % c.n_head);
            let kvh = head / group;
            let n_ctx = n_past + t + 1; // causal
            let lo = a.window.map_or(0, |w| n_ctx.saturating_sub(w));
            let qh = &q[i * hd..(i + 1) * hd];
            let mut scores: Vec<f32> = (lo..n_ctx)
                .map(|p| quant::dot_f32(qh, &k_at(p)[kvh * hd..][..hd]) * scale)
                .collect();
            match &a.sinks {
                Some(s) => ops::softmax_with_sink(&mut scores, s[head]),
                None => ops::softmax(&mut scores),
            }
            o.fill(0.0);
            for (p, &w) in (lo..n_ctx).zip(&scores) {
                quant::axpy(o, w, &v_at(p)[kvh * hd..][..hd]);
            }
            if let Some(gate) = &gate {
                for (o, g) in o.iter_mut().zip(&gate[i * hd..(i + 1) * hd]) {
                    *o *= ops::sigmoid(*g);
                }
            }
        });

        // the pass's rows into the cache: appended, or the last `ring` into the ring
        match cache {
            Cache::Full { k: kc, v: vc } => {
                // within the capacity reserved for the context
                debug_assert!(kc.len() + k.len() <= kc.capacity());
                kc.extend_from_slice(&k);
                vc.extend_from_slice(&v);
            }
            Cache::Ring { k: kc, v: vc, ring } => {
                for t in n.saturating_sub(ring)..n {
                    let at = ((n_past + t) % ring) * kv_dim;
                    kc[at..at + kv_dim].copy_from_slice(&k[t * kv_dim..(t + 1) * kv_dim]);
                    vc[at..at + kv_dim].copy_from_slice(&v[t * kv_dim..(t + 1) * kv_dim]);
                }
            }
        }

        matmul_new(&a.wo, &Input::new(&out, n))
    }

    /// With `saved`, also the (conv, recurrent) states after each token but the last
    fn delta_net(
        &self,
        d: &DeltaNet<'_>,
        h: &[f32],
        n: usize,
        conv_state: &mut [f32],
        s: &mut [f32],
        saved: Option<&mut Vec<(Vec<f32>, Vec<f32>)>>,
    ) -> Vec<f32> {
        let c = &self.cfg;
        let (sd, hk, hv) = (c.ssm_state, c.ssm_k_heads, c.ssm_v_heads);
        let conv_dim = c.conv_dim();
        let kk = c.conv_kernel;

        let input = Input::new(h, n);
        let qkv = matmul_new(&d.wqkv, &input);
        let z = matmul_new(&d.wgate, &input);
        let beta: Vec<f32> = matmul_new(&d.wbeta, &input)
            .into_iter()
            .map(ops::sigmoid)
            .collect();
        let alpha = matmul_new(&d.walpha, &input);
        // per token, per value head: log of the state decay
        let g: Vec<f32> = alpha
            .iter()
            .enumerate()
            .map(|(i, &al)| ops::softplus(al + d.dt_bias[i % hv]) * d.a[i % hv])
            .collect();

        // causal depthwise conv over [previous kernel-1 inputs, new inputs], then silu
        let hist = kk - 1;
        let mut conv_out = vec![0.0f32; n * conv_dim];
        let at = |row: usize, ch: usize| {
            if row < hist {
                conv_state[row * conv_dim + ch]
            } else {
                qkv[(row - hist) * conv_dim + ch]
            }
        };
        conv_out
            .par_chunks_mut(conv_dim)
            .enumerate()
            .for_each(|(t, out)| {
                for (ch, o) in out.iter_mut().enumerate() {
                    let w = &d.conv1d[ch * kk..(ch + 1) * kk];
                    let mut acc = 0.0;
                    for (j, wj) in w.iter().enumerate() {
                        acc += wj * at(t + j, ch);
                    }
                    *o = ops::silu(acc);
                }
            });
        // keep the last kernel-1 inputs for the next call
        let window = |end: usize| -> Vec<f32> {
            (end..end + hist)
                .flat_map(|row| (0..conv_dim).map(move |ch| (row, ch)))
                .map(|(r, ch)| at(r, ch))
                .collect()
        };
        let save = saved.is_some();
        let conv_saved: Vec<Vec<f32>> = if save {
            (1..n).map(window).collect()
        } else {
            Vec::new()
        };
        let new_state = window(n);
        conv_state.copy_from_slice(&new_state);

        // split q | k | v and l2-normalize q, k per head
        let (qd, kd) = (hk * sd, hk * sd);
        let mut q = vec![0.0f32; n * qd];
        let mut k = vec![0.0f32; n * kd];
        let mut v = vec![0.0f32; n * hv * sd];
        for t in 0..n {
            let row = &conv_out[t * conv_dim..(t + 1) * conv_dim];
            q[t * qd..(t + 1) * qd].copy_from_slice(&row[..qd]);
            k[t * kd..(t + 1) * kd].copy_from_slice(&row[qd..qd + kd]);
            v[t * hv * sd..(t + 1) * hv * sd].copy_from_slice(&row[qd + kd..]);
        }
        ops::l2_norm(&mut q, sd, c.eps);
        ops::l2_norm(&mut k, sd, c.eps);

        // gated delta rule, one value head per task. The state of a head is
        // stored transposed: row j holds S[:, j] (contiguous over the key dim).
        // Value heads are in tiled order: key head = value head % key heads.
        let scale = 1.0 / (sd as f32).sqrt();
        let mut o = vec![0.0f32; n * hv * sd];
        let o_heads: Vec<(Vec<f32>, Vec<f32>)> = s
            .par_chunks_mut(sd * sd)
            .enumerate()
            .map(|(head, st)| {
                let kh = head % hk;
                let mut out = vec![0.0f32; n * sd];
                let mut delta = vec![0.0f32; sd];
                // this head's state after each token but the last
                let mut hist = Vec::with_capacity(if save { (n - 1) * sd * sd } else { 0 });
                for t in 0..n {
                    let qt = &q[t * qd + kh * sd..][..sd];
                    let kt = &k[t * kd + kh * sd..][..sd];
                    let vt = &v[(t * hv + head) * sd..][..sd];
                    let b = beta[t * hv + head];
                    let decay = g[t * hv + head].exp();
                    st.iter_mut().for_each(|x| *x *= decay);
                    for (j, dj) in delta.iter_mut().enumerate() {
                        *dj = (vt[j] - quant::dot_f32(&st[j * sd..(j + 1) * sd], kt)) * b;
                    }
                    for (j, &dj) in delta.iter().enumerate() {
                        quant::axpy(&mut st[j * sd..(j + 1) * sd], dj, kt);
                    }
                    let ot = &mut out[t * sd..(t + 1) * sd];
                    for (j, oj) in ot.iter_mut().enumerate() {
                        *oj = quant::dot_f32(&st[j * sd..(j + 1) * sd], qt) * scale;
                    }
                    if save && t + 1 < n {
                        hist.extend_from_slice(st);
                    }
                }
                (out, hist)
            })
            .collect();
        for (head, (oh, _)) in o_heads.iter().enumerate() {
            for t in 0..n {
                o[(t * hv + head) * sd..][..sd].copy_from_slice(&oh[t * sd..(t + 1) * sd]);
            }
        }
        if let Some(saved) = saved {
            saved.clear();
            for (t, conv) in conv_saved.into_iter().enumerate() {
                let s: Vec<f32> = o_heads
                    .iter()
                    .flat_map(|(_, hist)| &hist[t * sd * sd..(t + 1) * sd * sd])
                    .copied()
                    .collect();
                saved.push((conv, s));
            }
        }

        // gated rms norm: norm(o) * silu(z), per head
        ops::rms_norm(&mut o, &d.norm, c.eps);
        o.iter_mut().zip(&z).for_each(|(o, z)| *o *= ops::silu(*z));

        matmul_new(&d.wout, &Input::new(&o, n))
    }
}

/// `down(silu(gate x) * up x)`
fn dense_ffn(gate: &Weight<'_>, up: &Weight<'_>, down: &Weight<'_>, input: &Input<'_>) -> Vec<f32> {
    let mut g = matmul_new(gate, input);
    let u = matmul_new(up, input);
    g.par_iter_mut()
        .zip(u.par_iter())
        .for_each(|(g, u)| *g = ops::silu(*g) * u);
    matmul_new(down, &Input::new(&g, input.n))
}

enum LayerState {
    Attention {
        k: Vec<f32>,
        v: Vec<f32>,
    },
    /// a sliding window layer's caches: position `p` in row `p % ring`, the
    /// last `ring` positions held, see [`Config::cpu_ring`]
    Window {
        k: Vec<f32>,
        v: Vec<f32>,
        ring: usize,
    },
    /// `saved`: the states after each token but the last of a verify pass
    DeltaNet {
        conv: Vec<f32>,
        s: Vec<f32>,
        saved: Vec<(Vec<f32>, Vec<f32>)>,
    },
    /// a slot of `inkling2_loop`: its kv caches (`Attention` or `Window`),
    /// and the conv state `[taps - 1][channels]`; `saved`: the conv inputs
    /// of a verify pass, `[taps - 1 + n][channels]` (`inkling::short_conv`)
    Inkling {
        kv: Box<LayerState>,
        conv: Vec<f32>,
        saved: Vec<f32>,
    },
}

/// The caches an attention layer reads and writes in a pass
enum Cache<'c> {
    /// a row per position, those of the pass appended
    Full {
        k: &'c mut Vec<f32>,
        v: &'c mut Vec<f32>,
    },
    /// see [`LayerState::Window`]
    Ring {
        k: &'c mut [f32],
        v: &'c mut [f32],
        ring: usize,
    },
}

/// An empty vector able to hold `n` values without reallocating
fn reserved(n: usize) -> Result<Vec<f32>, Error> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| Error::OutOfMemory {
        bytes: n.saturating_mul(4),
    })?;
    Ok(v)
}

fn zeroed(n: usize) -> Result<Vec<f32>, Error> {
    let mut v = reserved(n)?;
    v.resize(n, 0.0);
    Ok(v)
}

struct MtpState {
    kv: LayerState,
    /// rows entered: row `p` pairs token `p + 1` with the hidden state of token `p`
    rows: usize,
    /// normalized hidden state of the last token, waiting for its successor
    last: Vec<f32>,
    /// the output head rows of the draft vocabulary
    head: Option<(DraftVocab, Vec<u8>)>,
}

/// Base rows of a draft vocabulary: the lowest token ids. On Qwen3.8 (248k
/// tokens), 64k rows hold ~90% of the tokens of answers in English, French
/// and code, and read in a quarter of the time of the whole head; 32k (86%)
/// and 96k (96%) drafted slower.
pub(crate) const DRAFT_BASE: usize = 65536;
/// Rows of a draft vocabulary, at most
pub(crate) const DRAFT_CAP: usize = DRAFT_BASE + 16384;
/// Recent tokens a draft vocabulary's coverage is measured over, at most and
/// at least
const DRAFT_WINDOW: usize = 128;
const DRAFT_MIN_WINDOW: usize = 16;

/// The tokens the MTP head drafts from: the output head's rows for the lowest
/// token ids (BPE merge order, which roughly follows frequency), then for each
/// other token of the sequence as it comes. Reading these rows takes a few
/// percent of the time of the whole head, and a token outside them only costs
/// a rejection, but while too many recent tokens fall outside (another
/// language than most of the base's, say), drafts use the whole head.
pub(crate) struct DraftVocab {
    /// head row -> token
    ids: Vec<TokenId>,
    /// token -> whether it has a row
    has: Vec<bool>,
    /// whether each recent token had a row when it came, and how many did
    recent: VecDeque<bool>,
    hits: usize,
}

impl DraftVocab {
    /// None when the vocabulary is small enough to draft from all of it
    pub(crate) fn new(n_vocab: usize) -> Option<Self> {
        if n_vocab <= DRAFT_CAP {
            return None;
        }
        let mut has = vec![false; n_vocab];
        has[..DRAFT_BASE].fill(true);
        Some(DraftVocab {
            ids: (0..DRAFT_BASE as TokenId).collect(),
            has,
            recent: VecDeque::with_capacity(DRAFT_WINDOW),
            hits: 0,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.ids.len()
    }

    /// The tokens of the rows `rows`
    pub(crate) fn tokens(&self, rows: core::ops::Range<usize>) -> &[TokenId] {
        &self.ids[rows]
    }

    /// Whether drafts should use the whole output head: when fewer than 90%
    /// of the recent tokens had a row
    pub(crate) fn full(&self) -> bool {
        self.recent.len() >= DRAFT_MIN_WINDOW && 10 * self.hits < 9 * self.recent.len()
    }

    /// Give rows to the tokens of a pass that have none, returning the rows
    /// added. The first `counted` count toward the coverage: the others of a
    /// verify pass are drafts, which have rows.
    pub(crate) fn observe(
        &mut self,
        tokens: &[TokenId],
        counted: usize,
    ) -> core::ops::Range<usize> {
        let start = self.ids.len();
        for (i, &t) in tokens.iter().enumerate() {
            let has = self.has[t as usize];
            if i < counted {
                if self.recent.len() == DRAFT_WINDOW && self.recent.pop_front() == Some(true) {
                    self.hits -= 1;
                }
                self.recent.push_back(has);
                self.hits += has as usize;
            }
            if !has && self.ids.len() < DRAFT_CAP {
                self.has[t as usize] = true;
                self.ids.push(t);
            }
        }
        start..self.ids.len()
    }
}

/// Copy the output head rows of `tokens` to `dst`, one after the other
pub(crate) fn copy_head_rows(output: &Weight<'_>, tokens: &[TokenId], dst: &mut [u8]) {
    let rb = output.data.len() / output.rows;
    for (&t, d) in tokens.iter().zip(dst.chunks_exact_mut(rb)) {
        d.copy_from_slice(&output.data[t as usize * rb..(t as usize + 1) * rb]);
    }
}

/// The last verify pass, until [`Model::keep`]
struct Verify {
    pos0: usize,
    /// normalized hidden states of its tokens
    hidden: Vec<f32>,
}

/// The sequence processed so far: kv cache and recurrent states
pub struct State {
    layers: Vec<LayerState>,
    pub n_past: usize,
    /// tokens the state can hold
    pub context: usize,
    /// tokens a verify pass takes after the chosen one, at most
    draft: usize,
    /// floats per position in the kv caches
    kv_dim: usize,
    /// positions of a sliding window
    window: usize,
    mtp: Option<MtpState>,
    verify: Option<Verify>,
}

impl State {
    /// The memory the state holds, measured, but
    /// [`SessionMemory::transient`]: what its passes allocate
    pub fn memory(&self) -> SessionMemory {
        let mut m = SessionMemory::default();
        for ls in self.layers.iter().chain(self.mtp.as_ref().map(|m| &m.kv)) {
            // a slot of `inkling2_loop`: its conv state, and its caches as a layer's
            let ls = match ls {
                LayerState::Inkling { kv, conv, .. } => {
                    m.recurrent += 4 * conv.len();
                    &**kv
                }
                ls => ls,
            };
            match ls {
                LayerState::Attention { k, v } => {
                    m.kv += 4 * (k.capacity() + v.capacity());
                    m.kv_per_token += 2 * 4 * self.kv_dim;
                }
                // a fixed size, as a recurrent state
                LayerState::Window { k, v, .. } => m.recurrent += 4 * (k.len() + v.len()),
                LayerState::DeltaNet { conv, s, .. } => m.recurrent += 4 * (conv.len() + s.len()),
                LayerState::Inkling { .. } => unreachable!("a slot in a slot"),
            }
        }
        if let Some(mtp) = &self.mtp {
            m.scratch += 4 * mtp.last.len();
            m.draft_head = mtp.head.as_ref().map_or(0, |(_, rows)| rows.capacity());
        }
        m
    }

    /// A copy of the state: the kv caches of the `n_past` tokens, the delta net states
    pub fn snapshot(&self) -> Result<StateSnapshot, Error> {
        assert!(self.verify.is_none(), "snapshot between verify and keep");
        let layers = self
            .layers
            .iter()
            .map(|ls| ls.snapshot(self.n_past, self.kv_dim, self.window))
            .collect::<Result<_, _>>()?;
        let mtp = match &self.mtp {
            Some(m) => Some(MtpSnapshot {
                kv: m.kv.snapshot(m.rows, self.kv_dim, self.window)?,
                rows: m.rows,
                last: copied(&m.last)?,
            }),
            None => None,
        };
        Ok(StateSnapshot {
            n_past: self.n_past,
            layers,
            mtp,
        })
    }

    /// Replace the state with `snap`, from a state of the same model. Nothing
    /// changes when it doesn't fit.
    pub fn restore(&mut self, snap: &StateSnapshot) -> Result<(), Error> {
        if snap.n_past > self.context {
            return Err(Error::Snapshot("more tokens than the context holds"));
        }
        let (n_past, kv_dim, window) = (snap.n_past, self.kv_dim, self.window);
        let fits = self.layers.len() == snap.layers.len()
            && self
                .layers
                .iter()
                .zip(&snap.layers)
                .all(|(ls, s)| ls.fits(s, n_past, kv_dim, window))
            && match (&self.mtp, &snap.mtp) {
                (Some(m), Some(s)) => {
                    m.kv.fits(&s.kv, s.rows, kv_dim, window) && m.last.len() == s.last.len()
                }
                (Some(_), None) => {
                    return Err(Error::Snapshot("no MTP history, for a drafting session"));
                }
                (None, _) => true,
            };
        if !fits {
            return Err(Error::Snapshot("taken from another model"));
        }
        for (ls, s) in self.layers.iter_mut().zip(&snap.layers) {
            ls.restore(s, n_past, kv_dim);
        }
        if let (Some(m), Some(s)) = (&mut self.mtp, &snap.mtp) {
            m.kv.restore(&s.kv, s.rows, kv_dim);
            m.rows = s.rows;
            m.last.copy_from_slice(&s.last);
        }
        self.n_past = snap.n_past;
        self.verify = None;
        Ok(())
    }
}

/// See [`State::snapshot`]
pub struct StateSnapshot {
    n_past: usize,
    layers: Vec<LayerSnapshot>,
    mtp: Option<MtpSnapshot>,
}

impl StateSnapshot {
    pub fn bytes(&self) -> usize {
        let mtp = self
            .mtp
            .as_ref()
            .map_or(0, |m| m.kv.bytes() + 4 * m.last.len());
        self.layers.iter().map(LayerSnapshot::bytes).sum::<usize>() + mtp
    }
}

enum LayerSnapshot {
    Attention {
        k: Vec<f32>,
        v: Vec<f32>,
    },
    /// the rows of the positions a next token would see, in order
    Window {
        k: Vec<f32>,
        v: Vec<f32>,
    },
    DeltaNet {
        conv: Vec<f32>,
        s: Vec<f32>,
    },
    Inkling {
        kv: Box<LayerSnapshot>,
        conv: Vec<f32>,
    },
}

impl LayerSnapshot {
    fn bytes(&self) -> usize {
        match self {
            LayerSnapshot::Attention { k, v } | LayerSnapshot::Window { k, v } => {
                4 * (k.len() + v.len())
            }
            LayerSnapshot::DeltaNet { conv, s } => 4 * (conv.len() + s.len()),
            LayerSnapshot::Inkling { kv, conv } => kv.bytes() + 4 * conv.len(),
        }
    }
}

struct MtpSnapshot {
    kv: LayerSnapshot,
    rows: usize,
    last: Vec<f32>,
}

fn copied(v: &[f32]) -> Result<Vec<f32>, Error> {
    let mut c = reserved(v.len())?;
    c.extend_from_slice(v);
    Ok(c)
}

/// Positions of a sliding window a next token sees before its own, of the
/// `n_past` held
fn window_rows(n_past: usize, window: usize) -> usize {
    n_past.min(window.saturating_sub(1))
}

impl LayerState {
    /// The state after `n_past` tokens of `kv_dim` floats a row: the caches'
    /// first rows (they may hold more, left over from rejected tokens), or
    /// the rows a sliding window still sees
    fn snapshot(
        &self,
        n_past: usize,
        kv_dim: usize,
        window: usize,
    ) -> Result<LayerSnapshot, Error> {
        let kv = n_past * kv_dim;
        Ok(match self {
            LayerState::Attention { k, v } => LayerSnapshot::Attention {
                k: copied(&k[..kv])?,
                v: copied(&v[..kv])?,
            },
            LayerState::Window { k, v, ring } => {
                let first = n_past - window_rows(n_past, window);
                let rows = |c: &[f32]| -> Result<Vec<f32>, Error> {
                    let mut out = reserved((n_past - first) * kv_dim)?;
                    for p in first..n_past {
                        out.extend_from_slice(&c[(p % ring) * kv_dim..][..kv_dim]);
                    }
                    Ok(out)
                };
                LayerSnapshot::Window {
                    k: rows(k)?,
                    v: rows(v)?,
                }
            }
            LayerState::DeltaNet { conv, s, .. } => LayerSnapshot::DeltaNet {
                conv: copied(conv)?,
                s: copied(s)?,
            },
            LayerState::Inkling { kv, conv, .. } => LayerSnapshot::Inkling {
                kv: Box::new(kv.snapshot(n_past, kv_dim, window)?),
                conv: copied(conv)?,
            },
        })
    }

    /// Whether `snap` is a state of this layer after `n_past` tokens
    fn fits(&self, snap: &LayerSnapshot, n_past: usize, kv_dim: usize, window: usize) -> bool {
        match (self, snap) {
            (LayerState::Attention { k, .. }, LayerSnapshot::Attention { k: sk, .. }) => {
                let kv = n_past * kv_dim;
                sk.len() == kv && kv <= k.capacity()
            }
            (LayerState::Window { ring, .. }, LayerSnapshot::Window { k: sk, .. }) => {
                let rows = window_rows(n_past, window);
                sk.len() == rows * kv_dim && rows <= *ring
            }
            (LayerState::DeltaNet { conv, s, .. }, LayerSnapshot::DeltaNet { conv: sc, s: ss }) => {
                conv.len() == sc.len() && s.len() == ss.len()
            }
            (LayerState::Inkling { kv, conv, .. }, LayerSnapshot::Inkling { kv: sk, conv: sc }) => {
                conv.len() == sc.len() && kv.fits(sk, n_past, kv_dim, window)
            }
            _ => false,
        }
    }

    /// Copy `snap`, which [`LayerState::fits`]; within capacity, the caches don't reallocate
    fn restore(&mut self, snap: &LayerSnapshot, n_past: usize, kv_dim: usize) {
        match (self, snap) {
            (LayerState::Attention { k, v }, LayerSnapshot::Attention { k: sk, v: sv }) => {
                k.clear();
                k.extend_from_slice(sk);
                v.clear();
                v.extend_from_slice(sv);
            }
            (LayerState::Window { k, v, ring }, LayerSnapshot::Window { k: sk, v: sv }) => {
                let first = n_past - sk.len() / kv_dim;
                for (i, p) in (first..n_past).enumerate() {
                    let at = (p % *ring) * kv_dim;
                    k[at..at + kv_dim].copy_from_slice(&sk[i * kv_dim..(i + 1) * kv_dim]);
                    v[at..at + kv_dim].copy_from_slice(&sv[i * kv_dim..(i + 1) * kv_dim]);
                }
            }
            (
                LayerState::DeltaNet { conv, s, saved },
                LayerSnapshot::DeltaNet { conv: sc, s: ss },
            ) => {
                conv.copy_from_slice(sc);
                s.copy_from_slice(ss);
                saved.clear();
            }
            (
                LayerState::Inkling { kv, conv, saved },
                LayerSnapshot::Inkling { kv: sk, conv: sc },
            ) => {
                kv.restore(sk, n_past, kv_dim);
                conv.copy_from_slice(sc);
                saved.clear();
            }
            _ => unreachable!("checked by fits"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moe(k: usize, gating: Gating, norm_weights: bool) -> MoeConfig {
        MoeConfig {
            n_expert: 4,
            n_expert_used: k,
            n_ff_exp: 1,
            n_ff_shexp: 0,
            gating,
            norm_weights,
            weights_scale: 1.0,
            shared_gate: false,
        }
    }

    #[test]
    fn route_picks_and_weighs() {
        // probabilities 0.5, 0.25, 0.125, 0.125 (equal logits for experts 2, 3)
        let l = [2f32.ln() + 1.0, 1.0, 0.5f32.ln() + 1.0, 0.5f32.ln() + 1.0].map(|x| x - 1.0);
        let (ids, w) = route(&l, &moe(3, Gating::Softmax, true), None);
        // the lower expert first among equals
        assert_eq!(ids, [0, 1, 2]);
        let sum = 0.5 + 0.25 + 0.125;
        for (w, p) in w.iter().zip([0.5, 0.25, 0.125]) {
            assert!((w - p / sum).abs() < 1e-6, "{w} {}", p / sum);
        }
        let (ids, w) = route(&l, &moe(1, Gating::Softmax, true), None);
        assert_eq!((ids, w), (vec![0], vec![1.0]));
    }

    #[test]
    fn route_by_sigmoid_with_bias() {
        // sigmoid probabilities 0.5, 0.75, 0.25, 0.9
        let l = [0.0, 3f32.ln(), (1.0f32 / 3.0).ln(), 9f32.ln()];
        let p = [0.5, 0.75, 0.25, 0.9];
        let (ids, w) = route(&l, &moe(2, Gating::Sigmoid, false), None);
        assert_eq!(ids, [3, 1]);
        for (w, e) in w.iter().zip([0.9, 0.75]) {
            assert!((w - e).abs() < 1e-6, "{w} {e}");
        }
        // the bias changes the choice, not the weights
        let bias = [0.0, 0.0, 1.0, 0.0];
        let (ids, w) = route(&l, &moe(2, Gating::Sigmoid, true), Some(&bias));
        assert_eq!(ids, [2, 3]);
        for (w, i) in w.iter().zip([2, 3]) {
            let e = p[i] / (p[2] + p[3]);
            assert!((w - e).abs() < 1e-6, "{w} {e}");
        }
    }

    #[test]
    fn sink_takes_probability() {
        // a sink of the scores' logit: as a third score, dropped
        let mut x = [1.0f32, 1.0];
        ops::softmax_with_sink(&mut x, 1.0);
        assert!(x.iter().all(|&p| (p - 1.0 / 3.0).abs() < 1e-6), "{x:?}");
        // a very low sink takes nothing
        let mut y = [0.5f32, -0.25];
        let mut z = y;
        ops::softmax_with_sink(&mut y, -1e30);
        ops::softmax(&mut z);
        assert_eq!(y, z);
    }
}
