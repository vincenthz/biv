//! The decoder of the llama.cpp arch `inkling2_loop` (Inkling-2), ported from
//! llama.cpp `src/models/inkling2.cpp` and `inkling2-loop.cpp`.
//!
//! Every layer is attention without positional encoding, over a sliding
//! window or, every few layers, over all positions, with a learned relative
//! position bias and a sink logit per token and head; queries and keys are
//! clipped to an L2 norm, values RMS-normalized, and the keys, values and
//! attention output go through short causal convolutions with a residual
//! (plus taps computed from the input on the full attention layers). The
//! feed-forward is a mixture of experts with the SITU activation, whose output
//! goes through another short conv.
//!
//! The layers are in blocks: the first layer of a block starts a new
//! residual stream from its attention output, and the inputs of the
//! attentions, feed-forwards and output head are depth mixes, an attention of
//! learned queries over the normalized outputs of the earlier blocks and the
//! stream. The layers of the core (`loop.start..loop.end`) run `loop.count`
//! times, each pass from the prelude's output; the later passes mix messages
//! from the previous pass's blocks into their inputs through small low-rank
//! controllers. Each visit of a layer, a slot, has its own kv caches and conv
//! state: Inkling-2-Mini's 24 layers make 48 slots.

use super::{
    Cache, Config, Ffn, Gating, LayerState, Loader, Model, MoeConfig, State, Verify, reserved,
    route, zeroed,
};
use crate::Error;
use crate::engine::Pass;
use crate::gguf::{Gguf, Leaf, need, per_layer, tensor};
#[cfg(not(any(feature = "std", test)))]
use crate::math::Float;
use crate::memory::ModelMemory;
use crate::model::Moe;
use crate::ops::{self, Weight};
use crate::par::*;
use crate::prelude::*;
use crate::quant::{self, Q8Act};
use crate::tokenizer::TokenId;

mod cpu;
use cpu::Act;

/// Visits of the layers a model may have, llama.cpp's `LLAMA_MAX_LAYERS`
const MAX_SLOTS: usize = 512;

/// The eps of the weightless RMS norms of the depth mixes and controllers,
/// hard-coded by llama.cpp (the file's is the same f32 in Inkling-2-Mini)
pub(crate) const MIX_EPS: f32 = 1e-6;

/// The hyperparameters of `inkling2_loop` beyond [`Config`]'s
#[derive(Debug, Clone)]
pub struct InklingConfig {
    /// channels of the relative position bias, per head
    pub n_rel: usize,
    /// distances with a bias row in the full attention layers (farther ones
    /// have none); the sliding window layers' rows are the window's
    pub rel_extent: usize,
    /// L2 norm the queries and keys are clipped to
    pub qk_clip: f32,
    /// L2 norm the relative position channels are clipped to
    pub rel_clip: f32,
    /// the full attention layers scale their queries and relative position
    /// channels at position `pos` by `1 + alpha * ln(max(1, (pos + 1) / floor))`
    pub log_scale_floor: usize,
    pub log_scale_alpha: f32,
    /// the experts' SITU activation of gate `g` and up `u`: `beta *
    /// tanh(g / beta) * sigmoid(g) * linear_beta * tanh(u / linear_beta)`,
    /// the second factor only when `linear_beta` > 0
    pub situ_beta: f32,
    pub situ_linear_beta: f32,
    /// layers of a block
    pub block: usize,
    /// taps of the short convs, the current token's included
    pub conv: usize,
    /// `cap * tanh(logit / cap)`, none when 0
    pub softcap: f32,
    /// rows of the embeddings and output head, more than the vocabulary's
    pub vocab_padded: usize,
    /// the core layers, `loop_start..loop_end`, run `loop_count` times
    pub loop_start: usize,
    pub loop_end: usize,
    pub loop_count: usize,
    /// width of the controllers' bottleneck
    pub gate_rank: usize,
}

/// A visit of a layer: the slot of its caches and conv state, the physical
/// layer, and the pass of the core (0 outside it)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Visit {
    pub(crate) slot: usize,
    pub(crate) layer: usize,
    pub(crate) pass: usize,
}

impl InklingConfig {
    /// Slots of a model of `n_layer` physical layers: each layer once, and
    /// the core again for each pass after the first
    pub fn n_slot(&self, n_layer: usize) -> usize {
        n_layer + (self.loop_count - 1) * (self.loop_end - self.loop_start)
    }

    /// The physical layer of `slot`
    pub fn layer_of(&self, slot: usize, n_layer: usize) -> usize {
        if slot < n_layer {
            slot
        } else {
            self.loop_start + (slot - n_layer) % (self.loop_end - self.loop_start)
        }
    }

    /// The visits of a forward pass in order: the prelude, the passes of the
    /// core (the first in the layers' own slots, the others in slots
    /// `n_layer..`), the epilogue
    pub(crate) fn visits(&self, n_layer: usize) -> Vec<Visit> {
        let (start, end) = (self.loop_start, self.loop_end);
        let once = |i| Visit {
            slot: i,
            layer: i,
            pass: 0,
        };
        let mut v: Vec<Visit> = (0..start).map(once).collect();
        for pass in 0..self.loop_count {
            for layer in start..end {
                let slot = match pass {
                    0 => layer,
                    _ => n_layer + (pass - 1) * (end - start) + (layer - start),
                };
                v.push(Visit { slot, layer, pass });
            }
        }
        v.extend((end..n_layer).map(once));
        v
    }

    /// Channels of a slot's conv state: keys, values, attention output and
    /// feed-forward output
    pub fn conv_channels(&self, kv_dim: usize, n_embd: usize) -> usize {
        2 * kv_dim + 2 * n_embd
    }
}

/// Where the GPU backends keep the normalized blocks of the stream, regions
/// of a pass's rows in one buffer: the prelude's, the core's by the parity of
/// the pass (a pass reads the previous one's as its bank of messages), the
/// epilogue's, then the normalized stream and the previous pass's output
#[cfg_attr(
    not(any(all(feature = "metal", target_os = "macos"), feature = "cuda")),
    allow(dead_code)
)]
#[derive(Clone, Copy)]
pub(crate) struct StackPlan {
    pub(crate) prelude: usize,
    pub(crate) core: usize,
    pub(crate) epilogue: usize,
}

#[cfg_attr(
    not(any(all(feature = "metal", target_os = "macos"), feature = "cuda")),
    allow(dead_code)
)]
impl StackPlan {
    pub(crate) fn new(ink: &InklingConfig, n_layer: usize) -> Self {
        let blocks = |r: core::ops::Range<usize>| r.filter(|i| i.is_multiple_of(ink.block)).count();
        StackPlan {
            prelude: blocks(0..ink.loop_start),
            core: blocks(ink.loop_start..ink.loop_end),
            epilogue: blocks(ink.loop_end..n_layer),
        }
    }

    pub(crate) fn regions(&self) -> usize {
        self.prelude + 2 * self.core + self.epilogue + 2
    }

    /// The normalized stream of the visit
    pub(crate) fn cur(&self) -> usize {
        self.regions() - 2
    }

    /// The previous pass's normalized output
    pub(crate) fn bank_out(&self) -> usize {
        self.regions() - 1
    }

    /// The region of the `j`-th block pushed by a visit of pass `pass` of
    /// layer `layer`
    pub(crate) fn push(&self, ink: &InklingConfig, layer: usize, pass: usize, j: usize) -> usize {
        if layer < ink.loop_start {
            j
        } else if layer < ink.loop_end {
            self.prelude + (pass % 2) * self.core + j
        } else {
            self.prelude + 2 * self.core + j
        }
    }
}

impl Config {
    /// The config of an `inkling2_loop` GGUF, validated as llama.cpp does
    /// (`inkling2.cpp`, `inkling2-loop.cpp`), plus what this engine needs:
    /// the same head counts and expert sizes in every layer, one head size
    pub(super) fn inkling(g: &Gguf) -> Result<Config, Error> {
        let arch_name = "inkling2_loop";
        let q = &g.metadata.inkling2_loop;
        let bad = Error::UnsupportedModel;
        let u = |v: Option<u64>| v.map(|v| v as usize);
        let need = |v: Result<&u64, Error>| v.map(|&v| v as usize);
        let key = |k: &str| format!("{arch_name}.{k}");
        // a value of the file, wherever it is stored
        let get = |k: &str| -> Result<crate::gguf::Value, Error> {
            g.get(&key(k)).ok_or_else(|| Error::MissingKey(key(k)))
        };
        // an unsigned integer: missing when absent, unsupported when of
        // another type or negative
        let get_u = |k: &str| -> Result<usize, Error> {
            u64::from_value(&get(k)?)
                .map(|v| v as usize)
                .ok_or(bad("inkling2_loop: a loop.* value not an unsigned integer"))
        };

        let n_layer = need(need!(q, @arch_name, block_count))?;
        if u(q.nextn_predict_layers).unwrap_or(0) != 0 {
            return Err(bad("inkling2_loop with next-n layers"));
        }
        let n_embd = need(need!(q, @arch_name, embedding_length))?;
        // a per-layer value, one value or an array, the same for every layer;
        // missing when absent, unsupported when malformed
        let uniform = |k: &str| -> Result<usize, Error> {
            let v = get(k)?;
            let a = match per_layer(&v, n_layer) {
                Some(a) => a,
                None if v.as_array().is_some_and(|a| a.len() != n_layer) => {
                    return Err(bad(
                        "inkling2_loop: a per-layer array not of block_count layers",
                    ));
                }
                None => {
                    return Err(bad(
                        "inkling2_loop: a per-layer value not an unsigned integer or an array of them",
                    ));
                }
            };
            match a.split_first() {
                Some((&x, rest)) if rest.iter().all(|&y| y == x) => Ok(x as usize),
                _ => Err(bad(
                    "inkling2_loop: head counts or expert sizes that differ between layers",
                )),
            }
        };
        let n_head = uniform("attention.head_count")?;
        let n_head_kv = match g.contains(&key("attention.head_count_kv")) {
            true => uniform("attention.head_count_kv")?,
            false => n_head,
        };
        if n_head == 0 || n_head_kv == 0 || !n_head.is_multiple_of(n_head_kv) {
            return Err(bad("head_count not a multiple of head_count_kv"));
        }
        let head_dim = u(q.attention.key_length).unwrap_or(n_embd / n_head);
        let a = &q.attention;
        if [a.value_length, a.key_length_swa, a.value_length_swa]
            .iter()
            .any(|&l| u(l).unwrap_or(head_dim) != head_dim)
        {
            return Err(bad(
                "inkling2_loop: key and value lengths that differ, or differ in the sliding window layers",
            ));
        }
        let eps = *need!(q, @arch_name, attention.layer_norm_rms_epsilon)?;
        let swa = match &a.sliding_window_pattern {
            Some(p) if p.len() == n_layer => p.clone(),
            Some(_) => return Err(bad("sliding_window_pattern not of block_count layers")),
            None => {
                return Err(Error::MissingKey(key("attention.sliding_window_pattern")));
            }
        };
        let sliding_window = need(need!(q, @arch_name, attention.sliding_window))?;
        if u(q.rope.dimension_count).unwrap_or(0) != 0 {
            return Err(bad("inkling2_loop with rotary embeddings"));
        }

        let positive = |x: f32| x.is_finite() && x > 0.0;
        let ink = InklingConfig {
            n_rel: need(need!(q, @arch_name, attention.relative_length))?,
            rel_extent: need(need!(q, @arch_name, attention.relative_extent))?,
            qk_clip: *need!(q, @arch_name, attention.qk_clip)?,
            rel_clip: *need!(q, @arch_name, attention.relative_clip)?,
            log_scale_floor: need(need!(q, @arch_name, attention.log_scale_floor))?,
            log_scale_alpha: *need!(q, @arch_name, attention.log_scale_alpha)?,
            situ_beta: *need!(q, @arch_name, activation.situ_beta)?,
            situ_linear_beta: *need!(q, @arch_name, activation.situ_linear_beta)?,
            block: need(need!(q, @arch_name, attn_res.block_size))?,
            conv: need(need!(q, @arch_name, shortconv.l_cache))?,
            softcap: q.final_logit_softcapping.unwrap_or(0.0),
            vocab_padded: need(need!(q, @arch_name, vocab_size))?,
            loop_start: get_u("loop.start")?,
            loop_end: get_u("loop.end")?,
            loop_count: get_u("loop.count")?,
            gate_rank: get_u("loop.gate_rank")?,
        };
        let n_vocab = need(need!(q, @arch_name, vocab_size_unpadded))?;
        let varying = match g.get(&key("loop.varying")) {
            Some(v) => bool::from_value(&v).ok_or(bad("inkling2_loop: loop.varying not a bool"))?,
            None => false,
        };

        if ink.n_rel == 0
            || ink.rel_extent == 0
            || sliding_window == 0
            || ink.block == 0
            || ink.log_scale_floor == 0
        {
            return Err(bad(
                "inkling2_loop: a relative length or extent, sliding window, block size or log scale floor of 0",
            ));
        }
        if !(2..=64).contains(&ink.conv) {
            return Err(bad("shortconv.l_cache not in 2..=64"));
        }
        if n_vocab == 0 || n_vocab > ink.vocab_padded {
            return Err(bad("vocab_size_unpadded not in 1..=vocab_size"));
        }
        if ![
            eps,
            ink.qk_clip,
            ink.rel_clip,
            ink.situ_beta,
            ink.situ_linear_beta,
        ]
        .into_iter()
        .all(positive)
            || !(ink.log_scale_alpha.is_finite() && ink.log_scale_alpha >= 0.0)
            || !ink.softcap.is_finite()
            || ink.softcap < 0.0
        {
            return Err(bad(
                "inkling2_loop: an eps, clip, situ beta, log scale alpha or softcap out of range",
            ));
        }
        if ink.loop_count < if varying { 1 } else { 2 }
            || ink.gate_rank == 0
            || !(0 < ink.loop_start && ink.loop_start < ink.loop_end && ink.loop_end <= n_layer)
            || !ink.loop_start.is_multiple_of(ink.block)
            || !ink.loop_end.is_multiple_of(ink.block)
        {
            return Err(bad(
                "inkling2_loop: loop.start, end, count or gate_rank out of range",
            ));
        }
        // each visit of a layer has its caches: at most llama.cpp's
        // LLAMA_MAX_LAYERS slots
        let slots = (ink.loop_count - 1)
            .checked_mul(ink.loop_end - ink.loop_start)
            .and_then(|s| s.checked_add(n_layer));
        if slots.is_none_or(|s| s > MAX_SLOTS) {
            return Err(bad(
                "inkling2_loop: more than 512 slots (loop.count too large)",
            ));
        }
        if let Some(tokens) = &g.metadata.tokenizer.ggml.tokens
            && tokens.len() != n_vocab
        {
            return Err(bad(
                "vocab_size_unpadded is not the tokenizer's token count",
            ));
        }
        // the padded tables, the output head's when it has its own
        let rows = |info: Result<&crate::gguf::TensorInfo, Error>| -> Result<(), Error> {
            let info = info?;
            let expected = [n_embd as u64, ink.vocab_padded as u64];
            match info.dims == expected {
                true => Ok(()),
                false => Err(Error::TensorShape {
                    tensor: info.name.clone(),
                    dims: info.dims.clone(),
                    expected: expected.into(),
                }),
            }
        };
        rows(tensor!(g.tensors, token_embd.weight))?;
        if g.tensors.output.weight.is_some() {
            rows(tensor!(g.tensors, output.weight))?;
        }

        let n_expert = need(need!(q, @arch_name, expert_count))?;
        let n_expert_used = uniform("expert_used_count")?;
        if n_expert_used == 0 || n_expert_used > n_expert {
            return Err(bad("expert_used_count not in 1..=expert_count"));
        }
        let n_ff_exp = uniform("expert_feed_forward_length")?;
        if n_ff_exp == 0 {
            return Err(bad("expert_feed_forward_length of 0"));
        }
        // not in the file: llama.cpp's choices for the arch
        let moe = MoeConfig {
            n_expert,
            n_expert_used,
            n_ff_exp,
            n_ff_shexp: 0,
            gating: Gating::Sigmoid,
            norm_weights: true,
            weights_scale: q.expert_weights_scale.filter(|&s| s != 0.0).unwrap_or(1.0),
            shared_gate: false,
        };

        Ok(Config {
            arch: super::Arch::Inkling2Loop,
            n_layer,
            n_embd,
            n_ff: u(q.feed_forward_length).unwrap_or(0),
            n_vocab,
            n_head,
            n_head_kv,
            head_dim,
            n_rot: 0,
            rope_base: 10_000.0,
            eps,
            context_length: u(q.context_length).unwrap_or(4096),
            recurrent: vec![false; n_layer],
            swa,
            sliding_window,
            rope: vec![false; n_layer],
            attn_gate: false,
            attn_sinks: false,
            conv_kernel: 0,
            ssm_state: 0,
            ssm_k_heads: 0,
            ssm_v_heads: 0,
            moe: Some(moe),
            n_dense_lead: 0,
            inkling: Some(ink),
        })
    }
}

/// The controllers of a looped layer, mixing a message from the previous
/// pass into each of its inputs: queries, keys, values, feed-forward
pub(crate) struct Loop<'a> {
    /// `[4][n_embd]`: the queries of the messages' depth mix
    pub(crate) msg_query: Vec<f32>,
    /// `[rms(input) ‖ rms(message)]` to the bottleneck
    pub(crate) down: [Weight<'a>; 4],
    pub(crate) gain: [Weight<'a>; 4],
    pub(crate) forget: [Weight<'a>; 4],
}

/// A physical layer of `inkling2_loop`
pub(crate) struct InkLayer<'a> {
    pub(crate) swa: bool,
    /// `[n_q][n_embd]`: the depth mix queries of the attention inputs, one
    /// for all three on sliding window layers, one each on the others; none
    /// on the first layer, whose inputs are the stream
    pub(crate) attn_query: Option<Vec<f32>>,
    /// the norms of the attention inputs: one for all on sliding window
    /// layers, those of the queries, keys and values on the others
    pub(crate) norms: Vec<Vec<f32>>,
    pub(crate) wq: Weight<'a>,
    pub(crate) wk: Weight<'a>,
    pub(crate) wv: Weight<'a>,
    pub(crate) wo: Weight<'a>,
    /// the sigmoid output gate, per channel of the attention output
    pub(crate) wgate: Weight<'a>,
    /// the sink logit of each head
    pub(crate) wsink: Weight<'a>,
    /// the relative position channels `[n_head][n_rel]`
    pub(crate) wrel: Weight<'a>,
    /// `[n_head][head_dim]`, `[n_head][n_rel]`: scales before the clips
    pub(crate) q_clip: Vec<f32>,
    pub(crate) rel_clip: Vec<f32>,
    /// `[distance][n_rel]`: the bias of each distance, against the
    /// relative position channels
    pub(crate) rel_bias: Vec<f32>,
    /// `[channel][tap]`: the static taps of the short convs, the last on the
    /// current token
    pub(crate) k_conv: Vec<f32>,
    pub(crate) v_conv: Vec<f32>,
    pub(crate) post_conv: Vec<f32>,
    pub(crate) ffn_post_conv: Vec<f32>,
    /// full attention layers: the dynamic taps of the keys' and values'
    /// convs, `[n_head_kv][tap]` with tap 0 on the current token
    pub(crate) k_dyn: Option<Weight<'a>>,
    pub(crate) v_dyn: Option<Weight<'a>>,
    /// the depth mix query of the feed-forward input
    pub(crate) ffn_query: Vec<f32>,
    pub(crate) ffn_norm: Vec<f32>,
    pub(crate) moe: Box<Moe<'a>>,
    /// the core layers' controllers
    pub(crate) looped: Option<Loop<'a>>,
}

/// The layers of `inkling2_loop` and the vectors around them
pub(crate) struct Inkling<'a> {
    /// the norm of the embeddings
    pub(crate) tok_norm: Vec<f32>,
    /// the depth mix query before the output norm
    pub(crate) output_query: Vec<f32>,
    /// physical layers
    pub(crate) layers: Vec<InkLayer<'a>>,
    /// the visits of a pass, see [`InklingConfig::visits`]
    pub(crate) visits: Vec<Visit>,
}

impl<'g> Loader<'g> {
    fn ink_layer(
        &self,
        cfg: &Config,
        ink: &InklingConfig,
        i: usize,
    ) -> Result<InkLayer<'g>, Error> {
        let l = self;
        let t = &self.g.tensors;
        let (e, nh) = (cfg.n_embd, cfg.n_head);
        let (kv_dim, qd, k) = (cfg.kv_dim(), cfg.n_head * cfg.head_dim, ink.conv);
        let swa = cfg.swa[i];
        let ext = if swa {
            cfg.sliding_window
        } else {
            ink.rel_extent
        };
        let n_q = if swa { 1 } else { 3 };
        let attn_query = match i {
            0 => None,
            _ => Some(l.vec(tensor!(t, blk[i].attn_res_query.weight), n_q * e)?),
        };
        let norms = if swa {
            vec![l.vec(tensor!(t, blk[i].attn_norm.weight), e)?]
        } else {
            vec![
                l.vec(tensor!(t, blk[i].attn_q_inp_norm.weight), e)?,
                l.vec(tensor!(t, blk[i].attn_k_inp_norm.weight), e)?,
                l.vec(tensor!(t, blk[i].attn_v_inp_norm.weight), e)?,
            ]
        };
        let (k_dyn, v_dyn) = if swa {
            (None, None)
        } else {
            let rows = cfg.n_head_kv * k;
            (
                Some(l.weight(tensor!(t, blk[i].attn_k_conv_proj.weight), e, rows)?),
                Some(l.weight(tensor!(t, blk[i].attn_v_conv_proj.weight), e, rows)?),
            )
        };
        let looped = if (ink.loop_start..ink.loop_end).contains(&i) && ink.loop_count > 1 {
            let r = ink.gate_rank;
            let ctrl = |name: &str, cols: usize, rows: usize| -> Result<Weight<'g>, Error> {
                let full = format!("blk.{i}.{name}.weight");
                let info = t.get(&full).ok_or(Error::MissingTensor(full));
                l.weight(info, cols, rows)
            };
            let each = |part: &str, cols: usize, rows: usize| -> Result<[Weight<'g>; 4], Error> {
                Ok([
                    ctrl(&format!("loop_q_ctrl_{part}"), cols, rows)?,
                    ctrl(&format!("loop_k_ctrl_{part}"), cols, rows)?,
                    ctrl(&format!("loop_v_ctrl_{part}"), cols, rows)?,
                    ctrl(&format!("loop_ffn_ctrl_{part}"), cols, rows)?,
                ])
            };
            Some(Loop {
                msg_query: l.vec(tensor!(t, blk[i].loop_msg_query.weight), 4 * e)?,
                down: each("down", 2 * e, r)?,
                gain: each("gain", r, e)?,
                forget: each("forget", r, e)?,
            })
        } else {
            None
        };
        let Ffn::Moe(moe) = l.ffn(cfg, i)? else {
            unreachable!("inkling2_loop has no dense feed-forward");
        };
        if moe.shared.is_some() {
            return Err(Error::UnsupportedModel(
                "inkling2_loop with a shared expert",
            ));
        }
        Ok(InkLayer {
            swa,
            attn_query,
            norms,
            wq: l.weight(tensor!(t, blk[i].attn_q.weight), e, qd)?,
            wk: l.weight(tensor!(t, blk[i].attn_k.weight), e, kv_dim)?,
            wv: l.weight(tensor!(t, blk[i].attn_v.weight), e, kv_dim)?,
            wo: l.weight(tensor!(t, blk[i].attn_output.weight), qd, e)?,
            wgate: l.weight(tensor!(t, blk[i].attn_gate.weight), e, qd)?,
            wsink: l.weight(tensor!(t, blk[i].attn_sink_proj.weight), e, nh)?,
            wrel: l.weight(tensor!(t, blk[i].attn_rel.weight), e, nh * ink.n_rel)?,
            q_clip: l.vec(tensor!(t, blk[i].attn_q_clip.weight), qd)?,
            rel_clip: l.vec(tensor!(t, blk[i].attn_rel_clip.weight), nh * ink.n_rel)?,
            rel_bias: l.vec(tensor!(t, blk[i].attn_rel_bias.weight), ext * ink.n_rel)?,
            k_conv: l.vec(tensor!(t, blk[i].attn_k_conv.weight), kv_dim * k)?,
            v_conv: l.vec(tensor!(t, blk[i].attn_v_conv.weight), kv_dim * k)?,
            post_conv: l.vec(tensor!(t, blk[i].attn_post_conv.weight), e * k)?,
            ffn_post_conv: l.vec(tensor!(t, blk[i].ffn_post_conv.weight), e * k)?,
            k_dyn,
            v_dyn,
            ffn_query: l.vec(tensor!(t, blk[i].ffn_res_query.weight), e)?,
            ffn_norm: l.vec(tensor!(t, blk[i].ffn_norm.weight), e)?,
            moe,
            looped,
        })
    }
}

/// The full attention layers' scale of the queries and relative position
/// channels at position `pos`: `1 + alpha * ln(max(1, (pos + 1) / floor))`,
/// exactly 1 up to `pos = floor - 1`
pub(crate) fn tau(pos: usize, floor: usize, alpha: f32) -> f32 {
    let s = (pos as f32 + 1.0) * (1.0 / floor as f32);
    s.max(1.0).ln() * alpha + 1.0
}

/// The SITU activation of an expert's gate `g` and up `u` values:
/// `beta * tanh(g / beta) * sigmoid(g) * lb * tanh(u / lb)`, the second
/// factor only when `lb` > 0. Like ggml, multiplies by the reciprocals.
pub(crate) fn situ(g: f32, u: f32, beta: f32, lb: f32) -> f32 {
    let act = ((g * (1.0 / beta)).tanh() * beta) * ops::sigmoid(g);
    let u = if lb > 0.0 {
        (u * (1.0 / lb)).tanh() * lb
    } else {
        u
    };
    act * u
}

/// From this many rows (tokens), the loops over rows run in parallel; below,
/// on the current thread, their work being too small to pay for a dispatch
const PAR_ROWS: usize = 4;

/// Positions of a pass's queries (tokens times the positions each sees)
/// from which the heads of an attention run in parallel
const ATTN_PAR_POSITIONS: usize = 64;

/// `f(t, row)` on each row `t` of `width` values of `out`, in parallel from
/// [`PAR_ROWS`] rows
fn each_row(out: &mut [f32], width: usize, f: impl Fn(usize, &mut [f32]) + Sync + Send) {
    if out.len() / width >= PAR_ROWS {
        out.par_chunks_mut(width)
            .enumerate()
            .for_each(|(t, o)| f(t, o));
    } else {
        out.chunks_mut(width).enumerate().for_each(|(t, o)| f(t, o));
    }
}

/// The depth mix of `states` (each `n` rows of `e`, normalized) by each
/// query of `query` (`[n_q][e]`): per token and query, the states weighed by
/// the softmax of their dot products with the query over `sqrt(e)`. Returns
/// the `n` rows of each query.
pub(crate) fn mix(states: &[&[f32]], query: &[f32], e: usize, n: usize) -> Vec<Vec<f32>> {
    let scale = 1.0 / (e as f32).sqrt();
    query
        .chunks_exact(e)
        .map(|q| {
            let mut out = vec![0.0f32; n * e];
            each_row(&mut out, e, |t, o| {
                let mut p: Vec<f32> = states
                    .iter()
                    .map(|s| quant::dot_f32(&s[t * e..(t + 1) * e], q) * scale)
                    .collect();
                ops::softmax(&mut p);
                for (s, &p) in states.iter().zip(&p) {
                    quant::axpy(o, p, &s[t * e..(t + 1) * e]);
                }
            });
            out
        })
        .collect()
}

/// A copy of `x` normalized without weight, per row of `e`
fn normed(x: &[f32], e: usize) -> Vec<f32> {
    let mut y = x.to_vec();
    ops::rms(&mut y, e, MIX_EPS);
    y
}

/// Mix the message `msg` into the input `cur` (each `n` rows of `e`) with
/// controller `c` of `lp`: from `z = down · [rms(cur) ‖ rms(msg)]`,
/// `cur - tanh(forget · z) * cur + tanh(gain · z) * msg`
pub(crate) fn control(
    lp: &Loop<'_>,
    c: usize,
    cur: &[f32],
    msg: &[f32],
    e: usize,
    n: usize,
) -> Vec<f32> {
    controls(lp, &[c], &[cur], &[msg], e, n).swap_remove(0)
}

/// [`control`] with each controller of `cs` on its input and message, in
/// two parallel loops for all: the bottlenecks, then the outputs
pub(crate) fn controls(
    lp: &Loop<'_>,
    cs: &[usize],
    cur: &[&[f32]],
    msg: &[&[f32]],
    e: usize,
    n: usize,
) -> Vec<Vec<f32>> {
    let us: Vec<Vec<f32>> = cur
        .iter()
        .zip(msg)
        .map(|(cur, msg)| {
            let mut u = vec![0.0f32; 2 * n * e];
            for (t, row) in u.chunks_exact_mut(2 * e).enumerate() {
                row[..e].copy_from_slice(&cur[t * e..(t + 1) * e]);
                row[e..].copy_from_slice(&msg[t * e..(t + 1) * e]);
            }
            // each half normalized on its own
            ops::rms(&mut u, e, MIX_EPS);
            u
        })
        .collect();
    let mut q: Vec<Q8Act> = cs.iter().map(|_| Q8Act::default()).collect();
    let us: Vec<Act<'_>> = us
        .iter()
        .zip(&mut q)
        .zip(cs)
        .map(|((u, q), &c)| Act::new(u, 2 * e, q, &[&lp.down[c]]))
        .collect();
    let jobs: Vec<_> = cs.iter().zip(&us).map(|(&c, u)| (&lp.down[c], u)).collect();
    let zs = cpu::matmuls(&jobs);
    let r = zs[0].len() / n;
    let mut q: Vec<Q8Act> = cs.iter().map(|_| Q8Act::default()).collect();
    let zs: Vec<Act<'_>> = zs
        .iter()
        .zip(&mut q)
        .zip(cs)
        .map(|((z, q), &c)| Act::new(z, r, q, &[&lp.gain[c], &lp.forget[c]]))
        .collect();
    let jobs: Vec<cpu::Gate<'_>> = (0..cs.len())
        .map(|j| cpu::Gate {
            gain: &lp.gain[cs[j]],
            forget: &lp.forget[cs[j]],
            z: &zs[j],
            cur: cur[j],
            msg: msg[j],
        })
        .collect();
    cpu::gates(&jobs)
}

/// The short causal conv of the `n` rows of `x` (`width` channels each),
/// with a residual: `(x + conv) / sqrt((dynamic ? 2 : 1) * taps + 1)`. Per
/// channel `ch`, `conv` sums its taps `w[ch]` (`[width][taps]`) over its
/// `taps - 1` previous inputs, oldest first, and the current one (the last
/// tap); with `dynamic`, `(taps, head)`, plus each token's own taps of the
/// head of `head` channels of `ch`, `taps[t][ch / head * n + lag]` on the
/// input `lag` tokens back. The previous inputs come from the rows of
/// `state` (`[taps - 1][channels]`, oldest first) at channel `offset`,
/// which keep the last ones for the next call. With `saved` (`[taps - 1 +
/// n][channels]`), the rows of `state` and then of `x` are copied there at
/// the same channels: the state after token `t` is its rows `t + 1..t +
/// taps`, see [`keep_conv`].
pub(crate) fn short_conv(
    x: &[f32],
    width: usize,
    w: &[f32],
    dynamic: Option<(&[f32], usize)>,
    state: &mut [f32],
    offset: usize,
    saved: Option<&mut [f32]>,
) -> Vec<f32> {
    let taps = w.len() / width;
    let hist = taps - 1;
    let channels = state.len() / hist;
    let n = x.len() / width;
    let srow = |r: usize| r * channels + offset..r * channels + offset + width;
    if let Some(s) = saved {
        for r in 0..hist {
            s[srow(r)].copy_from_slice(&state[srow(r)]);
        }
        for t in 0..n {
            s[srow(hist + t)].copy_from_slice(&x[t * width..(t + 1) * width]);
        }
    }
    let mult = if dynamic.is_some() { 2 } else { 1 };
    let scale = 1.0 / ((mult * taps + 1) as f32).sqrt();
    let mut out = vec![0.0f32; n * width];
    // the inputs of row `r` of the state, then of `x`
    let at = |r: usize| -> &[f32] {
        match r < hist {
            true => &state[srow(r)],
            false => &x[(r - hist) * width..(r - hist + 1) * width],
        }
    };
    let mut src: Vec<&[f32]> = Vec::with_capacity(taps);
    for (t, out) in out.chunks_exact_mut(width).enumerate() {
        src.clear();
        src.extend((0..taps).map(|j| at(t + j)));
        let xt = &x[t * width..(t + 1) * width];
        for (ch, o) in out.iter_mut().enumerate() {
            let wc = &w[ch * taps..(ch + 1) * taps];
            let mut acc = 0.0;
            for (wj, s) in wc.iter().zip(&src) {
                acc += wj * s[ch];
            }
            if let Some((d, head)) = dynamic {
                let row = d.len() / n;
                let d = &d[t * row + ch / head * taps..][..taps];
                for (lag, dl) in d.iter().enumerate() {
                    acc += src[hist - lag][ch] * dl;
                }
            }
            *o = (xt[ch] + acc) * scale;
        }
    }
    // the last inputs, rows `n..n + taps - 1`, each read before written
    for r in 0..hist {
        match n + r < hist {
            true => state.copy_within(srow(n + r), srow(r).start),
            false => {
                let t = n + r - hist;
                state[srow(r)].copy_from_slice(&x[t * width..(t + 1) * width]);
            }
        }
    }
    out
}

/// The conv state of a slot after `kept` of the `verified` tokens of a
/// verify pass, from its `saved` inputs (see [`short_conv`])
pub(crate) fn keep_conv(conv: &mut [f32], saved: &[f32], kept: usize, verified: usize) {
    let channels = (saved.len() - conv.len()) / verified;
    conv.copy_from_slice(&saved[kept * channels..kept * channels + conv.len()]);
}

/// The attention bias of a key `d` positions back for a head whose
/// relative position channels are `r`, minus its sink logit `sink` (which
/// makes the softmax's own sink logit 0): the row `d` of the `[distance][n_rel]`
/// `table` against `r`, nothing past its last row
pub(crate) fn rel_bias(table: &[f32], r: &[f32], d: usize, sink: f32) -> f32 {
    let n_rel = r.len();
    let b = match table.get(d * n_rel..(d + 1) * n_rel) {
        Some(row) => quant::dot_f32(row, r),
        None => 0.0,
    };
    b - sink
}

impl<'a> Model<'a> {
    pub(super) fn load_inkling(g: &'a Gguf, cfg: Config) -> Result<Self, Error> {
        let l = Loader { g };
        let ink = cfg.inkling.as_ref().expect("an inkling config");
        let e = cfg.n_embd;
        let t = &g.tensors;
        let tok_embd = l.weight(tensor!(t, token_embd.weight), e, ink.vocab_padded)?;
        let padded = if t.output.weight.is_some() {
            l.weight(tensor!(t, output.weight), e, ink.vocab_padded)?
        } else {
            tok_embd
        };
        // the rows of the vocabulary's tokens only
        let rb = padded.data.len() / padded.rows;
        let output = Weight::new(padded.ty, e, cfg.n_vocab, &padded.data[..cfg.n_vocab * rb]);
        let layers = (0..cfg.n_layer)
            .map(|i| l.ink_layer(&cfg, ink, i))
            .collect::<Result<_, _>>()?;
        let root = |name: &str| -> Result<Vec<f32>, Error> {
            let info = t.get(name).ok_or_else(|| Error::MissingTensor(name.into()));
            l.vec(info, e)
        };
        let inkling = Inkling {
            tok_norm: root("token_embd_norm.weight")?,
            output_query: root("output_res_query.weight")?,
            layers,
            visits: ink.visits(cfg.n_layer),
        };
        Ok(Model {
            output_norm: l.vec(tensor!(t, output_norm.weight), e)?,
            memory: ModelMemory::from_gguf(g, &cfg, false),
            cfg,
            tok_embd,
            output,
            layers: Vec::new(),
            mtp: None,
            inv_freq: Vec::new(),
            inkling: Some(Box::new(inkling)),
        })
    }

    /// See [`Model::new_state`]: a slot per visit, its caches (a ring on
    /// sliding window layers) and conv state
    pub(super) fn inkling_state(&self, context: usize, draft: usize) -> Result<State, Error> {
        let c = &self.cfg;
        let ink = c.inkling.as_ref().expect("an inkling config");
        let kv_dim = c.kv_dim();
        let ring = c.cpu_ring(context, draft);
        let conv = (ink.conv - 1) * ink.conv_channels(kv_dim, c.n_embd);
        let n_slot = ink.n_slot(c.n_layer);
        let mut layers = Vec::with_capacity(n_slot);
        for slot in 0..n_slot {
            let kv = if c.is_swa(ink.layer_of(slot, c.n_layer)) {
                LayerState::Window {
                    k: zeroed(ring * kv_dim)?,
                    v: zeroed(ring * kv_dim)?,
                    ring,
                }
            } else {
                let n = context.saturating_mul(kv_dim);
                LayerState::Attention {
                    k: reserved(n)?,
                    v: reserved(n)?,
                }
            };
            layers.push(LayerState::Inkling {
                kv: Box::new(kv),
                conv: zeroed(conv)?,
                saved: Vec::new(),
            });
        }
        let state = State {
            layers,
            n_past: 0,
            context,
            draft,
            kv_dim,
            window: c.sliding_window,
            mtp: None,
            verify: None,
        };
        debug_assert_eq!(
            state.memory().allocated(),
            self.shape().cpu_session(context, draft, 1).allocated()
        );
        Ok(state)
    }

    /// See [`Model::run`]. Runs on a thread of the pool: its thousands of
    /// small parallel loops per token then start from a worker, instead of
    /// each waking one from the caller's thread and waiting for it.
    pub(super) fn run_inkling(
        &self,
        model: &Inkling<'_>,
        state: &mut State,
        tokens: &[TokenId],
        pass: Pass,
    ) -> Vec<Vec<f32>> {
        // a pass of few tokens is many short parallel loops: the threads
        // wait for the next one busy rather than asleep, see `cpu::busy`
        #[cfg(feature = "rayon")]
        return match tokens.len() <= cpu::BUSY_TOKENS {
            true => cpu::busy(|| self.run_inkling_here(model, state, tokens, pass)),
            false => rayon::scope(|_| self.run_inkling_here(model, state, tokens, pass)),
        };
        #[cfg(not(feature = "rayon"))]
        self.run_inkling_here(model, state, tokens, pass)
    }

    fn run_inkling_here(
        &self,
        model: &Inkling<'_>,
        state: &mut State,
        tokens: &[TokenId],
        pass: Pass,
    ) -> Vec<Vec<f32>> {
        let c = &self.cfg;
        let ink = c.inkling.as_ref().expect("an inkling config");
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

        let mut x = vec![0.0f32; n * e];
        for (t, &tok) in tokens.iter().enumerate() {
            self.tok_embd
                .dequantize_row(tok as usize, &mut x[t * e..(t + 1) * e]);
        }
        ops::rms_norm(&mut x, &model.tok_norm, c.eps);
        let tau: Vec<f32> = (0..n)
            .map(|t| tau(pos0 + t, ink.log_scale_floor, ink.log_scale_alpha))
            .collect();
        let save = pass == Pass::Verify;

        // the blocks of the stream, normalized, by index in `stack`
        let mut stack: Vec<Vec<f32>> = Vec::new();
        let mut blocks: Vec<usize> = Vec::new();
        // the prelude's output and blocks, which each pass of the core starts from
        let mut prelude = (Vec::new(), Vec::new());
        // the previous pass's blocks after its first, and its normalized output
        let mut bank: (Vec<usize>, Vec<f32>) = (Vec::new(), Vec::new());
        for v in &model.visits {
            let layer = &model.layers[v.layer];
            if v.layer == ink.loop_start {
                if v.pass == 0 {
                    prelude = (x.clone(), blocks.clone());
                } else {
                    let first = prelude.1.len() + 1;
                    bank = (blocks[first..].to_vec(), normed(&x, e));
                    // the last pass takes the prelude's output
                    match v.pass + 1 == ink.loop_count {
                        true => x = core::mem::take(&mut prelude.0),
                        false => x.clone_from(&prelude.0),
                    }
                    blocks.clone_from(&prelude.1);
                }
            }
            let msgs = match (&layer.looped, v.pass) {
                (Some(lp), 1..) => {
                    let states: Vec<&[f32]> = bank
                        .0
                        .iter()
                        .rev()
                        .map(|&b| stack[b].as_slice())
                        .chain([bank.1.as_slice()])
                        .collect();
                    Some(mix(&states, &lp.msg_query, e, n))
                }
                _ => None,
            };
            let ls = &mut state.layers[v.slot];
            let step = Step {
                n,
                pos0,
                tau: &tau,
                save,
            };
            self.ink_layer(
                layer,
                ls,
                &mut x,
                v.layer,
                &mut blocks,
                &mut stack,
                msgs.as_deref(),
                &step,
            );
        }
        state.n_past += n;

        if pass == Pass::Prefill {
            return Vec::new();
        }
        // the rows whose logits are wanted: every one for a verify, else the last
        let t0 = if pass == Pass::Verify { 0 } else { n - 1 };
        let m = n - t0;
        let cur = normed(&x[t0 * e..], e);
        let states: Vec<&[f32]> = blocks
            .iter()
            .rev()
            .map(|&b| &stack[b][t0 * e..])
            .chain([cur.as_slice()])
            .collect();
        let mut hidden = mix(&states, &model.output_query, e, m).swap_remove(0);
        ops::rms_norm(&mut hidden, &self.output_norm, c.eps);
        let mut q = Q8Act::default();
        let hidden = Act::new(&hidden, e, &mut q, &[&self.output]);
        let [mut logits]: [Vec<f32>; 1] = cpu::matmuls(&[(&self.output, &hidden)])
            .try_into()
            .expect("a matmul");
        if ink.softcap != 0.0 {
            let cap = ink.softcap;
            logits
                .par_iter_mut()
                .for_each(|z| *z = (*z * (1.0 / cap)).tanh() * cap);
        }
        if pass == Pass::Verify {
            state.verify = Some(Verify {
                pos0,
                hidden: Vec::new(),
            });
        }
        logits
            .chunks_exact(c.n_vocab)
            .map(<[f32]>::to_vec)
            .collect()
    }

    /// One visit of `layer` (physical index `ip`) over the stream `x`, in
    /// the slot `ls`: pushes onto `blocks` the stream it ends, the first
    /// layer of a block; `msgs`, the previous pass's messages of a looped one
    #[allow(clippy::too_many_arguments)]
    fn ink_layer(
        &self,
        layer: &InkLayer<'_>,
        ls: &mut LayerState,
        x: &mut Vec<f32>,
        ip: usize,
        blocks: &mut Vec<usize>,
        normed_blocks: &mut Vec<Vec<f32>>,
        msgs: Option<&[Vec<f32>]>,
        step: &Step<'_>,
    ) {
        let c = &self.cfg;
        let ink = c.inkling.as_ref().expect("an inkling config");
        let LayerState::Inkling { kv, conv, saved } = ls else {
            unreachable!("layer / state kind mismatch");
        };
        let (n, pos0) = (step.n, step.pos0);
        let (e, hd, nh) = (c.n_embd, c.head_dim, c.n_head);
        let (kv_dim, n_rel) = (c.kv_dim(), ink.n_rel);
        let qd = nh * hd;

        // the attention inputs: depth mixes, the stream itself in the first layer
        let cur = normed(x, e);
        let mut inputs = match &layer.attn_query {
            Some(q) => {
                let states: Vec<&[f32]> = blocks
                    .iter()
                    .rev()
                    .map(|&b| normed_blocks[b].as_slice())
                    .chain([cur.as_slice()])
                    .collect();
                mix(&states, q, e, n)
            }
            None => vec![x.clone()],
        };
        // one input for all on a sliding window layer, unless the messages
        // make the queries', keys' and values' differ
        if !layer.swa || msgs.is_some() {
            while inputs.len() < 3 {
                inputs.push(inputs[0].clone());
            }
        }
        if let Some(msgs) = msgs {
            let lp = layer.looped.as_ref().expect("a looped layer");
            let cs: Vec<usize> = (0..inputs.len()).collect();
            let cur: Vec<&[f32]> = inputs.iter().map(Vec::as_slice).collect();
            let msgs: Vec<&[f32]> = msgs.iter().map(Vec::as_slice).collect();
            inputs = controls(lp, &cs, &cur, &msgs[..cs.len()], e, n);
        }
        for (j, input) in inputs.iter_mut().enumerate() {
            ops::rms_norm(input, &layer.norms[j.min(layer.norms.len() - 1)], c.eps);
        }
        let mut q8: Vec<Q8Act> = inputs.iter().map(|_| Q8Act::default()).collect();
        // every weight that reads an input: quantized when any needs it
        let mut readers = vec![
            &layer.wq,
            &layer.wk,
            &layer.wv,
            &layer.wrel,
            &layer.wsink,
            &layer.wgate,
        ];
        readers.extend(&layer.k_dyn);
        readers.extend(&layer.v_dyn);
        let inp: Vec<Act<'_>> = inputs
            .iter()
            .zip(&mut q8)
            .map(|(x, q)| Act::new(x, e, q, &readers))
            .collect();
        let (iq, ik, iv) = (
            &inp[0],
            &inp[1.min(inp.len() - 1)],
            &inp[2.min(inp.len() - 1)],
        );

        // the projections of the inputs, in one parallel loop
        let mut jobs = vec![
            (&layer.wq, iq),
            (&layer.wk, ik),
            (&layer.wv, iv),
            (&layer.wrel, iq),
            (&layer.wsink, iq),
            (&layer.wgate, iq),
        ];
        jobs.extend(layer.k_dyn.as_ref().map(|w| (w, ik)));
        jobs.extend(layer.v_dyn.as_ref().map(|w| (w, iv)));
        let mut outs = cpu::matmuls(&jobs).into_iter();
        let mut next = || outs.next().expect("a matmul per projection");
        let (mut q, kraw, vraw, mut r, sink, gate) =
            (next(), next(), next(), next(), next(), next());
        let k_dyn = layer.k_dyn.as_ref().map(|_| next());
        let v_dyn = layer.v_dyn.as_ref().map(|_| next());
        // each buffer freed once used: see `Shape::inkling_cpu_session`
        drop(jobs);
        drop(inp);
        drop(q8);
        drop(inputs);

        // the inputs of the convs of a verify, from which `keep` takes the
        // conv state after any of its tokens: the four convs below write all
        // their channels
        let mut sv = match step.save {
            true => {
                let mut sv = core::mem::take(saved);
                sv.clear();
                sv.resize(conv.len() / (ink.conv - 1) * (ink.conv - 1 + n), 0.0);
                Some(sv)
            }
            false => None,
        };
        let mut k = short_conv(
            &kraw,
            kv_dim,
            &layer.k_conv,
            k_dyn.as_deref().map(|d| (d, hd)),
            conv,
            0,
            sv.as_deref_mut(),
        );
        let mut v = short_conv(
            &vraw,
            kv_dim,
            &layer.v_conv,
            v_dyn.as_deref().map(|d| (d, hd)),
            conv,
            kv_dim,
            sv.as_deref_mut(),
        );
        drop((kraw, vraw, k_dyn, v_dyn));
        ops::l2_clip(&mut q, hd, Some(&layer.q_clip), ink.qk_clip);
        ops::l2_clip(&mut k, hd, None, ink.qk_clip);
        ops::rms(&mut v, hd, c.eps);
        ops::l2_clip(&mut r, n_rel, Some(&layer.rel_clip), ink.rel_clip);
        if !layer.swa {
            for (t, &s) in step.tau.iter().enumerate() {
                q[t * qd..(t + 1) * qd].iter_mut().for_each(|x| *x *= s);
                let rr = &mut r[t * nh * n_rel..(t + 1) * nh * n_rel];
                rr.iter_mut().for_each(|x| *x *= s);
            }
        }

        let cache = match &mut **kv {
            LayerState::Attention { k, v } => {
                // drop positions left over from rejected tokens
                k.truncate(pos0 * kv_dim);
                v.truncate(pos0 * kv_dim);
                Cache::Full { k, v }
            }
            LayerState::Window { k, v, ring } => Cache::Ring { k, v, ring: *ring },
            _ => unreachable!("an inkling slot holds kv caches"),
        };
        let window = layer.swa.then_some(c.sliding_window);
        let heads = Heads {
            q: &q,
            k: &k,
            v: &v,
            r: &r,
            sink: &sink,
            gate: &gate,
        };
        let a = self.ink_attention(layer, &heads, n, pos0, window, cache);
        drop((q, k, v, r, sink, gate));
        let mut qa = Q8Act::default();
        let a = Act::new(&a, qd, &mut qa, &[&layer.wo]);
        let [out]: [Vec<f32>; 1] = cpu::matmuls(&[(&layer.wo, &a)])
            .try_into()
            .expect("a matmul");
        drop(qa);
        let attn = short_conv(
            &out,
            e,
            &layer.post_conv,
            None,
            conv,
            2 * kv_dim,
            sv.as_deref_mut(),
        );
        drop(out);

        // the first layer of a block starts a new stream, and the earlier one joins the blocks
        if ip.is_multiple_of(ink.block) {
            normed_blocks.push(cur);
            blocks.push(normed_blocks.len() - 1);
            *x = attn;
        } else {
            drop(cur);
            ops::add_inplace(x, &attn);
        }

        let cur = normed(x, e);
        let states: Vec<&[f32]> = blocks
            .iter()
            .rev()
            .map(|&b| normed_blocks[b].as_slice())
            .chain([cur.as_slice()])
            .collect();
        let mut h = mix(&states, &layer.ffn_query, e, n).swap_remove(0);
        drop(states);
        drop(cur);
        if let Some(msgs) = msgs {
            let lp = layer.looped.as_ref().expect("a looped layer");
            h = control(lp, 3, &h, &msgs[3], e, n);
        }
        ops::rms_norm(&mut h, &layer.ffn_norm, c.eps);
        let out = self.ink_moe(&layer.moe, &h, n);
        drop(h);
        let y = short_conv(
            &out,
            e,
            &layer.ffn_post_conv,
            None,
            conv,
            2 * kv_dim + e,
            sv.as_deref_mut(),
        );
        drop(out);
        ops::add_inplace(x, &y);
        if let Some(sv) = sv {
            *saved = sv;
        }
    }

    /// The gated attention output of each token and head, `[n][n_head *
    /// head_dim]`, over the keys and values of the pass and of `cache`, which
    /// it then appends those of the pass to
    fn ink_attention(
        &self,
        layer: &InkLayer<'_>,
        heads: &Heads<'_>,
        n: usize,
        n_past: usize,
        window: Option<usize>,
        cache: Cache<'_>,
    ) -> Vec<f32> {
        let c = &self.cfg;
        let ink = c.inkling.as_ref().expect("an inkling config");
        let (hd, kv_dim, n_rel) = (c.head_dim, c.kv_dim(), ink.n_rel);
        let Heads {
            q,
            k,
            v,
            r,
            sink,
            gate,
        } = *heads;

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
        // the output of row `i`, a head of a token
        let head = |i: usize, o: &mut [f32]| {
            let (t, head) = (i / c.n_head, i % c.n_head);
            let kvh = head / group;
            let pos = n_past + t;
            let n_ctx = pos + 1; // causal
            let lo = window.map_or(0, |w| n_ctx.saturating_sub(w));
            let qh = &q[i * hd..(i + 1) * hd];
            let rh = &r[i * n_rel..(i + 1) * n_rel];
            let mut scores: Vec<f32> = (lo..n_ctx)
                .map(|p| {
                    let b = rel_bias(&layer.rel_bias, rh, pos - p, sink[i]);
                    quant::dot_f32(qh, &k_at(p)[kvh * hd..][..hd]) * scale + b
                })
                .collect();
            // the sink is in the bias: the softmax's own is a zero logit
            ops::softmax_with_sink(&mut scores, 0.0);
            o.fill(0.0);
            for (p, &w) in (lo..n_ctx).zip(&scores) {
                quant::axpy(o, w, &v_at(p)[kvh * hd..][..hd]);
            }
            for (o, g) in o.iter_mut().zip(&gate[i * hd..(i + 1) * hd]) {
                *o *= ops::sigmoid(*g);
            }
        };
        let mut out = vec![0.0f32; n * c.n_head * hd];
        // each head of each token on its own thread, but all on the current
        // one when their positions are few: see `each_row`
        let span = window.map_or(n_past + n, |w| w.min(n_past + n));
        let width = match n * span >= ATTN_PAR_POSITIONS {
            true => hd,
            false => out.len(),
        };
        each_row(&mut out, width, |j, o| {
            for (i, o) in o.chunks_exact_mut(hd).enumerate() {
                head(j * (width / hd) + i, o);
            }
        });

        // the pass's rows into the cache: appended, or the last `ring` into the ring
        match cache {
            Cache::Full { k: kc, v: vc } => {
                // within the capacity reserved for the context
                debug_assert!(kc.len() + k.len() <= kc.capacity());
                kc.extend_from_slice(k);
                vc.extend_from_slice(v);
            }
            Cache::Ring { k: kc, v: vc, ring } => {
                for t in n.saturating_sub(ring)..n {
                    let at = ((n_past + t) % ring) * kv_dim;
                    kc[at..at + kv_dim].copy_from_slice(&k[t * kv_dim..(t + 1) * kv_dim]);
                    vc[at..at + kv_dim].copy_from_slice(&v[t * kv_dim..(t + 1) * kv_dim]);
                }
            }
        }
        out
    }

    /// [`Model::moe`] with SITU experts and no shared expert
    fn ink_moe(&self, m: &Moe<'_>, h: &[f32], n: usize) -> Vec<f32> {
        let e = self.cfg.n_embd;
        let mc = self.cfg.moe.as_ref().expect("a moe config");
        let ink = self.cfg.inkling.as_ref().expect("an inkling config");
        // the input of the router and of every expert's gate and up
        let mut q = Q8Act::default();
        let (gate, up) = (m.gate.expert(0), m.up.expert(0));
        let input = Act::new(h, e, &mut q, &[&m.router, &gate, &up]);
        let [logits]: [Vec<f32>; 1] = cpu::matmuls(&[(&m.router, &input)])
            .try_into()
            .expect("a matmul");
        // the (token, weight) pairs of each expert. `route` divides by the
        // weights' sum clamped to 1/16384 where llama.cpp adds 1e-20: the
        // same unless the 8 chosen probabilities sum below 6.1e-5.
        let mut routed: Vec<Vec<(usize, f32)>> = vec![Vec::new(); m.gate.n];
        for (t, l) in logits.chunks_exact(m.gate.n).enumerate() {
            let (ids, w) = route(l, mc, m.bias.as_deref());
            for (i, w) in ids.into_iter().zip(w) {
                routed[i].push((t, w));
            }
        }
        // each chosen expert over its tokens, `down(situ(gate x, up x))`,
        // summed in the order of the experts
        let chosen: Vec<cpu::Chosen<'_>> = routed
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.is_empty())
            .map(|(i, p)| cpu::Chosen {
                gate: m.gate.expert(i),
                up: m.up.expert(i),
                down: m.down.expert(i),
                toks: p.iter().map(|&(t, _)| t).collect(),
                weights: p.iter().map(|&(_, w)| w).collect(),
            })
            .collect();
        debug_assert_eq!(input.n(), n);
        cpu::experts(&chosen, &input, ink.situ_beta, ink.situ_linear_beta)
    }
}

/// What a visit of a layer needs of its pass
struct Step<'s> {
    n: usize,
    pos0: usize,
    /// per token, see [`tau`]
    tau: &'s [f32],
    save: bool,
}

/// The per-token values of a layer's attention: queries, keys, values,
/// relative position channels, sink logits and output gate logits
#[derive(Clone, Copy)]
struct Heads<'h> {
    q: &'h [f32],
    k: &'h [f32],
    v: &'h [f32],
    r: &'h [f32],
    sink: &'h [f32],
    gate: &'h [f32],
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::GgmlType;

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    fn norm(x: &[f32]) -> f32 {
        x.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    /// Inkling-2-Mini's loop: 24 layers, core 6..18 run 3 times
    fn mini() -> InklingConfig {
        InklingConfig {
            n_rel: 16,
            rel_extent: 1024,
            qk_clip: 22.627417,
            rel_clip: 8.0,
            log_scale_floor: 8192,
            log_scale_alpha: 0.1,
            situ_beta: 4.0,
            situ_linear_beta: 25.0,
            block: 6,
            conv: 4,
            softcap: 30.0,
            vocab_padded: 201024,
            loop_start: 6,
            loop_end: 18,
            loop_count: 3,
            gate_rank: 64,
        }
    }

    #[test]
    fn visits_run_the_core_three_times() {
        let ink = mini();
        assert_eq!(ink.n_slot(24), 48);
        let v = ink.visits(24);
        assert_eq!(v.len(), 48);
        // (slot, physical layer), in visit order
        let mut want: Vec<(usize, usize)> = (0..18).map(|i| (i, i)).collect();
        want.extend((24..48).map(|s| (s, 6 + (s - 24) % 12)));
        want.extend((18..24).map(|i| (i, i)));
        let got: Vec<(usize, usize)> = v.iter().map(|v| (v.slot, v.layer)).collect();
        assert_eq!(got, want);
        // each slot once, and its layer is the one it is visited as
        let mut slots: Vec<usize> = v.iter().map(|v| v.slot).collect();
        slots.sort();
        assert_eq!(slots, (0..48).collect::<Vec<_>>());
        assert!(v.iter().all(|v| ink.layer_of(v.slot, 24) == v.layer));
        // the passes: the core's second and third visits only
        let pass = |s: usize| v.iter().find(|v| v.slot == s).unwrap().pass;
        assert_eq!([pass(5), pass(6), pass(17), pass(18)], [0, 0, 0, 0]);
        assert_eq!([pass(24), pass(35), pass(36), pass(47)], [1, 1, 2, 2]);
        // the slots of the full attention layers 5, 11, 17, 23 (SWA pattern
        // [T,T,T,T,T,F] x 4)
        let full: Vec<usize> = (0..48).filter(|&s| ink.layer_of(s, 24) % 6 == 5).collect();
        assert_eq!(full, [5, 11, 17, 23, 29, 35, 41, 47]);
    }

    #[test]
    fn stack_regions_of_the_blocks_are_distinct() {
        let ink = mini();
        let plan = StackPlan::new(&ink, 24);
        assert_eq!((plan.prelude, plan.core, plan.epilogue), (1, 2, 1));
        assert_eq!(plan.regions(), 8);
        // the blocks pushed by a pass, in visit order
        let pushes = |pass: usize| -> Vec<usize> {
            let mut j = 0;
            let mut out = Vec::new();
            for v in ink.visits(24).iter().filter(|v| v.pass == pass) {
                if v.layer.is_multiple_of(ink.block) {
                    let phase_start = matches!(v.layer, 0 | 6 | 18);
                    if phase_start {
                        j = 0;
                    }
                    out.push(plan.push(&ink, v.layer, v.pass, j));
                    j += 1;
                }
            }
            out
        };
        // the prelude's and the epilogue's (pass 0), the core's by parity
        assert_eq!(pushes(0), [0, 1, 2, 5]);
        assert_eq!(pushes(1), [3, 4]);
        assert_eq!(pushes(2), [1, 2]);
        // the stream and the bank's output after every block
        assert_eq!((plan.cur(), plan.bank_out()), (6, 7));
        // a pass's core blocks never overwrite the previous pass's (its bank)
        for pass in 1..ink.loop_count {
            let (now, before) = (pushes(pass), pushes(pass - 1));
            let before: Vec<usize> = before
                .into_iter()
                .filter(|&r| (1..5).contains(&r))
                .collect();
            assert!(now.iter().all(|r| !before.contains(r)), "pass {pass}");
        }
    }

    #[test]
    fn tau_starts_past_the_floor() {
        assert_eq!(tau(0, 8192, 0.1), 1.0);
        assert_eq!(tau(8190, 8192, 0.1), 1.0);
        // (8191 + 1) / 8192 is exactly 1
        assert_eq!(tau(8191, 8192, 0.1), 1.0);
        // 1 + 0.1 ln(1 + 2^-13) = 1 + 1.2206286e-5
        let t = tau(8192, 8192, 0.1);
        assert!(t > 1.0 && close(t, 1.0000122, 2e-7), "{t}");
        // 1 + 0.1 ln 2 at twice the floor
        assert!(close(tau(16383, 8192, 0.1), 1.0693147, 1e-6));
        // no scale without alpha
        assert_eq!(tau(100_000, 8192, 0.0), 1.0);
    }

    #[test]
    fn clip_scales_down_long_rows_only() {
        let thr = 22.627417f32;
        // a row of norm 10 is unchanged, one of norm 100 scaled to the threshold
        let mut x = vec![6.0, 8.0, 0.0, 0.0, 60.0, 80.0, 0.0, 0.0];
        ops::l2_clip(&mut x, 4, None, thr);
        assert!(close(x[0], 6.0, 1e-5) && close(x[1], 8.0, 1e-5));
        assert!(close(norm(&x[4..]), thr, 1e-5));
        assert!(close(x[4] / x[5], 0.75, 1e-6));
        // the scale applies before the clip, per channel, and repeats per row
        let mut x = vec![3.0, 4.0, 3.0, 4.0];
        ops::l2_clip(&mut x, 2, Some(&[10.0, 0.0]), 8.0);
        // [30, 0] has norm 30 > 8
        assert!(close(x[0], 8.0, 1e-5) && x[1] == 0.0);
        assert!(close(x[2], 8.0, 1e-5) && x[3] == 0.0);
        // a scale per row of the group
        let mut x = vec![1.0, 1.0, 1.0, 1.0];
        ops::l2_clip(&mut x, 2, Some(&[1.0, 2.0, 3.0, 4.0]), 100.0);
        assert_eq!(x, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rms_has_no_weight() {
        let mut x = vec![3.0, -3.0, 3.0, -3.0, 1.0, 1.0, 1.0, 1.0];
        ops::rms(&mut x, 4, 0.0);
        assert_eq!(x, [1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0]);
        let mut x = vec![2.0, 0.0];
        ops::rms(&mut x, 2, 1e-6);
        // 2 / sqrt(2 + 1e-6)
        assert!(close(x[0], 1.4142132, 1e-6) && x[1] == 0.0);
    }

    #[test]
    fn situ_saturates_its_factors() {
        let (beta, lb) = (4.0, 25.0);
        let f = |g: f64, u: f64| {
            let s = 1.0 / (1.0 + (-g).exp());
            (4.0 * (g / 4.0).tanh() * s) * (25.0 * (u / 25.0).tanh())
        };
        for (g, u) in [(1.0, 2.0), (-3.0, 0.5), (10.0, -40.0), (0.25, 100.0)] {
            let got = situ(g as f32, u as f32, beta, lb);
            let want = f(g, u) as f32;
            assert!(
                close(got, want, 1e-5 * want.abs().max(1.0)),
                "{g} {u}: {got} {want}"
            );
        }
        // the gate is the factor with the sigmoid: swapped halves differ
        assert!(!close(
            situ(1.0, -3.0, beta, lb),
            situ(-3.0, 1.0, beta, lb),
            1e-3
        ));
        // 4 tanh(0.25) sigmoid(1) * 25 tanh(0.08) = 0.7161996 * 1.9957448
        assert!(close(situ(1.0, 2.0, beta, lb), 1.4293511, 2e-6));
        // both factors saturate: the gate's at beta, the up's at 25
        assert!(close(situ(40.0, 1e4, beta, lb), 4.0 * 25.0, 1e-4));
        // without a linear beta, up is linear
        assert!(close(situ(1.0, 2.0, beta, 0.0), 1.4323991, 2e-6));
        assert_eq!(situ(0.0, 5.0, beta, lb), 0.0);
    }

    #[test]
    fn static_conv_taps_oldest_first() {
        // one channel, 3 previous inputs s0..s2 (oldest first), 2 new ones
        let x = [10.0, 20.0];
        let scale = 1.0 / 5f32.sqrt();
        let run = |w: [f32; 4]| {
            let mut state = vec![1.0, 2.0, 3.0];
            let y = short_conv(&x, 1, &w, None, &mut state, 0, None);
            (y, state)
        };
        // the last tap on the current token: y = (x + x) / sqrt(5)
        let (y, state) = run([0.0, 0.0, 0.0, 1.0]);
        assert!(close(y[0], 20.0 * scale, 1e-5) && close(y[1], 40.0 * scale, 1e-5));
        // the state holds the last 3 inputs, oldest first
        assert_eq!(state, [3.0, 10.0, 20.0]);
        // the first tap on the input 3 tokens back
        let (y, _) = run([1.0, 0.0, 0.0, 0.0]);
        assert!(close(y[0], (10.0 + 1.0) * scale, 1e-5));
        assert!(close(y[1], (20.0 + 2.0) * scale, 1e-5));
        // all taps
        let (y, _) = run([1.0, 2.0, 3.0, 4.0]);
        let want = 10.0 + (1.0 + 2.0 * 2.0 + 3.0 * 3.0 + 4.0 * 10.0);
        assert!(close(y[0], want * scale, 1e-4));
    }

    #[test]
    fn dynamic_conv_taps_lag_from_the_current_token() {
        // two channels of a head each; the dynamic taps of token t are
        // `[head][lag]`, lag 0 the current token
        let x = [10.0, 100.0];
        let w = [0.0; 8];
        let mut dyn_taps = [0.0f32; 8];
        dyn_taps[0] = 1.0; // head 0, lag 0
        dyn_taps[4 + 3] = 1.0; // head 1, lag 3
        let mut state = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [3][2]
        let y = short_conv(&x, 2, &w, Some((&dyn_taps, 1)), &mut state, 0, None);
        // 1 / sqrt(2 * 4 + 1) = 1/3
        assert!(close(y[0], (10.0 + 10.0) / 3.0, 1e-5));
        // the oldest input of channel 1
        assert!(close(y[1], (100.0 + 2.0) / 3.0, 1e-5));
        assert_eq!(state, [3.0, 4.0, 5.0, 6.0, 10.0, 100.0]);
    }

    #[test]
    fn conv_keeps_its_channels_and_saves_each_token() {
        // a conv of width 1 at channel 2 of a state of 3 channels
        let mut state = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]; // [3][3]
        let x = [10.0, 20.0, 30.0];
        // the state's rows then the inputs, [3 + 3][3]
        let mut saved = vec![0.0; 6 * 3];
        let w = [0.0, 0.0, 0.0, 1.0];
        short_conv(&x, 1, &w, None, &mut state, 2, Some(&mut saved));
        assert_eq!(state, [1.0, 2.0, 10.0, 4.0, 5.0, 20.0, 7.0, 8.0, 30.0]);
        let ch2 = |s: &[f32]| s.iter().skip(2).step_by(3).copied().collect::<Vec<f32>>();
        assert_eq!(ch2(&saved), [3.0, 6.0, 9.0, 10.0, 20.0, 30.0]);
        assert!(
            saved
                .iter()
                .enumerate()
                .all(|(i, &v)| i % 3 == 2 || v == 0.0)
        );
        // after token 0: [6, 9, 10]; after token 1: [9, 10, 20]; after all, the state
        let mut s = vec![0.0; 9];
        keep_conv(&mut s, &saved, 1, 3);
        assert_eq!(ch2(&s), [6.0, 9.0, 10.0]);
        keep_conv(&mut s, &saved, 3, 3);
        assert_eq!(ch2(&s), ch2(&state));
        keep_conv(&mut s, &saved, 2, 3);
        assert_eq!(ch2(&s), [9.0, 10.0, 20.0]);
        // one token at a time from a kept state continues the same way
        let mut s = vec![1.0, 2.0, 9.0, 4.0, 5.0, 10.0, 7.0, 8.0, 20.0];
        let y = short_conv(&x[2..], 1, &w, None, &mut s, 2, None);
        assert_eq!(s, state);
        assert!(close(y[0], 60.0 / 5f32.sqrt(), 1e-4));
        // fewer tokens than the state's rows shift it
        let mut s = vec![1.0, 2.0, 3.0];
        short_conv(&[4.0], 1, &w, None, &mut s, 0, None);
        assert_eq!(s, [2.0, 3.0, 4.0]);
    }

    #[test]
    fn mix_weighs_states_by_their_scores() {
        let e = 4;
        // one state: itself, whatever the query
        let s = [1.0, 2.0, 3.0, 4.0];
        let out = mix(&[&s], &[5.0, -1.0, 0.0, 2.0], e, 1);
        assert_eq!(out[0], s);
        // two states and two tokens; the query is 2 ln 3 on channel 0, and
        // the scores are over sqrt(4)
        let a = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let ln3 = 3f32.ln();
        let query = [0.0, 0.0, 0.0, 0.0, 2.0 * ln3, 0.0, 0.0, 0.0];
        let out = mix(&[&a, &b], &query, e, 2);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], [0.5, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5]);
        // token 0: scores 0 and ln 3 -> 1/4, 3/4; token 1: both ln 3
        assert!(close(out[1][0], 0.75, 1e-6));
        assert!(close(out[1][4], 1.0, 1e-6) && close(out[1][5], 0.5, 1e-6));
    }

    fn f32_weight(data: &[f32]) -> &[u8] {
        // SAFETY: f32 has no padding and any bytes are valid u8
        unsafe { core::slice::from_raw_parts(data.as_ptr().cast(), 4 * data.len()) }
    }

    #[test]
    fn control_gates_the_input_and_message() {
        let (e, r) = (32, 32);
        // down: z0 = rms(cur)[0], z1 = rms(msg)[0]
        let mut down = vec![0.0f32; r * 2 * e];
        down[0] = 1.0;
        down[2 * e + e] = 1.0;
        // gain = tanh(0.5 z1), forget = tanh(0.25 z0) in every channel
        let mut gain = vec![0.0f32; e * r];
        let mut forget = vec![0.0f32; e * r];
        for j in 0..e {
            gain[j * r + 1] = 0.5;
            forget[j * r] = 0.25;
        }
        fn w(d: &[f32], cols: usize, rows: usize) -> Weight<'_> {
            Weight::new(GgmlType::F32, cols, rows, f32_weight(d))
        }
        let zero = vec![0.0f32; 2 * e * r];
        let lp = Loop {
            msg_query: vec![0.0; 4 * e],
            down: [
                w(&down, 2 * e, r),
                w(&zero, 2 * e, r),
                w(&down, 2 * e, r),
                w(&down, 2 * e, r),
            ],
            gain: [w(&gain, r, e); 4],
            forget: [w(&forget, r, e); 4],
        };
        // constant rows normalize to 1 (cur 3) and -1 (msg -2)
        let cur = vec![3.0f32; e];
        let msg = vec![-2.0f32; e];
        let out = control(&lp, 0, &cur, &msg, e, 1);
        let (g, f) = ((-0.5f32).tanh(), 0.25f32.tanh());
        let want = (3.0 - f * 3.0) + g * -2.0;
        assert!(
            out.iter().all(|&o| close(o, want, 1e-4)),
            "{} {want}",
            out[0]
        );
        // controller 1 has a zero bottleneck: the input as is
        let out = control(&lp, 1, &cur, &msg, e, 1);
        assert!(out.iter().all(|&o| o == 3.0));
    }

    #[test]
    fn rel_bias_by_distance_then_nothing() {
        // 2 distances of 2 channels
        let table = [1.0, 2.0, 3.0, 4.0];
        let r = [10.0, 100.0];
        assert_eq!(rel_bias(&table, &r, 0, 0.0), 210.0);
        assert_eq!(rel_bias(&table, &r, 1, 5.0), 430.0 - 5.0);
        // past the table's rows: the zero row, the sink only
        assert_eq!(rel_bias(&table, &r, 2, 5.0), -5.0);
        assert_eq!(rel_bias(&table, &r, 5000, -1.0), 1.0);
    }

    #[test]
    fn sink_folded_into_the_bias() {
        // softmax(s - t) with a zero sink is exp(s) / (exp(t) + sum exp(s))
        let (s, t) = ([0.5f32, -1.0, 2.0], 1.5f32);
        let mut p: Vec<f32> = s.iter().map(|x| x - t).collect();
        ops::softmax_with_sink(&mut p, 0.0);
        let den: f32 = t.exp() + s.iter().map(|x| x.exp()).sum::<f32>();
        for (p, s) in p.iter().zip(s) {
            assert!(close(*p, s.exp() / den, 1e-6));
        }
        // a sink below every score still takes its share
        let mut p = vec![-10.0f32];
        ops::softmax_with_sink(&mut p, 0.0);
        assert!(close(p[0], 1.0 / (1.0 + 10f32.exp()), 1e-9));
    }

    #[test]
    fn route_sigmoid_biased_choice_unbiased_weights() {
        let mc = MoeConfig {
            n_expert: 4,
            n_expert_used: 2,
            n_ff_exp: 1,
            n_ff_shexp: 0,
            gating: Gating::Sigmoid,
            norm_weights: true,
            weights_scale: 8.0,
            shared_gate: false,
        };
        let logits = [2.0f32, 1.0, 0.0, -1.0];
        // the bias makes expert 3 the second choice: sigmoid(-1) + 0.5 =
        // 0.769 > sigmoid(1) = 0.731
        let bias = [0.0f32, 0.0, 0.0, 0.5];
        let (ids, w) = route(&logits, &mc, Some(&bias));
        assert_eq!(ids, [0, 3]);
        let p = |x: f32| 1.0 / (1.0 + (-x).exp());
        let sum = p(2.0) + p(-1.0);
        assert!(close(w[0], p(2.0) / sum * 8.0, 1e-5));
        assert!(close(w[1], p(-1.0) / sum * 8.0, 1e-5));
    }
}

/// A tiny `inkling2_loop` GGUF written in memory: its config, the vectors
/// and states the model allocates against the estimates, and the same
/// logits whatever the passes
#[cfg(test)]
mod tiny {
    use super::*;
    use crate::memory::{Backend, Footprint};

    /// A metadata value of the test file
    enum V {
        U32(u32),
        F32(f32),
        Bool(bool),
        Str(&'static str),
        I32s(Vec<i32>),
        Bools(Vec<bool>),
    }

    const E: usize = 32;
    const N_LAYER: usize = 6;
    const N_HEAD: usize = 2;
    const HD: usize = 16;
    const KV: usize = HD;
    const N_REL: usize = 4;
    const EXTENT: usize = 5;
    const WINDOW: usize = 3;
    const TAPS: usize = 4;
    const N_EXPERT: usize = 4;
    const FF: usize = 8;
    const RANK: usize = 4;
    const VOCAB: usize = 36;
    const PADDED: usize = 40;

    fn swa(i: usize) -> bool {
        i.is_multiple_of(2)
    }

    fn looped(i: usize) -> bool {
        (2..4).contains(&i)
    }

    /// 6 layers in blocks of 2, the full attention ones odd; the core 2..4
    /// runs twice: 8 slots
    fn keys() -> Vec<(String, V)> {
        let k = |s: &str| format!("inkling2_loop.{s}");
        vec![
            ("general.architecture".into(), V::Str("inkling2_loop")),
            (k("block_count"), V::U32(N_LAYER as u32)),
            (k("context_length"), V::U32(64)),
            (k("embedding_length"), V::U32(E as u32)),
            (
                k("attention.head_count"),
                V::I32s(vec![N_HEAD as i32; N_LAYER]),
            ),
            (k("attention.head_count_kv"), V::U32(1)),
            (k("attention.layer_norm_rms_epsilon"), V::F32(1e-6)),
            (k("attention.key_length"), V::U32(HD as u32)),
            (k("attention.value_length"), V::U32(HD as u32)),
            (k("attention.sliding_window"), V::U32(WINDOW as u32)),
            (
                k("attention.sliding_window_pattern"),
                V::Bools((0..N_LAYER).map(swa).collect()),
            ),
            (k("attention.relative_length"), V::U32(N_REL as u32)),
            (k("attention.relative_extent"), V::U32(EXTENT as u32)),
            (k("attention.qk_clip"), V::F32(4.0)),
            (k("attention.relative_clip"), V::F32(2.0)),
            // tau > 1 from position 4
            (k("attention.log_scale_floor"), V::U32(4)),
            (k("attention.log_scale_alpha"), V::F32(0.1)),
            (k("expert_count"), V::U32(N_EXPERT as u32)),
            (k("expert_used_count"), V::U32(2)),
            (k("expert_feed_forward_length"), V::U32(FF as u32)),
            (k("expert_weights_scale"), V::F32(8.0)),
            (k("vocab_size"), V::U32(PADDED as u32)),
            (k("vocab_size_unpadded"), V::U32(VOCAB as u32)),
            (k("shortconv.l_cache"), V::U32(TAPS as u32)),
            (k("activation.situ_beta"), V::F32(4.0)),
            (k("activation.situ_linear_beta"), V::F32(25.0)),
            (k("attn_res.block_size"), V::U32(2)),
            (k("final_logit_softcapping"), V::F32(30.0)),
            (k("loop.start"), V::U32(2)),
            (k("loop.end"), V::U32(4)),
            (k("loop.count"), V::U32(2)),
            (k("loop.gate_rank"), V::U32(RANK as u32)),
            (k("loop.varying"), V::Bool(false)),
        ]
    }

    /// The tensors, as `(name, dims)`, all F32
    fn tensors() -> Vec<(String, Vec<usize>)> {
        let mut t: Vec<(String, Vec<usize>)> = vec![
            ("token_embd.weight".into(), vec![E, PADDED]),
            ("token_embd_norm.weight".into(), vec![E]),
            ("output_norm.weight".into(), vec![E]),
            ("output_res_query.weight".into(), vec![E]),
        ];
        let qd = N_HEAD * HD;
        for i in 0..N_LAYER {
            let mut add = |name: &str, dims: Vec<usize>| t.push((format!("blk.{i}.{name}"), dims));
            add("attn_q.weight", vec![E, qd]);
            add("attn_k.weight", vec![E, KV]);
            add("attn_v.weight", vec![E, KV]);
            add("attn_output.weight", vec![qd, E]);
            add("attn_gate.weight", vec![E, qd]);
            add("attn_sink_proj.weight", vec![E, N_HEAD]);
            add("attn_rel.weight", vec![E, N_HEAD * N_REL]);
            add("attn_q_clip.weight", vec![HD, N_HEAD]);
            add("attn_rel_clip.weight", vec![N_REL, N_HEAD]);
            let ext = if swa(i) { WINDOW } else { EXTENT };
            add("attn_rel_bias.weight", vec![N_REL, ext]);
            add("attn_k_conv.weight", vec![TAPS, KV]);
            add("attn_v_conv.weight", vec![TAPS, KV]);
            add("attn_post_conv.weight", vec![TAPS, E]);
            add("ffn_post_conv.weight", vec![TAPS, E]);
            add("ffn_res_query.weight", vec![E]);
            add("ffn_norm.weight", vec![E]);
            add("exp_probs_b.bias", vec![N_EXPERT]);
            add("ffn_gate_inp.weight", vec![E, N_EXPERT]);
            add("ffn_gate_up_exps.weight", vec![E, 2 * FF, N_EXPERT]);
            add("ffn_down_exps.weight", vec![FF, E, N_EXPERT]);
            if swa(i) {
                add("attn_norm.weight", vec![E]);
            } else {
                for n in ["q", "k", "v"] {
                    add(&format!("attn_{n}_inp_norm.weight"), vec![E]);
                }
                add("attn_k_conv_proj.weight", vec![E, TAPS]);
                add("attn_v_conv_proj.weight", vec![E, TAPS]);
            }
            match (i, swa(i)) {
                (0, _) => {}
                (_, true) => add("attn_res_query.weight", vec![E]),
                (_, false) => add("attn_res_query.weight", vec![E, 3]),
            }
            if looped(i) {
                add("loop_msg_query.weight", vec![E, 4]);
                for c in ["q", "k", "v", "ffn"] {
                    add(&format!("loop_{c}_ctrl_down.weight"), vec![2 * E, RANK]);
                    add(&format!("loop_{c}_ctrl_gain.weight"), vec![RANK, E]);
                    add(&format!("loop_{c}_ctrl_forget.weight"), vec![RANK, E]);
                }
            }
        }
        t
    }

    /// A GGUF of `keys` and `tensors`, the values pseudo-random: norms
    /// around 1, the rest around 0
    fn write(keys: &[(String, V)], tensors: &[(String, Vec<usize>)]) -> Vec<u8> {
        write_typed(keys, tensors, |_| false)
    }

    /// [`write`], the tensors named by `q8` in Q8_0 (blocks of a scale of
    /// 1/64 and pseudo-random values)
    fn write_typed(
        keys: &[(String, V)],
        tensors: &[(String, Vec<usize>)],
        q8: impl Fn(&str) -> bool,
    ) -> Vec<u8> {
        let mut b = Vec::new();
        let u32_ = |b: &mut Vec<u8>, x: u32| b.extend(x.to_le_bytes());
        let u64_ = |b: &mut Vec<u8>, x: u64| b.extend(x.to_le_bytes());
        let str_ = |b: &mut Vec<u8>, s: &str| {
            b.extend((s.len() as u64).to_le_bytes());
            b.extend(s.as_bytes());
        };
        u32_(&mut b, 0x4655_4747);
        u32_(&mut b, 3);
        u64_(&mut b, tensors.len() as u64);
        u64_(&mut b, keys.len() as u64);
        for (k, v) in keys {
            str_(&mut b, k);
            match v {
                V::U32(x) => {
                    u32_(&mut b, 4);
                    u32_(&mut b, *x);
                }
                V::F32(x) => {
                    u32_(&mut b, 6);
                    u32_(&mut b, x.to_bits());
                }
                V::Bool(x) => {
                    u32_(&mut b, 7);
                    b.push(*x as u8);
                }
                V::Str(s) => {
                    u32_(&mut b, 8);
                    str_(&mut b, s);
                }
                V::I32s(a) => {
                    u32_(&mut b, 9);
                    u32_(&mut b, 5);
                    u64_(&mut b, a.len() as u64);
                    a.iter().for_each(|&x| u32_(&mut b, x as u32));
                }
                V::Bools(a) => {
                    u32_(&mut b, 9);
                    u32_(&mut b, 7);
                    u64_(&mut b, a.len() as u64);
                    b.extend(a.iter().map(|&x| x as u8));
                }
            }
        }
        let mut offset = 0u64;
        for (name, dims) in tensors {
            str_(&mut b, name);
            u32_(&mut b, dims.len() as u32);
            dims.iter().for_each(|&d| u64_(&mut b, d as u64));
            u32_(&mut b, if q8(name) { 8 } else { 0 }); // Q8_0 or F32
            u64_(&mut b, offset);
            let n = dims.iter().product::<usize>() as u64;
            let bytes = if q8(name) { n / 32 * 34 } else { 4 * n };
            offset += bytes.div_ceil(32) * 32;
        }
        b.resize(b.len().div_ceil(32) * 32, 0);
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        for (name, dims) in tensors {
            let n: usize = dims.iter().product();
            let norm = name.contains("norm") || name.contains("clip");
            for i in 0..n {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                if q8(name) {
                    if i % 32 == 0 {
                        b.extend(0x2400u16.to_le_bytes());
                    }
                    b.push((seed >> 56) as u8 / 4);
                    continue;
                }
                let r = (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                let x = if norm { 1.0 + 0.2 * r } else { r };
                b.extend(x.to_le_bytes());
            }
            b.resize(b.len().div_ceil(32) * 32, 0);
        }
        b
    }

    fn gguf_of(keys: &[(String, V)]) -> Result<Gguf, Error> {
        Gguf::from_shards(vec![Box::new(write(keys, &tensors()))])
    }

    fn gguf() -> Gguf {
        gguf_of(&keys()).unwrap()
    }

    #[test]
    fn config_of_the_file() {
        let g = gguf();
        let c = Config::from_gguf(&g).unwrap();
        let ink = c.inkling.as_ref().unwrap();
        assert_eq!(
            (c.n_layer, c.n_embd, c.n_head, c.n_head_kv, c.head_dim),
            (6, 32, 2, 1, 16)
        );
        assert_eq!((c.n_vocab, ink.vocab_padded, c.sliding_window), (36, 40, 3));
        assert_eq!((ink.loop_start, ink.loop_end, ink.loop_count), (2, 4, 2));
        assert_eq!(ink.n_slot(c.n_layer), 8);
        assert_eq!(c.swa, [true, false, true, false, true, false]);
        let m = c.moe.as_ref().unwrap();
        assert_eq!((m.n_expert, m.n_expert_used, m.n_ff_exp), (4, 2, 8));
        assert_eq!(m.weights_scale, 8.0);
        // the new tensor names have their fields
        assert!(g.tensors.token_embd_norm.weight.is_some());
        assert!(g.tensors.output_res_query.weight.is_some());
        assert!(g.tensors.blk[2].loop_ffn_ctrl_gain.weight.is_some());
        assert!(g.tensors.blk[1].attn_v_conv_proj.weight.is_some());
        assert!(g.tensors.extra.is_empty());
    }

    #[test]
    fn malformed_keys() {
        let with = |key: &str, v: V| {
            let mut k = keys();
            let full = format!("inkling2_loop.{key}");
            k.retain(|(name, _)| *name != full);
            k.push((full, v));
            Config::from_gguf(&gguf_of(&k).unwrap())
        };
        // a per-layer array of another length than block_count
        let e = with("attention.head_count", V::I32s(vec![2; 5])).unwrap_err();
        assert!(matches!(e, Error::UnsupportedModel(_)), "{e}");
        // per-layer values that differ
        let e = with("attention.head_count", V::I32s(vec![2, 2, 2, 2, 2, 4])).unwrap_err();
        assert!(matches!(e, Error::UnsupportedModel(_)), "{e}");
        // of the wrong type, or negative: unsupported, as the patterns of
        // the other archs
        let e = with("attention.head_count", V::Str("2")).unwrap_err();
        assert!(matches!(e, Error::UnsupportedModel(_)), "{e}");
        let e = with("attention.head_count", V::I32s(vec![-2; 6])).unwrap_err();
        assert!(matches!(e, Error::UnsupportedModel(_)), "{e}");
        let e = with("loop.count", V::F32(2.0)).unwrap_err();
        assert!(matches!(e, Error::UnsupportedModel(_)), "{e}");
        let e = with("loop.varying", V::Str("yes")).unwrap_err();
        assert!(matches!(e, Error::UnsupportedModel(_)), "{e}");
        // absent: a missing key
        let mut k = keys();
        k.retain(|(name, _)| name != "inkling2_loop.loop.count");
        let e = Config::from_gguf(&gguf_of(&k).unwrap()).unwrap_err();
        assert!(
            matches!(&e, Error::MissingKey(k) if k.ends_with("loop.count")),
            "{e}"
        );
        // a loop the blocks don't divide
        let e = with("loop.start", V::U32(1)).unwrap_err();
        assert!(matches!(e, Error::UnsupportedModel(_)), "{e}");
        // more slots than llama.cpp's limit: 6 + 2 * 253 = 512 is the most
        assert!(with("loop.count", V::U32(254)).is_ok());
        for count in [255, u32::MAX] {
            let e = with("loop.count", V::U32(count)).unwrap_err();
            assert!(matches!(e, Error::UnsupportedModel(_)), "{e}");
        }
        // one value for every layer is an array of it
        let c = with("attention.head_count", V::U32(2)).unwrap();
        assert_eq!(c.n_head, 2);
    }

    #[test]
    fn per_layer_values() {
        use crate::gguf::Value;
        assert_eq!(per_layer(&Value::U32(3), 2), Some(vec![3, 3]));
        let a = Value::Array(vec![Value::I32(1), Value::I32(2)]);
        assert_eq!(per_layer(&a, 2), Some(vec![1, 2]));
        assert_eq!(per_layer(&a, 3), None);
        assert_eq!(per_layer(&Value::I32(-1), 2), None);
        assert_eq!(per_layer(&Value::Str("1".into()), 2), None);
    }

    /// The f32 vectors the model holds, by walking it
    fn loaded_params(m: &Model<'_>) -> usize {
        let ink = m.inkling.as_ref().unwrap();
        let mut n = m.output_norm.len() + ink.tok_norm.len() + ink.output_query.len();
        for l in &ink.layers {
            n += l.attn_query.as_ref().map_or(0, Vec::len);
            n += l.norms.iter().map(Vec::len).sum::<usize>();
            n += l.q_clip.len() + l.rel_clip.len() + l.rel_bias.len();
            n += l.k_conv.len() + l.v_conv.len() + l.post_conv.len() + l.ffn_post_conv.len();
            n += l.ffn_query.len() + l.ffn_norm.len();
            n += l.moe.bias.as_ref().map_or(0, Vec::len);
            n += l.looped.as_ref().map_or(0, |lp| lp.msg_query.len());
        }
        n
    }

    #[test]
    fn memory_of_the_model_and_its_states() {
        let g = gguf();
        let model = Model::load(&g).unwrap();
        let f = Footprint::from_gguf(&g).unwrap();
        let cpu = Backend::Cpu { batch: 64 };
        assert_eq!(model.memory.params, 4 * loaded_params(&model));
        assert_eq!(f.model(cpu), model.memory);
        for (context, draft) in [(16, 0), (16, 3), (40, 1)] {
            let st = model.new_state(context, draft).unwrap();
            let est = f.session(cpu, context, draft);
            assert_eq!(st.memory().allocated(), est.allocated());
            // 4 full attention slots over the context, 4 rings and 8 conv states
            let ring = model.cfg.cpu_ring(context, draft);
            assert_eq!(est.kv, 4 * 2 * 4 * context * KV);
            let conv = (TAPS - 1) * (2 * KV + 2 * E);
            assert_eq!(est.recurrent, 4 * (4 * 2 * ring * KV + 8 * conv));
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            let metal = Backend::Metal { max_batch: 256 };
            assert!(!f.supports(metal) && f.supports(cpu));
            assert_eq!(f.session(metal, 16, 0).allocated(), 0);
            assert_eq!(f.max_context(metal, 0, usize::MAX / 2), None);
        }
    }

    fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// Quantized weights that read the inputs of float ones: the inputs are
    /// quantized for them too
    #[test]
    fn quantized_readers_of_float_inputs() {
        let quantized = [
            "attn_gate",
            "attn_sink_proj",
            "attn_k_conv_proj",
            "attn_v_conv_proj",
        ];
        let file = write_typed(&keys(), &tensors(), |name| {
            quantized.iter().any(|q| name.contains(q))
        });
        let g = Gguf::from_shards(vec![Box::new(file)]).unwrap();
        assert_eq!(
            g.tensors.blk[1].attn_gate.weight.as_ref().unwrap().ty,
            crate::quant::GgmlType::Q8_0
        );
        let model = Model::load(&g).unwrap();
        let mut st = model.new_state(8, 0).unwrap();
        let l = model.forward(&mut st, &[1, 2, 3]);
        assert!(l.len() == VOCAB && l.iter().all(|z| z.is_finite()));
    }

    #[test]
    fn passes_agree() {
        let g = gguf();
        let model = Model::load(&g).unwrap();
        // past the window, the bias extent and the log scale floor
        let ids: Vec<TokenId> = (0..12).map(|i| (i * 7 + 3) % VOCAB as u32).collect();
        let n = ids.len();
        let fresh = || model.new_state(n + 4, 2).unwrap();

        // a decode step per token
        let mut st = fresh();
        let steps: Vec<Vec<f32>> = ids.iter().map(|&t| model.forward(&mut st, &[t])).collect();
        assert!(
            steps
                .iter()
                .all(|l| l.len() == VOCAB && l.iter().all(|z| z.is_finite()))
        );
        assert!(steps[n - 1].iter().all(|z| z.abs() <= 30.0));
        // the tokens matter
        assert!(maxdiff(&steps[0], &steps[n - 1]) > 1e-3);

        // one pass, and passes of 5
        for chunk in [n, 5] {
            let mut st = fresh();
            let (body, last) = ids.split_at(n - 1);
            body.chunks(chunk).for_each(|c| model.prefill(&mut st, c));
            let l = model.forward(&mut st, last);
            assert!(maxdiff(&l, &steps[n - 1]) < 1e-4, "passes of {chunk}");
        }

        // verify 3 at a time, keeping 1 or 2
        let mut st = fresh();
        model.prefill(&mut st, &ids[..2]);
        let (mut p, mut step) = (2, 0);
        while p + 3 <= n {
            let v = model.verify(&mut st, &ids[p..p + 3]);
            for (j, l) in v.iter().enumerate() {
                assert!(maxdiff(l, &steps[p + j]) < 1e-4, "verify at {}", p + j);
            }
            let keep = 1 + step % 2;
            model.keep(&mut st, keep);
            p += keep;
            step += 1;
        }
        let l = model.forward(&mut st, &ids[p..]);
        assert!(maxdiff(&l, &steps[n - 1]) < 1e-4);

        // snapshot halfway, restored in another state
        let mut st = fresh();
        model.prefill(&mut st, &ids[..n / 2]);
        let snap = st.snapshot().unwrap();
        model.prefill(&mut st, &ids[n / 2..n - 1]);
        let mut other = fresh();
        other.restore(&snap).unwrap();
        model.prefill(&mut other, &ids[n / 2..n - 1]);
        let a = model.forward(&mut st, &ids[n - 1..]);
        let b = model.forward(&mut other, &ids[n - 1..]);
        assert_eq!(a, b);
    }
}
