//! Memory use, in bytes: what a model and its sessions take, ahead of time
//! from the GGUF and the session parameters ([`Footprint`]), or once loaded
//! ([`Engine::model_memory`], [`Engine::session_memory`]) and while running
//! ([`Session::memory`], [`Engine::device_memory`]).
//!
//! [`Engine::model_memory`]: crate::engine::Engine::model_memory
//! [`Engine::session_memory`]: crate::engine::Engine::session_memory
//! [`Engine::device_memory`]: crate::engine::Engine::device_memory
//! [`Session::memory`]: crate::engine::Session::memory

use crate::Error;
use crate::gguf::{Gguf, tensor};
use crate::model::{Config, DRAFT_CAP, InklingConfig};
use crate::prelude::*;

/// The memory of a model, without its sessions
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ModelMemory {
    /// tensor data of the file but the MTP head's: mapped, not copied, so
    /// resident as it is read, and given back by the system under pressure
    pub weights: usize,
    /// tensor data of the MTP head, which only drafting sessions read
    pub mtp_weights: usize,
    /// the norm weights and other small vectors, converted to f32, and their
    /// copies on the GPU
    pub params: usize,
}

impl ModelMemory {
    pub fn total(&self) -> usize {
        self.weights + self.mtp_weights + self.params
    }

    /// The tensor data of `g`, split as [`ModelMemory::weights`] and
    /// [`ModelMemory::mtp_weights`], and the f32 vectors the model loads
    pub(crate) fn from_gguf(g: &Gguf, cfg: &Config, has_mtp: bool) -> Self {
        let mut m = ModelMemory {
            params: 4 * params(cfg, has_mtp),
            ..Default::default()
        };
        for name in &g.tensor_order {
            let bytes = g.tensors[name].n_bytes();
            let block = name
                .strip_prefix("blk.")
                .and_then(|rest| rest.split('.').next()?.parse::<usize>().ok());
            match block {
                Some(i) if i >= cfg.n_layer => m.mtp_weights += bytes,
                _ => m.weights += bytes,
            }
        }
        m
    }
}

/// The f32 values the model converts its small tensors to
fn params(c: &Config, has_mtp: bool) -> usize {
    if let Some(ink) = &c.inkling {
        return inkling_params(c, ink);
    }
    let e = c.n_embd;
    let sinks = if c.attn_sinks { c.n_head } else { 0 };
    let attention = 2 * e + 2 * c.head_dim + sinks;
    let delta_net = 2 * e + c.conv_kernel * c.conv_dim() + 2 * c.ssm_v_heads + c.ssm_state;
    let layers: usize = c
        .recurrent
        .iter()
        .map(|&r| if r { delta_net } else { attention })
        .sum();
    // enorm, hnorm, head norm and an attention layer
    let mtp = if has_mtp { 3 * e + attention } else { 0 };
    // output norm, rope frequencies
    layers + mtp + e + c.n_rot / 2
}

/// [`params`] of `inkling2_loop`: the norms, depth mix queries, clip
/// scales, bias tables, conv taps and router biases of each layer, and the
/// message queries of the looped ones
fn inkling_params(c: &Config, ink: &InklingConfig) -> usize {
    let (e, kv_dim, k) = (c.n_embd, c.kv_dim(), ink.conv);
    let n_expert = c.moe.as_ref().map_or(0, |m| m.n_expert);
    let layers: usize = (0..c.n_layer)
        .map(|i| {
            let swa = c.is_swa(i);
            let (ext, inputs) = match swa {
                true => (c.sliding_window, 1),
                false => (ink.rel_extent, 3),
            };
            // the mix queries of the attention inputs and their norms
            let query = if i > 0 { inputs * e } else { 0 };
            let attn = query
                + inputs * e
                + c.n_head * (c.head_dim + ink.n_rel)
                + ext * ink.n_rel
                + 2 * kv_dim * k
                + 2 * e * k;
            let looped = (ink.loop_start..ink.loop_end).contains(&i) && ink.loop_count > 1;
            attn + 2 * e + n_expert + if looped { 4 * e } else { 0 }
        })
        .sum();
    // embeddings norm, output mix query, output norm
    layers + 3 * e
}

/// The memory of a session, by what holds it
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SessionMemory {
    /// kv caches of the attention layers and the MTP head, for the whole
    /// context. The CPU reserves them without writing: the system only
    /// commits the pages of the tokens held, [`SessionMemory::kv_used`].
    pub kv: usize,
    /// states of a fixed size: the conv and recurrent states of the delta
    /// net layers, every copy the session keeps, and the kv caches of the
    /// sliding window layers
    pub recurrent: usize,
    /// output head rows of the MTP head's draft vocabulary
    pub draft_head: usize,
    /// activation and logits buffers allocated with the session
    pub scratch: usize,
    /// allocated by a pass and freed after it, at most: activations, logits,
    /// the delta net states of each token of a verify pass. An estimate on
    /// the CPU, for the engine's batch size.
    pub transient: usize,
    /// bytes of the kv caches per token held
    pub kv_per_token: usize,
}

impl SessionMemory {
    /// Held from the session's creation to its end
    pub fn allocated(&self) -> usize {
        self.kv + self.recurrent + self.draft_head + self.scratch
    }

    /// At most, during a pass
    pub fn peak(&self) -> usize {
        self.allocated() + self.transient
    }

    /// Bytes of the kv caches holding `n` tokens
    pub fn kv_used(&self, n: usize) -> usize {
        self.kv_per_token.saturating_mul(n).min(self.kv)
    }
}

/// The memory of a device, as it reports it
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceMemory {
    /// allocated by the process on the device: buffers, weights mapped to it
    pub allocated: usize,
    /// what the device can use without slowing down
    pub working_set: usize,
}

/// A backend and the parameters its memory depends on
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// prompt tokens per forward pass: 64 for [`Model`](crate::model::Model)
    Cpu { batch: usize },
    /// the `max_batch` of [`MetalEngine::new`](crate::metal::MetalEngine::new)
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal { max_batch: usize },
    /// the `max_batch` of [`CudaEngine::new`](crate::cuda::CudaEngine::new)
    #[cfg(feature = "cuda")]
    Cuda { max_batch: usize },
}

/// What session sizes depend on, besides the session parameters
#[derive(Clone, Copy)]
pub(crate) struct Shape<'a> {
    pub(crate) cfg: &'a Config,
    pub(crate) has_mtp: bool,
    /// bytes of a row of the output head
    pub(crate) head_row_bytes: usize,
}

impl Shape<'_> {
    /// The attention layers over the whole context, whose kv caches grow
    /// with it
    pub(crate) fn attention_layers(&self) -> usize {
        self.cfg.n_layer - self.delta_net_layers() - self.window_layers()
    }

    /// The attention layers over a sliding window
    pub(crate) fn window_layers(&self) -> usize {
        self.cfg.swa.iter().filter(|&&w| w).count()
    }

    pub(crate) fn delta_net_layers(&self) -> usize {
        self.cfg.recurrent.iter().filter(|&&r| r).count()
    }

    /// floats per position of a kv cache, keys or values
    pub(crate) fn kv_dim(&self) -> usize {
        self.cfg.n_head_kv * self.cfg.head_dim
    }

    /// floats of a delta net's conv and recurrent states
    pub(crate) fn delta_net_state(&self) -> (usize, usize) {
        let c = self.cfg;
        (
            c.conv_kernel.saturating_sub(1) * c.conv_dim(),
            c.ssm_v_heads * c.ssm_state * c.ssm_state,
        )
    }

    /// Bytes of the head rows of the draft vocabulary, when the vocabulary is
    /// too large to draft from all of it
    pub(crate) fn draft_head(&self) -> usize {
        if self.cfg.n_vocab > DRAFT_CAP {
            DRAFT_CAP * self.head_row_bytes
        } else {
            0
        }
    }

    /// What [`Model::new_state`](crate::model::Model::new_state) allocates for
    /// `context` tokens, drafting when `draft` > 0, and what its passes of
    /// `batch` tokens allocate
    pub(crate) fn cpu_session(&self, context: usize, draft: usize, batch: usize) -> SessionMemory {
        if let Some(ink) = &self.cfg.inkling {
            return self.inkling_cpu_session(ink, context, draft, batch);
        }
        let c = self.cfg;
        let (e, kv_dim) = (c.n_embd, self.kv_dim());
        let mtp = draft > 0 && self.has_mtp;
        let attention = self.attention_layers() + mtp as usize;
        let (conv, s) = self.delta_net_state();
        let state = 4 * (conv + s);
        let window = self.window_layers() * 2 * 4 * c.cpu_ring(context, draft) * kv_dim;

        // a verify pass keeps the states after each token but the last, and
        // each layer builds its copy before handing it over
        let saved = draft * (self.delta_net_layers() + 1) * state;
        // the residual stream, the normalized input and a layer's output, and
        // the largest layer's own activations
        let n = batch.max(draft + 1);
        let qd = c.n_head * c.head_dim;
        // a mixture of experts: the router logits, the output, and the shared
        // expert's or an expert's activations over its tokens
        let moe = c
            .moe
            .as_ref()
            .map_or(0, |m| m.n_expert + 2 * e + 2 * m.n_ff_shexp.max(m.n_ff_exp));
        let layer = (2 * c.n_ff)
            .max(5 * qd + 2 * kv_dim)
            .max(3 * c.conv_dim() + 3 * c.d_inner())
            .max(moe);
        let activations = 4 * n * (3 * e + layer);
        // the logits of a verify pass, and their copy per token
        let logits = 2 * 4 * (draft + 1) * c.n_vocab;

        SessionMemory {
            kv: attention * 2 * 4 * context.saturating_mul(kv_dim),
            recurrent: self.delta_net_layers() * state + window,
            draft_head: if mtp { self.draft_head() } else { 0 },
            scratch: if mtp { 4 * e } else { 0 },
            transient: saved + activations + logits,
            kv_per_token: attention * 2 * 4 * kv_dim,
        }
    }
}

impl Shape<'_> {
    /// [`Shape::cpu_session`] of `inkling2_loop`: a slot per layer visit,
    /// each with kv caches (a ring on sliding window layers) and a conv state
    fn inkling_cpu_session(
        &self,
        ink: &InklingConfig,
        context: usize,
        draft: usize,
        batch: usize,
    ) -> SessionMemory {
        let c = self.cfg;
        let (e, kv_dim) = (c.n_embd, self.kv_dim());
        let n_slot = ink.n_slot(c.n_layer);
        let full = (0..n_slot)
            .filter(|&s| !c.is_swa(ink.layer_of(s, c.n_layer)))
            .count();
        let window = (n_slot - full) * 2 * 4 * c.cpu_ring(context, draft) * kv_dim;
        let conv = 4 * (ink.conv - 1) * ink.conv_channels(kv_dim, e);

        // a verify pass keeps the conv inputs of its tokens after the state's
        // rows, `inkling::short_conv`, in each slot
        let hist = ink.conv - 1;
        let saved = match draft {
            0 => 0,
            _ => 4 * n_slot * (hist + draft + 1) * ink.conv_channels(kv_dim, e),
        };
        // per token, what a pass holds throughout: the stream, the prelude's
        // output, the previous pass's normalized one, the normalized blocks
        // and the messages of a looped layer
        let n = batch.max(draft + 1);
        let blocks = (0..n_slot)
            .filter(|&s| ink.layer_of(s, c.n_layer).is_multiple_of(ink.block))
            .count();
        let stream = (3 + blocks + 4) * e;
        // and the largest of a layer's steps, `inkling::ink_layer`: the
        // activations quantized for a matmul take 40 bytes per 32 values
        let q8 = |f: usize| f.div_ceil(32) * 10;
        let (qd, nh, r) = (c.n_head * c.head_dim, c.n_head, ink.gate_rank);
        // a controller: its input pair, bottleneck and output
        let control = 2 * e + q8(2 * e) + r + q8(r) + e;
        // the normalized stream and three attention inputs, with the
        // controllers of all three at once; then their projections; then
        // (smaller) the convs and the attention
        let inputs = 4 * e + 3 * control;
        let projections = 4 * e
            + 3 * q8(e)
            + 2 * qd
            + 2 * kv_dim
            + nh * (ink.n_rel + 1)
            + 2 * c.n_head_kv * ink.conv;
        // the feed-forward input, the router logits, the chosen experts'
        // SITU outputs and their quantization, their outputs (in a pass of
        // few tokens per expert, `inkling::cpu::experts`), and the output
        let experts = c.moe.as_ref().map_or(0, |m| {
            let (k, ff) = (m.n_expert_used, m.n_ff_exp);
            2 * e + q8(e) + m.n_expert + k * ff + q8(k * ff) + k * e
        });
        let layer = inputs.max(projections).max(2 * e + control).max(experts);
        let activations = 4 * n * (stream + layer);
        // the logits of a verify pass, and their copy per token
        let logits = 2 * 4 * (draft + 1) * c.n_vocab;

        SessionMemory {
            kv: full * 2 * 4 * context.saturating_mul(kv_dim),
            recurrent: n_slot * conv + window,
            draft_head: 0,
            scratch: 0,
            transient: saved + activations + logits,
            kv_per_token: full * 2 * 4 * kv_dim,
        }
    }
}

/// The memory a model takes and its sessions would, from its GGUF alone:
/// nothing is loaded or allocated
#[derive(Debug, Clone)]
pub struct Footprint {
    pub cfg: Config,
    /// whether the file has an MTP head to draft tokens with
    pub has_mtp: bool,
    /// bytes of a row of the output head
    pub head_row_bytes: usize,
    model: ModelMemory,
}

impl Footprint {
    pub fn from_gguf(g: &Gguf) -> Result<Self, Error> {
        let cfg = Config::from_gguf(g)?;
        let t = &g.tensors;
        let has_mtp = tensor!(t, blk[cfg.n_layer].nextn.eh_proj.weight).is_ok();
        let head = tensor!(t, output.weight).or(tensor!(t, token_embd.weight))?;
        Ok(Footprint {
            model: ModelMemory::from_gguf(g, &cfg, has_mtp),
            head_row_bytes: head.ty.row_size(cfg.n_embd),
            has_mtp,
            cfg,
        })
    }

    pub(crate) fn shape(&self) -> Shape<'_> {
        Shape {
            cfg: &self.cfg,
            has_mtp: self.has_mtp,
            head_row_bytes: self.head_row_bytes,
        }
    }

    /// What the model takes on `backend`
    pub fn model(&self, backend: Backend) -> ModelMemory {
        match backend {
            Backend::Cpu { .. } => self.model,
            // the small vectors are uploaded, the weights mapped
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Backend::Metal { .. } => ModelMemory {
                params: 2 * self.model.params,
                ..self.model
            },
            // the small vectors are uploaded, the weights copied
            #[cfg(feature = "cuda")]
            Backend::Cuda { .. } => ModelMemory {
                params: 2 * self.model.params,
                ..self.model
            },
        }
    }

    /// Whether `backend` runs the model: `inkling2_loop` runs on the CPU, and
    /// on Metal and CUDA when their kernels take the model's geometry
    pub fn supports(&self, backend: Backend) -> bool {
        match backend {
            Backend::Cpu { .. } => true,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Backend::Metal { .. } => crate::metal::supports(&self.cfg),
            #[cfg(feature = "cuda")]
            Backend::Cuda { .. } => crate::cuda::supports(&self.cfg),
        }
    }

    /// What a session on `backend` takes, created by
    /// [`Engine::new_session`](crate::engine::Engine::new_session) with
    /// `context` and `draft`; nothing when the backend doesn't run the model
    /// ([`Footprint::supports`]), its engine then failing to be created
    pub fn session(&self, backend: Backend, context: usize, draft: usize) -> SessionMemory {
        if !self.supports(backend) {
            return SessionMemory::default();
        }
        match backend {
            Backend::Cpu { batch } => self.shape().cpu_session(context, draft, batch),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Backend::Metal { max_batch } => self.shape().metal_session(context, draft, max_batch),
            #[cfg(feature = "cuda")]
            Backend::Cuda { max_batch } => self.shape().cuda_session(context, draft, max_batch),
        }
    }

    /// The longest context whose session, with the model, fits in `budget`
    /// bytes at its peak: weights included, the MTP head's when drafting.
    /// None when even an empty context doesn't fit, or the backend doesn't
    /// run the model.
    pub fn max_context(&self, backend: Backend, draft: usize, budget: usize) -> Option<usize> {
        if !self.supports(backend) {
            return None;
        }
        let m = self.model(backend);
        let model = m.weights + m.params + if draft > 0 { m.mtp_weights } else { 0 };
        let fits = |context| model + self.session(backend, context, draft).peak() <= budget;
        let (mut lo, mut hi) = (0, self.cfg.context_length);
        if !fits(lo) {
            return None;
        }
        if fits(hi) {
            return Some(hi);
        }
        // the longest that fits is in [lo, hi)
        while lo + 1 < hi {
            let mid = lo + (hi - lo) / 2;
            if fits(mid) { lo = mid } else { hi = mid }
        }
        Some(lo)
    }
}
