//! The CUDA path of `inkling2_loop` (Inkling-2), the GPU version of
//! [`crate::model::inkling`], whose math the kernels of `inkling.cu` follow,
//! laid out as the Metal path (`metal/inkling.rs`) is.
//!
//! A pass runs the visits of the layers in order over the stream `x` of its
//! tokens, as the CPU does. The normalized blocks the depth mixes read are
//! rows of [`InkScratch::stack`], each block in a region of its own, see
//! [`StackPlan`]. Each visit (slot) has its kv caches, f16 (a ring on the
//! sliding window layers, see [`Shape::cuda_ring`]), and its conv state
//! `[slots][taps - 1][channels]`, whose copies serve verify passes as the
//! delta nets' do (see `ssm_conv` in the kernels).
//!
//! The experts of the core's layers go to device memory before the others':
//! each pass reads them `loop.count` times.

use super::*;
use crate::model::InklingConfig;
use crate::model::inkling::{InkLayer, Inkling, MIX_EPS, StackPlan, Visit};

/// The taps of the short convs the kernels take
const CONV_TAPS: usize = 4;
/// Largest head size (`INK_MAXE` lanes of 32) and relative position channels
/// (a lane each) of the attention kernels
const MAX_HEAD_DIM: usize = 256;
const MAX_REL: usize = 32;
/// Most states of a depth mix, and its most queries (`INK_MAXS`, `INK_MAXQ`)
const MIX_STATES: usize = 8;
const MIX_QUERIES: usize = 4;
const MIX_THREADS: u32 = 256;
/// Blocks the attention of a short pass aims at, splitting the positions of
/// each (head, token) in up to [`ATTN_MAX_SPLITS`] ranges, see `ink_attn`
const ATTN_BLOCKS: usize = 256;
const ATTN_MAX_SPLITS: usize = 16;
const ATTN_WARPS: usize = 8;
/// Passes of this many tokens or more (without saved states) run their short
/// convs a thread per (channel, token), see `ink_conv_par`, rather than a
/// thread per channel through the tokens
const CONV_PAR_MIN: usize = 16;

#[repr(C)]
struct MixArgs {
    stack: DevPtr,
    cur: DevPtr,
    query: DevPtr,
    w: DevPtr,
    out: DevPtr,
    e: i32,
    n_states: i32,
    n_q: i32,
    copies: i32,
    scale: f32,
    eps: f32,
    norm_cur: i32,
    out_norm: i32,
    out_eps: f32,
    in_row0: i32,
    out_row0: i32,
    ostride: i32,
    off: [i32; MIX_STATES],
}

#[repr(C)]
struct RmsArgs {
    x: DevPtr,
    out: DevPtr,
    n: i32,
    eps: f32,
    istride: i32,
    ostride: i32,
}

#[repr(C)]
struct CtrlArgs {
    x: DevPtr,
    g: DevPtr,
    f: DevPtr,
    m: DevPtr,
    out: DevPtr,
    n: i32,
    pad: i32,
}

#[repr(C)]
struct InkConvArgs {
    x: DevPtr,
    w: DevPtr,
    state: DevPtr,
    out: DevPtr,
    dyn_: DevPtr,
    params: DevPtr,
    width: i32,
    n: i32,
    channels: i32,
    offset: i32,
    slots: i32,
    save: i32,
    add: i32,
    has_dyn: i32,
    head: i32,
    dyn_stride: i32,
    scale: f32,
    pad: i32,
}

#[repr(C)]
struct InkAttnArgs {
    q: DevPtr,
    k: DevPtr,
    v: DevPtr,
    r: DevPtr,
    q_clip: DevPtr,
    r_clip: DevPtr,
    k_cache: DevPtr,
    v_cache: DevPtr,
    sink: DevPtr,
    rel_bias: DevPtr,
    gate: DevPtr,
    out: DevPtr,
    part_o: DevPtr,
    part_ml: DevPtr,
    params: DevPtr,
    n_head: i32,
    n_kv: i32,
    hd: i32,
    n_rel: i32,
    eps: f32,
    scale: f32,
    clip: f32,
    rel_clip: f32,
    full: i32,
    floor: i32,
    alpha: f32,
    window: i32,
    mask: i32,
    ext: i32,
    splits: i32,
    pad: i32,
}

#[repr(C)]
struct SituArgs {
    gate: DevPtr,
    up: DevPtr,
    n: i32,
    beta: f32,
    ib: f32,
    lb: f32,
    ilb: f32,
    pad: i32,
}

#[repr(C)]
struct CapArgs {
    z: DevPtr,
    n: i32,
    cap: f32,
    inv: f32,
    pad: i32,
}

/// The kernels of `inkling.cu`
pub(super) struct InkKernels {
    mix: Func,
    rms: Func,
    control: Func,
    conv: Func,
    conv_par: Func,
    conv_state: Func,
    attn_prep: Func,
    attn: Func,
    attn_reduce: Func,
    situ: Func,
    moe_sum: Func,
    softcap: Func,
    copy: Func,
}

impl InkKernels {
    pub(super) fn load(f: impl Fn(&str) -> Result<Func, Error>) -> Result<Self, Error> {
        Ok(InkKernels {
            mix: f("ink_mix")?,
            rms: f("ink_rms")?,
            control: f("ink_control")?,
            conv: f("ink_conv")?,
            conv_par: f("ink_conv_par")?,
            conv_state: f("ink_conv_state")?,
            attn_prep: f("ink_attn_prep")?,
            attn: f("ink_attn")?,
            attn_reduce: f("ink_attn_reduce")?,
            situ: f("ink_situ")?,
            moe_sum: f("ink_moe_sum")?,
            softcap: f("ink_softcap")?,
            copy: f("ink_copy")?,
        })
    }
}

/// Whether the kernels take the model's geometry
pub(super) fn check(c: &Config, ink: &InklingConfig) -> Result<(), Error> {
    if !c.head_dim.is_multiple_of(32)
        || c.head_dim > MAX_HEAD_DIM
        || ink.n_rel > MAX_REL
        || ink.conv != CONV_TAPS
    {
        return Err(backend(
            "cuda: inkling2_loop needs heads of a multiple of 32 up to 256, at most 32 relative \
             position channels and convs of 4 taps",
        ));
    }
    // the output's depth mix: all the blocks and the stream
    let plan = StackPlan::new(ink, c.n_layer);
    if plan.prelude + plan.core + plan.epilogue + 1 > MIX_STATES {
        return Err(backend(format!(
            "cuda: inkling2_loop's depth mixes take at most {MIX_STATES} states"
        )));
    }
    // the long convs' keys and values go to rows of the attention inputs
    if c.kv_dim() > c.n_embd {
        return Err(backend(
            "cuda: inkling2_loop needs keys no wider than the stream",
        ));
    }
    Ok(())
}

/// Ranges of positions of each (head, token) of the attention of a pass of
/// `n` tokens: of short passes, to fill the device; of the others, one
fn attn_splits(c: &Config, n: usize) -> usize {
    if n > NT {
        return 1;
    }
    ATTN_BLOCKS.div_ceil(c.n_head * n).clamp(1, ATTN_MAX_SPLITS)
}

/// (token, range) pairs of the attention partials of passes of up to `b`
/// tokens, at most
fn attn_parts(c: &Config, b: usize) -> usize {
    (1..=b.min(NT))
        .map(|n| n * attn_splits(c, n))
        .max()
        .unwrap_or(0)
}

/// The controllers of a looped layer, see [`crate::model::inkling::Loop`]
struct GpuLoop {
    msg_query: Buf,
    down: [GpuWeight; 4],
    gain: [GpuWeight; 4],
    forget: [GpuWeight; 4],
}

/// A physical layer, see [`InkLayer`]
struct GpuInkLayer {
    swa: bool,
    /// the depth mix queries of the attention inputs, and how many
    attn_query: Option<Buf>,
    n_q: usize,
    /// `[n_norms][n_embd]`: one for all inputs, or one each
    norms: Buf,
    n_norms: usize,
    wq: GpuWeight,
    wk: GpuWeight,
    wv: GpuWeight,
    wo: GpuWeight,
    wgate: GpuWeight,
    wsink: GpuWeight,
    wrel: GpuWeight,
    q_clip: Buf,
    rel_clip: Buf,
    /// `[ext][n_rel]`
    rel_bias: Buf,
    ext: usize,
    k_conv: Buf,
    v_conv: Buf,
    post_conv: Buf,
    ffn_post_conv: Buf,
    k_dyn: Option<GpuWeight>,
    v_dyn: Option<GpuWeight>,
    ffn_query: Buf,
    ffn_norm: Buf,
    moe: GpuMoe,
    looped: Option<GpuLoop>,
}

/// The model on the device, see [`Inkling`]
pub(super) struct GpuInkling {
    tok_norm: Buf,
    output_query: Buf,
    layers: Vec<GpuInkLayer>,
    visits: Vec<Visit>,
    stack: StackPlan,
}

/// The order in which the layers' experts go to device memory: the core's,
/// which a pass reads `loop.count` times, then the others
fn experts_order(ink: &InklingConfig, n_layer: usize) -> Vec<usize> {
    (ink.loop_start..ink.loop_end)
        .chain(0..ink.loop_start)
        .chain(ink.loop_end..n_layer)
        .collect()
}

/// The model's weights through `p`, in the order of their priority for
/// device memory: the output head, every layer's attention, controllers and
/// router, then the experts (see [`experts_order`])
pub(super) fn place(
    model: &Model<'_>,
    m: &Inkling<'_>,
    p: &mut Placer,
) -> Result<(GpuWeight, GpuInkling), Error> {
    let c = &model.cfg;
    let ink = c.inkling.as_ref().expect("an inkling config");
    let output = p.weight(&model.output)?;
    let f32s = |v: &[f32]| Buf::from_f32(v);
    // a placeholder until the experts are placed, after every layer's
    let none = |x: &Experts<'_>, buf: &Arc<Buf>| GpuExperts {
        ty: 0,
        cols: x.cols,
        rows: x.rows,
        row_bytes: 0,
        stride: 0,
        offset: 0,
        n: x.n,
        buf: buf.clone(),
    };
    let mut layer = |l: &InkLayer<'_>| -> Result<GpuInkLayer, Error> {
        let looped = match &l.looped {
            Some(lp) => {
                let mut each = |w: &[Weight<'_>; 4]| -> Result<[GpuWeight; 4], Error> {
                    Ok([
                        p.weight(&w[0])?,
                        p.weight(&w[1])?,
                        p.weight(&w[2])?,
                        p.weight(&w[3])?,
                    ])
                };
                Some(GpuLoop {
                    msg_query: f32s(&lp.msg_query)?,
                    down: each(&lp.down)?,
                    gain: each(&lp.gain)?,
                    forget: each(&lp.forget)?,
                })
            }
            None => None,
        };
        let moe = &l.moe;
        let router = p.weight(&moe.router)?;
        Ok(GpuInkLayer {
            swa: l.swa,
            n_q: l.attn_query.as_ref().map_or(0, |q| q.len() / c.n_embd),
            attn_query: l.attn_query.as_deref().map(f32s).transpose()?,
            norms: f32s(&l.norms.concat())?,
            n_norms: l.norms.len(),
            wq: p.weight(&l.wq)?,
            wk: p.weight(&l.wk)?,
            wv: p.weight(&l.wv)?,
            wo: p.weight(&l.wo)?,
            wgate: p.weight(&l.wgate)?,
            wsink: p.weight(&l.wsink)?,
            wrel: p.weight(&l.wrel)?,
            q_clip: f32s(&l.q_clip)?,
            rel_clip: f32s(&l.rel_clip)?,
            rel_bias: f32s(&l.rel_bias)?,
            ext: l.rel_bias.len() / ink.n_rel,
            k_conv: f32s(&l.k_conv)?,
            v_conv: f32s(&l.v_conv)?,
            post_conv: f32s(&l.post_conv)?,
            ffn_post_conv: f32s(&l.ffn_post_conv)?,
            k_dyn: l.k_dyn.as_ref().map(|w| p.weight(w)).transpose()?,
            v_dyn: l.v_dyn.as_ref().map(|w| p.weight(w)).transpose()?,
            ffn_query: f32s(&l.ffn_query)?,
            ffn_norm: f32s(&l.ffn_norm)?,
            moe: GpuMoe {
                bias: moe.bias.as_deref().map(f32s).transpose()?,
                gate: none(&moe.gate, &router.buf),
                up: none(&moe.up, &router.buf),
                down: none(&moe.down, &router.buf),
                router,
                shared: None,
            },
            looped,
        })
    };
    let mut layers = m
        .layers
        .iter()
        .map(&mut layer)
        .collect::<Result<Vec<_>, _>>()?;
    for i in experts_order(ink, c.n_layer) {
        let (gm, moe) = (&mut layers[i].moe, &m.layers[i].moe);
        gm.gate = p.experts(&moe.gate)?;
        gm.up = p.experts(&moe.up)?;
        gm.down = p.experts(&moe.down)?;
    }
    let gpu = GpuInkling {
        tok_norm: f32s(&m.tok_norm)?,
        output_query: f32s(&m.output_query)?,
        layers,
        visits: m.visits.clone(),
        stack: StackPlan::new(ink, c.n_layer),
    };
    Ok((output, gpu))
}

/// Bytes on the device of the largest experts tensor of `m`
pub(super) fn largest_experts(m: &Inkling<'_>) -> usize {
    let x = |x: &Experts<'_>| x.data.len() / x.row_bytes() * device_row_bytes(x.ty, x.cols);
    m.layers
        .iter()
        .map(|l| x(&l.moe.gate).max(x(&l.moe.up)).max(x(&l.moe.down)))
        .max()
        .unwrap_or(0)
}

/// The buffers of a pass of `inkling2_loop` beyond [`Scratch`]'s
pub(super) struct InkScratch {
    /// the normalized blocks, see [`StackPlan`]: regions of `batch` rows
    stack: Buf,
    /// the prelude's output, which each pass of the core starts from
    prelude: Buf,
    /// `[3][batch][n_embd]`: the attention inputs
    inp: Buf,
    /// `[4][batch][n_embd]`: a looped layer's messages
    msg: Buf,
    /// a controller's input pair, bottleneck, gain and forget logits
    u: Buf,
    z: Buf,
    gain: Buf,
    forget: Buf,
    /// relative position channels and sink logits of the heads
    r: Buf,
    sink: Buf,
    /// `[2][batch][n_head_kv * taps]`: the keys' and values' dynamic taps
    dyn_: Buf,
    /// the attention's partial outputs and (max, sum) per (token, head,
    /// range) of a short pass, see `ink_attn`
    part_o: Buf,
    part_ml: Buf,
}

/// Bytes of the buffers of [`InkScratch`], in the order of its fields, for
/// passes of `b` tokens; none for the other archs
pub(super) fn scratch_bytes(c: &Config, b: usize) -> Vec<usize> {
    let Some(ink) = &c.inkling else {
        return Vec::new();
    };
    let e = c.n_embd;
    let regions = StackPlan::new(ink, c.n_layer).regions();
    let parts = attn_parts(c, b) * c.n_head;
    [
        regions * b * e,
        b * e,
        3 * b * e,
        4 * b * e,
        b * 2 * e,
        b * ink.gate_rank,
        b * e,
        b * e,
        b * c.n_head * ink.n_rel,
        b * c.n_head,
        2 * b * c.n_head_kv * ink.conv,
        parts * c.head_dim,
        parts * 2,
    ]
    .map(|n| sz(4 * n))
    .to_vec()
}

impl InkScratch {
    pub(super) fn new(c: &Config, b: usize) -> Result<Option<Self>, Error> {
        if c.inkling.is_none() {
            return Ok(None);
        }
        let mut bytes = scratch_bytes(c, b).into_iter();
        let mut next = || Buf::zeroed(bytes.next().expect("a size per buffer"));
        Ok(Some(InkScratch {
            stack: next()?,
            prelude: next()?,
            inp: next()?,
            msg: next()?,
            u: next()?,
            z: next()?,
            gain: next()?,
            forget: next()?,
            r: next()?,
            sink: next()?,
            dyn_: next()?,
            part_o: next()?,
            part_ml: next()?,
        }))
    }
}

/// The state of a slot of a session: its f16 kv caches (with `ring`, a
/// sliding window's, position `p` at row `p % ring`), and `slots` copies of
/// its conv state
pub(super) struct InkSlot {
    k: Buf,
    v: Buf,
    ring: Option<usize>,
    conv: Buf,
    swa: bool,
}

impl InkSlot {
    pub(super) fn bytes(&self) -> usize {
        self.k.bytes + self.v.bytes + self.conv.bytes
    }
}

/// See [`Session::snapshot`]: per slot, the kv rows a next token may see
/// and the current conv state
struct InkSnapshot {
    slots: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)>,
}

impl Shape<'_> {
    /// Bytes of a slot's conv state, one copy
    fn ink_conv_bytes(&self, ink: &InklingConfig) -> usize {
        4 * (ink.conv - 1) * ink.conv_channels(self.kv_dim(), self.cfg.n_embd)
    }

    /// The slots of sliding window layers, of the full attention ones
    fn ink_slots(&self, ink: &InklingConfig) -> (usize, usize) {
        let c = self.cfg;
        let n_slot = ink.n_slot(c.n_layer);
        let swa = (0..n_slot)
            .filter(|&s| c.is_swa(ink.layer_of(s, c.n_layer)))
            .count();
        (swa, n_slot - swa)
    }

    /// [`Shape::cuda_session`] of `inkling2_loop`
    pub(super) fn inkling_cuda_session(
        &self,
        ink: &InklingConfig,
        context: usize,
        draft: usize,
        max_batch: usize,
    ) -> SessionMemory {
        let c = self.cfg;
        let b = max_batch.max(1);
        let slots = self.cuda_draft(draft, b) + 1;
        let (swa, full) = self.ink_slots(ink);
        let ring = self.cuda_ring(context, b);
        let window_bytes = swa * 2 * sz(self.cuda_window_bytes(context, b));
        SessionMemory {
            kv: full * 2 * sz(kv_bytes(context, self.kv_dim()))
                + if ring.is_none() { window_bytes } else { 0 },
            recurrent: (swa + full) * sz(slots * self.ink_conv_bytes(ink))
                + if ring.is_some() { window_bytes } else { 0 },
            draft_head: 0,
            scratch: self.cuda_scratch(b, slots),
            // the logits of a verify pass and the embeddings of a pass, on the host
            transient: 4 * slots * c.n_vocab + 4 * b * c.n_embd,
            kv_per_token: (full + if ring.is_none() { swa } else { 0 }) * 2 * 2 * self.kv_dim(),
        }
    }
}

/// The slots of a session of `context` positions, `slots` copies of the conv
/// states, passes of up to `b` tokens
pub(super) fn new_slots(
    c: &Config,
    shape: &Shape<'_>,
    context: usize,
    slots: usize,
    b: usize,
) -> Result<Vec<InkSlot>, Error> {
    let ink = c.inkling.as_ref().expect("an inkling config");
    let ring = shape.cuda_ring(context, b);
    let window_bytes = shape.cuda_window_bytes(context, b);
    let full_bytes = kv_bytes(context, shape.kv_dim());
    let conv = slots * shape.ink_conv_bytes(ink);
    (0..ink.n_slot(c.n_layer))
        .map(|slot| {
            let swa = c.is_swa(ink.layer_of(slot, c.n_layer));
            let bytes = if swa { window_bytes } else { full_bytes };
            Ok(InkSlot {
                k: Buf::zeroed(bytes)?,
                v: Buf::zeroed(bytes)?,
                ring: if swa { ring } else { None },
                conv: Buf::zeroed(conv)?,
                swa,
            })
        })
        .collect()
}

/// Positions of a slot a next token may see, of the `n_past` before it
fn ink_rows(c: &Config, st: &InkSlot, n_past: usize) -> usize {
    if st.swa {
        n_past.min(c.sliding_window - 1)
    } else {
        n_past
    }
}

impl CudaSession<'_> {
    pub(super) fn ink_snapshot(&self) -> Result<Snapshot, Error> {
        assert!(
            self.verify_from.is_none(),
            "snapshot between verify and keep"
        );
        let e = self.e;
        e.d.bind();
        let c = e.cfg();
        let ink = c.inkling.as_ref().expect("an inkling config");
        let kv_row = 2 * c.kv_dim();
        let conv = e.model.shape().ink_conv_bytes(ink);
        let mut bytes = 0;
        let mut copy = |buf: &Buf, offset: usize, len: usize| -> Result<Vec<u8>, Error> {
            let mut out = Vec::new();
            out.try_reserve_exact(len)
                .map_err(|_| Error::OutOfMemory { bytes: len })?;
            out.resize(len, 0);
            buf.read(offset, &mut out);
            bytes += len;
            Ok(out)
        };
        let mut slots = Vec::with_capacity(self.ink.len());
        for st in &self.ink {
            let first = self.n_past - ink_rows(c, st, self.n_past);
            // the positions a next token may see, in order: runs of rows
            // that don't wrap around the ring
            let mut rows = |buf: &Buf| -> Result<Vec<u8>, Error> {
                let mut out = Vec::new();
                let mut p = first;
                while p < self.n_past {
                    let (row, end) = match st.ring {
                        Some(r) => (p % r, self.n_past.min(p - p % r + r)),
                        None => (p, self.n_past),
                    };
                    out.extend_from_slice(&copy(buf, row * kv_row, (end - p) * kv_row)?);
                    p = end;
                }
                Ok(out)
            };
            let (k, v) = (rows(&st.k)?, rows(&st.v)?);
            let cs = copy(&st.conv, self.slot * conv, conv)?;
            slots.push((k, v, cs));
        }
        Ok(Snapshot::new(self.n_past, bytes, InkSnapshot { slots }))
    }

    pub(super) fn ink_restore(&mut self, snapshot: &Snapshot) -> Result<(), Error> {
        let snap: &InkSnapshot = snapshot.data()?;
        let n_past = snapshot.n_past();
        if n_past > self.context {
            return Err(Error::Snapshot("more tokens than the context holds"));
        }
        let e = self.e;
        e.d.bind();
        let c = e.cfg();
        let ink = c.inkling.as_ref().expect("an inkling config");
        let kv_row = 2 * c.kv_dim();
        let conv = e.model.shape().ink_conv_bytes(ink);
        let fits = self.ink.len() == snap.slots.len()
            && self.ink.iter().zip(&snap.slots).all(|(st, (k, v, cs))| {
                let rows = ink_rows(c, st, n_past) * kv_row;
                k.len() == rows && v.len() == rows && cs.len() == conv
            });
        if !fits {
            return Err(Error::Snapshot("taken from another model"));
        }
        // the conv states go to slot 0
        for (st, (sk, sv, cs)) in self.ink.iter().zip(&snap.slots) {
            let first = n_past - sk.len() / kv_row;
            let mut p = first;
            while p < n_past {
                let (row, end) = match st.ring {
                    Some(r) => (p % r, n_past.min(p - p % r + r)),
                    None => (p, n_past),
                };
                let src = (p - first) * kv_row..(end - first) * kv_row;
                st.k.write(row * kv_row, &sk[src.clone()]);
                st.v.write(row * kv_row, &sv[src]);
                p = end;
            }
            st.conv.write(0, cs);
        }
        self.slot = 0;
        self.n_past = n_past;
        self.verify_from = None;
        Ok(())
    }
}

/// What a visit of a layer needs of its pass: the slot's state, the tokens,
/// the [`PassParams`] and the conv states' copies
struct Seg<'a> {
    st: &'a InkSlot,
    n: usize,
    params: DevPtr,
    slots: usize,
    save: bool,
}

impl Seg<'_> {
    /// Whether the short convs run a thread per (channel, token)
    fn par(&self) -> bool {
        self.n >= CONV_PAR_MIN && !self.save
    }
}

/// The rows of a depth mix: `n`, of the states and stream from `in_row0`, of
/// the outputs from `out_row0`
#[derive(Clone, Copy)]
struct Rows {
    n: usize,
    in_row0: usize,
    out_row0: usize,
}

impl Rows {
    fn all(n: usize) -> Self {
        Rows {
            n,
            in_row0: 0,
            out_row0: 0,
        }
    }
}

/// The outputs of a depth mix: each query's `copies` times from `out`,
/// `stride` floats apart, normalized by the rows of `norm` when given
struct MixOut {
    out: DevPtr,
    stride: usize,
    copies: usize,
    norm: Option<DevPtr>,
}

fn ink_scratch(s: &Scratch) -> &InkScratch {
    s.ink
        .as_ref()
        .expect("inkling scratch for an inkling model")
}

impl CudaEngine<'_> {
    fn ink(&self) -> &InkKernels {
        &self.k.ink
    }

    /// A pass of the `n` embeddings in `s.x` through the visits, the slots
    /// `st`, and the logits of its last `rows` into `s.logits`, see
    /// [`CudaEngine::encode_pass`]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ink_pass(
        &self,
        ink: &GpuInkling,
        s: &Scratch,
        st: &[InkSlot],
        n: usize,
        rows: usize,
        params: DevPtr,
        slots: usize,
        save: bool,
    ) {
        let c = self.cfg();
        let cfg = c.inkling.as_ref().expect("an inkling config");
        let e = c.n_embd;
        let is = ink_scratch(s);
        let plan = ink.stack;
        let x = s.x.ptr;
        let b = s.batch;
        let all = Rows::all(n);
        self.stage_begin();
        self.ink_norm(x, ink.tok_norm.ptr, x, n);

        // the blocks of the stream (regions of the stack), the prelude's,
        // the previous pass's after its first
        let mut blocks: Vec<usize> = Vec::new();
        let mut prelude_blocks: Vec<usize> = Vec::new();
        let mut bank: Vec<usize> = Vec::new();
        // blocks pushed by the phase (prelude, a pass of the core, epilogue)
        let mut pushed = 0;
        let mut phase = (0, 0);
        for v in &ink.visits {
            let layer = &ink.layers[v.layer];
            let here = (
                match v.layer {
                    l if l < cfg.loop_start => 0,
                    l if l < cfg.loop_end => 1,
                    _ => 2,
                },
                v.pass,
            );
            if here != phase {
                phase = here;
                pushed = 0;
            }
            let first_of_pass = v.layer == cfg.loop_start;
            if first_of_pass && v.pass == 0 {
                self.ink_copy(is.prelude.ptr, x, n * e);
                prelude_blocks.clone_from(&blocks);
            } else if first_of_pass {
                bank = blocks[prelude_blocks.len() + 1..].to_vec();
            }
            let msgs = match (&layer.looped, v.pass) {
                (Some(lp), 1..) => {
                    // the bank's states, then the previous pass's output,
                    // normalized at the pass's first layer
                    let states: Vec<usize> = bank
                        .iter()
                        .rev()
                        .copied()
                        .chain([plan.bank_out()])
                        .collect();
                    self.ink_mix(
                        s,
                        &states,
                        first_of_pass.then_some(x),
                        Some((&lp.msg_query, 4)),
                        Some(MixOut {
                            out: is.msg.ptr,
                            stride: b * e,
                            copies: 1,
                            norm: None,
                        }),
                        all,
                    );
                    if first_of_pass {
                        self.ink_copy(x, is.prelude.ptr, n * e);
                        blocks.clone_from(&prelude_blocks);
                    }
                    Some(lp)
                }
                _ => None,
            };
            let push = v
                .layer
                .is_multiple_of(cfg.block)
                .then(|| plan.push(cfg, v.layer, v.pass, pushed));
            pushed += push.is_some() as usize;
            let seg = Seg {
                st: &st[v.slot],
                n,
                params,
                slots,
                save,
            };
            self.ink_layer(s, ink, layer, &seg, &mut blocks, push, msgs);
        }

        // the rows with logits: the output mix, normalized, in `feat`
        if rows > 0 {
            let states: Vec<usize> = blocks.iter().rev().copied().chain([plan.cur()]).collect();
            self.ink_mix(
                s,
                &states,
                Some(x),
                Some((&ink.output_query, 1)),
                Some(MixOut {
                    out: s.feat.ptr,
                    stride: 0,
                    copies: 1,
                    norm: Some(self.output_norm.ptr),
                }),
                Rows {
                    n: rows,
                    in_row0: n - rows,
                    out_row0: 0,
                },
            );
            let feat = Act::new(s.feat.ptr, rows, e, &s.q8[1]);
            self.matmul(&self.output, &feat, s.logits.ptr);
            if cfg.softcap != 0.0 {
                let count = rows * c.n_vocab;
                let args = CapArgs {
                    z: s.logits.ptr,
                    n: count as i32,
                    cap: cfg.softcap,
                    inv: 1.0 / cfg.softcap,
                    pad: 0,
                };
                self.d.launch(
                    self.ink().softcap,
                    [count.div_ceil(256) as u32, 1, 1],
                    [256, 1, 1],
                    0,
                    &args,
                );
            }
        }
        self.stage_end();
    }

    /// One visit of `layer` over the stream `s.x`: with `push`, the first
    /// layer of a block, whose normalized input joins `blocks` in that region
    /// of the stack; with `lp`, a looped layer of a pass after the first, its
    /// messages in `msg`
    #[allow(clippy::too_many_arguments)]
    fn ink_layer(
        &self,
        s: &Scratch,
        ink: &GpuInkling,
        layer: &GpuInkLayer,
        g: &Seg<'_>,
        blocks: &mut Vec<usize>,
        push: Option<usize>,
        lp: Option<&GpuLoop>,
    ) {
        let c = self.cfg();
        let cfg = c.inkling.as_ref().expect("an inkling config");
        let (e, nkv, taps) = (c.n_embd, c.n_head_kv, cfg.conv);
        let (qd, kv_dim) = (c.n_head * c.head_dim, c.kv_dim());
        let is = ink_scratch(s);
        let plan = ink.stack;
        let (b, n) = (s.batch, g.n);
        let x = s.x.ptr;
        let rows = Rows::all(n);
        let inp = |j: usize| is.inp.f32_at(j * b * e);
        let norm = |j: usize| layer.norms.f32_at(j.min(layer.n_norms - 1) * e);
        let msg = |j: usize| is.msg.f32_at(j * b * e);

        // the attention inputs: depth mixes over the blocks and the
        // normalized stream (or the stream itself in the first layer),
        // controlled by the messages, normalized
        let cur = push.unwrap_or(plan.cur());
        let states: Vec<usize> = blocks.iter().rev().copied().chain([cur]).collect();
        let n_in = if !layer.swa || lp.is_some() { 3 } else { 1 };
        match &layer.attn_query {
            Some(q) => self.ink_mix(
                s,
                &states,
                Some(x),
                Some((q, layer.n_q)),
                Some(MixOut {
                    out: inp(0),
                    stride: b * e,
                    copies: n_in / layer.n_q,
                    norm: if lp.is_some() { None } else { Some(norm(0)) },
                }),
                rows,
            ),
            None => {
                self.ink_mix(s, &[cur], Some(x), None, None, rows);
                for j in 0..n_in {
                    match lp {
                        None => self.ink_norm(x, norm(j), inp(j), n),
                        Some(_) => self.ink_copy(inp(j), x, n * e),
                    }
                }
            }
        }
        if let Some(lp) = lp {
            for j in 0..3 {
                self.ink_control(s, lp, j, inp(j), msg(j), inp(j), n, norm(j));
            }
        }

        // the projections, of each input once quantized
        let dyn_rows = nkv * taps;
        let (k_dyn, v_dyn) = (is.dyn_.ptr, is.dyn_.f32_at(b * dyn_rows));
        let (ik, iv) = (1.min(n_in - 1), 2.min(n_in - 1));
        let mut jobs: Vec<(usize, &GpuWeight, DevPtr)> = vec![
            (0, &layer.wq, s.q.ptr),
            (0, &layer.wrel, is.r.ptr),
            (0, &layer.wsink, is.sink.ptr),
            (0, &layer.wgate, s.a1.ptr),
            (ik, &layer.wk, s.k.ptr),
        ];
        jobs.extend(layer.k_dyn.as_ref().map(|w| (ik, w, k_dyn)));
        jobs.push((iv, &layer.wv, s.v.ptr));
        jobs.extend(layer.v_dyn.as_ref().map(|w| (iv, w, v_dyn)));
        for j in 0..n_in {
            let ws: Vec<(&GpuWeight, DevPtr)> = jobs
                .iter()
                .filter(|job| job.0 == j)
                .map(|&(_, w, out)| (w, out))
                .collect();
            let a = Act::new(inp(j), n, e, &s.q8[0]);
            for group in ws.chunks(GROUP) {
                self.matmuls(group, &a);
            }
        }

        // the keys' and values' convs, in place, or of long passes into the
        // free attention inputs
        let par = g.par();
        let (kin, vin) = (s.k.ptr, s.v.ptr);
        let (k, v) = if par { (inp(0), inp(1)) } else { (kin, vin) };
        let kd = layer.k_dyn.as_ref().map(|_| k_dyn);
        let vd = layer.v_dyn.as_ref().map(|_| v_dyn);
        self.ink_conv(g, kin, &layer.k_conv, k, kd, kv_dim, 0, false);
        self.ink_conv(g, vin, &layer.v_conv, v, vd, kv_dim, kv_dim, false);
        if par {
            self.ink_conv_state(g, kin, kv_dim, 0);
            self.ink_conv_state(g, vin, kv_dim, kv_dim);
        }
        self.ink_attention(s, layer, g, k, v);
        let o = Act::new(s.o.ptr, n, qd, &s.q8[1]);
        self.matmul(&layer.wo, &o, s.y.ptr);

        // the output's conv, the residual's start at the first layer of a block
        let y = s.y.ptr;
        self.ink_conv(
            g,
            y,
            &layer.post_conv,
            x,
            None,
            e,
            2 * kv_dim,
            push.is_none(),
        );
        if par {
            self.ink_conv_state(g, y, e, 2 * kv_dim);
        }
        if let Some(p) = push {
            blocks.push(p);
        }

        // the feed-forward input: the depth mix with the stream, controlled,
        // normalized
        let states: Vec<usize> = blocks.iter().rev().copied().chain([plan.cur()]).collect();
        let h = s.h.ptr;
        self.ink_mix(
            s,
            &states,
            Some(x),
            Some((&layer.ffn_query, 1)),
            Some(MixOut {
                out: h,
                stride: 0,
                copies: 1,
                norm: if lp.is_some() {
                    None
                } else {
                    Some(layer.ffn_norm.ptr)
                },
            }),
            rows,
        );
        if let Some(lp) = lp {
            self.ink_control(s, lp, 3, h, msg(3), h, n, layer.ffn_norm.ptr);
        }
        self.ink_moe(s, &layer.moe, n);
        self.ink_conv(g, y, &layer.ffn_post_conv, x, None, e, 2 * kv_dim + e, true);
        if par {
            self.ink_conv_state(g, y, e, 2 * kv_dim + e);
        }
    }

    /// The depth mix over the stack regions `states` (the last one with
    /// `cur`, the stream normalized into it first), of each of the `n_q`
    /// queries of `query`
    fn ink_mix(
        &self,
        s: &Scratch,
        states: &[usize],
        cur: Option<DevPtr>,
        query: Option<(&Buf, usize)>,
        out: Option<MixOut>,
        rows: Rows,
    ) {
        let c = self.cfg();
        let e = c.n_embd;
        let is = ink_scratch(s);
        assert!(
            states.len() <= MIX_STATES,
            "a depth mix of {} states",
            states.len()
        );
        let mut off = [0i32; MIX_STATES];
        for (o, &r) in off.iter_mut().zip(states) {
            *o = (r * s.batch * e) as i32;
        }
        let stack = is.stack.ptr;
        let (query, n_q) = query.map_or((stack, 0), |(q, n)| (q.ptr, n));
        assert!(n_q <= MIX_QUERIES);
        let args = MixArgs {
            stack,
            cur: cur.unwrap_or(stack),
            query,
            w: out.as_ref().and_then(|o| o.norm).unwrap_or(stack),
            out: out.as_ref().map_or(stack, |o| o.out),
            e: e as i32,
            n_states: states.len() as i32,
            n_q: n_q as i32,
            copies: out.as_ref().map_or(1, |o| o.copies) as i32,
            scale: 1.0 / (e as f32).sqrt(),
            eps: MIX_EPS,
            norm_cur: cur.is_some() as i32,
            out_norm: out.as_ref().is_some_and(|o| o.norm.is_some()) as i32,
            out_eps: c.eps,
            in_row0: rows.in_row0 as i32,
            out_row0: rows.out_row0 as i32,
            ostride: out.as_ref().map_or(0, |o| o.stride) as i32,
            off,
        };
        self.d.launch(
            self.ink().mix,
            [rows.n as u32, 1, 1],
            [MIX_THREADS, 1, 1],
            0,
            &args,
        );
    }

    /// rms norm without weight of `rows` rows of `width`
    fn ink_rms(&self, x: DevPtr, out: DevPtr, rows: usize, width: usize, ostride: usize) {
        let args = RmsArgs {
            x,
            out,
            n: width as i32,
            eps: MIX_EPS,
            istride: width as i32,
            ostride: ostride as i32,
        };
        self.d
            .launch(self.ink().rms, [rows as u32, 1, 1], [256, 1, 1], 0, &args);
    }

    /// [`CudaEngine::rms_norm`] of rows of the stream with the weight `w`
    fn ink_norm(&self, x: DevPtr, w: DevPtr, out: DevPtr, rows: usize) {
        let e = self.cfg().n_embd;
        let args = NormArgs {
            x,
            w,
            out,
            gate: x,
            y: x,
            n: e as i32,
            ostride: e as i32,
            gated: 0,
            add: 0,
            eps: self.cfg().eps,
            pad: 0,
        };
        let threads = if e > 1024 { 256 } else { 32 };
        self.d.launch(
            self.k.rms_norm,
            [rows as u32, 1, 1],
            [threads, 1, 1],
            0,
            &args,
        );
    }

    fn ink_copy(&self, dst: DevPtr, src: DevPtr, n: usize) {
        self.elementwise(self.ink().copy, dst, src, n);
    }

    /// Controller `j` of `lp` mixing the message `m` into the input `x`, the
    /// result going to `out` (which may be `x`) rms-normalized with the
    /// weight `norm`: from `z = down · [rms(x) ‖ rms(m)]`, `x - tanh(forget ·
    /// z) * x + tanh(gain · z) * m`
    #[allow(clippy::too_many_arguments)]
    fn ink_control(
        &self,
        s: &Scratch,
        lp: &GpuLoop,
        j: usize,
        x: DevPtr,
        m: DevPtr,
        out: DevPtr,
        n: usize,
        norm: DevPtr,
    ) {
        let c = self.cfg();
        let (e, rank) = (
            c.n_embd,
            c.inkling.as_ref().expect("an inkling config").gate_rank,
        );
        let is = ink_scratch(s);
        self.ink_rms(x, is.u.ptr, n, e, 2 * e);
        self.ink_rms(m, is.u.f32_at(e), n, e, 2 * e);
        let u = Act::new(is.u.ptr, n, 2 * e, &s.q8[1]);
        self.matmul(&lp.down[j], &u, is.z.ptr);
        let z = Act::new(is.z.ptr, n, rank, &s.q8[1]);
        self.matmuls(
            &[(&lp.gain[j], is.gain.ptr), (&lp.forget[j], is.forget.ptr)],
            &z,
        );
        let args = CtrlArgs {
            x,
            g: is.gain.ptr,
            f: is.forget.ptr,
            m,
            out,
            n: (n * e) as i32,
            pad: 0,
        };
        self.d.launch(
            self.ink().control,
            [(n * e).div_ceil(256) as u32, 1, 1],
            [256, 1, 1],
            0,
            &args,
        );
        self.ink_norm(out, norm, out, n);
    }

    fn conv_args(
        &self,
        g: &Seg<'_>,
        x: DevPtr,
        out: DevPtr,
        width: usize,
        offset: usize,
    ) -> InkConvArgs {
        let c = self.cfg();
        let ink = c.inkling.as_ref().expect("an inkling config");
        InkConvArgs {
            x,
            w: 0,
            state: g.st.conv.ptr,
            out,
            dyn_: x,
            params: g.params,
            width: width as i32,
            n: g.n as i32,
            channels: ink.conv_channels(c.kv_dim(), c.n_embd) as i32,
            offset: offset as i32,
            slots: g.slots as i32,
            save: g.save as i32,
            add: 0,
            has_dyn: 0,
            head: c.head_dim as i32,
            dyn_stride: (c.n_head_kv * ink.conv) as i32,
            scale: 0.0,
            pad: 0,
        }
    }

    /// The short conv of the rows of `x` (`width` channels) into `out`,
    /// added to it with `add`, at channel `offset` of the slot's state, with
    /// the dynamic taps `dyn_` when given
    #[allow(clippy::too_many_arguments)]
    fn ink_conv(
        &self,
        g: &Seg<'_>,
        x: DevPtr,
        w: &Buf,
        out: DevPtr,
        dyn_: Option<DevPtr>,
        width: usize,
        offset: usize,
        add: bool,
    ) {
        let taps = CONV_TAPS;
        let mult = if dyn_.is_some() { 2 } else { 1 };
        let args = InkConvArgs {
            w: w.ptr,
            dyn_: dyn_.unwrap_or(x),
            add: add as i32,
            has_dyn: dyn_.is_some() as i32,
            scale: 1.0 / ((mult * taps + 1) as f32).sqrt(),
            ..self.conv_args(g, x, out, width, offset)
        };
        if g.par() {
            assert!(x != out, "a parallel conv in place");
            self.d.launch(
                self.ink().conv_par,
                [width.div_ceil(256) as u32, g.n as u32, 1],
                [256, 1, 1],
                0,
                &args,
            );
        } else {
            self.d.launch(
                self.ink().conv,
                [width.div_ceil(128) as u32, 1, 1],
                [128, 1, 1],
                0,
                &args,
            );
        }
    }

    /// The state update after [`CudaEngine::ink_conv`]'s of a parallel conv:
    /// the last 3 inputs `x` of `width` channels at `offset`
    fn ink_conv_state(&self, g: &Seg<'_>, x: DevPtr, width: usize, offset: usize) {
        let args = self.conv_args(g, x, x, width, offset);
        self.d.launch(
            self.ink().conv_state,
            [width.div_ceil(256) as u32, 1, 1],
            [256, 1, 1],
            0,
            &args,
        );
    }

    /// The attention of the queries in `s.q` over the slot's caches, which
    /// first take the keys `k` and values `v`, into `s.o`, gated by `s.a1`
    fn ink_attention(&self, s: &Scratch, layer: &GpuInkLayer, g: &Seg<'_>, k: DevPtr, v: DevPtr) {
        let c = self.cfg();
        let cfg = c.inkling.as_ref().expect("an inkling config");
        let is = ink_scratch(s);
        let (nh, nkv, hd, n) = (c.n_head, c.n_head_kv, c.head_dim, g.n);
        let splits = attn_splits(c, n);
        let args = InkAttnArgs {
            q: s.q.ptr,
            k,
            v,
            r: is.r.ptr,
            q_clip: layer.q_clip.ptr,
            r_clip: layer.rel_clip.ptr,
            k_cache: g.st.k.ptr,
            v_cache: g.st.v.ptr,
            sink: is.sink.ptr,
            rel_bias: layer.rel_bias.ptr,
            gate: s.a1.ptr,
            out: s.o.ptr,
            part_o: is.part_o.ptr,
            part_ml: is.part_ml.ptr,
            params: g.params,
            n_head: nh as i32,
            n_kv: nkv as i32,
            hd: hd as i32,
            n_rel: cfg.n_rel as i32,
            eps: c.eps,
            scale: 1.0 / (hd as f32).sqrt(),
            clip: cfg.qk_clip,
            rel_clip: cfg.rel_clip,
            full: !layer.swa as i32,
            floor: cfg.log_scale_floor as i32,
            alpha: cfg.log_scale_alpha,
            window: if layer.swa {
                c.sliding_window as i32
            } else {
                0
            },
            mask: g.st.ring.map_or(-1, |r| r as i32 - 1),
            ext: layer.ext as i32,
            splits: splits as i32,
            pad: 0,
        };
        let ink = self.ink();
        self.d.launch(
            ink.attn_prep,
            [(2 * nh + nkv) as u32, n as u32, 1],
            [32, 1, 1],
            0,
            &args,
        );
        self.d.launch(
            ink.attn,
            [nh as u32, n as u32, splits as u32],
            [(32 * ATTN_WARPS) as u32, 1, 1],
            0,
            &args,
        );
        if splits > 1 {
            self.d.launch(
                ink.attn_reduce,
                [nh as u32, n as u32, 1],
                [hd as u32, 1, 1],
                0,
                &args,
            );
        }
    }

    /// The mixture of experts of the normalized `n` tokens in `h` into `y`,
    /// with SITU experts, see [`CudaEngine::moe`]
    fn ink_moe(&self, s: &Scratch, m: &GpuMoe, n: usize) {
        let c = self.cfg();
        let cfg = c.inkling.as_ref().expect("an inkling config");
        let mc = c.moe.as_ref().expect("a moe config");
        let ms = s.moe.as_ref().expect("the moe buffers");
        let (e, k) = (c.n_embd, mc.n_expert_used);
        let pairs = n * k;
        let h = Act::new(s.h.ptr, n, e, &s.q8[0]);
        let (tiles, mma) = self.route(s, m, &h);
        let ff = mc.n_ff_exp;
        self.matmul_id(&m.gate, ms, &h, k, tiles, mma, ms.a1.ptr);
        self.matmul_id(&m.up, ms, &h, k, tiles, mma, ms.a2.ptr);
        let (beta, lb) = (cfg.situ_beta, cfg.situ_linear_beta);
        let count = pairs * ff;
        let args = SituArgs {
            gate: ms.a1.ptr,
            up: ms.a2.ptr,
            n: count as i32,
            beta,
            ib: 1.0 / beta,
            lb,
            ilb: if lb > 0.0 { 1.0 / lb } else { 0.0 },
            pad: 0,
        };
        self.d.launch(
            self.ink().situ,
            [count.div_ceil(256) as u32, 1, 1],
            [256, 1, 1],
            0,
            &args,
        );
        let a = Act::new(ms.a1.ptr, pairs, ff, &s.q8[1]);
        self.matmul_id(&m.down, ms, &a, 1, tiles, mma, ms.out.ptr);
        let args = CombineArgs {
            x: s.y.ptr,
            out: ms.out.ptr,
            weights: ms.weights.ptr,
            shared: 0,
            gate: 0,
            n_embd: e as i32,
            k: k as i32,
            mode: 0,
            pad: 0,
        };
        self.d.launch(
            self.ink().moe_sum,
            [e.div_ceil(256) as u32, n as u32, 1],
            [256, 1, 1],
            0,
            &args,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Inkling-2-Mini's geometry
    fn mini() -> Config {
        let n_layer = 24;
        Config {
            arch: crate::model::Arch::Inkling2Loop,
            n_layer,
            n_embd: 2304,
            n_ff: 0,
            n_vocab: 200058,
            n_head: 24,
            n_head_kv: 8,
            head_dim: 128,
            n_rot: 0,
            rope_base: 10_000.0,
            eps: 1e-6,
            context_length: 4096,
            recurrent: vec![false; n_layer],
            swa: (0..n_layer).map(|i| i % 6 != 5).collect(),
            sliding_window: 1024,
            rope: vec![false; n_layer],
            attn_gate: false,
            attn_sinks: false,
            conv_kernel: 0,
            ssm_state: 0,
            ssm_k_heads: 0,
            ssm_v_heads: 0,
            moe: Some(crate::model::MoeConfig {
                n_expert: 128,
                n_expert_used: 8,
                n_ff_exp: 1152,
                n_ff_shexp: 0,
                gating: crate::model::Gating::Sigmoid,
                norm_weights: true,
                weights_scale: 8.0,
                shared_gate: false,
            }),
            n_dense_lead: 0,
            inkling: Some(InklingConfig {
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
            }),
        }
    }

    #[test]
    fn check_takes_the_geometry_of_the_kernels() {
        let ok = |c: &Config| check(c, c.inkling.as_ref().unwrap()).is_ok();
        assert!(ok(&mini()));
        // 48 layers in blocks of 6: an output mix of 9 states
        let mut deep = mini();
        deep.n_layer = 48;
        let ink = deep.inkling.as_mut().unwrap();
        (ink.loop_start, ink.loop_end) = (12, 36);
        assert!(!ok(&deep));
        // keys wider than the stream
        let mut wide = mini();
        (wide.n_head, wide.n_head_kv) = (24, 24);
        assert!(!ok(&wide));
        // heads of 64 are taken, of 80 or 512 not
        let mut c = mini();
        c.head_dim = 64;
        assert!(ok(&c));
        c.head_dim = 80;
        assert!(!ok(&c));
        c.head_dim = 512;
        assert!(!ok(&c));
    }

    #[test]
    fn short_passes_split_the_positions() {
        let c = mini();
        // a decode step: 11 ranges of each of the 24 heads
        assert_eq!(attn_splits(&c, 1), 11);
        assert_eq!(attn_splits(&c, 8), 2);
        // a prefill: one
        assert_eq!(attn_splits(&c, 9), 1);
        assert_eq!(attn_splits(&c, 512), 1);
        // the partials hold the largest of the short passes'
        let most = (1..=NT).map(|n| n * attn_splits(&c, n)).max().unwrap();
        assert_eq!(attn_parts(&c, 512), most);
        assert_eq!(attn_parts(&c, 1), 11);
    }

    #[test]
    fn the_core_experts_go_first() {
        let c = mini();
        let ink = c.inkling.as_ref().unwrap();
        let order = experts_order(ink, c.n_layer);
        assert_eq!(&order[..12], &(6..18).collect::<Vec<_>>()[..]);
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(sorted, (0..24).collect::<Vec<_>>());
    }
}
