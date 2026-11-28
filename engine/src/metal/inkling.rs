//! The Metal path of `inkling2_loop` (Inkling-2), the GPU version of
//! [`crate::model::inkling`], whose math the kernels of `inkling.metal`
//! follow.
//!
//! A pass runs the visits of the layers in order over the stream `x` of all
//! its rows (the tokens of its sequences), as the CPU does: the depth mixes,
//! controllers, projections and the mixture of experts over all the rows at
//! once, the short convs and the attention per sequence. The normalized
//! blocks the mixes read are rows of `InkScratch::stack`, each block in a
//! region of its own: the prelude's, the core's of the current pass and of
//! the previous one (the bank of the messages), the epilogue's, plus the
//! normalized stream and the previous pass's normalized output.
//!
//! Each visit (slot) of a sequence has its kv caches, f16 (a ring on the
//! sliding window layers, see [`Shape::metal_ring`]), and its conv state
//! `[slots][taps - 1][channels]`, whose copies serve verify passes as the
//! delta nets' do (see `ssm_conv` in the shaders).

use super::*;
use crate::model::InklingConfig;
use crate::model::inkling::{InkLayer, Inkling, MIX_EPS, StackPlan, Visit};

/// The taps of the short convs the kernel takes
const CONV_TAPS: usize = 4;
/// Head size of the attention kernels
const HEAD_DIM: usize = 128;
/// Relative position channels per head the flash kernel takes
const N_REL: usize = 16;
/// Most states of a depth mix, and its most queries
const MIX_STATES: usize = 8;
const MIX_QUERIES: usize = 4;
/// Depth mixes of up to this many rows run in chunks of `MIX_CHUNK`
/// elements of the rows, a threadgroup each, in three dispatches (one
/// threadgroup per row reads its states at a core's bandwidth only: on an M2
/// Max ~25 us for a decode step's mix of 6 states and 3 queries); at most
/// `MIX_CHUNKS` per row, of `MIX_PART` partial sums each
const MIX_CHUNK_ROWS: usize = 16;
const MIX_CHUNK: usize = 256;
const MIX_CHUNKS: usize = 16;
const MIX_PART: usize = 40;
/// (query heads per kv head, simdgroups per threadgroup) of the
/// `ink_attn_split` kernels, see the instantiations in the shaders
const SPLIT_GROUPS: &[(usize, usize)] = &[(1, 2), (2, 2), (3, 4), (4, 2), (6, 3), (8, 4)];

#[repr(C)]
struct MixArgs {
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
    n: i32,
    eps: f32,
    istride: i32,
    ostride: i32,
}

#[repr(C)]
struct InkConvArgs {
    width: i32,
    n: i32,
    channels: i32,
    offset: i32,
    slot: i32,
    slots: i32,
    save: i32,
    add: i32,
    dyn_: i32,
    head: i32,
    dyn_stride: i32,
    scale: f32,
}

#[repr(C)]
struct InkAttnArgs {
    n_head: i32,
    n_kv: i32,
    hd: i32,
    n_rel: i32,
    pos0: i32,
    n: i32,
    eps: f32,
    scale: f32,
    clip: f32,
    rel_clip: f32,
    full: i32,
    floor: i32,
    alpha: f32,
    splits: i32,
    chunk: i32,
    window: i32,
    lo: i32,
    mask: i32,
    ext: i32,
}

#[repr(C)]
struct SituArgs {
    n: i32,
    beta: f32,
    ib: f32,
    lb: f32,
    ilb: f32,
}

#[repr(C)]
struct GateUpArgs {
    k: i32,
    rows: i32,
    x_div: i32,
    pad: i32,
    w_row_bytes: u64,
    x_row_bytes: u64,
    stride: u64,
    situ: SituArgs,
}

#[repr(C)]
struct CtrlArgs {
    e: i32,
    rank: i32,
    eps: f32,
    out_eps: f32,
    stride: i32,
    chunks: i32,
    w_stride: i32,
    pad: i32,
}

/// Passes of up to this many tokens run the controllers through
/// `ink_ctrl_*`, a threadgroup per row of the bottleneck and per 256
/// channels of the outputs, the input norms folded into the bottleneck's
/// dot products and the output norm's sums of squares into the outputs'
const CTRL_FAST_ROWS: usize = 8;

#[repr(C)]
struct CapArgs {
    n: i32,
    cap: f32,
    inv: f32,
}

/// Whether the Metal kernels take the model's geometry
pub(super) fn check(c: &Config, ink: &InklingConfig) -> Result<(), Error> {
    let group = c.n_head / c.n_head_kv;
    if c.head_dim != HEAD_DIM
        || ink.n_rel != N_REL
        || ink.conv != CONV_TAPS
        || !SPLIT_GROUPS.iter().any(|g| g.0 == group)
    {
        return Err(backend(
            "metal: inkling2_loop needs heads of 128, 16 relative position channels, convs of 4 taps \
             and 1, 2, 3, 4, 6 or 8 query heads per kv head",
        ));
    }
    // the output's depth mix: all the blocks and the stream
    let plan = StackPlan::new(ink, c.n_layer);
    if plan.prelude + plan.core + plan.epilogue + 1 > MIX_STATES {
        return Err(backend(format!(
            "metal: inkling2_loop's depth mixes take at most {MIX_STATES} states"
        )));
    }
    // the long convs' keys and values go to rows of the attention inputs
    // (`InkScratch::inp`), the output gate to `Scratch::a2`
    if c.kv_dim() > c.n_embd || c.n_head * c.head_dim > ffn_width(c).max(c.d_inner()) {
        return Err(backend(
            "metal: inkling2_loop needs keys no wider than the stream and query heads no wider \
             than the chosen experts",
        ));
    }
    Ok(())
}

/// The controllers of a looped layer, see [`crate::model::inkling::Loop`]
struct GpuLoop {
    msg_query: Buffer,
    down: [GpuWeight; 4],
    gain: [GpuWeight; 4],
    forget: [GpuWeight; 4],
}

/// A physical layer, see [`InkLayer`]
struct GpuInkLayer {
    swa: bool,
    /// the depth mix queries of the attention inputs, and how many
    attn_query: Option<Buffer>,
    n_q: usize,
    /// `[n_norms][n_embd]`: one for all inputs, or one each
    norms: Buffer,
    n_norms: usize,
    wq: GpuWeight,
    wk: GpuWeight,
    wv: GpuWeight,
    wo: GpuWeight,
    wgate: GpuWeight,
    wsink: GpuWeight,
    wrel: GpuWeight,
    q_clip: Buffer,
    rel_clip: Buffer,
    /// `[ext][n_rel]`
    rel_bias: Buffer,
    ext: usize,
    k_conv: Buffer,
    v_conv: Buffer,
    post_conv: Buffer,
    ffn_post_conv: Buffer,
    k_dyn: Option<GpuWeight>,
    v_dyn: Option<GpuWeight>,
    ffn_query: Buffer,
    ffn_norm: Buffer,
    moe: GpuMoe,
    looped: Option<GpuLoop>,
}

/// The model on the GPU, see [`Inkling`]
pub(super) struct GpuInkling {
    tok_norm: Buffer,
    output_query: Buffer,
    layers: Vec<GpuInkLayer>,
    visits: Vec<Visit>,
    /// simdgroups of the `ink_attn_split` kernel of the model's group
    split_nsg: usize,
    stack: StackPlan,
}

fn upload_all(device: &Device, parts: &[&[f32]]) -> Result<Buffer, Error> {
    upload(device, &parts.concat())
}

/// The up experts, of a fused `ffn_gate_up_exps` (`up` shares the data of
/// `gate`) in the mapping of the gate's, which a second one would count
/// again in the device's allocated size
fn wrap_up(
    device: &Device,
    g: &GpuExperts,
    gate: &Experts<'_>,
    up: &Experts<'_>,
) -> Result<GpuExperts, Error> {
    if !core::ptr::eq(gate.data, up.data) {
        return wrap_experts(device, up);
    }
    Ok(GpuExperts {
        buf: g.buf.clone(),
        offset: g.offset - gate.offset + up.offset,
        ty: up.ty,
        cols: up.cols,
        rows: up.rows,
        row_bytes: up.row_bytes(),
        stride: up.stride,
    })
}

impl GpuInkling {
    pub(super) fn new(device: &Device, c: &Config, m: &Inkling<'_>) -> Result<Self, Error> {
        let ink = c.inkling.as_ref().expect("an inkling config");
        let layer = |l: &InkLayer<'_>| -> Result<GpuInkLayer, Error> {
            let moe = &l.moe;
            let gate = wrap_experts(device, &moe.gate)?;
            let up = wrap_up(device, &gate, &moe.gate, &moe.up)?;
            let looped = match &l.looped {
                Some(lp) => {
                    let each = |w: &[Weight<'_>; 4]| -> Result<[GpuWeight; 4], Error> {
                        Ok([
                            wrap(device, &w[0])?,
                            wrap(device, &w[1])?,
                            wrap(device, &w[2])?,
                            wrap(device, &w[3])?,
                        ])
                    };
                    Some(GpuLoop {
                        msg_query: upload(device, &lp.msg_query)?,
                        down: each(&lp.down)?,
                        gain: each(&lp.gain)?,
                        forget: each(&lp.forget)?,
                    })
                }
                None => None,
            };
            let norms: Vec<&[f32]> = l.norms.iter().map(Vec::as_slice).collect();
            Ok(GpuInkLayer {
                swa: l.swa,
                n_q: l.attn_query.as_ref().map_or(0, |q| q.len() / c.n_embd),
                attn_query: match &l.attn_query {
                    Some(q) => Some(upload(device, q)?),
                    None => None,
                },
                norms: upload_all(device, &norms)?,
                n_norms: norms.len(),
                wq: wrap(device, &l.wq)?,
                wk: wrap(device, &l.wk)?,
                wv: wrap(device, &l.wv)?,
                wo: wrap(device, &l.wo)?,
                wgate: wrap(device, &l.wgate)?,
                wsink: wrap(device, &l.wsink)?,
                wrel: wrap(device, &l.wrel)?,
                q_clip: upload(device, &l.q_clip)?,
                rel_clip: upload(device, &l.rel_clip)?,
                rel_bias: upload(device, &l.rel_bias)?,
                ext: l.rel_bias.len() / ink.n_rel,
                k_conv: upload(device, &l.k_conv)?,
                v_conv: upload(device, &l.v_conv)?,
                post_conv: upload(device, &l.post_conv)?,
                ffn_post_conv: upload(device, &l.ffn_post_conv)?,
                k_dyn: match &l.k_dyn {
                    Some(w) => Some(wrap(device, w)?),
                    None => None,
                },
                v_dyn: match &l.v_dyn {
                    Some(w) => Some(wrap(device, w)?),
                    None => None,
                },
                ffn_query: upload(device, &l.ffn_query)?,
                ffn_norm: upload(device, &l.ffn_norm)?,
                moe: GpuMoe {
                    router: wrap(device, &moe.router)?,
                    bias: match &moe.bias {
                        Some(b) => Some(upload(device, b)?),
                        None => None,
                    },
                    gate,
                    up,
                    down: wrap_experts(device, &moe.down)?,
                    shared: None,
                },
                looped,
            })
        };
        let group = c.n_head / c.n_head_kv;
        Ok(GpuInkling {
            tok_norm: upload(device, &m.tok_norm)?,
            output_query: upload(device, &m.output_query)?,
            layers: m.layers.iter().map(layer).collect::<Result<_, _>>()?,
            visits: m.visits.clone(),
            split_nsg: SPLIT_GROUPS
                .iter()
                .find(|g| g.0 == group)
                .map(|g| g.1)
                .expect("checked by `check`"),
            stack: StackPlan::new(ink, c.n_layer),
        })
    }
}

/// The buffers of a pass of `inkling2_loop` beyond [`Scratch`]'s
pub(super) struct InkScratch {
    /// the normalized blocks, see [`StackPlan`]: regions of `batch` rows
    stack: Buffer,
    /// the prelude's output, which each pass of the core starts from
    prelude: Buffer,
    /// `[3][batch][n_embd]`: the attention inputs
    inp: Buffer,
    /// `[4][batch][n_embd]`: a looped layer's messages
    msg: Buffer,
    /// per controller of a step (3): its input pair, bottleneck, gain and
    /// forget logits
    u: Buffer,
    z: Buffer,
    gain: Buffer,
    forget: Buffer,
    /// relative position channels and sink logits of the heads
    r: Buffer,
    sink: Buffer,
    /// `[2][batch][n_head_kv * taps]`: the keys' and values' dynamic taps
    dyn_: Buffer,
    /// `[MIX_CHUNK_ROWS][MIX_CHUNKS][MIX_PART]`: a chunked depth mix's sums
    mix_part: Buffer,
}

/// Bytes of the buffers of [`InkScratch`], in the order of its fields, for
/// passes of `b` tokens; none for the other archs
pub(super) fn scratch_bytes(c: &Config, b: usize) -> Vec<usize> {
    let Some(ink) = &c.inkling else {
        return Vec::new();
    };
    let e = c.n_embd;
    let regions = StackPlan::new(ink, c.n_layer).regions();
    let f32s = |n: usize| 4 * n;
    vec![
        f32s(regions * b * e),
        f32s(b * e),
        f32s(3 * b * e),
        f32s(4 * b * e),
        f32s(3 * b * 2 * e),
        f32s(3 * b * ink.gate_rank),
        f32s(3 * b * e),
        f32s(3 * b * e),
        f32s(b * c.n_head * ink.n_rel),
        f32s(b * c.n_head),
        f32s(2 * b * c.n_head_kv * ink.conv),
        f32s(MIX_CHUNK_ROWS * MIX_CHUNKS * MIX_PART),
    ]
}

impl InkScratch {
    pub(super) fn new(
        e: &MetalEngine<'_>,
        a: &mut Alloc<'_>,
        b: usize,
    ) -> Result<Option<Self>, Error> {
        if e.ink.is_none() {
            return Ok(None);
        }
        let mut bytes = scratch_bytes(e.cfg(), b).into_iter();
        let mut next = || a.zeroed(bytes.next().expect("a size per buffer"));
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
            mix_part: next()?,
        }))
    }
}

/// The state of a slot of a sequence: its f16 kv caches (with `ring`, a
/// sliding window's, position `p` at row `p % ring`), and `slots` copies of
/// its conv state
pub(super) struct InkSlot {
    k: Buffer,
    v: Buffer,
    ring: Option<usize>,
    conv: Buffer,
    swa: bool,
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

    /// [`Shape::metal_seq`] of `inkling2_loop`
    pub(super) fn inkling_metal_seq(
        &self,
        ink: &InklingConfig,
        context: usize,
        draft: usize,
        max_batch: usize,
    ) -> SessionMemory {
        let c = self.cfg;
        let slots = self.metal_draft(draft, max_batch) + 1;
        let (swa, full) = self.ink_slots(ink);
        let ring = self.metal_ring(context, max_batch);
        let window_bytes = swa * 2 * pages(self.metal_window_bytes(context, max_batch));
        let conv = (swa + full) * pages(slots * self.ink_conv_bytes(ink));
        SessionMemory {
            kv: full * 2 * pages(kv_bytes(context, self.kv_dim()))
                + if ring.is_none() { window_bytes } else { 0 },
            recurrent: conv + if ring.is_some() { window_bytes } else { 0 },
            draft_head: 0,
            scratch: 0,
            transient: 4 * slots * c.n_vocab,
            kv_per_token: (full + if ring.is_none() { swa } else { 0 }) * 2 * 2 * self.kv_dim(),
        }
    }
}

impl Seq {
    /// [`Seq::new`] of `inkling2_loop`: a state per slot
    pub(super) fn new_inkling(
        e: &MetalEngine<'_>,
        context: usize,
        draft: usize,
        max_batch: usize,
    ) -> Result<Self, Error> {
        let c = e.cfg();
        let ink = c.inkling.as_ref().expect("an inkling config");
        let shape = e.model.shape();
        let memory = shape.metal_seq(context, draft, max_batch);
        let draft = shape.metal_draft(draft, max_batch);
        let slots = draft + 1;
        let (ring, window_bytes) = (
            shape.metal_ring(context, max_batch),
            shape.metal_window_bytes(context, max_batch),
        );
        let full_bytes = kv_bytes(context, shape.kv_dim());
        let conv = slots * shape.ink_conv_bytes(ink);
        let mut a = Alloc {
            dev: &e.device,
            bytes: 0,
        };
        let mut states = Vec::with_capacity(ink.n_slot(c.n_layer));
        for slot in 0..ink.n_slot(c.n_layer) {
            let swa = c.is_swa(ink.layer_of(slot, c.n_layer));
            let bytes = if swa { window_bytes } else { full_bytes };
            states.push(InkSlot {
                k: a.zeroed(bytes)?,
                v: a.zeroed(bytes)?,
                ring: if swa { ring } else { None },
                conv: a.zeroed(conv)?,
                swa,
            });
        }
        debug_assert_eq!(a.bytes, memory.allocated());
        Ok(Seq {
            layers: Vec::new(),
            mtp: None,
            draft,
            slots,
            slot: 0,
            verify_from: None,
            context,
            n_past: 0,
            memory,
            ink: states,
        })
    }

    /// Positions of a slot a next token may see, of the `n_past` before it
    fn ink_rows(c: &Config, st: &InkSlot, n_past: usize) -> usize {
        if st.swa {
            n_past.min(c.sliding_window - 1)
        } else {
            n_past
        }
    }

    pub(super) fn ink_snapshot(&self, e: &MetalEngine<'_>) -> Result<Snapshot, Error> {
        assert!(
            self.verify_from.is_none(),
            "snapshot between verify and keep"
        );
        let c = e.cfg();
        let ink = c.inkling.as_ref().expect("an inkling config");
        let kv_row = 2 * c.kv_dim();
        let conv = e.model.shape().ink_conv_bytes(ink);
        let mut bytes = 0;
        let mut slots = Vec::with_capacity(self.ink.len());
        for st in &self.ink {
            let first = self.n_past - Self::ink_rows(c, st, self.n_past);
            let rows = |buf: &Buffer| -> Result<Vec<u8>, Error> {
                let n = (self.n_past - first) * kv_row;
                let mut out = Vec::new();
                out.try_reserve_exact(n)
                    .map_err(|_| Error::OutOfMemory { bytes: n })?;
                // runs of rows that don't wrap around the ring
                let mut p = first;
                while p < self.n_past {
                    let (row, end) = match st.ring {
                        Some(r) => (p % r, self.n_past.min(p - p % r + r)),
                        None => (p, self.n_past),
                    };
                    out.extend_from_slice(host_bytes(buf, row * kv_row, (end - p) * kv_row));
                    p = end;
                }
                Ok(out)
            };
            let (k, v) = (rows(&st.k)?, rows(&st.v)?);
            let cs = copy_out(&st.conv, self.slot * conv, conv)?;
            bytes += k.len() + v.len() + cs.len();
            slots.push((k, v, cs));
        }
        Ok(Snapshot::new(self.n_past, bytes, InkSnapshot { slots }))
    }

    pub(super) fn ink_restore(
        &mut self,
        e: &MetalEngine<'_>,
        snapshot: &Snapshot,
    ) -> Result<(), Error> {
        let snap: &InkSnapshot = snapshot.data()?;
        let n_past = snapshot.n_past();
        if n_past > self.context {
            return Err(Error::Snapshot("more tokens than the context holds"));
        }
        let c = e.cfg();
        let ink = c.inkling.as_ref().expect("an inkling config");
        let kv_row = 2 * c.kv_dim();
        let conv = e.model.shape().ink_conv_bytes(ink);
        let fits = self.ink.len() == snap.slots.len()
            && self.ink.iter().zip(&snap.slots).all(|(st, (k, v, cs))| {
                let rows = Self::ink_rows(c, st, n_past) * kv_row;
                k.len() == rows && v.len() == rows && cs.len() == conv
            });
        if !fits {
            return Err(Error::Snapshot("taken from another model"));
        }
        // the conv states go to slot 0
        for (st, (sk, sv, cs)) in self.ink.iter().zip(&snap.slots) {
            let first = n_past - sk.len() / kv_row;
            for (j, p) in (first..n_past).enumerate() {
                let at = st.ring.map_or(p, |r| p % r) * kv_row;
                let src = j * kv_row..(j + 1) * kv_row;
                host_bytes(&st.k, at, kv_row).copy_from_slice(&sk[src.clone()]);
                host_bytes(&st.v, at, kv_row).copy_from_slice(&sv[src]);
            }
            host_bytes(&st.conv, 0, conv).copy_from_slice(cs);
        }
        self.slot = 0;
        self.n_past = n_past;
        self.verify_from = None;
        Ok(())
    }
}

/// Segments of this many tokens or more (without saved states) run their
/// short convs a thread per (channel, token), see `ink_conv_par`, rather than
/// a thread per channel through the tokens
const CONV_PAR_MIN: usize = 16;

fn conv_par(g: &InkSeg<'_>) -> bool {
    g.n >= CONV_PAR_MIN && !g.save
}

/// A sequence's tokens in a pass, as [`Seg`], with its slot of the visit
struct InkSeg<'a> {
    st: &'a InkSlot,
    row0: usize,
    n: usize,
    pos0: usize,
    slot: usize,
    slots: usize,
    save: bool,
}

/// The rows of a depth mix: of the states and stream from `in_row0`, of the
/// outputs from `out_row0`
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
struct MixOut<'a> {
    out: B<'a>,
    stride: usize,
    copies: usize,
    norm: Option<B<'a>>,
}

impl MetalEngine<'_> {
    fn ink_scratch<'s>(&self, s: &'s Scratch) -> &'s InkScratch {
        s.ink
            .as_ref()
            .expect("inkling scratch for an inkling model")
    }

    /// The depth mix over the stack regions `states` (the last one with
    /// `cur`, the stream normalized into it first), of each of the `n_q`
    /// queries of `query`
    #[allow(clippy::too_many_arguments)]
    fn ink_mix(
        &self,
        enc: &Encoder,
        s: &Scratch,
        states: &[usize],
        cur: Option<B<'_>>,
        query: Option<(&Buffer, usize)>,
        out: Option<MixOut<'_>>,
        rows: Rows,
    ) {
        let c = self.cfg();
        let e = c.n_embd;
        let is = self.ink_scratch(s);
        assert!(
            states.len() <= MIX_STATES,
            "a depth mix of {} states",
            states.len()
        );
        let mut off = [0i32; MIX_STATES];
        for (o, &r) in off.iter_mut().zip(states) {
            *o = (r * s.batch * e) as i32;
        }
        let (query, n_q) = query.unwrap_or((&is.stack, 0));
        assert!(n_q <= MIX_QUERIES);
        let args = MixArgs {
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
        Self::args(enc, &args);
        let stack = B::at(&is.stack);
        let o = out.as_ref().map_or(stack, |o| o.out);
        let chunks = e.div_ceil(MIX_CHUNK);
        if rows.n <= MIX_CHUNK_ROWS && e.is_multiple_of(4) && chunks <= MIX_CHUNKS {
            let part = B::at(&is.mix_part);
            let grid = [chunks, rows.n, 1];
            Self::bind(enc, &[stack, cur.unwrap_or(stack), B::at(query), part]);
            self.dispatch(enc, Kernel::InkMixPart, grid, [64, 1, 1], 0);
            Self::args(enc, &args);
            Self::bind(enc, &[stack, cur.unwrap_or(stack), part, o]);
            self.dispatch(enc, Kernel::InkMixApply, grid, [64, 1, 1], 0);
            if let Some(w) = out.as_ref().and_then(|o| o.norm).filter(|_| n_q > 0) {
                Self::args(enc, &args);
                Self::bind(enc, &[w, part, o]);
                self.dispatch(enc, Kernel::InkMixNorm, grid, [64, 1, 1], 0);
            }
            return;
        }
        Self::bind(
            enc,
            &[
                stack,
                cur.unwrap_or(stack),
                B::at(query),
                out.as_ref().and_then(|o| o.norm).unwrap_or(stack),
                o,
            ],
        );
        self.dispatch(enc, Kernel::InkMix, [rows.n, 1, 1], [256, 1, 1], 0);
    }

    /// rms norm without weight of `rows` rows of `width`
    #[allow(clippy::too_many_arguments)]
    fn ink_rms(
        &self,
        enc: &Encoder,
        x: B<'_>,
        out: B<'_>,
        rows: usize,
        width: usize,
        istride: usize,
        ostride: usize,
    ) {
        let args = RmsArgs {
            n: width as i32,
            eps: MIX_EPS,
            istride: istride as i32,
            ostride: ostride as i32,
        };
        Self::args(enc, &args);
        Self::bind(enc, &[x, out]);
        self.dispatch(enc, Kernel::InkRms, [rows, 1, 1], [256, 1, 1], 0);
    }

    /// [`MetalEngine::rms_norm`] with the weight at a binding
    fn ink_rms_norm(&self, enc: &Encoder, x: B<'_>, w: B<'_>, out: B<'_>, rows: usize) {
        let width = self.cfg().n_embd;
        let args = NormArgs {
            n: width as i32,
            eps: self.cfg().eps,
            gated: 0,
            ostride: width as i32,
        };
        Self::args(enc, &args);
        Self::bind(enc, &[x, w, out, x]);
        self.dispatch(enc, Kernel::RmsNorm, [rows, 1, 1], [256, 1, 1], 0);
    }

    /// [`MetalEngine::matmul`], also of weights whose rows the `mmv` kernels
    /// don't take (they read the inputs by 4 spans of 32): the controllers'
    /// gains and forgets, of rows of 64, then in passes of the rows `mv` takes
    fn ink_matmul(&self, enc: &Encoder, w: &GpuWeight, x: B<'_>, n: usize, out: B<'_>) {
        let mv_rows = if w.ty == GgmlType::F32 {
            8
        } else {
            MV_MAX_ROWS
        };
        if w.cols.is_multiple_of(128) || n <= mv_rows || n > MMV_MAX_ROWS {
            return self.matmul(enc, w, x, n, out);
        }
        for r in (0..n).step_by(mv_rows) {
            let x = x.f32_off(r * w.cols);
            self.matmul(enc, w, x, mv_rows.min(n - r), out.f32_off(r * w.rows));
        }
    }

    fn ink_copy(&self, enc: &Encoder, dst: B<'_>, src: B<'_>, n: usize) {
        Self::args(enc, &EltArgs { n: n as i32 });
        Self::bind(enc, &[dst, src]);
        self.dispatch(
            enc,
            Kernel::InkCopy,
            [n.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        );
    }

    /// The controllers `ctrls` of `lp`, together: controller `ctrls[j]`
    /// mixes the message `msgs[j]` into the input `xs[j]`, the result going to
    /// `outs[j]` (which may be `xs[j]`) rms-normalized with the weight
    /// `norm.0` plus `j * norm.1` floats, through set `j` of the buffers
    #[allow(clippy::too_many_arguments)]
    fn ink_control(
        &self,
        enc: &Encoder,
        s: &Scratch,
        lp: &GpuLoop,
        ctrls: &[usize],
        xs: &[B<'_>],
        msgs: &[B<'_>],
        outs: &[B<'_>],
        n: usize,
        norm: (B<'_>, usize),
    ) {
        let c = self.cfg();
        let (e, rank) = (
            c.n_embd,
            c.inkling.as_ref().expect("an inkling config").gate_rank,
        );
        let f32s = |w: &[GpuWeight; 4]| w.iter().all(|w| w.ty == GgmlType::F32);
        if n <= CTRL_FAST_ROWS
            && ctrls.len() <= 4
            && e.is_multiple_of(4)
            && e <= 4096
            && rank == 64
            && f32s(&lp.down)
            && f32s(&lp.gain)
            && f32s(&lp.forget)
        {
            return self.ink_control_short(enc, s, lp, ctrls, xs, msgs, outs, n, norm);
        }
        self.ink_control_long(enc, s, lp, ctrls, xs, msgs, outs, n);
        self.concurrent(|| {
            for (j, &o) in outs.iter().enumerate() {
                self.ink_rms_norm(enc, o, norm.0.f32_off(j * norm.1), o, n);
            }
        });
    }

    /// [`MetalEngine::ink_control`] of short passes, see [`CTRL_FAST_ROWS`]
    #[allow(clippy::too_many_arguments)]
    fn ink_control_short(
        &self,
        enc: &Encoder,
        s: &Scratch,
        lp: &GpuLoop,
        ctrls: &[usize],
        xs: &[B<'_>],
        msgs: &[B<'_>],
        outs: &[B<'_>],
        n: usize,
        norm: (B<'_>, usize),
    ) {
        let c = self.cfg();
        let ink = c.inkling.as_ref().expect("an inkling config");
        let (e, rank) = (c.n_embd, ink.gate_rank);
        let is = self.ink_scratch(s);
        // of ink_ctrl_out: 64 channels per threadgroup
        let chunks = e.div_ceil(64);
        let args = CtrlArgs {
            e: e as i32,
            rank: rank as i32,
            eps: MIX_EPS,
            out_eps: c.eps,
            stride: e as i32,
            chunks: chunks as i32,
            w_stride: norm.1 as i32,
            pad: 0,
        };
        let nc = ctrls.len();
        // the buffers of the 4 controller slots, the last repeated
        fn four<'b>(nc: usize, f: impl Fn(usize) -> B<'b>) -> [B<'b>; 4] {
            [0, 1, 2, 3].map(|j| f(j.min(nc - 1)))
        }
        fn weight(w: &GpuWeight) -> B<'_> {
            B(&w.buf, w.offset)
        }
        let (z, part) = (B::at(&is.z), B::at(&is.mix_part));
        let mut bufs: Vec<B<'_>> = Vec::with_capacity(22);
        bufs.extend(four(nc, |j| xs[j]));
        bufs.extend(four(nc, |j| msgs[j]));
        bufs.extend(four(nc, |j| weight(&lp.down[ctrls[j]])));
        bufs.push(z);
        Self::args(enc, &args);
        Self::bind(enc, &bufs);
        self.dispatch(enc, Kernel::InkCtrlDown, [rank, nc, n], [256, 1, 1], 0);
        bufs.truncate(8);
        bufs.extend(four(nc, |j| weight(&lp.gain[ctrls[j]])));
        bufs.extend(four(nc, |j| weight(&lp.forget[ctrls[j]])));
        bufs.extend(four(nc, |j| outs[j]));
        bufs.push(z);
        bufs.push(part);
        Self::args(enc, &args);
        Self::bind(enc, &bufs);
        self.dispatch(enc, Kernel::InkCtrlOut, [chunks, nc, n], [256, 1, 1], 0);
        let mut bufs: Vec<B<'_>> = four(nc, |j| outs[j]).to_vec();
        bufs.push(norm.0);
        bufs.push(part);
        Self::args(enc, &args);
        Self::bind(enc, &bufs);
        self.dispatch(
            enc,
            Kernel::InkCtrlNorm,
            [e.div_ceil(256), nc, n],
            [64, 1, 1],
            0,
        );
    }

    /// [`MetalEngine::ink_control`] of long passes, before the output norm
    #[allow(clippy::too_many_arguments)]
    fn ink_control_long(
        &self,
        enc: &Encoder,
        s: &Scratch,
        lp: &GpuLoop,
        ctrls: &[usize],
        xs: &[B<'_>],
        msgs: &[B<'_>],
        outs: &[B<'_>],
        n: usize,
    ) {
        let c = self.cfg();
        let ink = c.inkling.as_ref().expect("an inkling config");
        let (e, b, rank) = (c.n_embd, s.batch, ink.gate_rank);
        let is = self.ink_scratch(s);
        let u = |j: usize| B::at(&is.u).f32_off(j * b * 2 * e);
        let z = |j: usize| B::at(&is.z).f32_off(j * b * rank);
        let g = |j: usize| B::at(&is.gain).f32_off(j * b * e);
        let f = |j: usize| B::at(&is.forget).f32_off(j * b * e);
        self.concurrent(|| {
            for j in 0..ctrls.len() {
                self.ink_rms(enc, xs[j], u(j), n, e, e, 2 * e);
                self.ink_rms(enc, msgs[j], u(j).f32_off(e), n, e, e, 2 * e);
            }
        });
        self.concurrent(|| {
            for (j, &k) in ctrls.iter().enumerate() {
                self.matmul(enc, &lp.down[k], u(j), n, z(j));
            }
        });
        self.concurrent(|| {
            for (j, &k) in ctrls.iter().enumerate() {
                self.ink_matmul(enc, &lp.gain[k], z(j), n, g(j));
                self.ink_matmul(enc, &lp.forget[k], z(j), n, f(j));
            }
        });
        self.concurrent(|| {
            for j in 0..ctrls.len() {
                Self::args(enc, &EltArgs { n: (n * e) as i32 });
                Self::bind(enc, &[xs[j], g(j), f(j), msgs[j], outs[j]]);
                self.dispatch(
                    enc,
                    Kernel::InkControl,
                    [(n * e).div_ceil(256), 1, 1],
                    [256, 1, 1],
                    0,
                );
            }
        });
    }

    /// The short conv of the `g.n` rows of `x` (`width` channels) into
    /// `out`, added to it with `add`, at channel `offset` of the slot's
    /// state, with the dynamic taps `dyn_` when given
    #[allow(clippy::too_many_arguments)]
    fn ink_conv(
        &self,
        enc: &Encoder,
        g: &InkSeg<'_>,
        x: B<'_>,
        w: &Buffer,
        out: B<'_>,
        dyn_: Option<B<'_>>,
        width: usize,
        offset: usize,
        add: bool,
    ) {
        let c = self.cfg();
        let ink = c.inkling.as_ref().expect("an inkling config");
        let taps = ink.conv;
        let mult = if dyn_.is_some() { 2 } else { 1 };
        let args = InkConvArgs {
            width: width as i32,
            n: g.n as i32,
            channels: ink.conv_channels(c.kv_dim(), c.n_embd) as i32,
            offset: offset as i32,
            slot: g.slot as i32,
            slots: g.slots as i32,
            save: g.save as i32,
            add: add as i32,
            dyn_: dyn_.is_some() as i32,
            head: c.head_dim as i32,
            dyn_stride: (c.n_head_kv * taps) as i32,
            scale: 1.0 / ((mult * taps + 1) as f32).sqrt(),
        };
        Self::args(enc, &args);
        Self::bind(
            enc,
            &[x, B::at(w), B::at(&g.st.conv), out, dyn_.unwrap_or(x)],
        );
        if conv_par(g) {
            assert!(x.0 != out.0 || x.1 != out.1, "a parallel conv in place");
            self.dispatch(
                enc,
                Kernel::InkConvPar,
                [width.div_ceil(256), g.n, 1],
                [256, 1, 1],
                0,
            );
            return;
        }
        self.dispatch(
            enc,
            Kernel::InkConv,
            [width.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        );
    }

    /// The state update after [`MetalEngine::ink_conv`]'s of a `conv_par`
    /// segment: the last 3 inputs `x` of `width` channels at `offset`
    fn ink_conv_state(&self, enc: &Encoder, g: &InkSeg<'_>, x: B<'_>, width: usize, offset: usize) {
        let c = self.cfg();
        let ink = c.inkling.as_ref().expect("an inkling config");
        let args = InkConvArgs {
            width: width as i32,
            n: g.n as i32,
            channels: ink.conv_channels(c.kv_dim(), c.n_embd) as i32,
            offset: offset as i32,
            slot: g.slot as i32,
            slots: g.slots as i32,
            save: 0,
            add: 0,
            dyn_: 0,
            head: c.head_dim as i32,
            dyn_stride: 0,
            scale: 0.0,
        };
        Self::args(enc, &args);
        Self::bind(enc, &[x, B::at(&g.st.conv)]);
        self.dispatch(
            enc,
            Kernel::InkConvState,
            [width.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        );
    }

    /// One pass of the tokens of several sequences, see [`MetalEngine::pass`]
    pub(super) fn ink_pass(
        &self,
        ink: &GpuInkling,
        s: &Scratch,
        items: &mut [(&mut Seq, &[TokenId], Pass)],
    ) -> Vec<Vec<Vec<f32>>> {
        let c = self.cfg();
        let cfg = c.inkling.as_ref().expect("an inkling config");
        let e = c.n_embd;
        for (seq, tokens, pass) in items.iter() {
            let n = tokens.len();
            assert!(n > 0, "a pass of no tokens");
            assert!(
                seq.n_past + n <= seq.context,
                "context overflow ({} + {n} > {})",
                seq.n_past,
                seq.context
            );
            assert!(
                *pass != Pass::Verify || n <= seq.slots,
                "verify of {n} tokens, {} at most",
                seq.slots
            );
        }
        let n: usize = items.iter().map(|(_, t, _)| t.len()).sum();
        assert!(n <= s.batch, "a pass of {n} tokens, {} at most", s.batch);

        // the embeddings, on the CPU, straight into the shared buffer
        let mut places = Vec::with_capacity(items.len());
        let (mut row0, mut logit0) = (0, 0);
        let x_host = host(&s.x, 0, n * e);
        for (seq, tokens, pass) in items.iter() {
            for (t, &tok) in tokens.iter().enumerate() {
                let r = row0 + t;
                self.model
                    .tok_embd
                    .dequantize_row(tok as usize, &mut x_host[r * e..(r + 1) * e]);
            }
            places.push(Place {
                row0,
                n: tokens.len(),
                pos0: seq.n_past,
                mtp_rows: None,
                logit0,
            });
            row0 += tokens.len();
            logit0 += match pass {
                Pass::Prefill => 0,
                Pass::Forward => 1,
                Pass::Verify => tokens.len(),
            };
        }
        let logit_rows = logit0;
        assert!(
            logit_rows <= s.logit_rows,
            "{logit_rows} rows of logits, {} at most",
            s.logit_rows
        );

        let is = self.ink_scratch(s);
        let plan = ink.stack;
        let x = B::at(&s.x);
        let (b, rows) = (s.batch, Rows::all(n));
        self.submit(|enc| {
            self.rms_norm(enc, x, &ink.tok_norm, x, None, n, e);

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
                    self.ink_copy(enc, B::at(&is.prelude), x, n * e);
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
                            enc,
                            s,
                            &states,
                            first_of_pass.then_some(x),
                            Some((&lp.msg_query, 4)),
                            Some(MixOut {
                                out: B::at(&is.msg),
                                stride: b * e,
                                copies: 1,
                                norm: None,
                            }),
                            rows,
                        );
                        if first_of_pass {
                            self.ink_copy(enc, x, B::at(&is.prelude), n * e);
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
                let segs: Vec<InkSeg<'_>> = items
                    .iter()
                    .zip(&places)
                    .map(|((seq, _, pass), p)| InkSeg {
                        st: &seq.ink[v.slot],
                        row0: p.row0,
                        n: p.n,
                        pos0: p.pos0,
                        slot: seq.slot,
                        slots: seq.slots,
                        save: *pass == Pass::Verify,
                    })
                    .collect();
                self.ink_layer(enc, s, ink, layer, &segs, n, &mut blocks, push, msgs);
            }

            // the rows with logits: the output mix, normalized, gathered in `feat`
            if logit_rows > 0 {
                let states: Vec<usize> = blocks.iter().rev().copied().chain([plan.cur()]).collect();
                for ((_, _, pass), p) in items.iter().zip(&places) {
                    let rows = match pass {
                        Pass::Prefill => continue,
                        Pass::Forward => 1,
                        Pass::Verify => p.n,
                    };
                    self.ink_mix(
                        enc,
                        s,
                        &states,
                        Some(x),
                        Some((&ink.output_query, 1)),
                        Some(MixOut {
                            out: B::at(&s.feat),
                            stride: 0,
                            copies: 1,
                            norm: Some(B::at(&self.output_norm)),
                        }),
                        Rows {
                            n: rows,
                            in_row0: p.row0 + p.n - rows,
                            out_row0: p.logit0,
                        },
                    );
                }
                self.matmul(
                    enc,
                    &self.output,
                    B::at(&s.feat),
                    logit_rows,
                    B::at(&s.logits),
                );
                if cfg.softcap != 0.0 {
                    let count = logit_rows * c.n_vocab;
                    let args = CapArgs {
                        n: count as i32,
                        cap: cfg.softcap,
                        inv: 1.0 / cfg.softcap,
                    };
                    Self::args(enc, &args);
                    Self::bind(enc, &[B::at(&s.logits)]);
                    self.dispatch(
                        enc,
                        Kernel::InkSoftcap,
                        [count.div_ceil(256), 1, 1],
                        [256, 1, 1],
                        0,
                    );
                }
            }
        });

        items
            .iter_mut()
            .zip(&places)
            .map(|((seq, _, pass), p)| {
                seq.n_past += p.n;
                if *pass == Pass::Verify {
                    seq.verify_from = Some(p.pos0);
                }
                let rows = match pass {
                    Pass::Prefill => 0,
                    Pass::Forward => 1,
                    Pass::Verify => p.n,
                };
                (p.logit0..p.logit0 + rows)
                    .map(|t| host(&s.logits, t * c.n_vocab, c.n_vocab).to_vec())
                    .collect()
            })
            .collect()
    }

    /// One visit of `layer` over the stream `s.x` of `n` rows, the
    /// sequences `segs`: with `push`, the first layer of a block, whose
    /// normalized input joins `blocks` in that region of the stack; with
    /// `lp`, a looped layer of a pass after the first, its messages in `msg`
    #[allow(clippy::too_many_arguments)]
    fn ink_layer(
        &self,
        enc: &Encoder,
        s: &Scratch,
        ink: &GpuInkling,
        layer: &GpuInkLayer,
        segs: &[InkSeg<'_>],
        n: usize,
        blocks: &mut Vec<usize>,
        push: Option<usize>,
        lp: Option<&GpuLoop>,
    ) {
        let c = self.cfg();
        let cfg = c.inkling.as_ref().expect("an inkling config");
        let (e, hd, nh, nkv) = (c.n_embd, c.head_dim, c.n_head, c.n_head_kv);
        let (qd, kv_dim, n_rel, taps) = (nh * hd, c.kv_dim(), cfg.n_rel, cfg.conv);
        let is = self.ink_scratch(s);
        let plan = ink.stack;
        let b = s.batch;
        let x = B::at(&s.x);
        let rows = Rows::all(n);
        let inp = |j: usize| B::at(&is.inp).f32_off(j * b * e);
        let norm = |j: usize| B::at(&layer.norms).f32_off(j.min(layer.n_norms - 1) * e);
        let msg = |j: usize| B::at(&is.msg).f32_off(j * b * e);

        // the attention inputs: depth mixes over the blocks and the
        // normalized stream (or the stream itself in the first layer),
        // controlled by the messages, normalized
        let cur = push.unwrap_or(plan.cur());
        let states: Vec<usize> = blocks.iter().rev().copied().chain([cur]).collect();
        let n_in = if !layer.swa || lp.is_some() { 3 } else { 1 };
        match &layer.attn_query {
            Some(q) => {
                let copies = n_in / layer.n_q;
                self.ink_mix(
                    enc,
                    s,
                    &states,
                    Some(x),
                    Some((q, layer.n_q)),
                    Some(MixOut {
                        out: inp(0),
                        stride: b * e,
                        copies,
                        norm: if lp.is_some() { None } else { Some(norm(0)) },
                    }),
                    rows,
                );
            }
            None => {
                self.ink_mix(enc, s, &[cur], Some(x), None, None, rows);
                if lp.is_none() {
                    self.concurrent(|| {
                        for j in 0..n_in {
                            self.ink_rms_norm(enc, x, norm(j), inp(j), n);
                        }
                    });
                } else {
                    for j in 0..n_in {
                        self.ink_copy(enc, inp(j), x, n * e);
                    }
                }
            }
        }
        if let Some(lp) = lp {
            let ins: Vec<B<'_>> = (0..3).map(inp).collect();
            let ms: Vec<B<'_>> = (0..3).map(msg).collect();
            let w_stride = if layer.n_norms > 1 { e } else { 0 };
            self.ink_control(
                enc,
                s,
                lp,
                &[0, 1, 2],
                &ins,
                &ms,
                &ins,
                n,
                (norm(0), w_stride),
            );
        }
        let (iq, ik, iv) = (inp(0), inp(1.min(n_in - 1)), inp(2.min(n_in - 1)));

        // the projections
        let dyn_rows = nkv * taps;
        let k_dyn = B::at(&is.dyn_);
        let v_dyn = B::at(&is.dyn_).f32_off(b * dyn_rows);
        self.concurrent(|| {
            self.matmul(enc, &layer.wq, iq, n, B::at(&s.q));
            self.matmul(enc, &layer.wk, ik, n, B::at(&s.k));
            self.matmul(enc, &layer.wv, iv, n, B::at(&s.v));
            self.matmul(enc, &layer.wrel, iq, n, B::at(&is.r));
            self.matmul(enc, &layer.wsink, iq, n, B::at(&is.sink));
            self.matmul(enc, &layer.wgate, iq, n, B::at(&s.a2));
            if let Some(w) = &layer.k_dyn {
                self.matmul(enc, w, ik, n, k_dyn);
            }
            if let Some(w) = &layer.v_dyn {
                self.matmul(enc, w, iv, n, v_dyn);
            }
        });

        let group = nh / nkv;
        let window = layer.swa.then_some(c.sliding_window);
        for g in segs {
            let (gn, pos0, r) = (g.n, g.pos0, g.row0);
            // the keys' and values' convs, in place
            let (k, v) = (
                B::at(&s.k).f32_off(r * kv_dim),
                B::at(&s.v).f32_off(r * kv_dim),
            );
            let full = layer.k_dyn.is_some();
            // in place, or of long segments into the free attention inputs
            let (kin, vin) = (k, v);
            let (k, v) = if conv_par(g) {
                (inp(0).f32_off(r * kv_dim), inp(1).f32_off(r * kv_dim))
            } else {
                (k, v)
            };
            self.concurrent(|| {
                let kd = full.then(|| k_dyn.f32_off(r * dyn_rows));
                let vd = full.then(|| v_dyn.f32_off(r * dyn_rows));
                self.ink_conv(enc, g, kin, &layer.k_conv, k, kd, kv_dim, 0, false);
                self.ink_conv(enc, g, vin, &layer.v_conv, v, vd, kv_dim, kv_dim, false);
            });
            if conv_par(g) {
                self.concurrent(|| {
                    self.ink_conv_state(enc, g, kin, kv_dim, 0);
                    self.ink_conv_state(enc, g, vin, kv_dim, kv_dim);
                });
            }

            let lo = window.map_or(0, |w| (pos0 + 1).saturating_sub(w));
            let span = pos0 + gn - lo;
            let flash_split = (2..ATTN_FLASH_MIN).contains(&gn) && span >= ATTN_FLASH_SPLIT_CTX;
            let (lo, splits, chunk) = if flash_split {
                let lo = lo / ATTN_FLASH_C * ATTN_FLASH_C;
                let (splits, chunk) = flash_splits(c, pos0 + gn - lo, gn, s.batch);
                (lo, splits, chunk)
            } else {
                let (splits, chunk) = attn_splits(c, span, gn, s.batch);
                (lo, splits, chunk)
            };
            let args = InkAttnArgs {
                n_head: nh as i32,
                n_kv: nkv as i32,
                hd: hd as i32,
                n_rel: n_rel as i32,
                pos0: pos0 as i32,
                n: gn as i32,
                eps: c.eps,
                scale: 1.0 / (hd as f32).sqrt(),
                clip: cfg.qk_clip,
                rel_clip: cfg.rel_clip,
                full: !layer.swa as i32,
                floor: cfg.log_scale_floor as i32,
                alpha: cfg.log_scale_alpha,
                splits: splits as i32,
                chunk: chunk as i32,
                window: window.map_or(0, |w| w as i32),
                lo: lo as i32,
                mask: g.st.ring.map_or(-1, |r| r as i32 - 1),
                ext: layer.ext as i32,
            };
            let (q, rr, sink) = (
                B::at(&s.q).f32_off(r * qd),
                B::at(&is.r).f32_off(r * nh * n_rel),
                B::at(&is.sink).f32_off(r * nh),
            );
            let (gate, o) = (B::at(&s.a2).f32_off(r * qd), B::at(&s.o).f32_off(r * qd));
            let (kc, vc) = (B::at(&g.st.k), B::at(&g.st.v));
            Self::args(enc, &args);
            Self::bind(
                enc,
                &[
                    q,
                    k,
                    v,
                    rr,
                    B::at(&layer.q_clip),
                    B::at(&layer.rel_clip),
                    kc,
                    vc,
                ],
            );
            self.dispatch(
                enc,
                Kernel::InkAttnPrep,
                [2 * nh + nkv, gn, 1],
                [32, 1, 1],
                0,
            );

            let bias = B::at(&layer.rel_bias);
            // queries (half), scores, a float per row, relative channels and sink per row
            let flash_shmem =
                ATTN_FLASH_R * (2 * hd + 4 * ATTN_FLASH_C + 4) + ATTN_FLASH_R * 4 * (N_REL + 1);
            let blocks_of = (gn * group).div_ceil(ATTN_FLASH_R);
            if gn >= ATTN_FLASH_MIN {
                Self::args(enc, &args);
                Self::bind(enc, &[q, kc, vc, rr, sink, bias, gate, o]);
                self.dispatch(
                    enc,
                    Kernel::InkAttnFlash,
                    [blocks_of, nkv, 1],
                    [32 * ATTN_FLASH_NSG, 1, 1],
                    flash_shmem,
                );
            } else {
                let (po, pml) = (B::at(&s.part_o), B::at(&s.part_ml));
                Self::args(enc, &args);
                Self::bind(enc, &[q, kc, vc, rr, sink, bias, po, pml]);
                if flash_split {
                    self.dispatch(
                        enc,
                        Kernel::InkAttnFlashSplit,
                        [blocks_of, nkv, splits],
                        [32 * ATTN_FLASH_NSG, 1, 1],
                        flash_shmem,
                    );
                } else {
                    let nsg = ink.split_nsg;
                    self.dispatch(
                        enc,
                        Kernel::InkAttnSplit,
                        [nkv, splits, gn],
                        [32 * nsg, 1, 1],
                        4 * (2 * nsg * group + group * hd),
                    );
                }
                Self::args(enc, &args);
                Self::bind(enc, &[po, pml, gate, o]);
                // `ink_attn_reduce` takes up to 64 ranges
                debug_assert!(splits <= 64, "{splits} attention ranges");
                self.dispatch(enc, Kernel::InkAttnReduce, [nh, gn, 1], [hd, 1, 1], 0);
            }
        }
        self.matmul(enc, &layer.wo, B::at(&s.o), n, B::at(&s.y));
        // the output's conv, the residual's start at the first layer of a block
        for g in segs {
            let r = g.row0;
            let (y, xr) = (B::at(&s.y).f32_off(r * e), x.f32_off(r * e));
            self.ink_conv(
                enc,
                g,
                y,
                &layer.post_conv,
                xr,
                None,
                e,
                2 * kv_dim,
                push.is_none(),
            );
        }
        self.concurrent(|| {
            for g in segs.iter().filter(|g| conv_par(g)) {
                let y = B::at(&s.y).f32_off(g.row0 * e);
                self.ink_conv_state(enc, g, y, e, 2 * kv_dim);
            }
        });
        if let Some(p) = push {
            blocks.push(p);
        }

        // the feed-forward input: the depth mix with the stream, controlled,
        // normalized
        let states: Vec<usize> = blocks.iter().rev().copied().chain([plan.cur()]).collect();
        let h = B::at(&s.h);
        self.ink_mix(
            enc,
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
                    Some(B::at(&layer.ffn_norm))
                },
            }),
            rows,
        );
        if let Some(lp) = lp {
            self.ink_control(
                enc,
                s,
                lp,
                &[3],
                &[h],
                &[msg(3)],
                &[h],
                n,
                (B::at(&layer.ffn_norm), 0),
            );
        }
        self.ink_moe(enc, s, &layer.moe, n);
        for g in segs {
            let r = g.row0;
            let (y, xr) = (B::at(&s.y).f32_off(r * e), x.f32_off(r * e));
            self.ink_conv(
                enc,
                g,
                y,
                &layer.ffn_post_conv,
                xr,
                None,
                e,
                2 * kv_dim + e,
                true,
            );
        }
        self.concurrent(|| {
            for g in segs.iter().filter(|g| conv_par(g)) {
                let y = B::at(&s.y).f32_off(g.row0 * e);
                self.ink_conv_state(enc, g, y, e, 2 * kv_dim + e);
            }
        });
    }

    /// The mixture of experts of the normalized `n` tokens in `h` into `y`,
    /// with SITU experts, see [`MetalEngine::encode_moe`]
    fn ink_moe(&self, enc: &Encoder, s: &Scratch, moe: &GpuMoe, n: usize) {
        let c = self.cfg();
        let cfg = c.inkling.as_ref().expect("an inkling config");
        let m = s.moe.as_ref().expect("moe scratch for a moe model");
        let cm = c.moe.as_ref().expect("a moe config");
        let (ne, k) = (cm.n_expert, cm.n_expert_used);
        let (gate, up, down) = (&moe.gate, &moe.up, &moe.down);
        let h = B::at(&s.h);

        self.matmul(enc, &moe.router, h, n, B::at(&m.logits));
        let (mm_tiles, mv_tiles) = max_tiles(c, n, MM_ID_MIN);
        MetalEngine::args(
            enc,
            &RouteArgs {
                n_expert: ne as i32,
                k: k as i32,
                sigmoid: (cm.gating == Gating::Sigmoid) as i32,
                bias: moe.bias.is_some() as i32,
                norm: cm.norm_weights as i32,
                scale: cm.weights_scale,
            },
        );
        MetalEngine::bind(
            enc,
            &[
                B::at(&m.logits),
                B::at(&m.ids),
                B::at(&m.weights),
                B::at(moe.bias.as_ref().unwrap_or(&m.logits)),
            ],
        );
        let sigmoid = cm.gating == Gating::Sigmoid;
        if sigmoid && ne <= 256 && ne.is_multiple_of(4) && k <= 32 {
            self.dispatch(enc, Kernel::InkRoute, [n, 1, 1], [256, 1, 1], 0);
        } else {
            self.dispatch(enc, Kernel::MoeRoute, [n, 1, 1], [256, 1, 1], 4 * ne);
        }

        let (beta, lb) = (cfg.situ_beta, cfg.situ_linear_beta);
        let situ = SituArgs {
            n: 0,
            beta,
            ib: 1.0 / beta,
            lb,
            ilb: if lb > 0.0 { 1.0 / lb } else { 0.0 },
        };
        // passes of a pair per threadgroup row of q8_0 experts: the gate and
        // up rows together through SITU, then the down rows of all the pairs
        // of a token, weighed and summed
        let fused = n <= MV_ID_MAX_ROWS
            && k <= 8
            && [gate, up, down].iter().all(|w| w.ty == GgmlType::Q8_0)
            && gate.cols.is_multiple_of(32)
            && down.cols.is_multiple_of(32)
            && down.cols >= 8 * 32
            && gate.rows == down.cols;
        if fused {
            MetalEngine::args(
                enc,
                &GateUpArgs {
                    k: gate.cols as i32,
                    rows: gate.rows as i32,
                    x_div: k as i32,
                    pad: 0,
                    w_row_bytes: gate.row_bytes as u64,
                    x_row_bytes: (gate.cols * 4) as u64,
                    stride: gate.stride as u64,
                    situ,
                },
            );
            MetalEngine::bind(
                enc,
                &[
                    B(&gate.buf, gate.offset),
                    B(&up.buf, up.offset),
                    h,
                    B::at(&s.a1),
                    B::at(&m.ids),
                ],
            );
            self.dispatch(
                enc,
                Kernel::InkGateUpQ8_0,
                [gate.rows, n * k, 1],
                [128, 1, 1],
                0,
            );
            MetalEngine::args(
                enc,
                &MatIdArgs {
                    k: down.cols as i32,
                    rows: down.rows as i32,
                    x_div: k as i32,
                    pad: 0,
                    w_row_bytes: down.row_bytes as u64,
                    x_row_bytes: (down.cols * 4) as u64,
                    stride: down.stride as u64,
                },
            );
            MetalEngine::bind(
                enc,
                &[
                    B(&down.buf, down.offset),
                    B::at(&s.a1),
                    B::at(&s.y),
                    B::at(&m.ids),
                    B::at(&m.weights),
                ],
            );
            self.dispatch(
                enc,
                Kernel::InkDownQ8_0,
                [down.rows.div_ceil(4), n, 1],
                [256, 1, 1],
                0,
            );
            return;
        }

        let per_pair = n <= MV_ID_MAX_ROWS
            && [gate, up, down]
                .iter()
                .all(|w| mv_id_kernel(w.ty).is_some());
        if !per_pair {
            let args = SortArgs {
                n_pairs: (n * k) as i32,
                n_expert: ne as i32,
                mm_min: MM_ID_MIN as i32,
                mv_base: mm_tiles as i32,
            };
            let (rank, count) = (B::at(&m.rank), B(&m.rank, 4 * n * k));
            MetalEngine::args(enc, &args);
            MetalEngine::bind(enc, &[B::at(&m.ids), rank, count]);
            self.dispatch(enc, Kernel::MoeRank, [ne, 1, 1], [256, 1, 1], 4 * 8);
            MetalEngine::args(enc, &args);
            MetalEngine::bind(
                enc,
                &[
                    B::at(&m.ids),
                    B::at(&m.order),
                    B::at(&m.tiles),
                    B::at(&m.n_tiles),
                    rank,
                    count,
                ],
            );
            self.dispatch(enc, Kernel::MoeSort, [1, 1, 1], [256, 1, 1], 4 * 4 * ne);
        }

        let matmul = |w: &GpuExperts, x: B<'_>, x_div: usize, out: B<'_>| {
            MetalEngine::args(
                enc,
                &MatIdArgs {
                    k: w.cols as i32,
                    rows: w.rows as i32,
                    x_div: x_div as i32,
                    pad: 0,
                    w_row_bytes: w.row_bytes as u64,
                    x_row_bytes: (w.cols * 4) as u64,
                    stride: w.stride as u64,
                },
            );
            let wb = B(&w.buf, w.offset);
            if per_pair {
                let (kernel, rows_per_tg, threads) = mv_id_kernel(w.ty).expect("checked above");
                MetalEngine::bind(enc, &[wb, x, out, B::at(&m.ids)]);
                self.dispatch(
                    enc,
                    kernel,
                    [w.rows.div_ceil(rows_per_tg), n * k, 1],
                    [threads, 1, 1],
                    0,
                );
                return;
            }
            if n >= MM_ID_MIN {
                let (tiles, n_tiles) = (B::at(&m.tiles), B::at(&m.n_tiles));
                MetalEngine::bind(enc, &[wb, x, out, B::at(&m.order), tiles, n_tiles]);
                self.dispatch(
                    enc,
                    mm_id_kernel(w.ty).expect("checked at load"),
                    [mm_tiles, w.rows.div_ceil(64), 1],
                    [128, 1, 1],
                    8192,
                );
            }
            let (tiles, n_tiles) = (B(&m.tiles, 16 * mm_tiles), B(&m.n_tiles, 4));
            MetalEngine::bind(enc, &[wb, x, out, B::at(&m.order), tiles, n_tiles]);
            self.dispatch(
                enc,
                mmv_id_kernel(w.ty).expect("checked at load"),
                [w.rows.div_ceil(MMV_WEIGHT_ROWS), mv_tiles, 1],
                [64, 1, 1],
                0,
            );
        };
        self.concurrent(|| {
            matmul(gate, h, k, B::at(&s.a1));
            matmul(up, h, k, B::at(&s.a2));
        });
        let count = n * k * gate.rows;
        MetalEngine::args(
            enc,
            &SituArgs {
                n: count as i32,
                ..situ
            },
        );
        MetalEngine::bind(enc, &[B::at(&s.a1), B::at(&s.a2)]);
        self.dispatch(
            enc,
            Kernel::InkSitu,
            [count.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        );
        self.concurrent(|| matmul(down, B::at(&s.a1), 1, B::at(&m.out)));

        MetalEngine::args(
            enc,
            &CombineArgs {
                n_embd: c.n_embd as i32,
                k: k as i32,
                shared: 0,
            },
        );
        MetalEngine::bind(enc, &[B::at(&s.y), B::at(&m.out), B::at(&m.weights)]);
        self.dispatch(
            enc,
            Kernel::InkMoeSum,
            [c.n_embd.div_ceil(256), n, 1],
            [256, 1, 1],
            0,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Inkling-2-Mini's loop: 24 layers in blocks of 6, core 6..18 run 3 times
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

    /// Inkling-2-Mini's geometry
    fn mini_config() -> Config {
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
            inkling: Some(mini()),
        }
    }

    #[test]
    fn check_takes_the_geometry_of_the_kernels() {
        let ok = |c: &Config| check(c, c.inkling.as_ref().unwrap()).is_ok();
        let c = mini_config();
        assert!(ok(&c));
        // 3 query heads per kv head: `ink_attn_split_3`, not `attn_split`
        assert_eq!(c.n_head / c.n_head_kv, 3);
        assert!(!ATTN_GROUPS.iter().any(|g| g.0 == 3));
        // 48 layers in blocks of 6: an output mix of 9 states
        let mut deep = mini_config();
        deep.n_layer = 48;
        let ink = deep.inkling.as_mut().unwrap();
        (ink.loop_start, ink.loop_end) = (12, 36);
        assert!(!ok(&deep));
        // keys wider than the stream
        let mut wide = mini_config();
        (wide.n_head, wide.n_head_kv) = (24, 24);
        assert!(!ok(&wide));
        // query heads wider than the chosen experts' activations
        let mut narrow = mini_config();
        narrow.moe.as_mut().unwrap().n_expert_used = 2;
        assert!(!ok(&narrow));
        // other head sizes, relative channels or taps
        let mut c = mini_config();
        c.head_dim = 64;
        assert!(!ok(&c));
    }
}
