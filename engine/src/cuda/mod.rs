//! CUDA backend (NVIDIA GPUs). The kernels are compiled to PTX at build time
//! (`build.rs`, with nvcc), and loaded through the driver API, found at run
//! time. A pass is a sequence of kernel launches on the default stream.
//!
//! The weights are copied to the device, the output head and the layers'
//! weights first, then the experts of a mixture of experts. What doesn't fit
//! in the memory left for the sessions goes to pinned host memory mapped to
//! the device, which reads it across the bus: a model larger than the GPU's
//! memory runs, at the speed of the bus for that part. The token embeddings
//! stay on the host: a pass looks up its tokens' rows on the CPU.
//!
//! Not yet: the MTP head (sessions verify but don't draft) and batched
//! sequences (a [`Batch`](crate::engine::Batch) runs them one after another).

mod driver;

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use core::cell::Cell;
use core::sync::atomic::{AtomicU64, Ordering};

use driver::{Buf, DevPtr, Driver, Event, Func, Graph, HostBuf, Stream, driver, f32_bytes};

use crate::Error;
use crate::engine::{DraftLogits, Engine, Pass, Session, Snapshot};
use crate::memory::{DeviceMemory, ModelMemory, SessionMemory, Shape};
use crate::model::{Config, Experts, Ffn, Gating, Layer, Mixer, Model};
use crate::ops::Weight;
use crate::prelude::*;
use crate::quant::GgmlType;
use crate::tokenizer::TokenId;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));
/// The kernels assembled for [`ARCH`], the GPU of the build
const CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kernels.cubin"));
const ARCH: &str = env!("BIV_CUDA_ARCH");

/// Device memory [`CudaEngine::new`] leaves for the sessions
pub const DEFAULT_RESERVE: usize = 2 << 30;
/// Device memory left free besides the sessions': the driver's own, the
/// display's on a desktop
const SLACK: usize = 512 << 20;

/// Rows of the matmul kernels' tiles (`NT` in the kernels): input rows a
/// warp takes at once, pairs of a MoE tile
const NT: usize = 8;
const MM_ROWS: usize = 4;
/// Tiles of the tensor core kernels (`TM`, `TN`): weight rows, input rows
/// (pairs of a MoE tile). Passes of more than [`NT`] rows take them.
const TM: usize = 128;
const TN: usize = 64;
const MMA_THREADS: usize = 256;
const ATTN_WARPS: usize = 8;
/// Tiles of the flash attention of prefill (`FA_*` in the kernels): query
/// tokens, keys, warps
const FA_BQ: usize = 64;
const FA_BK: usize = 32;
const FA_WARPS: usize = 4;
/// Head sizes the flash attention takes
const FA_HEAD_DIMS: [usize; 3] = [64, 128, 256];
const GDN_ROWS: usize = 4;
const SD: usize = 128;
/// Passes of up to this many tokens replay as CUDA graphs (decode and
/// verify steps, where the launches would take longer than the work)
const GRAPH_TOKENS: usize = NT;

#[repr(C)]
#[derive(Clone, Copy)]
struct MatArgs {
    w: DevPtr,
    x: DevPtr,
    y: DevPtr,
    row_bytes: i64,
    cols: i32,
    rows: i32,
    n: i32,
    x_stride: i32,
    y_stride: i32,
    pad: i32,
    tiles: DevPtr,
    n_tiles: DevPtr,
    order: DevPtr,
    expert_stride: i64,
    x_div: i32,
    pad2: i32,
    xq: DevPtr,
    xds: DevPtr,
}

/// Matrices of one input in one launch, see `mm_group`
const GROUP: usize = 4;

#[repr(C)]
struct GroupArgs {
    m: [MatArgs; GROUP],
    end: [i32; GROUP],
    count: i32,
    pad: i32,
}

#[repr(C)]
struct QuantArgs {
    x: DevPtr,
    q: DevPtr,
    ds: DevPtr,
    cols: i32,
    pad: i32,
}

#[repr(C)]
struct EltArgs {
    a: DevPtr,
    b: DevPtr,
    n: i32,
    pad: i32,
}

#[repr(C)]
struct NormArgs {
    x: DevPtr,
    w: DevPtr,
    out: DevPtr,
    gate: DevPtr,
    y: DevPtr,
    n: i32,
    ostride: i32,
    gated: i32,
    add: i32,
    eps: f32,
    pad: i32,
}

#[repr(C)]
struct AttnArgs {
    qg: DevPtr,
    k: DevPtr,
    v: DevPtr,
    q_norm: DevPtr,
    k_norm: DevPtr,
    inv_freq: DevPtr,
    q: DevPtr,
    k_cache: DevPtr,
    v_cache: DevPtr,
    o: DevPtr,
    sinks: DevPtr,
    params: DevPtr,
    n_head: i32,
    n_kv: i32,
    hd: i32,
    n_rot: i32,
    n: i32,
    gated: i32,
    window: i32,
    mask: i32,
    has_sinks: i32,
    eps: f32,
    scale: f32,
}

#[repr(C)]
struct ConvArgs {
    qkv: DevPtr,
    w: DevPtr,
    state: DevPtr,
    out: DevPtr,
    params: DevPtr,
    conv_dim: i32,
    n: i32,
    slots: i32,
    save: i32,
}

#[repr(C)]
struct GdnArgs {
    conv_out: DevPtr,
    alpha: DevPtr,
    beta: DevPtr,
    dt_bias: DevPtr,
    a_neg: DevPtr,
    state: DevPtr,
    out: DevPtr,
    params: DevPtr,
    n: i32,
    hk: i32,
    hv: i32,
    conv_dim: i32,
    eps: f32,
    slots: i32,
    save: i32,
    pad: i32,
}

/// `PassParams` in the kernels: what changes from a pass to the next of the
/// same shape, read from device memory
#[repr(C)]
#[derive(Clone, Copy)]
struct PassParams {
    pos0: i32,
    slot: i32,
    pad: [i32; 2],
}

#[repr(C)]
struct RouteArgs {
    logits: DevPtr,
    ids: DevPtr,
    weights: DevPtr,
    bias: DevPtr,
    n_expert: i32,
    k: i32,
    sigmoid: i32,
    has_bias: i32,
    norm: i32,
    scale: f32,
}

#[repr(C)]
struct SortArgs {
    ids: DevPtr,
    order: DevPtr,
    tiles: DevPtr,
    n_tiles: DevPtr,
    n_pairs: i32,
    n_expert: i32,
    tile: i32,
    pad: i32,
}

#[repr(C)]
struct CombineArgs {
    x: DevPtr,
    out: DevPtr,
    weights: DevPtr,
    shared: DevPtr,
    gate: DevPtr,
    n_embd: i32,
    k: i32,
    mode: i32,
    pad: i32,
}

/// The weight types of the matmul kernels, in the order of their names
const TYPES: [(GgmlType, &str); 9] = [
    (GgmlType::F32, "f32"),
    (GgmlType::F16, "f16"),
    (GgmlType::BF16, "bf16"),
    (GgmlType::Q4_0, "q4_0"),
    (GgmlType::Q4_1, "q4_1"),
    (GgmlType::Q8_0, "q8_0"),
    (GgmlType::Q4K, "q4_K"),
    (GgmlType::Q5K, "q5_K"),
    (GgmlType::Q6K, "q6_K"),
];

/// Dynamic shared memory of the flash attention: the queries' tile and the
/// keys' and values' (f16, rows padded by 8)
fn fattn_shared(hd: usize) -> usize {
    (FA_BQ + 2 * FA_BK) * (hd + 8) * 2
}

fn type_index(ty: GgmlType) -> Option<usize> {
    TYPES.iter().position(|&(t, _)| t == ty)
}

struct Kernels {
    mm: [Func; 9],
    mm_id: [Func; 9],
    mm_group: [Func; 9],
    mm_glu: [Func; 9],
    mma: [Func; 9],
    mma_id: [Func; 9],
    /// the int8 kernels, for the types that have them
    mmq: [Option<Func>; 9],
    mmq_id: [Option<Func>; 9],
    quantize: Func,
    add: Func,
    swiglu: Func,
    rms_norm: Func,
    attn_prep: Func,
    attn: Func,
    /// flash attention, per head size of [`FA_HEAD_DIMS`]
    fattn: [Func; 3],
    ssm_conv: Func,
    gdn_prep: Func,
    gdn: Func,
    moe_route: Func,
    moe_sort: Func,
    moe_combine: Func,
}

impl Kernels {
    /// The machine code of the build's GPU, or the PTX compiled by the
    /// driver for another one
    fn load(d: &Driver) -> Result<Self, Error> {
        let m = match d.module(CUBIN, false) {
            Ok(m) => m,
            Err(cubin) => d.module(PTX.as_bytes(), true).map_err(|ptx| {
                backend(format!(
                    "{ptx} (and the kernels built for {ARCH}: {cubin}); \
                     build for this GPU (CUDA_COMPUTE_CAP), or update the driver"
                ))
            })?,
        };
        let f = |name: &str| d.function(m, name);
        let mut mm = Vec::with_capacity(TYPES.len());
        let mut mm_id = Vec::with_capacity(TYPES.len());
        let mut mm_group = Vec::with_capacity(TYPES.len());
        let mut mm_glu = Vec::with_capacity(TYPES.len());
        let mut mma = Vec::with_capacity(TYPES.len());
        let mut mma_id = Vec::with_capacity(TYPES.len());
        let mut mmq = Vec::with_capacity(TYPES.len());
        let mut mmq_id = Vec::with_capacity(TYPES.len());
        for (ty, name) in TYPES {
            let int8 = matches!(
                ty,
                GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q8_0 | GgmlType::Q4K | GgmlType::Q5K
            );
            mmq.push(int8.then(|| f(&format!("mmq_{name}"))).transpose()?);
            mmq_id.push(int8.then(|| f(&format!("mmq_id_{name}"))).transpose()?);
            mm.push(f(&format!("mm_{name}"))?);
            mm_id.push(f(&format!("mm_id_{name}"))?);
            mm_group.push(f(&format!("mm_group_{name}"))?);
            mm_glu.push(f(&format!("mm_glu_{name}"))?);
            mma.push(f(&format!("mma_{name}"))?);
            mma_id.push(f(&format!("mma_id_{name}"))?);
        }
        Ok(Kernels {
            mm: mm.try_into().unwrap_or_else(|_| unreachable!()),
            mm_id: mm_id.try_into().unwrap_or_else(|_| unreachable!()),
            mm_group: mm_group.try_into().unwrap_or_else(|_| unreachable!()),
            mm_glu: mm_glu.try_into().unwrap_or_else(|_| unreachable!()),
            mma: mma.try_into().unwrap_or_else(|_| unreachable!()),
            mma_id: mma_id.try_into().unwrap_or_else(|_| unreachable!()),
            mmq: mmq.try_into().unwrap_or_else(|_| unreachable!()),
            mmq_id: mmq_id.try_into().unwrap_or_else(|_| unreachable!()),
            quantize: f("quantize_q8")?,
            add: f("add_inplace")?,
            swiglu: f("swiglu")?,
            rms_norm: f("rms_norm")?,
            attn_prep: f("attn_prep")?,
            attn: f("attn")?,
            fattn: {
                let mut fs = Vec::new();
                for hd in FA_HEAD_DIMS {
                    let k = f(&format!("fattn_{hd}"))?;
                    d.max_shared(k, fattn_shared(hd))?;
                    fs.push(k);
                }
                fs.try_into().unwrap_or_else(|_| unreachable!())
            },
            ssm_conv: f("ssm_conv")?,
            gdn_prep: f("gdn_prep")?,
            gdn: f("gdn")?,
            moe_route: f("moe_route")?,
            moe_sort: f("moe_sort")?,
            moe_combine: f("moe_combine")?,
        })
    }
}

/// A weight matrix in memory the device reads
struct GpuWeight {
    ty: usize,
    cols: usize,
    rows: usize,
    row_bytes: usize,
    buf: Arc<Buf>,
}

/// A stack of expert matrices, see [`Experts`]
struct GpuExperts {
    ty: usize,
    cols: usize,
    rows: usize,
    row_bytes: usize,
    stride: usize,
    offset: usize,
    /// experts
    n: usize,
    buf: Arc<Buf>,
}

/// Slots of the [`Stage`] ring
const STAGE_SLOTS: usize = 3;

/// Device memory where a pass copies the tensors it reads in host memory
/// before their matmuls: they cross the bus at the copy engine's speed, where
/// the matmuls' reads reach half of it, and a prefill's once rather than once
/// per tile of tokens.
///
/// The feed-forward matrices go through a ring of slots of the largest, copied
/// on a stream of their own ahead of the matmuls, from the start of the pass:
/// the copies overlap the layers in device memory and the matmuls. Without a
/// ring (experts in host memory), a prefill copies an experts tensor to the
/// whole stage before its matmuls.
struct Stage {
    buf: Buf,
    /// bytes of a slot, 0 without a ring
    slot: usize,
    /// the experts tensor the stage holds a copy of, 0 when unknown: a graph's
    /// replay copies without the host knowing
    holds: AtomicU64,
    copier: Stream,
    /// the start of the pass, the end of a slot's copy, and of its matmul
    start: Event,
    copied: [Event; STAGE_SLOTS],
    freed: [Event; STAGE_SLOTS],
    plan: std::sync::Mutex<Plan>,
    /// held by a pass, as the stage and its events are its own until its end
    lock: std::sync::Mutex<()>,
}

/// The matrices a pass stages through the ring, in the order of its matmuls:
/// item `i` goes to slot `i % STAGE_SLOTS`
#[derive(Default)]
struct Plan {
    items: Vec<Arc<Buf>>,
    /// items whose copy is queued, given to a matmul, its matmul queued
    copying: usize,
    taken: usize,
    done: usize,
}

enum GpuMixer {
    Attention {
        wq: GpuWeight,
        wk: GpuWeight,
        wv: GpuWeight,
        wo: GpuWeight,
        q_norm: Buf,
        k_norm: Buf,
        sinks: Option<Buf>,
        rope: bool,
        window: Option<usize>,
    },
    DeltaNet {
        wqkv: GpuWeight,
        wgate: GpuWeight,
        wbeta: GpuWeight,
        walpha: GpuWeight,
        wout: GpuWeight,
        conv1d: Buf,
        dt_bias: Buf,
        a: Buf,
        norm: Buf,
    },
}

struct GpuShared {
    gate: GpuWeight,
    up: GpuWeight,
    down: GpuWeight,
    gate_inp: Option<GpuWeight>,
}

struct GpuMoe {
    router: GpuWeight,
    bias: Option<Buf>,
    gate: GpuExperts,
    up: GpuExperts,
    down: GpuExperts,
    shared: Option<GpuShared>,
}

enum GpuFfn {
    Dense {
        gate: GpuWeight,
        up: GpuWeight,
        down: GpuWeight,
    },
    Moe(Box<GpuMoe>),
}

struct GpuLayer {
    attn_norm: Buf,
    post_norm: Buf,
    mixer: GpuMixer,
    ffn: GpuFfn,
}

/// Where the weights go: the device while its budget lasts, else mapped host
/// memory. A tensor two weights share (a fused gate and up, a tied output
/// head) is copied once.
struct Placer {
    budget: usize,
    placed: BTreeMap<(usize, usize), Arc<Buf>>,
    device: usize,
    host: usize,
    /// bytes of the largest matrix and experts tensor in host memory
    largest_host: usize,
    largest_host_experts: usize,
    /// a dry run: the budget's accounting, no memory
    dry: bool,
    /// the host memory of the tensors that go there, and its bytes used:
    /// pinned at once before reading the model's data, as pinned memory the
    /// device reads at half the speed when pinned once the system's memory
    /// holds the data read (its pages scattered, it seems)
    arena: Option<(Arc<Buf>, usize)>,
}

/// The rows of `ty` the kernels read split: the quants of every block, then
/// the scales, each row padded to 16 bytes (see the kernels)
fn split_rows(ty: GgmlType) -> bool {
    split_layout(ty).is_some()
}

/// Bytes on the device of the largest feed-forward matrix and experts tensor
/// of `model`
fn largest_ffn(model: &Model<'_>) -> (usize, usize) {
    let w = |w: &Weight<'_>| w.rows * device_row_bytes(w.ty, w.cols);
    let x = |x: &Experts<'_>| x.data.len() / x.row_bytes() * device_row_bytes(x.ty, x.cols);
    let (mut dense, mut experts) = (0, 0);
    for l in &model.layers {
        match &l.ffn {
            Ffn::Dense { gate, up, down } => dense = dense.max(w(gate).max(w(up)).max(w(down))),
            Ffn::Moe(m) => {
                experts = experts.max(x(&m.gate).max(x(&m.up)).max(x(&m.down)));
                if let Some(s) = &m.shared {
                    dense = dense.max(w(&s.gate).max(w(&s.up)).max(w(&s.down)));
                }
            }
        }
    }
    (dense, experts)
}

/// Bytes of a [`Stage`] for the largest matrix and experts tensor
fn stage_bytes((matrix, experts): (usize, usize)) -> usize {
    match (matrix, experts) {
        (0, 0) => 0,
        (m, 0) => STAGE_SLOTS * m.next_multiple_of(256),
        (_, x) => x,
    }
}

/// (block bytes, offset and bytes of the part that goes after the quants)
fn split_layout(ty: GgmlType) -> Option<(usize, usize, usize)> {
    match ty {
        GgmlType::Q8_0 => Some((34, 0, 2)),
        GgmlType::Q4_0 => Some((18, 0, 2)),
        GgmlType::Q4_1 => Some((20, 0, 4)),
        GgmlType::Q6K => Some((210, 208, 2)),
        _ => None,
    }
}

/// Whether the matvecs of the type at `TYPES[ty]` take int8 activations
fn uses_q8(ty: usize) -> bool {
    !TYPES[ty].0.uses_f32_act()
}

/// Bytes of a row of `ty` and `cols` on the device
fn device_row_bytes(ty: GgmlType, cols: usize) -> usize {
    let bytes = ty.row_size(cols);
    if split_rows(ty) {
        bytes.next_multiple_of(16)
    } else {
        bytes
    }
}

/// The rows of `data` (of `ty` and `cols`) as the device holds them, when not
/// as they are
fn repack(ty: GgmlType, cols: usize, data: &[u8]) -> Option<Vec<u8>> {
    use crate::par::*;
    let (block, at, scale) = split_layout(ty)?;
    let (from, to) = (ty.row_size(cols), device_row_bytes(ty, cols));
    let rows = data.len() / from;
    let quants = block - scale;
    let blocks = from / block;
    let mut out = vec![0u8; rows * to];
    out.par_chunks_mut(to)
        .zip(data.par_chunks(from))
        .for_each(|(o, row)| {
            let (q, s) = o.split_at_mut(blocks * quants);
            for (i, b) in row.chunks_exact(block).enumerate() {
                s[i * scale..(i + 1) * scale].copy_from_slice(&b[at..at + scale]);
                let q = &mut q[i * quants..(i + 1) * quants];
                if at == 0 {
                    q.copy_from_slice(&b[scale..]);
                } else {
                    q.copy_from_slice(&b[..at]);
                }
            }
        });
    Some(out)
}

impl Placer {
    /// `data`, rows of `ty` and `cols`, in memory the device reads
    fn place(&mut self, ty: GgmlType, cols: usize, data: &[u8]) -> Result<Arc<Buf>, Error> {
        let key = (data.as_ptr() as usize, data.len());
        if let Some(b) = self.placed.get(&key) {
            return Ok(b.clone());
        }
        if self.dry {
            let bytes = data.len() / ty.row_size(cols) * device_row_bytes(ty, cols);
            if bytes <= self.budget {
                self.budget -= bytes;
                self.device += bytes;
            } else {
                self.budget = 0;
                self.host += bytes.next_multiple_of(256);
            }
            let b = Arc::new(Buf::none());
            self.placed.insert(key, b.clone());
            return Ok(b);
        }
        let packed = repack(ty, cols, data);
        let data = packed.as_deref().unwrap_or(data);
        let on_device = if data.len() <= self.budget {
            match Buf::from_bytes(data) {
                Ok(b) => Some(b),
                Err(Error::OutOfMemory { .. }) => None,
                Err(e) => return Err(e),
            }
        } else {
            None
        };
        let buf = match on_device {
            Some(b) => {
                self.budget -= data.len();
                self.device += data.len();
                b
            }
            None => {
                // nothing else goes to the device once a tensor didn't fit, so
                // that the layers before stay whole on it
                self.budget = 0;
                self.host += data.len();
                match &mut self.arena {
                    Some((arena, used)) if *used + data.len() <= arena.bytes => {
                        let b = Buf::carve(arena, *used, data);
                        *used += data.len().next_multiple_of(256);
                        b
                    }
                    _ => Buf::mapped(data)?,
                }
            }
        };
        let buf = Arc::new(buf);
        self.placed.insert(key, buf.clone());
        Ok(buf)
    }

    fn weight(&mut self, w: &Weight<'_>) -> Result<GpuWeight, Error> {
        let ty =
            type_index(w.ty).ok_or_else(|| backend(format!("cuda: no kernel for {:?}", w.ty)))?;
        if !w.cols.is_multiple_of(32) {
            return Err(backend("cuda: a weight of a width not a multiple of 32"));
        }
        debug_assert_eq!(w.data.len(), w.rows * w.ty.row_size(w.cols));
        let buf = self.place(w.ty, w.cols, w.data)?;
        if buf.is_host() {
            self.largest_host = self.largest_host.max(buf.bytes);
        }
        Ok(GpuWeight {
            ty,
            cols: w.cols,
            rows: w.rows,
            row_bytes: device_row_bytes(w.ty, w.cols),
            buf,
        })
    }

    fn experts(&mut self, x: &Experts<'_>) -> Result<GpuExperts, Error> {
        let ty =
            type_index(x.ty).ok_or_else(|| backend(format!("cuda: no kernel for {:?}", x.ty)))?;
        if !x.cols.is_multiple_of(32) {
            return Err(backend("cuda: experts of a width not a multiple of 32"));
        }
        // the stride and offset are whole rows
        let (from, to) = (x.row_bytes(), device_row_bytes(x.ty, x.cols));
        let buf = self.place(x.ty, x.cols, x.data)?;
        if buf.is_host() {
            self.largest_host_experts = self.largest_host_experts.max(buf.bytes);
        }
        Ok(GpuExperts {
            ty,
            cols: x.cols,
            rows: x.rows,
            row_bytes: to,
            stride: x.stride / from * to,
            offset: x.offset / from * to,
            n: x.n,
            buf,
        })
    }

    /// The attention or delta net weights of a layer
    fn mixer(&mut self, l: &Layer<'_>) -> Result<GpuMixer, Error> {
        Ok(match &l.mixer {
            Mixer::Attention(a) => GpuMixer::Attention {
                wq: self.weight(&a.wq)?,
                wk: self.weight(&a.wk)?,
                wv: self.weight(&a.wv)?,
                wo: self.weight(&a.wo)?,
                q_norm: Buf::from_f32(&a.q_norm)?,
                k_norm: Buf::from_f32(&a.k_norm)?,
                sinks: a.sinks.as_deref().map(Buf::from_f32).transpose()?,
                rope: a.rope,
                window: a.window,
            },
            Mixer::DeltaNet(d) => GpuMixer::DeltaNet {
                wqkv: self.weight(&d.wqkv)?,
                wgate: self.weight(&d.wgate)?,
                wbeta: self.weight(&d.wbeta)?,
                walpha: self.weight(&d.walpha)?,
                wout: self.weight(&d.wout)?,
                conv1d: Buf::from_f32(&d.conv1d)?,
                dt_bias: Buf::from_f32(&d.dt_bias)?,
                a: Buf::from_f32(&d.a)?,
                norm: Buf::from_f32(&d.norm)?,
            },
        })
    }

    /// The feed-forward weights of a layer but its experts', which go after
    /// every layer's
    fn ffn(&mut self, l: &Layer<'_>) -> Result<GpuFfn, Error> {
        Ok(match &l.ffn {
            Ffn::Dense { gate, up, down } => GpuFfn::Dense {
                gate: self.weight(gate)?,
                up: self.weight(up)?,
                down: self.weight(down)?,
            },
            Ffn::Moe(m) => {
                let shared = match &m.shared {
                    Some(s) => Some(GpuShared {
                        gate: self.weight(&s.gate)?,
                        up: self.weight(&s.up)?,
                        down: self.weight(&s.down)?,
                        gate_inp: s.gate_inp.as_ref().map(|w| self.weight(w)).transpose()?,
                    }),
                    None => None,
                };
                let router = self.weight(&m.router)?;
                // a placeholder until the experts are placed, after every layer
                let none = |x: &Experts<'_>| GpuExperts {
                    ty: 0,
                    cols: x.cols,
                    rows: x.rows,
                    row_bytes: 0,
                    stride: 0,
                    offset: 0,
                    n: x.n,
                    buf: router.buf.clone(),
                };
                GpuFfn::Moe(Box::new(GpuMoe {
                    bias: m.bias.as_deref().map(Buf::from_f32).transpose()?,
                    gate: none(&m.gate),
                    up: none(&m.up),
                    down: none(&m.down),
                    router,
                    shared,
                }))
            }
        })
    }
}

pub struct CudaEngine<'m> {
    model: &'m Model<'m>,
    d: &'static Driver,
    k: Kernels,
    layers: Vec<GpuLayer>,
    output_norm: Buf,
    output: GpuWeight,
    inv_freq: Buf,
    stage: Option<Stage>,
    max_batch: usize,
    name: String,
    /// bytes of weights on the device, and in mapped host memory
    weights_device: usize,
    weights_host: usize,
}

/// The arguments of `w x` into `out`, for the kernels of plain matrices
fn mat_args(w: &GpuWeight, x: &Act<'_>, out: DevPtr) -> MatArgs {
    MatArgs {
        w: w.buf.ptr,
        x: x.x,
        y: out,
        row_bytes: w.row_bytes as i64,
        cols: w.cols as i32,
        rows: w.rows as i32,
        n: x.n as i32,
        x_stride: w.cols as i32,
        y_stride: w.rows as i32,
        pad: 0,
        tiles: 0,
        n_tiles: 0,
        order: 0,
        expert_stride: 0,
        x_div: 1,
        pad2: 0,
        xq: 0,
        xds: 0,
    }
}

fn backend(msg: impl Into<String>) -> Error {
    Error::Backend(msg.into())
}

impl<'m> CudaEngine<'m> {
    /// An engine for passes of up to `max_batch` tokens, leaving
    /// [`DEFAULT_RESERVE`] of device memory for the sessions
    pub fn new(model: &'m Model<'m>, max_batch: usize) -> Result<Self, Error> {
        Self::with_reserve(model, max_batch, DEFAULT_RESERVE)
    }

    /// [`CudaEngine::new`], leaving `reserve` bytes of device memory for the
    /// sessions: the weights that don't fit in the rest go to host memory
    pub fn with_reserve(
        model: &'m Model<'m>,
        max_batch: usize,
        reserve: usize,
    ) -> Result<Self, Error> {
        let c = &model.cfg;
        if c.recurrent.contains(&true) && (c.ssm_state != SD || c.conv_kernel != 4) {
            return Err(backend(
                "cuda: the delta net kernels expect state size 128 and conv kernel 4",
            ));
        }
        if c.recurrent.contains(&false)
            && !(c.head_dim.is_multiple_of(32)
                && c.head_dim <= 512
                && c.n_head_kv > 0
                && c.n_head.is_multiple_of(c.n_head_kv)
                && c.n_rot.is_multiple_of(64)
                && c.n_rot <= c.head_dim)
        {
            return Err(backend("cuda: unsupported attention head geometry"));
        }
        if let Some(m) = &c.moe
            && m.n_expert > 4096
        {
            return Err(backend("cuda: more than 4096 experts"));
        }

        let d = driver()?;
        d.bind();
        if !d.can_map_host() {
            return Err(backend("cuda: the device can't map host memory"));
        }
        let k = Kernels::load(d)?;
        d.max_shared(k.attn, ATTN_WARPS * c.head_dim * 4)?;

        let params = 4 * model.memory.params;
        let (free, _) = d.memory_info();
        let placer = |dry, arena| {
            let mut p = Placer {
                budget: free.saturating_sub(reserve.saturating_add(params).saturating_add(SLACK)),
                placed: BTreeMap::new(),
                device: 0,
                host: 0,
                largest_host: 0,
                largest_host_experts: 0,
                dry,
                arena,
            };
            // the room of the stage, when the weights don't all fit
            if model.memory.weights > p.budget {
                p.budget = p.budget.saturating_sub(stage_bytes(largest_ffn(model)));
            }
            p
        };
        let mut dry = placer(true, None);
        Self::place(model, &mut dry)?;
        let arena = match dry.host {
            0 => None,
            n => Buf::mapped_uninit(n).ok().map(|b| (Arc::new(b), 0)),
        };
        let mut placer = placer(false, arena);
        let (output, layers) = Self::place(model, &mut placer)?;
        // what the arena holds more (a tensor that didn't fit the device after
        // all) is in its own memory
        placer.arena = None;

        Ok(CudaEngine {
            output_norm: Buf::from_f32(&model.output_norm)?,
            inv_freq: Buf::from_f32(&model.inv_freq)?,
            output,
            layers,
            name: d.name(),

            // without the room, the matmuls read host memory as they go
            stage: match stage_bytes((placer.largest_host, placer.largest_host_experts)) {
                0 => None,
                n => match Buf::zeroed(n) {
                    Ok(buf) => Some(Stage {
                        buf,
                        slot: match placer.largest_host_experts {
                            0 => placer.largest_host.next_multiple_of(256),
                            _ => 0,
                        },
                        holds: AtomicU64::new(0),
                        copier: Stream::new()?,
                        start: Event::new()?,
                        copied: [Event::new()?, Event::new()?, Event::new()?],
                        freed: [Event::new()?, Event::new()?, Event::new()?],
                        plan: std::sync::Mutex::new(Plan::default()),
                        lock: std::sync::Mutex::new(()),
                    }),
                    Err(_) => None,
                },
            },
            weights_device: placer.device,
            weights_host: placer.host,
            model,
            d,
            k,
            max_batch: max_batch.max(1),
        })
    }

    /// The model's weights through `placer`, in the order of their priority
    /// for device memory
    fn place(model: &Model<'_>, placer: &mut Placer) -> Result<(GpuWeight, Vec<GpuLayer>), Error> {
        // the output head first: it is read for every token's logits
        let output = placer.weight(&model.output)?;
        // then the layers' attention and delta nets (small, read at every
        // token), their feed-forwards, and the experts: what goes to host
        // memory is the large matrices of the last layers
        let mixers = model
            .layers
            .iter()
            .map(|l| placer.mixer(l))
            .collect::<Result<Vec<_>, _>>()?;
        let ffns = model
            .layers
            .iter()
            .map(|l| placer.ffn(l))
            .collect::<Result<Vec<_>, _>>()?;
        let mut layers = model
            .layers
            .iter()
            .zip(mixers.into_iter().zip(ffns))
            .map(|(l, (mixer, ffn))| {
                Ok(GpuLayer {
                    attn_norm: Buf::from_f32(&l.attn_norm)?,
                    post_norm: Buf::from_f32(&l.post_norm)?,
                    mixer,
                    ffn,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        for (gl, l) in layers.iter_mut().zip(&model.layers) {
            if let (GpuFfn::Moe(gm), Ffn::Moe(m)) = (&mut gl.ffn, &l.ffn) {
                gm.gate = placer.experts(&m.gate)?;
                gm.up = placer.experts(&m.up)?;
                gm.down = placer.experts(&m.down)?;
            }
        }

        Ok((output, layers))
    }

    fn cfg(&self) -> &Config {
        &self.model.cfg
    }

    /// Plan the ring of the stage for a pass: the feed-forward matrices in
    /// host memory, and queue the first copies. The pass holds the stage's
    /// lock (`CudaSession::pass`).
    fn stage_begin(&self) {
        let Some(st) = self.stage.as_ref().filter(|st| st.slot > 0) else {
            return;
        };
        let mut plan = st.plan.lock().unwrap_or_else(|p| p.into_inner());
        *plan = Plan::default();
        for l in &self.layers {
            if let GpuFfn::Dense { gate, up, down } = &l.ffn {
                for w in [gate, up, down] {
                    if w.buf.is_host() && w.buf.bytes <= st.slot {
                        plan.items.push(w.buf.clone());
                    }
                }
            }
        }
        if !plan.items.is_empty() {
            st.start.record();
            self.stage_copies(st, &mut plan);
        }
    }

    /// Queue the copies of the items whose slot is free, or will be once the
    /// matmuls queued are done
    fn stage_copies(&self, st: &Stage, plan: &mut Plan) {
        let end = (plan.done + STAGE_SLOTS).min(plan.items.len());
        if plan.copying >= end {
            return;
        }
        st.copier.run(|| {
            if plan.copying == 0 {
                st.start.wait();
            }
            while plan.copying < end {
                let i = plan.copying;
                let k = i % STAGE_SLOTS;
                if i >= STAGE_SLOTS {
                    st.freed[k].wait();
                }
                plan.items[i].copy_to_device(st.buf.at(k * st.slot), plan.items[i].bytes);
                st.copied[k].record();
                plan.copying += 1;
            }
        });
    }

    /// Where the matmul to queue next reads `buf`: its slot in the ring once
    /// copied, when it is the next item of the plan; `stage_done` after the
    /// matmul is queued. Else host memory.
    fn staged(&self, buf: &Buf) -> Option<DevPtr> {
        let st = self.stage.as_ref().filter(|st| st.slot > 0)?;
        let mut plan = st.plan.lock().unwrap_or_else(|p| p.into_inner());
        let i = plan.taken;
        if plan.items.get(i).is_none_or(|b| b.ptr != buf.ptr) {
            return None;
        }
        // at most 2 items of a matmul (gate and up) are taken and not done
        debug_assert!(i < plan.copying);
        let k = i % STAGE_SLOTS;
        st.copied[k].wait();
        plan.taken += 1;
        Some(st.buf.at(k * st.slot))
    }

    /// After queueing the matmul of the items `staged` gave: their slots free
    /// once it is done, for the next copies
    fn stage_done(&self) {
        let Some(st) = self.stage.as_ref().filter(|st| st.slot > 0) else {
            return;
        };
        let mut plan = st.plan.lock().unwrap_or_else(|p| p.into_inner());
        if plan.done == plan.taken {
            return;
        }
        while plan.done < plan.taken {
            st.freed[plan.done % STAGE_SLOTS].record();
            plan.done += 1;
        }
        self.stage_copies(st, &mut plan);
    }

    /// At the end of the pass: the current stream waits for any copy queued
    /// (a captured stream must join the capture's)
    fn stage_end(&self) {
        let Some(st) = self.stage.as_ref().filter(|st| st.slot > 0) else {
            return;
        };
        let mut plan = st.plan.lock().unwrap_or_else(|p| p.into_inner());
        debug_assert_eq!(plan.taken, plan.items.len(), "staged matrices not read");
        for i in plan.taken..plan.copying {
            st.copied[i % STAGE_SLOTS].wait();
        }
        *plan = Plan::default();
    }

    /// Where the matmuls of a prefill read the experts tensor `buf`: the
    /// stage after a copy to it, when `buf` is in host memory and fits
    fn staged_experts(&self, buf: &Buf) -> Option<DevPtr> {
        let st = self.stage.as_ref().filter(|st| st.slot == 0)?;
        if !buf.is_host() || buf.bytes > st.buf.bytes {
            return None;
        }
        if st.holds.swap(buf.ptr, Ordering::Relaxed) != buf.ptr {
            buf.copy_to_device(st.buf.ptr, buf.bytes);
        }
        Some(st.buf.ptr)
    }

    /// The int8 activations of `x`, quantized the first time
    fn q8(&self, x: &Act<'_>) -> (DevPtr, DevPtr) {
        if !x.quantized.get() {
            let args = QuantArgs {
                x: x.x,
                q: x.q8.q.ptr,
                ds: x.q8.ds.ptr,
                cols: x.cols as i32,
                pad: 0,
            };
            let grid = [(x.cols / 32).div_ceil(8) as u32, x.n as u32, 1];
            self.d.launch(self.k.quantize, grid, [256, 1, 1], 0, &args);
            x.quantized.set(true);
        }
        (x.q8.q.ptr, x.q8.ds.ptr)
    }

    /// `out[t][r] = dot(w.row(r), x[t])` for the rows of `x`
    fn matmul(&self, w: &GpuWeight, x: &Act<'_>, out: DevPtr) {
        let n = x.n;
        let mut args = mat_args(w, x, out);
        debug_assert_eq!(x.cols, w.cols);
        // the scalar kernel for a few rows: the tiles of the tensor cores
        // would cover the rows with few blocks
        if n > NT && w.rows > 64 {
            if let Some(p) = self.staged(&w.buf) {
                args.w = p;
            }
            let grid = [w.rows.div_ceil(TM) as u32, n.div_ceil(TN) as u32, 1];
            let f = match self.k.mmq[w.ty] {
                Some(f) => {
                    (args.xq, args.xds) = self.q8(x);
                    f
                }
                None => self.k.mma[w.ty],
            };
            self.d.launch(f, grid, [MMA_THREADS as u32, 1, 1], 0, &args);
            self.stage_done();
        } else {
            if uses_q8(w.ty) {
                (args.xq, args.xds) = self.q8(x);
            }
            if let Some(p) = self.staged(&w.buf) {
                args.w = p;
            }
            let grid = [w.rows.div_ceil(MM_ROWS) as u32, n.div_ceil(NT) as u32, 1];
            self.d.launch(
                self.k.mm[w.ty],
                grid,
                [(32 * MM_ROWS) as u32, 1, 1],
                0,
                &args,
            );
            self.stage_done();
        }
    }

    /// The matmuls of `ws` (weight, output) of one input: one launch when
    /// they are matvecs of one type
    fn matmuls(&self, ws: &[(&GpuWeight, DevPtr)], x: &Act<'_>) {
        let ty = ws[0].0.ty;
        if x.n > NT || ws.len() > GROUP || ws.iter().any(|(w, _)| w.ty != ty) {
            for &(w, out) in ws {
                self.matmul(w, x, out);
            }
            return;
        }
        let first = mat_args(ws[0].0, x, ws[0].1);
        let mut g = GroupArgs {
            m: [first; GROUP],
            end: [0; GROUP],
            count: ws.len() as i32,
            pad: 0,
        };
        let mut blocks = 0;
        let q8 = if uses_q8(ty) { self.q8(x) } else { (0, 0) };
        for (i, &(w, out)) in ws.iter().enumerate() {
            g.m[i] = mat_args(w, x, out);
            (g.m[i].xq, g.m[i].xds) = q8;
            blocks += w.rows.div_ceil(MM_ROWS);
            g.end[i] = blocks as i32;
        }
        self.d.launch(
            self.k.mm_group[ty],
            [blocks as u32, 1, 1],
            [(32 * MM_ROWS) as u32, 1, 1],
            0,
            &g,
        );
    }

    /// `silu(gate x) * up x` into `a1`, through `a2`
    fn glu(&self, s: &Scratch, gate: &GpuWeight, up: &GpuWeight, x: &Act<'_>) {
        if x.n <= NT && gate.ty == up.ty && gate.rows == up.rows {
            let q8 = if uses_q8(gate.ty) { self.q8(x) } else { (0, 0) };
            let mut g = mat_args(gate, x, s.a1.ptr);
            let mut u = mat_args(up, x, s.a2.ptr);
            (g.xq, g.xds) = q8;
            (u.xq, u.xds) = q8;
            if let Some(p) = self.staged(&gate.buf) {
                g.w = p;
            }
            if let Some(p) = self.staged(&up.buf) {
                u.w = p;
            }
            let args = GroupArgs {
                m: [g, u, g, g],
                end: [0; GROUP],
                count: 2,
                pad: 0,
            };
            self.d.launch(
                self.k.mm_glu[gate.ty],
                [gate.rows.div_ceil(MM_ROWS) as u32, 1, 1],
                [(32 * MM_ROWS) as u32, 1, 1],
                0,
                &args,
            );
            self.stage_done();
        } else {
            self.matmul(gate, x, s.a1.ptr);
            self.matmul(up, x, s.a2.ptr);
            self.elementwise(self.k.swiglu, s.a1.ptr, s.a2.ptr, x.n * gate.rows);
        }
    }

    /// The pairs of `ms` through their experts' matrices: input row `p /
    /// x_div` of `x`, output row `p`. The tiles are of [`TN`] pairs with
    /// `mma`, else of [`NT`].
    #[allow(clippy::too_many_arguments)]
    fn matmul_id(
        &self,
        w: &GpuExperts,
        ms: &MoeScratch,
        x: &Act<'_>,
        x_div: usize,
        max_tiles: usize,
        mma: bool,
        out: DevPtr,
    ) {
        let mut args = MatArgs {
            w: w.buf.at(w.offset),
            x: x.x,
            y: out,
            row_bytes: w.row_bytes as i64,
            cols: w.cols as i32,
            rows: w.rows as i32,
            n: 0,
            x_stride: x.cols as i32,
            y_stride: w.rows as i32,
            pad: 0,
            tiles: ms.tiles.ptr,
            n_tiles: ms.n_tiles.ptr,
            order: ms.order.ptr,
            expert_stride: w.stride as i64,
            x_div: x_div as i32,
            pad2: 0,
            xq: 0,
            xds: 0,
        };
        debug_assert_eq!(x.cols, w.cols);
        if mma {
            // when the pairs touch most experts: (1 - k / n)^n < 1 / 8 or so
            if x.n * x_div >= 2 * w.n
                && let Some(p) = self.staged_experts(&w.buf)
            {
                args.w = p + w.offset as DevPtr;
            }
            let grid = [w.rows.div_ceil(TM) as u32, max_tiles as u32, 1];
            let f = match self.k.mmq_id[w.ty] {
                Some(f) => {
                    (args.xq, args.xds) = self.q8(x);
                    f
                }
                None => self.k.mma_id[w.ty],
            };
            self.d.launch(f, grid, [MMA_THREADS as u32, 1, 1], 0, &args);
        } else {
            if uses_q8(w.ty) {
                (args.xq, args.xds) = self.q8(x);
            }
            let grid = [w.rows.div_ceil(MM_ROWS) as u32, max_tiles as u32, 1];
            self.d.launch(
                self.k.mm_id[w.ty],
                grid,
                [(32 * MM_ROWS) as u32, 1, 1],
                0,
                &args,
            );
        }
    }

    /// rms norm of `rows` rows of `width`, times silu(gate) when gated, into
    /// rows `ostride` apart; with `add`, after `x += add`
    #[allow(clippy::too_many_arguments)]
    fn rms_norm(
        &self,
        x: DevPtr,
        add: Option<DevPtr>,
        w: &Buf,
        out: DevPtr,
        gate: Option<DevPtr>,
        rows: usize,
        width: usize,
        ostride: usize,
    ) {
        let args = NormArgs {
            x,
            w: w.ptr,
            out,
            gate: gate.unwrap_or(x),
            y: add.unwrap_or(x),
            n: width as i32,
            ostride: ostride as i32,
            gated: gate.is_some() as i32,
            add: add.is_some() as i32,
            eps: self.cfg().eps,
            pad: 0,
        };
        let threads = if width > 1024 { 256 } else { 32 };
        self.d.launch(
            self.k.rms_norm,
            [rows as u32, 1, 1],
            [threads, 1, 1],
            0,
            &args,
        );
    }

    fn elementwise(&self, f: Func, a: DevPtr, b: DevPtr, n: usize) {
        let args = EltArgs {
            a,
            b,
            n: n as i32,
            pad: 0,
        };
        self.d
            .launch(f, [n.div_ceil(256) as u32, 1, 1], [256, 1, 1], 0, &args);
    }
}

impl Engine for CudaEngine<'_> {
    fn new_session(&self, context: usize, draft: usize) -> Result<Box<dyn Session + '_>, Error> {
        Ok(Box::new(CudaSession::new(self, context, draft)?))
    }

    fn batch_size(&self) -> usize {
        self.max_batch
    }

    // no MTP drafts yet
    fn draft_size(&self) -> usize {
        0
    }

    fn description(&self) -> String {
        let (major, minor) = self.d.compute_capability();
        let v = self.d.version();
        let mut s = format!(
            "cuda {}.{}, {} (sm_{major}{minor}, {} SMs), weights: {:.1} GB on the GPU",
            v / 1000,
            v % 1000 / 10,
            self.name,
            self.d.multiprocessors(),
            self.weights_device as f64 / 1e9,
        );
        if self.weights_host > 0 {
            s += &format!(", {:.1} GB in host memory", self.weights_host as f64 / 1e9);
        }
        s
    }

    // the small vectors are uploaded, the weights copied out of the mapping
    fn model_memory(&self) -> ModelMemory {
        let m = self.model.memory;
        ModelMemory {
            params: 2 * m.params,
            ..m
        }
    }

    fn session_memory(&self, context: usize, draft: usize) -> SessionMemory {
        self.model
            .shape()
            .cuda_session(context, draft, self.max_batch)
    }

    fn device_memory(&self) -> Option<DeviceMemory> {
        Some(DeviceMemory {
            allocated: self.d.device_bytes.load(Ordering::Relaxed),
            working_set: self.d.total_memory(),
        })
    }

    fn numerics(&self) -> Vec<(&'static str, String)> {
        vec![
            ("kind", "cuda".into()),
            ("device", self.name.clone()),
            ("kv_cache", "f16".into()),
        ]
    }
}

/// Bytes a buffer of `n` takes: at least the 4 a buffer of nothing gets
fn sz(n: usize) -> usize {
    n.max(4)
}

/// Bytes of a full kv cache, keys or values: f16
fn kv_bytes(context: usize, kv_dim: usize) -> usize {
    2 * context.saturating_mul(kv_dim)
}

/// Width of a dense feed-forward's activations: the dense layers', or the
/// shared expert's
fn ffn_width(c: &Config) -> usize {
    let dense = if c.moe.is_none() || c.n_dense_lead > 0 {
        c.n_ff
    } else {
        0
    };
    c.moe.as_ref().map_or(0, |m| m.n_ff_shexp).max(dense)
}

/// Bytes of the buffers of [`Scratch`] but `moe`, in the order of its fields
fn scratch_bytes(c: &Config, b: usize, logit_rows: usize) -> [usize; 18] {
    let (e, hd) = (c.n_embd, c.head_dim);
    let qd = c.n_head * hd;
    let kv_dim = c.kv_dim();
    let q_rows = if c.attn_gate { 2 * qd } else { qd };
    let ffn = ffn_width(c);
    let a1 = q_rows.max(c.conv_dim()).max(ffn);
    let a2 = c.d_inner().max(ffn);
    let o = qd.max(c.d_inner());
    let hv = c.ssm_v_heads;
    // the int8 inputs of the matmuls: the normalized hidden states, and the
    // others (attention / delta net outputs, feed-forward activations, the
    // pairs of a mixture of experts, the output head's input)
    let q0 = b * e;
    let pairs = c.moe.as_ref().map_or(0, |m| b * m.n_expert_used * m.n_ff_exp);
    let q1 = (b * qd.max(c.d_inner()).max(ffn))
        .max(pairs)
        .max(logit_rows * e);
    [
        b * e,
        b * e,
        b * e,
        b * a1,
        b * a2,
        b * kv_dim,
        b * kv_dim,
        b * qd,
        b * o,
        b * c.conv_dim(),
        b * hv,
        b * hv,
        b * e,
        logit_rows * c.n_vocab,
    ]
    .map(|n| sz(4 * n))
    .into_iter()
    .chain([q0, q0 / 32 * 8, q1, q1 / 32 * 8].map(sz))
    .collect::<Vec<_>>()
    .try_into()
    .unwrap_or_else(|_| unreachable!())
}

/// Tiles of `moe_sort` of up to `tile` pairs for `pairs` pairs, at most: the
/// full tiles and a last one per expert
fn max_tiles(n_expert: usize, pairs: usize, tile: usize) -> usize {
    pairs / tile + n_expert.min(pairs)
}

/// Bytes of the buffers of [`MoeScratch`], in the order of its fields
fn moe_bytes(c: &Config, b: usize) -> Option<[usize; 11]> {
    let m = c.moe.as_ref()?;
    let e = c.n_embd;
    let pairs = b * m.n_expert_used;
    Some(
        [
            4 * b * m.n_expert,
            4 * pairs,
            4 * pairs,
            4 * pairs,
            16 * max_tiles(m.n_expert, pairs, NT),
            4,
            4 * pairs * m.n_ff_exp,
            4 * pairs * m.n_ff_exp,
            4 * pairs * e,
            4 * b,
            4 * b * e,
        ]
        .map(sz),
    )
}

impl Shape<'_> {
    /// Tokens a verify pass checks after its first, at most: a verify pass is
    /// one pass
    fn cuda_draft(&self, draft: usize, max_batch: usize) -> usize {
        draft.min(max_batch.max(1) - 1)
    }

    /// Positions the caches of a sliding window layer hold, when fewer than
    /// the context: a pass writes its rows before its attention, so the
    /// window but a token's own position, and the pass's positions. A power
    /// of two, so that position `p` is at row `p & (ring - 1)`.
    fn cuda_ring(&self, context: usize, max_batch: usize) -> Option<usize> {
        let ring = (self.cfg.sliding_window + max_batch.max(1) - 1).next_power_of_two();
        (ring < context).then_some(ring)
    }

    /// Bytes of a sliding window layer's cache, keys or values
    fn cuda_window_bytes(&self, context: usize, max_batch: usize) -> usize {
        match self.cuda_ring(context, max_batch) {
            Some(ring) => kv_bytes(ring, self.kv_dim()),
            None => kv_bytes(context, self.kv_dim()),
        }
    }

    /// What the scratch of a session allocates, for passes of `b` tokens and
    /// `logit_rows` rows of logits
    fn cuda_scratch(&self, b: usize, logit_rows: usize) -> usize {
        let c = self.cfg;
        let mut bytes: usize = scratch_bytes(c, b, logit_rows).iter().sum();
        if let Some(moe) = moe_bytes(c, b) {
            bytes += moe.iter().sum::<usize>();
        }
        bytes
    }

    /// What [`CudaEngine::new_session`](Engine::new_session) allocates
    pub(crate) fn cuda_session(
        &self,
        context: usize,
        draft: usize,
        max_batch: usize,
    ) -> SessionMemory {
        let c = self.cfg;
        let b = max_batch.max(1);
        let slots = self.cuda_draft(draft, b) + 1;
        let attention = self.attention_layers();
        let (conv, s) = self.delta_net_state();
        let (window, ring) = (self.window_layers(), self.cuda_ring(context, b));
        let window_bytes = window * 2 * sz(self.cuda_window_bytes(context, b));
        SessionMemory {
            kv: attention * 2 * sz(kv_bytes(context, self.kv_dim()))
                + if ring.is_none() { window_bytes } else { 0 },
            recurrent: self.delta_net_layers() * (sz(slots * 4 * conv) + sz(slots * 4 * s))
                + if ring.is_some() { window_bytes } else { 0 },
            draft_head: 0,
            scratch: self.cuda_scratch(b, slots),
            // the logits of a verify pass and the embeddings of a pass, on the host
            transient: 4 * slots * c.n_vocab + 4 * b * c.n_embd,
            kv_per_token: (attention + if ring.is_none() { window } else { 0 })
                * 2
                * 2
                * self.kv_dim(),
        }
    }
}

/// Activation buffers, sized for passes of `batch` tokens and `logit_rows`
/// rows of logits
struct Scratch {
    batch: usize,
    logit_rows: usize,
    x: Buf,
    h: Buf,
    y: Buf,
    /// q+gate / qkv / ffn gate
    a1: Buf,
    /// z / ffn up
    a2: Buf,
    k: Buf,
    v: Buf,
    q: Buf,
    /// attention / delta net output
    o: Buf,
    conv_out: Buf,
    alpha: Buf,
    beta: Buf,
    /// normalized hidden states, the input of the output head
    feat: Buf,
    logits: Buf,
    /// int8 inputs: of the normalized hidden states `h`, of the others
    q8: [Q8Buf; 2],
    moe: Option<MoeScratch>,
}

/// Int8 activations, see `quantize_q8`
struct Q8Buf {
    q: Buf,
    ds: Buf,
}

/// The input of matmuls: `n` rows of `cols` floats, and their int8 version
/// in `q8`, quantized when a kernel first needs it
struct Act<'s> {
    x: DevPtr,
    n: usize,
    cols: usize,
    q8: &'s Q8Buf,
    quantized: Cell<bool>,
}

impl<'s> Act<'s> {
    fn new(x: DevPtr, n: usize, cols: usize, q8: &'s Q8Buf) -> Self {
        Act {
            x,
            n,
            cols,
            q8,
            quantized: Cell::new(false),
        }
    }
}

/// The buffers of a mixture of experts feed-forward, see `moe_route`
struct MoeScratch {
    /// router logits, per token
    logits: Buf,
    /// expert and weight of each pair
    ids: Buf,
    weights: Buf,
    /// pairs by expert, the tiles of them, and the count of tiles
    order: Buf,
    tiles: Buf,
    n_tiles: Buf,
    /// gate and up activations of each pair
    a1: Buf,
    a2: Buf,
    /// output of each pair
    out: Buf,
    /// the shared expert's gate logit, and output, per token
    gate: Buf,
    shared: Buf,
}

impl Scratch {
    fn new(c: &Config, b: usize, logit_rows: usize) -> Result<Self, Error> {
        let [
            x,
            h,
            y,
            a1,
            a2,
            k,
            v,
            q,
            o,
            conv_out,
            alpha,
            beta,
            feat,
            logits,
            q0,
            ds0,
            q1,
            ds1,
        ] = scratch_bytes(c, b, logit_rows).map(Buf::zeroed);
        let moe = match moe_bytes(c, b) {
            Some(sizes) => {
                let [
                    logits,
                    ids,
                    weights,
                    order,
                    tiles,
                    n_tiles,
                    a1,
                    a2,
                    out,
                    gate,
                    shared,
                ] = sizes.map(Buf::zeroed);
                Some(MoeScratch {
                    logits: logits?,
                    ids: ids?,
                    weights: weights?,
                    order: order?,
                    tiles: tiles?,
                    n_tiles: n_tiles?,
                    a1: a1?,
                    a2: a2?,
                    out: out?,
                    gate: gate?,
                    shared: shared?,
                })
            }
            None => None,
        };
        Ok(Scratch {
            batch: b,
            logit_rows,
            x: x?,
            h: h?,
            y: y?,
            a1: a1?,
            a2: a2?,
            k: k?,
            v: v?,
            q: q?,
            o: o?,
            conv_out: conv_out?,
            alpha: alpha?,
            beta: beta?,
            feat: feat?,
            logits: logits?,
            q8: [
                Q8Buf { q: q0?, ds: ds0? },
                Q8Buf { q: q1?, ds: ds1? },
            ],
            moe,
        })
    }
}

enum LayerState {
    /// with `ring`, a sliding window's: position `p` in row `p % ring`, see
    /// [`Shape::cuda_ring`]
    Attention { k: Buf, v: Buf, ring: Option<usize> },
    /// `slots` copies of each state, see `ssm_conv` in the kernels
    DeltaNet { conv: Buf, s: Buf },
}

struct CudaSnapshot {
    layers: Vec<LayerSnapshot>,
}

enum LayerSnapshot {
    Attention { k: Vec<u8>, v: Vec<u8> },
    DeltaNet { conv: Vec<u8>, s: Vec<u8> },
}

/// Positions of attention layer `i` a next token may see, of the `n_past`
/// before it: all, or a sliding window's but its own
fn window_rows(c: &Config, i: usize, n_past: usize) -> usize {
    if c.is_swa(i) {
        n_past.min(c.sliding_window - 1)
    } else {
        n_past
    }
}

struct CudaSession<'e> {
    e: &'e CudaEngine<'e>,
    layers: Vec<LayerState>,
    s: Scratch,
    /// copies of the delta net states (one more than the draft length), and
    /// the current one
    slots: usize,
    slot: usize,
    /// first position of the last verify, until `keep`
    verify_from: Option<usize>,
    context: usize,
    n_past: usize,
    memory: SessionMemory,
    stream: Stream,
    /// the device's [`PassParams`], and their copy, the embeddings and the
    /// logits of a graph pass in pinned host memory
    params: Buf,
    pinned_params: HostBuf,
    pinned_x: HostBuf,
    pinned_logits: HostBuf,
    /// the graphs of the passes of up to [`GRAPH_TOKENS`] tokens, by (tokens,
    /// logit rows, verify)
    graphs: Vec<((usize, usize, bool), Graph)>,
    use_graphs: bool,
}

impl<'e> CudaSession<'e> {
    fn new(e: &'e CudaEngine<'e>, context: usize, draft: usize) -> Result<Self, Error> {
        let c = e.cfg();
        let shape = e.model.shape();
        let b = e.max_batch;
        let memory = shape.cuda_session(context, draft, b);
        let slots = shape.cuda_draft(draft, b) + 1;
        let (conv, s) = shape.delta_net_state();
        let ring = shape.cuda_ring(context, b);
        let window_bytes = shape.cuda_window_bytes(context, b);
        let kv = kv_bytes(context, shape.kv_dim());
        let mut layers = Vec::with_capacity(c.n_layer);
        for (i, &r) in c.recurrent.iter().enumerate() {
            layers.push(if r {
                LayerState::DeltaNet {
                    conv: Buf::zeroed(slots * 4 * conv)?,
                    s: Buf::zeroed(slots * 4 * s)?,
                }
            } else if c.is_swa(i) {
                LayerState::Attention {
                    k: Buf::zeroed(window_bytes)?,
                    v: Buf::zeroed(window_bytes)?,
                    ring,
                }
            } else {
                LayerState::Attention {
                    k: Buf::zeroed(kv)?,
                    v: Buf::zeroed(kv)?,
                    ring: None,
                }
            });
        }
        let s = Scratch::new(c, b, slots)?;
        debug_assert_eq!(
            memory.allocated(),
            layers
                .iter()
                .map(|l| match l {
                    LayerState::Attention { k, v, .. } => k.bytes + v.bytes,
                    LayerState::DeltaNet { conv, s } => conv.bytes + s.bytes,
                })
                .sum::<usize>()
                + shape.cuda_scratch(b, slots)
        );
        Ok(CudaSession {
            e,
            layers,
            s,
            slots,
            slot: 0,
            verify_from: None,
            context,
            n_past: 0,
            memory,
            stream: Stream::new()?,
            params: Buf::zeroed(size_of::<PassParams>())?,
            pinned_params: HostBuf::new(size_of::<PassParams>())?,
            pinned_x: HostBuf::new(4 * GRAPH_TOKENS.min(b) * c.n_embd)?,
            pinned_logits: HostBuf::new(4 * slots * c.n_vocab)?,
            graphs: Vec::new(),
            use_graphs: std::env::var_os("BIV_CUDA_GRAPHS").is_none_or(|v| v != "0"),
        })
    }

    /// `tokens` in passes of the batch size, the last one `pass`
    fn run(&mut self, tokens: &[TokenId], pass: Pass) -> Vec<Vec<f32>> {
        let mut out = Vec::new();
        let chunks: Vec<&[TokenId]> = tokens.chunks(self.s.batch).collect();
        for (i, &chunk) in chunks.iter().enumerate() {
            let pass = if i == chunks.len() - 1 {
                pass
            } else {
                Pass::Prefill
            };
            out = self.pass(chunk, pass);
        }
        out
    }

    /// One pass of up to the batch size tokens
    fn pass(&mut self, tokens: &[TokenId], pass: Pass) -> Vec<Vec<f32>> {
        let e = self.e;
        let c = e.cfg();
        let emb = c.n_embd;
        let n = tokens.len();
        assert!(n > 0, "a pass of no tokens");
        assert!(
            self.n_past + n <= self.context,
            "context overflow ({} + {n} > {})",
            self.n_past,
            self.context
        );
        assert!(
            pass != Pass::Verify || n <= self.slots,
            "verify of {n} tokens, {} at most",
            self.slots
        );
        let s = &self.s;
        assert!(n <= s.batch, "a pass of {n} tokens, {} at most", s.batch);
        e.d.bind();

        let rows = match pass {
            Pass::Prefill => 0,
            Pass::Forward => 1,
            Pass::Verify => n,
        };
        assert!(rows <= s.logit_rows);
        let save = pass == Pass::Verify;
        let params = PassParams {
            pos0: self.n_past as i32,
            slot: self.slot as i32,
            pad: [0; 2],
        };
        let params_bytes = unsafe {
            core::slice::from_raw_parts(
                (&params as *const PassParams).cast::<u8>(),
                size_of::<PassParams>(),
            )
        };
        // the embedding lookup happens on the CPU
        let embed = |x: &mut [f32]| {
            for (t, &tok) in tokens.iter().enumerate() {
                e.model
                    .tok_embd
                    .dequantize_row(tok as usize, &mut x[t * emb..(t + 1) * emb]);
            }
        };
        // a stage's copies and reads are this pass's until its end
        let _stage = e
            .stage
            .as_ref()
            .map(|st| st.lock.lock().unwrap_or_else(|p| p.into_inner()));
        // a graph copies to the stage whatever the host thinks it holds, so its
        // capture and replay forget
        let forget = || {
            if let Some(st) = &e.stage {
                st.holds.store(0, Ordering::Relaxed);
            }
        };
        let logits = if self.use_graphs && n <= GRAPH_TOKENS {
            forget();
            let key = (n, rows, save);
            let x = &mut driver::f32_slice_mut(self.pinned_x.bytes_mut())[..n * emb];
            embed(x);
            self.pinned_params.bytes_mut()[..params_bytes.len()].copy_from_slice(params_bytes);
            let i = match self.graphs.iter().position(|(k, _)| *k == key) {
                Some(i) => i,
                None => {
                    let (s, slots) = (&self.s, self.slots);
                    let (px, pp, pl) = (&self.pinned_x, &self.pinned_params, &self.pinned_logits);
                    let params = self.params.ptr;
                    let g = self
                        .stream
                        .capture(|| {
                            px.upload(0, 4 * n * emb, s.x.ptr);
                            pp.upload(0, size_of::<PassParams>(), params);
                            e.encode_pass(s, &self.layers, n, rows, params, slots, save);
                            pl.download(0, 4 * rows * c.n_vocab, s.logits.ptr);
                        })
                        .unwrap_or_else(|err| panic!("{err}"));
                    self.graphs.push((key, g));
                    self.graphs.len() - 1
                }
            };
            forget();
            self.stream.launch_and_wait(&self.graphs[i].1);
            let all = &driver::f32_slice(self.pinned_logits.bytes())[..rows * c.n_vocab];
            all.chunks_exact(c.n_vocab).map(<[f32]>::to_vec).collect()
        } else {
            let mut x = vec![0f32; n * emb];
            embed(&mut x);
            s.x.write(0, f32_bytes(&x));
            self.params.write(0, params_bytes);
            let params = self.params.ptr;
            self.stream
                .run(|| e.encode_pass(s, &self.layers, n, rows, params, self.slots, save));
            e.d.sync();
            if rows > 0 {
                let all = s.logits.read_f32(0, rows * c.n_vocab);
                all.chunks_exact(c.n_vocab).map(<[f32]>::to_vec).collect()
            } else {
                Vec::new()
            }
        };

        if pass == Pass::Verify {
            self.verify_from = Some(self.n_past);
        }
        self.n_past += n;
        logits
    }

    fn snapshot(&self) -> Result<Snapshot, Error> {
        assert!(
            self.verify_from.is_none(),
            "snapshot between verify and keep"
        );
        let e = self.e;
        e.d.bind();
        let c = e.cfg();
        let kv_row = 2 * c.kv_dim();
        let (conv_slot, s_slot) = e.slot_bytes();
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
        let mut layers = Vec::with_capacity(self.layers.len());
        for (i, ls) in self.layers.iter().enumerate() {
            layers.push(match ls {
                // the positions a next token may see, in order
                LayerState::Attention { k, v, ring } => {
                    let first = self.n_past - window_rows(c, i, self.n_past);
                    let mut rows = |buf: &Buf| -> Result<Vec<u8>, Error> {
                        let mut out = Vec::new();
                        // runs of rows that don't wrap around the ring
                        let mut p = first;
                        while p < self.n_past {
                            let (row, end) = match ring {
                                Some(r) => (p % r, self.n_past.min(p - p % r + r)),
                                None => (p, self.n_past),
                            };
                            out.extend_from_slice(&copy(buf, row * kv_row, (end - p) * kv_row)?);
                            p = end;
                        }
                        Ok(out)
                    };
                    LayerSnapshot::Attention {
                        k: rows(k)?,
                        v: rows(v)?,
                    }
                }
                LayerState::DeltaNet { conv, s } => LayerSnapshot::DeltaNet {
                    conv: copy(conv, self.slot * conv_slot, conv_slot)?,
                    s: copy(s, self.slot * s_slot, s_slot)?,
                },
            });
        }
        Ok(Snapshot::new(self.n_past, bytes, CudaSnapshot { layers }))
    }

    fn restore(&mut self, snapshot: &Snapshot) -> Result<(), Error> {
        let snap: &CudaSnapshot = snapshot.data()?;
        let n_past = snapshot.n_past();
        if n_past > self.context {
            return Err(Error::Snapshot("more tokens than the context holds"));
        }
        let e = self.e;
        e.d.bind();
        let c = e.cfg();
        let kv_row = 2 * c.kv_dim();
        let (conv_slot, s_slot) = e.slot_bytes();
        let fits = self.layers.len() == snap.layers.len()
            && self
                .layers
                .iter()
                .zip(&snap.layers)
                .enumerate()
                .all(|(i, (ls, s))| match (ls, s) {
                    (LayerState::Attention { .. }, LayerSnapshot::Attention { k, .. }) => {
                        k.len() == window_rows(c, i, n_past) * kv_row
                    }
                    (LayerState::DeltaNet { .. }, LayerSnapshot::DeltaNet { conv, s }) => {
                        conv.len() == conv_slot && s.len() == s_slot
                    }
                    _ => false,
                });
        if !fits {
            return Err(Error::Snapshot("taken from another model"));
        }
        // the states go to slot 0
        for (ls, s) in self.layers.iter().zip(&snap.layers) {
            match (ls, s) {
                (
                    LayerState::Attention { k, v, ring },
                    LayerSnapshot::Attention { k: sk, v: sv },
                ) => {
                    let first = n_past - sk.len() / kv_row;
                    // runs of rows that don't wrap around the ring
                    let mut p = first;
                    while p < n_past {
                        let (row, end) = match ring {
                            Some(r) => (p % r, n_past.min(p - p % r + r)),
                            None => (p, n_past),
                        };
                        let src = (p - first) * kv_row..(end - first) * kv_row;
                        k.write(row * kv_row, &sk[src.clone()]);
                        v.write(row * kv_row, &sv[src]);
                        p = end;
                    }
                }
                (LayerState::DeltaNet { conv, s }, LayerSnapshot::DeltaNet { conv: sc, s: ss }) => {
                    conv.write(0, sc);
                    s.write(0, ss);
                }
                _ => unreachable!("checked above"),
            }
        }
        self.slot = 0;
        self.n_past = n_past;
        self.verify_from = None;
        Ok(())
    }
}

impl CudaEngine<'_> {
    /// Bytes of one slot of a delta net's conv and recurrent states
    fn slot_bytes(&self) -> (usize, usize) {
        let (conv, s) = self.model.shape().delta_net_state();
        (4 * conv, 4 * s)
    }

    /// A pass of the `n` embeddings in `s.x` through the layers, and the
    /// logits of its last `rows` into `s.logits`
    #[allow(clippy::too_many_arguments)]
    fn encode_pass(
        &self,
        s: &Scratch,
        layers: &[LayerState],
        n: usize,
        rows: usize,
        params: DevPtr,
        slots: usize,
        save: bool,
    ) {
        let c = self.cfg();
        let emb = c.n_embd;
        self.stage_begin();
        let first = &self.layers[0].attn_norm;
        self.rms_norm(s.x.ptr, None, first, s.h.ptr, None, n, emb, emb);
        for (i, (layer, ls)) in self.layers.iter().zip(layers).enumerate() {
            let next = self.layers.get(i + 1).map(|l| &l.attn_norm);
            self.encode_layer(s, layer, next, ls, n, params, slots, save);
        }
        if rows > 0 {
            let from = n - rows;
            self.rms_norm(
                s.x.f32_at(from * emb),
                None,
                &self.output_norm,
                s.feat.ptr,
                None,
                rows,
                emb,
                emb,
            );
            let feat = Act::new(s.feat.ptr, rows, emb, &s.q8[1]);
            self.matmul(&self.output, &feat, s.logits.ptr);
        }
        self.stage_end();
    }

    /// One decoder layer over the `n` rows of the residual stream in `s.x`,
    /// normalized in `s.h`, at positions `pos0..` of `ls`; then `s.h` the
    /// stream normalized by `next`, the next layer's norm
    #[allow(clippy::too_many_arguments)]
    fn encode_layer(
        &self,
        s: &Scratch,
        layer: &GpuLayer,
        next: Option<&Buf>,
        ls: &LayerState,
        n: usize,
        params: DevPtr,
        slots: usize,
        save: bool,
    ) {
        let c = self.cfg();
        let (emb, hd) = (c.n_embd, c.head_dim);
        let (x, h, y) = (s.x.ptr, s.h.ptr, s.y.ptr);

        let hx = Act::new(h, n, emb, &s.q8[0]);
        match &layer.mixer {
            GpuMixer::DeltaNet {
                wqkv,
                wgate,
                wbeta,
                walpha,
                wout,
                conv1d,
                dt_bias,
                a,
                norm,
            } => {
                let LayerState::DeltaNet { conv, s: st } = ls else {
                    unreachable!("layer / state kind mismatch")
                };
                let (sd, hk, hv) = (c.ssm_state, c.ssm_k_heads, c.ssm_v_heads);
                let conv_dim = c.conv_dim();
                self.matmuls(
                    &[
                        (wqkv, s.a1.ptr),
                        (wgate, s.a2.ptr),
                        (wbeta, s.beta.ptr),
                        (walpha, s.alpha.ptr),
                    ],
                    &hx,
                );

                let (slots, save) = (slots as i32, save as i32);
                let args = ConvArgs {
                    qkv: s.a1.ptr,
                    w: conv1d.ptr,
                    state: conv.ptr,
                    out: s.conv_out.ptr,
                    params,
                    conv_dim: conv_dim as i32,
                    n: n as i32,
                    slots,
                    save,
                };
                self.d.launch(
                    self.k.ssm_conv,
                    [conv_dim.div_ceil(32) as u32, 1, 1],
                    [32, 8, 1],
                    0,
                    &args,
                );
                let args = GdnArgs {
                    conv_out: s.conv_out.ptr,
                    alpha: s.alpha.ptr,
                    beta: s.beta.ptr,
                    dt_bias: dt_bias.ptr,
                    a_neg: a.ptr,
                    state: st.ptr,
                    out: s.o.ptr,
                    params,
                    n: n as i32,
                    hk: hk as i32,
                    hv: hv as i32,
                    conv_dim: conv_dim as i32,
                    eps: c.eps,
                    slots,
                    save,
                    pad: 0,
                };
                self.d
                    .launch(self.k.gdn_prep, [hk as u32, n as u32, 1], [32, 1, 1], 0, &args);
                self.d.launch(
                    self.k.gdn,
                    [(sd / GDN_ROWS) as u32, hv as u32, 1],
                    [32, GDN_ROWS as u32, 1],
                    0,
                    &args,
                );
                // norm(o) * silu(z), per head, in place
                self.rms_norm(s.o.ptr, None, norm, s.o.ptr, Some(s.a2.ptr), n * hv, sd, sd);
                let o = Act::new(s.o.ptr, n, wout.cols, &s.q8[1]);
                self.matmul(wout, &o, y);
            }
            GpuMixer::Attention {
                wq,
                wk,
                wv,
                wo,
                q_norm,
                k_norm,
                sinks,
                rope,
                window,
            } => {
                let LayerState::Attention { k: kc, v: vc, ring } = ls else {
                    unreachable!("layer / state kind mismatch")
                };
                self.matmuls(&[(wq, s.a1.ptr), (wk, s.k.ptr), (wv, s.v.ptr)], &hx);
                let args = AttnArgs {
                    qg: s.a1.ptr,
                    k: s.k.ptr,
                    v: s.v.ptr,
                    q_norm: q_norm.ptr,
                    k_norm: k_norm.ptr,
                    inv_freq: self.inv_freq.ptr,
                    q: s.q.ptr,
                    k_cache: kc.ptr,
                    v_cache: vc.ptr,
                    o: s.o.ptr,
                    sinks: sinks.as_ref().map_or(q_norm.ptr, |b| b.ptr),
                    params,
                    n_head: c.n_head as i32,
                    n_kv: c.n_head_kv as i32,
                    hd: hd as i32,
                    n_rot: if *rope { c.n_rot as i32 } else { 0 },
                    n: n as i32,
                    gated: c.attn_gate as i32,
                    window: window.map_or(0, |w| w as i32),
                    mask: ring.map_or(-1, |r| r as i32 - 1),
                    has_sinks: sinks.is_some() as i32,
                    eps: c.eps,
                    scale: 1.0 / (hd as f32).sqrt(),
                };
                self.d.launch(
                    self.k.attn_prep,
                    [(c.n_head + c.n_head_kv) as u32, n as u32, 1],
                    [32, 1, 1],
                    0,
                    &args,
                );
                match FA_HEAD_DIMS.iter().position(|&d| d == hd) {
                    Some(i) if n > NT => self.d.launch(
                        self.k.fattn[i],
                        [c.n_head as u32, n.div_ceil(FA_BQ) as u32, 1],
                        [(32 * FA_WARPS) as u32, 1, 1],
                        fattn_shared(hd) as u32,
                        &args,
                    ),
                    _ => self.d.launch(
                        self.k.attn,
                        [c.n_head as u32, n as u32, 1],
                        [(32 * ATTN_WARPS) as u32, 1, 1],
                        (ATTN_WARPS * hd * 4) as u32,
                        &args,
                    ),
                }
                let o = Act::new(s.o.ptr, n, wo.cols, &s.q8[1]);
                self.matmul(wo, &o, y);
            }
        }
        self.rms_norm(x, Some(y), &layer.post_norm, h, None, n, emb, emb);
        let hx = Act::new(h, n, emb, &s.q8[0]);
        match &layer.ffn {
            GpuFfn::Dense { gate, up, down } => {
                self.dense_ffn(s, gate, up, down, &hx, y);
                match next {
                    Some(w) => self.rms_norm(x, Some(y), w, h, None, n, emb, emb),
                    None => self.elementwise(self.k.add, x, y, n * emb),
                }
            }
            GpuFfn::Moe(m) => {
                self.moe(s, m, &hx);
                if let Some(w) = next {
                    self.rms_norm(x, None, w, h, None, n, emb, emb);
                }
            }
        }
    }

    /// `down(silu(gate h) * up h)` of the rows of `h` into `out`, through
    /// `a1` and `a2`
    fn dense_ffn(
        &self,
        s: &Scratch,
        gate: &GpuWeight,
        up: &GpuWeight,
        down: &GpuWeight,
        h: &Act<'_>,
        out: DevPtr,
    ) {
        let n = h.n;
        self.glu(s, gate, up, h);
        let a = Act::new(s.a1.ptr, n, gate.rows, &s.q8[1]);
        self.matmul(down, &a, out);
    }

    /// The mixture of experts of the normalized tokens `h`, added to the
    /// residual stream `x`
    fn moe(&self, s: &Scratch, m: &GpuMoe, h: &Act<'_>) {
        let n = h.n;
        let c = self.cfg();
        let mc = c.moe.as_ref().expect("a moe config");
        let ms = s.moe.as_ref().expect("the moe buffers");
        let (emb, k, ne) = (c.n_embd, mc.n_expert_used, mc.n_expert);
        let pairs = n * k;
        let mma = n > NT;
        let tile = if mma { TN } else { NT };

        self.matmul(&m.router, h, ms.logits.ptr);
        let args = RouteArgs {
            logits: ms.logits.ptr,
            ids: ms.ids.ptr,
            weights: ms.weights.ptr,
            bias: m.bias.as_ref().map_or(ms.logits.ptr, |b| b.ptr),
            n_expert: ne as i32,
            k: k as i32,
            sigmoid: (mc.gating == Gating::Sigmoid) as i32,
            has_bias: m.bias.is_some() as i32,
            norm: mc.norm_weights as i32,
            scale: mc.weights_scale,
        };
        self.d.launch(
            self.k.moe_route,
            [n as u32, 1, 1],
            [256, 1, 1],
            (2 * ne * 4) as u32,
            &args,
        );
        let args = SortArgs {
            ids: ms.ids.ptr,
            order: ms.order.ptr,
            tiles: ms.tiles.ptr,
            n_tiles: ms.n_tiles.ptr,
            n_pairs: pairs as i32,
            n_expert: ne as i32,
            tile: tile as i32,
            pad: 0,
        };
        self.d.launch(
            self.k.moe_sort,
            [1, 1, 1],
            [1024, 1, 1],
            (3 * ne * 4) as u32,
            &args,
        );

        let tiles = max_tiles(ne, pairs, tile);
        let ff = mc.n_ff_exp;
        self.matmul_id(&m.gate, ms, h, k, tiles, mma, ms.a1.ptr);
        self.matmul_id(&m.up, ms, h, k, tiles, mma, ms.a2.ptr);
        self.elementwise(self.k.swiglu, ms.a1.ptr, ms.a2.ptr, pairs * ff);
        let a = Act::new(ms.a1.ptr, pairs, ff, &s.q8[1]);
        self.matmul_id(&m.down, ms, &a, 1, tiles, mma, ms.out.ptr);

        let mode = match &m.shared {
            Some(sh) => {
                self.dense_ffn(s, &sh.gate, &sh.up, &sh.down, h, ms.shared.ptr);
                match &sh.gate_inp {
                    Some(g) => {
                        self.matmul(g, h, ms.gate.ptr);
                        1
                    }
                    None => 2,
                }
            }
            None => 0,
        };
        let args = CombineArgs {
            x: s.x.ptr,
            out: ms.out.ptr,
            weights: ms.weights.ptr,
            shared: ms.shared.ptr,
            gate: ms.gate.ptr,
            n_embd: emb as i32,
            k: k as i32,
            mode,
            pad: 0,
        };
        self.d.launch(
            self.k.moe_combine,
            [emb.div_ceil(256) as u32, n as u32, 1],
            [256, 1, 1],
            0,
            &args,
        );
    }
}

impl Session for CudaSession<'_> {
    fn forward(&mut self, tokens: &[TokenId]) -> Vec<f32> {
        self.run(tokens, Pass::Forward).pop().unwrap()
    }

    fn prefill(&mut self, tokens: &[TokenId]) {
        self.run(tokens, Pass::Prefill);
    }

    fn draft(
        &mut self,
        _next: TokenId,
        _n: usize,
        _pick: &mut dyn FnMut(&DraftLogits<'_>) -> TokenId,
    ) -> Vec<TokenId> {
        Vec::new()
    }

    fn verify(&mut self, tokens: &[TokenId]) -> Vec<Vec<f32>> {
        self.pass(tokens, Pass::Verify)
    }

    fn keep(&mut self, n: usize) {
        let pos0 = self.verify_from.take().expect("keep without verify");
        let verified = self.n_past - pos0;
        assert!(
            n >= 1 && n <= verified,
            "keep {n} of {verified} verified tokens"
        );
        // the slot after the n-th token holds the delta net states to go on from
        self.slot = (self.slot + n) % self.slots;
        // attention caches are overwritten from the new position
        self.n_past = pos0 + n;
    }

    fn n_past(&self) -> usize {
        self.n_past
    }

    fn memory(&self) -> SessionMemory {
        self.memory
    }

    fn snapshot(&self) -> Result<Snapshot, Error> {
        CudaSession::snapshot(self)
    }

    fn restore(&mut self, snapshot: &Snapshot) -> Result<(), Error> {
        CudaSession::restore(self, snapshot)
    }
}
