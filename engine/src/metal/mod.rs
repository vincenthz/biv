//! Metal backend: the whole forward pass of a batch is encoded into one
//! command buffer. Weights are not copied: the GPU reads the mmap'd GGUF
//! through no-copy buffers (unified memory). Small vectors (norms, conv
//! weights) are uploaded; activations and states live in shared buffers.

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLDispatchType, MTLLibrary, MTLResourceOptions,
    MTLSize,
};

use crate::Error;
use crate::engine::{Batch, DraftLogits, Engine, Pass, SeqId, Session, Snapshot, Work, insert};
use crate::memory::{DeviceMemory, ModelMemory, SessionMemory, Shape};
use crate::model::{
    Config, DRAFT_CAP, DraftVocab, Experts, Ffn, Gating, Layer, Mixer, Model, copy_head_rows,
};
use crate::ops::Weight;
use crate::prelude::*;
use crate::quant::GgmlType;
use crate::tokenizer::TokenId;

const SOURCE: &str = include_str!("kernels.metal");

/// Where the dispatches being encoded on this thread are, between
/// [`MetalEngine::concurrent`] groups. The encoders are concurrent: a
/// dispatch waits for the ones before it only behind a barrier, which every
/// dispatch has but those of a group after its first.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Group {
    /// not in a group
    None,
    /// before the first dispatch of a group
    First,
    /// after it
    In,
}

std::thread_local! {
    static GROUP: core::cell::Cell<Group> = const { core::cell::Cell::new(Group::None) };
}

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type Encoder = ProtocolObject<dyn MTLComputeCommandEncoder>;

/// Virtual memory page, the alignment of no-copy buffers
const PAGE: usize = if cfg!(target_arch = "aarch64") {
    16384
} else {
    4096
};

/// Matmul inputs of up to this many rows use the matvec kernels, one
/// threadgroup per input row: the weights of the second and third mostly come
/// from the cache (a verify pass of 2 tokens costs 1.2x a decode step, 3 1.6x)
const MV_MAX_ROWS: usize = 3;
/// Matmul inputs of up to this many rows use the `mmv` kernels, which read the
/// weights once for 8 input rows (on an M2 Max, a pass of 4 to 8 tokens costs
/// 1.75-1.9x a decode step, 12 to 16 3.2-3.4x); more use the `mm`
/// kernels, whose tiles are 32 rows (~4.7x from 5 to 32 tokens)
const MMV_MAX_ROWS: usize = 16;
/// Input rows and weight rows (2 simdgroups of 8) per `mmv` threadgroup
const MMV_ROWS: usize = 8;
const MMV_WEIGHT_ROWS: usize = 16;
/// The pairs of an expert are multiplied by `mm_id` in tiles of 32, the last
/// one too when it has this many pairs or more; the others by `mmv_id` in
/// tiles of 8. On an M2 Max, prefill of Qwen3.6 35B-A3B in passes of 256
/// tokens: 902 tok/s from 33 (no partial tile of 32), 937 from 12 to 16,
/// 931 from 8; in passes of 512, 1083 tok/s from 1, 1141 from 4, 1170 from 8,
/// 1174 from 16.
const MM_ID_MIN: usize = 16;
/// Passes of up to this many tokens multiply each of their pairs by its
/// expert with the `mv_id` kernels, a threadgroup row per pair, rather than
/// sorting them into tiles: in a decode step an expert has one pair, which
/// the 8 columns of an `mmv_id` tile would compute 8 times. Longer passes have
/// more tokens on the same experts, whose weights a tile reads once (Qwen3.6
/// 35B-A3B on an M2 Max, decode with 1 draft / 4 drafts: 83 / 87 tok/s up to 2,
/// 80 / 86 up to 3, 79 / 80 up to 8).
const MV_ID_MAX_ROWS: usize = 2;

/// Attention positions per split-K range, at least
const ATTN_CHUNK: usize = 32;
/// Split-K ranges per (token, head), at most. About 64 ranges keep the GPU
/// busy at any context length; more only add merge work.
const ATTN_MAX_SPLITS: usize = 64;
/// (query heads per kv head, simdgroups per threadgroup) of the `attn_split`
/// kernels, see the instantiations in the shaders
const ATTN_GROUPS: &[(usize, usize)] = &[(1, 2), (2, 2), (4, 2), (6, 3), (8, 4)];
const ATTN_HEAD_DIMS: &[usize] = &[128, 256];
/// Batches of at least this many tokens use the `attn_flash` kernel, which is
/// 2.2-2.4x faster from 64 tokens; with fewer query rows (decode, verify) it
/// can't fill the GPU and the split-K kernel is faster
const ATTN_FLASH_MIN: usize = 32;
/// Query rows, positions per step and simdgroups of the `attn_flash` kernels,
/// see the instantiations in the shaders. The K / V caches hold a multiple of
/// `ATTN_FLASH_C` positions.
const ATTN_FLASH_R: usize = 16;
const ATTN_FLASH_C: usize = 64;
const ATTN_FLASH_NSG: usize = 4;
/// Verify batches (2 tokens to `ATTN_FLASH_MIN`) of a sequence of at least
/// this many tokens use the split `attn_flash` kernel, which reads each K / V
/// tile once for all the tokens and heads of a block rather than once per
/// token: 13% less for a 5-token verify at 32k positions on M2 Max. Shorter,
/// its ranges are too short for its overheads.
const ATTN_FLASH_SPLIT_CTX: usize = 2048;

#[repr(C)]
struct MatArgs {
    k: i32,
    rows: i32,
    n: i32,
    pad: i32,
    w_row_bytes: u64,
    x_row_bytes: u64,
}

#[repr(C)]
struct MatIdArgs {
    k: i32,
    rows: i32,
    x_div: i32,
    pad: i32,
    w_row_bytes: u64,
    x_row_bytes: u64,
    stride: u64,
}

#[repr(C)]
struct RouteArgs {
    n_expert: i32,
    k: i32,
    sigmoid: i32,
    bias: i32,
    norm: i32,
    scale: f32,
}

#[repr(C)]
struct SortArgs {
    n_pairs: i32,
    n_expert: i32,
    mm_min: i32,
    mv_base: i32,
}

#[repr(C)]
struct CombineArgs {
    n_embd: i32,
    k: i32,
    shared: i32,
}

#[repr(C)]
struct EltArgs {
    n: i32,
}

#[repr(C)]
struct NormArgs {
    n: i32,
    eps: f32,
    gated: i32,
    ostride: i32,
}

#[repr(C)]
struct AttnArgs {
    n_head: i32,
    n_kv: i32,
    hd: i32,
    n_rot: i32,
    pos0: i32,
    eps: f32,
    scale: f32,
    splits: i32,
    chunk: i32,
    n: i32,
    gated: i32,
    window: i32,
    lo: i32,
    mask: i32,
    sinks: i32,
}

#[repr(C)]
struct ConvArgs {
    conv_dim: i32,
    n: i32,
    slot: i32,
    slots: i32,
    save: i32,
}

#[repr(C)]
struct GdnArgs {
    n: i32,
    hk: i32,
    hv: i32,
    conv_dim: i32,
    eps: f32,
    slot: i32,
    slots: i32,
    save: i32,
}

/// A buffer binding: buffer and byte offset
#[derive(Clone, Copy)]
struct B<'a>(&'a Buffer, usize);

impl<'a> B<'a> {
    fn at(buf: &'a Buffer) -> Self {
        B(buf, 0)
    }

    /// `n` f32 further
    fn f32_off(self, n: usize) -> Self {
        B(self.0, self.1 + 4 * n)
    }
}

/// A weight matrix seen by the GPU
#[derive(Clone)]
struct GpuWeight {
    buf: Buffer,
    offset: usize,
    ty: GgmlType,
    cols: usize,
    rows: usize,
    row_bytes: usize,
}

/// The experts' matrices of a layer seen by the GPU: expert `e`'s is
/// `stride` bytes after expert 0's, at `offset` in `buf`
struct GpuExperts {
    buf: Buffer,
    offset: usize,
    ty: GgmlType,
    cols: usize,
    rows: usize,
    row_bytes: usize,
    stride: usize,
}

/// The compute kernels, from `kernels.metal`, each with its MSL name. The
/// attention kernels are compiled for several head sizes and group sizes:
/// their variant here is the one of the model's geometry.
macro_rules! kernels {
    (|$hd:ident, $group:ident| $($(#[$m:meta])* $v:ident => $name:expr,)*) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        enum Kernel {
            $($(#[$m])* $v,)*
        }

        impl Kernel {
            /// in declaration order: `ALL[k as usize] == k`
            const ALL: &[Kernel] = &[$(Kernel::$v,)*];

            /// The MSL function, for attention heads of `hd` floats and query
            /// groups of `group` heads
            fn name(self, $hd: usize, $group: usize) -> String {
                let _ = ($hd, $group);
                match self {
                    $(Kernel::$v => String::from($name),)*
                }
            }
        }
    };
}

kernels! {
    |hd, group|
    MvQ4_0 => "mv_q4_0",
    MvQ4_1 => "mv_q4_1",
    MvQ8_0 => "mv_q8_0",
    MvQ5K => "mv_q5_K",
    MvQ6K => "mv_q6_K",
    MvF32 => "mv_f32",
    MvF16 => "mv_f16",
    MvBf16 => "mv_bf16",
    MmF32 => "mm_f32",
    MmF16 => "mm_f16",
    MmBf16 => "mm_bf16",
    MmQ4_0 => "mm_q4_0",
    MmQ4_1 => "mm_q4_1",
    MmQ8_0 => "mm_q8_0",
    MmQ4K => "mm_q4_K",
    MmQ5K => "mm_q5_K",
    MmQ6K => "mm_q6_K",
    MmvF32 => "mmv_f32",
    MmvF16 => "mmv_f16",
    MmvBf16 => "mmv_bf16",
    MmvQ4_0 => "mmv_q4_0",
    MmvQ4_1 => "mmv_q4_1",
    MmvQ8_0 => "mmv_q8_0",
    MmvQ4K => "mmv_q4_K",
    MmvQ5K => "mmv_q5_K",
    MmvQ6K => "mmv_q6_K",
    AddInplace => "add_inplace",
    Swiglu => "swiglu",
    RmsNorm => "rms_norm",
    AttnPrep => "attn_prep",
    AttnReduce => "attn_reduce",
    /// split-K attention, for short batches
    AttnSplit => format!("attn_split_{hd}_{group}"),
    AttnFlash => format!("attn_flash_{hd}"),
    /// flash attention split along the positions, for long contexts
    AttnFlashSplit => format!("attn_flash_split_{hd}"),
    SsmConv => "ssm_conv",
    Gdn128 => "gdn_128",
    MvIdQ4_0 => "mv_id_q4_0",
    MvIdQ4_1 => "mv_id_q4_1",
    MvIdQ8_0 => "mv_id_q8_0",
    MvIdQ5K => "mv_id_q5_K",
    MvIdQ6K => "mv_id_q6_K",
    MvIdF32 => "mv_id_f32",
    MvIdF16 => "mv_id_f16",
    MvIdBf16 => "mv_id_bf16",
    MmvIdF32 => "mmv_id_f32",
    MmvIdF16 => "mmv_id_f16",
    MmvIdBf16 => "mmv_id_bf16",
    MmvIdQ4_0 => "mmv_id_q4_0",
    MmvIdQ4_1 => "mmv_id_q4_1",
    MmvIdQ8_0 => "mmv_id_q8_0",
    MmvIdQ4K => "mmv_id_q4_K",
    MmvIdQ5K => "mmv_id_q5_K",
    MmvIdQ6K => "mmv_id_q6_K",
    MmIdF32 => "mm_id_f32",
    MmIdF16 => "mm_id_f16",
    MmIdBf16 => "mm_id_bf16",
    MmIdQ4_0 => "mm_id_q4_0",
    MmIdQ4_1 => "mm_id_q4_1",
    MmIdQ8_0 => "mm_id_q8_0",
    MmIdQ4K => "mm_id_q4_K",
    MmIdQ5K => "mm_id_q5_K",
    MmIdQ6K => "mm_id_q6_K",
    MoeRoute => "moe_route",
    MoeRank => "moe_rank",
    MoeSort => "moe_sort",
    MoeCombine => "moe_combine",
}

/// The pipeline of every [`Kernel`], indexed by it
struct Kernels {
    pipelines: Vec<Pipeline>,
}

impl Kernels {
    fn get(&self, k: Kernel) -> &Pipeline {
        &self.pipelines[k as usize]
    }
}

fn mv_kernel(ty: GgmlType) -> Option<(Kernel, usize, usize)> {
    // (kernel, rows per threadgroup, threads per threadgroup)
    Some(match ty {
        GgmlType::Q4_0 => (Kernel::MvQ4_0, 8, 64),
        GgmlType::Q4_1 => (Kernel::MvQ4_1, 8, 64),
        GgmlType::Q8_0 => (Kernel::MvQ8_0, 2, 128),
        GgmlType::Q5K => (Kernel::MvQ5K, 2, 64),
        GgmlType::Q6K => (Kernel::MvQ6K, 4, 64),
        GgmlType::F32 => (Kernel::MvF32, 4, 128),
        GgmlType::F16 => (Kernel::MvF16, 4, 128),
        GgmlType::BF16 => (Kernel::MvBf16, 4, 128),
        _ => return None,
    })
}

/// The `mv_id` kernels, of the same geometry as [`mv_kernel`]'s: a pair of
/// the experts' products per threadgroup row
fn mv_id_kernel(ty: GgmlType) -> Option<(Kernel, usize, usize)> {
    let (_, rows, threads) = mv_kernel(ty)?;
    let k = match ty {
        GgmlType::Q4_0 => Kernel::MvIdQ4_0,
        GgmlType::Q4_1 => Kernel::MvIdQ4_1,
        GgmlType::Q8_0 => Kernel::MvIdQ8_0,
        GgmlType::Q5K => Kernel::MvIdQ5K,
        GgmlType::Q6K => Kernel::MvIdQ6K,
        GgmlType::F32 => Kernel::MvIdF32,
        GgmlType::F16 => Kernel::MvIdF16,
        GgmlType::BF16 => Kernel::MvIdBf16,
        _ => return None,
    };
    Some((k, rows, threads))
}

/// The `mmv` kernels, for 4 to 16 input rows
fn mmv_kernel(ty: GgmlType) -> Option<Kernel> {
    Some(match ty {
        GgmlType::F32 => Kernel::MmvF32,
        GgmlType::F16 => Kernel::MmvF16,
        GgmlType::BF16 => Kernel::MmvBf16,
        GgmlType::Q4_0 => Kernel::MmvQ4_0,
        GgmlType::Q4_1 => Kernel::MmvQ4_1,
        GgmlType::Q8_0 => Kernel::MmvQ8_0,
        GgmlType::Q4K => Kernel::MmvQ4K,
        GgmlType::Q5K => Kernel::MmvQ5K,
        GgmlType::Q6K => Kernel::MmvQ6K,
        _ => return None,
    })
}

/// The `mmv_id` kernels, for the tiles of the experts' pairs
fn mmv_id_kernel(ty: GgmlType) -> Option<Kernel> {
    Some(match ty {
        GgmlType::F32 => Kernel::MmvIdF32,
        GgmlType::F16 => Kernel::MmvIdF16,
        GgmlType::BF16 => Kernel::MmvIdBf16,
        GgmlType::Q4_0 => Kernel::MmvIdQ4_0,
        GgmlType::Q4_1 => Kernel::MmvIdQ4_1,
        GgmlType::Q8_0 => Kernel::MmvIdQ8_0,
        GgmlType::Q4K => Kernel::MmvIdQ4K,
        GgmlType::Q5K => Kernel::MmvIdQ5K,
        GgmlType::Q6K => Kernel::MmvIdQ6K,
        _ => return None,
    })
}

/// The `mm_id` kernels, for the tiles of 32 pairs of passes of
/// [`MM_ID_MIN`] tokens or more
fn mm_id_kernel(ty: GgmlType) -> Option<Kernel> {
    Some(match ty {
        GgmlType::F32 => Kernel::MmIdF32,
        GgmlType::F16 => Kernel::MmIdF16,
        GgmlType::BF16 => Kernel::MmIdBf16,
        GgmlType::Q4_0 => Kernel::MmIdQ4_0,
        GgmlType::Q4_1 => Kernel::MmIdQ4_1,
        GgmlType::Q8_0 => Kernel::MmIdQ8_0,
        GgmlType::Q4K => Kernel::MmIdQ4K,
        GgmlType::Q5K => Kernel::MmIdQ5K,
        GgmlType::Q6K => Kernel::MmIdQ6K,
        _ => return None,
    })
}

fn mm_kernel(ty: GgmlType) -> Option<Kernel> {
    Some(match ty {
        GgmlType::F32 => Kernel::MmF32,
        GgmlType::F16 => Kernel::MmF16,
        GgmlType::BF16 => Kernel::MmBf16,
        GgmlType::Q4_0 => Kernel::MmQ4_0,
        GgmlType::Q4_1 => Kernel::MmQ4_1,
        GgmlType::Q8_0 => Kernel::MmQ8_0,
        GgmlType::Q4K => Kernel::MmQ4K,
        GgmlType::Q5K => Kernel::MmQ5K,
        GgmlType::Q6K => Kernel::MmQ6K,
        _ => return None,
    })
}

enum GpuMixer {
    Attention {
        wq: GpuWeight,
        wk: GpuWeight,
        wv: GpuWeight,
        wo: GpuWeight,
        q_norm: Buffer,
        k_norm: Buffer,
        /// see [`crate::model::Attention`]
        rope: bool,
        window: Option<usize>,
        sinks: Option<Buffer>,
    },
    DeltaNet {
        wqkv: GpuWeight,
        wgate: GpuWeight,
        wbeta: GpuWeight,
        walpha: GpuWeight,
        wout: GpuWeight,
        conv1d: Buffer,
        dt_bias: Buffer,
        a: Buffer,
        norm: Buffer,
    },
}

enum GpuFfn {
    Dense {
        gate: GpuWeight,
        up: GpuWeight,
        down: GpuWeight,
    },
    Moe(Box<GpuMoe>),
}

/// see [`crate::model::Moe`]
struct GpuMoe {
    router: GpuWeight,
    /// added to the probabilities to choose the experts
    bias: Option<Buffer>,
    gate: GpuExperts,
    up: GpuExperts,
    down: GpuExperts,
    shared: Option<GpuShared>,
}

/// see [`crate::model::SharedExpert`]
struct GpuShared {
    gate: GpuWeight,
    up: GpuWeight,
    down: GpuWeight,
    /// the one row of its gate, when it has one
    gate_inp: Option<GpuWeight>,
}

struct GpuLayer {
    attn_norm: Buffer,
    post_norm: Buffer,
    mixer: GpuMixer,
    ffn: GpuFfn,
}

/// The MTP head, see [`crate::model::Mtp`]
struct GpuMtp {
    enorm: Buffer,
    hnorm: Buffer,
    eh_proj: GpuWeight,
    layer: GpuLayer,
    head_norm: Buffer,
}

pub struct MetalEngine<'m> {
    model: &'m Model<'m>,
    device: Device,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    k: Kernels,
    layers: Vec<GpuLayer>,
    mtp: Option<GpuMtp>,
    output_norm: Buffer,
    output: GpuWeight,
    inv_freq: Buffer,
    attn_nsg: usize,
    max_batch: usize,
}

fn upload(device: &Device, data: &[f32]) -> Result<Buffer, Error> {
    let bytes = (data.len() * 4).max(4);
    let buf = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .ok_or(Error::OutOfMemory { bytes })?;
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            buf.contents().as_ptr() as *mut f32,
            data.len(),
        );
    }
    Ok(buf)
}

fn zeroed(device: &Device, bytes: usize) -> Result<Buffer, Error> {
    let buf = device
        .newBufferWithLength_options(bytes.max(4), MTLResourceOptions::StorageModeShared)
        .ok_or(Error::OutOfMemory { bytes })?;
    unsafe { std::ptr::write_bytes(buf.contents().as_ptr() as *mut u8, 0, bytes) };
    Ok(buf)
}

/// Expose mmap'd bytes to the GPU without copying them: the buffer of the
/// pages holding them, and their offset in it
fn map(device: &Device, data: &[u8]) -> Result<(Buffer, usize), Error> {
    let addr = data.as_ptr() as usize;
    let start = addr & !(PAGE - 1);
    let end = (addr + data.len()).next_multiple_of(PAGE);
    let ptr =
        NonNull::new(start as *mut c_void).ok_or_else(|| backend("metal: null weight pointer"))?;
    // read-only weights need no hazard tracking
    let opts =
        MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeUntracked;
    let buf = unsafe {
        device.newBufferWithBytesNoCopy_length_options_deallocator(ptr, end - start, opts, None)
    }
    .ok_or_else(|| backend("metal: cannot map the weights (no-copy buffer)"))?;
    Ok((buf, addr - start))
}

/// Expose mmap'd weights to the GPU without copying them
fn wrap(device: &Device, w: &Weight<'_>) -> Result<GpuWeight, Error> {
    if mm_kernel(w.ty).is_none() {
        return Err(backend(format!("metal: no matmul kernel for {:?}", w.ty)));
    }
    if !w.cols.is_multiple_of(32) {
        return Err(backend(format!(
            "metal: matmul width {} is not a multiple of 32",
            w.cols
        )));
    }
    let (buf, offset) = map(device, w.data)?;
    Ok(GpuWeight {
        buf,
        offset,
        ty: w.ty,
        cols: w.cols,
        rows: w.rows,
        row_bytes: w.ty.row_size(w.cols),
    })
}

/// Expose the experts' matrices to the GPU without copying them
fn wrap_experts(device: &Device, x: &Experts<'_>) -> Result<GpuExperts, Error> {
    if mmv_id_kernel(x.ty).is_none() || mm_id_kernel(x.ty).is_none() {
        return Err(backend(format!(
            "metal: no expert matmul kernel for {:?}",
            x.ty
        )));
    }
    // `mmv_id` takes 4 spans of 32 per step
    if !x.cols.is_multiple_of(128) {
        return Err(backend(format!(
            "metal: expert matmul width {} is not a multiple of 128",
            x.cols
        )));
    }
    let (buf, offset) = map(device, x.data)?;
    Ok(GpuExperts {
        buf,
        offset: offset + x.offset,
        ty: x.ty,
        cols: x.cols,
        rows: x.rows,
        row_bytes: x.row_bytes(),
        stride: x.stride,
    })
}

fn backend(msg: impl Into<String>) -> Error {
    Error::Backend(msg.into())
}

fn gpu_layer(device: &Device, l: &Layer<'_>) -> Result<GpuLayer, Error> {
    let mixer = match &l.mixer {
        Mixer::Attention(a) => GpuMixer::Attention {
            wq: wrap(device, &a.wq)?,
            wk: wrap(device, &a.wk)?,
            wv: wrap(device, &a.wv)?,
            wo: wrap(device, &a.wo)?,
            q_norm: upload(device, &a.q_norm)?,
            k_norm: upload(device, &a.k_norm)?,
            rope: a.rope,
            window: a.window,
            sinks: match &a.sinks {
                Some(s) => Some(upload(device, s)?),
                None => None,
            },
        },
        Mixer::DeltaNet(d) => GpuMixer::DeltaNet {
            wqkv: wrap(device, &d.wqkv)?,
            wgate: wrap(device, &d.wgate)?,
            wbeta: wrap(device, &d.wbeta)?,
            walpha: wrap(device, &d.walpha)?,
            wout: wrap(device, &d.wout)?,
            conv1d: upload(device, &d.conv1d)?,
            dt_bias: upload(device, &d.dt_bias)?,
            a: upload(device, &d.a)?,
            norm: upload(device, &d.norm)?,
        },
    };
    Ok(GpuLayer {
        attn_norm: upload(device, &l.attn_norm)?,
        post_norm: upload(device, &l.post_norm)?,
        mixer,
        ffn: match &l.ffn {
            Ffn::Dense { gate, up, down } => GpuFfn::Dense {
                gate: wrap(device, gate)?,
                up: wrap(device, up)?,
                down: wrap(device, down)?,
            },
            Ffn::Moe(m) => GpuFfn::Moe(Box::new(GpuMoe {
                router: wrap(device, &m.router)?,
                bias: match &m.bias {
                    Some(b) => Some(upload(device, b)?),
                    None => None,
                },
                gate: wrap_experts(device, &m.gate)?,
                up: wrap_experts(device, &m.up)?,
                down: wrap_experts(device, &m.down)?,
                shared: match &m.shared {
                    Some(s) => Some(GpuShared {
                        gate: wrap(device, &s.gate)?,
                        up: wrap(device, &s.up)?,
                        down: wrap(device, &s.down)?,
                        gate_inp: match &s.gate_inp {
                            Some(g) => Some(wrap(device, g)?),
                            None => None,
                        },
                    }),
                    None => None,
                },
            })),
        },
    })
}

impl<'m> MetalEngine<'m> {
    pub fn new(model: &'m Model<'m>, max_batch: usize) -> Result<Self, Error> {
        let c = &model.cfg;
        if c.recurrent.contains(&true) && (c.ssm_state != 128 || c.conv_kernel != 4) {
            return Err(backend(
                "metal: the delta net kernels expect state size 128 and conv kernel 4",
            ));
        }
        let group = c.n_head / c.n_head_kv;
        let attn_nsg = ATTN_GROUPS.iter().find(|g| g.0 == group).map(|g| g.1);
        let Some(attn_nsg) = attn_nsg.filter(|_| {
            ATTN_HEAD_DIMS.contains(&c.head_dim)
                && c.n_head.is_multiple_of(c.n_head_kv)
                && c.n_rot.is_multiple_of(64)
        }) else {
            return Err(backend("metal: unsupported attention head geometry"));
        };

        let device = MTLCreateSystemDefaultDevice().ok_or_else(|| backend("metal: no device"))?;
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(SOURCE), None)
            .map_err(|e| {
                backend(format!(
                    "metal: shader compilation failed: {}",
                    e.localizedDescription()
                ))
            })?;
        let mut pipelines = Vec::with_capacity(Kernel::ALL.len());
        for (i, &k) in Kernel::ALL.iter().enumerate() {
            debug_assert_eq!(k as usize, i);
            let name = k.name(c.head_dim, group);
            let f = library
                .newFunctionWithName(&NSString::from_str(&name))
                .ok_or_else(|| backend(format!("metal: missing kernel {name}")))?;
            let p = device
                .newComputePipelineStateWithFunction_error(&f)
                .map_err(|e| {
                    backend(format!(
                        "metal: pipeline {name}: {}",
                        e.localizedDescription()
                    ))
                })?;
            pipelines.push(p);
        }
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| backend("metal: no command queue"))?;

        let layers = model
            .layers
            .iter()
            .map(|l| gpu_layer(&device, l))
            .collect::<Result<_, _>>()?;
        let mtp = match &model.mtp {
            Some(m) => Some(GpuMtp {
                enorm: upload(&device, &m.enorm)?,
                hnorm: upload(&device, &m.hnorm)?,
                eh_proj: wrap(&device, &m.eh_proj)?,
                layer: gpu_layer(&device, &m.layer)?,
                head_norm: upload(&device, &m.head_norm)?,
            }),
            None => None,
        };

        Ok(MetalEngine {
            output_norm: upload(&device, &model.output_norm)?,
            output: wrap(&device, &model.output)?,
            inv_freq: upload(&device, &model.inv_freq)?,
            model,
            device,
            queue,
            k: Kernels { pipelines },
            attn_nsg,
            layers,
            mtp,
            max_batch: max_batch.max(1),
        })
    }

    fn cfg(&self) -> &Config {
        &self.model.cfg
    }

    fn has_sinks(&self, mixer: &GpuMixer) -> bool {
        matches!(mixer, GpuMixer::Attention { sinks: Some(_), .. })
    }

    fn dispatch(
        &self,
        enc: &Encoder,
        kernel: Kernel,
        grid: [usize; 3],
        threads: [usize; 3],
        shmem: usize,
    ) {
        enc.setComputePipelineState(self.k.get(kernel));
        if shmem > 0 {
            unsafe { enc.setThreadgroupMemoryLength_atIndex(shmem, 0) };
        }
        let size = |v: [usize; 3]| MTLSize {
            width: v[0],
            height: v[1],
            depth: v[2],
        };
        let barrier = GROUP.with(|g| match g.get() {
            Group::None => true,
            Group::First => {
                g.set(Group::In);
                true
            }
            Group::In => false,
        });
        if barrier {
            enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(size(grid), size(threads));
    }

    /// Encode the dispatches of `f` to run concurrently: they wait for the
    /// dispatches before them, not for each other, so none may read what
    /// another writes, nor two write the same bytes
    fn concurrent<R>(&self, f: impl FnOnce() -> R) -> R {
        GROUP.with(|g| {
            assert!(g.get() == Group::None, "nested concurrent groups");
            g.set(Group::First);
        });
        let r = f();
        GROUP.with(|g| g.set(Group::None));
        r
    }

    fn args<T>(enc: &Encoder, args: &T) {
        unsafe { enc.setBytes_length_atIndex(NonNull::from(args).cast(), size_of::<T>(), 0) };
    }

    fn bind(enc: &Encoder, bufs: &[B<'_>]) {
        for (i, b) in bufs.iter().enumerate() {
            unsafe { enc.setBuffer_offset_atIndex(Some(b.0), b.1, i + 1) };
        }
    }

    /// `out[t][r] = dot(w.row(r), x[t])` for `n` input rows
    fn matmul(&self, enc: &Encoder, w: &GpuWeight, x: B<'_>, n: usize, out: B<'_>) {
        let args = MatArgs {
            k: w.cols as i32,
            rows: w.rows as i32,
            n: n as i32,
            pad: 0,
            w_row_bytes: w.row_bytes as u64,
            x_row_bytes: (w.cols * 4) as u64,
        };
        Self::args(enc, &args);
        Self::bind(enc, &[B(&w.buf, w.offset), x, out]);
        match mv_kernel(w.ty) {
            // the float32 weights are small vectors' projections, too few rows to fill mmv
            Some((name, rows_per_tg, threads))
                if n <= MV_MAX_ROWS || (w.ty == GgmlType::F32 && n <= 8) =>
            {
                self.dispatch(
                    enc,
                    name,
                    [w.rows.div_ceil(rows_per_tg), n, 1],
                    [threads, 1, 1],
                    0,
                )
            }
            _ if n <= MMV_MAX_ROWS => self.dispatch(
                enc,
                mmv_kernel(w.ty).expect("checked at load"),
                [w.rows.div_ceil(MMV_WEIGHT_ROWS), n.div_ceil(MMV_ROWS), 1],
                [64, 1, 1],
                0,
            ),
            _ => {
                let name = mm_kernel(w.ty).expect("checked at load");
                self.dispatch(
                    enc,
                    name,
                    [n.div_ceil(32), w.rows.div_ceil(64), 1],
                    [128, 1, 1],
                    8192,
                )
            }
        }
    }

    /// rms norm of `rows` rows of `width`, optionally gated by silu(gate)
    #[allow(clippy::too_many_arguments)]
    fn rms_norm(
        &self,
        enc: &Encoder,
        x: B<'_>,
        w: &Buffer,
        out: B<'_>,
        gate: Option<B<'_>>,
        rows: usize,
        width: usize,
    ) {
        self.rms_norm_strided(enc, x, w, out, gate, rows, width, width)
    }

    /// [`MetalEngine::rms_norm`] with output rows `ostride` floats apart
    #[allow(clippy::too_many_arguments)]
    fn rms_norm_strided(
        &self,
        enc: &Encoder,
        x: B<'_>,
        w: &Buffer,
        out: B<'_>,
        gate: Option<B<'_>>,
        rows: usize,
        width: usize,
        ostride: usize,
    ) {
        let args = NormArgs {
            n: width as i32,
            eps: self.cfg().eps,
            gated: gate.is_some() as i32,
            ostride: ostride as i32,
        };
        Self::args(enc, &args);
        Self::bind(enc, &[x, B::at(w), out, gate.unwrap_or(x)]);
        let threads = if width >= 1024 { 256 } else { 32 };
        self.dispatch(enc, Kernel::RmsNorm, [rows, 1, 1], [threads, 1, 1], 0);
    }

    fn elementwise(&self, enc: &Encoder, kernel: Kernel, a: B<'_>, b: B<'_>, n: usize) {
        Self::args(enc, &EltArgs { n: n as i32 });
        Self::bind(enc, &[a, b]);
        self.dispatch(enc, kernel, [n.div_ceil(256), 1, 1], [256, 1, 1], 0);
    }
}

impl Engine for MetalEngine<'_> {
    fn new_session(&self, context: usize, draft: usize) -> Result<Box<dyn Session + '_>, Error> {
        Ok(Box::new(MetalSession::new(self, context, draft)?))
    }

    fn batch_size(&self) -> usize {
        self.max_batch
    }

    // a verify pass of 5 tokens costs 1.75x one of 1 (the matmul kernels read
    // the weights once for 8 tokens); 5 drafts are about as fast, 3 and 6 slower
    fn draft_size(&self) -> usize {
        4
    }

    fn description(&self) -> String {
        format!("metal, {}", self.device.name())
    }

    // the small vectors are uploaded, the weights mapped
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
            .metal_session(context, draft, self.max_batch)
    }

    fn device_memory(&self) -> Option<DeviceMemory> {
        Some(DeviceMemory {
            allocated: self.device.currentAllocatedSize(),
            working_set: self.device.recommendedMaxWorkingSetSize() as usize,
        })
    }

    // the batch size sets the split-K attention's splits
    fn numerics(&self) -> Vec<(&'static str, String)> {
        vec![
            ("kind", "metal".into()),
            ("device", self.device.name().to_string()),
            ("kv_cache", "f16".into()),
            ("max_batch", self.max_batch.to_string()),
        ]
    }

    fn new_batch(
        &self,
        tokens: usize,
        logits: usize,
    ) -> Result<Option<Box<dyn Batch + '_>>, Error> {
        let tokens = tokens.max(1);
        Ok(Some(Box::new(MetalBatch {
            e: self,
            s: Scratch::new(self, tokens, logits.clamp(1, tokens), self.mtp.is_some())?,
            seqs: Vec::new(),
        })))
    }
}

/// `len` f32 of a shared buffer, from `offset` f32
#[allow(clippy::mut_from_ref)]
fn host(buf: &Buffer, offset: usize, len: usize) -> &mut [f32] {
    assert!(4 * (offset + len) <= buf.length());
    unsafe {
        std::slice::from_raw_parts_mut((buf.contents().as_ptr() as *mut f32).add(offset), len)
    }
}

/// `len` bytes of a shared buffer, from `offset`
#[allow(clippy::mut_from_ref)]
fn host_bytes(buf: &Buffer, offset: usize, len: usize) -> &mut [u8] {
    assert!(offset + len <= buf.length());
    unsafe { std::slice::from_raw_parts_mut((buf.contents().as_ptr() as *mut u8).add(offset), len) }
}

/// A copy of `len` bytes of a shared buffer, from `offset`
fn copy_out(buf: &Buffer, offset: usize, len: usize) -> Result<Vec<u8>, Error> {
    let mut v = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|_| Error::OutOfMemory { bytes: len })?;
    v.extend_from_slice(host_bytes(buf, offset, len));
    Ok(v)
}

/// See [`Session::snapshot`]: the f16 kv caches of the tokens, the current
/// slot of the delta net states, the MTP head's history
struct MetalSnapshot {
    layers: Vec<LayerSnapshot>,
    mtp: Option<MtpSnapshot>,
}

enum LayerSnapshot {
    Attention { k: Vec<u8>, v: Vec<u8> },
    DeltaNet { conv: Vec<u8>, s: Vec<u8> },
}

struct MtpSnapshot {
    k: Vec<u8>,
    v: Vec<u8>,
    rows: usize,
    last: Vec<u8>,
}

enum LayerState {
    /// with `ring`, a sliding window's: position `p` in row `p % ring`, see
    /// [`Shape::metal_ring`]
    Attention {
        k: Buffer,
        v: Buffer,
        ring: Option<usize>,
    },
    /// `slots` copies of each state, see `ssm_conv` in the shaders
    DeltaNet { conv: Buffer, s: Buffer },
}

/// Activation buffers, sized for passes of `batch` tokens and `logit_rows`
/// rows of logits: a session's own, or shared by the sequences of a batch
struct Scratch {
    batch: usize,
    logit_rows: usize,
    x: Buffer,
    h: Buffer,
    y: Buffer,
    /// q+gate / qkv / ffn gate
    a1: Buffer,
    /// z / ffn up
    a2: Buffer,
    k: Buffer,
    v: Buffer,
    q: Buffer,
    /// attention / delta net output
    o: Buffer,
    conv_out: Buffer,
    /// split-K attention partials: accumulators and (max, sum)
    part_o: Buffer,
    part_ml: Buffer,
    alpha: Buffer,
    beta: Buffer,
    /// normalized hidden states, the input of the output and MTP heads
    feat: Buffer,
    /// logits of up to `logit_rows` tokens
    logits: Buffer,
    moe: Option<MoeScratch>,
    mtp: Option<MtpScratch>,
}

/// The buffers of a mixture of experts feed-forward, see `moe_route`
struct MoeScratch {
    /// router logits, per token
    logits: Buffer,
    /// expert and weight of each pair
    ids: Buffer,
    weights: Buffer,
    /// pairs by expert, their tiles and the tile counts, for `mm_id` then
    /// `mmv_id`
    order: Buffer,
    tiles: Buffer,
    n_tiles: Buffer,
    /// output of each pair
    out: Buffer,
    /// the shared expert's gate, per token
    gate: Buffer,
    /// rank of each pair among its expert's, then the pairs of each expert
    rank: Buffer,
}

/// The MTP head's input rows of a pass
struct MtpScratch {
    /// embeddings of the row tokens, at the rows of their sequence's tokens
    emb: Buffer,
    /// input rows, `[enorm(embedding) ‖ hnorm(hidden)]`
    input: Buffer,
}

/// The MTP head's history, see [`crate::model::Model::draft`]
struct MtpState {
    kv: LayerState,
    /// rows entered: row `p` pairs token `p + 1` with the hidden state of token `p`
    rows: usize,
    /// normalized hidden state of the last token, waiting for its successor
    last: Buffer,
    /// normalized output of the last draft, the hidden state of the next
    chain: Buffer,
    /// normalized hidden states of the tokens of the last verify, until `keep`
    verified: Buffer,
    /// the output head rows of the draft vocabulary
    head: Option<(DraftVocab, Buffer)>,
}

/// One sequence: its caches and states. Its passes run with a [`Scratch`],
/// its session's or its batch's.
struct Seq {
    layers: Vec<LayerState>,
    mtp: Option<MtpState>,
    /// tokens per draft
    draft: usize,
    /// copies of the delta net states (one more than the draft length), and
    /// the current one
    slots: usize,
    slot: usize,
    /// first position of the last verify, until `keep`
    verify_from: Option<usize>,
    context: usize,
    n_past: usize,
    /// without a scratch
    memory: SessionMemory,
}

struct MetalSession<'e> {
    e: &'e MetalEngine<'e>,
    seq: Seq,
    s: Scratch,
    memory: SessionMemory,
}

/// Bytes the device allocates for a buffer of `n`: whole pages
fn pages(n: usize) -> usize {
    n.max(4).next_multiple_of(PAGE)
}

/// Bytes of a kv cache, keys or values: f16, see `attn_flash`
fn kv_bytes(context: usize, kv_dim: usize) -> usize {
    2 * context
        .next_multiple_of(ATTN_FLASH_C)
        .saturating_mul(kv_dim)
}

/// Most tiles of `moe_sort` for `n` tokens, for `mm_id` and for `mmv_id`:
/// the full tiles, and a last one per expert (of `mm_min` pairs or more for
/// `mm_id`). Also the most of any pass of fewer tokens: `tiles` holds both
/// lists, the `mmv_id` one after the most `mm_id` tiles of the pass.
fn max_tiles(c: &Config, n: usize, mm_min: usize) -> (usize, usize) {
    c.moe.as_ref().map_or((0, 0), |m| {
        let pairs = n * m.n_expert_used;
        (
            pairs / 32 + m.n_expert.min(pairs / mm_min),
            pairs.div_ceil(8) + m.n_expert.min(pairs),
        )
    })
}

/// Width of a feed-forward's activations per token: the experts' of all of a
/// token's pairs, or the shared expert's
fn ffn_width(c: &Config) -> usize {
    let dense = if c.moe.is_none() || c.n_dense_lead > 0 {
        c.n_ff
    } else {
        0
    };
    c.moe
        .as_ref()
        .map_or(0, |m| (m.n_expert_used * m.n_ff_exp).max(m.n_ff_shexp))
        .max(dense)
}

/// Bytes of the buffers of [`Scratch`] but `moe` and `mtp`, in the order of
/// its fields, for passes of `b` tokens and `logit_rows` rows of logits
fn scratch_bytes(c: &Config, b: usize, logit_rows: usize) -> [usize; 16] {
    let kv_dim = c.n_head_kv * c.head_dim;
    let qd = c.n_head * c.head_dim;
    let f32s = |n: usize| 4 * n;
    [
        f32s(b * c.n_embd),
        f32s(b * c.n_embd),
        f32s(b * c.n_embd),
        f32s(b * c.conv_dim().max(2 * qd).max(ffn_width(c))),
        f32s(b * ffn_width(c).max(c.d_inner())),
        f32s(b * kv_dim),
        f32s(b * kv_dim),
        f32s(b * qd),
        f32s(b * qd.max(c.d_inner())),
        f32s(b * c.conv_dim()),
        f32s(attn_parts(c, b) * c.head_dim),
        f32s(2 * attn_parts(c, b)),
        f32s(b * c.ssm_v_heads),
        f32s(b * c.ssm_v_heads),
        f32s(b * c.n_embd),
        f32s(logit_rows * c.n_vocab),
    ]
}

/// Bytes of the buffers of [`MoeScratch`], in the order of its fields, for
/// passes of `b` tokens
fn moe_bytes(c: &Config, b: usize) -> Option<[usize; 9]> {
    let m = c.moe.as_ref()?;
    let pairs = b * m.n_expert_used;
    Some([
        4 * b * m.n_expert,
        4 * pairs,
        4 * pairs,
        4 * pairs,
        16 * {
            let (mm, mv) = max_tiles(c, b, MM_ID_MIN);
            mm + mv
        },
        8,
        4 * pairs * c.n_embd,
        4 * b,
        4 * (pairs + m.n_expert),
    ])
}

/// Bytes of the buffers of [`MtpScratch`]: `emb`, `input`
fn mtp_scratch_bytes(c: &Config, b: usize) -> [usize; 2] {
    [4 * b * c.n_embd, 4 * b * 2 * c.n_embd]
}

/// Bytes of the buffers of [`MtpState`] but its kv caches and head: `last`,
/// `chain`, `verified`
fn mtp_bytes(c: &Config, slots: usize) -> [usize; 3] {
    [4 * c.n_embd, 4 * c.n_embd, 4 * slots * c.n_embd]
}

impl Shape<'_> {
    /// Tokens a verify pass of a session asked for `draft` checks after its
    /// first, a verify pass being one chunk: the MTP head's drafts, or the
    /// proposals from the conversation (lookup), which need the room on a
    /// model without MTP head too
    fn metal_draft(&self, draft: usize, max_batch: usize) -> usize {
        draft.min(max_batch.max(1) - 1)
    }

    /// Whether a session asked for `draft` drafts with the MTP head: its
    /// state, kv cache and draft vocabulary
    fn metal_mtp(&self, draft: usize, max_batch: usize) -> bool {
        self.has_mtp && self.metal_draft(draft, max_batch) > 0
    }

    /// Positions the caches of a sliding window layer hold, when fewer than
    /// the context, for passes of up to `max_batch` tokens: a pass writes its
    /// rows before its attention, so the window but a token's own position
    /// and the pass's positions. A power of two, so that the kernels take
    /// position `p` at row `p & (ring - 1)`, and a multiple of `ATTN_FLASH_C`,
    /// so that the `attn_flash` tiles don't wrap.
    fn metal_ring(&self, context: usize, max_batch: usize) -> Option<usize> {
        let ring = (self.cfg.sliding_window + max_batch.max(1) - 1)
            .next_power_of_two()
            .max(ATTN_FLASH_C);
        (ring < context.next_multiple_of(ATTN_FLASH_C)).then_some(ring)
    }

    /// Bytes of a sliding window layer's cache, keys or values
    fn metal_window_bytes(&self, context: usize, max_batch: usize) -> usize {
        match self.metal_ring(context, max_batch) {
            Some(ring) => 2 * ring * self.kv_dim(),
            None => kv_bytes(context, self.kv_dim()),
        }
    }

    /// What a [`Seq`] allocates, in whole pages as the device does: a
    /// session's memory but its scratch
    fn metal_seq(&self, context: usize, draft: usize, max_batch: usize) -> SessionMemory {
        let c = self.cfg;
        let mtp = self.metal_mtp(draft, max_batch);
        let slots = self.metal_draft(draft, max_batch) + 1;
        let attention = self.attention_layers() + mtp as usize;
        let (conv, s) = self.delta_net_state();
        let (window, ring) = (self.window_layers(), self.metal_ring(context, max_batch));
        let window_bytes = window * 2 * pages(self.metal_window_bytes(context, max_batch));
        SessionMemory {
            // windows as long as the context grow with it
            kv: attention * 2 * pages(kv_bytes(context, self.kv_dim()))
                + if ring.is_none() { window_bytes } else { 0 },
            recurrent: self.delta_net_layers() * (pages(slots * 4 * conv) + pages(slots * 4 * s))
                + if ring.is_some() { window_bytes } else { 0 },
            draft_head: if mtp && self.draft_head() > 0 {
                pages(self.draft_head())
            } else {
                0
            },
            scratch: if mtp {
                mtp_bytes(c, slots).map(pages).iter().sum()
            } else {
                0
            },
            // the logits of a verify pass, copied out
            transient: 4 * slots * c.n_vocab,
            kv_per_token: (attention + if ring.is_none() { window } else { 0 })
                * 2
                * 2
                * self.kv_dim(),
        }
    }

    /// What a [`Scratch`] allocates for passes of `b` tokens and `logit_rows`
    /// rows of logits, with the MTP head's buffers when `mtp`
    fn metal_scratch(&self, b: usize, logit_rows: usize, mtp: bool) -> usize {
        let c = self.cfg;
        let mut bytes: usize = scratch_bytes(c, b, logit_rows).map(pages).iter().sum();
        if let Some(moe) = moe_bytes(c, b) {
            bytes += moe.map(pages).iter().sum::<usize>();
        }
        if mtp {
            bytes += mtp_scratch_bytes(c, b).map(pages).iter().sum::<usize>();
        }
        bytes
    }

    /// What [`MetalEngine::new_session`](Engine::new_session) allocates, in
    /// whole pages as the device does
    pub(crate) fn metal_session(
        &self,
        context: usize,
        draft: usize,
        max_batch: usize,
    ) -> SessionMemory {
        let b = max_batch.max(1);
        let seq = self.metal_seq(context, draft, b);
        let slots = self.metal_draft(draft, b) + 1;
        SessionMemory {
            scratch: seq.scratch + self.metal_scratch(b, slots, self.metal_mtp(draft, b)),
            ..seq
        }
    }
}

/// Allocations counting their bytes, in whole pages
struct Alloc<'d> {
    dev: &'d Device,
    bytes: usize,
}

impl Alloc<'_> {
    fn zeroed(&mut self, n: usize) -> Result<Buffer, Error> {
        self.bytes += pages(n);
        zeroed(self.dev, n)
    }
}

impl Scratch {
    fn new(e: &MetalEngine<'_>, b: usize, logit_rows: usize, mtp: bool) -> Result<Self, Error> {
        let c = e.cfg();
        let mut a = Alloc {
            dev: &e.device,
            bytes: 0,
        };
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
            part_o,
            part_ml,
            alpha,
            beta,
            feat,
            logits,
        ] = scratch_bytes(c, b, logit_rows);
        let s = Scratch {
            batch: b,
            logit_rows,
            x: a.zeroed(x)?,
            h: a.zeroed(h)?,
            y: a.zeroed(y)?,
            a1: a.zeroed(a1)?,
            a2: a.zeroed(a2)?,
            k: a.zeroed(k)?,
            v: a.zeroed(v)?,
            q: a.zeroed(q)?,
            o: a.zeroed(o)?,
            conv_out: a.zeroed(conv_out)?,
            part_o: a.zeroed(part_o)?,
            part_ml: a.zeroed(part_ml)?,
            alpha: a.zeroed(alpha)?,
            beta: a.zeroed(beta)?,
            feat: a.zeroed(feat)?,
            logits: a.zeroed(logits)?,
            moe: match moe_bytes(c, b) {
                Some([logits, ids, weights, order, tiles, n_tiles, out, gate, rank]) => {
                    Some(MoeScratch {
                        logits: a.zeroed(logits)?,
                        ids: a.zeroed(ids)?,
                        weights: a.zeroed(weights)?,
                        order: a.zeroed(order)?,
                        tiles: a.zeroed(tiles)?,
                        n_tiles: a.zeroed(n_tiles)?,
                        out: a.zeroed(out)?,
                        gate: a.zeroed(gate)?,
                        rank: a.zeroed(rank)?,
                    })
                }
                None => None,
            },
            mtp: if mtp {
                let [emb, input] = mtp_scratch_bytes(c, b);
                Some(MtpScratch {
                    emb: a.zeroed(emb)?,
                    input: a.zeroed(input)?,
                })
            } else {
                None
            },
        };
        debug_assert_eq!(a.bytes, e.model.shape().metal_scratch(b, logit_rows, mtp));
        Ok(s)
    }
}

impl Seq {
    /// A sequence whose verify passes fit in passes of `max_batch` tokens
    fn new(
        e: &MetalEngine<'_>,
        context: usize,
        draft: usize,
        max_batch: usize,
    ) -> Result<Self, Error> {
        let c = e.cfg();
        let shape = e.model.shape();
        let memory = shape.metal_seq(context, draft, max_batch);
        let mtp = shape.metal_mtp(draft, max_batch);
        let draft = shape.metal_draft(draft, max_batch);
        let slots = draft + 1;
        let kv_bytes = kv_bytes(context, shape.kv_dim());
        let (ring, window_bytes) = (
            shape.metal_ring(context, max_batch),
            shape.metal_window_bytes(context, max_batch),
        );
        let (conv, s) = shape.delta_net_state();
        let mut a = Alloc {
            dev: &e.device,
            bytes: 0,
        };
        let mut layers = Vec::with_capacity(c.n_layer);
        for (i, &r) in c.recurrent.iter().enumerate() {
            layers.push(if r {
                LayerState::DeltaNet {
                    conv: a.zeroed(slots * 4 * conv)?,
                    s: a.zeroed(slots * 4 * s)?,
                }
            } else if c.is_swa(i) {
                LayerState::Attention {
                    k: a.zeroed(window_bytes)?,
                    v: a.zeroed(window_bytes)?,
                    ring,
                }
            } else {
                LayerState::Attention {
                    k: a.zeroed(kv_bytes)?,
                    v: a.zeroed(kv_bytes)?,
                    ring: None,
                }
            });
        }
        let mtp = if mtp {
            let [last, chain, verified] = mtp_bytes(c, slots);
            Some(MtpState {
                kv: LayerState::Attention {
                    k: a.zeroed(kv_bytes)?,
                    v: a.zeroed(kv_bytes)?,
                    ring: None,
                },
                rows: 0,
                last: a.zeroed(last)?,
                chain: a.zeroed(chain)?,
                verified: a.zeroed(verified)?,
                head: match DraftVocab::new(c.n_vocab) {
                    Some(vocab) => {
                        let out = &e.model.output;
                        let rb = out.data.len() / out.rows;
                        let buf = a.zeroed(DRAFT_CAP * rb)?;
                        let n = vocab.len();
                        copy_head_rows(out, vocab.tokens(0..n), host_bytes(&buf, 0, n * rb));
                        Some((vocab, buf))
                    }
                    None => None,
                },
            })
        } else {
            None
        };
        debug_assert_eq!(a.bytes, memory.allocated());
        Ok(Seq {
            layers,
            mtp,
            draft,
            slots,
            slot: 0,
            verify_from: None,
            context,
            n_past: 0,
            memory,
        })
    }

    /// Keep the first `n` tokens of the last verify, dropping the others
    fn keep(&mut self, e: &MetalEngine<'_>, n: usize) {
        let pos0 = self.verify_from.take().expect("keep without verify");
        let verified = self.n_past - pos0;
        assert!(
            n >= 1 && n <= verified,
            "keep {n} of {verified} verified tokens"
        );
        // the slot after the n-th token holds the delta net states to go on from
        self.slot = (self.slot + n) % self.slots;
        self.n_past = pos0 + n;
        // attention caches are overwritten from the new position
        if let Some(m) = &mut self.mtp {
            let emb = e.cfg().n_embd;
            m.rows = m.rows.min(self.n_past - 1);
            host(&m.last, 0, emb).copy_from_slice(host(&m.verified, (n - 1) * emb, emb));
        }
    }

    fn snapshot(&self, e: &MetalEngine<'_>) -> Result<Snapshot, Error> {
        assert!(
            self.verify_from.is_none(),
            "snapshot between verify and keep"
        );
        let c = e.cfg();
        let kv_row = 2 * c.n_head_kv * c.head_dim;
        let (conv_slot, s_slot) = e.slot_bytes();
        let mut bytes = 0;
        let mut copy = |buf: &Buffer, offset: usize, len: usize| {
            bytes += len;
            copy_out(buf, offset, len)
        };
        let mut layers = Vec::with_capacity(self.layers.len());
        for (i, ls) in self.layers.iter().enumerate() {
            layers.push(match ls {
                // the positions a next token may see, in order
                LayerState::Attention { k, v, ring } => {
                    let first = self.n_past - window_rows(c, i, self.n_past);
                    let mut rows = |buf: &Buffer| -> Result<Vec<u8>, Error> {
                        let bytes = (self.n_past - first) * kv_row;
                        let mut out = Vec::new();
                        out.try_reserve_exact(bytes)
                            .map_err(|_| Error::OutOfMemory { bytes })?;
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
        let mtp = match &self.mtp {
            Some(MtpState {
                kv: LayerState::Attention { k, v, .. },
                rows,
                last,
                ..
            }) => Some(MtpSnapshot {
                k: copy(k, 0, rows * kv_row)?,
                v: copy(v, 0, rows * kv_row)?,
                rows: *rows,
                last: copy(last, 0, 4 * c.n_embd)?,
            }),
            Some(_) => unreachable!("the MTP head has an attention layer"),
            None => None,
        };
        Ok(Snapshot::new(
            self.n_past,
            bytes,
            MetalSnapshot { layers, mtp },
        ))
    }

    fn restore(&mut self, e: &MetalEngine<'_>, snapshot: &Snapshot) -> Result<(), Error> {
        let snap: &MetalSnapshot = snapshot.data()?;
        let n_past = snapshot.n_past();
        if n_past > self.context {
            return Err(Error::Snapshot("more tokens than the context holds"));
        }
        let c = e.cfg();
        let kv_row = 2 * c.n_head_kv * c.head_dim;
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
                })
            && match (&self.mtp, &snap.mtp) {
                (Some(_), Some(m)) => m.k.len() == m.rows * kv_row && m.last.len() == 4 * c.n_embd,
                (Some(_), None) => {
                    return Err(Error::Snapshot("no MTP history, for a drafting session"));
                }
                (None, _) => true,
            };
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
                    for (j, p) in (first..n_past).enumerate() {
                        let at = ring.map_or(p, |r| p % r) * kv_row;
                        let src = j * kv_row..(j + 1) * kv_row;
                        host_bytes(k, at, kv_row).copy_from_slice(&sk[src.clone()]);
                        host_bytes(v, at, kv_row).copy_from_slice(&sv[src]);
                    }
                }
                (LayerState::DeltaNet { conv, s }, LayerSnapshot::DeltaNet { conv: sc, s: ss }) => {
                    host_bytes(conv, 0, sc.len()).copy_from_slice(sc);
                    host_bytes(s, 0, ss.len()).copy_from_slice(ss);
                }
                _ => unreachable!("checked above"),
            }
        }
        if let (Some(m), Some(sm)) = (&mut self.mtp, &snap.mtp) {
            let LayerState::Attention { k, v, .. } = &m.kv else {
                unreachable!("the MTP head has an attention layer")
            };
            host_bytes(k, 0, sm.k.len()).copy_from_slice(&sm.k);
            host_bytes(v, 0, sm.v.len()).copy_from_slice(&sm.v);
            host_bytes(&m.last, 0, sm.last.len()).copy_from_slice(&sm.last);
            m.rows = sm.rows;
        }
        self.slot = 0;
        self.n_past = n_past;
        self.verify_from = None;
        Ok(())
    }
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

/// A sequence's tokens in a pass: rows `row0..row0 + n` of the activations,
/// at positions `pos0..` of its layer state `ls`. With `save`, a delta net
/// keeps its states after each token.
struct Seg<'a> {
    ls: &'a LayerState,
    row0: usize,
    n: usize,
    pos0: usize,
    slot: usize,
    slots: usize,
    save: bool,
}

/// Where a sequence's tokens are in a pass
struct Place {
    row0: usize,
    n: usize,
    pos0: usize,
    /// the MTP rows the pass completes, from the first without a row to the
    /// one before the last token
    mtp_rows: Option<(usize, usize)>,
    /// first row of its logits
    logit0: usize,
}

impl MetalEngine<'_> {
    /// Bytes of one slot of a delta net's conv and recurrent states
    fn slot_bytes(&self) -> (usize, usize) {
        let (conv, s) = self.model.shape().delta_net_state();
        (4 * conv, 4 * s)
    }

    /// Encode into one command buffer and wait for it
    fn submit(&self, encode: impl FnOnce(&Encoder)) {
        let cb = self
            .queue
            .commandBuffer()
            .expect("metal: no command buffer");
        let enc = cb
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
            .expect("metal: no compute encoder");
        encode(&enc);
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        if cb.status() == MTLCommandBufferStatus::Error {
            let msg = cb
                .error()
                .map(|e| e.localizedDescription().to_string())
                .unwrap_or_default();
            panic!("metal: command buffer failed: {msg}");
        }
    }

    /// One pass of the tokens of several sequences, with the activations in
    /// `s`: the matmuls, norms and feed-forwards over all the rows at once,
    /// the attention, delta nets and MTP head per sequence. Returns the
    /// logits each [`Pass`] asks for.
    fn pass(&self, s: &Scratch, items: &mut [(&mut Seq, &[TokenId], Pass)]) -> Vec<Vec<Vec<f32>>> {
        let c = self.cfg();
        let emb = c.n_embd;

        // the draft vocabularies take the tokens' head rows
        for (seq, tokens, pass) in items.iter_mut() {
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
            if let Some((vocab, buf)) = seq.mtp.as_mut().and_then(|m| m.head.as_mut()) {
                // a verify pass counts its first token only: the drafts have rows
                let counted = if *pass == Pass::Verify { 1 } else { n };
                let added = vocab.observe(tokens, counted);
                let out = &self.model.output;
                let rb = out.data.len() / out.rows;
                let dst = host_bytes(buf, added.start * rb, added.len() * rb);
                copy_head_rows(out, vocab.tokens(added), dst);
            }
        }

        let n: usize = items.iter().map(|(_, t, _)| t.len()).sum();
        assert!(n <= s.batch, "a pass of {n} tokens, {} at most", s.batch);
        let mut places = Vec::with_capacity(items.len());
        let (mut row0, mut logit0) = (0, 0);
        // the embedding lookup happens on the CPU, straight into the shared buffer
        let x_host = host(&s.x, 0, n * emb);
        for (seq, tokens, pass) in items.iter() {
            let len = tokens.len();
            for (t, &tok) in tokens.iter().enumerate() {
                let r = row0 + t;
                self.model
                    .tok_embd
                    .dequantize_row(tok as usize, &mut x_host[r * emb..(r + 1) * emb]);
            }
            let pos0 = seq.n_past;
            let mtp_rows = seq
                .mtp
                .as_ref()
                .map(|m| (m.rows, pos0 + len - 1))
                .filter(|(first, end)| first < end);
            // the embeddings of the MTP rows' tokens
            if let (Some(ms), Some((first, end))) = (&s.mtp, mtp_rows) {
                let from = row0 + first + 1 - pos0;
                host(&ms.emb, row0 * emb, (end - first) * emb)
                    .copy_from_slice(&x_host[from * emb..(row0 + len) * emb]);
            }
            places.push(Place {
                row0,
                n: len,
                pos0,
                mtp_rows,
                logit0,
            });
            row0 += len;
            logit0 += match pass {
                Pass::Prefill => 0,
                Pass::Forward => 1,
                Pass::Verify => len,
            };
        }
        let logit_rows = logit0;
        assert!(
            logit_rows <= s.logit_rows,
            "{logit_rows} rows of logits, {} at most",
            s.logit_rows
        );

        let (x, h, feat, logits) = (B::at(&s.x), B::at(&s.h), B::at(&s.feat), B::at(&s.logits));
        self.submit(|enc| {
            for (i, layer) in self.layers.iter().enumerate() {
                let segs: Vec<Seg<'_>> = items
                    .iter()
                    .zip(&places)
                    .map(|((seq, _, pass), p)| Seg {
                        ls: &seq.layers[i],
                        row0: p.row0,
                        n: p.n,
                        pos0: p.pos0,
                        slot: seq.slot,
                        slots: seq.slots,
                        save: *pass == Pass::Verify,
                    })
                    .collect();
                self.encode_layer(enc, s, layer, &segs, n);
            }

            // normalized hidden states: every row for the MTP head and
            // verify, else the last for its logits
            for ((seq, _, pass), p) in items.iter().zip(&places) {
                let all = seq.mtp.is_some() || *pass == Pass::Verify;
                if *pass == Pass::Prefill && !all {
                    continue;
                }
                let from = p.row0 + if all { 0 } else { p.n - 1 };
                let rows = p.row0 + p.n - from;
                self.rms_norm(
                    enc,
                    x.f32_off(from * emb),
                    &self.output_norm,
                    feat.f32_off(from * emb),
                    None,
                    rows,
                    emb,
                );
            }

            if let [(_, _, pass)] = &items[..] {
                // a sequence alone: its rows of logits are rows of `feat`
                let p = &places[0];
                match pass {
                    Pass::Prefill => {}
                    Pass::Forward => {
                        self.matmul(enc, &self.output, feat.f32_off((p.n - 1) * emb), 1, logits)
                    }
                    Pass::Verify => self.matmul(enc, &self.output, feat, p.n, logits),
                }
            } else if logit_rows > 0 {
                // the rows with logits, gathered in `h`, done with the layers
                for ((_, _, pass), p) in items.iter().zip(&places) {
                    let rows = match pass {
                        Pass::Prefill => continue,
                        Pass::Forward => 1,
                        Pass::Verify => p.n,
                    };
                    self.rms_norm(
                        enc,
                        x.f32_off((p.row0 + p.n - rows) * emb),
                        &self.output_norm,
                        h.f32_off(p.logit0 * emb),
                        None,
                        rows,
                        emb,
                    );
                }
                self.matmul(enc, &self.output, h, logit_rows, logits);
            }

            if let (Some(gm), Some(ms)) = (&self.mtp, &s.mtp) {
                for ((seq, _, _), p) in items.iter().zip(&places) {
                    let (Some(m), Some((first, end))) = (&seq.mtp, p.mtp_rows) else {
                        continue;
                    };
                    let rows = end - first;
                    // the first row's hidden state is the previous pass's
                    // last, unless a draft entered it
                    let prev = (first < p.pos0).then(|| B::at(&m.last));
                    let rest = feat.f32_off((p.row0 + first.max(p.pos0) - p.pos0) * emb);
                    let embs = B(&ms.emb, 4 * p.row0 * emb);
                    self.encode_mtp_input(enc, gm, ms, embs, rows, prev, rest);
                    self.matmul(enc, &gm.eh_proj, B::at(&ms.input), rows, x);
                    let seg = Seg {
                        ls: &m.kv,
                        row0: 0,
                        n: rows,
                        pos0: first,
                        slot: seq.slot,
                        slots: seq.slots,
                        save: false,
                    };
                    self.encode_layer(enc, s, &gm.layer, &[seg], rows);
                }
            }
        });

        items
            .iter_mut()
            .zip(&places)
            .map(|((seq, _, pass), p)| {
                seq.n_past += p.n;
                if let Some(m) = &mut seq.mtp {
                    if let Some((_, end)) = p.mtp_rows {
                        m.rows = end;
                    }
                    let last = (p.row0 + p.n - 1) * emb;
                    host(&m.last, 0, emb).copy_from_slice(host(&s.feat, last, emb));
                    if *pass == Pass::Verify {
                        host(&m.verified, 0, p.n * emb).copy_from_slice(host(
                            &s.feat,
                            p.row0 * emb,
                            p.n * emb,
                        ));
                    }
                }
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

    /// Up to `n` tokens the MTP head of `seq` predicts after `next`, with the
    /// activations in `s`, see [`Session::draft`]
    fn draft(
        &self,
        s: &Scratch,
        seq: &mut Seq,
        next: TokenId,
        n: usize,
        pick: &mut dyn FnMut(&DraftLogits<'_>) -> TokenId,
    ) -> Vec<TokenId> {
        let (Some(gm), Some(m), Some(ms)) = (&self.mtp, &seq.mtp, &s.mtp) else {
            return Vec::new();
        };
        let c = self.cfg();
        let emb = c.n_embd;
        // the last token's row waits for its successor, `next`
        assert_eq!(
            m.rows + 1,
            seq.n_past,
            "draft must follow a forward pass or keep"
        );
        let base = m.rows;
        let n = n.min(seq.draft);
        let mut drafts = Vec::with_capacity(n);
        let mut tok = next;
        for i in 0..n.min(seq.context - base) {
            self.model
                .tok_embd
                .dequantize_row(tok as usize, host(&ms.emb, 0, emb));
            self.submit(|enc| {
                // the first row is history, the chained ones are speculative
                // and overwritten by the next rows entered
                let hidden = if i == 0 { &m.last } else { &m.chain };
                self.encode_mtp_input(enc, gm, ms, B::at(&ms.emb), 1, None, B::at(hidden));
                self.matmul(enc, &gm.eh_proj, B::at(&ms.input), 1, B::at(&s.x));
                let seg = Seg {
                    ls: &m.kv,
                    row0: 0,
                    n: 1,
                    pos0: base + i,
                    slot: seq.slot,
                    slots: seq.slots,
                    save: false,
                };
                self.encode_layer(enc, s, &gm.layer, &[seg], 1);
                self.rms_norm(
                    enc,
                    B::at(&s.x),
                    &gm.head_norm,
                    B::at(&m.chain),
                    None,
                    1,
                    emb,
                );
                let head = match &m.head {
                    Some((vocab, buf)) if !vocab.full() => Some(GpuWeight {
                        buf: buf.clone(),
                        offset: 0,
                        rows: vocab.len(),
                        ..self.output.clone()
                    }),
                    _ => None,
                };
                self.matmul(
                    enc,
                    head.as_ref().unwrap_or(&self.output),
                    B::at(&m.chain),
                    1,
                    B::at(&s.logits),
                );
            });
            tok = match &m.head {
                Some((vocab, _)) if !vocab.full() => pick(&DraftLogits {
                    logits: host(&s.logits, 0, vocab.len()),
                    tokens: Some(vocab.tokens(0..vocab.len())),
                }),
                _ => pick(&DraftLogits {
                    logits: host(&s.logits, 0, c.n_vocab),
                    tokens: None,
                }),
            };
            drafts.push(tok);
        }
        if let Some(m) = &mut seq.mtp {
            m.rows = base + 1;
        }
        drafts
    }

    /// MTP input rows: `enorm` of the embeddings `embs` and `hnorm` of the
    /// hidden states, `prev` for the first row if given, then `rest`
    #[allow(clippy::too_many_arguments)]
    fn encode_mtp_input(
        &self,
        enc: &Encoder,
        gm: &GpuMtp,
        ms: &MtpScratch,
        embs: B<'_>,
        rows: usize,
        prev: Option<B<'_>>,
        rest: B<'_>,
    ) {
        let emb = self.cfg().n_embd;
        let input = B::at(&ms.input);
        self.rms_norm_strided(enc, embs, &gm.enorm, input, None, rows, emb, 2 * emb);
        let mut right = input.f32_off(emb);
        let mut rows = rows;
        if let Some(prev) = prev {
            self.rms_norm_strided(enc, prev, &gm.hnorm, right, None, 1, emb, 2 * emb);
            right = right.f32_off(2 * emb);
            rows -= 1;
        }
        if rows > 0 {
            self.rms_norm_strided(enc, rest, &gm.hnorm, right, None, rows, emb, 2 * emb);
        }
    }

    /// One decoder layer over the residual stream `s.x` of `n` tokens, the
    /// rows of the sequences `segs`
    fn encode_layer(
        &self,
        enc: &Encoder,
        s: &Scratch,
        layer: &GpuLayer,
        segs: &[Seg<'_>],
        n: usize,
    ) {
        let c = self.cfg();
        let (emb, hd) = (c.n_embd, c.head_dim);
        let (x, h, y) = (B::at(&s.x), B::at(&s.h), B::at(&s.y));

        self.rms_norm(enc, x, &layer.attn_norm, h, None, n, emb);
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
                let (sd, hk, hv) = (c.ssm_state, c.ssm_k_heads, c.ssm_v_heads);
                let conv_dim = c.conv_dim();
                self.concurrent(|| {
                    self.matmul(enc, wqkv, h, n, B::at(&s.a1));
                    self.matmul(enc, wgate, h, n, B::at(&s.a2));
                    self.matmul(enc, wbeta, h, n, B::at(&s.beta));
                    self.matmul(enc, walpha, h, n, B::at(&s.alpha));
                });

                for g in segs {
                    let LayerState::DeltaNet { conv, s: st } = g.ls else {
                        unreachable!("layer / state kind mismatch")
                    };
                    let (slot, slots, save) = (g.slot as i32, g.slots as i32, g.save as i32);
                    let r = g.row0;
                    MetalEngine::args(
                        enc,
                        &ConvArgs {
                            conv_dim: conv_dim as i32,
                            n: g.n as i32,
                            slot,
                            slots,
                            save,
                        },
                    );
                    MetalEngine::bind(
                        enc,
                        &[
                            B::at(&s.a1).f32_off(r * conv_dim),
                            B::at(conv1d),
                            B::at(conv),
                            B::at(&s.conv_out).f32_off(r * conv_dim),
                        ],
                    );
                    self.dispatch(
                        enc,
                        Kernel::SsmConv,
                        [conv_dim.div_ceil(256), 1, 1],
                        [256, 1, 1],
                        0,
                    );

                    let args = GdnArgs {
                        n: g.n as i32,
                        hk: hk as i32,
                        hv: hv as i32,
                        conv_dim: conv_dim as i32,
                        eps: c.eps,
                        slot,
                        slots,
                        save,
                    };
                    MetalEngine::args(enc, &args);
                    MetalEngine::bind(
                        enc,
                        &[
                            B::at(&s.conv_out).f32_off(r * conv_dim),
                            B::at(&s.alpha).f32_off(r * hv),
                            B::at(&s.beta).f32_off(r * hv),
                            B::at(dt_bias),
                            B::at(a),
                            B::at(st),
                            B::at(&s.o).f32_off(r * hv * sd),
                        ],
                    );
                    self.dispatch(enc, Kernel::Gdn128, [sd / 4, hv, 1], [32, 4, 1], 0);
                }

                // norm(o) * silu(z), per head, in place
                self.rms_norm(
                    enc,
                    B::at(&s.o),
                    norm,
                    B::at(&s.o),
                    Some(B::at(&s.a2)),
                    n * hv,
                    sd,
                );
                self.matmul(enc, wout, B::at(&s.o), n, y);
            }
            GpuMixer::Attention {
                wq,
                wk,
                wv,
                wo,
                q_norm,
                k_norm,
                rope,
                window,
                sinks,
            } => {
                self.concurrent(|| {
                    self.matmul(enc, wq, h, n, B::at(&s.a1));
                    self.matmul(enc, wk, h, n, B::at(&s.k));
                    self.matmul(enc, wv, h, n, B::at(&s.v));
                });

                let (qd, kv_dim) = (c.n_head * hd, c.n_head_kv * hd);
                let group = c.n_head / c.n_head_kv;
                let q_rows = if c.attn_gate { 2 * qd } else { qd };
                // a buffer for the kernels to bind when the layer has no sinks
                let sinks = sinks.as_ref().unwrap_or(q_norm);
                for g in segs {
                    let LayerState::Attention { k: kc, v: vc, ring } = g.ls else {
                        unreachable!("layer / state kind mismatch")
                    };
                    let (n, pos0, r) = (g.n, g.pos0, g.row0);
                    let (qg, q, o) = (
                        B::at(&s.a1).f32_off(r * q_rows),
                        B::at(&s.q).f32_off(r * qd),
                        B::at(&s.o).f32_off(r * qd),
                    );
                    // the positions the tokens see: from the first token's window
                    let lo = window.map_or(0, |w| (pos0 + 1).saturating_sub(w));
                    let span = pos0 + n - lo;
                    let flash_split =
                        (2..ATTN_FLASH_MIN).contains(&n) && span >= ATTN_FLASH_SPLIT_CTX;
                    let (lo, splits, chunk) = if flash_split {
                        // ranges of whole tiles
                        let lo = lo / ATTN_FLASH_C * ATTN_FLASH_C;
                        let (splits, chunk) = flash_splits(c, pos0 + n - lo, n, s.batch);
                        (lo, splits, chunk)
                    } else {
                        let (splits, chunk) = attn_splits(c, span, n, s.batch);
                        (lo, splits, chunk)
                    };
                    let args = AttnArgs {
                        n_head: c.n_head as i32,
                        n_kv: c.n_head_kv as i32,
                        hd: hd as i32,
                        n_rot: if *rope { c.n_rot as i32 } else { 0 },
                        pos0: pos0 as i32,
                        eps: c.eps,
                        scale: 1.0 / (hd as f32).sqrt(),
                        splits: splits as i32,
                        chunk: chunk as i32,
                        n: n as i32,
                        gated: c.attn_gate as i32,
                        window: window.map_or(0, |w| w as i32),
                        lo: lo as i32,
                        mask: ring.map_or(-1, |r| r as i32 - 1),
                        sinks: self.has_sinks(&layer.mixer) as i32,
                    };
                    MetalEngine::args(enc, &args);
                    MetalEngine::bind(
                        enc,
                        &[
                            qg,
                            B::at(&s.k).f32_off(r * kv_dim),
                            B::at(&s.v).f32_off(r * kv_dim),
                            B::at(q_norm),
                            B::at(k_norm),
                            B::at(&self.inv_freq),
                            q,
                            B::at(kc),
                            B::at(vc),
                        ],
                    );
                    self.dispatch(
                        enc,
                        Kernel::AttnPrep,
                        [c.n_head + c.n_head_kv, n, 1],
                        [32, 1, 1],
                        0,
                    );

                    if n >= ATTN_FLASH_MIN {
                        MetalEngine::args(enc, &args);
                        MetalEngine::bind(enc, &[q, B::at(kc), B::at(vc), qg, o, B::at(sinks)]);
                        // queries (half), scores, and a float per row
                        let shmem = ATTN_FLASH_R * (2 * hd + 4 * ATTN_FLASH_C + 4);
                        let blocks = (n * group).div_ceil(ATTN_FLASH_R);
                        self.dispatch(
                            enc,
                            Kernel::AttnFlash,
                            [blocks, c.n_head_kv, 1],
                            [32 * ATTN_FLASH_NSG, 1, 1],
                            shmem,
                        );
                    } else {
                        MetalEngine::args(enc, &args);
                        MetalEngine::bind(
                            enc,
                            &[q, B::at(kc), B::at(vc), B::at(&s.part_o), B::at(&s.part_ml)],
                        );
                        if flash_split {
                            let shmem = ATTN_FLASH_R * (2 * hd + 4 * ATTN_FLASH_C + 4);
                            let blocks = (n * group).div_ceil(ATTN_FLASH_R);
                            self.dispatch(
                                enc,
                                Kernel::AttnFlashSplit,
                                [blocks, c.n_head_kv, splits],
                                [32 * ATTN_FLASH_NSG, 1, 1],
                                shmem,
                            );
                        } else {
                            let shmem = 4 * (2 * self.attn_nsg * group + group * hd);
                            self.dispatch(
                                enc,
                                Kernel::AttnSplit,
                                [c.n_head_kv, splits, n],
                                [32 * self.attn_nsg, 1, 1],
                                shmem,
                            );
                        }

                        MetalEngine::args(enc, &args);
                        MetalEngine::bind(
                            enc,
                            &[B::at(&s.part_o), B::at(&s.part_ml), qg, o, B::at(sinks)],
                        );
                        self.dispatch(enc, Kernel::AttnReduce, [c.n_head, n, 1], [hd, 1, 1], 0);
                    }
                }

                self.matmul(enc, wo, B::at(&s.o), n, y);
            }
        }
        self.elementwise(enc, Kernel::AddInplace, x, y, n * emb);

        self.rms_norm(enc, x, &layer.post_norm, h, None, n, emb);
        match &layer.ffn {
            GpuFfn::Dense { gate, up, down } => {
                self.encode_dense_ffn(enc, s, gate, up, down, n, y);
                self.elementwise(enc, Kernel::AddInplace, x, y, n * emb);
            }
            GpuFfn::Moe(m) => self.encode_moe(
                enc,
                s,
                &m.router,
                m.bias.as_ref(),
                [&m.gate, &m.up, &m.down],
                m.shared.as_ref(),
                n,
            ),
        }
    }

    /// `down(silu(gate h) * up h)` of `n` tokens into `out`, through `a1`
    /// and `a2`
    #[allow(clippy::too_many_arguments)]
    fn encode_dense_ffn(
        &self,
        enc: &Encoder,
        s: &Scratch,
        gate: &GpuWeight,
        up: &GpuWeight,
        down: &GpuWeight,
        n: usize,
        out: B<'_>,
    ) {
        let h = B::at(&s.h);
        self.concurrent(|| {
            self.matmul(enc, gate, h, n, B::at(&s.a1));
            self.matmul(enc, up, h, n, B::at(&s.a2));
        });
        self.elementwise(
            enc,
            Kernel::Swiglu,
            B::at(&s.a1),
            B::at(&s.a2),
            n * gate.rows,
        );
        self.matmul(enc, down, B::at(&s.a1), n, out);
    }

    /// The mixture of experts of the normalized `n` tokens in `h`, added to
    /// the residual stream `x`
    #[allow(clippy::too_many_arguments)]
    fn encode_moe(
        &self,
        enc: &Encoder,
        s: &Scratch,
        router: &GpuWeight,
        bias: Option<&Buffer>,
        [gate, up, down]: [&GpuExperts; 3],
        shared: Option<&GpuShared>,
        n: usize,
    ) {
        let c = self.cfg();
        let m = s.moe.as_ref().expect("moe scratch for a moe model");
        let cm = c.moe.as_ref().expect("a moe config");
        let (ne, k) = (cm.n_expert, cm.n_expert_used);
        let h = B::at(&s.h);

        // the shared expert first: the experts then take `a1` and `a2`
        if let Some(sh) = shared {
            self.concurrent(|| {
                self.matmul(enc, &sh.gate, h, n, B::at(&s.a1));
                self.matmul(enc, &sh.up, h, n, B::at(&s.a2));
                if let Some(gate_inp) = &sh.gate_inp {
                    self.matmul(enc, gate_inp, h, n, B::at(&m.gate));
                }
                self.matmul(enc, router, h, n, B::at(&m.logits));
            });
            let rows = sh.gate.rows;
            self.elementwise(enc, Kernel::Swiglu, B::at(&s.a1), B::at(&s.a2), n * rows);
            self.matmul(enc, &sh.down, B::at(&s.a1), n, B::at(&s.y));
        } else {
            self.matmul(enc, router, h, n, B::at(&m.logits));
        }
        let (mm_tiles, mv_tiles) = max_tiles(c, n, MM_ID_MIN);
        MetalEngine::args(
            enc,
            &RouteArgs {
                n_expert: ne as i32,
                k: k as i32,
                sigmoid: (cm.gating == Gating::Sigmoid) as i32,
                bias: bias.is_some() as i32,
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
                // any buffer without a bias
                B::at(bias.unwrap_or(&m.logits)),
            ],
        );
        self.dispatch(enc, Kernel::MoeRoute, [n, 1, 1], [256, 1, 1], 4 * ne);

        // a matvec per pair, when every type of the experts has one
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
            // the ranks and counts, a threadgroup per expert
            let (rank, count) = (B::at(&m.rank), B(&m.rank, 4 * n * k));
            MetalEngine::args(enc, &args);
            MetalEngine::bind(enc, &[B::at(&m.ids), rank, count]);
            self.dispatch(enc, Kernel::MoeRank, [ne, 1, 1], [256, 1, 1], 4 * 8);
            // then the pairs in order and the tiles
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
            // an expert has at most a pair per token
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
        // a token's row for each of its pairs, then a row per pair; the
        // `mm_id` and `mmv_id` tiles of a matrix are disjoint pairs
        self.concurrent(|| {
            matmul(gate, h, k, B::at(&s.a1));
            matmul(up, h, k, B::at(&s.a2));
        });
        self.elementwise(
            enc,
            Kernel::Swiglu,
            B::at(&s.a1),
            B::at(&s.a2),
            n * k * gate.rows,
        );
        self.concurrent(|| matmul(down, B::at(&s.a1), 1, B::at(&m.out)));

        MetalEngine::args(
            enc,
            &CombineArgs {
                n_embd: c.n_embd as i32,
                k: k as i32,
                shared: match shared {
                    None => 0,
                    Some(GpuShared {
                        gate_inp: Some(_), ..
                    }) => 1,
                    Some(_) => 2,
                },
            },
        );
        MetalEngine::bind(
            enc,
            &[
                B::at(&s.x),
                B::at(&m.out),
                B::at(&m.weights),
                B::at(&s.y),
                B::at(&m.gate),
            ],
        );
        self.dispatch(
            enc,
            Kernel::MoeCombine,
            [c.n_embd.div_ceil(256), n, 1],
            [256, 1, 1],
            0,
        );
    }
}

/// Split-K attention partials the scratch buffers hold: one per head and range
/// for a single token, one per head and token for a full batch
fn attn_parts(c: &Config, max_batch: usize) -> usize {
    c.n_head * ATTN_MAX_SPLITS.max(max_batch)
}

/// Split-K ranges and their length for `n` tokens over `n_ctx` positions:
/// enough ranges to fill the GPU when there are few tokens, bounded by the
/// partial buffers
fn attn_splits(c: &Config, n_ctx: usize, n: usize, max_batch: usize) -> (usize, usize) {
    let splits = n_ctx
        .div_ceil(ATTN_CHUNK)
        .min(ATTN_MAX_SPLITS)
        .min(attn_parts(c, max_batch) / (n * c.n_head))
        .max(1);
    (splits, n_ctx.div_ceil(splits))
}

/// Split-K ranges and their length, a multiple of `ATTN_FLASH_C`, for the
/// split `attn_flash` kernel on `n` tokens over `n_ctx` positions
fn flash_splits(c: &Config, n_ctx: usize, n: usize, max_batch: usize) -> (usize, usize) {
    let max = (attn_parts(c, max_batch) / (n * c.n_head)).clamp(1, ATTN_MAX_SPLITS);
    let chunk = n_ctx.div_ceil(max).next_multiple_of(ATTN_FLASH_C);
    (n_ctx.div_ceil(chunk), chunk)
}

impl<'e> MetalSession<'e> {
    fn new(e: &'e MetalEngine<'e>, context: usize, draft: usize) -> Result<Self, Error> {
        let seq = Seq::new(e, context, draft, e.max_batch)?;
        let s = Scratch::new(e, e.max_batch, seq.slots, seq.mtp.is_some())?;
        let memory = e.model.shape().metal_session(context, draft, e.max_batch);
        debug_assert_eq!(
            memory.allocated(),
            seq.memory.allocated()
                + e.model
                    .shape()
                    .metal_scratch(s.batch, s.logit_rows, s.mtp.is_some())
        );
        Ok(MetalSession { e, seq, s, memory })
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
            out = self
                .e
                .pass(&self.s, &mut [(&mut self.seq, chunk, pass)])
                .pop()
                .expect("a result per sequence");
        }
        out
    }
}

impl Session for MetalSession<'_> {
    fn forward(&mut self, tokens: &[TokenId]) -> Vec<f32> {
        self.run(tokens, Pass::Forward)
            .pop()
            .expect("logits requested")
    }

    fn prefill(&mut self, tokens: &[TokenId]) {
        self.run(tokens, Pass::Prefill);
    }

    fn draft(
        &mut self,
        next: TokenId,
        n: usize,
        pick: &mut dyn FnMut(&DraftLogits<'_>) -> TokenId,
    ) -> Vec<TokenId> {
        self.e.draft(&self.s, &mut self.seq, next, n, pick)
    }

    fn verify(&mut self, tokens: &[TokenId]) -> Vec<Vec<f32>> {
        self.e
            .pass(&self.s, &mut [(&mut self.seq, tokens, Pass::Verify)])
            .pop()
            .expect("a result per sequence")
    }

    fn keep(&mut self, n: usize) {
        self.seq.keep(self.e, n)
    }

    fn n_past(&self) -> usize {
        self.seq.n_past
    }

    fn memory(&self) -> SessionMemory {
        self.memory
    }

    fn snapshot(&self) -> Result<Snapshot, Error> {
        self.seq.snapshot(self.e)
    }

    fn restore(&mut self, snapshot: &Snapshot) -> Result<(), Error> {
        self.seq.restore(self.e, snapshot)
    }
}

/// Sequences sharing one [`Scratch`], whose passes run together, see
/// [`Engine::new_batch`]
struct MetalBatch<'e> {
    e: &'e MetalEngine<'e>,
    s: Scratch,
    seqs: Vec<Option<Seq>>,
}

impl MetalBatch<'_> {
    fn seq(&self, seq: SeqId) -> &Seq {
        self.seqs[seq].as_ref().expect("a sequence of the batch")
    }

    fn seq_mut(&mut self, seq: SeqId) -> &mut Seq {
        self.seqs[seq].as_mut().expect("a sequence of the batch")
    }
}

impl Batch for MetalBatch<'_> {
    fn add(&mut self, context: usize, draft: usize) -> Result<SeqId, Error> {
        let seq = Seq::new(self.e, context, draft, self.s.batch)?;
        Ok(insert(&mut self.seqs, seq))
    }

    fn remove(&mut self, seq: SeqId) {
        self.seqs[seq] = None;
    }

    fn max_tokens(&self) -> usize {
        self.s.batch
    }

    fn max_logits(&self) -> usize {
        self.s.logit_rows
    }

    fn run(&mut self, work: &[Work<'_>]) -> Vec<Vec<Vec<f32>>> {
        // out of the slab for the pass, each once
        let mut taken: Vec<(SeqId, Seq)> = work
            .iter()
            .map(|w| {
                let seq = self.seqs[w.seq]
                    .take()
                    .expect("a sequence of the batch, once per pass");
                (w.seq, seq)
            })
            .collect();
        let mut items: Vec<(&mut Seq, &[TokenId], Pass)> = taken
            .iter_mut()
            .zip(work)
            .map(|((_, seq), w)| (seq, w.tokens, w.pass))
            .collect();
        let out = self.e.pass(&self.s, &mut items);
        drop(items);
        for (id, seq) in taken {
            self.seqs[id] = Some(seq);
        }
        out
    }

    fn draft(
        &mut self,
        seq: SeqId,
        next: TokenId,
        n: usize,
        pick: &mut dyn FnMut(&DraftLogits<'_>) -> TokenId,
    ) -> Vec<TokenId> {
        let seq = self.seqs[seq].as_mut().expect("a sequence of the batch");
        self.e.draft(&self.s, seq, next, n, pick)
    }

    fn keep(&mut self, seq: SeqId, n: usize) {
        let e = self.e;
        self.seq_mut(seq).keep(e, n)
    }

    fn n_past(&self, seq: SeqId) -> usize {
        self.seq(seq).n_past
    }

    fn seq_memory(&self, seq: SeqId) -> SessionMemory {
        self.seq(seq).memory
    }

    fn session_memory(&self, context: usize, draft: usize) -> SessionMemory {
        self.e.model.shape().metal_seq(context, draft, self.s.batch)
    }

    fn memory(&self) -> SessionMemory {
        let s = &self.s;
        SessionMemory {
            scratch: self
                .e
                .model
                .shape()
                .metal_scratch(s.batch, s.logit_rows, s.mtp.is_some()),
            ..Default::default()
        }
    }

    fn snapshot(&self, seq: SeqId) -> Result<Snapshot, Error> {
        self.seq(seq).snapshot(self.e)
    }

    fn restore(&mut self, seq: SeqId, snapshot: &Snapshot) -> Result<(), Error> {
        let e = self.e;
        self.seq_mut(seq).restore(e, snapshot)
    }
}
