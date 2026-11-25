// CUDA kernels of the forward pass. Each kernel takes one argument, a struct
// laid out as its `#[repr(C)]` twin in mod.rs. Activations are f32, the kv
// caches f16.
//
// The math follows the CPU path (model.rs, ops.rs): no fast-math, IEEE
// division and square roots, so that the two stay close.

#include <cuda_fp16.h>
#include <stdint.h>

typedef unsigned long long u64;

#define WARP 32

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// sum over the block (blockDim.x a multiple of 32, up to 1024), to every thread
__device__ float block_sum(float v) {
    __shared__ float part[32];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) part[w] = v;
    __syncthreads();
    float s = 0.f;
    for (int i = 0; i < nw; i++) s += part[i];
    return s;
}

__device__ __forceinline__ float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ float silu(float x) { return x * sigmoid(x); }

__device__ __forceinline__ float f16(const uint8_t *p) {
    return __half2float(__ushort_as_half(*(const uint16_t *) p));
}

// ---------------------------------------------------------------------------
// Weight rows, dequantized 32 values at a time: unit u of a row is its values
// [32u, 32u + 32), a block of the 32-element types, a sub-block of the k-quants
// (as quant.rs's dequantize_row lays them out).
//
// The rows of Q4_0, Q4_1, Q8_0 and Q6_K are split (mod.rs repacks them): the
// quants of every block, in their order, then the scales (f16 d, or d and m),
// the row padded to 16 bytes, so that the quants load 16 bytes at a time. A
// Q6_K block keeps its ql, qh and scales (208 bytes), its d goes after.

enum Type { F32 = 0, F16 = 1, BF16 = 2, Q4_0 = 3, Q4_1 = 4, Q8_0 = 5, Q4_K = 6, Q5_K = 7, Q6_K = 8 };

// the float-row kernels (mm_rows) split rows of these 3 types
template <int T> __host__ __device__ constexpr bool split_rows() { return T == Q4_0 || T == Q4_1 || T == Q8_0; }
// the matvecs take int8 activations (quantize_q8) for the quantized types
template <int T> __host__ __device__ constexpr bool q8_act() { return T >= Q4_0; }

__device__ __forceinline__ void scale_min_k4(int j, const uint8_t *q, int &s, int &m) {
    if (j < 4) {
        s = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        s = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// scale_min_k4 of the 12 bytes of scales as 3 words (one 4-aligned load
// each, where byte loads would each cross the bus to host memory)
__device__ __forceinline__ int byte_of(const uint32_t *w, int k) {
    return (int) ((w[k >> 2] >> (8 * (k & 3))) & 0xff);
}

__device__ __forceinline__ void scale_min_k4w(int j, const uint32_t *w, int &s, int &m) {
    if (j < 4) {
        s = byte_of(w, j) & 63;
        m = byte_of(w, j + 4) & 63;
    } else {
        s = (byte_of(w, j + 4) & 0xF) | ((byte_of(w, j - 4) >> 6) << 4);
        m = (byte_of(w, j + 4) >> 4) | ((byte_of(w, j) >> 6) << 4);
    }
}

__device__ __forceinline__ void load_scales_k(const uint8_t *b, uint32_t *w, float &d, float &dmin) {
    const uint32_t *p = (const uint32_t *) b;
    const uint32_t dm = p[0];
    w[0] = p[1];
    w[1] = p[2];
    w[2] = p[3];
    d = __half2float(__ushort_as_half((unsigned short) (dm & 0xffff)));
    dmin = __half2float(__ushort_as_half((unsigned short) (dm >> 16)));
}

template <int T> __device__ __forceinline__ void deq32(const uint8_t *row, int cols, int u, float *o);

template <> __device__ __forceinline__ void deq32<F32>(const uint8_t *row, int cols, int u, float *o) {
    const float4 *p = (const float4 *) (row + 128 * u);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const float4 v = p[i];
        o[4 * i] = v.x; o[4 * i + 1] = v.y; o[4 * i + 2] = v.z; o[4 * i + 3] = v.w;
    }
}

template <> __device__ __forceinline__ void deq32<F16>(const uint8_t *row, int cols, int u, float *o) {
    const __half2 *p = (const __half2 *) (row + 64 * u);
#pragma unroll
    for (int i = 0; i < 16; i++) {
        const float2 v = __half22float2(p[i]);
        o[2 * i] = v.x; o[2 * i + 1] = v.y;
    }
}

template <> __device__ __forceinline__ void deq32<BF16>(const uint8_t *row, int cols, int u, float *o) {
    const uint32_t *p = (const uint32_t *) (row + 64 * u);
#pragma unroll
    for (int i = 0; i < 16; i++) {
        const uint32_t w = p[i];
        o[2 * i] = __uint_as_float(w << 16);
        o[2 * i + 1] = __uint_as_float(w & 0xffff0000u);
    }
}

template <> __device__ __forceinline__ void deq32<Q4_0>(const uint8_t *row, int cols, int u, float *o) {
    const uint8_t *b = row + 16 * u;
    const float d = f16(row + cols / 2 + 2 * u);
#pragma unroll
    for (int j = 0; j < 16; j++) {
        const int q = b[j];
        o[j] = (float) ((q & 0xF) - 8) * d;
        o[j + 16] = (float) ((q >> 4) - 8) * d;
    }
}

template <> __device__ __forceinline__ void deq32<Q4_1>(const uint8_t *row, int cols, int u, float *o) {
    const uint8_t *b = row + 16 * u;
    const uint8_t *dm = row + cols / 2 + 4 * u;
    const float d = f16(dm), m = f16(dm + 2);
#pragma unroll
    for (int j = 0; j < 16; j++) {
        const int q = b[j];
        o[j] = (float) (q & 0xF) * d + m;
        o[j + 16] = (float) (q >> 4) * d + m;
    }
}

template <> __device__ __forceinline__ void deq32<Q8_0>(const uint8_t *row, int cols, int u, float *o) {
    const int8_t *b = (const int8_t *) row + 32 * u;
    const float d = f16(row + cols + 2 * u);
#pragma unroll
    for (int j = 0; j < 32; j++) o[j] = (float) b[j] * d;
}

// The split rows E values at a time (16 of Q8_0, 32 of Q4_0 / Q4_1): chunk c
// of a row is its values [E c, E c + E), 16 bytes of quants
template <int T> __host__ __device__ constexpr int chunk_values() { return T == Q8_0 ? 16 : 32; }

__device__ __forceinline__ void i8x4(uint32_t w, float d, float *o) {
    o[0] = (float) (int8_t) (w & 0xff) * d;
    o[1] = (float) (int8_t) ((w >> 8) & 0xff) * d;
    o[2] = (float) (int8_t) ((w >> 16) & 0xff) * d;
    o[3] = (float) (int8_t) (w >> 24) * d;
}

template <int T> __device__ __forceinline__ void chunk(uint4 v, const uint8_t *row, int cols, int c, float *o);

template <> __device__ __forceinline__ void chunk<Q8_0>(uint4 v, const uint8_t *row, int cols, int c, float *o) {
    const float d = f16(row + cols + 2 * (c >> 1));
    i8x4(v.x, d, o);
    i8x4(v.y, d, o + 4);
    i8x4(v.z, d, o + 8);
    i8x4(v.w, d, o + 12);
}

__device__ __forceinline__ void nib8(uint32_t w, int j, float d, float m, float *o) {
#pragma unroll
    for (int b = 0; b < 4; b++) {
        const uint32_t q = (w >> (8 * b)) & 0xff;
        o[j + b] = (float) (q & 0xF) * d + m;
        o[j + b + 16] = (float) (q >> 4) * d + m;
    }
}

template <> __device__ __forceinline__ void chunk<Q4_0>(uint4 v, const uint8_t *row, int cols, int c, float *o) {
    const float d = f16(row + cols / 2 + 2 * c);
    nib8(v.x, 0, d, -8.f * d, o);
    nib8(v.y, 4, d, -8.f * d, o);
    nib8(v.z, 8, d, -8.f * d, o);
    nib8(v.w, 12, d, -8.f * d, o);
}

template <> __device__ __forceinline__ void chunk<Q4_1>(uint4 v, const uint8_t *row, int cols, int c, float *o) {
    const uint8_t *dm = row + cols / 2 + 4 * c;
    const float d = f16(dm), m = f16(dm + 2);
    nib8(v.x, 0, d, m, o);
    nib8(v.y, 4, d, m, o);
    nib8(v.z, 8, d, m, o);
    nib8(v.w, 12, d, m, o);
}

template <> __device__ __forceinline__ void deq32<Q4_K>(const uint8_t *row, int cols, int u, float *o) {
    const uint8_t *b = row + 144 * (u >> 3);
    const int sub = u & 7;
    int s, m;
    uint32_t w[3];
    float d, dmin;
    load_scales_k(b, w, d, dmin);
    scale_min_k4w(sub, w, s, m);
    const float d1 = d * (float) s, m1 = dmin * (float) m;
    const uint8_t *q = b + 16 + 32 * (sub >> 1);
    const int shift = 4 * (sub & 1);
#pragma unroll
    for (int l = 0; l < 32; l++) o[l] = d1 * (float) ((q[l] >> shift) & 0xF) - m1;
}

template <> __device__ __forceinline__ void deq32<Q5_K>(const uint8_t *row, int cols, int u, float *o) {
    const uint8_t *b = row + 176 * (u >> 3);
    const int sub = u & 7;
    int s, m;
    uint32_t w[3];
    float d, dmin;
    load_scales_k(b, w, d, dmin);
    scale_min_k4w(sub, w, s, m);
    const float d1 = d * (float) s, m1 = dmin * (float) m;
    const uint8_t *qh = b + 16;
    const uint8_t *q = b + 48 + 32 * (sub >> 1);
    const int shift = 4 * (sub & 1);
    const int hbit = 1 << sub;
#pragma unroll
    for (int l = 0; l < 32; l++) {
        const int v = ((q[l] >> shift) & 0xF) + ((qh[l] & hbit) ? 16 : 0);
        o[l] = d1 * (float) v - m1;
    }
}

template <> __device__ __forceinline__ void deq32<Q6_K>(const uint8_t *row, int cols, int u, float *o) {
    const uint8_t *b = row + 208 * (u >> 3);
    const int sub = u & 7;
    const int n = sub >> 2, g = sub & 3;
    const uint8_t *ql = b + 64 * n + 32 * (g & 1);
    const uint8_t *qh = b + 128 + 32 * n;
    const int8_t *sc = (const int8_t *) (b + 192 + 8 * n);
    const float d = f16(row + 208 * (cols / 256) + 2 * (u >> 3));
    const int lshift = 4 * (g >> 1), hshift = 2 * g;
#pragma unroll
    for (int l = 0; l < 32; l++) {
        const int q = ((ql[l] >> lshift) & 0xF) | (((qh[l] >> hshift) & 3) << 4);
        o[l] = d * (float) sc[l / 16 + 2 * g] * (float) (q - 32);
    }
}

// ---------------------------------------------------------------------------
// y[t][r] = dot(w.row(r), x[t]): a warp per row, the lanes over its units,
// NT input rows at a time (the weights read once for them)

#define NT 8
#define MM_ROWS 4 // warps (rows) per block

struct MatArgs {
    u64 w;              // first row
    u64 x;
    u64 y;
    long long row_bytes;
    int cols;
    int rows;
    int n;              // input rows
    int x_stride;       // floats from an input row to the next
    int y_stride;
    int pad;
    // mm_id: the tiles of pairs by expert, see moe_sort
    u64 tiles;
    u64 n_tiles;
    u64 order;
    long long expert_stride; // bytes from an expert's matrix to the next
    int x_div;          // input row of pair p: p / x_div
    int pad2;
    // mmq: the input rows quantized (quantize_q8), rows of `cols`
    u64 xq;
    u64 xds;
};

template <int T>
__device__ __forceinline__ void mm_rows(const uint8_t *wr, int cols, const float *const *xs, int nt,
                                        float *acc) {
    const int lane = threadIdx.x & 31;
    if constexpr (split_rows<T>()) {
        // 16 bytes of quants per lane, a warp's loads contiguous
        constexpr int E = chunk_values<T>();
        const int chunks = cols / E;
        const uint4 *q = (const uint4 *) wr;
#pragma unroll 2
        for (int c = lane; c < chunks; c += WARP) {
            float w[E];
            chunk<T>(q[c], wr, cols, c, w);
#pragma unroll
            for (int t = 0; t < NT; t++) {
                if (t < nt) {
                    const float4 *x = (const float4 *) (xs[t] + E * c);
                    float s = 0.f;
#pragma unroll
                    for (int i = 0; i < E / 4; i++) {
                        const float4 v = x[i];
                        s += w[4 * i] * v.x + w[4 * i + 1] * v.y + w[4 * i + 2] * v.z + w[4 * i + 3] * v.w;
                    }
                    acc[t] += s;
                }
            }
        }
        return;
    }
    const int units = cols / 32;
    for (int u = lane; u < units; u += WARP) {
        float w[32];
        deq32<T>(wr, cols, u, w);
#pragma unroll
        for (int t = 0; t < NT; t++) {
            if (t < nt) {
                const float4 *x = (const float4 *) (xs[t] + 32 * u);
                float s = 0.f;
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    const float4 v = x[i];
                    s += w[4 * i] * v.x + w[4 * i + 1] * v.y + w[4 * i + 2] * v.z + w[4 * i + 3] * v.w;
                }
                acc[t] += s;
            }
        }
    }
}


// ---------------------------------------------------------------------------
// The weights of a row against int8 activations (quantize_q8), as the CPU's
// dot kernels: per unit of 32, d * d8 * dot(q, q8) (+ m * s8), the integer
// dot products 4 at a time (dp4a). A lane takes a chunk of 16 bytes of
// quants: Wq<T>::load reads chunk c of a row, dot its product with a row of
// activations (q8 its int8, ds its (d8, s8) per 32).

__device__ __forceinline__ int dot16(uint4 w, uint4 x, int s) {
    s = __dp4a((int) w.x, (int) x.x, s);
    s = __dp4a((int) w.y, (int) x.y, s);
    s = __dp4a((int) w.z, (int) x.z, s);
    return __dp4a((int) w.w, (int) x.w, s);
}

__device__ __forceinline__ uint4 lo4(uint4 v) {
    return make_uint4(v.x & 0x0F0F0F0Fu, v.y & 0x0F0F0F0Fu, v.z & 0x0F0F0F0Fu, v.w & 0x0F0F0F0Fu);
}

__device__ __forceinline__ uint4 hi4(uint4 v) { return lo4(make_uint4(v.x >> 4, v.y >> 4, v.z >> 4, v.w >> 4)); }

// bit `b` of each byte of v, as bit 4
__device__ __forceinline__ uint4 bit4(uint4 v, int b) {
    return make_uint4(((v.x >> b) & 0x01010101u) << 4, ((v.y >> b) & 0x01010101u) << 4,
                      ((v.z >> b) & 0x01010101u) << 4, ((v.w >> b) & 0x01010101u) << 4);
}

__device__ __forceinline__ uint4 or4(uint4 a, uint4 b) { return make_uint4(a.x | b.x, a.y | b.y, a.z | b.z, a.w | b.w); }

template <int T> struct Wq;

// chunk c: values [16c, 16c + 16), half of unit c / 2
template <> struct Wq<Q8_0> {
    static constexpr int VALUES = 16;
    uint4 q;
    float d;
    int c;
    __device__ __forceinline__ void load(const uint8_t *row, int cols, int c_) {
        c = c_;
        q = ((const uint4 *) row)[c];
        d = f16(row + cols + 2 * (c >> 1));
    }
    __device__ __forceinline__ float dot(const int8_t *q8, const float2 *ds) const {
        const int s = dot16(q, *(const uint4 *) (q8 + 16 * c), 0);
        return d * ds[c >> 1].x * (float) s;
    }
};

// chunk c: unit c, the low nibbles values [32c, 32c + 16), the high ones the
// next 16
template <> struct Wq<Q4_0> {
    static constexpr int VALUES = 32;
    uint4 q;
    float d;
    int c;
    __device__ __forceinline__ void load(const uint8_t *row, int cols, int c_) {
        c = c_;
        q = ((const uint4 *) row)[c];
        d = f16(row + cols / 2 + 2 * c);
    }
    __device__ __forceinline__ float dot(const int8_t *q8, const float2 *ds) const {
        const uint4 *x = (const uint4 *) (q8 + 32 * c);
        const int s = dot16(hi4(q), x[1], dot16(lo4(q), x[0], 0));
        const float2 xd = ds[c];
        return d * (xd.x * (float) s - 8.f * xd.y);
    }
};

template <> struct Wq<Q4_1> {
    static constexpr int VALUES = 32;
    uint4 q;
    float d, m;
    int c;
    __device__ __forceinline__ void load(const uint8_t *row, int cols, int c_) {
        c = c_;
        q = ((const uint4 *) row)[c];
        const uint8_t *dm = row + cols / 2 + 4 * c;
        d = f16(dm);
        m = f16(dm + 2);
    }
    __device__ __forceinline__ float dot(const int8_t *q8, const float2 *ds) const {
        const uint4 *x = (const uint4 *) (q8 + 32 * c);
        const int s = dot16(hi4(q), x[1], dot16(lo4(q), x[0], 0));
        const float2 xd = ds[c];
        return d * xd.x * (float) s + m * xd.y;
    }
};

// K-quants: chunk c of block c / 8: 16 bytes of the 32 of quants of sub-blocks
// (2j, 2j + 1), j = (c % 8) / 2, half hh = c % 2: values 16 hh.. of each. The
// lane of the first half adds the sub-blocks' min terms.
template <int T> struct WqK {
    static constexpr int VALUES = 32;
    uint4 lo, hi;
    float f0, f1, g0, g1;
    int u0, off;
    __device__ __forceinline__ void load(const uint8_t *row, int cols, int c) {
        constexpr int BB = T == Q4_K ? 144 : 176;
        const uint8_t *b = row + BB * (c >> 3);
        const int p = c & 7, j = p >> 1, hh = p & 1;
        const uint4 q = *(const uint4 *) (b + (T == Q4_K ? 16 : 48) + 16 * p);
        lo = lo4(q);
        hi = hi4(q);
        if (T == Q5_K) {
            const uint4 qh = *(const uint4 *) (b + 16 + 16 * hh);
            lo = or4(lo, bit4(qh, 2 * j));
            hi = or4(hi, bit4(qh, 2 * j + 1));
        }
        int s0, m0, s1, m1;
        uint32_t w[3];
        float d, dmin;
        load_scales_k(b, w, d, dmin);
        scale_min_k4w(2 * j, w, s0, m0);
        scale_min_k4w(2 * j + 1, w, s1, m1);
        f0 = d * (float) s0;
        f1 = d * (float) s1;
        g0 = hh ? 0.f : dmin * (float) m0;
        g1 = hh ? 0.f : dmin * (float) m1;
        u0 = 8 * (c >> 3) + 2 * j;
        off = 32 * u0 + 16 * hh;
    }
    __device__ __forceinline__ float dot(const int8_t *q8, const float2 *ds) const {
        const int s0 = dot16(lo, *(const uint4 *) (q8 + off), 0);
        const int s1 = dot16(hi, *(const uint4 *) (q8 + off + 32), 0);
        const float2 x0 = ds[u0], x1 = ds[u0 + 1];
        return f0 * x0.x * (float) s0 + f1 * x1.x * (float) s1 - g0 * x0.y - g1 * x1.y;
    }
};
template <> struct Wq<Q4_K> : WqK<Q4_K> {};
template <> struct Wq<Q5_K> : WqK<Q5_K> {};

// Q6_K: chunk c of block b = c / 8, half n = (c / 4) % 2, k = c % 4: 16 bytes of
// ql at 64 n + 16 k (the low nibbles values 128 n + 32 (k / 2) + 16 (k % 2)..,
// the high ones 64 further), the qh bits 2 (k / 2) and 4 + 2 (k / 2)
template <> struct Wq<Q6_K> {
    static constexpr int VALUES = 32;
    uint4 lo, hi;
    float f0, f1;
    int e;
    __device__ __forceinline__ void load(const uint8_t *row, int cols, int c) {
        const int b = c >> 3, n = (c >> 2) & 1, k = c & 3;
        const uint8_t *blk = row + 208 * b;
        const uint4 ql = *(const uint4 *) (blk + 64 * n + 16 * k);
        const uint4 qh = *(const uint4 *) (blk + 128 + 32 * n + 16 * (k & 1));
        const int hs = 2 * (k >> 1);
        const uint32_t *l = &ql.x, *h = &qh.x;
        uint32_t a[4], z[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            a[i] = __vsub4((l[i] & 0x0F0F0F0Fu) | (((h[i] >> hs) & 0x03030303u) << 4), 0x20202020u);
            z[i] = __vsub4(((l[i] >> 4) & 0x0F0F0F0Fu) | (((h[i] >> (hs + 4)) & 0x03030303u) << 4), 0x20202020u);
        }
        lo = make_uint4(a[0], a[1], a[2], a[3]);
        hi = make_uint4(z[0], z[1], z[2], z[3]);
        const uint2 scw = *(const uint2 *) (blk + 192 + 8 * n);
        const int8_t *sc = (const int8_t *) &scw;
        const float d = f16(row + 208 * (cols / 256) + 2 * b);
        f0 = d * (float) sc[k];
        f1 = d * (float) sc[k + 4];
        e = 256 * b + 128 * n + 32 * (k >> 1) + 16 * (k & 1);
    }
    __device__ __forceinline__ float dot(const int8_t *q8, const float2 *ds) const {
        const int s0 = dot16(lo, *(const uint4 *) (q8 + e), 0);
        const int s1 = dot16(hi, *(const uint4 *) (q8 + e + 64), 0);
        return f0 * ds[e >> 5].x * (float) s0 + f1 * ds[(e >> 5) + 2].x * (float) s1;
    }
};

// acc[t] += the lane's part of dot(w row, input row t), the input rows
// `xr` of `a`: int8 for the quantized types, else floats
// acc[m][t] += the lane's part of dot(row wr[m], int8 input row xr[t]), for
// M rows of one width. A lane loads U chunks of each row before their dot
// products, the loads unconditional (past the end, the last chunk again):
// host memory takes several loads in flight per warp to approach the bus's
// speed (the experts' matvecs, which read it), device memory does better
// with one.
template <int T, int M, int U>
__device__ __forceinline__ void rows_dot_q8(const MatArgs &a, const uint8_t *const *wr, const int *xr, int nt,
                                            float (*acc)[NT]) {
    const int lane = threadIdx.x & 31;
    const int8_t *q8[NT];
    const float2 *ds[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) {
        q8[t] = (const int8_t *) a.xq + (size_t) xr[t] * a.cols;
        ds[t] = (const float2 *) a.xds + (size_t) xr[t] * (a.cols / 32);
    }
    const int chunks = a.cols / Wq<T>::VALUES;
    for (int c0 = lane; c0 < chunks; c0 += WARP * U) {
        Wq<T> w[M][U];
#pragma unroll
        for (int u = 0; u < U; u++)
#pragma unroll
            for (int m = 0; m < M; m++) w[m][u].load(wr[m], a.cols, min(c0 + WARP * u, chunks - 1));
#pragma unroll
        for (int u = 0; u < U; u++) {
            if (c0 + WARP * u < chunks) {
#pragma unroll
                for (int m = 0; m < M; m++)
#pragma unroll
                    for (int t = 0; t < NT; t++)
                        if (t < nt) acc[m][t] += w[m][u].dot(q8[t], ds[t]);
            }
        }
    }
}

template <int T, int U = 1>
__device__ __forceinline__ void rows_dot(const MatArgs &a, const uint8_t *wr, const int *xr, int nt, float *acc) {
    if constexpr (q8_act<T>()) {
        rows_dot_q8<T, 1, U>(a, &wr, xr, nt, (float (*)[NT]) acc);
    } else {
        const float *xs[NT];
#pragma unroll
        for (int t = 0; t < NT; t++) xs[t] = (const float *) a.x + (size_t) xr[t] * a.x_stride;
        mm_rows<T>(wr, a.cols, xs, nt, acc);
    }
}

template <int T>
__device__ __forceinline__ void mm_body(const MatArgs &a, int bx) {
    const int row = bx * MM_ROWS + (threadIdx.x >> 5);
    if (row >= a.rows) return;
    const int t0 = blockIdx.y * NT;
    const int nt = min(NT, a.n - t0);
    int xr[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) xr[t] = t0 + min(t, nt - 1);
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) acc[t] = 0.f;
    rows_dot<T>(a, (const uint8_t *) a.w + row * a.row_bytes, xr, nt, acc);
    float *y = (float *) a.y;
#pragma unroll
    for (int t = 0; t < NT; t++) {
        if (t < nt) {
            const float v = warp_sum(acc[t]);
            if ((threadIdx.x & 31) == 0) y[(size_t) (t0 + t) * a.y_stride + row] = v;
        }
    }
}

// Matrices of one input in one launch: blocks [end[i-1], end[i]) are m[i]'s
#define GROUP 4
struct GroupArgs { MatArgs m[GROUP]; int end[GROUP]; int count; int pad; };

// silu(gate x) * up x into gate's output (gate and up of the same shape)
template <int T>
__device__ __forceinline__ void mm_glu_body(const MatArgs &g, const MatArgs &u) {
    const int row = blockIdx.x * MM_ROWS + (threadIdx.x >> 5);
    if (row >= g.rows) return;
    const int t0 = blockIdx.y * NT;
    const int nt = min(NT, g.n - t0);
    int xr[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) xr[t] = t0 + min(t, nt - 1);
    float acc[2][NT];
#pragma unroll
    for (int t = 0; t < NT; t++) acc[0][t] = acc[1][t] = 0.f;
    float *ag = acc[0], *au = acc[1];
    const uint8_t *wr[2] = {(const uint8_t *) g.w + row * g.row_bytes, (const uint8_t *) u.w + row * u.row_bytes};
    if constexpr (q8_act<T>()) {
        rows_dot_q8<T, 2, 1>(g, wr, xr, nt, acc);
    } else {
        rows_dot<T>(g, wr[0], xr, nt, ag);
        rows_dot<T>(u, wr[1], xr, nt, au);
    }
    float *y = (float *) g.y;
#pragma unroll
    for (int t = 0; t < NT; t++) {
        if (t < nt) {
            const float a = warp_sum(ag[t]), b = warp_sum(au[t]);
            if ((threadIdx.x & 31) == 0) y[(size_t) (t0 + t) * g.y_stride + row] = silu(a) * b;
        }
    }
}

struct Tile { int expert; int start; int count; int pad; };

// The pairs of one tile (all of one expert) through the expert's matrix:
// input row order[i] / x_div, output row order[i]
template <int T>
__device__ __forceinline__ void mm_id_body(const MatArgs &a) {
    if ((int) blockIdx.y >= *(const int *) a.n_tiles) return;
    const int row = blockIdx.x * MM_ROWS + (threadIdx.x >> 5);
    if (row >= a.rows) return;
    const Tile tile = ((const Tile *) a.tiles)[blockIdx.y];
    const int *order = (const int *) a.order;
    const int nt = tile.count;
    int pair[NT], xr[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) {
        pair[t] = order[tile.start + min(t, nt - 1)];
        xr[t] = pair[t] / a.x_div;
    }
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) acc[t] = 0.f;
    const uint8_t *w = (const uint8_t *) a.w + tile.expert * a.expert_stride + row * a.row_bytes;
    rows_dot<T, 4>(a, w, xr, nt, acc);
    float *y = (float *) a.y;
#pragma unroll
    for (int t = 0; t < NT; t++) {
        if (t < nt) {
            const float v = warp_sum(acc[t]);
            if ((threadIdx.x & 31) == 0) y[(size_t) pair[t] * a.y_stride + row] = v;
        }
    }
}

#define MM_KERNELS(name, T)                                                                      \
    extern "C" __global__ void __launch_bounds__(WARP * MM_ROWS) mm_##name(const MatArgs a) {    \
        mm_body<T>(a, blockIdx.x);                                                               \
    }                                                                                            \
    extern "C" __global__ void __launch_bounds__(WARP * MM_ROWS) mm_group_##name(const GroupArgs g) { \
        int i = 0;                                                                               \
        while (i + 1 < g.count && (int) blockIdx.x >= g.end[i]) i++;                             \
        mm_body<T>(g.m[i], blockIdx.x - (i ? g.end[i - 1] : 0));                                 \
    }                                                                                            \
    extern "C" __global__ void __launch_bounds__(WARP * MM_ROWS) mm_glu_##name(const GroupArgs g) { \
        mm_glu_body<T>(g.m[0], g.m[1]);                                                          \
    }                                                                                            \
    extern "C" __global__ void __launch_bounds__(WARP * MM_ROWS) mm_id_##name(const MatArgs a) { \
        mm_id_body<T>(a);                                                                        \
    }

MM_KERNELS(f32, F32)
MM_KERNELS(f16, F16)
MM_KERNELS(bf16, BF16)
MM_KERNELS(q4_0, Q4_0)
MM_KERNELS(q4_1, Q4_1)
MM_KERNELS(q8_0, Q8_0)
MM_KERNELS(q4_K, Q4_K)
MM_KERNELS(q5_K, Q5_K)
MM_KERNELS(q6_K, Q6_K)

// ---------------------------------------------------------------------------
// The same products on the tensor cores, for passes of more than NT rows: a
// block computes TM weight rows by TN input rows, TK columns at a time, the
// weights dequantized and the inputs rounded to f16 in shared memory (as
// Metal's simdgroup tiles), f32 accumulators (mma.sync m16n8k16). 8 warps,
// each 32 weight rows by 32 input rows.

#define TM 128
#define TN 64
#define TK 64
#define TPAD (TK + 8) // halves per shared row: ldmatrix rows on distinct banks
#define MMA_THREADS 256

__device__ __forceinline__ void ldsm_x4(uint32_t *r, const __half *p) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}

__device__ __forceinline__ void mma16816(float *c, const uint32_t *a, const uint32_t *b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t pack_half2(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *(const uint32_t *) &h;
}

// 8 floats as 8 halves, 16 bytes
__device__ __forceinline__ uint4 pack8(const float *v) {
    return make_uint4(pack_half2(v[0], v[1]), pack_half2(v[2], v[3]), pack_half2(v[4], v[5]),
                      pack_half2(v[6], v[7]));
}

// mm (ID false): input rows blockIdx.y * TN.., output rows the same. mm_id
// (ID true): the pairs of tile blockIdx.y, input row order[i] / x_div,
// output row order[i].
template <int T, bool ID>
__device__ __forceinline__ void mma_body(const MatArgs &a) {
    __shared__ __align__(16) __half ws[TM * TPAD];
    __shared__ __align__(16) __half xs[TN * TPAD];
    __shared__ int orow[TN], irow[TN];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int r0 = blockIdx.x * TM;
    const uint8_t *w = (const uint8_t *) a.w;
    int nt;
    if (ID) {
        if ((int) blockIdx.y >= *(const int *) a.n_tiles) return;
        const Tile tile = ((const Tile *) a.tiles)[blockIdx.y];
        nt = tile.count;
        w += tile.expert * a.expert_stride;
        if (tid < TN) {
            const int p = ((const int *) a.order)[tile.start + min(tid, nt - 1)];
            orow[tid] = p;
            irow[tid] = p / a.x_div;
        }
    } else {
        const int t0 = blockIdx.y * TN;
        nt = min(TN, a.n - t0);
        if (tid < TN) {
            const int t = t0 + min(tid, nt - 1);
            orow[tid] = t;
            irow[tid] = t;
        }
    }
    __syncthreads();

    // loaders: a weight row's unit (2 threads per row), 16 columns of an input row
    const int lr = tid >> 1, lu = tid & 1;
    const uint8_t *wrow = w + (size_t) min(r0 + lr, a.rows - 1) * a.row_bytes;
    const int xc = tid >> 2, xo = 16 * (tid & 3);
    const float *xrow = (const float *) a.x + (size_t) irow[xc] * a.x_stride;
    const int units = a.cols / 32;
    const int wm = warp & 3, wn = warp >> 2;

    float acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[i][j][c] = 0.f;

    for (int k0 = 0; k0 < a.cols; k0 += TK) {
        uint4 *wd = (uint4 *) (ws + lr * TPAD + 32 * lu);
        const int u = k0 / 32 + lu;
        if (u < units) {
            float v[32];
            deq32<T>(wrow, a.cols, u, v);
#pragma unroll
            for (int i = 0; i < 4; i++) wd[i] = pack8(v + 8 * i);
        } else {
#pragma unroll
            for (int i = 0; i < 4; i++) wd[i] = make_uint4(0, 0, 0, 0);
        }
        uint4 *xd = (uint4 *) (xs + xc * TPAD + xo);
        if (k0 + xo < a.cols) {
            const float4 *src = (const float4 *) (xrow + k0 + xo);
            float v[16];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const float4 f = src[i];
                v[4 * i] = f.x; v[4 * i + 1] = f.y; v[4 * i + 2] = f.z; v[4 * i + 3] = f.w;
            }
            xd[0] = pack8(v);
            xd[1] = pack8(v + 8);
        } else {
            xd[0] = make_uint4(0, 0, 0, 0);
            xd[1] = make_uint4(0, 0, 0, 0);
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < TK; kk += 16) {
            uint32_t af[2][4], bf[4][2];
#pragma unroll
            for (int i = 0; i < 2; i++)
                ldsm_x4(af[i], ws + (32 * wm + 16 * i + (lane & 15)) * TPAD + kk + (lane >> 4) * 8);
#pragma unroll
            for (int j = 0; j < 2; j++) {
                uint32_t r[4];
                ldsm_x4(r, xs + (32 * wn + 16 * j + (lane & 7) + ((lane >> 4) & 1) * 8) * TPAD + kk +
                               ((lane >> 3) & 1) * 8);
                bf[2 * j][0] = r[0];
                bf[2 * j][1] = r[1];
                bf[2 * j + 1][0] = r[2];
                bf[2 * j + 1][1] = r[3];
            }
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) mma16816(acc[i][j], af[i], bf[j]);
        }
        __syncthreads();
    }

    float *y = (float *) a.y;
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int r = r0 + 32 * wm + 16 * i + (lane >> 2);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int c = 32 * wn + 8 * j + 2 * (lane & 3);
#pragma unroll
            for (int h = 0; h < 2; h++) {
                if (c + h < nt) {
                    float *yr = y + (size_t) orow[c + h] * a.y_stride;
                    if (r < a.rows) yr[r] = acc[i][j][h];
                    if (r + 8 < a.rows) yr[r + 8] = acc[i][j][2 + h];
                }
            }
        }
    }
}

#define MMA_KERNELS(name, T)                                                                       \
    extern "C" __global__ void __launch_bounds__(MMA_THREADS) mma_##name(const MatArgs a) {        \
        mma_body<T, false>(a);                                                                     \
    }                                                                                              \
    extern "C" __global__ void __launch_bounds__(MMA_THREADS) mma_id_##name(const MatArgs a) {     \
        mma_body<T, true>(a);                                                                      \
    }

MMA_KERNELS(f32, F32)
MMA_KERNELS(f16, F16)
MMA_KERNELS(bf16, BF16)
MMA_KERNELS(q4_0, Q4_0)
MMA_KERNELS(q4_1, Q4_1)
MMA_KERNELS(q8_0, Q8_0)
MMA_KERNELS(q4_K, Q4_K)
MMA_KERNELS(q5_K, Q5_K)
MMA_KERNELS(q6_K, Q6_K)

// ---------------------------------------------------------------------------
// Int8 activations, as the CPU's Q8Act: per 32 values, d = amax / 127, the
// values rounded to multiples of d, and s = d * sum(q). Rows of `cols`: q as
// int8, (d, s) as float2 per 32. One warp per 32 values.

struct QuantArgs { u64 x; u64 q; u64 ds; int cols; int pad; };

extern "C" __global__ void quantize_q8(const QuantArgs a) {
    const int blocks = a.cols / 32;
    const int b = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (b >= blocks) return;
    const size_t i = (size_t) blockIdx.y * a.cols + 32 * b + lane;
    const float v = ((const float *) a.x)[i];
    const float amax = warp_max(fabsf(v));
    const float d = amax / 127.f;
    const float id = d != 0.f ? 1.f / d : 0.f;
    const int q = (int) roundf(v * id);
    ((int8_t *) a.q)[i] = (int8_t) q;
    int sum = q;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (lane == 0) ((float2 *) a.ds)[(size_t) blockIdx.y * blocks + b] = make_float2(d, d * (float) sum);
}

// Unit u of a weight row as 32 int8 (8 words) q, its scale d and m: the unit's
// dot product with activations (q8, d8, s8) is d * d8 * dot(q, q8) + m * s8,
// as quant.rs's scalar dot kernels compute it.
template <int T> __device__ __forceinline__ void q8unit(const uint8_t *row, int cols, int u, uint32_t *q, float &d, float &m);

template <> __device__ __forceinline__ void q8unit<Q8_0>(const uint8_t *row, int cols, int u, uint32_t *q, float &d, float &m) {
    const uint4 *b = (const uint4 *) (row + 32 * u);
    d = f16(row + cols + 2 * u);
    m = 0.f;
    const uint4 v0 = b[0], v1 = b[1];
    q[0] = v0.x; q[1] = v0.y; q[2] = v0.z; q[3] = v0.w;
    q[4] = v1.x; q[5] = v1.y; q[6] = v1.z; q[7] = v1.w;
}

template <> __device__ __forceinline__ void q8unit<Q4_0>(const uint8_t *row, int cols, int u, uint32_t *q, float &d, float &m) {
    const uint4 v = *(const uint4 *) (row + 16 * u);
    d = f16(row + cols / 2 + 2 * u);
    m = 0.f;
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; i++) {
        q[i] = __vsub4(w[i] & 0x0F0F0F0Fu, 0x08080808u);
        q[4 + i] = __vsub4((w[i] >> 4) & 0x0F0F0F0Fu, 0x08080808u);
    }
}

template <> __device__ __forceinline__ void q8unit<Q4_1>(const uint8_t *row, int cols, int u, uint32_t *q, float &d, float &m) {
    const uint4 v = *(const uint4 *) (row + 16 * u);
    const uint8_t *dm = row + cols / 2 + 4 * u;
    d = f16(dm);
    m = f16(dm + 2);
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; i++) {
        q[i] = w[i] & 0x0F0F0F0Fu;
        q[4 + i] = (w[i] >> 4) & 0x0F0F0F0Fu;
    }
}

template <> __device__ __forceinline__ void q8unit<Q4_K>(const uint8_t *row, int cols, int u, uint32_t *q, float &d, float &m) {
    const uint8_t *b = row + 144 * (u >> 3);
    const int sub = u & 7;
    int sc, mn;
    uint32_t sw[3];
    float bd, bmin;
    load_scales_k(b, sw, bd, bmin);
    scale_min_k4w(sub, sw, sc, mn);
    d = bd * (float) sc;
    m = -bmin * (float) mn;
    const uint32_t *w = (const uint32_t *) (b + 16 + 32 * (sub >> 1));
    const int shift = 4 * (sub & 1);
#pragma unroll
    for (int i = 0; i < 8; i++) q[i] = (w[i] >> shift) & 0x0F0F0F0Fu;
}

template <> __device__ __forceinline__ void q8unit<Q5_K>(const uint8_t *row, int cols, int u, uint32_t *q, float &d, float &m) {
    const uint8_t *b = row + 176 * (u >> 3);
    const int sub = u & 7;
    int sc, mn;
    uint32_t sw[3];
    float bd, bmin;
    load_scales_k(b, sw, bd, bmin);
    scale_min_k4w(sub, sw, sc, mn);
    d = bd * (float) sc;
    m = -bmin * (float) mn;
    const uint32_t *qh = (const uint32_t *) (b + 16);
    const uint32_t *w = (const uint32_t *) (b + 48 + 32 * (sub >> 1));
    const int shift = 4 * (sub & 1);
#pragma unroll
    for (int i = 0; i < 8; i++) q[i] = ((w[i] >> shift) & 0x0F0F0F0Fu) | (((qh[i] >> sub) & 0x01010101u) << 4);
}

// The int8 tensor cores (mma.sync m16n8k32): a block computes TM weight rows
// by TN input rows, QTK columns at a time; per unit of 32 columns, the int32
// products scaled as the CPU scales them, into f32 accumulators. 8 warps,
// each 32 weight rows by 32 input rows.

#define QTK 128
#define QPAD (QTK + 16) // bytes per shared row: ldmatrix rows on distinct banks
#define QU (QTK / 32)

// An int32 of magnitude below 2^22 as a float, exactly: the integer added to
// the bits of 1.5 * 2^23, which takes 2 full-rate instructions where cvt takes
// one at an eighth of the rate. A unit's product is at most 32 * 127 * 128.
__device__ __forceinline__ float small_i2f(int v) {
    return __int_as_float(v + 0x4B400000) - 12582912.f;
}

__device__ __forceinline__ void imma16832(int *c, const uint32_t *a, const uint32_t *b) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void ldsm_x4_b(uint32_t *r, const int8_t *p) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}

template <int T, bool ID>
__device__ __forceinline__ void mmq_body(const MatArgs &a) {
    constexpr bool MIN = T == Q4_1 || T == Q4_K || T == Q5_K;
    __shared__ __align__(16) int8_t ws[TM * QPAD];
    __shared__ __align__(16) int8_t xs[TN * QPAD];
    __shared__ float wd[TM][QU], wmn[TM][QU];
    __shared__ float2 xds[TN][QU];
    __shared__ int orow[TN], irow[TN];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int r0 = blockIdx.x * TM;
    const uint8_t *w = (const uint8_t *) a.w;
    int nt;
    if (ID) {
        if ((int) blockIdx.y >= *(const int *) a.n_tiles) return;
        const Tile tile = ((const Tile *) a.tiles)[blockIdx.y];
        nt = tile.count;
        w += tile.expert * a.expert_stride;
        if (tid < TN) {
            const int p = ((const int *) a.order)[tile.start + min(tid, nt - 1)];
            orow[tid] = p;
            irow[tid] = p / a.x_div;
        }
    } else {
        const int t0 = blockIdx.y * TN;
        nt = min(TN, a.n - t0);
        if (tid < TN) {
            const int t = t0 + min(tid, nt - 1);
            orow[tid] = t;
            irow[tid] = t;
        }
    }
    __syncthreads();

    // loaders: 2 units of a weight row (2 threads per row), 32 bytes of an
    // input row and the scales of a unit
    const int lr = tid >> 1, lu = 2 * (tid & 1);
    const uint8_t *wrow = w + (size_t) min(r0 + lr, a.rows - 1) * a.row_bytes;
    const int xc = tid >> 2, xu = tid & 3;
    const int units = a.cols / 32;
    const int8_t *xqrow = (const int8_t *) a.xq + (size_t) irow[xc] * a.cols;
    const float2 *xdsrow = (const float2 *) a.xds + (size_t) irow[xc] * units;
    const int wm = warp & 3, wn = warp >> 2;
    const int g = lane >> 2, tq = lane & 3;

    float acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[i][j][c] = 0.f;

    for (int k0 = 0; k0 < a.cols; k0 += QTK) {
        const int ub = k0 / 32;
#pragma unroll
        for (int uu = 0; uu < 2; uu++) {
            const int u = ub + lu + uu;
            uint32_t q[8];
            float d = 0.f, m = 0.f;
            if (u < units) {
                q8unit<T>(wrow, a.cols, u, q, d, m);
            } else {
#pragma unroll
                for (int i = 0; i < 8; i++) q[i] = 0;
            }
            uint4 *dst = (uint4 *) (ws + lr * QPAD + 32 * (lu + uu));
            dst[0] = make_uint4(q[0], q[1], q[2], q[3]);
            dst[1] = make_uint4(q[4], q[5], q[6], q[7]);
            wd[lr][lu + uu] = d;
            if (MIN) wmn[lr][lu + uu] = m;
        }
        {
            const int u = ub + xu;
            uint4 *dst = (uint4 *) (xs + xc * QPAD + 32 * xu);
            if (u < units) {
                const uint4 *src = (const uint4 *) (xqrow + 32 * u);
                dst[0] = src[0];
                dst[1] = src[1];
                xds[xc][xu] = xdsrow[u];
            } else {
                dst[0] = make_uint4(0, 0, 0, 0);
                dst[1] = make_uint4(0, 0, 0, 0);
                xds[xc][xu] = make_float2(0.f, 0.f);
            }
        }
        __syncthreads();
#pragma unroll
        for (int uu = 0; uu < QU; uu++) {
            uint32_t af[2][4], bf[4][2];
#pragma unroll
            for (int i = 0; i < 2; i++)
                ldsm_x4_b(af[i], ws + (32 * wm + 16 * i + (lane & 15)) * QPAD + 32 * uu + (lane >> 4) * 16);
#pragma unroll
            for (int j = 0; j < 2; j++) {
                uint32_t r[4];
                ldsm_x4_b(r, xs + (32 * wn + 16 * j + (lane & 7) + ((lane >> 4) & 1) * 8) * QPAD + 32 * uu +
                                 ((lane >> 3) & 1) * 16);
                bf[2 * j][0] = r[0];
                bf[2 * j][1] = r[1];
                bf[2 * j + 1][0] = r[2];
                bf[2 * j + 1][1] = r[3];
            }
            float rd[2][2], rm[2][2];
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int r = 32 * wm + 16 * i + g + 8 * h;
                    rd[i][h] = wd[r][uu];
                    rm[i][h] = MIN ? wmn[r][uu] : 0.f;
                }
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int c = 32 * wn + 8 * j + 2 * tq;
                const float2 x0 = xds[c][uu], x1 = xds[c + 1][uu];
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    int ci[4] = {0, 0, 0, 0};
                    imma16832(ci, af[i], bf[j]);
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        float &a0 = acc[i][j][2 * h], &a1 = acc[i][j][2 * h + 1];
                        a0 += rd[i][h] * x0.x * small_i2f(ci[2 * h]);
                        a1 += rd[i][h] * x1.x * small_i2f(ci[2 * h + 1]);
                        if (MIN) {
                            a0 += rm[i][h] * x0.y;
                            a1 += rm[i][h] * x1.y;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    float *y = (float *) a.y;
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int r = r0 + 32 * wm + 16 * i + g;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int c = 32 * wn + 8 * j + 2 * tq;
#pragma unroll
            for (int h = 0; h < 2; h++) {
                if (c + h < nt) {
                    float *yr = y + (size_t) orow[c + h] * a.y_stride;
                    if (r < a.rows) yr[r] = acc[i][j][h];
                    if (r + 8 < a.rows) yr[r + 8] = acc[i][j][2 + h];
                }
            }
        }
    }
}

#define MMQ_KERNELS(name, T)                                                                       \
    extern "C" __global__ void __launch_bounds__(MMA_THREADS) mmq_##name(const MatArgs a) {        \
        mmq_body<T, false>(a);                                                                     \
    }                                                                                              \
    extern "C" __global__ void __launch_bounds__(MMA_THREADS) mmq_id_##name(const MatArgs a) {     \
        mmq_body<T, true>(a);                                                                      \
    }

MMQ_KERNELS(q4_0, Q4_0)
MMQ_KERNELS(q4_1, Q4_1)
MMQ_KERNELS(q8_0, Q8_0)
MMQ_KERNELS(q4_K, Q4_K)
MMQ_KERNELS(q5_K, Q5_K)

// ---------------------------------------------------------------------------
// elementwise

struct EltArgs { u64 a; u64 b; int n; int pad; };

// a += b
extern "C" __global__ void add_inplace(const EltArgs e) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < e.n) ((float *) e.a)[i] += ((const float *) e.b)[i];
}

// a = silu(a) * b
extern "C" __global__ void swiglu(const EltArgs e) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < e.n) {
        float *a = (float *) e.a;
        a[i] = silu(a[i]) * ((const float *) e.b)[i];
    }
}

// ---------------------------------------------------------------------------
// rms norm of rows of n, times w, optionally times silu(gate) (rows of n
// too); output rows ostride apart. With `add`, first x += y (the residual
// stream). One block per row, NORM_THREADS threads, the row in registers up
// to NORM_MAX values a thread.

struct NormArgs { u64 x; u64 w; u64 out; u64 gate; u64 y; int n; int ostride; int gated; int add; float eps; int pad; };

#define NORM_MAX 32

extern "C" __global__ void rms_norm(const NormArgs a) {
    const int row = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    float *x = (float *) a.x + (size_t) row * a.n;
    const float *y = (const float *) a.y + (size_t) row * a.n;
    const float *w = (const float *) a.w;
    const float *g = (const float *) a.gate + (size_t) row * a.n;
    float *out = (float *) a.out + (size_t) row * a.ostride;
    float ss = 0.f;
    if (a.n <= nt * NORM_MAX) {
        float v[NORM_MAX];
#pragma unroll
        for (int j = 0; j < NORM_MAX; j++) {
            const int i = tid + nt * j;
            if (i < a.n) {
                float e = x[i];
                if (a.add) e += y[i];
                v[j] = e;
                ss += e * e;
            }
        }
        if (a.add) {
#pragma unroll
            for (int j = 0; j < NORM_MAX; j++) {
                const int i = tid + nt * j;
                if (i < a.n) x[i] = v[j];
            }
        }
        ss = block_sum(ss);
        const float scale = 1.f / sqrtf(ss / (float) a.n + a.eps);
#pragma unroll
        for (int j = 0; j < NORM_MAX; j++) {
            const int i = tid + nt * j;
            if (i < a.n) {
                float e = v[j] * (scale * w[i]);
                if (a.gated) e *= silu(g[i]);
                out[i] = e;
            }
        }
        return;
    }
    for (int i = tid; i < a.n; i += nt) {
        float e = x[i];
        if (a.add) {
            e += y[i];
            x[i] = e;
        }
        ss += e * e;
    }
    ss = block_sum(ss);
    const float scale = 1.f / sqrtf(ss / (float) a.n + a.eps);
    for (int i = tid; i < a.n; i += nt) {
        float e = x[i] * (scale * w[i]);
        if (a.gated) e *= silu(g[i]);
        out[i] = e;
    }
}

// ---------------------------------------------------------------------------
// What changes from a pass of a session to the next of the same shape, in
// device memory: a pass replays as a CUDA graph

struct PassParams {
    int pos0; // position of the pass's first token
    int slot; // of the delta net states, see ssm_conv
    int pad[2];
};

// ---------------------------------------------------------------------------
// attention

struct AttnArgs {
    u64 qg;       // projected queries, followed by their output gate per head when gated
    u64 k;        // projected keys and values of the pass
    u64 v;
    u64 q_norm;
    u64 k_norm;
    u64 inv_freq;
    u64 q;        // normalized, rotated queries
    u64 k_cache;  // f16, position p at row p & mask
    u64 v_cache;
    u64 o;
    u64 sinks;
    u64 params;   // PassParams
    int n_head;
    int n_kv;
    int hd;
    int n_rot;    // rotated dims; 0: no rope
    int n;        // tokens of the pass
    int gated;
    int window;   // positions a token sees, its own included; 0: all
    int mask;     // -1, or the rows of the ring - 1
    int has_sinks;
    float eps;
    float scale;
};

// Per (head, token), one warp: rms norm of q or k, NeoX rope on the first
// n_rot dims, then q to `q`, or k and v to the caches at the token's position.
// Heads [0, n_head) are queries, [n_head, n_head + n_kv) keys. Lane l holds
// dims l + 32j, so rope pairs (i, i + n_rot/2) stay in a lane when n_rot/2 is
// a multiple of 32.
extern "C" __global__ void attn_prep(const AttnArgs a) {
    constexpr int MAXE = 16; // head_dim up to 512
    const int head = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    const int hd = a.hd, E = hd / 32, kv_dim = a.n_kv * hd;
    const int pos = ((const PassParams *) a.params)->pos0 + t;
    const bool is_q = head < a.n_head;
    const int kh = head - a.n_head;
    const float *src = is_q ? (const float *) a.qg + (size_t) (t * a.n_head + head) * (a.gated ? 2 : 1) * hd
                            : (const float *) a.k + (size_t) t * kv_dim + kh * hd;
    const float *w = (const float *) (is_q ? a.q_norm : a.k_norm);

    float vals[MAXE];
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < MAXE; j++) {
        if (j < E) {
            vals[j] = src[lane + 32 * j];
            ss += vals[j] * vals[j];
        }
    }
    ss = warp_sum(ss);
    const float scale = 1.f / sqrtf(ss / (float) hd + a.eps);
#pragma unroll
    for (int j = 0; j < MAXE; j++) {
        if (j < E) vals[j] *= scale * w[lane + 32 * j];
    }

    const int m = a.n_rot / 64; // rotated pairs per lane
    const float *inv_freq = (const float *) a.inv_freq;
#pragma unroll
    for (int j = 0; j < MAXE / 2; j++) {
        if (j < m) {
            const int i = lane + 32 * j;
            float s, c;
            sincosf((float) pos * inv_freq[i], &s, &c);
            const float x0 = vals[j], x1 = vals[j + m];
            vals[j] = x0 * c - x1 * s;
            vals[j + m] = x0 * s + x1 * c;
        }
    }

    if (is_q) {
        float *dst = (float *) a.q + (size_t) (t * a.n_head + head) * hd;
#pragma unroll
        for (int j = 0; j < MAXE; j++) {
            if (j < E) dst[lane + 32 * j] = vals[j];
        }
    } else {
        const size_t row = (size_t) (pos & a.mask) * kv_dim + kh * hd;
        __half *kd = (__half *) a.k_cache + row;
        __half *vd = (__half *) a.v_cache + row;
        const float *vs = (const float *) a.v + (size_t) t * kv_dim + kh * hd;
#pragma unroll
        for (int j = 0; j < MAXE; j++) {
            if (j < E) {
                kd[lane + 32 * j] = __float2half(vals[j]);
                vd[lane + 32 * j] = __float2half(vs[lane + 32 * j]);
            }
        }
    }
}

#define ATTN_WARPS 8

// Causal attention, one block per (query head, token): each warp an online
// softmax over every ATTN_WARPS-th position the token sees (lane l holding
// dims l + 32j), then the warps merged, with the head's sink logit in the
// denominator, and the output gated by sigmoid(gate) when gated.
extern "C" __global__ void __launch_bounds__(WARP * ATTN_WARPS) attn(const AttnArgs a) {
    constexpr int MAXE = 16;
    __shared__ float sm[ATTN_WARPS], sl[ATTN_WARPS];
    extern __shared__ float sacc[]; // [ATTN_WARPS][hd]
    const int head = blockIdx.x, t = blockIdx.y;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int hd = a.hd, E = hd / 32, kv_dim = a.n_kv * hd;
    const int kvh = head / (a.n_head / a.n_kv);
    const int pos = ((const PassParams *) a.params)->pos0 + t;
    const int lo = a.window > 0 ? max(pos + 1 - a.window, 0) : 0;

    const float *qh = (const float *) a.q + (size_t) (t * a.n_head + head) * hd;
    float q[MAXE], acc[MAXE];
#pragma unroll
    for (int j = 0; j < MAXE; j++) {
        q[j] = j < E ? qh[lane + 32 * j] : 0.f;
        acc[j] = 0.f;
    }
    float m = -INFINITY, l = 0.f;
    const __half *kc = (const __half *) a.k_cache + kvh * hd + lane;
    const __half *vc = (const __half *) a.v_cache + kvh * hd + lane;
    for (int p = lo + w; p <= pos; p += ATTN_WARPS) {
        const size_t row = (size_t) (p & a.mask) * kv_dim;
        float s = 0.f;
#pragma unroll
        for (int j = 0; j < MAXE; j++) {
            if (j < E) s += q[j] * __half2float(kc[row + 32 * j]);
        }
        s = warp_sum(s) * a.scale;
        const float mn = fmaxf(m, s);
        const float corr = expf(m - mn), e = expf(s - mn);
        l = l * corr + e;
#pragma unroll
        for (int j = 0; j < MAXE; j++) {
            if (j < E) acc[j] = acc[j] * corr + e * __half2float(vc[row + 32 * j]);
        }
        m = mn;
    }
    if (lane == 0) {
        sm[w] = m;
        sl[w] = l;
    }
#pragma unroll
    for (int j = 0; j < MAXE; j++) {
        if (j < E) sacc[w * hd + lane + 32 * j] = acc[j];
    }
    __syncthreads();

    float M = a.has_sinks ? ((const float *) a.sinks)[head] : -INFINITY;
    for (int i = 0; i < ATTN_WARPS; i++) M = fmaxf(M, sm[i]);
    float L = a.has_sinks ? expf(((const float *) a.sinks)[head] - M) : 0.f;
    for (int i = 0; i < ATTN_WARPS; i++) L += sl[i] * expf(sm[i] - M);
    const float inv = 1.f / L;
    const float *gate = (const float *) a.qg + (size_t) (t * a.n_head + head) * 2 * hd + hd;
    float *o = (float *) a.o + (size_t) (t * a.n_head + head) * hd;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float v = 0.f;
        for (int i = 0; i < ATTN_WARPS; i++) v += sacc[i * hd + d] * expf(sm[i] - M);
        v *= inv;
        if (a.gated) v *= sigmoid(gate[d]);
        o[d] = v;
    }
}

// Flash attention of a pass of more than a few tokens, on the f16 tensor
// cores: a block of FA_WARPS warps takes FA_BQ query tokens of a head (16 a
// warp), the keys FA_BK at a time through shared memory: S = Q K^T, the
// online softmax (log2 domain) in registers, O += P V. The queries round to
// f16; the sink logit joins the denominator at the end.

#define FA_BQ 64
#define FA_BK 32
#define FA_WARPS 4

__device__ __forceinline__ void ldsm_x4_t(uint32_t *r, const __half *p) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}

template <int D>
__device__ __forceinline__ void fattn_body(const AttnArgs &a) {
    constexpr int P = D + 8; // halves per shared row
    extern __shared__ __align__(16) __half fsm[];
    __half *qs = fsm, *ks = qs + FA_BQ * P, *vs = ks + FA_BK * P;
    const int head = blockIdx.x, t0 = blockIdx.y * FA_BQ;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int pos0 = ((const PassParams *) a.params)->pos0;
    const int kvh = head / (a.n_head / a.n_kv);
    const int kv_dim = a.n_kv * D;
    const int nq = min(FA_BQ, a.n - t0);
    const float sl2 = a.scale * 1.4426950408889634f; // scores in log2 units

    for (int i = tid; i < FA_BQ * D / 4; i += 32 * FA_WARPS) {
        const int r = i / (D / 4), c = 4 * (i % (D / 4));
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (r < nq) v = *(const float4 *) ((const float *) a.q + ((size_t) (t0 + r) * a.n_head + head) * D + c);
        *(uint2 *) (qs + r * P + c) = make_uint2(pack_half2(v.x, v.y), pack_half2(v.z, v.w));
    }

    // this lane's rows of the warp's 16: g and g + 8
    const int g = lane >> 2, tq = lane & 3;
    int rpos[2], rlo[2];
#pragma unroll
    for (int h = 0; h < 2; h++) {
        rpos[h] = pos0 + t0 + 16 * warp + g + 8 * h;
        rlo[h] = a.window > 0 ? rpos[h] + 1 - a.window : 0;
    }
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    float o[D / 8][4];
#pragma unroll
    for (int j = 0; j < D / 8; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;

    const int last = pos0 + t0 + nq - 1;
    const int first = a.window > 0 ? max(pos0 + t0 + 1 - a.window, 0) : 0;
    const __half *kc = (const __half *) a.k_cache + kvh * D, *vc = (const __half *) a.v_cache + kvh * D;
    for (int p0 = first; p0 <= last; p0 += FA_BK) {
        __syncthreads();
        for (int i = tid; i < FA_BK * D / 8; i += 32 * FA_WARPS) {
            const int r = i / (D / 8), c = 8 * (i % (D / 8));
            const int p = p0 + r;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = kv;
            if (p <= last) {
                const size_t row = (size_t) (p & a.mask) * kv_dim + c;
                kv = *(const uint4 *) (kc + row);
                vv = *(const uint4 *) (vc + row);
            }
            *(uint4 *) (ks + r * P + c) = kv;
            *(uint4 *) (vs + r * P + c) = vv;
        }
        __syncthreads();

        float sc[4][4];
#pragma unroll
        for (int j = 0; j < 4; j++) sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.f;
#pragma unroll
        for (int kk = 0; kk < D; kk += 16) {
            uint32_t af[4], bf[4][2];
            ldsm_x4(af, qs + (16 * warp + (lane & 15)) * P + kk + (lane >> 4) * 8);
#pragma unroll
            for (int j = 0; j < 2; j++) {
                uint32_t r[4];
                ldsm_x4(r, ks + (16 * j + (lane & 7) + ((lane >> 4) & 1) * 8) * P + kk + ((lane >> 3) & 1) * 8);
                bf[2 * j][0] = r[0];
                bf[2 * j][1] = r[1];
                bf[2 * j + 1][0] = r[2];
                bf[2 * j + 1][1] = r[3];
            }
#pragma unroll
            for (int j = 0; j < 4; j++) mma16816(sc[j], af, bf[j]);
        }

        // mask, scale, the online softmax of the two rows
#pragma unroll
        for (int h = 0; h < 2; h++) {
            float mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; j++)
#pragma unroll
                for (int c = 0; c < 2; c++) {
                    const int p = p0 + 8 * j + 2 * tq + c;
                    float &v = sc[j][2 * h + c];
                    v = (p <= rpos[h] && p >= rlo[h]) ? v * sl2 : -INFINITY;
                    mx = fmaxf(mx, v);
                }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            const float mn = fmaxf(m[h], mx);
            const float corr = mn == -INFINITY ? 1.f : exp2f(m[h] - mn);
            float sum = 0.f;
#pragma unroll
            for (int j = 0; j < 4; j++)
#pragma unroll
                for (int c = 0; c < 2; c++) {
                    float &v = sc[j][2 * h + c];
                    v = mn == -INFINITY ? 0.f : exp2f(v - mn);
                    sum += v;
                }
            l[h] = l[h] * corr + sum;
            m[h] = mn;
#pragma unroll
            for (int j = 0; j < D / 8; j++) {
                o[j][2 * h] *= corr;
                o[j][2 * h + 1] *= corr;
            }
        }

        // O += P V, P as the A fragments of two k16 steps
#pragma unroll
        for (int ks2 = 0; ks2 < 2; ks2++) {
            uint32_t pa[4] = {pack_half2(sc[2 * ks2][0], sc[2 * ks2][1]), pack_half2(sc[2 * ks2][2], sc[2 * ks2][3]),
                              pack_half2(sc[2 * ks2 + 1][0], sc[2 * ks2 + 1][1]),
                              pack_half2(sc[2 * ks2 + 1][2], sc[2 * ks2 + 1][3])};
#pragma unroll
            for (int j = 0; j < D / 16; j++) {
                uint32_t r[4];
                ldsm_x4_t(r, vs + (16 * ks2 + (lane & 7) + ((lane >> 3) & 1) * 8) * P + 16 * j + (lane >> 4) * 8);
                const uint32_t b0[2] = {r[0], r[1]}, b1[2] = {r[2], r[3]};
                mma16816(o[2 * j], pa, b0);
                mma16816(o[2 * j + 1], pa, b1);
            }
        }
    }

    // the row sums over the 4 lanes of a row, the sink, the gate
    const float sink = a.has_sinks ? ((const float *) a.sinks)[head] * 1.4426950408889634f : -INFINITY;
#pragma unroll
    for (int h = 0; h < 2; h++) {
        l[h] += __shfl_xor_sync(0xffffffffu, l[h], 1);
        l[h] += __shfl_xor_sync(0xffffffffu, l[h], 2);
        const int r = 16 * warp + g + 8 * h;
        const float M = fmaxf(m[h], sink);
        const float corr = M == -INFINITY ? 0.f : exp2f(m[h] - M);
        const float L = l[h] * corr + (a.has_sinks ? exp2f(sink - M) : 0.f);
        const float inv = corr / L;
        if (r < nq) {
            const size_t row = (size_t) (t0 + r) * a.n_head + head;
            float *out = (float *) a.o + row * D;
            const float *gate = (const float *) a.qg + row * 2 * D + D;
#pragma unroll
            for (int j = 0; j < D / 8; j++) {
                const int d = 8 * j + 2 * tq;
                float v0 = o[j][2 * h] * inv, v1 = o[j][2 * h + 1] * inv;
                if (a.gated) {
                    v0 *= sigmoid(gate[d]);
                    v1 *= sigmoid(gate[d + 1]);
                }
                *(float2 *) (out + d) = make_float2(v0, v1);
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(32 * FA_WARPS) fattn_64(const AttnArgs a) { fattn_body<64>(a); }
extern "C" __global__ void __launch_bounds__(32 * FA_WARPS) fattn_128(const AttnArgs a) { fattn_body<128>(a); }
extern "C" __global__ void __launch_bounds__(32 * FA_WARPS) fattn_256(const AttnArgs a) { fattn_body<256>(a); }

// ---------------------------------------------------------------------------
// delta net. The states have `slots` copies: a pass reads and writes slot
// `slot`, or with `save` (verify, at most `slots` tokens) writes the state
// after token t to slot (slot + 1 + t) % slots, so that it can resume from any
// of them.

struct ConvArgs { u64 qkv; u64 w; u64 state; u64 out; u64 params; int conv_dim; int n; int slots; int save; };

// causal depthwise conv (kernel 4) + silu. A block of CONV_C channels by
// CONV_T threads over the tokens (t, t + CONV_T, ...); the state holds the
// previous 3 inputs, as [slots][3][conv_dim]. The block reads the state
// before any thread writes it.
#define CONV_C 32
#define CONV_T 8
extern "C" __global__ void __launch_bounds__(CONV_C * CONV_T) ssm_conv(const ConvArgs a) {
    const int cd = a.conv_dim;
    const int c = blockIdx.x * CONV_C + threadIdx.x;
    const bool ok = c < cd;
    float *state = (float *) a.state;
    const float *qkv = (const float *) a.qkv;
    float *out = (float *) a.out;
    const int slot = ((const PassParams *) a.params)->slot;
    float *st = state + (size_t) slot * 3 * cd + c;
    float s[3] = {0.f, 0.f, 0.f};
    float4 wc = make_float4(0.f, 0.f, 0.f, 0.f);
    if (ok) {
        s[0] = st[0]; s[1] = st[cd]; s[2] = st[2 * cd];
        wc = *(const float4 *) ((const float *) a.w + 4 * c);
    }
    __syncthreads();
    if (!ok) return;
    // input t, the state's for t < 0
    auto in = [&](int t) { return t < 0 ? s[3 + t] : qkv[(size_t) t * cd + c]; };
    for (int t = threadIdx.y; t < a.n; t += CONV_T) {
        const float x3 = in(t - 3), x2 = in(t - 2), x1 = in(t - 1), x0 = in(t);
        // as the CPU sums: w0 s0 + w1 s1 + w2 s2 + w3 x, in order
        float acc = 0.f;
        acc += wc.x * x3;
        acc += wc.y * x2;
        acc += wc.z * x1;
        acc += wc.w * x0;
        out[(size_t) t * cd + c] = silu(acc);
        if (a.save || t == a.n - 1) {
            float *sv = a.save ? state + (size_t) ((slot + 1 + t) % a.slots) * 3 * cd + c : st;
            sv[0] = x2; sv[cd] = x1; sv[2 * cd] = x0;
        }
    }
}

struct GdnArgs {
    u64 conv_out; u64 alpha; u64 beta; u64 dt_bias; u64 a_neg; u64 state; u64 out; u64 params;
    int n; int hk; int hv; int conv_dim; float eps; int slots; int save; int pad;
};

__device__ __forceinline__ float dot4(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

#define SD 128
#define GDN_ROWS 4

// In place, per (key head, token), one warp: q and k over their l2 norms
// (q also times 1/sqrt(SD), the output's scale), and per value head of the
// key head (value heads are in tiled order: key head = head % hk) the decay
// exp(softplus(alpha + dt) * a) over alpha and sigmoid(beta) over beta.
extern "C" __global__ void gdn_prep(const GdnArgs a) {
    const int kh = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    float *base = (float *) a.conv_out + (size_t) t * a.conv_dim;
    float4 *qp = (float4 *) (base + kh * SD) + lane;
    float4 *kp = (float4 *) (base + a.hk * SD + kh * SD) + lane;
    float4 q4 = *qp, k4 = *kp;
    const float qs = 1.f / sqrtf(warp_sum(dot4(q4, q4)) + a.eps) * (1.f / sqrtf((float) SD));
    const float ks = 1.f / sqrtf(warp_sum(dot4(k4, k4)) + a.eps);
    q4.x *= qs; q4.y *= qs; q4.z *= qs; q4.w *= qs;
    k4.x *= ks; k4.y *= ks; k4.z *= ks; k4.w *= ks;
    *qp = q4;
    *kp = k4;
    if (lane < a.hv / a.hk) {
        const int h = kh + a.hk * lane;
        float *alpha = (float *) a.alpha + (size_t) t * a.hv + h;
        float *beta = (float *) a.beta + (size_t) t * a.hv + h;
        const float x = *alpha + ((const float *) a.dt_bias)[h];
        const float sp = x > 20.f ? x : log1pf(expf(x));
        *alpha = expf(sp * ((const float *) a.a_neg)[h]);
        *beta = sigmoid(*beta);
    }
}

// The recurrence after gdn_prep. One warp per state row (a value index),
// GDN_ROWS rows per block; each lane holds 4 consecutive key elements. The
// state row j holds S[:, j]. Per token: S *= decay, d = (v - S.k) * beta,
// S += k d, y = S.q, the last as (decayed S).q + d (k.q), so that the three
// dot products reduce together. The next token's inputs load during a step.
extern "C" __global__ void __launch_bounds__(WARP * GDN_ROWS) gdn(const GdnArgs a) {
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int head = blockIdx.y;
    const int row = blockIdx.x * GDN_ROWS + ty;
    const int kh = head % a.hk;

    float *s_ptr = (float *) a.state + ((size_t) head * SD + row) * SD + 4 * tx;
    const size_t slot_size = (size_t) a.hv * SD * SD;
    const int slot = ((const PassParams *) a.params)->slot;
    float4 ls = *(float4 *) (s_ptr + slot * slot_size);

    const float *g = (const float *) a.alpha, *bt = (const float *) a.beta;
    const float *cv = (const float *) a.conv_out;
    const size_t qo = kh * SD + 4 * tx, ko = a.hk * SD + kh * SD + 4 * tx, vo = 2 * a.hk * SD + head * SD + row;
    float *out = (float *) a.out + (size_t) head * SD + row;
    const size_t ostride = (size_t) a.hv * SD;

    float4 q4 = *(const float4 *) (cv + qo), k4 = *(const float4 *) (cv + ko);
    float vj = cv[vo], decay = g[head], b = bt[head];
    for (int t = 0; t < a.n; t++) {
        float4 nq = q4, nk = k4;
        float nv = vj, ndecay = decay, nb = b;
        if (t + 1 < a.n) {
            const float *nx = cv + (size_t) (t + 1) * a.conv_dim;
            nq = *(const float4 *) (nx + qo);
            nk = *(const float4 *) (nx + ko);
            nv = nx[vo];
            ndecay = g[(t + 1) * a.hv + head];
            nb = bt[(t + 1) * a.hv + head];
        }
        ls.x *= decay; ls.y *= decay; ls.z *= decay; ls.w *= decay;
        float sk = dot4(ls, k4), sq = dot4(ls, q4), kq = dot4(k4, q4);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            sk += __shfl_xor_sync(0xffffffffu, sk, o);
            sq += __shfl_xor_sync(0xffffffffu, sq, o);
            kq += __shfl_xor_sync(0xffffffffu, kq, o);
        }
        const float d = (vj - sk) * b;
        ls.x += k4.x * d; ls.y += k4.y * d; ls.z += k4.z * d; ls.w += k4.w * d;
        if (tx == 0) out[(size_t) t * ostride] = sq + d * kq;
        if (a.save) *(float4 *) (s_ptr + ((slot + 1 + t) % a.slots) * slot_size) = ls;
        q4 = nq; k4 = nk; vj = nv; decay = ndecay; b = nb;
    }
    if (!a.save) *(float4 *) (s_ptr + slot * slot_size) = ls;
}

// ---------------------------------------------------------------------------
// mixture of experts. A pass of n tokens through k experts each makes n*k
// (token, expert) pairs, pair t*k + j for the j-th expert of token t:
// moe_route picks them, moe_sort groups them by expert into tiles of up to NT
// (or TN) pairs, mm_id (or mma_id) multiplies each tile by its expert's matrix, and moe_combine
// sums the pairs of each token.

struct RouteArgs {
    u64 logits; u64 ids; u64 weights; u64 bias;
    int n_expert;
    int k;
    int sigmoid;  // gating: sigmoid of each logit, else softmax of the logits
    int has_bias; // a bias per expert adds to the probabilities to choose them
    int norm;     // the weights over their sum
    float scale;  // of the weights
};

// the larger of two (score, expert), the lower expert among equals
__device__ __forceinline__ void best_of(float &v, int &i, float v2, int i2) {
    if (v2 > v || (v2 == v && i2 < i)) {
        v = v2;
        i = i2;
    }
}

// One block per token: its probabilities, its k experts of highest
// probability plus bias (the lowest index among equals), and their
// probabilities over their sum (or as they are), scaled. `p` holds the
// scores, a chosen expert's set to -inf.
extern "C" __global__ void moe_route(const RouteArgs a) {
    extern __shared__ float p[]; // [n_expert] scores, then [n_expert] probabilities
    float *prob = p + a.n_expert;
    __shared__ float red_v[32];
    __shared__ int red_i[32];
    const int ne = a.n_expert, tid = threadIdx.x, nt = blockDim.x;
    const int lane = tid & 31, sg = tid >> 5, nsg = nt >> 5;
    const float *l = (const float *) a.logits + (size_t) blockIdx.x * ne;
    const float *bias = (const float *) a.bias;

    if (a.sigmoid) {
        for (int i = tid; i < ne; i += nt) prob[i] = sigmoid(l[i]);
    } else {
        float m = -INFINITY;
        for (int i = tid; i < ne; i += nt) m = fmaxf(m, l[i]);
        m = warp_max(m);
        if (lane == 0) red_v[sg] = m;
        __syncthreads();
        m = red_v[0];
        for (int s = 1; s < nsg; s++) m = fmaxf(m, red_v[s]);
        __syncthreads();
        float sum = 0.f;
        for (int i = tid; i < ne; i += nt) {
            prob[i] = expf(l[i] - m);
            sum += prob[i];
        }
        sum = block_sum(sum);
        const float inv = 1.f / sum;
        for (int i = tid; i < ne; i += nt) prob[i] *= inv;
    }
    __syncthreads();
    for (int i = tid; i < ne; i += nt) p[i] = prob[i] + (a.has_bias ? bias[i] : 0.f);
    __syncthreads();

    int *ids = (int *) a.ids + (size_t) blockIdx.x * a.k;
    float *weights = (float *) a.weights + (size_t) blockIdx.x * a.k;
    float total = 0.f;
    for (int j = 0; j < a.k; j++) {
        float v = -INFINITY;
        int bi = ne;
        for (int i = tid; i < ne; i += nt) best_of(v, bi, p[i], i);
        for (int o = 16; o > 0; o >>= 1) {
            best_of(v, bi, __shfl_down_sync(0xffffffffu, v, o), __shfl_down_sync(0xffffffffu, bi, o));
        }
        if (lane == 0) {
            red_v[sg] = v;
            red_i[sg] = bi;
        }
        __syncthreads();
        v = red_v[0];
        bi = red_i[0];
        for (int s = 1; s < nsg; s++) best_of(v, bi, red_v[s], red_i[s]);
        total += prob[bi];
        if (tid == 0) {
            ids[j] = bi;
            weights[j] = prob[bi];
            p[bi] = -INFINITY;
        }
        __syncthreads();
    }
    if (tid == 0) {
        const float sum = a.norm ? fmaxf(total, 1.f / 16384.f) : 1.f;
        for (int j = 0; j < a.k; j++) {
            weights[j] = a.scale == 1.f ? weights[j] / sum : weights[j] / sum * a.scale;
        }
    }
}

struct SortArgs { u64 ids; u64 order; u64 tiles; u64 n_tiles; int n_pairs; int n_expert; int tile; int pad; };

// One block: the pairs grouped by expert in `order`, and the tiles of up to
// `tile` pairs of each expert (NT for mm_id, TN for mma_id). Shared: a count, a start and a fill per expert.
extern "C" __global__ void moe_sort(const SortArgs a) {
    extern __shared__ int sh[];
    int *count = sh, *start = sh + a.n_expert, *fill = sh + 2 * a.n_expert;
    const int *ids = (const int *) a.ids;
    for (int e = threadIdx.x; e < a.n_expert; e += blockDim.x) {
        count[e] = 0;
        fill[e] = 0;
    }
    __syncthreads();
    for (int p = threadIdx.x; p < a.n_pairs; p += blockDim.x) atomicAdd(&count[ids[p]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        Tile *tiles = (Tile *) a.tiles;
        int s = 0, nt = 0;
        for (int e = 0; e < a.n_expert; e++) {
            start[e] = s;
            const int c = count[e];
            for (int o = 0; o < c; o += a.tile) tiles[nt++] = Tile{e, s + o, min(a.tile, c - o), 0};
            s += c;
        }
        *(int *) a.n_tiles = nt;
    }
    __syncthreads();
    // the order within an expert doesn't change the numbers: each pair's dot
    // products are its own
    int *order = (int *) a.order;
    for (int p = threadIdx.x; p < a.n_pairs; p += blockDim.x) {
        const int e = ids[p];
        order[start[e] + atomicAdd(&fill[e], 1)] = p;
    }
}

struct CombineArgs { u64 x; u64 out; u64 weights; u64 shared; u64 gate; int n_embd; int k; int mode; int pad; };

// x[t] += sum_j weights[t*k + j] * out[t*k + j], plus the shared expert's
// output: mode 0 none, 1 times sigmoid(gate[t]), 2 as it is
extern "C" __global__ void moe_combine(const CombineArgs a) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (d >= a.n_embd) return;
    const float *out = (const float *) a.out, *w = (const float *) a.weights;
    const float *shared = (const float *) a.shared;
    float acc = 0.f;
    for (int j = 0; j < a.k; j++) acc += w[t * a.k + j] * out[((size_t) t * a.k + j) * a.n_embd + d];
    if (a.mode == 1) {
        acc += sigmoid(((const float *) a.gate)[t]) * shared[(size_t) t * a.n_embd + d];
    } else if (a.mode == 2) {
        acc += shared[(size_t) t * a.n_embd + d];
    }
    ((float *) a.x)[(size_t) t * a.n_embd + d] += acc;
}
