// Metal kernels for the qwen35 and qwen35moe graphs.
//
// The quantized matvec (mv_*), the simdgroup matmul (mm) and the dequantize
// helpers are adapted from ggml-metal (llama.cpp, MIT). The delta net kernel is
// adapted from ggml's gated_delta_net, fused with the q/k l2 norm and the gate
// math. The remaining kernels are written for this model, among them the
// expert products over tiles of the tokens routed to the same expert, which
// read the expert's weights once per tile: `mm_id` (the matmul, 32 tokens)
// and `mmv_id` (the matvec, up to 8).

#include <metal_stdlib>
using namespace metal;

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

#define QK4_0 32
#define QK_K 256

typedef struct { half d;         uint8_t qs[16]; } block_q4_0;
typedef struct { half d; half m; uint8_t qs[16]; } block_q4_1;
typedef struct { half d;         int8_t  qs[32]; } block_q8_0;
typedef struct { half d; half dmin; uint8_t scales[12]; uint8_t qs[128]; } block_q4_K;
typedef struct { half d; half dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; } block_q5_K;
typedef struct { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; half d; } block_q6_K;
// 16 bf16 values, the "block" of the float matmul path
typedef struct { ushort v[16]; } block_bf16;

struct MatArgs {
    int   k;      // input width
    int   rows;   // weight rows = output width
    int   n;      // input rows (tokens)
    int   pad;
    ulong w_row_bytes;
    ulong x_row_bytes;
};

// Fast math lets the compiler assume that no value is INF, so a score that
// must not count is a finite value far below any real score. The start of a
// running max is SCORE_FLOOR, and a masked score is MASKED_SCORE, below it.
// Every exp of a score minus a running max then has a finite argument of at
// most about FLT_MAX/4 in magnitude, and a masked or empty term gives
// exp(-FLT_MAX/8) or less, which is 0, as exp(-INFINITY) is.
constant float SCORE_FLOOR  = -FLT_MAX/8;
constant float MASKED_SCORE = -FLT_MAX/4;

// SIGMOID_OVERFLOW_X is ops::SIGMOID_OVERFLOW_X, which the engine defines
// before this source when it compiles the library.
#ifndef SIGMOID_OVERFLOW_X
#error "the engine defines SIGMOID_OVERFLOW_X"
#endif

// At or above SIGMOID_OVERFLOW_X, the forms of ggml; below it exp(-x) can
// overflow, so the form that computes exp(x)
inline float sigmoid(float x) {
    if (x < SIGMOID_OVERFLOW_X) {
        const float e = exp(x);
        return e / (1.0f + e);
    }
    return 1.0f / (1.0f + exp(-x));
}
inline float silu(float x) {
    return x < SIGMOID_OVERFLOW_X ? x * sigmoid(x) : x / (1.0f + exp(-x));
}
inline float bf16_to_f32(ushort h) { return as_type<float>(uint(h) << 16); }

// ---------------------------------------------------------------------------
// matvec: q4_0 / q4_1 (ggml mul_vec_q_n_f32_impl)

inline float block_q_n_dot_y(device const block_q4_0 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;
    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    device const uint16_t * qs = ((device const uint16_t *) qb_curr + 1 + il/2);
    for (int i = 0; i < 8; i += 2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F);
        acc[1] += yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[2] += yl[i + 8] * (qs[i / 2] & 0x00F0);
        acc[3] += yl[i + 9] * (qs[i / 2] & 0xF000);
    }
    return d * (sumy * -8.f + acc[0] + acc[1] + acc[2] + acc[3]);
}

inline float block_q_n_dot_y(device const block_q4_1 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;
    float m = qb_curr->m;
    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    device const uint16_t * qs = ((device const uint16_t *) qb_curr + 2 + il/2);
    for (int i = 0; i < 8; i += 2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F);
        acc[1] += yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[2] += yl[i + 8] * (qs[i / 2] & 0x00F0);
        acc[3] += yl[i + 9] * (qs[i / 2] & 0xF000);
    }
    return d * (acc[0] + acc[1] + acc[2] + acc[3]) + sumy * m;
}

template<typename block_q_type, short NR0, short NSG>
kernel void mv_q_n(
        constant MatArgs & args [[buffer(0)]],
        device const char  * src0 [[buffer(1)]],
        device const char  * src1 [[buffer(2)]],
        device       float * dst  [[buffer(3)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NW = 32;
    constexpr short NQ = 16;

    const int nb = args.k/QK4_0;
    const int r0 = (tgpig.x*NSG + sgitg)*NR0;
    const int r1 = tgpig.y;

    device const float * y = (device const float *) (src1 + r1*args.x_row_bytes);

    device const block_q_type * ax[NR0];
    FOR_UNROLL (int row = 0; row < NR0; ++row) {
        const int rr = min(r0 + row, args.rows - 1);
        ax[row] = (device const block_q_type *) (src0 + rr*args.w_row_bytes);
    }

    float sumf[NR0] = {0.f};

    const short ix = (tiisg/(NW/NQ));
    const short il = (tiisg%(NW/NQ))*8;

    float yl[16];
    device const float * yb = y + ix*QK4_0 + il;

    for (int ib = ix; ib < nb; ib += NQ) {
        float sumy[2] = { 0.f, 0.f };

        FOR_UNROLL (short i = 0; i < 8; i += 2) {
            sumy[0]  += yb[i +  0] + yb[i +  1];
            yl[i + 0] = yb[i +  0];
            yl[i + 1] = yb[i +  1]/256.f;

            sumy[1]  += yb[i + 16] + yb[i + 17];
            yl[i + 8] = yb[i + 16]/16.f;
            yl[i + 9] = yb[i + 17]/4096.f;
        }

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            sumf[row] += block_q_n_dot_y(ax[row] + ib, sumy[0] + sumy[1], yl, il);
        }

        yb += QK4_0 * 16;
    }

    device float * dst_f32 = dst + r1*args.rows;
    for (int row = 0; row < NR0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && r0 + row < args.rows) {
            dst_f32[r0 + row] = tot;
        }
    }
}

template [[host_name("mv_q4_0")]] kernel void mv_q_n<block_q4_0, 4, 2>(constant MatArgs &, device const char *, device const char *, device float *, uint3, ushort, ushort);
template [[host_name("mv_q4_1")]] kernel void mv_q_n<block_q4_1, 4, 2>(constant MatArgs &, device const char *, device const char *, device float *, uint3, ushort, ushort);

// ---------------------------------------------------------------------------
// matvec: q5_K (ggml kernel_mul_mv_q5_K_f32_impl, nr0 = 1, nsg = 2)

kernel void mv_q5_K(
        constant MatArgs & args [[buffer(0)]],
        device const char  * src0 [[buffer(1)]],
        device const char  * src1 [[buffer(2)]],
        device       float * dst  [[buffer(3)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NSG = 2;
    constexpr int nr0 = 1;

    const int nb = args.k/QK_K;
    const int r1 = tgpig.y;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    if (first_row >= args.rows) {
        return;
    }

    device const block_q5_K * x = (device const block_q5_K *) (src0 + first_row*args.w_row_bytes);
    device const float     * yy = (device const float      *) (src1 + r1*args.x_row_bytes);

    float sumf[nr0] = {0.f};
    float yl[16], yh[16];

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short tid = tiisg/4;
    const short ix  = tiisg%4;
    const short iq  = tid/4;
    const short ir  = tid%4;

    const short l0 = 8*ir;
    const short q_offset = 32*iq + l0;
    const short y_offset = 64*iq + l0;

    const uint8_t hm1 = 1u << (2*iq);
    const uint8_t hm2 = hm1 << 1;
    const uint8_t hm3 = hm1 << 4;
    const uint8_t hm4 = hm2 << 4;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    device const float * y1 = yy + ix*QK_K + y_offset;

    for (int i = ix; i < nb; i += 4) {
        device const uint8_t * q1 = x[i].qs + q_offset;
        device const uint8_t * qh = x[i].qh + l0;
        device const half * dh = &x[i].d;
        device const uint16_t * a = (device const uint16_t *)x[i].scales + iq;

        device const float * y2 = y1 + 128;
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short l = 0; l < 8; ++l) {
            yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];
            yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
        }

        for (short row = 0; row < nr0; ++row) {
            device const uint8_t * q2 = q1 + 64;

            sc16[0] = a[0] & kmask1;
            sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

            float4 acc1 = {0.f};
            float4 acc2 = {0.f};
            FOR_UNROLL (short l = 0; l < 8; ++l) {
                uint8_t h = qh[l];
                acc1[0] += yl[l+0] * (q1[l] & 0x0F);
                acc1[1] += yl[l+8] * (q1[l] & 0xF0);
                acc1[2] += yh[l+0] * (q2[l] & 0x0F);
                acc1[3] += yh[l+8] * (q2[l] & 0xF0);
                acc2[0] += h & hm1 ? yl[l+0] : 0.f;
                acc2[1] += h & hm2 ? yl[l+8] : 0.f;
                acc2[2] += h & hm3 ? yh[l+0] : 0.f;
                acc2[3] += h & hm4 ? yh[l+8] : 0.f;
            }

            sumf[row] += dh[0] * (sc8[0] * (acc1[0]      + 16.f*acc2[0]) +
                                  sc8[1] * (acc1[1]/16.f + 16.f*acc2[1]) +
                                  sc8[4] * (acc1[2]      + 16.f*acc2[2]) +
                                  sc8[5] * (acc1[3]/16.f + 16.f*acc2[3])) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
        }

        y1 += 4 * QK_K;
    }

    device float * dst_f32 = dst + r1*args.rows;
    for (int row = 0; row < nr0 && first_row + row < args.rows; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = tot;
        }
    }
}

// ---------------------------------------------------------------------------
// matvec: q6_K (ggml kernel_mul_mv_q6_K_f32_impl, nr0 = 2, nsg = 2)

kernel void mv_q6_K(
        constant MatArgs & args [[buffer(0)]],
        device const char  * src0 [[buffer(1)]],
        device const char  * src1 [[buffer(2)]],
        device       float * dst  [[buffer(3)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NSG = 2;
    constexpr int nr0 = 2;

    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const int nb = args.k/QK_K;
    const int r1 = tgpig.y;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;

    device const float * yy = (device const float *) (src1 + r1*args.x_row_bytes);

    float sumf[nr0] = { 0.f };
    float yl[16];

    const short tid = tiisg/2;
    const short ix  = tiisg%2;
    const short ip  = tid/8;
    const short il  = tid%8;
    const short l0  = 4*il;
    const short is  = 8*ip + l0/16;

    const short y_offset   = 128*ip + l0;
    const short q_offset_l =  64*ip + l0;
    const short q_offset_h =  32*ip + l0;

    for (int i = ix; i < nb; i += 2) {
        device const float * y = yy + i * QK_K + y_offset;

        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = y[l +  0];
            yl[4*l + 1] = y[l + 32];
            yl[4*l + 2] = y[l + 64];
            yl[4*l + 3] = y[l + 96];
        }

        for (short row = 0; row < nr0; ++row) {
            const int rr = min(first_row + row, args.rows - 1);
            device const block_q6_K * x = (device const block_q6_K *) (src0 + rr*args.w_row_bytes) + i;
            device const uint8_t * q1 = x->ql + q_offset_l;
            device const uint8_t * q2 = q1 + 32;
            device const uint8_t * qh = x->qh + q_offset_h;
            device const int8_t  * sc = x->scales + is;

            float4 sums = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }

            sumf[row] += x->d * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);
        }
    }

    device float * dst_f32 = dst + r1*args.rows;
    for (int row = 0; row < nr0 && first_row + row < args.rows; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

// ---------------------------------------------------------------------------
// matvec: float weights, one simdgroup per row, 4 rows per threadgroup

inline float4 load4(device const float * p) { return *(device const float4 *) p; }
inline float4 load4(device const half  * p) { return float4(*(device const half4 *) p); }
inline float4 load4(device const ushort * p) {
    ushort4 u = *(device const ushort4 *) p;
    return float4(bf16_to_f32(u.x), bf16_to_f32(u.y), bf16_to_f32(u.z), bf16_to_f32(u.w));
}

template<typename T>
kernel void mv_float(
        constant MatArgs & args [[buffer(0)]],
        device const char  * src0 [[buffer(1)]],
        device const char  * src1 [[buffer(2)]],
        device       float * dst  [[buffer(3)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const int r = tgpig.x*4 + sgitg;
    const int t = tgpig.y;
    if (r >= args.rows) {
        return;
    }
    device const T     * w = (device const T     *) (src0 + r*args.w_row_bytes);
    device const float * y = (device const float *) (src1 + t*args.x_row_bytes);
    float4 acc = 0.f;
    for (int i = 4*tiisg; i < args.k; i += 128) {
        acc += load4(w + i) * load4(y + i);
    }
    const float tot = simd_sum(acc[0] + acc[1] + acc[2] + acc[3]);
    if (tiisg == 0) {
        dst[t*args.rows + r] = tot;
    }
}

template [[host_name("mv_f32")]]  kernel void mv_float<float> (constant MatArgs &, device const char *, device const char *, device float *, uint3, ushort, ushort);
template [[host_name("mv_f16")]]  kernel void mv_float<half>  (constant MatArgs &, device const char *, device const char *, device float *, uint3, ushort, ushort);
template [[host_name("mv_bf16")]] kernel void mv_float<ushort>(constant MatArgs &, device const char *, device const char *, device float *, uint3, ushort, ushort);

// ---------------------------------------------------------------------------
// dequantize 16 weights into a 4x4 matrix (ggml dequantize.h)

#define QK_NL 16

template <typename type4x4>
void dequantize_f32(device const float4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4x4>
void dequantize_f16(device const half4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4x4>
void dequantize_bf16(device const block_bf16 * src, short il, thread type4x4 & reg) {
    for (int i = 0; i < 16; i++) {
        reg[i/4][i%4] = bf16_to_f32(src->v[i]);
    }
}

template <typename type4x4>
void dequantize_q4_0(device const block_q4_0 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 1);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float md = -8.h * xb->d;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    float4x4 reg_f;
    for (int i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = d1 * (qs[i] & mask0) + md;
        reg_f[i/2][2*(i%2) + 1] = d2 * (qs[i] & mask1) + md;
    }
    reg = (type4x4) reg_f;
}

template <typename type4x4>
void dequantize_q4_1(device const block_q4_1 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 2);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float  m = xb->m;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    float4x4 reg_f;
    for (int i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = ((qs[i] & mask0) * d1) + m;
        reg_f[i/2][2*(i%2) + 1] = ((qs[i] & mask1) * d2) + m;
    }
    reg = (type4x4) reg_f;
}

template <typename type4x4>
void dequantize_q8_0(device const block_q8_0 *xb, short il, thread type4x4 & reg) {
    device const packed_char4 * qs = (device const packed_char4 *) xb->qs;
    const float d = xb->d;

    float4x4 reg_f;
    for (int i = 0; i < 4; ++i) {
        reg_f[i] = float4(qs[4*il + i]) * d;
    }
    reg = (type4x4) reg_f;
}

static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)), uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

template <typename type4x4>
void dequantize_q4_K(device const block_q4_K * xb, short il, thread type4x4 & reg) {
    device const uchar * q = xb->qs;

    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d   = il < 2 ? xb->d : xb->d / 16.h;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

template <typename type4x4>
void dequantize_q5_K(device const block_q5_K *xb, short il, thread type4x4 & reg) {
    device const uint8_t * q  = xb->qs;
    device const uint8_t * qh = xb->qh;

    short is = (il/4) * 2;
    q  = q + 32 * (il/4) + 16 * (il&1);
    qh = qh + 16 * (il&1);
    uint8_t ul = 1 << (il/2);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d = il < 2 ? xb->d : xb->d / 16.f;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask  = il<2 ? 0x0F : 0xF0;
    const float qh_val = il<2 ? 16.f : 256.f;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * ((q[i] & mask) + (qh[i] & ul ? qh_val : 0)) - ml;
    }
}

template <typename type4x4>
void dequantize_q6_K(device const block_q6_K *xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const uint16_t * ql = (device const uint16_t *)xb->ql;
    device const uint16_t * qh = (device const uint16_t *)xb->qh;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    ql = ql + 32*(il/8) + 16*((il/2)&1) + 8*(il&1);
    qh = qh + 16*(il/8) + 8*(il&1);
    float sc = scales[(il%2) + 2 * ((il/2))];
    il = (il/2) & 3;

    const uint32_t kmask1 = il>1 ? (il>2 ? 0xC0C0C0C0 : 0x30303030) : (il>0 ? 0x0C0C0C0C : 0x03030303);
    const uint32_t kmask2 = il>1 ? 0xF0F0F0F0                       : 0x0F0F0F0F;
    const float ml = d_all * sc * 32.f;
    const float dl0 = d_all * sc;
    const float dl1 = dl0 / 256.f;
    const float dl2 = dl0 / (256.f * 256.f);
    const float dl3 = dl0 / (256.f * 256.f * 256.f);
    const uint8_t shr_h = il>2 ? 2 : 0;
    const uint8_t shl_h = il>1 ? 0 : (il>0 ? 2 : 4);
    const uint8_t shr_l = il>1 ? 4 : 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t  low = (ql[2*i] | (uint32_t)(ql[2*i+1] << 16)) & kmask2;
        const uint32_t high = (qh[2*i] | (uint32_t)(qh[2*i+1] << 16)) & kmask1;
        const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = dl0 *  ((half)(q & 0xFF))       - ml;
        reg[i][1] = dl1 * ((float)(q & 0xFF00))     - ml;
        reg[i][2] = dl2 * ((float)(q & 0xFF0000))   - ml;
        reg[i][3] = dl3 * ((float)(q & 0xFF000000)) - ml;
    }
}

// ---------------------------------------------------------------------------
// matmul for up to 8 input rows on simdgroup matrices, the operands placed in
// the matrix registers directly (no threadgroup memory, no barriers):
// C[row][token] += W[row][k] Y[token][k], 8 weight rows x 8 tokens per
// fragment. In an 8x8 fragment lane l holds elements (fm, fn) and (fm, fn + 1)
// (the layout MLX uses). Over a span of 32 weights of a row, a lane takes
// elements 8j + fn and 8j + fn + 1 for j = 0..3: `deq_span` gives them, for
// the span at `s` spans (32 weights) from the row start.

inline void deq_span(device const block_q4_0 * x, int s, short fn, thread float2 * w) {
    device const block_q4_0 * b = x + s;
    device const ushort * qs = (device const ushort *) b->qs;
    const ushort q0 = qs[fn/2], q1 = qs[fn/2 + 4];
    const float d = b->d, m = -8.f*d;
    w[0] = float2(d*float(q0 & 0xF) + m,        d*float((q0 >> 8) & 0xF) + m);
    w[1] = float2(d*float(q1 & 0xF) + m,        d*float((q1 >> 8) & 0xF) + m);
    w[2] = float2(d*float((q0 >> 4) & 0xF) + m, d*float(q0 >> 12) + m);
    w[3] = float2(d*float((q1 >> 4) & 0xF) + m, d*float(q1 >> 12) + m);
}

inline void deq_span(device const block_q4_1 * x, int s, short fn, thread float2 * w) {
    device const block_q4_1 * b = x + s;
    device const ushort * qs = (device const ushort *) b->qs;
    const ushort q0 = qs[fn/2], q1 = qs[fn/2 + 4];
    const float d = b->d, m = b->m;
    w[0] = float2(d*float(q0 & 0xF) + m,        d*float((q0 >> 8) & 0xF) + m);
    w[1] = float2(d*float(q1 & 0xF) + m,        d*float((q1 >> 8) & 0xF) + m);
    w[2] = float2(d*float((q0 >> 4) & 0xF) + m, d*float(q0 >> 12) + m);
    w[3] = float2(d*float((q1 >> 4) & 0xF) + m, d*float(q1 >> 12) + m);
}

inline void deq_span(device const block_q8_0 * x, int s, short fn, thread float2 * w) {
    device const block_q8_0 * b = x + s;
    const float d = b->d;
    FOR_UNROLL (short j = 0; j < 4; j++) {
        w[j] = d*float2(b->qs[8*j + fn], b->qs[8*j + fn + 1]);
    }
}

inline void deq_span(device const block_q4_K * x, int s, short fn, thread float2 * w) {
    device const block_q4_K * b = x + s/8;
    const int g = (s%8)/2, h = s%2;
    const uchar2 sc = get_scale_min_k4_just2(2*g + h, 0, b->scales);
    const float d = float(b->d)*sc[0], m = float(b->dmin)*sc[1];
    FOR_UNROLL (short j = 0; j < 4; j++) {
        const ushort q = *(device const ushort *) (b->qs + 32*g + 8*j + fn) >> (4*h);
        w[j] = float2(d*float(q & 0xF) - m, d*float((q >> 8) & 0xF) - m);
    }
}

inline void deq_span(device const block_q5_K * x, int s, short fn, thread float2 * w) {
    device const block_q5_K * b = x + s/8;
    const int g = (s%8)/2, h = s%2;
    const uchar2 sc = get_scale_min_k4_just2(2*g + h, 0, b->scales);
    const float d = float(b->d)*sc[0], m = float(b->dmin)*sc[1];
    FOR_UNROLL (short j = 0; j < 4; j++) {
        const ushort ql = *(device const ushort *) (b->qs + 32*g + 8*j + fn) >> (4*h);
        const ushort qh = *(device const ushort *) (b->qh + 8*j + fn) >> (2*g + h);
        w[j] = float2(d*float((ql & 0xF) | ((qh & 1) << 4)) - m,
                      d*float(((ql >> 8) & 0xF) | (((qh >> 8) & 1) << 4)) - m);
    }
}

inline void deq_span(device const block_q6_K * x, int s, short fn, thread float2 * w) {
    device const block_q6_K * b = x + s/8;
    const int n = (s%8)/4, qq = s%4;
    const float d = b->d;
    device const int8_t * sc = b->scales + 8*n + 2*qq;
    const float d0 = d*sc[0], d1 = d*sc[1];
    FOR_UNROLL (short j = 0; j < 4; j++) {
        const short l = 8*j + fn;
        const ushort ql = *(device const ushort *) (b->ql + 64*n + 32*(qq & 1) + l) >> (4*(qq >> 1));
        const ushort qh = *(device const ushort *) (b->qh + 32*n + l) >> (2*qq);
        const float dl = j < 2 ? d0 : d1;
        w[j] = float2(dl*float(int((ql & 0xF) | ((qh & 3) << 4)) - 32),
                      dl*float(int(((ql >> 8) & 0xF) | (((qh >> 8) & 3) << 4)) - 32));
    }
}

inline void deq_span(device const float4x4 * x, int s, short fn, thread float2 * w) {
    device const float * f = (device const float *) x + 32*s + fn;
    FOR_UNROLL (short j = 0; j < 4; j++) {
        w[j] = *(device const float2 *) (f + 8*j);
    }
}

inline void deq_span(device const half4x4 * x, int s, short fn, thread float2 * w) {
    device const half * f = (device const half *) x + 32*s + fn;
    FOR_UNROLL (short j = 0; j < 4; j++) {
        w[j] = float2(*(device const half2 *) (f + 8*j));
    }
}

inline void deq_span(device const block_bf16 * x, int s, short fn, thread float2 * w) {
    device const ushort * f = (device const ushort *) x + 32*s + fn;
    FOR_UNROLL (short j = 0; j < 4; j++) {
        w[j] = float2(bf16_to_f32(f[8*j]), bf16_to_f32(f[8*j + 1]));
    }
}

// RT row fragments and TT input fragments (8 rows each) per simdgroup, U
// spans per step (loads first)
template<typename block_q, short RT, short NSG, short U, short TT>
kernel void mmv(
        constant MatArgs & args [[buffer(0)]],
        device const char  * src0 [[buffer(1)]],
        device const char  * src1 [[buffer(2)]],
        device       float * dst  [[buffer(3)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short qid = tiisg/4;
    const short fm = (qid & 4) + (tiisg/2)%4;
    const short fn = (qid & 2)*2 + (tiisg%2)*2;

    const int r0 = (tgpig.x*NSG + sgitg)*8*RT;
    const int r1 = tgpig.y*8*TT;
    const int nt = min(8*TT, args.n - r1);
    const int ns = args.k/32;

    device const float * y[TT][2];
    FOR_UNROLL (short tt = 0; tt < TT; tt++) {
        FOR_UNROLL (short i = 0; i < 2; i++) {
            y[tt][i] = (device const float *) (src1 + (r1 + min(8*tt + fn + i, nt - 1))*args.x_row_bytes) + fm;
        }
    }

    device const block_q * x[RT];
    FOR_UNROLL (short rt = 0; rt < RT; rt++) {
        x[rt] = (device const block_q *) (src0 + min(r0 + 8*rt + fm, args.rows - 1)*args.w_row_bytes);
    }

    simdgroup_float8x8 c[RT][TT];
    FOR_UNROLL (short rt = 0; rt < RT; rt++) {
        FOR_UNROLL (short tt = 0; tt < TT; tt++) {
            c[rt][tt] = make_filled_simdgroup_matrix<float, 8>(0.f);
        }
    }

    for (int is = 0; is < ns; is += U) {
        float2 w[U][RT][4];
        FOR_UNROLL (short u = 0; u < U; u++) {
            FOR_UNROLL (short rt = 0; rt < RT; rt++) {
                deq_span(x[rt], is + u, fn, w[u][rt]);
            }
        }
        FOR_UNROLL (short u = 0; u < U; u++) {
            FOR_UNROLL (short tt = 0; tt < TT; tt++) {
                simdgroup_float8x8 b[4];
                FOR_UNROLL (short j = 0; j < 4; j++) {
                    b[j].thread_elements()[0] = y[tt][0][32*(is + u) + 8*j];
                    b[j].thread_elements()[1] = y[tt][1][32*(is + u) + 8*j];
                }
                FOR_UNROLL (short rt = 0; rt < RT; rt++) {
                    FOR_UNROLL (short j = 0; j < 4; j++) {
                        simdgroup_float8x8 a;
                        a.thread_elements()[0] = w[u][rt][j][0];
                        a.thread_elements()[1] = w[u][rt][j][1];
                        simdgroup_multiply_accumulate(c[rt][tt], a, b[j], c[rt][tt]);
                    }
                }
            }
        }
    }

    FOR_UNROLL (short rt = 0; rt < RT; rt++) {
        const int row = r0 + 8*rt + fm;
        FOR_UNROLL (short tt = 0; tt < TT; tt++) {
            const int t = 8*tt + fn;
            if (row < args.rows) {
                if (t < nt)     dst[(r1 + t)*args.rows + row]     = c[rt][tt].thread_elements()[0];
                if (t + 1 < nt) dst[(r1 + t + 1)*args.rows + row] = c[rt][tt].thread_elements()[1];
            }
        }
    }
}

// one row fragment, one input fragment and 4 spans per step, 2 simdgroups:
// the best measured on an M2 Max for every type, from 1 to 8 input rows (two
// input fragments per simdgroup were no faster over a whole pass of 16)
#define MMV_T(name, block) \
template [[host_name(name)]] kernel void mmv<block, 1, 2, 4, 1>(constant MatArgs &, device const char *, device const char *, device float *, uint3, ushort, ushort);
MMV_T("mmv_f32",  float4x4)
MMV_T("mmv_f16",  half4x4)
MMV_T("mmv_bf16", block_bf16)
MMV_T("mmv_q4_0", block_q4_0)
MMV_T("mmv_q4_1", block_q4_1)
MMV_T("mmv_q8_0", block_q8_0)
MMV_T("mmv_q4_K", block_q4_K)
MMV_T("mmv_q5_K", block_q5_K)
MMV_T("mmv_q6_K", block_q6_K)

// two and four input fragments per simdgroup: one threadgroup takes 16 or
// 32 input rows, so the weights are read once for that many rows
#define MMV16_T(name, block) \
template [[host_name(name)]] kernel void mmv<block, 1, 2, 4, 2>(constant MatArgs &, device const char *, device const char *, device float *, uint3, ushort, ushort);
#define MMV32_T(name, block) \
template [[host_name(name)]] kernel void mmv<block, 1, 2, 4, 4>(constant MatArgs &, device const char *, device const char *, device float *, uint3, ushort, ushort);
MMV16_T("mmv16_f32", float4x4)
MMV16_T("mmv16_f16", half4x4)
MMV16_T("mmv16_bf16", block_bf16)
MMV16_T("mmv16_q4_0", block_q4_0)
MMV16_T("mmv16_q4_1", block_q4_1)
MMV16_T("mmv16_q8_0", block_q8_0)
MMV16_T("mmv16_q4_K", block_q4_K)
MMV16_T("mmv16_q5_K", block_q5_K)
MMV16_T("mmv16_q6_K", block_q6_K)
MMV32_T("mmv32_f32", float4x4)
MMV32_T("mmv32_f16", half4x4)
MMV32_T("mmv32_bf16", block_bf16)
MMV32_T("mmv32_q4_0", block_q4_0)
MMV32_T("mmv32_q4_1", block_q4_1)
MMV32_T("mmv32_q8_0", block_q8_0)
MMV32_T("mmv32_q4_K", block_q4_K)
MMV32_T("mmv32_q5_K", block_q5_K)
MMV32_T("mmv32_q6_K", block_q6_K)

// ---------------------------------------------------------------------------
// matmul with simdgroup matrices: 64 weight rows x 32 input rows per
// threadgroup of 4 simdgroups (ggml kernel_mul_mm, no tensor API).
// `k` must be a multiple of 32.

constant constexpr int MM_NR0 = 64;
constant constexpr int MM_NR1 = 32;

// The products of a tile over the whole of `k`: `x` the weight row of this
// thread's loads, at its first block, `y` its input row, at its first float.
// `mc` gets the 32 rows x 16 inputs of this simdgroup.
template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
inline void mm_tile(
        int k,
        device const block_q * x,
        device const float * y,
        threadgroup char * shmem,
        ushort tiitg,
        ushort sgitg,
        thread simdgroup_float8x8 * mc) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const short il0 = (tiitg % NL0);
    short il = il0;

    simdgroup_half8x8  ma[4];
    simdgroup_half8x8  mb[2];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < k; loop_k += NK) {
        half4x4 temp_a;
        dequantize_func(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = (tiitg/NL0)/8;

            const short lx = (tiitg/NL0)%8;
            const short ly = i%8;

            const short ib = 8*sx + sy;

            *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
        }

        {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            *(threadgroup half2x4 *)(sb + 64*ib + 8*ly) = (half2x4)(*((device const float2x4 *) y));
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const half * lsmb = (sb + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8*64;
            lsmb += 4*64;
        }
    }
}

// Stage the products of `mm_tile` in `shmem`: weight row r of input j at
// j*MM_NR0 + r
inline void mm_stage(threadgroup char * shmem, ushort sgitg, thread simdgroup_float8x8 * mc) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*MM_NR0;

    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*MM_NR0*(i/4), MM_NR0, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
}

template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
kernel void mm(
        constant MatArgs & args [[buffer(0)]],
        device const char * src0 [[buffer(1)]],
        device const char * src1 [[buffer(2)]],
        device       char * dst  [[buffer(3)]],
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {

    constexpr int NR0 = MM_NR0;
    constexpr int NR1 = MM_NR1;
    constexpr int NL0 = 2;
    constexpr int NL1 = 4;

    const int ne0 = args.rows;
    const int ne1 = args.n;

    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    const short nr0 = (ne0 - r0 < NR0) ? (ne0 - r0) : NR0;
    const short nr1 = (ne1 - r1 < NR1) ? (ne1 - r1) : NR1;

    // a thread shouldn't load data outside of the matrix
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    device const block_q * x = (device const block_q *)(src0 + args.w_row_bytes*(r0 + lr0)) + (tiitg % NL0)/nl;
    device const float * y = (device const float *)(src1 + args.x_row_bytes*(r1 + lr1) + 4*8*(tiitg % NL1));

    simdgroup_float8x8 mc[8];
    mm_tile<block_q, nl, dequantize_func>(args.k, x, y, shmem, tiitg, sgitg, mc);

    if (r0 + NR0 <= ne0 && r1 + NR1 <= ne1) {
        device float * C = (device float *) dst +
            (r0 + 32*(sgitg &  1)) +
            (r1 + 16*(sgitg >> 1)) * ne0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + 8*ne0*(i/4), ne0, 0, false);
        }
    } else {
        // partial tile: stage in threadgroup memory, write what is inside the matrix
        mm_stage(shmem, sgitg, mc);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float * D = (device float *) dst + r0 + (r1 + j)*ne0;
                threadgroup float * C = ((threadgroup float *) shmem) + j*NR0;
                for (int i = 0; i < nr0; i++) {
                    D[i] = C[i];
                }
            }
        }
    }
}

typedef void (mm_t)(constant MatArgs &, device const char *, device const char *, device char *, threadgroup char *, uint3, ushort, ushort);

template [[host_name("mm_f32")]]  kernel mm_t mm<float4x4,   1,     dequantize_f32>;
template [[host_name("mm_f16")]]  kernel mm_t mm<half4x4,    1,     dequantize_f16>;
template [[host_name("mm_bf16")]] kernel mm_t mm<block_bf16, 1,     dequantize_bf16>;
template [[host_name("mm_q4_0")]] kernel mm_t mm<block_q4_0, 2,     dequantize_q4_0>;
template [[host_name("mm_q4_1")]] kernel mm_t mm<block_q4_1, 2,     dequantize_q4_1>;
template [[host_name("mm_q8_0")]] kernel mm_t mm<block_q8_0, 2,     dequantize_q8_0>;
template [[host_name("mm_q4_K")]] kernel mm_t mm<block_q4_K, QK_NL, dequantize_q4_K>;
template [[host_name("mm_q5_K")]] kernel mm_t mm<block_q5_K, QK_NL, dequantize_q5_K>;
template [[host_name("mm_q6_K")]] kernel mm_t mm<block_q6_K, QK_NL, dequantize_q6_K>;


// ---------------------------------------------------------------------------
// element-wise

struct EltArgs { int n; };

kernel void add_inplace(
        constant EltArgs & a [[buffer(0)]],
        device float * x [[buffer(1)]],
        device const float * y [[buffer(2)]],
        uint i [[thread_position_in_grid]]) {
    if ((int) i < a.n) {
        x[i] += y[i];
    }
}

// gate = silu(gate) * up
kernel void swiglu(
        constant EltArgs & a [[buffer(0)]],
        device float * gate [[buffer(1)]],
        device const float * up [[buffer(2)]],
        uint i [[thread_position_in_grid]]) {
    if ((int) i < a.n) {
        gate[i] = silu(gate[i]) * up[i];
    }
}

// ---------------------------------------------------------------------------
// rms norm of `rows` rows of `n` floats, scaled by w[n]; optionally gated by
// silu(gate) (the delta net output norm). Output rows are `ostride` floats
// apart. One threadgroup per row.

struct NormArgs { int n; float eps; int gated; int ostride; };

kernel void rms_norm(
        constant NormArgs & a [[buffer(0)]],
        device const float * x [[buffer(1)]],
        device const float * w [[buffer(2)]],
        device       float * out [[buffer(3)]],
        device const float * gate [[buffer(4)]],
        uint   row  [[threadgroup_position_in_grid]],
        ushort tid  [[thread_position_in_threadgroup]],
        ushort ntg  [[threads_per_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[32];

    device const float * xr = x + row*a.n;
    float ss = 0.f;
    for (int i = tid; i < a.n; i += ntg) {
        ss += xr[i]*xr[i];
    }
    ss = simd_sum(ss);
    if (ntg > 32) {
        if (lane == 0) {
            partial[sg] = ss;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ss = lane < (ntg + 31)/32 ? partial[lane] : 0.f;
        ss = simd_sum(ss);
    }
    // the order of ops::rms_norm, with correctly rounded divisions and sqrt
    const float mean = precise::divide(ss, float(a.n));
    const float scale = precise::divide(1.0f, precise::sqrt(mean + a.eps));

    device float * o = out + row*a.ostride;
    for (int i = tid; i < a.n; i += ntg) {
        float v = xr[i]*scale*w[i];
        if (a.gated) {
            v *= silu(gate[row*a.n + i]);
        }
        o[i] = v;
    }
}

// ---------------------------------------------------------------------------
// full attention

struct AttnArgs {
    int   n_head;
    int   n_kv;
    int   hd;
    int   n_rot;
    int   pos0;   // position of the first token of the batch
    float eps;
    float scale;
    int   splits; // attention: position ranges per (token, head)
    int   chunk;  // attention: positions per range
    int   n;      // tokens in the batch
};

// Per (head, token), 32 threads: q / k rms norm, NeoX rope on the first n_rot
// dims, then q to q_out and k, v to the f16 cache at the token's position.
// Heads [0, n_head) are queries, [n_head, n_head + n_kv) are keys.
// Lane l holds dims l + 32*j, so rope pairs (i, i + n_rot/2) stay in a lane
// when n_rot/2 is a multiple of 32.
kernel void attn_prep(
        constant AttnArgs & a [[buffer(0)]],
        device const float * qg      [[buffer(1)]],
        device const float * k       [[buffer(2)]],
        device const float * v       [[buffer(3)]],
        device const float * q_norm  [[buffer(4)]],
        device const float * k_norm  [[buffer(5)]],
        device const float * inv_freq [[buffer(6)]],
        device       float * q_out   [[buffer(7)]],
        device       half  * k_cache [[buffer(8)]],
        device       half  * v_cache [[buffer(9)]],
        uint2  tg   [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int MAXE = 16; // head_dim up to 512
    const int head = tg.x;
    const int t    = tg.y;
    const int hd   = a.hd;
    const int E    = hd/32;
    const int kv_dim = a.n_kv*hd;
    const int pos  = a.pos0 + t;

    const bool is_q = head < a.n_head;
    const int kh = head - a.n_head;

    device const float * src = is_q ? qg + (t*a.n_head + head)*2*hd : k + t*kv_dim + kh*hd;
    device const float * w   = is_q ? q_norm : k_norm;

    float vals[MAXE];
    float ss = 0.f;
    for (int j = 0; j < E; j++) {
        vals[j] = src[lane + 32*j];
        ss += vals[j]*vals[j];
    }
    ss = simd_sum(ss);
    // the order of ops::rms_norm, with correctly rounded divisions and sqrt
    const float mean = precise::divide(ss, float(hd));
    const float scale = precise::divide(1.0f, precise::sqrt(mean + a.eps));
    for (int j = 0; j < E; j++) {
        vals[j] *= scale * w[lane + 32*j];
    }

    const int m = a.n_rot/64; // rotated pairs per lane
    for (int j = 0; j < m; j++) {
        const int i = lane + 32*j;
        const float theta = pos * inv_freq[i];
        const float c = precise::cos(theta);
        const float s = precise::sin(theta);
        const float x0 = vals[j];
        const float x1 = vals[j + m];
        vals[j]     = x0*c - x1*s;
        vals[j + m] = x0*s + x1*c;
    }

    if (is_q) {
        device float * dst = q_out + (t*a.n_head + head)*hd;
        for (int j = 0; j < E; j++) {
            dst[lane + 32*j] = vals[j];
        }
    } else {
        device half * kd = k_cache + pos*kv_dim + kh*hd;
        device half * vd = v_cache + pos*kv_dim + kh*hd;
        device const float * vs = v + t*kv_dim + kh*hd;
        for (int j = 0; j < E; j++) {
            kd[lane + 32*j] = (half) vals[j];
            vd[lane + 32*j] = (half) vs[lane + 32*j];
        }
    }
}

// Split-K causal attention ("flash decoding"). A threadgroup takes one token,
// one kv head with its G query heads, and one range of `chunk` positions, so
// K / V rows are read from memory once for the G heads. Each simdgroup takes
// HPS of the heads (few enough to keep q and the accumulators in registers)
// and runs an online softmax over a strided part of the range; the
// simdgroups are then merged into one partial (max, sum, acc) per query head,
// and attn_reduce merges the ranges.
// Lane l holds the HD/32 contiguous dims [l*E, l*E + E).
template <int HD, int G, int HPS, int NSG>
kernel void attn_split(
        constant AttnArgs & a [[buffer(0)]],
        device const float * q       [[buffer(1)]],
        device const half  * k_cache [[buffer(2)]],
        device const half  * v_cache [[buffer(3)]],
        device       float * part_o  [[buffer(4)]],
        device       float * part_ml [[buffer(5)]],
        threadgroup  float * shm     [[threadgroup(0)]],
        uint3  tg   [[threadgroup_position_in_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    constexpr int E  = HD/32;
    constexpr int HG = G/HPS;   // head groups
    constexpr int PS = NSG/HG;  // simdgroups per head group, on strided positions
    static_assert(G % HPS == 0 && NSG % HG == 0, "attn_split geometry");
    const int hg = sg % HG;
    const int ps = sg / HG;
    const int kvh   = tg.x;
    const int split = tg.y;
    const int t     = tg.z;
    const int kv_dim = a.n_kv*HD;
    const int n_ctx = a.pos0 + t + 1;
    const int p0 = split*a.chunk;
    const int p1 = min(p0 + a.chunk, n_ctx);

    float qv[HPS][E];
    float acc[HPS][E];
    float m[HPS];
    float l[HPS];
    for (int h = 0; h < HPS; h++) {
        const int head = kvh*G + hg*HPS + h;
        device const float4 * qh = (device const float4 *) (q + (t*a.n_head + head)*HD + lane*E);
        for (int j = 0; j < E/4; j++) {
            const float4 v = qh[j] * a.scale;
            for (int i = 0; i < 4; i++) {
                qv[h][4*j + i]  = v[i];
                acc[h][4*j + i] = 0.f;
            }
        }
        m[h] = SCORE_FLOOR;
        l[h] = 0.f;
    }

    for (int p = p0 + ps; p < p1; p += PS) {
        device const half4 * kp = (device const half4 *) (k_cache + p*kv_dim + kvh*HD + lane*E);
        device const half4 * vp = (device const half4 *) (v_cache + p*kv_dim + kvh*HD + lane*E);
        float kf[E];
        float vf[E];
        for (int j = 0; j < E/4; j++) {
            const float4 kj = (float4) kp[j];
            const float4 vj = (float4) vp[j];
            for (int i = 0; i < 4; i++) {
                kf[4*j + i] = kj[i];
                vf[4*j + i] = vj[i];
            }
        }
        for (int h = 0; h < HPS; h++) {
            float s = 0.f;
            for (int j = 0; j < E; j++) {
                s += qv[h][j] * kf[j];
            }
            s = simd_sum(s);
            const float m_new = max(m[h], s);
            const float c  = exp(m[h] - m_new);
            const float pe = exp(s - m_new);
            l[h] = l[h]*c + pe;
            for (int j = 0; j < E; j++) {
                acc[h][j] = acc[h][j]*c + pe*vf[j];
            }
            m[h] = m_new;
        }
    }

    // merge the position strides: shm = [PS][G] maxima, [PS][G] sums, [G][HD] accumulator
    threadgroup float * sm = shm;
    threadgroup float * sl = sm + PS*G;
    threadgroup float * so = sl + PS*G;
    if (lane == 0) {
        for (int h = 0; h < HPS; h++) {
            sm[ps*G + hg*HPS + h] = m[h];
            sl[ps*G + hg*HPS + h] = l[h];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float f[HPS];
    for (int h = 0; h < HPS; h++) {
        float M = SCORE_FLOOR;
        for (int r = 0; r < PS; r++) {
            M = max(M, sm[r*G + hg*HPS + h]);
        }
        // an empty stride has kept its start value
        f[h] = m[h] == SCORE_FLOOR ? 0.f : exp(m[h] - M);
    }
    for (int r = 0; r < PS; r++) {
        if (ps == r) {
            for (int h = 0; h < HPS; h++) {
                for (int j = 0; j < E; j++) {
                    threadgroup float * o = so + (hg*HPS + h)*HD + lane*E + j;
                    *o = (r == 0 ? 0.f : *o) + f[h]*acc[h][j];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int part = (t*a.n_head + kvh*G)*a.splits + split; // + g*splits
    for (int i = tid; i < G*HD; i += NSG*32) {
        const int g = i / HD;
        part_o[(part + g*a.splits)*HD + i % HD] = so[i];
    }
    if (tid < G) {
        const int g = tid;
        float M = SCORE_FLOOR;
        for (int r = 0; r < PS; r++) {
            M = max(M, sm[r*G + g]);
        }
        float L = 0.f;
        for (int r = 0; r < PS; r++) {
            // an empty stride has kept its start value
            L += sm[r*G + g] == SCORE_FLOOR ? 0.f : sl[r*G + g]*exp(sm[r*G + g] - M);
        }
        part_ml[2*(part + g*a.splits)]     = M;
        part_ml[2*(part + g*a.splits) + 1] = L;
    }
}

// two heads per simdgroup: q and the accumulators of three or more spill
#define ATTN_SPLIT(hd, g, hps, nsg) \
template [[host_name("attn_split_" #hd "_" #g)]] kernel void attn_split<hd, g, hps, nsg>( \
    constant AttnArgs &, device const float *, device const half *, device const half *, \
    device float *, device float *, threadgroup float *, uint3, ushort, ushort, ushort);
ATTN_SPLIT(128, 1, 1, 2) ATTN_SPLIT(128, 2, 2, 2) ATTN_SPLIT(128, 4, 2, 2) ATTN_SPLIT(128, 6, 2, 3) ATTN_SPLIT(128, 8, 2, 4)
ATTN_SPLIT(256, 1, 1, 2) ATTN_SPLIT(256, 2, 2, 2) ATTN_SPLIT(256, 4, 2, 2) ATTN_SPLIT(256, 6, 2, 3) ATTN_SPLIT(256, 8, 2, 4)

// Merge the `splits` partials of one (head, token), one thread per dim, and
// gate the result by sigmoid(gate), the second half of each head of the
// joint q projection.
kernel void attn_reduce(
        constant AttnArgs & a [[buffer(0)]],
        device const float * part_o  [[buffer(1)]],
        device const float * part_ml [[buffer(2)]],
        device const float * qg      [[buffer(3)]],
        device       float * out     [[buffer(4)]],
        uint2  tg  [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    const int head = tg.x;
    const int t    = tg.y;
    const int hd   = a.hd;
    const int part = (t*a.n_head + head)*a.splits;

    // split 0 always holds position 0, so M is a real score
    float M = SCORE_FLOOR;
    for (int s = 0; s < a.splits; s++) {
        M = max(M, part_ml[2*(part + s)]);
    }
    float L = 0.f;
    float o = 0.f;
    for (int s = 0; s < a.splits; s++) {
        const float w = exp(part_ml[2*(part + s)] - M);
        L += w * part_ml[2*(part + s) + 1];
        o += w * part_o[(part + s)*hd + tid];
    }
    const float gate = qg[(t*a.n_head + head)*2*hd + hd + tid];
    out[(t*a.n_head + head)*hd + tid] = o / L * sigmoid(gate);
}

// Causal attention over a batch of tokens ("flash attention"), for prefill.
// A threadgroup takes R query rows of one kv head, the rows being the
// (token, head) pairs of its G query heads, head fastest, and walks the
// positions they see, C at a time:
// - S = Q·Kᵀ: each simdgroup computes QR x QC 8x8 tiles of the R x C scores,
//   so each K tile is loaded once per threadgroup, into threadgroup memory
// - online softmax, R/NSG rows per simdgroup, masked past each row's position
// - O += P·V: each simdgroup holds the R x HD/NSG slice of the output in
//   registers, so each V tile is loaded once per threadgroup and used for R/8
//   row tiles
// The output is gated as in attn_reduce. The loops over the register arrays
// must be unrolled: dynamically indexed, the arrays go to memory and the
// kernel is several times slower.
// The K / V caches must hold a multiple of C positions: the last range reads
// past the batch, and those positions are masked.
// SPLIT: the threadgroup takes the range of `chunk` positions tg.z (a multiple
// of C) and writes split-K partials as attn_split does, for attn_reduce,
// instead of the output: with few query rows (verify), so that the
// threadgroups fill the GPU while each K / V tile is read once for the R rows.
template <int HD, int R, int C, int NSG, int QR, bool SPLIT>
kernel void attn_flash(
        constant AttnArgs & a [[buffer(0)]],
        device const float * q       [[buffer(1)]],
        device const half  * k_cache [[buffer(2)]],
        device const half  * v_cache [[buffer(3)]],
        device       float * qg      [[buffer(4)]], // SPLIT: part_o
        device       float * out     [[buffer(5)]], // SPLIT: part_ml
        threadgroup  half  * shm     [[threadgroup(0)]],
        uint3  tg   [[threadgroup_position_in_grid]],
        uint3  ntg  [[threadgroups_per_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    constexpr int RT = R/8;                // row tiles
    constexpr int CT = C/8;                // position tiles
    constexpr int DT = HD/8;               // head dim tiles
    constexpr int QC = RT*CT/(NSG*QR);     // score position tiles per simdgroup
    constexpr int OD = DT/NSG;             // output head dim tiles per simdgroup
    constexpr int NQ = R/NSG;              // softmax rows per simdgroup
    constexpr int CL = C/32;               // positions per lane in the softmax
    static_assert(RT % QR == 0 && (RT/QR)*(CT/QC) == NSG && DT % NSG == 0 && R % NSG == 0 && C % 32 == 0,
                  "attn_flash geometry");

    threadgroup half  * sq = shm;                               // [R][HD] queries, scaled
    threadgroup float * ss = (threadgroup float *) (sq + R*HD); // [R][C] scores, then probabilities
    threadgroup float * sf = ss + R*C;                          // [R] rescale factors, then 1/sum

    const int G      = a.n_head/a.n_kv;
    const int kvh    = tg.y;
    const int rows   = a.n*G;
    const int kv_dim = a.n_kv*HD;
    const int row0   = (ntg.x - 1 - tg.x)*R; // the blocks that see the most positions first

    for (int i = tid; i < R*HD/4; i += NSG*32) {
        const int g = row0 + i/(HD/4);
        float4 v = 0.f;
        if (g < rows) {
            const int t = g/G;
            const int head = kvh*G + g%G;
            v = ((device const float4 *) (q + (t*a.n_head + head)*HD))[i%(HD/4)] * a.scale;
        }
        ((threadgroup half4 *) sq)[i] = (half4) v;
    }

    // softmax state of this simdgroup's rows, and the last position each sees;
    // rows past the batch see what the last one does
    int   last[NQ];
    float M[NQ];
    float L[NQ];
    FOR_UNROLL (int j = 0; j < NQ; j++) {
        last[j] = a.pos0 + min(row0 + sg*NQ + j, rows - 1)/G;
        M[j] = SCORE_FLOOR;
        L[j] = 0.f;
    }
    const int last_end = a.pos0 + (min(row0 + R, rows) - 1)/G + 1;
    const int begin = SPLIT ? (int) tg.z*a.chunk : 0;
    const int end   = SPLIT ? min(begin + a.chunk, last_end) : last_end;

    // this simdgroup's score tiles
    const int qr0 = (sg % (RT/QR))*QR;
    const int qc0 = (sg / (RT/QR))*QC;
    // a lane holds elements (lr, lc) and (lr, lc + 1) of each 8x8 matrix
    const int lr = (lane & 16)/4 + (lane & 6)/2;
    const int lc = (lane & 8)/2 + (lane & 1)*2;

    simdgroup_float8x8 mo[RT][OD];
    FOR_UNROLL (int rt = 0; rt < RT; rt++) {
        FOR_UNROLL (int k = 0; k < OD; k++) {
            mo[rt][k] = make_filled_simdgroup_matrix<float, 8>(0.f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int p0 = begin; p0 < end; p0 += C) {
        {
            simdgroup_float8x8 ms[QR][QC];
            FOR_UNROLL (int i = 0; i < QR; i++) {
                FOR_UNROLL (int j = 0; j < QC; j++) {
                    ms[i][j] = make_filled_simdgroup_matrix<float, 8>(0.f);
                }
            }
            device const half * pk = k_cache + (p0 + 8*qc0)*kv_dim + kvh*HD;
            _Pragma("clang loop unroll_count(4)") for (int d = 0; d < DT; d++) {
                simdgroup_half8x8 mq[QR];
                FOR_UNROLL (int i = 0; i < QR; i++) {
                    simdgroup_load(mq[i], sq + 8*(qr0 + i)*HD + 8*d, HD);
                }
                FOR_UNROLL (int j = 0; j < QC; j++) {
                    simdgroup_half8x8 mk;
                    simdgroup_load(mk, pk + 8*j*kv_dim + 8*d, kv_dim, 0, true);
                    FOR_UNROLL (int i = 0; i < QR; i++) {
                        simdgroup_multiply_accumulate(ms[i][j], mq[i], mk, ms[i][j]);
                    }
                }
            }
            FOR_UNROLL (int i = 0; i < QR; i++) {
                FOR_UNROLL (int j = 0; j < QC; j++) {
                    simdgroup_store(ms[i][j], ss + 8*(qr0 + i)*C + 8*(qc0 + j), C);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (int j = 0; j < NQ; j++) {
            const int r = sg*NQ + j;
            float s[CL];
            float m = M[j];
            FOR_UNROLL (int c = 0; c < CL; c++) {
                const int i = lane + 32*c;
                s[c] = p0 + i <= last[j] && p0 + i < end ? ss[r*C + i] : MASKED_SCORE;
                m = max(m, s[c]);
            }
            m = simd_max(m);
            const float f = exp(M[j] - m);
            float l = 0.f;
            FOR_UNROLL (int c = 0; c < CL; c++) {
                const float pe = exp(s[c] - m);
                ss[r*C + lane + 32*c] = pe;
                l += pe;
            }
            L[j] = L[j]*f + simd_sum(l);
            M[j] = m;
            if (lane == 0) {
                sf[r] = f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (int rt = 0; rt < RT; rt++) {
            const float f = sf[8*rt + lr];
            FOR_UNROLL (int k = 0; k < OD; k++) {
                thread auto & e = mo[rt][k].thread_elements();
                e[0] *= f;
                e[1] *= f;
            }
        }
        device const half * pv = v_cache + p0*kv_dim + kvh*HD + 8*sg*OD;
        FOR_UNROLL (int ct = 0; ct < CT; ct++) {
            simdgroup_half8x8 mv[OD];
            FOR_UNROLL (int k = 0; k < OD; k++) {
                simdgroup_load(mv[k], pv + 8*ct*kv_dim + 8*k, kv_dim);
            }
            FOR_UNROLL (int rt = 0; rt < RT; rt++) {
                simdgroup_float8x8 mp;
                simdgroup_load(mp, ss + 8*rt*C + 8*ct, C);
                FOR_UNROLL (int k = 0; k < OD; k++) {
                    simdgroup_multiply_accumulate(mo[rt][k], mp, mv[k], mo[rt][k]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (SPLIT) {
        device float * part_o  = qg;
        device float * part_ml = out;
        if (lane == 0) {
            FOR_UNROLL (int j = 0; j < NQ; j++) {
                const int g = row0 + sg*NQ + j;
                if (g < rows) {
                    const int part = ((g/G)*a.n_head + kvh*G + g%G)*a.splits + tg.z;
                    part_ml[2*part]     = M[j];
                    part_ml[2*part + 1] = L[j];
                }
            }
        }
        FOR_UNROLL (int rt = 0; rt < RT; rt++) {
            const int g = row0 + 8*rt + lr;
            if (g >= rows) {
                break;
            }
            const int part = ((g/G)*a.n_head + kvh*G + g%G)*a.splits + tg.z;
            device float * o = part_o + part*HD;
            FOR_UNROLL (int k = 0; k < OD; k++) {
                thread auto & e = mo[rt][k].thread_elements();
                const int i = 8*(sg*OD + k) + lc;
                o[i]     = e[0];
                o[i + 1] = e[1];
            }
        }
        return;
    }
    if (lane == 0) {
        FOR_UNROLL (int j = 0; j < NQ; j++) {
            sf[sg*NQ + j] = 1.f/L[j];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    FOR_UNROLL (int rt = 0; rt < RT; rt++) {
        const int g = row0 + 8*rt + lr;
        if (g >= rows) {
            break;
        }
        const int t = g/G;
        const int head = kvh*G + g%G;
        const float inv = sf[8*rt + lr];
        device const float * gate = qg + (t*a.n_head + head)*2*HD + HD;
        device float * o = out + (t*a.n_head + head)*HD;
        FOR_UNROLL (int k = 0; k < OD; k++) {
            thread auto & e = mo[rt][k].thread_elements();
            const int i = 8*(sg*OD + k) + lc;
            o[i]     = e[0] * inv * sigmoid(gate[i]);
            o[i + 1] = e[1] * inv * sigmoid(gate[i + 1]);
        }
    }
}

// 16 rows, 64 positions, 4 simdgroups of 2 x 2 score tiles: the fastest of the
// shapes tried on M2 Max for head size 256 (2.4x the split-K kernel)
#define ATTN_FLASH(name, hd, split) \
template [[host_name(name)]] kernel void attn_flash<hd, 16, 64, 4, 2, split>( \
    constant AttnArgs &, device const float *, device const half *, device const half *, \
    device float *, device float *, threadgroup half *, uint3, uint3, ushort, ushort, ushort);
ATTN_FLASH("attn_flash_128", 128, false) ATTN_FLASH("attn_flash_256", 256, false)
ATTN_FLASH("attn_flash_split_128", 128, true) ATTN_FLASH("attn_flash_split_256", 256, true)

// ---------------------------------------------------------------------------
// gated delta net

// The recurrent states of a delta net layer (conv window, delta rule state)
// have `slots` copies, `slot` being the current one. A pass updates it in
// place, or with `save` (verify, at most `slots` tokens) writes the state after
// token t to slot (slot + 1 + t) % slots, so that it can resume from any of them.

struct ConvArgs { int conv_dim; int n; int slot; int slots; int save; };

// causal depthwise conv (kernel 4) + silu, one thread per channel; the state
// holds the previous 3 inputs, as [slots][3][conv_dim]
kernel void ssm_conv(
        constant ConvArgs & a [[buffer(0)]],
        device const float * qkv   [[buffer(1)]],
        device const float * w     [[buffer(2)]],
        device       float * state [[buffer(3)]],
        device       float * out   [[buffer(4)]],
        uint c [[thread_position_in_grid]]) {
    const int cd = a.conv_dim;
    if ((int) c >= cd) {
        return;
    }
    device float * st = state + a.slot*3*cd + c;
    float s0 = st[0], s1 = st[cd], s2 = st[2*cd];
    const float4 wc = *(device const float4 *) (w + 4*c);
    for (int t = 0; t < a.n; t++) {
        const float x = qkv[t*cd + c];
        const float acc = wc[0]*s0 + wc[1]*s1 + wc[2]*s2 + wc[3]*x;
        out[t*cd + c] = silu(acc);
        s0 = s1; s1 = s2; s2 = x;
        if (a.save) {
            device float * sv = state + ((a.slot + 1 + t) % a.slots)*3*cd + c;
            sv[0] = s0; sv[cd] = s1; sv[2*cd] = s2;
        }
    }
    if (!a.save) {
        st[0] = s0; st[cd] = s1; st[2*cd] = s2;
    }
}

struct GdnArgs {
    int   n;        // tokens
    int   hk;       // key heads
    int   hv;       // value heads
    int   conv_dim;
    float eps;
    int   slot;     // state slots, see ssm_conv
    int   slots;
    int   save;
};

// One simdgroup per state row (a value index), NSG rows per threadgroup; each
// lane holds 4 consecutive key elements (head size 128). The state row j holds
// S[:, j]. Fused: l2 norm of q and k, decay = exp(softplus(alpha + dt) * a),
// beta = sigmoid(beta). Value heads are in tiled order: key head = head % hk.
template<short NSG>
kernel void gdn(
        constant GdnArgs & a [[buffer(0)]],
        device const float * conv_out [[buffer(1)]],
        device const float * alpha    [[buffer(2)]],
        device const float * beta     [[buffer(3)]],
        device const float * dt_bias  [[buffer(4)]],
        device const float * a_neg    [[buffer(5)]],
        device       float * state    [[buffer(6)]],
        device       float * out      [[buffer(7)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]]) {
    constexpr int SD = 128;
    const uint tx = tpitg.x;
    const uint ty = tpitg.y;
    const int head = tgpig.y;
    const int row  = tgpig.x*NSG + ty;
    const int kh   = head % a.hk;

    // this thread's elements, at this offset in every slot
    device float * s_ptr = state + (head*SD + row)*SD + 4*tx;
    const long slot_size = (long) a.hv*SD*SD;
    float4 ls = *(device float4 *) (s_ptr + a.slot*slot_size);

    const float scale = rsqrt((float) SD);
    const float dt = dt_bias[head];
    const float an = a_neg[head];

    for (int t = 0; t < a.n; t++) {
        device const float * base = conv_out + t*a.conv_dim;
        float4 q4 = *(device const float4 *) (base + kh*SD + 4*tx);
        float4 k4 = *(device const float4 *) (base + a.hk*SD + kh*SD + 4*tx);
        const float vj = base[2*a.hk*SD + head*SD + row];

        q4 *= rsqrt(simd_sum(dot(q4, q4)) + a.eps);
        k4 *= rsqrt(simd_sum(dot(k4, k4)) + a.eps);

        const float x  = alpha[t*a.hv + head] + dt;
        const float sp = x > 20.f ? x : precise::log(1.f + precise::exp(x));
        const float b  = sigmoid(beta[t*a.hv + head]);

        ls *= precise::exp(sp * an);
        const float s_k = simd_sum(dot(ls, k4));
        const float d = (vj - s_k) * b;
        ls += k4 * d;
        const float y = simd_sum(dot(ls, q4));
        if (tx == 0) {
            out[(t*a.hv + head)*SD + row] = y * scale;
        }
        if (a.save) {
            *(device float4 *) (s_ptr + ((a.slot + 1 + t) % a.slots)*slot_size) = ls;
        }
    }

    if (!a.save) {
        *(device float4 *) (s_ptr + a.slot*slot_size) = ls;
    }
}

template [[host_name("gdn_128")]] kernel void gdn<4>(constant GdnArgs &, device const float *, device const float *, device const float *, device const float *, device const float *, device float *, device float *, uint3, uint3);

// ---------------------------------------------------------------------------
// mixture of experts. A pass of n tokens through k experts each makes n*k
// (token, expert) pairs, pair t*k + j for the j-th expert of token t:
// `moe_route` picks them, `moe_sort` groups them by expert into tiles, of 32
// pairs for `mm_id` and of up to 8 for `mmv_id` (an expert's last few), which
// multiply each tile by its expert's matrix (the weights read once per tile),
// and `moe_combine` sums the pairs of each token.

struct RouteArgs { int n_expert; int k; };

// the larger of two (probability, expert), the lower expert among equals
inline void best_of(thread float & v, thread int & i, float v2, int i2) {
    if (v2 > v || (v2 == v && i2 < i)) {
        v = v2;
        i = i2;
    }
}

// One threadgroup per token: the softmax of its router logits, its k experts
// of highest probability, and their probabilities over their sum. `p` holds
// the probabilities, a chosen expert's set to -1.
kernel void moe_route(
        constant RouteArgs & args [[buffer(0)]],
        device const float * logits [[buffer(1)]],
        device int   * ids     [[buffer(2)]],
        device float * weights [[buffer(3)]],
        threadgroup float * p [[threadgroup(0)]],
        uint   tg   [[threadgroup_position_in_grid]],
        ushort tid  [[thread_position_in_threadgroup]],
        ushort ntg  [[threads_per_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red_v[32];
    threadgroup int   red_i[32];
    const int ne = args.n_expert;
    const short nsg = (ntg + 31)/32;
    device const float * l = logits + tg*ne;

    float m = SCORE_FLOOR;
    for (int i = tid; i < ne; i += ntg) {
        m = max(m, l[i]);
    }
    m = simd_max(m);
    if (lane == 0) red_v[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = red_v[0];
    for (short s = 1; s < nsg; s++) m = max(m, red_v[s]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum = 0.f;
    for (int i = tid; i < ne; i += ntg) {
        p[i] = exp(l[i] - m);
        sum += p[i];
    }
    sum = simd_sum(sum);
    if (lane == 0) red_v[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = 0.f;
    for (short s = 0; s < nsg; s++) sum += red_v[s];
    const float inv = 1.f/sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tid; i < ne; i += ntg) {
        p[i] *= inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = 0.f;
    for (int j = 0; j < args.k; j++) {
        float v = -2.f;
        int   bi = ne;
        for (int i = tid; i < ne; i += ntg) {
            best_of(v, bi, p[i], i);
        }
        for (ushort o = 16; o > 0; o /= 2) {
            best_of(v, bi, simd_shuffle_down(v, o), simd_shuffle_down(bi, o));
        }
        if (lane == 0) {
            red_v[sg] = v;
            red_i[sg] = bi;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        v = red_v[0];
        bi = red_i[0];
        for (short s = 1; s < nsg; s++) best_of(v, bi, red_v[s], red_i[s]);
        total += v;
        if (tid == 0) {
            ids[tg*args.k + j] = bi;
            weights[tg*args.k + j] = v;
            p[bi] = -1.f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
    if (tid == 0) {
        const float s = 1.f/max(total, 6.103515625e-5f);
        for (int j = 0; j < args.k; j++) {
            weights[tg*args.k + j] *= s;
        }
    }
}

struct SortArgs { int n_pairs; int n_expert; int mm_min; int mv_base; };

// One threadgroup: the pairs in order of expert (`order`, by pair index within
// an expert), and the tiles of an expert's pairs, as (expert, first in
// `order`, pairs). Tiles of 32 for `mm_id`: the full ones, and the last when
// it has `mm_min` pairs or more; the pairs left in tiles of 8 for `mmv_id`,
// from `tiles[mv_base]`. `n_tiles` gets the number of each. `tmp` holds
// 4 * n_expert counters.
kernel void moe_sort(
        constant SortArgs & args [[buffer(0)]],
        device const int * ids   [[buffer(1)]],
        device uint      * order [[buffer(2)]],
        device uint4     * tiles [[buffer(3)]],
        device uint      * n_tiles [[buffer(4)]],
        threadgroup uint * tmp [[threadgroup(0)]],
        ushort tid [[thread_position_in_threadgroup]],
        ushort ntg [[threads_per_threadgroup]]) {
    const int ne = args.n_expert;
    threadgroup uint * count = tmp;
    threadgroup uint * start = tmp + ne;
    threadgroup uint * mm0 = tmp + 2*ne;
    threadgroup uint * mv0 = tmp + 3*ne;
    for (int e = tid; e < ne; e += ntg) {
        uint c = 0;
        for (int p = 0; p < args.n_pairs; p++) {
            c += ids[p] == e;
        }
        count[e] = c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // pairs of an expert in tiles of 32 for mm_id, the rest in tiles of 8
    const uint mm_min = args.mm_min;
    #define MM_PAIRS(c) ((c) % 32 >= mm_min ? (c) : (c) / 32 * 32)
    if (tid == 0) {
        uint s = 0, tm = 0, tv = 0;
        for (int e = 0; e < ne; e++) {
            const uint c = count[e], m = MM_PAIRS(c);
            start[e] = s;
            mm0[e] = tm;
            mv0[e] = tv;
            s += c;
            tm += (m + 31)/32;
            tv += (c - m + 7)/8;
        }
        n_tiles[0] = tm;
        n_tiles[1] = tv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = tid; e < ne; e += ntg) {
        const uint c = count[e];
        if (c == 0) continue;
        uint s = start[e];
        for (int p = 0; p < args.n_pairs; p++) {
            if (ids[p] == e) order[s++] = p;
        }
        const uint m = MM_PAIRS(c);
        for (uint i = 0; i < m; i += 32) {
            tiles[mm0[e] + i/32] = uint4(e, start[e] + i, min(32u, m - i), 0);
        }
        for (uint i = m; i < c; i += 8) {
            tiles[args.mv_base + mv0[e] + (i - m)/8] = uint4(e, start[e] + i, min(8u, c - i), 0);
        }
    }
    #undef MM_PAIRS
}

struct MatIdArgs {
    int   k;       // input width
    int   rows;    // weight rows of an expert = output width
    int   x_div;   // pair p reads input row p / x_div
    int   pad;
    ulong w_row_bytes;
    ulong x_row_bytes;
    ulong stride;  // bytes from an expert's matrix to the next
};

// `mmv` (one row fragment, one input fragment, 2 simdgroups, 4 spans per
// step) over the pairs of a tile: the weights of its expert, the input rows
// of its pairs, an output row per pair
template<typename block_q>
kernel void mmv_id(
        constant MatIdArgs & args [[buffer(0)]],
        device const char  * src0 [[buffer(1)]],
        device const char  * src1 [[buffer(2)]],
        device       float * dst  [[buffer(3)]],
        device const uint  * order [[buffer(4)]],
        device const uint4 * tiles [[buffer(5)]],
        device const uint  * n_tiles [[buffer(6)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    if (tgpig.y >= n_tiles[0]) {
        return;
    }
    const uint4 tile = tiles[tgpig.y];
    const short qid = tiisg/4;
    const short fm = (qid & 4) + (tiisg/2)%4;
    const short fn = (qid & 2)*2 + (tiisg%2)*2;
    const short U = 4;

    const int r0 = (tgpig.x*2 + sgitg)*8;
    const int nt = tile.z;
    const int ns = args.k/32;
    src0 += tile.x*args.stride;

    device const float * y[2];
    FOR_UNROLL (short i = 0; i < 2; i++) {
        const uint p = order[tile.y + min(fn + i, nt - 1)];
        y[i] = (device const float *) (src1 + (p/args.x_div)*args.x_row_bytes) + fm;
    }
    device const block_q * x = (device const block_q *) (src0 + min(r0 + fm, args.rows - 1)*args.w_row_bytes);

    simdgroup_float8x8 c = make_filled_simdgroup_matrix<float, 8>(0.f);
    for (int is = 0; is < ns; is += U) {
        float2 w[U][4];
        FOR_UNROLL (short u = 0; u < U; u++) {
            deq_span(x, is + u, fn, w[u]);
        }
        FOR_UNROLL (short u = 0; u < U; u++) {
            FOR_UNROLL (short j = 0; j < 4; j++) {
                simdgroup_float8x8 a, b;
                b.thread_elements()[0] = y[0][32*(is + u) + 8*j];
                b.thread_elements()[1] = y[1][32*(is + u) + 8*j];
                a.thread_elements()[0] = w[u][j][0];
                a.thread_elements()[1] = w[u][j][1];
                simdgroup_multiply_accumulate(c, a, b, c);
            }
        }
    }

    const int row = r0 + fm;
    if (row < args.rows) {
        if (fn < nt)     dst[order[tile.y + fn]*args.rows + row]     = c.thread_elements()[0];
        if (fn + 1 < nt) dst[order[tile.y + fn + 1]*args.rows + row] = c.thread_elements()[1];
    }
}

#define MMV_ID(name, block) \
template [[host_name(name)]] kernel void mmv_id<block>(constant MatIdArgs &, device const char *, device const char *, device float *, device const uint *, device const uint4 *, device const uint *, uint3, ushort, ushort);
MMV_ID("mmv_id_f32",  float4x4)
MMV_ID("mmv_id_f16",  half4x4)
MMV_ID("mmv_id_bf16", block_bf16)
MMV_ID("mmv_id_q4_0", block_q4_0)
MMV_ID("mmv_id_q4_1", block_q4_1)
MMV_ID("mmv_id_q8_0", block_q8_0)
MMV_ID("mmv_id_q4_K", block_q4_K)
MMV_ID("mmv_id_q5_K", block_q5_K)
MMV_ID("mmv_id_q6_K", block_q6_K)

// `mm` over a tile of up to 32 pairs: the
// weights of its expert, the input rows of its pairs, an output row per pair.
// A threadgroup per tile and 64 weight rows, the tiles along x.
template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
kernel void mm_id(
        constant MatIdArgs & args [[buffer(0)]],
        device const char  * src0 [[buffer(1)]],
        device const char  * src1 [[buffer(2)]],
        device       float * dst  [[buffer(3)]],
        device const uint  * order [[buffer(4)]],
        device const uint4 * tiles [[buffer(5)]],
        device const uint  * n_tiles [[buffer(6)]],
        threadgroup  char  * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    if (tgpig.x >= n_tiles[0]) {
        return;
    }
    const uint4 tile = tiles[tgpig.x];
    constexpr int NL0 = 2;
    constexpr int NL1 = 4;

    const int ne0 = args.rows;
    const int r0 = tgpig.y*MM_NR0;
    const short nr0 = (ne0 - r0 < MM_NR0) ? (ne0 - r0) : MM_NR0;
    const short nr1 = tile.z;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    device const block_q * x = (device const block_q *)(src0 + tile.x*args.stride + args.w_row_bytes*(r0 + lr0)) + (tiitg % NL0)/nl;
    const uint p = order[tile.y + lr1];
    device const float * y = (device const float *)(src1 + (p/args.x_div)*args.x_row_bytes + 4*8*(tiitg % NL1));

    simdgroup_float8x8 mc[8];
    mm_tile<block_q, nl, dequantize_func>(args.k, x, y, shmem, tiitg, sgitg, mc);
    mm_stage(shmem, sgitg, mc);

    // each pair's row, 64 weight rows at a time
    threadgroup const float * C = (threadgroup const float *) shmem;
    for (int i = tiitg; i < nr1*MM_NR0; i += 128) {
        const int j = i/MM_NR0, r = i%MM_NR0;
        if (r < nr0) {
            dst[order[tile.y + j]*ne0 + r0 + r] = C[i];
        }
    }
}

typedef void (mm_id_t)(constant MatIdArgs &, device const char *, device const char *, device float *, device const uint *, device const uint4 *, device const uint *, threadgroup char *, uint3, ushort, ushort);

template [[host_name("mm_id_f32")]]  kernel mm_id_t mm_id<float4x4,   1,     dequantize_f32>;
template [[host_name("mm_id_f16")]]  kernel mm_id_t mm_id<half4x4,    1,     dequantize_f16>;
template [[host_name("mm_id_bf16")]] kernel mm_id_t mm_id<block_bf16, 1,     dequantize_bf16>;
template [[host_name("mm_id_q4_0")]] kernel mm_id_t mm_id<block_q4_0, 2,     dequantize_q4_0>;
template [[host_name("mm_id_q4_1")]] kernel mm_id_t mm_id<block_q4_1, 2,     dequantize_q4_1>;
template [[host_name("mm_id_q8_0")]] kernel mm_id_t mm_id<block_q8_0, 2,     dequantize_q8_0>;
template [[host_name("mm_id_q4_K")]] kernel mm_id_t mm_id<block_q4_K, QK_NL, dequantize_q4_K>;
template [[host_name("mm_id_q5_K")]] kernel mm_id_t mm_id<block_q5_K, QK_NL, dequantize_q5_K>;
template [[host_name("mm_id_q6_K")]] kernel mm_id_t mm_id<block_q6_K, QK_NL, dequantize_q6_K>;

struct CombineArgs { int n_embd; int k; int shared; };

// x[t] += sum_j weights[t*k + j] * out[t*k + j] + sigmoid(gate[t]) * shared[t]
kernel void moe_combine(
        constant CombineArgs & args [[buffer(0)]],
        device float       * x       [[buffer(1)]],
        device const float * out     [[buffer(2)]],
        device const float * weights [[buffer(3)]],
        device const float * shared  [[buffer(4)]],
        device const float * gate    [[buffer(5)]],
        uint2 gid [[thread_position_in_grid]]) {
    const int d = gid.x, t = gid.y;
    if (d >= args.n_embd) {
        return;
    }
    float acc = 0.f;
    for (int j = 0; j < args.k; j++) {
        acc += weights[t*args.k + j]*out[(t*args.k + j)*args.n_embd + d];
    }
    if (args.shared) {
        acc += sigmoid(gate[t])*shared[t*args.n_embd + d];
    }
    x[t*args.n_embd + d] += acc;
}
