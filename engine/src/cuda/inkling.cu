// ---------------------------------------------------------------------------
// inkling2_loop (Inkling-2): included at the end of kernels.cu, whose helpers
// it uses. See engine/src/model/inkling.rs for the math these kernels follow,
// and cuda/inkling.rs for how they are launched. Like the delta nets', the
// kernels that depend on the position or the slot of the conv states read
// them from the PassParams in device memory, so that a pass replays as a
// CUDA graph.

#define INK_MAXS 8
#define INK_MAXQ 4

// The depth mix of one row (a block per row): the states, at float offsets
// `off[s]` of `stack` (rows of `e`), the last one optionally rms(cur) of the
// row of `cur`, written there first; then per query q of `query` ([n_q][e])
// the softmax of the states' dot products with it times `scale`, and the
// states weighed by it, written `copies` times from output slot q * copies
// (slots `ostride` floats apart), and with `out_norm` rms-normalized and
// scaled by row q of `w`. No query: only the norm.
struct InkMixArgs {
    u64 stack; u64 cur; u64 query; u64 w; u64 out;
    int e;
    int n_states;
    int n_q;
    int copies;
    float scale;
    float eps;      // of rms(cur)
    int norm_cur;
    int out_norm;
    float out_eps;
    int in_row0;    // the row of states and cur of block 0
    int out_row0;   // the row of the outputs of block 0
    int ostride;
    int off[INK_MAXS];
};

extern "C" __global__ void ink_mix(const InkMixArgs a) {
    __shared__ float dots[32][INK_MAXQ * INK_MAXS];
    const int e = a.e, S = a.n_states, tid = threadIdx.x, nt = blockDim.x;
    const int lane = tid & 31, w = tid >> 5, nw = nt >> 5;
    const size_t r = (size_t) a.in_row0 + blockIdx.x;
    float *stack = (float *) a.stack;

    // each thread reads back only the elements it wrote
    if (a.norm_cur) {
        const float *x = (const float *) a.cur + r * e;
        float *c = stack + a.off[S - 1] + r * e;
        float ss = 0.f;
        for (int i = tid; i < e; i += nt) ss += x[i] * x[i];
        ss = block_sum(ss);
        const float scale = 1.f / sqrtf(ss / (float) e + a.eps);
        for (int i = tid; i < e; i += nt) c[i] = x[i] * scale;
    }
    if (a.n_q == 0) return;

    const float *query = (const float *) a.query;
    float d[INK_MAXQ * INK_MAXS];
#pragma unroll
    for (int k = 0; k < INK_MAXQ * INK_MAXS; k++) d[k] = 0.f;
    for (int i = tid; i < e; i += nt) {
#pragma unroll
        for (int s = 0; s < INK_MAXS; s++) {
            if (s < S) {
                const float v = stack[a.off[s] + r * e + i];
#pragma unroll
                for (int q = 0; q < INK_MAXQ; q++) {
                    if (q < a.n_q) d[q * INK_MAXS + s] += v * query[q * e + i];
                }
            }
        }
    }
#pragma unroll
    for (int k = 0; k < INK_MAXQ * INK_MAXS; k++) {
        const float v = warp_sum(d[k]);
        if (lane == 0) dots[w][k] = v;
    }
    __syncthreads();
    // the softmax of each query's scores, in every thread
    float p[INK_MAXQ * INK_MAXS];
#pragma unroll
    for (int q = 0; q < INK_MAXQ; q++) {
        float m = -INFINITY;
#pragma unroll
        for (int s = 0; s < INK_MAXS; s++) {
            float v = 0.f;
            for (int g = 0; g < nw; g++) v += dots[g][q * INK_MAXS + s];
            v *= a.scale;
            p[q * INK_MAXS + s] = v;
            if (s < S) m = fmaxf(m, v);
        }
        float sum = 0.f;
#pragma unroll
        for (int s = 0; s < INK_MAXS; s++) {
            const float v = s < S ? expf(p[q * INK_MAXS + s] - m) : 0.f;
            p[q * INK_MAXS + s] = v;
            sum += v;
        }
        const float inv = 1.f / sum;
#pragma unroll
        for (int s = 0; s < INK_MAXS; s++) p[q * INK_MAXS + s] *= inv;
    }

    const float *wn = (const float *) a.w;
    for (int q = 0; q < a.n_q; q++) {
        float *o = (float *) a.out + (size_t) q * a.copies * a.ostride + ((size_t) a.out_row0 + blockIdx.x) * e;
        float ss = 0.f;
        for (int i = tid; i < e; i += nt) {
            float v = 0.f;
#pragma unroll
            for (int s = 0; s < INK_MAXS; s++) {
                if (s < S) v += p[q * INK_MAXS + s] * stack[a.off[s] + r * e + i];
            }
            o[i] = v;
            ss += v * v;
        }
        if (a.out_norm) {
            ss = block_sum(ss);
            const float scale = 1.f / sqrtf(ss / (float) e + a.out_eps);
            const float *wq = wn + q * e;
            for (int i = tid; i < e; i += nt) o[i] = o[i] * (scale * wq[i]);
        }
        for (int c = 1; c < a.copies; c++) {
            float *oc = o + (size_t) c * a.ostride;
            for (int i = tid; i < e; i += nt) oc[i] = o[i];
        }
    }
}

// rms norm without weight of rows of `n`, input rows `istride` floats apart,
// output rows `ostride`; a block per row
struct InkRmsArgs { u64 x; u64 out; int n; float eps; int istride; int ostride; };

extern "C" __global__ void ink_rms(const InkRmsArgs a) {
    const int tid = threadIdx.x, nt = blockDim.x;
    const float *x = (const float *) a.x + (size_t) blockIdx.x * a.istride;
    float *o = (float *) a.out + (size_t) blockIdx.x * a.ostride;
    float ss = 0.f;
    for (int i = tid; i < a.n; i += nt) ss += x[i] * x[i];
    ss = block_sum(ss);
    const float scale = 1.f / sqrtf(ss / (float) a.n + a.eps);
    for (int i = tid; i < a.n; i += nt) o[i] = x[i] * scale;
}

// a loop controller's output: (x - tanh(f) x) + tanh(g) m, out may be x
struct InkCtrlArgs { u64 x; u64 g; u64 f; u64 m; u64 out; int n; int pad; };

extern "C" __global__ void ink_control(const InkCtrlArgs a) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= a.n) return;
    const float x = ((const float *) a.x)[i];
    const float f = ((const float *) a.f)[i], g = ((const float *) a.g)[i];
    ((float *) a.out)[i] = (x - tanhf(f) * x) + tanhf(g) * ((const float *) a.m)[i];
}

// The short causal conv (4 taps, the last on the current token) with a
// residual, of `n` rows of `width` channels: y = (x + conv) * scale, plus
// with `dyn` each token's own taps, `dyn` [n][dyn_stride] at (channel /
// head) * 4 + lag, lag 0 the current token. The state holds the previous 3
// inputs of `channels` channels, as [slots][3][channels], this conv's from
// `offset`; slots as for ssm_conv. With `add`, out += y, else out = y.
struct InkConvArgs {
    u64 x; u64 w; u64 state; u64 out; u64 dyn; u64 params;
    int width;
    int n;
    int channels;
    int offset;
    int slots;
    int save;
    int add;
    int has_dyn;
    int head;
    int dyn_stride;
    float scale;
    int pad;
};

// the taps of channel c on its inputs x3 (3 back) .. x0 (current), as the
// CPU sums them: the static ones oldest first, then the dynamic ones by lag
__device__ __forceinline__ float ink_taps(const InkConvArgs &a, int c, int t, float x3, float x2, float x1,
                                          float x0) {
    const float4 wc = ((const float4 *) a.w)[c];
    float acc = 0.f;
    acc += wc.x * x3;
    acc += wc.y * x2;
    acc += wc.z * x1;
    acc += wc.w * x0;
    if (a.has_dyn) {
        const float *d = (const float *) a.dyn + (size_t) t * a.dyn_stride + (c / a.head) * 4;
        acc += x0 * d[0];
        acc += x1 * d[1];
        acc += x2 * d[2];
        acc += x3 * d[3];
    }
    return (x0 + acc) * a.scale;
}

// a thread per channel through the tokens; in place allowed (out == x)
extern "C" __global__ void ink_conv(const InkConvArgs a) {
    const int W = a.width, C = a.channels;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= W) return;
    const int slot = ((const PassParams *) a.params)->slot;
    float *state = (float *) a.state;
    float *st = state + (size_t) slot * 3 * C + a.offset + c;
    const float *x = (const float *) a.x;
    float *out = (float *) a.out;
    float s0 = st[0], s1 = st[C], s2 = st[2 * C];
    for (int t = 0; t < a.n; t++) {
        const float xt = x[(size_t) t * W + c];
        const float y = ink_taps(a, c, t, s0, s1, s2, xt);
        if (a.add) {
            out[(size_t) t * W + c] += y;
        } else {
            out[(size_t) t * W + c] = y;
        }
        s0 = s1;
        s1 = s2;
        s2 = xt;
        if (a.save) {
            float *sv = state + (size_t) ((slot + 1 + t) % a.slots) * 3 * C + a.offset + c;
            sv[0] = s0;
            sv[C] = s1;
            sv[2 * C] = s2;
        }
    }
    if (!a.save) {
        st[0] = s0;
        st[C] = s1;
        st[2 * C] = s2;
    }
}

// ink_conv of long passes without saved states: a thread per (channel,
// token), the inputs before the pass from the state; `out` is not `x`. The
// state then takes the pass's last 3 inputs, by ink_conv_state (n >= 3).
extern "C" __global__ void ink_conv_par(const InkConvArgs a) {
    const int W = a.width, C = a.channels;
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c >= W) return;
    const int slot = ((const PassParams *) a.params)->slot;
    const float *st = (const float *) a.state + (size_t) slot * 3 * C + a.offset + c;
    const float *x = (const float *) a.x;
    auto in = [&](int j) { return j < 0 ? st[(3 + j) * C] : x[(size_t) j * W + c]; };
    const float y = ink_taps(a, c, t, in(t - 3), in(t - 2), in(t - 1), in(t));
    float *o = (float *) a.out + (size_t) t * W + c;
    if (a.add) {
        *o += y;
    } else {
        *o = y;
    }
}

extern "C" __global__ void ink_conv_state(const InkConvArgs a) {
    const int W = a.width, C = a.channels;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= W) return;
    const int slot = ((const PassParams *) a.params)->slot;
    float *st = (float *) a.state + (size_t) slot * 3 * C + a.offset + c;
    const float *x = (const float *) a.x;
    for (int j = 0; j < 3; j++) st[j * C] = x[(size_t) (a.n - 3 + j) * W + c];
}

// ---------------------------------------------------------------------------
// inkling attention: no rope, a relative position bias per (token, head,
// distance), minus a sink logit per (token, head) whose own softmax logit is
// zero, an output gate per channel

struct InkAttnArgs {
    u64 q; u64 k; u64 v; u64 r;
    u64 q_clip; u64 r_clip;
    u64 k_cache; u64 v_cache;  // f16, position p at row p & mask
    u64 sink; u64 rel_bias; u64 gate; u64 out;
    u64 part_o; u64 part_ml;
    u64 params;
    int n_head;
    int n_kv;
    int hd;
    int n_rel;
    float eps;      // of the values' rms norm
    float scale;    // of the scores
    float clip;     // the L2 norm of queries and keys at most
    float rel_clip; // and of the relative position channels
    int full;       // tau scales the queries and relative position channels
    int floor_;     // tau = 1 + alpha ln(max(1, (pos + 1) / floor))
    float alpha;
    int window;     // positions a token sees, its own included; 0: all
    int mask;       // -1, or the rows of the ring - 1
    int ext;        // distances with a bias row; farther, none
    int splits;     // ranges of positions per (token, head)
    int pad;
};

#define INK_MAXE 8 // head size up to 256

// Per (head, token), one warp (lane l holding dims l + 32j). Heads [0,
// n_head): q clipped (scaled by q_clip first) and times tau, in place.
// [n_head, n_head + n_kv): k clipped, v rms-normalized, to the f16 caches at
// the token's position. [n_head + n_kv, 2 n_head + n_kv): r clipped (scaled
// by r_clip first) and times tau, in place.
extern "C" __global__ void ink_attn_prep(const InkAttnArgs a) {
    const int head = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    const int hd = a.hd, E = hd / 32, kv_dim = a.n_kv * hd;
    const int pos = ((const PassParams *) a.params)->pos0 + t;
    float tau = 1.f;
    if (a.full) tau = logf(fmaxf(((float) pos + 1.f) * (1.f / (float) a.floor_), 1.f)) * a.alpha + 1.f;

    if (head < a.n_head) {
        float *src = (float *) a.q + (size_t) (t * a.n_head + head) * hd;
        const float *sc = (const float *) a.q_clip + head * hd;
        float vals[INK_MAXE];
        float ss = 0.f;
#pragma unroll
        for (int j = 0; j < INK_MAXE; j++) {
            if (j < E) {
                vals[j] = src[lane + 32 * j] * sc[lane + 32 * j];
                ss += vals[j] * vals[j];
            }
        }
        ss = warp_sum(ss);
        const float inv = 1.f / fmaxf(sqrtf(ss), a.clip);
#pragma unroll
        for (int j = 0; j < INK_MAXE; j++) {
            if (j < E) src[lane + 32 * j] = vals[j] * inv * a.clip * tau;
        }
    } else if (head < a.n_head + a.n_kv) {
        const int kh = head - a.n_head;
        const float *ks = (const float *) a.k + (size_t) t * kv_dim + kh * hd;
        const float *vs = (const float *) a.v + (size_t) t * kv_dim + kh * hd;
        float kv[INK_MAXE], vv[INK_MAXE];
        float sk = 0.f, sv = 0.f;
#pragma unroll
        for (int j = 0; j < INK_MAXE; j++) {
            if (j < E) {
                kv[j] = ks[lane + 32 * j];
                vv[j] = vs[lane + 32 * j];
                sk += kv[j] * kv[j];
                sv += vv[j] * vv[j];
            }
        }
        sk = warp_sum(sk);
        sv = warp_sum(sv);
        const float inv = 1.f / fmaxf(sqrtf(sk), a.clip);
        const float vscale = 1.f / sqrtf(sv / (float) hd + a.eps);
        const size_t row = (size_t) (pos & a.mask) * kv_dim + kh * hd;
        __half *kd = (__half *) a.k_cache + row;
        __half *vd = (__half *) a.v_cache + row;
#pragma unroll
        for (int j = 0; j < INK_MAXE; j++) {
            if (j < E) {
                kd[lane + 32 * j] = __float2half(kv[j] * inv * a.clip);
                vd[lane + 32 * j] = __float2half(vv[j] * vscale);
            }
        }
    } else {
        const int rh = head - a.n_head - a.n_kv, nr = a.n_rel;
        float *src = (float *) a.r + (size_t) (t * a.n_head + rh) * nr;
        const float x = lane < nr ? src[lane] * ((const float *) a.r_clip)[rh * nr + lane] : 0.f;
        const float ss = warp_sum(x * x);
        const float inv = 1.f / fmaxf(sqrtf(ss), a.rel_clip);
        if (lane < nr) src[lane] = x * inv * a.rel_clip * tau;
    }
}

#define INK_ATTN_WARPS 8

// Attention of one (query head, token) over a range of the positions it
// sees, range blockIdx.z of `splits`: each warp an online softmax over every
// INK_ATTN_WARPS-th position (lane l holding dims l + 32j, and channel l of
// the relative position channels, so that the bias's dot product joins the
// score's reduction), then the warps merged. One range: the output, with the
// zero sink logit, gated by sigmoid(gate). Else the range's (max, sum) and
// its output weighed by exp(m - max), for ink_attn_reduce.
extern "C" __global__ void __launch_bounds__(WARP * INK_ATTN_WARPS) ink_attn(const InkAttnArgs a) {
    __shared__ float sm[INK_ATTN_WARPS], sl[INK_ATTN_WARPS];
    __shared__ float sacc[INK_ATTN_WARPS][32 * INK_MAXE];
    const int head = blockIdx.x, t = blockIdx.y, split = blockIdx.z;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int hd = a.hd, E = hd / 32, kv_dim = a.n_kv * hd, nr = a.n_rel;
    const int kvh = head / (a.n_head / a.n_kv);
    const int pos = ((const PassParams *) a.params)->pos0 + t;
    const int n_ctx = pos + 1;
    const int first = a.window > 0 ? max(n_ctx - a.window, 0) : 0;
    const int chunk = (n_ctx - first + a.splits - 1) / a.splits;
    const int p0 = first + split * chunk, p1 = min(p0 + chunk, n_ctx);
    const int row = t * a.n_head + head;

    const float *qh = (const float *) a.q + (size_t) row * hd;
    float q[INK_MAXE], acc[INK_MAXE];
#pragma unroll
    for (int j = 0; j < INK_MAXE; j++) {
        q[j] = j < E ? qh[lane + 32 * j] * a.scale : 0.f;
        acc[j] = 0.f;
    }
    const float rv = lane < nr ? ((const float *) a.r)[(size_t) row * nr + lane] : 0.f;
    const float sk = ((const float *) a.sink)[row];
    const float *bias = (const float *) a.rel_bias;
    float m = -INFINITY, l = 0.f;
    const __half *kc = (const __half *) a.k_cache + kvh * hd + lane;
    const __half *vc = (const __half *) a.v_cache + kvh * hd + lane;
    for (int p = p0 + w; p < p1; p += INK_ATTN_WARPS) {
        const size_t at = (size_t) (p & a.mask) * kv_dim;
        const int d = pos - p;
        float s = (lane < nr && d < a.ext) ? bias[d * nr + lane] * rv : 0.f;
#pragma unroll
        for (int j = 0; j < INK_MAXE; j++) {
            if (j < E) s += q[j] * __half2float(kc[at + 32 * j]);
        }
        s = warp_sum(s) - sk;
        const float mn = fmaxf(m, s);
        const float corr = expf(m - mn), e = expf(s - mn);
        l = l * corr + e;
#pragma unroll
        for (int j = 0; j < INK_MAXE; j++) {
            if (j < E) acc[j] = acc[j] * corr + e * __half2float(vc[at + 32 * j]);
        }
        m = mn;
    }
    if (lane == 0) {
        sm[w] = m;
        sl[w] = l;
    }
#pragma unroll
    for (int j = 0; j < INK_MAXE; j++) {
        if (j < E) sacc[w][lane + 32 * j] = acc[j];
    }
    __syncthreads();

    // a single range has the sink's zero logit
    float M = a.splits == 1 ? 0.f : -INFINITY;
    for (int i = 0; i < INK_ATTN_WARPS; i++) M = fmaxf(M, sm[i]);
    float f[INK_ATTN_WARPS];
    float L = a.splits == 1 ? expf(-M) : 0.f;
#pragma unroll
    for (int i = 0; i < INK_ATTN_WARPS; i++) {
        // warps without positions (-inf) add nothing, nor do empty ranges
        f[i] = sm[i] == -INFINITY ? 0.f : expf(sm[i] - M);
        L += sl[i] * f[i];
    }
    for (int dd = threadIdx.x; dd < hd; dd += blockDim.x) {
        float o = 0.f;
#pragma unroll
        for (int i = 0; i < INK_ATTN_WARPS; i++) o += sacc[i][dd] * f[i];
        if (a.splits == 1) {
            const size_t i = (size_t) row * hd + dd;
            ((float *) a.out)[i] = o / L * sigmoid(((const float *) a.gate)[i]);
        } else {
            ((float *) a.part_o)[((size_t) row * a.splits + split) * hd + dd] = o;
        }
    }
    if (a.splits > 1 && threadIdx.x == 0) {
        float *ml = (float *) a.part_ml + 2 * ((size_t) row * a.splits + split);
        ml[0] = M;
        ml[1] = L;
    }
}

// Merge the ranges of one (head, token) with the zero sink logit, a thread
// per dim, gated by sigmoid(gate)
extern "C" __global__ void ink_attn_reduce(const InkAttnArgs a) {
    const int head = blockIdx.x, t = blockIdx.y, dd = threadIdx.x;
    const int hd = a.hd, S = a.splits;
    const size_t row = (size_t) t * a.n_head + head;
    const float *ml = (const float *) a.part_ml + 2 * row * S;
    const float *po = (const float *) a.part_o + row * S * hd;
    float M = 0.f;
    for (int s = 0; s < S; s++) M = fmaxf(M, ml[2 * s]);
    float L = expf(-M), o = 0.f;
    for (int s = 0; s < S; s++) {
        const float m = ml[2 * s];
        const float f = m == -INFINITY ? 0.f : expf(m - M);
        L += ml[2 * s + 1] * f;
        if (dd < hd) o += po[s * hd + dd] * f;
    }
    if (dd < hd) {
        const size_t i = row * hd + dd;
        ((float *) a.out)[i] = o / L * sigmoid(((const float *) a.gate)[i]);
    }
}

// ---------------------------------------------------------------------------
// inkling feed-forward and output

// gate = SITU(gate, up): (tanh(g / beta) beta) sigmoid(g) * (tanh(u / lb) lb),
// the second factor u itself when lb is 0; `ib`, `ilb` the reciprocals
struct InkSituArgs { u64 gate; u64 up; int n; float beta; float ib; float lb; float ilb; int pad; };

extern "C" __global__ void ink_situ(const InkSituArgs a) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= a.n) return;
    float *gate = (float *) a.gate;
    const float g = gate[i], up = ((const float *) a.up)[i];
    const float act = (tanhf(g * a.ib) * a.beta) * sigmoid(g);
    const float u = a.lb > 0.f ? tanhf(up * a.ilb) * a.lb : up;
    gate[i] = act * u;
}

// y[t] = sum_j weights[t*k + j] * out[t*k + j] (moe_combine without the
// residual: the sum goes through a conv first)
extern "C" __global__ void ink_moe_sum(const CombineArgs a) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (d >= a.n_embd) return;
    const float *out = (const float *) a.out, *w = (const float *) a.weights;
    float acc = 0.f;
    for (int j = 0; j < a.k; j++) acc += w[t * a.k + j] * out[((size_t) t * a.k + j) * a.n_embd + d];
    ((float *) a.x)[(size_t) t * a.n_embd + d] = acc;
}

// z = tanh(z / cap) cap, `inv` the reciprocal
struct InkCapArgs { u64 z; int n; float cap; float inv; int pad; };

extern "C" __global__ void ink_softcap(const InkCapArgs a) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a.n) {
        float *z = (float *) a.z;
        z[i] = tanhf(z[i] * a.inv) * a.cap;
    }
}

// a = b, of n floats
extern "C" __global__ void ink_copy(const EltArgs e) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < e.n) ((float *) e.a)[i] = ((const float *) e.b)[i];
}
