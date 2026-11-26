// ---------------------------------------------------------------------------
// inkling2_loop (Inkling-2): appended to kernels.metal, whose helpers and
// FOR_UNROLL it uses. See engine/src/model/inkling.rs for the math these
// kernels follow, and metal/inkling.rs for how they are encoded.

// The depth mix of one token (a threadgroup per row): the states, at float
// offsets `off[s]` of `stack` (rows of `e`), the last one optionally
// rms(cur) of the row of `cur`, written there first; then per query q of
// `query` ([n_q][e]) the softmax of the states' dot products with it times
// `scale`, and the states weighed by it, written `copies` times from output
// slot q * copies (slots `ostride` floats apart), and with `out_norm`
// rms-normalized and scaled by row q of `w`. No query: only the norm.
#define INK_MAXS 8
#define INK_MAXQ 4

struct InkMixArgs {
    int   e;
    int   n_states;
    int   n_q;
    int   copies;
    float scale;
    float eps;      // of rms(cur)
    int   norm_cur;
    int   out_norm;
    float out_eps;
    int   in_row0;  // the row of states and cur of threadgroup 0
    int   out_row0; // the row of the outputs of threadgroup 0
    int   ostride;
    int   off[INK_MAXS];
};

// the sum over the threadgroup of `v`, `red` 32 floats; every thread gets it
inline float ink_tg_sum(float v, threadgroup float * red, ushort lane, ushort sg, ushort nsg) {
    v = simd_sum(v);
    if (nsg > 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
            red[sg] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        v = simd_sum(lane < nsg ? red[lane] : 0.f);
    }
    return v;
}

kernel void ink_mix(
        constant InkMixArgs & a [[buffer(0)]],
        device       float * stack [[buffer(1)]],
        device const float * cur   [[buffer(2)]],
        device const float * query [[buffer(3)]],
        device const float * w     [[buffer(4)]],
        device       float * out   [[buffer(5)]],
        uint   row  [[threadgroup_position_in_grid]],
        ushort tid  [[thread_position_in_threadgroup]],
        ushort ntg  [[threads_per_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[32];
    threadgroup float dots[8][INK_MAXQ*INK_MAXS];
    const int e = a.e;
    const int S = a.n_states;
    const ushort nsg = (ntg + 31)/32;
    const int r = a.in_row0 + row;

    // each thread reads back only the elements it wrote
    if (a.norm_cur) {
        device const float * x = cur + r*e;
        device float * c = stack + a.off[S - 1] + r*e;
        float ss = 0.f;
        for (int i = tid; i < e; i += ntg) {
            ss += x[i]*x[i];
        }
        ss = ink_tg_sum(ss, red, lane, sg, nsg);
        const float scale = 1.0f/sqrt(ss/e + a.eps);
        for (int i = tid; i < e; i += ntg) {
            c[i] = x[i]*scale;
        }
    }
    if (a.n_q == 0) {
        return;
    }

    float d[INK_MAXQ*INK_MAXS];
    FOR_UNROLL (int k = 0; k < INK_MAXQ*INK_MAXS; k++) {
        d[k] = 0.f;
    }
    for (int i = tid; i < e; i += ntg) {
        FOR_UNROLL (int s = 0; s < INK_MAXS; s++) {
            if (s < S) {
                const float v = stack[a.off[s] + r*e + i];
                FOR_UNROLL (int q = 0; q < INK_MAXQ; q++) {
                    if (q < a.n_q) {
                        d[q*INK_MAXS + s] += v*query[q*e + i];
                    }
                }
            }
        }
    }
    FOR_UNROLL (int k = 0; k < INK_MAXQ*INK_MAXS; k++) {
        const float v = simd_sum(d[k]);
        if (lane == 0) {
            dots[sg][k] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // the softmax of each query's scores, in every thread
    float p[INK_MAXQ*INK_MAXS];
    FOR_UNROLL (int q = 0; q < INK_MAXQ; q++) {
        float m = -INFINITY;
        FOR_UNROLL (int s = 0; s < INK_MAXS; s++) {
            float v = 0.f;
            for (int g = 0; g < nsg; g++) {
                v += dots[g][q*INK_MAXS + s];
            }
            v *= a.scale;
            p[q*INK_MAXS + s] = v;
            if (s < S) {
                m = max(m, v);
            }
        }
        float sum = 0.f;
        FOR_UNROLL (int s = 0; s < INK_MAXS; s++) {
            const float v = s < S ? exp(p[q*INK_MAXS + s] - m) : 0.f;
            p[q*INK_MAXS + s] = v;
            sum += v;
        }
        const float inv = 1.0f/sum;
        FOR_UNROLL (int s = 0; s < INK_MAXS; s++) {
            p[q*INK_MAXS + s] *= inv;
        }
    }

    for (int q = 0; q < a.n_q; q++) {
        device float * o = out + q*a.copies*a.ostride + (a.out_row0 + row)*e;
        float ss = 0.f;
        for (int i = tid; i < e; i += ntg) {
            float v = 0.f;
            FOR_UNROLL (int s = 0; s < INK_MAXS; s++) {
                if (s < S) {
                    v += p[q*INK_MAXS + s]*stack[a.off[s] + r*e + i];
                }
            }
            o[i] = v;
            ss += v*v;
        }
        if (a.out_norm) {
            ss = ink_tg_sum(ss, red, lane, sg, nsg);
            const float scale = rsqrt(ss/e + a.out_eps);
            device const float * wq = w + q*e;
            for (int i = tid; i < e; i += ntg) {
                o[i] = o[i]*scale*wq[i];
            }
        }
        for (int c = 1; c < a.copies; c++) {
            device float * oc = o + c*a.ostride;
            for (int i = tid; i < e; i += ntg) {
                oc[i] = o[i];
            }
        }
    }
}

// rms norm without weight of rows of `n`, input rows `istride` floats apart,
// output rows `ostride`; a threadgroup per row
struct InkRmsArgs { int n; float eps; int istride; int ostride; };

kernel void ink_rms(
        constant InkRmsArgs & a [[buffer(0)]],
        device const float * x   [[buffer(1)]],
        device       float * out [[buffer(2)]],
        uint   row  [[threadgroup_position_in_grid]],
        ushort tid  [[thread_position_in_threadgroup]],
        ushort ntg  [[threads_per_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[32];
    device const float * xr = x + row*a.istride;
    float ss = 0.f;
    for (int i = tid; i < a.n; i += ntg) {
        ss += xr[i]*xr[i];
    }
    ss = ink_tg_sum(ss, red, lane, sg, (ntg + 31)/32);
    const float scale = 1.0f/sqrt(ss/a.n + a.eps);
    device float * o = out + row*a.ostride;
    for (int i = tid; i < a.n; i += ntg) {
        o[i] = xr[i]*scale;
    }
}

// a loop controller's output: (x - tanh(f)*x) + tanh(g)*m
kernel void ink_control(
        constant EltArgs & a [[buffer(0)]],
        device const float * x   [[buffer(1)]],
        device const float * g   [[buffer(2)]],
        device const float * f   [[buffer(3)]],
        device const float * m   [[buffer(4)]],
        device       float * out [[buffer(5)]],
        uint i [[thread_position_in_grid]]) {
    if ((int) i < a.n) {
        const float xi = x[i];
        out[i] = (xi - precise::tanh(f[i])*xi) + precise::tanh(g[i])*m[i];
    }
}

// The short causal conv (4 taps, the last on the current token) with a
// residual, of `n` rows of `width` channels, one thread per channel:
// y = (x + conv) * scale, plus with `dyn` each token's own taps, `dyn`
// [n][dyn_stride] at (channel / head) * 4 + lag, lag 0 the current token. The
// state holds the previous 3 inputs of `channels` channels, as
// [slots][3][channels], this conv's from `offset`; slots as for ssm_conv.
// With `add`, out += y, else out = y (in place allowed: out == x).
struct InkConvArgs {
    int   width;
    int   n;
    int   channels;
    int   offset;
    int   slot;
    int   slots;
    int   save;
    int   add;
    int   dyn;
    int   head;
    int   dyn_stride;
    float scale;
};

kernel void ink_conv(
        constant InkConvArgs & a [[buffer(0)]],
        device const float * x     [[buffer(1)]],
        device const float * w     [[buffer(2)]],
        device       float * state [[buffer(3)]],
        device       float * out   [[buffer(4)]],
        device const float * dyn   [[buffer(5)]],
        uint c [[thread_position_in_grid]]) {
    const int W = a.width;
    const int C = a.channels;
    if ((int) c >= W) {
        return;
    }
    device float * st = state + a.slot*3*C + a.offset + c;
    float s0 = st[0], s1 = st[C], s2 = st[2*C];
    const float4 wc = *(device const float4 *) (w + 4*c);
    device const float * dc = dyn + (c / a.head)*4;
    for (int t = 0; t < a.n; t++) {
        const float xt = x[t*W + c];
        float acc = 0.f;
        acc += wc[0]*s0;
        acc += wc[1]*s1;
        acc += wc[2]*s2;
        acc += wc[3]*xt;
        if (a.dyn) {
            const float4 d = *(device const float4 *) (dc + t*a.dyn_stride);
            acc += xt*d[0];
            acc += s2*d[1];
            acc += s1*d[2];
            acc += s0*d[3];
        }
        const float y = (xt + acc)*a.scale;
        if (a.add) {
            out[t*W + c] += y;
        } else {
            out[t*W + c] = y;
        }
        s0 = s1; s1 = s2; s2 = xt;
        if (a.save) {
            device float * sv = state + ((a.slot + 1 + t) % a.slots)*3*C + a.offset + c;
            sv[0] = s0; sv[C] = s1; sv[2*C] = s2;
        }
    }
    if (!a.save) {
        st[0] = s0; st[C] = s1; st[2*C] = s2;
    }
}

// ---------------------------------------------------------------------------
// inkling attention: no rope, a relative position bias per (token, head,
// distance), minus a sink logit per (token, head) whose own softmax logit is
// zero, an output gate per channel

struct InkAttnArgs {
    int   n_head;
    int   n_kv;
    int   hd;
    int   n_rel;
    int   pos0;     // position of the first token
    int   n;        // tokens
    float eps;      // of the values' rms norm
    float scale;    // of the scores
    float clip;     // the L2 norm of queries and keys at most
    float rel_clip; // and of the relative position channels
    int   full;     // tau scales the queries and relative position channels
    int   floor_;   // tau = 1 + alpha ln(max(1, (pos + 1) / floor))
    float alpha;
    int   splits;   // split-K ranges per (token, head)
    int   chunk;    // positions per range
    int   window;   // positions a token sees, its own included; 0: all
    int   lo;       // first position of the ranges
    int   mask;     // cache row of position p: p & mask
    int   ext;      // distances with a bias row; farther, none
};

inline int ink_first(constant InkAttnArgs & a, int pos) {
    return a.window > 0 ? max(pos + 1 - a.window, 0) : 0;
}

inline float ink_tau(constant InkAttnArgs & a, int pos) {
    if (!a.full) {
        return 1.f;
    }
    const float s = ((float) pos + 1.0f)*(1.0f/(float) a.floor_);
    return precise::log(max(s, 1.0f))*a.alpha + 1.0f;
}

// Per (head, token), 32 threads (HD = 128: 4 dims per lane). Heads [0, n_head):
// q clipped (scaled by q_clip first) and times tau, in place. [n_head, n_head
// + n_kv): k clipped, v rms-normalized, to the f16 caches at the token's
// position. [n_head + n_kv, 2 n_head + n_kv): r clipped (scaled by rel_clip
// first) and times tau, in place.
kernel void ink_attn_prep(
        constant InkAttnArgs & a [[buffer(0)]],
        device       float * q       [[buffer(1)]],
        device const float * k       [[buffer(2)]],
        device const float * v       [[buffer(3)]],
        device       float * r       [[buffer(4)]],
        device const float * q_clip  [[buffer(5)]],
        device const float * r_clip  [[buffer(6)]],
        device       half  * k_cache [[buffer(7)]],
        device       half  * v_cache [[buffer(8)]],
        uint2  tg   [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int E = 4;
    const int head = tg.x;
    const int t    = tg.y;
    const int hd   = a.hd;
    const int kv_dim = a.n_kv*hd;
    const int pos  = a.pos0 + t;
    const float tau = ink_tau(a, pos);

    if (head < a.n_head) {
        device float * src = q + (t*a.n_head + head)*hd;
        device const float * sc = q_clip + head*hd;
        float vals[E];
        float ss = 0.f;
        for (int j = 0; j < E; j++) {
            vals[j] = src[lane + 32*j]*sc[lane + 32*j];
            ss += vals[j]*vals[j];
        }
        ss = simd_sum(ss);
        const float inv = 1.0f/max(sqrt(ss), a.clip);
        for (int j = 0; j < E; j++) {
            src[lane + 32*j] = vals[j]*inv*a.clip*tau;
        }
    } else if (head < a.n_head + a.n_kv) {
        const int kh = head - a.n_head;
        device const float * ks = k + t*kv_dim + kh*hd;
        device const float * vs = v + t*kv_dim + kh*hd;
        float kv[E];
        float vv[E];
        float sk = 0.f, sv = 0.f;
        for (int j = 0; j < E; j++) {
            kv[j] = ks[lane + 32*j];
            vv[j] = vs[lane + 32*j];
            sk += kv[j]*kv[j];
            sv += vv[j]*vv[j];
        }
        sk = simd_sum(sk);
        sv = simd_sum(sv);
        const float inv = 1.0f/max(sqrt(sk), a.clip);
        const float vscale = 1.0f/sqrt(sv/hd + a.eps);
        device half * kd = k_cache + (pos & a.mask)*kv_dim + kh*hd;
        device half * vd = v_cache + (pos & a.mask)*kv_dim + kh*hd;
        for (int j = 0; j < E; j++) {
            kd[lane + 32*j] = (half) (kv[j]*inv*a.clip);
            vd[lane + 32*j] = (half) (vv[j]*vscale);
        }
    } else {
        const int rh = head - a.n_head - a.n_kv;
        const int nr = a.n_rel;
        device float * src = r + (t*a.n_head + rh)*nr;
        const float x = lane < nr ? src[lane]*r_clip[rh*nr + lane] : 0.f;
        const float ss = simd_sum(x*x);
        const float inv = 1.0f/max(sqrt(ss), a.rel_clip);
        if (lane < nr) {
            src[lane] = x*inv*a.rel_clip*tau;
        }
    }
}

// Split-K attention, as attn_split (one token, one kv head and its G query
// heads, a range of positions; each simdgroup HPS heads on every PS-th
// position), HD = 128, with the bias: lane l < n_rel holds channel l of the
// heads' relative position channels, so that the bias dot product joins the
// score's reduction. The zero sink is added in ink_attn_reduce.
template <int G, int HPS, int NSG>
kernel void ink_attn_split(
        constant InkAttnArgs & a [[buffer(0)]],
        device const float * q        [[buffer(1)]],
        device const half  * k_cache  [[buffer(2)]],
        device const half  * v_cache  [[buffer(3)]],
        device const float * r        [[buffer(4)]],
        device const float * sink     [[buffer(5)]],
        device const float * rel_bias [[buffer(6)]],
        device       float * part_o   [[buffer(7)]],
        device       float * part_ml  [[buffer(8)]],
        threadgroup  float * shm      [[threadgroup(0)]],
        uint3  tg   [[threadgroup_position_in_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    constexpr int HD = 128;
    constexpr int E  = HD/32;
    constexpr int HG = G/HPS;
    constexpr int PS = NSG/HG;
    static_assert(G % HPS == 0 && NSG % HG == 0, "ink_attn_split geometry");
    const int hg = sg % HG;
    const int ps = sg / HG;
    const int kvh   = tg.x;
    const int split = tg.y;
    const int t     = tg.z;
    const int kv_dim = a.n_kv*HD;
    const int pos   = a.pos0 + t;
    const int n_ctx = pos + 1;
    const int first = ink_first(a, pos);
    const int p0 = a.lo + split*a.chunk;
    const int p1 = min(p0 + a.chunk, n_ctx);
    const int nr = a.n_rel;

    float qv[HPS][E];
    float acc[HPS][E];
    float rv[HPS];
    float sk[HPS];
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
        rv[h] = lane < nr ? r[(t*a.n_head + head)*nr + lane] : 0.f;
        sk[h] = sink[t*a.n_head + head];
        m[h] = -INFINITY;
        l[h] = 0.f;
    }

    const int start = p0 + ps + (max(first - p0 - ps, 0) + PS - 1)/PS*PS;
    for (int p = start; p < p1; p += PS) {
        device const half4 * kp = (device const half4 *) (k_cache + (p & a.mask)*kv_dim + kvh*HD + lane*E);
        device const half4 * vp = (device const half4 *) (v_cache + (p & a.mask)*kv_dim + kvh*HD + lane*E);
        const int d = pos - p;
        const float rb = (lane < nr && d < a.ext) ? rel_bias[d*nr + lane] : 0.f;
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
            float s = rb*rv[h];
            for (int j = 0; j < E; j++) {
                s += qv[h][j] * kf[j];
            }
            s = simd_sum(s) - sk[h];
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
        float M = -INFINITY;
        for (int i = 0; i < PS; i++) {
            M = max(M, sm[i*G + hg*HPS + h]);
        }
        f[h] = m[h] == -INFINITY ? 0.f : exp(m[h] - M);
    }
    for (int i = 0; i < PS; i++) {
        if (ps == i) {
            for (int h = 0; h < HPS; h++) {
                for (int j = 0; j < E; j++) {
                    threadgroup float * o = so + (hg*HPS + h)*HD + lane*E + j;
                    *o = (i == 0 ? 0.f : *o) + f[h]*acc[h][j];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int part = (t*a.n_head + kvh*G)*a.splits + split;
    for (int i = tid; i < G*HD; i += NSG*32) {
        const int g = i / HD;
        part_o[(part + g*a.splits)*HD + i % HD] = so[i];
    }
    if (tid < G) {
        const int g = tid;
        float M = -INFINITY;
        for (int i = 0; i < PS; i++) {
            M = max(M, sm[i*G + g]);
        }
        float L = 0.f;
        for (int i = 0; i < PS; i++) {
            L += sm[i*G + g] == -INFINITY ? 0.f : sl[i*G + g]*exp(sm[i*G + g] - M);
        }
        part_ml[2*(part + g*a.splits)]     = M;
        part_ml[2*(part + g*a.splits) + 1] = L;
    }
}

#define INK_ATTN_SPLIT(g, hps, nsg) \
template [[host_name("ink_attn_split_" #g)]] kernel void ink_attn_split<g, hps, nsg>( \
    constant InkAttnArgs &, device const float *, device const half *, device const half *, \
    device const float *, device const float *, device const float *, \
    device float *, device float *, threadgroup float *, uint3, ushort, ushort, ushort);
INK_ATTN_SPLIT(1, 1, 2) INK_ATTN_SPLIT(2, 2, 2) INK_ATTN_SPLIT(3, 3, 4) INK_ATTN_SPLIT(4, 2, 2)
INK_ATTN_SPLIT(6, 2, 3) INK_ATTN_SPLIT(8, 2, 4)

// Merge the splits (at most 64) of one (head, token) with the zero sink
// logit, one thread per dim, gated by sigmoid(gate), gate [n][n_head * hd]:
// the splits' maxima and sums read by a thread each, the partial outputs
// summed in order (loops of a static count, the splits past the last
// weighed 0)
kernel void ink_attn_reduce(
        constant InkAttnArgs & a [[buffer(0)]],
        device const float * part_o  [[buffer(1)]],
        device const float * part_ml [[buffer(2)]],
        device const float * gate    [[buffer(3)]],
        device       float * out     [[buffer(4)]],
        uint2  tg   [[threadgroup_position_in_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    constexpr int MAXS = 64;
    threadgroup float ws[MAXS];
    threadgroup float red[2][4];
    const int head = tg.x;
    const int t    = tg.y;
    const int hd   = a.hd;
    const int part = (t*a.n_head + head)*a.splits;
    const int sp   = min((int) tid, a.splits - 1);
    const bool in  = (int) tid < a.splits;
    // the sink's logit is 0: ranges without positions (M -inf, L 0) add nothing
    const float m = in ? part_ml[2*(part + sp)] : -INFINITY;
    const float l = in ? part_ml[2*(part + sp) + 1] : 0.f;
    float M = simd_max(max(m, 0.f));
    if (lane == 0) {
        red[0][sg] = M;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    M = max(max(red[0][0], red[0][1]), max(red[0][2], red[0][3]));
    const float w = in ? exp(m - M) : 0.f;
    if (tid < MAXS) {
        ws[tid] = w;
    }
    const float wl = simd_sum(w*l);
    if (lane == 0) {
        red[1][sg] = wl;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float L = exp(-M) + ((red[1][0] + red[1][1]) + (red[1][2] + red[1][3]));
    float o = 0.f;
    FOR_UNROLL (int s = 0; s < MAXS; s++) {
        const float v = part_o[(part + min(s, a.splits - 1))*hd + tid];
        o += ws[s]*v;
    }
    const int i = (t*a.n_head + head)*hd + tid;
    out[i] = o / L * sigmoid(gate[i]);
}

// Flash attention, as attn_flash (R query rows of one kv head, the (token,
// head) pairs of its G heads, head fastest; C positions per step), HD = 128,
// with the bias added to each score in the softmax step: the rows' relative
// position channels (16) and sink logits are in threadgroup memory, the bias
// table row of each distance read from the device. Without SPLIT, the
// online softmax starts from the zero sink logit (M = 0, L = 1) and the
// output is gated by sigmoid(gate); with it, split-K partials as attn_flash.
template <int R, int C, int NSG, int QR, bool SPLIT>
kernel void ink_attn_flash(
        constant InkAttnArgs & a [[buffer(0)]],
        device const float * q        [[buffer(1)]],
        device const half  * k_cache  [[buffer(2)]],
        device const half  * v_cache  [[buffer(3)]],
        device const float * r        [[buffer(4)]],
        device const float * sink     [[buffer(5)]],
        device const float * rel_bias [[buffer(6)]],
        device       float * gate     [[buffer(7)]], // SPLIT: part_o
        device       float * out      [[buffer(8)]], // SPLIT: part_ml
        threadgroup  half  * shm      [[threadgroup(0)]],
        uint3  tg   [[threadgroup_position_in_grid]],
        uint3  ntg  [[threadgroups_per_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    constexpr int HD = 128;
    constexpr int NR = 16;                 // relative position channels
    constexpr int RT = R/8;
    constexpr int CT = C/8;
    constexpr int DT = HD/8;
    constexpr int QC = RT*CT/(NSG*QR);
    constexpr int OD = DT/NSG;
    constexpr int NQ = R/NSG;
    constexpr int CL = C/32;
    static_assert(RT % QR == 0 && (RT/QR)*(CT/QC) == NSG && DT % NSG == 0 && R % NSG == 0 && C % 32 == 0,
                  "ink_attn_flash geometry");

    threadgroup half  * sq = shm;                               // [R][HD] queries, scaled
    threadgroup float * ss = (threadgroup float *) (sq + R*HD); // [R][C] scores, then probabilities
    threadgroup float * sf = ss + R*C;                          // [R] rescale factors, then 1/sum
    threadgroup float * sr = sf + R;                            // [R][NR] relative position channels
    threadgroup float * sk = sr + R*NR;                         // [R] sink logits

    const int G      = a.n_head/a.n_kv;
    const int kvh    = tg.y;
    const int rows   = a.n*G;
    const int kv_dim = a.n_kv*HD;
    const int row0   = (ntg.x - 1 - tg.x)*R;

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
    for (int i = tid; i < R*NR; i += NSG*32) {
        const int g = min(row0 + i/NR, rows - 1);
        const int t = g/G;
        const int head = kvh*G + g%G;
        sr[i] = r[(t*a.n_head + head)*NR + i%NR];
        if (i % NR == 0) {
            sk[i/NR] = sink[t*a.n_head + head];
        }
    }

    int   first[NQ];
    int   last[NQ];
    float M[NQ];
    float L[NQ];
    FOR_UNROLL (int j = 0; j < NQ; j++) {
        last[j] = a.pos0 + min(row0 + sg*NQ + j, rows - 1)/G;
        first[j] = ink_first(a, last[j]);
        M[j] = SPLIT ? -FLT_MAX/2 : 0.f;
        L[j] = SPLIT ? 0.f : 1.f;
    }
    const int last_end = a.pos0 + (min(row0 + R, rows) - 1)/G + 1;
    const int begin = SPLIT ? a.lo + (int) tg.z*a.chunk : ink_first(a, a.pos0 + row0/G) / C * C;
    const int end   = SPLIT ? min(begin + a.chunk, last_end) : last_end;

    const int qr0 = (sg % (RT/QR))*QR;
    const int qc0 = (sg / (RT/QR))*QC;
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
            device const half * pk = k_cache + ((p0 & a.mask) + 8*qc0)*kv_dim + kvh*HD;
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
            const int rw = sg*NQ + j;
            threadgroup const float4 * rr = (threadgroup const float4 *) (sr + rw*NR);
            float s[CL];
            float m = M[j];
            FOR_UNROLL (int c = 0; c < CL; c++) {
                const int i = lane + 32*c;
                const int p = p0 + i;
                float v = -INFINITY;
                if (p <= last[j] && p >= first[j] && p < end) {
                    const int d = last[j] - p;
                    float b = 0.f;
                    if (d < a.ext) {
                        device const float4 * tb = (device const float4 *) (rel_bias + d*NR);
                        const float4 x = tb[0]*rr[0] + tb[1]*rr[1] + tb[2]*rr[2] + tb[3]*rr[3];
                        b = (x[0] + x[1]) + (x[2] + x[3]);
                    }
                    v = ss[rw*C + i] + (b - sk[rw]);
                }
                s[c] = v;
                m = max(m, s[c]);
            }
            m = simd_max(m);
            const float f = exp(M[j] - m);
            float l = 0.f;
            FOR_UNROLL (int c = 0; c < CL; c++) {
                const float pe = exp(s[c] - m);
                ss[rw*C + lane + 32*c] = pe;
                l += pe;
            }
            L[j] = L[j]*f + simd_sum(l);
            M[j] = m;
            if (lane == 0) {
                sf[rw] = f;
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
        device const half * pv = v_cache + (p0 & a.mask)*kv_dim + kvh*HD + 8*sg*OD;
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
        device float * part_o  = gate;
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
        device const float * gt = gate + (t*a.n_head + head)*HD;
        device float * o = out + (t*a.n_head + head)*HD;
        FOR_UNROLL (int k = 0; k < OD; k++) {
            thread auto & e = mo[rt][k].thread_elements();
            const int i = 8*(sg*OD + k) + lc;
            o[i]     = e[0] * inv * sigmoid(gt[i]);
            o[i + 1] = e[1] * inv * sigmoid(gt[i + 1]);
        }
    }
}

#define INK_ATTN_FLASH(name, split) \
template [[host_name(name)]] kernel void ink_attn_flash<16, 64, 4, 2, split>( \
    constant InkAttnArgs &, device const float *, device const half *, device const half *, \
    device const float *, device const float *, device const float *, \
    device float *, device float *, threadgroup half *, uint3, uint3, ushort, ushort, ushort);
INK_ATTN_FLASH("ink_attn_flash", false) INK_ATTN_FLASH("ink_attn_flash_split", true)

// ---------------------------------------------------------------------------
// inkling feed-forward and output

// gate = SITU(gate, up): (tanh(g / beta) beta) sigmoid(g) * (tanh(u / lb) lb),
// the second factor u itself when lb is 0; `ib`, `ilb` the reciprocals
struct InkSituArgs { int n; float beta; float ib; float lb; float ilb; };

kernel void ink_situ(
        constant InkSituArgs & a [[buffer(0)]],
        device       float * gate [[buffer(1)]],
        device const float * up   [[buffer(2)]],
        uint i [[thread_position_in_grid]]) {
    if ((int) i < a.n) {
        const float g = gate[i];
        const float act = (precise::tanh(g*a.ib)*a.beta)*sigmoid(g);
        const float u = a.lb > 0.f ? precise::tanh(up[i]*a.ilb)*a.lb : up[i];
        gate[i] = act*u;
    }
}

// out[t] = sum_j weights[t*k + j] * pairs[t*k + j]
kernel void ink_moe_sum(
        constant CombineArgs & args [[buffer(0)]],
        device       float * out     [[buffer(1)]],
        device const float * pairs   [[buffer(2)]],
        device const float * weights [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]) {
    const int d = gid.x, t = gid.y;
    if (d >= args.n_embd) {
        return;
    }
    float acc = 0.f;
    for (int j = 0; j < args.k; j++) {
        acc += weights[t*args.k + j]*pairs[(t*args.k + j)*args.n_embd + d];
    }
    out[t*args.n_embd + d] = acc;
}

// z = tanh(z / cap) cap, `inv` the reciprocal
struct InkCapArgs { int n; float cap; float inv; };

kernel void ink_softcap(
        constant InkCapArgs & a [[buffer(0)]],
        device float * z [[buffer(1)]],
        uint i [[thread_position_in_grid]]) {
    if ((int) i < a.n) {
        z[i] = precise::tanh(z[i]*a.inv)*a.cap;
    }
}

kernel void ink_copy(
        constant EltArgs & a [[buffer(0)]],
        device       float * dst [[buffer(1)]],
        device const float * src [[buffer(2)]],
        uint i [[thread_position_in_grid]]) {
    if ((int) i < a.n) {
        dst[i] = src[i];
    }
}

// ---------------------------------------------------------------------------
// ink_mix of a few rows, each row in chunks of INK_MIX_CH elements, a
// threadgroup of 64 threads (a float4 each) per (chunk, row), in three
// dispatches: the chunks' partial sums (the scores, with the raw stream for
// the normalized one, and its sum of squares), then the outputs from the
// totals (with out_norm unnormalized, and their partial sums of squares),
// then with out_norm the normalization. `part` is [rows][16][40]: 32 scores,
// the stream's sum of squares, the outputs' per query.
#define INK_MIX_CH 256
#define INK_MIX_P 40

// the sums over the 64 threads of the first N of `v`, written to `dst`
// (loops of a static count: a run time bound in an unrolled loop is slow)
template <int N>
inline void ink_mix_sums(thread const float * v, device float * dst,
                         threadgroup float (*tg)[INK_MIX_P], ushort lane, ushort sg) {
    FOR_UNROLL (int k = 0; k < N; k++) {
        const float t = simd_sum(v[k]);
        if (lane == 0) {
            tg[sg][k] = t;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        for (int k = lane; k < N; k += 32) {
            dst[k] = tg[0][k] + tg[1][k];
        }
    }
}

kernel void ink_mix_part(
        constant InkMixArgs & a [[buffer(0)]],
        device const float * stack [[buffer(1)]],
        device const float * cur   [[buffer(2)]],
        device const float * query [[buffer(3)]],
        device       float * part  [[buffer(4)]],
        uint2  tg   [[threadgroup_position_in_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[2][INK_MIX_P];
    const int e = a.e;
    const int S = a.n_states;
    const int r = a.in_row0 + tg.y;
    const bool in = tg.x*INK_MIX_CH + 4*tid < e;
    // past the row's end, its start (the loads unconditional, then dropped)
    const int i = in ? tg.x*INK_MIX_CH + 4*tid : 0;
    float4 st[INK_MAXS];
    FOR_UNROLL (int s = 0; s < INK_MAXS; s++) {
        device const float * src = (s == S - 1 && a.norm_cur) ? cur + r*e : stack + a.off[s] + r*e;
        st[s] = *(device const float4 *) (src + i);
        st[s] = (in && s < S) ? st[s] : 0.f;
    }
    float v[INK_MIX_P];
    FOR_UNROLL (int q = 0; q < INK_MAXQ; q++) {
        float4 qv = *(device const float4 *) (query + min(q, max(a.n_q - 1, 0))*e + i);
        qv = (in && q < a.n_q) ? qv : 0.f;
        FOR_UNROLL (int s = 0; s < INK_MAXS; s++) {
            v[q*INK_MAXS + s] = dot(st[s], qv);
        }
    }
    float4 c = 0.f;
    FOR_UNROLL (int s = 0; s < INK_MAXS; s++) {
        if (s == S - 1) {
            c = st[s];
        }
    }
    v[32] = a.norm_cur ? dot(c, c) : 0.f;
    FOR_UNROLL (int k = 33; k < INK_MIX_P; k++) {
        v[k] = 0.f;
    }
    // the scores of no query: only the stream's sum of squares
    if (a.n_q == 0) {
        v[0] = v[32];
        ink_mix_sums<1>(v, part + (tg.y*16 + tg.x)*INK_MIX_P + 32, red, lane, sg);
        return;
    }
    ink_mix_sums<33>(v, part + (tg.y*16 + tg.x)*INK_MIX_P, red, lane, sg);
}

kernel void ink_mix_apply(
        constant InkMixArgs & a [[buffer(0)]],
        device       float * stack [[buffer(1)]],
        device const float * cur   [[buffer(2)]],
        device       float * part  [[buffer(3)]],
        device       float * out   [[buffer(4)]],
        uint2  tg   [[threadgroup_position_in_grid]],
        uint2  ntg  [[threadgroups_per_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[2][INK_MIX_P];
    const int e = a.e;
    const int S = a.n_states;
    const int nq = a.n_q;
    const int r = a.in_row0 + tg.y;
    const bool in = tg.x*INK_MIX_CH + 4*tid < e;
    const int i = in ? tg.x*INK_MIX_CH + 4*tid : 0;
    device float * pr = part + tg.y*16*INK_MIX_P;

    // the totals: score q * 8 + s in lane q * 8 + s, the stream's sum of squares
    float t = 0.f, sc = 0.f;
    FOR_UNROLL (uint ch = 0; ch < 16; ch++) {
        const float vt = pr[ch*INK_MIX_P + lane];
        const float vc = pr[ch*INK_MIX_P + 32];
        t += ch < ntg.x ? vt : 0.f;
        sc += ch < ntg.x ? vc : 0.f;
    }
    const float scale_c = a.norm_cur ? 1.0f/sqrt(sc/e + a.eps) : 1.f;

    float4 st[INK_MAXS];
    FOR_UNROLL (int s = 0; s < INK_MAXS; s++) {
        const bool c = s == S - 1 && a.norm_cur;
        st[s] = *(device const float4 *) ((c ? cur : stack + a.off[s]) + r*e + i);
        st[s] = (in && s < S) ? st[s]*(c ? scale_c : 1.f) : 0.f;
        if (in && c) {
            *(device float4 *) (stack + a.off[s] + r*e + i) = st[s];
        }
    }
    if (nq == 0) {
        return;
    }

    // the softmax of the scores of query q = lane / 8, in its lanes
    const ushort ls = lane % INK_MAXS;
    const float v = t*(ls == S - 1 ? scale_c : 1.f)*a.scale;
    float m = ls < S ? v : -INFINITY;
    FOR_UNROLL (ushort o = 1; o < INK_MAXS; o *= 2) {
        m = max(m, simd_shuffle_xor(m, o));
    }
    const float pe = ls < S ? exp(v - m) : 0.f;
    float sum = pe;
    FOR_UNROLL (ushort o = 1; o < INK_MAXS; o *= 2) {
        sum += simd_shuffle_xor(sum, o);
    }
    const float prob = pe*(1.0f/sum);

    float ss[INK_MIX_P];
    FOR_UNROLL (int k = 0; k < INK_MIX_P; k++) {
        ss[k] = 0.f;
    }
    FOR_UNROLL (int q = 0; q < INK_MAXQ; q++) {
        float4 o = 0.f;
        // the states past S are 0, their probabilities too
        FOR_UNROLL (int s = 0; s < INK_MAXS; s++) {
            o += simd_shuffle(prob, (ushort) (q*INK_MAXS + s))*st[s];
        }
        ss[q] = dot(o, o);
        if (in && q < nq) {
            device float * oq = out + q*a.copies*a.ostride + (a.out_row0 + tg.y)*e + i;
            const int copies = a.out_norm ? 1 : a.copies;
            for (int c = 0; c < copies; c++) {
                *(device float4 *) (oq + c*a.ostride) = o;
            }
        }
    }
    if (a.out_norm) {
        ink_mix_sums<INK_MAXQ>(ss, pr + tg.x*INK_MIX_P + 33, red, lane, sg);
    }
}

kernel void ink_mix_norm(
        constant InkMixArgs & a [[buffer(0)]],
        device const float * w    [[buffer(1)]],
        device const float * part [[buffer(2)]],
        device       float * out  [[buffer(3)]],
        uint2  tg   [[threadgroup_position_in_grid]],
        uint2  ntg  [[threadgroups_per_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const int e = a.e;
    const int i = tg.x*INK_MIX_CH + 4*tid;
    device const float * pr = part + tg.y*16*INK_MIX_P + 33;
    float t = 0.f;
    FOR_UNROLL (uint ch = 0; ch < 16; ch++) {
        const float v = pr[ch*INK_MIX_P + lane % INK_MAXQ];
        t += ch < ntg.x ? v : 0.f;
    }
    for (int q = 0; q < a.n_q; q++) {
        const float scale = rsqrt(simd_shuffle(t, (ushort) q)/e + a.out_eps);
        if (i >= e) {
            continue;
        }
        device float * oq = out + q*a.copies*a.ostride + (a.out_row0 + tg.y)*e + i;
        const float4 y = *(device const float4 *) oq * scale * *(device const float4 *) (w + q*e + i);
        for (int c = 0; c < a.copies; c++) {
            *(device float4 *) (oq + c*a.ostride) = y;
        }
    }
}

// ---------------------------------------------------------------------------
// inkling mixture of experts of short passes (a threadgroup row per pair)

// moe_route of sigmoid gating, one threadgroup of 256 threads per token, a
// thread per expert (n_expert <= 256, k <= 32): each chosen expert's place is its rank
// among the probabilities plus bias, the lower expert first among equals, as
// moe_route picks them; the weights the same.
kernel void ink_route(
        constant RouteArgs & args [[buffer(0)]],
        device const float * logits  [[buffer(1)]],
        device int         * ids     [[buffer(2)]],
        device float       * weights [[buffer(3)]],
        device const float * bias    [[buffer(4)]],
        uint   tg  [[threadgroup_position_in_grid]],
        ushort tid [[thread_position_in_threadgroup]]) {
    threadgroup float sel[32];
    threadgroup float4 p4[64];
    const int ne = args.n_expert;
    const int k = args.k;
    float w = 0.f, pv = -INFINITY;
    if (tid < ne) {
        w = sigmoid(logits[tg*ne + tid]);
        pv = w + (args.bias ? bias[tid] : 0.f);
    }
    // 256 threads: the probabilities past ne -inf
    ((threadgroup float *) p4)[tid] = pv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < ne) {
        int rank = 0;
        FOR_UNROLL (int j4 = 0; j4 < 64; j4++) {
            const float4 q = p4[j4];
            const int4 jj = int4(4*j4) + int4(0, 1, 2, 3);
            const bool4 b = q > pv || (q == pv && jj < int4((int) tid));
            rank += (b[0] ? 1 : 0) + (b[1] ? 1 : 0) + (b[2] ? 1 : 0) + (b[3] ? 1 : 0);
        }
        if (rank < k) {
            ids[tg*k + rank] = tid;
            sel[rank] = w;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int) tid < k) {
        float total = 0.f;
        for (int j = 0; j < k; j++) {
            total += sel[j];
        }
        const float s = (args.norm ? 1.f/max(total, 6.103515625e-5f) : 1.f) * args.scale;
        weights[tg*k + tid] = sel[tid] * s;
    }
}

struct InkGateUpArgs {
    int   k;       // input width
    int   rows;    // gate rows of an expert = output width
    int   x_div;   // pair p reads input row p / x_div
    int   pad;
    ulong w_row_bytes;
    ulong x_row_bytes;
    ulong stride;  // bytes from an expert's matrix to the next
    int   n;       // SITU, as InkSituArgs (n unused)
    float beta;
    float ib;
    float lb;
    float ilb;
};

// SITU(gate, up) of a row of a pair's expert, as mv_id_q8_0 of the gate
// and of the up rows (the same sums) then ink_situ: act [pair][rows]
kernel void ink_gate_up_q8_0(
        constant InkGateUpArgs & a [[buffer(0)]],
        device const char  * gate [[buffer(1)]],
        device const char  * up   [[buffer(2)]],
        device const char  * src1 [[buffer(3)]],
        device       float * act  [[buffer(4)]],
        device const int   * ids  [[buffer(5)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NR0 = 1;
    constexpr short NR = 2*NR0;
    constexpr short NSG = 4;
    constexpr short NW = 32;
    constexpr short NQ = 8;
    threadgroup float shmem[NW*NR];

    const uint p = tgpig.y;
    const int nb = a.k/32;
    const int r0 = tgpig.x*NR0;
    device const float * y = (device const float *) (src1 + (p/a.x_div)*a.x_row_bytes);

    device const block_q8_0 * ax[NR];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const int rr = min(r0 + row, a.rows - 1);
        ax[row]       = (device const block_q8_0 *) (gate + ids[p]*a.stride + rr*a.w_row_bytes);
        ax[row + NR0] = (device const block_q8_0 *) (up   + ids[p]*a.stride + rr*a.w_row_bytes);
    }

    float sumf[NR] = { 0.f };

    const short ix = tiisg/(NW/NQ);
    const short il = tiisg%(NW/NQ);
    const int ib0 = sgitg*NQ + ix;

    float yl[NQ];
    device const float * yb = y + ib0*32 + il*NQ;

    for (int ib = ib0; ib < nb; ib += NSG*NQ) {
        FOR_UNROLL (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }
        FOR_UNROLL (short row = 0; row < NR; row++) {
            device const int8_t * qs = ax[row][ib].qs + il*NQ;
            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NQ; ++i) {
                sumq += qs[i] * yl[i];
            }
            sumf[row] += sumq*ax[row][ib].d;
        }
        yb += NSG*NQ*32;
    }

    FOR_UNROLL (short row = 0; row < NR; ++row) {
        if (sgitg == 0) {
            shmem[NW*row + tiisg] = 0.f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    FOR_UNROLL (short row = 0; row < NR; ++row) {
        if (tiisg == 0) {
            shmem[NW*row + sgitg] = sumf[row];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot[NR];
    FOR_UNROLL (short row = 0; row < NR; ++row) {
        tot[row] = simd_sum(shmem[NW*row + tiisg]);
    }
    if (tiisg == 0 && sgitg == 0) {
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            if (r0 + row < a.rows) {
                const float g = tot[row];
                const float u = tot[row + NR0];
                const float act_g = (precise::tanh(g*a.ib)*a.beta)*sigmoid(g);
                const float up_u = a.lb > 0.f ? precise::tanh(u*a.ilb)*a.lb : u;
                act[p*a.rows + r0 + row] = act_g*up_u;
            }
        }
    }
}

// The experts' down rows of a token, weighed and summed: a simdgroup per
// pair (k <= 8) on 4 output rows, its 4 x nb blocks taken 8 at a time
// (4 lanes of 8 quants each), the pairs summed in order: out [token][rows]
// = sum_j weights[t*k + j] * down_{ids[t*k + j]} . act[t*k + j]
kernel void ink_down_q8_0(
        constant MatIdArgs & a [[buffer(0)]],
        device const char  * src0    [[buffer(1)]],
        device const float * act     [[buffer(2)]],
        device       float * out     [[buffer(3)]],
        device const int   * ids     [[buffer(4)]],
        device const float * weights [[buffer(5)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NR = 4;
    threadgroup float part[8][NR];
    const int t  = tgpig.y;
    const int kk = a.x_div;          // the pairs of a token
    const int nb = a.k/32;
    const int r0 = tgpig.x*NR;
    const short ix = tiisg/4;
    const short il = tiisg%4;

    float s[NR] = { 0.f };
    if (sgitg < kk) {
        const int pair = t*kk + sgitg;
        device const char * w = src0 + ids[pair]*a.stride;
        device const float * x = act + pair*a.k + il*8;
        device const block_q8_0 * rows[NR];
        FOR_UNROLL (short r = 0; r < NR; r++) {
            rows[r] = (device const block_q8_0 *) (w + min(r0 + r, a.rows - 1)*a.w_row_bytes);
        }
        int row = 0, b = ix;
        for (int u = ix; u < NR*nb; u += 8) {
            const float4 y0 = *(device const float4 *) (x + b*32);
            const float4 y1 = *(device const float4 *) (x + b*32 + 4);
            device const block_q8_0 * blk = rows[0] + b;
            FOR_UNROLL (short r = 1; r < NR; r++) {
                if (row == r) {
                    blk = rows[r] + b;
                }
            }
            device const int8_t * qs = blk->qs + il*8;
            const float sumq = (qs[0]*y0[0] + qs[1]*y0[1] + qs[2]*y0[2] + qs[3]*y0[3])
                             + (qs[4]*y1[0] + qs[5]*y1[1] + qs[6]*y1[2] + qs[7]*y1[3]);
            const float v = sumq*blk->d;
            FOR_UNROLL (short r = 0; r < NR; r++) {
                s[r] += row == r ? v : 0.f;
            }
            b += 8;
            if (b >= nb) {
                b -= nb;
                row++;
            }
        }
    }
    FOR_UNROLL (short r = 0; r < NR; r++) {
        s[r] = simd_sum(s[r]);
    }
    if (tiisg == 0) {
        FOR_UNROLL (short r = 0; r < NR; r++) {
            part[sgitg][r] = s[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0 && tiisg < NR && r0 + tiisg < a.rows) {
        float acc = 0.f;
        for (int j = 0; j < kk; j++) {
            acc += weights[t*kk + j]*part[j][tiisg];
        }
        out[t*a.rows + r0 + tiisg] = acc;
    }
}

// ---------------------------------------------------------------------------
// The loop controllers of short passes, a few (x, m) pairs at once (`ctrl`
// along the grid's y, tokens along z): z = down . [rms(x); rms(m)] (the norms
// as scales of the dot products), then per channel out = (x - tanh(f) x) +
// tanh(g) m with g = gain . z, f = forget . z, then out rms-normalized with
// its weight. Up to 4 controllers: their buffers bound one after the other.
struct InkCtrlArgs {
    int   e;
    int   rank;
    float eps;      // of the input norms
    float out_eps;  // of the output's norm
    int   stride;   // floats between the tokens of x, m, out
    int   chunks;   // of ink_ctrl_out: partial sums of squares per row
    int   w_stride; // floats between the output norms' weights of the controllers
    int   pad;
};

#define INK_CTRL_SEL(p, j) ((j) == 0 ? p##0 : (j) == 1 ? p##1 : (j) == 2 ? p##2 : p##3)

// a row of z per threadgroup (256 threads): z [ctrl][token][rank]
kernel void ink_ctrl_down(
        constant InkCtrlArgs & a [[buffer(0)]],
        device const float * x0 [[buffer(1)]], device const float * x1 [[buffer(2)]],
        device const float * x2 [[buffer(3)]], device const float * x3 [[buffer(4)]],
        device const float * m0 [[buffer(5)]], device const float * m1 [[buffer(6)]],
        device const float * m2 [[buffer(7)]], device const float * m3 [[buffer(8)]],
        device const float * w0 [[buffer(9)]], device const float * w1 [[buffer(10)]],
        device const float * w2 [[buffer(11)]], device const float * w3 [[buffer(12)]],
        device       float * z  [[buffer(13)]],
        uint3  tg   [[threadgroup_position_in_grid]],
        uint3  ntg  [[threadgroups_per_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    threadgroup float4 red[8];
    const int e4 = a.e/4;
    const int row = tg.x, j = tg.y, t = tg.z;
    device const float4 * x = (device const float4 *) (INK_CTRL_SEL(x, j) + t*a.stride);
    device const float4 * m = (device const float4 *) (INK_CTRL_SEL(m, j) + t*a.stride);
    device const float4 * w = (device const float4 *) (INK_CTRL_SEL(w, j) + row*2*a.e);
    float4 acc = 0.f; // dot x, dot m, ss x, ss m
    FOR_UNROLL (int k = 0; k < 4; k++) {
        const int i = tid + 256*k;
        if (i < e4) {
            const float4 xv = x[i], mv = m[i];
            acc += float4(dot(xv, w[i]), dot(mv, w[e4 + i]), dot(xv, xv), dot(mv, mv));
        }
    }
    acc = float4(simd_sum(acc[0]), simd_sum(acc[1]), simd_sum(acc[2]), simd_sum(acc[3]));
    if (lane == 0) {
        red[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float4 s = 0.f;
        for (int g = 0; g < 8; g++) {
            s += red[g];
        }
        const float sx = 1.0f/sqrt(s[2]/a.e + a.eps);
        const float sm = 1.0f/sqrt(s[3]/a.e + a.eps);
        z[(j*ntg.z + t)*a.rank + row] = s[0]*sx + s[1]*sm;
    }
}

// 8 channels per simdgroup, 64 per threadgroup, rank 64 (a gain or forget
// row read by 16 lanes, 2 rows at a time): out (may be x), part
// [ctrl][token][chunks] the sums of squares of out by chunks of 256 channels
kernel void ink_ctrl_out(
        constant InkCtrlArgs & a [[buffer(0)]],
        device const float * x0 [[buffer(1)]], device const float * x1 [[buffer(2)]],
        device const float * x2 [[buffer(3)]], device const float * x3 [[buffer(4)]],
        device const float * m0 [[buffer(5)]], device const float * m1 [[buffer(6)]],
        device const float * m2 [[buffer(7)]], device const float * m3 [[buffer(8)]],
        device const float * g0 [[buffer(9)]], device const float * g1 [[buffer(10)]],
        device const float * g2 [[buffer(11)]], device const float * g3 [[buffer(12)]],
        device const float * f0 [[buffer(13)]], device const float * f1 [[buffer(14)]],
        device const float * f2 [[buffer(15)]], device const float * f3 [[buffer(16)]],
        device       float * o0 [[buffer(17)]], device       float * o1 [[buffer(18)]],
        device       float * o2 [[buffer(19)]], device       float * o3 [[buffer(20)]],
        device const float * z    [[buffer(21)]],
        device       float * part [[buffer(22)]],
        uint3  tg   [[threadgroup_position_in_grid]],
        uint3  ntg  [[threadgroups_per_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[8];
    const int j = tg.y, t = tg.z;
    const int c0 = tg.x*64 + sg*8;
    const float4 zv = ((device const float4 *) (z + (j*ntg.z + t)*a.rank))[lane % 16];
    device const float4 * gw = (device const float4 *) INK_CTRL_SEL(g, j);
    device const float4 * fw = (device const float4 *) INK_CTRL_SEL(f, j);
    // lane l: channel c0 + 2 r + l / 16 of pass r, float4 l % 16 of its row
    float gs[4], fs[4];
    FOR_UNROLL (int r = 0; r < 4; r++) {
        const int c = min(c0 + 2*r + lane/16, a.e - 1);
        gs[r] = dot(gw[c*16 + lane % 16], zv);
        fs[r] = dot(fw[c*16 + lane % 16], zv);
    }
    FOR_UNROLL (int r = 0; r < 4; r++) {
        FOR_UNROLL (ushort o = 1; o < 16; o *= 2) {
            gs[r] += simd_shuffle_xor(gs[r], o);
            fs[r] += simd_shuffle_xor(fs[r], o);
        }
    }
    // channel c0 + i in lane i < 8: of pass i / 2, from lane 16 (i % 2)
    float gi = 0.f, fi = 0.f;
    FOR_UNROLL (int r = 0; r < 4; r++) {
        const float g = simd_shuffle(gs[r], (ushort) (16*(lane % 2)));
        const float f = simd_shuffle(fs[r], (ushort) (16*(lane % 2)));
        gi = lane/2 == r ? g : gi;
        fi = lane/2 == r ? f : fi;
    }
    const int c = c0 + lane;
    float y = 0.f;
    if (lane < 8 && c < a.e) {
        const float xi = INK_CTRL_SEL(x, j)[t*a.stride + c];
        const float mi = INK_CTRL_SEL(m, j)[t*a.stride + c];
        y = (xi - precise::tanh(fi)*xi) + precise::tanh(gi)*mi;
        INK_CTRL_SEL(o, j)[t*a.stride + c] = y;
    }
    const float ss = simd_sum(y*y);
    if (lane == 0) {
        red[sg] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0.f;
        for (int g = 0; g < 8; g++) {
            s += red[g];
        }
        part[(j*ntg.z + t)*a.chunks + tg.x] = s;
    }
}

// out *= rsqrt(ss/e + out_eps) * w, ss the sum of ink_ctrl_out's chunks, a
// float4 per thread, 64 per threadgroup; w of controller j at j * w_stride
kernel void ink_ctrl_norm(
        constant InkCtrlArgs & a [[buffer(0)]],
        device       float * o0 [[buffer(1)]], device       float * o1 [[buffer(2)]],
        device       float * o2 [[buffer(3)]], device       float * o3 [[buffer(4)]],
        device const float * w    [[buffer(5)]],
        device const float * part [[buffer(6)]],
        uint3  tg   [[threadgroup_position_in_grid]],
        uint3  ntg  [[threadgroups_per_grid]],
        ushort tid  [[thread_index_in_threadgroup]]) {
    const int j = tg.y, t = tg.z;
    const int i = (tg.x*64 + tid)*4;
    if (i >= a.e) {
        return;
    }
    device const float * pr = part + (j*ntg.z + t)*a.chunks;
    float ss = 0.f;
    FOR_UNROLL (int k = 0; k < 64; k++) {
        const float v = pr[k];
        ss += k < a.chunks ? v : 0.f;
    }
    const float scale = rsqrt(ss/a.e + a.out_eps);
    device float4 * o = (device float4 *) (INK_CTRL_SEL(o, j) + t*a.stride + i);
    *o = *o * scale * *(device const float4 *) (w + j*a.w_stride + i);
}

// ---------------------------------------------------------------------------
// ink_conv of long passes without saved states: a thread per (channel,
// token), the taps from the inputs or (before the pass) the state, the same
// sums as ink_conv; `out` is not `x`. The state then takes the pass's last 3
// inputs, by ink_conv_state (n >= 3).
kernel void ink_conv_par(
        constant InkConvArgs & a [[buffer(0)]],
        device const float * x     [[buffer(1)]],
        device const float * w     [[buffer(2)]],
        device const float * state [[buffer(3)]],
        device       float * out   [[buffer(4)]],
        device const float * dyn   [[buffer(5)]],
        uint2 gid [[thread_position_in_grid]]) {
    const int W = a.width;
    const int C = a.channels;
    const int c = gid.x, t = gid.y;
    if (c >= W) {
        return;
    }
    device const float * st = state + a.slot*3*C + a.offset + c;
    // the input j tokens before t
    float in[4];
    FOR_UNROLL (int j = 0; j < 4; j++) {
        const int tj = t - j;
        in[j] = tj >= 0 ? x[tj*W + c] : st[(3 + tj)*C];
    }
    const float xt = in[0], s2 = in[1], s1 = in[2], s0 = in[3];
    const float4 wc = *(device const float4 *) (w + 4*c);
    float acc = 0.f;
    acc += wc[0]*s0;
    acc += wc[1]*s1;
    acc += wc[2]*s2;
    acc += wc[3]*xt;
    if (a.dyn) {
        const float4 d = *(device const float4 *) (dyn + (c / a.head)*4 + t*a.dyn_stride);
        acc += xt*d[0];
        acc += s2*d[1];
        acc += s1*d[2];
        acc += s0*d[3];
    }
    const float y = (xt + acc)*a.scale;
    if (a.add) {
        out[t*W + c] += y;
    } else {
        out[t*W + c] = y;
    }
}

kernel void ink_conv_state(
        constant InkConvArgs & a [[buffer(0)]],
        device const float * x     [[buffer(1)]],
        device       float * state [[buffer(2)]],
        uint c [[thread_position_in_grid]]) {
    const int W = a.width;
    const int C = a.channels;
    if ((int) c >= W) {
        return;
    }
    device float * st = state + a.slot*3*C + a.offset + c;
    st[0]   = x[(a.n - 3)*W + c];
    st[C]   = x[(a.n - 2)*W + c];
    st[2*C] = x[(a.n - 1)*W + c];
}
