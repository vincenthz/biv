//! CPU compute: matmul (threaded with rayon) over quantized weights and the small
//! element-wise ops of the graph.

use core::cell::OnceCell;

use crate::math::Float;
use crate::par::{self, *};
use crate::prelude::*;
use crate::quant::{self, GgmlType, Q8Act};

/// A 2D weight matrix in ggml layout: `rows` rows of `cols` contiguous elements
#[derive(Clone, Copy)]
pub struct Weight<'a> {
    pub ty: GgmlType,
    pub cols: usize,
    pub rows: usize,
    pub data: &'a [u8],
    row_bytes: usize,
}

impl<'a> Weight<'a> {
    pub fn new(ty: GgmlType, cols: usize, rows: usize, data: &'a [u8]) -> Self {
        let row_bytes = ty.row_size(cols);
        assert_eq!(data.len(), row_bytes * rows, "weight size mismatch");
        Weight {
            ty,
            cols,
            rows,
            data,
            row_bytes,
        }
    }

    pub fn row(&self, r: usize) -> &'a [u8] {
        &self.data[r * self.row_bytes..(r + 1) * self.row_bytes]
    }

    /// Dequantize row `r` (e.g. an embedding lookup)
    pub fn dequantize_row(&self, r: usize, out: &mut [f32]) {
        quant::dequantize_row(self.ty, self.row(r), &mut out[..self.cols]);
    }
}

/// Matmul input: `n` rows of activations, quantized on first use and shared
/// by every weight that consumes it
pub struct Input<'x> {
    pub x: &'x [f32],
    pub n: usize,
    q: OnceCell<Q8Act>,
}

impl<'x> Input<'x> {
    pub fn new(x: &'x [f32], n: usize) -> Self {
        debug_assert_eq!(x.len() % n, 0);
        Input {
            x,
            n,
            q: OnceCell::new(),
        }
    }

    fn q8(&self) -> &Q8Act {
        self.q.get_or_init(|| {
            let k = self.x.len() / self.n;
            let mut q = Q8Act::default();
            // quantize rows in parallel; each row is block-aligned
            let rows: Vec<Q8Act> = self.x.par_chunks(k).map(Q8Act::quantize).collect();
            for r in rows {
                q.qs.extend(r.qs);
                q.d.extend(r.d);
                q.s.extend(r.s);
            }
            q
        })
    }
}

struct SyncPtr(*mut f32);
unsafe impl Sync for SyncPtr {}
unsafe impl Send for SyncPtr {}

/// `out[t][r] = dot(w.row(r), x[t])` for every input row `t`
pub fn matmul(w: &Weight<'_>, input: &Input<'_>, out: &mut [f32]) {
    let n = input.n;
    let k = w.cols;
    assert_eq!(input.x.len(), n * k, "matmul input size mismatch");
    assert_eq!(out.len(), n * w.rows, "matmul output size mismatch");
    let xq = if w.ty.uses_f32_act() {
        None
    } else {
        Some(input.q8())
    };

    let threads = par::threads();
    // enough tasks to balance across cores, each a run of whole rows
    let chunk = (w.rows / (threads * 8)).clamp(1, 256);
    let n_chunks = w.rows.div_ceil(chunk);
    let out_ptr = SyncPtr(out.as_mut_ptr());
    let rows = w.rows;

    (0..n_chunks).into_par_iter().for_each(|c| {
        let out_ptr = &out_ptr;
        let r0 = c * chunk;
        let r1 = (r0 + chunk).min(rows);
        let xf = |t: usize| &input.x[t * k..(t + 1) * k];
        let xqs = |t: usize| xq.map(|q| q.slice(t * k, k));
        // SAFETY: each (t, r) output is written by exactly one task
        let put = |t: usize, r: usize, v: f32| unsafe { *out_ptr.0.add(t * rows + r) = v };
        for r in r0..r1 {
            let wr = w.row(r);
            // the weight row stays hot in L1 while it meets every input row
            let mut t = 0;
            while t + 4 <= n {
                let xq4 = xq.map(|_| core::array::from_fn(|i| xqs(t + i).unwrap()));
                let v = quant::dot_x4(w.ty, wr, core::array::from_fn(|i| xf(t + i)), xq4);
                for (i, v) in v.into_iter().enumerate() {
                    put(t + i, r, v);
                }
                t += 4;
            }
            for t in t..n {
                put(t, r, quant::dot(w.ty, wr, xf(t), xqs(t)));
            }
        }
    });
}

pub fn matmul_new(w: &Weight<'_>, input: &Input<'_>) -> Vec<f32> {
    let mut out = vec![0.0; input.n * w.rows];
    matmul(w, input, &mut out);
    out
}

/// RMS norm of each `w.len()`-sized group of `x`, scaled by `w`
pub fn rms_norm(x: &mut [f32], w: &[f32], eps: f32) {
    for g in x.chunks_exact_mut(w.len()) {
        let ss = quant::dot_f32(g, g) / g.len() as f32;
        let scale = 1.0 / (ss + eps).sqrt();
        for (v, w) in g.iter_mut().zip(w) {
            *v *= scale * w;
        }
    }
}

pub fn rms_norm_into(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    out.copy_from_slice(x);
    rms_norm(out, w, eps);
}

/// `x / sqrt(sum(x^2) + eps)` on each `group`-sized slice
pub fn l2_norm(x: &mut [f32], group: usize, eps: f32) {
    for g in x.chunks_exact_mut(group) {
        let scale = 1.0 / (quant::dot_f32(g, g) + eps).sqrt();
        g.iter_mut().for_each(|v| *v *= scale);
    }
}

/// The input below which `sigmoid` computes `exp(x) / (1 + exp(x))` in
/// place of `1 / (1 + exp(-x))`, on the CPU and in the Metal kernels, which
/// get it as a define. Below it `exp(-x)` can overflow: ln(f32::MAX) is about
/// 88.7, and the margin of about 8.7 covers the error of the fast exp of
/// Metal. At or above it both paths keep the forms of ggml, so their results
/// do not change there.
pub const SIGMOID_OVERFLOW_X: f32 = -80.0;

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    if x < SIGMOID_OVERFLOW_X {
        let e = Float::exp(x);
        e / (1.0 + e)
    } else {
        1.0 / (1.0 + Float::exp(-x))
    }
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

#[inline]
pub fn softplus(x: f32) -> f32 {
    // same threshold as ggml
    if x > 20.0 {
        x
    } else {
        Float::ln_1p(Float::exp(x))
    }
}

pub fn add_inplace(x: &mut [f32], y: &[f32]) {
    x.iter_mut().zip(y).for_each(|(a, b)| *a += b);
}

/// NeoX rotary embedding of the first `2 * inv_freq.len()` dims of each head
pub fn rope_neox(x: &mut [f32], head_dim: usize, pos: usize, inv_freq: &[f32]) {
    let half = inv_freq.len();
    for head in x.chunks_exact_mut(head_dim) {
        for (i, &f) in inv_freq.iter().enumerate() {
            let (sin, cos) = (pos as f32 * f).sin_cos();
            let (a, b) = (head[i], head[i + half]);
            head[i] = a * cos - b * sin;
            head[i + half] = a * sin + b * cos;
        }
    }
}

pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = Float::exp(*v - max);
        sum += *v;
    }
    let inv = 1.0 / sum;
    x.iter_mut().for_each(|v| *v *= inv);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Inputs at and above SIGMOID_OVERFLOW_X, in steps of this size
    const STEP: f32 = 0.25;
    /// The inputs far below SIGMOID_OVERFLOW_X, where exp(-x) overflows
    const DEAD: [f32; 3] = [-104.0, -1.0e4, f32::MIN];

    #[test]
    fn sigmoid_keeps_the_ggml_form_at_and_above_the_overflow_input() {
        let mut x = SIGMOID_OVERFLOW_X;
        while x < -SIGMOID_OVERFLOW_X {
            let ggml = 1.0 / (1.0 + Float::exp(-x));
            assert_eq!(sigmoid(x).to_bits(), ggml.to_bits(), "x = {x}");
            assert_eq!(silu(x).to_bits(), (x * ggml).to_bits(), "x = {x}");
            x += STEP;
        }
    }

    #[test]
    fn sigmoid_below_the_overflow_input_is_finite() {
        for x in DEAD {
            assert_eq!(sigmoid(x), 0.0, "x = {x}");
            assert!(silu(x).is_finite(), "x = {x}");
        }
    }
}
