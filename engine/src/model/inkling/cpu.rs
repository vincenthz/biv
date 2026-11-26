//! The matmuls of `inkling2_loop` on the CPU, in few parallel loops: its
//! many small steps per token (48 slots, each with its projections,
//! controllers and experts) would otherwise each pay a dispatch to the
//! threads and wait for its slowest task. In a pass of few tokens, the
//! threads wait for the next loop busy ([`busy`]).
//!
//! The values are those of [`crate::ops::matmul`] bit for bit: each output
//! is the same [`quant::dot`] of a weight row and an activation row, only
//! computed in tiles of rows and tokens (on aarch64, Q8_0 weights: the
//! same instructions in the same order per output, the loads of a block
//! shared by the tile), and the steps after a matmul (SITU, the
//! controllers' gates, the experts' weighted sum) are done by the tasks
//! that computed its rows.

use super::situ;
#[cfg(not(any(feature = "std", test)))]
use crate::math::Float;
use crate::ops::Weight;
use crate::par::{self, *};
use crate::prelude::*;
use crate::quant::{self, Q8Act, Q8Slice};

/// Multiply-adds below which a parallel loop stays on the current thread:
/// low, the threads waiting busy in a pass of few tokens ([`busy`]), and a
/// pass of many having much more work
const SEQ: usize = 1 << 16;

/// Passes of at most this many tokens run with the threads busy, [`busy`]
#[cfg(feature = "rayon")]
pub(crate) const BUSY_TOKENS: usize = 32;

/// `f` with the other threads of the pool waiting for its parallel loops in
/// a busy loop, taking any work of the pool, rather than asleep: a pass of
/// few tokens is hundreds of loops of tens of microseconds, and between two
/// of them a sleeping thread takes about as long to wake up again
#[cfg(feature = "rayon")]
pub(crate) fn busy<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    use core::sync::atomic::{AtomicBool, Ordering};
    /// Ends the busy loops however `f` ends, a panic included
    struct Done<'a>(&'a AtomicBool);
    impl Drop for Done<'_> {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }
    let done = AtomicBool::new(false);
    rayon::scope(|s| {
        for _ in 1..rayon::current_num_threads() {
            s.spawn(|_| {
                while !done.load(Ordering::Acquire) {
                    if rayon::yield_now() != Some(rayon::Yield::Executed) {
                        core::hint::spin_loop();
                    }
                }
            });
        }
        let _done = Done(&done);
        f()
    })
}

/// Rows of a task: a multiple of the tiles' and of the quantization
/// blocks', enough tasks per thread to balance them
fn chunk_rows(rows: usize, align: usize) -> usize {
    let chunk = (rows / (par::threads() * 8)).clamp(align, 256);
    chunk / align * align
}

/// `f` on each task, in parallel when the work is at least [`SEQ`]
fn run<T: Send>(tasks: Vec<T>, work: usize, f: impl Fn(T) + Sync + Send) {
    if work < SEQ {
        tasks.into_iter().for_each(f);
    } else {
        tasks.into_par_iter().for_each(f);
    }
}

#[derive(Clone, Copy)]
struct SyncPtr<T>(*mut T);
unsafe impl<T> Sync for SyncPtr<T> {}
unsafe impl<T> Send for SyncPtr<T> {}

impl<T> SyncPtr<T> {
    /// The pointer, through the wrapper: a closure using it captures the
    /// wrapper, not its field
    fn get(&self) -> *mut T {
        self.0
    }
}

/// `n` rows of `k` activations, with their int8 quantization
/// ([`Q8Act`]) when a weight needs it
#[derive(Clone, Copy)]
pub(crate) struct Act<'a> {
    pub(crate) x: &'a [f32],
    pub(crate) k: usize,
    q: Option<(&'a [i8], &'a [f32], &'a [f32])>,
}

impl<'a> Act<'a> {
    /// `x`, its quantization in `q` when any of `weights` needs it
    pub(crate) fn new(x: &'a [f32], k: usize, q: &'a mut Q8Act, weights: &[&Weight<'_>]) -> Self {
        let quantize = weights.iter().any(|w| !w.ty.uses_f32_act());
        if quantize {
            quantize_rows(x, k, q);
        }
        Act::of(x, k, quantize.then_some(&*q))
    }

    /// `x` and, quantized already, `q`
    fn of(x: &'a [f32], k: usize, q: Option<&'a Q8Act>) -> Self {
        Act {
            x,
            k,
            q: q.map(|q| (&q.qs[..], &q.d[..], &q.s[..])),
        }
    }

    pub(crate) fn n(&self) -> usize {
        self.x.len() / self.k
    }

    fn row(&self, t: usize) -> &'a [f32] {
        &self.x[t * self.k..(t + 1) * self.k]
    }

    fn q8(&self, t: usize) -> Option<Q8Slice<'a>> {
        let k = self.k;
        self.q.map(|(qs, d, s)| Q8Slice {
            qs: &qs[t * k..(t + 1) * k],
            d: &d[t * k / 32..(t + 1) * k / 32],
            s: &s[t * k / 32..(t + 1) * k / 32],
        })
    }
}

/// A block of 32 values quantized as [`Q8Act::quantize_into`] does: its
/// scale and `scale * sum`
fn quantize_block(xb: &[f32], qb: &mut [i8]) -> (f32, f32) {
    let amax = xb.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    let mut sum = 0i32;
    for (q, &v) in qb.iter_mut().zip(xb) {
        let r = (v * id).round() as i32;
        *q = r as i8;
        sum += r;
    }
    (d, d * sum as f32)
}

/// [`Q8Act::quantize_into`] of the `x.len() / k` rows of `x`, in parallel
/// over rows when there are many
fn quantize_rows(x: &[f32], k: usize, q: &mut Q8Act) {
    assert!(
        k.is_multiple_of(32),
        "activation length must be a multiple of 32"
    );
    let nb = x.len() / 32;
    q.qs.resize(x.len(), 0);
    q.d.resize(nb, 0.0);
    q.s.resize(nb, 0.0);
    if x.len() >= 16 * k {
        x.par_chunks(k)
            .zip(q.qs.par_chunks_mut(k))
            .zip(q.d.par_chunks_mut(k / 32))
            .zip(q.s.par_chunks_mut(k / 32))
            .for_each(|(((x, q), d), s)| quantize_row(x, q, d, s));
    } else {
        quantize_row(x, &mut q.qs, &mut q.d, &mut q.s);
    }
}

/// The blocks of `x` into `q`, their scales `d` and sums `s`
fn quantize_row(x: &[f32], q: &mut [i8], d: &mut [f32], s: &mut [f32]) {
    let blocks = x.as_chunks::<32>().0.iter().zip(q.as_chunks_mut::<32>().0);
    for ((xb, qb), (d, s)) in blocks.zip(d.iter_mut().zip(s)) {
        (*d, *s) = quantize_block(xb, qb);
    }
}

/// `put(i, r, dot(w.row(r), x[toks[i]]))` for each row `r` of `rows` and
/// each `i`: the value of [`quant::dot`]
fn dot_rows(
    w: &Weight<'_>,
    rows: core::ops::Range<usize>,
    x: &Act<'_>,
    toks: &[usize],
    put: &mut impl FnMut(usize, usize, f32),
) {
    #[cfg(target_arch = "aarch64")]
    if w.ty == quant::GgmlType::Q8_0 {
        return neon::dot_rows_q8_0(w, rows, x, toks, put);
    }
    // as `ops::matmul`: tokens by four, then one by one
    let xq = |i: usize| x.q8(toks[i]);
    for r in rows {
        let wr = w.row(r);
        let mut i = 0;
        while i + 4 <= toks.len() {
            let q4 = x.q.map(|_| core::array::from_fn(|j| xq(i + j).unwrap()));
            let xf = core::array::from_fn(|j| x.row(toks[i + j]));
            for (j, v) in quant::dot_x4(w.ty, wr, xf, q4).into_iter().enumerate() {
                put(i + j, r, v);
            }
            i += 4;
        }
        for (i, &t) in toks.iter().enumerate().skip(i) {
            put(i, r, quant::dot(w.ty, wr, x.row(t), x.q8(t)));
        }
    }
}

/// `0..n`
pub(crate) fn all(n: usize) -> Vec<usize> {
    (0..n).collect()
}

/// The matmul of each weight and input of `jobs`, `[n][rows]` each, in one
/// parallel loop over runs of rows: [`crate::ops::matmul`]'s values
pub(crate) fn matmuls(jobs: &[(&Weight<'_>, &Act<'_>)]) -> Vec<Vec<f32>> {
    let mut outs: Vec<Vec<f32>> = jobs
        .iter()
        .map(|(w, x)| vec![0.0; x.n() * w.rows])
        .collect();
    let ptrs: Vec<SyncPtr<f32>> = outs.iter_mut().map(|o| SyncPtr(o.as_mut_ptr())).collect();
    let toks: Vec<Vec<usize>> = jobs.iter().map(|(_, x)| all(x.n())).collect();
    for (w, x) in jobs {
        assert_eq!(w.cols, x.k, "matmul input size mismatch");
    }
    let total: usize = jobs.iter().map(|(w, _)| w.rows).sum();
    let chunk = chunk_rows(total, 4);
    let tasks: Vec<(usize, usize)> = jobs
        .iter()
        .enumerate()
        .flat_map(|(j, (w, _))| (0..w.rows.div_ceil(chunk)).map(move |c| (j, c * chunk)))
        .collect();
    let work = jobs.iter().map(|(w, x)| x.n() * w.rows * w.cols).sum();
    run(tasks, work, |(j, r0)| {
        let (w, x) = jobs[j];
        let out = ptrs[j];
        let rows = w.rows;
        // SAFETY: each (t, r) output of a job is written by exactly one task
        let mut put = |t: usize, r: usize, v: f32| unsafe { *out.get().add(t * rows + r) = v };
        dot_rows(w, r0..(r0 + chunk).min(rows), x, &toks[j], &mut put);
    });
    outs
}

/// A controller of a looped layer from its bottleneck `z`: its gain and
/// forget gates, its input and message, `[n][e]` each
pub(crate) struct Gate<'a> {
    pub(crate) gain: &'a Weight<'a>,
    pub(crate) forget: &'a Weight<'a>,
    pub(crate) z: &'a Act<'a>,
    pub(crate) cur: &'a [f32],
    pub(crate) msg: &'a [f32],
}

/// The output of each controller of `jobs`, `[n][e]`, in one parallel
/// loop: `cur - tanh(forget · z) * cur + tanh(gain · z) * msg`
pub(crate) fn gates(jobs: &[Gate<'_>]) -> Vec<Vec<f32>> {
    let mut outs: Vec<Vec<f32>> = jobs.iter().map(|j| vec![0.0; j.cur.len()]).collect();
    let ptrs: Vec<SyncPtr<f32>> = outs.iter_mut().map(|o| SyncPtr(o.as_mut_ptr())).collect();
    let toks: Vec<Vec<usize>> = jobs.iter().map(|j| all(j.z.n())).collect();
    let total: usize = jobs.iter().map(|j| j.gain.rows).sum();
    let chunk = chunk_rows(total, 4);
    let tasks: Vec<(usize, usize)> = jobs
        .iter()
        .enumerate()
        .flat_map(|(j, job)| (0..job.gain.rows.div_ceil(chunk)).map(move |c| (j, c * chunk)))
        .collect();
    let work = jobs
        .iter()
        .map(|j| 2 * j.z.n() * j.gain.rows * j.gain.cols)
        .sum();
    run(tasks, work, |(j, r0)| {
        let Gate {
            gain,
            forget,
            z,
            cur,
            msg,
        } = jobs[j];
        let (out, e) = (ptrs[j], gain.rows);
        let rows = r0..(r0 + chunk).min(e);
        // SAFETY: each (t, r) output of a job is written by exactly one
        // task, the forget gate first, then the output from it
        let at = |t: usize, r: usize| unsafe { &mut *out.get().add(t * e + r) };
        dot_rows(forget, rows.clone(), z, &toks[j], &mut |t, r, f| {
            *at(t, r) = f
        });
        dot_rows(gain, rows, z, &toks[j], &mut |t, r, g| {
            let (o, x, m) = (at(t, r), cur[t * e + r], msg[t * e + r]);
            *o = (x - o.tanh() * x) + g.tanh() * m;
        });
    });
    outs
}

/// For each job `j` of `jobs`, `a(j, r0, r1)` on the runs of rows `r0..r1`
/// of its `ra.0` rows, `ra.1` at most each, then, once they are all done,
/// `b(j, r0, r1)` on the runs of its `rb.0` rows, `rb.1` at most each: in
/// parallel when the `work` is at least [`SEQ`], a job's `b` not waiting
/// for the other jobs' `a`
fn chained(
    jobs: usize,
    ra: (usize, usize),
    rb: (usize, usize),
    work: usize,
    a: &(impl Fn(usize, usize, usize) + Sync),
    b: &(impl Fn(usize, usize, usize) + Sync),
) {
    let runs = |(rows, run): (usize, usize)| {
        (0..rows)
            .step_by(run)
            .map(move |r| (r, (r + run).min(rows)))
    };
    #[cfg(feature = "rayon")]
    if work >= SEQ {
        use core::sync::atomic::{AtomicUsize, Ordering};
        // the runs of `a` left per job
        let left: Vec<AtomicUsize> = (0..jobs)
            .map(|_| AtomicUsize::new(ra.0.div_ceil(ra.1)))
            .collect();
        rayon::scope(|s| {
            for (j, left) in left.iter().enumerate() {
                for (r0, r1) in runs(ra) {
                    s.spawn(move |s| {
                        a(j, r0, r1);
                        // the last run of job j: its writes are seen by the
                        // tasks spawned below
                        if left.fetch_sub(1, Ordering::AcqRel) == 1 {
                            for (r0, r1) in runs(rb) {
                                s.spawn(move |_| b(j, r0, r1));
                            }
                        }
                    });
                }
            }
        });
        return;
    }
    let _ = work;
    for j in 0..jobs {
        runs(ra).for_each(|(r0, r1)| a(j, r0, r1));
        runs(rb).for_each(|(r0, r1)| b(j, r0, r1));
    }
}

/// An expert chosen by tokens of a pass: its gate, up and down, and its
/// tokens with their weights, in token order
pub(crate) struct Chosen<'w> {
    pub(crate) gate: Weight<'w>,
    pub(crate) up: Weight<'w>,
    pub(crate) down: Weight<'w>,
    pub(crate) toks: Vec<usize>,
    pub(crate) weights: Vec<f32>,
}

/// The mixture of experts output `[n][e]` of the tokens `h`: the sum over
/// each token's experts, in the order of `chosen`, of its weight times
/// `down(situ(gate h, up h))`. With few tokens per expert, each expert's
/// gates and ups with SITU and the quantization of its output, then its
/// downs, each in runs of rows ([`chained`]), the outputs then summed; else
/// two parallel loops: the gates and ups of all experts, then the downs with
/// the weighted sum, each task a run of rows of every expert.
pub(crate) fn experts(chosen: &[Chosen<'_>], h: &Act<'_>, beta: f32, lb: f32) -> Vec<f32> {
    let Some(first) = chosen.first() else {
        return vec![0.0; h.x.len()];
    };
    let (ff, e) = (first.gate.rows, first.down.rows);
    let quantize = chosen.iter().any(|c| !c.down.ty.uses_f32_act());
    // each expert's SITU outputs, `[tokens][ff]`, from row `at[j]`
    let at: Vec<usize> = chosen
        .iter()
        .scan(0, |n, c| {
            let at = *n;
            *n += c.toks.len();
            Some(at)
        })
        .collect();
    let rows: usize = chosen.iter().map(|c| c.toks.len()).sum();
    let mut act = vec![0.0f32; rows * ff];
    let mut q = Q8Act::default();
    if quantize {
        assert!(
            ff.is_multiple_of(32),
            "expert length must be a multiple of 32"
        );
        q.qs.resize(rows * ff, 0);
        q.d.resize(rows * ff / 32, 0.0);
        q.s.resize(rows * ff / 32, 0.0);
    }

    // rows r0..r1 of expert j: the gates and ups, SITU, then the
    // quantization of whole blocks
    let (pa, pq) = (
        SyncPtr(act.as_mut_ptr()),
        (
            SyncPtr(q.qs.as_mut_ptr()),
            SyncPtr(q.d.as_mut_ptr()),
            SyncPtr(q.s.as_mut_ptr()),
        ),
    );
    let gate_up = |j: usize, r0: usize, r1: usize| {
        let c = &chosen[j];
        // SAFETY: rows r0..r1 of the expert's outputs are this task's only
        let a = |i: usize, r: usize| unsafe { &mut *pa.get().add((at[j] + i) * ff + r) };
        dot_rows(&c.gate, r0..r1, h, &c.toks, &mut |i, r, g| *a(i, r) = g);
        dot_rows(&c.up, r0..r1, h, &c.toks, &mut |i, r, u| {
            let g = a(i, r);
            *g = situ(*g, u, beta, lb);
        });
        if quantize {
            for i in 0..c.toks.len() {
                let row = (at[j] + i) * ff;
                for b in (r0..r1).step_by(32) {
                    // SAFETY: as above, the blocks of rows r0..r1
                    let (xb, qb, d, s) = unsafe {
                        (
                            core::slice::from_raw_parts(pa.get().add(row + b), 32),
                            core::slice::from_raw_parts_mut(pq.0.get().add(row + b), 32),
                            &mut *pq.1.get().add((row + b) / 32),
                            &mut *pq.2.get().add((row + b) / 32),
                        )
                    };
                    (*d, *s) = quantize_block(xb, qb);
                }
            }
        }
    };
    let downs: Vec<Vec<usize>> = chosen
        .iter()
        .zip(&at)
        .map(|(c, &at)| (at..at + c.toks.len()).collect())
        .collect();
    let mut out = vec![0.0f32; h.x.len()];
    let work = rows * ff * (2 * h.k + e);

    // few tokens per expert (decode, verify): no wait between all the gates
    // and ups and all the downs. Each expert's gates and ups run in runs of
    // rows, and the last of them to end starts its downs, in runs of rows
    // too; the outputs are then summed in the order of the experts
    if chosen.iter().all(|c| c.toks.len() <= 4) {
        let mut ys = vec![0.0f32; rows * e];
        let py = SyncPtr(ys.as_mut_ptr());
        let down = |j: usize, r0: usize, r1: usize| {
            let (n, at) = (chosen[j].toks.len(), at[j]);
            // SAFETY: expert j's SITU outputs, all written before its downs
            // start; rows r0..r1 of its outputs are this task's only
            let act = unsafe {
                let x = core::slice::from_raw_parts(pa.get().add(at * ff), n * ff);
                let q = quantize.then(|| {
                    (
                        core::slice::from_raw_parts(pq.0.get().add(at * ff) as *const i8, n * ff),
                        core::slice::from_raw_parts(
                            pq.1.get().add(at * ff / 32) as *const f32,
                            n * ff / 32,
                        ),
                        core::slice::from_raw_parts(
                            pq.2.get().add(at * ff / 32) as *const f32,
                            n * ff / 32,
                        ),
                    )
                });
                Act { x, k: ff, q }
            };
            dot_rows(
                &chosen[j].down,
                r0..r1,
                &act,
                &all(n),
                &mut |i, r, y| unsafe {
                    *py.get().add((at + i) * e + r) = y;
                },
            );
        };
        // about 6 runs of each per thread
        let runs = (6 * par::threads()).div_ceil(chosen.len());
        let gu_rows = ff.div_ceil(runs).next_multiple_of(32);
        let down_rows = e.div_ceil(runs).next_multiple_of(4);
        chained(
            chosen.len(),
            (ff, gu_rows),
            (e, down_rows),
            work,
            &gate_up,
            &down,
        );
        for (c, &at) in chosen.iter().zip(&at) {
            for (i, (&t, &w)) in c.toks.iter().zip(&c.weights).enumerate() {
                let y = &ys[(at + i) * e..(at + i + 1) * e];
                quant::axpy(&mut out[t * e..(t + 1) * e], w, y);
            }
        }
        return out;
    }

    let chunk = chunk_rows(chosen.len() * ff, if quantize { 32 } else { 4 });
    let tasks: Vec<(usize, usize)> = (0..chosen.len())
        .flat_map(|j| (0..ff.div_ceil(chunk)).map(move |c| (j, c * chunk)))
        .collect();
    run(tasks, 2 * rows * ff * h.k, |(j, r0)| {
        gate_up(j, r0, (r0 + chunk).min(ff))
    });

    // the downs, each output row of every expert in turn, in runs of rows
    // that start at multiples of 8: each token's weighted sum is then the
    // values of `quant::axpy` over the whole row (fused multiply-adds by 8
    // with AVX2)
    let act = Act::of(&act, ff, quantize.then_some(&q));
    let po = SyncPtr(out.as_mut_ptr());
    let chunk = chunk_rows(e, 8);
    let most = chosen.iter().map(|c| c.toks.len()).max().unwrap_or(0);
    let tasks: Vec<usize> = (0..e.div_ceil(chunk)).map(|c| c * chunk).collect();
    run(tasks, rows * ff * e, |r0| {
        let r1 = (r0 + chunk).min(e);
        let m = r1 - r0;
        let mut ys = vec![0.0f32; most * m];
        for (c, toks) in chosen.iter().zip(&downs) {
            dot_rows(&c.down, r0..r1, &act, toks, &mut |i, r, y| {
                ys[i * m + r - r0] = y
            });
            for (i, (&t, &w)) in c.toks.iter().zip(&c.weights).enumerate() {
                // SAFETY: rows r0..r1 of the output are this task's only
                let o = unsafe { core::slice::from_raw_parts_mut(po.get().add(t * e + r0), m) };
                quant::axpy(o, w, &ys[i * m..(i + 1) * m]);
            }
        }
    });
    out
}

/// The Q8_0 kernel: tiles of up to 4 rows and 4 tokens, each output with
/// the instructions of `quant::neon::dot_q8_0` in the same order
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::Act;
    use crate::ops::Weight;
    use crate::quant::f16_to_f32;
    use core::arch::aarch64::*;

    /// The dots of the `R` rows from `w`, `rb` bytes apart, with the `T`
    /// activation rows `xq` (`nb` blocks, scales `xd`)
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn tile<const R: usize, const T: usize>(
        w: *const u8,
        rb: usize,
        xq: [*const i8; T],
        xd: [*const f32; T],
        nb: usize,
    ) -> [[f32; T]; R] {
        unsafe {
            let mut acc = [[vdupq_n_f32(0.0); T]; R];
            for i in 0..nb {
                let x0: [int8x16_t; T] = core::array::from_fn(|t| vld1q_s8(xq[t].add(32 * i)));
                let x1: [int8x16_t; T] = core::array::from_fn(|t| vld1q_s8(xq[t].add(32 * i + 16)));
                let dx: [f32; T] = core::array::from_fn(|t| *xd[t].add(i));
                for (r, acc) in acc.iter_mut().enumerate() {
                    let b = w.add(r * rb + 34 * i);
                    let w0 = vld1q_s8(b.add(2) as *const i8);
                    let w1 = vld1q_s8(b.add(18) as *const i8);
                    let dw = f16_to_f32((b as *const u16).read_unaligned());
                    for t in 0..T {
                        let p = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0, x0[t]), w1, x1[t]);
                        acc[t] = vfmaq_n_f32(acc[t], vcvtq_f32_s32(p), dw * dx[t]);
                    }
                }
            }
            acc.map(|a| a.map(|a| vaddvq_f32(a)))
        }
    }

    /// [`super::dot_rows`] of a Q8_0 weight
    pub(super) fn dot_rows_q8_0(
        w: &Weight<'_>,
        rows: core::ops::Range<usize>,
        x: &Act<'_>,
        toks: &[usize],
        put: &mut impl FnMut(usize, usize, f32),
    ) {
        let (qs, d, _) = x.q.expect("quantized activations required");
        let k = x.k;
        let nb = k / 32;
        let rb = w.data.len() / w.rows;
        assert!(rows.end <= w.rows && toks.iter().all(|&t| t < x.n()));
        assert!(qs.len() >= x.n() * k && d.len() >= x.n() * nb && w.cols == k);
        let xq = |t: usize| qs[t * k..].as_ptr();
        let xd = |t: usize| d[t * nb..].as_ptr();
        // the tile of the rows from `r` and the tokens from `i`
        macro_rules! tile {
            ($r:expr, $i:expr, $R:literal, $T:literal) => {{
                let (r, i) = ($r, $i);
                // SAFETY: the rows and blocks are in bounds, checked above
                let v = unsafe {
                    tile::<$R, $T>(
                        w.data.as_ptr().add(r * rb),
                        rb,
                        core::array::from_fn(|j| xq(toks[i + j])),
                        core::array::from_fn(|j| xd(toks[i + j])),
                        nb,
                    )
                };
                for (dr, v) in v.iter().enumerate() {
                    for (dt, &v) in v.iter().enumerate() {
                        put(i + dt, r + dr, v);
                    }
                }
            }};
        }
        let n = toks.len();
        let mut r = rows.start;
        while r < rows.end {
            let mut i = 0;
            if r + 4 <= rows.end {
                while i + 4 <= n {
                    tile!(r, i, 4, 4);
                    i += 4;
                }
                match n - i {
                    3 => tile!(r, i, 4, 3),
                    2 => tile!(r, i, 4, 2),
                    1 => tile!(r, i, 4, 1),
                    _ => {}
                }
                r += 4;
            } else {
                while i + 4 <= n {
                    tile!(r, i, 1, 4);
                    i += 4;
                }
                for i in i..n {
                    tile!(r, i, 1, 1);
                }
                r += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{self, Input};
    use crate::quant::GgmlType;

    /// Pseudo-random values in [-1, 1)
    fn values(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s >> 8) as f32 / (1u32 << 23) as f32 - 1.0
            })
            .collect()
    }

    /// `rows` rows of `cols` Q8_0 blocks
    fn q8_0(cols: usize, rows: usize, seed: u32) -> Vec<u8> {
        let v = values(cols * rows, seed);
        let mut out = Vec::new();
        for b in v.as_chunks::<32>().0 {
            let amax = b.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            let d = amax / 127.0;
            // an f16 scale: d rounded to 10 bits of mantissa, as bits
            let h = half_bits(d);
            out.extend(h.to_le_bytes());
            for &x in b {
                out.push((x / d).round().clamp(-127.0, 127.0) as i8 as u8);
            }
        }
        out
    }

    /// The f16 bits of a normal positive `x` (truncated)
    fn half_bits(x: f32) -> u16 {
        let b = x.to_bits();
        let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
        ((exp as u16) << 10) | ((b >> 13) & 0x3ff) as u16
    }

    #[test]
    fn quantized_rows_as_q8act() {
        let x = values(64 * 20, 3);
        let mut q = Q8Act::default();
        quantize_rows(&x, 64, &mut q);
        let want = Q8Act::quantize(&x);
        assert_eq!((q.qs, q.d, q.s), (want.qs, want.d, want.s));
    }

    #[test]
    fn matmuls_as_ops() {
        let (cols, rows) = (96, 37);
        let wq = q8_0(cols, rows, 1);
        let wf: Vec<u8> = values(cols * rows, 2)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        for ty in [GgmlType::Q8_0, GgmlType::F32] {
            let data = if ty == GgmlType::F32 { &wf } else { &wq };
            let w = Weight::new(ty, cols, rows, data);
            for n in [1, 2, 3, 4, 5, 7, 9] {
                let x = values(n * cols, 10 + n as u32);
                let want = ops::matmul_new(&w, &Input::new(&x, n));
                let mut q = Q8Act::default();
                let act = Act::new(&x, cols, &mut q, &[&w]);
                let got = matmuls(&[(&w, &act)]);
                assert_eq!(
                    got[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "{ty:?} n={n}"
                );
            }
        }
    }

    #[test]
    fn experts_as_ops() {
        // 8 experts of Q8_0 weights, enough work for the parallel loops
        let (e, ff, k) = (256, 128, 8);
        let gate: Vec<Vec<u8>> = (0..k).map(|j| q8_0(e, ff, 20 + j)).collect();
        let up: Vec<Vec<u8>> = (0..k).map(|j| q8_0(e, ff, 40 + j)).collect();
        let down: Vec<Vec<u8>> = (0..k).map(|j| q8_0(ff, e, 60 + j)).collect();
        fn w(d: &[u8], cols: usize, rows: usize) -> Weight<'_> {
            Weight::new(GgmlType::Q8_0, cols, rows, d)
        }
        let (beta, lb) = (2.0, 3.0);
        // decode, a verify, and a pass of more tokens per expert than few
        for n in [1, 3, 6] {
            let h = values(n * e, 7 + n as u32);
            let chosen: Vec<Chosen<'_>> = (0..k as usize)
                .map(|j| {
                    // the tokens of expert j: every token but some
                    let toks: Vec<usize> = (0..n).filter(|t| (t + j) % 4 != 3).collect();
                    let weights = toks.iter().map(|&t| 0.1 + (t + j) as f32 * 0.01).collect();
                    Chosen {
                        gate: w(&gate[j], e, ff),
                        up: w(&up[j], e, ff),
                        down: w(&down[j], ff, e),
                        toks,
                        weights,
                    }
                })
                .filter(|c| !c.toks.is_empty())
                .collect();
            let mut q = Q8Act::default();
            let input = Act::new(&h, e, &mut q, &[&chosen[0].gate]);
            let got = experts(&chosen, &input, beta, lb);
            // the same with `ops::matmul` over each expert's rows
            let mut want = vec![0.0f32; n * e];
            for c in &chosen {
                let x: Vec<f32> = c
                    .toks
                    .iter()
                    .flat_map(|&t| h[t * e..(t + 1) * e].to_vec())
                    .collect();
                let m = c.toks.len();
                let x = Input::new(&x, m);
                let (g, u) = (ops::matmul_new(&c.gate, &x), ops::matmul_new(&c.up, &x));
                let a: Vec<f32> = g
                    .iter()
                    .zip(&u)
                    .map(|(&g, &u)| situ(g, u, beta, lb))
                    .collect();
                let y = ops::matmul_new(&c.down, &Input::new(&a, m));
                for ((&t, &wt), y) in c.toks.iter().zip(&c.weights).zip(y.chunks_exact(e)) {
                    quant::axpy(&mut want[t * e..(t + 1) * e], wt, y);
                }
            }
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "n={n}"
            );
        }
    }
}
