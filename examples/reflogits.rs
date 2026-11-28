//! CPU logits against reference logits for the same token ids, e.g. llama.cpp's
//! `llama-results` converted to raw files: `<ref>.tokens.u32` (u32 ids),
//! `<ref>.logits.f32` (row-major `[rows][n_vocab]` f32) and an optional
//! `<ref>.meta` (`key=value` lines: `n_vocab`, `start_row`, `rows`) when the
//! logits cover only the positions `start_row..start_row + rows`.
//!
//! `reflogits <gguf> <ref> [--mode M] [--warm N] [--context N] [--limit N]
//! [--out FILE] [--tsv FILE] [--all]`, modes:
//! - `prefill` (default): one prefill of all but the last token, a forward of
//!   the last; the last position only
//! - `all`: one verify pass over every compared position (the prefill path
//!   with logits after each token); every position. A verify pass keeps the
//!   recurrent states after each token, which on a delta net model (Qwen3.5)
//!   is megabytes per token: use `verify=N` there
//! - `token`: a forward per token (the decode path); every position
//! - `chunk=N`: a forward per N tokens (the prefill path); the last position
//!   of every chunk
//! - `verify=N`: a verify (logits after each token) and keep per N tokens, in a
//!   session taking N - 1 draft tokens; every position
//!
//! The positions before the reference's first row are prefilled in chunks of
//! `--warm` tokens (512) before the mode starts. Per position: argmax
//! agreement, max |logit difference| and KL(ref || cpu) of the softmaxes over
//! the reference's vocabulary (the unpadded one), then their distribution.
//! `--out` writes the cpu logits of the compared positions, row-major f32, and
//! `<FILE>.meta`; `--tsv` writes the per-position table of every position.
//! `--against <prefix>` loads no model (the gguf argument is ignored, `-` will
//! do) and compares the reference with another logits file of the same tokens,
//! e.g. a second llama.cpp run or a `--out` of an earlier run (its `.meta` has
//! `start_row` when the positions are contiguous, i.e. not in `chunk=N` mode).
//! The engine is the model itself, i.e. the CPU backend, or with
//! `--gpu [max_batch]` the GPU one (Metal on macOS, CUDA with the `cuda`
//! feature; `--metal` is the same), of passes of `max_batch` tokens (512,
//! more for a longer `verify=N`).
use std::fs;
use std::process::exit;
use std::time::Instant;

use biv::{
    engine::{Engine, Session},
    model::Model,
};

#[derive(Clone, Copy, Debug)]
enum Mode {
    Prefill,
    All,
    Token,
    Chunk(usize),
    Verify(usize),
}

impl Mode {
    fn parse(s: &str) -> Option<Mode> {
        let n = |v: &str| v.parse::<usize>().ok().filter(|&n| n > 0);
        Some(match s.split_once('=') {
            None if s == "prefill" => Mode::Prefill,
            None if s == "all" => Mode::All,
            None if s == "token" => Mode::Token,
            Some(("chunk", v)) => Mode::Chunk(n(v)?),
            Some(("verify", v)) => Mode::Verify(n(v)?),
            _ => return None,
        })
    }
}

struct Args {
    model: String,
    reference: String,
    mode: Mode,
    warm: usize,
    context: Option<usize>,
    limit: Option<usize>,
    out: Option<String>,
    tsv: Option<String>,
    against: Option<String>,
    all: bool,
    /// the GPU engine, of passes of this many tokens
    gpu: Option<usize>,
}

const USAGE: &str = "usage: reflogits <gguf> <ref prefix> [--mode prefill|all|token|chunk=N|verify=N] \
                     [--warm N] [--context N] [--limit N] [--out FILE] [--tsv FILE] [--against PREFIX] [--all] [--gpu [BATCH]]";

fn die(msg: impl std::fmt::Display) -> ! {
    eprintln!("reflogits: {msg}");
    exit(2)
}

fn args() -> Args {
    let mut it = std::env::args().skip(1).peekable();
    let mut pos = Vec::new();
    let mut a = Args {
        model: String::new(),
        reference: String::new(),
        mode: Mode::Prefill,
        warm: 512,
        context: None,
        limit: None,
        out: None,
        tsv: None,
        against: None,
        all: false,
        gpu: None,
    };
    let num = |v: Option<String>| -> usize {
        v.and_then(|v| v.parse().ok()).unwrap_or_else(|| die(USAGE))
    };
    while let Some(s) = it.next() {
        match s.as_str() {
            "--mode" => {
                let v = it.next().unwrap_or_else(|| die(USAGE));
                a.mode = Mode::parse(&v).unwrap_or_else(|| die(format!("bad mode {v:?}")));
            }
            "--warm" => a.warm = num(it.next()).max(1),
            "--context" => a.context = Some(num(it.next())),
            "--limit" => a.limit = Some(num(it.next())),
            "--out" => a.out = Some(it.next().unwrap_or_else(|| die(USAGE))),
            "--tsv" => a.tsv = Some(it.next().unwrap_or_else(|| die(USAGE))),
            "--against" => a.against = Some(it.next().unwrap_or_else(|| die(USAGE))),
            "--all" => a.all = true,
            "--gpu" | "--metal" => {
                // an optional batch size
                let batch = it.peek().and_then(|v| v.parse().ok());
                if batch.is_some() {
                    it.next();
                }
                a.gpu = Some(batch.unwrap_or(512));
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                exit(0)
            }
            _ if s.starts_with("--") => die(format!("unknown option {s}\n{USAGE}")),
            _ => pos.push(s),
        }
    }
    let [model, reference] = <[String; 2]>::try_from(pos).unwrap_or_else(|_| die(USAGE));
    a.model = model;
    a.reference = reference;
    a
}

/// The reference: every token, and the logits of `rows` positions from `start`
struct Reference {
    tokens: Vec<u32>,
    logits: Vec<f32>,
    n_vocab: usize,
    start: usize,
    rows: usize,
}

impl Reference {
    fn load(prefix: &str) -> Reference {
        let read = |ext: &str| {
            let p = format!("{prefix}.{ext}");
            fs::read(&p).unwrap_or_else(|e| die(format!("{p}: {e}")))
        };
        let tokens: Vec<u32> = read("tokens.u32")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| u32::from_le_bytes(b))
            .collect();
        let logits: Vec<f32> = read("logits.f32")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b))
            .collect();
        let meta = fs::read_to_string(format!("{prefix}.meta")).unwrap_or_default();
        let key = |k: &str| -> Option<usize> {
            meta.lines()
                .filter_map(|l| l.split_once('='))
                .find(|(a, _)| a.trim() == k)
                .and_then(|(_, v)| v.trim().parse().ok())
        };
        let start = key("start_row").unwrap_or(0);
        let rows = key("rows").unwrap_or(tokens.len() - start);
        let n_vocab = key("n_vocab").unwrap_or(logits.len() / rows.max(1));
        if rows == 0 || start + rows > tokens.len() || rows * n_vocab != logits.len() {
            die(format!(
                "{prefix}: {} tokens, {} logits do not make {rows} rows of {n_vocab} from {start}",
                tokens.len(),
                logits.len()
            ));
        }
        Reference {
            tokens,
            logits,
            n_vocab,
            start,
            rows,
        }
    }

    /// The logits at position `pos`, if the reference has them
    fn row(&self, pos: usize) -> Option<&[f32]> {
        let r = pos.checked_sub(self.start).filter(|&r| r < self.rows)?;
        Some(&self.logits[r * self.n_vocab..(r + 1) * self.n_vocab])
    }
}

/// Prefill `ids` in chunks of `chunk` tokens
fn warm(s: &mut dyn Session, ids: &[u32], chunk: usize) {
    for c in ids.chunks(chunk) {
        s.prefill(c);
    }
}

/// The comparison at one position
struct Stat {
    pos: usize,
    ref_top: usize,
    ref_logit: f32,
    /// the reference's top logit minus its second
    ref_margin: f32,
    cpu_top: usize,
    cpu_logit: f32,
    /// how many cpu logits beat the cpu logit of the reference's top token
    rank: usize,
    max_diff: f32,
    mean_diff: f64,
    /// the mean of cpu - reference: a shift of every logit, which the softmax ignores
    shift: f64,
    /// max |cpu - reference - shift|
    max_centered: f64,
    kl: f64,
}

fn top2(v: &[f32]) -> (usize, f32, f32) {
    let (mut i1, mut l1, mut l2) = (0, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (i, &l) in v.iter().enumerate() {
        if l > l1 {
            (l2, i1, l1) = (l1, i, l);
        } else if l > l2 {
            l2 = l;
        }
    }
    (i1, l1, l2)
}

/// log of the softmax, in f64
fn log_softmax(v: &[f32]) -> Vec<f64> {
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let s: f64 = v.iter().map(|&x| (x as f64 - m).exp()).sum();
    let ls = s.ln();
    v.iter().map(|&x| x as f64 - m - ls).collect()
}

fn compare(pos: usize, r: &[f32], c: &[f32]) -> Stat {
    let n = r.len().min(c.len());
    let (r, c) = (&r[..n], &c[..n]);
    let (ref_top, ref_logit, ref_second) = top2(r);
    let (cpu_top, cpu_logit, _) = top2(c);
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f64;
    let mut sum_signed = 0.0f64;
    for (a, b) in r.iter().zip(c) {
        let d = (a - b).abs();
        // NaN propagates to the worst case
        max_diff = if d.is_nan() {
            f32::NAN
        } else {
            max_diff.max(d)
        };
        sum_diff += d as f64;
        sum_signed += *b as f64 - *a as f64;
    }
    let shift = sum_signed / n as f64;
    let max_centered = r
        .iter()
        .zip(c)
        .map(|(a, b)| (*b as f64 - *a as f64 - shift).abs())
        .fold(0.0f64, |m, d| if d.is_nan() { f64::NAN } else { m.max(d) });
    let (lr, lc) = (log_softmax(r), log_softmax(c));
    let kl = lr.iter().zip(&lc).map(|(a, b)| a.exp() * (a - b)).sum();
    let rank = c.iter().filter(|&&x| x > c[ref_top]).count();
    Stat {
        pos,
        ref_top,
        ref_logit,
        ref_margin: ref_logit - ref_second,
        cpu_top,
        cpu_logit,
        rank,
        max_diff,
        mean_diff: sum_diff / n as f64,
        shift,
        max_centered,
        kl,
    }
}

fn line(s: &Stat, tok: u32) {
    println!(
        "{:6} {:7} {:>4} {:7} {:9.4} {:7} {:9.4} {:8.4} {:5} {:9.5} {:9.6} {:9.5} {:9.5} {:10.3e}",
        s.pos,
        tok,
        if s.ref_top == s.cpu_top { "ok" } else { "FLIP" },
        s.ref_top,
        s.ref_logit,
        s.cpu_top,
        s.cpu_logit,
        s.ref_margin,
        s.rank,
        s.max_diff,
        s.mean_diff,
        s.shift,
        s.max_centered,
        s.kl
    );
}

fn header() {
    println!(
        "{:>6} {:>7} {:>4} {:>7} {:>9} {:>7} {:>9} {:>8} {:>5} {:>9} {:>9} {:>9} {:>9} {:>10}",
        "pos",
        "token",
        "top1",
        "ref",
        "logit",
        "cpu",
        "logit",
        "margin",
        "rank",
        "max|d|",
        "mean|d|",
        "shift",
        "max|d-s|",
        "KL"
    );
}

/// The value at quantile `q` of sorted `v`
fn quantile(v: &[f64], q: f64) -> f64 {
    v[((v.len() - 1) as f64 * q).round() as usize]
}

fn dist(name: &str, mut v: Vec<f64>) -> (f64, f64) {
    v.sort_by(f64::total_cmp);
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let worst = *v.last().unwrap();
    println!(
        "{name:>8}: worst {worst:.4e}  mean {mean:.4e}  p50 {:.4e}  p90 {:.4e}  p99 {:.4e}",
        quantile(&v, 0.5),
        quantile(&v, 0.9),
        quantile(&v, 0.99)
    );
    (worst, mean)
}

fn main() {
    let a = args();
    let reference = Reference::load(&a.reference);
    let mut ids = reference.tokens.clone();
    if let Some(l) = a.limit {
        ids.truncate(l);
    }
    let n = ids.len();
    if n == 0 || n <= reference.start {
        die(format!(
            "{n} tokens leave no reference row (the first is at {})",
            reference.start
        ));
    }
    let first = reference.start;
    let mode = match a.mode {
        Mode::All => Mode::Verify(n - first),
        m => m,
    };

    let mut stats: Vec<Stat> = Vec::new();
    let mut kept: Vec<(usize, Vec<f32>)> = Vec::new();
    let mut vocab_note = false;
    let mut record = |pos: usize, logits: Vec<f32>| {
        let Some(r) = reference.row(pos) else {
            return;
        };
        if logits.len() != r.len() && !vocab_note {
            println!(
                "note: {} cpu logits, {} reference ones: comparing the first {}",
                logits.len(),
                r.len(),
                r.len().min(logits.len())
            );
            vocab_note = true;
        }
        stats.push(compare(pos, r, &logits));
        if a.out.is_some() {
            kept.push((pos, logits));
        }
    };

    if let Some(other) = &a.against {
        // two logits files of the same tokens, no model: "cpu" is `other`
        let o = Reference::load(other);
        if o.tokens[..n.min(o.tokens.len())] != ids[..n.min(o.tokens.len())] {
            die(format!("{other}: not the tokens of {}", a.reference));
        }
        println!(
            "reference {}: rows {first}..{} of {}; against {other}: rows {}..{} of {}",
            a.reference,
            first + reference.rows,
            reference.n_vocab,
            o.start,
            o.start + o.rows,
            o.n_vocab
        );
        for pos in first..n {
            if let Some(l) = o.row(pos) {
                record(pos, l.to_vec());
            }
        }
    } else {
        let t = Instant::now();
        let g = biv_cli::open_gguf(&a.model).unwrap_or_else(|e| die(e));
        let model = Model::load(&g).unwrap_or_else(|e| die(e));
        let context = a.context.unwrap_or(n).max(n);
        let draft = match mode {
            Mode::Verify(c) => c - 1,
            _ => 0,
        };
        let engine: Box<dyn Engine + '_> = match a.gpu {
            None => Box::new(&model),
            Some(batch) => biv_cli::gpu_engine(&model, batch.max(draft + 1))
                .unwrap_or_else(|e| die(e))
                .unwrap_or_else(|| die("no GPU backend in this build")),
        };
        let mut s = engine
            .new_session(context, draft)
            .unwrap_or_else(|e| die(e));
        println!(
            "model {} loaded in {:?}: {}; reference {}: {} tokens, rows {}..{} of {} logits",
            a.model,
            t.elapsed(),
            engine.description(),
            a.reference,
            reference.tokens.len(),
            first,
            first + reference.rows,
            reference.n_vocab
        );
        println!(
            "mode {:?} ({mode:?}), {n} tokens, context {context}",
            a.mode
        );

        let t = Instant::now();
        match mode {
            Mode::All => unreachable!("all is one verify pass"),
            Mode::Prefill => {
                if n > 1 {
                    s.prefill(&ids[..n - 1]);
                }
                let l = s.forward(&ids[n - 1..]);
                record(n - 1, l);
            }
            Mode::Token => {
                warm(s.as_mut(), &ids[..first], a.warm);
                for p in first..n {
                    let l = s.forward(&ids[p..p + 1]);
                    record(p, l);
                }
            }
            Mode::Chunk(c) => {
                warm(s.as_mut(), &ids[..first], a.warm);
                let mut p = first;
                for chunk in ids[first..].chunks(c) {
                    let l = s.forward(chunk);
                    p += chunk.len();
                    record(p - 1, l);
                }
            }
            Mode::Verify(c) => {
                warm(s.as_mut(), &ids[..first], a.warm);
                let mut p = first;
                for chunk in ids[first..].chunks(c) {
                    let ls = s.verify(chunk);
                    s.keep(chunk.len());
                    for (i, l) in ls.into_iter().enumerate() {
                        record(p + i, l);
                    }
                    p += chunk.len();
                }
            }
        }
        let el = t.elapsed();
        println!(
            "ran {n} tokens in {el:?} ({:.1} tok/s)",
            n as f64 / el.as_secs_f64(),
        );
    }
    println!("{} positions compared", stats.len());
    if stats.is_empty() {
        die("no position compared");
    }

    // every position when few, else the flips and the worst ones by KL
    header();
    if a.all || stats.len() <= 64 {
        for st in &stats {
            line(st, ids[st.pos]);
        }
    } else {
        let mut idx: Vec<usize> = (0..stats.len()).collect();
        idx.sort_by(|&x, &y| stats[y].kl.total_cmp(&stats[x].kl));
        let mut show: Vec<usize> = idx[..10.min(idx.len())].to_vec();
        show.extend((0..stats.len()).filter(|&i| stats[i].ref_top != stats[i].cpu_top));
        show.sort();
        show.dedup();
        for i in show {
            line(&stats[i], ids[stats[i].pos]);
        }
    }

    let agree = stats.iter().filter(|s| s.ref_top == s.cpu_top).count();
    let nonfinite = stats
        .iter()
        .filter(|s| !s.max_diff.is_finite() || !s.kl.is_finite())
        .count();
    let flips: Vec<&Stat> = stats.iter().filter(|s| s.ref_top != s.cpu_top).collect();
    let worst_flip_margin = flips.iter().map(|s| s.ref_margin).fold(0.0f32, f32::max);
    println!(
        "top-1 agreement: {agree}/{} ({:.2}%), largest reference margin among flips {worst_flip_margin:.4}{}",
        stats.len(),
        100.0 * agree as f64 / stats.len() as f64,
        if nonfinite > 0 {
            format!(", {nonfinite} positions NOT FINITE")
        } else {
            String::new()
        }
    );
    let (dmax, dmean) = dist("max|d|", stats.iter().map(|s| s.max_diff as f64).collect());
    dist("mean|d|", stats.iter().map(|s| s.mean_diff).collect());
    let (cmax, cmean) = dist("max|d-s|", stats.iter().map(|s| s.max_centered).collect());
    let (kmax, kmean) = dist("KL", stats.iter().map(|s| s.kl).collect());
    let worst_d = stats
        .iter()
        .max_by(|x, y| x.max_diff.total_cmp(&y.max_diff))
        .unwrap();
    let worst_k = stats.iter().max_by(|x, y| x.kl.total_cmp(&y.kl)).unwrap();
    println!(
        "worst max|d| at {}, worst KL at {}",
        worst_d.pos, worst_k.pos
    );
    // how often each side's top token is the prompt's next token: a quality
    // check that doesn't depend on the noise between the two
    let next: Vec<(&Stat, usize)> = stats
        .iter()
        .filter_map(|s| Some((s, *reference.tokens.get(s.pos + 1)? as usize)))
        .collect();
    let hits = |f: fn(&Stat) -> usize| next.iter().filter(|(s, t)| f(s) == *t).count();
    let (ref_next, cpu_next) = (hits(|s| s.ref_top), hits(|s| s.cpu_top));
    let pct = |h: usize| 100.0 * h as f64 / next.len().max(1) as f64;
    println!(
        "next-token accuracy over {} positions: reference {:.2}%, cpu {:.2}%",
        next.len(),
        pct(ref_next),
        pct(cpu_next)
    );
    println!(
        "SUMMARY ref={} mode={} positions={} top1={:.2}% maxd_worst={dmax:.4e} maxd_mean={dmean:.4e} maxc_worst={cmax:.4e} maxc_mean={cmean:.4e} kl_worst={kmax:.4e} kl_mean={kmean:.4e} next_ref={:.2}% next_cpu={:.2}%",
        a.reference,
        match &a.against {
            Some(o) => format!("against:{o}"),
            None => format!("{:?}", a.mode),
        },
        stats.len(),
        100.0 * agree as f64 / stats.len() as f64,
        pct(ref_next),
        pct(cpu_next)
    );

    if let Some(path) = &a.tsv {
        let mut t = String::from(
            "pos\ttoken\tref_top\tref_logit\tref_margin\tcpu_top\tcpu_logit\trank\tmax_diff\tmean_diff\tshift\tmax_centered\tkl\n",
        );
        for st in &stats {
            t.push_str(&format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                st.pos,
                ids[st.pos],
                st.ref_top,
                st.ref_logit,
                st.ref_margin,
                st.cpu_top,
                st.cpu_logit,
                st.rank,
                st.max_diff,
                st.mean_diff,
                st.shift,
                st.max_centered,
                st.kl
            ));
        }
        fs::write(path, t).unwrap_or_else(|e| die(format!("{path}: {e}")));
        println!("wrote {} positions to {path}", stats.len());
    }

    if let Some(out) = &a.out {
        let mut bytes = Vec::new();
        let width = kept.first().map_or(0, |k| k.1.len());
        for (_, l) in &kept {
            bytes.extend(l.iter().flat_map(|x| x.to_le_bytes()));
        }
        fs::write(out, bytes).unwrap_or_else(|e| die(format!("{out}: {e}")));
        let positions: Vec<String> = kept.iter().map(|k| k.0.to_string()).collect();
        let p0 = kept.first().map_or(0, |k| k.0);
        let contiguous = kept.iter().enumerate().all(|(i, k)| k.0 == p0 + i);
        fs::write(
            format!("{out}.meta"),
            format!(
                "n_tokens={n}\nn_vocab={width}\n{}rows={}\nmode={:?}\npositions={}\n",
                if contiguous {
                    format!("start_row={p0}\n")
                } else {
                    String::new()
                },
                kept.len(),
                a.mode,
                positions.join(",")
            ),
        )
        .unwrap_or_else(|e| die(format!("{out}.meta: {e}")));
        println!("wrote {} rows of {width} logits to {out}", kept.len());
    }
}
