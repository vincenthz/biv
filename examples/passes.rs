//! The logits of an engine must not depend on how the tokens are split into
//! passes, beyond rounding: passes of several sizes (prefill kernels, verify
//! kernels, decode steps) against one prompt, verify + keep against decode
//! steps, a snapshot restored into another session, and two sequences in
//! one batch against each alone. On the GPU engine (Metal on macOS, CUDA with
//! the `cuda` feature), the CPU with `--cpu`.
//!
//! `passes <gguf> [--cpu] [n_tokens] [batch]`
use std::time::Instant;

use biv::{
    engine::{Engine, Pass, Work},
    model::Model,
    sampler::argmax,
    tokenizer::Tokenizer,
};

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn line(what: &str, a: &[f32], reference: &[f32]) {
    println!(
        "{what}: max |diff| {:.4}, argmax {} / {}",
        maxdiff(a, reference),
        argmax(a),
        argmax(reference)
    );
}

const TEXT: &str = "Rust is a multi-paradigm, general-purpose programming language that \
    emphasizes performance, type safety, and concurrency. It enforces memory safety, meaning \
    that all references point to valid memory, without a garbage collector. To simultaneously \
    enforce memory safety and prevent data races, its borrow checker tracks the object lifetime \
    of all references in a program during compilation. Rust was influenced by ideas from \
    functional programming, including immutability, higher-order functions, algebraic data \
    types, and pattern matching. ";

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let cpu = args.iter().any(|a| a == "--cpu");
    args.retain(|a| a != "--cpu");
    let path = args
        .first()
        .expect("usage: passes <gguf> [--cpu] [n_tokens] [batch]");
    let n: usize = args.get(1).map_or(1500, |s| s.parse().unwrap());
    let batch: usize = args.get(2).map_or(256, |s| s.parse().unwrap());
    let g = biv_cli::open_gguf(path).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let model = Model::load(&g).unwrap();
    let gpu = if cpu {
        None
    } else {
        biv_cli::gpu_engine(&model, batch).unwrap()
    };
    let engine: &dyn Engine = match &gpu {
        Some(g) => g.as_ref(),
        None => &model,
    };
    println!("{}, batch {}", engine.description(), engine.batch_size());

    // the text numbered, so that it doesn't repeat exactly
    let mut ids = Vec::new();
    let mut i = 0;
    while ids.len() < n {
        ids.extend(tok.encode(&format!("{i}. {TEXT}")));
        i += 1;
    }
    ids.truncate(n);
    let context = n + 8;
    let (body, last) = ids.split_at(n - 1);

    // prefill in passes of `chunk`, then a forward of the last token
    let run = |chunk: usize| -> Vec<f32> {
        let mut s = engine.new_session(context, 0).unwrap();
        let t = Instant::now();
        for c in body.chunks(chunk) {
            s.prefill(c);
        }
        let l = s.forward(last);
        eprintln!("passes of {chunk}: {:?}", t.elapsed());
        l
    };
    let reference = run(engine.batch_size());
    for chunk in [64, 31, 7, 1] {
        line(&format!("passes of {chunk}"), &run(chunk), &reference);
    }

    // decode steps from the middle, against verify + keep of 3 tokens with
    // 1 or 2 kept, and against two sequences in one batch
    let mid = n / 2;
    let steps = 48.min(n - mid - 1);
    let mut s = engine.new_session(context, 2).unwrap();
    s.prefill(&ids[..mid]);
    let decode: Vec<Vec<f32>> = (0..steps)
        .map(|j| s.forward(&ids[mid + j..mid + j + 1]))
        .collect();
    let mut v = engine.new_session(context, 2).unwrap();
    v.prefill(&ids[..mid]);
    let (mut p, mut step, mut worst, mut flips) = (mid, 0, 0.0f32, 0);
    while p + 3 <= mid + steps {
        let ls = v.verify(&ids[p..p + 3]);
        for (j, l) in ls.iter().enumerate() {
            if let Some(d) = decode.get(p - mid + j) {
                worst = worst.max(maxdiff(l, d));
                flips += (argmax(l) != argmax(d)) as usize;
            }
        }
        let keep = 1 + step % 2;
        v.keep(keep);
        p += keep;
        step += 1;
    }
    println!(
        "verify + keep ({step} verifies): max |diff| vs decode {worst:.4}, {flips} argmax flips"
    );

    // a snapshot after the prompt's first half, restored in another session
    let mut a = engine.new_session(context, 0).unwrap();
    a.prefill(&ids[..mid]);
    let snap = a.snapshot().unwrap();
    a.prefill(&ids[mid..n - 1]);
    let la = a.forward(last);
    let mut b = engine.new_session(context, 0).unwrap();
    b.prefill(&ids[..7]);
    b.restore(&snap).unwrap();
    b.prefill(&ids[mid..n - 1]);
    let lb = b.forward(last);
    line("snapshot / restore vs straight", &lb, &la);
    println!(
        "  snapshot of {} tokens: {} bytes",
        snap.n_past(),
        snap.bytes()
    );

    // two sequences in one batch: the prompt, and its second half alone,
    // prefilled together, then decoded together
    if let Some(mut bt) = engine.new_batch(engine.batch_size(), 2).unwrap() {
        let x = bt.add(context, 2).unwrap();
        let y = bt.add(context, 2).unwrap();
        let other = &ids[mid..];
        let half = bt.max_tokens() / 2;
        let (mut px, mut py) = (0, 0);
        while px < mid || py < other.len() - 1 {
            let wx = &ids[px..mid.min(px + half)];
            let wy = &other[py..(other.len() - 1).min(py + half)];
            let mut work = Vec::new();
            if !wx.is_empty() {
                work.push(Work {
                    seq: x,
                    tokens: wx,
                    pass: Pass::Prefill,
                });
            }
            if !wy.is_empty() {
                work.push(Work {
                    seq: y,
                    tokens: wy,
                    pass: Pass::Prefill,
                });
            }
            bt.run(&work);
            px += wx.len();
            py += wy.len();
        }
        let mut alone = engine.new_session(context, 0).unwrap();
        alone.prefill(&other[..other.len() - 1]);
        let ly_alone = alone.forward(&other[other.len() - 1..]);
        let (mut wx, mut wy) = (0.0f32, 0.0f32);
        let mut ly = Vec::new();
        for j in 0..steps {
            let tx = &ids[mid + j..mid + j + 1];
            let mut work = vec![Work {
                seq: x,
                tokens: tx,
                pass: Pass::Forward,
            }];
            if j == 0 {
                work.push(Work {
                    seq: y,
                    tokens: &other[other.len() - 1..],
                    pass: Pass::Forward,
                });
            }
            let mut out = bt.run(&work);
            wx = wx.max(maxdiff(&out[0][0], &decode[j]));
            if j == 0 {
                ly = out.pop().unwrap().pop().unwrap();
                wy = maxdiff(&ly, &ly_alone);
            }
        }
        println!(
            "batch of 2: max |diff| vs alone {wx:.4} (decode steps), {wy:.4} (the other's last, argmax {} / {})",
            argmax(&ly),
            argmax(&ly_alone)
        );
    }
}
