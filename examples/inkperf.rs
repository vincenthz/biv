//! Prefill and decode rates at a context: `inkperf <gguf> [--cpu] [-c N]
//! [-n M] [-b B] [-r R] [-f tokens.u32]`: prefills N tokens (pseudo-random
//! ids, or the first N of a token file) in one `forward`, then decodes M
//! greedy tokens from there, R times; prints the best and median rates.
use std::time::Instant;

use biv::{engine::Engine, model::Model, sampler::argmax};

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let cpu = args.iter().any(|a| a == "--cpu");
    args.retain(|a| a != "--cpu");
    let path = args[0].clone();
    let get = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .map(|i| args[i + 1].clone())
    };
    let num = |k: &str, d: usize| get(k).map_or(d, |v| v.parse().unwrap());
    let (ctx, tg, bs, reps) = (num("-c", 1852), num("-n", 64), num("-b", 256), num("-r", 3));

    let g = biv_cli::open_gguf(&path).unwrap();
    let model = Model::load(&g).unwrap();
    let gpu = if cpu {
        None
    } else {
        biv_cli::gpu_engine(&model, bs).unwrap()
    };
    let engine: &dyn Engine = match &gpu {
        Some(g) => g.as_ref(),
        None => &model,
    };
    let toks: Vec<u32> = match get("-f") {
        Some(f) => std::fs::read(f)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .take(ctx)
            .collect(),
        None => {
            let mut x: u32 = 12345;
            (0..ctx)
                .map(|_| {
                    x = x.wrapping_mul(1103515245).wrapping_add(12345);
                    (x >> 8) % 50000 + 1000
                })
                .collect()
        }
    };
    assert_eq!(toks.len(), ctx, "not enough tokens");
    let (mut pp, mut dec) = (Vec::new(), Vec::new());
    for _ in 0..reps {
        let mut s = engine.new_session(ctx + tg + 1, 0).unwrap();
        let t = Instant::now();
        let mut next = argmax(&s.forward(&toks));
        pp.push(ctx as f64 / t.elapsed().as_secs_f64());
        let t = Instant::now();
        for _ in 0..tg {
            next = argmax(&s.forward(&[next]));
        }
        dec.push(tg as f64 / t.elapsed().as_secs_f64());
    }
    let show = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        format!("best {:7.2} median {:7.2}", v[v.len() - 1], v[v.len() / 2])
    };
    println!(
        "{}: ctx {ctx}: prefill {} tok/s, decode {tg}: {} tok/s",
        engine.description(),
        show(&mut pp),
        show(&mut dec)
    );
}
