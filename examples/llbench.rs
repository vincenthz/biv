//! TEMPORARY: llama-bench-like pp/tg measurement. `llbench <gguf> [--cpu] [-p 512] [-n 128] [-r 3] [-b 256]`
use std::time::Instant;

use biv::{engine::Engine, model::Model, sampler::argmax};

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let cpu = args.iter().any(|a| a == "--cpu");
    args.retain(|a| a != "--cpu");
    let path = args[0].clone();
    let get = |k: &str, d: usize| {
        args.iter()
            .position(|a| a == k)
            .map(|i| args[i + 1].parse().unwrap())
            .unwrap_or(d)
    };
    let (pp, tg, reps, bs) = (get("-p", 512), get("-n", 128), get("-r", 3), get("-b", 256));

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
    println!("{}", engine.description());
    let sm = engine.session_memory(8192, 4);
    println!(
        "session(8k ctx, 4 drafts): scratch {:.1} MB, kv {:.1} MB, recurrent {:.1} MB",
        sm.scratch as f64 / 1e6,
        sm.kv as f64 / 1e6,
        sm.recurrent as f64 / 1e6
    );

    // pseudo-random token ids, like llama-bench
    let mut x: u32 = 12345;
    let toks: Vec<u32> = (0..pp)
        .map(|_| {
            x = x.wrapping_mul(1103515245).wrapping_add(12345);
            (x >> 8) % 50000 + 1000
        })
        .collect();

    let run_pp = || {
        let mut s = engine.new_session(pp + 1, 0).unwrap();
        let t = Instant::now();
        let _ = s.forward(&toks);
        pp as f64 / t.elapsed().as_secs_f64()
    };
    let run_tg = || {
        let mut s = engine.new_session(tg + 2, 0).unwrap();
        let mut next = toks[0];
        let t = Instant::now();
        for _ in 0..tg {
            next = argmax(&s.forward(&[next]));
        }
        tg as f64 / t.elapsed().as_secs_f64()
    };
    let stats = |v: &[f64]| {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        let sd = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
        (m, sd)
    };
    run_pp();
    let v: Vec<f64> = (0..reps).map(|_| run_pp()).collect();
    let (m, sd) = stats(&v);
    println!("pp{pp}: {m:8.2} ± {sd:.2} tok/s");
    run_tg();
    let v: Vec<f64> = (0..reps).map(|_| run_tg()).collect();
    let (m, sd) = stats(&v);
    println!("tg{tg}: {m:8.2} ± {sd:.2} tok/s");
}
