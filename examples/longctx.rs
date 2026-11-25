//! Decode speed as the context grows: fill the context up to each depth, then
//! time a few decode steps there. `longctx <gguf> [depth...]`, Metal unless `--cpu`.
use std::time::Instant;

use biv::{engine::Engine, model::Model, sampler::argmax, tokenizer::Tokenizer};

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let cpu = args.iter().any(|a| a == "--cpu");
    args.retain(|a| a != "--cpu");
    let path = args
        .first()
        .expect("usage: longctx <gguf> [depth...]")
        .clone();
    let mut depths: Vec<usize> = args[1..].iter().map(|s| s.parse().unwrap()).collect();
    if depths.is_empty() {
        depths = vec![0, 1024, 4096, 8192, 16384];
    }
    depths.sort();

    let g = biv_cli::open_gguf(&path).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let model = Model::load(&g).unwrap();
    let gpu = if cpu {
        None
    } else {
        biv_cli::gpu_engine(&model, 256).unwrap()
    };
    let engine: &dyn Engine = match &gpu {
        Some(g) => g.as_ref(),
        None => &model,
    };
    println!("{}", engine.description());

    let decode_steps = 16;
    let max = depths.last().unwrap() + decode_steps * depths.len() + 1;
    let mut s = engine.new_session(max, 0).unwrap();

    // filler text, so attention isn't over a degenerate sequence
    let text = "The history of computing is a story of abstraction layered upon abstraction: switches became gates, gates became circuits, circuits became processors, and processors became the substrate on which languages, operating systems and networks were built. ";
    let filler = tok.encode(text);
    let mut next = filler[0];
    for &d in &depths {
        let fill = d.saturating_sub(s.n_past());
        if fill > 0 {
            let toks: Vec<_> = filler.iter().copied().cycle().take(fill).collect();
            let t = Instant::now();
            next = argmax(&s.forward(&toks));
            let e = t.elapsed().as_secs_f64();
            print!(
                "prefill {fill:6} -> {:6}: {:7.1} tok/s | ",
                s.n_past(),
                fill as f64 / e
            );
        } else {
            print!("{:42}", "");
        }
        let t = Instant::now();
        for _ in 0..decode_steps {
            next = argmax(&s.forward(&[next]));
        }
        println!(
            "decode at {:6}: {:5.1} tok/s",
            d,
            decode_steps as f64 / t.elapsed().as_secs_f64()
        );
    }
}
