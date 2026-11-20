//! Greedy generation from a raw prompt, for comparison with llama.cpp's llama-simple
use std::io::Write;
use std::time::Instant;

use biv::{model::Model, tokenizer::Tokenizer};

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap();
    let prompt = args.next().unwrap_or("The capital of France is".into());
    let n_gen: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(16);
    let g = biv_cli::open_gguf(&path).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let model = Model::load(&g).unwrap();
    let ids = tok.encode_with_special(&prompt);
    let mut st = model.new_state(ids.len() + n_gen, 0).unwrap();
    eprintln!("prompt tokens: {ids:?}");
    let t0 = Instant::now();
    let mut logits = model.forward(&mut st, &ids);
    eprintln!("prefill {} tokens: {:?}", ids.len(), t0.elapsed());
    let mut top: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    top.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (i, l) in &top[..5] {
        eprintln!("  {i:6} {l:8.3} {:?}", tok.decode(&[*i as u32]));
    }
    print!("{prompt}");
    let t1 = Instant::now();
    for _ in 0..n_gen {
        let next = argmax(&logits);
        print!("{}", tok.decode(&[next]));
        std::io::stdout().flush().unwrap();
        if tok.eog.contains(&next) {
            break;
        }
        logits = model.forward(&mut st, &[next]);
    }
    println!();
    eprintln!(
        "decode {n_gen} tokens: {:?} ({:.2} tok/s)",
        t1.elapsed(),
        n_gen as f64 / t1.elapsed().as_secs_f64()
    );
}
