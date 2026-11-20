//! Final logits must not depend on how the prompt is split into forward passes
use biv::{model::Model, tokenizer::Tokenizer};

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let g = biv_cli::open_gguf(&path).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let model = Model::load(&g).unwrap();
    let text = "Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, type safety, and concurrency. It enforces memory safety, meaning that all references point to valid memory, without a garbage collector. To simultaneously enforce memory safety and prevent data races, its borrow checker tracks the object lifetime of all references in a program during compilation. Rust was influenced by ideas from functional programming, including immutability, higher-order functions, algebraic data types, and pattern matching. It also supports object-oriented programming via structs, enums, traits, and methods. Software developer Graydon Hoare created Rust";
    let ids = tok.encode(text);
    println!("{} tokens", ids.len());
    let run = |chunk: usize| {
        let mut st = model.new_state(ids.len(), 0).unwrap();
        let (body, last) = ids.split_at(ids.len() - 1);
        for c in body.chunks(chunk) {
            model.prefill(&mut st, c);
        }
        model.forward(&mut st, last)
    };
    let (ca, cb) = (64, 5);
    let a = run(ca);
    let b = run(cb);
    let top = |l: &[f32]| {
        let mut v: Vec<_> = l.iter().copied().enumerate().collect();
        v.sort_by(|x, y| y.1.total_cmp(&x.1));
        v.truncate(3);
        v
    };
    let maxdiff = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    println!(
        "chunk {ca}: {:?}\nchunk {cb}: {:?}\nmax |diff| = {maxdiff}",
        top(&a),
        top(&b)
    );
    println!("next: {:?}", tok.decode(&[top(&a)[0].0 as u32]));
}
