//! Logits of fixed token ids, for bit-identity checks between builds:
//! `dumplogits <gguf> <cpu|metal> <out.f32> [n_tokens]`. The ids are a
//! deterministic pseudo-random sequence of the vocabulary; they go through
//! passes of 1, 16, 64 and 3 tokens in turn, then a verify of 5 and a keep
//! of 3, and the logits of every pass are written one after the other.
use std::io::Write;

use biv::engine::Engine;
use biv::model::Model;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [path, backend, out] = &args[..3] else {
        panic!("usage: dumplogits <gguf> <cpu|metal> <out.f32> [n_tokens]");
    };
    let n: usize = args.get(3).map_or(600, |s| s.parse().unwrap());
    let g = biv_cli::open_gguf(path).unwrap();
    let model = Model::load(&g).unwrap();
    let vocab = model.cfg.n_vocab as u64;
    let mut seed = 0x9e3779b97f4a7c15u64;
    let ids: Vec<u32> = (0..n + 16)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % vocab.min(150_000)) as u32
        })
        .collect();
    let engine: Box<dyn Engine + '_> = match backend.as_str() {
        "cpu" => Box::new(&model),
        #[cfg(target_os = "macos")]
        "metal" => Box::new(biv::metal::MetalEngine::new(&model, 256).unwrap()),
        b => panic!("unknown backend {b}"),
    };
    let mut s = engine.new_session(n + 16, 4).unwrap();
    let mut f = std::io::BufWriter::new(std::fs::File::create(out).unwrap());
    let mut put = |l: &[f32]| {
        for v in l {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    };
    let mut p = 0;
    for &len in [1usize, 16, 64, 3].iter().cycle() {
        if p + len > n {
            break;
        }
        put(&s.forward(&ids[p..p + len]));
        p += len;
    }
    let l = s.verify(&ids[p..p + 5]);
    for l in &l {
        put(l);
    }
    s.keep(3);
    put(&s.forward(&ids[p + 3..p + 4]));
    println!(
        "{backend}: {} tokens, {} logits per row",
        s.n_past(),
        l[0].len()
    );
}
