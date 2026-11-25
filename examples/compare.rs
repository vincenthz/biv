//! GPU vs CPU logits on the same sequence: prefill (matmul kernels), then
//! decode steps (matvec kernels). Metal on macOS, CUDA with the `cuda` feature.
//!
//! `compare <gguf> [repeat] [context]`: the prompt repeated `repeat` times
#[cfg(any(target_os = "macos", feature = "cuda"))]
mod imp {
    use std::time::Instant;

    use biv::{engine::Engine, model::Model, sampler::argmax, tokenizer::Tokenizer};

    fn top(l: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut v: Vec<_> = l.iter().copied().enumerate().collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v.truncate(k);
        v
    }

    fn report(what: &str, a: &[f32], b: &[f32]) {
        let maxdiff = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let (ta, tb) = (top(a, 5), top(b, 5));
        let same = ta.iter().map(|x| x.0).eq(tb.iter().map(|x| x.0));
        println!(
            "{what}: max |diff| {maxdiff:.4}, top-5 {}",
            if same { "identical" } else { "DIFFERENT" }
        );
        if !same {
            println!("  cpu {ta:?}\n  gpu {tb:?}");
        }
    }

    pub fn main() {
        let mut args = std::env::args().skip(1);
        let path = args.next().unwrap();
        let repeat: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(1);
        let context: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(8192);
        let g = biv_cli::open_gguf(&path).unwrap();
        let tok = Tokenizer::from_gguf(&g).unwrap();
        let model = Model::load(&g).unwrap();
        let t = Instant::now();
        let engine = biv_cli::gpu_engine(&model, 256).unwrap().unwrap();
        println!(
            "gpu engine ready in {:?}: {}",
            t.elapsed(),
            engine.description()
        );

        let ids = tok.encode(&"Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, type safety, and concurrency. It enforces memory safety, meaning that all references point to valid memory, without a garbage collector. Software developer Graydon Hoare created Rust. ".repeat(repeat));
        let ids = &ids[..ids.len() - 1]; // end mid-sentence: " Rust"
        let ids = ids.to_vec();
        let mut cpu = (&model).new_session(context, 0).unwrap();
        let mut gpu = engine.new_session(context, 0).unwrap();

        let t = Instant::now();
        let lc = cpu.forward(&ids);
        let tc = t.elapsed();
        let t = Instant::now();
        let lg = gpu.forward(&ids);
        let tg = t.elapsed();
        println!("prefill {} tokens: cpu {tc:?}, gpu {tg:?}", ids.len());
        report("prefill", &lc, &lg);

        // decode a few greedy tokens with both, feeding the cpu choice to both
        let mut next = argmax(&lc);
        let mut text = vec![next];
        for step in 0..8 {
            let lc = cpu.forward(&[next]);
            let lg = gpu.forward(&[next]);
            report(&format!("decode {step}"), &lc, &lg);
            next = argmax(&lc);
            text.push(next);
        }
        println!("cpu text: {:?}", tok.decode(&text));

        // gpu decode speed
        let n = 32;
        let t = Instant::now();
        for _ in 0..n {
            let l = gpu.forward(&[next]);
            next = argmax(&l);
        }
        println!(
            "gpu decode: {:.1} tok/s",
            n as f64 / t.elapsed().as_secs_f64()
        );
    }
}

fn main() {
    #[cfg(any(target_os = "macos", feature = "cuda"))]
    imp::main();
    #[cfg(not(any(target_os = "macos", feature = "cuda")))]
    eprintln!("compare needs a GPU backend: Metal (macOS) or the cuda feature");
}
