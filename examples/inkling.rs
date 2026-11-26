//! Checks of the `inkling2_loop` CPU path from token ids, without the
//! tokenizer: greedy text (to compare with llama.cpp's llama-simple), and the
//! same logits whatever the passes (chunks, decode steps, verify + keep,
//! snapshot / restore).
//!
//! `inkling <gguf> greedy [ids,comma,separated] [n]`
//! `inkling <gguf> check [ids...] [n_gen]`: greedy ids, then the checks over them
//! `inkling <gguf> long [n_tokens]`: passes of 64 against 7, past the window
use std::time::Instant;

use biv::model::{Model, State};

/// GPT-2's bytes as printable characters, reversed
fn byte_decoder() -> std::collections::HashMap<char, u8> {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32)
        .chain(0xa1..=0xac)
        .chain(0xae..=0xff)
        .collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.iter()
        .zip(&cs)
        .map(|(&b, &c)| (char::from_u32(c).unwrap(), b as u8))
        .collect()
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32
}

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap();
    let mode = args.next().unwrap_or("greedy".into());
    let g = biv_cli::open_gguf(&path).unwrap();
    let vocab = g.metadata.tokenizer.ggml.tokens.clone().unwrap();
    let bd = byte_decoder();
    let decode = |ids: &[u32]| -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .flat_map(|&i| {
                let s = &vocab[i as usize];
                match s.chars().all(|c| bd.contains_key(&c)) {
                    true => s.chars().map(|c| bd[&c]).collect::<Vec<u8>>(),
                    false => s.as_bytes().to_vec(),
                }
            })
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    };
    let t0 = Instant::now();
    let model = Model::load(&g).unwrap();
    let c = &model.cfg;
    eprintln!(
        "{} layers, n_vocab {}, heads {}/{} x {}, window {}, loaded in {:?}",
        c.n_layer,
        c.n_vocab,
        c.n_head,
        c.n_head_kv,
        c.head_dim,
        c.sliding_window,
        t0.elapsed()
    );
    // "The capital of France is" by llama-tokenize
    let france = vec![371, 151451, 149879, 151194, 149884];
    let ids_arg = |s: Option<String>| -> Vec<u32> {
        match s {
            Some(s) if s.contains(',') || s.parse::<u32>().is_ok() => {
                s.split(',').map(|x| x.trim().parse().unwrap()).collect()
            }
            _ => france.clone(),
        }
    };
    let greedy = |prompt: &[u32], n_gen: usize| -> (Vec<u32>, Vec<Vec<f32>>) {
        let mut st = model.new_state(prompt.len() + n_gen + 1, 0).unwrap();
        let t = Instant::now();
        let mut logits = model.forward(&mut st, prompt);
        eprintln!("prefill {} tokens: {:?}", prompt.len(), t.elapsed());
        let t = Instant::now();
        let mut out = Vec::new();
        let mut all = vec![logits.clone()];
        for _ in 0..n_gen {
            let next = argmax(&logits);
            out.push(next);
            logits = model.forward(&mut st, &[next]);
            all.push(logits.clone());
        }
        eprintln!(
            "decode {n_gen} tokens: {:?} ({:.2} tok/s)",
            t.elapsed(),
            n_gen as f64 / t.elapsed().as_secs_f64()
        );
        (out, all)
    };

    match mode.as_str() {
        "greedy" => {
            let prompt = ids_arg(args.next());
            let n: usize = args.next().map_or(40, |s| s.parse().unwrap());
            let (out, all) = greedy(&prompt, n);
            let l = &all[0];
            let zmax = l.iter().fold(0.0f32, |m, z| m.max(z.abs()));
            eprintln!("logits {} values, max |z| {zmax}", l.len());
            let mut top: Vec<(usize, f32)> = l.iter().copied().enumerate().collect();
            top.sort_by(|a, b| b.1.total_cmp(&a.1));
            for (i, z) in &top[..5] {
                eprintln!("  {i:6} {z:8.3} {:?}", decode(&[*i as u32]));
            }
            println!("ids: {out:?}");
            println!("{}{}", decode(&prompt), decode(&out));
        }
        "check" => {
            let prompt = ids_arg(args.next());
            let n_gen: usize = args.next().map_or(24, |s| s.parse().unwrap());
            let (out, all) = greedy(&prompt, n_gen);
            let mut ids = prompt.clone();
            ids.extend(&out);
            println!("{}", decode(&ids));
            // the logits after each token, from a decode step each
            let n = ids.len();
            let reference = &all[all.len() - 1];
            let fresh = || model.new_state(n + 4, 2).unwrap();
            // one pass, and passes of 5 tokens
            for chunk in [n, 5, 3] {
                let mut st = fresh();
                let (body, last) = ids.split_at(n - 1);
                for c in body.chunks(chunk) {
                    model.prefill(&mut st, c);
                }
                let l = model.forward(&mut st, last);
                println!(
                    "passes of {chunk}: max |diff| vs decode {} (argmax {} / {})",
                    maxdiff(&l, reference),
                    argmax(&l),
                    argmax(reference)
                );
            }
            // verify + keep, accepting 1 or 2 of 3 tokens, against decode steps
            let mut st = fresh();
            let k = prompt.len();
            model.prefill(&mut st, &ids[..k]);
            let mut worst = 0.0f32;
            let mut p = k;
            let mut step = 0;
            while p + 3 <= n {
                let v = model.verify(&mut st, &ids[p..p + 3]);
                // the logits after ids[p + j] were all[p + j - k + 1]
                for (j, l) in v.iter().enumerate() {
                    worst = worst.max(maxdiff(l, &all[p + j - k + 1]));
                }
                let keep = 1 + step % 2;
                model.keep(&mut st, keep);
                p += keep;
                step += 1;
            }
            let l = model.forward(&mut st, &ids[p..]);
            worst = worst.max(maxdiff(&l, reference));
            println!("verify + keep: max |diff| vs decode {worst}");
            // snapshot halfway, run on, restore and finish
            let mut st = fresh();
            let half = n / 2;
            model.prefill(&mut st, &ids[..half]);
            let snap = st.snapshot().unwrap();
            model.prefill(&mut st, &ids[half..n - 1]);
            let mut other: State = fresh();
            other.restore(&snap).unwrap();
            model.prefill(&mut other, &ids[half..n - 1]);
            let a = model.forward(&mut st, &ids[n - 1..]);
            let b = model.forward(&mut other, &ids[n - 1..]);
            println!(
                "snapshot / restore: max |diff| {} vs straight, {} vs decode",
                maxdiff(&a, &b),
                maxdiff(&b, reference)
            );
            let m = st.memory();
            println!("session memory {m:?}, snapshot {} bytes", snap.bytes());
        }
        "long" => {
            let n: usize = args.next().map_or(1500, |s| s.parse().unwrap());
            // a text long enough: greedy ids of a prompt, repeated with variations
            let mut ids: Vec<u32> = Vec::new();
            let mut i = 0u32;
            while ids.len() < n {
                ids.extend(&france);
                ids.push(13 + (i % 50));
                i += 1;
            }
            ids.truncate(n);
            let run = |chunk: usize| -> Vec<f32> {
                let mut st = model.new_state(n, 0).unwrap();
                let t = Instant::now();
                let (body, last) = ids.split_at(n - 1);
                for c in body.chunks(chunk) {
                    model.prefill(&mut st, c);
                }
                let l = model.forward(&mut st, last);
                eprintln!("passes of {chunk}: {:?}", t.elapsed());
                l
            };
            let a = run(64);
            let b = run(7);
            println!(
                "{n} tokens, passes of 64 vs 7: max |diff| {}, argmax {} / {}",
                maxdiff(&a, &b),
                argmax(&a),
                argmax(&b)
            );
        }
        _ => panic!("mode: greedy, check or long"),
    }
}
