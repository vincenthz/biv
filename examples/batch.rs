//! Continuous batching: chats on threads of their own sharing one engine.
//! `batch <gguf> [--cpu] [-n tokens] [-d draft] [-j chats...]`
//!
//! First, sequences prefilled with different prompts then run in one pass
//! have the same top-1 token as each alone, and logits close to its.
//! Then 8 prompts are answered greedily by `j` chats at a time (each chat
//! takes the next prompt when it is done), for each `j`: the tokens per
//! second of all the chats together, and how many answers are the text of
//! the same chat alone (they may part by rounding: the matmul kernels
//! depend on the rows of the whole pass).
use std::sync::Mutex;
use std::time::Instant;

use biv::{
    batch::{self, BatchOptions},
    chat::{Chat, ChatOptions, Event},
    engine::{self, Engine, Pass, Work},
    model::Model,
    sampler::{Sampler, SamplerConfig, argmax},
    tokenizer::Tokenizer,
};

const PROMPTS: [&str; 8] = [
    "Explain in detail how a CPU cache hierarchy works and why it matters for performance.",
    "Write a Rust function that parses an IPv4 address from a string, with tests.",
    "Summarize the causes of the French Revolution in a few paragraphs.",
    "What are the trade-offs between B-trees and LSM trees for a storage engine?",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "How does public key cryptography work? Explain it to a curious teenager.",
    "Give a recipe for a vegetarian lasagna, with the steps in order.",
    "Compare TCP and QUIC: connection setup, head-of-line blocking, congestion control.",
];

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let cpu = args.iter().any(|a| a == "--cpu");
    args.retain(|a| a != "--cpu");
    let flag = |name: &str| args.iter().position(|a| a == name).map(|i| &args[i + 1]);
    let max_tokens = flag("-n").map_or(128, |v| v.parse().unwrap());
    let draft: Option<usize> = flag("-d").map(|v| v.parse().unwrap());
    let jobs: Vec<usize> = match args.iter().position(|a| a == "-j") {
        Some(i) => args[i + 1..].iter().map_while(|a| a.parse().ok()).collect(),
        None => vec![1, 2, 4, 8],
    };
    let path = args
        .first()
        .expect("usage: batch <gguf> [--cpu] [-n tokens] [-d draft] [-j chats...]");

    let g = biv_cli::open_gguf(path).unwrap();
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

    one_pass(engine, &tok);

    let opts = ChatOptions {
        thinking: false,
        max_tokens,
        context: 4096,
        draft,
        ..Default::default()
    };
    // each prompt alone, on the engine itself
    let t = Instant::now();
    let alone: Vec<(String, usize)> = PROMPTS
        .iter()
        .map(|p| answer(engine, &tok, &opts, p))
        .collect();
    let tokens: usize = alone.iter().map(|a| a.1).sum();
    let base = tokens as f64 / t.elapsed().as_secs_f64();
    println!("alone: {tokens} tokens, {base:.1} tok/s");

    prefix(engine, &tok, &opts);

    for &j in &jobs {
        let next = Mutex::new(0);
        let answers = Mutex::new(vec![(String::new(), 0); PROMPTS.len()]);
        let t = Instant::now();
        batch::serve(engine, &BatchOptions::default(), |h| {
            std::thread::scope(|s| {
                for _ in 0..j {
                    s.spawn(|| {
                        loop {
                            let i = {
                                let mut n = next.lock().unwrap();
                                *n += 1;
                                *n - 1
                            };
                            let Some(p) = PROMPTS.get(i) else { break };
                            let a = answer(&h, &tok, &opts, p);
                            answers.lock().unwrap()[i] = a;
                        }
                    });
                }
            })
        })
        .unwrap();
        let answers = answers.into_inner().unwrap();
        let tokens: usize = answers.iter().map(|a| a.1).sum();
        let rate = tokens as f64 / t.elapsed().as_secs_f64();
        let same = answers
            .iter()
            .zip(&alone)
            .filter(|(a, b)| a.0 == b.0)
            .count();
        println!(
            "{j} chats: {tokens} tokens, {rate:.1} tok/s ({:.2}x), {same}/{} answers as alone",
            rate / base,
            PROMPTS.len()
        );
        for (i, (a, b)) in answers.iter().zip(&alone).enumerate() {
            if a.0 != b.0 {
                let at =
                    a.0.chars()
                        .zip(b.0.chars())
                        .take_while(|(x, y)| x == y)
                        .count();
                println!("  prompt {i} parts at char {at} of {}", b.0.chars().count());
            }
        }
    }
}

/// The greedy answer to `prompt` by a new chat on `engine`, and its tokens
fn answer(
    engine: &dyn Engine,
    tok: &Tokenizer,
    opts: &ChatOptions,
    prompt: &str,
) -> (String, usize) {
    reply(&mut Chat::new(engine, tok, opts.clone()).unwrap(), prompt)
}

/// The greedy answer of `chat` to `prompt`, and its tokens
fn reply(chat: &mut Chat<'_>, prompt: &str) -> (String, usize) {
    let mut sampler = Sampler::new(SamplerConfig {
        temperature: 0.0,
        ..Default::default()
    });
    let mut text = String::new();
    let s = chat
        .send(
            prompt,
            &mut sampler,
            |ev| {
                if let Event::Answer(a) = ev {
                    text.push_str(a)
                }
            },
            || false,
        )
        .unwrap();
    (text, s.gen_tokens)
}

/// A system prompt saved by one chat of the scheduler and restored in
/// another, one at a time: the answer is the same as on the engine itself
fn prefix(engine: &dyn Engine, tok: &Tokenizer, opts: &ChatOptions) {
    let opts = ChatOptions {
        system: Some("You are a terse assistant. Answer in at most three sentences.".into()),
        ..opts.clone()
    };
    let prompt = PROMPTS[3];
    let mut saver = Chat::new(engine, tok, opts.clone()).unwrap();
    saver.save().unwrap();
    let expected = reply(&mut saver, prompt);
    let restored = batch::serve(engine, &BatchOptions::default(), |h| {
        let mut saver = Chat::new(&h, tok, opts.clone()).unwrap();
        let prefix = saver.save().unwrap();
        drop(saver);
        let mut chat = Chat::new(&h, tok, opts.clone()).unwrap();
        chat.restore(&prefix).unwrap();
        reply(&mut chat, prompt)
    })
    .unwrap();
    assert_eq!(
        restored, expected,
        "answer after a prefix restored in a batch"
    );
    println!("prefix saved and restored through the scheduler: same answer");
}

/// Prompts of different lengths prefilled one by one, then their last
/// tokens in one pass, against the same prompts in sessions of their own
fn one_pass(engine: &dyn Engine, tok: &Tokenizer) {
    let prompts: Vec<Vec<u32>> = PROMPTS[..4].iter().map(|p| tok.encode(p)).collect();
    let mut b = engine::new_batch(engine, 256, 16).unwrap();
    let seqs: Vec<usize> = prompts.iter().map(|_| b.add(1024, 0).unwrap()).collect();
    for (&seq, p) in seqs.iter().zip(&prompts) {
        let body = &p[..p.len() - 1];
        b.run(&[Work {
            seq,
            tokens: body,
            pass: Pass::Prefill,
        }]);
    }
    let work: Vec<Work<'_>> = seqs
        .iter()
        .zip(&prompts)
        .map(|(&seq, p)| Work {
            seq,
            tokens: &p[p.len() - 1..],
            pass: Pass::Forward,
        })
        .collect();
    let batched = b.run(&work);
    let mut worst = 0f32;
    for (p, logits) in prompts.iter().zip(&batched) {
        let mut s = engine.new_session(1024, 0).unwrap();
        let alone = s.forward(p);
        let diff = alone
            .iter()
            .zip(&logits[0])
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert_eq!(argmax(&alone), argmax(&logits[0]), "top-1 token in a batch");
        worst = worst.max(diff);
    }
    println!(
        "one pass of {} sequences: same top-1 token as alone, logits within {worst:.3}",
        prompts.len()
    );
}
