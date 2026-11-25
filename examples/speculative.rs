//! Speculative decoding with the MTP head: greedy answers with and without
//! drafts must be the same text. `speculative <gguf> [--cpu] [-n tokens] [-d draft]`
use std::time::Instant;

use biv::{
    chat::{Chat, ChatOptions, Event},
    engine::Engine,
    model::Model,
    sampler::{Sampler, SamplerConfig},
    tokenizer::Tokenizer,
};

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let cpu = args.iter().any(|a| a == "--cpu");
    args.retain(|a| a != "--cpu");
    let flag = |name: &str, default: usize| {
        args.iter()
            .position(|a| a == name)
            .map(|i| args[i + 1].parse().unwrap())
            .unwrap_or(default)
    };
    let (max_tokens, draft) = (flag("-n", 256), flag("-d", 1));
    let path = args
        .first()
        .expect("usage: speculative <gguf> [--cpu] [-n tokens] [-d draft]");

    let g = biv_cli::open_gguf(path).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let model = Model::load(&g).unwrap();
    assert!(model.has_mtp(), "the model has no MTP head");
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

    let prompts = [
        "Explain in detail how a CPU cache hierarchy works and why it matters for performance.",
        "Write a Rust function that parses an IPv4 address from a string, with tests.",
    ];
    for prompt in prompts {
        let mut texts = Vec::new();
        for d in [0, draft] {
            let opts = ChatOptions {
                thinking: false,
                max_tokens,
                context: 4096,
                draft: Some(d),
                ..Default::default()
            };
            let mut chat = Chat::new(engine, &tok, opts).unwrap();
            let mut sampler = Sampler::new(SamplerConfig {
                temperature: 0.0,
                ..Default::default()
            });
            // one turn first, so that the timed one continues a conversation
            chat.send("Hello", &mut sampler, |_| {}, || false).unwrap();
            let mut text = String::new();
            let t = Instant::now();
            let mut gen_start = t;
            let s = chat
                .send(
                    prompt,
                    &mut sampler,
                    |ev| match ev {
                        Event::PromptDone => gen_start = Instant::now(),
                        Event::Answer(a) => text.push_str(a),
                        _ => {}
                    },
                    || false,
                )
                .unwrap();
            let rate = s.gen_tokens as f64 / gen_start.elapsed().as_secs_f64();
            print!(
                "draft {d}: {} tokens, {rate:5.2} tok/s ({:.1}s)",
                s.gen_tokens,
                t.elapsed().as_secs_f64()
            );
            if s.drafted > 0 {
                print!(
                    ", drafts {}/{} right ({:.0}%)",
                    s.accepted,
                    s.drafted,
                    100.0 * s.accepted as f64 / s.drafted as f64
                );
            }
            println!();
            texts.push(text);
        }
        match texts[0]
            .chars()
            .zip(texts[1].chars())
            .position(|(a, b)| a != b)
        {
            None if texts[0].len() == texts[1].len() => println!("  same text"),
            at => {
                let at = at.unwrap_or(texts[0].len().min(texts[1].len()));
                let ctx = |t: &str| {
                    t.chars()
                        .skip(at.saturating_sub(40))
                        .take(80)
                        .collect::<String>()
                };
                println!(
                    "  DIFFERENT at char {at}:\n  plain: {:?}\n  draft: {:?}",
                    ctx(&texts[0]),
                    ctx(&texts[1])
                );
            }
        }
    }
}
