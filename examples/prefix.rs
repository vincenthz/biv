//! Saved conversation prefixes: a chat restored from a prefix answers like the
//! chat that saved it, without processing the prefix again.
//! `prefix <gguf> [--cpu] [-n tokens]`
use std::time::Instant;

use biv::{
    chat::{Chat, ChatOptions, Event},
    engine::Engine,
    model::Model,
    sampler::{Sampler, SamplerConfig},
    tokenizer::Tokenizer,
    transcript::{TokenEncoding, Transcript},
};

const SYSTEM: &str = "You are a terse assistant for a team of embedded systems engineers. \
Answer in at most three sentences. Prefer concrete numbers, part names and register \
names to general advice. When a question is ambiguous, state the assumption you make \
instead of asking. Never use bullet lists or headings. If you don't know, say so in one \
sentence. Units are SI; temperatures are in Celsius. The team works mostly with ARM \
Cortex-M microcontrollers, RISC-V soft cores on FPGAs, and Linux on application \
processors, writing C and Rust.";

/// Greedy answer to `user`, and the time to process its prompt
fn ask(chat: &mut Chat<'_>, user: &str) -> (String, usize, f64) {
    let mut sampler = Sampler::new(SamplerConfig {
        temperature: 0.0,
        ..Default::default()
    });
    let mut text = String::new();
    let t = Instant::now();
    let mut prompt_time = 0.0;
    let stats = chat
        .send(
            user,
            &mut sampler,
            |ev| match ev {
                Event::PromptDone => prompt_time = t.elapsed().as_secs_f64(),
                Event::Answer(a) => text.push_str(a),
                _ => {}
            },
            || false,
        )
        .unwrap();
    (text, stats.prompt_tokens, prompt_time)
}

fn check(what: &str, a: &str, b: &str) {
    if a == b {
        println!("  {what}: same text");
    } else {
        println!("  {what}: DIFFERENT\n    {a:?}\n    {b:?}");
    }
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let cpu = args.iter().any(|a| a == "--cpu");
    args.retain(|a| a != "--cpu");
    let max_tokens = args
        .iter()
        .position(|a| a == "-n")
        .map(|i| args[i + 1].parse().unwrap())
        .unwrap_or(48);
    let path = args
        .first()
        .expect("usage: prefix <gguf> [--cpu] [-n tokens]");

    let g = biv_cli::open_gguf(path).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let model = Model::load(&g).unwrap();
    #[cfg(target_os = "macos")]
    let metal = (!cpu).then(|| biv::metal::MetalEngine::new(&model, 256).unwrap());
    #[cfg(target_os = "macos")]
    let engine: &dyn Engine = match &metal {
        Some(m) => m,
        None => &model,
    };
    #[cfg(not(target_os = "macos"))]
    let engine: &dyn Engine = {
        let _ = cpu;
        &model
    };
    println!("{}", engine.description());

    let (q1, q2) = (
        "How do I measure the stack high-water mark?",
        "And on the RISC-V cores?",
    );
    let opts = ChatOptions {
        system: Some(SYSTEM.into()),
        thinking: false,
        max_tokens,
        context: 4096,
        ..Default::default()
    };

    // the system prompt processed with the first message
    let mut plain = Chat::new(engine, &tok, opts.clone()).unwrap();
    let (a1, n, t) = ask(&mut plain, q1);
    println!("no prefix: first prompt {n} tokens in {t:.2}s");

    // the system prompt saved first
    let mut saver = Chat::new(engine, &tok, opts.clone()).unwrap();
    let t = Instant::now();
    let system = saver.save().unwrap();
    println!(
        "saved the system prompt: {} tokens, {} MB, in {:.2}s",
        system.n_past(),
        system.bytes() >> 20,
        t.elapsed().as_secs_f64()
    );
    let (b1, n, t) = ask(&mut saver, q1);
    println!("after save: first prompt {n} tokens in {t:.2}s");
    check("vs the system prompt processed with the message", &a1, &b1);
    let turn1 = saver.save().unwrap();
    let (b2, ..) = ask(&mut saver, q2);

    // restored in another chat, whose own system prompt is ignored
    let other = ChatOptions {
        system: None,
        ..opts.clone()
    };
    let mut restored = Chat::new(engine, &tok, other.clone()).unwrap();
    restored.restore(&system).unwrap();
    let (c1, n, t) = ask(&mut restored, q1);
    println!("restored: first prompt {n} tokens in {t:.2}s");
    check("restored vs the chat that saved it", &b1, &c1);

    // back to the end of the first turn, twice
    for i in 0..2 {
        restored.restore(&turn1).unwrap();
        let (c2, ..) = ask(&mut restored, q2);
        check(
            &format!("second turn from a saved first turn ({i})"),
            &b2,
            &c2,
        );
    }

    // the same conversations as token ids, processed again
    let mut replayer = Chat::new(engine, &tok, opts.clone()).unwrap();
    let empty = replayer.transcript();
    println!(
        "transcript before the first message: {} tokens",
        empty.len()
    );
    let (r1, ..) = ask(&mut replayer, q1);
    check("first answer after an empty transcript", &a1, &r1);
    let transcript = replayer.transcript();
    assert_eq!(transcript.len(), replayer.n_tokens());
    let (processed, rest) = transcript.split();
    assert_eq!(processed.len(), replayer.n_past());
    println!(
        "transcript of the first turn: {} tokens, {} pending, {} generated",
        transcript.len(),
        rest.len(),
        transcript.generated.len()
    );
    // through bytes, in both token encodings (the byte format does not
    // carry the conversation yet, so only the token fields round-trip)
    for encoding in [TokenEncoding::Full, TokenEncoding::Compact] {
        let bytes = transcript.to_bytes(encoding);
        println!("  {encoding:?}: {} bytes", bytes.len());
        let back = Transcript::from_bytes(&bytes).unwrap();
        assert_eq!(back.tokens, transcript.tokens);
        assert_eq!(back.pending, transcript.pending);
        assert_eq!(back.generated, transcript.generated);
    }
    let mut replayed = Chat::new(engine, &tok, other.clone()).unwrap();
    for i in 0..2 {
        let t = Instant::now();
        replayed.restore_transcript(&transcript).unwrap();
        println!(
            "restored the transcript in {:.2}s",
            t.elapsed().as_secs_f64()
        );
        assert_eq!(replayed.transcript(), transcript);
        let (r2, ..) = ask(&mut replayed, q2);
        check(&format!("second turn from the transcript ({i})"), &b2, &r2);
    }
    let mut bad = transcript.clone();
    bad.tokens.push(tok.n_vocab() as u32);
    match replayed.restore_transcript(&bad) {
        Err(e) => println!("  token id past the vocabulary: {e}"),
        Ok(()) => println!("  token id past the vocabulary: RESTORED"),
    }

    // drafting settings may differ, but a drafting session needs the MTP history
    let mut no_draft = Chat::new(
        engine,
        &tok,
        ChatOptions {
            draft: Some(0),
            ..opts.clone()
        },
    )
    .unwrap();
    no_draft.restore(&system).unwrap();
    let (d1, ..) = ask(&mut no_draft, q1);
    check("restored without drafts", &b1, &d1);
    let undrafted = Chat::new(
        engine,
        &tok,
        ChatOptions {
            draft: Some(0),
            ..opts.clone()
        },
    )
    .unwrap()
    .save()
    .unwrap();
    let mut drafting = Chat::new(engine, &tok, opts.clone()).unwrap();
    match drafting.restore(&undrafted) {
        Err(e) => println!("  drafting chat, prefix without MTP history: {e}"),
        Ok(()) => println!("  drafting chat, prefix without MTP history: RESTORED"),
    }
}
