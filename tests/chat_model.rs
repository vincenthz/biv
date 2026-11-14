//! Chat tests on a real model, ignored without one:
//! `BIV_TEST_MODEL=<gguf> cargo test --release --test chat_model -- --ignored`
//! On macOS they run on Metal, elsewhere on the CPU.

use biv::chat::{Chat, ChatOptions, Event, Stop, TurnStats};
use biv::engine::Engine;
use biv::model::Model;
use biv::sampler::{Sampler, SamplerConfig, SamplerOverrides};
use biv::tokenizer::Tokenizer;

/// The variable that names the GGUF file
const MODEL_ENV: &str = "BIV_TEST_MODEL";
const CONTEXT: usize = 2048;
/// per answer, so that a test takes seconds rather than minutes
const MAX_TOKENS: usize = 48;
/// Metal prompt tokens per forward pass, as biv-cli uses
const METAL_BATCH: usize = 256;
const SEED: u64 = 0x5EED;
const FIRST: &str = "Name three prime numbers above 100.";
const SECOND: &str = "Which of them is the largest?";
/// A message that the model answers in a few tokens
const SHORT: &str = "Reply with the single word yes.";
/// The answer tokens of a turn that stops at its max_tokens
const FEW_TOKENS: usize = 4;
/// The context room above the prompt of a turn that stops at the context
const ROOM: usize = 4;

fn model_path() -> String {
    std::env::var(MODEL_ENV).unwrap_or_else(|_| panic!("{MODEL_ENV} must name a GGUF file"))
}

#[cfg(target_os = "macos")]
fn engine<'m>(model: &'m Model<'m>) -> Box<dyn Engine + 'm> {
    Box::new(biv::metal::MetalEngine::new(model, METAL_BATCH).expect("metal engine"))
}

#[cfg(not(target_os = "macos"))]
fn engine<'m>(model: &'m Model<'m>) -> Box<dyn Engine + 'm> {
    Box::new(model)
}

fn options() -> ChatOptions {
    ChatOptions {
        context: CONTEXT,
        max_tokens: MAX_TOKENS,
        ..ChatOptions::default()
    }
}

/// One turn; the token events must give the tokens of the stats, in order
fn send(chat: &mut Chat<'_>, text: &str, sampler: &mut Sampler) -> TurnStats {
    let mut streamed = Vec::new();
    let stats = chat
        .send(
            text,
            sampler,
            |ev| {
                if let Event::Token(t) = ev {
                    streamed.push(t);
                }
            },
            || false,
        )
        .expect("turn");
    assert_eq!(streamed, stats.tokens);
    stats
}

#[test]
#[ignore = "needs a model file"]
fn chat_turns_on_model() {
    let gguf = biv_cli::open_gguf(model_path()).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let model = Model::load(&gguf).expect("model");
    let engine = engine(&model);
    // the publisher's values: sampled, not greedy, so the seed matters
    let none = SamplerOverrides {
        temperature: None,
        top_k: None,
        top_p: None,
        min_p: None,
        presence_penalty: None,
        repeat_penalty: None,
    };
    let cfg = SamplerConfig::from_gguf(&gguf, none, SEED).expect("sampler config");

    // the tokens of each answer, and the chat's history of them
    let mut chat = Chat::new(engine.as_ref(), &tok, options()).expect("chat");
    let mut sampler = Sampler::new(cfg.clone());
    let first = send(&mut chat, FIRST, &mut sampler);
    assert_eq!(first.tokens.len(), first.gen_tokens);
    assert_eq!(chat.generated(), &first.tokens[..]);
    let prefix = chat.save().expect("save");
    let rng_state = sampler.rng_state();
    let second = send(&mut chat, SECOND, &mut sampler);
    assert_eq!(second.tokens.len(), second.gen_tokens);
    let both: Vec<_> = first.tokens.iter().chain(&second.tokens).copied().collect();
    assert_eq!(chat.generated(), &both[..]);

    // another chat, from the saved conversation and the sampler state,
    // gives the same second answer
    let mut other = Chat::new(engine.as_ref(), &tok, options()).expect("chat");
    other.restore(&prefix).expect("restore");
    assert_eq!(other.generated(), &first.tokens[..]);
    let mut resumed = Sampler::resume(cfg, rng_state).expect("resume");
    let again = send(&mut other, SECOND, &mut resumed);
    assert_eq!(again.tokens, second.tokens);
    assert_eq!(other.generated(), &both[..]);
    assert_eq!(resumed.rng_state(), sampler.rng_state());
}

#[test]
#[ignore = "needs a model file"]
fn each_stop_reason_on_model() {
    let gguf = biv_cli::open_gguf(model_path()).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let model = Model::load(&gguf).expect("model");
    let engine = engine(&model);
    let none = SamplerOverrides {
        temperature: None,
        top_k: None,
        top_p: None,
        min_p: None,
        presence_penalty: None,
        repeat_penalty: None,
    };
    let cfg = SamplerConfig::from_gguf(&gguf, none, SEED).expect("sampler config");
    let turn = |opts: ChatOptions, text: &str, stop: &dyn Fn() -> bool| {
        let mut chat = Chat::new(engine.as_ref(), &tok, opts).expect("chat");
        let mut sampler = Sampler::new(cfg.clone());
        chat.send(text, &mut sampler, |_| {}, stop).expect("turn")
    };

    // the model ends a short answer
    let end = ChatOptions {
        thinking: false,
        ..options()
    };
    let ended = turn(end, SHORT, &|| false);
    assert_eq!(ended.stop, Stop::End);

    let few = ChatOptions {
        max_tokens: FEW_TOKENS,
        ..options()
    };
    let cut = turn(few, FIRST, &|| false);
    assert_eq!(cut.stop, Stop::MaxTokens);
    assert_eq!(cut.gen_tokens, FEW_TOKENS);

    let cancelled = turn(options(), FIRST, &|| true);
    assert_eq!(cancelled.stop, Stop::Cancel);
    assert!(cancelled.gen_tokens < MAX_TOKENS);

    // a context with a few tokens of room above the prompt
    let tight = ChatOptions {
        context: cut.prompt_tokens + ROOM,
        ..options()
    };
    let full = turn(tight, FIRST, &|| false);
    assert_eq!(full.stop, Stop::ContextFull);
}
