//! Lookup: tokens proposed from the conversation, verified in one pass.
//! Greedy answers with and without lookup must be the same text, on the
//! engine itself and in a batch, with the model's MTP head drafting too when
//! it has one, and on a model without one (whose sessions still need the
//! room to verify the proposals).
//! `lookup <gguf> [--cpu] [-n tokens] [-l lookup]`
use biv::{
    batch::{self, BatchOptions},
    chat::{Chat, ChatOptions, Event, TurnStats},
    engine::Engine,
    model::Model,
    sampler::{Sampler, SamplerConfig},
    tokenizer::Tokenizer,
};

/// Answers that copy from the prompt, so that lookup proposes tokens
const PROMPTS: [&str; 2] = [
    "Repeat this sentence three times, exactly, one per line: the quick brown fox jumps over \
     the lazy dog near the old river bank.",
    "Here is a function:\n\nfn add(a: u32, b: u32) -> u32 {\n    a + b\n}\n\nWrite it again, \
     renamed to plus, with a doc comment.",
];

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
    let (max_tokens, lookup) = (flag("-n", 96), flag("-l", 7));
    let path = args
        .first()
        .expect("usage: lookup <gguf> [--cpu] [-n tokens] [-l lookup]");

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
    let mtp = if model.has_mtp() {
        "MTP head"
    } else {
        "no MTP head"
    };
    println!("{}, {mtp}", engine.description());

    // the engine's drafts, and none: lookup alone
    for draft in [None, Some(0)] {
        let opts = |lookup| ChatOptions {
            thinking: false,
            max_tokens,
            context: 4096,
            draft,
            lookup,
            ..Default::default()
        };
        let mut looked_up = 0;
        for prompt in PROMPTS {
            let (plain, _) = answer(engine, &tok, &opts(0), prompt);
            let (text, s) = answer(engine, &tok, &opts(lookup), prompt);
            let batched = batch::serve(engine, &BatchOptions::default(), |h| {
                answer(&h, &tok, &opts(lookup), prompt)
            })
            .unwrap();
            println!(
                "draft {draft:?}: {} tokens, {}/{} looked up accepted, {}/{} drafted accepted",
                s.gen_tokens, s.looked_up_accepted, s.looked_up, s.accepted, s.drafted
            );
            looked_up += s.looked_up;
            assert_eq!(text, plain, "the answer changed with lookup");
            assert_eq!(
                batched.0, plain,
                "the answer changed with lookup, in a batch"
            );
        }
        assert!(looked_up > 0, "lookup proposed nothing");
    }
    println!("same answers with and without lookup, alone and in a batch");
}

/// The greedy answer of a new chat on `engine` to `prompt`, after a first
/// turn, and its stats
fn answer(
    engine: &dyn Engine,
    tok: &Tokenizer,
    opts: &ChatOptions,
    prompt: &str,
) -> (String, TurnStats) {
    let mut chat = Chat::new(engine, tok, opts.clone()).unwrap();
    let mut sampler = Sampler::new(SamplerConfig {
        temperature: 0.0,
        ..Default::default()
    });
    chat.send("Hello", &mut sampler, |_| {}, || false).unwrap();
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
    (text, s)
}
