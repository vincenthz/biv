//! The prompts of the compiled chat template against the built-in format's:
//! the same conversation (a system prompt, tools, function calls and their
//! results, a question after them, and a prefix saved before the first
//! message), greedy, rendered both ways. Where the transcripts part, it shows
//! both sides.
//! `template <gguf> [--cpu] [--think]`
use biv::{
    chat::{Chat, ChatOptions, Event, ToolCall},
    engine::Engine,
    model::Model,
    sampler::{Sampler, SamplerConfig},
    tokenizer::{TokenId, Tokenizer},
};

const WEATHER: &str = r#"{"type": "function", "function": {"name": "get_weather", "description": "Current weather in a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}}}"#;
const TIME: &str = r#"{"type": "function", "function": {"name": "get_time", "description": "Current local time in an IANA time zone", "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": ["timezone"]}}}"#;

fn greedy() -> Sampler {
    Sampler::new(SamplerConfig {
        temperature: 0.0,
        ..Default::default()
    })
}

fn result(c: &ToolCall) -> String {
    match c.name.as_str() {
        "get_weather" => r#"{"temperature": 18, "condition": "light rain"}"#.into(),
        "get_time" => r#"{"time": "2026-10-30T19:42:00+09:00"}"#.into(),
        _ => r#"{"error": "unknown function"}"#.into(),
    }
}

/// A turn: the user's message, or the results of the last answer's calls
fn turn(chat: &mut Chat<'_>, user: Option<&str>, calls: &[ToolCall]) -> Vec<ToolCall> {
    let ignore = |_: Event<'_>| {};
    let stats = match user {
        Some(u) => chat.send(u, &mut greedy(), ignore, || false),
        None => {
            let results: Vec<String> = calls.iter().map(result).collect();
            chat.send_tool_responses(&results, &mut greedy(), ignore, || false)
        }
    }
    .unwrap();
    stats.tool_calls
}

/// The conversation, from `chat` as it is: the tokens after each turn
fn converse(chat: &mut Chat<'_>) -> Vec<Vec<TokenId>> {
    let mut after = Vec::new();
    let mut calls = turn(
        chat,
        Some("What's the weather in Paris, and the time in Tokyo?"),
        &[],
    );
    after.push(chat.transcript().tokens);
    for _ in 0..2 {
        if calls.is_empty() {
            break;
        }
        calls = turn(chat, None, &calls);
        after.push(chat.transcript().tokens);
    }
    turn(
        chat,
        Some("Thanks. Say it again in one short sentence."),
        &[],
    );
    after.push(chat.transcript().tokens);
    after
}

fn show(tok: &Tokenizer, tokens: &[TokenId]) -> String {
    let bytes: Vec<u8> = tokens
        .iter()
        .flat_map(|&t| tok.token_bytes(t).to_vec())
        .collect();
    format!("{:?}", String::from_utf8_lossy(&bytes))
}

/// Whether the transcripts of both ways are the same, after each turn
fn compare(
    tok: &Tokenizer,
    what: &str,
    compiled: &[Vec<TokenId>],
    builtin: &[Vec<TokenId>],
) -> bool {
    for (i, (a, b)) in compiled.iter().zip(builtin).enumerate() {
        if a != b {
            let at = a.iter().zip(b).take_while(|(x, y)| x == y).count();
            let from = at.saturating_sub(12);
            println!(
                "{what}: turn {}: they part at token {at} of {} / {}",
                i + 1,
                a.len(),
                b.len()
            );
            println!(
                "  compiled: {}",
                show(tok, &a[from..(at + 24).min(a.len())])
            );
            println!(
                "  built-in: {}",
                show(tok, &b[from..(at + 24).min(b.len())])
            );
            return false;
        }
    }
    if compiled.len() != builtin.len() {
        println!("{what}: {} turns / {}", compiled.len(), builtin.len());
        return false;
    }
    let n = compiled.last().map_or(0, |t| t.len());
    println!("{what}: the same, {} turns, {n} tokens", compiled.len());
    true
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let cpu = args.iter().any(|a| a == "--cpu");
    let think = args.iter().any(|a| a == "--think");
    args.retain(|a| !a.starts_with("--"));
    let path = args
        .first()
        .expect("usage: template <gguf> [--cpu] [--think]");

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

    let opts = ChatOptions {
        system: Some("You are a helpful assistant.".into()),
        thinking: think,
        tools: vec![WEATHER.into(), TIME.into()],
        max_tokens: if think { 384 } else { 160 },
        context: 8192,
        ..Default::default()
    };
    let builtin = ChatOptions {
        builtin_format: true,
        ..opts.clone()
    };
    let mut chat = Chat::new(engine, &tok, opts.clone()).unwrap();
    let Some(t) = chat.template() else {
        println!("the model's chat template is not a compiled one");
        return;
    };
    println!("compiled template: {}, thinking {think}", t.name());
    let mut ok = compare(
        &tok,
        "conversation",
        &converse(&mut chat),
        &converse(&mut Chat::new(engine, &tok, builtin.clone()).unwrap()),
    );

    // the system block saved before the first message, then the conversation
    // in another chat, restored from it
    let saved = |opts: &ChatOptions| {
        let mut saver = Chat::new(engine, &tok, opts.clone()).unwrap();
        let prefix = saver.save().unwrap();
        let mut chat = Chat::new(engine, &tok, opts.clone()).unwrap();
        chat.restore(&prefix).unwrap();
        (prefix.n_past(), converse(&mut chat))
    };
    let (n_compiled, compiled) = saved(&opts);
    let (n_builtin, built) = saved(&builtin);
    println!("saved prefix: {n_compiled} tokens / {n_builtin}");
    ok &= n_compiled == n_builtin;
    ok &= compare(&tok, "from the saved prefix", &compiled, &built);
    println!("{}", if ok { "SAME" } else { "DIFFERENT" });
}
