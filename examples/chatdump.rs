//! A chat, greedy, with what the chat showed (tagged by event) and the whole
//! sequence it processed, token by token, special tokens marked.
//! `chatdump <gguf> [--cpu] [--no-think] [--effort L] [--builtin] [--tools]
//! [-n N] [--json out.json] <message>...`
//! The messages are sent in turn; with `--tools`, the system prompt and the
//! two functions of `examples/template.rs`, whose calls get canned results
//! (up to three rounds per message). `--json` writes the conversation as
//! OpenAI-style messages, for rendering the chat template out of band.
use std::fmt::Write as _;

use biv::{
    chat::{Chat, ChatOptions, Event, ReasoningEffort, ToolCall},
    engine::Engine,
    model::Model,
    sampler::{Sampler, SamplerConfig},
    tokenizer::Tokenizer,
};

const SYSTEM: &str = "You are a helpful assistant.";
const WEATHER: &str = r#"{"type": "function", "function": {"name": "get_weather", "description": "Current weather in a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}}}"#;
const TIME: &str = r#"{"type": "function", "function": {"name": "get_time", "description": "Current local time in an IANA time zone", "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": ["timezone"]}}}"#;

fn result(c: &ToolCall) -> String {
    match c.name.as_str() {
        "get_weather" => r#"{"temperature": 18, "condition": "light rain"}"#.into(),
        "get_time" => r#"{"time": "2026-10-30T19:42:00+09:00"}"#.into(),
        _ => r#"{"error": "unknown function"}"#.into(),
    }
}

fn js(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => {
                let _ = write!(o, "\\u{:04x}", c as u32);
            }
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// What one answer showed
#[derive(Default)]
struct Shown {
    thinking: String,
    answer: String,
    calls: Vec<ToolCall>,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: chatdump <gguf> [options] <message>...");
    let (mut cpu, mut think, mut builtin, mut tools, mut effort, mut n) =
        (false, true, false, false, ReasoningEffort::XHigh, 512);
    let mut json_out = None;
    let mut messages = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--cpu" => cpu = true,
            "--no-think" => think = false,
            "--builtin" => builtin = true,
            "--tools" => tools = true,
            "--effort" => effort = ReasoningEffort::parse(&args.next().unwrap()).unwrap(),
            "-n" => n = args.next().unwrap().parse().unwrap(),
            "--json" => json_out = args.next(),
            _ => messages.push(a),
        }
    }
    let g = biv_cli::open_gguf(&path).unwrap();
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
    let opts = ChatOptions {
        system: tools.then(|| SYSTEM.into()),
        tools: if tools {
            vec![WEATHER.into(), TIME.into()]
        } else {
            Vec::new()
        },
        thinking: think,
        effort,
        max_tokens: n,
        context: 8192,
        builtin_format: builtin,
        ..Default::default()
    };
    let mut chat = Chat::new(engine, &tok, opts).unwrap();
    println!(
        "{} | template {:?} | thinking {think} effort {}",
        engine.description(),
        chat.template().map(|t| t.name()),
        effort.as_str()
    );
    let mut conv: Vec<String> = Vec::new();
    if tools {
        conv.push(format!(
            r#"{{"role": "system", "content": {}}}"#,
            js(SYSTEM)
        ));
    }
    for m in &messages {
        conv.push(format!(r#"{{"role": "user", "content": {}}}"#, js(m)));
        let mut results: Option<Vec<String>> = None;
        for _round in 0..4 {
            let mut sampler = Sampler::new(SamplerConfig {
                temperature: 0.0,
                ..Default::default()
            });
            let mut last = "";
            let mut shown = Shown::default();
            let on_event = |ev: Event<'_>| {
                let (tag, text) = match ev {
                    Event::PromptDone => return,
                    Event::Thinking(t) => {
                        shown.thinking.push_str(t);
                        ("THINK", t.to_string())
                    }
                    Event::ThinkingDone => ("THINK_DONE", String::new()),
                    Event::Answer(t) => {
                        shown.answer.push_str(t);
                        ("ANSWER", t.to_string())
                    }
                    Event::ToolCall(c) => {
                        shown.calls.push(c.clone());
                        ("CALL", format!("{c:?}"))
                    }
                };
                if tag != last || tag == "CALL" {
                    print!("\n[{tag}] ");
                    last = tag;
                }
                print!("{text}");
            };
            let stats = match &results {
                None => chat.send(m, &mut sampler, on_event, || false),
                Some(r) => chat.send_tool_responses(r, &mut sampler, on_event, || false),
            }
            .unwrap();
            println!(
                "\n-- {} prompt tokens, {} generated, truncated {}, {} calls",
                stats.prompt_tokens,
                stats.gen_tokens,
                stats.truncated,
                stats.tool_calls.len()
            );
            let mut msg = String::from(r#"{"role": "assistant""#);
            if !shown.thinking.is_empty() {
                let _ = write!(msg, r#", "reasoning_content": {}"#, js(&shown.thinking));
            }
            let _ = write!(msg, r#", "content": {}"#, js(&shown.answer));
            if !shown.calls.is_empty() {
                msg.push_str(r#", "tool_calls": ["#);
                for (i, c) in shown.calls.iter().enumerate() {
                    let args: Vec<String> = c
                        .arguments
                        .iter()
                        .map(|(k, v)| format!("{}: {}", js(k), js(v)))
                        .collect();
                    let _ = write!(
                        msg,
                        r#"{}{{"type": "function", "function": {{"name": {}, "arguments": {{{}}}}}}}"#,
                        if i > 0 { ", " } else { "" },
                        js(&c.name),
                        args.join(", ")
                    );
                }
                msg.push(']');
            }
            msg.push('}');
            conv.push(msg);
            if !tools || stats.tool_calls.is_empty() {
                break;
            }
            let r: Vec<String> = stats.tool_calls.iter().map(result).collect();
            for (c, r) in stats.tool_calls.iter().zip(&r) {
                conv.push(format!(
                    r#"{{"role": "tool", "name": {}, "content": {}}}"#,
                    js(&c.name),
                    js(r)
                ));
            }
            results = Some(r);
        }
    }
    let t = chat.transcript();
    println!("-- sequence ({} tokens):", t.tokens.len());
    let mut line = String::new();
    for &id in &t.tokens {
        let text = String::from_utf8_lossy(tok.token_bytes(id)).into_owned();
        if tok.is_special(id) {
            if !line.is_empty() {
                println!("  {line:?}");
                line.clear();
            }
            println!("{id:>7} {text}");
        } else {
            line.push_str(&text);
        }
    }
    if !line.is_empty() {
        println!("  {line:?}");
    }
    let ids: Vec<String> = t.tokens.iter().map(|i| i.to_string()).collect();
    println!("-- ids: {}", ids.join(","));
    if let Some(out) = json_out {
        let tools_json = if tools {
            format!("[{WEATHER}, {TIME}]")
        } else {
            "null".into()
        };
        let effort = if think { effort.as_str() } else { "none" };
        std::fs::write(
            &out,
            format!(
                "{{\"effort\": {}, \"tools\": {tools_json}, \"messages\": [\n{}\n], \"ids\": [{}]}}\n",
                js(effort),
                conv.join(",\n"),
                ids.join(",")
            ),
        )
        .unwrap();
    }
}
