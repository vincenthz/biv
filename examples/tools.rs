//! Function calling: tools in the system block, the model's calls, their
//! results sent back, and a saved prefix holding the tools.
//! `tools <gguf> [--cpu]`
use biv::{
    chat::{Chat, ChatOptions, Event, ReasoningEffort, ToolCall},
    engine::Engine,
    model::Model,
    sampler::{Sampler, SamplerConfig},
    tokenizer::{PromptPart, Tokenizer},
};

const WEATHER: &str = r#"{"type": "function", "function": {"name": "get_weather", "description": "Current weather in a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}"#;
const TIME: &str = r#"{"type": "function", "function": {"name": "get_time", "description": "Current local time in an IANA time zone", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "description": "e.g. Europe/Paris"}}, "required": ["timezone"]}}}"#;
const SYSTEM: &str = "You are a helpful assistant.";

/// The template's text around the tool definitions, copied from the GGUF's chat template
const TOOLS_HEADER: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
const TOOLS_FOOTER: &str = "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

struct Turn {
    thinking: String,
    answer: String,
    calls: Vec<ToolCall>,
    prompt_tokens: usize,
}

fn greedy() -> Sampler {
    Sampler::new(SamplerConfig {
        temperature: 0.0,
        ..Default::default()
    })
}

fn collect(send: impl FnOnce(&mut dyn FnMut(Event<'_>)) -> biv::chat::TurnStats) -> Turn {
    let (mut thinking, mut answer) = (String::new(), String::new());
    let stats = send(&mut |ev| match ev {
        Event::Thinking(t) => thinking.push_str(t),
        Event::Answer(a) => answer.push_str(a),
        _ => {}
    });
    Turn {
        thinking,
        answer,
        calls: stats.tool_calls,
        prompt_tokens: stats.prompt_tokens,
    }
}

fn ask(chat: &mut Chat<'_>, user: &str) -> Turn {
    collect(|f| chat.send(user, &mut greedy(), f, || false).unwrap())
}

fn respond(chat: &mut Chat<'_>, responses: &[String]) -> Turn {
    collect(|f| {
        chat.send_tool_responses(responses, &mut greedy(), f, || false)
            .unwrap()
    })
}

fn show(what: &str, t: &Turn) {
    println!("{what}:");
    if !t.thinking.is_empty() {
        println!("  thinking: {} chars", t.thinking.len());
    }
    if !t.answer.is_empty() {
        println!("  answer: {:?}", t.answer);
    }
    for c in &t.calls {
        println!("  call: {} {:?}", c.name, c.arguments);
    }
}

/// A made-up result for a call
fn result(c: &ToolCall) -> String {
    match c.name.as_str() {
        "get_weather" => {
            r#"{"temperature": 18, "condition": "light rain", "unit": "celsius"}"#.into()
        }
        "get_time" => r#"{"time": "2026-10-30T19:42:00+09:00"}"#.into(),
        _ => r#"{"error": "unknown function"}"#.into(),
    }
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let cpu = args.iter().any(|a| a == "--cpu");
    args.retain(|a| a != "--cpu");
    let path = args.first().expect("usage: tools <gguf> [--cpu]");

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

    let tools = vec![WEATHER.to_string(), TIME.to_string()];
    let opts = ChatOptions {
        system: Some(SYSTEM.into()),
        thinking: false,
        tools: tools.clone(),
        max_tokens: 512,
        context: 8192,
        ..Default::default()
    };
    let question = "What's the weather in Paris, and what time is it in Tokyo?";

    // the first prompt, as the chat template renders it
    let rendered = format!(
        "<|im_start|>system\n{TOOLS_HEADER}\n{WEATHER}\n{TIME}{TOOLS_FOOTER}\n\n{SYSTEM}<|im_end|>\n\
         <|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    let expected = tok.encode_with_special(&rendered);
    use PromptPart::{Content, Template};
    let parts = [
        Template("<|im_start|>system\n"),
        Template(TOOLS_HEADER),
        Template("\n"),
        Content(WEATHER),
        Template("\n"),
        Content(TIME),
        Template(TOOLS_FOOTER),
        Template("\n\n"),
        Content(SYSTEM),
        Template("<|im_end|>\n<|im_start|>user\n"),
        Content(question),
        Template("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    ];
    let by_parts = tok.encode_prompt(&parts);
    let separately: Vec<_> = parts
        .iter()
        .flat_map(|p| match *p {
            Template(s) => tok.encode_with_special(s),
            Content(s) => tok.encode(s),
        })
        .collect();
    println!(
        "tokenization of the first prompt ({} tokens): by parts {}, parts tokenized separately {}",
        expected.len(),
        if expected == by_parts {
            "same"
        } else {
            "DIFFERENT"
        },
        if expected == separately {
            "same"
        } else {
            "different"
        },
    );

    let mut chat = Chat::new(engine, &tok, opts.clone()).unwrap();
    let t1 = ask(&mut chat, question);
    show("tools question", &t1);
    println!(
        "  prompt: {} tokens, {} rendered by hand",
        t1.prompt_tokens,
        if t1.prompt_tokens == expected.len() {
            "same as"
        } else {
            "DIFFERENT from"
        }
    );
    let responses: Vec<String> = t1.calls.iter().map(result).collect();
    let t2 = respond(&mut chat, &responses);
    show("after the tool responses", &t2);

    let t3 = ask(
        &mut Chat::new(engine, &tok, opts.clone()).unwrap(),
        "What is the capital of Japan?",
    );
    show("question without a tool", &t3);

    // the system prompt and tools saved, then restored in a chat without tools
    let mut saver = Chat::new(engine, &tok, opts.clone()).unwrap();
    let prefix = saver.save().unwrap();
    println!(
        "saved system prompt with tools: {} tokens, {} MB",
        prefix.n_past(),
        prefix.bytes() >> 20
    );
    let a = ask(&mut saver, question);
    let mut restored = Chat::new(
        engine,
        &tok,
        ChatOptions {
            tools: Vec::new(),
            system: None,
            ..opts.clone()
        },
    )
    .unwrap();
    restored.restore(&prefix).unwrap();
    let b = ask(&mut restored, question);
    println!(
        "restored prefix: {} calls, {} as the chat that saved it",
        b.calls.len(),
        if (&a.calls, &a.answer) == (&b.calls, &b.answer) {
            "same"
        } else {
            "DIFFERENT"
        }
    );

    let thinking = ChatOptions {
        thinking: true,
        effort: ReasoningEffort::Low,
        max_tokens: 2048,
        ..opts.clone()
    };
    let mut chat = Chat::new(engine, &tok, thinking).unwrap();
    let t4 = ask(&mut chat, question);
    show("tools question, thinking", &t4);
    let responses: Vec<String> = t4.calls.iter().map(result).collect();
    if !responses.is_empty() {
        show(
            "after the tool responses, thinking",
            &respond(&mut chat, &responses),
        );
    }
}
