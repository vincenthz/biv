//! T5: the tokens of a prompt that Chat renders with the template of the
//! model file against the tokens of the same conversation rendered with
//! the template directly. Chat uses the template for every prompt, so
//! both must give the same tokens for every scripted conversation.

mod common;

use biv::Error;
use biv::chat::Conversation as ChatConversation;
use biv::chat::{Chat, ChatOptions, Message, ReasoningEffort, Said, Step};
use biv::template::{Origin, Template, Value, json};
use biv::tokenizer::{PromptPart, TokenId, Tokenizer};
use common::{NO_DRAFT, Script, model_path, sampler};

/// The template of the production GGUF, byte for byte (tests/templates)
const TEMPLATE: &str = include_str!("templates/corpus/gguf-Qwen3.8-27B-Q4_1.jinja");
const CONTEXT: usize = 4096;
const ROOM: usize = 64;
const VARIABLES: [&str; 9] = [
    "messages",
    "tools",
    "documents",
    "add_generation_prompt",
    "bos_token",
    "eos_token",
    "enable_thinking",
    "preserve_thinking",
    "reasoning_effort",
];
const WEATHER: &str = r#"{"type": "function", "function": {"name": "get_weather", "description": "Current weather in a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "options": {"type": "object"}}, "required": ["city"]}}}"#;
/// An argument as a model can write it: tojson of its parsed value gives
/// other bytes (`{"unit": "c", "days": 2}`)
const RAW_OPTIONS: &str = r#"{ "unit":"c",  "days":2 }"#;
const ENDED: &str = "<|im_end|>";

/// A first answer with tool calls, scripted, then their responses as the
/// next turn
struct ToolTurn {
    /// the text that the model writes after the generation prompt
    written: String,
    /// the same answer as a message: its reasoning and its calls
    reasoning: Option<&'static str>,
    calls: Vec<(&'static str, Vec<(&'static str, &'static str)>)>,
    responses: Vec<&'static str>,
}

/// The text of one call in the XML parameter form of the template
fn call_text(name: &str, args: &[(&str, &str)]) -> String {
    let mut t = format!("<tool_call>\n<function={name}>\n");
    for (k, v) in args {
        t.push_str(&format!("<parameter={k}>\n{v}\n</parameter>\n"));
    }
    t + "</function>\n</tool_call>"
}

struct Conversation {
    name: &'static str,
    system: Option<&'static str>,
    tools: Vec<&'static str>,
    thinking: bool,
    effort: ReasoningEffort,
    history: Vec<Message<'static>>,
    user: &'static str,
    tool_turn: Option<ToolTurn>,
}

fn conversations() -> Vec<Conversation> {
    let plain = |name, thinking, history, user| Conversation {
        name,
        system: None,
        tools: Vec::new(),
        thinking,
        effort: ReasoningEffort::Medium,
        history,
        user,
        tool_turn: None,
    };
    vec![
        plain("user, thinking", true, vec![], "Name a prime number."),
        plain("user, no thinking", false, vec![], "Name a prime number."),
        plain(
            "two turns, thinking",
            true,
            vec![
                Message::User("Name a prime number."),
                Message::Assistant("Seven."),
            ],
            "And the next one?",
        ),
        plain(
            "reasoned answer as the model wrote it",
            true,
            vec![
                Message::User("Name a prime number."),
                Message::AssistantWithReasoning {
                    reasoning: "\nSeven has no divisor but 1 and itself.\n",
                    answer: "\n\nSeven.",
                },
            ],
            "And the next one?",
        ),
        Conversation {
            name: "system and tools, no thinking",
            system: Some("You are a helpful assistant."),
            tools: vec![WEATHER],
            thinking: false,
            effort: ReasoningEffort::Medium,
            history: vec![],
            user: "What is the weather in Paris?",
            tool_turn: None,
        },
        Conversation {
            name: "effort low",
            system: None,
            tools: Vec::new(),
            thinking: true,
            effort: ReasoningEffort::Low,
            history: vec![],
            user: "Name a prime number.",
            tool_turn: None,
        },
        Conversation {
            name: "effort xhigh",
            system: None,
            tools: Vec::new(),
            thinking: true,
            effort: ReasoningEffort::XHigh,
            history: vec![],
            user: "Name a prime number.",
            tool_turn: None,
        },
        tool_case(
            "a tool call and its response, no thinking",
            false,
            None,
            vec![("get_weather", vec![("city", "Paris")])],
            vec![r#"{"temperature": 18}"#],
        ),
        tool_case(
            "a tool call whose argument bytes tojson would change",
            false,
            None,
            vec![(
                "get_weather",
                vec![("city", "Paris"), ("options", RAW_OPTIONS)],
            )],
            vec![r#"{"temperature": 18}"#],
        ),
        tool_case(
            "two tool calls after reasoning, two responses",
            true,
            Some("The user wants the weather of two cities.\n"),
            vec![
                ("get_weather", vec![("city", "Paris")]),
                ("get_weather", vec![("city", "Rome")]),
            ],
            vec![r#"{"temperature": 18}"#, r#"{"temperature": 24}"#],
        ),
    ]
}

fn tool_case(
    name: &'static str,
    thinking: bool,
    reasoning: Option<&'static str>,
    calls: Vec<(&'static str, Vec<(&'static str, &'static str)>)>,
    responses: Vec<&'static str>,
) -> Conversation {
    // the model ends its reasoning, then writes two newlines before the
    // first call, as after the plain generation prompt
    let mut written = match reasoning {
        Some(r) => format!("{r}</think>\n\n"),
        None => String::new(),
    };
    let texts: Vec<String> = calls.iter().map(|(n, a)| call_text(n, a)).collect();
    written.push_str(&texts.join("\n"));
    Conversation {
        name,
        system: None,
        tools: vec![WEATHER],
        thinking,
        effort: ReasoningEffort::Low,
        history: vec![],
        user: "What is the weather in Paris and in Rome?",
        tool_turn: Some(ToolTurn {
            written,
            reasoning,
            calls,
            responses,
        }),
    }
}

fn effort_word(e: ReasoningEffort) -> &'static str {
    match e {
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::XHigh => "xhigh",
    }
}

/// The prompt tokens of Chat: the chat renders the conversation with the
/// template, and the D7 prefix check decides whether to reuse or reset.
/// For a single turn, the tokens are the render of the conversation. For a
/// tool turn, the first answer runs to its end (the Chat builds the
/// Said::Assistant from the answer), then the render of the full
/// conversation with the tool responses gives the tokens.
fn old_tokens(tok: &Tokenizer, t: &Template, c: &Conversation) -> Vec<TokenId> {
    let end = tok.token_id(ENDED).expect("the model has the token");
    let mut script = Vec::new();
    if let Some(t) = &c.tool_turn {
        script = tok.encode_with_special(&t.written);
        script.push(end);
    }
    script.push(end);
    let engine = Script::new(script, tok.n_vocab());
    let opts = ChatOptions {
        system: c.system.map(String::from),
        thinking: c.thinking,
        effort: c.effort,
        context: CONTEXT,
        max_tokens: ROOM,
        draft: Some(NO_DRAFT),
        lookup: 0,
        tools: c.tools.iter().map(|t| t.to_string()).collect(),
        ..ChatOptions::default()
    };
    let mut chat = Chat::new(&engine, tok, opts).expect("chat");
    let mut s = sampler();
    let mut to_end = |chat: &mut Chat<'_>| {
        while !matches!(chat.step(&mut s, |_| {}).expect("step"), Step::Done(..)) {}
    };
    chat.begin_with_history(&c.history, c.user).expect("turn");
    if c.tool_turn.is_none() {
        let n = chat.turn_prompt_tokens().expect("an open turn");
        to_end(&mut chat);
        return engine.fed.take()[..n].to_vec();
    }
    // tool case: run the first turn to build the Said::Assistant, then
    // render the conversation with the tool responses from the test side,
    // which verifies that Chat built the correct conversation
    let tt = c.tool_turn.as_ref().unwrap();
    to_end(&mut chat);
    let conv = chat.conversation();
    // the conversation has: the history messages, the user message, and
    // the assistant's answer (from answer_said)
    let mut messages = Vec::new();
    render_conversation_messages(conv, &mut messages);
    for r in &tt.responses {
        messages.push(message("tool", r, None));
    }
    render_with_messages(tok, t, c, messages)
}

fn text(s: &str) -> Value {
    Value::str(s, Origin::Content)
}

fn message(role: &str, content: &str, reasoning: Option<&str>) -> Value {
    message_with(role, content, reasoning, Vec::new())
}

fn message_with(
    role: &str,
    content: &str,
    reasoning: Option<&str>,
    extra: Vec<(Value, Value)>,
) -> Value {
    let mut m = vec![(text("role"), text(role)), (text("content"), text(content))];
    if let Some(r) = reasoning {
        m.push((text("reasoning_content"), text(r)));
    }
    m.extend(extra);
    Value::map(m)
}

/// The messages of the Chat's conversation as template values
fn render_conversation_messages(conv: &ChatConversation, messages: &mut Vec<Value>) {
    if let Some(s) = &conv.system {
        messages.push(message("system", s, None));
    }
    for m in &conv.messages {
        messages.push(match m {
            Said::User(u) => message("user", u, None),
            Said::Tool(r) => message("tool", r, None),
            Said::Assistant {
                reasoning,
                answer,
                tool_calls,
            } => {
                let mut extra = Vec::new();
                if !tool_calls.is_empty() {
                    let calls = tool_calls
                        .iter()
                        .map(|c| {
                            let arguments = c
                                .arguments
                                .iter()
                                .map(|(k, v)| (text(k), text(v)))
                                .collect();
                            Value::map(vec![
                                (text("type"), text("function")),
                                (
                                    text("function"),
                                    Value::map(vec![
                                        (text("name"), text(&c.name)),
                                        (text("arguments"), Value::map(arguments)),
                                    ]),
                                ),
                            ])
                        })
                        .collect();
                    extra.push((text("tool_calls"), Value::list(calls)));
                }
                message_with("assistant", answer, reasoning.as_deref(), extra)
            }
        });
    }
}

/// Render `messages` with the template, as `new_tokens` does
fn render_with_messages(
    tok: &Tokenizer,
    t: &Template,
    c: &Conversation,
    messages: Vec<Value>,
) -> Vec<TokenId> {
    let tools = if c.tools.is_empty() {
        Value::None
    } else {
        Value::list(
            c.tools
                .iter()
                .map(|t| json::parse(t, Origin::Content).expect("a tool"))
                .collect(),
        )
    };
    let mut vars = vec![
        ("messages".to_string(), Value::list(messages)),
        ("tools".to_string(), tools),
        ("documents".to_string(), Value::None),
        ("add_generation_prompt".to_string(), Value::Bool(true)),
        ("bos_token".to_string(), Value::str("", Origin::Template)),
        ("eos_token".to_string(), Value::str("", Origin::Template)),
        ("enable_thinking".to_string(), Value::Bool(c.thinking)),
        ("preserve_thinking".to_string(), Value::Bool(true)),
    ];
    if c.thinking {
        vars.push((
            "reasoning_effort".to_string(),
            Value::str(effort_word(c.effort), Origin::Template),
        ));
    }
    let segments = t.render(&vars, None).expect("it renders");
    let parts: Vec<PromptPart<'_>> = segments
        .iter()
        .map(|s| match s.origin {
            Origin::Template => PromptPart::Template(&s.text),
            Origin::Content => PromptPart::Content(&s.text),
        })
        .collect();
    tok.encode_prompt(&parts)
}

/// The prompt tokens of the new path: the template of the model file
fn new_tokens(tok: &Tokenizer, t: &Template, c: &Conversation) -> Vec<TokenId> {
    let mut messages = Vec::new();
    if let Some(s) = c.system {
        messages.push(message("system", s, None));
    }
    for m in &c.history {
        messages.push(match *m {
            Message::User(u) => message("user", u, None),
            Message::Assistant(a) => message("assistant", a, None),
            Message::AssistantWithReasoning { reasoning, answer } => {
                message("assistant", answer, Some(reasoning))
            }
        });
    }
    messages.push(message("user", c.user, None));
    if let Some(t) = &c.tool_turn {
        let calls = t
            .calls
            .iter()
            .map(|(name, args)| {
                let arguments = args.iter().map(|(k, v)| (text(k), text(v))).collect();
                Value::map(vec![
                    (text("type"), text("function")),
                    (
                        text("function"),
                        Value::map(vec![
                            (text("name"), text(name)),
                            (text("arguments"), Value::map(arguments)),
                        ]),
                    ),
                ])
            })
            .collect();
        messages.push(message_with(
            "assistant",
            "",
            t.reasoning,
            vec![(text("tool_calls"), Value::list(calls))],
        ));
        for r in &t.responses {
            messages.push(message("tool", r, None));
        }
    }
    let tools = if c.tools.is_empty() {
        Value::None
    } else {
        Value::list(
            c.tools
                .iter()
                .map(|t| json::parse(t, Origin::Content).expect("a tool"))
                .collect(),
        )
    };
    let mut vars = vec![
        ("messages".to_string(), Value::list(messages)),
        ("tools".to_string(), tools),
        ("documents".to_string(), Value::None),
        ("add_generation_prompt".to_string(), Value::Bool(true)),
        ("bos_token".to_string(), Value::str("", Origin::Template)),
        ("eos_token".to_string(), Value::str("", Origin::Template)),
        ("enable_thinking".to_string(), Value::Bool(c.thinking)),
        ("preserve_thinking".to_string(), Value::Bool(true)),
    ];
    if c.thinking {
        vars.push((
            "reasoning_effort".to_string(),
            Value::str(effort_word(c.effort), Origin::Template),
        ));
    }
    let segments = t.render(&vars, None).expect("it renders");
    let parts: Vec<PromptPart<'_>> = segments
        .iter()
        .map(|s| match s.origin {
            Origin::Template => PromptPart::Template(&s.text),
            Origin::Content => PromptPart::Content(&s.text),
        })
        .collect();
    tok.encode_prompt(&parts)
}

#[test]
#[ignore = "needs a model file"]
fn the_template_of_the_model_file_gives_the_tokens_of_the_fixed_template() {
    let gguf = biv_cli::open_gguf(model_path()).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let template = Template::load(TEMPLATE, &VARIABLES).expect("it loads");
    let mut differ = Vec::new();
    for c in conversations() {
        let (old, new) = (
            old_tokens(&tok, &template, &c),
            new_tokens(&tok, &template, &c),
        );
        if old == new {
            println!("same: {}", c.name);
            continue;
        }
        let at = old
            .iter()
            .zip(&new)
            .position(|(a, b)| a != b)
            .unwrap_or(old.len().min(new.len()));
        let show = |t: &[TokenId]| {
            format!(
                "{:?}",
                tok.decode(&t[at.saturating_sub(4)..(at + 6).min(t.len())])
            )
        };
        differ.push(format!(
            "{}: {} old tokens, {} new tokens, first difference at {at}: old {} new {}",
            c.name,
            old.len(),
            new.len(),
            show(&old),
            show(&new)
        ));
    }
    for d in &differ {
        println!("DIFFER {d}");
    }
    assert!(differ.is_empty(), "{} conversations differ", differ.len());
}

/// A template whose generation prompts with thinking on and off do not
/// share a prefix fails the probe: the reasoning end token cannot be
/// found from them.
#[test]
#[ignore = "needs a model file"]
fn misaligned_generation_prompts_fail_the_probe() {
    // a template that renders reasoning in a past message (so the probe
    // finds it), but whose generation prompts with thinking on and off
    // start with different texts
    let source = r#"{%- for message in messages -%}
<|im_start|>{{ message.role }}
{%- if message.reasoning_content is defined and message.reasoning_content %}
<think>
{{ message.reasoning_content }}</think>
{% endif %}
{{ message.content }}<|im_end|>
{% endfor -%}
{%- if add_generation_prompt -%}
{%- if enable_thinking -%}
<|im_start|>assistant
<think>
{%- else -%}
<|im_start|>assistant
NO_THINK_PROMPT
{%- endif -%}
{%- endif -%}"#;
    let gguf = biv_cli::open_gguf(model_path()).expect("gguf");
    let mut tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    tok.chat_template = Some(source.to_string());
    let engine = Script::new(vec![], tok.n_vocab());
    let opts = ChatOptions {
        thinking: true,
        ..ChatOptions::default()
    };
    match Chat::new(&engine, &tok, opts) {
        Err(Error::Probe(..)) => {}
        Err(e) => panic!("expected a probe error, got: {e}"),
        Ok(_) => panic!("must fail"),
    }
}
