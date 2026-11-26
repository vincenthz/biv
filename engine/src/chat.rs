//! Multi-turn chat on top of the model, following the GGUF's chat template.
//! When it is one of the templates compiled into [`crate::template`], found by
//! the hash of its text, that code renders the prompts. Otherwise built-in
//! formats do: ChatML with `<think>` reasoning blocks (Qwen), Flint's role
//! tokens (`<|user|>`, `<|assistant|>`, …), or Inkling-2's blocks
//! (`<|message_model|><|content_text|>…<|end_message|>`), told apart by the
//! vocabulary. The format, from the vocabulary either way, is what reads the
//! answers: the reasoning blocks, the function calls, the end of an answer.
//!
//! Every turn only appends to the sequence: the model state is kept between
//! turns and only the new tokens are processed. This suits the recurrent
//! layers, whose state cannot be rewound (beyond the last verify pass of
//! speculative decoding), and is why past reasoning stays in the history
//! (`preserve_thinking`), even with templates that would drop it. An answer
//! stays as the model generated it: a turn renders only what follows it, the
//! template's end of the answer and the new messages, from a conversation
//! where stand-ins take the place of those before (see [`Chat::template_turn`]).
//!
//! With tools, the model may answer with function calls ([`Event::ToolCall`]),
//! whose results go back with [`Chat::send_tool_responses`].
//!
//! A conversation can be saved at any point between turns and restored later,
//! in the same chat or another one on the same engine: [`Chat::save`] before the
//! first message saves the processed system prompt, so that conversations
//! starting with it don't process it again. Its token ids alone
//! ([`Chat::transcript`]) take a few bytes per token and restore in any chat on
//! a model with the same vocabulary, by processing them again
//! ([`Chat::restore_transcript`]).

use alloc::borrow::Cow;
use alloc::collections::VecDeque;

use crate::Error;
use crate::engine::{DraftLogits, Engine, Session, Snapshot};
use crate::memory::SessionMemory;
use crate::prelude::*;
use crate::sampler::Sampler;
use crate::template::{self as compiled, rt};
use crate::tokenizer::{PromptPart, StreamDecoder, TokenId, Tokenizer};
pub use crate::transcript::Transcript;

use PromptPart::{Content, Template};

/// The chat template's text before the tool definitions in the system block
const TOOLS_HEADER: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
/// And after them
const TOOLS_FOOTER: &str = "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// The Flint template's text before the tool definitions, which follow each
/// other without separators
const FLINT_TOOLS_HEADER: &str = "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>";
const FLINT_TOOLS_FOOTER: &str = "</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>...</tool_call>\n";

/// The chat format of a model, from its special tokens
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    /// `<|im_start|>role\n...<|im_end|>\n`; thinking is switched on or off by
    /// the generation prompt
    ChatMl { im_end: TokenId },
    /// `<|role|>...` with no end tokens: the next role's token ends an answer.
    /// The model opens its reasoning block itself, when it wants to.
    Flint { think: TokenId },
    /// Inkling-2's blocks, see [`Inkling`]
    Inkling(Inkling),
}

/// Inkling-2's tokens. A message is blocks, each a role's header, a content
/// type and the content up to `<|end_message|>`; an answer is the model's
/// blocks (reasoning, text, function calls), the model writing their headers,
/// then `<|content_model_end_sampling|>`. There is no generation prompt, and
/// the reasoning is switched by an effort level written once, before the first
/// message: `0` for none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Inkling {
    /// the headers: the model's, the others'
    model: TokenId,
    user: TokenId,
    system: TokenId,
    tool: TokenId,
    /// the content types
    text: TokenId,
    thinking: TokenId,
    /// `{"name": NAME, "args": {…}}`, after a header with the name, or not
    call: TokenId,
    /// a function's error and a call as text, which only show as text
    error: Option<TokenId>,
    call_text: Option<TokenId>,
    /// `<|end_message|>`, which ends every block
    end: TokenId,
    /// `<|content_model_end_sampling|>`, the end of an answer
    end_sampling: TokenId,
}

/// Where an Inkling answer is, between its tokens
#[derive(Debug, Clone, PartialEq, Eq)]
enum Block {
    Between,
    /// after a header, the text before the content type: a function's name
    Header(String),
    Thinking,
    Text,
    /// a function call's JSON so far, after the header's text
    Call {
        name: String,
        json: String,
    },
    /// content that is not shown (an image's, a declaration's)
    Hidden,
}

/// Reads an Inkling answer token by token into [`Event`]s: the blocks as the
/// reference's parsers take them, the header and its name optional, text not
/// trimmed, a function's name from its JSON, else from the header
struct InklingReader {
    f: Inkling,
    block: Block,
    decoder: StreamDecoder,
    /// as in [`Chat::answer`]: the answer doesn't start with newlines, nor do
    /// they show at its end
    skip_leading_newlines: bool,
    held_newlines: String,
}

impl InklingReader {
    fn new(f: Inkling) -> Self {
        InklingReader {
            f,
            block: Block::Between,
            decoder: StreamDecoder::default(),
            skip_leading_newlines: true,
            held_newlines: String::new(),
        }
    }

    /// Whether `token`, which the model wrote, ends the answer: another role's
    /// header, which only the next turn writes
    fn ends(&self, token: TokenId) -> bool {
        token == self.f.user || token == self.f.system || token == self.f.tool
    }

    /// The end of the block, if one is open
    fn finish(&mut self, calls: &mut Vec<ToolCall>, on_event: &mut dyn FnMut(Event<'_>)) {
        self.decoder = StreamDecoder::default();
        match core::mem::replace(&mut self.block, Block::Between) {
            Block::Thinking => {
                self.held_newlines.clear();
                self.skip_leading_newlines = true;
                on_event(Event::ThinkingDone);
            }
            Block::Call { name, json } => match ToolCall::parse_inkling(&name, &json) {
                Some(c) => {
                    on_event(Event::ToolCall(&c));
                    calls.push(c);
                }
                // not a call after all: show it as it was written
                None => on_event(Event::Answer(&json)),
            },
            _ => {}
        }
    }

    /// Read `token`, which neither ends the answer nor the block
    fn token(
        &mut self,
        token: TokenId,
        tok: &Tokenizer,
        calls: &mut Vec<ToolCall>,
        on_event: &mut dyn FnMut(Event<'_>),
    ) {
        let f = self.f;
        if token == f.end {
            self.finish(calls, on_event);
        } else if token == f.model {
            // a header before the block's end ends it too
            self.finish(calls, on_event);
            self.block = Block::Header(String::new());
        } else if token == f.thinking
            || token == f.text
            || token == f.call
            || Some(token) == f.error
            || Some(token) == f.call_text
        {
            // the header is optional
            let name = match core::mem::replace(&mut self.block, Block::Between) {
                Block::Header(name) => name,
                open => {
                    self.block = open;
                    self.finish(calls, on_event);
                    String::new()
                }
            };
            self.block = if token == f.thinking {
                self.held_newlines.clear();
                self.skip_leading_newlines = true;
                Block::Thinking
            } else if token == f.call {
                Block::Call {
                    name,
                    json: String::new(),
                }
            } else {
                Block::Text
            };
        } else if tok.is_special(token) {
            // another content type, or a token of no block
            if let Block::Between | Block::Header(_) = self.block {
                self.block = Block::Hidden;
            }
        } else {
            let mut text = self.decoder.push(tok.token_bytes(token));
            let thinking = match &mut self.block {
                Block::Header(s) | Block::Call { json: s, .. } => {
                    s.push_str(&text);
                    return;
                }
                Block::Hidden => return,
                Block::Thinking => true,
                // text outside any block shows too
                Block::Text | Block::Between => false,
            };
            if self.skip_leading_newlines {
                let trimmed = text.trim_start_matches('\n');
                if !trimmed.is_empty() {
                    self.skip_leading_newlines = false;
                }
                text = trimmed.to_string();
            }
            let body = text.trim_end_matches('\n');
            if !body.is_empty() {
                let shown = core::mem::take(&mut self.held_newlines) + body;
                on_event(if thinking {
                    Event::Thinking(&shown)
                } else {
                    Event::Answer(&shown)
                });
            }
            self.held_newlines.push_str(&text[body.len()..]);
        }
    }

    /// At the end of the answer, however it ends: the end of the open block,
    /// and the tokens that close the answer in the history, as the template
    /// writes it (`…<|end_message|><|content_model_end_sampling|>`)
    fn close(
        &mut self,
        calls: &mut Vec<ToolCall>,
        on_event: &mut dyn FnMut(Event<'_>),
    ) -> Vec<TokenId> {
        let mut closing = Vec::with_capacity(2);
        if self.block != Block::Between {
            closing.push(self.f.end);
        }
        self.finish(calls, on_event);
        closing.push(self.f.end_sampling);
        closing
    }
}

/// The members of the JSON object `text`, which is valid JSON: the text of
/// each key (quoted) and value, as written
fn json_members(text: &str) -> Vec<(&str, &str)> {
    let b = text.as_bytes();
    let ws = |mut i: usize| {
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') {
            i += 1;
        }
        i
    };
    let string_end = |mut i: usize| {
        i += 1;
        while i < b.len() && b[i] != b'"' {
            i += if b[i] == b'\\' { 2 } else { 1 };
        }
        (i + 1).min(b.len())
    };
    // a value ends at a ',' or the '}' of its object
    let value_end = |mut i: usize| {
        let mut depth = 0usize;
        while i < b.len() {
            match b[i] {
                b'"' => {
                    i = string_end(i);
                    continue;
                }
                b'{' | b'[' => depth += 1,
                b'}' | b']' | b',' if depth == 0 => break,
                b'}' | b']' => depth -= 1,
                _ => {}
            }
            i += 1;
        }
        i
    };
    let mut members = Vec::new();
    // past the '{'
    let mut i = ws(0) + 1;
    loop {
        i = ws(i);
        if b.get(i) != Some(&b'"') {
            return members;
        }
        let key_end = string_end(i);
        let key = &text[i..key_end];
        // past the ':'
        let start = ws(ws(key_end) + 1);
        let end = value_end(start);
        members.push((key, text[start..end].trim_end()));
        i = ws(end);
        if b.get(i) != Some(&b',') {
            return members;
        }
        i += 1;
    }
}

/// The value of `key` in the members of a JSON object, the last one when it
/// has several (as [`rt::Json::parse`])
fn json_member<'t>(members: &[(&str, &'t str)], key: &str) -> Option<&'t str> {
    let named = |k: &str| matches!(rt::Json::parse(k), Ok(rt::Json::Str(k)) if k == key);
    members
        .iter()
        .rev()
        .find(|(k, _)| named(k))
        .map(|&(_, v)| v)
}

/// JSON text without the whitespace between its tokens, as the Inkling
/// template writes a function's arguments
fn json_compact(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let (mut string, mut escaped) = (false, false);
    for c in text.chars() {
        if string {
            match c {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => string = false,
                _ => {}
            }
        } else if c == '"' {
            string = true;
        } else if matches!(c, ' ' | '\t' | '\n' | '\r') {
            continue;
        }
        out.push(c);
    }
    out
}

/// Message text as the Flint template writes it: its `flint_text` macro
/// drops NULs and reasoning tags, and escapes the role tokens
fn flint_text(t: &str) -> Cow<'_, str> {
    const ROLES: [&str; 7] = [
        "begin_of_text",
        "end_of_text",
        "system",
        "developer",
        "user",
        "assistant",
        "observation",
    ];
    let mut t = Cow::Borrowed(t);
    for drop in ["\0", "<think>", "</think>"] {
        if t.contains(drop) {
            t = Cow::Owned(t.replace(drop, ""));
        }
    }
    for role in ROLES {
        let token = format!("<|{role}|>");
        if t.contains(&token) {
            t = Cow::Owned(t.replace(&token, &format!("&lt;|{role}|&gt;")));
        }
    }
    t
}

/// The value of a tool's `"function"` key, the function the Flint template
/// writes, or the whole tool without one
fn tool_function(tool: &str) -> &str {
    // the keys of the outer object: skip strings and nested values
    let b = tool.as_bytes();
    let (mut i, mut depth) = (0, 0);
    let string_end = |mut i: usize| {
        i += 1;
        while i < b.len() && b[i] != b'"' {
            i += if b[i] == b'\\' { 2 } else { 1 };
        }
        i + 1
    };
    while i < b.len() {
        match b[i] {
            b'{' | b'[' => depth += 1,
            b'}' | b']' => depth -= 1,
            b'"' => {
                let end = string_end(i).min(b.len());
                if depth == 1 && &tool[i..end] == "\"function\"" {
                    let rest = tool[end..].trim_start();
                    if let Some(v) = rest.strip_prefix(':') {
                        let v = v.trim_start();
                        let start = tool.len() - v.len();
                        // the value's end: back to depth 1 at a ',' or the '}'
                        let (mut j, mut d) = (start, 0);
                        while j < b.len() {
                            match b[j] {
                                b'"' => {
                                    j = string_end(j);
                                    continue;
                                }
                                b'{' | b'[' => d += 1,
                                b'}' | b']' if d == 0 => break,
                                b'}' | b']' => d -= 1,
                                b',' if d == 0 => break,
                                _ => {}
                            }
                            j += 1;
                        }
                        return tool[start..j.min(b.len())].trim_end();
                    }
                }
                i = end;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    tool
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    Low,
    Medium,
    /// Inkling-2's 0.9, its default; for Qwen the same as `XHigh`, as its
    /// templates have it
    High,
    XHigh,
}

impl ReasoningEffort {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "low" => Self::Low,
            "medium" => Self::Medium,
            "high" => Self::High,
            "xhigh" => Self::XHigh,
            _ => return None,
        })
    }

    /// As chat templates name it (`reasoning_effort`)
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
        }
    }

    /// The system instruction the template adds when thinking is enabled
    fn instructions(self) -> Option<&'static str> {
        match self {
            Self::High | Self::XHigh => Some(
                "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
            ),
            Self::Low => Some(
                "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
            ),
            Self::Medium => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChatOptions {
    pub system: Option<String>,
    /// whether the model reasons before answering, and how much: for models
    /// that can be told, see [`Chat::controls_thinking`]
    pub thinking: bool,
    pub effort: ReasoningEffort,
    /// per answer, thinking included
    pub max_tokens: usize,
    pub context: usize,
    /// prompt tokens processed per forward pass, by default the engine's choice
    pub batch: Option<usize>,
    /// tokens drafted by the MTP head per step, 0 for none, by default the
    /// engine's choice
    pub draft: Option<usize>,
    /// tokens proposed per step from earlier text of the conversation, at
    /// most, 0 for none: when what was just generated appeared before, the
    /// tokens that followed it (a quote, code being rewritten). The session
    /// holds a copy of the recurrent states per token of a step.
    pub lookup: usize,
    /// functions the model may call, each a JSON object as the chat template's
    /// `tojson` writes it, e.g. `{"type": "function", "function": {"name":
    /// "get_weather", "description": "...", "parameters": {...}}}`. A compiled
    /// template writes it again from its `name`, `description` and
    /// `parameters`, the other keys left out.
    pub tools: Vec<String>,
    /// render the prompts with the built-in formats even when the model's
    /// chat template is a compiled one
    pub builtin_format: bool,
}

impl Default for ChatOptions {
    fn default() -> Self {
        ChatOptions {
            system: None,
            thinking: true,
            effort: ReasoningEffort::XHigh,
            max_tokens: 8192,
            context: 16384,
            batch: None,
            draft: None,
            lookup: 0,
            tools: Vec::new(),
            builtin_format: false,
        }
    }
}

/// Pieces of the answer, as they are generated
pub enum Event<'a> {
    /// the prompt is processed, generation starts
    PromptDone,
    Thinking(&'a str),
    Answer(&'a str),
    /// the reasoning block is complete
    ThinkingDone,
    /// a function call, to answer with [`Chat::send_tool_responses`]
    ToolCall(&'a ToolCall),
}

/// A function call the model made
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub name: String,
    /// parameters as the model wrote them: strings verbatim, other values as
    /// JSON (numbers, booleans, arrays, objects)
    pub arguments: Vec<(String, String)>,
}

impl ToolCall {
    /// From the text between `<tool_call>` and `</tool_call>` in `format`
    fn parse_as(format: Format, text: &str) -> Option<Self> {
        match format {
            Format::ChatMl { .. } => Self::parse(text),
            Format::Flint { .. } => Self::parse_flint(text),
            Format::Inkling(_) => Self::parse_inkling("", text),
        }
    }

    /// Inkling's: `{"name": NAME, "args": {KEY: VALUE, …}}`, after a header
    /// whose text is the name too, or nothing. The JSON's name is the one;
    /// the arguments that are not strings are their JSON as the model wrote
    /// it (numbers as written), without the whitespace between its tokens.
    fn parse_inkling(header: &str, json: &str) -> Option<Self> {
        let json = json.trim();
        let j = rt::Json::parse(json).ok()?;
        let rt::Json::Object(_) = j else {
            return None;
        };
        let name = match j.get("name") {
            Some(n) => n.as_str()?.trim(),
            None => header.trim(),
        };
        if name.is_empty() {
            return None;
        }
        let arguments = match j.get("args") {
            None | Some(rt::Json::Null) => Vec::new(),
            Some(rt::Json::Object(args)) => {
                let text = json_members(json_member(&json_members(json), "args")?);
                args.iter()
                    .map(|(k, v)| {
                        let v = match v {
                            rt::Json::Str(s) => s.clone(),
                            _ => json_compact(json_member(&text, k)?),
                        };
                        Some((k.clone(), v))
                    })
                    .collect::<Option<_>>()?
            }
            Some(_) => return None,
        };
        Some(ToolCall {
            name: name.to_string(),
            arguments,
        })
    }

    /// From the text between `<tool_call>` and `</tool_call>`:
    /// `<function=NAME>`, then `<parameter=NAME>\nVALUE\n</parameter>` for each
    /// parameter, then `</function>`
    fn parse(text: &str) -> Option<Self> {
        let (name, mut rest) = text
            .trim_start()
            .strip_prefix("<function=")?
            .split_once('>')?;
        let mut arguments = Vec::new();
        loop {
            rest = rest.trim_start();
            let Some(param) = rest.strip_prefix("<parameter=") else {
                break;
            };
            let (param, after) = param.split_once('>')?;
            let (value, after) = after.split_once("</parameter>")?;
            let value = value.strip_prefix('\n').unwrap_or(value);
            let value = value.strip_suffix('\n').unwrap_or(value);
            arguments.push((param.trim().to_string(), value.to_string()));
            rest = after;
        }
        if rest.strip_prefix("</function>")?.trim().is_empty() {
            Some(ToolCall {
                name: name.trim().to_string(),
                arguments,
            })
        } else {
            None
        }
    }
}

impl ToolCall {
    /// Flint's: `NAME`, then `<arg_key>KEY</arg_key><arg_value>VALUE</arg_value>`
    /// for each argument
    fn parse_flint(text: &str) -> Option<Self> {
        let (name, mut rest) = text.split_at(text.find("<arg_key>").unwrap_or(text.len()));
        let name = name.trim();
        if name.is_empty() || name.contains('<') {
            return None;
        }
        let mut arguments = Vec::new();
        loop {
            rest = rest.trim_start();
            if rest.is_empty() {
                break;
            }
            let (key, after) = rest.strip_prefix("<arg_key>")?.split_once("</arg_key>")?;
            let value = after.trim_start().strip_prefix("<arg_value>")?;
            let (value, after) = value.split_once("</arg_value>")?;
            arguments.push((key.trim().to_string(), value.to_string()));
            rest = after;
        }
        Some(ToolCall {
            name: name.to_string(),
            arguments,
        })
    }
}

#[derive(Debug, Default, Clone)]
pub struct TurnStats {
    pub prompt_tokens: usize,
    pub gen_tokens: usize,
    /// stopped by max_tokens or the context size rather than by the model
    pub truncated: bool,
    /// tokens drafted by the MTP head or proposed from the conversation, and
    /// how many of them were right
    pub drafted: usize,
    pub accepted: usize,
    /// of those, proposed from the conversation, and right
    pub looked_up: usize,
    pub looked_up_accepted: usize,
    /// the function calls of the answer, as given to [`Event::ToolCall`]
    pub tool_calls: Vec<ToolCall>,
}

/// A conversation up to some point, see [`Chat::save`]
pub struct Prefix {
    snapshot: Snapshot,
    pending: Vec<TokenId>,
    generated: Vec<TokenId>,
    history: Vec<TokenId>,
    calls: Vec<String>,
}

impl Prefix {
    /// Tokens processed
    pub fn n_past(&self) -> usize {
        self.snapshot.n_past()
    }

    /// Memory held
    pub fn bytes(&self) -> usize {
        self.snapshot.bytes()
    }
}

/// The draft length of a chat's session: room to verify the MTP head's drafts
/// and the proposals from the conversation
fn session_draft(engine: &dyn Engine, opts: &ChatOptions) -> usize {
    opts.draft
        .unwrap_or_else(|| engine.draft_size())
        .max(opts.lookup)
}

pub struct Chat<'m> {
    engine: &'m dyn Engine,
    tok: &'m Tokenizer,
    pub opts: ChatOptions,
    session: Box<dyn Session + 'm>,
    /// generated but not yet processed tokens, fed before the next prompt
    pending: Vec<TokenId>,
    /// every generated token, for the sampler penalties
    generated: Vec<TokenId>,
    /// every token of the sequence, for the proposals from the conversation
    history: Vec<TokenId>,
    /// consecutive steps whose proposal from the conversation was all right
    lookup_run: usize,
    format: Format,
    /// the compiled chat template that renders the prompts, if any
    template: Option<compiled::Template>,
    /// ChatML's and Flint's reasoning end and call tokens, not Inkling's
    think_end: Option<TokenId>,
    tool_call: Option<TokenId>,
    tool_call_end: Option<TokenId>,
    /// the names of the last answer's function calls, which the results
    /// sent back are named by
    calls: Vec<String>,
}

/// Stand-ins for the messages before a turn, see [`Chat::template_turn`]:
/// text no message has, which the templates write as it is (private use
/// characters, not whitespace)
const STAND_IN_QUERY: &str = "\u{f8fe}";
const STAND_IN_ANSWER: &str = "\u{f8ff}";

/// A tool of [`ChatOptions::tools`] as the compiled templates take it
fn parse_tool(json: &str) -> Result<rt::Tool, Error> {
    let bad = |why: &str| Error::Template(format!("tool {json:?}: {why}"));
    let j = rt::Json::parse(json).map_err(|e| bad(&e.0))?;
    let f = j.get("function").unwrap_or(&j);
    let Some(name) = f.get("name").and_then(rt::Json::as_str) else {
        return Err(bad("no name"));
    };
    Ok(rt::Tool {
        name: name.into(),
        description: f
            .get("description")
            .and_then(rt::Json::as_str)
            .map(Into::into),
        parameters: f
            .get("parameters")
            .cloned()
            .unwrap_or(rt::Json::Object(Vec::new())),
    })
}

/// The tokenizer's parts of a rendered prompt
fn prompt_parts(s: &rt::Str) -> Vec<PromptPart<'_>> {
    s.0.iter()
        .map(|seg| {
            if seg.lit {
                Template(&seg.text)
            } else {
                Content(&seg.text)
            }
        })
        .collect()
}

/// What follows `mark`, data of `s`
fn after(s: &rt::Str, mark: &str) -> Option<rt::Str> {
    let i =
        s.0.iter()
            .position(|seg| !seg.lit && seg.text.contains(mark))?;
    let seg = &s.0[i];
    let at = seg.text.find(mark)? + mark.len();
    let mut rest = rt::Str::default();
    rest.push(&seg.text[at..], false);
    for seg in &s.0[i + 1..] {
        rest.push(&seg.text, seg.lit);
    }
    Some(rest)
}

/// The longest end of `a` that starts `b`
fn overlap(a: &[TokenId], b: &[TokenId]) -> usize {
    (0..=a.len().min(b.len()))
        .rev()
        .find(|&k| a[a.len() - k..] == b[..k])
        .unwrap_or(0)
}

impl<'m> Chat<'m> {
    pub fn new(
        engine: &'m dyn Engine,
        tok: &'m Tokenizer,
        opts: ChatOptions,
    ) -> Result<Self, Error> {
        let id = |s: &str| tok.token_id(s).ok_or_else(|| Error::MissingToken(s.into()));
        let has = |s: &str| tok.token_id(s).is_some();
        let format = if !has("<|im_start|>") && has("<|assistant|>") && has("<|observation|>") {
            Format::Flint {
                think: id("<think>")?,
            }
        } else if !has("<|im_start|>") && has("<|message_model|>") && has("<|end_message|>") {
            Format::Inkling(Inkling {
                model: id("<|message_model|>")?,
                user: id("<|message_user|>")?,
                system: id("<|message_system|>")?,
                tool: id("<|message_tool|>")?,
                text: id("<|content_text|>")?,
                thinking: id("<|content_thinking|>")?,
                call: id("<|content_invoke_tool_json|>")?,
                error: tok.token_id("<|content_tool_error|>"),
                call_text: tok.token_id("<|content_invoke_tool_text|>"),
                end: id("<|end_message|>")?,
                end_sampling: id("<|content_model_end_sampling|>")?,
            })
        } else {
            id("<|im_start|>")?;
            Format::ChatMl {
                im_end: id("<|im_end|>")?,
            }
        };
        let template = if opts.builtin_format {
            None
        } else {
            tok.chat_template().and_then(compiled::Template::recognize)
        };
        let session = engine.new_session(opts.context, session_draft(engine, &opts))?;
        let (think_end, tool_call, tool_call_end) = match format {
            Format::Inkling(_) => (None, None, None),
            _ => (
                Some(id("</think>")?),
                Some(id("<tool_call>")?),
                Some(id("</tool_call>")?),
            ),
        };
        Ok(Chat {
            engine,
            tok,
            session,
            opts,
            pending: Vec::new(),
            generated: Vec::new(),
            history: Vec::new(),
            lookup_run: 0,
            format,
            template,
            think_end,
            tool_call,
            tool_call_end,
            calls: Vec::new(),
        })
    }

    pub fn reset(&mut self) -> Result<(), Error> {
        // release the old state before allocating the new one
        self.session = Box::new(Empty);
        self.session = self
            .engine
            .new_session(self.opts.context, session_draft(self.engine, &self.opts))?;
        self.pending.clear();
        self.generated.clear();
        self.history.clear();
        self.lookup_run = 0;
        self.calls.clear();
        Ok(())
    }

    pub fn n_past(&self) -> usize {
        self.session.n_past()
    }

    /// The compiled chat template that renders the prompts, or `None` for the
    /// built-in formats: the model's template is not a compiled one, or
    /// [`ChatOptions::builtin_format`]
    pub fn template(&self) -> Option<compiled::Template> {
        self.template
    }

    /// Whether [`ChatOptions::thinking`] and [`ChatOptions::effort`] have an
    /// effect: Flint decides by itself when to reason, and how much
    pub fn controls_thinking(&self) -> bool {
        matches!(self.format, Format::ChatMl { .. } | Format::Inkling(_))
    }

    /// Whether [`ChatOptions::thinking`] and [`ChatOptions::effort`] are
    /// taken at the start of a conversation only, not per message: Inkling-2
    /// is told the effort once, before the first message (thinking off is
    /// the effort `none`), so changing them takes a new conversation
    pub fn thinking_per_conversation(&self) -> bool {
        matches!(self.format, Format::Inkling(_))
    }

    /// The memory of the conversation's session
    pub fn memory(&self) -> SessionMemory {
        self.session.memory()
    }

    /// The memory of the session a chat on `engine` with `opts` would make,
    /// without making it
    pub fn session_memory(engine: &dyn Engine, opts: &ChatOptions) -> SessionMemory {
        engine.session_memory(opts.context, session_draft(engine, opts))
    }

    /// Save the conversation so far, to [`Chat::restore`] it in this chat or
    /// another on the same engine. Before the first message, the system prompt
    /// is processed now, so that the prefix holds it.
    pub fn save(&mut self) -> Result<Prefix, Error> {
        if self.session.n_past() == 0 {
            let system = match (self.template, self.format) {
                (Some(t), _) => self.opening(t)?,
                (None, Format::Inkling(_)) => self.inkling_system()?,
                (None, _) => {
                    let mut parts = Vec::new();
                    let prompt = self.system_prompt();
                    self.system_block(prompt.as_deref(), &mut parts);
                    self.tok.encode_prompt(&parts)
                }
            };
            if system.len() >= self.opts.context {
                return Err(Error::ContextFull {
                    n_past: 0,
                    prompt: system.len(),
                    context: self.opts.context,
                });
            }
            self.prefill(&system);
            self.history.extend_from_slice(&system);
        }
        Ok(Prefix {
            snapshot: self.session.snapshot()?,
            pending: self.pending.clone(),
            generated: self.generated.clone(),
            history: self.history.clone(),
            calls: self.calls.clone(),
        })
    }

    /// Go back to a saved conversation. Its system prompt, reasoning
    /// instructions and tools are those of the chat that saved it, whatever
    /// `opts` says.
    pub fn restore(&mut self, prefix: &Prefix) -> Result<(), Error> {
        self.session.restore(&prefix.snapshot)?;
        self.pending.clone_from(&prefix.pending);
        self.generated.clone_from(&prefix.generated);
        self.history.clone_from(&prefix.history);
        self.calls.clone_from(&prefix.calls);
        self.lookup_run = 0;
        Ok(())
    }

    /// Tokens of the conversation so far: those processed, and the end of the
    /// last answer, processed with the next message
    pub fn n_tokens(&self) -> usize {
        self.history.len() + self.pending.len()
    }

    /// The conversation so far as token ids, to [`Chat::restore_transcript`]
    /// in any chat whose model has the same vocabulary. Nothing is processed:
    /// before the first message it is empty, unlike [`Chat::save`].
    pub fn transcript(&self) -> Transcript {
        let mut tokens = Vec::with_capacity(self.n_tokens());
        tokens.extend_from_slice(&self.history);
        tokens.extend_from_slice(&self.pending);
        Transcript {
            tokens,
            pending: self.pending.len(),
            generated: self.generated.clone(),
        }
    }

    /// Replace the conversation with `transcript`, processing its tokens
    /// again. Its system prompt, reasoning instructions and tools are those
    /// of the chat that made it, whatever `opts` says, as with
    /// [`Chat::restore`]. The answers that follow are those of the chat
    /// that made it up to rounding: its tokens were processed in other
    /// passes. Nothing changes when it doesn't fit.
    pub fn restore_transcript(&mut self, transcript: &Transcript) -> Result<(), Error> {
        let Transcript {
            tokens,
            pending,
            generated,
        } = transcript;
        if *pending > tokens.len() {
            return Err(Error::Transcript("more pending tokens than tokens"));
        }
        if *pending > 0 && *pending == tokens.len() {
            return Err(Error::Transcript("pending tokens without a conversation"));
        }
        let n_vocab = self.tok.n_vocab();
        if tokens
            .iter()
            .chain(generated)
            .any(|&t| t as usize >= n_vocab)
        {
            return Err(Error::Transcript("token id past the vocabulary"));
        }
        // the processed tokens and those of the next message's end
        if tokens.len() >= self.opts.context {
            return Err(Error::ContextFull {
                n_past: 0,
                prompt: tokens.len(),
                context: self.opts.context,
            });
        }
        if self.session.n_past() > 0 {
            self.reset()?;
        }
        let (processed, rest) = transcript.split();
        self.prefill(processed);
        self.history = processed.to_vec();
        self.pending = rest.to_vec();
        self.generated.clone_from(generated);
        self.lookup_run = 0;
        // not in a transcript: the results of its last calls go unnamed
        self.calls.clear();
        Ok(())
    }

    /// Tokens drafted by the MTP head per step
    fn draft(&self) -> usize {
        self.opts.draft.unwrap_or_else(|| self.engine.draft_size())
    }

    /// Append `tokens` to the sequence, in passes of the batch size
    fn prefill(&mut self, tokens: &[TokenId]) {
        let batch = self
            .opts
            .batch
            .unwrap_or_else(|| self.engine.batch_size())
            .max(1);
        for chunk in tokens.chunks(batch) {
            self.session.prefill(chunk);
        }
    }

    /// The parameters of the compiled templates
    fn template_params<'p>(
        &self,
        messages: &'p [rt::Message],
        tools: &'p [rt::Tool],
        generation: bool,
    ) -> compiled::Params<'p> {
        compiled::Params {
            messages,
            tools,
            add_generation_prompt: generation,
            enable_thinking: Some(self.opts.thinking),
            // the history keeps past reasoning, see the module
            preserve_thinking: Some(true),
            // Inkling-2's template has no switch but the effort: none is
            // thinking off, as llama.cpp has it
            reasoning_effort: Some(match self.format {
                Format::Inkling(_) if !self.opts.thinking => "none",
                _ => self.opts.effort.as_str(),
            }),
            add_vision_id: false,
        }
    }

    /// `messages` after the system prompt, rendered by the compiled template `t`
    fn render(
        &self,
        t: compiled::Template,
        messages: &[rt::Message],
        generation: bool,
    ) -> Result<rt::Str, Error> {
        let tools = self
            .opts
            .tools
            .iter()
            .map(|t| parse_tool(t))
            .collect::<Result<Vec<_>, _>>()?;
        let mut all = Vec::with_capacity(messages.len() + 1);
        if let Some(system) = self.opts.system.as_deref().filter(|s| !s.trim().is_empty()) {
            all.push(rt::Message::new(rt::Role::System, system));
        }
        all.extend_from_slice(messages);
        t.render(&self.template_params(&all, &tools, generation))
            .map_err(|e| Error::Template(e.0))
    }

    fn encode(&self, s: &rt::Str) -> Vec<TokenId> {
        self.tok.encode_prompt(&prompt_parts(s))
    }

    /// What the compiled template `t` writes before the first message, as far
    /// as [`Chat::save`] processes it: up to the last special token before
    /// the message's text (`<|im_start|>` of `<|im_start|>user\n`, `<|user|>`),
    /// where conversations with another first message part
    fn opening(&self, t: compiled::Template) -> Result<Vec<TokenId>, Error> {
        let first = |q: &str| -> Result<Vec<TokenId>, Error> {
            let query = [rt::Message::new(rt::Role::User, q)];
            Ok(self.encode(&self.render(t, &query, false)?))
        };
        let (mut a, b) = (first(STAND_IN_QUERY)?, first(STAND_IN_ANSWER)?);
        let common = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
        let mut cut = a[..common]
            .iter()
            .rposition(|&id| self.tok.is_special(id))
            .unwrap_or(0);
        // Inkling's message starts with two: `<|message_user|><|content_text|>`
        if let Format::Inkling(f) = self.format
            && cut > 0
            && a[cut - 1] == f.user
        {
            cut -= 1;
        }
        a.truncate(cut);
        Ok(a)
    }

    /// The tokens of the next turn with the compiled template `t`: what is left
    /// of the previous answer, `messages` and the generation prompt.
    ///
    /// Before the first answer, the conversation is rendered whole, less what
    /// [`Chat::save`] processed of it. After it, the answers are in the history
    /// as the model generated them, and the template renders `messages` after
    /// stand-ins for the conversation so far: a query, then an answer whose
    /// text marks where the turn starts, at the template's end of the answer.
    /// The end token the model generated ([`Chat::pending`]) starts it already.
    fn template_turn(
        &self,
        t: compiled::Template,
        messages: &[rt::Message],
    ) -> Result<Vec<TokenId>, Error> {
        if self.generated.is_empty() {
            let tokens = self.encode(&self.render(t, messages, true)?);
            if self.session.n_past() == 0 {
                return Ok(tokens);
            }
            let opening = self.opening(t)?;
            return match tokens.strip_prefix(opening.as_slice()) {
                Some(rest) => Ok(rest.to_vec()),
                None => Err(Error::Template(
                    "the first message changes what comes before it".into(),
                )),
            };
        }
        let mut all = vec![
            rt::Message::new(rt::Role::User, STAND_IN_QUERY),
            rt::Message::new(rt::Role::Assistant, STAND_IN_ANSWER),
        ];
        all.extend_from_slice(messages);
        let rendered = self.render(t, &all, true)?;
        let Some(rest) = after(&rendered, STAND_IN_ANSWER) else {
            return Err(Error::Template("an answer's text is not written".into()));
        };
        let tokens = self.encode(&rest);
        let k = match self.format {
            // the template's end of the answer is its last block's end, then
            // the end of sampling, which ends what the answer left
            // ([`InklingReader::close`]) however it ended: in a block, after
            // one, after text outside any
            Format::Inkling(f) => match tokens.iter().position(|&t| t == f.end_sampling) {
                Some(i) => i + 1,
                None => {
                    return Err(Error::Template("an answer's end is not written".into()));
                }
            },
            _ => overlap(&self.pending, &tokens),
        };
        let mut out = self.pending.clone();
        out.extend_from_slice(&tokens[k..]);
        Ok(out)
    }

    /// The system prompt as the chat template writes it, if any
    fn system_prompt(&self) -> Option<Cow<'_, str>> {
        let system = self.opts.system.as_deref()?;
        let system = match self.format {
            Format::ChatMl { .. } => Cow::Borrowed(system.trim()),
            Format::Flint { .. } => flint_text(system),
            // as it is, unless blank
            Format::Inkling(_) if system.trim().is_empty() => return None,
            Format::Inkling(_) => Cow::Borrowed(system),
        };
        (!system.is_empty()).then_some(system)
    }

    /// Inkling's effort level, as its template writes the levels'
    fn inkling_effort(&self) -> &'static str {
        if !self.opts.thinking {
            return "0";
        }
        match self.opts.effort {
            ReasoningEffort::Low => "0.2",
            ReasoningEffort::Medium => "0.7",
            ReasoningEffort::High => "0.9",
            ReasoningEffort::XHigh => "0.99",
        }
    }

    /// Inkling's system blocks, as its template writes them before the first
    /// message: the tools (as JSON, compact, the keys sorted), the system
    /// prompt, the effort level
    fn inkling_system(&self) -> Result<Vec<TokenId>, Error> {
        let mut tools = None;
        if !self.opts.tools.is_empty() {
            let specs = self
                .opts
                .tools
                .iter()
                .map(|t| {
                    let t = parse_tool(t)?;
                    let parameters = rt::json(&t.parameters);
                    Ok(rt::Value::map(vec![
                        (
                            "description",
                            rt::Value::data(t.description.as_deref().unwrap_or("")),
                        ),
                        ("name", rt::Value::data(&t.name)),
                        (
                            "parameters",
                            if rt::truthy(&parameters) {
                                parameters
                            } else {
                                rt::Value::map(Vec::new())
                            },
                        ),
                        ("type", rt::Value::lit("function")),
                    ]))
                })
                .collect::<Result<Vec<_>, Error>>()?;
            let u = rt::Value::Undefined;
            let json = rt::f_tojson_sep(
                &rt::Value::list(specs),
                &u,
                &u,
                &rt::Value::Bool(true),
                ",",
                ":",
            )
            .map_err(|e| Error::Template(e.0))?;
            tools = Some(rt::to_str(&json).text());
        }
        let system = self.system_prompt();
        let mut parts = Vec::new();
        if let Some(t) = &tools {
            parts.extend([
                Template("<|message_system|>tool_declare<|content_xml|>"),
                Content(t),
                Template("<|end_message|>"),
            ]);
        }
        if let Some(s) = &system {
            parts.extend([
                Template("<|message_system|><|content_text|>"),
                Content(s),
                Template("<|end_message|>"),
            ]);
        }
        parts.extend([
            Template("<|message_system|><|content_text|>Thinking effort level: "),
            Content(self.inkling_effort()),
            Template("<|end_message|>"),
        ]);
        Ok(self.tok.encode_prompt(&parts))
    }

    /// The tokens of the next turn in Inkling's built-in format: what is left
    /// of the previous answer, the system blocks on the first turn, `message`.
    /// No generation prompt: the model writes its header itself.
    fn inkling_turn(&self, message: &[PromptPart<'_>]) -> Result<Vec<TokenId>, Error> {
        let mut t = self.pending.clone();
        if self.session.n_past() == 0 {
            t.extend(self.inkling_system()?);
        }
        t.extend(self.tok.encode_prompt(message));
        Ok(t)
    }

    /// The system block, as the chat template writes it, with the
    /// [`Chat::system_prompt`]: the reasoning instructions, the tools, the
    /// system prompt; nothing without any of them. Flint's has the system
    /// prompt, then the tools in a block of their own.
    fn system_block<'s>(&'s self, system: Option<&'s str>, parts: &mut Vec<PromptPart<'s>>) {
        if let Format::Flint { .. } = self.format {
            if let Some(s) = system {
                parts.extend([Template("<|system|>"), Content(s)]);
            }
            if !self.opts.tools.is_empty() {
                parts.push(Template(FLINT_TOOLS_HEADER));
                for tool in &self.opts.tools {
                    parts.push(Content(tool_function(tool.trim())));
                }
                parts.push(Template(FLINT_TOOLS_FOOTER));
            }
            return;
        }
        let reasoning = if self.opts.thinking {
            self.opts.effort.instructions()
        } else {
            None
        };
        let tools = !self.opts.tools.is_empty();
        if reasoning.is_none() && !tools && system.is_none() {
            return;
        }
        parts.push(Template("<|im_start|>system\n"));
        if let Some(r) = reasoning {
            parts.push(Template(r));
            if tools || system.is_some() {
                parts.push(Template("\n\n"));
            }
        }
        if tools {
            parts.push(Template(TOOLS_HEADER));
            for tool in &self.opts.tools {
                parts.extend([Template("\n"), Content(tool.trim())]);
            }
            parts.push(Template(TOOLS_FOOTER));
            if system.is_some() {
                parts.push(Template("\n\n"));
            }
        }
        if let Some(s) = system {
            parts.push(Content(s));
        }
        parts.push(Template("<|im_end|>\n"));
    }

    /// The tokens of the next turn in the built-in format: what is left of the
    /// previous answer, the system block on the first turn, `message` and the
    /// generation prompt
    fn turn_tokens(&self, message: &[PromptPart<'_>]) -> Vec<TokenId> {
        let chatml = matches!(self.format, Format::ChatMl { .. });
        let system = self.system_prompt();
        let mut parts = Vec::new();
        if chatml && !self.pending.is_empty() {
            parts.push(Template("\n"));
        }
        if self.session.n_past() == 0 {
            self.system_block(system.as_deref(), &mut parts);
        }
        parts.extend_from_slice(message);
        if chatml {
            parts.push(Template("<|im_start|>assistant\n"));
            parts.push(Template(if self.opts.thinking {
                "<think>\n"
            } else {
                "<think>\n\n</think>\n\n"
            }));
        } else {
            parts.push(Template("<|assistant|>"));
        }
        let mut t = self.pending.clone();
        t.extend(self.tok.encode_prompt(&parts));
        t
    }

    /// Send a user message and stream the answer through `on_event`.
    /// `stop` is polled between tokens to interrupt the generation.
    pub fn send(
        &mut self,
        user: &str,
        sampler: &mut Sampler,
        on_event: impl FnMut(Event<'_>),
        stop: impl Fn() -> bool,
    ) -> Result<TurnStats, Error> {
        let prompt = match (self.template, self.format) {
            (Some(t), _) => self.template_turn(t, &[rt::Message::new(rt::Role::User, user)])?,
            (None, Format::ChatMl { .. }) => self.turn_tokens(&[
                Template("<|im_start|>user\n"),
                Content(user.trim()),
                Template("<|im_end|>\n"),
            ]),
            (None, Format::Flint { .. }) => {
                self.turn_tokens(&[Template("<|user|>"), Content(&flint_text(user))])
            }
            (None, Format::Inkling(_)) => self.inkling_turn(&[
                Template("<|message_user|><|content_text|>"),
                Content(user),
                Template("<|end_message|>"),
            ])?,
        };
        self.answer(prompt, sampler, on_event, stop)
    }

    /// Send the results of the function calls of the last answer, in their
    /// order, and stream the answer as [`Chat::send`] does. Each is named by
    /// its call's function, for templates that write it, when there are as
    /// many as calls.
    pub fn send_tool_responses(
        &mut self,
        responses: &[impl AsRef<str>],
        sampler: &mut Sampler,
        on_event: impl FnMut(Event<'_>),
        stop: impl Fn() -> bool,
    ) -> Result<TurnStats, Error> {
        let names: &[String] = if self.calls.len() == responses.len() {
            &self.calls
        } else {
            &[]
        };
        if let Some(t) = self.template {
            let messages: Vec<rt::Message> = responses
                .iter()
                .enumerate()
                .map(|(i, r)| rt::Message {
                    name: names.get(i).cloned(),
                    ..rt::Message::new(rt::Role::Tool, r.as_ref())
                })
                .collect();
            let prompt = self.template_turn(t, &messages)?;
            return self.answer(prompt, sampler, on_event, stop);
        }
        if let Format::Inkling(_) = self.format {
            let mut message = Vec::new();
            for (i, r) in responses.iter().enumerate() {
                message.push(Template("<|message_tool|>"));
                if let Some(name) = names.get(i) {
                    message.push(Content(name));
                }
                message.extend([
                    Template("<|content_text|>"),
                    Content(r.as_ref()),
                    Template("<|end_message|>"),
                ]);
            }
            let prompt = self.inkling_turn(&message)?;
            return self.answer(prompt, sampler, on_event, stop);
        }
        let flint: Vec<Cow<'_, str>> = match self.format {
            Format::ChatMl { .. } | Format::Inkling(_) => Vec::new(),
            Format::Flint { .. } => responses.iter().map(|r| flint_text(r.as_ref())).collect(),
        };
        let mut message = Vec::new();
        if let Format::ChatMl { .. } = self.format {
            message.push(Template("<|im_start|>user"));
            for r in responses {
                message.extend([
                    Template("\n<tool_response>\n"),
                    Content(r.as_ref().trim()),
                    Template("\n</tool_response>"),
                ]);
            }
            message.push(Template("<|im_end|>\n"));
        } else {
            for r in &flint {
                message.extend([
                    Template("<|observation|><tool_response>"),
                    Content(r),
                    Template("</tool_response>"),
                ]);
            }
        }
        let prompt = self.turn_tokens(&message);
        self.answer(prompt, sampler, on_event, stop)
    }

    /// Process `prompt` and generate the answer
    fn answer(
        &mut self,
        prompt: Vec<TokenId>,
        sampler: &mut Sampler,
        mut on_event: impl FnMut(Event<'_>),
        stop: impl Fn() -> bool,
    ) -> Result<TurnStats, Error> {
        if self.session.n_past() + prompt.len() + 1 >= self.opts.context {
            return Err(Error::ContextFull {
                n_past: self.session.n_past(),
                prompt: prompt.len(),
                context: self.opts.context,
            });
        }
        self.pending.clear();

        let mut stats = TurnStats {
            prompt_tokens: prompt.len(),
            ..Default::default()
        };
        self.history.extend_from_slice(&prompt);
        let (body, last) = prompt.split_at(prompt.len() - 1);
        self.prefill(body);
        let mut logits = self.session.forward(last);
        on_event(Event::PromptDone);

        let (mut thinking, think) = match self.format {
            Format::ChatMl { .. } => (self.opts.thinking, None),
            // the model opens its reasoning block itself
            Format::Flint { think } => (false, Some(think)),
            // all its blocks are the reader's
            Format::Inkling(_) => (false, None),
        };
        let mut inkling = match self.format {
            Format::Inkling(f) => Some(InklingReader::new(f)),
            _ => None,
        };
        let mut decoder = StreamDecoder::default();
        // the template puts "\n\n" after </think>: don't echo it
        let mut skip_leading_newlines = !thinking;
        // newlines at the end of the text so far, shown only if more follows
        let mut held_newlines = String::new();
        // the text of the function call being generated
        let mut call: Option<String> = None;
        let mut spec = Speculation::default();
        loop {
            let next = self.choose(&mut logits, &spec, sampler);
            stats.gen_tokens += 1;
            self.generated.push(next);

            if let Some(r) = &mut inkling
                && (self.is_end(next) || r.ends(next))
            {
                // the end of sampling, never `<|endoftext|>`, closes the
                // answer in the history, after the open block's end
                let closing = r.close(&mut stats.tool_calls, &mut on_event);
                self.pending.extend(closing);
                break;
            }
            if self.is_end(next) {
                if let Some(text) = call.take() {
                    on_event(Event::Answer(&format!("<tool_call>{text}")));
                }
                // the end marker closes the message in the history; Flint's
                // is the next role's token, which the next turn writes
                if let Format::ChatMl { im_end } = self.format {
                    self.pending.push(im_end);
                }
                break;
            }
            if let Some(r) = &mut inkling {
                r.token(next, self.tok, &mut stats.tool_calls, &mut on_event);
            } else if Some(next) == think && !thinking && call.is_none() {
                thinking = true;
                held_newlines.clear();
                skip_leading_newlines = true;
            } else if Some(next) == self.think_end && thinking {
                thinking = false;
                skip_leading_newlines = true;
                held_newlines.clear();
                on_event(Event::ThinkingDone);
            } else if Some(next) == self.tool_call && !thinking && call.is_none() {
                // the newlines before a call are not part of the answer
                held_newlines.clear();
                call = Some(String::new());
            } else if Some(next) == self.tool_call_end && call.is_some() {
                let text = call.take().unwrap_or_default();
                match ToolCall::parse_as(self.format, &text) {
                    Some(c) => {
                        on_event(Event::ToolCall(&c));
                        stats.tool_calls.push(c);
                    }
                    // not a call after all: show it as it was written
                    None => on_event(Event::Answer(&format!("<tool_call>{text}</tool_call>"))),
                }
                skip_leading_newlines = true;
            } else {
                let mut text = decoder.push(self.tok.token_bytes(next));
                if let Some(c) = &mut call {
                    c.push_str(&text);
                } else {
                    if skip_leading_newlines {
                        let trimmed = text.trim_start_matches('\n');
                        if !trimmed.is_empty() {
                            skip_leading_newlines = false;
                        }
                        text = trimmed.to_string();
                    }
                    let body = text.trim_end_matches('\n');
                    if !body.is_empty() {
                        let shown = core::mem::take(&mut held_newlines) + body;
                        on_event(if thinking {
                            Event::Thinking(&shown)
                        } else {
                            Event::Answer(&shown)
                        });
                    }
                    held_newlines.push_str(&text[body.len()..]);
                }
            }

            let full = self.session.n_past() - spec.drafts.len() + 2 >= self.opts.context;
            if stats.gen_tokens >= self.opts.max_tokens || full || stop() {
                stats.truncated = true;
                // close the message so the history stays well-formed
                self.pending.push(next);
                if let Some(r) = &mut inkling {
                    let closing = r.close(&mut stats.tool_calls, &mut on_event);
                    self.pending.extend(closing);
                    break;
                }
                if thinking {
                    self.pending.extend(self.think_end);
                }
                if let Some(text) = call.take() {
                    on_event(Event::Answer(&format!("<tool_call>{text}")));
                    self.pending.extend(self.tool_call_end);
                }
                if let Format::ChatMl { im_end } = self.format {
                    self.pending.push(im_end);
                }
                break;
            }
            logits = self.advance(next, &mut spec, &mut stats, sampler);
        }
        self.settle(&mut spec);
        self.calls = stats.tool_calls.iter().map(|c| c.name.clone()).collect();
        Ok(stats)
    }

    /// Whether `token` ends an answer
    fn is_end(&self, token: TokenId) -> bool {
        matches!(self.format, Format::ChatMl { im_end } if token == im_end)
            || self.tok.eog.contains(&token)
    }

    /// The token after the sequence, from `logits`, the logits after it. With
    /// a draft drawn from a distribution q (the MTP head's, while sampling),
    /// the draft is taken with a chance of p/q, else a token is drawn from
    /// max(p - q, 0) (speculative sampling): either way the token follows the
    /// sampler's distribution p. A draft of the likeliest token, or one from
    /// the conversation, is right when the sampled token is the same.
    fn choose(&self, logits: &mut [f32], spec: &Speculation, sampler: &mut Sampler) -> TokenId {
        let (Some(&draft), Some(Some(q))) = (spec.drafts.front(), spec.q.front()) else {
            return sampler.sample(logits, &self.generated);
        };
        let p = normalized(sampler.distribution(logits, &self.generated));
        speculative_choice(&p, q, draft, sampler)
    }

    /// The logits after `next`: those of the last verify if `next` is its
    /// next draft, else from a new draft and verify
    fn advance(
        &mut self,
        next: TokenId,
        spec: &mut Speculation,
        stats: &mut TurnStats,
        sampler: &mut Sampler,
    ) -> Vec<f32> {
        self.history.push(next);
        if let Some(draft) = spec.drafts.pop_front() {
            spec.q.pop_front();
            let logits = spec.logits.pop_front().expect("logits for every draft");
            if draft == next {
                spec.kept += 1;
                stats.accepted += 1;
                if spec.looked_up {
                    stats.looked_up_accepted += 1;
                }
                return logits;
            }
        }
        self.settle(spec);

        // room for the drafts in the context
        let room = self.opts.context.saturating_sub(self.session.n_past() + 2);
        let (drafts, q, looked_up) = self.propose(next, room, sampler);
        if drafts.is_empty() {
            return self.session.forward(&[next]);
        }
        stats.drafted += drafts.len();
        if looked_up {
            stats.looked_up += drafts.len();
        }
        let tokens: Vec<TokenId> = core::iter::once(next)
            .chain(drafts.iter().copied())
            .collect();
        let mut logits = VecDeque::from(self.session.verify(&tokens));
        let first = logits.pop_front().expect("logits for every token");
        *spec = Speculation {
            proposed: drafts.len(),
            drafts: drafts.into(),
            q: q.into(),
            logits,
            kept: 1,
            looked_up,
        };
        first
    }

    /// The tokens to verify after `next`, at most `room`, the distributions
    /// they were drawn from (none for the likeliest token or one from the
    /// conversation), and whether they come from the conversation rather than
    /// the MTP head. A continuation found
    /// after a long match is taken alone; after a short one, only if the MTP
    /// head drafts its first tokens too. It is cut to `LOOKUP_SHORT` tokens
    /// unless the last steps' continuations were all right: a copy is running.
    fn propose(
        &mut self,
        next: TokenId,
        room: usize,
        sampler: &mut Sampler,
    ) -> (Vec<TokenId>, Vec<Option<Dist>>, bool) {
        let mut found = if self.opts.lookup > 0 {
            continuation(&self.history, self.opts.lookup)
        } else {
            None
        };
        let long = if self.lookup_run >= LOOKUP_RUN {
            self.opts.lookup
        } else {
            self.opts.lookup.min(LOOKUP_SHORT)
        };
        if let Some((_, c)) = &mut found {
            c.truncate(long.min(room));
        }
        if let Some((m, c)) = &found
            && *m >= LOOKUP_STRONG
            && !c.is_empty()
        {
            let c = found.unwrap().1;
            return (c, Vec::new(), true);
        }
        let mut q = Vec::new();
        let draft = self.draft();
        let drafts = if draft > 0 {
            let greedy = sampler.greedy();
            self.session.draft(next, draft.min(room), &mut |l| {
                if greedy {
                    return l.argmax();
                }
                let d = normalized(sampler.distribution_of(l.candidates()));
                let t = sampler.draw(&d);
                q.push(Some(d));
                t
            })
        } else {
            Vec::new()
        };
        match found {
            Some((_, c))
                if c.len() >= LOOKUP_AGREE
                    && drafts.len() >= LOOKUP_AGREE
                    && c[..LOOKUP_AGREE] == drafts[..LOOKUP_AGREE] =>
            {
                (c, Vec::new(), true)
            }
            _ => (drafts, q, false),
        }
    }

    /// Keep the verified tokens of the last draft that matched what was sampled
    fn settle(&mut self, spec: &mut Speculation) {
        if spec.kept > 0 {
            self.session.keep(spec.kept);
        }
        if spec.looked_up && spec.kept == spec.proposed + 1 {
            self.lookup_run += 1;
        } else if spec.kept > 0 {
            self.lookup_run = 0;
        }
        *spec = Speculation::default();
    }
}

/// (token, probability) pairs
type Dist = Vec<(TokenId, f32)>;

/// `draft`, drawn from `q`, with a chance of p/q, else a token drawn from
/// max(p - q, 0): a token drawn from `p` (both normalized)
fn speculative_choice(
    p: &[(TokenId, f32)],
    q: &[(TokenId, f32)],
    draft: TokenId,
    sampler: &mut Sampler,
) -> TokenId {
    let prob = |d: &[(TokenId, f32)], t| d.iter().find(|c| c.0 == t).map_or(0.0, |c| c.1);
    if sampler.uniform() * prob(q, draft) < prob(p, draft) {
        return draft;
    }
    let residual: Dist = p
        .iter()
        .map(|&(t, pt)| (t, (pt - prob(q, t)).max(0.0)))
        .filter(|c| c.1 > 0.0)
        .collect();
    if residual.is_empty() {
        sampler.draw(p)
    } else {
        sampler.draw(&residual)
    }
}

/// `d` scaled to a sum of 1
fn normalized(mut d: Dist) -> Dist {
    let sum: f32 = d.iter().map(|c| c.1).sum();
    for c in &mut d {
        c.1 /= sum;
    }
    d
}

/// A match of at least this many tokens is proposed without the MTP head's
/// agreement, and one of fewer is not proposed at all (`LOOKUP_MIN`)
const LOOKUP_STRONG: usize = 8;
const LOOKUP_MIN: usize = 3;
/// Leading tokens a continuation and the MTP head's drafts must share to
/// propose the continuation after a short match
const LOOKUP_AGREE: usize = 2;
/// Tokens proposed from the conversation per step, unless the last
/// `LOOKUP_RUN` steps' proposals were all right
const LOOKUP_SHORT: usize = 7;
const LOOKUP_RUN: usize = 2;
/// Longest match looked for
const LOOKUP_MAX_MATCH: usize = 32;

/// What followed the latest earlier occurrence of the longest suffix of
/// `history` (at least `LOOKUP_MIN` tokens long), up to `n` tokens, and the
/// suffix length. The occurrence may overlap the suffix: a repeating pattern
/// continues with its own period.
fn continuation(history: &[TokenId], n: usize) -> Option<(usize, Vec<TokenId>)> {
    let len = history.len();
    let last = *history.last()?;
    let (mut best, mut at) = (0, 0);
    for j in (0..len - 1).rev() {
        if history[j] != last {
            continue;
        }
        let m = (1..LOOKUP_MAX_MATCH.min(j + 1))
            .take_while(|&k| history[j - k] == history[len - 1 - k])
            .count()
            + 1;
        if m > best {
            (best, at) = (m, j);
            if m == LOOKUP_MAX_MATCH {
                break;
            }
        }
    }
    if best < LOOKUP_MIN {
        return None;
    }
    // history followed by the continuation, where position len + i repeats j + 1 + i
    let mut c: Vec<TokenId> = Vec::with_capacity(n);
    for i in 0..n {
        let p = at + 1 + i;
        c.push(if p < len { history[p] } else { c[p - len] });
    }
    Some((best, c))
}

/// The last verify: its drafts not yet compared with the sampled tokens, the
/// logits after each, and its tokens kept so far
#[derive(Default)]
struct Speculation {
    drafts: VecDeque<TokenId>,
    /// the distribution each draft was drawn from, if not a single token
    q: VecDeque<Option<Dist>>,
    logits: VecDeque<Vec<f32>>,
    kept: usize,
    /// drafts of the verify, and whether they come from the conversation
    proposed: usize,
    looked_up: bool,
}

/// Placeholder session while a new one is being created
struct Empty;

impl Session for Empty {
    fn forward(&mut self, _: &[TokenId]) -> Vec<f32> {
        unreachable!()
    }
    fn prefill(&mut self, _: &[TokenId]) {
        unreachable!()
    }
    fn draft(
        &mut self,
        _: TokenId,
        _: usize,
        _: &mut dyn FnMut(&DraftLogits<'_>) -> TokenId,
    ) -> Vec<TokenId> {
        unreachable!()
    }
    fn verify(&mut self, _: &[TokenId]) -> Vec<Vec<f32>> {
        unreachable!()
    }
    fn keep(&mut self, _: usize) {
        unreachable!()
    }
    fn n_past(&self) -> usize {
        0
    }
    fn memory(&self) -> SessionMemory {
        SessionMemory::default()
    }
    fn snapshot(&self) -> Result<Snapshot, Error> {
        unreachable!()
    }
    fn restore(&mut self, _: &Snapshot) -> Result<(), Error> {
        unreachable!()
    }
}

#[cfg(test)]
mod tests {
    use super::{Cow, ToolCall, continuation, flint_text, speculative_choice, tool_function};
    use crate::prelude::*;
    use crate::sampler::{Sampler, SamplerConfig};

    /// `high` is `xhigh` for Qwen, as its templates resolve it
    #[test]
    fn high_effort_is_qwens_xhigh() {
        use super::{ReasoningEffort, compiled, rt};
        assert_eq!(
            ReasoningEffort::High.instructions(),
            ReasoningEffort::XHigh.instructions()
        );
        let messages = [rt::Message::new(rt::Role::User, "Q")];
        for t in [
            compiled::Template::Qwen35,
            compiled::Template::Qwen36,
            compiled::Template::Qwen38,
        ] {
            let render = |effort: ReasoningEffort| {
                let p = compiled::Params {
                    messages: &messages,
                    add_generation_prompt: true,
                    enable_thinking: Some(true),
                    reasoning_effort: Some(effort.as_str()),
                    ..Default::default()
                };
                t.render(&p).unwrap().text()
            };
            let high = render(ReasoningEffort::parse("high").unwrap());
            assert_eq!(high, render(ReasoningEffort::XHigh), "{t:?}");
        }
    }

    #[test]
    fn flint_text_escapes_roles() {
        assert!(matches!(flint_text("plain"), Cow::Borrowed("plain")));
        assert_eq!(
            flint_text("a\0<think>b</think> <|user|><|observation|>"),
            "ab &lt;|user|&gt;&lt;|observation|&gt;"
        );
        // one pass per replacement, in the template's order
        assert_eq!(flint_text("<th<think>ink>"), "<think>");
        assert_eq!(flint_text("<|us\0er|>"), "&lt;|user|&gt;");
    }

    #[test]
    fn tool_function_takes_the_inner_object() {
        let f = r#"{"name": "a", "parameters": {"x": [1, {"y": "}"}]}}"#;
        assert_eq!(
            tool_function(&format!(r#"{{"type": "function", "function": {f}}}"#)),
            f
        );
        assert_eq!(
            tool_function(&format!(r#"{{"function":{f},"type":"function"}}"#)),
            f
        );
        // a "function" key deeper down, or in a string, is not the tool's
        let t = r#"{"name": "f", "parameters": {"function": {}}, "d": "\"function\": 1"}"#;
        assert_eq!(tool_function(t), t);
    }

    #[test]
    fn flint_tool_call_parses() {
        let c = ToolCall::parse_flint("get_weather <arg_key>city</arg_key><arg_value>Paris</arg_value>\n<arg_key> days </arg_key> <arg_value>[1, 2]</arg_value>").unwrap();
        assert_eq!(c.name, "get_weather");
        assert_eq!(
            c.arguments,
            [
                ("city".into(), "Paris".into()),
                ("days".into(), "[1, 2]".into())
            ]
        );
        assert_eq!(ToolCall::parse_flint("now").unwrap().arguments, []);
        assert!(ToolCall::parse_flint("").is_none());
        assert!(ToolCall::parse_flint("f<arg_key>x</arg_key>").is_none());
        assert!(
            ToolCall::parse_flint("f<arg_key>x</arg_key><arg_value>1</arg_value> junk").is_none()
        );
    }

    #[test]
    fn speculative_choice_follows_the_target() {
        // q overlaps p partly, puts mass on a token p excludes, misses one p has
        let p = [(0, 0.5), (1, 0.3), (2, 0.2)];
        let q = [(0, 0.2), (1, 0.6), (3, 0.2)];
        let mut s = Sampler::new(SamplerConfig {
            seed: 7,
            ..Default::default()
        });
        let n = 400_000;
        let mut count = [0usize; 4];
        let mut accepted = 0;
        for _ in 0..n {
            let d = s.draw(&q);
            let t = speculative_choice(&p, &q, d, &mut s);
            accepted += (t == d) as usize;
            count[t as usize] += 1;
        }
        for &(t, pt) in p.iter() {
            let f = count[t as usize] as f64 / n as f64;
            assert!((f - pt as f64).abs() < 0.005, "token {t}: {f} vs {pt}");
        }
        assert_eq!(count[3], 0);
        // the acceptance is the overlap: min(p, q) summed = 0.2 + 0.3
        let a = accepted as f64 / n as f64;
        assert!((a - 0.5).abs() < 0.005, "acceptance {a}");
    }

    #[test]
    fn continuation_of_the_longest_latest_match() {
        // "1 2 3 4 5" then "9 2 3 4 5 6", now "... 3 4 5": the longer match wins
        let h = [7, 1, 2, 3, 4, 5, 8, 9, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5];
        assert_eq!(continuation(&h, 2), Some((5, vec![8, 9])));
        // too short a match
        assert_eq!(continuation(&[1, 2, 3, 9, 2, 3], 2), None);
        assert_eq!(continuation(&[5], 2), None);
    }

    #[test]
    fn continuation_overlapping_its_match() {
        // a period of 2: "a b a b a" continues "b a b"
        let h = [0, 1, 0, 1, 0];
        assert_eq!(continuation(&h, 3), Some((3, vec![1, 0, 1])));
        // a run of one token
        assert_eq!(continuation(&[4, 4, 4, 4], 3), Some((3, vec![4, 4, 4])));
    }
}

#[cfg(test)]
mod inkling_tests {
    //! Inkling-2's chat on a scripted engine: the model's answers are given,
    //! and the sequence the chat processes is checked against the reference,
    //! Hugging Face `tokenizers`' ids of the Jinja2 render (transformers'
    //! settings) of the whole conversation

    use core::cell::RefCell;
    use std::sync::OnceLock;

    use super::*;
    use crate::memory::ModelMemory;
    use crate::sampler::SamplerConfig;

    fn tok() -> Option<&'static Tokenizer> {
        static TOK: OnceLock<Option<Tokenizer>> = OnceLock::new();
        TOK.get_or_init(crate::tokenizer::tests::inkling2).as_ref()
    }

    /// An engine whose model generates the tokens of `script`, in order,
    /// whatever it is given, and whose session's sequence is `seq`
    struct Scripted {
        n_vocab: usize,
        script: RefCell<VecDeque<TokenId>>,
        seq: RefCell<Vec<TokenId>>,
    }

    impl Scripted {
        fn new(tok: &Tokenizer, answers: &[&str]) -> Self {
            Scripted {
                n_vocab: tok.n_vocab(),
                script: RefCell::new(
                    answers
                        .iter()
                        .flat_map(|a| tok.encode_with_special(a))
                        .collect(),
                ),
                seq: RefCell::new(Vec::new()),
            }
        }
    }

    impl Engine for Scripted {
        fn new_session(&self, _: usize, _: usize) -> Result<Box<dyn Session + '_>, Error> {
            self.seq.borrow_mut().clear();
            Ok(Box::new(ScriptedSession(self)))
        }
        fn batch_size(&self) -> usize {
            // passes that don't end where the blocks do
            7
        }
        fn draft_size(&self) -> usize {
            0
        }
        fn description(&self) -> String {
            "scripted".into()
        }
        fn model_memory(&self) -> ModelMemory {
            ModelMemory::default()
        }
        fn session_memory(&self, _: usize, _: usize) -> SessionMemory {
            SessionMemory::default()
        }
        fn numerics(&self) -> Vec<(&'static str, String)> {
            Vec::new()
        }
    }

    struct ScriptedSession<'a>(&'a Scripted);

    impl Session for ScriptedSession<'_> {
        fn forward(&mut self, tokens: &[TokenId]) -> Vec<f32> {
            self.0.seq.borrow_mut().extend_from_slice(tokens);
            let next = self
                .0
                .script
                .borrow_mut()
                .pop_front()
                .expect("a scripted token");
            let mut logits = vec![0.0; self.0.n_vocab];
            logits[next as usize] = 1.0;
            logits
        }
        fn prefill(&mut self, tokens: &[TokenId]) {
            self.0.seq.borrow_mut().extend_from_slice(tokens);
        }
        fn draft(
            &mut self,
            _: TokenId,
            _: usize,
            _: &mut dyn FnMut(&DraftLogits<'_>) -> TokenId,
        ) -> Vec<TokenId> {
            Vec::new()
        }
        fn verify(&mut self, _: &[TokenId]) -> Vec<Vec<f32>> {
            unreachable!()
        }
        fn keep(&mut self, _: usize) {}
        fn n_past(&self) -> usize {
            self.0.seq.borrow().len()
        }
        fn memory(&self) -> SessionMemory {
            SessionMemory::default()
        }
        fn snapshot(&self) -> Result<Snapshot, Error> {
            let seq = self.0.seq.borrow().clone();
            Ok(Snapshot::new(seq.len(), 0, seq))
        }
        fn restore(&mut self, s: &Snapshot) -> Result<(), Error> {
            *self.0.seq.borrow_mut() = s.data::<Vec<TokenId>>()?.clone();
            Ok(())
        }
    }

    fn greedy() -> Sampler {
        Sampler::new(SamplerConfig {
            temperature: 0.0,
            ..Default::default()
        })
    }

    /// What a turn showed: the reasoning, the text, the calls
    #[derive(Debug, Default, PartialEq)]
    struct Shown {
        thinking: String,
        done: usize,
        answer: String,
        calls: Vec<ToolCall>,
    }

    /// A turn: `user`'s message, or else the results of the last calls
    fn turn(chat: &mut Chat<'_>, user: Option<&str>, results: &[&str]) -> Shown {
        let mut s = Shown::default();
        let on_event = |ev: Event<'_>| match ev {
            Event::Thinking(t) => s.thinking.push_str(t),
            Event::ThinkingDone => s.done += 1,
            Event::Answer(t) => s.answer.push_str(t),
            Event::ToolCall(c) => s.calls.push(c.clone()),
            Event::PromptDone => {}
        };
        let stats = match user {
            Some(u) => chat.send(u, &mut greedy(), on_event, || false),
            None => chat.send_tool_responses(results, &mut greedy(), on_event, || false),
        }
        .unwrap();
        assert_eq!(stats.tool_calls, s.calls);
        s
    }

    const WEATHER: &str = r#"{"type": "function", "function": {"name": "get_weather", "description": "Current weather in a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}}}"#;
    const TIME: &str = r#"{"type": "function", "function": {"name": "get_time", "description": "Current local time in an IANA time zone", "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": ["timezone"]}}}"#;

    /// The model's answers of the conversation with tools
    const TOOL_ANSWERS: [&str; 3] = [
        r#"<|message_model|><|content_thinking|>Two tools.<|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Paris"}}<|end_message|><|message_model|>get_time<|content_invoke_tool_json|>{"name":"get_time","args":{"timezone":"Asia/Tokyo"}}<|end_message|><|content_model_end_sampling|>"#,
        "<|message_model|><|content_text|>It is 18 °C with light rain in Paris, and 19:42 in Tokyo.<|end_message|><|content_model_end_sampling|>",
        "<|message_model|><|content_text|>Paris: 18 °C, rain; Tokyo: 19:42.<|end_message|><|content_model_end_sampling|>",
    ];
    #[rustfmt::skip]
    const TOOLS: &[TokenId] = &[
        200002, 2251, 62, 76529, 200024, 58, 90, 1, 70469, 1, 25, 1, 36515, 154046, 149881,
        149878, 152652, 1, 11, 1, 930, 1, 25, 1, 832, 62, 30847, 1, 11, 1, 71968, 1, 25, 90, 1,
        71961, 1, 25, 90, 1, 2958, 1, 25, 90, 1, 70469, 1, 25, 1, 3278, 150231, 1, 11, 1, 1008,
        1, 25, 1, 1485, 1, 92, 92, 11, 1, 69812, 1, 25, 58, 1, 2958, 1, 60, 11, 1, 1008, 1, 25,
        1, 3156, 1, 92, 11, 1, 1008, 1, 25, 1, 36404, 1, 92, 11, 90, 1, 70469, 1, 25, 1, 36515,
        150368, 150038, 149881, 149916, 220, 148818, 150038, 152964, 1, 11, 1, 930, 1, 25, 1,
        832, 62, 848, 1, 11, 1, 71968, 1, 25, 90, 1, 71961, 1, 25, 90, 1, 848, 3746, 1, 25, 90,
        1, 1008, 1, 25, 1, 1485, 1, 92, 92, 11, 1, 69812, 1, 25, 58, 1, 848, 3746, 1, 60, 11, 1,
        1008, 1, 25, 1, 3156, 1, 92, 11, 1, 1008, 1, 25, 1, 36404, 1, 92, 60, 200010, 200002,
        200004, 1735, 149914, 149878, 160385, 171159, 13, 200010, 200002, 200004, 34182, 279,
        154554, 150628, 25, 220, 15, 13, 68436, 200010, 200000, 200004, 1775, 6, 82, 149876,
        154046, 149881, 152717, 11, 149880, 149876, 150038, 149881, 182416, 30, 200010, 200001,
        200008, 4348, 152804, 13, 200010, 200001, 832, 62, 30847, 200049, 90, 1, 930, 1, 25, 1,
        832, 62, 30847, 1, 11, 1, 70747, 1, 25, 90, 1, 2958, 1, 25, 1, 3967, 1, 92, 92, 200010,
        200001, 832, 62, 848, 200049, 90, 1, 930, 1, 25, 1, 832, 62, 848, 1, 11, 1, 70747, 1,
        25, 90, 1, 848, 3746, 1, 25, 1, 4997, 14, 34664, 1, 92, 92, 200010, 200006, 200003, 832,
        62, 30847, 200004, 90, 1, 70799, 1, 25, 220, 68073, 11, 196980, 32667, 1, 25, 196980,
        1638, 154005, 1, 92, 200010, 200003, 832, 62, 848, 200004, 90, 1, 848, 1, 25, 196980,
        68111, 21, 12, 68026, 12, 68120, 51, 68028, 25, 68872, 25, 68018, 10, 68609, 25, 68018,
        1, 92, 200010, 200001, 200004, 1256, 149884, 220, 68073, 153620, 34, 149899, 150932,
        154005, 149881, 152717, 11, 149880, 220, 68028, 25, 68872, 149881, 182416, 13, 200010,
        200006, 200000, 200004, 78028, 13, 159336, 149921, 151433, 149881, 150011, 150921,
        156021, 13, 200010, 200001, 200004, 3967, 25, 220, 68073, 153620, 34, 11, 154005, 26,
        182416, 25, 220, 68028, 25, 68872, 13, 200010, 200006
    ];
    /// The conversation with tools, thinking on (xhigh), in a chat from `opts`
    fn tools_conversation(tok: &Tokenizer, opts: &ChatOptions, saved: bool) -> Vec<TokenId> {
        let engine = Scripted::new(tok, &TOOL_ANSWERS);
        let mut chat = Chat::new(&engine, tok, opts.clone()).unwrap();
        if saved {
            let prefix = chat.save().unwrap();
            chat.reset().unwrap();
            chat.restore(&prefix).unwrap();
        }
        let s = turn(
            &mut chat,
            Some("What's the weather in Paris, and the time in Tokyo?"),
            &[],
        );
        assert_eq!(s.thinking, "Two tools.");
        assert_eq!((s.done, s.answer.as_str()), (1, ""));
        let call = |name: &str, k: &str, v: &str| ToolCall {
            name: name.into(),
            arguments: vec![(k.into(), v.into())],
        };
        assert_eq!(
            s.calls,
            [
                call("get_weather", "city", "Paris"),
                call("get_time", "timezone", "Asia/Tokyo")
            ]
        );
        let s = turn(
            &mut chat,
            None,
            &[
                r#"{"temperature": 18, "condition": "light rain"}"#,
                r#"{"time": "2026-10-30T19:42:00+09:00"}"#,
            ],
        );
        assert_eq!(
            s.answer,
            "It is 18 °C with light rain in Paris, and 19:42 in Tokyo."
        );
        let s = turn(
            &mut chat,
            Some("Thanks. Say it again in one short sentence."),
            &[],
        );
        assert_eq!(s.answer, "Paris: 18 °C, rain; Tokyo: 19:42.");
        let t = chat.transcript();
        // everything is processed but the end of the last answer
        assert_eq!(*engine.seq.borrow(), t.tokens[..t.tokens.len() - t.pending]);
        assert_eq!(t.tokens[t.tokens.len() - t.pending..], [200006]);
        t.tokens
    }

    #[test]
    fn inkling_chat_needs_no_qwen_tokens() {
        let Some(tok) = tok() else {
            std::eprintln!("no Inkling-2-Mini GGUF: skipped");
            return;
        };
        let engine = Scripted::new(tok, &[]);
        let chat = Chat::new(&engine, tok, ChatOptions::default()).unwrap();
        assert!(matches!(chat.format, Format::Inkling(_)));
        assert_eq!(chat.template(), Some(compiled::Template::Inkling2));
        assert!(chat.controls_thinking() && chat.thinking_per_conversation());
        assert_eq!(tok.eog, [200006, 199999]);
    }

    #[test]
    fn inkling_conversation_with_tools() {
        let Some(tok) = tok() else {
            std::eprintln!("no Inkling-2-Mini GGUF: skipped");
            return;
        };
        let opts = ChatOptions {
            system: Some("You are a helpful assistant.".into()),
            tools: vec![WEATHER.into(), TIME.into()],
            ..Default::default()
        };
        let builtin = ChatOptions {
            builtin_format: true,
            ..opts.clone()
        };
        for (opts, saved) in [
            (&opts, false),
            (&builtin, false),
            (&opts, true),
            (&builtin, true),
        ] {
            let what = (opts.builtin_format, saved);
            assert_eq!(tools_conversation(tok, opts, saved), TOOLS, "{what:?}");
        }
    }

    /// The model's answers of the conversation without thinking: reasoning
    /// anyway in the second
    const PLAIN_ANSWERS: [&str; 2] = [
        "<|message_model|><|content_text|>Hello!<|end_message|><|content_model_end_sampling|>",
        "<|message_model|><|content_thinking|>Short.<|end_message|><|message_model|><|content_text|>Bye!<|end_message|><|content_model_end_sampling|>",
    ];
    #[rustfmt::skip]
    const PLAIN: &[TokenId] = &[
        200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 200010, 200000, 200004, 153725,
        198, 200010, 200001, 200004, 32269, 0, 200010, 200006, 200000, 200004, 48135, 200010,
        200001, 200008, 31485, 13, 200010, 200001, 200004, 48135, 0, 200010, 200006
    ];
    #[test]
    fn inkling_conversation_without_thinking() {
        let Some(tok) = tok() else {
            std::eprintln!("no Inkling-2-Mini GGUF: skipped");
            return;
        };
        for builtin_format in [false, true] {
            let opts = ChatOptions {
                thinking: false,
                // a blank system prompt is none
                system: Some(" \n".into()),
                builtin_format,
                ..Default::default()
            };
            let engine = Scripted::new(tok, &PLAIN_ANSWERS);
            let mut chat = Chat::new(&engine, tok, opts).unwrap();
            assert_eq!(turn(&mut chat, Some(" Hi\n"), &[]).answer, "Hello!");
            let s = turn(&mut chat, Some("Bye"), &[]);
            assert_eq!((s.thinking.as_str(), s.done), ("Short.", 1));
            assert_eq!(s.answer, "Bye!");
            assert_eq!(chat.transcript().tokens, PLAIN, "built-in {builtin_format}");
        }
    }

    /// The first turn's tokens, the same whatever the effort and the way
    /// they are rendered
    #[test]
    fn inkling_effort_levels() {
        let Some(tok) = tok() else {
            std::eprintln!("no Inkling-2-Mini GGUF: skipped");
            return;
        };
        let answer =
            ["<|message_model|><|content_text|>A<|end_message|><|content_model_end_sampling|>"];
        for (thinking, effort, level) in [
            (true, ReasoningEffort::Low, "0.2"),
            (true, ReasoningEffort::Medium, "0.7"),
            (true, ReasoningEffort::High, "0.9"),
            (true, ReasoningEffort::XHigh, "0.99"),
            (false, ReasoningEffort::XHigh, "0"),
            (false, ReasoningEffort::Low, "0"),
        ] {
            let first = |builtin_format: bool| {
                let opts = ChatOptions {
                    thinking,
                    effort,
                    builtin_format,
                    ..Default::default()
                };
                let engine = Scripted::new(tok, &answer);
                let mut chat = Chat::new(&engine, tok, opts).unwrap();
                turn(&mut chat, Some("Q"), &[]);
                chat.transcript().tokens
            };
            let want = tok.encode_with_special(&format!(
                "<|message_system|><|content_text|>Thinking effort level: {level}<|end_message|><|message_user|><|content_text|>Q<|end_message|>{}",
                answer[0]
            ));
            assert_eq!(first(false), want, "{thinking} {effort:?}");
            assert_eq!(first(true), want, "{thinking} {effort:?}");
        }
    }

    /// However an answer ends, the sequence goes on as the template writes
    /// the end of an answer: `…<|end_message|><|content_model_end_sampling|>`
    #[test]
    fn inkling_answer_endings() {
        let Some(tok) = tok() else {
            std::eprintln!("no Inkling-2-Mini GGUF: skipped");
            return;
        };
        let opening = "<|message_system|><|content_text|>Thinking effort level: 0.99<|end_message|><|message_user|><|content_text|>Q<|end_message|>";
        let next = "<|message_user|><|content_text|>Next<|end_message|>";
        let done =
            "<|message_model|><|content_text|>B<|end_message|><|content_model_end_sampling|>";
        // generated, max_tokens, the answer as the sequence has it
        for (generated, max_tokens, closed) in [
            (
                "<|message_model|><|content_text|>Hi<|end_message|><|endoftext|>",
                100,
                "<|message_model|><|content_text|>Hi<|end_message|><|content_model_end_sampling|>",
            ),
            (
                "<|message_model|><|content_text|>Hi<|content_model_end_sampling|>",
                100,
                "<|message_model|><|content_text|>Hi<|end_message|><|content_model_end_sampling|>",
            ),
            (
                // cut by max_tokens in a block
                "<|message_model|><|content_text|>Hello",
                3,
                "<|message_model|><|content_text|>Hello<|end_message|><|content_model_end_sampling|>",
            ),
            (
                "<|message_model|><|content_text|>Hi<|end_message|>",
                4,
                "<|message_model|><|content_text|>Hi<|end_message|><|content_model_end_sampling|>",
            ),
            (
                "<|message_model|>",
                1,
                "<|message_model|><|end_message|><|content_model_end_sampling|>",
            ),
            // another role's header ends the answer, and is left out
            (
                "<|message_model|><|content_text|>Hi<|end_message|><|message_user|>",
                100,
                "<|message_model|><|content_text|>Hi<|end_message|><|content_model_end_sampling|>",
            ),
            // nothing but the end: an answer without content
            (
                "<|content_model_end_sampling|>",
                100,
                "<|content_model_end_sampling|>",
            ),
            // text outside any block, after one or before any: no block is
            // open, the sequence keeps it as the model wrote it
            (
                "<|message_model|><|content_text|>Hi<|end_message|>\n<|content_model_end_sampling|>",
                100,
                "<|message_model|><|content_text|>Hi<|end_message|>\n<|content_model_end_sampling|>",
            ),
            (
                "<|message_model|><|content_text|>Hi<|end_message|>ok",
                5,
                "<|message_model|><|content_text|>Hi<|end_message|>ok<|content_model_end_sampling|>",
            ),
            (
                "Hello<|content_model_end_sampling|>",
                100,
                "Hello<|content_model_end_sampling|>",
            ),
        ] {
            for builtin_format in [false, true] {
                let opts = ChatOptions {
                    max_tokens,
                    builtin_format,
                    ..Default::default()
                };
                let engine = Scripted::new(tok, &[generated, done]);
                let mut chat = Chat::new(&engine, tok, opts).unwrap();
                turn(&mut chat, Some("Q"), &[]);
                chat.opts.max_tokens = 100;
                assert_eq!(turn(&mut chat, Some("Next"), &[]).answer, "B");
                let want = tok.encode_with_special(&format!("{opening}{closed}{next}{done}"));
                assert_eq!(
                    chat.transcript().tokens,
                    want,
                    "{generated:?}, built-in {builtin_format}"
                );
            }
        }
    }

    /// The answer's blocks as events, from tokens: whatever the reference
    /// parsers take
    #[test]
    fn inkling_reader() {
        let Some(tok) = tok() else {
            std::eprintln!("no Inkling-2-Mini GGUF: skipped");
            return;
        };
        let engine = Scripted::new(tok, &[]);
        let Format::Inkling(f) = Chat::new(&engine, tok, ChatOptions::default())
            .unwrap()
            .format
        else {
            panic!("not Inkling's format");
        };
        let read = |text: &str| {
            let mut r = InklingReader::new(f);
            let mut s = Shown::default();
            let mut calls = Vec::new();
            let mut on_event = |ev: Event<'_>| match ev {
                Event::Thinking(t) => s.thinking.push_str(t),
                Event::ThinkingDone => s.done += 1,
                Event::Answer(t) => s.answer.push_str(t),
                Event::ToolCall(c) => s.calls.push(c.clone()),
                Event::PromptDone => {}
            };
            for t in tok.encode_with_special(text) {
                r.token(t, tok, &mut calls, &mut on_event);
            }
            let closing = r.close(&mut calls, &mut on_event);
            assert_eq!(calls, s.calls);
            (s, closing)
        };
        let call = |name: &str, args: &[(&str, &str)]| ToolCall {
            name: name.into(),
            arguments: args.iter().map(|&(k, v)| (k.into(), v.into())).collect(),
        };
        // text blocks are joined, not trimmed; leading newlines dropped
        let (s, closing) = read(
            "<|message_model|><|content_text|>\n\nA \n<|end_message|><|message_model|><|content_text|>B<|end_message|>",
        );
        assert_eq!((s.answer.as_str(), &closing[..]), ("A \nB", &[200006][..]));
        // a call without the name before it, JSON with spaces and values of
        // every kind: the JSON's name is the one
        let (s, _) = read(
            r#"<|message_model|><|content_invoke_tool_json|>{"name": "f", "args": {"s": "x y", "n": 1.5, "o": {"b": [1, null], "a": true}}}<|end_message|><|message_model|>other<|content_invoke_tool_json|>{"name":"g","args":{}}<|end_message|>"#,
        );
        assert_eq!(
            s.calls,
            [
                call(
                    "f",
                    &[
                        ("s", "x y"),
                        ("n", "1.5"),
                        ("o", r#"{"b":[1,null],"a":true}"#)
                    ]
                ),
                call("g", &[])
            ]
        );
        // numbers as the model wrote them, past i64 and f64 too; a key
        // given twice is its last value, as Python's json
        let (s, _) = read(
            r#"<|message_model|><|content_invoke_tool_json|>{"name":"f","args":{"id": 12345678901234567890, "x":1e400,"y":[1E5, "a \" ]"],"id":-0.10}, "args2": 1}<|end_message|>"#,
        );
        assert_eq!(
            s.calls,
            [call(
                "f",
                &[("id", "-0.10"), ("x", "1e400"), ("y", r#"[1E5,"a \" ]"]"#)]
            )]
        );
        // the header's name when the JSON has none; no header at all
        let (s, _) = read(
            r#"<|message_model|> h <|content_invoke_tool_json|>{"args":{"k":"v"}}<|end_message|><|content_text|>after<|end_message|>"#,
        );
        assert_eq!(
            (s.calls, s.answer.as_str()),
            (vec![call("h", &[("k", "v")])], "after")
        );
        // not a call: shown as it was written
        let (s, _) =
            read("<|message_model|><|content_invoke_tool_json|>{\"args\": 1}<|end_message|>");
        assert_eq!((s.calls.len(), s.answer.as_str()), (0, "{\"args\": 1}"));
        // a call cut short at the end, complete or not
        let (s, closing) =
            read(r#"<|message_model|><|content_invoke_tool_json|>{"name":"f","args":{}}"#);
        assert_eq!(
            (s.calls, &closing[..]),
            (vec![call("f", &[])], &[200010, 200006][..])
        );
        let (s, _) = read(r#"<|message_model|><|content_invoke_tool_json|>{"name":"f","ar"#);
        assert_eq!(
            (s.calls.len(), s.answer.as_str()),
            (0, r#"{"name":"f","ar"#)
        );
        // reasoning, then text; an error and a call as text show as text; an
        // image's block does not
        let (s, closing) = read(
            "<|message_model|><|content_thinking|>\nhmm\n<|end_message|><|message_model|><|content_text|>\nok<|end_message|><|message_model|><|content_tool_error|>E<|end_message|><|message_model|><|content_invoke_tool_text|>T<|end_message|><|message_model|><|content_image|>img<|end_message|><|message_model|><|content_text|>",
        );
        assert_eq!((s.thinking.as_str(), s.done), ("hmm", 1));
        assert_eq!(
            (s.answer.as_str(), &closing[..]),
            ("okET", &[200010, 200006][..])
        );
    }
}
