//! Multi-turn chat on top of the model, with the chat template of the model
//! file ([`crate::template`]).
//!
//! For each turn the chat renders the whole conversation with the template
//! ([`Conversation`]). When the tokens that the session holds start the
//! tokens of the render, the session goes on and only the rest is
//! processed. Otherwise the session resets and processes the whole render:
//! the recurrent layers cannot rewind (beyond the last verify pass of
//! speculative decoding), so a session cannot keep part of what it holds. A
//! template that keeps past reasoning (`preserve_thinking`) gives a reuse
//! when it writes an answer as the model wrote it. A template that drops
//! past reasoning, or trims an answer, gives a reset ([`TurnStats::reset`]).
//!
//! A turn runs to its end in [`Chat::send`], or one pass of the model at a
//! time: [`Chat::begin`] opens it, each [`Chat::step`] does one pass and gives
//! the tokens chosen after it, and [`Chat::cancel`] ends it between two steps.
//! A caller that runs the model on its own schedule asks for the next pass
//! only when it has taken the tokens of the last one.
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

use alloc::collections::VecDeque;

use crate::Error;
pub use crate::conversation::{Conversation, Said};
use crate::engine::{DraftLogits, Engine, Session, Snapshot};
use crate::memory::SessionMemory;
use crate::prelude::*;
use crate::sampler::Sampler;
use crate::template::{self, ChatFormat, EFFORT_WORDS, Origin, Template, ToolCallForm, Value};
use crate::tokenizer::{PromptPart, StreamDecoder, TokenId, Tokenizer};
pub use crate::transcript::Transcript;

/// The top-level names that a chat sets in each render: the template is
/// loaded with them, and a render sets each one. `reasoning_effort` is set
/// only when thinking is on; `bos_token` and `eos_token` are set to empty
/// strings (the tokenizer handles the actual tokens).
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
const BOS_TOKEN: &str = "bos_token";
const EOS_TOKEN: &str = "eos_token";
/// A chat keeps the reasoning of earlier answers in the prompt, so that a
/// template that writes it as the model wrote it gives a reuse
const PRESERVE_THINKING: bool = true;

/// The tokens that the close of a stopped turn adds after the context: the
/// last token chosen, the end of the open block (`</think>` or `</tool_call>`)
/// and the turn end. One block at most is open, because a call opens only
/// after the reasoning block. A turn starts only with this room above its
/// prompt, and stops at the end of the context while this room is left, so
/// that the session holds the close before the next prompt.
pub const CLOSE_ROOM_TOKENS: usize = 3;

/// The chat template of the model and its probed properties: the texts and
/// special tokens that a chat needs during generation
struct Format {
    template: Template,
    /// the special token that ends a turn
    turn_end: TokenId,
    /// the special token that ends the reasoning block, when the template
    /// writes reasoning: from the generation prompt with thinking off minus
    /// the generation prompt with thinking on, trimmed to one special token
    reasoning_end: Option<TokenId>,
    /// the special tokens that open and close a tool call (only the
    /// [`ToolCallForm::XmlParameters`] form), when the template writes them
    tool_call: Option<(TokenId, TokenId)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    Low,
    Medium,
    XHigh,
}

impl ReasoningEffort {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "low" => Self::Low,
            "medium" => Self::Medium,
            "high" | "xhigh" => Self::XHigh,
            _ => return None,
        })
    }

    /// The effort word as the template reads it
    pub fn word(self) -> &'static str {
        match self {
            Self::Low => EFFORT_WORDS[0],
            Self::Medium => EFFORT_WORDS[1],
            Self::XHigh => EFFORT_WORDS[2],
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChatOptions {
    pub system: Option<String>,
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
    /// "get_weather", "description": "...", "parameters": {...}}}`
    pub tools: Vec<String>,
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
        }
    }
}

/// Pieces of the answer, as they are generated
pub enum Event<'a> {
    /// the prompt is processed, generation starts
    PromptDone,
    /// a token of the answer, as soon as it is chosen and before its text:
    /// the ids of [`TurnStats::tokens`], in order, the end marker included
    /// when the model writes it
    Token(TokenId),
    Thinking(&'a str),
    Answer(&'a str),
    /// the reasoning block is complete
    ThinkingDone,
    /// a function call, to answer with [`Chat::send_tool_responses`]
    ToolCall(&'a ToolCall),
}

/// A message of an earlier conversation, for [`Chat::send_with_history`]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Message<'a> {
    User(&'a str),
    /// the answer, without its reasoning
    Assistant(&'a str),
    /// an answer with its reasoning, as the model wrote them: `reasoning` is
    /// the text of the tokens after `<think>\n` and before `</think>`, and
    /// `answer` the text of the tokens after `</think>`, without the end of
    /// the message. Neither is trimmed, so a chat that goes on from them
    /// holds the tokens of the chat that wrote them, when the tokenizer gives
    /// their text the same tokens. An answer that stopped in its reasoning
    /// has an empty `answer`. A special token in either text is plain text,
    /// as in all the content of a message.
    AssistantWithReasoning {
        reasoning: &'a str,
        answer: &'a str,
    },
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

/// Why a turn ended
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    /// the model wrote its end marker
    End,
    /// the answer reached `max_tokens`
    MaxTokens,
    /// the session reached the end of the context
    ContextFull,
    /// the caller ended the turn: [`Chat::cancel`], or the `stop` function
    /// of [`Chat::send`]
    Cancel,
}

#[derive(Debug, Clone)]
pub struct TurnStats {
    pub prompt_tokens: usize,
    pub gen_tokens: usize,
    /// why the turn ended
    pub stop: Stop,
    /// whether the session was reset for this turn: the rendered prompt did
    /// not start with the tokens the session held
    pub reset: bool,
    /// tokens drafted by the MTP head or proposed from the conversation, and
    /// how many of them were right
    pub drafted: usize,
    pub accepted: usize,
    /// of those, proposed from the conversation, and right
    pub looked_up: usize,
    pub looked_up_accepted: usize,
    /// the function calls of the answer, as given to [`Event::ToolCall`]
    pub tool_calls: Vec<ToolCall>,
    /// the tokens sampled for the answer, in order, the end marker included
    /// when the model wrote it
    pub tokens: Vec<TokenId>,
}

/// What one [`Chat::step`] did
#[derive(Debug, Clone)]
pub enum Step {
    /// a pass over a part of the prompt; more of it is left
    Prompt,
    /// the tokens chosen after the pass, in order; the turn goes on
    Tokens(Vec<TokenId>),
    /// the tokens chosen after the last pass, in order, and the end of the
    /// turn
    Done(Vec<TokenId>, TurnStats),
}

/// An open turn: the state between two steps
struct Turn {
    stats: TurnStats,
    phase: Phase,
    /// how many messages the conversation had before this turn: on cancel
    /// at done 0, the conversation is truncated back to this count
    messages_before: usize,
}

enum Phase {
    /// the prompt, of which `done` tokens are processed; it starts with the
    /// `pending` tokens that the chat held before the turn
    Prompt {
        prompt: Vec<TokenId>,
        done: usize,
        pending: usize,
    },
    /// the answer, with `next` chosen and not yet processed
    Answer { answer: Answer, next: TokenId },
}

/// The state of an answer between two steps
struct Answer {
    thinking: bool,
    decoder: StreamDecoder,
    /// the template puts "\n\n" after </think>: don't echo it
    skip_leading_newlines: bool,
    /// newlines at the end of the text so far, shown only if more follows
    held_newlines: String,
    /// the text of the function call being generated
    call: Option<String>,
    spec: Speculation,
}

impl Answer {
    fn new(thinking: bool) -> Self {
        Answer {
            thinking,
            decoder: StreamDecoder::default(),
            skip_leading_newlines: !thinking,
            held_newlines: String::new(),
            call: None,
            spec: Speculation::default(),
        }
    }
}

/// A conversation up to some point, see [`Chat::save`]
pub struct Prefix {
    snapshot: Snapshot,
    pending: Vec<TokenId>,
    generated: Vec<TokenId>,
    history: Vec<TokenId>,
    conversation: Conversation,
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
    format: Format,
    conversation: Conversation,
    session: Box<dyn Session + 'm>,
    /// tokens not yet processed, fed before the next prompt: the end of the
    /// last answer, or the rest of a cancelled prompt
    pending: Vec<TokenId>,
    /// every generated token, for the sampler penalties
    generated: Vec<TokenId>,
    /// every token of the sequence, for the proposals from the conversation
    history: Vec<TokenId>,
    /// consecutive steps whose proposal from the conversation was all right
    lookup_run: usize,
    /// the turn between [`Chat::begin`] and its end
    turn: Option<Turn>,
}

/// Load the template of the model file, probe it, and build the [`Format`]
/// that a chat uses during generation. Fails when the model has no template,
/// when the template does not load or probe, or when a token the probes
/// need is not in the vocabulary.
fn load_format(tok: &Tokenizer) -> Result<Format, Error> {
    let source = tok.chat_template.as_ref().ok_or(Error::TemplateMissing)?;
    let template = Template::load(source, &VARIABLES).map_err(Error::Template)?;
    let is_special = |text: &str| tok.token_id(text).is_some_and(|id| tok.is_special(id));
    let probe = ChatFormat::probe(&template, &is_special).map_err(Error::Probe)?;

    let turn_end = tok
        .token_id(&probe.turn_end)
        .ok_or_else(|| Error::MissingToken(probe.turn_end.clone()))?;

    // the reasoning end token: the generation prompt with thinking off
    // minus the one with thinking on, trimmed, must be one special token
    let reasoning_end = if probe.reasoning.is_some() {
        let diff = probe
            .generation_prompt_plain
            .strip_prefix(&probe.generation_prompt_thinking)
            .ok_or(Error::Probe(template::ProbeError::TemplateReasoningUnclear))?;
        let trimmed = diff.trim();
        if trimmed.is_empty() || !is_special(trimmed) {
            return Err(Error::Probe(template::ProbeError::TemplateReasoningUnclear));
        }
        Some(
            tok.token_id(trimmed)
                .ok_or_else(|| Error::MissingToken(trimmed.into()))?,
        )
    } else {
        None
    };

    // tool call tokens: only the XmlParameters form
    let tool_call = if let Some(tc) = &probe.tool_call {
        match tc.form {
            ToolCallForm::XmlParameters => {
                let open = tok
                    .token_id(&tc.open)
                    .ok_or_else(|| Error::MissingToken(tc.open.clone()))?;
                let close = tok
                    .token_id(&tc.close)
                    .ok_or_else(|| Error::MissingToken(tc.close.clone()))?;
                Some((open, close))
            }
            other => return Err(Error::ToolCallForm(other)),
        }
    } else {
        None
    };

    Ok(Format {
        template,
        turn_end,
        reasoning_end,
        tool_call,
    })
}

impl<'m> Chat<'m> {
    pub fn new(
        engine: &'m dyn Engine,
        tok: &'m Tokenizer,
        opts: ChatOptions,
    ) -> Result<Self, Error> {
        let format = load_format(tok)?;
        if !opts.tools.is_empty() && format.tool_call.is_none() {
            return Err(Error::ToolsUnsupported);
        }
        if opts.thinking && format.reasoning_end.is_none() {
            return Err(Error::ThinkingUnsupported);
        }
        let conversation = Conversation::new(opts.system.clone(), opts.tools.clone());
        let session = engine.new_session(opts.context, session_draft(engine, &opts))?;
        Ok(Chat {
            engine,
            tok,
            format,
            conversation,
            session,
            opts,
            pending: Vec::new(),
            generated: Vec::new(),
            history: Vec::new(),
            lookup_run: 0,
            turn: None,
        })
    }

    /// A chat that goes on from `session`, a session of `engine` made as
    /// [`Chat::new`] makes one (`opts.context`, and the draft size of `opts`
    /// or of the engine, at least `opts.lookup`), for example one rebuilt by
    /// repeating the session calls of another chat, or one whose calls are
    /// recorded. `pending` are the tokens generated but not yet processed,
    /// fed before the next prompt, `generated` every token generated so far,
    /// for the sampler penalties, and `history` every token of the
    /// conversation, for prompt lookup; all empty for a session that holds
    /// nothing. `conversation` is the messages the chat has held.
    /// A session that holds tokens gets no system block: its system prompt,
    /// reasoning instructions and tools are those it holds, whatever `opts`
    /// says, as for [`Chat::restore`]. Fails with [`Error::ContextFull`] when
    /// the session and `pending` hold more tokens than `opts.context`.
    pub fn from_session(
        engine: &'m dyn Engine,
        tok: &'m Tokenizer,
        opts: ChatOptions,
        session: Box<dyn Session + 'm>,
        pending: Vec<TokenId>,
        generated: Vec<TokenId>,
        history: Vec<TokenId>,
        conversation: Conversation,
    ) -> Result<Self, Error> {
        if session.n_past() + pending.len() > opts.context {
            return Err(Error::ContextFull {
                n_past: session.n_past(),
                prompt: pending.len(),
                context: opts.context,
            });
        }
        let format = load_format(tok)?;
        if !opts.tools.is_empty() && format.tool_call.is_none() {
            return Err(Error::ToolsUnsupported);
        }
        if opts.thinking && format.reasoning_end.is_none() {
            return Err(Error::ThinkingUnsupported);
        }
        Ok(Chat {
            engine,
            tok,
            format,
            conversation,
            session,
            opts,
            pending,
            generated,
            history,
            lookup_run: 0,
            turn: None,
        })
    }

    pub fn reset(&mut self) -> Result<(), Error> {
        self.session = Box::new(Empty);
        self.session = self
            .engine
            .new_session(self.opts.context, session_draft(self.engine, &self.opts))?;
        self.pending.clear();
        self.generated.clear();
        self.history.clear();
        self.lookup_run = 0;
        self.turn = None;
        self.conversation = Conversation::new(self.opts.system.clone(), self.opts.tools.clone());
        Ok(())
    }

    pub fn n_past(&self) -> usize {
        self.session.n_past()
    }

    pub fn memory(&self) -> SessionMemory {
        self.session.memory()
    }

    pub fn session_memory(engine: &dyn Engine, opts: &ChatOptions) -> SessionMemory {
        engine.session_memory(opts.context, session_draft(engine, opts))
    }

    pub fn generated(&self) -> &[TokenId] {
        &self.generated
    }

    pub fn history(&self) -> &[TokenId] {
        &self.history
    }

    pub fn pending(&self) -> &[TokenId] {
        &self.pending
    }

    pub fn turn_prompt_tokens(&self) -> Option<usize> {
        self.turn.as_ref().map(|turn| turn.stats.prompt_tokens)
    }

    /// The conversation of the chat
    pub fn conversation(&self) -> &Conversation {
        &self.conversation
    }

    /// Save the conversation so far, to [`Chat::restore`] it in this chat or
    /// another on the same engine. Before the first message, the system prompt
    /// is processed now, so that the prefix holds it.
    pub fn save(&mut self) -> Result<Prefix, Error> {
        self.no_turn()?;
        if self.session.n_past() == 0 && self.conversation.has_messages() {
            let system = self.render_tokens(false)?;
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
            conversation: self.conversation.clone(),
        })
    }

    pub fn restore(&mut self, prefix: &Prefix) -> Result<(), Error> {
        self.no_turn()?;
        self.session.restore(&prefix.snapshot)?;
        self.pending.clone_from(&prefix.pending);
        self.generated.clone_from(&prefix.generated);
        self.history.clone_from(&prefix.history);
        self.conversation = prefix.conversation.clone();
        self.lookup_run = 0;
        Ok(())
    }

    pub fn n_tokens(&self) -> usize {
        self.history.len() + self.pending.len()
    }

    pub fn transcript(&self) -> Transcript {
        let mut tokens = Vec::with_capacity(self.n_tokens());
        tokens.extend_from_slice(&self.history);
        tokens.extend_from_slice(&self.pending);
        Transcript {
            tokens,
            pending: self.pending.len(),
            generated: self.generated.clone(),
            conversation: Some(self.conversation.clone()),
        }
    }

    /// A version 1 transcript holds no messages, so a chat that renders
    /// with its template cannot use it.
    pub fn restore_transcript(&mut self, _transcript: &Transcript) -> Result<(), Error> {
        Err(Error::TranscriptNoMessages)
    }

    fn no_turn(&self) -> Result<(), Error> {
        match self.turn {
            Some(_) => Err(Error::TurnOpen),
            None => Ok(()),
        }
    }

    fn draft(&self) -> usize {
        self.opts.draft.unwrap_or_else(|| self.engine.draft_size())
    }

    fn batch(&self) -> usize {
        self.opts
            .batch
            .unwrap_or_else(|| self.engine.batch_size())
            .max(1)
    }

    fn prefill(&mut self, tokens: &[TokenId]) {
        for chunk in tokens.chunks(self.batch()) {
            self.session.prefill(chunk);
        }
    }

    /// Render the whole conversation with the template, with or without the
    /// generation prompt at the end, and encode the result
    fn render_tokens(&self, generation: bool) -> Result<Vec<TokenId>, Error> {
        let (messages, tools) = self.conversation.values()?;
        let mut vars = vec![
            ("messages".into(), messages),
            ("tools".into(), tools),
            ("documents".into(), Value::None),
            ("add_generation_prompt".into(), Value::Bool(generation)),
            (BOS_TOKEN.into(), Value::str("", Origin::Template)),
            (EOS_TOKEN.into(), Value::str("", Origin::Template)),
            ("enable_thinking".into(), Value::Bool(self.opts.thinking)),
            ("preserve_thinking".into(), Value::Bool(PRESERVE_THINKING)),
        ];
        if self.opts.thinking {
            vars.push((
                "reasoning_effort".into(),
                Value::str(self.opts.effort.word(), Origin::Template),
            ));
        }
        let segments = self
            .format
            .template
            .render(&vars, None)
            .map_err(Error::Template)?;
        let parts: Vec<PromptPart<'_>> = segments
            .iter()
            .map(|s| match s.origin {
                Origin::Template => PromptPart::Template(&s.text),
                Origin::Content => PromptPart::Content(&s.text),
            })
            .collect();
        Ok(self.tok.encode_prompt(&parts))
    }

    pub fn send(
        &mut self,
        user: &str,
        sampler: &mut Sampler,
        on_event: impl FnMut(Event<'_>),
        stop: impl Fn() -> bool,
    ) -> Result<TurnStats, Error> {
        self.begin(user)?;
        self.run(sampler, on_event, stop)
    }

    pub fn send_with_history(
        &mut self,
        history: &[Message<'_>],
        user: &str,
        sampler: &mut Sampler,
        on_event: impl FnMut(Event<'_>),
        stop: impl Fn() -> bool,
    ) -> Result<TurnStats, Error> {
        self.begin_with_history(history, user)?;
        self.run(sampler, on_event, stop)
    }

    pub fn send_tool_responses(
        &mut self,
        responses: &[impl AsRef<str>],
        sampler: &mut Sampler,
        on_event: impl FnMut(Event<'_>),
        stop: impl Fn() -> bool,
    ) -> Result<TurnStats, Error> {
        self.begin_tool_responses(responses)?;
        self.run(sampler, on_event, stop)
    }

    pub fn begin(&mut self, user: &str) -> Result<(), Error> {
        self.begin_with_history(&[], user)
    }

    /// Start a turn as [`Chat::send_with_history`] does, to [`Chat::step`]
    pub fn begin_with_history(&mut self, history: &[Message<'_>], user: &str) -> Result<(), Error> {
        self.no_turn()?;
        let messages_before = self.conversation.messages.len();
        for m in history {
            self.conversation.messages.push(match *m {
                Message::User(text) => Said::User(text.to_string()),
                Message::Assistant(text) => Said::Assistant {
                    reasoning: None,
                    answer: text.to_string(),
                    tool_calls: Vec::new(),
                },
                Message::AssistantWithReasoning { reasoning, answer } => Said::Assistant {
                    reasoning: Some(reasoning.to_string()),
                    answer: answer.to_string(),
                    tool_calls: Vec::new(),
                },
            });
        }
        self.conversation
            .messages
            .push(Said::User(user.to_string()));
        self.begin_turn(messages_before)
    }

    /// Start a turn as [`Chat::send_tool_responses`] does, to [`Chat::step`]
    pub fn begin_tool_responses(&mut self, responses: &[impl AsRef<str>]) -> Result<(), Error> {
        self.no_turn()?;
        let messages_before = self.conversation.messages.len();
        for r in responses {
            self.conversation
                .messages
                .push(Said::Tool(r.as_ref().to_string()));
        }
        self.begin_turn(messages_before)
    }

    /// Open the turn: render the conversation, do the D7 prefix check, and
    /// set up the prompt. On error, the conversation is rolled back.
    fn begin_turn(&mut self, messages_before: usize) -> Result<(), Error> {
        let rendered = match self.render_tokens(true) {
            Ok(r) => r,
            Err(e) => {
                self.conversation.messages.truncate(messages_before);
                return Err(e);
            }
        };

        // D7: if the rendered tokens start with what the session holds
        // (history) followed by the unprocessed end of the last answer
        // (pending), the session goes on; otherwise it resets
        let held_len = self.history.len() + self.pending.len();
        let prefix_matches = rendered.len() >= held_len
            && rendered[..self.history.len()] == self.history[..]
            && rendered[self.history.len()..held_len] == self.pending[..];
        let (prompt, reset, pending_count) = if prefix_matches {
            let pc = self.pending.len();
            self.pending.clear();
            (rendered[self.history.len()..].to_vec(), false, pc)
        } else {
            // reset: clear the session and process the whole render
            self.session = Box::new(Empty);
            self.session = match self
                .engine
                .new_session(self.opts.context, session_draft(self.engine, &self.opts))
            {
                Ok(s) => s,
                Err(e) => {
                    self.conversation.messages.truncate(messages_before);
                    return Err(e);
                }
            };
            self.generated.clear();
            self.history.clear();
            self.pending.clear();
            self.lookup_run = 0;
            (rendered, true, 0)
        };

        if self.session.n_past() + prompt.len() + CLOSE_ROOM_TOKENS > self.opts.context {
            self.conversation.messages.truncate(messages_before);
            return Err(Error::ContextFull {
                n_past: self.session.n_past(),
                prompt: prompt.len(),
                context: self.opts.context,
            });
        }

        let stats = TurnStats {
            prompt_tokens: prompt.len(),
            gen_tokens: 0,
            stop: Stop::End,
            reset,
            drafted: 0,
            accepted: 0,
            looked_up: 0,
            looked_up_accepted: 0,
            tool_calls: Vec::new(),
            tokens: Vec::new(),
        };
        self.turn = Some(Turn {
            stats,
            phase: Phase::Prompt {
                prompt,
                done: 0,
                pending: pending_count,
            },
            messages_before,
        });
        Ok(())
    }

    fn run(
        &mut self,
        sampler: &mut Sampler,
        mut on_event: impl FnMut(Event<'_>),
        stop: impl Fn() -> bool,
    ) -> Result<TurnStats, Error> {
        loop {
            match self.step(sampler, &mut on_event)? {
                Step::Done(_, stats) => return Ok(stats),
                Step::Tokens(_) if stop() => return self.cancel(&mut on_event),
                Step::Prompt | Step::Tokens(_) => {}
            }
        }
    }

    pub fn step(
        &mut self,
        sampler: &mut Sampler,
        mut on_event: impl FnMut(Event<'_>),
    ) -> Result<Step, Error> {
        let Turn {
            mut stats,
            phase,
            messages_before,
        } = self.turn.take().ok_or(Error::NoTurn)?;
        let (answer, logits) = match phase {
            Phase::Prompt {
                prompt,
                done,
                pending,
            } => {
                let body = prompt.len() - 1;
                if done < body {
                    let end = body.min(done + self.batch());
                    self.session.prefill(&prompt[done..end]);
                    self.history.extend_from_slice(&prompt[done..end]);
                    self.turn = Some(Turn {
                        stats,
                        phase: Phase::Prompt {
                            prompt,
                            done: end,
                            pending,
                        },
                        messages_before,
                    });
                    return Ok(Step::Prompt);
                }
                let last = &prompt[body..];
                self.history.extend_from_slice(last);
                let logits = self.session.forward(last);
                on_event(Event::PromptDone);
                (Answer::new(self.opts.thinking), logits)
            }
            Phase::Answer { mut answer, next } => {
                let logits = self.advance(next, &mut answer.spec, &mut stats, sampler);
                (answer, logits)
            }
        };
        self.choose_tokens(
            stats,
            answer,
            logits,
            sampler,
            &mut on_event,
            messages_before,
        )
    }

    pub fn cancel(&mut self, mut on_event: impl FnMut(Event<'_>)) -> Result<TurnStats, Error> {
        let Turn {
            mut stats,
            phase,
            messages_before,
        } = self.turn.take().ok_or(Error::NoTurn)?;
        stats.stop = Stop::Cancel;
        match phase {
            Phase::Prompt {
                prompt,
                done: 0,
                pending,
            } => {
                self.pending.extend_from_slice(&prompt[..pending]);
                self.conversation.messages.truncate(messages_before);
            }
            Phase::Prompt { prompt, done, .. } => {
                self.pending.extend_from_slice(&prompt[done..]);
                self.close_message(self.opts.thinking, false);
            }
            Phase::Answer { mut answer, next } => {
                self.close_answer(&mut answer, next, &mut on_event);
                self.settle(&mut answer.spec);
            }
        }
        Ok(stats)
    }

    fn choose_tokens(
        &mut self,
        mut stats: TurnStats,
        mut a: Answer,
        mut logits: Vec<f32>,
        sampler: &mut Sampler,
        on_event: &mut impl FnMut(Event<'_>),
        messages_before: usize,
    ) -> Result<Step, Error> {
        let first = stats.tokens.len();
        loop {
            let next = match self.choose(&mut logits, &a.spec, sampler) {
                Ok(next) => next,
                Err(e) => {
                    self.close_message(a.thinking, a.call.is_some());
                    self.settle(&mut a.spec);
                    return Err(e);
                }
            };
            stats.gen_tokens += 1;
            stats.tokens.push(next);
            self.generated.push(next);
            sampler.accept(next);
            on_event(Event::Token(next));

            if next == self.format.turn_end || self.tok.eog.contains(&next) {
                if let Some(text) = a.call.take() {
                    on_event(Event::Answer(&format!("<tool_call>{text}")));
                }
                self.pending.push(self.format.turn_end);
                stats.stop = Stop::End;
                return Ok(self.finish(stats, a, first, messages_before));
            }
            if self.format.reasoning_end == Some(next) && a.thinking {
                a.thinking = false;
                a.skip_leading_newlines = true;
                a.held_newlines.clear();
                on_event(Event::ThinkingDone);
            } else if self.format.tool_call.map(|(open, _)| open) == Some(next)
                && !a.thinking
                && a.call.is_none()
            {
                a.held_newlines.clear();
                a.call = Some(String::new());
            } else if self.format.tool_call.map(|(_, close)| close) == Some(next)
                && a.call.is_some()
            {
                let text = a.call.take().unwrap_or_default();
                match ToolCall::parse(&text) {
                    Some(c) => {
                        on_event(Event::ToolCall(&c));
                        stats.tool_calls.push(c);
                    }
                    None => on_event(Event::Answer(&format!("<tool_call>{text}</tool_call>"))),
                }
                a.skip_leading_newlines = true;
            } else {
                let mut text = a.decoder.push(self.tok.token_bytes(next));
                if let Some(c) = &mut a.call {
                    c.push_str(&text);
                } else {
                    if a.skip_leading_newlines {
                        let trimmed = text.trim_start_matches('\n');
                        if !trimmed.is_empty() {
                            a.skip_leading_newlines = false;
                        }
                        text = trimmed.to_string();
                    }
                    let body = text.trim_end_matches('\n');
                    if !body.is_empty() {
                        let shown = core::mem::take(&mut a.held_newlines) + body;
                        on_event(if a.thinking {
                            Event::Thinking(&shown)
                        } else {
                            Event::Answer(&shown)
                        });
                    }
                    a.held_newlines.push_str(&text[body.len()..]);
                }
            }

            let full = self.session.n_past() - a.spec.drafts.len() + CLOSE_ROOM_TOKENS
                >= self.opts.context;
            let reason = if stats.gen_tokens >= self.opts.max_tokens {
                Some(Stop::MaxTokens)
            } else if full {
                Some(Stop::ContextFull)
            } else {
                None
            };
            if let Some(reason) = reason {
                stats.stop = reason;
                self.close_answer(&mut a, next, on_event);
                return Ok(self.finish(stats, a, first, messages_before));
            }
            if a.spec.drafts.front() == Some(&next) {
                logits = self.advance(next, &mut a.spec, &mut stats, sampler);
                continue;
            }
            let tokens = stats.tokens[first..].to_vec();
            self.turn = Some(Turn {
                stats,
                phase: Phase::Answer { answer: a, next },
                messages_before,
            });
            return Ok(Step::Tokens(tokens));
        }
    }

    /// Build the [`Said::Assistant`] from the generated tokens
    fn answer_said(&self, stats: &TurnStats) -> Said {
        let tokens = &stats.tokens;
        let end = if tokens
            .last()
            .is_some_and(|&t| t == self.format.turn_end || self.tok.eog.contains(&t))
        {
            tokens.len() - 1
        } else {
            tokens.len()
        };
        let content = &tokens[..end];

        // when thinking is on, split the tokens at the reasoning end
        // token; when thinking is off, the model did not reason
        let (reasoning, answer) = if self.opts.thinking {
            if let Some(end_id) = self.format.reasoning_end {
                if let Some(pos) = content.iter().position(|&t| t == end_id) {
                    let r = self.tok.decode(&content[..pos]);
                    let after = &content[pos + 1..];
                    let a = self.tok.decode(after);
                    let a = a.trim_start_matches('\n').to_string();
                    (Some(r), a)
                } else {
                    // the reasoning did not end (truncated while thinking)
                    let r = self.tok.decode(content);
                    (Some(r), String::new())
                }
            } else {
                (None, self.tok.decode(content))
            }
        } else {
            (None, self.tok.decode(content))
        };

        let answer = if !stats.tool_calls.is_empty() {
            String::new()
        } else {
            answer
        };

        Said::Assistant {
            reasoning,
            answer,
            tool_calls: stats.tool_calls.clone(),
        }
    }

    fn finish(
        &mut self,
        stats: TurnStats,
        mut a: Answer,
        first: usize,
        _messages_before: usize,
    ) -> Step {
        self.settle(&mut a.spec);
        self.conversation.messages.push(self.answer_said(&stats));
        Step::Done(stats.tokens[first..].to_vec(), stats)
    }

    fn close_answer(
        &mut self,
        a: &mut Answer,
        next: TokenId,
        on_event: &mut impl FnMut(Event<'_>),
    ) {
        self.pending.push(next);
        let call_open = a.call.is_some();
        if let Some(text) = a.call.take() {
            on_event(Event::Answer(&format!("<tool_call>{text}")));
        }
        self.close_message(a.thinking, call_open);
    }

    fn close_message(&mut self, thinking: bool, call_open: bool) {
        if thinking {
            if let Some(id) = self.format.reasoning_end {
                self.pending.push(id);
            }
        }
        if call_open {
            if let Some((_, close)) = self.format.tool_call {
                self.pending.push(close);
            }
        }
        self.pending.push(self.format.turn_end);
    }

    fn choose(
        &self,
        logits: &mut [f32],
        spec: &Speculation,
        sampler: &mut Sampler,
    ) -> Result<TokenId, Error> {
        let (Some(&draft), Some(Some(q))) = (spec.drafts.front(), spec.q.front()) else {
            return sampler.sample(logits, &self.generated);
        };
        let p = normalized(sampler.distribution(logits, &self.generated)?);
        Ok(speculative_choice(&p, q, draft, sampler))
    }

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

    fn propose(
        &mut self,
        next: TokenId,
        room: usize,
        sampler: &mut Sampler,
    ) -> (Vec<TokenId>, Vec<Option<Dist>>, bool) {
        if sampler.masked() {
            return (Vec::new(), Vec::new(), false);
        }
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

type Dist = Vec<(TokenId, f32)>;

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

fn normalized(mut d: Dist) -> Dist {
    let sum: f32 = d.iter().map(|c| c.1).sum();
    for c in &mut d {
        c.1 /= sum;
    }
    d
}

const LOOKUP_STRONG: usize = 8;
const LOOKUP_MIN: usize = 3;
const LOOKUP_AGREE: usize = 2;
const LOOKUP_SHORT: usize = 7;
const LOOKUP_RUN: usize = 2;
const LOOKUP_MAX_MATCH: usize = 32;

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
    let mut c: Vec<TokenId> = Vec::with_capacity(n);
    for i in 0..n {
        let p = at + 1 + i;
        c.push(if p < len { history[p] } else { c[p - len] });
    }
    Some((best, c))
}

#[derive(Default)]
struct Speculation {
    drafts: VecDeque<TokenId>,
    q: VecDeque<Option<Dist>>,
    logits: VecDeque<Vec<f32>>,
    kept: usize,
    proposed: usize,
    looked_up: bool,
}

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
    use super::{continuation, speculative_choice};
    use crate::prelude::*;
    use crate::sampler::{Sampler, SamplerConfig};

    #[test]
    fn speculative_choice_follows_the_target() {
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
        let a = accepted as f64 / n as f64;
        assert!((a - 0.5).abs() < 0.005, "acceptance {a}");
    }

    #[test]
    fn continuation_of_the_longest_latest_match() {
        let h = [7, 1, 2, 3, 4, 5, 8, 9, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5];
        assert_eq!(continuation(&h, 2), Some((5, vec![8, 9])));
        assert_eq!(continuation(&[1, 2, 3, 9, 2, 3], 2), None);
        assert_eq!(continuation(&[5], 2), None);
    }

    #[test]
    fn continuation_overlapping_its_match() {
        let h = [0, 1, 0, 1, 0];
        assert_eq!(continuation(&h, 3), Some((3, vec![1, 0, 1])));
    }
}
