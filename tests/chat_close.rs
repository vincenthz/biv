//! The close of a turn that stops at the end of the context. Only the
//! tokenizer comes from the model file; a scripted engine chooses each token,
//! so the tests need no GPU, but they are ignored without a model file:
//! `BIV_TEST_MODEL=<gguf> cargo test --release --test chat_close -- --ignored`

mod common;

use biv::Error;
use biv::chat::{CLOSE_ROOM_TOKENS, Chat, ChatOptions, Event, Stop, TurnStats};
use biv::tokenizer::{TokenId, Tokenizer};
use common::{NO_DRAFT, Script, model_path, sampler};

const MESSAGE: &str = "What time is it?";
/// A text that the tokenizer gives as one token
const WORD: &str = " word";
/// A function that the model can call
const TOOL: &str = r#"{"type": "function", "function": {"name": "get_time", "description": "The time now", "parameters": {"type": "object", "properties": {}}}}"#;
/// The context of the turn that measures the prompt: far above any prompt
const WIDE_CONTEXT: usize = 4096;
/// The answer of the turn that measures the prompt: one token
const PROBE_TOKENS: usize = 1;
/// The tokens that the scripted engine can choose in one session, more than
/// any turn of these tests generates
const SCRIPT_TOKENS: usize = 16;
/// The per-answer cap, above the script, so that only the context stops a turn
const MAX_TOKENS: usize = SCRIPT_TOKENS + 1;
/// The tokens that a turn generates before the context leaves only the room
/// of the close
const STEPS_BEFORE_THE_CLOSE: usize = 2;

fn options(context: usize, thinking: bool, tools: Vec<String>) -> ChatOptions {
    ChatOptions {
        system: None,
        thinking,
        context,
        max_tokens: MAX_TOKENS,
        draft: Some(NO_DRAFT),
        lookup: 0,
        tools,
        ..ChatOptions::default()
    }
}

fn send(chat: &mut Chat<'_>) -> Result<TurnStats, Error> {
    chat.send(MESSAGE, &mut sampler(), |_: Event<'_>| {}, || false)
}

fn id(tok: &Tokenizer, text: &str) -> TokenId {
    tok.token_id(text)
        .unwrap_or_else(|| panic!("the tokenizer has {text}"))
}

fn word(tok: &Tokenizer) -> TokenId {
    let ids = tok.encode(WORD);
    assert_eq!(ids.len(), 1, "{WORD:?} is one token");
    ids[0]
}

/// The prompt tokens of the first turn with `thinking` and `tools`
fn prompt_tokens(engine: &Script, tok: &Tokenizer, thinking: bool, tools: &[String]) -> usize {
    let opts = ChatOptions {
        max_tokens: PROBE_TOKENS,
        ..options(WIDE_CONTEXT, thinking, tools.to_vec())
    };
    let mut chat = Chat::new(engine, tok, opts).expect("chat");
    send(&mut chat).expect("turn").prompt_tokens
}

/// One turn of `script` in a context with STEPS_BEFORE_THE_CLOSE steps above
/// the prompt and the room of the close. The turn stops at the end of the
/// context with its close in the context, and a chat goes on from it.
fn close_at_the_end_of_the_context(
    tok: &Tokenizer,
    script: Vec<TokenId>,
    thinking: bool,
    tools: Vec<String>,
) -> Vec<TokenId> {
    let engine = Script::new(script, tok.n_vocab());
    let prompt = prompt_tokens(&engine, tok, thinking, &tools);
    let context = prompt + STEPS_BEFORE_THE_CLOSE + CLOSE_ROOM_TOKENS;
    let opts = options(context, thinking, tools);
    let mut chat = Chat::new(&engine, tok, opts.clone()).expect("chat");
    let stats = send(&mut chat).expect("turn");
    assert_eq!(stats.stop, Stop::ContextFull);
    // a step for each token below the room of the close, and the last token
    assert_eq!(stats.gen_tokens, STEPS_BEFORE_THE_CLOSE + 1);
    assert_eq!(chat.n_past() + chat.pending().len(), context);

    let pending = chat.pending().to_vec();
    let session = engine.session_at(chat.n_past());
    Chat::from_session(
        &engine,
        tok,
        opts,
        session,
        pending.clone(),
        chat.generated().to_vec(),
        chat.history().to_vec(),
        chat.conversation().clone(),
    )
    .expect("a chat goes on from the turn");
    pending
}

#[test]
#[ignore = "needs a model file"]
fn a_turn_that_fills_the_context_while_thinking_keeps_its_close_in_the_context() {
    let gguf = biv_cli::open_gguf(model_path()).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let script = vec![word(&tok); SCRIPT_TOKENS];
    let pending = close_at_the_end_of_the_context(&tok, script, true, Vec::new());
    assert_eq!(pending[1..], [id(&tok, "</think>"), id(&tok, "<|im_end|>")]);
}

#[test]
#[ignore = "needs a model file"]
fn a_turn_that_fills_the_context_in_a_call_keeps_its_close_in_the_context() {
    let gguf = biv_cli::open_gguf(model_path()).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let mut script = vec![id(&tok, "<tool_call>")];
    script.resize(SCRIPT_TOKENS, word(&tok));
    let pending = close_at_the_end_of_the_context(&tok, script, false, vec![TOOL.into()]);
    assert_eq!(
        pending[1..],
        [id(&tok, "</tool_call>"), id(&tok, "<|im_end|>")]
    );
}

/// A call opens only after the reasoning block: a <tool_call> while thinking
/// is text of the reasoning, so a close holds one block end at most
#[test]
#[ignore = "needs a model file"]
fn a_call_does_not_open_while_thinking() {
    let gguf = biv_cli::open_gguf(model_path()).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let mut script = vec![id(&tok, "<tool_call>")];
    script.resize(SCRIPT_TOKENS, word(&tok));
    let pending = close_at_the_end_of_the_context(&tok, script, true, vec![TOOL.into()]);
    assert_eq!(pending[1..], [id(&tok, "</think>"), id(&tok, "<|im_end|>")]);
}

/// A prompt that leaves less than the room of the close is refused, and a
/// prompt that leaves the room gives one token and its close
#[test]
#[ignore = "needs a model file"]
fn a_prompt_needs_the_room_of_the_close() {
    let gguf = biv_cli::open_gguf(model_path()).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let engine = Script::new(vec![word(&tok); SCRIPT_TOKENS], tok.n_vocab());
    let prompt = prompt_tokens(&engine, &tok, true, &[]);

    let short = options(prompt + CLOSE_ROOM_TOKENS - 1, true, Vec::new());
    let mut chat = Chat::new(&engine, &tok, short).expect("chat");
    assert!(matches!(send(&mut chat), Err(Error::ContextFull { .. })));

    let context = prompt + CLOSE_ROOM_TOKENS;
    let room = options(context, true, Vec::new());
    let mut chat = Chat::new(&engine, &tok, room).expect("chat");
    let stats = send(&mut chat).expect("turn");
    assert_eq!(stats.stop, Stop::ContextFull);
    assert_eq!(stats.gen_tokens, 1);
    assert_eq!(chat.n_past() + chat.pending().len(), context);
}
