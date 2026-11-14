//! A chat that goes on from the messages of an earlier conversation. Only
//! the tokenizer comes from the model file; a scripted engine chooses each
//! token, so the test needs no GPU, but it is ignored without a model file:
//! `BIV_TEST_MODEL=<gguf> cargo test --release --test chat_history -- --ignored`

mod common;

use biv::chat::{Chat, ChatOptions, Message, ReasoningEffort, Step, Stop};
use biv::tokenizer::{PromptPart, TokenId, Tokenizer};
use common::{NO_DRAFT, Script, model_path, sampler};

use PromptPart::{Content, Template};

const CONTEXT: usize = 4096;
/// The answer of the turn: one token, enough to see the prompt
const ONE_TOKEN: usize = 1;
const FIRST: &str = "Name a prime number.";
const ANSWER: &str = "Seven.";
const SECOND: &str = "And the next one?";
/// A text that the tokenizer gives as one token, the answer of the script
const WORD: &str = " word";
/// The reasoning of an earlier answer. It starts with a newline, so its
/// text joins the newline at the end of the generation prompt unless the
/// two are tokenized apart.
const REASONING: &str = "\nSeven has no divisor but 1 and itself.\n";
/// The answer after the reasoning. It starts with the newlines that the
/// model writes after `</think>`, which the answer text of the events drops.
const REASONED_ANSWER: &str = "\n\nSeven.";
/// The answer room of the chats that reason
const REASONED_ROOM: usize = 64;
const THINK_END: &str = "</think>";
const IM_END: &str = "<|im_end|>";

/// The chat template of Qwen3.8 for a user message, an earlier answer with
/// no reasoning, and the next user message, with thinking on and no system
/// block (reasoning effort medium gives no reasoning instructions)
fn template(tok: &Tokenizer) -> Vec<biv::tokenizer::TokenId> {
    tok.encode_prompt(&[
        Template("<|im_start|>user\n"),
        Content(FIRST),
        Template("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        Content(ANSWER),
        Template("<|im_end|>\n<|im_start|>user\n"),
        Content(SECOND),
        Template("<|im_end|>\n<|im_start|>assistant\n<think>\n"),
    ])
}

/// Thinking on and no system block (reasoning effort medium gives no
/// reasoning instructions), so the prompt is `template`
fn options() -> ChatOptions {
    options_with_room(ONE_TOKEN)
}

/// The options of [`options`] with an answer room of `max_tokens`
fn options_with_room(max_tokens: usize) -> ChatOptions {
    ChatOptions {
        system: None,
        thinking: true,
        effort: ReasoningEffort::Medium,
        context: CONTEXT,
        max_tokens,
        draft: Some(NO_DRAFT),
        lookup: 0,
        tools: Vec::new(),
        ..ChatOptions::default()
    }
}

#[test]
#[ignore = "needs a model file"]
fn a_chat_takes_the_messages_of_an_earlier_conversation_as_the_template_writes_them() {
    let gguf = biv_cli::open_gguf(model_path()).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let word = tok.encode(WORD);
    let engine = Script::new(word, tok.n_vocab());
    let mut chat = Chat::new(&engine, &tok, options()).expect("chat");
    let history = [Message::User(FIRST), Message::Assistant(ANSWER)];
    let stats = chat
        .send_with_history(&history, SECOND, &mut sampler(), |_| {}, || false)
        .expect("turn");
    let expected = template(&tok);
    assert_eq!(stats.prompt_tokens, expected.len());
    assert_eq!(*engine.fed.borrow(), expected);
}

#[test]
#[ignore = "needs a model file"]
fn an_open_turn_gives_its_prompt_tokens_before_the_prompt_pass() {
    let gguf = biv_cli::open_gguf(model_path()).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let word = tok.encode(WORD);
    let engine = Script::new(word, tok.n_vocab());
    let mut chat = Chat::new(&engine, &tok, options()).expect("chat");
    assert_eq!(chat.turn_prompt_tokens(), None);
    let history = [Message::User(FIRST), Message::Assistant(ANSWER)];
    chat.begin_with_history(&history, SECOND).expect("turn");
    assert_eq!(chat.turn_prompt_tokens(), Some(template(&tok).len()));
    assert!(engine.fed.borrow().is_empty(), "no prompt pass yet");
    let mut sampler = sampler();
    while !matches!(
        chat.step(&mut sampler, |_| {}).expect("step"),
        Step::Done(..)
    ) {}
    assert_eq!(chat.turn_prompt_tokens(), None);
}

fn special(tok: &Tokenizer, text: &str) -> TokenId {
    tok.token_id(text)
        .unwrap_or_else(|| panic!("the model has the token {text}"))
}

/// The tokens that a chat took when it answered FIRST with `script`, then
/// SECOND with the next tokens of the script, all in one session. It checks
/// that the first turn stopped as `stop` says.
fn reused_chat_tokens(
    tok: &Tokenizer,
    script: Vec<TokenId>,
    room: usize,
    stop: Stop,
) -> Vec<TokenId> {
    let engine = Script::new(script, tok.n_vocab());
    let mut chat = Chat::new(&engine, tok, options_with_room(room)).expect("chat");
    let first = chat
        .send(FIRST, &mut sampler(), |_| {}, || false)
        .expect("first turn");
    assert_eq!(first.stop, stop);
    chat.send(SECOND, &mut sampler(), |_| {}, || false)
        .expect("second turn");
    engine.fed.take()
}

/// The tokens that a new chat took when it got FIRST and `answer` as
/// history, then answered SECOND with WORD and the end of the message
fn history_chat_tokens(tok: &Tokenizer, answer: Message<'_>, room: usize) -> Vec<TokenId> {
    let script = [tok.encode(WORD), vec![special(tok, IM_END)]].concat();
    let engine = Script::new(script, tok.n_vocab());
    let mut chat = Chat::new(&engine, tok, options_with_room(room)).expect("chat");
    chat.send_with_history(
        &[Message::User(FIRST), answer],
        SECOND,
        &mut sampler(),
        |_| {},
        || false,
    )
    .expect("turn");
    engine.fed.take()
}

#[test]
#[ignore = "needs a model file"]
fn a_chat_given_an_answer_with_its_reasoning_holds_the_tokens_of_the_chat_that_wrote_it() {
    let gguf = biv_cli::open_gguf(model_path()).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let reasoning = tok.encode(REASONING);
    assert_ne!(
        tok.encode(&format!("\n{REASONING}")),
        [tok.encode("\n"), reasoning.clone()].concat(),
        "the reasoning joins the newline of the prompt when the two are tokenized whole"
    );
    let script = [
        reasoning,
        vec![special(&tok, THINK_END)],
        tok.encode(REASONED_ANSWER),
        vec![special(&tok, IM_END)],
        tok.encode(WORD),
        vec![special(&tok, IM_END)],
    ]
    .concat();
    let reused = reused_chat_tokens(&tok, script, REASONED_ROOM, Stop::End);
    let answer = Message::AssistantWithReasoning {
        reasoning: REASONING,
        answer: REASONED_ANSWER,
    };
    assert_eq!(history_chat_tokens(&tok, answer, REASONED_ROOM), reused);
}

#[test]
#[ignore = "needs a model file"]
fn a_chat_given_an_answer_that_stopped_in_its_reasoning_holds_the_tokens_of_the_chat_that_wrote_it()
{
    let gguf = biv_cli::open_gguf(model_path()).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let reasoning = tok.encode(REASONING);
    // the first turn stops at its room, in the reasoning
    let room = reasoning.len();
    let script = [reasoning, tok.encode(WORD), vec![special(&tok, IM_END)]].concat();
    let reused = reused_chat_tokens(&tok, script, room, Stop::MaxTokens);
    let answer = Message::AssistantWithReasoning {
        reasoning: REASONING,
        answer: "",
    };
    assert_eq!(history_chat_tokens(&tok, answer, room), reused);
}
