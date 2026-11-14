//! The counts of the passes of the batch scheduler. Only the tokenizer comes
//! from the model file; a scripted engine chooses each token, so the tests
//! need no GPU, but they are ignored without a model file:
//! `BIV_TEST_MODEL=<gguf> cargo test --release --test batch_passes -- --ignored`

mod common;

use std::thread;
use std::time::Duration;

use biv::batch::{self, BatchOptions, Passes};
use biv::chat::{Chat, ChatOptions, Event};
use biv::engine::Engine;
use biv::tokenizer::{TokenId, Tokenizer};
use common::{NO_DRAFT, Script, model_path, sampler};

/// The answer of each of `chats` chats at the same time with drafts, on a
/// batch whose passes take at most `decode_rows` decode rows, and the
/// passes after them
fn drafted_answers(
    engine: &Script,
    tok: &Tokenizer,
    chats: usize,
    decode_rows: usize,
) -> (Vec<Vec<TokenId>>, Passes) {
    let message = WORD.repeat(MESSAGE_WORDS);
    let options = ChatOptions {
        draft: Some(DRAFTS),
        ..options()
    };
    let batch = BatchOptions {
        decode_rows,
        ..batch_options()
    };
    batch::serve(engine, &batch, |handle| {
        let answers = thread::scope(|scope| {
            let threads: Vec<_> = (0..chats)
                .map(|_| {
                    scope.spawn(|| {
                        let engine: &dyn Engine = &handle;
                        let mut chat = Chat::new(engine, tok, options.clone()).expect("chat");
                        chat.send(&message, &mut sampler(), |_: Event<'_>| {}, || false)
                            .expect("send")
                            .tokens
                    })
                })
                .collect();
            threads
                .into_iter()
                .map(|t| t.join().expect("a chat"))
                .collect()
        });
        (answers, handle.passes())
    })
    .expect("serve")
}

/// A text that the tokenizer gives as one token
const WORD: &str = " word";
/// The words of the message, more than one pass of prompt tokens
const MESSAGE_WORDS: usize = 100;
/// The tokens that the scripted engine can choose in one session
const SCRIPT_TOKENS: usize = 16;
/// The per-answer cap, below the script, so that it stops each turn
const MAX_TOKENS: usize = 6;
/// The prompt tokens of each pass of these tests
const BATCH: usize = 32;
/// A context far above any turn of these tests
const CONTEXT: usize = 4096;
/// The batch of the scheduler: the tokens and the logits of one pass
const PASS_TOKENS: usize = 64;
const PASS_LOGITS: usize = 8;
/// The wait of the scheduler for the sessions of the last pass
const PASS_WAIT: Duration = Duration::from_millis(5);
/// The chats of the test of two chats
const TWO_CHATS: usize = 2;
/// The decode rows of a pass in the test of the limit: the two cost levels
/// of the Metal matmul kernels, the rows that one `mmv` threadgroup takes and
/// the most rows of `mmv`
#[cfg(target_os = "macos")]
const DECODE_ROW_LIMITS: [usize; 2] = [biv::metal::MMV_ROWS, biv::metal::MMV_MAX_ROWS];
/// The chats at a time in the test of the limit
const BUSY_CHATS: [usize; 5] = [1, 2, 4, 8, 16];
/// The drafts that each chat asks for in the test of the limit
const DRAFTS: usize = 7;
/// The tokens of the script in the test of the limit: the answer and the
/// drafts after its last token
const DRAFT_SCRIPT_TOKENS: usize = 64;

fn options() -> ChatOptions {
    ChatOptions {
        system: None,
        thinking: false,
        context: CONTEXT,
        max_tokens: MAX_TOKENS,
        batch: Some(BATCH),
        draft: Some(NO_DRAFT),
        lookup: 0,
        ..ChatOptions::default()
    }
}

fn batch_options() -> BatchOptions {
    BatchOptions {
        tokens: Some(PASS_TOKENS),
        logits: PASS_LOGITS,
        wait: PASS_WAIT,
        decode_rows: PASS_TOKENS,
    }
}

fn word(tok: &Tokenizer) -> TokenId {
    let ids = tok.encode(WORD);
    assert_eq!(ids.len(), 1, "{WORD:?} is one token");
    ids[0]
}

/// One turn of `chats` chats at the same time, each on a thread of its own,
/// and the passes of the scheduler after them
fn passes_of(engine: &Script, tok: &Tokenizer, chats: usize) -> Passes {
    let message = WORD.repeat(MESSAGE_WORDS);
    batch::serve(engine, &batch_options(), |handle| {
        thread::scope(|scope| {
            for _ in 0..chats {
                scope.spawn(|| {
                    let engine: &dyn Engine = &handle;
                    let mut chat = Chat::new(engine, tok, options()).expect("chat");
                    chat.send(&message, &mut sampler(), |_: Event<'_>| {}, || false)
                        .expect("send");
                });
            }
        });
        handle.passes()
    })
    .expect("serve")
}

/// The passes count each token that the sessions took. With one chat no
/// pass holds the tokens of two sequences, and each pass holds one.
#[test]
#[ignore = "needs a model file"]
fn the_passes_of_one_chat_hold_its_tokens_and_none_is_shared() {
    let gguf = biv_cli::open_gguf(model_path()).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let engine = Script::new(vec![word(&tok); SCRIPT_TOKENS], tok.n_vocab());

    let passes = passes_of(&engine, &tok, 1);

    assert_eq!(passes.tokens, engine.fed.borrow().len() as u64);
    assert_eq!(passes.shared, 0);
    assert!(passes.passes > 1, "the prompt is more than one pass");
    // each pass holds the one sequence, and the scripted engine drafts nothing
    assert_eq!(passes.sequences, passes.passes);
    assert_eq!((passes.drafts, passes.draft_tokens), (0, 0));
    assert!(passes.pass_time > Duration::ZERO);
}

/// With two chats the passes hold the tokens of both, the same tokens as
/// two turns alone, in at most as many passes
#[test]
#[ignore = "needs a model file"]
fn the_passes_of_two_chats_hold_the_tokens_of_both() {
    let gguf = biv_cli::open_gguf(model_path()).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let alone_engine = Script::new(vec![word(&tok); SCRIPT_TOKENS], tok.n_vocab());
    let alone = passes_of(&alone_engine, &tok, 1);
    let engine = Script::new(vec![word(&tok); SCRIPT_TOKENS], tok.n_vocab());

    let passes = passes_of(&engine, &tok, TWO_CHATS);

    assert_eq!(passes.tokens, engine.fed.borrow().len() as u64);
    assert_eq!(passes.tokens, TWO_CHATS as u64 * alone.tokens);
    assert!(passes.shared <= passes.passes);
    assert!(passes.passes <= TWO_CHATS as u64 * alone.passes);
    // a pass holds one or two sequences, and a shared pass holds two
    assert_eq!(passes.sequences, passes.passes + passes.shared);
    assert_eq!(passes.sequences, TWO_CHATS as u64 * alone.sequences);
}

/// With drafts, no pass holds more decode rows than the limit of the batch,
/// whatever the chats at a time, and each chat gets the answer of a chat
/// alone: the scripted engine has no rounding
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs a model file"]
fn the_decode_rows_of_a_pass_stay_within_the_limit() {
    let gguf = biv_cli::open_gguf(model_path()).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let script = vec![word(&tok); DRAFT_SCRIPT_TOKENS];
    for limit in DECODE_ROW_LIMITS {
        let alone_engine = Script::new(script.clone(), tok.n_vocab());
        let (alone, alone_passes) = drafted_answers(&alone_engine, &tok, 1, limit);
        assert!(alone_passes.drafts > 0, "a chat alone drafts");
        for chats in BUSY_CHATS {
            let engine = Script::new(script.clone(), tok.n_vocab());
            let (answers, passes) = drafted_answers(&engine, &tok, chats, limit);
            assert!(
                passes.widest_decode <= limit as u64,
                "{chats} chats, limit {limit}: a pass of {} decode rows",
                passes.widest_decode
            );
            for answer in answers {
                assert_eq!(answer, alone[0], "{chats} chats, limit {limit}");
            }
        }
    }
}
