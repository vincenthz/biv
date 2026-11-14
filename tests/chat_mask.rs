//! A token mask on the sampler of a chat. Only the tokenizer comes from the
//! model file; a scripted engine offers each token, so the tests need no
//! GPU, but they are ignored without a model file:
//! `BIV_TEST_MODEL=<gguf> cargo test --release --test chat_mask -- --ignored`

mod common;

use std::cell::RefCell;
use std::rc::Rc;

use biv::Error;
use biv::chat::{Chat, ChatOptions, Event, Stop};
use biv::sampler::TokenMask;
use biv::tokenizer::{TokenId, Tokenizer};
use common::{NO_DRAFT, Script, model_path, sampler};

const MESSAGE: &str = "Say a word.";
/// The token that the script offers at each step
const OFFERED: &str = " word";
/// The token that the mask allows in its place
const ALLOWED: &str = " other";
/// The allowed tokens before the mask allows only the end of the message
const ALLOWED_STEPS: usize = 3;
/// The tokens that the scripted engine can offer in one session, more than
/// the turn generates
const SCRIPT_TOKENS: usize = 16;
const CONTEXT: usize = 4096;
const MAX_TOKENS: usize = SCRIPT_TOKENS;
/// Prompt lookup is on: a proposal while the mask is set would reach
/// verify, which the scripted session refuses
const LOOKUP: usize = 8;

/// A mask that allows ALLOWED for ALLOWED_STEPS tokens, then only the end of
/// the message, and records each token that it is told of
struct Steps {
    allowed: TokenId,
    end: TokenId,
    advanced: Rc<RefCell<Vec<TokenId>>>,
}

impl TokenMask for Steps {
    fn mask(&self, logits: &mut [f32]) {
        let next = if self.advanced.borrow().len() < ALLOWED_STEPS {
            self.allowed
        } else {
            self.end
        };
        for (t, l) in logits.iter_mut().enumerate() {
            if t as TokenId != next {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    fn advance(&mut self, token: TokenId) {
        self.advanced.borrow_mut().push(token);
    }
}

/// A mask that allows no token
struct Nothing;

impl TokenMask for Nothing {
    fn mask(&self, logits: &mut [f32]) {
        logits.fill(f32::NEG_INFINITY);
    }

    fn advance(&mut self, _token: TokenId) {}
}

fn options() -> ChatOptions {
    ChatOptions {
        system: None,
        thinking: false,
        context: CONTEXT,
        max_tokens: MAX_TOKENS,
        draft: Some(NO_DRAFT),
        lookup: LOOKUP,
        tools: Vec::new(),
        ..ChatOptions::default()
    }
}

fn one(tok: &Tokenizer, text: &str) -> TokenId {
    let ids = tok.encode(text);
    assert_eq!(ids.len(), 1, "{text:?} is one token");
    ids[0]
}

#[test]
#[ignore = "needs a model file"]
fn a_masked_chat_chooses_only_the_tokens_that_the_mask_allows() {
    let gguf = biv_cli::open_gguf(model_path()).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let offered = one(&tok, OFFERED);
    let allowed = one(&tok, ALLOWED);
    let end = tok
        .token_id("<|im_end|>")
        .expect("the tokenizer has <|im_end|>");
    let engine = Script::new(vec![offered; SCRIPT_TOKENS], tok.n_vocab());
    let mut chat = Chat::new(&engine, &tok, options()).expect("chat");
    let advanced = Rc::new(RefCell::new(Vec::new()));
    let mut s = sampler();
    s.set_mask(Box::new(Steps {
        allowed,
        end,
        advanced: Rc::clone(&advanced),
    }));
    let stats = chat
        .send(MESSAGE, &mut s, |_: Event<'_>| {}, || false)
        .expect("turn");

    let mut expected = vec![allowed; ALLOWED_STEPS];
    expected.push(end);
    assert_eq!(stats.tokens, expected, "the offered token is never chosen");
    assert_eq!(
        *advanced.borrow(),
        expected,
        "the mask is told each chosen token"
    );
    assert_eq!(stats.drafted, 0, "nothing is drafted while the mask is set");
    assert!(matches!(stats.stop, Stop::End));
}

#[test]
#[ignore = "needs a model file"]
fn a_mask_that_allows_no_token_ends_the_turn_with_a_named_error() {
    let gguf = biv_cli::open_gguf(model_path()).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let engine = Script::new(vec![one(&tok, OFFERED); SCRIPT_TOKENS], tok.n_vocab());
    let mut chat = Chat::new(&engine, &tok, options()).expect("chat");
    let mut s = sampler();
    s.set_mask(Box::new(Nothing));
    let turn = chat.send(MESSAGE, &mut s, |_: Event<'_>| {}, || false);
    assert!(matches!(turn, Err(Error::MaskAllowsNoToken)));
    let end = tok
        .token_id("<|im_end|>")
        .expect("the tokenizer has <|im_end|>");
    assert_eq!(chat.pending(), [end], "the open message is closed");
}
