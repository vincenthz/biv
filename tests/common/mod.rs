//! What the chat tests share: the model file, and an engine whose tokens are
//! scripted, so that a test needs only the tokenizer of the model file and
//! no GPU.

#![allow(dead_code, reason = "each test crate uses a part of this module")]

use std::cell::RefCell;

use biv::Error;
use biv::engine::{DraftLogits, Engine, Session, Snapshot};
use biv::memory::{ModelMemory, SessionMemory};
use biv::sampler::{Sampler, SamplerConfig};
use biv::tokenizer::TokenId;

/// The variable that names the GGUF file
pub const MODEL_ENV: &str = "BIV_TEST_MODEL";
/// Prompt tokens per prefill of the scripted engine
const SCRIPT_BATCH: usize = 64;
/// The scripted engine drafts nothing
pub const NO_DRAFT: usize = 0;
/// A greedy sampler: the one-hot logits of the script choose each token
const GREEDY: f32 = 0.0;
/// The logit of the scripted token; every other token has 0
const CHOSEN_LOGIT: f32 = 1.0;

pub fn model_path() -> String {
    std::env::var(MODEL_ENV).unwrap_or_else(|_| panic!("{MODEL_ENV} must name a GGUF file"))
}

/// An engine whose sessions choose the tokens of `script`, in order: the
/// logits of each forward, and of each token of a verify, put the next token
/// of the script first. Its drafts are the next tokens of the script, so
/// each draft is right.
pub struct Script {
    pub tokens: Vec<TokenId>,
    pub vocab: usize,
    /// every token that its sessions took, prefilled or forwarded, in order
    pub fed: RefCell<Vec<TokenId>>,
}

struct ScriptSession<'s> {
    script: &'s Script,
    n_past: usize,
    /// the tokens of the script that the logits gave so far
    given: usize,
    /// `n_past` and `given` before the last verify, until its keep
    verified_from: Option<(usize, usize)>,
}

impl Script {
    pub fn new(tokens: Vec<TokenId>, vocab: usize) -> Self {
        Script {
            tokens,
            vocab,
            fed: RefCell::default(),
        }
    }

    /// A session that holds `n_past` tokens, as a replay would rebuild one
    pub fn session_at(&self, n_past: usize) -> Box<dyn Session + '_> {
        Box::new(ScriptSession {
            script: self,
            n_past,
            given: 0,
            verified_from: None,
        })
    }
}

impl Engine for Script {
    fn new_session(&self, _context: usize, _draft: usize) -> Result<Box<dyn Session + '_>, Error> {
        Ok(self.session_at(0))
    }

    fn batch_size(&self) -> usize {
        SCRIPT_BATCH
    }

    fn draft_size(&self) -> usize {
        NO_DRAFT
    }

    fn description(&self) -> String {
        "script".into()
    }

    fn numerics(&self) -> Vec<(&'static str, String)> {
        Vec::new()
    }

    fn model_memory(&self) -> ModelMemory {
        ModelMemory::default()
    }

    fn session_memory(&self, _context: usize, _draft: usize) -> SessionMemory {
        SessionMemory::default()
    }
}

impl ScriptSession<'_> {
    /// Logits that put the next token of the script first
    fn next_logits(&mut self) -> Vec<f32> {
        let token = *self
            .script
            .tokens
            .get(self.given)
            .expect("the script holds a token for each logits");
        self.given += 1;
        let mut logits = vec![0.0; self.script.vocab];
        logits[token as usize] = CHOSEN_LOGIT;
        logits
    }
}

impl Session for ScriptSession<'_> {
    fn forward(&mut self, tokens: &[TokenId]) -> Vec<f32> {
        self.n_past += tokens.len();
        self.script.fed.borrow_mut().extend_from_slice(tokens);
        self.next_logits()
    }

    fn prefill(&mut self, tokens: &[TokenId]) {
        self.n_past += tokens.len();
        self.script.fed.borrow_mut().extend_from_slice(tokens);
    }

    /// The `n` tokens of the script after `next`, the token of the last
    /// logits
    fn draft(
        &mut self,
        _next: TokenId,
        n: usize,
        _pick: &mut dyn FnMut(&DraftLogits<'_>) -> TokenId,
    ) -> Vec<TokenId> {
        self.script.tokens[self.given..self.given + n].to_vec()
    }

    fn verify(&mut self, tokens: &[TokenId]) -> Vec<Vec<f32>> {
        self.verified_from = Some((self.n_past, self.given));
        self.n_past += tokens.len();
        self.script.fed.borrow_mut().extend_from_slice(tokens);
        tokens.iter().map(|_| self.next_logits()).collect()
    }

    fn keep(&mut self, n: usize) {
        let (n_past, given) = self.verified_from.take().expect("a keep follows a verify");
        self.n_past = n_past + n;
        self.given = given + n;
    }

    fn n_past(&self) -> usize {
        self.n_past
    }

    fn memory(&self) -> SessionMemory {
        SessionMemory::default()
    }

    fn snapshot(&self) -> Result<Snapshot, Error> {
        unreachable!("these tests take no snapshot")
    }

    fn restore(&mut self, _snapshot: &Snapshot) -> Result<(), Error> {
        unreachable!("these tests restore no snapshot")
    }
}

/// A greedy sampler, which takes the token of the script
pub fn sampler() -> Sampler {
    Sampler::new(SamplerConfig {
        temperature: GREEDY,
        ..SamplerConfig::default()
    })
}
