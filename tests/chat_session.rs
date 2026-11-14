//! Chat::from_session on a real model, ignored without one:
//! `BIV_TEST_MODEL=<gguf> cargo test --release --test chat_session -- --ignored`
//! On macOS it runs on Metal, elsewhere on the CPU.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Mutex, PoisonError};

use biv::Error;
use biv::chat::{Chat, ChatOptions, Event, TurnStats};
use biv::engine::{DraftLogits, Engine, Session, Snapshot};
use biv::memory::{ModelMemory, SessionMemory};
use biv::model::Model;
use biv::sampler::{Sampler, SamplerConfig};
use biv::tokenizer::{TokenId, Tokenizer};

/// The variable that names the GGUF file
const MODEL_ENV: &str = "BIV_TEST_MODEL";
const CONTEXT: usize = 2048;
/// per answer, so that the test takes seconds rather than minutes
const MAX_TOKENS: usize = 48;
/// Metal prompt tokens per forward pass, as biv-cli uses
const METAL_BATCH: usize = 256;
const SEED: u64 = 0x5EED;
const SYSTEM: &str = "You are a terse assistant. Answer in one sentence.";
const MESSAGE: &str = "Name three prime numbers above 100.";
const FOLLOW_UP: &str = "Which of them is the largest?";

/// One test at a time holds the model, whatever the test threads
static MODEL: Mutex<()> = Mutex::new(());

fn model_path() -> String {
    std::env::var(MODEL_ENV).unwrap_or_else(|_| panic!("{MODEL_ENV} must name a GGUF file"))
}

#[cfg(target_os = "macos")]
fn engine<'m>(model: &'m Model<'m>) -> Box<dyn Engine + 'm> {
    Box::new(biv::metal::MetalEngine::new(model, METAL_BATCH).expect("metal engine"))
}

#[cfg(not(target_os = "macos"))]
fn engine<'m>(model: &'m Model<'m>) -> Box<dyn Engine + 'm> {
    Box::new(model)
}

/// A call that changes the state of a session
#[derive(Debug, Clone)]
enum Call {
    Prefill(Vec<TokenId>),
    Forward(Vec<TokenId>),
    /// the token, the draft length, and the tokens the pick chose
    Draft(TokenId, usize, Vec<TokenId>),
    Verify(Vec<TokenId>),
    Keep(usize),
}

type Log = Rc<RefCell<Vec<Call>>>;

/// An engine whose sessions record their calls
struct Recording<'e> {
    inner: &'e dyn Engine,
    log: Log,
}

struct RecordingSession<'e> {
    inner: Box<dyn Session + 'e>,
    log: Log,
}

impl Engine for Recording<'_> {
    fn new_session(&self, context: usize, draft: usize) -> Result<Box<dyn Session + '_>, Error> {
        Ok(Box::new(RecordingSession {
            inner: self.inner.new_session(context, draft)?,
            log: self.log.clone(),
        }))
    }

    fn batch_size(&self) -> usize {
        self.inner.batch_size()
    }

    fn draft_size(&self) -> usize {
        self.inner.draft_size()
    }

    fn description(&self) -> String {
        self.inner.description()
    }

    fn numerics(&self) -> Vec<(&'static str, String)> {
        self.inner.numerics()
    }

    fn model_memory(&self) -> ModelMemory {
        self.inner.model_memory()
    }

    fn session_memory(&self, context: usize, draft: usize) -> SessionMemory {
        self.inner.session_memory(context, draft)
    }
}

impl Session for RecordingSession<'_> {
    fn forward(&mut self, tokens: &[TokenId]) -> Vec<f32> {
        self.log.borrow_mut().push(Call::Forward(tokens.to_vec()));
        self.inner.forward(tokens)
    }

    fn prefill(&mut self, tokens: &[TokenId]) {
        self.log.borrow_mut().push(Call::Prefill(tokens.to_vec()));
        self.inner.prefill(tokens)
    }

    fn draft(
        &mut self,
        next: TokenId,
        n: usize,
        pick: &mut dyn FnMut(&DraftLogits<'_>) -> TokenId,
    ) -> Vec<TokenId> {
        let mut picked = Vec::new();
        let drafted = self.inner.draft(next, n, &mut |logits| {
            let t = pick(logits);
            picked.push(t);
            t
        });
        self.log.borrow_mut().push(Call::Draft(next, n, picked));
        drafted
    }

    fn verify(&mut self, tokens: &[TokenId]) -> Vec<Vec<f32>> {
        self.log.borrow_mut().push(Call::Verify(tokens.to_vec()));
        self.inner.verify(tokens)
    }

    fn keep(&mut self, n: usize) {
        self.log.borrow_mut().push(Call::Keep(n));
        self.inner.keep(n)
    }

    fn n_past(&self) -> usize {
        self.inner.n_past()
    }

    fn memory(&self) -> SessionMemory {
        self.inner.memory()
    }

    fn snapshot(&self) -> Result<Snapshot, Error> {
        self.inner.snapshot()
    }

    fn restore(&mut self, snapshot: &Snapshot) -> Result<(), Error> {
        self.inner.restore(snapshot)
    }
}

/// Make the calls of `log` on `session`, in order
fn replay(session: &mut dyn Session, log: &[Call]) {
    for call in log {
        match call {
            Call::Prefill(t) => session.prefill(t),
            Call::Forward(t) => {
                session.forward(t);
            }
            Call::Draft(next, n, picked) => {
                let mut picks = picked.iter();
                session.draft(*next, *n, &mut |_| {
                    *picks.next().expect("a draft picks no more than it did")
                });
            }
            Call::Verify(t) => {
                session.verify(t);
            }
            Call::Keep(n) => session.keep(*n),
        }
    }
}

fn options() -> ChatOptions {
    ChatOptions {
        system: Some(SYSTEM.into()),
        context: CONTEXT,
        max_tokens: MAX_TOKENS,
        ..ChatOptions::default()
    }
}

/// A session made as Chat::new makes one
fn new_session<'m>(engine: &'m dyn Engine, opts: &ChatOptions) -> Box<dyn Session + 'm> {
    let draft = opts.draft.unwrap_or_else(|| engine.draft_size());
    engine
        .new_session(opts.context, draft.max(opts.lookup))
        .expect("session")
}

/// The text of the answer, thinking included, and its stats
fn send(chat: &mut Chat<'_>) -> (String, TurnStats) {
    let mut sampler = Sampler::new(SamplerConfig {
        seed: SEED,
        ..SamplerConfig::default()
    });
    let mut text = String::new();
    let stats = chat
        .send(
            MESSAGE,
            &mut sampler,
            |ev| match ev {
                Event::Thinking(t) | Event::Answer(t) => text.push_str(t),
                _ => {}
            },
            || false,
        )
        .expect("turn");
    (text, stats)
}

#[test]
#[ignore = "needs a model file"]
fn chat_from_a_replayed_session() {
    let _model = MODEL.lock().unwrap_or_else(PoisonError::into_inner);
    let gguf = biv_cli::open_gguf(model_path()).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let model = Model::load(&gguf).expect("model");
    let engine = engine(&model);
    let opts = options();

    // too many pending tokens for the context
    let empty = new_session(engine.as_ref(), &opts);
    let overfull = vec![0; opts.context + 1];
    assert!(matches!(
        Chat::from_session(
            engine.as_ref(),
            &tok,
            opts.clone(),
            empty,
            overfull,
            Vec::new(),
            Vec::new(),
            biv::chat::Conversation::new(None, Vec::new()),
        ),
        Err(Error::ContextFull { .. })
    ));

    // a chat that processes its system prompt, its session calls recorded
    let rec = Recording {
        inner: engine.as_ref(),
        log: Log::default(),
    };
    let mut first = Chat::new(&rec, &tok, opts.clone()).expect("chat");
    first.save().expect("save");
    let log = rec.log.borrow().clone();
    assert!(!log.is_empty());

    // the same calls on a new session, in a chat made from it
    let mut session = new_session(engine.as_ref(), &opts);
    replay(session.as_mut(), &log);
    let history = first.history().to_vec();
    let mut second = Chat::from_session(
        engine.as_ref(),
        &tok,
        opts,
        session,
        Vec::new(),
        Vec::new(),
        history,
        first.conversation().clone(),
    )
    .expect("chat from session");
    assert_eq!(second.n_past(), first.n_past());

    // they answer the same message with the same text
    let (text_a, stats_a) = send(&mut first);
    let (text_b, stats_b) = send(&mut second);
    assert!(!text_a.is_empty());
    assert_eq!(text_a, text_b);
    assert_eq!(stats_a.gen_tokens, stats_b.gen_tokens);
    assert_eq!(first.n_past(), second.n_past());
}

/// One turn with `sampler`
fn turn(chat: &mut Chat<'_>, text: &str, sampler: &mut Sampler) -> TurnStats {
    chat.send(text, sampler, |_| {}, || false).expect("turn")
}

#[test]
#[ignore = "needs a model file"]
fn chat_from_a_session_replayed_after_an_answer() {
    let _model = MODEL.lock().unwrap_or_else(PoisonError::into_inner);
    let gguf = biv_cli::open_gguf(model_path()).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let model = Model::load(&gguf).expect("model");
    let engine = engine(&model);
    let opts = options();
    let cfg = SamplerConfig {
        seed: SEED,
        ..SamplerConfig::default()
    };

    // one answer, its session calls recorded, then what a rebuild needs
    let rec = Recording {
        inner: engine.as_ref(),
        log: Log::default(),
    };
    let mut first = Chat::new(&rec, &tok, opts.clone()).expect("chat");
    let mut sampler = Sampler::new(cfg.clone());
    let one = turn(&mut first, MESSAGE, &mut sampler);
    assert!(!one.tokens.is_empty());
    let log = rec.log.borrow().clone();
    let n_past = first.n_past();
    let pending = first.pending().to_vec();
    // the end of the answer waits for the next prompt
    assert!(!pending.is_empty());
    let generated = first.generated().to_vec();
    let history = first.history().to_vec();
    let rng_state = sampler.rng_state();

    // the original chat goes on
    let two = turn(&mut first, FOLLOW_UP, &mut sampler);
    assert!(!two.tokens.is_empty());

    // a new session, the same calls, a chat made from it with the same tokens
    // and sampler state
    let mut session = new_session(engine.as_ref(), &opts);
    replay(session.as_mut(), &log);
    let mut second = Chat::from_session(
        engine.as_ref(),
        &tok,
        opts,
        session,
        pending,
        generated,
        history,
        first.conversation().clone(),
    )
    .expect("chat from session");
    assert_eq!(second.n_past(), n_past);
    let mut resumed = Sampler::resume(cfg, rng_state).expect("resume");
    let again = turn(&mut second, FOLLOW_UP, &mut resumed);

    // the same next answer
    assert_eq!(again.tokens, two.tokens);
    assert_eq!(second.generated(), first.generated());
    assert_eq!(second.history(), first.history());
    assert_eq!(second.n_past(), first.n_past());
    assert_eq!(resumed.rng_state(), sampler.rng_state());
}
