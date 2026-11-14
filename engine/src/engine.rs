//! Compute backends behind one interface: an engine holds the weights, a
//! session holds one sequence (kv cache, recurrent states), and a batch
//! holds sequences whose passes run together.

use alloc::sync::Arc;
use core::any::Any;

use crate::Error;
use crate::memory::{DeviceMemory, ModelMemory, SessionMemory};
use crate::model::{Model, State, StateSnapshot};
use crate::par;
use crate::prelude::*;
use crate::tokenizer::TokenId;

pub trait Engine {
    /// A new empty sequence able to hold `context` tokens, verifying up to
    /// `draft` tokens after the chosen one per step, and drafting with the
    /// model's MTP head when it has one and `draft` isn't 0
    fn new_session(&self, context: usize, draft: usize) -> Result<Box<dyn Session + '_>, Error>;

    /// Prompt tokens per forward pass this backend works best with
    fn batch_size(&self) -> usize;

    /// Tokens drafted per step this backend works best with
    fn draft_size(&self) -> usize;

    fn description(&self) -> String;

    /// The memory of the model on this backend
    fn model_memory(&self) -> ModelMemory;

    /// The memory of a session [`Engine::new_session`] would make with
    /// `context` and `draft`, without making it
    fn session_memory(&self, context: usize, draft: usize) -> SessionMemory;

    /// The memory of the device the backend computes on, when it has its own
    /// accounting: allocated now, and its budget
    fn device_memory(&self) -> Option<DeviceMemory> {
        None
    }

    /// What decides the numbers this backend computes, besides the code, the
    /// weights and how the tokens are batched: kernel variants, cache types,
    /// sizes, as (name, value) pairs. Equal pairs on the same code and model
    /// should give the same bits.
    fn numerics(&self) -> Vec<(&'static str, String)>;

    /// Sequences whose tokens go through the model together, in passes of up
    /// to `tokens` tokens of which up to `logits` have logits, when the
    /// backend can: the weights are then read once for all of them. None when
    /// it can't; [`new_batch`] then runs them one after another.
    fn new_batch(
        &self,
        tokens: usize,
        logits: usize,
    ) -> Result<Option<Box<dyn Batch + '_>>, Error> {
        let _ = (tokens, logits);
        Ok(None)
    }
}

/// [`Engine::new_batch`], or sequences run one after another when the
/// backend doesn't batch
pub fn new_batch<'e>(
    engine: &'e dyn Engine,
    tokens: usize,
    logits: usize,
) -> Result<Box<dyn Batch + 'e>, Error> {
    Ok(match engine.new_batch(tokens, logits)? {
        Some(b) => b,
        None => Box::new(Sequential {
            engine,
            tokens: tokens.max(1),
            logits: logits.max(1),
            seqs: Vec::new(),
        }),
    })
}

/// What a pass computes after its tokens
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pass {
    /// no logits ([`Session::prefill`])
    Prefill,
    /// logits of the last token ([`Session::forward`])
    Forward,
    /// logits of every token, and the states to go back to any of them
    /// ([`Session::verify`])
    Verify,
}

/// A sequence of a [`Batch`]
pub type SeqId = usize;

/// The host time of the command buffers of an engine since it started, by
/// part, see [`Batch::host_time`]. A difference over a pass is the host time
/// of the pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostTime {
    /// from the start of each command buffer to its commit: the encode of
    /// its kernels
    pub encode: core::time::Duration,
    /// the wait for each command buffer after its commit, its device time
    /// included
    pub wait: core::time::Duration,
    /// the copies of the rows of logits of each pass out of the shared
    /// buffer, after its wait
    pub readback: core::time::Duration,
    /// the first part of the wait: from the commit of each command buffer to
    /// the time that the driver began to schedule it
    pub queue: core::time::Duration,
    /// the second part of the wait: from that time to the start of the
    /// command buffer on the device. The third part is the device time of
    /// [`Batch::gpu_time`].
    pub schedule: core::time::Duration,
    /// the last part of the wait: from the end of each command buffer on the
    /// device to the return of the wait for it
    pub completion: core::time::Duration,
    /// the command buffers for which the device did not give its four times
    /// in order: the parts of the wait and the device time hold nothing of
    /// them. A reader that reports or decides on these times refuses them
    /// by name when this count is not 0.
    pub untimed: u64,
}

impl HostTime {
    /// No host time
    pub const ZERO: HostTime = HostTime {
        encode: core::time::Duration::ZERO,
        wait: core::time::Duration::ZERO,
        readback: core::time::Duration::ZERO,
        queue: core::time::Duration::ZERO,
        schedule: core::time::Duration::ZERO,
        completion: core::time::Duration::ZERO,
        untimed: 0,
    };

    /// The host time from `before` to this one
    pub fn since(self, before: HostTime) -> HostTime {
        HostTime {
            encode: self.encode - before.encode,
            wait: self.wait - before.wait,
            readback: self.readback - before.readback,
            queue: self.queue - before.queue,
            schedule: self.schedule - before.schedule,
            completion: self.completion - before.completion,
            untimed: self.untimed - before.untimed,
        }
    }

    /// The host time of this one and `other`
    pub fn plus(self, other: HostTime) -> HostTime {
        HostTime {
            encode: self.encode + other.encode,
            wait: self.wait + other.wait,
            readback: self.readback + other.readback,
            queue: self.queue + other.queue,
            schedule: self.schedule + other.schedule,
            completion: self.completion + other.completion,
            untimed: self.untimed + other.untimed,
        }
    }
}

/// A sequence's part of a batched pass
#[derive(Clone, Copy, Debug)]
pub struct Work<'a> {
    pub seq: SeqId,
    pub tokens: &'a [TokenId],
    pub pass: Pass,
}

impl Work<'_> {
    /// Rows of logits it asks for
    pub fn logits(&self) -> usize {
        match self.pass {
            Pass::Prefill => 0,
            Pass::Forward => 1,
            Pass::Verify => self.tokens.len(),
        }
    }
}

/// Sequences whose passes run together: [`Batch::run`] takes the tokens of
/// several, and the weights are read once for all of them. Each sequence is
/// otherwise a [`Session`], through the methods of the same names.
///
/// The logits of a sequence depend on the other tokens of its passes (the
/// matmul kernels are chosen by the rows of the whole pass), by rounding.
pub trait Batch {
    /// A new empty sequence, as [`Engine::new_session`] makes
    fn add(&mut self, context: usize, draft: usize) -> Result<SeqId, Error>;

    /// Drop a sequence and its memory
    fn remove(&mut self, seq: SeqId);

    /// Tokens of a pass, at most
    fn max_tokens(&self) -> usize;

    /// The time that the device of the engine spent on the work of this
    /// batch's engine since it started, or None when the engine has no
    /// device clock. A difference over a pass is the device time of the
    /// pass.
    fn gpu_time(&self) -> Option<core::time::Duration>;

    /// The host time of the command buffers of this batch's engine since it
    /// started, by part, or None when the engine has no command buffers
    fn host_time(&self) -> Option<HostTime>;

    /// Rows of logits of a pass, at most
    fn max_logits(&self) -> usize;

    /// Each work's tokens through its sequence, all in one pass, returning
    /// for each work the logits its [`Pass`] asks for. A sequence appears at
    /// most once, a verify has at most the sequence's draft length + 1 tokens
    /// and must be followed by [`Batch::keep`]; the tokens and the logits are
    /// within [`Batch::max_tokens`] and [`Batch::max_logits`].
    fn run(&mut self, work: &[Work<'_>]) -> Vec<Vec<Vec<f32>>>;

    /// [`Session::draft`] of `seq`, alone
    fn draft(
        &mut self,
        seq: SeqId,
        next: TokenId,
        n: usize,
        pick: &mut dyn FnMut(&DraftLogits<'_>) -> TokenId,
    ) -> Vec<TokenId>;

    /// [`Session::keep`]
    fn keep(&mut self, seq: SeqId, n: usize);

    /// [`Session::n_past`]
    fn n_past(&self, seq: SeqId) -> usize;

    /// [`Session::memory`], but the buffers the batch shares between its
    /// sequences: [`Batch::memory`]
    fn seq_memory(&self, seq: SeqId) -> SessionMemory;

    /// What [`Batch::add`] would allocate, as [`Engine::session_memory`]
    fn session_memory(&self, context: usize, draft: usize) -> SessionMemory;

    /// The buffers shared by the sequences: activations, logits
    fn memory(&self) -> SessionMemory;

    /// [`Session::snapshot`]
    fn snapshot(&self, seq: SeqId) -> Result<Snapshot, Error>;

    /// [`Session::restore`]
    fn restore(&mut self, seq: SeqId, snapshot: &Snapshot) -> Result<(), Error>;
}

/// A [`Batch`] of sessions run one after another, for backends without one
struct Sequential<'e> {
    engine: &'e dyn Engine,
    tokens: usize,
    logits: usize,
    seqs: Vec<Option<Box<dyn Session + 'e>>>,
}

impl<'e> Sequential<'e> {
    fn seq(&self, seq: SeqId) -> &(dyn Session + 'e) {
        self.seqs[seq].as_deref().expect("a sequence of the batch")
    }

    fn seq_mut(&mut self, seq: SeqId) -> &mut (dyn Session + 'e) {
        self.seqs[seq]
            .as_deref_mut()
            .expect("a sequence of the batch")
    }
}

impl Batch for Sequential<'_> {
    fn gpu_time(&self) -> Option<core::time::Duration> {
        None
    }

    fn host_time(&self) -> Option<HostTime> {
        None
    }

    fn add(&mut self, context: usize, draft: usize) -> Result<SeqId, Error> {
        let session = self.engine.new_session(context, draft)?;
        Ok(insert(&mut self.seqs, session))
    }

    fn remove(&mut self, seq: SeqId) {
        self.seqs[seq] = None;
    }

    fn max_tokens(&self) -> usize {
        self.tokens
    }

    fn max_logits(&self) -> usize {
        self.logits
    }

    fn run(&mut self, work: &[Work<'_>]) -> Vec<Vec<Vec<f32>>> {
        work.iter()
            .map(|w| {
                let s = self.seq_mut(w.seq);
                match w.pass {
                    Pass::Prefill => {
                        s.prefill(w.tokens);
                        Vec::new()
                    }
                    Pass::Forward => vec![s.forward(w.tokens)],
                    Pass::Verify => s.verify(w.tokens),
                }
            })
            .collect()
    }

    fn draft(
        &mut self,
        seq: SeqId,
        next: TokenId,
        n: usize,
        pick: &mut dyn FnMut(&DraftLogits<'_>) -> TokenId,
    ) -> Vec<TokenId> {
        self.seq_mut(seq).draft(next, n, pick)
    }

    fn keep(&mut self, seq: SeqId, n: usize) {
        self.seq_mut(seq).keep(n)
    }

    fn n_past(&self, seq: SeqId) -> usize {
        self.seq(seq).n_past()
    }

    fn seq_memory(&self, seq: SeqId) -> SessionMemory {
        self.seq(seq).memory()
    }

    fn session_memory(&self, context: usize, draft: usize) -> SessionMemory {
        self.engine.session_memory(context, draft)
    }

    fn memory(&self) -> SessionMemory {
        SessionMemory::default()
    }

    fn snapshot(&self, seq: SeqId) -> Result<Snapshot, Error> {
        self.seq(seq).snapshot()
    }

    fn restore(&mut self, seq: SeqId, snapshot: &Snapshot) -> Result<(), Error> {
        self.seq_mut(seq).restore(snapshot)
    }
}

/// Put `item` in the first free place of `slab`, returning its index
pub(crate) fn insert<T>(slab: &mut Vec<Option<T>>, item: T) -> usize {
    match slab.iter().position(Option::is_none) {
        Some(i) => {
            slab[i] = Some(item);
            i
        }
        None => {
            slab.push(Some(item));
            slab.len() - 1
        }
    }
}

/// A sequence. Speculative decoding is three calls: [`Session::draft`] guesses
/// the tokens after the one just chosen, [`Session::verify`] appends the chosen
/// token and the guesses, and [`Session::keep`] keeps the ones that held.
pub trait Session {
    /// Append `tokens`, returning the logits of the last one
    fn forward(&mut self, tokens: &[TokenId]) -> Vec<f32>;

    /// Append `tokens` without computing logits
    fn prefill(&mut self, tokens: &[TokenId]);

    /// Up to `n` tokens likely to follow `next`, the token chosen after the
    /// sequence, at most the session's draft length, each chosen by `pick`
    /// from the MTP head's logits ([`DraftLogits::argmax`] for the likeliest).
    /// Empty when the session doesn't draft.
    fn draft(
        &mut self,
        next: TokenId,
        n: usize,
        pick: &mut dyn FnMut(&DraftLogits<'_>) -> TokenId,
    ) -> Vec<TokenId>;

    /// Append `tokens` provisionally, returning the logits after each one:
    /// the chosen token and up to the session's draft length more, drafted or
    /// not. [`Session::keep`] must follow.
    fn verify(&mut self, tokens: &[TokenId]) -> Vec<Vec<f32>>;

    /// Keep the first `n` tokens of the last verify, dropping the others
    fn keep(&mut self, n: usize);

    /// Tokens in the sequence
    fn n_past(&self) -> usize;

    /// The memory of the session: allocated, and [`SessionMemory::kv_used`]
    /// with [`Session::n_past`] for the part of the kv caches in use
    fn memory(&self) -> SessionMemory;

    /// A copy of the sequence, to [`Session::restore`] in sessions of the same
    /// engine. Not between [`Session::verify`] and [`Session::keep`].
    fn snapshot(&self) -> Result<Snapshot, Error>;

    /// Replace the sequence with `snapshot`, taken from a session of the same
    /// engine. The draft length may differ, but a drafting session needs a
    /// snapshot from a drafting one: the MTP head's history is part of it.
    fn restore(&mut self, snapshot: &Snapshot) -> Result<(), Error>;
}

/// The MTP head's logits for a drafted token: `logits[i]` is for token
/// `tokens[i]`, or for token `i` when the head covers the whole vocabulary
pub struct DraftLogits<'a> {
    pub logits: &'a [f32],
    pub tokens: Option<&'a [TokenId]>,
}

impl DraftLogits<'_> {
    pub fn token(&self, i: usize) -> TokenId {
        self.tokens.map_or(i as TokenId, |t| t[i])
    }

    /// The likeliest token
    pub fn argmax(&self) -> TokenId {
        self.token(crate::sampler::argmax(self.logits) as usize)
    }

    /// (token, logit) pairs
    pub fn candidates(&self) -> Vec<(TokenId, f32)> {
        self.logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (self.token(i), l))
            .collect()
    }
}

/// The state of a sequence after its tokens, in the layout of the backend that
/// made it: the attention caches of those tokens, and the recurrent states,
/// which only processing the whole sequence again could rebuild
pub struct Snapshot {
    n_past: usize,
    bytes: usize,
    data: Arc<dyn Any + Send + Sync>,
}

impl Snapshot {
    pub(crate) fn new<T: Any + Send + Sync>(n_past: usize, bytes: usize, data: T) -> Self {
        Snapshot {
            n_past,
            bytes,
            data: Arc::new(data),
        }
    }

    /// The same snapshot, its data shared rather than copied
    #[cfg(feature = "std")]
    pub(crate) fn share(&self) -> Snapshot {
        Snapshot {
            data: Arc::clone(&self.data),
            ..*self
        }
    }

    pub(crate) fn data<T: Any>(&self) -> Result<&T, Error> {
        self.data
            .downcast_ref()
            .ok_or(Error::Snapshot("taken from another backend"))
    }

    /// Tokens in the sequence
    pub fn n_past(&self) -> usize {
        self.n_past
    }

    /// Memory held
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl<T: Engine + ?Sized> Engine for &T {
    fn new_session(&self, context: usize, draft: usize) -> Result<Box<dyn Session + '_>, Error> {
        (**self).new_session(context, draft)
    }

    fn batch_size(&self) -> usize {
        (**self).batch_size()
    }

    fn draft_size(&self) -> usize {
        (**self).draft_size()
    }

    fn description(&self) -> String {
        (**self).description()
    }

    fn model_memory(&self) -> ModelMemory {
        (**self).model_memory()
    }

    fn session_memory(&self, context: usize, draft: usize) -> SessionMemory {
        (**self).session_memory(context, draft)
    }

    fn device_memory(&self) -> Option<DeviceMemory> {
        (**self).device_memory()
    }

    fn numerics(&self) -> Vec<(&'static str, String)> {
        (**self).numerics()
    }

    fn new_batch(
        &self,
        tokens: usize,
        logits: usize,
    ) -> Result<Option<Box<dyn Batch + '_>>, Error> {
        (**self).new_batch(tokens, logits)
    }
}

impl Engine for Model<'_> {
    fn new_session(&self, context: usize, draft: usize) -> Result<Box<dyn Session + '_>, Error> {
        Ok(Box::new(CpuSession {
            model: self,
            state: self.new_state(context, draft > 0)?,
            draft,
        }))
    }

    fn batch_size(&self) -> usize {
        64
    }

    // a pass of 2 tokens costs 1.45x one of 1, and the cost grows from there
    fn draft_size(&self) -> usize {
        1
    }

    fn description(&self) -> String {
        format!("cpu, {} threads", par::threads())
    }

    fn model_memory(&self) -> ModelMemory {
        self.memory
    }

    fn session_memory(&self, context: usize, draft: usize) -> SessionMemory {
        self.shape().cpu_session(context, draft, self.batch_size())
    }

    // not the thread count: each output row is computed whole by one thread
    fn numerics(&self) -> Vec<(&'static str, String)> {
        vec![
            ("kind", "cpu".into()),
            ("kernels", crate::quant::kernels().into()),
            ("kv_cache", "f32".into()),
        ]
    }
}

struct CpuSession<'m> {
    model: &'m Model<'m>,
    state: State,
    draft: usize,
}

impl Session for CpuSession<'_> {
    fn forward(&mut self, tokens: &[TokenId]) -> Vec<f32> {
        self.model.forward(&mut self.state, tokens)
    }

    fn prefill(&mut self, tokens: &[TokenId]) {
        self.model.prefill(&mut self.state, tokens)
    }

    fn draft(
        &mut self,
        next: TokenId,
        n: usize,
        pick: &mut dyn FnMut(&DraftLogits<'_>) -> TokenId,
    ) -> Vec<TokenId> {
        self.model
            .draft(&mut self.state, next, n.min(self.draft), pick)
    }

    fn verify(&mut self, tokens: &[TokenId]) -> Vec<Vec<f32>> {
        self.model.verify(&mut self.state, tokens)
    }

    fn keep(&mut self, n: usize) {
        self.model.keep(&mut self.state, n)
    }

    fn n_past(&self) -> usize {
        self.state.n_past
    }

    fn memory(&self) -> SessionMemory {
        let estimate = self.model.session_memory(self.state.context, self.draft);
        SessionMemory {
            transient: estimate.transient,
            ..self.state.memory()
        }
    }

    fn snapshot(&self) -> Result<Snapshot, Error> {
        let snap = self.state.snapshot()?;
        Ok(Snapshot::new(self.state.n_past, snap.bytes(), snap))
    }

    fn restore(&mut self, snapshot: &Snapshot) -> Result<(), Error> {
        self.state.restore(snapshot.data::<StateSnapshot>()?)
    }
}

#[cfg(test)]
mod tests {
    use super::HostTime;
    use core::time::Duration;

    /// Host times of two reads of the counters, in milliseconds
    const BEFORE_MS: [u64; 6] = [3, 5, 7, 1, 2, 3];
    const AFTER_MS: [u64; 6] = [4, 9, 8, 2, 4, 6];

    fn time(ms: [u64; 6], untimed: u64) -> HostTime {
        let [encode, wait, readback, queue, schedule, completion] = ms.map(Duration::from_millis);
        HostTime {
            encode,
            wait,
            readback,
            queue,
            schedule,
            completion,
            untimed,
        }
    }

    #[test]
    fn a_host_time_since_an_earlier_one_adds_back_to_it() {
        let (before, after) = (time(BEFORE_MS, 2), time(AFTER_MS, 3));
        let pass = after.since(before);
        assert_eq!(pass, time([1, 4, 1, 1, 2, 3], 1));
        assert_eq!(before.plus(pass), after);
        assert_eq!(HostTime::ZERO.plus(pass), pass);
    }
}
