//! Continuous batching: chats on threads of their own share an engine whose
//! passes take the tokens of every chat ready for one, so that the weights
//! are read once for all of them.
//!
//! [`serve`] runs a scheduler on the calling thread, which owns the engine
//! and its [`Batch`], and gives a [`Handle`] to a closure run on another
//! thread. The handle is an [`Engine`]: a [`Chat`](crate::chat::Chat) made
//! on it works as on any engine, its session's calls being requests to the
//! scheduler. Between passes the scheduler takes the requests that came: the
//! tokens of each sequence (a prompt, the token chosen, a verify), and the
//! other calls (drafts, keep, snapshots), which it serves at once. A pass
//! takes the decode steps first (forward and verify, in their order), then
//! the prompts in pieces that fill the rest of [`Batch::max_tokens`].
//! Sequences join and leave between any two passes.
//!
//! After a pass the scheduler waits, up to [`BatchOptions::wait`], for the
//! sequences it answered to send their next tokens, so that they make the
//! next pass together rather than one after the other.
//!
//! A sequence's logits depend by rounding on the other tokens of its passes,
//! see [`Batch`]: the answers of batched chats can differ from those of the
//! same chats alone.

use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::time::{Duration, Instant};

use crate::Error;
use crate::engine::{
    self, Batch, DraftLogits, Engine, HostTime, Pass, SeqId, Session, Snapshot, Work,
};
use crate::memory::{DeviceMemory, ModelMemory, SessionMemory};
use crate::prelude::*;
use crate::tokenizer::TokenId;

/// How a [`serve`] scheduler batches
#[derive(Debug, Clone)]
pub struct BatchOptions {
    /// tokens per pass, at most: the engine's [`Engine::batch_size`] when None
    pub tokens: Option<usize>,
    /// rows of logits per pass, at most: decode steps, and the tokens of
    /// verify passes
    pub logits: usize,
    /// how long a pass waits for the sequences of the previous one
    pub wait: Duration,
    /// decode rows per pass, at most: the forward and verify tokens of its
    /// sequences. A draft shrinks so that the busy sequences fit, and a
    /// pass takes no more, unless one sequence alone needs more.
    pub decode_rows: usize,
}

/// The decode rows of a pass of [`BatchOptions::default`]: on Metal, the
/// most rows that the `mmv` matmul kernels take, above which the `mm`
/// kernels take a pass and it costs more; elsewhere no limit
#[cfg(all(feature = "metal", target_os = "macos"))]
const DEFAULT_DECODE_ROWS: usize = crate::metal::MMV_MAX_ROWS;
#[cfg(not(all(feature = "metal", target_os = "macos")))]
const DEFAULT_DECODE_ROWS: usize = usize::MAX;

impl Default for BatchOptions {
    fn default() -> Self {
        BatchOptions {
            tokens: None,
            logits: 64,
            wait: Duration::from_millis(5),
            decode_rows: DEFAULT_DECODE_ROWS,
        }
    }
}

/// Run `f` on a thread of its own with a [`Handle`] to `engine`, serving the
/// sessions made on it from this thread until `f` returns and every clone
/// of the handle is dropped. A panic of `f` is resumed here.
pub fn serve<R: Send>(
    engine: &dyn Engine,
    opts: &BatchOptions,
    f: impl FnOnce(Handle) -> R + Send,
) -> Result<R, Error> {
    let tokens = opts.tokens.unwrap_or_else(|| engine.batch_size()).max(1);
    let mut batch = engine::new_batch(engine, tokens, opts.logits)?;
    let mut numerics = engine.numerics();
    numerics.push(("batch_tokens", batch.max_tokens().to_string()));
    numerics.push(("batch_logits", batch.max_logits().to_string()));
    let (tx, rx) = mpsc::channel();
    let handle = Handle {
        tx,
        info: Arc::new(Info {
            description: format!(
                "{}, batched passes of {} tokens",
                engine.description(),
                batch.max_tokens()
            ),
            batch_size: batch.max_tokens(),
            draft_size: engine.draft_size(),
            model_memory: engine.model_memory(),
            numerics,
        }),
    };
    let pass_gpu_time = batch.gpu_time().map(|_| Duration::ZERO);
    let pass_host_time = batch.host_time().map(|_| HostTime::ZERO);
    Ok(std::thread::scope(|scope| {
        let user = scope.spawn(move || f(handle));
        Scheduler {
            batch: &mut *batch,
            engine,
            wait: opts.wait,
            decode_rows: opts.decode_rows.max(1),
            queue: Vec::new(),
            answered: Vec::new(),
            last_pass: Vec::new(),
            passes: Passes {
                passes: 0,
                shared: 0,
                tokens: 0,
                sequences: 0,
                pass_time: Duration::ZERO,
                pass_gpu_time,
                pass_host_time,
                drafts: 0,
                draft_tokens: 0,
                draft_time: Duration::ZERO,
                draft_picks: 0,
                draft_pick_time: Duration::ZERO,
                widest_decode: 0,
            },
        }
        .run(rx);
        user.join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
    }))
}

/// An [`Engine`] whose sessions are served by a [`serve`] scheduler. Clones
/// share it; it can be sent to and shared between threads.
#[derive(Clone)]
pub struct Handle {
    tx: Sender<Request>,
    info: Arc<Info>,
}

/// The passes a [`serve`] scheduler ran since it started
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Passes {
    /// all passes
    pub passes: u64,
    /// the passes that held the tokens of more than one sequence
    pub shared: u64,
    /// the tokens of all passes
    pub tokens: u64,
    /// the sequences of all passes, each counted once for each pass that
    /// holds its tokens: over `passes`, the mean sequences of a pass
    pub sequences: u64,
    /// the time of the engine in the passes
    pub pass_time: Duration,
    /// the device time of the passes, a part of `pass_time`, or None when
    /// the engine has no device clock ([`Batch::gpu_time`])
    pub pass_gpu_time: Option<Duration>,
    /// the host time of the passes by part, a part of `pass_time`, or None
    /// when the engine has no command buffers ([`Batch::host_time`]). What
    /// `pass_time` holds past the parts is the work of a pass before its
    /// command buffer, and past its readback.
    pub pass_host_time: Option<HostTime>,
    /// the drafts that the scheduler served: each one runs for one sequence,
    /// outside the passes
    pub drafts: u64,
    /// the tokens of those drafts
    pub draft_tokens: u64,
    /// the time of the engine in those drafts, with the picks of their chats
    pub draft_time: Duration,
    /// the picks of those drafts: each one a round trip of the logits of a
    /// drafted token to its chat and of its pick back
    pub draft_picks: u64,
    /// the time of those round trips, a part of `draft_time`
    pub draft_pick_time: Duration,
    /// the most decode rows of one pass
    pub widest_decode: u64,
}

/// What the handle answers without asking the scheduler
struct Info {
    description: String,
    batch_size: usize,
    draft_size: usize,
    model_memory: ModelMemory,
    numerics: Vec<(&'static str, String)>,
}

enum Request {
    Add {
        context: usize,
        draft: usize,
        reply: Sender<Result<SeqId, Error>>,
    },
    Remove(SeqId),
    /// tokens through the sequence, then the logits its pass asks for
    Run {
        seq: SeqId,
        tokens: Vec<TokenId>,
        pass: Pass,
        reply: Sender<Vec<Vec<f32>>>,
    },
    /// the logits of each draft go out on `reply`, the token picked comes
    /// back on `picks`, and the drafts at the end
    Draft {
        seq: SeqId,
        next: TokenId,
        n: usize,
        reply: Sender<DraftMsg>,
        picks: Receiver<TokenId>,
    },
    Keep {
        seq: SeqId,
        n: usize,
    },
    SeqMemory {
        seq: SeqId,
        reply: Sender<SessionMemory>,
    },
    SessionMemory {
        context: usize,
        draft: usize,
        reply: Sender<SessionMemory>,
    },
    SharedMemory {
        reply: Sender<SessionMemory>,
    },
    DeviceMemory {
        reply: Sender<Option<DeviceMemory>>,
    },
    Passes {
        reply: Sender<Passes>,
    },
    Snapshot {
        seq: SeqId,
        reply: Sender<Result<Snapshot, Error>>,
    },
    Restore {
        seq: SeqId,
        snapshot: Snapshot,
        reply: Sender<Result<(), Error>>,
    },
}

enum DraftMsg {
    Logits {
        logits: Vec<f32>,
        tokens: Option<Vec<TokenId>>,
    },
    Done(Vec<TokenId>),
}

const STOPPED: &str = "batch scheduler stopped";

impl Handle {
    /// Send the request `make` builds around a reply channel, and wait for the reply
    fn call<T>(&self, make: impl FnOnce(Sender<T>) -> Request) -> T {
        let (tx, rx) = mpsc::channel();
        self.tx.send(make(tx)).expect(STOPPED);
        rx.recv().expect(STOPPED)
    }

    /// The memory of the buffers the scheduler's batch shares between its
    /// sequences, which their sessions' [`Session::memory`] leaves out
    pub fn shared_memory(&self) -> SessionMemory {
        self.call(|reply| Request::SharedMemory { reply })
    }

    /// The passes the scheduler ran since it started
    pub fn passes(&self) -> Passes {
        self.call(|reply| Request::Passes { reply })
    }
}

impl Engine for Handle {
    fn new_session(&self, context: usize, draft: usize) -> Result<Box<dyn Session + '_>, Error> {
        let seq = self.call(|reply| Request::Add {
            context,
            draft,
            reply,
        })?;
        Ok(Box::new(Proxy {
            h: self,
            seq,
            n_past: 0,
            verify_from: None,
        }))
    }

    fn batch_size(&self) -> usize {
        self.info.batch_size
    }

    fn draft_size(&self) -> usize {
        self.info.draft_size
    }

    fn description(&self) -> String {
        self.info.description.clone()
    }

    fn model_memory(&self) -> ModelMemory {
        self.info.model_memory
    }

    fn session_memory(&self, context: usize, draft: usize) -> SessionMemory {
        self.call(|reply| Request::SessionMemory {
            context,
            draft,
            reply,
        })
    }

    fn device_memory(&self) -> Option<DeviceMemory> {
        self.call(|reply| Request::DeviceMemory { reply })
    }

    fn numerics(&self) -> Vec<(&'static str, String)> {
        self.info.numerics.clone()
    }
}

/// A session of a [`Handle`]: a sequence of the scheduler's batch
struct Proxy<'h> {
    h: &'h Handle,
    seq: SeqId,
    /// as the scheduler's, without asking
    n_past: usize,
    verify_from: Option<usize>,
}

impl Proxy<'_> {
    fn run(&mut self, tokens: &[TokenId], pass: Pass) -> Vec<Vec<f32>> {
        let logits = self.h.call(|reply| Request::Run {
            seq: self.seq,
            tokens: tokens.to_vec(),
            pass,
            reply,
        });
        self.n_past += tokens.len();
        logits
    }
}

impl Session for Proxy<'_> {
    fn forward(&mut self, tokens: &[TokenId]) -> Vec<f32> {
        self.run(tokens, Pass::Forward)
            .pop()
            .expect("logits requested")
    }

    fn prefill(&mut self, tokens: &[TokenId]) {
        self.run(tokens, Pass::Prefill);
    }

    fn draft(
        &mut self,
        next: TokenId,
        n: usize,
        pick: &mut dyn FnMut(&DraftLogits<'_>) -> TokenId,
    ) -> Vec<TokenId> {
        let (reply, from_scheduler) = mpsc::channel();
        let (to_scheduler, picks) = mpsc::channel();
        self.h
            .tx
            .send(Request::Draft {
                seq: self.seq,
                next,
                n,
                reply,
                picks,
            })
            .expect(STOPPED);
        loop {
            match from_scheduler.recv().expect(STOPPED) {
                DraftMsg::Logits { logits, tokens } => {
                    let tok = pick(&DraftLogits {
                        logits: &logits,
                        tokens: tokens.as_deref(),
                    });
                    to_scheduler.send(tok).expect(STOPPED);
                }
                DraftMsg::Done(drafts) => return drafts,
            }
        }
    }

    fn verify(&mut self, tokens: &[TokenId]) -> Vec<Vec<f32>> {
        let pos0 = self.n_past;
        let logits = self.run(tokens, Pass::Verify);
        self.verify_from = Some(pos0);
        logits
    }

    fn keep(&mut self, n: usize) {
        let pos0 = self.verify_from.take().expect("keep without verify");
        self.h
            .tx
            .send(Request::Keep { seq: self.seq, n })
            .expect(STOPPED);
        self.n_past = pos0 + n;
    }

    fn n_past(&self) -> usize {
        self.n_past
    }

    fn memory(&self) -> SessionMemory {
        self.h.call(|reply| Request::SeqMemory {
            seq: self.seq,
            reply,
        })
    }

    fn snapshot(&self) -> Result<Snapshot, Error> {
        self.h.call(|reply| Request::Snapshot {
            seq: self.seq,
            reply,
        })
    }

    fn restore(&mut self, snapshot: &Snapshot) -> Result<(), Error> {
        self.h.call(|reply| Request::Restore {
            seq: self.seq,
            snapshot: snapshot.share(),
            reply,
        })?;
        self.n_past = snapshot.n_past();
        self.verify_from = None;
        Ok(())
    }
}

impl Drop for Proxy<'_> {
    fn drop(&mut self) {
        // the scheduler may be gone already, when it panicked
        let _ = self.h.tx.send(Request::Remove(self.seq));
    }
}

/// Tokens a sequence waits to have processed
struct Pending {
    seq: SeqId,
    tokens: Vec<TokenId>,
    /// processed by earlier passes, as prefill
    done: usize,
    pass: Pass,
    reply: Sender<Vec<Vec<f32>>>,
}

impl Pending {
    fn left(&self) -> usize {
        self.tokens.len() - self.done
    }
}

/// A piece of a pending request in a pass: `len` tokens of `queue[i]`
struct Piece {
    i: usize,
    len: usize,
    pass: Pass,
}

struct Scheduler<'b, 'e> {
    batch: &'b mut (dyn Batch + 'e),
    engine: &'b dyn Engine,
    wait: Duration,
    decode_rows: usize,
    /// in the order they came
    queue: Vec<Pending>,
    /// the sequences the last pass answered, until they send tokens again
    answered: Vec<SeqId>,
    /// the sequences of the last pass
    last_pass: Vec<SeqId>,
    passes: Passes,
}

impl Scheduler<'_, '_> {
    /// Serve requests until every handle is dropped
    fn run(mut self, rx: Receiver<Request>) {
        let mut open = true;
        loop {
            if self.queue.is_empty() {
                if !open {
                    return;
                }
                match rx.recv() {
                    Ok(r) => self.handle(r),
                    Err(_) => return,
                }
            }
            loop {
                match rx.try_recv() {
                    Ok(r) => self.handle(r),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        open = false;
                        break;
                    }
                }
            }
            // the sequences of the last pass are about to come back
            let mut deadline = Instant::now() + self.wait;
            while open && !self.answered.is_empty() && !self.full() {
                let now = Instant::now();
                if now >= deadline {
                    break;
                }
                match rx.recv_timeout(deadline - now) {
                    Ok(r) => {
                        // a draft keeps the device busy: the wait starts again after it
                        let drafted = matches!(r, Request::Draft { .. });
                        self.handle(r);
                        if drafted {
                            deadline = Instant::now() + self.wait;
                        }
                    }
                    Err(RecvTimeoutError::Timeout) => break,
                    Err(RecvTimeoutError::Disconnected) => open = false,
                }
            }
            self.answered.clear();
            if !self.queue.is_empty() {
                self.pass();
            }
        }
    }

    /// Whether the queue holds a full pass already
    fn full(&self) -> bool {
        self.queue.iter().map(Pending::left).sum::<usize>() >= self.batch.max_tokens()
    }

    /// The drafts that `seq` may take, so that the decode rows of the busy
    /// sequences, each a token and its drafts, fit in `decode_rows`. The
    /// busy sequences are `seq`, those of the last pass and those that wait.
    fn draft_room(&self, seq: SeqId) -> usize {
        let mut busy: Vec<SeqId> = self.queue.iter().map(|p| p.seq).collect();
        busy.extend(&self.last_pass);
        busy.push(seq);
        busy.sort_unstable();
        busy.dedup();
        (self.decode_rows / busy.len()).saturating_sub(1)
    }

    fn handle(&mut self, r: Request) {
        let room = match &r {
            Request::Draft { seq, .. } => self.draft_room(*seq),
            _ => 0,
        };
        let b = &mut *self.batch;
        match r {
            Request::Add {
                context,
                draft,
                reply,
            } => {
                let _ = reply.send(b.add(context, draft));
            }
            Request::Remove(seq) => {
                self.answered.retain(|&s| s != seq);
                self.queue.retain(|p| p.seq != seq);
                b.remove(seq);
            }
            Request::Run {
                seq,
                tokens,
                pass,
                reply,
            } => {
                self.answered.retain(|&s| s != seq);
                if tokens.is_empty() {
                    // as a session would, for a prefill; a pass of nothing otherwise
                    assert_eq!(pass, Pass::Prefill, "a pass of no tokens");
                    let _ = reply.send(Vec::new());
                    return;
                }
                self.queue.push(Pending {
                    seq,
                    tokens,
                    done: 0,
                    pass,
                    reply,
                });
            }
            Request::Draft {
                seq,
                next,
                n,
                reply,
                picks,
            } => {
                let n = n.min(room);
                if n == 0 {
                    let _ = reply.send(DraftMsg::Done(Vec::new()));
                    return;
                }
                let start = Instant::now();
                let (mut pick_count, mut pick_time) = (0, Duration::ZERO);
                let drafts = b.draft(seq, next, n, &mut |l| {
                    let round_trip = Instant::now();
                    let msg = DraftMsg::Logits {
                        logits: l.logits.to_vec(),
                        tokens: l.tokens.map(<[TokenId]>::to_vec),
                    };
                    // a chat gone takes any token
                    let pick = if reply.send(msg).is_err() {
                        0
                    } else {
                        picks.recv().unwrap_or(0)
                    };
                    pick_count += 1;
                    pick_time += round_trip.elapsed();
                    pick
                });
                self.passes.draft_picks += pick_count;
                self.passes.draft_pick_time += pick_time;
                self.passes.drafts += 1;
                self.passes.draft_tokens += drafts.len() as u64;
                self.passes.draft_time += start.elapsed();
                let _ = reply.send(DraftMsg::Done(drafts));
            }
            Request::Keep { seq, n } => b.keep(seq, n),
            Request::SeqMemory { seq, reply } => {
                let _ = reply.send(b.seq_memory(seq));
            }
            Request::SessionMemory {
                context,
                draft,
                reply,
            } => {
                let _ = reply.send(b.session_memory(context, draft));
            }
            Request::SharedMemory { reply } => {
                let _ = reply.send(b.memory());
            }
            Request::DeviceMemory { reply } => {
                let _ = reply.send(self.engine.device_memory());
            }
            Request::Passes { reply } => {
                let _ = reply.send(self.passes);
            }
            Request::Snapshot { seq, reply } => {
                let _ = reply.send(b.snapshot(seq));
            }
            Request::Restore {
                seq,
                snapshot,
                reply,
            } => {
                let _ = reply.send(b.restore(seq, &snapshot));
            }
        }
    }

    /// One pass: the decode steps, then pieces of the prompts, within the
    /// batch's tokens and logits
    fn pass(&mut self) {
        let (mut tokens, mut logits) = (self.batch.max_tokens(), self.batch.max_logits());
        let mut decode_left = self.decode_rows;
        let mut pieces: Vec<Piece> = Vec::new();
        for decode in [true, false] {
            for (i, p) in self.queue.iter().enumerate() {
                if pieces.iter().any(|q| q.i == i) || tokens == 0 {
                    continue;
                }
                let left = p.left();
                let piece = match p.pass {
                    // whole, or not at all; alone, whatever its size
                    Pass::Verify if decode => {
                        if (left > tokens || left > logits || left > decode_left)
                            && !pieces.is_empty()
                        {
                            continue;
                        }
                        Piece {
                            i,
                            len: left,
                            pass: Pass::Verify,
                        }
                    }
                    Pass::Forward
                        if decode
                            && left <= tokens
                            && logits > 0
                            && (left <= decode_left || pieces.is_empty()) =>
                    {
                        Piece {
                            i,
                            len: left,
                            pass: Pass::Forward,
                        }
                    }
                    // a prompt, or the start of a long forward
                    Pass::Prefill if !decode => Piece {
                        i,
                        len: left.min(tokens),
                        pass: Pass::Prefill,
                    },
                    Pass::Forward if !decode && left > 1 => Piece {
                        i,
                        len: (left - 1).min(tokens),
                        pass: Pass::Prefill,
                    },
                    _ => continue,
                };
                tokens = tokens.saturating_sub(piece.len);
                if decode {
                    decode_left = decode_left.saturating_sub(piece.len);
                }
                logits = logits.saturating_sub(match piece.pass {
                    Pass::Prefill => 0,
                    Pass::Forward => 1,
                    Pass::Verify => piece.len,
                });
                pieces.push(piece);
            }
        }

        let work: Vec<Work<'_>> = pieces
            .iter()
            .map(|piece| {
                let p = &self.queue[piece.i];
                Work {
                    seq: p.seq,
                    tokens: &p.tokens[p.done..p.done + piece.len],
                    pass: piece.pass,
                }
            })
            .collect();
        let start = Instant::now();
        let gpu_before = self.batch.gpu_time();
        let host_before = self.batch.host_time();
        let out = self.batch.run(&work);
        self.passes.pass_time += start.elapsed();
        if let (Some(total), Some(before), Some(after)) = (
            self.passes.pass_gpu_time.as_mut(),
            gpu_before,
            self.batch.gpu_time(),
        ) {
            *total += after - before;
        }
        if let (Some(total), Some(before), Some(after)) = (
            self.passes.pass_host_time.as_mut(),
            host_before,
            self.batch.host_time(),
        ) {
            *total = total.plus(after.since(before));
        }
        // a sequence has at most one request in the queue, so each piece is
        // the tokens of another sequence
        self.passes.passes += 1;
        self.passes.sequences += pieces.len() as u64;
        let decode_rows: usize = pieces
            .iter()
            .filter(|p| p.pass != Pass::Prefill)
            .map(|p| p.len)
            .sum();
        self.passes.widest_decode = self.passes.widest_decode.max(decode_rows as u64);
        self.last_pass = pieces.iter().map(|p| self.queue[p.i].seq).collect();
        if pieces.len() > 1 {
            self.passes.shared += 1;
        }
        self.passes.tokens += pieces.iter().map(|p| p.len as u64).sum::<u64>();

        let mut finished = Vec::new();
        for (piece, logits) in pieces.iter().zip(out) {
            let p = &mut self.queue[piece.i];
            p.done += piece.len;
            if p.left() == 0 {
                let _ = p.reply.send(logits);
                self.answered.push(p.seq);
                finished.push(piece.i);
            }
        }
        let mut i = 0;
        self.queue.retain(|_| {
            i += 1;
            !finished.contains(&(i - 1))
        });
    }
}
