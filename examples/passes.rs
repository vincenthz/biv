//! The time of one batched pass for each mix of sequences and tokens.
//! `passes <gguf>`
//!
//! Each sequence holds the same prompt. A pass is a verify of `tokens`
//! tokens for each of `seqs` sequences, which gives logits for every token,
//! as a decode step with drafts does. `idle` more sequences hold the prompt
//! in the batch but are not in the pass. Each case runs one pass to warm,
//! then `ITERS` timed passes, and prints the mean time of a pass, its ratio
//! to a pass of one token of one sequence, and the GPU time of a pass.
use std::time::{Duration, Instant};

use biv::{
    engine::{self, Batch, Engine, Pass, Work},
    model::Model,
    tokenizer::Tokenizer,
};

/// The prompt of each sequence
const PROMPT: &str =
    "Explain in detail how a CPU cache hierarchy works and why it matters for performance.";
/// The tokens of a pass and the rows of logits of a pass, at most
const PASS_TOKENS: usize = 256;
const PASS_LOGITS: usize = 64;
/// The context of each sequence, far above the tokens of the timed passes
const CONTEXT: usize = 4096;
/// The timed passes of each case, after one pass to warm
const ITERS: usize = 8;
/// Milliseconds in a second, for the times of a pass
const MS_PER_S: f64 = 1e3;
/// Each case: the sequences of a pass, the tokens of each sequence, and the
/// idle sequences of the batch. The first case is the reference: one token
/// of one sequence.
const CASES: [(usize, usize, usize); 32] = [
    (1, 1, 0),
    (1, 2, 0),
    (1, 3, 0),
    (1, 4, 0),
    (1, 6, 0),
    (1, 8, 0),
    (1, 12, 0),
    (1, 16, 0),
    (1, 17, 0),
    (1, 24, 0),
    (1, 32, 0),
    (1, 48, 0),
    // sequences of one row
    (2, 1, 0),
    (4, 1, 0),
    (6, 1, 0),
    (8, 1, 0),
    (12, 1, 0),
    (16, 1, 0),
    (24, 1, 0),
    (32, 1, 0),
    // sequences with drafts
    (2, 6, 0),
    (3, 6, 0),
    (4, 6, 0),
    (8, 6, 0),
    // about 30 rows in other shapes
    (6, 5, 0),
    (8, 4, 0),
    (16, 2, 0),
    (30, 1, 0),
    // the same pass with idle sequences in the batch
    (6, 5, 10),
    (6, 5, 26),
    (16, 2, 16),
    (1, 1, 31),
];

fn main() {
    let path = std::env::args().nth(1).expect("usage: passes <gguf>");
    let g = biv_cli::open_gguf(&path).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let model = Model::load(&g).unwrap();
    #[cfg(target_os = "macos")]
    let metal = biv::metal::MetalEngine::new(&model, PASS_TOKENS).unwrap();
    #[cfg(target_os = "macos")]
    let engine: &dyn Engine = &metal;
    #[cfg(not(target_os = "macos"))]
    let engine: &dyn Engine = &model;
    println!("{}", engine.description());

    let prompt = tok.encode(PROMPT);
    let mut batch = engine::new_batch(engine, PASS_TOKENS, PASS_LOGITS).unwrap();
    let mut reference = None;
    for (seqs, tokens, idle) in CASES {
        let ids: Vec<usize> = (0..seqs)
            // a sequence verifies its drafts and one token more
            .map(|_| batch.add(CONTEXT, tokens - 1).unwrap())
            .collect();
        let idle_ids: Vec<usize> = (0..idle)
            .map(|_| batch.add(CONTEXT, tokens - 1).unwrap())
            .collect();
        for &seq in ids.iter().chain(&idle_ids) {
            batch.run(&[Work {
                seq,
                tokens: &prompt,
                pass: Pass::Prefill,
            }]);
        }
        let step: Vec<_> = prompt.iter().copied().cycle().take(tokens).collect();
        let work: Vec<Work<'_>> = ids
            .iter()
            .map(|&seq| Work {
                seq,
                tokens: &step,
                pass: Pass::Verify,
            })
            .collect();
        let pass = |batch: &mut dyn Batch| {
            batch.run(&work);
            for &seq in &ids {
                batch.keep(seq, tokens);
            }
        };
        pass(&mut *batch);
        let start = Instant::now();
        let gpu_start = batch.gpu_time();
        let host_start = batch.host_time();
        for _ in 0..ITERS {
            pass(&mut *batch);
        }
        let per_pass = |d: Duration| d.as_secs_f64() * MS_PER_S / ITERS as f64;
        let ms = per_pass(start.elapsed());
        let gpu_ms = match (gpu_start, batch.gpu_time()) {
            (Some(a), Some(b)) => format!("{:7.2}", per_pass(b - a)),
            _ => "no device clock".into(),
        };
        let host_ms = match (host_start, batch.host_time()) {
            (Some(a), Some(b)) => {
                let h = b.since(a);
                assert_eq!(
                    h.untimed, 0,
                    "CommandBufferTimeMissing: {} command buffers of the case have no times",
                    h.untimed
                );
                format!(
                    "encode {:7.2} wait {:7.2} (queue {:6.2} schedule {:6.2} completion {:6.2}) readback {:7.2}",
                    per_pass(h.encode),
                    per_pass(h.wait),
                    per_pass(h.queue),
                    per_pass(h.schedule),
                    per_pass(h.completion),
                    per_pass(h.readback)
                )
            }
            _ => "no command buffers".into(),
        };
        let base = *reference.get_or_insert(ms);
        println!(
            "seqs={seqs:2} tokens={tokens:2} idle={idle:2} rows={:3}: {ms:7.2} ms a pass, \
             {:5.2}x one token, GPU {gpu_ms} ms, host {host_ms} ms",
            seqs * tokens,
            ms / base
        );
        for seq in ids.into_iter().chain(idle_ids) {
            batch.remove(seq);
        }
    }
}
