//! The logits of the `mmv32` kernels against those of the `mm` kernels, on
//! the same tokens, for prompt passes and verify passes of 17 to 32 rows.
//! The CPU logits give the size of the rounding that the Metal kernels had
//! before. Ignored without a model file:
//! `BIV_TEST_MODEL=<gguf> cargo test --release --test metal_kernels -- --ignored --nocapture`
#![cfg(target_os = "macos")]

use biv::engine::Engine;
use biv::metal::{MMV16_ROWS, MetalEngine};
use biv::model::Model;
use biv::sampler::argmax;
use biv::tokenizer::{TokenId, Tokenizer};

/// The variable that names the GGUF file
const MODEL_ENV: &str = "BIV_TEST_MODEL";
/// Metal prompt tokens per forward pass, as biv-cli uses
const METAL_BATCH: usize = 256;
const CONTEXT: usize = 2048;
/// The session of a prompt pass does not draft
const NO_DRAFT: usize = 0;
/// The rows of the passes that `mmv32` takes from `mm`: the fewest, one
/// between and the most
const PASS_ROWS: [usize; 3] = [17, 24, 32];
/// The prompt tokens before each verify pass
const PREFIX: usize = 40;
/// The largest differences of the logits of `mmv32` from those of `mm`:
/// absolute, and relative to the largest absolute logit. The source is one
/// measure on an Apple M3 Max on 2026-11-13: the verify passes of 17, 24
/// and 32 rows gave 0.017576 and 6.679e-4, and the prompt passes less. Plain
/// rounding, the CPU logits against those of `mm`, is 0.44 and 1.7e-2. The
/// margin is small on purpose, because the GPU gives the same bits on each
/// run: a change of the GPU, the driver or a kernel must fail this test and
/// get a new measure, not a wider const.
const MAX_ABS_LOGIT_DIFFERENCE: f32 = 0.018;
const MAX_REL_LOGIT_DIFFERENCE: f32 = 6.7e-4;
/// A text of more than `PREFIX` and the most `PASS_ROWS` tokens
const TEXT: &str = "Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, type safety, and concurrency. It enforces memory safety, meaning that all references point to valid memory, without a garbage collector. Software developer Graydon Hoare created Rust. ";

fn model_path() -> String {
    std::env::var(MODEL_ENV).unwrap_or_else(|_| panic!("{MODEL_ENV} must name a GGUF file"))
}

/// The largest absolute difference of two logit vectors, and the same
/// divided by the largest absolute logit of `reference`
fn difference(reference: &[f32], other: &[f32]) -> (f32, f32) {
    let abs = reference
        .iter()
        .zip(other)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    let scale = reference.iter().map(|a| a.abs()).fold(0.0, f32::max);
    (abs, abs / scale)
}

/// The differences of the rows of logits of `other` from `reference`
#[derive(Default)]
struct Differences {
    abs: f32,
    rel: f32,
    /// rows whose likeliest token differs
    argmax: usize,
    rows: usize,
}

impl Differences {
    fn add(&mut self, reference: &[Vec<f32>], other: &[Vec<f32>]) {
        assert_eq!(reference.len(), other.len());
        for (a, b) in reference.iter().zip(other) {
            let (abs, rel) = difference(a, b);
            self.abs = self.abs.max(abs);
            self.rel = self.rel.max(rel);
            self.argmax += usize::from(argmax(a) != argmax(b));
            self.rows += 1;
        }
    }
}

impl std::fmt::Display for Differences {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "largest abs {:.6}, largest rel {:.3e}, likeliest token differs in {} of {} rows",
            self.abs, self.rel, self.argmax, self.rows
        )
    }
}

/// The logits of the last token of a prompt pass of each of `PASS_ROWS`
/// tokens, and of each token of a verify pass of each of `PASS_ROWS` tokens
/// after `PREFIX` tokens
fn logits(engine: &dyn Engine, ids: &[TokenId]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut prompt = Vec::new();
    let mut verify = Vec::new();
    for rows in PASS_ROWS {
        let mut s = engine.new_session(CONTEXT, NO_DRAFT).expect("session");
        prompt.push(s.forward(&ids[..rows]));
        // a verify takes the chosen token and up to the draft length more
        let mut s = engine.new_session(CONTEXT, rows - 1).expect("session");
        s.prefill(&ids[..PREFIX]);
        verify.extend(s.verify(&ids[PREFIX..PREFIX + rows]));
    }
    (prompt, verify)
}

/// The passes of 17 to 32 rows on `mmv32` give the logits of `mm` within
/// the tolerance
#[test]
#[ignore = "needs a model file"]
fn the_mmv32_logits_are_within_the_tolerance_of_mm() {
    let gguf = biv_cli::open_gguf(model_path()).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let model = Model::load(&gguf).expect("model");
    let ids = tok.encode(&TEXT.repeat(2));
    assert!(ids.len() >= PREFIX + PASS_ROWS[PASS_ROWS.len() - 1]);

    let mm = MetalEngine::with_mmv_rows(&model, METAL_BATCH, MMV16_ROWS).expect("mm engine");
    let mmv32 = MetalEngine::new(&model, METAL_BATCH).expect("mmv32 engine");
    let (mm_prompt, mm_verify) = logits(&mm, &ids);
    let (new_prompt, new_verify) = logits(&mmv32, &ids);
    let (cpu_prompt, cpu_verify) = logits(&&model, &ids);

    let kernels: Vec<(&str, Differences)> = [
        ("prompt pass", &cpu_prompt, &mm_prompt, &new_prompt),
        ("verify pass", &cpu_verify, &mm_verify, &new_verify),
    ]
    .into_iter()
    .map(|(what, reference, mm, new)| {
        let mut kernels = Differences::default();
        kernels.add(mm, new);
        let mut mm_cpu = Differences::default();
        mm_cpu.add(reference, mm);
        let mut new_cpu = Differences::default();
        new_cpu.add(reference, new);
        println!("{what}, mmv32 from mm:  {kernels}");
        println!("{what}, mm from cpu:    {mm_cpu}");
        println!("{what}, mmv32 from cpu: {new_cpu}");
        (what, kernels)
    })
    .collect();
    for (what, kernels) in kernels {
        assert!(kernels.abs <= MAX_ABS_LOGIT_DIFFERENCE, "{what}: {kernels}");
        assert!(kernels.rel <= MAX_REL_LOGIT_DIFFERENCE, "{what}: {kernels}");
    }
}
