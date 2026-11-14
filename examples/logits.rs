//! The logits of a model at each position of a list of token ids, with no
//! tokenizer: a model whose GGUF holds no tokenizer, or a check of exact
//! logits on CPU against Metal.
//!
//! ```text
//! logits <gguf> --ids 1,2,3 --rows 5,9 --engine cpu|metal
//! ```
//!
//! The ids go in one at a time, so each pass takes one token, and after
//! each one the example prints the position, the id, the argmax of the
//! logits with its logit, and the logit of each chosen row. Each value is
//! printed as the shortest text that reads back to the same f32. Each input
//! is required, and a missing or wrong input fails by name.

use std::process::ExitCode;

use biv::engine::Engine;
use biv::model::Model;

/// The exit code of a usage error (sysexits `EX_USAGE`)
const USAGE_EXIT_CODE: u8 = 64;
/// The exit code of a model that does not load or run
const MODEL_EXIT_CODE: u8 = 65;
const USAGE: &str = "usage: logits <gguf> --ids <id,id,..> --rows <row,row,..> --engine cpu|metal";
const IDS_FLAG: &str = "--ids";
const ROWS_FLAG: &str = "--rows";
const ENGINE_FLAG: &str = "--engine";
const CPU: &str = "cpu";
const METAL: &str = "metal";
const LIST_SEPARATOR: char = ',';
/// The tokens of each forward pass: one, so each pass gives the logits of
/// one position
const PASS_TOKENS: usize = 1;
/// The drafts of the session: none, because nothing is drafted
const NO_DRAFT: usize = 0;

/// The engine that runs the passes
#[derive(Clone, Copy)]
enum EngineKind {
    Cpu,
    Metal,
}

/// The inputs of a run
struct Inputs {
    model: String,
    ids: Vec<u32>,
    rows: Vec<usize>,
    engine: EngineKind,
}

/// The name and the text of the error of a run
struct Failure {
    name: &'static str,
    text: String,
    code: u8,
}

impl Failure {
    fn usage(name: &'static str, text: impl Into<String>) -> Self {
        Failure {
            name,
            text: text.into(),
            code: USAGE_EXIT_CODE,
        }
    }

    fn model(name: &'static str, text: impl Into<String>) -> Self {
        Failure {
            name,
            text: text.into(),
            code: MODEL_EXIT_CODE,
        }
    }
}

/// The numbers of a list flag: each one, or the failure of the first that
/// is not a number
fn numbers<T: std::str::FromStr>(flag: &str, list: &str) -> Result<Vec<T>, Failure> {
    list.split(LIST_SEPARATOR)
        .map(|word| {
            word.parse().map_err(|_| {
                Failure::usage("NotANumber", format!("{flag}: {word:?} is not a number"))
            })
        })
        .collect()
}

fn parse(words: &[String]) -> Result<Inputs, Failure> {
    let (model, mut rest) = match words {
        [model, rest @ ..] => (model.clone(), rest),
        [] => return Err(Failure::usage("ModelMissing", "no GGUF path")),
    };
    let (mut ids, mut rows, mut engine) = (None, None, None);
    while let [flag, value, more @ ..] = rest {
        match flag.as_str() {
            IDS_FLAG => ids = Some(numbers::<u32>(IDS_FLAG, value)?),
            ROWS_FLAG => rows = Some(numbers::<usize>(ROWS_FLAG, value)?),
            ENGINE_FLAG => {
                engine = Some(match value.as_str() {
                    CPU => EngineKind::Cpu,
                    METAL => EngineKind::Metal,
                    other => {
                        return Err(Failure::usage(
                            "EngineUnknown",
                            format!("{ENGINE_FLAG}: {other:?} is not {CPU} or {METAL}"),
                        ));
                    }
                })
            }
            other => return Err(Failure::usage("FlagUnknown", format!("{other:?}"))),
        }
        rest = more;
    }
    if let [flag] = rest {
        return Err(Failure::usage(
            "ValueMissing",
            format!("{flag} has no value"),
        ));
    }
    Ok(Inputs {
        model,
        ids: ids.ok_or_else(|| Failure::usage("IdsMissing", format!("no {IDS_FLAG}")))?,
        rows: rows.ok_or_else(|| Failure::usage("RowsMissing", format!("no {ROWS_FLAG}")))?,
        engine: engine
            .ok_or_else(|| Failure::usage("EngineMissing", format!("no {ENGINE_FLAG}")))?,
    })
}

/// The engine of `kind` on `model`
fn engine<'m>(model: &'m Model<'m>, kind: EngineKind) -> Result<Box<dyn Engine + 'm>, Failure> {
    match kind {
        EngineKind::Cpu => Ok(Box::new(model)),
        #[cfg(target_os = "macos")]
        EngineKind::Metal => biv::metal::MetalEngine::new(model, PASS_TOKENS)
            .map(|e| Box::new(e) as Box<dyn Engine + 'm>)
            .map_err(|e| Failure::model("MetalEngineFailed", e.to_string())),
        #[cfg(not(target_os = "macos"))]
        EngineKind::Metal => Err(Failure::usage(
            "MetalUnavailable",
            "this platform has no Metal",
        )),
    }
}

fn run(inputs: &Inputs) -> Result<(), Failure> {
    let gguf = biv_cli::open_gguf(&inputs.model)
        .map_err(|e| Failure::model("ModelUnreadable", e.to_string()))?;
    let model = Model::load(&gguf).map_err(|e| Failure::model("ModelRefused", e.to_string()))?;
    let n_vocab = model.cfg.n_vocab;
    if let Some(id) = inputs.ids.iter().find(|id| **id as usize >= n_vocab) {
        return Err(Failure::usage(
            "IdPastVocabulary",
            format!("{id} is not below the {n_vocab} tokens of the model"),
        ));
    }
    if let Some(row) = inputs.rows.iter().find(|row| **row >= n_vocab) {
        return Err(Failure::usage(
            "RowPastVocabulary",
            format!("{row} is not below the {n_vocab} tokens of the model"),
        ));
    }
    let engine = engine(&model, inputs.engine)?;
    println!("engine: {}", engine.description());
    let mut session = engine
        .new_session(inputs.ids.len(), NO_DRAFT)
        .map_err(|e| Failure::model("SessionRefused", e.to_string()))?;
    for (position, id) in inputs.ids.iter().enumerate() {
        let logits = session.forward(&[*id]);
        let (argmax, top) = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .ok_or_else(|| Failure::model("NoLogits", "the pass gave no logits"))?;
        let rows: Vec<String> = inputs
            .rows
            .iter()
            .map(|row| format!("{row}:{}", logits[*row]))
            .collect();
        println!(
            "position={position} id={id} argmax={argmax} argmax_logit={top} rows={}",
            rows.join(&LIST_SEPARATOR.to_string())
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    let words: Vec<String> = std::env::args().skip(1).collect();
    match parse(&words).and_then(|inputs| run(&inputs)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(Failure { name, text, code }) => {
            eprintln!("logits: {name}: {text}\n{USAGE}");
            ExitCode::from(code)
        }
    }
}
