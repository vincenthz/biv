//! A fingerprint of the inference stack, to detect any change in what it
//! generates: what should decide the output (model, platform, backend,
//! generation settings), and a probe, a fixed run whose logits are recorded
//! bit for bit. The code isn't hashed: the logits show whether it changed the
//! output, as they show a change nothing listed accounts for (a new OS shader
//! compiler, another machine).
//!
//! The format is text, a `key=value` per line, in a fixed order. In values,
//! `\`, newlines and carriage returns are written `\\`, `\n` and `\r`.
//!
//! Hashes are 64-bit FNV-1a: enough to tell that something changed, not to
//! resist forgery.

use std::path::Path;

use biv::chat::{Chat, ChatOptions, Event};
use biv::engine::Engine;
use biv::gguf::Gguf;
use biv::sampler::{Sampler, SamplerConfig};
use biv::tokenizer::{TokenId, Tokenizer};

pub const FORMAT: &str = "biv-fingerprint/1";

/// Changes whenever the probe runs differently
const PROBE_VERSION: u64 = 1;
const PROBE_CONTEXT: usize = 1024;
/// Prompt tokens per pass; the last `PROBE_TAIL` go in a pass of their own.
/// On Metal, passes of 32 tokens or more use the flash attention kernel, and
/// shorter ones the split-K kernel.
const PROBE_BATCH: usize = 64;
const PROBE_TAIL: usize = 16;
/// Tokens generated greedily, one per pass, then again with drafts
const PROBE_DECODE: usize = 12;
const PROBE_DRAFT: usize = 2;
/// Logits recorded with their value, besides the hash of them all
const PROBE_TOP: usize = 8;
const PROBE_PROMPT: &str = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
<|im_start|>user\nSummarize the following notes in two sentences.\n\n\
The probe passes through every kernel: matrix products over 64 tokens, attention over \
long and short batches, the recurrent layers, decoding one token at a time, and drafts \
verified in one pass. Numbers such as 3.14159, 1e-6 and 0x7FFF, punctuation like (a, b) \
-> {c; d}, and some code: `fn main() { println!(\"hi\"); }`. Quelques mots en français, \
ein paar Wörter auf Deutsch, 日本語の文章も少し, and an emoji 🦀. The weather on Tuesday \
was cold and windy, so the meeting moved indoors; the budget review is postponed to \
next week, pending the final numbers from the finance team.<|im_end|>\n\
<|im_start|>assistant\n<think>\n\n</think>\n\n";
const PROBE_SYSTEM: &str = "You are a helpful assistant.";
const PROBE_QUESTION: &str = "Write a haiku about the sea.";
const PROBE_CHAT_TOKENS: usize = 24;

/// `key=value` lines, in order
#[derive(Default)]
pub struct Fingerprint {
    lines: Vec<(String, String)>,
}

fn escape(v: &str) -> String {
    v.replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

fn unescape(v: &str) -> String {
    let mut out = String::with_capacity(v.len());
    let mut chars = v.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some(c) => out.push(c),
            None => out.push('\\'),
        }
    }
    out
}

impl Fingerprint {
    fn push(&mut self, key: impl Into<String>, value: impl ToString) {
        self.lines.push((key.into(), value.to_string()));
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.lines
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        let mut f = Fingerprint::default();
        for (n, line) in text.lines().enumerate() {
            if line.is_empty() {
                continue;
            }
            let (k, v) = line
                .split_once('=')
                .ok_or_else(|| format!("line {}: not key=value", n + 1))?;
            f.push(k, unescape(v));
        }
        Ok(f)
    }

    pub fn to_text(&self) -> String {
        self.lines
            .iter()
            .map(|(k, v)| format!("{k}={}\n", escape(v)))
            .collect()
    }

    fn hash(&self) -> String {
        hash(self.to_text().as_bytes())
    }

    /// The lines of the stack: not the header, the probe, or the run's info
    fn stack(&self) -> impl Iterator<Item = &(String, String)> {
        self.lines.iter().filter(|(k, _)| {
            !k.starts_with("probe.")
                && !k.starts_with("info.")
                && !["format", "stack_id", "output_id"].contains(&k.as_str())
        })
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// FNV-1a, 64 bits, in hex: unlike std's hasher, the same on every build
fn hash(bytes: &[u8]) -> String {
    let h = bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3)
    });
    format!("{h:016x}")
}

fn command(program: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(program)
        .args(args)
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (out.status.success() && !s.is_empty()).then_some(s)
}

/// The OS release: its libm and, on macOS, the Metal shader compiler
fn os() -> String {
    if cfg!(target_os = "macos") {
        let version = command("sw_vers", &["-productVersion"]).unwrap_or_default();
        let build = command("sw_vers", &["-buildVersion"]).unwrap_or_default();
        return format!("macOS {version} ({build})");
    }
    if cfg!(target_os = "linux") {
        let release = std::fs::read_to_string("/etc/os-release").unwrap_or_default();
        let name = release
            .lines()
            .find_map(|l| l.strip_prefix("PRETTY_NAME="))
            .map(|n| n.trim_matches('"').to_string())
            .unwrap_or("Linux".into());
        let kernel = std::fs::read_to_string("/proc/sys/kernel/osrelease").unwrap_or_default();
        return format!("{name}, kernel {}", kernel.trim());
    }
    std::env::consts::OS.to_string()
}

fn cpu() -> String {
    if cfg!(target_os = "macos") {
        return command("sysctl", &["-n", "machdep.cpu.brand_string"]).unwrap_or_default();
    }
    std::fs::read_to_string("/proc/cpuinfo")
        .unwrap_or_default()
        .lines()
        .find_map(|l| l.strip_prefix("model name").and_then(|r| r.split_once(':')))
        .map(|(_, v)| v.trim().to_string())
        .unwrap_or_default()
}

/// How the chat generates, as configured
pub struct Generation<'a> {
    pub chat: &'a ChatOptions,
    pub sampler: &'a SamplerConfig,
    /// a new seed every run: sampled text cannot be reproduced
    pub random_seed: bool,
}

/// What should decide the generated text, besides the code. The weights are
/// only in it through `model_sha256`, the SHA-256 of the model's file (for a
/// split model, the XOR of its shards'), when the caller knows it; the
/// probe's logits show a change to them either way.
pub fn stack(
    model_sha256: Option<[u8; 32]>,
    gguf: &Gguf,
    engine: &dyn Engine,
    generation: &Generation<'_>,
) -> Fingerprint {
    let mut f = Fingerprint::default();
    let general = &gguf.metadata.general;
    f.push("model.name", general.name.as_deref().unwrap_or_default());
    f.push(
        "model.architecture",
        general.architecture.as_deref().unwrap_or_default(),
    );
    if let Some(sum) = model_sha256 {
        f.push("model.sha256", hex(&sum));
    }
    f.push("platform.os", os());
    for (k, v) in engine.numerics() {
        f.push(format!("backend.{k}"), v);
    }
    let c = generation.chat;
    f.push("generation.context", c.context);
    f.push(
        "generation.batch",
        c.batch.unwrap_or_else(|| engine.batch_size()),
    );
    f.push(
        "generation.draft",
        c.draft.unwrap_or_else(|| engine.draft_size()),
    );
    // the renderer of the prompts: a compiled chat template, or the built-in formats
    let template = gguf
        .metadata
        .tokenizer
        .chat_template
        .as_deref()
        .and_then(biv::template::Template::recognize)
        .filter(|_| !c.builtin_format);
    f.push(
        "generation.chat_template",
        template.map_or("builtin", |t| t.name()),
    );
    f.push("generation.thinking", c.thinking);
    f.push(
        "generation.effort",
        format!("{:?}", c.effort).to_lowercase(),
    );
    f.push("generation.max_tokens", c.max_tokens);
    if let Some(system) = &c.system {
        f.push("generation.system_hash", hash(system.as_bytes()));
    }
    if !c.tools.is_empty() {
        f.push("generation.tools_hash", hash(c.tools.join("\n").as_bytes()));
    }
    let s = generation.sampler;
    f.push("generation.temperature", s.temperature);
    f.push("generation.top_k", s.top_k);
    f.push("generation.top_p", s.top_p);
    f.push("generation.min_p", s.min_p);
    f.push("generation.presence_penalty", s.presence_penalty);
    f.push("generation.repeat_penalty", s.repeat_penalty);
    f.push("generation.repeat_last_n", s.repeat_last_n);
    let seed = if generation.random_seed {
        "random".into()
    } else {
        s.seed.to_string()
    };
    f.push("generation.seed", seed);
    f
}

/// Facts about the run that don't change the output
pub fn info(model_path: &Path, threads: usize) -> Fingerprint {
    let mut f = Fingerprint::default();
    f.push("info.cpu", cpu());
    f.push("info.threads", threads);
    f.push("info.model_path", model_path.display());
    f
}

/// Token indices by decreasing logit, ties by index
fn top(logits: &[f32], k: usize) -> Vec<usize> {
    let order = |&a: &usize, &b: &usize| logits[b].total_cmp(&logits[a]).then(a.cmp(&b));
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    let k = k.min(idx.len());
    if k < idx.len() {
        idx.select_nth_unstable_by(k, order);
        idx.truncate(k);
    }
    idx.sort_unstable_by(order);
    idx
}

fn argmax(logits: &[f32]) -> TokenId {
    top(logits, 1)[0] as TokenId
}

fn joined<T: ToString>(items: impl IntoIterator<Item = T>) -> String {
    items
        .into_iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

/// `<key>.logits`, the hash of all the logits' bits, and `<key>.top`, the
/// largest ones as `id:value`
fn record(f: &mut Fingerprint, key: &str, logits: &[f32]) {
    let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
    f.push(format!("{key}.logits"), hash(&bytes));
    let top = top(logits, PROBE_TOP)
        .into_iter()
        .map(|i| format!("{i}:{}", logits[i]));
    f.push(format!("{key}.top"), joined(top));
}

/// [`PROBE_PROMPT`] in the model's chat format: ChatML, or Flint's role tokens
fn probe_prompt(tok: &Tokenizer) -> String {
    if tok.token_id("<|im_start|>").is_some() || tok.token_id("<|assistant|>").is_none() {
        return PROBE_PROMPT.into();
    }
    PROBE_PROMPT
        .replace("<|im_start|>system\n", "<|system|>")
        .replace("<|im_end|>\n<|im_start|>user\n", "<|user|>")
        .replace(
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            "<|assistant|>",
        )
}

/// The fixed run: a prompt in passes of several sizes, greedy decoding, the
/// same with drafts from the restored prompt, then a sampled chat turn
pub fn probe(engine: &dyn Engine, tok: &Tokenizer) -> Result<Fingerprint, biv::Error> {
    let mut f = Fingerprint::default();
    f.push("probe.version", PROBE_VERSION);
    let prompt = tok.encode_with_special(&probe_prompt(tok));
    f.push("probe.tokens", joined(&prompt));
    let mut session = engine.new_session(PROBE_CONTEXT, PROBE_DRAFT)?;

    let (head, tail) = prompt.split_at(prompt.len() - PROBE_TAIL);
    let mut logits = Vec::new();
    for (i, pass) in head.chunks(PROBE_BATCH).chain([tail]).enumerate() {
        logits = session.forward(pass);
        let key = format!("probe.prompt.{}", i + 1);
        f.push(format!("{key}.tokens"), pass.len());
        record(&mut f, &key, &logits);
    }
    let snapshot = session.snapshot()?;
    let after_prompt = logits.clone();

    for i in 1..=PROBE_DECODE {
        let next = argmax(&logits);
        logits = session.forward(&[next]);
        let key = format!("probe.decode.{i}");
        f.push(format!("{key}.token"), next);
        record(&mut f, &key, &logits);
    }

    session.restore(&snapshot)?;
    logits = after_prompt;
    let mut decoded = 0;
    let mut pass = 0;
    while decoded < PROBE_DECODE {
        pass += 1;
        let next = argmax(&logits);
        let mut tokens = vec![next];
        tokens.extend(session.draft(next, PROBE_DRAFT, &mut |l| l.argmax()));
        let mut rows = session.verify(&tokens);
        let kept = 1
            + (1..tokens.len())
                .take_while(|&i| tokens[i] == argmax(&rows[i - 1]))
                .count();
        session.keep(kept);
        let key = format!("probe.verify.{pass}");
        f.push(format!("{key}.tokens"), joined(&tokens));
        f.push(format!("{key}.kept"), kept);
        for (r, row) in rows.iter().enumerate() {
            record(&mut f, &format!("{key}.{}", r + 1), row);
        }
        logits = rows.swap_remove(kept - 1);
        decoded += kept;
    }
    drop(session);

    let opts = ChatOptions {
        system: Some(PROBE_SYSTEM.into()),
        thinking: false,
        max_tokens: PROBE_CHAT_TOKENS,
        context: PROBE_CONTEXT,
        batch: Some(PROBE_BATCH),
        draft: Some(0),
        ..Default::default()
    };
    let mut chat = Chat::new(engine, tok, opts)?;
    let mut sampler = Sampler::new(SamplerConfig {
        temperature: 0.7,
        top_k: 20,
        top_p: 0.95,
        seed: 1,
        ..Default::default()
    });
    let mut text = String::new();
    // the reasoning too, which a model like Flint starts with in any case
    let on_event = |ev: Event<'_>| {
        if let Event::Answer(a) | Event::Thinking(a) = ev {
            text.push_str(a)
        }
    };
    let stats = chat.send(PROBE_QUESTION, &mut sampler, on_event, || false)?;
    f.push("probe.chat.tokens", stats.gen_tokens);
    f.push("probe.chat.text", text);
    Ok(f)
}

/// The whole fingerprint, with an id for the stack and one for the output
pub fn document(stack: Fingerprint, probe: Fingerprint, info: Fingerprint) -> Fingerprint {
    let mut f = Fingerprint::default();
    f.push("format", FORMAT);
    f.push("stack_id", stack.hash());
    f.push("output_id", probe.hash());
    f.lines.extend(stack.lines);
    f.lines.extend(probe.lines);
    f.lines.extend(info.lines);
    f
}

/// A logits record of the probe: its phase, where it comes from, its input,
/// and the key of its values
struct Record<'a> {
    phase: &'a str,
    what: String,
    input: Option<String>,
    key: &'a str,
}

fn records(f: &Fingerprint) -> Vec<Record<'_>> {
    let mut out = Vec::new();
    for (k, _) in &f.lines {
        let Some(key) = k.strip_suffix(".logits") else {
            continue;
        };
        let get = |k: String| f.get(&k).map(String::from);
        let (phase, what, input) = match key.split('.').collect::<Vec<_>>()[..] {
            ["probe", "prompt", n] => {
                let tokens = get(format!("{key}.tokens"));
                let what = format!(
                    "prompt pass {n} ({} tokens)",
                    tokens.as_deref().unwrap_or("?")
                );
                ("prompt", what, tokens)
            }
            ["probe", "decode", n] => (
                "decode",
                format!("decode step {n}"),
                get(format!("{key}.token")),
            ),
            ["probe", "verify", n, r] => {
                let tokens = get(format!("probe.verify.{n}.tokens")).unwrap_or_default();
                let i = r.parse::<usize>().unwrap_or(1).saturating_sub(1);
                let token = tokens.split(' ').nth(i).map(String::from);
                ("verify", format!("verify pass {n}, row {r}"), token)
            }
            _ => continue,
        };
        out.push(Record {
            phase,
            what,
            input,
            key,
        });
    }
    out
}

/// The `id:value` pairs of a record
fn top_of(f: &Fingerprint, key: &str) -> Vec<(u64, f64)> {
    f.get(&format!("{key}.top"))
        .unwrap_or_default()
        .split(' ')
        .filter_map(|p| {
            let (i, v) = p.split_once(':')?;
            Some((i.parse().ok()?, v.parse().unwrap_or(f64::NAN)))
        })
        .collect()
}

/// What differs between a saved fingerprint and a new one, as lines for
/// people, and whether they are the same; `decode` gives a token's text
pub fn compare(
    old: &Fingerprint,
    new: &Fingerprint,
    decode: impl Fn(TokenId) -> String,
) -> Result<(Vec<String>, bool), String> {
    if old.get("format") != Some(FORMAT) {
        return Err(format!("not a {FORMAT} file"));
    }
    let mut lines = Vec::new();
    let stack_same = old.get("stack_id") == new.get("stack_id");
    let output_same = old.get("output_id") == new.get("output_id");

    if stack_same {
        lines.push("stack: same".to_string());
    } else {
        lines.push("stack: different".to_string());
        let find =
            |f: &Fingerprint, k: &str| f.stack().find(|(n, _)| n == k).map(|(_, v)| v.clone());
        for (k, v) in old.stack() {
            match find(new, k) {
                Some(w) if &w == v => {}
                Some(w) => lines.push(format!("  {k}: {v} -> {w}")),
                None => lines.push(format!("  {k}: {v} -> (none)")),
            }
        }
        for (k, w) in new.stack() {
            if find(old, k).is_none() {
                lines.push(format!("  {k}: (none) -> {w}"));
            }
        }
    }

    if output_same {
        lines.push("output: identical".to_string());
        return Ok((lines, stack_same));
    }
    lines.push("output: different".to_string());
    let (vo, vn) = (old.get("probe.version"), new.get("probe.version"));
    if vo != vn {
        lines.push(format!(
            "  probe version {} -> {}: the runs differ, they cannot be compared",
            vo.unwrap_or("(none)"),
            vn.unwrap_or("(none)")
        ));
        return Ok((lines, false));
    }
    let show = |id: u64| format!("{id} {:?}", decode(id as TokenId));
    let tokens = |f: &Fingerprint| -> Vec<String> {
        let t = f.get("probe.tokens").unwrap_or_default();
        t.split(' ').map(String::from).collect()
    };
    let (to, tn) = (tokens(old), tokens(new));
    if to != tn {
        let i = to.iter().zip(&tn).take_while(|(a, b)| a == b).count();
        lines.push(format!(
            "  prompt tokens ({} -> {}) differ from token {i}: {} -> {}: the tokenizer changed",
            to.len(),
            tn.len(),
            to.get(i).map_or("(end)", |v| v),
            tn.get(i).map_or("(end)", |v| v),
        ));
        return Ok((lines, false));
    }

    let (ro, rn) = (records(old), records(new));
    let (mut identical, mut compared, mut max_diff) = (0, 0, 0f64);
    let mut first_diff = None;
    let mut reordered = (0, None);
    // where the inputs stopped being the same, per phase: decoding and
    // drafting both start from the prompt, so each is compared to its end
    let mut diverged = Vec::new();
    for (o, n) in ro.iter().zip(&rn) {
        if diverged.contains(&o.phase) {
            continue;
        }
        if o.key != n.key || o.input != n.input {
            // after a token chosen differently, or other drafts
            lines.push(format!(
                "  {}: other input ({} -> {}), not compared from there",
                o.what,
                o.input.as_deref().unwrap_or("?"),
                n.input.as_deref().unwrap_or("?"),
            ));
            diverged.push(o.phase);
            continue;
        }
        compared += 1;
        let hash = format!("{}.logits", o.key);
        if old.get(&hash) == new.get(&hash) {
            identical += 1;
            continue;
        }
        let (a, b) = (top_of(old, o.key), top_of(new, n.key));
        let diff = a
            .iter()
            .filter_map(|(i, v)| b.iter().find(|(j, _)| j == i).map(|(_, w)| (v - w).abs()))
            .fold(0f64, f64::max);
        max_diff = max_diff.max(diff);
        first_diff.get_or_insert(format!("{}, top logits differ by up to {diff:.3e}", o.what));
        let ids = |t: &[(u64, f64)]| t.iter().map(|t| t.0).collect::<Vec<_>>();
        if a.first().map(|t| t.0) != b.first().map(|t| t.0) {
            lines.push(format!(
                "  different token after {}: {} -> {}",
                o.what,
                a.first().map_or("(none)".into(), |t| show(t.0)),
                b.first().map_or("(none)".into(), |t| show(t.0)),
            ));
            diverged.push(o.phase);
        } else if ids(&a) != ids(&b) {
            reordered.0 += 1;
            reordered.1.get_or_insert(o.what.clone());
        }
    }
    lines.push(format!(
        "  logits: {identical} of {compared} compared identical, {} recorded",
        ro.len()
    ));
    if let Some(d) = first_diff {
        lines.push(format!("  first difference: {d}"));
        lines.push(format!(
            "  largest difference of the top logits: {max_diff:.3e}"
        ));
    }
    if let (n, Some(what)) = reordered {
        let times = if n == 1 {
            "once".into()
        } else {
            format!("{n} times")
        };
        lines.push(format!(
            "  same top token, other top {PROBE_TOP} order: {times}, first after {what}"
        ));
    }
    let (co, cn) = (old.get("probe.chat.text"), new.get("probe.chat.text"));
    if co == cn {
        lines.push("  chat text: same".to_string());
    } else {
        lines.push(format!(
            "  chat text: {:?}\n          -> {:?}",
            co.unwrap_or_default(),
            cn.unwrap_or_default()
        ));
    }
    Ok((lines, false))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_orders_by_value_then_index() {
        assert_eq!(top(&[1.0, 3.0, 2.0, 3.0, -1.0], 3), [1, 3, 2]);
        assert_eq!(top(&[0.5, 0.5], 8), [0, 1]);
        assert_eq!(argmax(&[0.0, -0.0, 0.0]), 0);
    }

    #[test]
    fn text_reads_back() {
        let mut f = Fingerprint::default();
        f.push("a", "x=1\ny\\n\r");
        f.push("b.c", "");
        let text = f.to_text();
        assert_eq!(text, "a=x=1\\ny\\\\n\\r\nb.c=\n");
        assert_eq!(Fingerprint::parse(&text).unwrap().lines, f.lines);
        assert!(Fingerprint::parse("no equal sign").is_err());
    }

    fn doc(prompt: &[f32], decoded: &[&[f32]], verify: &[f32], device: &str) -> Fingerprint {
        let mut stack = Fingerprint::default();
        stack.push("backend.device", device);
        let mut p = Fingerprint::default();
        p.push("probe.version", PROBE_VERSION);
        p.push("probe.tokens", "1 2 3 4");
        p.push("probe.prompt.1.tokens", 4);
        record(&mut p, "probe.prompt.1", prompt);
        let mut prev = prompt;
        for (i, l) in decoded.iter().enumerate() {
            p.push(format!("probe.decode.{}.token", i + 1), argmax(prev));
            record(&mut p, &format!("probe.decode.{}", i + 1), l);
            prev = l;
        }
        p.push("probe.verify.1.tokens", argmax(prompt));
        p.push("probe.verify.1.kept", 1);
        record(&mut p, "probe.verify.1.1", verify);
        p.push("probe.chat.text", "hi");
        let doc = document(stack, p, Fingerprint::default());
        // as read back from a file
        Fingerprint::parse(&doc.to_text()).unwrap()
    }

    fn check(old: &Fingerprint, new: &Fingerprint) -> (String, bool) {
        let (lines, same) = compare(old, new, |id| format!("t{id}")).unwrap();
        (lines.join("\n"), same)
    }

    #[test]
    fn compare_reports_what_changed() {
        let p = [0.0, 2.0, 1.0];
        let base = doc(
            &p,
            &[&[3.0, 0.0, 1.0], &[0.0, 0.0, 5.0]],
            &[1.0, 0.0, 0.0],
            "gpu",
        );

        let (text, same) = check(&base, &base);
        assert!(same);
        assert_eq!(text, "stack: same\noutput: identical");

        let other = doc(
            &p,
            &[&[3.0, 0.0, 1.0], &[0.0, 0.0, 5.0]],
            &[1.0, 0.0, 0.0],
            "cpu",
        );
        let (text, same) = check(&base, &other);
        assert!(!same);
        assert!(text.contains("backend.device: gpu -> cpu"), "{text}");
        assert!(text.contains("output: identical"), "{text}");

        // the second decoded token changes: the decode phase stops there, the
        // verify pass from the prompt is still compared
        let other = doc(
            &p,
            &[&[3.0, 4.0, 1.0], &[0.0, 0.0, 5.0]],
            &[1.0, 0.0, 0.0],
            "gpu",
        );
        let (text, same) = check(&base, &other);
        assert!(!same);
        assert!(
            text.contains(r#"different token after decode step 1: 0 "t0" -> 1 "t1""#),
            "{text}"
        );
        assert!(
            text.contains("logits: 2 of 3 compared identical, 4 recorded"),
            "{text}"
        );
        assert!(
            text.contains("top logits differ by up to 4.000e0"),
            "{text}"
        );
        assert!(text.contains("chat text: same"), "{text}");
    }
}
