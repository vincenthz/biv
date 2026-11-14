//! An agent loop whose tools do I/O: list a directory, read a file, wait for a
//! file another process writes. The model calls tools until it can answer; the
//! tools run here, on a workspace in a temporary directory, where a simulated
//! release build writes its result after a few seconds.
//! `agent <gguf> [--cpu] [--think] [task]`
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant};

use biv::{
    chat::{Chat, ChatOptions, Event, ReasoningEffort, ToolCall},
    engine::Engine,
    model::Model,
    sampler::{Sampler, SamplerConfig},
    tokenizer::Tokenizer,
};

const TOOLS: [&str; 3] = [
    r#"{"type": "function", "function": {"name": "list_dir", "description": "List the entries of a directory of the workspace; directories end with /", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Directory relative to the workspace root, . for the root"}}, "required": ["path"]}}}"#,
    r#"{"type": "function", "function": {"name": "read_file", "description": "Read a text file of the workspace", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "File relative to the workspace root"}}, "required": ["path"]}}}"#,
    r#"{"type": "function", "function": {"name": "wait_for_file", "description": "Wait until a file of the workspace exists, then read it. For results that other processes write, such as builds.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "File relative to the workspace root"}, "timeout_seconds": {"type": "number", "description": "How long to wait at most"}}, "required": ["path", "timeout_seconds"]}}}"#,
];

const TASK: &str = "A release build is running in the workspace and writes build/status.json when it \
finishes. Wait for it, then tell me whether it passed and how full the flash is, and what my notes say \
to do next, with the version number it implies.";

/// Tool calls answered before giving up
const MAX_STEPS: usize = 8;
/// Bytes of a file returned at most
const MAX_READ: u64 = 64 * 1024;
/// Seconds `wait_for_file` waits at most, whatever the model asks
const MAX_WAIT: f64 = 120.0;

/// The workspace of the firmware project
fn workspace() -> PathBuf {
    let root = std::env::temp_dir().join(format!("biv-agent-{}", std::process::id()));
    let files = [
        (
            "README.md",
            "# zephyr\n\nFirmware for the sensor board (STM32F405, 512 KB flash).\n",
        ),
        (
            "config.toml",
            "name = \"zephyr\"\nversion = \"2.4.1\"\n\n[build]\ntarget = \"thumbv7em-none-eabihf\"\nprofile = \"release\"\n",
        ),
        (
            "notes/todo.txt",
            "- once the release build passes, bump the minor version and tag it\n- if flash use goes over 80%, enable LTO before adding the BLE stack\n",
        ),
        ("build/.keep", ""),
    ];
    for (name, content) in files {
        let path = root.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }
    root
}

/// The release build, writing its result after `secs`
fn start_build(root: &Path, secs: u64) {
    let status = root.join("build/status.json");
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(secs));
        // written whole, then renamed: a reader never sees half of it
        let tmp = status.with_extension("tmp");
        std::fs::write(
            &tmp,
            r#"{"result": "success", "warnings": 3, "flash_used_kb": 431, "flash_total_kb": 512}"#,
        )
        .unwrap();
        std::fs::rename(tmp, status).unwrap();
    });
}

/// A path of the workspace: relative, without `..`
fn resolve(root: &Path, path: &str) -> Result<PathBuf, String> {
    let rel = Path::new(path.trim());
    if rel.is_absolute()
        || rel
            .components()
            .any(|c| matches!(c, Component::ParentDir | Component::Prefix(_)))
    {
        return Err(format!("{path}: outside the workspace"));
    }
    Ok(root.join(rel))
}

fn read(path: &Path) -> Result<String, String> {
    use std::io::Read;
    let f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut bytes = Vec::new();
    f.take(MAX_READ)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn json_string(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn arg<'a>(call: &'a ToolCall, name: &str) -> Result<&'a str, String> {
    call.arguments
        .iter()
        .find(|(k, _)| k == name)
        .map(|(_, v)| v.as_str())
        .ok_or(format!("missing parameter {name}"))
}

/// Run a call: its result as JSON, `{"error": ...}` when it fails
fn run(root: &Path, call: &ToolCall) -> String {
    let result = match call.name.as_str() {
        "list_dir" => arg(call, "path")
            .and_then(|p| resolve(root, p))
            .and_then(|dir| {
                let mut entries: Vec<String> = std::fs::read_dir(dir)
                    .map_err(|e| e.to_string())?
                    .filter_map(Result::ok)
                    .map(|e| {
                        let name = e.file_name().to_string_lossy().into_owned();
                        if e.path().is_dir() { name + "/" } else { name }
                    })
                    .collect();
                entries.sort();
                Ok(format!(
                    "{{\"entries\": [{}]}}",
                    entries
                        .iter()
                        .map(|e| json_string(e))
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            }),
        "read_file" => arg(call, "path")
            .and_then(|p| resolve(root, p))
            .and_then(|path| read(&path))
            .map(|content| format!("{{\"content\": {}}}", json_string(&content))),
        "wait_for_file" => (|| {
            let path = resolve(root, arg(call, "path")?)?;
            let timeout: f64 = arg(call, "timeout_seconds")?
                .trim()
                .parse()
                .map_err(|_| "timeout_seconds: not a number")?;
            let timeout = Duration::from_secs_f64(timeout.clamp(0.0, MAX_WAIT));
            let start = Instant::now();
            while !path.exists() {
                if start.elapsed() >= timeout {
                    return Err(format!("not there after {:.0} s", timeout.as_secs_f64()));
                }
                std::thread::sleep(Duration::from_millis(200));
            }
            let content = read(&path)?;
            Ok(format!(
                "{{\"waited_seconds\": {:.1}, \"content\": {}}}",
                start.elapsed().as_secs_f64(),
                json_string(&content)
            ))
        })(),
        name => Err(format!("unknown function {name}")),
    };
    result.unwrap_or_else(|e| format!("{{\"error\": {}}}", json_string(&e)))
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let cpu = args.iter().any(|a| a == "--cpu");
    let think = args.iter().any(|a| a == "--think");
    args.retain(|a| a != "--cpu" && a != "--think");
    let path = args
        .first()
        .expect("usage: agent <gguf> [--cpu] [--think] [task]")
        .clone();
    let task = args.get(1).cloned().unwrap_or(TASK.into());

    let g = biv_cli::open_gguf(&path).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let model = Model::load(&g).unwrap();
    #[cfg(target_os = "macos")]
    let metal = (!cpu).then(|| biv::metal::MetalEngine::new(&model, 256).unwrap());
    #[cfg(target_os = "macos")]
    let engine: &dyn Engine = match &metal {
        Some(m) => m,
        None => &model,
    };
    #[cfg(not(target_os = "macos"))]
    let engine: &dyn Engine = {
        let _ = cpu;
        &model
    };
    println!("{}", engine.description());

    let root = workspace();
    println!("workspace {}\n\n> {task}\n", root.display());
    let opts = ChatOptions {
        system: Some(
            "You work on the files of a firmware project through the tools. Be brief.".into(),
        ),
        thinking: think,
        effort: ReasoningEffort::Low,
        tools: TOOLS.iter().map(|t| t.to_string()).collect(),
        max_tokens: 2048,
        context: 16384,
        ..Default::default()
    };
    let mut chat = Chat::new(engine, &tok, opts).unwrap();
    let mut sampler = Sampler::new(SamplerConfig {
        temperature: 0.0,
        ..Default::default()
    });
    // whether the text printed so far ends in the middle of a line
    let mut mid_line = false;
    let mut on_event = |ev: Event<'_>| match ev {
        // the text events show the answer
        Event::Token(_) => {}
        Event::Thinking(t) => {
            print!("\x1b[2m{t}\x1b[0m");
            mid_line = !t.ends_with('\n');
        }
        Event::ThinkingDone => {
            println!("\n");
            mid_line = false;
        }
        Event::Answer(a) => {
            print!("{a}");
            mid_line = !a.ends_with('\n');
        }
        Event::ToolCall(c) => {
            if std::mem::take(&mut mid_line) {
                println!();
            }
            let args: Vec<String> = c
                .arguments
                .iter()
                .map(|(k, v)| format!("{k}={v:?}"))
                .collect();
            println!("-> {}({})", c.name, args.join(", "));
        }
        Event::PromptDone => {}
    };

    start_build(&root, 10);
    let mut stats = chat
        .send(&task, &mut sampler, &mut on_event, || false)
        .unwrap();
    let mut steps = 0;
    while !stats.tool_calls.is_empty() && steps < MAX_STEPS {
        steps += stats.tool_calls.len();
        let results: Vec<String> = stats
            .tool_calls
            .iter()
            .map(|call| {
                let t = Instant::now();
                let result = run(&root, call);
                let shown: String = result.chars().take(100).collect();
                println!(
                    "<- {} in {:.1}s: {shown}{}",
                    call.name,
                    t.elapsed().as_secs_f64(),
                    if shown.len() < result.len() {
                        "..."
                    } else {
                        ""
                    }
                );
                result
            })
            .collect();
        println!();
        stats = chat
            .send_tool_responses(&results, &mut sampler, &mut on_event, || false)
            .unwrap();
    }
    if !stats.tool_calls.is_empty() {
        println!("\n(stopped after {steps} tool calls)");
    }
    println!();
    let _ = std::fs::remove_dir_all(&root);
}
