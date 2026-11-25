//! jinja2rust [--name NAME] [--rt PATH] [-o OUT.rs] INPUT
//! jinja2rust registry OUT_DIR INPUT...
//!
//! INPUT is a Jinja chat template, or a GGUF whose `tokenizer.chat_template`
//! is compiled. The first form prints the Rust code of one template, or
//! writes it to OUT.rs. The second writes a module of all of them to OUT_DIR:
//! a file per template, named after it, the runtime, and `mod.rs`.

use std::io::{BufReader, Read};
use std::process::exit;

use jinja2rust::{Options, Source, generate, registry as registry_files, rustfmt};

fn usage() -> ! {
    eprintln!(
        "usage: jinja2rust [--name NAME] [--rt PATH] [-o OUT.rs] TEMPLATE.jinja|MODEL.gguf\n       \
         jinja2rust registry OUT_DIR TEMPLATE.jinja|MODEL.gguf..."
    );
    exit(2)
}

/// The template of `input`, a GGUF or a template file
fn read_template(input: &str) -> String {
    let fail = |e: std::io::Error| -> ! {
        eprintln!("{input}: {e}");
        exit(1)
    };
    match gguf_template(input) {
        Ok(Some(t)) => t,
        Ok(None) => std::fs::read_to_string(input).unwrap_or_else(|e| fail(e)),
        Err(e) => fail(e),
    }
}

fn stem(input: &str) -> String {
    std::path::Path::new(input)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or(input.to_owned())
}

fn registry(mut args: impl Iterator<Item = String>) {
    let Some(dir) = args.next() else { usage() };
    let inputs: Vec<String> = args.collect();
    if inputs.is_empty() {
        usage();
    }
    let texts: Vec<(String, String)> = inputs.iter().map(|i| (stem(i), read_template(i))).collect();
    let sources: Vec<Source<'_>> = texts
        .iter()
        .map(|(name, text)| Source { name, text })
        .collect();
    let files = registry_files(&sources).unwrap_or_else(|e| {
        eprintln!("{e}");
        exit(1)
    });
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| {
        eprintln!("{dir}: {e}");
        exit(1)
    });
    for (name, code) in files {
        let path = std::path::Path::new(&dir).join(&name);
        let code = rustfmt(&code).unwrap_or_else(|e| {
            eprintln!("rustfmt {name}: {e}");
            exit(1)
        });
        std::fs::write(&path, code).unwrap_or_else(|e| {
            eprintln!("{}: {e}", path.display());
            exit(1)
        });
        eprintln!("{}", path.display());
    }
}

/// The chat template in a GGUF's metadata, or `None` if not a GGUF
fn gguf_template(path: &str) -> std::io::Result<Option<String>> {
    struct Rd(BufReader<std::fs::File>);
    impl Rd {
        fn bytes(&mut self, n: usize) -> std::io::Result<Vec<u8>> {
            let mut b = vec![0; n];
            self.0.read_exact(&mut b)?;
            Ok(b)
        }
        fn u32(&mut self) -> std::io::Result<u32> {
            Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
        }
        fn u64(&mut self) -> std::io::Result<u64> {
            Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
        }
        fn skip(&mut self, n: u64) -> std::io::Result<()> {
            self.0.seek_relative(n as i64)
        }
    }
    let mut r = Rd(BufReader::new(std::fs::File::open(path)?));
    if r.bytes(4)? != b"GGUF" {
        return Ok(None);
    }
    let _version = r.u32()?;
    let _tensors = r.u64()?;
    let n_kv = r.u64()?;
    // the sizes of the scalar types, by type id
    let size = |t: u32| match t {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        10..=12 => 8,
        _ => 0,
    };
    for _ in 0..n_kv {
        let n = r.u64()? as usize;
        let key = r.bytes(n)?;
        match r.u32()? {
            8 => {
                let n = r.u64()? as usize;
                let s = r.bytes(n)?;
                if key == b"tokenizer.chat_template" {
                    return Ok(Some(String::from_utf8_lossy(&s).into_owned()));
                }
            }
            9 => {
                let et = r.u32()?;
                let n = r.u64()?;
                if et == 8 {
                    for _ in 0..n {
                        let len = r.u64()?;
                        r.skip(len)?;
                    }
                } else {
                    r.skip(n * size(et))?;
                }
            }
            t => r.skip(size(t))?,
        }
    }
    Ok(Some(String::new()).filter(|s| !s.is_empty()))
}

fn main() {
    let mut args = std::env::args().skip(1).peekable();
    if args.peek().is_some_and(|a| a == "registry") {
        args.next();
        return registry(args);
    }
    let (mut name, mut rt_path, mut out, mut input) = (None, "super::rt".to_owned(), None, None);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--name" => name = Some(args.next().unwrap_or_else(|| usage())),
            "--rt" => rt_path = args.next().unwrap_or_else(|| usage()),
            "-o" => out = Some(args.next().unwrap_or_else(|| usage())),
            "-h" | "--help" => usage(),
            _ if input.is_none() => input = Some(a),
            _ => usage(),
        }
    }
    let Some(input) = input else { usage() };
    let source = read_template(&input);
    let name = name.unwrap_or_else(|| stem(&input));
    let generated = match generate(
        &source,
        &Options {
            name: &name,
            rt_path: &rt_path,
        },
    ) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("{input}: {e}");
            exit(1)
        }
    };
    eprintln!("{name}: hash {:#018x}", generated.hash);
    for p in &generated.params {
        eprintln!("  {}: {}", p.name, p.kind.rust_type());
    }
    match out {
        Some(path) => std::fs::write(&path, generated.code).unwrap_or_else(|e| {
            eprintln!("{path}: {e}");
            exit(1)
        }),
        None => print!("{}", generated.code),
    }
}
