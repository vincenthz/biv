//! The template renderer of biv against the reference renderer, on the
//! corpus of tests/templates: each template on each case of
//! conversations.json, against the bytes or the failure that
//! tools/template_fixtures.py recorded with jinja2 and transformers.

use std::path::{Path, PathBuf};

use biv::template::{Date, ErrorKind, Origin, Template, TemplateError, Value, json};

const CORPUS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/templates/corpus");
const EXPECTED: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/templates/expected");
const CONVERSATIONS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/templates/conversations.json"
);

/// The date of the fixtures (tools/template_fixtures.py FIXED_DATE)
const FIXED_DATE: Date = Date {
    year: 2026,
    month: 1,
    day: 5,
};

/// The top-level names that biv sets in each render, as transformers passes
/// them: the messages, the tools and the documents (none when there are
/// none), the generation prompt, and the variables of each case
const VARIABLES: [&str; 9] = [
    "messages",
    "tools",
    "documents",
    "add_generation_prompt",
    "bos_token",
    "eos_token",
    "enable_thinking",
    "preserve_thinking",
    "reasoning_effort",
];

/// The variables of a case whose values are texts of the template, not of
/// the caller: the special token texts
const TEMPLATE_TEXT_VARIABLES: [&str; 2] = ["bos_token", "eos_token"];

/// The templates that the subset refuses at load, and why
const REFUSED_AT_LOAD: [(&str, &str); 5] = [
    ("CohereForAI-c4ai-command-r-plus-tool_use", "indent"),
    ("Kimi-K2-Instruct", "the list method append"),
    ("Kimi-K2-Thinking", "the list method append"),
    ("fireworks-ai-llama-3-firefunction-v2", "the filter block"),
    ("meetkai-functionary-medium-v3.2", "unique"),
];

fn read_json(path: &Path, origin: Origin) -> Value {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    json::parse(&text, origin).unwrap_or_else(|e| panic!("{}: {e:?}", path.display()))
}

fn field(v: &Value, key: &str) -> Option<Value> {
    match v {
        Value::Map(m) => Value::map_get_str(m, key),
        _ => None,
    }
}

fn text(v: &Value) -> String {
    match v {
        Value::Str(s) => s.as_str().into(),
        other => panic!("not a string: {other:?}"),
    }
}

fn items(v: &Value) -> Vec<Value> {
    match v {
        Value::List(l) => l.to_vec(),
        other => panic!("not a list: {other:?}"),
    }
}

/// A value of a case with the origin of a template text
fn template_text(v: &Value) -> Value {
    Value::str(&text(v), Origin::Template)
}

/// The variables of the render of `case`
fn variables(case: &Value) -> Vec<(String, Value)> {
    let mut vars = vec![
        (
            "messages".to_string(),
            field(case, "messages").expect("messages"),
        ),
        (
            "tools".to_string(),
            field(case, "tools").unwrap_or(Value::None),
        ),
        ("documents".to_string(), Value::None),
        (
            "add_generation_prompt".to_string(),
            field(case, "add_generation_prompt").expect("add_generation_prompt"),
        ),
    ];
    let Some(Value::Map(case_vars)) = field(case, "variables") else {
        panic!("a case with no variables");
    };
    for (k, v) in case_vars.iter() {
        let k = text(k);
        let v = if TEMPLATE_TEXT_VARIABLES.contains(&k.as_str()) || k == "reasoning_effort" {
            template_text(v)
        } else {
            v.clone()
        };
        vars.push((k, v));
    }
    vars
}

/// The failure type of the reference for a failure of biv
fn reference_failure(e: &TemplateError) -> &'static str {
    match e.kind {
        ErrorKind::Raised(_) => "TemplateError",
        ErrorKind::Type(_) => "TypeError",
        ErrorKind::Undefined(_) => "UndefinedError",
        ErrorKind::Value(_) => "ValueError",
        _ => "a failure of biv only",
    }
}

fn corpus() -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(CORPUS)
        .expect("the corpus")
        .map(|e| e.expect("a corpus file").path())
        .filter(|p| p.extension().is_some_and(|x| x == "jinja"))
        .collect();
    paths.sort();
    paths
}

#[test]
fn each_template_renders_each_case_as_the_reference_does() {
    let cases = items(
        &field(
            &read_json(Path::new(CONVERSATIONS), Origin::Content),
            "cases",
        )
        .expect("cases"),
    );
    let mut refused = Vec::new();
    let mut differ = Vec::new();
    let mut variable_unknown = Vec::new();
    let mut compared = 0;
    for path in corpus() {
        let stem = path
            .file_stem()
            .expect("a name")
            .to_string_lossy()
            .into_owned();
        let source = std::fs::read_to_string(&path).expect("a template");
        let template = match Template::load(&source, &VARIABLES) {
            Ok(t) => t,
            Err(e) => {
                refused.push((stem, e.to_string()));
                continue;
            }
        };
        let fixture = read_json(
            &Path::new(EXPECTED).join(format!("{stem}.json")),
            Origin::Content,
        );
        for (case, expected) in cases
            .iter()
            .zip(items(&field(&fixture, "cases").expect("cases")))
        {
            let name = text(&field(case, "name").expect("a name"));
            compared += 1;
            let got = template.render(&variables(case), Some(FIXED_DATE));
            match (got, field(&expected, "output"), field(&expected, "failure")) {
                (Ok(segments), Some(out), None) => {
                    let rendered: String = segments.iter().map(|s| s.text.as_str()).collect();
                    if rendered != text(&out) {
                        differ.push(format!("{stem} {name}: the bytes differ\n--- biv\n{rendered}\n--- reference\n{}", text(&out)));
                    }
                }
                (Err(e), Some(_), None) if matches!(e.kind, ErrorKind::VariableUnknown(_)) => {
                    variable_unknown.push(format!("{stem} {name}: {e}"));
                }
                (Err(e), None, Some(failure)) => {
                    if reference_failure(&e) != text(&failure) {
                        differ.push(format!(
                            "{stem} {name}: biv {e}, the reference {}",
                            text(&failure)
                        ));
                    }
                }
                (Ok(_), None, Some(failure)) => {
                    differ.push(format!(
                        "{stem} {name}: biv renders, the reference fails with {}",
                        text(&failure)
                    ));
                }
                (Err(e), Some(_), None) => {
                    differ.push(format!("{stem} {name}: biv {e}, the reference renders"))
                }
                _ => panic!("{stem} {name}: a fixture with neither an output nor a failure"),
            }
        }
    }
    println!(
        "compared={compared} differ={} variable_unknown={}",
        differ.len(),
        variable_unknown.len()
    );
    for v in &variable_unknown {
        println!("variable_unknown: {v}");
    }
    let refused_names: Vec<(&str, &str)> = refused
        .iter()
        .map(|(s, e)| (s.as_str(), e.as_str()))
        .collect();
    for (stem, why) in REFUSED_AT_LOAD {
        assert!(
            refused_names
                .iter()
                .any(|(s, e)| *s == stem && e.contains(why)),
            "{stem} is not refused at load for {why}: {refused_names:?}"
        );
    }
    assert_eq!(
        refused.len(),
        REFUSED_AT_LOAD.len(),
        "refused at load: {refused_names:?}"
    );
    assert!(
        differ.is_empty(),
        "{} cases differ:\n{}",
        differ.len(),
        differ.join("\n")
    );
}

/// The variables of the case `name` of conversations.json, with the
/// content of its last message set to `content`
fn case_with_content(name: &str, content: &str) -> Vec<(String, Value)> {
    let cases = items(
        &field(
            &read_json(Path::new(CONVERSATIONS), Origin::Content),
            "cases",
        )
        .expect("cases"),
    );
    let case = cases
        .iter()
        .find(|c| field(c, "name").is_some_and(|n| text(&n) == name))
        .unwrap_or_else(|| panic!("no case {name}"));
    let mut vars = variables(case);
    let messages = vars
        .iter_mut()
        .find(|(n, _)| n == "messages")
        .expect("messages");
    let mut list = items(&messages.1);
    let Some(Value::Map(last)) = list.pop() else {
        panic!("a message that is not a mapping");
    };
    let mut last = last.to_vec();
    for (k, v) in last.iter_mut() {
        if text(k) == "content" {
            *v = Value::str(content, Origin::Content);
        }
    }
    list.push(Value::map(last));
    messages.1 = Value::list(list);
    vars
}

/// The cases of the probes of the origin tests, in order: one user
/// message, then one user message with tools, for a template that the
/// reference also fails with no tools
const PROBE_CASES: [&str; 2] = ["user", "tools"];

/// The first probe case that `template` renders, and its segments, with
/// the content of its last message set to `content`
fn probe(
    template: &Template,
    content: &str,
) -> Option<(&'static str, Vec<biv::template::Segment>)> {
    PROBE_CASES.iter().find_map(|case| {
        template
            .render(&case_with_content(case, content), Some(FIXED_DATE))
            .ok()
            .map(|s| (*case, s))
    })
}

/// The templates of the corpus that load, with their names
fn loaded() -> Vec<(String, Template)> {
    corpus()
        .into_iter()
        .filter_map(|p| {
            let stem = p
                .file_stem()
                .expect("a name")
                .to_string_lossy()
                .into_owned();
            let source = std::fs::read_to_string(&p).expect("a template");
            Template::load(&source, &VARIABLES).ok().map(|t| (stem, t))
        })
        .collect()
}

fn joined(segments: &[biv::template::Segment], origin: Origin) -> String {
    segments
        .iter()
        .filter(|s| s.origin == origin)
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join("\u{0}")
}

#[test]
fn a_message_with_the_text_of_the_template_stays_content() {
    let mut checked = 0;
    for (stem, template) in loaded() {
        // a template that the reference fails on every probe case is
        // skipped: the corpus test checks those failures
        let Some((case, probe)) = probe(&template, "x") else {
            continue;
        };
        // all the text of the template in the probe, with its special token
        // texts, as the content of the message
        let template_text: String = probe
            .iter()
            .filter(|s| s.origin == Origin::Template)
            .map(|s| s.text.as_str())
            .collect();
        let segments = template
            .render(&case_with_content(case, &template_text), Some(FIXED_DATE))
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let content = joined(&segments, Origin::Content);
        assert!(
            content.contains(template_text.trim()),
            "{stem}: the text of the message is not all Content"
        );
        checked += 1;
    }
    println!("origin_checks={checked}");
    assert!(checked > 0, "no template rendered the probe");
}

/// The variable that names the GGUF files whose vocabularies the special
/// token test reads, separated by ':'
const VOCABULARIES_ENV: &str = "BIV_TEST_VOCABULARIES";

#[test]
#[ignore = "needs GGUF files: BIV_TEST_VOCABULARIES=<a.gguf>:<b.gguf>"]
fn a_message_with_special_token_texts_gives_no_control_token() {
    use biv::tokenizer::{PromptPart, Tokenizer};
    let paths = std::env::var(VOCABULARIES_ENV)
        .unwrap_or_else(|_| panic!("{VOCABULARIES_ENV} must name GGUF files"));
    let mut checked = 0;
    for path in paths.split(':') {
        let gguf = biv_cli::open_gguf(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let tok = Tokenizer::from_gguf(&gguf).unwrap_or_else(|e| panic!("{path}: {e}"));
        let specials: Vec<(u32, String)> = (0..tok.n_vocab() as u32)
            .filter(|id| tok.is_special(*id) && !tok.token_text(*id).is_empty())
            .map(|id| (id, tok.token_text(id).to_string()))
            .collect();
        let tokens = |segments: &[biv::template::Segment]| -> Vec<u32> {
            let parts: Vec<PromptPart<'_>> = segments
                .iter()
                .map(|s| match s.origin {
                    Origin::Template => PromptPart::Template(&s.text),
                    Origin::Content => PromptPart::Content(&s.text),
                })
                .collect();
            tok.encode_prompt(&parts)
        };
        let control = |ids: &[u32]| ids.iter().filter(|id| tok.is_special(**id)).count();
        for (stem, template) in loaded() {
            let Some((case, empty)) = probe(&template, "") else {
                continue;
            };
            let template_text = joined(&empty, Origin::Template);
            // the special token texts of this vocabulary that the template writes
            let found: Vec<&str> = specials
                .iter()
                .filter(|(_, t)| template_text.contains(t.as_str()))
                .map(|(_, t)| t.as_str())
                .collect();
            if found.is_empty() {
                continue;
            }
            let message = found.join(" ");
            let with = template
                .render(&case_with_content(case, &message), Some(FIXED_DATE))
                .unwrap_or_else(|e| panic!("{stem}: {e}"));
            assert_eq!(
                control(&tokens(&with)),
                control(&tokens(&empty)),
                "{stem} with {path}: the message {message:?} gives a control token"
            );
            // the check can fail: the same text as template text gives them
            assert!(
                control(&tok.encode_with_special(&message)) >= found.len(),
                "{stem}"
            );
            checked += 1;
        }
    }
    println!("special_token_checks={checked}");
    assert!(
        checked > 0,
        "no template wrote a special token of these vocabularies"
    );
}

/// The variable that names the GGUF files of the probe test, and the
/// corpus template of each, as <a.gguf>=<template>:<b.gguf>=<template>
const PROBE_PAIRS_ENV: &str = "BIV_TEST_PROBE_PAIRS";

#[test]
#[ignore = "needs GGUF files: BIV_TEST_PROBE_PAIRS=<a.gguf>=<template stem>:..."]
fn the_probes_give_the_texts_of_each_local_template() {
    use biv::template::ChatFormat;
    use biv::tokenizer::Tokenizer;
    let pairs = std::env::var(PROBE_PAIRS_ENV)
        .unwrap_or_else(|_| panic!("{PROBE_PAIRS_ENV} must name pairs"));
    for pair in pairs.split(':') {
        let (path, stem) = pair.split_once('=').expect("<gguf>=<template stem>");
        let gguf = biv_cli::open_gguf(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let tok = Tokenizer::from_gguf(&gguf).unwrap_or_else(|e| panic!("{path}: {e}"));
        let is_special = |t: &str| tok.token_id(t).is_some_and(|id| tok.is_special(id));
        let source = std::fs::read_to_string(Path::new(CORPUS).join(format!("{stem}.jinja")))
            .expect("the template");
        let template = Template::load(&source, &VARIABLES).expect("it loads");
        let format =
            ChatFormat::probe(&template, &is_special).unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        println!("probe {stem}: {format:?}");
        // the texts of the two local templates, which chat.rs fixes in code now
        use biv::template::{Reasoning, ToolCall, ToolCallForm};
        assert_eq!(format.turn_end, "<|im_end|>", "{stem}");
        assert_eq!(
            format.generation_prompt_thinking, "<|im_start|>assistant\n<think>\n",
            "{stem}"
        );
        assert_eq!(
            format.generation_prompt_plain, "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            "{stem}"
        );
        assert_eq!(
            format.tool_call,
            Some(ToolCall {
                open: "<tool_call>".into(),
                close: "</tool_call>".into(),
                form: ToolCallForm::XmlParameters,
            }),
            "{stem}"
        );
        // the larger template keeps the reasoning of an earlier answer and
        // takes the three efforts; the smaller drops that reasoning and
        // writes no effort
        let (reasoning, efforts) = if stem.contains("27B") {
            (
                Some(Reasoning {
                    open: "<think>\n".into(),
                    close: "\n</think>\n\n".into(),
                }),
                Some(biv::template::EFFORT_WORDS.to_vec()),
            )
        } else {
            (None, None)
        };
        assert_eq!(format.reasoning, reasoning, "{stem}");
        assert_eq!(format.efforts, efforts, "{stem}");
        let mut outcomes: std::collections::BTreeMap<String, usize> = Default::default();
        for (_, t) in loaded() {
            let key = match ChatFormat::probe(&t, &is_special) {
                Ok(_) => "ok".to_string(),
                Err(e) => format!("{e:?}")
                    .split('(')
                    .next()
                    .unwrap_or_default()
                    .to_string(),
            };
            *outcomes.entry(key).or_default() += 1;
        }
        println!("probe outcomes with the vocabulary of {stem}: {outcomes:?}");
    }
}
