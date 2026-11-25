//! The engine's compiled templates against the generator and templates/.

extern crate alloc;

#[allow(dead_code)]
#[path = "../../../engine/src/template/mod.rs"]
mod template;

use std::path::{Path, PathBuf};

use template::Template;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn source(t: Template) -> String {
    std::fs::read_to_string(root().join(format!("templates/{}.jinja", t.name()))).unwrap()
}

/// The engine's template module is what the generator makes of templates/
#[test]
fn generated_code_is_current() {
    let texts: Vec<(&str, String)> = Template::ALL
        .iter()
        .map(|&t| (t.name(), source(t)))
        .collect();
    let sources: Vec<jinja2rust::Source<'_>> = texts
        .iter()
        .map(|(name, text)| jinja2rust::Source { name, text })
        .collect();
    let dir = root().join("../../engine/src/template");
    for (file, code) in jinja2rust::registry(&sources).unwrap() {
        let code = jinja2rust::rustfmt(&code).unwrap();
        let committed = std::fs::read_to_string(dir.join(&file)).unwrap();
        assert!(
            code == committed,
            "engine/src/template/{file} is stale: in tools/jinja2rust, \
             cargo run -- registry ../../engine/src/template templates/*.jinja"
        );
    }
}

/// Every template is recognized by its text
#[test]
fn templates_are_recognized() {
    for t in Template::ALL {
        assert_eq!(Template::recognize(&source(t)), Some(t));
    }
    assert_eq!(Template::recognize("{{ messages }}"), None);
}
