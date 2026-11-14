//! The constructs and the named failures of the template subset, one
//! case each. The outputs and the failures of the reference cases came
//! from the renderer of transformers 5.5.2 with jinja2 3.1.6 on
//! 2026-11-14, with the variables of VARIABLES and the date 2026-01-05.

use super::{Date, ErrorKind, Origin, Template, Value, json};
use crate::prelude::*;

/// The cases and their outcomes in the reference, which
/// tools/template_unit_cases.py writes
const UNIT_CASES: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tests/templates/unit_cases.json"
));

const DATE: Date = Date {
    year: 2026,
    month: 1,
    day: 5,
};

fn unit_cases() -> Value {
    json::parse(UNIT_CASES, Origin::Content).expect("the unit cases")
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

fn variables() -> Vec<(String, Value)> {
    let Some(Value::Map(m)) = field(&unit_cases(), "variables") else {
        panic!("the variables are a mapping");
    };
    m.iter().map(|(k, v)| (text(k), v.clone())).collect()
}

/// The names that the cases set
fn names(vars: &[(String, Value)]) -> Vec<&str> {
    vars.iter().map(|(n, _)| n.as_str()).collect()
}

fn render(source: &str, date: Option<Date>) -> Result<String, super::TemplateError> {
    let vars = variables();
    let t = Template::load(source, &names(&vars))?;
    let segments = t.render(&vars, date)?;
    Ok(segments.iter().map(|s| s.text.as_str()).collect())
}

/// The failure type of the reference for a failure of biv
fn reference_failure(e: &ErrorKind) -> &'static str {
    match e {
        ErrorKind::Raised(_) => "TemplateError",
        ErrorKind::Type(_) => "TypeError",
        ErrorKind::Undefined(_) => "UndefinedError",
        _ => "a failure of biv only",
    }
}

#[test]
fn each_construct_renders_as_the_reference() {
    let Some(Value::List(cases)) = field(&unit_cases(), "cases") else {
        panic!("the cases are a list");
    };
    for case in cases.iter() {
        let name = text(&field(case, "name").expect("a name"));
        let source = text(&field(case, "source").expect("a source"));
        match (
            render(&source, Some(DATE)),
            field(case, "output"),
            field(case, "failure"),
        ) {
            (Ok(out), Some(want), None) => assert_eq!(out, text(&want), "{name}"),
            (Err(e), None, Some(failure)) => {
                assert_eq!(reference_failure(&e.kind), text(&failure), "{name}: {e}")
            }
            (Ok(out), None, Some(failure)) => panic!(
                "{name}: renders {out:?}, the reference fails with {}",
                text(&failure)
            ),
            (Err(e), Some(want), None) => {
                panic!("{name}: {e}, the reference renders {:?}", text(&want))
            }
            _ => panic!("{name}: a case with neither an output nor a failure"),
        }
    }
}

/// The failures of biv only: an outcome that the subset or the rulings give
#[test]
fn each_failure_of_the_subset_has_its_name() {
    let fails = |source: &str, date: Option<Date>| render(source, date).expect_err(source).kind;
    // an unset top-level name used as a value fails at load
    assert_eq!(
        fails("{{ unset_name }}", Some(DATE)),
        ErrorKind::VariableUnknown("unset_name".into())
    );
    assert_eq!(
        fails("{{ unset_x.y if false else '' }}", Some(DATE)),
        ErrorKind::VariableUnknown("unset_x".into())
    );
    // a name that the template sets, read before the set, fails at render
    assert_eq!(
        fails(
            "{% if false %}{% set late = 1 %}{% endif %}{{ late }}",
            Some(DATE)
        ),
        ErrorKind::VariableUnknown("late".into())
    );
    // a filter that the reference does not have loads, is listed, and fails
    // when a render reaches it
    let vars = variables();
    let t = Template::load("{% if t %}{{ 1 | from_json }}{% endif %}", &names(&vars))
        .expect("it loads");
    assert_eq!(
        t.report().unknown_filters,
        vec![("from_json".to_string(), 1)]
    );
    assert_eq!(
        t.render(&vars, Some(DATE)).expect_err("it is reached").kind,
        ErrorKind::FilterUnknown("from_json".into())
    );
    let unreached = Template::load("{% if n %}{{ 1 | from_json }}{% endif %}ok", &names(&vars))
        .expect("it loads");
    let out: String = unreached
        .render(&vars, Some(DATE))
        .expect("not reached")
        .iter()
        .map(|s| s.text.as_str())
        .collect();
    assert_eq!(out, "ok");
    // a generator iterated a second time, where Python gives no items
    assert_eq!(
        fails(
            "{% set g = words | map('upper') %}{{ g | list }}{{ g | list }}",
            Some(DATE)
        ),
        ErrorKind::GeneratorReused
    );
    // a Markup string in an operation other than output, + and safe
    for (source, what) in [
        ("{{ 'a'|safe | upper }}", "a filter other than safe"),
        ("{{ ('a'|safe).strip() }}", "a method"),
        ("{{ ('a'|safe) == 'a' }}", "a comparison"),
        ("{{ ('ab'|safe)[0] }}", "an item"),
        ("{% for c in 'ab'|safe %}{% endfor %}", "an iteration"),
    ] {
        assert_eq!(
            fails(source, Some(DATE)),
            ErrorKind::MarkupUnsupported(what),
            "{source}"
        );
    }
    // the constructs that the subset removed because no template of the
    // corpus uses them fail by name
    for (source, what) in [
        (
            "{% for x in words %}{% else %}e{% endfor %}",
            "the else of a for loop",
        ),
        ("{{ 7 // 2 }}", "the operators // and **"),
        ("{{ 2 ** 3 }}", "the operators // and **"),
        ("{{ words | first }}", "first"),
        ("{{ words | last }}", "last"),
        ("{{ 'a'.title() }}", "a method outside the subset"),
        ("{{ 'a'.rsplit() }}", "a method outside the subset"),
        ("{{ 'a'.find('a') }}", "a method outside the subset"),
        ("{{ 'a'.count('a') }}", "a method outside the subset"),
        ("{{ d.keys() }}", "a method outside the subset"),
        ("{{ d.values() }}", "a method outside the subset"),
        ("{{ 1 is number }}", "a test outside the subset"),
        ("{{ 1 is odd }}", "a test outside the subset"),
        ("{{ 1 is sameas 1 }}", "a test outside the subset"),
        (
            "{{ words | select('odd') | list }}",
            "a test outside the subset",
        ),
        (
            "{% for x in words %}{{ loop.revindex }}{% endfor %}",
            "a loop attribute outside the subset",
        ),
    ] {
        assert_eq!(
            fails(source, Some(DATE)),
            ErrorKind::Unsupported(what),
            "{source}"
        );
    }
    // the date
    assert_eq!(
        fails("{{ strftime_now('%Y') }}", None),
        ErrorKind::DateMissing
    );
    assert_eq!(
        fails("{{ strftime_now('%H') }}", Some(DATE)),
        ErrorKind::DateFormat('H')
    );
    // the constructs that the subset leaves out fail at load
    for (source, what) in [
        ("{% filter upper %}a{% endfilter %}", "the filter block"),
        ("{{ [1].append(2) }}", "the list method append"),
        ("{{ 'a' | indent }}", "indent"),
        ("{{ 'a' | unique }}", "unique"),
        ("{{ 'a'.format() }}", "a method outside the subset"),
        ("{% include 'x' %}", "include"),
    ] {
        assert_eq!(
            fails(source, Some(DATE)),
            ErrorKind::Unsupported(what),
            "{source}"
        );
    }
    // text that is not a template
    assert!(matches!(
        fails("{% if %}", Some(DATE)),
        ErrorKind::Syntax(_)
    ));
    assert!(matches!(
        fails("{{ 1 + }}", Some(DATE)),
        ErrorKind::Syntax(_)
    ));
    assert!(matches!(
        fails("{% for x in y %}", Some(DATE)),
        ErrorKind::Syntax(_)
    ));
}

/// The options of a template: the names that it reads only under a guard
/// or as a test
#[test]
fn the_options_of_a_template_are_its_guarded_reads() {
    let vars = variables();
    let t = Template::load(
        "{% if opt_a is defined %}{{ opt_a }}{% endif %}{{ opt_b | default('x') }}{% for i in opt_c %}{% endfor %}{{ 'k' in opt_d }}",
        &names(&vars),
    )
    .expect("it loads");
    assert_eq!(t.report().options, vec!["opt_a", "opt_b", "opt_c", "opt_d"]);
}

/// The origin of each byte: the text of a value of the caller stays
/// Content through slices, trims, splits, case changes and joins
#[test]
fn the_text_of_a_value_keeps_its_origin() {
    let vars = vec![("m".to_string(), Value::str(" <|x|> ab ", Origin::Content))];
    let t = Template::load(
        "<|s|>{{ m | trim }}|{{ m.split() | join(',') }}|{{ m[1:5] | upper }}<|e|>",
        &["m"],
    )
    .expect("it loads");
    let segments = t.render(&vars, None).expect("it renders");
    let of = |o: Origin| -> String {
        segments
            .iter()
            .filter(|s| s.origin == o)
            .map(|s| s.text.as_str())
            .collect()
    };
    assert_eq!(of(Origin::Template), "<|s|>|,|<|e|>");
    assert_eq!(of(Origin::Content), "<|x|> ab<|x|>ab<|X|");
}
