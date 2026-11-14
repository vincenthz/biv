//! The chat template of a model file (GGUF key tokenizer.chat_template): a
//! subset of Jinja, rendered as the reference renderer of transformers
//! renders it (jinja2, trim_blocks, lstrip_blocks, its tojson and
//! raise_exception), for every architecture that biv supports.
//!
//! The subset holds each construct that the corpus of tests/templates
//! uses, except a few that fail at load by name (TemplateUnsupported). The
//! rendered text keeps the origin of each byte: the text of the template,
//! or the text of a value of the caller, so that a special token text in a
//! message stays plain text.

mod check;
mod eval;
mod filters;
pub mod json;
mod lexer;
mod parser;
mod probe;
#[cfg(test)]
mod tests;
mod value;

use core::fmt;

pub use check::LoadReport;
pub use probe::{ChatFormat, EFFORT_WORDS, ProbeError, Reasoning, ToolCall, ToolCallForm};
pub use value::{OStr, Origin, Segment, Value};

use crate::prelude::*;

/// What went wrong in a template, with its line and column (0 when the
/// renderer gives no column)
#[derive(Debug, Clone, PartialEq)]
pub struct TemplateError {
    pub kind: ErrorKind,
    pub line: u32,
    pub column: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ErrorKind {
    /// text that is not a template
    Syntax(&'static str),
    /// a construct of the reference that the subset leaves out
    Unsupported(&'static str),
    /// a top-level name that the caller does not set, used other than
    /// under a guard or as a test
    VariableUnknown(String),
    /// a filter that the reference does not have, reached by a render
    FilterUnknown(String),
    /// a test that the reference does not have, reached by a render
    TestUnknown(String),
    /// raise_exception of the template, with its message
    Raised(String),
    /// a TypeError of the reference
    Type(String),
    /// an UndefinedError of the reference: an unset value used as a value
    Undefined(String),
    /// a ValueError of the reference
    Value(String),
    /// another runtime error of the reference
    Runtime(String),
    /// macro calls nested past the bound
    Recursion(String),
    /// a generator iterated a second time: Python gives no items there
    GeneratorReused,
    /// an operation on a Markup string (from the safe filter) other than
    /// output, +, and the safe filter
    MarkupUnsupported(&'static str),
    /// strftime_now with no date given
    DateMissing,
    /// strftime_now with a format code outside %Y %m %d %b %B
    DateFormat(char),
}

impl fmt::Display for TemplateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (name, detail): (&str, String) = match &self.kind {
            ErrorKind::Syntax(s) => ("TemplateSyntax", (*s).into()),
            ErrorKind::Unsupported(s) => ("TemplateUnsupported", (*s).into()),
            ErrorKind::VariableUnknown(n) => ("TemplateVariableUnknown", n.clone()),
            ErrorKind::FilterUnknown(n) => ("TemplateFilterUnknown", n.clone()),
            ErrorKind::TestUnknown(n) => ("TemplateTestUnknown", n.clone()),
            ErrorKind::Raised(m) => ("TemplateRaised", m.clone()),
            ErrorKind::Type(m) => ("TemplateTypeError", m.clone()),
            ErrorKind::Undefined(m) => ("TemplateUndefined", m.clone()),
            ErrorKind::Value(m) => ("TemplateValueError", m.clone()),
            ErrorKind::Runtime(m) => ("TemplateRuntimeError", m.clone()),
            ErrorKind::Recursion(m) => ("TemplateRecursion", m.clone()),
            ErrorKind::GeneratorReused => ("TemplateGeneratorReused", String::new()),
            ErrorKind::MarkupUnsupported(what) => ("TemplateMarkupUnsupported", (*what).into()),
            ErrorKind::DateMissing => ("TemplateDateMissing", String::new()),
            ErrorKind::DateFormat(c) => ("TemplateDateFormat", format!("%{c}")),
        };
        write!(
            f,
            "{name} at line {} column {}: {detail}",
            self.line, self.column
        )
    }
}

/// The date that strftime_now formats: biv reads no clock, so the caller
/// gives it
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Date {
    pub year: i32,
    pub month: u8,
    pub day: u8,
}

/// A loaded template
#[derive(Debug)]
pub struct Template {
    nodes: Vec<parser::Node>,
    report: LoadReport,
}

impl Template {
    /// Load `source`. `variables` are the top-level names that the caller
    /// sets in each render; the load checks the other names that the
    /// template reads (see [`LoadReport::options`]).
    pub fn load(source: &str, variables: &[&str]) -> Result<Template, TemplateError> {
        let src = lexer::normalize(source);
        let toks = lexer::lex(&src)?;
        let nodes = parser::parse(toks)?;
        let report = check::check(&nodes, variables)?;
        Ok(Template { nodes, report })
    }

    /// What the load found
    pub fn report(&self) -> &LoadReport {
        &self.report
    }

    /// The text of the template with `variables`, as runs of one origin
    pub fn render(
        &self,
        variables: &[(String, Value)],
        date: Option<Date>,
    ) -> Result<Vec<Segment>, TemplateError> {
        let mut r = eval::Renderer::new(variables, date);
        let mut out = OStr::default();
        r.render(&self.nodes, &mut out)?;
        Ok(out.segments().collect())
    }
}
