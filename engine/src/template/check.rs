//! The checks of a template at load: the constructs that the subset leaves
//! out, the filters and the tests that the reference does not have, and
//! the reads of the top-level names that the caller does not set (P6).
//!
//! A name that the template never assigns and the caller does not set is
//! an option of the template when each read of it is a guard (is defined,
//! is undefined, the default filter) or a test use (a truth value, a test,
//! a loop over it, a membership in it), or sits in a branch that a guard
//! on the same name protects. Any other read fails the load by name.

use super::filters::{FILTERS, METHODS, REFERENCE_FILTERS, REFERENCE_TESTS, TESTS};
use super::parser::{CmpOp, Expr, ExprKind, Node, Target};
use super::{ErrorKind, TemplateError};
use crate::prelude::*;

/// The global names of the renderer
const GLOBALS: [&str; 5] = [
    "range",
    "namespace",
    "raise_exception",
    "strftime_now",
    "loop",
];

/// What a template reads and names, as the load finds it
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LoadReport {
    /// the top-level names that the caller does not set and that the
    /// template reads only under a guard or as a test
    pub options: Vec<String>,
    /// the filters that the reference does not have, with their lines: a
    /// render that reaches one fails by name
    pub unknown_filters: Vec<(String, u32)>,
    /// the tests that the reference does not have, with their lines
    pub unknown_tests: Vec<(String, u32)>,
}

/// How the value of an expression is used
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Use {
    /// a truth value, a test, a loop or a membership: Undefined is valid
    Test,
    /// a guard: is defined or the default filter
    Guard,
    /// any other use: the value itself
    Value,
}

struct Checker<'a> {
    /// the names that the check classifies
    names: Vec<String>,
    report: &'a mut LoadReport,
    /// the names whose reads are all guards or tests so far
    seen: Vec<String>,
}

fn unsupported_err(kind: ErrorKind, line: u32) -> TemplateError {
    TemplateError {
        kind,
        line,
        column: 0,
    }
}

/// The names that the template assigns anywhere: sets, loop targets,
/// macros and their parameters
fn assigned(nodes: &[Node], out: &mut Vec<String>) {
    let add =
        |t: &Target, out: &mut Vec<String>| out.extend(t.names().into_iter().map(String::from));
    for n in nodes {
        match n {
            Node::Set { target, .. } => add(target, out),
            Node::SetBlock { name, body } => {
                out.push(name.clone());
                assigned(body, out);
            }
            Node::For { target, body, .. } => {
                add(target, out);
                assigned(body, out);
            }
            Node::If {
                branches,
                otherwise,
            } => {
                for (_, b) in branches {
                    assigned(b, out);
                }
                assigned(otherwise, out);
            }
            Node::Macro(def) => {
                out.push(def.name.clone());
                out.extend(def.params.iter().map(|(p, _)| p.clone()));
                assigned(&def.body, out);
            }
            Node::Text(_)
            | Node::Output(_)
            | Node::SetAttr { .. }
            | Node::Break
            | Node::Continue => {}
        }
    }
}

/// The names that `test` proves set when it is true, and when it is false
fn guards(test: &Expr, names: &[String]) -> (Vec<String>, Vec<String>) {
    let ours = |n: &str| names.iter().any(|m| m == n);
    match &test.kind {
        ExprKind::Test {
            value,
            name,
            negated,
            ..
        } => {
            let ExprKind::Name(n) = &value.kind else {
                return (Vec::new(), Vec::new());
            };
            if !ours(n) {
                return (Vec::new(), Vec::new());
            }
            let set_when_true = match name.as_str() {
                "defined" => !negated,
                "undefined" => *negated,
                _ => return (Vec::new(), Vec::new()),
            };
            if set_when_true {
                (vec![n.clone()], Vec::new())
            } else {
                (Vec::new(), vec![n.clone()])
            }
        }
        ExprKind::Name(n) if ours(n) => (vec![n.clone()], Vec::new()),
        ExprKind::Not(x) => {
            let (t, f) = guards(x, names);
            (f, t)
        }
        ExprKind::And(a, b) => {
            let (mut t, _) = guards(a, names);
            t.extend(guards(b, names).0);
            (t, Vec::new())
        }
        ExprKind::Or(a, b) => {
            let (_, mut f) = guards(a, names);
            f.extend(guards(b, names).1);
            (Vec::new(), f)
        }
        _ => (Vec::new(), Vec::new()),
    }
}

impl Checker<'_> {
    fn nodes(&mut self, nodes: &[Node], safe: &[String]) -> Result<(), TemplateError> {
        for n in nodes {
            self.node(n, safe)?;
        }
        Ok(())
    }

    fn node(&mut self, n: &Node, safe: &[String]) -> Result<(), TemplateError> {
        match n {
            Node::Text(_) | Node::Break | Node::Continue => Ok(()),
            Node::Output(e) => self.expr(e, Use::Value, safe),
            Node::If {
                branches,
                otherwise,
            } => {
                let mut false_so_far: Vec<String> = Vec::new();
                for (test, body) in branches {
                    let mut s = safe.to_vec();
                    s.extend(false_so_far.iter().cloned());
                    self.expr(test, Use::Test, &s)?;
                    let (t, f) = guards(test, &self.names);
                    let mut body_safe = s.clone();
                    body_safe.extend(t);
                    self.nodes(body, &body_safe)?;
                    false_so_far.extend(f);
                }
                let mut s = safe.to_vec();
                s.extend(false_so_far);
                self.nodes(otherwise, &s)
            }
            Node::For {
                iter, filter, body, ..
            } => {
                self.expr(iter, Use::Test, safe)?;
                if let Some(f) = filter {
                    self.expr(f, Use::Test, safe)?;
                }
                self.nodes(body, safe)
            }
            Node::Set { value, .. } | Node::SetAttr { value, .. } => {
                self.expr(value, Use::Value, safe)
            }
            Node::SetBlock { body, .. } => self.nodes(body, safe),
            Node::Macro(def) => {
                for (_, d) in &def.params {
                    if let Some(d) = d {
                        self.expr(d, Use::Value, safe)?;
                    }
                }
                self.nodes(&def.body, safe)
            }
        }
    }

    fn expr(&mut self, e: &Expr, using: Use, safe: &[String]) -> Result<(), TemplateError> {
        let line = e.line;
        match &e.kind {
            ExprKind::Name(n) => {
                if self.names.iter().any(|m| m == n) {
                    if using == Use::Value && !safe.iter().any(|s| s == n) {
                        return Err(unsupported_err(ErrorKind::VariableUnknown(n.clone()), line));
                    }
                    if !self.seen.iter().any(|s| s == n) {
                        self.seen.push(n.clone());
                    }
                }
                Ok(())
            }
            ExprKind::Str(_)
            | ExprKind::Int(_)
            | ExprKind::Float(_)
            | ExprKind::Bool(_)
            | ExprKind::None => Ok(()),
            ExprKind::List(items) | ExprKind::Tuple(items) => {
                for i in items {
                    self.expr(i, Use::Value, safe)?;
                }
                Ok(())
            }
            ExprKind::Dict(items) => {
                for (k, v) in items {
                    self.expr(k, Use::Value, safe)?;
                    self.expr(v, Use::Value, safe)?;
                }
                Ok(())
            }
            ExprKind::Attr(o, _) => self.expr(o, Use::Value, safe),
            ExprKind::Item(o, i) => {
                self.expr(o, Use::Value, safe)?;
                self.expr(i, Use::Value, safe)
            }
            ExprKind::Slice {
                value,
                start,
                stop,
                step,
            } => {
                self.expr(value, Use::Value, safe)?;
                for p in [start, stop, step].into_iter().flatten() {
                    self.expr(p, Use::Value, safe)?;
                }
                Ok(())
            }
            ExprKind::Call { func, args, kwargs } => {
                if let ExprKind::Attr(_, m) = &func.kind
                    && !METHODS.contains(&m.as_str())
                {
                    return Err(unsupported_err(
                        ErrorKind::Unsupported("a method outside the subset"),
                        line,
                    ));
                }
                self.expr(func, Use::Value, safe)?;
                for a in args {
                    self.expr(a, Use::Value, safe)?;
                }
                for (_, a) in kwargs {
                    self.expr(a, Use::Value, safe)?;
                }
                Ok(())
            }
            ExprKind::Filter {
                value,
                name,
                args,
                kwargs,
            } => {
                if !REFERENCE_FILTERS.contains(&name.as_str()) {
                    self.report.unknown_filters.push((name.clone(), line));
                } else if !FILTERS.contains(&name.as_str()) {
                    return Err(unsupported_err(
                        ErrorKind::Unsupported(filter_name(name)),
                        line,
                    ));
                }
                let inner = if name == "default" || name == "d" {
                    Use::Guard
                } else {
                    Use::Value
                };
                self.expr(value, inner, safe)?;
                for a in args {
                    self.expr(a, Use::Value, safe)?;
                }
                for (_, a) in kwargs {
                    self.expr(a, Use::Value, safe)?;
                }
                Ok(())
            }
            ExprKind::Test {
                value, name, args, ..
            } => {
                if !REFERENCE_TESTS.contains(&name.as_str()) {
                    self.report.unknown_tests.push((name.clone(), line));
                } else if !TESTS.contains(&name.as_str()) {
                    return Err(unsupported_err(
                        ErrorKind::Unsupported("a test outside the subset"),
                        line,
                    ));
                }
                self.expr(value, Use::Test, safe)?;
                for a in args {
                    self.expr(a, Use::Value, safe)?;
                }
                Ok(())
            }
            ExprKind::Neg(x) | ExprKind::Pos(x) => self.expr(x, Use::Value, safe),
            ExprKind::Not(x) => self.expr(x, Use::Test, safe),
            ExprKind::And(a, b) => {
                // the left side is a truth value; the right side is the value
                // of the whole, in the same use, under the guards of the left
                self.expr(a, Use::Test, safe)?;
                let mut s = safe.to_vec();
                s.extend(guards(a, &self.names).0);
                self.expr(b, using, &s)
            }
            ExprKind::Or(a, b) => {
                self.expr(a, Use::Test, safe)?;
                let mut s = safe.to_vec();
                s.extend(guards(a, &self.names).1);
                self.expr(b, using, &s)
            }
            ExprKind::Binary(_, a, b) => {
                self.expr(a, Use::Value, safe)?;
                self.expr(b, Use::Value, safe)
            }
            ExprKind::Compare(first, ops) => {
                let mut left = first.as_ref();
                for (op, right) in ops {
                    let membership = matches!(op, CmpOp::In | CmpOp::NotIn);
                    self.expr(left, Use::Value, safe)?;
                    // a membership in an unset value is a test of it
                    self.expr(right, if membership { Use::Test } else { Use::Value }, safe)?;
                    left = right;
                }
                Ok(())
            }
            ExprKind::Cond {
                test,
                then,
                otherwise,
            } => {
                self.expr(test, Use::Test, safe)?;
                let (t, f) = guards(test, &self.names);
                let mut st = safe.to_vec();
                st.extend(t);
                self.expr(then, using, &st)?;
                if let Some(o) = otherwise {
                    let mut sf = safe.to_vec();
                    sf.extend(f);
                    self.expr(o, using, &sf)?;
                }
                Ok(())
            }
        }
    }
}

fn filter_name(name: &str) -> &'static str {
    REFERENCE_FILTERS
        .iter()
        .find(|f| **f == name)
        .copied()
        .unwrap_or("a filter")
}

/// The checks of the statements of a template, with the names that the
/// caller sets in each render
pub fn check(nodes: &[Node], set_by_caller: &[&str]) -> Result<LoadReport, TemplateError> {
    let mut assigned_names = Vec::new();
    assigned(nodes, &mut assigned_names);
    let mut read = Vec::new();
    names_read(nodes, &mut read);
    let names: Vec<String> = read
        .into_iter()
        .filter(|n| {
            !assigned_names.contains(n)
                && !set_by_caller.contains(&n.as_str())
                && !GLOBALS.contains(&n.as_str())
        })
        .collect();
    let mut report = LoadReport::default();
    let mut c = Checker {
        names,
        report: &mut report,
        seen: Vec::new(),
    };
    c.nodes(nodes, &[])?;
    let mut options = c.seen;
    options.sort();
    report.options = options;
    Ok(report)
}

/// The names that the template reads
fn names_read(nodes: &[Node], out: &mut Vec<String>) {
    fn expr(e: &Expr, out: &mut Vec<String>) {
        let mut sub = |x: &Expr| expr(x, out);
        match &e.kind {
            ExprKind::Name(n) => {
                if !out.contains(n) {
                    out.push(n.clone());
                }
            }
            ExprKind::Str(_)
            | ExprKind::Int(_)
            | ExprKind::Float(_)
            | ExprKind::Bool(_)
            | ExprKind::None => {}
            ExprKind::List(items) | ExprKind::Tuple(items) => items.iter().for_each(sub),
            ExprKind::Dict(items) => {
                for (k, v) in items {
                    sub(k);
                    sub(v);
                }
            }
            ExprKind::Attr(o, _) | ExprKind::Neg(o) | ExprKind::Pos(o) | ExprKind::Not(o) => sub(o),
            ExprKind::Item(a, b)
            | ExprKind::And(a, b)
            | ExprKind::Or(a, b)
            | ExprKind::Binary(_, a, b) => {
                sub(a);
                sub(b);
            }
            ExprKind::Slice {
                value,
                start,
                stop,
                step,
            } => {
                sub(value);
                for p in [start, stop, step].into_iter().flatten() {
                    sub(p);
                }
            }
            ExprKind::Call { func, args, kwargs } => {
                sub(func);
                args.iter().for_each(&mut sub);
                kwargs.iter().for_each(|(_, a)| sub(a));
            }
            ExprKind::Filter {
                value,
                args,
                kwargs,
                ..
            } => {
                sub(value);
                args.iter().for_each(&mut sub);
                kwargs.iter().for_each(|(_, a)| sub(a));
            }
            ExprKind::Test { value, args, .. } => {
                sub(value);
                args.iter().for_each(&mut sub);
            }
            ExprKind::Compare(first, ops) => {
                sub(first);
                ops.iter().for_each(|(_, x)| sub(x));
            }
            ExprKind::Cond {
                test,
                then,
                otherwise,
            } => {
                sub(test);
                sub(then);
                if let Some(o) = otherwise {
                    sub(o);
                }
            }
        }
    }
    for n in nodes {
        match n {
            Node::Text(_) | Node::Break | Node::Continue => {}
            Node::Output(e) | Node::Set { value: e, .. } => expr(e, out),
            Node::SetAttr { name, value, .. } => {
                if !out.contains(name) {
                    out.push(name.clone());
                }
                expr(value, out);
            }
            Node::If {
                branches,
                otherwise,
            } => {
                for (t, b) in branches {
                    expr(t, out);
                    names_read(b, out);
                }
                names_read(otherwise, out);
            }
            Node::For {
                iter, filter, body, ..
            } => {
                expr(iter, out);
                if let Some(f) = filter {
                    expr(f, out);
                }
                names_read(body, out);
            }
            Node::SetBlock { body, .. } => names_read(body, out),
            Node::Macro(def) => {
                for (_, d) in &def.params {
                    if let Some(d) = d {
                        expr(d, out);
                    }
                }
                names_read(&def.body, out);
            }
        }
    }
}
