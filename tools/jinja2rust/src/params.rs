//! The template's parameters as native Rust: the standard ones of chat
//! templates (`messages`, `tools`, `add_generation_prompt`, the special
//! tokens) have their types, and the template's own flags get one from how
//! the template reads them.

use std::collections::{BTreeMap, BTreeSet};

use minijinja::machinery::ast;
use minijinja::value::ValueKind;

use crate::GenError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// `&'a [rt::Message]`
    Messages,
    /// `&'a [rt::Tool]`
    Tools,
    /// `bool`: read for its truth only, where false and undefined are alike
    Bool,
    /// `Option<bool>`: tested for being defined, or for `true`/`false`
    OptBool,
    /// `Option<&'a str>`
    OptStr,
    /// `&'a str`, the template's own text: a special token
    Token,
}

impl Kind {
    pub fn rust_type(self) -> &'static str {
        match self {
            Kind::Messages => "&'a [rt::Message]",
            Kind::Tools => "&'a [rt::Tool]",
            Kind::Bool => "bool",
            Kind::OptBool => "Option<bool>",
            Kind::OptStr => "Option<&'a str>",
            Kind::Token => "&'a str",
        }
    }

    /// The template value of field `f` of the parameters
    pub fn to_value(self, f: &str) -> String {
        match self {
            Kind::Messages => format!("rt::messages(self.{f})"),
            Kind::Tools => format!("rt::tools(self.{f})"),
            Kind::Bool => format!("rt::Value::Bool(self.{f})"),
            Kind::OptBool => format!("rt::opt_bool(self.{f})"),
            Kind::OptStr => format!("rt::opt_str(self.{f})"),
            Kind::Token => format!("rt::Value::lit(self.{f})"),
        }
    }
}

pub struct Param {
    pub name: String,
    pub kind: Kind,
    /// the lines that read it
    pub lines: BTreeSet<usize>,
}

fn standard(name: &str) -> Option<Kind> {
    Some(match name {
        "messages" => Kind::Messages,
        "tools" => Kind::Tools,
        "add_generation_prompt" => Kind::Bool,
        "bos_token" | "eos_token" => Kind::Token,
        _ => return None,
    })
}

/// How an expression's value is used by what holds it
#[derive(Clone, Copy, PartialEq, Eq)]
enum Use {
    /// for its truth
    Cond,
    /// as a string
    Str,
    /// compared to a bool
    Bool,
    /// tested for being defined or none
    Defined,
    /// otherwise: not a flag or a string
    Other,
}

#[derive(Default)]
struct Usage {
    cond: bool,
    str_: bool,
    bool_: bool,
    defined: bool,
    other: Option<usize>,
    lines: BTreeSet<usize>,
}

struct Walk<'n> {
    names: &'n BTreeSet<String>,
    usage: BTreeMap<String, Usage>,
}

/// The types of the context variables `names`, from how `stmts` reads them
pub fn infer(stmts: &[ast::Stmt<'_>], names: &BTreeSet<String>) -> Result<Vec<Param>, GenError> {
    let mut w = Walk {
        names,
        usage: BTreeMap::new(),
    };
    w.stmts(stmts);
    let mut out = Vec::new();
    for name in names {
        let u = w.usage.remove(name).unwrap_or_default();
        let kind = match standard(name) {
            Some(k) => k,
            None => {
                let fail = |line: usize, msg: &str| GenError {
                    line,
                    msg: format!("`{name}`: {msg}"),
                };
                let first = u.lines.first().copied().unwrap_or(0);
                if let Some(line) = u.other {
                    return Err(fail(line, "read other than as a flag or a string"));
                }
                if u.str_ && u.bool_ {
                    return Err(fail(first, "read both as a flag and as a string"));
                }
                if u.str_ {
                    Kind::OptStr
                } else if u.bool_ || u.defined {
                    Kind::OptBool
                } else {
                    debug_assert!(u.cond || u.lines.is_empty());
                    Kind::Bool
                }
            }
        };
        out.push(Param {
            name: name.clone(),
            kind,
            lines: u.lines,
        });
    }
    // the standard ones first, in their order, then the template's own
    let rank = |n: &str| {
        [
            "messages",
            "tools",
            "add_generation_prompt",
            "bos_token",
            "eos_token",
        ]
        .iter()
        .position(|s| *s == n)
        .unwrap_or(usize::MAX)
    };
    out.sort_by(|a, b| (rank(&a.name), &a.name).cmp(&(rank(&b.name), &b.name)));
    Ok(out)
}

fn const_kind(e: &ast::Expr<'_>) -> Option<ValueKind> {
    match e {
        ast::Expr::Const(c) => Some(c.value.kind()),
        _ => None,
    }
}

/// The use of an operand compared to `other`
fn compared_to(other: &ast::Expr<'_>) -> Use {
    match const_kind(other) {
        Some(ValueKind::String) => Use::Str,
        Some(ValueKind::Bool) => Use::Bool,
        _ => Use::Other,
    }
}

impl Walk<'_> {
    fn stmts(&mut self, stmts: &[ast::Stmt<'_>]) {
        for s in stmts {
            self.stmt(s);
        }
    }

    fn stmt(&mut self, s: &ast::Stmt<'_>) {
        match s {
            ast::Stmt::Template(t) => self.stmts(&t.children),
            ast::Stmt::EmitExpr(e) => self.expr(&e.expr, Use::Str),
            ast::Stmt::EmitRaw(_) | ast::Stmt::Continue(_) | ast::Stmt::Break(_) => {}
            ast::Stmt::ForLoop(f) => {
                self.expr(&f.iter, Use::Other);
                if let Some(c) = &f.filter_expr {
                    self.expr(c, Use::Cond);
                }
                self.stmts(&f.body);
                self.stmts(&f.else_body);
            }
            ast::Stmt::IfCond(c) => {
                self.expr(&c.expr, Use::Cond);
                self.stmts(&c.true_body);
                self.stmts(&c.false_body);
            }
            ast::Stmt::WithBlock(w) => {
                for (_, e) in &w.assignments {
                    self.expr(e, Use::Other);
                }
                self.stmts(&w.body);
            }
            ast::Stmt::Set(s) => self.expr(&s.expr, Use::Other),
            ast::Stmt::SetBlock(s) => self.stmts(&s.body),
            ast::Stmt::AutoEscape(a) => self.stmts(&a.body),
            ast::Stmt::FilterBlock(f) => self.stmts(&f.body),
            ast::Stmt::Macro(m) => {
                m.defaults.iter().for_each(|d| self.expr(d, Use::Other));
                self.stmts(&m.body);
            }
            ast::Stmt::CallBlock(c) => self.stmts(&c.macro_decl.body),
            ast::Stmt::Do(d) => {
                self.expr(&d.call.expr, Use::Other);
                d.call.args.iter().for_each(|a| self.arg(a));
            }
        }
    }

    fn arg(&mut self, a: &ast::CallArg<'_>) {
        match a {
            ast::CallArg::Pos(e)
            | ast::CallArg::Kwarg(_, e)
            | ast::CallArg::PosSplat(e)
            | ast::CallArg::KwargSplat(e) => self.expr(e, Use::Other),
        }
    }

    fn expr(&mut self, e: &ast::Expr<'_>, how: Use) {
        match e {
            ast::Expr::Var(v) => {
                if !self.names.contains(v.id) {
                    return;
                }
                let line = v.span().start_line as usize;
                let u = self.usage.entry(v.id.to_owned()).or_default();
                u.lines.insert(line);
                match how {
                    Use::Cond => u.cond = true,
                    Use::Str => u.str_ = true,
                    Use::Bool => u.bool_ = true,
                    Use::Defined => u.defined = true,
                    Use::Other => {
                        u.other.get_or_insert(line);
                    }
                }
            }
            ast::Expr::Const(_) => {}
            ast::Expr::UnaryOp(u) => match u.op {
                ast::UnaryOpKind::Not => self.expr(&u.expr, Use::Cond),
                ast::UnaryOpKind::Neg => self.expr(&u.expr, Use::Other),
            },
            ast::Expr::BinOp(b) => {
                use ast::BinOpKind as K;
                match b.op {
                    // Python's `and`/`or` give an operand: it is used as the whole
                    K::ScAnd | K::ScOr => {
                        self.expr(&b.left, how);
                        self.expr(&b.right, how);
                    }
                    K::Eq | K::Ne | K::Lt | K::Lte | K::Gt | K::Gte => {
                        self.expr(&b.left, compared_to(&b.right));
                        self.expr(&b.right, compared_to(&b.left));
                    }
                    K::In => {
                        // `x in ('a', 'b')` or `'a' in x`
                        let left = match &b.right {
                            ast::Expr::Const(c)
                                if c.value.kind() == ValueKind::String
                                    || c.value.try_iter().is_ok_and(|mut i| {
                                        i.all(|v| v.kind() == ValueKind::String)
                                    }) =>
                            {
                                Use::Str
                            }
                            _ => Use::Other,
                        };
                        let right = match const_kind(&b.left) {
                            Some(ValueKind::String) => Use::Str,
                            _ => Use::Other,
                        };
                        self.expr(&b.left, left);
                        self.expr(&b.right, right);
                    }
                    K::Concat => {
                        self.expr(&b.left, Use::Str);
                        self.expr(&b.right, Use::Str);
                    }
                    K::Add => {
                        let str_ = |e: &ast::Expr<'_>| const_kind(e) == Some(ValueKind::String);
                        let (l, r) = if str_(&b.left) || str_(&b.right) {
                            (Use::Str, Use::Str)
                        } else {
                            (Use::Other, Use::Other)
                        };
                        self.expr(&b.left, l);
                        self.expr(&b.right, r);
                    }
                    _ => {
                        self.expr(&b.left, Use::Other);
                        self.expr(&b.right, Use::Other);
                    }
                }
            }
            ast::Expr::Compare(c) => {
                self.expr(&c.expr, Use::Other);
                c.ops.iter().for_each(|o| self.expr(&o.expr, Use::Other));
            }
            ast::Expr::IfExpr(i) => {
                self.expr(&i.test_expr, Use::Cond);
                self.expr(&i.true_expr, how);
                if let Some(e) = &i.false_expr {
                    self.expr(e, how);
                }
            }
            ast::Expr::Filter(f) => {
                if let Some(inner) = &f.expr {
                    let child = match f.name {
                        "default" | "d" => {
                            // undefined becomes the default: the parameter is optional,
                            // of the default's type
                            let d = f.args.first().and_then(|a| match a {
                                ast::CallArg::Pos(e) => const_kind(e),
                                _ => None,
                            });
                            match d {
                                Some(ValueKind::String) => Use::Str,
                                Some(ValueKind::Bool) => Use::Bool,
                                _ => Use::Other,
                            }
                        }
                        "string" | "trim" | "upper" | "lower" => Use::Str,
                        "safe" => how,
                        _ => Use::Other,
                    };
                    self.expr(inner, child);
                }
                f.args.iter().for_each(|a| self.arg(a));
            }
            ast::Expr::Test(t) => {
                let child = match t.name {
                    "defined" | "undefined" | "none" => Use::Defined,
                    "true" | "false" | "boolean" => Use::Bool,
                    "string" => Use::Str,
                    _ => Use::Other,
                };
                self.expr(&t.expr, child);
                t.args.iter().for_each(|a| self.arg(a));
            }
            ast::Expr::Slice(s) => {
                self.expr(&s.expr, Use::Other);
                for e in [&s.start, &s.stop, &s.step].into_iter().flatten() {
                    self.expr(e, Use::Other);
                }
            }
            ast::Expr::GetAttr(g) => self.expr(&g.expr, Use::Other),
            ast::Expr::GetItem(g) => {
                self.expr(&g.expr, Use::Other);
                self.expr(&g.subscript_expr, Use::Other);
            }
            ast::Expr::Call(c) => {
                self.expr(&c.expr, Use::Other);
                c.args.iter().for_each(|a| self.arg(a));
            }
            ast::Expr::List(l) => l.items.iter().for_each(|e| self.expr(e, Use::Other)),
            ast::Expr::Tuple(t) => t.items.iter().for_each(|e| self.expr(e, Use::Other)),
            ast::Expr::Map(m) => {
                m.keys.iter().for_each(|e| self.expr(e, Use::Other));
                m.values.iter().for_each(|e| self.expr(e, Use::Other));
            }
        }
    }
}
