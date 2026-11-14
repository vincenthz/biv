//! The parser of templates: the statements and the expressions of the
//! subset, with the grammar of the reference parser (jinja2).

use alloc::boxed::Box;

use super::lexer::{Tok, Token};
use super::{ErrorKind, TemplateError};
use crate::prelude::*;

/// A statement of a template
#[derive(Debug, Clone)]
pub enum Node {
    /// text of the template
    Text(String),
    /// {{ expr }}
    Output(Expr),
    /// if, the elifs, and else
    If {
        branches: Vec<(Expr, Vec<Node>)>,
        otherwise: Vec<Node>,
    },
    For {
        target: Target,
        iter: Expr,
        /// the condition of `for x in y if cond`
        filter: Option<Expr>,
        body: Vec<Node>,
    },
    /// set name = expr, or set a, b = expr
    Set {
        target: Target,
        value: Expr,
    },
    /// set ns.attr = expr
    SetAttr {
        name: String,
        attr: String,
        value: Expr,
        line: u32,
    },
    /// {% set name %}...{% endset %}
    SetBlock {
        name: String,
        body: Vec<Node>,
    },
    Macro(Rc<MacroDef>),
    Break,
    Continue,
}

use alloc::rc::Rc;

/// A macro of a template
#[derive(Debug)]
pub struct MacroDef {
    pub name: String,
    pub params: Vec<(String, Option<Expr>)>,
    pub body: Vec<Node>,
}

/// The target of a set or of a for
#[derive(Debug, Clone)]
pub enum Target {
    Name(String),
    Tuple(Vec<String>),
}

impl Target {
    pub fn names(&self) -> Vec<&str> {
        match self {
            Target::Name(n) => vec![n.as_str()],
            Target::Tuple(ns) => ns.iter().map(String::as_str).collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub line: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Concat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    In,
    NotIn,
}

#[derive(Debug, Clone)]
pub enum ExprKind {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    None,
    List(Vec<Expr>),
    Tuple(Vec<Expr>),
    Dict(Vec<(Expr, Expr)>),
    Name(String),
    Attr(Box<Expr>, String),
    Item(Box<Expr>, Box<Expr>),
    Slice {
        value: Box<Expr>,
        start: Option<Box<Expr>>,
        stop: Option<Box<Expr>>,
        step: Option<Box<Expr>>,
    },
    Call {
        func: Box<Expr>,
        args: Vec<Expr>,
        kwargs: Vec<(String, Expr)>,
    },
    Filter {
        value: Box<Expr>,
        name: String,
        args: Vec<Expr>,
        kwargs: Vec<(String, Expr)>,
    },
    Test {
        value: Box<Expr>,
        name: String,
        args: Vec<Expr>,
        negated: bool,
    },
    Neg(Box<Expr>),
    Pos(Box<Expr>),
    Not(Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    Compare(Box<Expr>, Vec<(CmpOp, Expr)>),
    Cond {
        test: Box<Expr>,
        then: Box<Expr>,
        otherwise: Option<Box<Expr>>,
    },
}

/// The statements of jinja2 that the subset leaves out: each fails at load
const UNSUPPORTED_TAGS: [&str; 14] = [
    "filter",
    "call",
    "raw",
    "include",
    "import",
    "from",
    "extends",
    "block",
    "with",
    "do",
    "autoescape",
    "generation",
    "trans",
    "print",
];

/// The methods that change a value in place: the sandbox of the reference
/// refuses them on its values, and the subset leaves them out
const MUTATING_METHODS: [&str; 9] = [
    "append",
    "pop",
    "extend",
    "insert",
    "remove",
    "update",
    "clear",
    "setdefault",
    "popitem",
];

struct Parser {
    toks: Vec<Token>,
    at: usize,
}

type Res<T> = Result<T, TemplateError>;

/// The positional and the keyword arguments of a call
type Args = (Vec<Expr>, Vec<(String, Expr)>);

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.at).map(|t| &t.tok)
    }

    fn peek_at(&self, n: usize) -> Option<&Tok> {
        self.toks.get(self.at + n).map(|t| &t.tok)
    }

    fn line(&self) -> u32 {
        self.toks
            .get(self.at)
            .or(self.toks.last())
            .map_or(1, |t| t.line)
    }

    fn err(&self, kind: ErrorKind) -> TemplateError {
        let t = self.toks.get(self.at).or(self.toks.last());
        TemplateError {
            kind,
            line: t.map_or(1, |t| t.line),
            column: t.map_or(1, |t| t.column),
        }
    }

    fn next(&mut self) -> Res<Tok> {
        let t = self
            .toks
            .get(self.at)
            .ok_or_else(|| self.err(ErrorKind::Syntax("the template ends inside a tag")))?;
        self.at += 1;
        Ok(t.tok.clone())
    }

    fn is_op(&self, op: &str) -> bool {
        matches!(self.peek(), Some(Tok::Op(o)) if *o == op)
    }

    fn is_name(&self, name: &str) -> bool {
        matches!(self.peek(), Some(Tok::Name(n)) if n == name)
    }

    fn eat_op(&mut self, op: &str) -> bool {
        let yes = self.is_op(op);
        if yes {
            self.at += 1;
        }
        yes
    }

    fn eat_name(&mut self, name: &str) -> bool {
        let yes = self.is_name(name);
        if yes {
            self.at += 1;
        }
        yes
    }

    fn expect_op(&mut self, op: &'static str) -> Res<()> {
        if self.eat_op(op) {
            Ok(())
        } else {
            Err(self.err(ErrorKind::Syntax(op)))
        }
    }

    fn expect_end(&mut self, end: &Tok) -> Res<()> {
        if self.peek() == Some(end) {
            self.at += 1;
            Ok(())
        } else {
            Err(self.err(ErrorKind::Syntax("the end of the tag")))
        }
    }

    fn ident(&mut self) -> Res<String> {
        match self.next()? {
            Tok::Name(n) => Ok(n),
            _ => {
                self.at -= 1;
                Err(self.err(ErrorKind::Syntax("a name")))
            }
        }
    }

    /// The statements up to a block whose tag is one of `ends`; the tag
    /// stays in the stream, after its BlockStart
    fn body(&mut self, ends: &[&str]) -> Res<(Vec<Node>, Option<String>)> {
        let mut nodes = Vec::new();
        loop {
            match self.peek() {
                None => return Ok((nodes, None)),
                Some(Tok::Data(d)) => {
                    nodes.push(Node::Text(d.clone()));
                    self.at += 1;
                }
                Some(Tok::VarStart) => {
                    self.at += 1;
                    let e = self.tuple(true)?;
                    self.expect_end(&Tok::VarEnd)?;
                    nodes.push(Node::Output(e));
                }
                Some(Tok::BlockStart) => {
                    if let Some(Tok::Name(tag)) = self.peek_at(1)
                        && ends.contains(&tag.as_str())
                    {
                        let tag = tag.clone();
                        self.at += 2;
                        return Ok((nodes, Some(tag)));
                    }
                    self.at += 1;
                    nodes.push(self.statement()?);
                }
                Some(_) => return Err(self.err(ErrorKind::Syntax("a token outside a tag"))),
            }
        }
    }

    /// A body that must end with one of `ends`
    fn body_to(&mut self, ends: &[&str]) -> Res<(Vec<Node>, String)> {
        match self.body(ends)? {
            (nodes, Some(tag)) => Ok((nodes, tag)),
            (_, None) => Err(self.err(ErrorKind::Syntax("a block with no end"))),
        }
    }

    fn statement(&mut self) -> Res<Node> {
        let tag = self.ident()?;
        match tag.as_str() {
            "if" => {
                let mut branches = Vec::new();
                let mut test = self.tuple(false)?;
                let otherwise;
                loop {
                    self.expect_end(&Tok::BlockEnd)?;
                    let (body, end) = self.body_to(&["elif", "else", "endif"])?;
                    branches.push((test, body));
                    match end.as_str() {
                        "elif" => test = self.tuple(false)?,
                        "else" => {
                            self.expect_end(&Tok::BlockEnd)?;
                            let (body, _) = self.body_to(&["endif"])?;
                            otherwise = body;
                            break;
                        }
                        _ => {
                            otherwise = Vec::new();
                            break;
                        }
                    }
                }
                self.expect_end(&Tok::BlockEnd)?;
                Ok(Node::If {
                    branches,
                    otherwise,
                })
            }
            "for" => {
                let target = self.target()?;
                if !self.eat_name("in") {
                    return Err(self.err(ErrorKind::Syntax("in")));
                }
                let iter = self.tuple_no_cond()?;
                let filter = if self.eat_name("if") {
                    Some(self.expression(true)?)
                } else {
                    None
                };
                if self.is_name("recursive") {
                    return Err(self.err(ErrorKind::Unsupported("a recursive loop")));
                }
                self.expect_end(&Tok::BlockEnd)?;
                let (body, end) = self.body_to(&["else", "endfor"])?;
                if end == "else" {
                    return Err(self.err(ErrorKind::Unsupported("the else of a for loop")));
                }
                self.expect_end(&Tok::BlockEnd)?;
                Ok(Node::For {
                    target,
                    iter,
                    filter,
                    body,
                })
            }
            "set" => {
                let line = self.line();
                let first = self.ident()?;
                if self.eat_op(".") {
                    let attr = self.ident()?;
                    self.expect_op("=")?;
                    let value = self.tuple(true)?;
                    self.expect_end(&Tok::BlockEnd)?;
                    return Ok(Node::SetAttr {
                        name: first,
                        attr,
                        value,
                        line,
                    });
                }
                if self.peek() == Some(&Tok::BlockEnd) {
                    self.at += 1;
                    let (body, _) = self.body_to(&["endset"])?;
                    self.expect_end(&Tok::BlockEnd)?;
                    return Ok(Node::SetBlock { name: first, body });
                }
                if self.is_op("|") {
                    return Err(self.err(ErrorKind::Unsupported("a filter on a set block")));
                }
                let mut names = vec![first];
                while self.eat_op(",") {
                    names.push(self.ident()?);
                }
                self.expect_op("=")?;
                let value = self.tuple(true)?;
                self.expect_end(&Tok::BlockEnd)?;
                let target = if names.len() == 1 {
                    Target::Name(names.pop().expect("one name"))
                } else {
                    Target::Tuple(names)
                };
                Ok(Node::Set { target, value })
            }
            "macro" => {
                let name = self.ident()?;
                self.expect_op("(")?;
                let mut params = Vec::new();
                while !self.eat_op(")") {
                    if !params.is_empty() {
                        self.expect_op(",")?;
                        if self.eat_op(")") {
                            break;
                        }
                    }
                    let p = self.ident()?;
                    let default = if self.eat_op("=") {
                        Some(self.expression(true)?)
                    } else {
                        None
                    };
                    params.push((p, default));
                }
                self.expect_end(&Tok::BlockEnd)?;
                let (body, _) = self.body_to(&["endmacro"])?;
                self.expect_end(&Tok::BlockEnd)?;
                Ok(Node::Macro(Rc::new(MacroDef { name, params, body })))
            }
            "break" => {
                self.expect_end(&Tok::BlockEnd)?;
                Ok(Node::Break)
            }
            "continue" => {
                self.expect_end(&Tok::BlockEnd)?;
                Ok(Node::Continue)
            }
            t if UNSUPPORTED_TAGS.contains(&t) => {
                self.at -= 1;
                Err(self.err(ErrorKind::Unsupported(unsupported_tag(t))))
            }
            _ => {
                self.at -= 1;
                Err(self.err(ErrorKind::Syntax("an unknown tag")))
            }
        }
    }

    fn target(&mut self) -> Res<Target> {
        let paren = self.eat_op("(");
        let mut names = vec![self.ident()?];
        while self.eat_op(",") {
            if paren && self.is_op(")") {
                break;
            }
            names.push(self.ident()?);
        }
        if paren {
            self.expect_op(")")?;
        }
        Ok(if names.len() == 1 && !paren {
            Target::Name(names.pop().expect("one name"))
        } else {
            Target::Tuple(names)
        })
    }

    /// An expression, or a tuple of them without parentheses
    fn tuple(&mut self, with_cond: bool) -> Res<Expr> {
        let line = self.line();
        let first = self.expression(with_cond)?;
        if !self.is_op(",") {
            return Ok(first);
        }
        let mut items = vec![first];
        while self.eat_op(",") {
            if self.at_tuple_end() {
                break;
            }
            items.push(self.expression(with_cond)?);
        }
        Ok(Expr {
            kind: ExprKind::Tuple(items),
            line,
        })
    }

    fn tuple_no_cond(&mut self) -> Res<Expr> {
        self.tuple(false)
    }

    fn at_tuple_end(&self) -> bool {
        matches!(
            self.peek(),
            Some(Tok::VarEnd | Tok::BlockEnd) | Some(Tok::Op(")" | "]" | "}"))
        ) || self.is_name("in")
            || self.is_name("if")
    }

    fn expression(&mut self, with_cond: bool) -> Res<Expr> {
        if with_cond { self.cond() } else { self.or() }
    }

    fn cond(&mut self) -> Res<Expr> {
        let line = self.line();
        let mut e = self.or()?;
        while self.eat_name("if") {
            let test = self.or()?;
            let otherwise = if self.eat_name("else") {
                Some(Box::new(self.cond()?))
            } else {
                None
            };
            e = Expr {
                kind: ExprKind::Cond {
                    test: Box::new(test),
                    then: Box::new(e),
                    otherwise,
                },
                line,
            };
        }
        Ok(e)
    }

    fn or(&mut self) -> Res<Expr> {
        let line = self.line();
        let mut e = self.and()?;
        while self.eat_name("or") {
            let r = self.and()?;
            e = Expr {
                kind: ExprKind::Or(Box::new(e), Box::new(r)),
                line,
            };
        }
        Ok(e)
    }

    fn and(&mut self) -> Res<Expr> {
        let line = self.line();
        let mut e = self.not()?;
        while self.eat_name("and") {
            let r = self.not()?;
            e = Expr {
                kind: ExprKind::And(Box::new(e), Box::new(r)),
                line,
            };
        }
        Ok(e)
    }

    fn not(&mut self) -> Res<Expr> {
        let line = self.line();
        if self.eat_name("not") {
            let e = self.not()?;
            return Ok(Expr {
                kind: ExprKind::Not(Box::new(e)),
                line,
            });
        }
        self.compare()
    }

    fn compare(&mut self) -> Res<Expr> {
        let line = self.line();
        let first = self.math1()?;
        let mut ops = Vec::new();
        loop {
            let op = match self.peek() {
                Some(Tok::Op("==")) => CmpOp::Eq,
                Some(Tok::Op("!=")) => CmpOp::Ne,
                Some(Tok::Op("<")) => CmpOp::Lt,
                Some(Tok::Op("<=")) => CmpOp::Le,
                Some(Tok::Op(">")) => CmpOp::Gt,
                Some(Tok::Op(">=")) => CmpOp::Ge,
                Some(Tok::Name(n)) if n == "in" => CmpOp::In,
                Some(Tok::Name(n))
                    if n == "not" && matches!(self.peek_at(1), Some(Tok::Name(m)) if m == "in") =>
                {
                    self.at += 1;
                    CmpOp::NotIn
                }
                _ => break,
            };
            self.at += 1;
            ops.push((op, self.math1()?));
        }
        Ok(if ops.is_empty() {
            first
        } else {
            Expr {
                kind: ExprKind::Compare(Box::new(first), ops),
                line,
            }
        })
    }

    fn binary(&mut self, ops: &[(&str, BinOp)], next: fn(&mut Self) -> Res<Expr>) -> Res<Expr> {
        let line = self.line();
        let mut e = next(self)?;
        'outer: loop {
            for &(tok, op) in ops {
                if self.eat_op(tok) {
                    let r = next(self)?;
                    e = Expr {
                        kind: ExprKind::Binary(op, Box::new(e), Box::new(r)),
                        line,
                    };
                    continue 'outer;
                }
            }
            return Ok(e);
        }
    }

    fn math1(&mut self) -> Res<Expr> {
        self.binary(&[("+", BinOp::Add), ("-", BinOp::Sub)], Self::concat)
    }

    fn concat(&mut self) -> Res<Expr> {
        self.binary(&[("~", BinOp::Concat)], Self::math2)
    }

    fn math2(&mut self) -> Res<Expr> {
        self.binary(
            &[("*", BinOp::Mul), ("/", BinOp::Div), ("%", BinOp::Mod)],
            Self::pow,
        )
    }

    fn pow(&mut self) -> Res<Expr> {
        let e = self.unary_filtered()?;
        if self.is_op("//") || self.is_op("**") {
            return Err(self.err(ErrorKind::Unsupported("the operators // and **")));
        }
        Ok(e)
    }

    fn unary_filtered(&mut self) -> Res<Expr> {
        self.unary(true)
    }

    fn unary(&mut self, with_filter: bool) -> Res<Expr> {
        let line = self.line();
        let e = if self.eat_op("-") {
            let inner = self.unary(false)?;
            Expr {
                kind: ExprKind::Neg(Box::new(inner)),
                line,
            }
        } else if self.eat_op("+") {
            let inner = self.unary(false)?;
            Expr {
                kind: ExprKind::Pos(Box::new(inner)),
                line,
            }
        } else {
            self.primary()?
        };
        let e = self.postfix(e)?;
        if with_filter {
            self.filter_expr(e)
        } else {
            Ok(e)
        }
    }

    fn primary(&mut self) -> Res<Expr> {
        let line = self.line();
        let kind = match self.next()? {
            Tok::Name(n) => match n.as_str() {
                "true" | "True" => ExprKind::Bool(true),
                "false" | "False" => ExprKind::Bool(false),
                "none" | "None" => ExprKind::None,
                _ => ExprKind::Name(n),
            },
            Tok::Str(s) => {
                let mut s = s;
                while let Some(Tok::Str(more)) = self.peek() {
                    s.push_str(more);
                    self.at += 1;
                }
                ExprKind::Str(s)
            }
            Tok::Int(i) => ExprKind::Int(i),
            Tok::Float(f) => ExprKind::Float(f),
            Tok::Op("(") => {
                if self.eat_op(")") {
                    ExprKind::Tuple(Vec::new())
                } else {
                    let first = self.expression(true)?;
                    if self.eat_op(")") {
                        return Ok(first);
                    }
                    let mut items = vec![first];
                    while self.eat_op(",") {
                        if self.is_op(")") {
                            break;
                        }
                        items.push(self.expression(true)?);
                    }
                    self.expect_op(")")?;
                    ExprKind::Tuple(items)
                }
            }
            Tok::Op("[") => {
                let mut items = Vec::new();
                while !self.eat_op("]") {
                    if !items.is_empty() {
                        self.expect_op(",")?;
                        if self.eat_op("]") {
                            break;
                        }
                    }
                    items.push(self.expression(true)?);
                }
                ExprKind::List(items)
            }
            Tok::Op("{") => {
                let mut items = Vec::new();
                while !self.eat_op("}") {
                    if !items.is_empty() {
                        self.expect_op(",")?;
                        if self.eat_op("}") {
                            break;
                        }
                    }
                    let k = self.expression(true)?;
                    self.expect_op(":")?;
                    let v = self.expression(true)?;
                    items.push((k, v));
                }
                ExprKind::Dict(items)
            }
            _ => {
                self.at -= 1;
                return Err(self.err(ErrorKind::Syntax("an expression")));
            }
        };
        Ok(Expr { kind, line })
    }

    fn postfix(&mut self, mut e: Expr) -> Res<Expr> {
        loop {
            let line = self.line();
            if self.eat_op(".") {
                let attr = match self.next()? {
                    Tok::Name(n) => n,
                    Tok::Int(i) => format!("{i}"),
                    _ => return Err(self.err(ErrorKind::Syntax("an attribute"))),
                };
                e = Expr {
                    kind: ExprKind::Attr(Box::new(e), attr),
                    line,
                };
            } else if self.eat_op("[") {
                e = self.subscript(e, line)?;
            } else if self.is_op("(") {
                if let ExprKind::Attr(_, m) = &e.kind
                    && MUTATING_METHODS.contains(&m.as_str())
                {
                    return Err(self.err(ErrorKind::Unsupported(mutating(m))));
                }
                self.at += 1;
                let (args, kwargs) = self.call_args()?;
                e = Expr {
                    kind: ExprKind::Call {
                        func: Box::new(e),
                        args,
                        kwargs,
                    },
                    line,
                };
            } else {
                return Ok(e);
            }
        }
    }

    fn subscript(&mut self, value: Expr, line: u32) -> Res<Expr> {
        let part = |p: &mut Self| -> Res<Option<Box<Expr>>> {
            if p.is_op(":") || p.is_op("]") {
                Ok(None)
            } else {
                Ok(Some(Box::new(p.expression(true)?)))
            }
        };
        let start = part(self)?;
        if self.eat_op("]") {
            let index = start.ok_or_else(|| self.err(ErrorKind::Syntax("an index")))?;
            return Ok(Expr {
                kind: ExprKind::Item(Box::new(value), index),
                line,
            });
        }
        self.expect_op(":")?;
        let stop = part(self)?;
        let step = if self.eat_op(":") { part(self)? } else { None };
        self.expect_op("]")?;
        Ok(Expr {
            kind: ExprKind::Slice {
                value: Box::new(value),
                start,
                stop,
                step,
            },
            line,
        })
    }

    /// The arguments of a call, after its "("
    fn call_args(&mut self) -> Res<Args> {
        let (mut args, mut kwargs) = (Vec::new(), Vec::new());
        while !self.eat_op(")") {
            if !(args.is_empty() && kwargs.is_empty()) {
                self.expect_op(",")?;
                if self.eat_op(")") {
                    break;
                }
            }
            if self.is_op("*") || self.is_op("**") {
                return Err(self.err(ErrorKind::Unsupported("an argument list with *")));
            }
            if let (Some(Tok::Name(n)), Some(Tok::Op("="))) = (self.peek(), self.peek_at(1)) {
                let n = n.clone();
                self.at += 2;
                kwargs.push((n, self.expression(true)?));
            } else {
                args.push(self.expression(true)?);
            }
        }
        Ok((args, kwargs))
    }

    fn filter_expr(&mut self, mut e: Expr) -> Res<Expr> {
        loop {
            let line = self.line();
            if self.eat_op("|") {
                let name = self.ident()?;
                let (args, kwargs) = if self.eat_op("(") {
                    self.call_args()?
                } else {
                    (Vec::new(), Vec::new())
                };
                e = Expr {
                    kind: ExprKind::Filter {
                        value: Box::new(e),
                        name,
                        args,
                        kwargs,
                    },
                    line,
                };
            } else if self.eat_name("is") {
                let negated = self.eat_name("not");
                let name = self.ident()?;
                let args = if self.eat_op("(") {
                    let (args, kwargs) = self.call_args()?;
                    if !kwargs.is_empty() {
                        return Err(
                            self.err(ErrorKind::Unsupported("a keyword argument of a test"))
                        );
                    }
                    args
                } else if self.test_arg_follows() {
                    let a = self.primary()?;
                    vec![self.postfix(a)?]
                } else {
                    Vec::new()
                };
                e = Expr {
                    kind: ExprKind::Test {
                        value: Box::new(e),
                        name,
                        args,
                        negated,
                    },
                    line,
                };
            } else {
                return Ok(e);
            }
        }
    }

    /// A test takes one argument with no parentheses, as in jinja2, when a
    /// name, a literal or a bracket follows that is not else, or, and or is
    fn test_arg_follows(&self) -> bool {
        match self.peek() {
            Some(Tok::Name(n)) => !matches!(n.as_str(), "else" | "or" | "and" | "is"),
            Some(Tok::Str(_) | Tok::Int(_) | Tok::Float(_)) => true,
            Some(Tok::Op("(" | "[" | "{")) => true,
            _ => false,
        }
    }
}

fn unsupported_tag(tag: &str) -> &'static str {
    match tag {
        "filter" => "the filter block",
        "call" => "the call block",
        "raw" => "the raw block",
        "include" => "include",
        "import" | "from" => "import",
        "extends" | "block" => "template inheritance",
        "with" => "the with block",
        "do" => "the do statement",
        "autoescape" => "the autoescape block",
        "generation" => "the generation block",
        "trans" => "the trans block",
        _ => "the print statement",
    }
}

fn mutating(method: &str) -> &'static str {
    match method {
        "append" => "the list method append",
        "pop" => "the method pop",
        "extend" => "the list method extend",
        "insert" => "the list method insert",
        "remove" => "the list method remove",
        "update" => "the dict method update",
        "clear" => "the method clear",
        "setdefault" => "the dict method setdefault",
        _ => "the dict method popitem",
    }
}

/// The statements of the tokens of a template
pub fn parse(toks: Vec<Token>) -> Res<Vec<Node>> {
    let mut p = Parser { toks, at: 0 };
    match p.body(&[])? {
        (nodes, None) => Ok(nodes),
        (_, Some(_)) => Err(p.err(ErrorKind::Syntax("an end tag with no start"))),
    }
}
