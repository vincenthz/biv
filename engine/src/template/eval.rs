//! The renderer of templates: the statements and the expressions of the
//! subset, with the semantics of the reference renderer (jinja2 in the
//! immutable sandbox of transformers).
//!
//! An unset top-level name gives Undefined, as in jinja2. Its use under a
//! guard (is defined, the default filter) or as a test (a truth value, a
//! test, a loop over it, a membership in it) is the semantics of the
//! language. Any other use fails by name, TemplateVariableUnknown. Below
//! the top level, an attribute or an item that is not set is the Undefined
//! of jinja2 in all its uses.

use alloc::boxed::Box;
use alloc::rc::Rc;
use core::cell::RefCell;

use super::filters;
use super::parser::{BinOp, CmpOp, Expr, ExprKind, MacroDef, Node, Target};
use super::value::{
    Function, LoopState, MacroRef, Number, OStr, Origin, Undefined, Value, py_eq, to_ostr,
};
use super::{Date, ErrorKind, TemplateError};
use crate::prelude::*;

pub type Res<T> = Result<T, TemplateError>;

/// The deepest nesting of macro calls: a template that recurses past it
/// fails by name, as Python's recursion limit makes jinja2 fail
const MACRO_DEPTH_BOUND: usize = 64;

/// What a statement leaves to the loop around it
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Flow {
    Next,
    Break,
    Continue,
}

pub struct Renderer<'a> {
    /// the variables of the render
    globals: &'a [(String, Value)],
    /// the scopes: the first is the scope of the template; a loop iteration
    /// and a macro call each push one
    frames: Vec<Vec<(String, Value)>>,
    /// the first frame of the macro that runs, whose names hide those of
    /// the frames below it, except the scope of the template
    base: usize,
    /// the macros of the template, by the index of their values
    macros: Vec<Rc<MacroDef>>,
    date: Option<Date>,
    depth: usize,
}

pub fn err(kind: ErrorKind, line: u32) -> TemplateError {
    TemplateError {
        kind,
        line,
        column: 0,
    }
}

/// A TypeError of Python
pub fn type_err(line: u32, msg: String) -> TemplateError {
    err(ErrorKind::Type(msg), line)
}

/// The UndefinedError of jinja2 for a use of `u` that needs a value
pub fn undefined_err(u: &Undefined, line: u32) -> TemplateError {
    if u.top {
        err(ErrorKind::VariableUnknown(u.name.clone()), line)
    } else {
        err(
            ErrorKind::Undefined(format!("'{}' is undefined", u.name)),
            line,
        )
    }
}

/// The value of `v` for a use that is no guard and no test: an unset
/// top-level name fails by name; an unset attribute stays Undefined
pub fn not_top(v: Value, line: u32) -> Res<Value> {
    match &v {
        Value::Undefined(u) if u.top => Err(undefined_err(u, line)),
        _ => Ok(v),
    }
}

/// The value of `v` for a use that needs a value, as arithmetic does:
/// every Undefined fails
pub fn defined(v: Value, line: u32) -> Res<Value> {
    match &v {
        Value::Undefined(u) => Err(undefined_err(u, line)),
        _ => Ok(v),
    }
}

impl<'a> Renderer<'a> {
    pub fn new(globals: &'a [(String, Value)], date: Option<Date>) -> Self {
        Renderer {
            globals,
            frames: vec![Vec::new()],
            base: 0,
            macros: Vec::new(),
            date,
            depth: 0,
        }
    }

    pub fn date(&self) -> Option<Date> {
        self.date
    }

    fn lookup(&self, name: &str, line: u32) -> Value {
        let found = |frame: &Vec<(String, Value)>| {
            frame
                .iter()
                .rev()
                .find(|(n, _)| n == name)
                .map(|(_, v)| v.clone())
        };
        for i in (self.base..self.frames.len()).rev() {
            if let Some(v) = found(&self.frames[i]) {
                return v;
            }
        }
        if self.base > 0
            && let Some(v) = found(&self.frames[0])
        {
            return v;
        }
        if let Some((_, v)) = self.globals.iter().find(|(n, _)| n == name) {
            return v.clone();
        }
        match name {
            "range" => Value::Function(Function::Range),
            "namespace" => Value::Function(Function::Namespace),
            "raise_exception" => Value::Function(Function::RaiseException),
            "strftime_now" => Value::Function(Function::StrftimeNow),
            _ => Value::Undefined(Rc::new(Undefined {
                name: name.into(),
                top: true,
                line,
            })),
        }
    }

    fn assign(&mut self, name: &str, v: Value) {
        let frame = self.frames.last_mut().expect("a frame");
        match frame.iter_mut().find(|(n, _)| n == name) {
            Some(slot) => slot.1 = v,
            None => frame.push((name.into(), v)),
        }
    }

    fn assign_target(&mut self, target: &Target, v: Value, line: u32) -> Res<()> {
        match target {
            Target::Name(n) => self.assign(n, v),
            Target::Tuple(names) => {
                let items = filters::iterate(&v, line)?;
                if items.len() != names.len() {
                    return Err(err(
                        ErrorKind::Value(format!(
                            "not enough values to unpack (expected {}, got {})",
                            names.len(),
                            items.len()
                        )),
                        line,
                    ));
                }
                for (n, item) in names.iter().zip(items) {
                    self.assign(n, item);
                }
            }
        }
        Ok(())
    }

    pub fn render(&mut self, nodes: &[Node], out: &mut OStr) -> Res<()> {
        match self.nodes(nodes, out)? {
            Flow::Next => Ok(()),
            // jinja2 refuses break and continue outside a loop at compile
            Flow::Break | Flow::Continue => Err(err(
                ErrorKind::Syntax("a break or continue outside a loop"),
                0,
            )),
        }
    }

    fn nodes(&mut self, nodes: &[Node], out: &mut OStr) -> Res<Flow> {
        for node in nodes {
            let flow = self.node(node, out)?;
            if flow != Flow::Next {
                return Ok(flow);
            }
        }
        Ok(Flow::Next)
    }

    fn node(&mut self, node: &Node, out: &mut OStr) -> Res<Flow> {
        match node {
            Node::Text(t) => out.push(t, Origin::Template),
            Node::Output(e) => {
                let v = not_top(self.eval(e)?, e.line)?;
                // the reference prints a generator with its address
                if matches!(v, Value::Generator(_)) {
                    return Err(err(
                        ErrorKind::Unsupported("the output of a generator"),
                        e.line,
                    ));
                }
                out.push_ostr(&to_ostr(&v));
            }
            Node::If {
                branches,
                otherwise,
            } => {
                for (test, body) in branches {
                    if self.eval(test)?.truthy() {
                        return self.nodes(body, out);
                    }
                }
                return self.nodes(otherwise, out);
            }
            Node::For {
                target,
                iter,
                filter,
                body,
            } => return self.for_loop(target, iter, filter.as_ref(), body, out),
            Node::Set { target, value } => {
                let v = not_top(self.eval(value)?, value.line)?;
                self.assign_target(target, v, value.line)?;
            }
            Node::SetAttr {
                name,
                attr,
                value,
                line,
            } => {
                let v = not_top(self.eval(value)?, value.line)?;
                match self.lookup(name, *line) {
                    Value::Namespace(ns) => {
                        let mut ns = ns.borrow_mut();
                        match ns.iter_mut().find(|(n, _)| n == attr) {
                            Some(slot) => slot.1 = v,
                            None => ns.push((attr.clone(), v)),
                        }
                    }
                    Value::Undefined(u) => return Err(undefined_err(&u, *line)),
                    _ => {
                        return Err(err(
                            ErrorKind::Runtime(
                                "cannot assign attribute on non-namespace object".into(),
                            ),
                            *line,
                        ));
                    }
                }
            }
            Node::SetBlock { name, body } => {
                let mut inner = OStr::default();
                // a set block has its own scope for its body
                self.frames.push(Vec::new());
                let flow = self.nodes(body, &mut inner);
                self.frames.pop();
                flow?;
                self.assign(name, Value::ostr(inner));
            }
            Node::Macro(def) => {
                let index = self.macros.len();
                self.macros.push(def.clone());
                self.assign(&def.name, Value::Macro(Rc::new(MacroRef { index })));
            }
            Node::Break => return Ok(Flow::Break),
            Node::Continue => return Ok(Flow::Continue),
        }
        Ok(Flow::Next)
    }

    fn for_loop(
        &mut self,
        target: &Target,
        iter: &Expr,
        filter: Option<&Expr>,
        body: &[Node],
        out: &mut OStr,
    ) -> Res<Flow> {
        // a loop over an unset value runs no iteration
        let seq = self.eval(iter)?;
        let all = filters::iterate(&seq, iter.line)?;
        let items = match filter {
            None => all,
            Some(cond) => {
                let mut kept = Vec::new();
                for item in all {
                    self.frames.push(Vec::new());
                    let r = self
                        .assign_target(target, item.clone(), cond.line)
                        .and_then(|()| self.eval(cond));
                    self.frames.pop();
                    if r?.truthy() {
                        kept.push(item);
                    }
                }
                kept
            }
        };
        let length = items.len();
        for (i, item) in items.iter().enumerate() {
            self.frames.push(Vec::new());
            let state = LoopState {
                index0: i,
                length,
                previtem: i.checked_sub(1).map(|p| items[p].clone()),
                nextitem: items.get(i + 1).cloned(),
            };
            self.assign("loop", Value::Loop(Rc::new(state)));
            let flow = self
                .assign_target(target, item.clone(), iter.line)
                .and_then(|()| self.nodes(body, out));
            self.frames.pop();
            match flow? {
                Flow::Break => break,
                Flow::Next | Flow::Continue => {}
            }
        }
        Ok(Flow::Next)
    }

    pub fn eval(&mut self, e: &Expr) -> Res<Value> {
        let line = e.line;
        Ok(match &e.kind {
            ExprKind::Str(s) => Value::str(s, Origin::Template),
            ExprKind::Int(i) => Value::Int(*i),
            ExprKind::Float(f) => Value::Float(*f),
            ExprKind::Bool(b) => Value::Bool(*b),
            ExprKind::None => Value::None,
            ExprKind::List(items) => {
                let mut v = Vec::new();
                for i in items {
                    v.push(not_top(self.eval(i)?, i.line)?);
                }
                Value::list(v)
            }
            ExprKind::Tuple(items) => {
                let mut v = Vec::new();
                for i in items {
                    v.push(not_top(self.eval(i)?, i.line)?);
                }
                Value::Tuple(Rc::new(v))
            }
            ExprKind::Dict(items) => {
                let mut v: Vec<(Value, Value)> = Vec::new();
                for (k, val) in items {
                    let k = not_top(self.eval(k)?, k.line)?;
                    let val = not_top(self.eval(val)?, val.line)?;
                    match v.iter_mut().find(|(old, _)| py_eq(old, &k)) {
                        Some(slot) => slot.1 = val,
                        None => v.push((k, val)),
                    }
                }
                Value::map(v)
            }
            ExprKind::Name(n) => self.lookup(n, line),
            ExprKind::Attr(obj, attr) => {
                let o = defined(self.eval(obj)?, line)?;
                filters::getattr(&o, attr, line)?
            }
            ExprKind::Item(obj, index) => {
                let o = defined(self.eval(obj)?, line)?;
                let i = not_top(self.eval(index)?, line)?;
                filters::getitem(&o, &i, line)?
            }
            ExprKind::Slice {
                value,
                start,
                stop,
                step,
            } => {
                let v = defined(self.eval(value)?, line)?;
                let mut part = |p: &Option<Box<Expr>>| -> Res<Option<i64>> {
                    match p {
                        None => Ok(None),
                        Some(p) => match defined(self.eval(p)?, line)? {
                            Value::None => Ok(None),
                            Value::Int(i) => Ok(Some(i)),
                            Value::Bool(b) => Ok(Some(i64::from(b))),
                            other => Err(type_err(
                                line,
                                format!(
                                    "slice indices must be integers or None, not {}",
                                    other.type_name()
                                ),
                            )),
                        },
                    }
                };
                let (a, b, c) = (part(start)?, part(stop)?, part(step)?);
                filters::slice(&v, a, b, c, line)?
            }
            ExprKind::Call { func, args, kwargs } => self.call(func, args, kwargs, line)?,
            ExprKind::Filter {
                value,
                name,
                args,
                kwargs,
            } => {
                let v = self.eval(value)?;
                let mut a = Vec::new();
                for x in args {
                    a.push(not_top(self.eval(x)?, x.line)?);
                }
                let mut k = Vec::new();
                for (n, x) in kwargs {
                    k.push((n.clone(), not_top(self.eval(x)?, x.line)?));
                }
                filters::filter(name, v, &a, &k, line)?
            }
            ExprKind::Test {
                value,
                name,
                args,
                negated,
            } => {
                let v = self.eval(value)?;
                let mut a = Vec::new();
                for x in args {
                    a.push(not_top(self.eval(x)?, x.line)?);
                }
                let yes = filters::test(name, &v, &a, line)?;
                Value::Bool(yes != *negated)
            }
            ExprKind::Neg(x) => {
                let v = defined(self.eval(x)?, line)?;
                match v.number() {
                    Some(Number::Int(i)) => Value::Int(-i),
                    Some(Number::Float(f)) => Value::Float(-f),
                    None => {
                        return Err(type_err(
                            line,
                            format!("bad operand type for unary -: '{}'", v.type_name()),
                        ));
                    }
                }
            }
            ExprKind::Pos(x) => {
                let v = defined(self.eval(x)?, line)?;
                match v.number() {
                    Some(Number::Int(i)) => Value::Int(i),
                    Some(Number::Float(f)) => Value::Float(f),
                    None => {
                        return Err(type_err(
                            line,
                            format!("bad operand type for unary +: '{}'", v.type_name()),
                        ));
                    }
                }
            }
            ExprKind::Not(x) => Value::Bool(!self.eval(x)?.truthy()),
            ExprKind::And(a, b) => {
                let l = self.eval(a)?;
                if l.truthy() { self.eval(b)? } else { l }
            }
            ExprKind::Or(a, b) => {
                let l = self.eval(a)?;
                if l.truthy() { l } else { self.eval(b)? }
            }
            ExprKind::Binary(op, a, b) => {
                let (l, r) = (self.eval(a)?, self.eval(b)?);
                filters::binary(*op, l, r, line)?
            }
            ExprKind::Compare(first, ops) => {
                let mut left = self.eval(first)?;
                for (op, right) in ops {
                    let r = self.eval(right)?;
                    if !compare(*op, &left, &r, line)? {
                        return Ok(Value::Bool(false));
                    }
                    left = r;
                }
                Value::Bool(true)
            }
            ExprKind::Cond {
                test,
                then,
                otherwise,
            } => {
                if self.eval(test)?.truthy() {
                    self.eval(then)?
                } else {
                    match otherwise {
                        Some(o) => self.eval(o)?,
                        // jinja2 gives Undefined for an if expression with no
                        // else whose test is false
                        None => Value::Undefined(Rc::new(Undefined {
                            name: "the else of an if expression".into(),
                            top: false,
                            line,
                        })),
                    }
                }
            }
        })
    }

    fn call(
        &mut self,
        func: &Expr,
        args: &[Expr],
        kwargs: &[(String, Expr)],
        line: u32,
    ) -> Res<Value> {
        let f = defined(self.eval(func)?, line)?;
        let mut a = Vec::new();
        for x in args {
            a.push(not_top(self.eval(x)?, x.line)?);
        }
        let mut k = Vec::new();
        for (n, x) in kwargs {
            k.push((n.clone(), not_top(self.eval(x)?, x.line)?));
        }
        match f {
            Value::Macro(m) => self.call_macro(m.index, &a, &k, line),
            Value::Function(func) => filters::call_function(self, func, &a, &k, line),
            Value::Method(m) => filters::call_method(&m.0, &m.1, &a, &k, line),
            other => Err(type_err(
                line,
                format!("'{}' object is not callable", other.type_name()),
            )),
        }
    }

    fn call_macro(
        &mut self,
        index: usize,
        args: &[Value],
        kwargs: &[(String, Value)],
        line: u32,
    ) -> Res<Value> {
        let def = self.macros[index].clone();
        if args.len() > def.params.len() {
            return Err(type_err(
                line,
                format!(
                    "macro '{}' takes not more than {} argument(s)",
                    def.name,
                    def.params.len()
                ),
            ));
        }
        if self.depth >= MACRO_DEPTH_BOUND {
            return Err(err(ErrorKind::Recursion(def.name.clone()), line));
        }
        // the frame of the macro: its parameters
        let mut frame = Vec::new();
        for (i, (p, default)) in def.params.iter().enumerate() {
            let v = if let Some(v) = args.get(i) {
                v.clone()
            } else if let Some((_, v)) = kwargs.iter().find(|(n, _)| n == p) {
                v.clone()
            } else if let Some(d) = default {
                self.eval(d)?
            } else {
                // jinja2 gives an unset parameter as Undefined
                Value::Undefined(Rc::new(Undefined {
                    name: p.clone(),
                    top: false,
                    line,
                }))
            };
            frame.push((p.clone(), v));
        }
        if let Some((n, _)) = kwargs
            .iter()
            .find(|(n, _)| !def.params.iter().any(|(p, _)| p == n))
        {
            return Err(type_err(
                line,
                format!("macro '{}' takes no keyword argument '{n}'", def.name),
            ));
        }
        let saved = self.base;
        self.frames.push(frame);
        self.base = self.frames.len() - 1;
        self.depth += 1;
        let mut out = OStr::default();
        let r = self.nodes(&def.body, &mut out);
        self.depth -= 1;
        self.frames.pop();
        self.base = saved;
        r?;
        Ok(Value::ostr(out))
    }

    /// The value of a namespace() call
    pub fn namespace(&self, args: &[Value], kwargs: &[(String, Value)], line: u32) -> Res<Value> {
        let mut attrs: Vec<(String, Value)> = Vec::new();
        for a in args {
            match a {
                Value::Map(m) => {
                    for (k, v) in m.iter() {
                        let Value::Str(k) = k else {
                            return Err(type_err(line, "keywords must be strings".into()));
                        };
                        attrs.push((k.as_str().into(), v.clone()));
                    }
                }
                other => {
                    return Err(type_err(
                        line,
                        format!(
                            "'{}' object is not iterable as a namespace",
                            other.type_name()
                        ),
                    ));
                }
            }
        }
        for (k, v) in kwargs {
            match attrs.iter_mut().find(|(n, _)| n == k) {
                Some(slot) => slot.1 = v.clone(),
                None => attrs.push((k.clone(), v.clone())),
            }
        }
        Ok(Value::Namespace(Rc::new(RefCell::new(attrs))))
    }
}

/// One comparison of Python
fn compare(op: CmpOp, l: &Value, r: &Value, line: u32) -> Res<bool> {
    match op {
        CmpOp::Eq | CmpOp::Ne => {
            let l = not_top(l.clone(), line)?;
            let r = not_top(r.clone(), line)?;
            filters::plain(&l, "a comparison", line)?;
            filters::plain(&r, "a comparison", line)?;
            let eq = py_eq(&l, &r);
            Ok(if op == CmpOp::Eq { eq } else { !eq })
        }
        CmpOp::In | CmpOp::NotIn => {
            // a membership in an unset value is a test of it: it holds nothing
            let l = not_top(l.clone(), line)?;
            let yes = filters::contains(r, &l, line)?;
            Ok(if op == CmpOp::In { yes } else { !yes })
        }
        CmpOp::Lt | CmpOp::Le | CmpOp::Gt | CmpOp::Ge => {
            let l = defined(l.clone(), line)?;
            let r = defined(r.clone(), line)?;
            let ord = filters::order(&l, &r, line, op_symbol(op))?;
            Ok(match op {
                CmpOp::Lt => ord.is_lt(),
                CmpOp::Le => ord.is_le(),
                CmpOp::Gt => ord.is_gt(),
                _ => ord.is_ge(),
            })
        }
    }
}

fn op_symbol(op: CmpOp) -> &'static str {
    match op {
        CmpOp::Lt => "<",
        CmpOp::Le => "<=",
        CmpOp::Gt => ">",
        _ => ">=",
    }
}

/// The binary operator `op` of Python, with the operands as they come
pub fn binop_name(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Mod => "%",
        BinOp::Concat => "~",
    }
}
