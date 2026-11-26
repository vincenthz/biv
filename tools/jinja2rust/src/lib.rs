//! Compiles a chat template, the Jinja that transformers renders with
//! `trim_blocks` and `lstrip_blocks`, into Rust code over the `rt` runtime
//! (`rt/rt.rs`): a `Params` struct of the template's parameters as native
//! types, and a `render` function from them to the rendered prompt, whose
//! text keeps the template's own apart from the data's.
//!
//! The parameters are the variables the template reads but never sets. The
//! standard ones have their types (`messages: &[rt::Message]`, `tools:
//! &[rt::Tool]`, `add_generation_prompt: bool`, the special tokens), the
//! others get one from how the template reads them (see [`params`]).
//!
//! The translation is direct: Jinja's scopes become Rust blocks, its variables
//! Rust variables (`g_` at the template's top level, `v_` below), its macros
//! functions (`m_`), over the runtime's dynamic values. Whatever the runtime
//! doesn't implement is refused with the line it is on, so a template either
//! compiles whole or not at all.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;

pub mod params;

use minijinja::machinery::{Span, ast, parse};
use minijinja::syntax::SyntaxConfig;
use minijinja::value::{Value as JValue, ValueKind};

#[derive(Debug)]
pub struct GenError {
    pub line: usize,
    pub msg: String,
}

impl fmt::Display for GenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.msg)
    }
}

impl std::error::Error for GenError {}

type G<T = Code> = Result<T, GenError>;

fn fail<T>(span: Span, msg: impl Into<String>) -> G<T> {
    Err(GenError {
        line: span.start_line as usize,
        msg: msg.into(),
    })
}

pub struct Options<'a> {
    /// what the template is, for the generated file's doc comment
    pub name: &'a str,
    /// the path of the runtime module from the generated one
    pub rt_path: &'a str,
}

pub struct Generated {
    pub code: String,
    /// the template's parameters, the context variables it reads
    pub params: Vec<params::Param>,
    /// FNV-1a of the template's text, to recognize it in a model
    pub hash: u64,
}

pub fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Where a context variable's value goes in the code before the template's
/// parameters are known: the variable's field, or undefined without one
const CTX_MARK: &str = "@@ctx:";

/// A Jinja name as a Rust field
fn field(name: &str) -> String {
    const KEYWORDS: [&str; 39] = [
        "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else", "enum",
        "extern", "false", "fn", "for", "gen", "if", "impl", "in", "let", "loop", "match", "mod",
        "move", "mut", "pub", "ref", "return", "self", "Self", "static", "struct", "super",
        "trait", "true", "type", "unsafe", "use", "where", "while",
    ];
    if KEYWORDS.contains(&name) {
        format!("r#{name}")
    } else {
        name.to_owned()
    }
}

/// `source` with transformers' `{% generation %}` and `{% endgeneration %}`
/// tags, which mark the assistant's text for training masks and render their
/// body as it is, made `{% if true %}` and `{% endif %}`: their whitespace
/// control kept, on the same line
fn rewrite_generation(source: &str) -> std::borrow::Cow<'_, str> {
    // the tag's word at the start of `rest`: its offset, length and new word
    let tag = |rest: &str| -> Option<(usize, usize, &'static str)> {
        let inner = rest.strip_prefix("{%")?;
        let inner = inner.strip_prefix(['-', '+']).unwrap_or(inner).trim_start();
        let (word, by) = [("endgeneration", "endif"), ("generation", "if true")]
            .into_iter()
            .find(|(w, _)| inner.starts_with(w))?;
        let after = inner[word.len()..].trim_start();
        let after = after.strip_prefix(['-', '+']).unwrap_or(after);
        after
            .starts_with("%}")
            .then_some((rest.len() - inner.len(), word.len(), by))
    };
    let (mut out, mut done, mut at) = (String::new(), 0, 0);
    while let Some(i) = source[at..].find("{%") {
        let start = at + i;
        at = start + 2;
        if let Some((offset, len, by)) = tag(&source[start..]) {
            out.push_str(&source[done..start + offset]);
            out.push_str(by);
            done = start + offset + len;
        }
    }
    if done == 0 {
        return std::borrow::Cow::Borrowed(source);
    }
    out.push_str(&source[done..]);
    std::borrow::Cow::Owned(out)
}

pub fn generate(source: &str, opts: &Options<'_>) -> Result<Generated, GenError> {
    let config = SyntaxConfig::builder()
        .trim_blocks(true)
        .lstrip_blocks(true)
        .build()
        .expect("the default syntax");
    // the hash below is of the text as it is
    let text = rewrite_generation(source);
    let root = parse(&text, "template", config).map_err(|e| GenError {
        line: e.line().unwrap_or(0),
        msg: e.detail().unwrap_or("syntax error").to_string(),
    })?;
    let ast::Stmt::Template(t) = &root else {
        unreachable!("parse gives a template")
    };
    let mut g = Gen::default();
    g.collect_macros(&t.children)?;
    let top = assigned(&t.children);

    g.line("/// Render the template");
    g.open("pub fn render(params: &Params<'_>) -> rt::R<rt::Str> {");
    g.line("let ctx = &params.context();");
    g.line("let mut out = rt::Str::default();");
    for name in &top {
        g.line(&format!("let mut g_{name} = {CTX_MARK}{name}@@;"));
    }
    g.frames = vec![Frame::new("g_", top.iter().cloned())];
    g.stmts(&t.children)?;
    g.line("Ok(out)");
    g.close("}");
    let render = std::mem::take(&mut g.w);

    for stmt in &t.children {
        if let ast::Stmt::Macro(m) = stmt {
            g.gen_macro(m)?;
        }
    }
    let macros = std::mem::take(&mut g.w);

    // a parameter is a name the template never sets: then all its reads are
    // the parameter's
    let mut bound = HashSet::new();
    bound_names(&t.children, &mut bound);
    if let Some(name) = g.context.iter().find(|n| bound.contains(*n)) {
        return Err(GenError {
            line: 0,
            msg: format!("`{name}` is read from the context, and set by the template"),
        });
    }
    let params = params::infer(&t.children, &g.context)?;

    let hash = fnv1a(source);
    let mut code = String::new();
    code.push_str(&format!(
        "//! The chat template of {}, generated by jinja2rust: do not edit.\n\n\
         #![allow(unused_mut, unused_variables, unused_assignments, unused_imports, unreachable_code)]\n\
         #![allow(clippy::all)]\n\n\
         use {} as rt;\n\
         use alloc::vec;\n\
         use alloc::vec::Vec;\n\n\
         /// FNV-1a of the template's text\n\
         pub const TEMPLATE_HASH: u64 = {:#018x};\n\n",
        opts.name, opts.rt_path, hash
    ));
    code.push_str(&params_code(&params));
    code.push_str(&render);
    code.push_str(&macros);
    // the context variables' values, now that the parameters are known
    let mut resolved = String::with_capacity(code.len());
    let mut rest = code.as_str();
    while let Some(at) = rest.find(CTX_MARK) {
        resolved.push_str(&rest[..at]);
        let after = &rest[at + CTX_MARK.len()..];
        let end = after.find("@@").expect("a closed mark");
        let name = &after[..end];
        if params.iter().any(|p| p.name == name) {
            resolved.push_str(&format!("ctx.{}.clone()", field(name)));
        } else {
            resolved.push_str("rt::Value::Undefined");
        }
        rest = &after[end + 2..];
    }
    resolved.push_str(rest);
    Ok(Generated {
        code: resolved,
        params,
        hash,
    })
}

/// The `Params` the caller fills, and the context the template reads
fn params_code(params: &[params::Param]) -> String {
    let mut c = String::new();
    c.push_str("/// The template's parameters\n");
    c.push_str("#[derive(Clone, Copy, Debug, Default)]\n");
    c.push_str("pub struct Params<'a> {\n");
    for p in params {
        let lines: Vec<String> = p.lines.iter().map(|l| l.to_string()).collect();
        let what = match p.kind {
            params::Kind::Messages => "the conversation".to_owned(),
            params::Kind::Tools => "the functions the model may call".to_owned(),
            _ => format!("`{}`", p.name),
        };
        let note = match p.kind {
            params::Kind::OptBool | params::Kind::OptStr => ": `None` leaves it undefined",
            params::Kind::Token => ": the special token",
            _ => "",
        };
        c.push_str(&format!(
            "    /// {what}{note} (line{} {})\n",
            if lines.len() > 1 { "s" } else { "" },
            lines.join(", ")
        ));
        c.push_str(&format!(
            "    pub {}: {},\n",
            field(&p.name),
            p.kind.rust_type()
        ));
    }
    c.push_str("}\n\n");
    c.push_str("/// The parameters as the template's values\n");
    c.push_str("struct Ctx {\n");
    for p in params {
        c.push_str(&format!("    {}: rt::Value,\n", field(&p.name)));
    }
    c.push_str("}\n\n");
    c.push_str("impl Params<'_> {\n");
    c.push_str("    fn context(&self) -> Ctx {\n");
    c.push_str("        Ctx {\n");
    for p in params {
        let f = field(&p.name);
        c.push_str(&format!("            {f}: {},\n", p.kind.to_value(&f)));
    }
    c.push_str("        }\n    }\n}\n\n");
    c
}

/// Every name a template sets anywhere: assigned, a loop's, a macro's or a
/// `with`'s
fn bound_names(stmts: &[ast::Stmt<'_>], out: &mut HashSet<String>) {
    let target = |e: &ast::Expr<'_>, out: &mut HashSet<String>| {
        let mut names = HashSet::new();
        expr_vars(e, &mut names);
        // `ns.x = …` sets an attribute: `ns` is read
        if !matches!(e, ast::Expr::GetAttr(_)) {
            out.extend(names);
        }
    };
    for s in stmts {
        match s {
            ast::Stmt::Template(t) => bound_names(&t.children, out),
            ast::Stmt::ForLoop(f) => {
                target(&f.target, out);
                bound_names(&f.body, out);
                bound_names(&f.else_body, out);
            }
            ast::Stmt::IfCond(c) => {
                bound_names(&c.true_body, out);
                bound_names(&c.false_body, out);
            }
            ast::Stmt::WithBlock(w) => {
                w.assignments.iter().for_each(|(t, _)| target(t, out));
                bound_names(&w.body, out);
            }
            ast::Stmt::Set(s) => target(&s.target, out),
            ast::Stmt::SetBlock(s) => {
                target(&s.target, out);
                bound_names(&s.body, out);
            }
            ast::Stmt::Macro(m) => {
                m.args.iter().for_each(|a| target(a, out));
                bound_names(&m.body, out);
            }
            ast::Stmt::AutoEscape(a) => bound_names(&a.body, out),
            ast::Stmt::FilterBlock(f) => bound_names(&f.body, out),
            ast::Stmt::CallBlock(c) => bound_names(&c.macro_decl.body, out),
            _ => {}
        }
    }
}

/// Rust code for a Jinja value: a place (a variable) or a value expression
enum Code {
    Place(String),
    Value(String),
}

impl Code {
    /// an owned `rt::Value`
    fn val(self) -> String {
        match self {
            Code::Place(p) => format!("{p}.clone()"),
            Code::Value(v) => v,
        }
    }

    /// a `&rt::Value`
    fn r(self) -> String {
        match self {
            Code::Place(p) => format!("&{p}"),
            Code::Value(v) => format!("&{v}"),
        }
    }
}

struct Frame {
    prefix: &'static str,
    names: HashSet<String>,
    is_loop: bool,
}

impl Frame {
    fn new(prefix: &'static str, names: impl IntoIterator<Item = String>) -> Frame {
        Frame {
            prefix,
            names: names.into_iter().collect(),
            is_loop: false,
        }
    }
}

#[derive(Clone)]
struct MacroInfo {
    params: Vec<String>,
    /// the template's top-level variables it reads, passed by the caller
    captured: Vec<String>,
}

#[derive(Default)]
struct Gen {
    macros: HashMap<String, MacroInfo>,
    frames: Vec<Frame>,
    /// the context variables the template reads
    context: BTreeSet<String>,
    tmp: usize,
    depth: usize,
    w: String,
    indent: usize,
}

/// The variables a scope's body assigns: `set` here or in its `if`s
fn assigned(stmts: &[ast::Stmt<'_>]) -> Vec<String> {
    fn targets(e: &ast::Expr<'_>, out: &mut Vec<String>) {
        match e {
            ast::Expr::Var(v) => {
                if !out.iter().any(|n| n == v.id) {
                    out.push(v.id.to_owned());
                }
            }
            ast::Expr::Tuple(t) => t.items.iter().for_each(|e| targets(e, out)),
            ast::Expr::List(l) => l.items.iter().for_each(|e| targets(e, out)),
            _ => {}
        }
    }
    fn walk(stmts: &[ast::Stmt<'_>], out: &mut Vec<String>) {
        for s in stmts {
            match s {
                ast::Stmt::Set(s) => targets(&s.target, out),
                ast::Stmt::SetBlock(s) => targets(&s.target, out),
                ast::Stmt::IfCond(c) => {
                    walk(&c.true_body, out);
                    walk(&c.false_body, out);
                }
                _ => {}
            }
        }
    }
    let mut out = Vec::new();
    walk(stmts, &mut out);
    out
}

/// Every variable name statements mention, nested scopes included
fn mentioned(stmts: &[ast::Stmt<'_>], out: &mut HashSet<String>) {
    for s in stmts {
        match s {
            ast::Stmt::Template(t) => mentioned(&t.children, out),
            ast::Stmt::EmitExpr(e) => expr_vars(&e.expr, out),
            ast::Stmt::EmitRaw(_) => {}
            ast::Stmt::ForLoop(f) => {
                expr_vars(&f.target, out);
                expr_vars(&f.iter, out);
                if let Some(e) = &f.filter_expr {
                    expr_vars(e, out);
                }
                mentioned(&f.body, out);
                mentioned(&f.else_body, out);
            }
            ast::Stmt::IfCond(c) => {
                expr_vars(&c.expr, out);
                mentioned(&c.true_body, out);
                mentioned(&c.false_body, out);
            }
            ast::Stmt::WithBlock(w) => {
                for (t, e) in &w.assignments {
                    expr_vars(t, out);
                    expr_vars(e, out);
                }
                mentioned(&w.body, out);
            }
            ast::Stmt::Set(s) => {
                expr_vars(&s.target, out);
                expr_vars(&s.expr, out);
            }
            ast::Stmt::SetBlock(s) => {
                expr_vars(&s.target, out);
                if let Some(f) = &s.filter {
                    expr_vars(f, out);
                }
                mentioned(&s.body, out);
            }
            ast::Stmt::AutoEscape(a) => mentioned(&a.body, out),
            ast::Stmt::FilterBlock(f) => mentioned(&f.body, out),
            ast::Stmt::Macro(m) => mentioned(&m.body, out),
            ast::Stmt::CallBlock(c) => {
                expr_vars(&c.call.expr, out);
                c.call.args.iter().for_each(|a| arg_vars(a, out));
                mentioned(&c.macro_decl.body, out);
            }
            ast::Stmt::Continue(_) | ast::Stmt::Break(_) => {}
            ast::Stmt::Do(d) => {
                expr_vars(&d.call.expr, out);
                d.call.args.iter().for_each(|a| arg_vars(a, out));
            }
        }
    }
}

fn arg_vars(a: &ast::CallArg<'_>, out: &mut HashSet<String>) {
    match a {
        ast::CallArg::Pos(e)
        | ast::CallArg::Kwarg(_, e)
        | ast::CallArg::PosSplat(e)
        | ast::CallArg::KwargSplat(e) => expr_vars(e, out),
    }
}

fn expr_vars(e: &ast::Expr<'_>, out: &mut HashSet<String>) {
    match e {
        ast::Expr::Var(v) => {
            out.insert(v.id.to_owned());
        }
        ast::Expr::Const(_) => {}
        ast::Expr::Slice(s) => {
            expr_vars(&s.expr, out);
            for e in [&s.start, &s.stop, &s.step].into_iter().flatten() {
                expr_vars(e, out);
            }
        }
        ast::Expr::UnaryOp(u) => expr_vars(&u.expr, out),
        ast::Expr::BinOp(b) => {
            expr_vars(&b.left, out);
            expr_vars(&b.right, out);
        }
        ast::Expr::Compare(c) => {
            expr_vars(&c.expr, out);
            c.ops.iter().for_each(|o| expr_vars(&o.expr, out));
        }
        ast::Expr::IfExpr(i) => {
            expr_vars(&i.test_expr, out);
            expr_vars(&i.true_expr, out);
            if let Some(e) = &i.false_expr {
                expr_vars(e, out);
            }
        }
        ast::Expr::Filter(f) => {
            if let Some(e) = &f.expr {
                expr_vars(e, out);
            }
            f.args.iter().for_each(|a| arg_vars(a, out));
        }
        ast::Expr::Test(t) => {
            expr_vars(&t.expr, out);
            t.args.iter().for_each(|a| arg_vars(a, out));
        }
        ast::Expr::GetAttr(g) => expr_vars(&g.expr, out),
        ast::Expr::GetItem(g) => {
            expr_vars(&g.expr, out);
            expr_vars(&g.subscript_expr, out);
        }
        ast::Expr::Call(c) => {
            expr_vars(&c.expr, out);
            c.args.iter().for_each(|a| arg_vars(a, out));
        }
        ast::Expr::List(l) => l.items.iter().for_each(|e| expr_vars(e, out)),
        ast::Expr::Tuple(t) => t.items.iter().for_each(|e| expr_vars(e, out)),
        ast::Expr::Map(m) => {
            m.keys.iter().for_each(|e| expr_vars(e, out));
            m.values.iter().for_each(|e| expr_vars(e, out));
        }
    }
}

fn target_list<'a>(e: &'a ast::Expr<'a>) -> Option<Vec<&'a str>> {
    let items = match e {
        ast::Expr::Var(v) => return Some(vec![v.id]),
        ast::Expr::Tuple(t) => &t.items,
        ast::Expr::List(l) => &l.items,
        _ => return None,
    };
    items
        .iter()
        .map(|e| match e {
            ast::Expr::Var(v) => Some(v.id),
            _ => None,
        })
        .collect()
}

fn lit(s: &str) -> String {
    format!("{s:?}")
}

/// The positional and keyword arguments of a call, by parameter name
fn bind_args<'e>(
    span: Span,
    what: &str,
    params: &[&str],
    args: &'e [ast::CallArg<'e>],
) -> G<Vec<Option<&'e ast::Expr<'e>>>> {
    let mut out: Vec<Option<&ast::Expr<'_>>> = vec![None; params.len()];
    let mut pos = 0;
    for a in args {
        match a {
            ast::CallArg::Pos(e) => {
                if pos >= params.len() {
                    return fail(span, format!("too many arguments to {what}"));
                }
                out[pos] = Some(e);
                pos += 1;
            }
            ast::CallArg::Kwarg(name, e) => match params.iter().position(|p| p == name) {
                Some(i) => out[i] = Some(e),
                None => return fail(span, format!("{what} has no argument {name}")),
            },
            _ => return fail(span, format!("argument splats to {what}")),
        }
    }
    Ok(out)
}

impl Gen {
    fn line(&mut self, s: &str) {
        for _ in 0..self.indent {
            self.w.push_str("    ");
        }
        self.w.push_str(s);
        self.w.push('\n');
    }

    fn open(&mut self, s: &str) {
        self.line(s);
        self.indent += 1;
    }

    fn close(&mut self, s: &str) {
        self.indent -= 1;
        self.line(s);
    }

    fn tmp(&mut self) -> String {
        self.tmp += 1;
        format!("t{}", self.tmp)
    }

    fn collect_macros(&mut self, stmts: &[ast::Stmt<'_>]) -> G<()> {
        let top: HashSet<String> = assigned(stmts).into_iter().collect();
        let mut reads: Vec<(String, HashSet<String>)> = Vec::new();
        for s in stmts {
            if let ast::Stmt::Macro(m) = s {
                let mut params = Vec::new();
                for a in &m.args {
                    match a {
                        ast::Expr::Var(v) => params.push(v.id.to_owned()),
                        _ => return fail(m.span(), "a macro parameter that is not a name"),
                    }
                }
                let mut seen = HashSet::new();
                mentioned(&m.body, &mut seen);
                m.defaults.iter().for_each(|d| expr_vars(d, &mut seen));
                reads.push((m.name.to_owned(), seen));
                self.macros.insert(
                    m.name.to_owned(),
                    MacroInfo {
                        params,
                        captured: Vec::new(),
                    },
                );
            }
        }
        // what each macro captures, through the macros it calls
        let mut captured: HashMap<String, BTreeSet<String>> = HashMap::new();
        loop {
            let mut changed = false;
            for (name, seen) in &reads {
                let params = &self.macros[name].params;
                let mut c: BTreeSet<String> = seen
                    .iter()
                    .filter(|v| top.contains(*v) && !params.contains(v))
                    .cloned()
                    .collect();
                for callee in seen.iter().filter(|v| self.macros.contains_key(*v)) {
                    if let Some(cc) = captured.get(callee) {
                        c.extend(cc.iter().filter(|v| !params.contains(v)).cloned());
                    }
                }
                if captured.get(name) != Some(&c) {
                    captured.insert(name.clone(), c);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        for (name, c) in captured {
            self.macros.get_mut(&name).unwrap().captured = c.into_iter().collect();
        }
        Ok(())
    }

    fn gen_macro(&mut self, m: &ast::Macro<'_>) -> G<()> {
        let info = self.macros[m.name].clone();
        let mut sig = String::from("ctx: &Ctx");
        for p in &info.params {
            sig.push_str(&format!(", mut v_{p}: rt::Value"));
        }
        for c in &info.captured {
            sig.push_str(&format!(", g_{c}: rt::Value"));
        }
        self.line("");
        self.line(&format!("/// The `{}` macro", m.name));
        self.open(&format!("fn m_{}({sig}) -> rt::R {{", m.name));
        self.line("let mut out = rt::Str::default();");
        self.frames = vec![
            Frame::new("g_", info.captured.iter().cloned()),
            Frame::new("v_", info.params.iter().cloned()),
        ];
        let first_default = info.params.len() - m.defaults.len();
        for (i, d) in m.defaults.iter().enumerate() {
            let p = &info.params[first_default + i];
            let d = self.expr(d)?.val();
            self.line(&format!("if v_{p}.is_undefined() {{ v_{p} = {d}; }}"));
        }
        let locals: Vec<String> = assigned(&m.body)
            .into_iter()
            .filter(|n| !info.params.contains(n))
            .collect();
        self.declare(&locals);
        self.stmts(&m.body)?;
        self.line("Ok(rt::Value::Str(out))");
        self.close("}");
        Ok(())
    }

    /// Declare the variables a new scope assigns, from their values outside
    /// it, and add them to the innermost frame
    fn declare(&mut self, names: &[String]) {
        for n in names {
            if self.frames.last().unwrap().names.contains(n) {
                continue;
            }
            let outer = self.outer(n).val();
            self.line(&format!("let mut v_{n} = {outer};"));
        }
        let frame = self.frames.last_mut().unwrap();
        frame.names.extend(names.iter().cloned());
    }

    fn var(&mut self, name: &str) -> Code {
        match self.local(name) {
            Some(code) => code,
            None => {
                self.context.insert(name.to_owned());
                Code::Place(format!("ctx.{}", field(name)))
            }
        }
    }

    fn local(&self, name: &str) -> Option<Code> {
        self.frames
            .iter()
            .rev()
            .find(|f| f.names.contains(name))
            .map(|f| Code::Place(format!("{}{name}", f.prefix)))
    }

    /// The value a scope's own variable starts from: the variable outside the
    /// scope, or the context's, if the template reads it elsewhere (resolved
    /// at the end, see [`CTX_MARK`])
    fn outer(&self, name: &str) -> Code {
        self.local(name)
            .unwrap_or_else(|| Code::Value(format!("{CTX_MARK}{name}@@")))
    }

    fn in_loop(&self) -> bool {
        self.frames.iter().any(|f| f.is_loop)
    }

    fn stmts(&mut self, stmts: &[ast::Stmt<'_>]) -> G<()> {
        for s in stmts {
            self.stmt(s)?;
        }
        Ok(())
    }

    fn stmt(&mut self, s: &ast::Stmt<'_>) -> G<()> {
        match s {
            ast::Stmt::Template(t) => self.stmts(&t.children)?,
            ast::Stmt::EmitRaw(r) => self.line(&format!("out.push({}, true);", lit(r.raw))),
            ast::Stmt::EmitExpr(e) => match &e.expr {
                ast::Expr::Const(c) if c.value.kind() == ValueKind::String => {
                    let s = c.value.as_str().unwrap();
                    self.line(&format!("out.push({}, true);", lit(s)));
                }
                expr => {
                    let v = self.expr(expr)?.r();
                    self.line(&format!("rt::emit(&mut out, {v});"));
                }
            },
            ast::Stmt::IfCond(c) => {
                let cond = self.cond(&c.expr)?;
                self.open(&format!("if {cond} {{"));
                self.stmts(&c.true_body)?;
                // `elif`s: an `else` holding a single `if`
                let mut rest = &c.false_body;
                while let [ast::Stmt::IfCond(elif)] = rest.as_slice() {
                    let cond = self.cond(&elif.expr)?;
                    self.indent -= 1;
                    self.line(&format!("}} else if {cond} {{"));
                    self.indent += 1;
                    self.stmts(&elif.true_body)?;
                    rest = &elif.false_body;
                }
                if !rest.is_empty() {
                    self.indent -= 1;
                    self.line("} else {");
                    self.indent += 1;
                    self.stmts(rest)?;
                }
                self.close("}");
            }
            ast::Stmt::ForLoop(f) => self.for_loop(f, f.span())?,
            ast::Stmt::Set(set) => {
                let v = self.expr(&set.expr)?.val();
                self.assign(&set.target, v, set.span())?;
            }
            ast::Stmt::SetBlock(set) => {
                self.open("{");
                self.line("let mut out = rt::Str::default();");
                self.stmts(&set.body)?;
                let mut v = Code::Value("rt::Value::Str(out)".into());
                if let Some(f) = &set.filter {
                    v = self.filter_chain(f, v)?;
                }
                let v = v.val();
                self.assign(&set.target, v, set.span())?;
                self.close("}");
            }
            ast::Stmt::WithBlock(w) => {
                self.open("{");
                let mut names = Vec::new();
                let mut values = Vec::new();
                for (t, e) in &w.assignments {
                    let Some(ts) = target_list(t) else {
                        return fail(w.span(), "a `with` target that is not a name");
                    };
                    if ts.len() != 1 {
                        return fail(w.span(), "tuple targets in `with`");
                    }
                    values.push(self.expr(e)?.val());
                    names.push(ts[0].to_owned());
                }
                for (n, v) in names.iter().zip(values) {
                    self.line(&format!("let mut v_{n} = {v};"));
                }
                self.frames.push(Frame::new("v_", names.iter().cloned()));
                let locals = assigned(&w.body);
                self.declare(&locals);
                self.stmts(&w.body)?;
                self.frames.pop();
                self.close("}");
            }
            ast::Stmt::Macro(m) => {
                if self.frames.len() != 1 || self.frames[0].prefix != "g_" || self.indent != 1 {
                    return fail(m.span(), "a macro that is not at the template's top level");
                }
            }
            ast::Stmt::Do(d) => {
                let call = self.call(&d.call, d.span())?.val();
                self.line(&format!("let _ = {call};"));
            }
            ast::Stmt::Continue(c) => {
                if !self.in_loop() {
                    return fail(c.span(), "continue outside a loop");
                }
                self.line("continue;");
            }
            ast::Stmt::Break(b) => {
                if !self.in_loop() {
                    return fail(b.span(), "break outside a loop");
                }
                self.line("break;");
            }
            ast::Stmt::AutoEscape(a) => return fail(a.span(), "autoescape blocks"),
            ast::Stmt::FilterBlock(f) => return fail(f.span(), "filter blocks"),
            ast::Stmt::CallBlock(c) => return fail(c.span(), "call blocks"),
        }
        Ok(())
    }

    fn assign(&mut self, target: &ast::Expr<'_>, value: String, span: Span) -> G<()> {
        match target {
            ast::Expr::Var(v) => {
                let Code::Place(p) = self.var(v.id) else {
                    unreachable!("assigned variables are declared")
                };
                self.line(&format!("{p} = {value};"));
            }
            ast::Expr::GetAttr(g) => {
                let base = self.expr(&g.expr)?.r();
                self.line(&format!("rt::set_attr({base}, {}, {value})?;", lit(g.name)));
            }
            ast::Expr::Tuple(_) => {
                let Some(names) = target_list(target) else {
                    return fail(span, "a tuple target that is not names");
                };
                let t = self.tmp();
                self.line(&format!(
                    "let {t} = rt::unpack(&{value}, {})?;",
                    names.len()
                ));
                for (i, n) in names.iter().enumerate() {
                    let Code::Place(p) = self.var(n) else {
                        unreachable!("assigned variables are declared")
                    };
                    self.line(&format!("{p} = {t}[{i}].clone();"));
                }
            }
            _ => return fail(span, "an assignment to something that is not a name"),
        }
        Ok(())
    }

    fn for_loop(&mut self, f: &ast::ForLoop<'_>, span: Span) -> G<()> {
        if f.recursive {
            return fail(span, "recursive loops");
        }
        let Some(targets) = target_list(&f.target) else {
            return fail(span, "a loop target that is not names");
        };
        let targets: Vec<String> = targets.iter().map(|s| s.to_string()).collect();
        self.depth += 1;
        let d = self.depth;
        let (items, i, item) = (format!("items{d}"), format!("i{d}"), format!("item{d}"));
        let iter = self.expr(&f.iter)?.r();
        self.open("{");
        let bind = |g: &mut Gen, item: &str| {
            if targets.len() == 1 {
                g.line(&format!("let mut v_{} = {item}.clone();", targets[0]));
            } else {
                let t = g.tmp();
                g.line(&format!(
                    "let {t} = rt::unpack({item}, {})?;",
                    targets.len()
                ));
                for (k, n) in targets.iter().enumerate() {
                    g.line(&format!("let mut v_{n} = {t}[{k}].clone();"));
                }
            }
        };
        match &f.filter_expr {
            None => self.line(&format!("let {items} = rt::iter({iter})?;")),
            Some(cond) => {
                self.line(&format!("let mut {items} = Vec::new();"));
                self.open(&format!("for {item} in rt::iter({iter})? {{"));
                self.line(&format!("let {item} = &{item};"));
                bind(self, &item);
                self.frames.push(Frame::new("v_", targets.iter().cloned()));
                let c = self.cond(cond)?;
                self.frames.pop();
                self.line(&format!("if {c} {{ {items}.push({item}.clone()); }}"));
                self.close("}");
            }
        }
        self.open(&format!(
            "for ({i}, {item}) in {items}.iter().enumerate() {{"
        ));
        let mut seen = HashSet::new();
        mentioned(&f.body, &mut seen);
        let mut names = targets.clone();
        if seen.contains("loop") {
            self.line(&format!("let v_loop = rt::loop_info(&{items}, {i});"));
            names.push("loop".into());
        }
        bind(self, &item);
        let mut frame = Frame::new("v_", names);
        frame.is_loop = true;
        // the body's variables start from their values outside the loop
        let locals: Vec<String> = assigned(&f.body)
            .into_iter()
            .filter(|n| !frame.names.contains(n))
            .collect();
        for n in &locals {
            let outer = self.outer(n).val();
            self.line(&format!("let mut v_{n} = {outer};"));
        }
        frame.names.extend(locals);
        self.frames.push(frame);
        self.stmts(&f.body)?;
        self.frames.pop();
        self.close("}");
        if !f.else_body.is_empty() {
            self.open(&format!("if {items}.is_empty() {{"));
            self.frames.push(Frame::new("v_", []));
            let locals = assigned(&f.else_body);
            self.declare(&locals);
            self.stmts(&f.else_body)?;
            self.frames.pop();
            self.close("}");
        }
        self.close("}");
        self.depth -= 1;
        Ok(())
    }

    /// Rust code for a condition: a `bool`
    fn cond(&mut self, e: &ast::Expr<'_>) -> G<String> {
        Ok(match e {
            ast::Expr::Const(c) if c.value.kind() == ValueKind::Bool => {
                format!("{}", c.value.is_true())
            }
            ast::Expr::UnaryOp(u) if matches!(u.op, ast::UnaryOpKind::Not) => {
                format!("!{}", self.cond_atom(&u.expr)?)
            }
            ast::Expr::BinOp(b) if matches!(b.op, ast::BinOpKind::ScAnd) => {
                format!(
                    "{} && {}",
                    self.cond_atom(&b.left)?,
                    self.cond_atom(&b.right)?
                )
            }
            ast::Expr::BinOp(b) if matches!(b.op, ast::BinOpKind::ScOr) => {
                format!(
                    "{} || {}",
                    self.cond_atom(&b.left)?,
                    self.cond_atom(&b.right)?
                )
            }
            ast::Expr::BinOp(b) => match self.compare_op(&b.op) {
                Some(op) => {
                    let (l, r) = (self.expr(&b.left)?.r(), self.expr(&b.right)?.r());
                    compare(op, &l, &r)
                }
                None => format!("rt::truthy({})", self.expr(e)?.r()),
            },
            ast::Expr::Compare(c) => self.chain(c)?,
            ast::Expr::Test(t) => self.test(t, t.span())?,
            _ => format!("rt::truthy({})", self.expr(e)?.r()),
        })
    }

    /// A condition that can be an operand of `!`, `&&` and `||`
    fn cond_atom(&mut self, e: &ast::Expr<'_>) -> G<String> {
        let c = self.cond(e)?;
        let simple = c.starts_with("rt::") && !c.contains(" && ") && !c.contains(" || ")
            || c.starts_with('!') && !c.contains(' ')
            || c == "true"
            || c == "false";
        Ok(if simple { c } else { format!("({c})") })
    }

    fn compare_op(&self, op: &ast::BinOpKind) -> Option<&'static str> {
        Some(match op {
            ast::BinOpKind::Eq => "==",
            ast::BinOpKind::Ne => "!=",
            ast::BinOpKind::Lt => "<",
            ast::BinOpKind::Lte => "<=",
            ast::BinOpKind::Gt => ">",
            ast::BinOpKind::Gte => ">=",
            ast::BinOpKind::In => "in",
            _ => return None,
        })
    }

    /// `a < b < c`, Python's chained comparison
    fn chain(&mut self, c: &ast::Compare<'_>) -> G<String> {
        let first = self.tmp();
        let mut code = format!("{{ let {first} = {}; ", self.expr(&c.expr)?.val());
        let mut prev = first;
        let mut closes = 1;
        for (k, op) in c.ops.iter().enumerate() {
            let t = self.tmp();
            code.push_str(&format!("let {t} = {}; ", self.expr(&op.expr)?.val()));
            let name = match op.op {
                ast::CompareOpKind::Eq => "==",
                ast::CompareOpKind::Ne => "!=",
                ast::CompareOpKind::Lt => "<",
                ast::CompareOpKind::Lte => "<=",
                ast::CompareOpKind::Gt => ">",
                ast::CompareOpKind::Gte => ">=",
                ast::CompareOpKind::In => "in",
                ast::CompareOpKind::NotIn => "not in",
            };
            code.push_str(&compare(name, &format!("&{prev}"), &format!("&{t}")));
            if k + 1 < c.ops.len() {
                code.push_str(" && { ");
                closes += 1;
            }
            prev = t;
        }
        for _ in 0..closes {
            code.push_str(" }");
        }
        Ok(code)
    }

    fn test(&mut self, t: &ast::Test<'_>, span: Span) -> G<String> {
        let v = self.expr(&t.expr)?.r();
        let simple = |name: &str| format!("rt::is_{name}({v})");
        Ok(match t.name {
            "defined" | "undefined" | "none" | "string" | "mapping" | "iterable" | "sequence"
            | "true" | "false" | "boolean" | "number" | "integer" | "float"
                if t.args.is_empty() =>
            {
                simple(t.name)
            }
            "eq" | "equalto" | "==" | "ne" | "!=" | "in" if t.args.len() == 1 => {
                let ast::CallArg::Pos(a) = &t.args[0] else {
                    return fail(span, format!("a keyword argument to the {} test", t.name));
                };
                let a = self.expr(a)?.r();
                let op = match t.name {
                    "ne" | "!=" => "!=",
                    "in" => "in",
                    _ => "==",
                };
                compare(op, &v, &a)
            }
            name => return fail(span, format!("the {name} test")),
        })
    }

    fn expr(&mut self, e: &ast::Expr<'_>) -> G<Code> {
        Ok(match e {
            ast::Expr::Var(v) => self.var(v.id),
            ast::Expr::Const(c) => Code::Value(self.constant(&c.value, c.span())?),
            ast::Expr::Slice(s) => {
                let v = self.expr(&s.expr)?.r();
                let mut parts = Vec::new();
                for p in [&s.start, &s.stop, &s.step] {
                    parts.push(match p {
                        Some(p) => self.expr(p)?.r(),
                        None => "&rt::Value::Undefined".into(),
                    });
                }
                Code::Value(format!("rt::slice({v}, {})?", parts.join(", ")))
            }
            ast::Expr::UnaryOp(u) => match u.op {
                ast::UnaryOpKind::Not => Code::Value(format!("rt::Value::Bool({})", self.cond(e)?)),
                ast::UnaryOpKind::Neg => {
                    Code::Value(format!("rt::neg({})?", self.expr(&u.expr)?.r()))
                }
            },
            ast::Expr::BinOp(b) => {
                if self.compare_op(&b.op).is_some() {
                    return Ok(Code::Value(format!("rt::Value::Bool({})", self.cond(e)?)));
                }
                match b.op {
                    ast::BinOpKind::ScAnd | ast::BinOpKind::ScOr => {
                        // Python's: the operand that decides, not a bool
                        let t = self.tmp();
                        let l = self.expr(&b.left)?.val();
                        let r = self.expr(&b.right)?.val();
                        let (yes, no) = if matches!(b.op, ast::BinOpKind::ScAnd) {
                            (r, t.clone())
                        } else {
                            (t.clone(), r)
                        };
                        Code::Value(format!(
                            "{{ let {t} = {l}; if rt::truthy(&{t}) {{ {yes} }} else {{ {no} }} }}"
                        ))
                    }
                    ast::BinOpKind::Add => {
                        let (l, r) = (self.expr(&b.left)?.r(), self.expr(&b.right)?.r());
                        Code::Value(format!("rt::add({l}, {r})?"))
                    }
                    ast::BinOpKind::Concat => {
                        let (l, r) = (self.expr(&b.left)?.r(), self.expr(&b.right)?.r());
                        Code::Value(format!("rt::concat({l}, {r})"))
                    }
                    op => {
                        let name = match op {
                            ast::BinOpKind::Sub => "-",
                            ast::BinOpKind::Mul => "*",
                            ast::BinOpKind::Div => "/",
                            ast::BinOpKind::FloorDiv => "//",
                            ast::BinOpKind::Rem => "%",
                            ast::BinOpKind::Pow => "**",
                            _ => unreachable!("handled above"),
                        };
                        let (l, r) = (self.expr(&b.left)?.r(), self.expr(&b.right)?.r());
                        Code::Value(format!("rt::arith({}, {l}, {r})?", lit(name)))
                    }
                }
            }
            ast::Expr::Compare(_) | ast::Expr::Test(_) => {
                Code::Value(format!("rt::Value::Bool({})", self.cond(e)?))
            }
            ast::Expr::IfExpr(i) => {
                let c = self.cond(&i.test_expr)?;
                let yes = self.expr(&i.true_expr)?.val();
                let no = match &i.false_expr {
                    Some(e) => self.expr(e)?.val(),
                    None => "rt::Value::Undefined".into(),
                };
                Code::Value(format!("if {c} {{ {yes} }} else {{ {no} }}"))
            }
            ast::Expr::Filter(f) => {
                let Some(inner) = &f.expr else {
                    return fail(f.span(), "a filter without a value");
                };
                let v = self.expr(inner)?;
                self.filter(f, v, f.span())?
            }
            ast::Expr::GetAttr(g) => {
                let v = self.expr(&g.expr)?.r();
                Code::Value(format!("rt::getattr({v}, {})", lit(g.name)))
            }
            ast::Expr::GetItem(g) => {
                let v = self.expr(&g.expr)?.r();
                let i = self.expr(&g.subscript_expr)?.r();
                Code::Value(format!("rt::getitem({v}, {i})?"))
            }
            ast::Expr::Call(c) => self.call(c, c.span())?,
            ast::Expr::List(l) => Code::Value(self.list(&l.items)?),
            ast::Expr::Tuple(t) => Code::Value(self.list(&t.items)?),
            ast::Expr::Map(m) => {
                let mut entries = Vec::new();
                for (k, v) in m.keys.iter().zip(&m.values) {
                    let ast::Expr::Const(k) = k else {
                        return fail(m.span(), "a mapping key that is not a constant");
                    };
                    let Some(k) = k.value.as_str() else {
                        return fail(m.span(), "a mapping key that is not a string");
                    };
                    entries.push(format!("({}, {})", lit(k), self.expr(v)?.val()));
                }
                Code::Value(format!("rt::Value::map(vec![{}])", entries.join(", ")))
            }
        })
    }

    fn list(&mut self, items: &[ast::Expr<'_>]) -> G<String> {
        let mut out = Vec::new();
        for i in items {
            out.push(self.expr(i)?.val());
        }
        Ok(format!("rt::Value::list(vec![{}])", out.join(", ")))
    }

    fn constant(&mut self, v: &JValue, span: Span) -> G<String> {
        Ok(match v.kind() {
            ValueKind::Undefined => "rt::Value::Undefined".into(),
            ValueKind::None => "rt::Value::None".into(),
            ValueKind::Bool => format!("rt::Value::Bool({})", v.is_true()),
            ValueKind::Number if v.is_integer() => {
                format!("rt::Value::Int({})", v.as_i64().unwrap())
            }
            ValueKind::Number => {
                let f = f64::try_from(v.clone()).unwrap();
                format!("rt::Value::Float({f:?})")
            }
            ValueKind::String => format!("rt::Value::lit({})", lit(v.as_str().unwrap())),
            ValueKind::Seq => {
                let mut items = Vec::new();
                for x in v.try_iter().unwrap() {
                    items.push(self.constant(&x, span)?);
                }
                format!("rt::Value::list(vec![{}])", items.join(", "))
            }
            ValueKind::Map => {
                let mut entries = Vec::new();
                for k in v.try_iter().unwrap() {
                    let Some(ks) = k.as_str() else {
                        return fail(span, "a mapping key that is not a string");
                    };
                    let x = v.get_item(&k).unwrap();
                    entries.push(format!("({}, {})", lit(ks), self.constant(&x, span)?));
                }
                format!("rt::Value::map(vec![{}])", entries.join(", "))
            }
            kind => return fail(span, format!("a constant of kind {kind:?}")),
        })
    }

    /// The filters of a `set` block's filter expression, applied to `v`
    fn filter_chain(&mut self, f: &ast::Expr<'_>, v: Code) -> G<Code> {
        let ast::Expr::Filter(f) = f else {
            return fail(f.span(), "a set block filter that is not a filter");
        };
        let v = match &f.expr {
            Some(inner) => self.filter_chain(inner, v)?,
            None => v,
        };
        self.filter(f, v, f.span())
    }

    fn arg(&mut self, a: Option<&ast::Expr<'_>>) -> G<String> {
        Ok(match a {
            Some(e) => self.expr(e)?.r(),
            None => "&rt::Value::Undefined".into(),
        })
    }

    fn filter(&mut self, f: &ast::Filter<'_>, v: Code, span: Span) -> G<Code> {
        let what = format!("the {} filter", f.name);
        let args = |params: &[&str]| bind_args(span, &what, params, &f.args);
        Ok(Code::Value(match f.name {
            "safe" => {
                args(&[])?;
                return Ok(v);
            }
            "trim" => {
                let a = args(&["chars"])?;
                format!("rt::f_trim({}, {})?", v.r(), self.arg(a[0])?)
            }
            "length" | "count" => {
                args(&[])?;
                format!("rt::f_length({})?", v.r())
            }
            "string" => {
                args(&[])?;
                format!("rt::f_string({})", v.r())
            }
            "items" | "first" | "last" | "list" => {
                args(&[])?;
                format!("rt::f_{}({})?", f.name, v.r())
            }
            "upper" | "lower" => {
                args(&[])?;
                format!("rt::f_{}({})", f.name, v.r())
            }
            "default" | "d" => {
                let a = args(&["default_value", "boolean"])?;
                let d = match a[0] {
                    Some(e) => self.expr(e)?.r(),
                    None => "&rt::Value::lit(\"\")".into(),
                };
                format!("rt::f_default({}, {d}, {})", v.r(), self.arg(a[1])?)
            }
            "replace" => {
                let a = args(&["old", "new", "count"])?;
                if a[2].is_some() {
                    return fail(span, "the replace filter's count");
                }
                format!(
                    "rt::f_replace({}, {}, {})?",
                    v.r(),
                    self.arg(a[0])?,
                    self.arg(a[1])?
                )
            }
            "join" => {
                let a = args(&["d", "attribute"])?;
                if a[1].is_some() {
                    return fail(span, "the join filter's attribute");
                }
                let sep = match a[0] {
                    Some(e) => self.expr(e)?.r(),
                    None => "&rt::Value::lit(\"\")".into(),
                };
                format!("rt::f_join({}, {sep})?", v.r())
            }
            "tojson" => {
                // transformers' own: (x, ensure_ascii=False, indent=None,
                // separators=None, sort_keys=False)
                let a = args(&["ensure_ascii", "indent", "separators", "sort_keys"])?;
                // separators: none, or a constant pair of strings
                let separators = match a[2].map(|s| s.as_const()) {
                    None => None,
                    Some(Some(c)) if c.is_none() => None,
                    Some(Some(c))
                        if matches!(c.kind(), ValueKind::Seq)
                            && c.len() == Some(2)
                            && c.try_iter()
                                .is_ok_and(|mut i| i.all(|x| x.kind() == ValueKind::String)) =>
                    {
                        let mut i = c.try_iter().expect("a sequence");
                        let mut next = || i.next().expect("two items").as_str().unwrap().to_owned();
                        Some((next(), next()))
                    }
                    Some(_) => {
                        return fail(span, "the tojson filter's separators, but for two strings");
                    }
                };
                match separators {
                    None => format!(
                        "rt::f_tojson({}, {}, {}, {})?",
                        v.r(),
                        self.arg(a[0])?,
                        self.arg(a[1])?,
                        self.arg(a[3])?
                    ),
                    Some((item, key)) => format!(
                        "rt::f_tojson_sep({}, {}, {}, {}, {}, {})?",
                        v.r(),
                        self.arg(a[0])?,
                        self.arg(a[1])?,
                        self.arg(a[3])?,
                        lit(&item),
                        lit(&key)
                    ),
                }
            }
            "dictsort" => {
                let a = args(&["case_sensitive", "by", "reverse"])?;
                if a[1].is_some() || a[2].is_some() {
                    return fail(span, "the dictsort filter's by and reverse");
                }
                format!("rt::f_dictsort({}, {})?", v.r(), self.arg(a[0])?)
            }
            "float" => {
                let a = args(&["default"])?;
                format!("rt::f_float({}, {})?", v.r(), self.arg(a[0])?)
            }
            name => return fail(span, format!("the {name} filter")),
        }))
    }

    fn call(&mut self, c: &ast::Call<'_>, span: Span) -> G<Code> {
        match &c.expr {
            ast::Expr::Var(v) if self.macros.contains_key(v.id) => {
                let info = self.macros[v.id].clone();
                let params: Vec<&str> = info.params.iter().map(|s| s.as_str()).collect();
                let bound = bind_args(span, &format!("the {} macro", v.id), &params, &c.args)?;
                let mut args = vec!["ctx".to_owned()];
                for a in bound {
                    args.push(match a {
                        Some(e) => self.expr(e)?.val(),
                        None => "rt::Value::Undefined".into(),
                    });
                }
                for name in &info.captured {
                    args.push(format!("g_{name}.clone()"));
                }
                Ok(Code::Value(format!("m_{}({})?", v.id, args.join(", "))))
            }
            ast::Expr::Var(v) if v.id == "namespace" => {
                let mut kwargs = Vec::new();
                for a in &c.args {
                    let ast::CallArg::Kwarg(k, e) = a else {
                        return fail(span, "namespace() with other than keyword arguments");
                    };
                    kwargs.push(format!("({}, {})", lit(k), self.expr(e)?.val()));
                }
                Ok(Code::Value(format!(
                    "rt::namespace(vec![{}])",
                    kwargs.join(", ")
                )))
            }
            ast::Expr::Var(v) if v.id == "raise_exception" => {
                let a = bind_args(span, "raise_exception", &["message"], &c.args)?;
                let msg = self.arg(a[0])?;
                Ok(Code::Value(format!("rt::raise_exception({msg})?")))
            }
            ast::Expr::Var(v) if v.id == "range" => {
                let mut args = Vec::new();
                for a in &c.args {
                    let ast::CallArg::Pos(e) = a else {
                        return fail(span, "range() with keyword arguments");
                    };
                    args.push(self.expr(e)?.val());
                }
                Ok(Code::Value(format!("rt::range(&[{}])?", args.join(", "))))
            }
            ast::Expr::GetAttr(g) => {
                let obj = self.expr(&g.expr)?.r();
                let mut args = Vec::new();
                for a in &c.args {
                    let ast::CallArg::Pos(e) = a else {
                        return fail(span, format!("keyword arguments to the {} method", g.name));
                    };
                    args.push(self.expr(e)?.val());
                }
                Ok(Code::Value(format!(
                    "rt::call_method({obj}, {}, &[{}])?",
                    lit(g.name),
                    args.join(", ")
                )))
            }
            ast::Expr::Var(v) => fail(span, format!("the {} function", v.id)),
            _ => fail(span, "a call of something that is not a name or a method"),
        }
    }
}

/// A comparison of two `&rt::Value`s, as a `bool`
fn compare(op: &str, l: &str, r: &str) -> String {
    match op {
        "==" => format!("rt::eq({l}, {r})"),
        "!=" => format!("!rt::eq({l}, {r})"),
        "in" => format!("rt::contains({r}, {l})?"),
        "not in" => format!("!rt::contains({r}, {l})?"),
        op => format!("rt::cmp({}, {l}, {r})?", lit(op)),
    }
}

/// The runtime the generated code runs on, copied next to it
pub const RUNTIME: &str = include_str!("../rt/rt.rs");

/// A template of a registry: its name, a Rust module name, and its text
pub struct Source<'a> {
    pub name: &'a str,
    pub text: &'a str,
}

/// The files of a registry of templates, by file name: each template's
/// module, the runtime (`rt.rs`), and `mod.rs`, which recognizes a template
/// by its hash and renders it from parameters common to all of them
pub fn registry(sources: &[Source<'_>]) -> Result<Vec<(String, String)>, GenError> {
    let mut files = Vec::new();
    let mut templates = Vec::new();
    for s in sources {
        let ok = s
            .name
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_lowercase())
            && s.name
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
        if !ok || ["rt", "mod"].contains(&s.name) {
            return Err(GenError {
                line: 0,
                msg: format!("{:?} is not a name for a template module", s.name),
            });
        }
        let g = generate(
            s.text,
            &Options {
                name: s.name,
                rt_path: "super::rt",
            },
        )
        .map_err(|e| GenError {
            line: e.line,
            msg: format!("{}: {}", s.name, e.msg),
        })?;
        files.push((format!("{}.rs", s.name), g.code.clone()));
        templates.push((s.name, g));
    }
    files.push(("rt.rs".into(), RUNTIME.into()));
    files.push(("mod.rs".into(), router(&templates)?));
    Ok(files)
}

fn variant(name: &str) -> String {
    name.split('_')
        .map(|w| {
            let mut c = w.chars();
            c.next()
                .map(|f| f.to_ascii_uppercase().to_string() + c.as_str())
                .unwrap_or_default()
        })
        .collect()
}

/// `mod.rs` of a registry
fn router(templates: &[(&str, Generated)]) -> Result<String, GenError> {
    use params::Kind;
    // the common parameters: every template's, a flag optional where one
    // template tells undefined from false
    let mut common: Vec<(String, Kind, Vec<&str>)> = Vec::new();
    for (name, g) in templates {
        for p in &g.params {
            match common.iter_mut().find(|c| c.0 == p.name) {
                Some(c) => {
                    c.1 = match (c.1, p.kind) {
                        (a, b) if a == b => a,
                        (Kind::Bool, Kind::OptBool) | (Kind::OptBool, Kind::Bool) => Kind::OptBool,
                        (a, b) => {
                            return Err(GenError {
                                line: 0,
                                msg: format!(
                                    "`{}` is {} in one template and {} in {name}",
                                    p.name,
                                    a.rust_type(),
                                    b.rust_type()
                                ),
                            });
                        }
                    };
                    c.2.push(name);
                }
                None => common.push((p.name.clone(), p.kind, vec![name])),
            }
        }
    }
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
    common.sort_by(|a, b| (rank(&a.0), &a.0).cmp(&(rank(&b.0), &b.0)));

    let mut c = String::new();
    c.push_str(
        "//! The chat templates compiled by jinja2rust: do not edit.\n\
         //!\n\
         //! [`Template::recognize`] finds a model's chat template by the hash of\n\
         //! its text, and [`Template::render`] renders it from [`Params`], the\n\
         //! parameters of all the templates: each reads those it has.\n\n",
    );
    c.push_str("pub mod rt;\n\n");
    for (name, _) in templates {
        c.push_str(&format!("pub mod {name};\n"));
    }
    c.push_str("\n#[derive(Clone, Copy, Debug, PartialEq, Eq)]\npub enum Template {\n");
    for (name, _) in templates {
        c.push_str(&format!("    {},\n", variant(name)));
    }
    c.push_str("}\n\n");

    c.push_str("/// The parameters of the templates: each reads those it has\n");
    c.push_str("#[derive(Clone, Copy, Debug, Default)]\npub struct Params<'a> {\n");
    for (name, kind, readers) in &common {
        let what = match kind {
            Kind::Messages => "the conversation".to_owned(),
            Kind::Tools => "the functions the model may call".to_owned(),
            _ => format!("`{name}`"),
        };
        let note = match kind {
            Kind::OptBool | Kind::OptStr => ": `None` leaves it undefined",
            Kind::Token => ": the special token",
            _ => "",
        };
        let who = if readers.len() == templates.len() {
            "all".to_owned()
        } else {
            readers.join(", ")
        };
        c.push_str(&format!("    /// {what}{note} ({who})\n"));
        c.push_str(&format!("    pub {}: {},\n", field(name), kind.rust_type()));
    }
    c.push_str("}\n\n");

    c.push_str("impl Template {\n");
    c.push_str(&format!(
        "    pub const ALL: [Template; {}] = [\n",
        templates.len()
    ));
    for (name, _) in templates {
        c.push_str(&format!("        Template::{},\n", variant(name)));
    }
    c.push_str("    ];\n\n");

    c.push_str("    pub fn name(self) -> &'static str {\n        match self {\n");
    for (name, _) in templates {
        c.push_str(&format!(
            "            Template::{} => {name:?},\n",
            variant(name)
        ));
    }
    c.push_str("        }\n    }\n\n");

    c.push_str("    /// FNV-1a of the template's text\n");
    c.push_str("    pub fn hash(self) -> u64 {\n        match self {\n");
    for (name, _) in templates {
        c.push_str(&format!(
            "            Template::{} => {name}::TEMPLATE_HASH,\n",
            variant(name)
        ));
    }
    c.push_str("        }\n    }\n\n");

    c.push_str("    /// The compiled template whose text is `source`, if any\n");
    c.push_str("    pub fn recognize(source: &str) -> Option<Template> {\n");
    c.push_str("        let hash = rt::fnv1a(source);\n");
    c.push_str("        Self::ALL.into_iter().find(|t| t.hash() == hash)\n    }\n\n");

    c.push_str("    /// The parameters the template reads, the others of [`Params`] ignored\n");
    c.push_str("    pub fn params(self) -> &'static [&'static str] {\n        match self {\n");
    for (name, g) in templates {
        let names: Vec<String> = g.params.iter().map(|p| format!("{:?}", p.name)).collect();
        c.push_str(&format!(
            "            Template::{} => &[{}],\n",
            variant(name),
            names.join(", ")
        ));
    }
    c.push_str("        }\n    }\n\n");

    c.push_str(
        "    pub fn render(self, p: &Params<'_>) -> rt::R<rt::Str> {\n        match self {\n",
    );
    for (name, g) in templates {
        c.push_str(&format!(
            "            Template::{} => {name}::render(&{name}::Params {{\n",
            variant(name)
        ));
        for p in &g.params {
            let f = field(&p.name);
            let generic = common.iter().find(|c| c.0 == p.name).unwrap().1;
            let value = match (generic, p.kind) {
                (Kind::OptBool, Kind::Bool) => format!("p.{f}.unwrap_or(false)"),
                _ => format!("p.{f}"),
            };
            c.push_str(&format!("                {f}: {value},\n"));
        }
        c.push_str("            }),\n");
    }
    c.push_str("        }\n    }\n}\n");
    Ok(c)
}

/// `code` as rustfmt formats it, for generated files that `cargo fmt` leaves
/// as they are
pub fn rustfmt(code: &str) -> std::io::Result<String> {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let mut child = Command::new("rustfmt")
        .args(["--edition", "2024", "--emit", "stdout"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    child.stdin.take().unwrap().write_all(code.as_bytes())?;
    let out = child.wait_with_output()?;
    if !out.status.success() {
        return Err(std::io::Error::other(
            String::from_utf8_lossy(&out.stderr).into_owned(),
        ));
    }
    String::from_utf8(out.stdout).map_err(std::io::Error::other)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_tags_render_their_body() {
        for (from, to) in [
            (
                "{%- generation -%}{{- x -}}{%- endgeneration -%}",
                "{%- if true -%}{{- x -}}{%- endif -%}",
            ),
            (
                "a{% generation %}b{%+  endgeneration\t+%}",
                "a{% if true %}b{%+  endif\t+%}",
            ),
            // other tags and text are left as they are
            (
                "{% set generation = 1 %}{{ 'generation' }}{% generations %}",
                "{% set generation = 1 %}{{ 'generation' }}{% generations %}",
            ),
        ] {
            assert_eq!(rewrite_generation(from), to);
        }
        // the hash is the original text's, which models have
        let source = "{%- generation -%}{{- messages -}}{%- endgeneration -%}";
        let opts = Options {
            name: "t",
            rt_path: "rt",
        };
        assert_eq!(generate(source, &opts).unwrap().hash, fnv1a(source));
    }
}
