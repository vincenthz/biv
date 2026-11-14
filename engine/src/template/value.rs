//! The values of a template render: the Python values that a chat
//! template sees, and strings that keep the origin of each byte.

use alloc::rc::Rc;
use core::cell::RefCell;
use core::fmt::Write;

use crate::prelude::*;

/// Where the bytes of a rendered prompt come from. The tokenizer finds the
/// special tokens only in the text of the template; the text of a message
/// or a tool is plain text, also when it holds the text of a special token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// a literal of the template, or a value that the caller marks as one
    Template,
    /// the text of a message, a tool or another value of the caller
    Content,
}

/// A run of rendered text of one origin
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub text: String,
    pub origin: Origin,
}

/// A string with the origin of each byte, as runs: each run ends at its
/// byte offset, and the runs cover the whole text in order
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OStr {
    text: String,
    runs: Vec<(usize, Origin)>,
    /// a Markup string of the reference (the safe filter): a + with a plain
    /// string escapes the plain one for HTML
    markup: bool,
}

impl OStr {
    pub fn new(text: &str, origin: Origin) -> Self {
        let mut s = OStr::default();
        s.push(text, origin);
        s
    }

    pub fn as_str(&self) -> &str {
        &self.text
    }

    pub fn is_markup(&self) -> bool {
        self.markup
    }

    /// This string as a plain string: the ~ operator of the reference
    /// joins with str, so its result is no Markup
    pub fn into_plain(mut self) -> OStr {
        self.markup = false;
        self
    }

    /// This string as a Markup string
    pub fn into_markup(mut self) -> OStr {
        self.markup = true;
        self
    }

    /// The HTML escape of markupsafe, with the origin of each char
    pub fn html_escaped(&self) -> OStr {
        self.map_chars(|c, out| match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&#34;"),
            '\'' => out.push_str("&#39;"),
            c => out.push(c),
        })
    }

    pub fn len(&self) -> usize {
        self.text.len()
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    pub fn push(&mut self, text: &str, origin: Origin) {
        if text.is_empty() {
            return;
        }
        self.text.push_str(text);
        let end = self.text.len();
        match self.runs.last_mut() {
            Some((last, o)) if *o == origin => *last = end,
            _ => self.runs.push((end, origin)),
        }
    }

    pub fn push_ostr(&mut self, other: &OStr) {
        let mut start = 0;
        for &(end, origin) in &other.runs {
            self.push(&other.text[start..end], origin);
            start = end;
        }
    }

    /// The bytes `start..end`, which are on char boundaries, with their
    /// origins
    pub fn slice(&self, start: usize, end: usize) -> OStr {
        let mut out = OStr::default();
        let mut run_start = 0;
        for &(run_end, origin) in &self.runs {
            let (a, b) = (run_start.max(start), run_end.min(end));
            if a < b {
                out.push(&self.text[a..b], origin);
            }
            run_start = run_end;
        }
        out
    }

    /// The origin of the byte at `at`
    pub fn origin_at(&self, at: usize) -> Origin {
        self.runs
            .iter()
            .find(|(end, _)| at < *end)
            .map(|&(_, o)| o)
            .unwrap_or(Origin::Template)
    }

    /// Each char mapped by `f`, with the origin of the char it comes from
    pub fn map_chars(&self, f: impl Fn(char, &mut String)) -> OStr {
        let mut out = OStr::default();
        let mut buf = String::new();
        for (at, c) in self.text.char_indices() {
            buf.clear();
            f(c, &mut buf);
            out.push(&buf, self.origin_at(at));
        }
        out
    }

    /// The runs as segments
    pub fn segments(&self) -> impl Iterator<Item = Segment> + '_ {
        let mut start = 0;
        self.runs.iter().map(move |&(end, origin)| {
            let s = Segment {
                text: self.text[start..end].into(),
                origin,
            };
            start = end;
            s
        })
    }
}

/// An undefined value: a name or an attribute that has no value
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Undefined {
    /// the name or the attribute
    pub name: String,
    /// a top-level name of the render, not an attribute or an item
    pub top: bool,
    /// the line of the read
    pub line: u32,
}

/// A user macro of the template
#[derive(Debug)]
pub struct MacroRef {
    pub index: usize,
}

/// A value of the render: the Python values of the reference renderer
#[derive(Debug, Clone)]
pub enum Value {
    Undefined(Rc<Undefined>),
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(Rc<OStr>),
    List(Rc<Vec<Value>>),
    /// a tuple: a list that prints with parentheses
    Tuple(Rc<Vec<Value>>),
    /// a dict, in the order of insertion
    Map(Rc<Vec<(Value, Value)>>),
    /// a namespace(), whose attributes a set can change
    Namespace(Rc<RefCell<Vec<(String, Value)>>>),
    /// a macro of the template
    Macro(Rc<MacroRef>),
    /// a global function of the renderer
    Function(Function),
    /// a method of a value, not yet called
    Method(Rc<(Value, String)>),
    /// a generator of the reference, as map, select and items give it
    Generator(Rc<Generator>),
    /// the loop object of a for loop
    Loop(Rc<LoopState>),
}

/// A generator of the reference: Python iterates it once, so a second
/// iteration fails by name in biv, where Python gives no items
#[derive(Debug)]
pub struct Generator {
    pub items: Vec<Value>,
    pub used: core::cell::Cell<bool>,
}

/// The global functions of the renderer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Function {
    Range,
    Namespace,
    RaiseException,
    StrftimeNow,
}

/// The state of a for loop that its loop object shows
#[derive(Debug)]
pub struct LoopState {
    pub index0: usize,
    pub length: usize,
    pub previtem: Option<Value>,
    pub nextitem: Option<Value>,
}

impl Value {
    pub fn str(text: &str, origin: Origin) -> Value {
        Value::Str(Rc::new(OStr::new(text, origin)))
    }

    pub fn ostr(s: OStr) -> Value {
        Value::Str(Rc::new(s))
    }

    pub fn list(items: Vec<Value>) -> Value {
        Value::List(Rc::new(items))
    }

    pub fn map(items: Vec<(Value, Value)>) -> Value {
        Value::Map(Rc::new(items))
    }

    /// The value at `key` of a map
    pub fn map_get(items: &[(Value, Value)], key: &Value) -> Option<Value> {
        items
            .iter()
            .find(|(k, _)| py_eq(k, key))
            .map(|(_, v)| v.clone())
    }

    /// The value at the string key `key` of a map
    pub fn map_get_str(items: &[(Value, Value)], key: &str) -> Option<Value> {
        items
            .iter()
            .find(|(k, _)| matches!(k, Value::Str(s) if s.as_str() == key))
            .map(|(_, v)| v.clone())
    }

    /// The truth value of Python
    pub fn truthy(&self) -> bool {
        match self {
            Value::Undefined(_) | Value::None => false,
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::Float(f) => *f != 0.0,
            Value::Str(s) => !s.is_empty(),
            Value::List(l) | Value::Tuple(l) => !l.is_empty(),
            Value::Map(m) => !m.is_empty(),
            Value::Namespace(_)
            | Value::Macro(_)
            | Value::Function(_)
            | Value::Method(_)
            | Value::Loop(_)
            | Value::Generator(_) => true,
        }
    }

    /// The Python type name, as TypeError messages give it
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Undefined(_) => "Undefined",
            Value::None => "NoneType",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::Str(_) => "str",
            Value::List(_) => "list",
            Value::Tuple(_) => "tuple",
            Value::Map(_) => "dict",
            Value::Namespace(_) => "Namespace",
            Value::Macro(_) => "Macro",
            Value::Function(_) => "function",
            Value::Method(_) => "method",
            Value::Loop(_) => "LoopContext",
            Value::Generator(_) => "generator",
        }
    }

    /// The number of a value that Python takes as a number: an int, a float
    /// or a bool
    pub fn number(&self) -> Option<Number> {
        match self {
            Value::Bool(b) => Some(Number::Int(i64::from(*b))),
            Value::Int(i) => Some(Number::Int(*i)),
            Value::Float(f) => Some(Number::Float(*f)),
            _ => None,
        }
    }
}

/// A Python number
#[derive(Debug, Clone, Copy)]
pub enum Number {
    Int(i64),
    Float(f64),
}

impl Number {
    pub fn as_f64(self) -> f64 {
        match self {
            Number::Int(i) => i as f64,
            Number::Float(f) => f,
        }
    }
}

/// The == of Python
pub fn py_eq(a: &Value, b: &Value) -> bool {
    if let (Some(x), Some(y)) = (a.number(), b.number()) {
        return match (x, y) {
            (Number::Int(x), Number::Int(y)) => x == y,
            _ => x.as_f64() == y.as_f64(),
        };
    }
    match (a, b) {
        (Value::Undefined(_), Value::Undefined(_)) | (Value::None, Value::None) => true,
        (Value::Str(x), Value::Str(y)) => x.as_str() == y.as_str(),
        (Value::List(x), Value::List(y)) | (Value::Tuple(x), Value::Tuple(y)) => {
            x.len() == y.len() && x.iter().zip(y.iter()).all(|(a, b)| py_eq(a, b))
        }
        (Value::Map(x), Value::Map(y)) => {
            x.len() == y.len()
                && x.iter()
                    .all(|(k, v)| Value::map_get(y, k).is_some_and(|w| py_eq(v, &w)))
        }
        (Value::Namespace(x), Value::Namespace(y)) => Rc::ptr_eq(x, y),
        (Value::Function(x), Value::Function(y)) => x == y,
        (Value::Macro(x), Value::Macro(y)) => Rc::ptr_eq(x, y),
        _ => false,
    }
}

/// The repr of a Python float: the shortest digits that read back to the
/// same float, in positional form for exponents from -4 to 15, else in
/// exponent form
pub fn float_repr(f: f64, out: &mut String) {
    if f.is_nan() {
        out.push_str("nan");
        return;
    }
    if f.is_infinite() {
        out.push_str(if f < 0.0 { "-inf" } else { "inf" });
        return;
    }
    // {:e} gives the shortest digits that read back to the same float
    let e = format!("{f:e}");
    let (mantissa, exp) = e.split_once('e').expect("{:e} gives an exponent");
    let exp: i32 = exp.parse().expect("an exponent");
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(m) => ("-", m),
        None => ("", mantissa),
    };
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    out.push_str(sign);
    const POSITIONAL_LOW: i32 = -4;
    const POSITIONAL_HIGH: i32 = 16;
    if (POSITIONAL_LOW..POSITIONAL_HIGH).contains(&exp) {
        let point = exp + 1;
        if point <= 0 {
            out.push_str("0.");
            for _ in 0..-point {
                out.push('0');
            }
            out.push_str(&digits);
        } else {
            let point = point as usize;
            if digits.len() <= point {
                out.push_str(&digits);
                for _ in digits.len()..point {
                    out.push('0');
                }
                out.push_str(".0");
            } else {
                out.push_str(&digits[..point]);
                out.push('.');
                out.push_str(&digits[point..]);
            }
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let _ = write!(out, "e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    }
}

/// The quote that the repr of a Python str takes
fn repr_quote(s: &str) -> char {
    if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    }
}

/// One char of the repr of a Python str with `quote`
fn repr_char(c: char, quote: char, out: &mut String) {
    match c {
        '\\' => out.push_str("\\\\"),
        '\n' => out.push_str("\\n"),
        '\r' => out.push_str("\\r"),
        '\t' => out.push_str("\\t"),
        c if c == quote => {
            out.push('\\');
            out.push(c);
        }
        c if (c as u32) < 0x20 || c as u32 == 0x7f => {
            let _ = write!(out, "\\x{:02x}", c as u32);
        }
        c => out.push(c),
    }
}

/// The repr of a Python str
pub fn str_repr(s: &str, out: &mut String) {
    let quote = repr_quote(s);
    out.push(quote);
    for c in s.chars() {
        repr_char(c, quote, out);
    }
    out.push(quote);
}

/// The repr of a value, as Python prints it inside a list or a dict
pub fn repr(v: &Value, out: &mut String) {
    match v {
        Value::Str(s) => str_repr(s.as_str(), out),
        Value::List(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                repr(item, out);
            }
            out.push(']');
        }
        Value::Tuple(items) => {
            out.push('(');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                repr(item, out);
            }
            if items.len() == 1 {
                out.push(',');
            }
            out.push(')');
        }
        Value::Map(items) => {
            out.push('{');
            for (i, (k, v)) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                repr(k, out);
                out.push_str(": ");
                repr(v, out);
            }
            out.push('}');
        }
        other => scalar_str(other, out),
    }
}

/// The str of a value that is not a string, a list or a dict
pub fn scalar_str(v: &Value, out: &mut String) {
    match v {
        Value::Undefined(_) => {}
        Value::None => out.push_str("None"),
        Value::Bool(true) => out.push_str("True"),
        Value::Bool(false) => out.push_str("False"),
        Value::Int(i) => {
            let _ = write!(out, "{i}");
        }
        Value::Float(f) => float_repr(*f, out),
        Value::Str(s) => out.push_str(s.as_str()),
        Value::Namespace(_) => out.push_str("<Namespace>"),
        Value::Macro(_) => out.push_str("<Macro>"),
        Value::Function(_) | Value::Method(_) => out.push_str("<function>"),
        Value::Loop(_) => out.push_str("<LoopContext>"),
        Value::Generator(_) => out.push_str("<generator>"),
        Value::List(_) | Value::Tuple(_) | Value::Map(_) => repr(v, out),
    }
}

/// The str of a value, as the output of the template gives it, with the
/// origins of its strings. A list or a dict prints as its repr, as Template
/// text with the origins of the strings that it holds.
pub fn to_ostr(v: &Value) -> OStr {
    match v {
        Value::Str(s) => (**s).clone(),
        Value::List(_) | Value::Tuple(_) | Value::Map(_) => {
            let mut out = OStr::default();
            repr_ostr(v, &mut out);
            out
        }
        other => {
            let mut s = String::new();
            scalar_str(other, &mut s);
            OStr::new(&s, Origin::Template)
        }
    }
}

fn repr_ostr(v: &Value, out: &mut OStr) {
    let t = Origin::Template;
    match v {
        Value::Str(s) => {
            // the quotes are Template text, and each char keeps its origin
            let quote = repr_quote(s.as_str());
            let mut q = String::new();
            q.push(quote);
            out.push(&q, t);
            let mut buf = String::new();
            for (at, c) in s.as_str().char_indices() {
                buf.clear();
                repr_char(c, quote, &mut buf);
                out.push(&buf, s.origin_at(at));
            }
            out.push(&q, t);
        }
        Value::List(items) | Value::Tuple(items) => {
            let tuple = matches!(v, Value::Tuple(_));
            out.push(if tuple { "(" } else { "[" }, t);
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(", ", t);
                }
                repr_ostr(item, out);
            }
            if tuple && items.len() == 1 {
                out.push(",", t);
            }
            out.push(if tuple { ")" } else { "]" }, t);
        }
        Value::Map(items) => {
            out.push("{", t);
            for (i, (k, v)) in items.iter().enumerate() {
                if i > 0 {
                    out.push(", ", t);
                }
                repr_ostr(k, out);
                out.push(": ", t);
                repr_ostr(v, out);
            }
            out.push("}", t);
        }
        other => {
            let mut s = String::new();
            scalar_str(other, &mut s);
            out.push(&s, t);
        }
    }
}
