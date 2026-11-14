//! JSON for templates: a parser of the values that a caller gives to a
//! render (tools, the arguments of tool calls), and the tojson filter of
//! the reference renderer, which is Python's json.dumps with
//! ensure_ascii=False by default.

use alloc::rc::Rc;
use core::fmt::Write;

use super::value::{OStr, Origin, Value, float_repr};
use crate::prelude::*;

/// A JSON text that does not parse, at this byte
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonError {
    pub at: usize,
    pub what: &'static str,
}

/// The value of the JSON text `text`. Each string gets `origin`. A number
/// with no fraction and no exponent is an int, as in Python.
pub fn parse(text: &str, origin: Origin) -> Result<Value, JsonError> {
    let mut p = Parser {
        s: text.as_bytes(),
        at: 0,
        origin,
    };
    let v = p.value()?;
    p.space();
    if p.at != p.s.len() {
        return Err(p.err("text after the value"));
    }
    Ok(v)
}

struct Parser<'a> {
    s: &'a [u8],
    at: usize,
    origin: Origin,
}

impl Parser<'_> {
    fn err(&self, what: &'static str) -> JsonError {
        JsonError { at: self.at, what }
    }

    fn space(&mut self) {
        while self
            .s
            .get(self.at)
            .is_some_and(|b| matches!(b, b' ' | b'\t' | b'\n' | b'\r'))
        {
            self.at += 1;
        }
    }

    fn eat(&mut self, word: &[u8]) -> bool {
        if self.s[self.at..].starts_with(word) {
            self.at += word.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self) -> Result<Value, JsonError> {
        self.space();
        match self.s.get(self.at) {
            None => Err(self.err("no value")),
            Some(b'{') => {
                self.at += 1;
                let mut items = Vec::new();
                self.space();
                if self.eat(b"}") {
                    return Ok(Value::map(items));
                }
                loop {
                    self.space();
                    if self.s.get(self.at) != Some(&b'"') {
                        return Err(self.err("a key that is not a string"));
                    }
                    let key = self.string()?;
                    self.space();
                    if !self.eat(b":") {
                        return Err(self.err("no colon after a key"));
                    }
                    let v = self.value()?;
                    items.push((key, v));
                    self.space();
                    if self.eat(b"}") {
                        return Ok(Value::map(items));
                    }
                    if !self.eat(b",") {
                        return Err(self.err("no comma in an object"));
                    }
                }
            }
            Some(b'[') => {
                self.at += 1;
                let mut items = Vec::new();
                self.space();
                if self.eat(b"]") {
                    return Ok(Value::list(items));
                }
                loop {
                    items.push(self.value()?);
                    self.space();
                    if self.eat(b"]") {
                        return Ok(Value::list(items));
                    }
                    if !self.eat(b",") {
                        return Err(self.err("no comma in an array"));
                    }
                }
            }
            Some(b'"') => self.string(),
            Some(b't') if self.eat(b"true") => Ok(Value::Bool(true)),
            Some(b'f') if self.eat(b"false") => Ok(Value::Bool(false)),
            Some(b'n') if self.eat(b"null") => Ok(Value::None),
            Some(_) => self.number(),
        }
    }

    fn hex4(&mut self) -> Result<u32, JsonError> {
        let hex = self
            .s
            .get(self.at..self.at + 4)
            .ok_or_else(|| self.err("a short unicode escape"))?;
        let hex = core::str::from_utf8(hex).map_err(|_| self.err("a unicode escape"))?;
        let v = u32::from_str_radix(hex, 16).map_err(|_| self.err("a unicode escape"))?;
        self.at += 4;
        Ok(v)
    }

    fn string(&mut self) -> Result<Value, JsonError> {
        const SURROGATE_HIGH: core::ops::Range<u32> = 0xD800..0xDC00;
        const SURROGATE_LOW: core::ops::Range<u32> = 0xDC00..0xE000;
        const SURROGATE_BASE: u32 = 0x10000;
        const SURROGATE_SHIFT: u32 = 10;
        self.at += 1;
        let mut out = String::new();
        loop {
            let start = self.at;
            while self
                .s
                .get(self.at)
                .is_some_and(|b| *b != b'"' && *b != b'\\' && *b >= 0x20)
            {
                self.at += 1;
            }
            let run = core::str::from_utf8(&self.s[start..self.at])
                .map_err(|_| self.err("a string that is not UTF-8"))?;
            out.push_str(run);
            match self.s.get(self.at) {
                Some(b'"') => {
                    self.at += 1;
                    return Ok(Value::str(&out, self.origin));
                }
                Some(b'\\') => {
                    self.at += 1;
                    let c = *self.s.get(self.at).ok_or_else(|| self.err("an escape"))?;
                    self.at += 1;
                    match c {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let mut code = self.hex4()?;
                            if SURROGATE_HIGH.contains(&code) && self.eat(b"\\u") {
                                let low = self.hex4()?;
                                if !SURROGATE_LOW.contains(&low) {
                                    return Err(self.err("a lone surrogate"));
                                }
                                code = SURROGATE_BASE
                                    + ((code - SURROGATE_HIGH.start) << SURROGATE_SHIFT)
                                    + (low - SURROGATE_LOW.start);
                            }
                            out.push(
                                char::from_u32(code).ok_or_else(|| self.err("a lone surrogate"))?,
                            );
                        }
                        _ => return Err(self.err("an unknown escape")),
                    }
                }
                _ => return Err(self.err("an unterminated string")),
            }
        }
    }

    fn number(&mut self) -> Result<Value, JsonError> {
        let start = self.at;
        let mut float = false;
        while let Some(&b) = self.s.get(self.at) {
            match b {
                b'0'..=b'9' | b'-' | b'+' => {}
                b'.' | b'e' | b'E' => float = true,
                _ => break,
            }
            self.at += 1;
        }
        let text =
            core::str::from_utf8(&self.s[start..self.at]).map_err(|_| self.err("a number"))?;
        if text.is_empty() {
            return Err(self.err("an unknown value"));
        }
        if float {
            text.parse()
                .map(Value::Float)
                .map_err(|_| self.err("a number"))
        } else {
            text.parse()
                .map(Value::Int)
                .map_err(|_| self.err("an integer out of range"))
        }
    }
}

/// The options of tojson that the corpus uses
#[derive(Debug, Clone, Copy)]
pub struct DumpOptions {
    pub ensure_ascii: bool,
    pub indent: Option<usize>,
}

/// A value that json.dumps refuses, as its TypeError message gives it
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NotSerializable(pub String);

/// json.dumps of `v`, with the origin of each string. The punctuation and
/// the keywords are Template text.
pub fn dump(v: &Value, opts: DumpOptions) -> Result<OStr, NotSerializable> {
    let mut out = OStr::default();
    write_value(v, opts, 0, &mut out)?;
    Ok(out)
}

fn newline(opts: DumpOptions, level: usize, out: &mut OStr) {
    if let Some(indent) = opts.indent {
        let mut s = String::from("\n");
        for _ in 0..indent * level {
            s.push(' ');
        }
        out.push(&s, Origin::Template);
    }
}

fn write_value(
    v: &Value,
    opts: DumpOptions,
    level: usize,
    out: &mut OStr,
) -> Result<(), NotSerializable> {
    let t = Origin::Template;
    // with an indent, Python's item separator has no space
    let item_sep = if opts.indent.is_some() { "," } else { ", " };
    match v {
        Value::None => out.push("null", t),
        Value::Bool(true) => out.push("true", t),
        Value::Bool(false) => out.push("false", t),
        Value::Int(i) => out.push(&format!("{i}"), t),
        Value::Float(f) => out.push(&float_json(*f), t),
        Value::Str(s) => write_string(s, opts, out),
        Value::List(items) | Value::Tuple(items) => {
            if items.is_empty() {
                out.push("[]", t);
                return Ok(());
            }
            out.push("[", t);
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(item_sep, t);
                }
                newline(opts, level + 1, out);
                write_value(item, opts, level + 1, out)?;
            }
            newline(opts, level, out);
            out.push("]", t);
        }
        Value::Map(items) => {
            if items.is_empty() {
                out.push("{}", t);
                return Ok(());
            }
            out.push("{", t);
            for (i, (k, v)) in items.iter().enumerate() {
                if i > 0 {
                    out.push(item_sep, t);
                }
                newline(opts, level + 1, out);
                write_key(k, opts, out)?;
                out.push(": ", t);
                write_value(v, opts, level + 1, out)?;
            }
            newline(opts, level, out);
            out.push("}", t);
        }
        other => {
            return Err(NotSerializable(format!(
                "Object of type {} is not JSON serializable",
                other.type_name()
            )));
        }
    }
    Ok(())
}

/// A key of a dict: Python turns a number, a bool or None into a string
fn write_key(k: &Value, opts: DumpOptions, out: &mut OStr) -> Result<(), NotSerializable> {
    let text = match k {
        Value::Str(s) => {
            write_string(s, opts, out);
            return Ok(());
        }
        Value::None => "null".into(),
        Value::Bool(true) => "true".into(),
        Value::Bool(false) => "false".into(),
        Value::Int(i) => format!("{i}"),
        Value::Float(f) => float_json(*f),
        other => {
            return Err(NotSerializable(format!(
                "keys must be str, int, float, bool or None, not {}",
                other.type_name()
            )));
        }
    };
    out.push(&format!("\"{text}\""), Origin::Template);
    Ok(())
}

/// A float as json.dumps writes it
fn float_json(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f < 0.0 {
            "-Infinity".into()
        } else {
            "Infinity".into()
        };
    }
    let mut s = String::new();
    float_repr(f, &mut s);
    s
}

fn write_string(s: &Rc<OStr>, opts: DumpOptions, out: &mut OStr) {
    const ASCII_END: u32 = 0x80;
    const CONTROL_END: u32 = 0x20;
    const BMP_END: u32 = 0x10000;
    out.push("\"", Origin::Template);
    let mut buf = String::new();
    for (at, c) in s.as_str().char_indices() {
        buf.clear();
        match c {
            '"' => buf.push_str("\\\""),
            '\\' => buf.push_str("\\\\"),
            '\n' => buf.push_str("\\n"),
            '\r' => buf.push_str("\\r"),
            '\t' => buf.push_str("\\t"),
            '\u{8}' => buf.push_str("\\b"),
            '\u{c}' => buf.push_str("\\f"),
            c if (c as u32) < CONTROL_END => {
                let _ = write!(buf, "\\u{:04x}", c as u32);
            }
            c if opts.ensure_ascii && c as u32 >= ASCII_END => {
                let code = c as u32;
                if code < BMP_END {
                    let _ = write!(buf, "\\u{code:04x}");
                } else {
                    let mut units = [0u16; 2];
                    for unit in c.encode_utf16(&mut units) {
                        let _ = write!(buf, "\\u{:04x}", *unit);
                    }
                }
            }
            c => buf.push(c),
        }
        out.push(&buf, s.origin_at(at));
    }
    out.push("\"", Origin::Template);
}
