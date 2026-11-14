//! The Python semantics of the values of a render: attribute and item
//! access, slices, operators, membership and order, the methods of str and
//! dict, the global functions, and the filters and tests of the subset.

use alloc::rc::Rc;
use core::cmp::Ordering;

use super::eval::{Renderer, Res, binop_name, defined, err, not_top, type_err};
use super::json::{self, DumpOptions};
use super::parser::BinOp;
use super::value::{Function, Generator, Number, OStr, Origin, Undefined, Value, py_eq, to_ostr};
use super::{Date, ErrorKind};
use crate::prelude::*;

/// The filters of the reference renderer: those of jinja2, with the tojson
/// of transformers. A template may name another filter, which fails when
/// a render reaches it, as in the reference.
pub const REFERENCE_FILTERS: [&str; 52] = [
    "abs",
    "attr",
    "batch",
    "capitalize",
    "center",
    "count",
    "d",
    "default",
    "dictsort",
    "e",
    "escape",
    "filesizeformat",
    "first",
    "float",
    "forceescape",
    "format",
    "groupby",
    "indent",
    "int",
    "items",
    "join",
    "last",
    "length",
    "list",
    "lower",
    "map",
    "max",
    "min",
    "pprint",
    "random",
    "reject",
    "rejectattr",
    "replace",
    "reverse",
    "round",
    "safe",
    "select",
    "selectattr",
    "slice",
    "sort",
    "string",
    "striptags",
    "sum",
    "title",
    "tojson",
    "trim",
    "truncate",
    "unique",
    "upper",
    "urlencode",
    "urlize",
    "wordcount",
];

/// The filters of the reference that the subset has
pub const FILTERS: [&str; 22] = [
    "count",
    "d",
    "default",
    "dictsort",
    "int",
    "items",
    "join",
    "length",
    "list",
    "lower",
    "map",
    "reject",
    "rejectattr",
    "replace",
    "safe",
    "select",
    "selectattr",
    "sort",
    "string",
    "tojson",
    "trim",
    "upper",
];

/// The tests of the reference renderer (jinja2)
pub const REFERENCE_TESTS: [&str; 39] = [
    "boolean",
    "callable",
    "defined",
    "divisibleby",
    "eq",
    "equalto",
    "escaped",
    "even",
    "false",
    "filter",
    "float",
    "ge",
    "gt",
    "greaterthan",
    "in",
    "integer",
    "iterable",
    "le",
    "lessthan",
    "lower",
    "lt",
    "mapping",
    "ne",
    "none",
    "number",
    "odd",
    "sameas",
    "sequence",
    "string",
    "test",
    "true",
    "undefined",
    "upper",
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
];

/// The tests of the reference that the subset has: those that the corpus uses
pub const TESTS: [&str; 11] = [
    "boolean",
    "defined",
    "equalto",
    "false",
    "iterable",
    "mapping",
    "none",
    "sequence",
    "string",
    "true",
    "undefined",
];

/// The methods of str and dict that the subset has. A call of another
/// method fails at load.
pub const METHODS: [&str; 11] = [
    "split",
    "strip",
    "lstrip",
    "rstrip",
    "startswith",
    "endswith",
    "replace",
    "upper",
    "lower",
    "items",
    "get",
];
const DICT_METHODS: [&str; 2] = ["items", "get"];
const STR_METHODS: [&str; 9] = [
    "split",
    "strip",
    "lstrip",
    "rstrip",
    "startswith",
    "endswith",
    "replace",
    "upper",
    "lower",
];

/// The whitespace of Python's str.isspace
fn py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

fn undefined(name: &str, line: u32) -> Value {
    Value::Undefined(Rc::new(Undefined {
        name: name.into(),
        top: false,
        line,
    }))
}

/// The items of a for loop over `v`, or of list(v)
pub fn iterate(v: &Value, line: u32) -> Res<Vec<Value>> {
    plain(v, "an iteration", line)?;
    Ok(match v {
        // an unset value iterates as empty
        Value::Undefined(_) => Vec::new(),
        Value::Str(s) => s
            .as_str()
            .char_indices()
            .map(|(at, c)| Value::ostr(s.slice(at, at + c.len_utf8())))
            .collect(),
        Value::List(l) | Value::Tuple(l) => l.to_vec(),
        Value::Generator(g) => {
            if g.used.replace(true) {
                return Err(err(ErrorKind::GeneratorReused, line));
            }
            g.items.clone()
        }
        Value::Map(m) => m.iter().map(|(k, _)| k.clone()).collect(),
        other => {
            return Err(type_err(
                line,
                format!("'{}' object is not iterable", other.type_name()),
            ));
        }
    })
}

/// The items of the input of map, select, reject, selectattr and
/// rejectattr: jinja2 iterates the input only when it is true
fn iterate_if_true(v: &Value, line: u32) -> Res<Vec<Value>> {
    if v.truthy() {
        iterate(v, line)
    } else {
        Ok(Vec::new())
    }
}

fn generator(items: Vec<Value>) -> Value {
    Value::Generator(Rc::new(Generator {
        items,
        used: core::cell::Cell::new(false),
    }))
}

/// Fail by name on a Markup string: the subset has its output, +, and the
/// safe filter only
pub fn plain(v: &Value, what: &'static str, line: u32) -> Res<()> {
    match v {
        Value::Str(s) if s.is_markup() => Err(err(ErrorKind::MarkupUnsupported(what), line)),
        _ => Ok(()),
    }
}

/// obj.attr, as jinja2 gives it: an attribute of Python first (a method),
/// then an item
pub fn getattr(obj: &Value, attr: &str, line: u32) -> Res<Value> {
    Ok(match obj {
        Value::Map(m) => {
            if DICT_METHODS.contains(&attr) {
                return Ok(Value::Method(Rc::new((obj.clone(), attr.into()))));
            }
            Value::map_get_str(m, attr).unwrap_or_else(|| undefined(attr, line))
        }
        Value::Str(_) if STR_METHODS.contains(&attr) => {
            Value::Method(Rc::new((obj.clone(), attr.into())))
        }
        Value::Namespace(ns) => ns
            .borrow()
            .iter()
            .find(|(n, _)| n == attr)
            .map(|(_, v)| v.clone())
            .unwrap_or_else(|| undefined(attr, line)),
        Value::Loop(l) => {
            let i = l.index0;
            let n = l.length;
            match attr {
                "index" => Value::Int((i + 1) as i64),
                "index0" => Value::Int(i as i64),
                "first" => Value::Bool(i == 0),
                "last" => Value::Bool(i + 1 == n),
                "length" => Value::Int(n as i64),
                "previtem" => l.previtem.clone().unwrap_or_else(|| undefined(attr, line)),
                "nextitem" => l.nextitem.clone().unwrap_or_else(|| undefined(attr, line)),
                _ => {
                    return Err(err(
                        ErrorKind::Unsupported("a loop attribute outside the subset"),
                        line,
                    ));
                }
            }
        }
        _ => undefined(attr, line),
    })
}

/// The index `i` of a sequence of `len`, from its end when negative
fn index_of(i: i64, len: usize) -> Option<usize> {
    let i = if i < 0 { i + len as i64 } else { i };
    usize::try_from(i).ok().filter(|i| *i < len)
}

/// obj[key], as jinja2 gives it: an item first, then an attribute for a
/// string key, else Undefined
pub fn getitem(obj: &Value, key: &Value, line: u32) -> Res<Value> {
    plain(obj, "an item", line)?;
    let int_key = match key {
        Value::Int(i) => Some(*i),
        Value::Bool(b) => Some(i64::from(*b)),
        _ => None,
    };
    let found = match (obj, int_key) {
        (Value::Map(m), _) => Value::map_get(m, key),
        (Value::List(l) | Value::Tuple(l), Some(i)) => index_of(i, l.len()).map(|i| l[i].clone()),
        (Value::Str(s), Some(i)) => {
            let chars: Vec<(usize, char)> = s.as_str().char_indices().collect();
            index_of(i, chars.len()).map(|i| {
                let (at, c) = chars[i];
                Value::ostr(s.slice(at, at + c.len_utf8()))
            })
        }
        _ => None,
    };
    if let Some(v) = found {
        return Ok(v);
    }
    Ok(match key {
        Value::Str(k) => getattr(obj, k.as_str(), line)?,
        other => {
            let mut s = String::new();
            super::value::repr(other, &mut s);
            undefined(&s, line)
        }
    })
}

/// The indices of a slice of Python over `len` items
fn slice_indices(len: usize, start: Option<i64>, stop: Option<i64>, step: i64) -> Vec<usize> {
    let n = len as i64;
    let clamp = |v: i64, low: i64, high: i64| v.max(low).min(high);
    let norm = |v: i64| if v < 0 { v + n } else { v };
    let mut out = Vec::new();
    if step > 0 {
        let a = start.map_or(0, |v| clamp(norm(v), 0, n));
        let b = stop.map_or(n, |v| clamp(norm(v), 0, n));
        let mut i = a;
        while i < b {
            out.push(i as usize);
            i += step;
        }
    } else {
        let a = start.map_or(n - 1, |v| clamp(norm(v), -1, n - 1));
        let b = stop.map_or(-1, |v| clamp(norm(v), -1, n - 1));
        let mut i = a;
        while i > b {
            out.push(i as usize);
            i += step;
        }
    }
    out
}

pub fn slice(
    v: &Value,
    start: Option<i64>,
    stop: Option<i64>,
    step: Option<i64>,
    line: u32,
) -> Res<Value> {
    plain(v, "a slice", line)?;
    let step = step.unwrap_or(1);
    if step == 0 {
        return Err(err(
            ErrorKind::Value("slice step cannot be zero".into()),
            line,
        ));
    }
    Ok(match v {
        Value::Str(s) => {
            let chars: Vec<(usize, char)> = s.as_str().char_indices().collect();
            let mut out = OStr::default();
            for i in slice_indices(chars.len(), start, stop, step) {
                let (at, c) = chars[i];
                out.push_ostr(&s.slice(at, at + c.len_utf8()));
            }
            Value::ostr(out)
        }
        Value::List(l) => Value::list(
            slice_indices(l.len(), start, stop, step)
                .into_iter()
                .map(|i| l[i].clone())
                .collect(),
        ),
        Value::Tuple(l) => Value::Tuple(Rc::new(
            slice_indices(l.len(), start, stop, step)
                .into_iter()
                .map(|i| l[i].clone())
                .collect(),
        )),
        other => {
            return Err(type_err(
                line,
                format!("'{}' object is not subscriptable", other.type_name()),
            ));
        }
    })
}

/// The binary operator `op` of Python
pub fn binary(op: BinOp, l: Value, r: Value, line: u32) -> Res<Value> {
    if op == BinOp::Concat {
        let (l, r) = (not_top(l, line)?, not_top(r, line)?);
        let mut s = to_ostr(&l);
        s.push_ostr(&to_ostr(&r));
        return Ok(Value::ostr(s.into_plain()));
    }
    let (l, r) = (defined(l, line)?, defined(r, line)?);
    let unsupported = |l: &Value, r: &Value| {
        type_err(
            line,
            format!(
                "unsupported operand type(s) for {}: '{}' and '{}'",
                binop_name(op),
                l.type_name(),
                r.type_name()
            ),
        )
    };
    match (op, &l, &r) {
        (BinOp::Add, Value::Str(a), Value::Str(b)) => {
            // Markup + str escapes the str, and gives Markup
            let markup = a.is_markup() || b.is_markup();
            let part = |s: &OStr, other_markup: bool| {
                if other_markup && !s.is_markup() {
                    s.html_escaped()
                } else {
                    s.clone()
                }
            };
            let mut s = part(a, b.is_markup());
            s.push_ostr(&part(b, a.is_markup()));
            return Ok(Value::ostr(if markup { s.into_markup() } else { s }));
        }
        (BinOp::Add, Value::Str(_), other) => {
            return Err(type_err(
                line,
                format!(
                    "can only concatenate str (not \"{}\") to str",
                    other.type_name()
                ),
            ));
        }
        (BinOp::Add, Value::List(a), Value::List(b)) => {
            return Ok(Value::list(a.iter().chain(b.iter()).cloned().collect()));
        }
        (BinOp::Add, Value::List(_), other) => {
            return Err(type_err(
                line,
                format!(
                    "can only concatenate list (not \"{}\") to list",
                    other.type_name()
                ),
            ));
        }
        (BinOp::Mul, Value::Str(s), Value::Int(n)) | (BinOp::Mul, Value::Int(n), Value::Str(s)) => {
            let mut out = OStr::default();
            for _ in 0..(*n).max(0) {
                out.push_ostr(s);
            }
            return Ok(Value::ostr(out));
        }
        (_, Value::Str(s), _) | (_, _, Value::Str(s)) if s.is_markup() && op != BinOp::Add => {
            return Err(err(
                ErrorKind::MarkupUnsupported("an operator other than +"),
                line,
            ));
        }
        (BinOp::Mod, Value::Str(_), _) => {
            return Err(err(
                ErrorKind::Unsupported("string formatting with %"),
                line,
            ));
        }
        _ => {}
    }
    let (Some(a), Some(b)) = (l.number(), r.number()) else {
        return Err(unsupported(&l, &r));
    };
    let zero = || err(ErrorKind::Runtime("division by zero".into()), line);
    Ok(match (a, b) {
        (Number::Int(x), Number::Int(y)) => match op {
            BinOp::Add => Value::Int(x.checked_add(y).ok_or_else(|| overflow(line))?),
            BinOp::Sub => Value::Int(x.checked_sub(y).ok_or_else(|| overflow(line))?),
            BinOp::Mul => Value::Int(x.checked_mul(y).ok_or_else(|| overflow(line))?),
            BinOp::Div => {
                if y == 0 {
                    return Err(zero());
                }
                Value::Float(x as f64 / y as f64)
            }
            BinOp::Mod => {
                if y == 0 {
                    return Err(zero());
                }
                let m = x % y;
                Value::Int(if m != 0 && ((m < 0) != (y < 0)) {
                    m + y
                } else {
                    m
                })
            }
            BinOp::Concat => unreachable!("concat is handled above"),
        },
        (a, b) => {
            let (x, y) = (a.as_f64(), b.as_f64());
            match op {
                BinOp::Add => Value::Float(x + y),
                BinOp::Sub => Value::Float(x - y),
                BinOp::Mul => Value::Float(x * y),
                BinOp::Div => {
                    if y == 0.0 {
                        return Err(zero());
                    }
                    Value::Float(x / y)
                }
                BinOp::Mod => {
                    if y == 0.0 {
                        return Err(zero());
                    }
                    let m = libm::fmod(x, y);
                    Value::Float(if m != 0.0 && ((m < 0.0) != (y < 0.0)) {
                        m + y
                    } else {
                        m
                    })
                }
                BinOp::Concat => unreachable!("concat is handled above"),
            }
        }
    })
}

/// Python ints have no bound; an i64 that overflows fails by name
fn overflow(line: u32) -> super::TemplateError {
    err(ErrorKind::Unsupported("an integer past 64 bits"), line)
}

/// `item in container`, as Python gives it
pub fn contains(container: &Value, item: &Value, line: u32) -> Res<bool> {
    Ok(match container {
        Value::Undefined(_) => false,
        Value::Str(s) => match item {
            Value::Str(i) => s.as_str().contains(i.as_str()),
            other => {
                return Err(type_err(
                    line,
                    format!(
                        "'in <string>' requires string as left operand, not {}",
                        other.type_name()
                    ),
                ));
            }
        },
        Value::List(l) | Value::Tuple(l) => l.iter().any(|v| py_eq(v, item)),
        Value::Generator(_) => iterate(container, line)?.iter().any(|v| py_eq(v, item)),
        Value::Map(m) => m.iter().any(|(k, _)| py_eq(k, item)),
        other => {
            return Err(type_err(
                line,
                format!("argument of type '{}' is not iterable", other.type_name()),
            ));
        }
    })
}

/// The order of two values, as < of Python gives it
pub fn order(l: &Value, r: &Value, line: u32, symbol: &str) -> Res<Ordering> {
    plain(l, "a comparison", line)?;
    plain(r, "a comparison", line)?;
    if let (Some(a), Some(b)) = (l.number(), r.number()) {
        return match (a, b) {
            (Number::Int(x), Number::Int(y)) => Ok(x.cmp(&y)),
            (a, b) => a
                .as_f64()
                .partial_cmp(&b.as_f64())
                .ok_or_else(|| err(ErrorKind::Unsupported("an order of nan"), line)),
        };
    }
    match (l, r) {
        (Value::Str(a), Value::Str(b)) => Ok(a.as_str().cmp(b.as_str())),
        (Value::List(a), Value::List(b)) | (Value::Tuple(a), Value::Tuple(b)) => {
            for (x, y) in a.iter().zip(b.iter()) {
                if !py_eq(x, y) {
                    return order(x, y, line, symbol);
                }
            }
            Ok(a.len().cmp(&b.len()))
        }
        _ => Err(type_err(
            line,
            format!(
                "'{symbol}' not supported between instances of '{}' and '{}'",
                l.type_name(),
                r.type_name()
            ),
        )),
    }
}

/// The value of a keyword argument `name`, or of the positional argument
/// at `place`
fn arg<'v>(
    args: &'v [Value],
    kwargs: &'v [(String, Value)],
    place: usize,
    name: &str,
) -> Option<&'v Value> {
    kwargs
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, v)| v)
        .or_else(|| args.get(place))
}

fn as_str(v: &Value, line: u32, what: &str) -> Res<Rc<OStr>> {
    match v {
        Value::Str(s) => Ok(s.clone()),
        other => Err(type_err(
            line,
            format!("{what} must be str, not {}", other.type_name()),
        )),
    }
}

fn as_bool(v: Option<&Value>) -> bool {
    v.is_some_and(Value::truthy)
}

/// The text of a value as jinja2's soft_str gives it: a string as it is,
/// another value as its str
fn soft_str(v: &Value) -> OStr {
    to_ostr(v)
}

/// The global function `f`
pub fn call_function(
    r: &mut Renderer<'_>,
    f: Function,
    args: &[Value],
    kwargs: &[(String, Value)],
    line: u32,
) -> Res<Value> {
    match f {
        Function::Namespace => r.namespace(args, kwargs, line),
        Function::RaiseException => {
            let msg = args.first().map(soft_str).unwrap_or_default();
            Err(err(ErrorKind::Raised(msg.as_str().into()), line))
        }
        Function::Range => {
            let ints: Vec<i64> = args
                .iter()
                .map(|a| match a {
                    Value::Int(i) => Ok(*i),
                    Value::Bool(b) => Ok(i64::from(*b)),
                    other => Err(type_err(
                        line,
                        format!(
                            "'{}' object cannot be interpreted as an integer",
                            other.type_name()
                        ),
                    )),
                })
                .collect::<Res<_>>()?;
            let (start, stop, step) = match ints[..] {
                [stop] => (0, stop, 1),
                [start, stop] => (start, stop, 1),
                [start, stop, step] => (start, stop, step),
                _ => {
                    return Err(type_err(
                        line,
                        format!("range expected 1 to 3 arguments, got {}", ints.len()),
                    ));
                }
            };
            if step == 0 {
                return Err(err(
                    ErrorKind::Value("range() arg 3 must not be zero".into()),
                    line,
                ));
            }
            let mut out = Vec::new();
            let mut i = start;
            while (step > 0 && i < stop) || (step < 0 && i > stop) {
                out.push(Value::Int(i));
                i += step;
            }
            Ok(Value::list(out))
        }
        Function::StrftimeNow => {
            let date = r.date().ok_or_else(|| err(ErrorKind::DateMissing, line))?;
            let fmt = as_str(args.first().unwrap_or(&Value::None), line, "the format")?;
            Ok(Value::str(
                &strftime(date, fmt.as_str(), line)?,
                Origin::Template,
            ))
        }
    }
}

const MONTHS: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];
/// The letters of the short name of a month
const SHORT_MONTH: usize = 3;

/// A date in the format codes of the corpus: %Y %m %d %b %B and %%
fn strftime(date: Date, fmt: &str, line: u32) -> Res<String> {
    let mut out = String::new();
    let mut chars = fmt.chars();
    let month = MONTHS
        .get(usize::from(date.month).wrapping_sub(1))
        .ok_or_else(|| err(ErrorKind::Value(format!("month {}", date.month)), line))?;
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('Y') => out.push_str(&format!("{}", date.year)),
            Some('m') => out.push_str(&format!("{:02}", date.month)),
            Some('d') => out.push_str(&format!("{:02}", date.day)),
            Some('b') => out.push_str(&month[..SHORT_MONTH]),
            Some('B') => out.push_str(month),
            Some('%') => out.push('%'),
            Some(other) => return Err(err(ErrorKind::DateFormat(other), line)),
            None => return Err(err(ErrorKind::DateFormat('%'), line)),
        }
    }
    Ok(out)
}

/// str.split of Python: on runs of whitespace with no separator, else on
/// each separator, at most `max` times when `max` is 0 or more
fn split(s: &OStr, sep: Option<&str>, max: i64) -> Vec<Value> {
    let text = s.as_str();
    let mut cuts: Vec<(usize, usize)> = Vec::new();
    match sep {
        Some(sep) => {
            let found: Vec<(usize, &str)> = text.match_indices(sep).collect();
            let mut taken = Vec::new();
            for (at, m) in found {
                if max >= 0 && taken.len() as i64 >= max {
                    break;
                }
                taken.push((at, at + m.len()));
            }
            let mut start = 0;
            for (a, b) in taken {
                cuts.push((start, a));
                start = b;
            }
            cuts.push((start, text.len()));
        }
        None => {
            let words: Vec<(usize, usize)> = {
                let mut v = Vec::new();
                let mut start = None;
                for (at, c) in text.char_indices() {
                    match (py_space(c), start) {
                        (true, Some(s)) => {
                            v.push((s, at));
                            start = None;
                        }
                        (false, None) => start = Some(at),
                        _ => {}
                    }
                }
                if let Some(s) = start {
                    v.push((s, text.len()));
                }
                v
            };
            if max < 0 || (words.len() as i64) <= max + 1 {
                cuts = words;
            } else {
                let keep = max as usize;
                cuts.extend_from_slice(&words[..keep]);
                cuts.push((words[keep].0, text.trim_end_matches(py_space).len()));
            }
        }
    }
    cuts.into_iter()
        .map(|(a, b)| Value::ostr(s.slice(a, b)))
        .collect()
}

/// str.strip of Python: the chars of `chars`, or whitespace
fn strip(s: &OStr, chars: Option<&str>, left: bool, right: bool) -> OStr {
    let text = s.as_str();
    let drop = |c: char| chars.map_or_else(|| py_space(c), |cs| cs.contains(c));
    let start = if left {
        text.len() - text.trim_start_matches(drop).len()
    } else {
        0
    };
    let end = if right {
        text.trim_end_matches(drop).len()
    } else {
        text.len()
    };
    if start >= end {
        OStr::default()
    } else {
        s.slice(start, end)
    }
}

fn upper(s: &OStr) -> OStr {
    s.map_chars(|c, out| out.extend(c.to_uppercase()))
}

fn lower(s: &OStr) -> OStr {
    s.map_chars(|c, out| out.extend(c.to_lowercase()))
}

/// str.replace of Python
fn replace(s: &OStr, old: &OStr, new: &OStr, count: i64) -> OStr {
    let text = s.as_str();
    let mut out = OStr::default();
    let mut start = 0;
    let mut done = 0;
    if old.is_empty() {
        // Python puts new between each char and at both ends
        let mut ats: Vec<usize> = text.char_indices().map(|(a, _)| a).collect();
        ats.push(text.len());
        for (i, at) in ats.iter().enumerate() {
            if count < 0 || done < count {
                out.push_ostr(new);
                done += 1;
            }
            if let Some(next) = ats.get(i + 1) {
                out.push_ostr(&s.slice(*at, *next));
            }
        }
        return out;
    }
    for (at, _) in text.match_indices(old.as_str()) {
        if count >= 0 && done >= count {
            break;
        }
        out.push_ostr(&s.slice(start, at));
        out.push_ostr(new);
        start = at + old.len();
        done += 1;
    }
    out.push_ostr(&s.slice(start, text.len()));
    out
}

fn opt_str(v: Option<&Value>, line: u32, what: &str) -> Res<Option<Rc<OStr>>> {
    match v {
        None | Some(Value::None) => Ok(None),
        Some(v) => as_str(v, line, what).map(Some),
    }
}

fn opt_int(v: Option<&Value>, default: i64, line: u32) -> Res<i64> {
    match v {
        None | Some(Value::None) => Ok(default),
        Some(Value::Int(i)) => Ok(*i),
        Some(Value::Bool(b)) => Ok(i64::from(*b)),
        Some(other) => Err(type_err(
            line,
            format!("an integer is required, not {}", other.type_name()),
        )),
    }
}

/// A method of a str or a dict
pub fn call_method(
    obj: &Value,
    name: &str,
    args: &[Value],
    kwargs: &[(String, Value)],
    line: u32,
) -> Res<Value> {
    if let Value::Map(m) = obj {
        return Ok(match name {
            "items" => Value::list(
                m.iter()
                    .map(|(k, v)| Value::Tuple(Rc::new(vec![k.clone(), v.clone()])))
                    .collect(),
            ),
            _ => {
                let key = arg(args, kwargs, 0, "key")
                    .ok_or_else(|| type_err(line, "get expected at least 1 argument".into()))?;
                Value::map_get(m, key).unwrap_or_else(|| {
                    arg(args, kwargs, 1, "default")
                        .cloned()
                        .unwrap_or(Value::None)
                })
            }
        });
    }
    let Value::Str(s) = obj else {
        return Err(type_err(
            line,
            format!("'{}' object has no method {name}", obj.type_name()),
        ));
    };
    plain(obj, "a method", line)?;
    Ok(match name {
        "split" => {
            let sep = opt_str(arg(args, kwargs, 0, "sep"), line, "the separator")?;
            if sep.as_ref().is_some_and(|s| s.is_empty()) {
                return Err(err(ErrorKind::Value("empty separator".into()), line));
            }
            let max = opt_int(arg(args, kwargs, 1, "maxsplit"), -1, line)?;
            Value::list(split(s, sep.as_ref().map(|s| s.as_str()), max))
        }
        "strip" | "lstrip" | "rstrip" => {
            let chars = opt_str(arg(args, kwargs, 0, "chars"), line, "the chars")?;
            Value::ostr(strip(
                s,
                chars.as_ref().map(|c| c.as_str()),
                name != "rstrip",
                name != "lstrip",
            ))
        }
        "startswith" | "endswith" => {
            let what = arg(args, kwargs, 0, "prefix")
                .ok_or_else(|| type_err(line, format!("{name} takes at least 1 argument")))?;
            let options: Vec<Value> = match what {
                Value::Tuple(t) => t.to_vec(),
                other => vec![other.clone()],
            };
            let mut yes = false;
            for o in options {
                let o = as_str(&o, line, name)?;
                yes |= if name == "startswith" {
                    s.as_str().starts_with(o.as_str())
                } else {
                    s.as_str().ends_with(o.as_str())
                };
            }
            Value::Bool(yes)
        }
        "replace" => {
            let old = as_str(
                arg(args, kwargs, 0, "old").unwrap_or(&Value::None),
                line,
                "old",
            )?;
            let new = as_str(
                arg(args, kwargs, 1, "new").unwrap_or(&Value::None),
                line,
                "new",
            )?;
            let count = opt_int(arg(args, kwargs, 2, "count"), -1, line)?;
            Value::ostr(replace(s, &old, &new, count))
        }
        "upper" => Value::ostr(upper(s)),
        "lower" => Value::ostr(lower(s)),
        _ => {
            return Err(err(
                ErrorKind::Unsupported("a method outside the subset"),
                line,
            ));
        }
    })
}

/// The value at the dotted path `path` of `v`, as the attribute argument
/// of jinja2's filters reads it
fn attr_path(v: &Value, path: &str, line: u32) -> Res<Value> {
    let mut cur = v.clone();
    for part in path.split('.') {
        cur = match part.parse::<i64>() {
            Ok(i) => getitem(&cur, &Value::Int(i), line)?,
            Err(_) => getitem(&cur, &Value::str(part, Origin::Template), line)?,
        };
    }
    Ok(cur)
}

/// The key of a sort: a lowered string when the sort is not case
/// sensitive
fn sort_key(v: &Value, case_sensitive: bool) -> Value {
    match v {
        Value::Str(s) if !case_sensitive => Value::ostr(lower(s)),
        other => other.clone(),
    }
}

fn sorted(mut items: Vec<(Value, Value)>, reverse: bool, line: u32) -> Res<Vec<(Value, Value)>> {
    let mut failed = None;
    items.sort_by(|a, b| {
        order(&a.0, &b.0, line, "<").unwrap_or_else(|e| {
            failed.get_or_insert(e);
            Ordering::Equal
        })
    });
    if let Some(e) = failed {
        return Err(e);
    }
    if reverse {
        items.reverse();
    }
    Ok(items)
}

/// The filter `name` with the value `v`
pub fn filter(
    name: &str,
    v: Value,
    args: &[Value],
    kwargs: &[(String, Value)],
    line: u32,
) -> Res<Value> {
    if name == "default" || name == "d" {
        let fallback = arg(args, kwargs, 0, "default_value")
            .cloned()
            .unwrap_or_else(|| Value::str("", Origin::Template));
        let boolean = as_bool(arg(args, kwargs, 1, "boolean"));
        return Ok(match &v {
            Value::Undefined(_) => fallback,
            other if boolean && !other.truthy() => fallback,
            _ => v,
        });
    }
    if !REFERENCE_FILTERS.contains(&name) {
        return Err(err(ErrorKind::FilterUnknown(name.into()), line));
    }
    if name != "safe" {
        plain(&v, "a filter other than safe", line)?;
    }
    let v = not_top(v, line)?;
    Ok(match name {
        "length" | "count" => {
            let n = match &v {
                Value::Undefined(_) => 0,
                Value::Str(s) => s.as_str().chars().count(),
                Value::List(l) | Value::Tuple(l) => l.len(),
                Value::Map(m) => m.len(),
                other => {
                    return Err(type_err(
                        line,
                        format!("object of type '{}' has no len()", other.type_name()),
                    ));
                }
            };
            Value::Int(n as i64)
        }
        "string" => Value::ostr(soft_str(&v)),
        "safe" => Value::ostr(soft_str(&v).into_markup()),
        "trim" => {
            let chars = opt_str(arg(args, kwargs, 0, "chars"), line, "the chars")?;
            Value::ostr(strip(
                &soft_str(&v),
                chars.as_ref().map(|c| c.as_str()),
                true,
                true,
            ))
        }
        "upper" => Value::ostr(upper(&soft_str(&v))),
        "lower" => Value::ostr(lower(&soft_str(&v))),
        "replace" => {
            let old = as_str(
                arg(args, kwargs, 0, "old").unwrap_or(&Value::None),
                line,
                "old",
            )?;
            let new = as_str(
                arg(args, kwargs, 1, "new").unwrap_or(&Value::None),
                line,
                "new",
            )?;
            let count = opt_int(arg(args, kwargs, 2, "count"), -1, line)?;
            Value::ostr(replace(&soft_str(&v), &old, &new, count))
        }
        "items" => match &v {
            Value::Undefined(_) => generator(Vec::new()),
            Value::Map(m) => generator(
                m.iter()
                    .map(|(k, x)| Value::Tuple(Rc::new(vec![k.clone(), x.clone()])))
                    .collect(),
            ),
            _ => {
                return Err(type_err(
                    line,
                    "Can only get item pairs from a mapping.".into(),
                ));
            }
        },
        "list" => Value::list(iterate(&v, line)?),
        "join" => {
            let sep = arg(args, kwargs, 0, "d").map(soft_str).unwrap_or_default();
            let attribute = opt_str(arg(args, kwargs, 1, "attribute"), line, "the attribute")?;
            let mut out = OStr::default();
            for (i, item) in iterate(&v, line)?.iter().enumerate() {
                if i > 0 {
                    out.push_ostr(&sep);
                }
                let item = match &attribute {
                    Some(a) => attr_path(item, a.as_str(), line)?,
                    None => item.clone(),
                };
                out.push_ostr(&soft_str(&item));
            }
            Value::ostr(out)
        }
        "int" => {
            let fallback = arg(args, kwargs, 0, "default")
                .cloned()
                .unwrap_or(Value::Int(0));
            match &v {
                Value::Int(i) => Value::Int(*i),
                Value::Bool(b) => Value::Int(i64::from(*b)),
                Value::Float(f) if f.is_finite() => Value::Int(*f as i64),
                Value::Str(s) => {
                    let t = s.as_str().trim_matches(py_space);
                    let t: String = t.chars().filter(|c| *c != '_').collect();
                    match t.parse::<i64>() {
                        Ok(i) => Value::Int(i),
                        Err(_) => match t.parse::<f64>() {
                            Ok(f) if f.is_finite() => Value::Int(f as i64),
                            _ => fallback,
                        },
                    }
                }
                _ => fallback,
            }
        }
        "tojson" => {
            // the positions of the tojson of transformers:
            // tojson(x, ensure_ascii, indent, separators, sort_keys)
            let ensure_ascii = as_bool(arg(args, kwargs, 0, "ensure_ascii"));
            let indent = match arg(args, kwargs, 1, "indent") {
                None | Some(Value::None) => None,
                Some(Value::Int(i)) => Some(
                    usize::try_from(*i).map_err(|_| type_err(line, "a negative indent".into()))?,
                ),
                Some(other) => {
                    return Err(type_err(
                        line,
                        format!("an indent of {}", other.type_name()),
                    ));
                }
            };
            if arg(args, kwargs, 2, "separators").is_some()
                || arg(args, kwargs, 3, "sort_keys").is_some()
            {
                return Err(err(
                    ErrorKind::Unsupported("tojson with separators or sort_keys"),
                    line,
                ));
            }
            let dumped = json::dump(
                &v,
                DumpOptions {
                    ensure_ascii,
                    indent,
                },
            )
            .map_err(|e| type_err(line, e.0))?;
            Value::ostr(dumped)
        }
        "selectattr" | "rejectattr" | "select" | "reject" => {
            let by_attr = name.ends_with("attr");
            let keep = !name.starts_with("reject");
            let (attribute, test_at) = if by_attr {
                let a = as_str(args.first().unwrap_or(&Value::None), line, "the attribute")?;
                (Some(a), 1)
            } else {
                (None, 0)
            };
            let test_name = match args.get(test_at) {
                Some(t) => Some(as_str(t, line, "the test")?),
                None => None,
            };
            let rest = args.get(test_at + 1..).unwrap_or(&[]);
            let mut out = Vec::new();
            for item in iterate_if_true(&v, line)? {
                let subject = match &attribute {
                    Some(a) => attr_path(&item, a.as_str(), line)?,
                    None => item.clone(),
                };
                let passed = match &test_name {
                    Some(t) => test(t.as_str(), &subject, rest, line)?,
                    None => subject.truthy(),
                };
                if passed == keep {
                    out.push(item);
                }
            }
            generator(out)
        }
        "map" => {
            let items = iterate_if_true(&v, line)?;
            let mut out = Vec::new();
            if let Some(a) = kwargs
                .iter()
                .find(|(n, _)| n == "attribute")
                .map(|(_, a)| a)
            {
                let a = as_str(a, line, "the attribute")?;
                let fallback = kwargs
                    .iter()
                    .find(|(n, _)| n == "default")
                    .map(|(_, d)| d.clone());
                for item in items {
                    let got = attr_path(&item, a.as_str(), line)?;
                    out.push(match (&got, &fallback) {
                        (Value::Undefined(_), Some(d)) => d.clone(),
                        _ => got,
                    });
                }
            } else {
                let f = as_str(args.first().unwrap_or(&Value::None), line, "the filter")?;
                let rest = args.get(1..).unwrap_or(&[]);
                for item in items {
                    out.push(filter(f.as_str(), item, rest, &[], line)?);
                }
            }
            generator(out)
        }
        "dictsort" => {
            let case_sensitive = as_bool(arg(args, kwargs, 0, "case_sensitive"));
            let by = opt_str(arg(args, kwargs, 1, "by"), line, "by")?;
            let reverse = as_bool(arg(args, kwargs, 2, "reverse"));
            let Value::Map(m) = &v else {
                return Err(type_err(line, "dictsort needs a mapping".into()));
            };
            let by_value = match by.as_ref().map(|b| b.as_str()) {
                None | Some("key") => false,
                Some("value") => true,
                Some(_) => {
                    return Err(err(
                        ErrorKind::Value("You can only sort by either \"key\" or \"value\"".into()),
                        line,
                    ));
                }
            };
            let keyed: Vec<(Value, Value)> = m
                .iter()
                .map(|(k, x)| {
                    let key = sort_key(if by_value { x } else { k }, case_sensitive);
                    (key, Value::Tuple(Rc::new(vec![k.clone(), x.clone()])))
                })
                .collect();
            Value::list(
                sorted(keyed, reverse, line)?
                    .into_iter()
                    .map(|(_, x)| x)
                    .collect(),
            )
        }
        "sort" => {
            let reverse = as_bool(arg(args, kwargs, 0, "reverse"));
            let case_sensitive = as_bool(arg(args, kwargs, 1, "case_sensitive"));
            let attribute = opt_str(arg(args, kwargs, 2, "attribute"), line, "the attribute")?;
            let mut keyed = Vec::new();
            for item in iterate(&v, line)? {
                let key = match &attribute {
                    Some(a) => attr_path(&item, a.as_str(), line)?,
                    None => item.clone(),
                };
                keyed.push((sort_key(&key, case_sensitive), item));
            }
            Value::list(
                sorted(keyed, reverse, line)?
                    .into_iter()
                    .map(|(_, x)| x)
                    .collect(),
            )
        }
        other => {
            // the parse refuses a filter of the reference outside the subset
            return Err(err(ErrorKind::Unsupported(unsupported_filter(other)), line));
        }
    })
}

fn unsupported_filter(name: &str) -> &'static str {
    REFERENCE_FILTERS
        .iter()
        .find(|f| **f == name)
        .copied()
        .unwrap_or("a filter")
}

/// The test `name` of `v`
pub fn test(name: &str, v: &Value, args: &[Value], line: u32) -> Res<bool> {
    plain(v, "a test", line)?;
    let other = || -> Res<&Value> {
        args.first()
            .ok_or_else(|| type_err(line, format!("the test {name} takes an argument")))
    };
    Ok(match name {
        "defined" => !matches!(v, Value::Undefined(_)),
        "undefined" => matches!(v, Value::Undefined(_)),
        "none" => matches!(v, Value::None),
        "string" => matches!(v, Value::Str(_)),
        "mapping" => matches!(v, Value::Map(_)),
        "iterable" => matches!(
            v,
            Value::Undefined(_) | Value::Str(_) | Value::List(_) | Value::Tuple(_) | Value::Map(_)
        ),
        "sequence" => matches!(
            v,
            Value::Undefined(_) | Value::Str(_) | Value::List(_) | Value::Tuple(_) | Value::Map(_)
        ),
        "boolean" => matches!(v, Value::Bool(_)),
        "true" => matches!(v, Value::Bool(true)),
        "false" => matches!(v, Value::Bool(false)),
        "equalto" => py_eq(&not_top(v.clone(), line)?, other()?),
        _ if REFERENCE_TESTS.contains(&name) => {
            return Err(err(
                ErrorKind::Unsupported("a test outside the subset"),
                line,
            ));
        }
        _ => return Err(err(ErrorKind::TestUnknown(name.into()), line)),
    })
}
