//! The lexer of templates, with the settings of the reference renderer:
//! trim_blocks, lstrip_blocks, and no trailing newline kept.

use super::{ErrorKind, TemplateError};
use crate::prelude::*;

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    /// template text between tags
    Data(String),
    VarStart,
    VarEnd,
    BlockStart,
    BlockEnd,
    Name(String),
    Str(String),
    Int(i64),
    Float(f64),
    Op(&'static str),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub tok: Tok,
    pub line: u32,
    pub column: u32,
}

/// The operators, longest first, so that the first match is the longest
const OPS: [&str; 26] = [
    "//", "**", "==", "!=", "<=", ">=", "+", "-", "*", "/", "%", "~", "<", ">", "=", "(", ")", "[",
    "]", "{", "}", ",", ".", ":", "|", ";",
];

/// The line breaks of Python's str.splitlines, which the reference lexer
/// turns into "\n"
fn is_line_break(c: char) -> bool {
    matches!(
        c,
        '\n' | '\r'
            | '\u{b}'
            | '\u{c}'
            | '\u{1c}'
            | '\u{1d}'
            | '\u{1e}'
            | '\u{85}'
            | '\u{2028}'
            | '\u{2029}'
    )
}

/// The source as the reference lexer reads it: each line break of
/// splitlines is "\n", and one trailing line break is gone
pub fn normalize(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut chars = source.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\r' && chars.peek() == Some(&'\n') {
            chars.next();
            out.push('\n');
        } else if is_line_break(c) {
            out.push('\n');
        } else {
            out.push(c);
        }
    }
    if out.ends_with('\n') {
        out.pop();
    }
    out
}

struct Lexer<'a> {
    src: &'a str,
    at: usize,
    out: Vec<Token>,
}

/// What strips the start of the text after a tag
#[derive(Debug, Clone, Copy, PartialEq)]
enum After {
    /// nothing: a variable tag, or a tag that ends with +
    Keep,
    /// one newline: the trim_blocks of a block or a comment
    TrimNewline,
    /// all whitespace: a tag that ends with -
    StripSpace,
}

impl<'a> Lexer<'a> {
    fn position(&self, at: usize) -> (u32, u32) {
        let before = &self.src[..at];
        let line = before.matches('\n').count() + 1;
        let column = before.rfind('\n').map_or(at, |n| at - n - 1) + 1;
        (line as u32, column as u32)
    }

    fn err(&self, at: usize, kind: ErrorKind) -> TemplateError {
        let (line, column) = self.position(at);
        TemplateError { kind, line, column }
    }

    fn push(&mut self, at: usize, tok: Tok) {
        let (line, column) = self.position(at);
        self.out.push(Token { tok, line, column });
    }

    fn run(mut self) -> Result<Vec<Token>, TemplateError> {
        let mut after = After::Keep;
        // the text starts a line: at the start of the template, or after a
        // tag whose trim took the newline
        let mut line_start = true;
        while self.at < self.src.len() {
            let rest = &self.src[self.at..];
            let next = ["{{", "{%", "{#"]
                .iter()
                .filter_map(|open| rest.find(open))
                .min();
            let data_end = next.map_or(self.src.len(), |n| self.at + n);
            let mut data = &self.src[self.at..data_end];
            let data_at = self.at;
            match after {
                After::Keep => {}
                After::TrimNewline => data = data.strip_prefix('\n').unwrap_or(data),
                After::StripSpace => data = data.trim_start(),
            }
            let Some(_) = next else {
                if !data.is_empty() {
                    self.push(data_at, Tok::Data(data.into()));
                }
                break;
            };
            let open = &self.src[data_end..data_end + 2];
            let modifier = self.src[data_end + 2..].chars().next();
            let is_block = open == "{%" || open == "{#";
            let mut text = data;
            if modifier == Some('-') {
                text = text.trim_end();
            } else if is_block && modifier != Some('+') {
                // lstrip_blocks: spaces and tabs before a block that starts
                // its line
                let line_from = text.rfind('\n').map(|n| n + 1);
                let tail = &text[line_from.unwrap_or(0)..];
                if (line_from.is_some() || line_start)
                    && tail.chars().all(|c| c == ' ' || c == '\t')
                {
                    text = &text[..text.len() - tail.len()];
                }
            }
            if !text.is_empty() {
                self.push(data_at, Tok::Data(text.into()));
            }
            self.at = data_end + 2;
            if matches!(modifier, Some('-') | Some('+')) {
                self.at += 1;
            }
            after = match open {
                "{#" => self.comment(data_end)?,
                "{{" => {
                    self.push(data_end, Tok::VarStart);
                    self.expression("}}", data_end)?
                }
                _ => {
                    self.push(data_end, Tok::BlockStart);
                    self.expression("%}", data_end)?
                }
            };
            line_start = after == After::TrimNewline && self.src[self.at..].starts_with('\n');
        }
        Ok(self.out)
    }

    /// The end of a tag at `self.at`: its modifier and what it strips after
    fn tag_end(&mut self, close: &str, block: bool) -> Option<After> {
        let rest = &self.src[self.at..];
        for (modifier, after) in [
            ("-", After::StripSpace),
            ("+", After::Keep),
            (
                "",
                if block {
                    After::TrimNewline
                } else {
                    After::Keep
                },
            ),
        ] {
            if rest.starts_with(modifier) && rest[modifier.len()..].starts_with(close) {
                // a lone + before the close is only a modifier of a block
                if modifier == "+" && !block {
                    continue;
                }
                self.at += modifier.len() + close.len();
                return Some(after);
            }
        }
        None
    }

    fn comment(&mut self, open_at: usize) -> Result<After, TemplateError> {
        let rest = &self.src[self.at..];
        let Some(end) = rest.find("#}") else {
            return Err(self.err(open_at, ErrorKind::Syntax("a comment with no end")));
        };
        let body_end = self.at + end;
        let stripped = self.src[..body_end].ends_with('-') && body_end > self.at;
        self.at = body_end + 2;
        Ok(if stripped {
            After::StripSpace
        } else {
            After::TrimNewline
        })
    }

    fn expression(&mut self, close: &'static str, open_at: usize) -> Result<After, TemplateError> {
        let block = close == "%}";
        // the open brackets: the close of the tag counts only outside them
        let mut depth = 0usize;
        loop {
            // whitespace between tokens
            let rest = &self.src[self.at..];
            let skipped = rest.len() - rest.trim_start().len();
            self.at += skipped;
            if self.at >= self.src.len() {
                return Err(self.err(open_at, ErrorKind::Syntax("a tag with no end")));
            }
            let start = self.at;
            if depth == 0
                && let Some(after) = self.tag_end(close, block)
            {
                self.push(start, if block { Tok::BlockEnd } else { Tok::VarEnd });
                return Ok(after);
            }
            let rest = &self.src[self.at..];
            let c = rest.chars().next().expect("text is left");
            if c.is_ascii_alphabetic() || c == '_' {
                let len = rest
                    .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                    .unwrap_or(rest.len());
                self.at += len;
                self.push(start, Tok::Name(rest[..len].into()));
            } else if c.is_ascii_digit() {
                let tok = self.number()?;
                self.push(start, tok);
            } else if c == '\'' || c == '"' {
                let s = self.string(c)?;
                self.push(start, Tok::Str(s));
            } else if let Some(op) = OPS.iter().find(|op| rest.starts_with(**op)) {
                match *op {
                    "(" | "[" | "{" => depth += 1,
                    ")" | "]" | "}" => {
                        depth = depth.checked_sub(1).ok_or_else(|| {
                            self.err(start, ErrorKind::Syntax("a close with no open"))
                        })?;
                    }
                    _ => {}
                }
                self.at += op.len();
                self.push(start, Tok::Op(op));
            } else {
                return Err(self.err(start, ErrorKind::Syntax("an unknown character")));
            }
        }
    }

    fn number(&mut self) -> Result<Tok, TemplateError> {
        let start = self.at;
        let bytes = self.src.as_bytes();
        let digits = |at: &mut usize| {
            while *at < bytes.len() && (bytes[*at].is_ascii_digit() || bytes[*at] == b'_') {
                *at += 1;
            }
        };
        let mut at = self.at;
        digits(&mut at);
        let mut float = false;
        if at + 1 < bytes.len() && bytes[at] == b'.' && bytes[at + 1].is_ascii_digit() {
            float = true;
            at += 1;
            digits(&mut at);
        }
        if at < bytes.len() && (bytes[at] == b'e' || bytes[at] == b'E') {
            let mut e = at + 1;
            if e < bytes.len() && (bytes[e] == b'+' || bytes[e] == b'-') {
                e += 1;
            }
            if e < bytes.len() && bytes[e].is_ascii_digit() {
                float = true;
                at = e;
                digits(&mut at);
            }
        }
        self.at = at;
        let text: String = self.src[start..at].chars().filter(|c| *c != '_').collect();
        let bad = || self.err(start, ErrorKind::Syntax("a number"));
        if float {
            text.parse().map(Tok::Float).map_err(|_| bad())
        } else {
            text.parse().map(Tok::Int).map_err(|_| bad())
        }
    }

    /// A string literal, unescaped as Python's unicode-escape codec does
    fn string(&mut self, quote: char) -> Result<String, TemplateError> {
        let start = self.at;
        self.at += 1;
        let mut out = String::new();
        let mut chars = self.src[self.at..].char_indices();
        while let Some((i, c)) = chars.next() {
            if c == quote {
                self.at += i + 1;
                return Ok(out);
            }
            if c != '\\' {
                out.push(c);
                continue;
            }
            let Some((_, e)) = chars.next() else { break };
            match e {
                '\n' => {}
                '\\' => out.push('\\'),
                '\'' => out.push('\''),
                '"' => out.push('"'),
                'a' => out.push('\u{7}'),
                'b' => out.push('\u{8}'),
                'f' => out.push('\u{c}'),
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                'v' => out.push('\u{b}'),
                'x' | 'u' | 'U' => {
                    let len = match e {
                        'x' => 2,
                        'u' => 4,
                        _ => 8,
                    };
                    let hex: String = chars.by_ref().take(len).map(|(_, c)| c).collect();
                    let code = (hex.len() == len)
                        .then(|| u32::from_str_radix(&hex, 16).ok())
                        .flatten()
                        .and_then(char::from_u32)
                        .ok_or_else(|| self.err(start, ErrorKind::Syntax("a string escape")))?;
                    out.push(code);
                }
                '0'..='7' => {
                    let mut code = e.to_digit(8).expect("an octal digit");
                    for _ in 0..2 {
                        match chars.clone().next() {
                            Some((_, d)) if d.is_digit(8) => {
                                chars.next();
                                code = code * 8 + d.to_digit(8).expect("an octal digit");
                            }
                            _ => break,
                        }
                    }
                    out.push(char::from_u32(code).expect("an octal escape is below 512"));
                }
                'N' => {
                    return Err(self.err(start, ErrorKind::Unsupported("a named unicode escape")));
                }
                other => {
                    out.push('\\');
                    out.push(other);
                }
            }
        }
        Err(self.err(start, ErrorKind::Syntax("a string with no end")))
    }
}

/// The tokens of a normalized source
pub fn lex(src: &str) -> Result<Vec<Token>, TemplateError> {
    Lexer {
        src,
        at: 0,
        out: Vec::new(),
    }
    .run()
}
