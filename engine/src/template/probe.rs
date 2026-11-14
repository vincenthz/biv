//! The probes of a chat template at load: small conversations with marker
//! values, whose renders give the texts that a chat needs, for any
//! template. A probe never guesses a text: a render that does not give one
//! clear value fails the probe by name.

use super::value::{Origin, Value};
use super::{Template, TemplateError};
use crate::prelude::*;

/// Markers that a template cannot hold: private-use chars around a letter
const MARK_USER: &str = "\u{e000}U\u{e001}";
const MARK_SECOND: &str = "\u{e000}S\u{e001}";
const MARK_ANSWER: &str = "\u{e000}A\u{e001}";
const MARK_REASONING: [&str; 2] = ["\u{e000}R\u{e001}", "\u{e000}Q\u{e001}"];
const MARK_TOOL: &str = "\u{e000}T\u{e001}";
const MARK_KEY: &str = "\u{e000}K\u{e001}";
const MARK_VALUE: &str = "\u{e000}V\u{e001}";

/// The effort words of biv, in order
pub const EFFORT_WORDS: [&str; 3] = ["low", "medium", "xhigh"];

/// Why a probe failed
#[derive(Debug, Clone, PartialEq)]
pub enum ProbeError {
    /// the template failed on a probe conversation
    Render(&'static str, TemplateError),
    /// two renders that differ only at the end of a conversation do not
    /// share their start
    TemplateNotPrefixStable(&'static str),
    /// the text after an answer is not one special token
    TemplateNoTurnEnd(String),
    /// add_generation_prompt adds nothing
    TemplateNoGenerationPrompt,
    /// the reasoning texts depend on the reasoning, or appear in no clear place
    TemplateReasoningUnclear,
    /// the effort changes the render, but no effort word of biv renders
    TemplateEffortUnclear,
    /// a tool call is written with only one of its two special tokens
    TemplateToolCallTokenMissing,
    /// a tool call is written in a form that no parser of biv reads
    ToolCallFormUnknown(String),
}

/// The texts of the reasoning block of an answer
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reasoning {
    /// before the reasoning, after the start of the answer
    pub open: String,
    /// between the reasoning and the answer
    pub close: String,
}

/// The forms of the body of a tool call that biv reads
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallForm {
    /// <function=NAME> then <parameter=KEY>VALUE</parameter>, then </function>
    XmlParameters,
    /// {"name": NAME, "arguments": {KEY: VALUE}}
    JsonObject,
}

/// The texts of a tool call
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub open: String,
    pub close: String,
    pub form: ToolCallForm,
}

/// What the probes give for a template
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatFormat {
    /// the special token that ends a turn
    pub turn_end: String,
    /// the text that add_generation_prompt adds, with thinking on and off
    pub generation_prompt_thinking: String,
    pub generation_prompt_plain: String,
    /// None: the template writes no reasoning of an earlier answer
    pub reasoning: Option<Reasoning>,
    /// the effort words of biv that the template takes; None: the effort
    /// does not change the render
    pub efforts: Option<Vec<&'static str>>,
    /// None: the template writes no tool call
    pub tool_call: Option<ToolCall>,
}

fn s(text: &str) -> Value {
    Value::str(text, Origin::Content)
}

fn message(role: &str, content: &str, extra: Vec<(Value, Value)>) -> Value {
    let mut m = vec![(s("role"), s(role)), (s("content"), s(content))];
    m.extend(extra);
    Value::map(m)
}

/// The variables of a probe render, as biv sets them
fn vars(
    messages: Vec<Value>,
    generation: bool,
    thinking: bool,
    effort: Option<&str>,
    tools: bool,
) -> Vec<(String, Value)> {
    let tool = Value::map(vec![
        (s("type"), s("function")),
        (
            s("function"),
            Value::map(vec![
                (s("name"), s(MARK_TOOL)),
                (s("description"), s("")),
                (s("parameters"), Value::map(vec![(s("type"), s("object"))])),
            ]),
        ),
    ]);
    let mut v = vec![
        ("messages".to_string(), Value::list(messages)),
        (
            "tools".to_string(),
            if tools {
                Value::list(vec![tool])
            } else {
                Value::None
            },
        ),
        ("documents".to_string(), Value::None),
        ("add_generation_prompt".to_string(), Value::Bool(generation)),
        ("bos_token".to_string(), Value::str("", Origin::Template)),
        ("eos_token".to_string(), Value::str("", Origin::Template)),
        ("enable_thinking".to_string(), Value::Bool(thinking)),
        ("preserve_thinking".to_string(), Value::Bool(true)),
    ];
    if let Some(e) = effort {
        v.push((
            "reasoning_effort".to_string(),
            Value::str(e, Origin::Template),
        ));
    }
    v
}

fn render(
    t: &Template,
    probe: &'static str,
    vars: &[(String, Value)],
) -> Result<String, ProbeError> {
    let segments = t
        .render(vars, None)
        .map_err(|e| ProbeError::Render(probe, e))?;
    Ok(segments.into_iter().map(|s| s.text).collect())
}

/// `long` less its start `short`, which it must start with
fn after<'a>(long: &'a str, short: &str, probe: &'static str) -> Result<&'a str, ProbeError> {
    long.strip_prefix(short)
        .ok_or(ProbeError::TemplateNotPrefixStable(probe))
}

impl ChatFormat {
    /// The probes of `t`. `is_special` tells whether a text is one special
    /// token of the vocabulary of the model.
    pub fn probe(
        t: &Template,
        is_special: &dyn Fn(&str) -> bool,
    ) -> Result<ChatFormat, ProbeError> {
        let user = || message("user", MARK_USER, Vec::new());
        // P2: the generation prompt
        let no_prompt = render(
            t,
            "generation",
            &vars(vec![user()], false, false, None, false),
        )?;
        let plain = render(
            t,
            "generation",
            &vars(vec![user()], true, false, None, false),
        )?;
        let thinking = render(
            t,
            "generation",
            &vars(vec![user()], true, true, Some(EFFORT_WORDS[1]), false),
        )?;
        let generation_prompt_plain = after(&plain, &no_prompt, "generation")?.to_string();
        let generation_prompt_thinking = after(&thinking, &no_prompt, "generation")?.to_string();
        if generation_prompt_plain.is_empty() {
            return Err(ProbeError::TemplateNoGenerationPrompt);
        }
        // P1: the turn end, after an answer and before the next message
        let answered = |reasoning: Option<&str>| {
            let mut extra = Vec::new();
            if let Some(r) = reasoning {
                extra.push((s("reasoning_content"), s(r)));
            }
            vec![
                user(),
                message("assistant", MARK_ANSWER, extra),
                message("user", MARK_SECOND, Vec::new()),
            ]
        };
        let turn = render(
            t,
            "turn end",
            &vars(answered(None), false, false, None, false),
        )?;
        let answer_end = turn
            .find(MARK_ANSWER)
            .ok_or(ProbeError::TemplateNoTurnEnd(String::new()))?
            + MARK_ANSWER.len();
        let second = turn
            .find(MARK_SECOND)
            .ok_or(ProbeError::TemplateNoTurnEnd(String::new()))?;
        let between = &turn[answer_end..second];
        let turn_end = between
            .split_whitespace()
            .find(|w| !w.is_empty())
            .unwrap_or_default()
            .to_string();
        // the text after the answer starts with one special token
        let starts = between.trim_start();
        if turn_end.is_empty() || !is_special(&turn_end) || !starts.starts_with(&turn_end) {
            return Err(ProbeError::TemplateNoTurnEnd(between.to_string()));
        }
        // P3: the reasoning of an earlier answer, with two markers
        let mut found: Vec<Option<Reasoning>> = Vec::new();
        for marker in MARK_REASONING {
            let text = render(
                t,
                "reasoning",
                &vars(
                    answered(Some(marker)),
                    false,
                    true,
                    Some(EFFORT_WORDS[1]),
                    false,
                ),
            )?;
            let Some(r_at) = text.find(marker) else {
                found.push(None);
                continue;
            };
            let a_at = text
                .find(MARK_ANSWER)
                .ok_or(ProbeError::TemplateReasoningUnclear)?;
            let u_end = text
                .find(MARK_USER)
                .ok_or(ProbeError::TemplateReasoningUnclear)?
                + MARK_USER.len();
            if !(u_end <= r_at && r_at < a_at) {
                return Err(ProbeError::TemplateReasoningUnclear);
            }
            let close = text[r_at + marker.len()..a_at].to_string();
            // the open text: the last special token before the reasoning in
            // the turn of the answer, with only whitespace after it
            let block = &text[u_end..r_at];
            let turn = &block[block.rfind(&turn_end).map_or(0, |e| e + turn_end.len())..];
            let open =
                last_special_open(turn, is_special).ok_or(ProbeError::TemplateReasoningUnclear)?;
            found.push(Some(Reasoning { open, close }));
        }
        let reasoning = match (&found[0], &found[1]) {
            (a, b) if a == b => a.clone(),
            _ => return Err(ProbeError::TemplateReasoningUnclear),
        };
        // P4: the efforts
        let mut renders = Vec::new();
        for word in EFFORT_WORDS {
            renders.push((
                word,
                t.render(&vars(vec![user()], true, true, Some(word), false), None),
            ));
        }
        let ok: Vec<(&'static str, String)> = renders
            .iter()
            .filter_map(|(w, r)| {
                r.as_ref()
                    .ok()
                    .map(|seg| (*w, seg.iter().map(|s| s.text.as_str()).collect()))
            })
            .collect();
        let efforts = if ok.len() == EFFORT_WORDS.len() && ok.windows(2).all(|w| w[0].1 == w[1].1) {
            None
        } else if ok.is_empty() {
            return Err(ProbeError::TemplateEffortUnclear);
        } else {
            Some(ok.iter().map(|(w, _)| *w).collect())
        };
        // P5: the tool call
        let call = Value::map(vec![
            (s("type"), s("function")),
            (
                s("function"),
                Value::map(vec![
                    (s("name"), s(MARK_TOOL)),
                    (
                        s("arguments"),
                        Value::map(vec![(s(MARK_KEY), s(MARK_VALUE))]),
                    ),
                ]),
            ),
        ]);
        let calling = vec![
            user(),
            message(
                "assistant",
                "",
                vec![(s("tool_calls"), Value::list(vec![call]))],
            ),
            message("user", MARK_SECOND, Vec::new()),
        ];
        let text = render(t, "tool call", &vars(calling, false, false, None, true))?;
        let tool_call = tool_call_of(&text, is_special)?;
        Ok(ChatFormat {
            turn_end,
            generation_prompt_thinking,
            generation_prompt_plain,
            reasoning,
            efforts,
            tool_call,
        })
    }
}

/// The last special token of `turn` that only whitespace follows, with
/// that whitespace
fn last_special_open(turn: &str, is_special: &dyn Fn(&str) -> bool) -> Option<String> {
    let bounds: Vec<usize> = (0..=turn.len())
        .filter(|i| turn.is_char_boundary(*i))
        .collect();
    for &i in bounds.iter().rev() {
        for &j in bounds.iter().filter(|j| **j > i) {
            if turn[j..].trim().is_empty() && is_special(&turn[i..j]) {
                return Some(turn[i..].to_string());
            }
        }
    }
    None
}

/// The tool call of the render `text` of the probe call, between the user
/// marker and the second user marker
fn tool_call_of(
    text: &str,
    is_special: &dyn Fn(&str) -> bool,
) -> Result<Option<ToolCall>, ProbeError> {
    let (Some(u), Some(second)) = (text.find(MARK_USER), text.find(MARK_SECOND)) else {
        return Ok(None);
    };
    let region = &text[u + MARK_USER.len()..second];
    let Some(name_at) = region.find(MARK_TOOL) else {
        return Ok(None);
    };
    let value_end = region
        .find(MARK_VALUE)
        .map(|v| v + MARK_VALUE.len())
        .ok_or(ProbeError::ToolCallFormUnknown(region.into()))?;
    let xml_start = format!("<function={MARK_TOOL}>");
    let (form, body_start, body_end) =
        if region[..name_at + MARK_TOOL.len() + 1].ends_with(&xml_start) {
            let end = region[value_end..]
                .find("</function>")
                .map(|e| value_end + e + "</function>".len())
                .ok_or(ProbeError::ToolCallFormUnknown(region.into()))?;
            (
                ToolCallForm::XmlParameters,
                name_at + MARK_TOOL.len() + 1 - xml_start.len(),
                end,
            )
        } else if let Some(open) = region[..name_at].rfind('{') {
            let end = region[value_end..]
                .find("}}")
                .map(|e| value_end + e + "}}".len())
                .ok_or(ProbeError::ToolCallFormUnknown(region.into()))?;
            (ToolCallForm::JsonObject, open, end)
        } else {
            return Err(ProbeError::ToolCallFormUnknown(region.into()));
        };
    // the special tokens just before and just after the body
    let before = region[..body_start].trim_end();
    let after = region[body_end..].trim_start();
    let bounds =
        |t: &str| -> Vec<usize> { (0..=t.len()).filter(|i| t.is_char_boundary(*i)).collect() };
    // the longest special token that ends the text before the body, and the
    // longest that starts the text after it
    let open = bounds(before)
        .into_iter()
        .find(|i| *i < before.len() && is_special(&before[*i..]))
        .map_or("", |i| &before[i..]);
    let close = bounds(after)
        .into_iter()
        .rev()
        .find(|j| *j > 0 && is_special(&after[..*j]))
        .map_or("", |j| &after[..j]);
    let open_special = !open.is_empty();
    let close_special = !close.is_empty();
    match (open_special, close_special) {
        (true, true) => Ok(Some(ToolCall {
            open: open.into(),
            close: close.into(),
            form,
        })),
        (false, false) => Err(ProbeError::ToolCallFormUnknown(region.into())),
        _ => Err(ProbeError::TemplateToolCallTokenMissing),
    }
}
