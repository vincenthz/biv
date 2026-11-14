//! The conversation of a chat as its chat template reads it: the system
//! prompt, the tools, and the messages with their roles. A chat renders all
//! of it for each turn, see [`crate::chat`].

use crate::Error;
use crate::chat::ToolCall;
use crate::prelude::*;
use crate::template::{Origin, Value, json};

/// A message of a conversation, with the fields that the template reads
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Said {
    User(String),
    /// an answer: its reasoning (`None` for an answer given without it),
    /// its text after the reasoning, and its function calls
    Assistant {
        reasoning: Option<String>,
        answer: String,
        tool_calls: Vec<ToolCall>,
    },
    /// the result of a function call
    Tool(String),
}

/// A conversation: what a chat renders for each turn
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conversation {
    /// the system prompt, the first message when set
    pub system: Option<String>,
    /// functions the model may call, each a JSON object
    pub tools: Vec<String>,
    pub messages: Vec<Said>,
}

/// Text of a caller, which stays plain text in the prompt
fn text(s: &str) -> Value {
    Value::str(s, Origin::Content)
}

fn message(role: &str, content: &str, mut fields: Vec<(Value, Value)>) -> Value {
    let mut m = vec![(text("role"), text(role)), (text("content"), text(content))];
    m.append(&mut fields);
    Value::map(m)
}

/// A call as the template reads it. Each argument is the string that the
/// model wrote, so the template writes the bytes of the model.
fn call(c: &ToolCall) -> Value {
    let arguments = c
        .arguments
        .iter()
        .map(|(k, v)| (text(k), text(v)))
        .collect();
    let function = Value::map(vec![
        (text("name"), text(&c.name)),
        (text("arguments"), Value::map(arguments)),
    ]);
    Value::map(vec![
        (text("type"), text("function")),
        (text("function"), function),
    ])
}

impl Conversation {
    /// A conversation with no messages yet
    pub fn new(system: Option<String>, tools: Vec<String>) -> Self {
        Conversation {
            system,
            tools,
            messages: Vec::new(),
        }
    }

    /// Whether a render has a message: the template of a conversation with
    /// none has no text to give
    pub fn has_messages(&self) -> bool {
        self.system.is_some() || !self.messages.is_empty()
    }

    /// The messages, then the tools (`None` when there are none), as the
    /// template reads them. Fails with [`Error::ToolDefinition`] when a tool
    /// is not JSON.
    pub(crate) fn values(&self) -> Result<(Value, Value), Error> {
        let mut messages = Vec::with_capacity(self.messages.len() + 1);
        if let Some(s) = &self.system {
            messages.push(message("system", s, Vec::new()));
        }
        for m in &self.messages {
            messages.push(match m {
                Said::User(u) => message("user", u, Vec::new()),
                Said::Tool(r) => message("tool", r, Vec::new()),
                Said::Assistant {
                    reasoning,
                    answer,
                    tool_calls,
                } => {
                    let mut fields = Vec::new();
                    if let Some(r) = reasoning {
                        fields.push((text("reasoning_content"), text(r)));
                    }
                    if !tool_calls.is_empty() {
                        let calls = tool_calls.iter().map(call).collect();
                        fields.push((text("tool_calls"), Value::list(calls)));
                    }
                    message("assistant", answer, fields)
                }
            });
        }
        let tools = if self.tools.is_empty() {
            Value::None
        } else {
            let mut values = Vec::with_capacity(self.tools.len());
            for (index, t) in self.tools.iter().enumerate() {
                let v = json::parse(t, Origin::Content)
                    .map_err(|error| Error::ToolDefinition { index, error })?;
                values.push(v);
            }
            Value::list(values)
        };
        Ok((Value::list(messages), tools))
    }
}
