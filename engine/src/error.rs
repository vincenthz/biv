//! The errors of every fallible step: parsing a GGUF, loading a model and its
//! tokenizer from it, starting a backend or a session, and chatting.

use core::fmt;

use crate::prelude::*;
use crate::quant::GgmlType;
use crate::template::json::JsonError;
use crate::template::{ProbeError, TemplateError, ToolCallForm};

#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// The GGUF data ends in the middle of a field
    Truncated,
    /// The data is not a GGUF file
    BadMagic,
    UnsupportedVersion(u32),
    /// A metadata value of an unknown type
    UnknownValueType(u32),
    /// A tensor of a type this crate doesn't know
    UnknownTensorType {
        tensor: String,
        ty: u32,
    },
    /// A tensor whose data lies past the end of its shard
    TensorOutOfBounds(String),

    /// A metadata key the model needs is missing, or has the wrong type
    MissingKey(String),
    MissingTensor(String),
    /// A tensor of another shape than the model's hyperparameters give
    TensorShape {
        tensor: String,
        dims: Vec<u64>,
        expected: Vec<u64>,
    },
    /// A weight of a type that can be dequantized but not multiplied
    NoKernel {
        tensor: String,
        ty: GgmlType,
    },
    /// `general.architecture` is not `qwen35` or `qwen35moe`
    UnsupportedArchitecture(String),
    /// The tokenizer is not GPT-2 style BPE with the `qwen35` pre-tokenizer
    UnsupportedTokenizer {
        model: String,
        pre: String,
    },
    /// A combination of hyperparameters the engine has no code for
    UnsupportedModel(&'static str),
    /// A token the chat template or the byte-level BPE needs
    MissingToken(String),

    /// An allocation failed
    OutOfMemory {
        bytes: usize,
    },
    /// The message doesn't fit in what is left of the context
    ContextFull {
        n_past: usize,
        prompt: usize,
        context: usize,
    },
    /// A snapshot that can't be restored in this session
    Snapshot(&'static str),
    /// A sampler seed of 0: the caller names the seed
    ZeroSeed,
    /// A sampler state of 0, which the random generator never holds
    ZeroRngState,
    /// The token mask of the sampler allows no token, so no token can
    /// come next
    MaskAllowsNoToken,
    /// A turn is open: step it to its end or cancel it first
    TurnOpen,
    /// No turn is open to step or cancel
    NoTurn,
    /// A transcript that can't be read, or restored in this chat
    Transcript(&'static str),
    /// A transcript of version 1, which holds the tokens of a conversation
    /// but not its messages, so a chat cannot render it
    TranscriptNoMessages,
    /// The model file has no chat template (`tokenizer.chat_template`)
    TemplateMissing,
    /// The chat template does not load, or fails on a render
    Template(TemplateError),
    /// A probe of the chat template gives no clear text
    Probe(ProbeError),
    /// A tool of the chat is not JSON
    ToolDefinition {
        index: usize,
        error: JsonError,
    },
    /// A chat with tools on a template that writes no tool call
    ToolsUnsupported,
    /// A chat with tools on a template whose tool calls have a form that
    /// Chat does not read
    ToolCallForm(ToolCallForm),
    /// A turn with thinking on a template whose generation prompts give no
    /// special token that ends the reasoning
    ThinkingUnsupported,
    /// A reasoning effort whose word the template does not take
    EffortUnsupported(&'static str),
    /// The GPU backend failed: no device, shader compilation, ...
    Backend(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Truncated => f.write_str("gguf: unexpected end of data"),
            Error::BadMagic => f.write_str("gguf: bad magic, not a GGUF file"),
            Error::UnsupportedVersion(v) => write!(f, "gguf: unsupported version {v}"),
            Error::UnknownValueType(ty) => write!(f, "gguf: unknown value type {ty}"),
            Error::UnknownTensorType { tensor, ty } => {
                write!(f, "gguf: tensor {tensor}: unsupported type {ty}")
            }
            Error::TensorOutOfBounds(tensor) => write!(f, "gguf: tensor {tensor} out of bounds"),
            Error::MissingKey(key) => write!(f, "gguf: missing key {key}"),
            Error::MissingTensor(tensor) => write!(f, "missing tensor {tensor}"),
            Error::TensorShape {
                tensor,
                dims,
                expected,
            } => {
                write!(f, "{tensor}: shape {dims:?}, expected {expected:?}")
            }
            Error::NoKernel { tensor, ty } => write!(f, "{tensor}: no matmul kernel for {ty:?}"),
            Error::UnsupportedArchitecture(arch) => {
                write!(
                    f,
                    "unsupported architecture {arch} (expected qwen35 or qwen35moe)"
                )
            }
            Error::UnsupportedTokenizer { model, pre } => {
                write!(
                    f,
                    "unsupported tokenizer {model} / {pre} (expected gpt2 / qwen35)"
                )
            }
            Error::UnsupportedModel(what) => write!(f, "unsupported model: {what}"),
            Error::MissingToken(token) => write!(f, "vocabulary has no {token:?} token"),
            Error::OutOfMemory { bytes } => write!(f, "cannot allocate {} MB", bytes >> 20),
            Error::ContextFull {
                n_past,
                prompt,
                context,
            } => {
                write!(f, "context full ({n_past} + {prompt} tokens > {context})")
            }
            Error::Snapshot(why) => write!(f, "cannot restore the snapshot: {why}"),
            Error::ZeroSeed => f.write_str("sampler seed 0: name a seed"),
            Error::ZeroRngState => f.write_str("sampler state 0: not a state of the generator"),
            Error::MaskAllowsNoToken => f.write_str("sampler: the token mask allows no token"),
            Error::TurnOpen => f.write_str("chat: a turn is open"),
            Error::NoTurn => f.write_str("chat: no turn is open"),
            Error::Transcript(why) => write!(f, "invalid transcript: {why}"),
            Error::TranscriptNoMessages => {
                f.write_str("transcript of version 1: it holds no messages to render")
            }
            Error::TemplateMissing => f.write_str("the model file has no chat template"),
            Error::Template(e) => write!(f, "chat template: {e}"),
            Error::Probe(e) => write!(f, "chat template probe: {e:?}"),
            Error::ToolDefinition { index, error } => write!(
                f,
                "tool {index} is not JSON: {} at byte {}",
                error.what, error.at
            ),
            Error::ToolsUnsupported => f.write_str("the chat template writes no tool call"),
            Error::ToolCallForm(form) => {
                write!(f, "chat: no reader for tool calls of the form {form:?}")
            }
            Error::ThinkingUnsupported => {
                f.write_str("the chat template gives no special token that ends the reasoning")
            }
            Error::EffortUnsupported(word) => {
                write!(
                    f,
                    "the chat template does not take the reasoning effort {word}"
                )
            }
            Error::Backend(msg) => f.write_str(msg),
        }
    }
}

impl core::error::Error for Error {}
