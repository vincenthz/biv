//! The errors of every fallible step: parsing a GGUF, loading a model and its
//! tokenizer from it, starting a backend or a session, and chatting.

use core::fmt;

use crate::prelude::*;
use crate::quant::GgmlType;

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
    /// `general.architecture` is not `qwen35`, `qwen35moe` or `flint`
    UnsupportedArchitecture(String),
    /// The tokenizer is not GPT-2 style BPE with the `qwen35` or `flint` pre-tokenizer
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
    /// A transcript that can't be read, or restored in this chat
    Transcript(&'static str),
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
                    "unsupported architecture {arch} (expected qwen35, qwen35moe or flint)"
                )
            }
            Error::UnsupportedTokenizer { model, pre } => {
                write!(
                    f,
                    "unsupported tokenizer {model} / {pre} (expected gpt2 / qwen35 or flint)"
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
            Error::Transcript(why) => write!(f, "invalid transcript: {why}"),
            Error::Backend(msg) => f.write_str(msg),
        }
    }
}

impl core::error::Error for Error {}
