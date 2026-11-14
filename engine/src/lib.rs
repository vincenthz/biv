//! Pure Rust inference for Qwen3.8 (llama.cpp `qwen35` and `qwen35moe` GGUF models), on CPU or Metal.
//!
//! No I/O: a model is parsed from the bytes of its GGUF, however the caller
//! got them (mmap, embedded, read into memory). The crate is `no_std` + `alloc`
//! without its default features; `std` brings std's float functions, `rayon` the
//! multithreaded CPU compute, `metal` the Metal backend.

#![no_std]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "std")]
pub mod batch;
pub mod chat;
pub mod conversation;
pub mod engine;
mod error;
pub mod gguf;
pub mod memory;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;
pub mod model;
pub mod ops;
pub mod quant;
pub mod sampler;
pub mod template;
pub mod tokenizer;
pub mod transcript;

mod math;
mod par;

pub use error::Error;

/// What std's prelude would bring
mod prelude {
    pub(crate) use alloc::boxed::Box;
    pub(crate) use alloc::string::{String, ToString};
    pub(crate) use alloc::vec::Vec;
    pub(crate) use alloc::{format, vec};
}
