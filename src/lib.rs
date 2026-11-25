//! What the command line tools add around `biv`: model files on disk,
//! memory-mapped, and the Hugging Face cache.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::{fmt, io};

use biv::gguf::{Gguf, Shard};
use biv::tokenizer::Tokenizer;
use memmap2::Mmap;

pub mod bench;
pub mod fingerprint;
pub mod hf;

/// A model file that cannot be used, and why
#[derive(Debug)]
pub enum Error {
    /// The file (or one of its shards) cannot be opened or mapped
    Io { path: PathBuf, source: io::Error },
    /// The file is not a GGUF the engine can read
    Gguf { path: PathBuf, source: biv::Error },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io { path, source } => write!(f, "{}: {source}", path.display()),
            Error::Gguf { path, source } => write!(f, "{}: {source}", path.display()),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io { source, .. } => Some(source),
            Error::Gguf { source, .. } => Some(source),
        }
    }
}

/// All the shard paths of a (possibly split) GGUF, given any of its shards
fn shard_paths(path: &Path) -> Vec<PathBuf> {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    // <prefix>-00001-of-00002.gguf
    if let Some(stem) = name.strip_suffix(".gguf")
        && let Some((head, total)) = stem.rsplit_once("-of-")
        && let Some((prefix, _idx)) = head.rsplit_once('-')
        && let Ok(n) = total.parse::<usize>()
    {
        let width = total.len();
        return (1..=n)
            .map(|i| path.with_file_name(format!("{prefix}-{i:0width$}-of-{total}.gguf")))
            .collect();
    }
    vec![path.to_path_buf()]
}

fn map(path: &Path) -> Result<Mmap, Error> {
    let io = |source| Error::Io {
        path: path.to_path_buf(),
        source,
    };
    let file = File::open(path).map_err(io)?;
    // SAFETY: the model files are read-only for the lifetime of the process
    unsafe { Mmap::map(&file) }.map_err(io)
}

/// Map a GGUF file, and the other shards of a split one
pub fn open_gguf(path: impl AsRef<Path>) -> Result<Gguf, Error> {
    let path = path.as_ref();
    let mut shards: Vec<Shard> = Vec::new();
    for p in shard_paths(path) {
        shards.push(Box::new(map(&p)?));
    }
    Gguf::from_shards(shards).map_err(|source| Error::Gguf {
        path: path.to_path_buf(),
        source,
    })
}

/// The tokenizer of a GGUF file, without the model: only the metadata at the
/// start of the first shard is read from disk, not the tensors
pub fn open_tokenizer(path: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let path = path.as_ref();
    let first = shard_paths(path).swap_remove(0);
    Tokenizer::from_gguf_bytes(&map(&first)?).map_err(|source| Error::Gguf {
        path: path.to_path_buf(),
        source,
    })
}

/// The GPU engine of this build for passes of up to `max_batch` tokens:
/// Metal on macOS, CUDA with the `cuda` feature; None without one
pub fn gpu_engine<'m>(
    model: &'m biv::model::Model<'m>,
    max_batch: usize,
) -> Result<Option<Box<dyn biv::engine::Engine + 'm>>, biv::Error> {
    #[cfg(target_os = "macos")]
    return Ok(Some(Box::new(biv::metal::MetalEngine::new(
        model, max_batch,
    )?)));
    #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
    return Ok(Some(Box::new(biv::cuda::CudaEngine::new(
        model, max_batch,
    )?)));
    #[cfg(not(any(target_os = "macos", feature = "cuda")))]
    {
        let _ = (model, max_batch);
        Ok(None)
    }
}
