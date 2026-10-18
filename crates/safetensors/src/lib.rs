//! Validated `SafeTensors` metadata and bounded encoded-payload readers.
//!
//! File-backed artifacts retain one opened file, not its payload in host heap.
//! In-memory artifacts explicitly retain supplied bytes. Both use the upstream
//! metadata validator and expose identical format-level access; models own
//! parameter interpretation, decoding, placement and backend materialization.
//!
//! Files must remain unmodified in place through every alias while any artifact
//! or reader exists. Path replacement/unlink cannot redirect an opened source.
//! This is descriptor retention, not an enforced immutable filesystem snapshot.

mod index;
mod reader;

use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ribn_foundation::ScalarType;
use safetensors::tensor::Dtype;

pub use reader::ArtifactReader;
use reader::{FileCursor, Storage};

/// Validated artifact with shared metadata and one retained byte/file source.
#[derive(Clone)]
pub struct SafeTensorArtifact {
    inner: Arc<ArtifactData>,
}

struct ArtifactData {
    storage: Storage,
    index: BTreeMap<String, TensorEntry>,
    source: Option<PathBuf>,
    len: usize,
}

struct TensorEntry {
    scalar_type: ScalarType,
    shape: Vec<usize>,
    range: Range<usize>,
}

impl SafeTensorArtifact {
    /// Validate and retain supplied in-memory bytes, including their full payload.
    ///
    /// # Errors
    /// Rejects invalid `SafeTensors` metadata, layout or payload length.
    pub fn from_bytes(bytes: impl Into<Arc<[u8]>>) -> Result<Self, ArtifactError> {
        let bytes = bytes.into();
        let prefix = bytes
            .get(..8)
            .ok_or_else(|| ArtifactError::format("header too short"))?;
        let header_len = index::header_len(prefix, bytes.len())?;
        let index = index::parse(&bytes[8..8 + header_len], bytes.len())?;
        Ok(Self {
            inner: Arc::new(ArtifactData {
                len: bytes.len(),
                storage: Storage::Bytes(bytes),
                index,
                source: None,
            }),
        })
    }

    /// Open and validate only the header, retaining the selected file for reads.
    /// Header length is capped at the upstream format limit of 100,000,000 bytes.
    /// Payload reads allocate no internal buffers. Clones/readers share the file
    /// but have independent cursors and keep it alive after this owner is dropped.
    ///
    /// # Errors
    /// Rejects I/O failures, invalid metadata/layout and incomplete/trailing payloads.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, ArtifactError> {
        let path = path.into();
        let io_error = |error: std::io::Error| ArtifactError::Io {
            path: path.clone(),
            message: error.to_string(),
        };
        let mut file = File::open(&path).map_err(io_error)?;
        let metadata = file.metadata().map_err(io_error)?;
        if !metadata.is_file() {
            return Err(ArtifactError::format("source is not a regular file"));
        }
        let len = usize::try_from(metadata.len()).map_err(ArtifactError::format)?;
        if len < 8 {
            return Err(ArtifactError::format("header too short"));
        }
        let mut prefix = [0; 8];
        file.read_exact(&mut prefix).map_err(io_error)?;
        let header_len = index::header_len(&prefix, len)?;
        let mut header = vec![0; header_len];
        file.read_exact(&mut header).map_err(io_error)?;
        let index = index::parse(&header, len)?;
        Ok(Self {
            inner: Arc::new(ArtifactData {
                storage: Storage::File(Mutex::new(FileCursor {
                    file,
                    position: Some(u64::try_from(8 + header_len).map_err(ArtifactError::format)?),
                })),
                index,
                source: Some(path),
                len,
            }),
        })
    }

    #[must_use]
    pub fn source(&self) -> Option<&Path> {
        self.inner.source.as_deref()
    }

    /// Tensor names in lexical order, from retained metadata.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        self.inner.index.keys().cloned().collect()
    }

    /// Borrow format metadata; call [`ArtifactTensor::reader`] to access bytes.
    ///
    /// # Errors
    /// Returns [`ArtifactError::MissingTensor`] when `name` is absent.
    pub fn tensor(&self, name: &str) -> Result<ArtifactTensor<'_>, ArtifactError> {
        let entry = self
            .inner
            .index
            .get(name)
            .ok_or_else(|| ArtifactError::MissingTensor(name.to_owned()))?;
        Ok(ArtifactTensor {
            artifact: self,
            entry,
        })
    }

    /// Read the complete selected artifact, including its header, e.g. for hashing.
    #[must_use]
    pub fn reader(&self) -> ArtifactReader {
        ArtifactReader::new(self.clone(), 0..self.len())
    }

    /// Encoded artifact bytes retained in application heap, excluding metadata,
    /// allocator overhead and OS page cache. File-backed artifacts return zero.
    #[must_use]
    pub fn resident_payload_bytes(&self) -> usize {
        match &self.inner.storage {
            Storage::Bytes(bytes) => bytes.len(),
            Storage::File(_) => 0,
        }
    }

    /// Complete encoded artifact length, regardless of backing storage.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.len == 0
    }

    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.inner.index.len()
    }
}

/// Format metadata and access to encoded bytes, not an execution tensor.
/// The payload reader owns its source independently of this borrowed view.
pub struct ArtifactTensor<'a> {
    artifact: &'a SafeTensorArtifact,
    entry: &'a TensorEntry,
}

impl ArtifactTensor<'_> {
    #[must_use]
    pub fn scalar_type(&self) -> &ScalarType {
        &self.entry.scalar_type
    }

    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.entry.shape
    }

    /// Encoded payload length; this does not allocate or read it.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entry.range.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entry.range.is_empty()
    }

    /// Independent, owning reader bounded to this tensor's encoded payload.
    #[must_use]
    pub fn reader(&self) -> ArtifactReader {
        ArtifactReader::new(self.artifact.clone(), self.entry.range.clone())
    }
}

fn scalar_type(dtype: Dtype) -> ScalarType {
    match dtype {
        Dtype::F16 => ScalarType::F16,
        Dtype::BF16 => ScalarType::Bf16,
        Dtype::F32 => ScalarType::F32,
        Dtype::F64 => ScalarType::F64,
        Dtype::I8 => ScalarType::I8,
        Dtype::U8 => ScalarType::U8,
        Dtype::I16 => ScalarType::I16,
        Dtype::U16 => ScalarType::U16,
        Dtype::I32 => ScalarType::I32,
        Dtype::U32 => ScalarType::U32,
        Dtype::I64 => ScalarType::I64,
        Dtype::U64 => ScalarType::U64,
        Dtype::BOOL => ScalarType::Bool,
        _ => ScalarType::Named(dtype.to_string()),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArtifactError {
    Io { path: PathBuf, message: String },
    InvalidFormat(String),
    MissingTensor(String),
}

impl ArtifactError {
    fn format(error: impl fmt::Display) -> Self {
        Self::InvalidFormat(error.to_string())
    }
}

impl fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, message } => write!(
                f,
                "could not read SafeTensors artifact {}: {message}",
                path.display()
            ),
            Self::InvalidFormat(message) => write!(f, "invalid SafeTensors artifact: {message}"),
            Self::MissingTensor(name) => write!(f, "SafeTensors artifact has no tensor {name:?}"),
        }
    }
}

impl std::error::Error for ArtifactError {}

#[cfg(test)]
mod tests;
