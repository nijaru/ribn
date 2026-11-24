//! Pinned-descriptor GGUF preflight for Inkling 2 / Inkling 2 Loop artifacts.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use engine_gguf::{GgufFile, GgufTokenizer, TensorDataReader};

use crate::Inkling2Error;
use crate::config::Inkling2Config;
use crate::parameters::{ParameterId, ParameterSpec, specifications};

/// GGML tensor encodings supported by the Inkling 2 CPU reference path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorEncoding {
    F32,
    F16,
    Q8_0,
    Iq4Nl,
}

impl TensorEncoding {
    /// Map a raw GGML value type ID to a supported encoding.
    ///
    /// # Errors
    /// Rejects unsupported GGML type IDs explicitly rather than silently falling back.
    pub fn from_ggml_type(value_type: u32, tensor_name: &str) -> Result<Self, Inkling2Error> {
        match value_type {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            8 => Ok(Self::Q8_0),
            20 => Ok(Self::Iq4Nl),
            other => Err(Inkling2Error::new(
                tensor_name,
                format!("unsupported GGML tensor type {other}"),
            )),
        }
    }
}

/// Validated GGUF source retaining one opened descriptor for metadata,
/// tokenizer, and bounded tensor readers without reading bulk weights.
pub struct Inkling2Source {
    file: GgufFile,
    config: Inkling2Config,
    tokenizer: GgufTokenizer,
    specs: BTreeMap<ParameterId, ParameterSpec>,
    encodings: BTreeMap<ParameterId, TensorEncoding>,
}

impl Inkling2Source {
    /// Open and preflight an Inkling 2 GGUF artifact without reading tensor payloads.
    ///
    /// Validates architecture metadata, loop/slot geometry, complete tensor
    /// inventory (exact names, shapes, supported encodings, and in-bounds byte
    /// ranges), no unexpected extra tensors, and embedded tokenizer invariants.
    ///
    /// # Errors
    /// Returns [`Inkling2Error`] if any metadata, tensor, or tokenizer check fails.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, Inkling2Error> {
        let path = path.into();
        let file = GgufFile::open(&path)
            .map_err(|error| Inkling2Error::new(path.display().to_string(), error.to_string()))?;
        Self::from_gguf(file)
    }

    /// Preflight an already-opened [`GgufFile`].
    ///
    /// # Errors
    /// Returns [`Inkling2Error`] if any metadata, tensor, or tokenizer check fails.
    pub fn from_gguf(file: GgufFile) -> Result<Self, Inkling2Error> {
        let config = Inkling2Config::from_gguf(&file)?;
        let tokenizer = file
            .tokenizer()
            .map_err(|error| Inkling2Error::new("tokenizer", error.to_string()))?;

        if tokenizer.model() != "gpt2" {
            return Err(Inkling2Error::new(
                "tokenizer.ggml.model",
                format!("expected \"gpt2\", got {:?}", tokenizer.model()),
            ));
        }
        if !matches!(tokenizer.pretokenizer(), "inkling2" | "gpt-4o") {
            return Err(Inkling2Error::new(
                "tokenizer.ggml.pre",
                format!(
                    "expected \"inkling2\" or \"gpt-4o\", got {:?}",
                    tokenizer.pretokenizer()
                ),
            ));
        }
        if tokenizer.tokens().len() != config.vocab_unpadded() {
            return Err(Inkling2Error::new(
                "tokenizer vocabulary",
                format!(
                    "token count {} does not equal vocab_size_unpadded {}",
                    tokenizer.tokens().len(),
                    config.vocab_unpadded()
                ),
            ));
        }
        if usize::try_from(tokenizer.eos_token_id()).is_ok_and(|id| id >= config.vocab_unpadded()) {
            return Err(Inkling2Error::new(
                "tokenizer.ggml.eos_token_id",
                format!(
                    "EOS token ID {} is outside vocab_size_unpadded {}",
                    tokenizer.eos_token_id(),
                    config.vocab_unpadded()
                ),
            ));
        }

        let all_specs = specifications(&config);
        let mut expected_names = BTreeSet::new();
        let mut specs = BTreeMap::new();
        let mut encodings = BTreeMap::new();

        for spec in all_specs {
            expected_names.insert(spec.name.clone());
            let Some(info) = file.tensor(&spec.name) else {
                if spec.required {
                    return Err(Inkling2Error::new(
                        &spec.name,
                        "missing required tensor in GGUF artifact",
                    ));
                }
                continue;
            };
            if !spec.matches_dimensions(info.dimensions()) {
                return Err(Inkling2Error::new(
                    &spec.name,
                    format!(
                        "expected dimensions {:?}, got {:?}",
                        spec.dims,
                        info.dimensions()
                    ),
                ));
            }
            let encoding = TensorEncoding::from_ggml_type(info.value_type(), &spec.name)?;
            // Verify encoded byte range lies within the file bounds.
            let _ = file
                .tensor_data_range(&spec.name)
                .map_err(|error| Inkling2Error::new(&spec.name, error.to_string()))?;
            encodings.insert(spec.id, encoding);
            specs.insert(spec.id, spec);
        }

        for tensor in file.tensors() {
            if !expected_names.contains(tensor.name()) {
                return Err(Inkling2Error::new(
                    tensor.name(),
                    "unexpected tensor not covered by Inkling 2 inventory",
                ));
            }
        }

        Ok(Self {
            file,
            config,
            tokenizer,
            specs,
            encodings,
        })
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        self.file.path()
    }

    #[must_use]
    pub const fn config(&self) -> &Inkling2Config {
        &self.config
    }

    #[must_use]
    pub const fn tokenizer(&self) -> &GgufTokenizer {
        &self.tokenizer
    }

    #[must_use]
    pub const fn file(&self) -> &GgufFile {
        &self.file
    }

    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.specs.len()
    }

    #[must_use]
    pub fn parameter_spec(&self, id: ParameterId) -> Option<&ParameterSpec> {
        self.specs.get(&id)
    }

    #[must_use]
    pub fn parameter_encoding(&self, id: ParameterId) -> Option<TensorEncoding> {
        self.encodings.get(&id).copied()
    }

    /// Open a bounded reader for one validated parameter.
    ///
    /// # Errors
    /// Returns [`Inkling2Error`] if the parameter is absent or cannot be opened.
    pub fn open_parameter(&self, id: ParameterId) -> Result<TensorDataReader, Inkling2Error> {
        let spec = self.specs.get(&id).ok_or_else(|| {
            Inkling2Error::new(format!("{id:?}"), "parameter absent from artifact")
        })?;
        self.file
            .open_tensor(&spec.name)
            .map_err(|error| Inkling2Error::new(&spec.name, error.to_string()))
    }
}
