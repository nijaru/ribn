//! Whole-inventory preflight followed by retained, encoded parameter sources.

use crate::{Gemma4Config, Gemma4Error, ParameterId, ParameterSpec};
use ribn_foundation::ScalarType;
use ribn_hf::{LocalModelPackage, WeightLayout};
use ribn_safetensors::{ArtifactReader, SafeTensorArtifact};
use std::collections::BTreeMap;

/// Logical parameter plus its selected BF16 encoding. No decoded/device storage.
pub struct StoredParameter {
    spec: ParameterSpec,
    reader: ArtifactReader,
}
impl StoredParameter {
    #[must_use]
    pub const fn spec(&self) -> &ParameterSpec {
        &self.spec
    }
    /// A fresh bounded reader of the exact source validated by preflight.
    #[must_use]
    pub fn reader(&self) -> ArtifactReader {
        self.reader.clone()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Exclusion {
    SharedTailProjection,
    NonText,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExcludedParameter {
    pub name: String,
    pub reason: Exclusion,
}

/// One validated text source. All parameter readers retain the selected files,
/// including after the package/source paths disappear. Config/index and members
/// must stay unchanged while opening; files must not mutate through any alias
/// while readers live. This is not an atomic package or executable GPU snapshot.
pub struct Gemma4Source {
    config: Gemma4Config,
    parameters: BTreeMap<ParameterId, StoredParameter>,
    excluded: Vec<ExcludedParameter>,
    encoded_bytes: usize,
    media: Inventory,
    vision_config: Option<serde_json::Value>,
}

pub(crate) struct Inventory {
    pub(crate) files: Vec<SafeTensorArtifact>,
    pub(crate) inventory: BTreeMap<String, usize>,
}
impl Inventory {
    fn open(package: &LocalModelPackage) -> Result<Self, Gemma4Error> {
        let (paths, indexed_count) = match package.weight_layout() {
            WeightLayout::Single { path } => (vec![path], None),
            WeightLayout::Sharded {
                shards,
                parameter_count,
                ..
            } => (shards, Some(parameter_count)),
        };
        let mut files = Vec::new();
        let mut inventory = BTreeMap::new();
        for path in paths {
            let artifact = SafeTensorArtifact::open(&path)
                .map_err(|e| Gemma4Error::new(path.display().to_string(), e.to_string()))?;
            let index = files.len();
            for name in artifact.names() {
                if package.weight_file(&name) != Some(path.as_path()) {
                    return Err(Gemma4Error::new(
                        name,
                        "tensor is absent from its shard index or routed to a different file",
                    ));
                }
                match inventory.entry(name) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert(index);
                    }
                    std::collections::btree_map::Entry::Occupied(entry) => {
                        return Err(Gemma4Error::new(
                            entry.key(),
                            "duplicate tensor across shards",
                        ));
                    }
                }
            }
            files.push(artifact);
        }
        if indexed_count.is_some_and(|n| n != inventory.len()) {
            return Err(Gemma4Error::new(
                "shard index",
                "advertises parameters absent from the selected files",
            ));
        }
        Ok(Self { files, inventory })
    }
}

impl Gemma4Source {
    /// Resolve all shards and validate the entire text inventory before returning
    /// any selected parameter. Inactive stored tail K/V/K-norm is optional; all
    /// active parameters are mandatory. Unknown text parameters reject. Known
    /// media namespaces are explicitly excluded, not claimed as supported.
    ///
    /// # Errors
    /// Rejects unsupported config, missing/duplicate/misindexed parameters, wrong
    /// shapes/dtypes, unknown namespaces, or unreadable/malformed artifacts.
    pub fn open(package: &LocalModelPackage) -> Result<Self, Gemma4Error> {
        let config = Gemma4Config::from_json(package.config())?;
        let Inventory {
            files,
            mut inventory,
        } = Inventory::open(package)?;
        let mut parameters = BTreeMap::new();
        let mut excluded = Vec::new();
        let mut encoded_bytes = 0_usize;
        for spec in crate::parameters::specifications(&config) {
            let inactive = match spec.id {
                ParameterId::Layer { layer, parameter } => {
                    parameter.needs_owner() && config.layers[layer].kv_owner != layer
                }
                _ => false,
            };
            let Some(index) = inventory.remove(&spec.name) else {
                if inactive {
                    continue;
                }
                return Err(Gemma4Error::new(
                    &spec.name,
                    "missing active text parameter",
                ));
            };
            let tensor = files[index]
                .tensor(&spec.name)
                .map_err(|e| Gemma4Error::new(&spec.name, e.to_string()))?;
            if tensor.scalar_type() != &ScalarType::Bf16 || tensor.shape() != spec.shape {
                return Err(Gemma4Error::new(
                    &spec.name,
                    format!(
                        "expected BF16 {:?}, found {:?} {:?}",
                        spec.shape,
                        tensor.scalar_type(),
                        tensor.shape()
                    ),
                ));
            }
            if inactive {
                excluded.push(ExcludedParameter {
                    name: spec.name,
                    reason: Exclusion::SharedTailProjection,
                });
            } else {
                encoded_bytes = encoded_bytes.checked_add(spec.byte_len()?).ok_or_else(|| {
                    Gemma4Error::new("weights", "aggregate encoded byte size overflows")
                })?;
                parameters.insert(
                    spec.id,
                    StoredParameter {
                        spec,
                        reader: tensor.reader(),
                    },
                );
            }
        }
        for name in inventory.keys() {
            if !["model.audio_tower.", "model.vision_tower."]
                .iter()
                .any(|p| name.starts_with(p))
                && name != "model.embed_audio.embedding_projection.weight"
                && name != "model.embed_vision.embedding_projection.weight"
            {
                return Err(Gemma4Error::new(
                    name,
                    "unexpected parameter (including untied output heads)",
                ));
            }
            excluded.push(ExcludedParameter {
                name: name.clone(),
                reason: Exclusion::NonText,
            });
        }
        Ok(Self {
            config,
            parameters,
            excluded,
            encoded_bytes,
            media: Inventory { files, inventory },
            vision_config: package.config().get("vision_config").cloned(),
        })
    }

    /// Select vision parameters from the same retained files as the text source.
    /// Unused vision semantics do not affect text-only loading.
    /// # Errors
    /// Rejects missing/unsupported vision configuration or an incomplete inventory.
    pub fn vision_source(&self) -> Result<crate::vision::Gemma4VisionSource, Gemma4Error> {
        let raw = self
            .vision_config
            .as_ref()
            .ok_or_else(|| Gemma4Error::new("vision_config", "missing configuration"))?;
        let config = crate::vision::Gemma4VisionConfig::parse(raw, self.config.hidden_size())?;
        crate::vision::Gemma4VisionSource::select(config, &self.media)
    }

    #[must_use]
    pub const fn config(&self) -> &Gemma4Config {
        &self.config
    }
    #[must_use]
    pub fn parameters(&self) -> &BTreeMap<ParameterId, StoredParameter> {
        &self.parameters
    }
    #[must_use]
    pub fn parameter(&self, id: ParameterId) -> Option<&StoredParameter> {
        self.parameters.get(&id)
    }
    /// The output head and input embedding are the exact same logical parameter;
    /// a backend must preserve that allocation alias without scaling its weights.
    #[must_use]
    #[expect(
        clippy::missing_panics_doc,
        reason = "private complete-preflight invariant; callers cannot construct or mutate the inventory"
    )]
    pub fn output_projection(&self) -> &StoredParameter {
        self.parameters
            .get(&ParameterId::TokenEmbedding)
            .expect("complete preflight always contains the token embedding")
    }
    /// Unique active encoded bytes, not VRAM residency or loading/upload peak.
    #[must_use]
    pub const fn encoded_bytes(&self) -> usize {
        self.encoded_bytes
    }
    #[must_use]
    pub fn excluded(&self) -> &[ExcludedParameter] {
        &self.excluded
    }
}
