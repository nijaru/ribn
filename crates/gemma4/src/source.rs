//! Whole-inventory preflight followed by retained, encoded parameter sources.

use crate::{Gemma4Config, Gemma4Error, ParameterId, ParameterSpec};
use ribn_foundation::ScalarType;
use ribn_hf::{LocalModelPackage, WeightLayout};
use ribn_safetensors::{ArtifactReader, SafeTensorArtifact};
use serde_json::Value;
use std::collections::BTreeMap;

const AFFINE_GROUP_SIZE: usize = 64;
const HF_TEXT_PREFIX: &str = "model.language_model.";
const MLX_TEXT_PREFIX: &str = "language_model.model.";

/// Native weight encoding of a validated Gemma 4 package.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Gemma4SourceEncoding {
    /// Full unconverted BF16 checkpoint.
    Bf16,
    /// Pre-quantized 8-bit affine group-64 checkpoint (MLX layout).
    AffineQ8G64,
    /// Pre-quantized 4-bit affine group-64 checkpoint (MLX layout).
    AffineQ4G64,
}

impl Gemma4SourceEncoding {
    /// Quantization bit width for pre-quantized affine packages, or `None` for BF16.
    #[must_use]
    pub const fn affine_bits(self) -> Option<u8> {
        match self {
            Self::Bf16 => None,
            Self::AffineQ8G64 => Some(8),
            Self::AffineQ4G64 => Some(4),
        }
    }
}

/// Encoded storage for one validated Gemma 4 parameter.
#[derive(Clone)]
pub enum ParameterPayload {
    /// Uncompressed BF16 tensor bytes.
    Bf16(ArtifactReader),
    /// Pre-quantized affine group-64 weight (`U32` packed weights + BF16 `scales` + BF16 `biases`).
    AffineGroup64 {
        bits: u8,
        packed: ArtifactReader,
        scales: ArtifactReader,
        biases: ArtifactReader,
    },
}

/// Logical parameter plus its validated encoded storage. No decoded/device storage.
pub struct StoredParameter {
    spec: ParameterSpec,
    payload: ParameterPayload,
    encoded_bytes: usize,
}
impl StoredParameter {
    #[must_use]
    pub const fn spec(&self) -> &ParameterSpec {
        &self.spec
    }
    /// Exact encoded storage bytes for this parameter (`BF16` or `packed + scales + biases`).
    #[must_use]
    pub const fn encoded_bytes(&self) -> usize {
        self.encoded_bytes
    }
    /// Validated encoded payload (`Bf16` or `AffineGroup64`).
    #[must_use]
    pub const fn payload(&self) -> &ParameterPayload {
        &self.payload
    }
    /// A fresh bounded BF16 reader for uncompressed parameters.
    ///
    /// # Errors
    /// Rejects pre-quantized parameters rather than exposing packed integer words as BF16.
    pub fn bf16_reader(&self) -> Result<ArtifactReader, Gemma4Error> {
        match &self.payload {
            ParameterPayload::Bf16(reader) => Ok(reader.clone()),
            ParameterPayload::AffineGroup64 { bits, .. } => Err(Gemma4Error::new(
                &self.spec.name,
                format!("parameter is pre-quantized ({bits}-bit affine); BF16 reader unavailable"),
            )),
        }
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
    encoding: Gemma4SourceEncoding,
    parameters: BTreeMap<ParameterId, StoredParameter>,
    excluded: Vec<ExcludedParameter>,
    encoded_bytes: usize,
    media: Inventory,
    vision_config: Option<Value>,
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

fn parse_quant_object(
    obj: &serde_json::Map<String, Value>,
) -> Result<Gemma4SourceEncoding, Gemma4Error> {
    if let Some(method) = obj.get("quant_method") {
        return Err(Gemma4Error::new(
            "quantization",
            format!(
                "unsupported quantization method {method}; only MLX affine group-64 Q4/Q8 is supported"
            ),
        ));
    }
    if let Some(mode) = obj.get("mode")
        && mode.as_str() != Some("affine")
    {
        return Err(Gemma4Error::new(
            "quantization.mode",
            format!("unsupported quantization mode {mode}; expected \"affine\""),
        ));
    }
    let group_size = obj
        .get("group_size")
        .and_then(Value::as_u64)
        .ok_or_else(|| Gemma4Error::new("quantization.group_size", "missing integer group_size"))?;
    if group_size != AFFINE_GROUP_SIZE as u64 {
        return Err(Gemma4Error::new(
            "quantization.group_size",
            format!(
                "unsupported group_size {group_size}; only group_size {AFFINE_GROUP_SIZE} is supported"
            ),
        ));
    }
    let bits = obj
        .get("bits")
        .and_then(Value::as_u64)
        .ok_or_else(|| Gemma4Error::new("quantization.bits", "missing integer bits"))?;
    match bits {
        4 => Ok(Gemma4SourceEncoding::AffineQ4G64),
        8 => Ok(Gemma4SourceEncoding::AffineQ8G64),
        other => Err(Gemma4Error::new(
            "quantization.bits",
            format!(
                "unsupported bit width {other}; only 4-bit and 8-bit affine quantization are supported"
            ),
        )),
    }
}

fn parse_source_encoding(config: &Value) -> Result<Gemma4SourceEncoding, Gemma4Error> {
    let parse_opt = |key: &str| -> Result<Option<Gemma4SourceEncoding>, Gemma4Error> {
        let Some(raw) = config.get(key) else {
            return Ok(None);
        };
        if raw.is_null() {
            return Ok(None);
        }
        let obj = raw
            .as_object()
            .ok_or_else(|| Gemma4Error::new(key, "must be a JSON object"))?;
        parse_quant_object(obj).map(Some)
    };
    let q1 = parse_opt("quantization")?;
    let q2 = parse_opt("quantization_config")?;
    match (q1, q2) {
        (None, None) => Ok(Gemma4SourceEncoding::Bf16),
        (Some(enc), None) | (None, Some(enc)) => Ok(enc),
        (Some(a), Some(b)) if a == b => Ok(a),
        (Some(a), Some(b)) => Err(Gemma4Error::new(
            "quantization",
            format!("quantization ({a:?}) and quantization_config ({b:?}) disagree"),
        )),
    }
}

fn detect_text_prefix(inventory: &BTreeMap<String, usize>) -> Result<&'static str, Gemma4Error> {
    let has_hf = inventory.contains_key("model.language_model.embed_tokens.weight");
    let has_mlx = inventory.contains_key("language_model.model.embed_tokens.weight");
    match (has_hf, has_mlx) {
        (true, true) => Err(Gemma4Error::new(
            "weights",
            "ambiguous text namespace (both model.language_model. and language_model.model. present)",
        )),
        (false, true) => Ok(MLX_TEXT_PREFIX),
        _ => Ok(HF_TEXT_PREFIX),
    }
}

fn is_known_non_text(name: &str, text_prefix: &str) -> bool {
    if text_prefix == MLX_TEXT_PREFIX {
        [
            "vision_tower.",
            "audio_tower.",
            "embed_vision.",
            "embed_audio.",
        ]
        .iter()
        .any(|p| name.starts_with(p))
            || ["model.audio_tower.", "model.vision_tower."]
                .iter()
                .any(|p| name.starts_with(p))
            || name == "model.embed_audio.embedding_projection.weight"
            || name == "model.embed_vision.embedding_projection.weight"
    } else {
        ["model.audio_tower.", "model.vision_tower."]
            .iter()
            .any(|p| name.starts_with(p))
            || name == "model.embed_audio.embedding_projection.weight"
            || name == "model.embed_vision.embedding_projection.weight"
    }
}

impl Gemma4Source {
    /// Resolve all shards and validate the entire text inventory before returning
    /// any selected parameter. Inactive stored tail K/V/K-norm is optional; all
    /// active parameters are mandatory. Unknown text parameters reject. Known
    /// media namespaces are explicitly excluded, not claimed as supported.
    ///
    /// Supports both uncompressed BF16 HF checkpoints (`model.language_model.*`) and
    /// pre-quantized MLX affine group-64 Q4/Q8 checkpoints (`language_model.model.*`
    /// or `model.language_model.*`).
    ///
    /// # Errors
    /// Rejects unsupported config/quantization, missing/duplicate/misindexed parameters,
    /// wrong shapes/dtypes, unknown namespaces, or unreadable/malformed artifacts.
    #[expect(
        clippy::too_many_lines,
        reason = "whole-inventory BF16 and pre-quantized affine group-64 validation"
    )]
    pub fn open(package: &LocalModelPackage) -> Result<Self, Gemma4Error> {
        let config = Gemma4Config::from_json(package.config())?;
        let encoding = parse_source_encoding(package.config())?;
        let Inventory {
            files,
            mut inventory,
        } = Inventory::open(package)?;
        let text_prefix = detect_text_prefix(&inventory)?;
        let mut parameters = BTreeMap::new();
        let mut excluded = Vec::new();
        let mut encoded_bytes = 0_usize;
        let mut quantized_count = 0_usize;

        for spec in crate::parameters::specifications_with_prefix(&config, text_prefix) {
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

            let quant_stem = if encoding.affine_bits().is_some()
                && spec.shape.len() == 2
                && spec.shape[1].is_multiple_of(AFFINE_GROUP_SIZE)
            {
                spec.name.strip_suffix(".weight")
            } else {
                None
            };

            if let (Some(bits), Some(stem)) = (encoding.affine_bits(), quant_stem)
                && tensor.scalar_type() == &ScalarType::U32
            {
                let rows = spec.shape[0];
                let cols = spec.shape[1];
                let pack_factor = usize::from(32 / bits);
                let expected_packed = [rows, cols / pack_factor];
                if tensor.shape() != expected_packed {
                    return Err(Gemma4Error::new(
                        &spec.name,
                        format!(
                            "expected U32 {expected_packed:?} for {bits}-bit affine weight, found U32 {:?}",
                            tensor.shape()
                        ),
                    ));
                }
                let scales_name = format!("{stem}.scales");
                let biases_name = format!("{stem}.biases");
                let scales_idx = inventory.remove(&scales_name).ok_or_else(|| {
                    Gemma4Error::new(&scales_name, "missing affine quantization scales tensor")
                })?;
                let biases_idx = inventory.remove(&biases_name).ok_or_else(|| {
                    Gemma4Error::new(&biases_name, "missing affine quantization biases tensor")
                })?;
                let scales = files[scales_idx]
                    .tensor(&scales_name)
                    .map_err(|e| Gemma4Error::new(&scales_name, e.to_string()))?;
                let biases = files[biases_idx]
                    .tensor(&biases_name)
                    .map_err(|e| Gemma4Error::new(&biases_name, e.to_string()))?;
                let expected_sb = [rows, cols / AFFINE_GROUP_SIZE];
                if scales.scalar_type() != &ScalarType::Bf16 || scales.shape() != expected_sb {
                    return Err(Gemma4Error::new(
                        &scales_name,
                        format!(
                            "expected BF16 {expected_sb:?}, found {:?} {:?}",
                            scales.scalar_type(),
                            scales.shape()
                        ),
                    ));
                }
                if biases.scalar_type() != &ScalarType::Bf16 || biases.shape() != expected_sb {
                    return Err(Gemma4Error::new(
                        &biases_name,
                        format!(
                            "expected BF16 {expected_sb:?}, found {:?} {:?}",
                            biases.scalar_type(),
                            biases.shape()
                        ),
                    ));
                }

                if inactive {
                    for name in [spec.name, scales_name, biases_name] {
                        excluded.push(ExcludedParameter {
                            name,
                            reason: Exclusion::SharedTailProjection,
                        });
                    }
                } else {
                    let packed_bytes = expected_packed[0]
                        .checked_mul(expected_packed[1])
                        .and_then(|n| n.checked_mul(4))
                        .ok_or_else(|| {
                            Gemma4Error::new(&spec.name, "packed weight byte size overflows")
                        })?;
                    let sb_bytes = expected_sb[0]
                        .checked_mul(expected_sb[1])
                        .and_then(|n| n.checked_mul(4))
                        .ok_or_else(|| {
                            Gemma4Error::new(&spec.name, "scale/bias byte size overflows")
                        })?;
                    let param_bytes = packed_bytes.checked_add(sb_bytes).ok_or_else(|| {
                        Gemma4Error::new(&spec.name, "quantized parameter byte size overflows")
                    })?;
                    encoded_bytes = encoded_bytes.checked_add(param_bytes).ok_or_else(|| {
                        Gemma4Error::new("weights", "aggregate encoded byte size overflows")
                    })?;
                    quantized_count += 1;
                    parameters.insert(
                        spec.id,
                        StoredParameter {
                            spec,
                            payload: ParameterPayload::AffineGroup64 {
                                bits,
                                packed: tensor.reader(),
                                scales: scales.reader(),
                                biases: biases.reader(),
                            },
                            encoded_bytes: param_bytes,
                        },
                    );
                }
                continue;
            }

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
                let param_bytes = spec.byte_len()?;
                encoded_bytes = encoded_bytes.checked_add(param_bytes).ok_or_else(|| {
                    Gemma4Error::new("weights", "aggregate encoded byte size overflows")
                })?;
                parameters.insert(
                    spec.id,
                    StoredParameter {
                        spec,
                        payload: ParameterPayload::Bf16(tensor.reader()),
                        encoded_bytes: param_bytes,
                    },
                );
            }
        }

        if encoding != Gemma4SourceEncoding::Bf16 && quantized_count == 0 {
            return Err(Gemma4Error::new(
                "quantization",
                format!(
                    "config declares {encoding:?} quantization, but no quantized U32 tensors were found"
                ),
            ));
        }

        for name in inventory.keys() {
            if !is_known_non_text(name, text_prefix) {
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
            encoding,
            parameters,
            excluded,
            encoded_bytes,
            media: Inventory { files, inventory },
            vision_config: package.config().get("vision_config").cloned(),
        })
    }

    /// Native weight encoding of this validated source package.
    #[must_use]
    pub const fn encoding(&self) -> Gemma4SourceEncoding {
        self.encoding
    }

    /// Reject pre-quantized sources cleanly on BF16-only backends.
    ///
    /// # Errors
    /// Returns [`Gemma4Error`] when this source is pre-quantized.
    pub fn require_bf16(&self, backend: &str) -> Result<(), Gemma4Error> {
        if self.encoding == Gemma4SourceEncoding::Bf16 {
            Ok(())
        } else {
            Err(Gemma4Error::new(
                backend,
                format!(
                    "pre-quantized {:?} package requires the Metal model-wide backend",
                    self.encoding
                ),
            ))
        }
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
