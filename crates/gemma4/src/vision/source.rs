//! Vision selection shares the text source's retained artifact files.
use super::{Gemma4VisionConfig, VisionParameterId, VisionParameterSpec};
use crate::source::Inventory;
use crate::{CpuParameter, Gemma4Error};
use ribn_foundation::ScalarType;
use ribn_safetensors::ArtifactReader;
use std::collections::{BTreeMap, BTreeSet};

pub struct StoredVisionParameter {
    spec: VisionParameterSpec,
    reader: ArtifactReader,
}
impl StoredVisionParameter {
    #[must_use]
    pub const fn spec(&self) -> &VisionParameterSpec {
        &self.spec
    }
    #[must_use]
    pub fn reader(&self) -> ArtifactReader {
        self.reader.clone()
    }
}
/// Complete vision encoder/projector inventory, retaining the same immutable files
/// selected by the text source. Not a decoder or an application media interface.
pub struct Gemma4VisionSource {
    config: Gemma4VisionConfig,
    parameters: BTreeMap<VisionParameterId, StoredVisionParameter>,
    bytes: usize,
}
impl Gemma4VisionSource {
    pub(crate) fn select(
        config: Gemma4VisionConfig,
        inventory: &Inventory,
    ) -> Result<Self, Gemma4Error> {
        let mut parameters = BTreeMap::new();
        let mut selected = BTreeSet::new();
        let mut bytes = 0_usize;
        for spec in config.specifications() {
            let index = inventory.inventory.get(&spec.name).ok_or_else(|| {
                Gemma4Error::new(&spec.name, "missing vision parameter or clipping buffer")
            })?;
            let tensor = inventory.files[*index]
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
            bytes = bytes
                .checked_add(spec.byte_len()?)
                .ok_or_else(|| Gemma4Error::new("vision weights", "byte size overflow"))?;
            selected.insert(spec.name.clone());
            parameters.insert(
                spec.id,
                StoredVisionParameter {
                    spec,
                    reader: tensor.reader(),
                },
            );
        }
        for name in inventory.inventory.keys() {
            if (name.starts_with("model.vision_tower.") || name.starts_with("model.embed_vision."))
                && !selected.contains(name)
            {
                return Err(Gemma4Error::new(name, "unexpected vision parameter"));
            }
        }
        Ok(Self {
            config,
            parameters,
            bytes,
        })
    }
    #[must_use]
    pub const fn config(&self) -> &Gemma4VisionConfig {
        &self.config
    }
    #[must_use]
    pub fn parameters(&self) -> &BTreeMap<VisionParameterId, StoredVisionParameter> {
        &self.parameters
    }
    #[must_use]
    pub const fn encoded_bytes(&self) -> usize {
        self.bytes
    }
}

/// Native host BF16 vision storage. No text weights or fictitious text roles.
pub struct Gemma4VisionCpuWeights {
    config: Gemma4VisionConfig,
    parameters: BTreeMap<VisionParameterId, CpuParameter>,
    bytes: usize,
}
impl Gemma4VisionCpuWeights {
    /// Materialize the entire validated vision inventory, then validate clip ranges.
    /// # Errors
    /// Rejects truncated/trailing sources, allocation failure or invalid clip bounds.
    pub fn load(source: &Gemma4VisionSource) -> Result<Self, Gemma4Error> {
        let mut parameters = BTreeMap::new();
        let mut bytes = 0_usize;
        for (&id, stored) in source.parameters() {
            let parameter = CpuParameter::read(&stored.spec.shape, stored.reader())?;
            bytes = bytes
                .checked_add(stored.spec.byte_len()?)
                .ok_or_else(|| Gemma4Error::new("vision weights", "byte size overflow"))?;
            parameters.insert(id, parameter);
        }
        if bytes != source.bytes {
            return Err(Gemma4Error::new(
                "vision weights",
                "materialized byte count disagrees",
            ));
        }
        let weights = Self {
            config: source.config.clone(),
            parameters,
            bytes,
        };
        if weights.config.clipped {
            for layer in 0..weights.config.layers {
                for linear in super::VisionLinear::ALL {
                    weights.bounds(layer, linear)?;
                }
            }
        }
        Ok(weights)
    }
    #[must_use]
    pub const fn config(&self) -> &Gemma4VisionConfig {
        &self.config
    }
    #[must_use]
    pub const fn bytes(&self) -> usize {
        self.bytes
    }
    #[must_use]
    pub fn parameter(&self, id: VisionParameterId) -> Option<&CpuParameter> {
        self.parameters.get(&id)
    }
    pub(super) fn get(&self, id: VisionParameterId) -> Result<&CpuParameter, Gemma4Error> {
        self.parameter(id)
            .ok_or_else(|| Gemma4Error::new("vision weights", format!("missing {id:?}")))
    }
    pub(super) fn bounds(
        &self,
        layer: usize,
        linear: super::VisionLinear,
    ) -> Result<[f32; 4], Gemma4Error> {
        use super::VisionLinearPart::{InputMax, InputMin, OutputMax, OutputMin};
        let mut bounds = [0.0; 4];
        for (slot, part) in bounds
            .iter_mut()
            .zip([InputMin, InputMax, OutputMin, OutputMax])
        {
            *slot = f32::from_bits(
                u32::from(
                    self.get(VisionParameterId::Linear {
                        layer,
                        linear,
                        part,
                    })?
                    .values()[0],
                ) << 16,
            );
        }
        if bounds.iter().any(|v| v.is_nan()) || bounds[0] > bounds[1] || bounds[2] > bounds[3] {
            return Err(Gemma4Error::new(
                "vision clipping",
                "NaN or reversed activation bounds",
            ));
        }
        Ok(bounds)
    }
}
