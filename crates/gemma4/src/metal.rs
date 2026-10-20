//! Prepared Gemma weights resident on one Metal device.
//!
//! Loading uploads the already validated host inventory without conversion and
//! keeps the tied output head as the single token-embedding buffer.

use crate::{Gemma4Config, Gemma4CpuWeights, Gemma4Error, LayerParameter, ParameterId};
use engine_metal::{MetalBf16Buffer, MetalBf16Executor};
use std::collections::BTreeMap;

/// Every active BF16 parameter uploaded once to one Metal device.
pub struct Gemma4MetalWeights {
    parameters: BTreeMap<ParameterId, MetalBf16Buffer>,
    bytes: u64,
    config: Gemma4Config,
}

impl Gemma4MetalWeights {
    /// Upload the validated host inventory, preserving native BF16 encodings.
    /// # Errors
    /// Returns the first Metal upload failure; no partial set is published.
    pub fn load(
        host: &Gemma4CpuWeights,
        executor: &MetalBf16Executor,
    ) -> Result<Self, Gemma4Error> {
        let mut parameters = BTreeMap::new();
        let mut bytes = 0_u64;
        for (id, parameter) in host.parameters() {
            let buffer = executor
                .upload(parameter.values())
                .map_err(|error| Gemma4Error::new("Metal weight", error.to_string()))?;
            bytes =
                bytes
                    .checked_add(u64::try_from(parameter.values().len() * 2).map_err(|_| {
                        Gemma4Error::new("Metal weight", "parameter bytes exceed u64")
                    })?)
                    .ok_or_else(|| Gemma4Error::new("Metal weight", "total bytes overflow"))?;
            parameters.insert(id, buffer);
        }
        Ok(Self {
            parameters,
            bytes,
            config: host.config().clone(),
        })
    }

    #[must_use]
    pub fn parameter(&self, id: ParameterId) -> Option<&MetalBf16Buffer> {
        self.parameters.get(&id)
    }

    /// Decoder-layer parameter; inactive shared-tail roles are absent by design.
    /// # Errors
    /// Rejects an unknown layer or role instead of substituting another owner.
    pub fn layer(
        &self,
        layer: usize,
        parameter: LayerParameter,
    ) -> Result<&MetalBf16Buffer, Gemma4Error> {
        self.parameter(ParameterId::Layer { layer, parameter })
            .ok_or_else(|| {
                Gemma4Error::new(
                    "Metal weight",
                    format!("layer {layer} has no active {parameter:?} parameter"),
                )
            })
    }

    /// Encoded bytes uploaded once; tied-head aliasing does not double count.
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.parameters.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.parameters.is_empty()
    }

    #[must_use]
    pub const fn config(&self) -> &Gemma4Config {
        &self.config
    }
}

impl std::fmt::Debug for Gemma4MetalWeights {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Gemma4MetalWeights")
            .field("parameters", &self.parameters.len())
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}
