//! Ordinary package loading for the Gemma 4 Metal executor.

use crate::{GemmaProcessor, ProcessorLimits, TextConfig, TextError, TextOwner};
use engine_gemma4::{Gemma4CpuWeights, Gemma4Executor, Gemma4MetalModel, Gemma4Source};
use ribn::Engine;
use ribn::driver::{Driver, DriverConfig};
use ribn_hf::LocalModelPackage;
use std::sync::Arc;

/// Retained model size reported after a successful load.
#[derive(Clone, Copy, Debug)]
pub struct GemmaLoadReport {
    pub model_bytes: u64,
}

impl TextOwner {
    /// Load one unchanged Gemma 4 package on the default Metal device.
    ///
    /// # Errors
    /// Rejects invalid package assets, weight loading or Metal preparation, and
    /// text-owner assembly failures.
    pub fn load_gemma(
        path: impl Into<std::path::PathBuf>,
    ) -> Result<(Self, GemmaLoadReport), TextError> {
        let package = LocalModelPackage::open(path).map_err(TextError::processor)?;
        let source = Gemma4Source::open(&package).map_err(TextError::processor)?;
        let host = Gemma4CpuWeights::load(&source).map_err(TextError::processor)?;
        let model_bytes = host.bytes();
        let model = Arc::new(Gemma4MetalModel::load(host).map_err(TextError::processor)?);
        let executor = Gemma4Executor::metal(model).map_err(TextError::processor)?;
        let engine = Engine::with_defaults(executor).map_err(TextError::Engine)?;
        let driver = DriverConfig::for_engine(&engine);
        let (shutdown, handle) = Driver::spawn(engine, driver).map_err(TextError::Owner)?;
        let processor = Arc::new(
            GemmaProcessor::open(&package, ProcessorLimits::default())
                .map_err(TextError::processor)?,
        );
        let owner = Self::new(processor, shutdown, handle, TextConfig::default())?;
        Ok((owner, GemmaLoadReport { model_bytes }))
    }
}
