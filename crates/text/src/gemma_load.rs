//! Ordinary package loading for the Gemma 4 executors.

use crate::{GemmaProcessor, ProcessorLimits, TextConfig, TextError, TextOwner};
use engine_gemma4::{Gemma4CpuWeights, Gemma4Executor, Gemma4Source};
use ribn::Engine;
use ribn::driver::{Driver, DriverConfig};
use ribn_hf::LocalModelPackage;
use std::sync::Arc;

/// Retained model size reported after a successful load.
#[derive(Clone, Copy, Debug)]
pub struct GemmaLoadReport {
    pub model_bytes: u64,
}

#[cfg(feature = "cuda")]
const CUDA_POOL_BYTES: u64 = 20 << 30;
#[cfg(feature = "cuda")]
const CUDA_MAX_POSITIONS: usize = 4096;
#[cfg(feature = "cuda")]
const CUDA_SCRATCH_ENTRIES: usize = 8 * CUDA_MAX_POSITIONS;

impl TextOwner {
    /// Load one unchanged Gemma 4 package on the default Metal device.
    ///
    /// # Errors
    /// Rejects invalid package assets, weight loading or Metal preparation, and
    /// text-owner assembly failures.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn load_gemma(
        path: impl Into<std::path::PathBuf>,
    ) -> Result<(Self, GemmaLoadReport), TextError> {
        let package = LocalModelPackage::open(path).map_err(TextError::processor)?;
        let source = Gemma4Source::open(&package).map_err(TextError::processor)?;
        let host = Gemma4CpuWeights::load(&source).map_err(TextError::processor)?;
        let model_bytes = host.bytes();
        let model =
            Arc::new(engine_gemma4::Gemma4MetalModel::load(host).map_err(TextError::processor)?);
        let executor = Gemma4Executor::metal(model).map_err(TextError::processor)?;
        assemble(&package, executor, model_bytes)
    }

    /// Load one unchanged Gemma 4 package on the default CUDA device.
    ///
    /// The device cache is bounded to 4K positions and a 20 GiB shared pool;
    /// longer requests reject at admission. This is the experimental greedy path.
    ///
    /// # Errors
    /// Rejects invalid package assets, CUDA device preparation, weight upload or
    /// text-owner assembly failures.
    #[cfg(feature = "cuda")]
    pub fn load_gemma_cuda(
        path: impl Into<std::path::PathBuf>,
    ) -> Result<(Self, GemmaLoadReport), TextError> {
        let package = LocalModelPackage::open(path).map_err(TextError::processor)?;
        let source = Gemma4Source::open(&package).map_err(TextError::processor)?;
        let host = Gemma4CpuWeights::load(&source).map_err(TextError::processor)?;
        let model_bytes = host.bytes();
        let context = cudarc::driver::CudaContext::new(0).map_err(TextError::processor)?;
        let stream = context.new_stream().map_err(TextError::processor)?;
        let pool = ribn_foundation::BytePool::new(CUDA_POOL_BYTES).shared();
        let model = engine_gemma4::Gemma4CudaModel::load(
            &source,
            host,
            stream,
            &pool,
            CUDA_MAX_POSITIONS,
            CUDA_SCRATCH_ENTRIES,
        )
        .map_err(TextError::processor)?;
        let executor = Gemma4Executor::cuda(Arc::new(std::sync::Mutex::new(model)))
            .map_err(TextError::processor)?;
        assemble(&package, executor, model_bytes)
    }
}

fn assemble<E>(
    package: &LocalModelPackage,
    executor: Gemma4Executor<E>,
    model_bytes: u64,
) -> Result<(TextOwner, GemmaLoadReport), TextError>
where
    E: engine_gemma4::GemmaDecode + 'static,
{
    let engine = Engine::with_defaults(executor).map_err(TextError::Engine)?;
    let driver = DriverConfig::for_engine(&engine);
    let (shutdown, handle) = Driver::spawn(engine, driver).map_err(TextError::Owner)?;
    let processor = Arc::new(
        GemmaProcessor::open(package, ProcessorLimits::default()).map_err(TextError::processor)?,
    );
    let owner = TextOwner::new(processor, shutdown, handle, TextConfig::default())?;
    Ok((owner, GemmaLoadReport { model_bytes }))
}
