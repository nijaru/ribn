//! An opaque stream domain for native ordered execution.
use super::{
    CudaBf16Attention, CudaBf16ComputeError, CudaBf16Elementwise, CudaBf16Error, CudaBf16Gemm,
    CudaBf16Matrix, CudaBf16RmsNorm, CudaBf16RotaryApply, CudaBf16RotaryFactors, CudaBf16Weights,
};
use cudarc::driver::{CudaContext, CudaStream};
use engine_core::WeightTensorSpec;
use ribn_foundation::BytePool;
use std::sync::Arc;

/// A private nonblocking CUDA stream and its associated BF16 resource factory.
///
/// No public operation exposes the stream or permits capture. A caller-owned raw
/// stream cannot establish this invariant: capture can defer commands into a graph
/// that neither a later stream drain nor an event query proves was executed.
/// Resources retain the stream independently of this factory's lifetime.
pub struct CudaBf16Stream {
    // This Arc must not escape through public APIs or callbacks. In particular,
    // capture/graph replay is not part of this execution domain's contract.
    pub(super) stream: Arc<CudaStream>,
}
impl CudaBf16Stream {
    /// Create a private nonblocking stream in the supplied live context.
    /// # Errors
    /// Returns context/stream creation failures.
    pub fn new(context: &Arc<CudaContext>) -> Result<Self, CudaBf16Error> {
        Ok(Self {
            stream: context
                .new_stream()
                .map_err(|e| super::error("BF16 private stream", e))?,
        })
    }
    /// Allocate a BF16 matrix in this domain.
    /// # Errors
    /// As [`CudaBf16Matrix::allocate`].
    pub fn matrix(
        &self,
        rows: usize,
        cols: usize,
        pool: &Arc<BytePool>,
        staging_bytes: usize,
    ) -> Result<CudaBf16Matrix, CudaBf16Error> {
        CudaBf16Matrix::allocate(rows, cols, Arc::clone(&self.stream), pool, staging_bytes)
    }
    /// Allocate one immutable BF16 weight inventory in this domain.
    /// # Errors
    /// As [`CudaBf16Weights::allocate`].
    pub fn weights(
        &self,
        specs: Vec<WeightTensorSpec>,
        pool: &Arc<BytePool>,
        staging_bytes: usize,
    ) -> Result<CudaBf16Weights, CudaBf16Error> {
        CudaBf16Weights::allocate(specs, Arc::clone(&self.stream), pool, staging_bytes)
    }
    /// Prepare the domain's charged cuBLAS workspace.
    /// # Errors
    /// As [`CudaBf16Gemm::new`], including retained initialization failure.
    pub fn gemm(
        &self,
        pool: &Arc<BytePool>,
        workspace_bytes: usize,
    ) -> Result<CudaBf16Gemm, CudaBf16ComputeError> {
        CudaBf16Gemm::new(Arc::clone(&self.stream), pool, workspace_bytes)
    }
    /// Prepare grouped RMS.
    /// # Errors
    /// As [`CudaBf16RmsNorm::new`].
    pub fn rms_norm(&self) -> Result<CudaBf16RmsNorm, CudaBf16ComputeError> {
        CudaBf16RmsNorm::new(Arc::clone(&self.stream))
    }
    /// Prepare elementwise operations.
    /// # Errors
    /// As [`CudaBf16Elementwise::new`].
    pub fn elementwise(&self) -> Result<CudaBf16Elementwise, CudaBf16ComputeError> {
        CudaBf16Elementwise::new(Arc::clone(&self.stream))
    }
    /// Prepare rotary application.
    /// # Errors
    /// As [`CudaBf16RotaryApply::new`].
    pub fn rotary(&self) -> Result<CudaBf16RotaryApply, CudaBf16ComputeError> {
        CudaBf16RotaryApply::new(Arc::clone(&self.stream))
    }
    /// Prepare model-supplied rotary frequencies and bounded position storage.
    /// # Errors
    /// As [`CudaBf16RotaryFactors::new`].
    pub fn rotary_factors(
        &self,
        pool: &Arc<BytePool>,
        frequencies: &[f32],
        capacity_rows: usize,
    ) -> Result<CudaBf16RotaryFactors, CudaBf16ComputeError> {
        CudaBf16RotaryFactors::new(Arc::clone(&self.stream), pool, frequencies, capacity_rows)
    }
    /// Prepare attention with charged bounded scratch.
    /// # Errors
    /// As [`CudaBf16Attention::new`].
    pub fn attention(
        &self,
        pool: &Arc<BytePool>,
        scratch_entries: usize,
    ) -> Result<CudaBf16Attention, CudaBf16ComputeError> {
        CudaBf16Attention::new(Arc::clone(&self.stream), pool, scratch_entries)
    }
    /// Explicit blocking drain of this domain. This does not publish a command
    /// transaction's output or retire its access guards.
    /// # Errors
    /// Returns driver completion errors.
    pub fn synchronize(&self) -> Result<(), CudaBf16Error> {
        self.stream
            .synchronize()
            .map_err(|e| super::error("BF16 domain completion", e))
    }
}
