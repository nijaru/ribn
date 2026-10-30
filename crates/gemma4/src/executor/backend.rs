//! Concrete decoder operations and model-wide completion ownership.
use super::Gemma4Executor;
use crate::cpu::Gemma4CpuDecoder;
use crate::{Gemma4CpuWeights, Gemma4Error};
use std::sync::Arc;

/// Sequence-level decoder operations; execution failure does not permit replay.
/// Teardown is separate: it may succeed even when continuation is invalid.
pub trait GemmaDecode: Send {
    /// Append a prompt chunk without returning logits.
    ///
    /// Success certifies completed continuation at the accepted end, not pending
    /// device work. A backend may retain a full-head completion path internally.
    /// # Errors
    /// As [`Self::prefill`]. Failed execution retains resources for settlement and
    /// close; an error never permits replay of a potentially advanced continuation.
    fn advance_prefill(&mut self, tokens: &[u32]) -> Result<(), Gemma4Error>;
    /// Append a prompt chunk and return logits for its last token.
    /// # Errors
    /// Rejects invalid tokens, excess context or a backend execution failure.
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<u16>, Gemma4Error>;
    /// Append one generated token and return the next logits.
    /// # Errors
    /// As [`Self::prefill`].
    fn decode(&mut self, token: u32) -> Result<Vec<u16>, Gemma4Error>;
    /// Tokens currently owned by this continuation.
    fn position(&self) -> usize;
    /// Release sequence resources after model-wide completion is established.
    /// A successful close is idempotent; the sequence will not execute again.
    /// # Errors
    /// Failed cleanup must retain its resources for another close attempt.
    fn close(&mut self) -> Result<(), Gemma4Error>;
}

/// Creation and completion capture the same concrete model. Completion cannot
/// depend on finding an admitted decoder: construction may have failed earlier.
pub(super) struct Backend<D> {
    pub(super) create: Box<dyn FnMut() -> Result<D, Gemma4Error> + Send>,
    pub(super) settle: Box<dyn FnMut() -> Result<(), Gemma4Error> + Send>,
}

impl GemmaDecode for Gemma4CpuDecoder {
    fn advance_prefill(&mut self, tokens: &[u32]) -> Result<(), Gemma4Error> {
        Self::advance_prefill(self, tokens)
    }
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<u16>, Gemma4Error> {
        Self::prefill(self, tokens)
    }
    fn decode(&mut self, token: u32) -> Result<Vec<u16>, Gemma4Error> {
        Self::decode(self, token)
    }
    fn position(&self) -> usize {
        Self::position(self)
    }
    fn close(&mut self) -> Result<(), Gemma4Error> {
        // CPU calls leave no asynchronous accesses, including on error.
        Ok(())
    }
}

impl Gemma4Executor<Gemma4CpuDecoder> {
    /// Bind the executor to host CPU weights shared by every sequence.
    /// # Errors
    /// Rejects a configuration whose context exceeds the runtime's u32 limits.
    pub fn cpu(weights: Arc<Gemma4CpuWeights>) -> Result<Self, Gemma4Error> {
        let config = weights.config().clone();
        Self::new(
            "gemma4 cpu",
            config.max_position_embeddings(),
            config,
            Backend {
                create: Box::new(move || Gemma4CpuDecoder::shared(weights.clone())),
                // No model-wide device work exists on this backend.
                settle: Box::new(|| Ok(())),
            },
        )
    }
}

#[cfg(feature = "cuda")]
impl GemmaDecode for crate::Gemma4CudaDecoder {
    fn advance_prefill(&mut self, tokens: &[u32]) -> Result<(), Gemma4Error> {
        Self::advance_prefill(self, tokens)
    }
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<u16>, Gemma4Error> {
        Self::prefill(self, tokens)
    }
    fn decode(&mut self, token: u32) -> Result<Vec<u16>, Gemma4Error> {
        Self::decode(self, token)
    }
    fn position(&self) -> usize {
        Self::position(self)
    }
    fn close(&mut self) -> Result<(), Gemma4Error> {
        self.retire_caches()
    }
}

#[cfg(feature = "cuda")]
impl Gemma4Executor<crate::Gemma4CudaDecoder> {
    /// Bind the executor without consuming the caller's CUDA model owner.
    /// # Errors
    /// Rejects a poisoned lock or an invalid prepared cache bound; the caller
    /// retains its model handle for explicit retirement after failed binding.
    pub fn cuda(
        model: &Arc<std::sync::Mutex<crate::Gemma4CudaModel>>,
    ) -> Result<Self, Gemma4Error> {
        let (config, capacity) = {
            let model = model
                .lock()
                .map_err(|_| Gemma4Error::new("executor", "CUDA model lock poisoned"))?;
            (model.config().clone(), model.max_positions())
        };
        let model = Arc::clone(model);
        let completion = Arc::clone(&model);
        Self::new(
            "gemma4 cuda",
            capacity,
            config,
            Backend {
                create: Box::new(move || crate::Gemma4CudaDecoder::shared(&model)),
                settle: Box::new(move || {
                    completion
                        .lock()
                        .map_err(|_| Gemma4Error::new("executor", "CUDA model lock poisoned"))?
                        .retire_failure()
                        .map_err(|error| Gemma4Error::new("CUDA retirement", error.to_string()))
                }),
            },
        )
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl GemmaDecode for crate::Gemma4MetalDecoder {
    fn advance_prefill(&mut self, tokens: &[u32]) -> Result<(), Gemma4Error> {
        // Retain the qualified full-head/readback completion path until head-free
        // device execution and its completion ownership are qualified together.
        Self::prefill(self, tokens).map(drop)
    }
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<u16>, Gemma4Error> {
        Self::prefill(self, tokens)
    }
    fn decode(&mut self, token: u32) -> Result<Vec<u16>, Gemma4Error> {
        Self::decode(self, token)
    }
    fn position(&self) -> usize {
        Self::position(self)
    }
    fn close(&mut self) -> Result<(), Gemma4Error> {
        // The backend settles the shared model's device access envelope before
        // releasing this decoder's host-owned continuation.
        Ok(())
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl Gemma4Executor<crate::Gemma4MetalDecoder> {
    /// Bind the executor to an uploaded Metal model shared by every sequence.
    /// # Errors
    /// Rejects a configuration whose context exceeds the runtime's u32 limits.
    pub fn metal(model: Arc<crate::Gemma4MetalModel>) -> Result<Self, Gemma4Error> {
        let config = model.config().clone();
        let completion = Arc::clone(&model);
        Self::new(
            "gemma4 metal",
            config.max_position_embeddings(),
            config,
            Backend {
                create: Box::new(move || crate::Gemma4MetalDecoder::shared(model.clone())),
                // Retain the SAME model even if no decoder was constructed. An
                // uncertain Graph submission cannot release its access envelope.
                settle: Box::new(move || completion.retire_failure()),
            },
        )
    }
}
