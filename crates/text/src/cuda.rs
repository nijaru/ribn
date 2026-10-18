//! GGUF/CUDA assembly for the text facade.
//!
//! Loading is deliberately separate from facade behavior so the preprocessing,
//! delivery and batching paths are exercisable without a device.

#[cfg(test)]
mod overhead;
#[cfg(test)]
mod tests;

use std::path::PathBuf;
use std::sync::Arc;

use engine_qwen::{QwenCuda, QwenLoadOptions};
use ribn::driver::{Driver, DriverConfig};
use ribn::{Engine, EngineConfig, GenerationExecutor, SchedulePolicy};

use crate::error::TextError;
use crate::model::{TextConfig, TextOwner};
use crate::process::{GgufProcessor, ProcessorLimits};

pub use engine_qwen::MemoryReport;

/// Model loading and text-layer limits for the currently supported path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoadOptions {
    pub device: u16,
    pub context_tokens: u32,
    /// Prepared sequence capacity and the token runtime's active-sequence limit.
    /// Application permits, HTTP slots and physical kernel batching are separate.
    pub max_sequences: usize,
    /// Same-sequence prefill chunk size, or `None` for the serial prefill path.
    pub prefill_chunk_members: Option<usize>,
    /// Prefer sampling a whole prefill chunk's final row. Enabled by default;
    /// unsupported modes and prompt shapes keep the serial final step.
    pub sampled_prefill_final: bool,
    pub weight_budget_bytes: Option<u64>,
    /// Continuation bytes the prepared execution may charge in total. `None` keeps the
    /// conservative context-sized reservation; a request is charged for the tokens it
    /// can reach either way.
    pub continuation_capacity_bytes: Option<u64>,
    pub headroom_bytes: u64,
    /// Text-layer assembly settings.
    pub text: TextConfig,
    /// Text preprocessing bounds. `max_prompt_tokens` is additionally clamped to
    /// the configured context length.
    pub limits: ProcessorLimits,
}

impl Default for LoadOptions {
    fn default() -> Self {
        let qwen = QwenLoadOptions::default();
        Self {
            device: qwen.device,
            context_tokens: qwen.context_tokens,
            max_sequences: qwen.max_sequences,
            prefill_chunk_members: qwen.prefill_chunk_members,
            sampled_prefill_final: qwen.sampled_prefill_final,
            weight_budget_bytes: qwen.weight_budget_bytes,
            continuation_capacity_bytes: qwen.continuation_capacity_bytes,
            headroom_bytes: qwen.headroom_bytes,
            text: TextConfig::default(),
            limits: ProcessorLimits::default(),
        }
    }
}

impl From<LoadOptions> for QwenLoadOptions {
    fn from(value: LoadOptions) -> Self {
        Self {
            device: value.device,
            context_tokens: value.context_tokens,
            max_sequences: value.max_sequences,
            prefill_chunk_members: value.prefill_chunk_members,
            sampled_prefill_final: value.sampled_prefill_final,
            weight_budget_bytes: value.weight_budget_bytes,
            continuation_capacity_bytes: value.continuation_capacity_bytes,
            headroom_bytes: value.headroom_bytes,
            ..Self::default()
        }
    }
}

// Validate the frontend's count/credit envelope before allocating model resources.
fn engine_config(max_sequences: usize) -> Result<EngineConfig, TextError> {
    let defaults = EngineConfig::default();
    if max_sequences == 0 {
        return Err(TextError::InvalidInput("sequence limit must be nonzero"));
    }
    max_sequences
        .checked_add(defaults.max_queued_requests)
        .ok_or(TextError::InvalidInput("text request count overflows"))?;
    // The driver's full-mailbox admission rule must not undercut active capacity.
    let active_events = max_sequences
        .checked_mul(defaults.max_events_per_request)
        .ok_or(TextError::InvalidInput("text event credit count overflows"))?;
    Ok(EngineConfig {
        max_active_requests: max_sequences,
        max_buffered_events: defaults.max_buffered_events.max(active_events),
        ..defaults
    })
}

fn configured_engine(
    executor: impl GenerationExecutor + 'static,
    config: EngineConfig,
) -> Result<Engine, TextError> {
    let policy = SchedulePolicy {
        max_batch_tokens: SchedulePolicy::default()
            .max_batch_tokens
            .min(executor.info().limits.max_batch_tokens),
        ..SchedulePolicy::default()
    };
    Engine::new(executor, config, policy).map_err(TextError::Engine)
}

impl TextOwner {
    /// Load a supported local GGUF model and prepare its CUDA executor.
    ///
    /// Returns the lifecycle owner and the prepared memory report.
    ///
    /// # Errors
    /// Returns an explicit unsupported-architecture error instead of inferring
    /// execution support from the existence of GGUF metadata.
    pub fn load(
        path: impl Into<PathBuf>,
        options: LoadOptions,
    ) -> Result<(Self, MemoryReport), TextError> {
        let config = engine_config(options.max_sequences)?;
        let limits = ProcessorLimits {
            max_prompt_tokens: options
                .limits
                .max_prompt_tokens
                .min(options.context_tokens as usize),
            ..options.limits
        };
        if !limits.is_valid() {
            return Err(TextError::InvalidInput(
                "text preprocessing bounds and context length must be nonzero",
            ));
        }
        let (executor, tokenizer) =
            QwenCuda::load_gguf(path, options.into()).map_err(TextError::Execution)?;
        let processor = Arc::new(GgufProcessor::new(tokenizer, limits));
        let memory = executor.memory_report();
        let engine = configured_engine(executor, config)?;
        let driver = DriverConfig::for_engine(&engine);
        let (shutdown, handle) = Driver::spawn(engine, driver).map_err(TextError::Owner)?;
        let owner = Self::new(processor, shutdown, handle, options.text)?;
        Ok((owner, memory))
    }
}
