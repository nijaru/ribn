use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use crate::{QwenGguf, QwenLayerKind as GgufQwenLayerKind};
use cudarc::driver::{CudaContext, CudaStream};
use engine_core::{DeviceId, StateRequirement, WeightBinding};
use engine_nvidia::NvidiaBackend;
use engine_nvidia::{
    CudaQwen35Decode, CudaQwen35ServingDispatcher, CudaQwen35Weights, GemvMode, QwenCudaProfile,
    QwenLayerKind, StagedTensorSource, wrap_f32_stream,
};
use ribn::{ExecutionError, ExecutorInfo, GenerationLimits};

use crate::execution::{QwenExecution, model_error};
use crate::state;

/// Same-sequence prefill chunk size this backend uses when a caller enables
/// prefill chunking without naming a size.
///
/// Eight is the largest qualified lane (the backend bounds it by its own declared
/// capacity) and the size the prefill qualification measured, so it is both the fastest
/// measured choice and the one with parity evidence. The test below fails if the backend
/// capacity moves without this measured default being re-decided.
pub const DEFAULT_PREFILL_CHUNK_MEMBERS: usize = 8;

/// Preparation limits, not live request scheduling policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QwenLoadOptions {
    pub device: u16,
    pub context_tokens: u32,
    pub max_sequences: usize,
    /// Backend-native projection mode. Warp is the existing qualified default;
    /// scalar is the correctness oracle and integer-dot is explicitly lossy.
    pub gemv_mode: GemvMode,
    /// Experimental compatible decode subgroups. Disabled until device and
    /// mixed-workload performance gates pass; does not change continuation policy.
    pub grouped_decode: bool,
    /// Same-sequence prefill chunk size, or `None` for the serial prefill
    /// path. Enabled by default: the lane is backend-local, and its parity and
    /// serving-effect gates are recorded in
    /// `benchmarks/qwen-prefill-qualification.md`.
    pub prefill_chunk_members: Option<usize>,
    /// Sample a whole prefill chunk's final row instead of re-running that token
    /// on the serial path. Only the warp projection without the chunk-parallel
    /// scan honours it; every other mode and prompt shape keeps the serial step.
    pub sampled_prefill_final: bool,
    /// Optional upper bound for staged weights. Automatic mode uses observed
    /// free device memory minus request-state reservations and headroom.
    pub weight_budget_bytes: Option<u64>,
    /// Continuation bytes the loaded execution may charge in total. `None` keeps the
    /// conservative reservation for `max_sequences` sequences that each reach the
    /// whole context, which is a loading-time policy, not what a request is charged:
    /// since roadmap 4a a request is charged for the tokens it can actually reach.
    pub continuation_capacity_bytes: Option<u64>,
    /// Explicit reserve for preparation/workspace allocations and other users.
    /// This is not a proof that every future CUDA allocation will succeed.
    pub headroom_bytes: u64,
}

impl Default for QwenLoadOptions {
    fn default() -> Self {
        Self {
            device: 0,
            context_tokens: 4096,
            max_sequences: 1,
            gemv_mode: GemvMode::default(),
            grouped_decode: false,
            prefill_chunk_members: Some(DEFAULT_PREFILL_CHUNK_MEMBERS),
            sampled_prefill_final: true,
            weight_budget_bytes: None,
            continuation_capacity_bytes: None,
            headroom_bytes: 512 << 20,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryReport {
    pub free_before_preparation_bytes: u64,
    pub weight_budget_bytes: u64,
    pub reserved_sequence_bytes: u64,
    pub free_after_preparation_bytes: u64,
}

pub(crate) type Backend = NvidiaBackend<CudaQwen35ServingDispatcher>;

pub(crate) fn load(
    path: PathBuf,
    options: QwenLoadOptions,
) -> Result<(QwenExecution<Backend>, MemoryReport), ExecutionError> {
    if options.context_tokens == 0 || options.max_sequences == 0 {
        return Err(ExecutionError::new(
            "Qwen context and sequence limits must be nonzero",
        ));
    }
    let provider =
        QwenGguf::open_with_kv_block_tokens(path, options.context_tokens).map_err(model_error)?;
    validate_cuda_profile(provider.config())?;
    if u64::from(options.context_tokens) > provider.config().context_length() {
        return Err(ExecutionError::new(
            "requested context exceeds the Qwen artifact limit",
        ));
    }
    let vocabulary_size = u32::try_from(
        provider
            .file()
            .tokenizer()
            .map_err(model_error)?
            .tokens()
            .len(),
    )
    .map_err(model_error)?;
    let device = DeviceId::new(options.device);
    let continuation_capacity_bytes =
        continuation_capacity(provider.state_requirements(), &options)?;
    let context = CudaContext::new(usize::from(options.device)).map_err(model_error)?;
    let stream = context.default_stream();
    let free_before_preparation_bytes =
        u64::try_from(context.mem_get_info().map_err(model_error)?.0).map_err(model_error)?;
    let available = free_before_preparation_bytes.checked_sub(continuation_capacity_bytes)
        .and_then(|bytes| bytes.checked_sub(options.headroom_bytes))
        .ok_or_else(|| ExecutionError::new(format!("Qwen state requires {continuation_capacity_bytes} bytes plus {} bytes headroom, but only {free_before_preparation_bytes} bytes are free", options.headroom_bytes)))?;
    let weight_budget_bytes = options.weight_budget_bytes.unwrap_or(available);
    if weight_budget_bytes == 0 || weight_budget_bytes > available {
        return Err(ExecutionError::new(format!(
            "weight budget {weight_budget_bytes} exceeds the available {available} bytes after state/headroom reservations"
        )));
    }
    let staged = Arc::new(stage_weights(
        &provider,
        device,
        &context,
        &stream,
        weight_budget_bytes,
    )?);
    let layer_kinds = layer_kinds(&provider)?;
    let mut executor = CudaQwen35Decode::new(
        &context,
        stream.clone(),
        staged,
        layer_kinds,
        provider.config().rms_norm_epsilon,
    )
    .map_err(model_error)?;
    executor.set_gemv_mode(options.gemv_mode);
    let dispatcher =
        CudaQwen35ServingDispatcher::new(&context, executor, stream.clone(), options.max_sequences)
            .map_err(model_error)?
            .with_grouped_decode(options.grouped_decode);
    // The chunk lane is part of preparation, so its scratch is allocated
    // before the post-preparation memory check below accounts for it.
    let dispatcher = prepare_dispatcher(dispatcher, &options)?;
    // Preparation is outside the scheduler. Publish no ready owner until all
    // uploads and preparation work have completed.
    stream.synchronize().map_err(model_error)?;
    let free_after_preparation_bytes =
        u64::try_from(context.mem_get_info().map_err(model_error)?.0).map_err(model_error)?;
    if free_after_preparation_bytes < continuation_capacity_bytes {
        return Err(ExecutionError::new(format!(
            "prepared Qwen leaves {free_after_preparation_bytes} free bytes, below its {continuation_capacity_bytes}-byte continuation capacity"
        )));
    }
    let requirements: Arc<[StateRequirement]> = provider.state_requirements().into();
    let backend = prepare_execution(dispatcher, device);
    let info = ExecutorInfo {
        name: provider.artifact_id().to_owned(),
        limits: GenerationLimits {
            context_tokens: options.context_tokens,
            max_sequences: options.max_sequences,
            max_batch_tokens: 128,
            max_decode_tokens: 1,
        },
    };
    let execution = QwenExecution::new(
        backend,
        info,
        requirements,
        state::manager(device, continuation_capacity_bytes),
        vocabulary_size,
    );
    Ok((
        execution,
        MemoryReport {
            free_before_preparation_bytes,
            weight_budget_bytes,
            reserved_sequence_bytes: continuation_capacity_bytes,
            free_after_preparation_bytes,
        },
    ))
}

/// Apply the preparation-time lane toggles that cannot be set on construction.
///
/// The chunk lane and sampled-final preference are part of preparation, so their
/// scratch must exist before the post-preparation memory check accounts for it.
fn prepare_dispatcher(
    dispatcher: CudaQwen35ServingDispatcher,
    options: &QwenLoadOptions,
) -> Result<CudaQwen35ServingDispatcher, ExecutionError> {
    let Some(members) = options.prefill_chunk_members else {
        return Ok(dispatcher);
    };
    Ok(dispatcher
        .with_prefill_chunk(members)
        .map_err(model_error)?
        .with_sampled_prefill_final(options.sampled_prefill_final))
}

/// Continuation bytes this loaded execution may charge in total.
///
/// The default keeps the conservative loading-time reservation for `max_sequences`
/// sequences that each reach the whole context. It is a policy for how much device
/// memory the weights may not use, not what a request is charged: a request is charged
/// for the tokens it can actually reach (roadmap 4a).
fn continuation_capacity(
    requirements: &[StateRequirement],
    options: &QwenLoadOptions,
) -> Result<u64, ExecutionError> {
    let reserved = state::capacity(requirements, options.max_sequences)
        .ok_or_else(|| ExecutionError::new("Qwen state reservation overflow"))?;
    let capacity = options.continuation_capacity_bytes.unwrap_or(reserved);
    if capacity == 0 {
        return Err(ExecutionError::new(
            "Qwen continuation capacity must be nonzero",
        ));
    }
    Ok(capacity)
}

/// Wrap the prepared dispatcher in the submission bookkeeping the engine drives.
///
/// The production serving dispatcher resolves staged tensors from its own store, so it needs
/// no name binding; the empty binding still records the device the batch executes on.
fn prepare_execution(dispatcher: CudaQwen35ServingDispatcher, device: DeviceId) -> Backend {
    NvidiaBackend::new(device, WeightBinding::empty(device), dispatcher)
}

/// The current physical executor executes one declared geometry and numerical profile.
/// Describe what the artifact declares and reject disagreements before CUDA access; valid
/// model metadata is not backend support.
fn validate_cuda_profile(config: &crate::QwenConfig) -> Result<(), ExecutionError> {
    let declared = QwenCudaProfile {
        embedding_length: u64::from(config.embedding_length),
        feed_forward_length: u64::from(config.feed_forward_length),
        attention_heads: u64::from(config.attention_heads),
        kv_heads: u64::from(config.kv_heads),
        key_length: u64::from(config.key_length),
        value_length: u64::from(config.value_length),
        rope_dimensions: u64::from(config.rope_dimensions),
        rope_frequency_base: config.rope_frequency_base,
        rms_norm_epsilon: config.rms_norm_epsilon,
        ssm_group_count: u64::from(config.ssm_group_count),
        ssm_inner_size: u64::from(config.ssm_inner_size),
        ssm_state_size: u64::from(config.ssm_state_size),
        ssm_time_step_rank: u64::from(config.ssm_time_step_rank),
        ssm_conv_kernel: u64::from(config.ssm_conv_kernel),
    };
    match QwenCudaProfile::SUPPORTED.mismatch(&declared) {
        Some(mismatch) => Err(ExecutionError::new(mismatch.to_string())),
        None => Ok(()),
    }
}

fn stage_weights(
    provider: &QwenGguf,
    device: DeviceId,
    context: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    budget: u64,
) -> Result<CudaQwen35Weights, ExecutionError> {
    let mut names = vec![
        "token_embd.weight".to_owned(),
        "output_norm.weight".to_owned(),
        "output.weight".to_owned(),
    ];
    let count = provider
        .config()
        .language_layer_count()
        .ok_or_else(|| ExecutionError::new("invalid Qwen language layer count"))?;
    for layer in 0..count {
        let binding = provider
            .layer_weight_binding(device, layer)
            .map_err(model_error)?;
        names.extend(
            binding
                .tensors()
                .iter()
                .map(|tensor| tensor.name().to_owned()),
        );
    }
    let tensors = names
        .iter()
        .map(|name| {
            let reader = provider.open_tensor(name).map_err(model_error)?;
            let spec = reader.spec().clone();
            let value_type = reader.value_type();
            let encoded_bytes = reader.remaining();
            Ok(if matches!(value_type, 0 | 1) {
                StagedTensorSource {
                    spec,
                    value_type,
                    encoded_bytes,
                    reader: Box::new(io::empty()),
                    f32_blocks: Some(Box::new(wrap_f32_stream(reader))),
                }
            } else {
                StagedTensorSource {
                    spec,
                    value_type,
                    encoded_bytes,
                    reader: Box::new(reader),
                    f32_blocks: None,
                }
            })
        })
        .collect::<Result<Vec<_>, ExecutionError>>()?;
    CudaQwen35Weights::stage(context, stream, budget, tensors).map_err(model_error)
}

fn layer_kinds(provider: &QwenGguf) -> Result<Vec<QwenLayerKind>, ExecutionError> {
    let count = provider
        .config()
        .language_layer_count()
        .ok_or_else(|| ExecutionError::new("invalid Qwen language layer count"))?;
    (0..count)
        .map(|layer| {
            provider
                .layer_kind(layer)
                .map(|kind| match kind {
                    GgufQwenLayerKind::Recurrent => QwenLayerKind::Recurrent,
                    GgufQwenLayerKind::FullAttention => QwenLayerKind::FullAttention,
                })
                .map_err(model_error)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::QwenCuda;
    use crate::gguf::config::tests::{encode, metadata};
    use engine_gguf::MetadataValue;

    #[test]
    fn incompatible_model_semantics_reject_before_tokenizer_or_device_access() {
        let path =
            std::env::temp_dir().join(format!("ribn-qwen-profile-{}.gguf", std::process::id()));
        let baseline = metadata();
        std::fs::write(&path, encode(&baseline)).unwrap();
        let model = QwenGguf::open(&path).unwrap();
        validate_cuda_profile(model.config()).unwrap();
        drop(model);
        for (changes, diagnostic) in [
            (
                vec![(
                    "qwen35.attention.layer_norm_rms_epsilon",
                    MetadataValue::F32(1.0e-5),
                )],
                "rms_norm_epsilon",
            ),
            (
                vec![("qwen35.rope.freq_base", MetadataValue::F32(10_000.0))],
                "rope_frequency_base",
            ),
            (
                vec![("qwen35.rope.dimension_count", MetadataValue::U64(128))],
                "rope_dimensions",
            ),
            // Identical aggregate Q/K/V projection widths, different head semantics.
            (
                vec![
                    ("qwen35.attention.head_count", MetadataValue::U64(48)),
                    ("qwen35.attention.head_count_kv", MetadataValue::U64(8)),
                    ("qwen35.attention.key_length", MetadataValue::U64(128)),
                    ("qwen35.attention.value_length", MetadataValue::U64(128)),
                ],
                "attention_heads",
            ),
            (
                vec![("qwen35.ssm.group_count", MetadataValue::U64(8))],
                "ssm_group_count",
            ),
        ] {
            let mut values = baseline.clone();
            for (key, value) in changes {
                values.insert(key.to_owned(), value);
            }
            std::fs::write(&path, encode(&values)).unwrap();
            let result = QwenCuda::load_gguf(
                &path,
                QwenLoadOptions {
                    device: u16::MAX,
                    ..QwenLoadOptions::default()
                },
            );
            let Err(error) = result else {
                panic!("accepted unsupported {diagnostic}")
            };
            assert!(
                error.to_string().contains(diagnostic),
                "{diagnostic}: {error}"
            );
        }
        std::fs::remove_file(path).unwrap();
    }
}

#[cfg(test)]
mod default_lane_tests {
    use super::*;

    #[test]
    fn qualified_prefill_default_equals_the_declared_backend_capacity() {
        assert_eq!(
            DEFAULT_PREFILL_CHUNK_MEMBERS,
            engine_nvidia::MAX_BATCH_MEMBERS,
            "the qualified prefill chunk default must be re-decided when the backend \
             lane capacity changes"
        );
    }
}
