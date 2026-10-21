//! Host-orchestrated Gemma 4 BF16 decoder over the qualified native components.
//!
//! One token is processed at a time. Embedding lookup, PLE lookup and logit
//! soft-capping stay on the host; projections, norms, rotary, attention and the
//! elementwise layer algebra run on the device. KV owners keep device-resident
//! caches; shared-tail layers read their owner's cache. K/V rows append through
//! completion-owned device copies. Execution remains synchronous and one token at
//! a time, so this is correctness-first, not a competitive path.

use super::Gemma4CudaWeights;
use crate::cpu::CpuParameter;
use crate::{
    AttentionKind, Gemma4Config, Gemma4CpuWeights, Gemma4Error, Gemma4Source, LayerGeometry,
    LayerParameter, ParameterId,
};
use cudarc::driver::CudaStream;
use engine_nvidia::{
    AttentionGeometry, AttentionSpan, CudaBf16Attention, CudaBf16ComputeError, CudaBf16Elementwise,
    CudaBf16Error, CudaBf16Gemm, CudaBf16Matrix, CudaBf16RmsNorm, CudaBf16RotaryApply,
    CudaBf16RotaryFactors,
};
use ribn_foundation::BytePool;
use std::sync::{Arc, Mutex};

// The allocation owner requires nonempty staging even for device-only buffers.
const DEVICE_ONLY_STAGING: usize = 1;
const WEIGHT_STAGING: usize = 1024 * 1024;
const GEMM_WORKSPACE: usize = 8 * 1024 * 1024;

/// Device-resident keys and values for one KV owner layer.
struct DeviceCache {
    keys: CudaBf16Matrix,
    values: CudaBf16Matrix,
}

/// Per-geometry working set; local and global layers differ in head width.
struct GeometryBuffers {
    width: usize,
    cos: CudaBf16Matrix,
    sin: CudaBf16Matrix,
    normed: CudaBf16Matrix,
    queries: CudaBf16Matrix,
    query_norm: CudaBf16Matrix,
    query_rotary: CudaBf16Matrix,
    keys: CudaBf16Matrix,
    key_norm: CudaBf16Matrix,
    key_rotary: CudaBf16Matrix,
    values: CudaBf16Matrix,
    value_norm: CudaBf16Matrix,
    attended: CudaBf16Matrix,
    projected: CudaBf16Matrix,
    post_attention: CudaBf16Matrix,
    ffn_norm: CudaBf16Matrix,
    gate: CudaBf16Matrix,
    up: CudaBf16Matrix,
    activated: CudaBf16Matrix,
    product: CudaBf16Matrix,
    down: CudaBf16Matrix,
    post_feedforward: CudaBf16Matrix,
    ple_gate: CudaBf16Matrix,
    ple_activated: CudaBf16Matrix,
    ple_product: CudaBf16Matrix,
    ple_projected: CudaBf16Matrix,
    post_ple: CudaBf16Matrix,
}

/// Model-wide buffers shared by every layer.
struct Buffers {
    hidden: CudaBf16Matrix,
    hidden_next: CudaBf16Matrix,
    final_norm: CudaBf16Matrix,
    logits: CudaBf16Matrix,
    ple_projection: CudaBf16Matrix,
    ple_projected_scaled: CudaBf16Matrix,
    ple_normed: CudaBf16Matrix,
    ple_context: CudaBf16Matrix,
    local: GeometryBuffers,
    global: GeometryBuffers,
}

struct Executors {
    gemm: CudaBf16Gemm,
    rms: CudaBf16RmsNorm,
    rotary: CudaBf16RotaryApply,
    elementwise: CudaBf16Elementwise,
    attention: CudaBf16Attention,
}

/// Host inventory plus one device weight set, executors and reusable buffers.
pub struct Gemma4CudaModel {
    host: Arc<Gemma4CpuWeights>,
    cuda: Gemma4CudaWeights,
    executors: Executors,
    buffers: Buffers,
    factors: Vec<CudaBf16RotaryFactors>,
    stream: Arc<CudaStream>,
    pool: Arc<BytePool>,
    max_positions: usize,
    failure: Option<CudaBf16ComputeError>,
}

impl Gemma4CudaModel {
    /// Upload the validated inventory and prepare one token-step working set.
    /// # Errors
    /// Rejects invalid geometry, allocation/compilation failures or a load failure
    /// whose storage could not be retired.
    pub fn load(
        source: &Gemma4Source,
        host: Gemma4CpuWeights,
        stream: Arc<CudaStream>,
        pool: &Arc<BytePool>,
        max_positions: usize,
        scratch_entries: usize,
    ) -> Result<Self, Gemma4Error> {
        let config = host.config().clone();
        if max_positions == 0 || max_positions > config.max_position_embeddings() {
            return Err(Gemma4Error::new(
                "CUDA decoder",
                "cache capacity outside the model context",
            ));
        }
        let cuda = match Gemma4CudaWeights::load(source, Arc::clone(&stream), pool, WEIGHT_STAGING)
        {
            Ok(cuda) => cuda,
            Err(mut failure) => {
                let message = failure.to_string();
                let retired = failure.retire().is_ok();
                let detail = if retired {
                    message
                } else {
                    format!("{message}; load retirement unresolved")
                };
                return Err(Gemma4Error::new("CUDA decoder", detail));
            }
        };
        let executors = Executors {
            gemm: CudaBf16Gemm::new(Arc::clone(&stream), pool, GEMM_WORKSPACE)
                .map_err(|e| backend(&e))?,
            rms: CudaBf16RmsNorm::new(Arc::clone(&stream)).map_err(|e| backend(&e))?,
            rotary: CudaBf16RotaryApply::new(Arc::clone(&stream)).map_err(|e| backend(&e))?,
            elementwise: CudaBf16Elementwise::new(Arc::clone(&stream)).map_err(|e| backend(&e))?,
            attention: CudaBf16Attention::new(Arc::clone(&stream), pool, scratch_entries)
                .map_err(|e| backend(&e))?,
        };
        let hidden = config.hidden_size();
        let ple = config.ple_dim();
        let packed = config.packed_ple_dim();
        let vocab = config.vocab_size();
        let buffers = Buffers {
            hidden: matrix(1, hidden, &stream, pool, hidden * 2)?,
            hidden_next: matrix(1, hidden, &stream, pool, DEVICE_ONLY_STAGING)?,
            final_norm: matrix(1, hidden, &stream, pool, DEVICE_ONLY_STAGING)?,
            logits: matrix(1, vocab, &stream, pool, vocab * 2)?,
            ple_projection: matrix(1, packed, &stream, pool, DEVICE_ONLY_STAGING)?,
            ple_projected_scaled: matrix(1, packed, &stream, pool, DEVICE_ONLY_STAGING)?,
            ple_normed: matrix(1, packed, &stream, pool, packed * 2)?,
            ple_context: matrix(1, ple, &stream, pool, ple * 2)?,
            local: GeometryBuffers::new(&config, AttentionWidth::Local, &stream, pool)?,
            global: GeometryBuffers::new(&config, AttentionWidth::Global, &stream, pool)?,
        };
        let mut factors = Vec::with_capacity(config.layers().len());
        for layer in 0..config.layers().len() {
            let width = config.layers()[layer].head_dim;
            let frequencies = (0..width / 2)
                .map(|pair| config.rotary_inverse_frequency(layer, pair))
                .collect::<Result<Vec<_>, _>>()?;
            factors.push(
                CudaBf16RotaryFactors::new(Arc::clone(&stream), pool, &frequencies, 1)
                    .map_err(|e| backend(&e))?,
            );
        }
        Ok(Self {
            host: Arc::new(host),
            cuda,
            executors,
            buffers,
            factors,
            stream,
            pool: Arc::clone(pool),
            max_positions,
            failure: None,
        })
    }

    #[must_use]
    pub fn config(&self) -> &Gemma4Config {
        self.host.config()
    }

    #[must_use]
    pub fn max_positions(&self) -> usize {
        self.max_positions
    }

    /// Retire a failed device operation retained by the last step.
    /// # Errors
    /// Returns the backend's drain/cleanup failure for another attempt.
    pub fn retire_failure(&mut self) -> Result<(), CudaBf16Error> {
        match self.failure.as_mut() {
            Some(failure) => failure.retire(),
            None => Ok(()),
        }
    }

    fn step(
        &mut self,
        tokens: &[u32],
        position: usize,
        caches: &mut [Option<DeviceCache>],
    ) -> Result<Vec<u16>, Gemma4Error> {
        if let Some(failure) = self.failure.as_ref() {
            return Err(Gemma4Error::new(
                "CUDA decoder",
                format!("device failure is not retired: {}", failure.cause()),
            ));
        }
        let result = (|| {
            let Self {
                host,
                cuda,
                executors,
                buffers,
                factors,
                ..
            } = self;
            // The caller validates the entire range before any cache mutation.
            for (offset, &token) in tokens.iter().enumerate() {
                run_token(
                    host.config(),
                    host,
                    cuda,
                    executors,
                    buffers,
                    factors,
                    caches,
                    position + offset,
                    token,
                )?;
            }
            project_logits(host.config(), cuda, executors, buffers)
        })();
        match result {
            Ok(logits) => Ok(logits),
            Err(failure) => {
                let message = failure.cause().to_string();
                self.failure = Some(failure);
                Err(Gemma4Error::new("CUDA decoder", message))
            }
        }
    }
}

/// One sequence's continuation state.
pub struct Gemma4CudaDecoder {
    model: Arc<Mutex<Gemma4CudaModel>>,
    caches: Vec<Option<DeviceCache>>,
    position: usize,
}

impl Gemma4CudaDecoder {
    /// Bind one continuation to a shared device model.
    /// # Errors
    /// Rejects a poisoned model lock or cache-allocation failure.
    pub fn shared(model: Arc<Mutex<Gemma4CudaModel>>) -> Result<Self, Gemma4Error> {
        let guard = model
            .lock()
            .map_err(|_| Gemma4Error::new("CUDA decoder", "model lock poisoned"))?;
        let config = guard.config();
        let mut caches = Vec::with_capacity(config.layers().len());
        for (layer, geometry) in config.layers().iter().enumerate() {
            caches.push(
                (geometry.kv_owner == layer)
                    .then(|| {
                        DeviceCache::allocate(
                            guard.max_positions,
                            config.num_key_value_heads() * geometry.head_dim,
                            &guard.stream,
                            &guard.pool,
                        )
                    })
                    .transpose()?,
            );
        }
        drop(guard);
        Ok(Self {
            model,
            caches,
            position: 0,
        })
    }

    /// Prepare and bind a fresh model and sequence.
    /// # Errors
    /// As [`Gemma4CudaModel::load`] or [`Self::shared`].
    pub fn load(
        source: &Gemma4Source,
        host: Gemma4CpuWeights,
        stream: Arc<CudaStream>,
        pool: &Arc<BytePool>,
        max_positions: usize,
        scratch_entries: usize,
    ) -> Result<Self, Gemma4Error> {
        let model =
            Gemma4CudaModel::load(source, host, stream, pool, max_positions, scratch_entries)?;
        Self::shared(Arc::new(Mutex::new(model)))
    }

    /// Append a prompt chunk and return logits for its last token. Intermediate
    /// tokens advance all layer/cache state without computing discarded logits.
    /// # Errors
    /// Rejects an empty chunk, invalid tokens or excess capacity before mutation.
    /// A device failure faults the shared model and retains its retirement owner.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<u16>, Gemma4Error> {
        if tokens.is_empty() {
            return Err(Gemma4Error::new("CUDA decoder", "empty prompt chunk"));
        }
        let mut model = self
            .model
            .lock()
            .map_err(|_| Gemma4Error::new("CUDA decoder", "model lock poisoned"))?;
        if tokens
            .iter()
            .any(|&token| token as usize >= model.config().vocab_size())
        {
            return Err(Gemma4Error::new(
                "CUDA decoder",
                "token outside the vocabulary",
            ));
        }
        let end = self
            .position
            .checked_add(tokens.len())
            .filter(|&end| end <= model.max_positions)
            .ok_or_else(|| {
                Gemma4Error::new("CUDA decoder", "chunk exceeds the device cache capacity")
            })?;
        model.config().validate_text_tokens(tokens)?;
        let logits = model.step(tokens, self.position, &mut self.caches)?;
        self.position = end;
        Ok(logits)
    }

    /// Append one token and return its post-soft-cap logits.
    /// # Errors
    /// Rejects invalid tokens or a step past the cache capacity.
    pub fn decode(&mut self, token: u32) -> Result<Vec<u16>, Gemma4Error> {
        self.prefill(&[token])
    }

    #[must_use]
    pub fn position(&self) -> usize {
        self.position
    }

    /// Clear continuation state; weights stay bound.
    pub fn reset(&mut self) {
        self.position = 0;
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "one decoder step's semantic owners, kept explicit"
)]
fn run_token(
    config: &Gemma4Config,
    host: &Gemma4CpuWeights,
    cuda: &mut Gemma4CudaWeights,
    executors: &mut Executors,
    buffers: &mut Buffers,
    factors: &mut [CudaBf16RotaryFactors],
    caches: &mut [Option<DeviceCache>],
    position: usize,
    token: u32,
) -> Result<(), CudaBf16ComputeError> {
    let Buffers {
        hidden,
        hidden_next,
        ple_projection,
        ple_projected_scaled,
        ple_normed,
        ple_context,
        local,
        global,
        ..
    } = buffers;
    let Scale {
        embed,
        ple_embed,
        ple_projection: ple_projection_factor,
    } = scales(config);
    let ple = config.ple_dim();
    let packed = config.packed_ple_dim();

    let embedding = parameter(host, ParameterId::TokenEmbedding)?;
    let ple_embedding = parameter(host, ParameterId::PleEmbedding)?;
    let hidden_row = scaled_row(row(embedding, token)?, embed);
    hidden.upload(&encode(&hidden_row))?;
    let identity = scaled_row(row(ple_embedding, token)?, ple_embed);

    cuda.linear_into(
        ParameterId::PleProjection,
        &mut executors.gemm,
        hidden,
        ple_projection,
    )?;
    executors.elementwise.scale_into(
        ple_projection,
        ple_projection_factor,
        ple_projected_scaled,
    )?;
    cuda.rms_norm_into(
        ParameterId::PleNorm,
        &mut executors.rms,
        ple_projected_scaled,
        ple_normed,
    )?;
    let projection = decode_bits(&ple_normed.read_chunk(0, packed * 2)?);

    for (layer, geometry) in config.layers().iter().enumerate() {
        let context = crate::ple_combine(
            &projection[layer * ple..(layer + 1) * ple],
            &identity[layer * ple..(layer + 1) * ple],
        )
        .map_err(|_| CudaBf16Error::Invalid("PLE combine failed"))?;
        ple_context.upload(&encode(&context))?;
        let buffers = if geometry.head_dim == local.width {
            &mut *local
        } else {
            &mut *global
        };
        layer_forward(
            config,
            host,
            cuda,
            executors,
            factors,
            caches,
            hidden,
            hidden_next,
            ple_context,
            buffers,
            layer,
            geometry,
            position,
        )?;
    }

    Ok(())
}

fn project_logits(
    config: &Gemma4Config,
    cuda: &mut Gemma4CudaWeights,
    executors: &mut Executors,
    buffers: &mut Buffers,
) -> Result<Vec<u16>, CudaBf16ComputeError> {
    let Buffers {
        hidden,
        final_norm,
        logits,
        ..
    } = buffers;
    cuda.rms_norm_into(
        ParameterId::FinalNorm,
        &mut executors.rms,
        hidden,
        final_norm,
    )?;
    cuda.linear_into(
        ParameterId::TokenEmbedding,
        &mut executors.gemm,
        final_norm,
        logits,
    )?;
    let raw = decode_bits(&logits.read_chunk(0, config.vocab_size() * 2)?);
    let capped = crate::softcap(&raw, config.final_logit_softcapping())
        .map_err(|_| CudaBf16Error::Invalid("soft cap failed"))?;
    Ok(capped)
}

#[expect(
    clippy::too_many_arguments,
    reason = "one layer's semantic inputs and stage buffers, kept explicit"
)]
#[expect(
    clippy::too_many_lines,
    reason = "one layer's ordered semantic stages, kept explicit"
)]
fn layer_forward(
    config: &Gemma4Config,
    host: &Gemma4CpuWeights,
    cuda: &mut Gemma4CudaWeights,
    executors: &mut Executors,
    factors: &mut [CudaBf16RotaryFactors],
    caches: &mut [Option<DeviceCache>],
    hidden: &mut CudaBf16Matrix,
    hidden_next: &mut CudaBf16Matrix,
    ple_context: &CudaBf16Matrix,
    buffers: &mut GeometryBuffers,
    layer: usize,
    geometry: &LayerGeometry,
    position: usize,
) -> Result<(), CudaBf16ComputeError> {
    let Executors {
        gemm,
        rms,
        rotary,
        elementwise,
        attention,
    } = executors;
    let heads = config.num_attention_heads();
    let kv_heads = config.num_key_value_heads();
    let width = geometry.head_dim;
    let window = match geometry.kind {
        AttentionKind::Sliding => Some(config.sliding_window()),
        AttentionKind::Full => None,
    };

    cuda.rms_norm_into(
        layer_id(layer, LayerParameter::InputNorm),
        rms,
        hidden,
        &mut buffers.normed,
    )?;
    cuda.linear_into(
        layer_id(layer, LayerParameter::Query),
        gemm,
        &mut buffers.normed,
        &mut buffers.queries,
    )?;
    cuda.rms_norm_into(
        layer_id(layer, LayerParameter::QueryNorm),
        rms,
        &mut buffers.queries,
        &mut buffers.query_norm,
    )?;
    let position_u32 =
        u32::try_from(position).map_err(|_| CudaBf16Error::Invalid("position exceeds u32"))?;
    factors[layer].prepare_into(&[position_u32], &mut buffers.cos, &mut buffers.sin)?;
    rotary.apply_into(
        &mut buffers.cos,
        &mut buffers.sin,
        &mut buffers.query_norm,
        &mut buffers.query_rotary,
    )?;

    if geometry.kv_owner == layer {
        cuda.linear_into(
            layer_id(layer, LayerParameter::Key),
            gemm,
            &mut buffers.normed,
            &mut buffers.keys,
        )?;
        cuda.rms_norm_into(
            layer_id(layer, LayerParameter::KeyNorm),
            rms,
            &mut buffers.keys,
            &mut buffers.key_norm,
        )?;
        rotary.apply_into(
            &mut buffers.cos,
            &mut buffers.sin,
            &mut buffers.key_norm,
            &mut buffers.key_rotary,
        )?;
        cuda.linear_into(
            layer_id(layer, LayerParameter::Value),
            gemm,
            &mut buffers.normed,
            &mut buffers.values,
        )?;
        cuda.value_norm_into(layer, rms, &mut buffers.values, &mut buffers.value_norm)?;
        let cache = caches[layer]
            .as_mut()
            .ok_or(CudaBf16Error::Invalid("missing KV owner cache"))?;
        cache.keys.copy_rows_from(
            position,
            &buffers.key_rotary,
            0..buffers.key_rotary.shape().0,
        )?;
        cache.values.copy_rows_from(
            position,
            &buffers.value_norm,
            0..buffers.value_norm.shape().0,
        )?;
    }

    let cache = caches[geometry.kv_owner]
        .as_ref()
        .ok_or(CudaBf16Error::Invalid("missing KV owner cache"))?;
    attention.attend_into(
        &buffers.query_rotary,
        &cache.keys,
        &cache.values,
        AttentionGeometry {
            heads,
            kv_heads,
            width,
            window,
        },
        AttentionSpan {
            query_start: position,
            key_start: 0,
            key_sequence: position + 1,
        },
        &mut buffers.attended,
    )?;
    cuda.linear_into(
        layer_id(layer, LayerParameter::AttentionOutput),
        gemm,
        &mut buffers.attended,
        &mut buffers.projected,
    )?;
    cuda.rms_norm_into(
        layer_id(layer, LayerParameter::PostAttentionNorm),
        rms,
        &mut buffers.projected,
        &mut buffers.post_attention,
    )?;
    elementwise.add_into(hidden, &buffers.post_attention, hidden_next)?;

    cuda.rms_norm_into(
        layer_id(layer, LayerParameter::PreFeedforwardNorm),
        rms,
        hidden_next,
        &mut buffers.ffn_norm,
    )?;
    cuda.linear_into(
        layer_id(layer, LayerParameter::Gate),
        gemm,
        &mut buffers.ffn_norm,
        &mut buffers.gate,
    )?;
    cuda.linear_into(
        layer_id(layer, LayerParameter::Up),
        gemm,
        &mut buffers.ffn_norm,
        &mut buffers.up,
    )?;
    elementwise.gelu_tanh_into(&buffers.gate, &mut buffers.activated)?;
    elementwise.multiply_into(&buffers.activated, &buffers.up, &mut buffers.product)?;
    cuda.linear_into(
        layer_id(layer, LayerParameter::Down),
        gemm,
        &mut buffers.product,
        &mut buffers.down,
    )?;
    cuda.rms_norm_into(
        layer_id(layer, LayerParameter::PostFeedforwardNorm),
        rms,
        &mut buffers.down,
        &mut buffers.post_feedforward,
    )?;
    elementwise.add_into(hidden_next, &buffers.post_feedforward, hidden)?;

    cuda.linear_into(
        layer_id(layer, LayerParameter::PleGate),
        gemm,
        hidden,
        &mut buffers.ple_gate,
    )?;
    elementwise.gelu_tanh_into(&buffers.ple_gate, &mut buffers.ple_activated)?;
    elementwise.multiply_into(
        &buffers.ple_activated,
        ple_context,
        &mut buffers.ple_product,
    )?;
    cuda.linear_into(
        layer_id(layer, LayerParameter::PleProjection),
        gemm,
        &mut buffers.ple_product,
        &mut buffers.ple_projected,
    )?;
    cuda.rms_norm_into(
        layer_id(layer, LayerParameter::PostPleNorm),
        rms,
        &mut buffers.ple_projected,
        &mut buffers.post_ple,
    )?;
    elementwise.add_into(hidden, &buffers.post_ple, hidden_next)?;
    let scalar = host
        .layer(layer, LayerParameter::LayerScalar)
        .map_err(|_| CudaBf16Error::Invalid("missing layer scalar"))?;
    elementwise.scale_into(hidden_next, load(scalar.values()[0]), hidden)?;
    Ok(())
}

impl GeometryBuffers {
    fn new(
        config: &Gemma4Config,
        kind: AttentionWidth,
        stream: &Arc<CudaStream>,
        pool: &Arc<BytePool>,
    ) -> Result<Self, Gemma4Error> {
        let width = kind.width(config);
        let heads = config.num_attention_heads();
        let kv_heads = config.num_key_value_heads();
        let hidden = config.hidden_size();
        let intermediate = config.intermediate_size();
        let ple = config.ple_dim();
        let q = heads * width;
        let kv = kv_heads * width;
        Ok(Self {
            width,
            cos: matrix(1, width, stream, pool, DEVICE_ONLY_STAGING)?,
            sin: matrix(1, width, stream, pool, DEVICE_ONLY_STAGING)?,
            normed: matrix(1, hidden, stream, pool, DEVICE_ONLY_STAGING)?,
            queries: matrix(1, q, stream, pool, DEVICE_ONLY_STAGING)?,
            query_norm: matrix(1, q, stream, pool, DEVICE_ONLY_STAGING)?,
            query_rotary: matrix(1, q, stream, pool, DEVICE_ONLY_STAGING)?,
            keys: matrix(1, kv, stream, pool, DEVICE_ONLY_STAGING)?,
            key_norm: matrix(1, kv, stream, pool, DEVICE_ONLY_STAGING)?,
            key_rotary: matrix(1, kv, stream, pool, DEVICE_ONLY_STAGING)?,
            values: matrix(1, kv, stream, pool, DEVICE_ONLY_STAGING)?,
            value_norm: matrix(1, kv, stream, pool, DEVICE_ONLY_STAGING)?,
            attended: matrix(1, q, stream, pool, DEVICE_ONLY_STAGING)?,
            projected: matrix(1, hidden, stream, pool, DEVICE_ONLY_STAGING)?,
            post_attention: matrix(1, hidden, stream, pool, DEVICE_ONLY_STAGING)?,
            ffn_norm: matrix(1, hidden, stream, pool, DEVICE_ONLY_STAGING)?,
            gate: matrix(1, intermediate, stream, pool, DEVICE_ONLY_STAGING)?,
            up: matrix(1, intermediate, stream, pool, DEVICE_ONLY_STAGING)?,
            activated: matrix(1, intermediate, stream, pool, DEVICE_ONLY_STAGING)?,
            product: matrix(1, intermediate, stream, pool, DEVICE_ONLY_STAGING)?,
            down: matrix(1, hidden, stream, pool, DEVICE_ONLY_STAGING)?,
            post_feedforward: matrix(1, hidden, stream, pool, DEVICE_ONLY_STAGING)?,
            ple_gate: matrix(1, ple, stream, pool, DEVICE_ONLY_STAGING)?,
            ple_activated: matrix(1, ple, stream, pool, DEVICE_ONLY_STAGING)?,
            ple_product: matrix(1, ple, stream, pool, DEVICE_ONLY_STAGING)?,
            ple_projected: matrix(1, hidden, stream, pool, DEVICE_ONLY_STAGING)?,
            post_ple: matrix(1, hidden, stream, pool, DEVICE_ONLY_STAGING)?,
        })
    }
}

#[derive(Clone, Copy)]
enum AttentionWidth {
    Local,
    Global,
}

impl AttentionWidth {
    fn width(self, config: &Gemma4Config) -> usize {
        let kind = match self {
            Self::Local => AttentionKind::Sliding,
            Self::Global => AttentionKind::Full,
        };
        config
            .layers()
            .iter()
            .find(|layer| layer.kind == kind)
            .map_or(0, |layer| layer.head_dim)
    }
}

impl DeviceCache {
    fn allocate(
        capacity: usize,
        columns: usize,
        stream: &Arc<CudaStream>,
        pool: &Arc<BytePool>,
    ) -> Result<Self, Gemma4Error> {
        Ok(Self {
            keys: matrix(capacity, columns, stream, pool, DEVICE_ONLY_STAGING)?,
            values: matrix(capacity, columns, stream, pool, DEVICE_ONLY_STAGING)?,
        })
    }
}

struct Scale {
    embed: f32,
    ple_embed: f32,
    ple_projection: f32,
}

fn scales(config: &Gemma4Config) -> Scale {
    #[expect(
        clippy::cast_precision_loss,
        reason = "model dimensions are small positive integers"
    )]
    let hidden = config.hidden_size() as f32;
    #[expect(
        clippy::cast_precision_loss,
        reason = "model dimensions are small positive integers"
    )]
    let ple = config.ple_dim() as f32;
    Scale {
        embed: bf16_factor(hidden.sqrt()),
        ple_embed: bf16_factor(ple.sqrt()),
        ple_projection: 1.0 / hidden.sqrt(),
    }
}

fn layer_id(layer: usize, parameter: LayerParameter) -> ParameterId {
    ParameterId::Layer { layer, parameter }
}

fn parameter(host: &Gemma4CpuWeights, id: ParameterId) -> Result<&CpuParameter, CudaBf16Error> {
    host.parameter(id)
        .ok_or(CudaBf16Error::Invalid("missing active model parameter"))
}

fn row(parameter: &CpuParameter, token: u32) -> Result<&[u16], CudaBf16Error> {
    let shape = parameter.shape();
    let token = token as usize;
    if shape.len() != 2 || token >= shape[0] {
        return Err(CudaBf16Error::Invalid("token outside its embedding table"));
    }
    Ok(&parameter.values()[token * shape[1]..(token + 1) * shape[1]])
}

fn scaled_row(values: &[u16], factor: f32) -> Vec<u16> {
    values
        .iter()
        .map(|&bits| crate::bf16(load(bits) * factor))
        .collect()
}

fn matrix(
    rows: usize,
    columns: usize,
    stream: &Arc<CudaStream>,
    pool: &Arc<BytePool>,
    staging: usize,
) -> Result<CudaBf16Matrix, Gemma4Error> {
    CudaBf16Matrix::allocate(rows, columns, Arc::clone(stream), pool, staging)
        .map_err(|e| Gemma4Error::new("CUDA decoder", e.to_string()))
}

fn backend(error: &CudaBf16ComputeError) -> Gemma4Error {
    Gemma4Error::new("CUDA decoder", error.to_string())
}

fn load(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn bf16_factor(value: f32) -> f32 {
    load(crate::bf16(value))
}

fn encode(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn decode_bits(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect()
}
