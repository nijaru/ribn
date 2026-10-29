//! Host-orchestrated Gemma 4 BF16 decoder over the qualified native components.
//!
//! Prepared sixteen/eight-row groups and scalar tails share one layer implementation.
//! Embedding lookup, PLE lookup and logit
//! soft-capping stay on the host; projections, norms, rotary, attention and the
//! elementwise layer algebra run on the device. KV owners keep device-resident
//! caches; shared-tail layers read their owner's full group span. K/V rows append
//! through completion-owned device copies. Execution remains synchronous and
//! greedy-only; this is not a competitive serving claim.

use super::{Gemma4CudaWeights, host::HostLookups};
use crate::cpu::CpuParameter;
use crate::{
    AttentionKind, Gemma4Config, Gemma4Error, Gemma4Source, LayerGeometry, LayerParameter,
    ParameterId,
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

mod buffers;
mod preparation;
use buffers::{Buffers, CacheConstruction, DeviceCache, GeometryBuffers};
use preparation::PREFILL_LANES;

struct Executors {
    gemm: CudaBf16Gemm,
    rms: CudaBf16RmsNorm,
    rotary: CudaBf16RotaryApply,
    elementwise: CudaBf16Elementwise,
    attention: CudaBf16Attention,
}

/// Host lookup tables plus one device weight set, executors and reusable buffers.
pub struct Gemma4CudaModel {
    host: HostLookups,
    cuda: Gemma4CudaWeights,
    executors: Executors,
    buffers: Buffers,
    prefill: Vec<Buffers>,
    factors: Vec<CudaBf16RotaryFactors>,
    stream: Arc<CudaStream>,
    pool: Arc<BytePool>,
    max_positions: usize,
    failure: Option<CudaBf16ComputeError>,
    cache_retirement: Option<CacheConstruction>,
}

impl Gemma4CudaModel {
    /// Upload the inventory and prepare bounded scalar/qualified prefill storage.
    /// `scratch_entries` is an attention-scratch ceiling, not an allocation grant;
    /// it must cover one query against the entire declared cache reach.
    /// # Errors
    /// Rejects invalid geometry, allocation/compilation failures or a load failure
    /// whose storage could not be retired.
    pub fn load(
        source: &Gemma4Source,
        stream: Arc<CudaStream>,
        pool: &Arc<BytePool>,
        max_positions: usize,
        scratch_entries: usize,
    ) -> Result<Self, Gemma4Error> {
        let config = source.config();
        let device = stream
            .context()
            .name()
            .map_err(|error| Gemma4Error::new("CUDA decoder preparation", error.to_string()))?;
        let prepared = preparation::prepare(config, max_positions, scratch_entries, &device)?;
        let host = HostLookups::load(source)?;
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
            attention: CudaBf16Attention::new(Arc::clone(&stream), pool, prepared.scratch_entries)
                .map_err(|e| backend(&e))?,
        };
        let buffers = Buffers::new(config, 1, &stream, pool)?;
        let mut prefill = Vec::with_capacity(PREFILL_LANES.len());
        for rows in PREFILL_LANES {
            if rows <= prepared.rows {
                prefill.push(Buffers::new(config, rows, &stream, pool)?);
            }
        }
        let mut factors = Vec::with_capacity(config.layers().len());
        for layer in 0..config.layers().len() {
            let width = config.layers()[layer].head_dim;
            let frequencies = (0..width / 2)
                .map(|pair| config.rotary_inverse_frequency(layer, pair))
                .collect::<Result<Vec<_>, _>>()?;
            factors.push(
                CudaBf16RotaryFactors::new(Arc::clone(&stream), pool, &frequencies, prepared.rows)
                    .map_err(|e| backend(&e))?,
            );
        }
        Ok(Self {
            host,
            cuda,
            executors,
            buffers,
            prefill,
            factors,
            stream,
            pool: Arc::clone(pool),
            max_positions,
            failure: None,
            cache_retirement: None,
        })
    }

    #[must_use]
    pub fn config(&self) -> &Gemma4Config {
        &self.host.config
    }

    /// BF16 payload retained for host embedding lookups and layer scalars.
    /// Excludes metadata, transient buffers, device storage and process RSS.
    #[must_use]
    pub fn host_lookup_bytes(&self) -> usize {
        self.host.bytes()
    }

    #[must_use]
    pub fn max_positions(&self) -> usize {
        self.max_positions
    }

    /// Largest prepared physical prefill width. Qualified preparation chooses
    /// sixteen, eight or scalar according to reach and scratch; other
    /// geometry/devices retain scalar execution.
    #[must_use]
    pub fn prefill_rows(&self) -> usize {
        self.prefill
            .last()
            .map_or(1, |buffers| buffers.hidden.shape().0)
    }

    /// Establish associated-stream completion and retire any retained operation.
    /// This also covers transfer errors whose retirement lives in allocations,
    /// not in a compute-error record. A faulted model is never reopened.
    /// # Errors
    /// Failed drain/cleanup leaves this same owner available for another attempt.
    pub fn retire_failure(&mut self) -> Result<(), CudaBf16Error> {
        self.stream
            .synchronize()
            .map_err(|source| CudaBf16Error::Transport {
                operation: "Gemma model completion",
                source: Box::new(source),
            })?;
        if let Some(failure) = self.failure.as_mut() {
            failure.retire()?;
        }
        if let Some(caches) = self.cache_retirement.as_mut() {
            caches.close()?;
        }
        self.cache_retirement = None;
        Ok(())
    }

    fn step(
        &mut self,
        tokens: &[u32],
        position: usize,
        caches: &mut [Option<DeviceCache>],
    ) -> Result<Vec<u16>, Gemma4Error> {
        if self.cache_retirement.is_some() {
            return Err(Gemma4Error::new(
                "CUDA decoder",
                "cache construction cleanup is unresolved",
            ));
        }
        if let Some(failure) = self.failure.as_ref() {
            return Err(Gemma4Error::new(
                "CUDA decoder",
                format!("model is faulted: {}", failure.cause()),
            ));
        }
        let result = (|| {
            let Self {
                host,
                cuda,
                executors,
                buffers,
                prefill,
                factors,
                ..
            } = self;
            // The caller validates the whole accepted range before cache mutation.
            // All selected working sets and maximum-reach scratch already exist.
            let mut offset = 0;
            for batch in prefill.iter_mut().rev() {
                let rows = batch.hidden.shape().0;
                while tokens.len() - offset >= rows {
                    run_rows(
                        &host.config,
                        host,
                        cuda,
                        executors,
                        batch,
                        factors,
                        caches,
                        position + offset,
                        &tokens[offset..offset + rows],
                    )?;
                    offset += rows;
                }
                if offset == tokens.len() {
                    return project_logits(&host.config, cuda, executors, batch);
                }
            }
            for (relative, token) in tokens[offset..].iter().enumerate() {
                run_rows(
                    &host.config,
                    host,
                    cuda,
                    executors,
                    buffers,
                    factors,
                    caches,
                    position + offset + relative,
                    std::slice::from_ref(token),
                )?;
            }
            project_logits(&host.config, cuda, executors, buffers)
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
    /// Bind one continuation without consuming the caller's model owner.
    /// # Errors
    /// Rejects a poisoned lock or cache allocation/cleanup failure. Keep the
    /// model handle to retry [`Gemma4CudaModel::retire_failure`] on failure.
    pub fn shared(model: &Arc<Mutex<Gemma4CudaModel>>) -> Result<Self, Gemma4Error> {
        let mut guard = model
            .lock()
            .map_err(|_| Gemma4Error::new("CUDA decoder", "model lock poisoned"))?;
        let caches = guard.allocate_caches()?;
        drop(guard);
        Ok(Self {
            model: Arc::clone(model),
            caches,
            position: 0,
        })
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

    /// Release sequence allocations while retaining every failed close for retry.
    pub(crate) fn retire_caches(&mut self) -> Result<(), Gemma4Error> {
        for cache in self.caches.iter_mut().flatten() {
            cache
                .close()
                .map_err(|error| Gemma4Error::new("CUDA cache retirement", error.to_string()))?;
        }
        Ok(())
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
fn run_rows(
    config: &Gemma4Config,
    host: &HostLookups,
    cuda: &mut Gemma4CudaWeights,
    executors: &mut Executors,
    buffers: &mut Buffers,
    factors: &mut [CudaBf16RotaryFactors],
    caches: &mut [Option<DeviceCache>],
    position: usize,
    tokens: &[u32],
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

    let embedding = &host.token_embedding;
    let ple_embedding = &host.ple_embedding;
    let rows = tokens.len();
    let positions: Vec<u32> = (position..position + rows)
        .map(|value| {
            u32::try_from(value).map_err(|_| CudaBf16Error::Invalid("position exceeds u32"))
        })
        .collect::<Result<_, _>>()?;
    let mut hidden_rows = Vec::with_capacity(rows * config.hidden_size());
    let mut identity = Vec::with_capacity(rows * packed);
    for &token in tokens {
        hidden_rows.extend(scaled_row(row(embedding, token)?, embed));
        identity.extend(scaled_row(row(ple_embedding, token)?, ple_embed));
    }
    hidden.upload(&encode(&hidden_rows))?;

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
    let projection = decode_bits(&ple_normed.read_chunk(0, rows * packed * 2)?);

    for (layer, geometry) in config.layers().iter().enumerate() {
        let mut context = Vec::with_capacity(rows * ple);
        for row in 0..rows {
            let base = row * packed + layer * ple;
            context.extend(
                crate::ple_combine(&projection[base..base + ple], &identity[base..base + ple])
                    .map_err(|_| CudaBf16Error::Invalid("PLE combine failed"))?,
            );
        }
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
            &positions,
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
        final_input,
        final_norm,
        logits,
        ..
    } = buffers;
    let rows = hidden.shape().0;
    let input = if rows == 1 {
        hidden
    } else {
        final_input.copy_rows_from(0, hidden, rows - 1..rows)?;
        final_input
    };
    cuda.rms_norm_into(
        ParameterId::FinalNorm,
        &mut executors.rms,
        input,
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
    host: &HostLookups,
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
    positions: &[u32],
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
    factors[layer].prepare_into(positions, &mut buffers.cos, &mut buffers.sin)?;
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
            key_sequence: position + positions.len(),
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
    elementwise.scale_into(hidden_next, load(host.layer_scalars[layer]), hidden)?;
    Ok(())
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
