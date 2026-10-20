//! Prepared Gemma weights resident on one Metal device.
//!
//! Loading uploads the already validated host inventory without conversion and
//! keeps the tied output head as the single token-embedding buffer.

use crate::{Gemma4Config, Gemma4CpuWeights, Gemma4Error, LayerParameter, ParameterId};
use engine_metal::{MetalBf16Buffer, MetalBf16Executor};
use std::collections::BTreeMap;
use std::sync::Arc;

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

use crate::AttentionKind;
use crate::cpu::{
    add, argmax, factors, gelu_tanh, multiply, ple_combine, scale, softcap, validate_inventory,
};
use engine_metal::{MetalAttentionConfig, MetalBf16Error};

/// Host orchestration for one Metal decoder step.
struct MetalOps<'a> {
    executor: &'a MetalBf16Executor,
}

fn metal_error(error: &MetalBf16Error) -> Gemma4Error {
    Gemma4Error::new("Metal decoder", error.to_string())
}

impl MetalOps<'_> {
    fn linear(
        &self,
        input: &[u16],
        rows: usize,
        weight: &MetalBf16Buffer,
    ) -> Result<Vec<u16>, Gemma4Error> {
        if rows == 0 || !input.len().is_multiple_of(rows) {
            return Err(Gemma4Error::new("Metal decoder", "linear input rows"));
        }
        let inner = input.len() / rows;
        if inner == 0 || !weight.elements().is_multiple_of(inner) {
            return Err(Gemma4Error::new("Metal decoder", "linear weight shape"));
        }
        let columns = weight.elements() / inner;
        let buffer = self
            .executor
            .upload(input)
            .map_err(|error| metal_error(&error))?;
        let output = self
            .executor
            .linear(&buffer, weight, rows, columns, inner)
            .map_err(|error| metal_error(&error))?;
        Ok(output.bits())
    }

    fn rms(
        &self,
        input: &[u16],
        rows: usize,
        width: usize,
        epsilon: f32,
        scale: Option<&MetalBf16Buffer>,
    ) -> Result<Vec<u16>, Gemma4Error> {
        let buffer = self
            .executor
            .upload(input)
            .map_err(|error| metal_error(&error))?;
        let output = self
            .executor
            .rms_norm(&buffer, rows, width, epsilon, scale)
            .map_err(|error| metal_error(&error))?;
        Ok(output.bits())
    }

    fn rotary(
        &self,
        input: &[u16],
        rows: usize,
        heads: usize,
        width: usize,
        cosine: &[u16],
        sine: &[u16],
    ) -> Result<Vec<u16>, Gemma4Error> {
        let input = self
            .executor
            .upload(input)
            .map_err(|error| metal_error(&error))?;
        let cosine = self
            .executor
            .upload(cosine)
            .map_err(|error| metal_error(&error))?;
        let sine = self
            .executor
            .upload(sine)
            .map_err(|error| metal_error(&error))?;
        let output = self
            .executor
            .rotary_apply(&input, &cosine, &sine, rows, heads, width)
            .map_err(|error| metal_error(&error))?;
        Ok(output.bits())
    }

    fn attention(
        &self,
        query: &[u16],
        keys: &[u16],
        values: &[u16],
        config: MetalAttentionConfig,
        query_start: usize,
        key_start: usize,
    ) -> Result<Vec<u16>, Gemma4Error> {
        let query = self
            .executor
            .upload(query)
            .map_err(|error| metal_error(&error))?;
        let keys = self
            .executor
            .upload(keys)
            .map_err(|error| metal_error(&error))?;
        let values = self
            .executor
            .upload(values)
            .map_err(|error| metal_error(&error))?;
        let output = self
            .executor
            .attention(&query, &keys, &values, config, query_start, key_start)
            .map_err(|error| metal_error(&error))?;
        Ok(output.bits())
    }
}

struct OwnerCache {
    keys: Vec<u16>,
    values: Vec<u16>,
}

/// Host orchestration of one sequence over Metal-resident Gemma weights.
///
/// Matmuls, normalization, rotary and attention run on the GPU; embeddings,
/// elementwise algebra and the soft cap reuse the validated host primitives so
/// this slice needs no additional device kernels. The cache keeps complete
/// per-owner history and masks local spans rather than evicting them.
/// Host inventory, device-resident weights and one Metal executor, shared by
/// every sequence that runs on this device.
pub struct Gemma4MetalModel {
    host: Arc<Gemma4CpuWeights>,
    metal: Gemma4MetalWeights,
    executor: MetalBf16Executor,
}

impl Gemma4MetalModel {
    /// Upload the validated inventory once.
    /// # Errors
    /// Rejects an inconsistent inventory or a Metal preparation failure.
    pub fn load(host: Gemma4CpuWeights) -> Result<Self, Gemma4Error> {
        let executor = MetalBf16Executor::new().map_err(|error| metal_error(&error))?;
        let host = Arc::new(host);
        let config = host.config().clone();
        validate_inventory(&config, &host)?;
        let metal = Gemma4MetalWeights::load(&host, &executor)?;
        Ok(Self {
            host,
            metal,
            executor,
        })
    }

    #[must_use]
    pub fn config(&self) -> &Gemma4Config {
        self.host.config()
    }
}

pub struct Gemma4MetalDecoder {
    model: Arc<Gemma4MetalModel>,
    caches: Vec<Option<OwnerCache>>,
    position: usize,
}

impl Gemma4MetalDecoder {
    /// Upload the inventory and prepare decoder state.
    /// # Errors
    /// Rejects an inventory whose shapes disagree with its source configuration
    /// or a Metal preparation failure.
    pub fn load(host: Gemma4CpuWeights) -> Result<Self, Gemma4Error> {
        Self::shared(Arc::new(Gemma4MetalModel::load(host)?))
    }

    /// Prepare one sequence over an already-loaded model.
    /// # Errors
    /// Rejects a key/value owner that follows its consumer.
    pub fn shared(model: Arc<Gemma4MetalModel>) -> Result<Self, Gemma4Error> {
        let config = model.config();
        let mut caches = Vec::with_capacity(config.layers().len());
        for (layer, geometry) in config.layers().iter().enumerate() {
            if geometry.kv_owner > layer {
                return Err(Gemma4Error::new(
                    "Metal decoder",
                    "key/value owner must not follow its consumer",
                ));
            }
            caches.push((geometry.kv_owner == layer).then(|| OwnerCache {
                keys: Vec::new(),
                values: Vec::new(),
            }));
        }
        Ok(Self {
            model,
            caches,
            position: 0,
        })
    }

    #[must_use]
    pub const fn position(&self) -> usize {
        self.position
    }

    /// Clear continuation state; weights stay bound.
    pub fn reset(&mut self) {
        for cache in self.caches.iter_mut().flatten() {
            cache.keys.clear();
            cache.values.clear();
        }
        self.position = 0;
    }

    /// Process a prompt chunk and return post-soft-cap logits for its last token.
    /// # Errors
    /// Rejects empty/out-of-range or media tokens and a chunk past the context.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<u16>, Gemma4Error> {
        self.step(tokens)
    }

    /// Process one token and return its post-soft-cap logits.
    /// # Errors
    /// As [`Self::prefill`].
    pub fn decode(&mut self, token: u32) -> Result<Vec<u16>, Gemma4Error> {
        self.step(&[token])
    }

    fn step(&mut self, tokens: &[u32]) -> Result<Vec<u16>, Gemma4Error> {
        let sequence = tokens.len();
        let end = self
            .position
            .checked_add(sequence)
            .ok_or_else(|| Gemma4Error::new("Metal decoder", "position overflow"))?;
        let config = self.model.config();
        if sequence == 0
            || end > config.max_position_embeddings()
            || tokens
                .iter()
                .any(|&token| token as usize >= config.vocab_size())
        {
            return Err(Gemma4Error::new(
                "Metal decoder",
                "empty, overlong or out-of-range token chunk",
            ));
        }
        config.validate_text_tokens(tokens)?;
        let lengths: Vec<_> = self
            .caches
            .iter()
            .map(|cache| {
                cache
                    .as_ref()
                    .map(|cache| (cache.keys.len(), cache.values.len()))
            })
            .collect();
        let position = self.position;
        let result = run_step(
            &self.model.host,
            &self.model.metal,
            &MetalOps {
                executor: &self.model.executor,
            },
            &mut self.caches,
            position,
            tokens,
        );
        match result {
            Ok(logits) => {
                self.position = end;
                Ok(logits)
            }
            Err(error) => {
                for (cache, lengths) in self.caches.iter_mut().zip(lengths) {
                    if let (Some(cache), Some((keys, values))) = (cache, lengths) {
                        cache.keys.truncate(keys);
                        cache.values.truncate(values);
                    }
                }
                Err(error)
            }
        }
    }
}

fn run_step(
    host: &Gemma4CpuWeights,
    metal: &Gemma4MetalWeights,
    ops: &MetalOps<'_>,
    caches: &mut [Option<OwnerCache>],
    position: usize,
    tokens: &[u32],
) -> Result<Vec<u16>, Gemma4Error> {
    let config = host.config();
    let sequence = tokens.len();
    let hidden_size = config.hidden_size();
    let ple = config.ple_dim();
    let packed = config.packed_ple_dim();
    let epsilon = config.rms_norm_eps();
    let embedding = host
        .parameter(ParameterId::TokenEmbedding)
        .ok_or_else(|| Gemma4Error::new("Metal decoder", "missing token embedding"))?;
    let embed_scale = f32_value(bf16_f32(usize_to_f32(hidden_size).sqrt()));
    let mut hidden = Vec::with_capacity(sequence * hidden_size);
    for &token in tokens {
        let row = token_row(embedding, token as usize)?;
        hidden.extend(scale(row, embed_scale)?);
    }
    let ple_embedding = host
        .parameter(ParameterId::PleEmbedding)
        .ok_or_else(|| Gemma4Error::new("Metal decoder", "missing per-layer embedding"))?;
    let ple_scale = f32_value(bf16_f32(usize_to_f32(ple).sqrt()));
    let mut identity = Vec::with_capacity(sequence * packed);
    for &token in tokens {
        let row = token_row(ple_embedding, token as usize)?;
        identity.extend(scale(row, ple_scale)?);
    }
    let projection = ops.linear(
        &hidden,
        sequence,
        metal
            .parameter(ParameterId::PleProjection)
            .ok_or_else(|| Gemma4Error::new("Metal decoder", "missing PLE projection"))?,
    )?;
    let projection = scale(&projection, 1.0 / usize_to_f32(hidden_size).sqrt())?;
    let ple_norm = metal
        .parameter(ParameterId::PleNorm)
        .ok_or_else(|| Gemma4Error::new("Metal decoder", "missing PLE norm"))?;
    let mut per_layer = Vec::with_capacity(config.layers().len());
    for layer in 0..config.layers().len() {
        let mut slice = Vec::with_capacity(sequence * ple);
        let mut token_slice = Vec::with_capacity(sequence * ple);
        for token in 0..sequence {
            let base = token * packed + layer * ple;
            slice.extend_from_slice(&projection[base..base + ple]);
            token_slice.extend_from_slice(&identity[base..base + ple]);
        }
        let context = ops.rms(&slice, sequence, ple, epsilon, Some(ple_norm))?;
        per_layer.push(ple_combine(&context, &token_slice)?);
    }

    let positions: Vec<u32> = (position..position + sequence)
        .map(|value| {
            u32::try_from(value)
                .map_err(|_| Gemma4Error::new("Metal decoder", "position exceeds u32"))
        })
        .collect::<Result<_, _>>()?;
    for (layer, geometry) in config.layers().iter().enumerate() {
        hidden = layer_forward(
            config,
            metal,
            caches,
            ops,
            geometry,
            layer,
            position,
            sequence,
            epsilon,
            &positions,
            &hidden,
            &per_layer[layer],
        )?;
    }
    let last = &hidden[(sequence - 1) * hidden_size..];
    let normalized = ops.rms(
        last,
        1,
        hidden_size,
        epsilon,
        Some(
            metal
                .parameter(ParameterId::FinalNorm)
                .ok_or_else(|| Gemma4Error::new("Metal decoder", "missing final norm"))?,
        ),
    )?;
    let logits = ops.linear(&normalized, 1, embedding_buffer(metal)?)?;
    softcap(&logits, config.final_logit_softcapping())
}

fn embedding_buffer(metal: &Gemma4MetalWeights) -> Result<&MetalBf16Buffer, Gemma4Error> {
    metal
        .parameter(ParameterId::TokenEmbedding)
        .ok_or_else(|| Gemma4Error::new("Metal decoder", "missing token embedding"))
}

fn token_row(parameter: &crate::CpuParameter, token: usize) -> Result<&[u16], Gemma4Error> {
    let shape = parameter.shape();
    if shape.len() != 2 || token >= shape[0] {
        return Err(Gemma4Error::new(
            "Metal decoder",
            "token exceeds its embedding table",
        ));
    }
    Ok(&parameter.values()[token * shape[1]..(token + 1) * shape[1]])
}

#[expect(
    clippy::cast_precision_loss,
    reason = "model dimensions are small positive integers"
)]
fn usize_to_f32(value: usize) -> f32 {
    value as f32
}

fn bf16_f32(value: f32) -> u16 {
    crate::bf16(value)
}

fn f32_value(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

#[expect(
    clippy::too_many_arguments,
    reason = "one decoder layer's semantic inputs, kept explicit"
)]
fn layer_forward(
    config: &Gemma4Config,
    metal: &Gemma4MetalWeights,
    caches: &mut [Option<OwnerCache>],
    ops: &MetalOps<'_>,
    geometry: &crate::LayerGeometry,
    layer: usize,
    position: usize,
    sequence: usize,
    epsilon: f32,
    positions: &[u32],
    hidden: &[u16],
    per_layer: &[u16],
) -> Result<Vec<u16>, Gemma4Error> {
    let hidden = attention_forward(
        config, metal, caches, ops, geometry, layer, position, sequence, epsilon, positions, hidden,
    )?;
    feedforward_forward(metal, ops, layer, sequence, epsilon, &hidden, per_layer)
}

#[expect(
    clippy::too_many_arguments,
    reason = "one attention block's semantic inputs, kept explicit"
)]
fn attention_forward(
    config: &Gemma4Config,
    metal: &Gemma4MetalWeights,
    caches: &mut [Option<OwnerCache>],
    ops: &MetalOps<'_>,
    geometry: &crate::LayerGeometry,
    layer: usize,
    position: usize,
    sequence: usize,
    epsilon: f32,
    positions: &[u32],
    hidden: &[u16],
) -> Result<Vec<u16>, Gemma4Error> {
    let hidden_size = config.hidden_size();
    let heads = config.num_attention_heads();
    let kv_heads = config.num_key_value_heads();
    let width = geometry.head_dim;
    let normed = ops.rms(
        hidden,
        sequence,
        hidden_size,
        epsilon,
        Some(metal.layer(layer, LayerParameter::InputNorm)?),
    )?;
    let queries = ops.linear(
        &normed,
        sequence,
        metal.layer(layer, LayerParameter::Query)?,
    )?;
    let queries = ops.rms(
        &queries,
        sequence * heads,
        width,
        epsilon,
        Some(metal.layer(layer, LayerParameter::QueryNorm)?),
    )?;
    let (cosine, sine) = factors(config, layer, positions)?;
    let queries = ops.rotary(&queries, sequence, heads, width, &cosine, &sine)?;
    if geometry.kv_owner == layer {
        let keys = ops.linear(&normed, sequence, metal.layer(layer, LayerParameter::Key)?)?;
        let keys = ops.rms(
            &keys,
            sequence * kv_heads,
            width,
            epsilon,
            Some(metal.layer(layer, LayerParameter::KeyNorm)?),
        )?;
        let keys = ops.rotary(&keys, sequence, kv_heads, width, &cosine, &sine)?;
        let values = ops.linear(
            &normed,
            sequence,
            metal.layer(layer, LayerParameter::Value)?,
        )?;
        let values = ops.rms(&values, sequence * kv_heads, width, epsilon, None)?;
        let cache = caches[layer]
            .as_mut()
            .ok_or_else(|| Gemma4Error::new("Metal decoder", "owner cache missing"))?;
        cache.keys.extend_from_slice(&keys);
        cache.values.extend_from_slice(&values);
    }
    let cache = caches[geometry.kv_owner]
        .as_ref()
        .ok_or_else(|| Gemma4Error::new("Metal decoder", "owner cache missing"))?;
    let window = match geometry.kind {
        AttentionKind::Sliding => Some(config.sliding_window()),
        AttentionKind::Full => None,
    };
    let attended = ops.attention(
        &queries,
        &cache.keys,
        &cache.values,
        MetalAttentionConfig {
            heads,
            kv_heads,
            width,
            window,
        },
        position,
        0,
    )?;
    let projected = ops.linear(
        &attended,
        sequence,
        metal.layer(layer, LayerParameter::AttentionOutput)?,
    )?;
    let normed = ops.rms(
        &projected,
        sequence,
        hidden_size,
        epsilon,
        Some(metal.layer(layer, LayerParameter::PostAttentionNorm)?),
    )?;
    add(hidden, &normed)
}

fn feedforward_forward(
    metal: &Gemma4MetalWeights,
    ops: &MetalOps<'_>,
    layer: usize,
    sequence: usize,
    epsilon: f32,
    hidden: &[u16],
    per_layer: &[u16],
) -> Result<Vec<u16>, Gemma4Error> {
    let hidden_size = metal
        .layer(layer, LayerParameter::PreFeedforwardNorm)?
        .elements();
    let residual = hidden.to_vec();
    let normed = ops.rms(
        hidden,
        sequence,
        hidden_size,
        epsilon,
        Some(metal.layer(layer, LayerParameter::PreFeedforwardNorm)?),
    )?;
    let gate = ops.linear(&normed, sequence, metal.layer(layer, LayerParameter::Gate)?)?;
    let up = ops.linear(&normed, sequence, metal.layer(layer, LayerParameter::Up)?)?;
    let product = multiply(&gelu_tanh(&gate)?, &up)?;
    let down = ops.linear(
        &product,
        sequence,
        metal.layer(layer, LayerParameter::Down)?,
    )?;
    let normed = ops.rms(
        &down,
        sequence,
        hidden_size,
        epsilon,
        Some(metal.layer(layer, LayerParameter::PostFeedforwardNorm)?),
    )?;
    let hidden = add(&residual, &normed)?;
    let residual = hidden.clone();
    let gate = ops.linear(
        &hidden,
        sequence,
        metal.layer(layer, LayerParameter::PleGate)?,
    )?;
    let product = multiply(&gelu_tanh(&gate)?, per_layer)?;
    let projected = ops.linear(
        &product,
        sequence,
        metal.layer(layer, LayerParameter::PleProjection)?,
    )?;
    let normed = ops.rms(
        &projected,
        sequence,
        hidden_size,
        epsilon,
        Some(metal.layer(layer, LayerParameter::PostPleNorm)?),
    )?;
    let hidden = add(&residual, &normed)?;
    let scalar = metal.layer(layer, LayerParameter::LayerScalar)?.bits();
    scale(&hidden, f32_value(scalar[0]))
}

/// Token index of the largest soft-capped logit.
/// # Errors
/// Rejects an empty logit row.
pub fn metal_argmax(logits: &[u16]) -> Result<u32, Gemma4Error> {
    argmax(logits).ok_or_else(|| Gemma4Error::new("Metal decoder", "empty logits"))
}
