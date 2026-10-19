//! One host CPU Gemma 4 forward step: cache, layers, PLE and output head.

use crate::cpu::{
    AttentionConfig, CpuParameter, Gemma4CpuWeights, add, apply_rotary, attention, bf16, factors,
    gelu_tanh, multiply, ple_combine, project, rms_norm, rms_norm_scale_free, scale, softcap,
};
use crate::{AttentionKind, Gemma4Config, Gemma4Error, LayerGeometry, LayerParameter, ParameterId};

#[derive(Clone, Copy)]
struct Scales {
    embed: f32,
    ple_embed: f32,
    ple_projection: f32,
}

struct OwnerCache {
    keys: Vec<u16>,
    values: Vec<u16>,
}

/// Host Gemma 4 decoder state for one sequence: BF16 weights, hybrid KV and position.
///
/// This is a correctness-first CPU execution path. It keeps complete per-owner
/// history and applies sliding-window masks instead of evicting local spans, so the
/// arithmetic matches the model while cache eviction/donor trimming remain unproven.
pub struct Gemma4CpuDecoder {
    config: Gemma4Config,
    weights: Gemma4CpuWeights,
    caches: Vec<Option<OwnerCache>>,
    position: usize,
    scales: Scales,
}

impl Gemma4CpuDecoder {
    /// Prepare decoder state, binding it to the weights' validated source config.
    /// # Errors
    /// Rejects inconsistent owners, missing active K/V parameters or an inventory
    /// whose shapes disagree with its source configuration.
    pub fn new(weights: Gemma4CpuWeights) -> Result<Self, Gemma4Error> {
        let config = weights.config().clone();
        validate_inventory(&config, &weights)?;
        let mut caches = Vec::with_capacity(config.layers().len());
        for (layer, geometry) in config.layers().iter().enumerate() {
            if geometry.kv_owner > layer {
                return Err(Gemma4Error::new(
                    "CPU decoder",
                    "key/value owner must not follow its consumer",
                ));
            }
            let owner = geometry.kv_owner == layer;
            if owner {
                for parameter in [
                    LayerParameter::Key,
                    LayerParameter::Value,
                    LayerParameter::KeyNorm,
                ] {
                    weights.layer(layer, parameter)?;
                }
            }
            caches.push(owner.then(|| OwnerCache {
                keys: Vec::new(),
                values: Vec::new(),
            }));
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "model dimensions are small positive integers"
        )]
        let hidden = config.hidden_size() as f32;
        #[expect(
            clippy::cast_precision_loss,
            reason = "per-layer widths are small positive integers"
        )]
        let ple = config.ple_dim() as f32;
        Ok(Self {
            scales: Scales {
                embed: bf16_f32(hidden.sqrt()),
                ple_embed: bf16_f32(ple.sqrt()),
                ple_projection: 1.0 / hidden.sqrt(),
            },
            config,
            weights,
            caches,
            position: 0,
        })
    }

    /// Clear all continuation state; weights and configuration stay bound.
    pub fn reset(&mut self) {
        for cache in self.caches.iter_mut().flatten() {
            cache.keys.clear();
            cache.values.clear();
        }
        self.position = 0;
    }

    #[must_use]
    pub const fn position(&self) -> usize {
        self.position
    }

    /// Process a prompt chunk and return post-soft-cap logits for its last token.
    /// # Errors
    /// Rejects empty/out-of-range tokens or a chunk beyond the context bound.
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
            .ok_or_else(|| Gemma4Error::new("CPU decoder", "position overflow"))?;
        if sequence == 0
            || end > self.config.max_position_embeddings()
            || tokens
                .iter()
                .any(|&token| token as usize >= self.config.vocab_size())
        {
            return Err(Gemma4Error::new(
                "CPU decoder",
                "empty, overlong or out-of-range token chunk",
            ));
        }
        self.config.validate_text_tokens(tokens)?;
        let lengths: Vec<_> = self
            .caches
            .iter()
            .map(|cache| {
                cache
                    .as_ref()
                    .map(|cache| (cache.keys.len(), cache.values.len()))
            })
            .collect();
        let logits = run_step(
            &self.config,
            &self.weights,
            &mut self.caches,
            self.position,
            self.scales,
            tokens,
        );
        match logits {
            Ok(logits) => {
                self.position = end;
                Ok(logits)
            }
            Err(error) => {
                // A failed step must not leave partly advanced continuation state.
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

fn expect_shape(actual: &[usize], expected: &[usize], name: &str) -> Result<(), Gemma4Error> {
    if actual != expected {
        return Err(Gemma4Error::new(
            "CPU decoder",
            format!("{name} shape {actual:?} disagrees with the source config {expected:?}"),
        ));
    }
    Ok(())
}

/// Bind the inventory to the configuration recorded with its validated source.
fn validate_inventory(
    config: &Gemma4Config,
    weights: &Gemma4CpuWeights,
) -> Result<(), Gemma4Error> {
    let hidden = config.hidden_size();
    let ple = config.ple_dim();
    let packed = config.packed_ple_dim();
    let heads = config.num_attention_heads();
    let kv_heads = config.num_key_value_heads();
    let intermediate = config.intermediate_size();
    let parameter = |id: ParameterId| {
        weights
            .parameter(id)
            .ok_or_else(|| Gemma4Error::new("CPU decoder", format!("missing {id:?} parameter")))
    };
    expect_shape(
        parameter(ParameterId::TokenEmbedding)?.shape(),
        &[config.vocab_size(), hidden],
        "token embedding",
    )?;
    let embedding = parameter(ParameterId::PleEmbedding)?.shape();
    if embedding.len() != 2 || embedding[0] < config.vocab_size() || embedding[1] != packed {
        return Err(Gemma4Error::new(
            "CPU decoder",
            "per-layer embedding shape disagrees with the source config",
        ));
    }
    expect_shape(
        parameter(ParameterId::PleProjection)?.shape(),
        &[packed, hidden],
        "PLE projection",
    )?;
    expect_shape(parameter(ParameterId::PleNorm)?.shape(), &[ple], "PLE norm")?;
    expect_shape(
        parameter(ParameterId::FinalNorm)?.shape(),
        &[hidden],
        "final norm",
    )?;
    for (layer, geometry) in config.layers().iter().enumerate() {
        let width = geometry.head_dim;
        let shape = |role: LayerParameter| weights.layer(layer, role).map(CpuParameter::shape);
        expect_shape(
            shape(LayerParameter::Query)?,
            &[heads * width, hidden],
            "query",
        )?;
        expect_shape(
            shape(LayerParameter::AttentionOutput)?,
            &[hidden, heads * width],
            "attention output",
        )?;
        expect_shape(shape(LayerParameter::QueryNorm)?, &[width], "query norm")?;
        for role in [
            LayerParameter::InputNorm,
            LayerParameter::PostAttentionNorm,
            LayerParameter::PreFeedforwardNorm,
            LayerParameter::PostFeedforwardNorm,
        ] {
            expect_shape(shape(role)?, &[hidden], "hidden norm")?;
        }
        expect_shape(
            shape(LayerParameter::Gate)?,
            &[intermediate, hidden],
            "gate",
        )?;
        expect_shape(shape(LayerParameter::Up)?, &[intermediate, hidden], "up")?;
        expect_shape(
            shape(LayerParameter::Down)?,
            &[hidden, intermediate],
            "down",
        )?;
        expect_shape(shape(LayerParameter::PleGate)?, &[ple, hidden], "PLE gate")?;
        expect_shape(
            shape(LayerParameter::PleProjection)?,
            &[hidden, ple],
            "PLE layer projection",
        )?;
        expect_shape(
            shape(LayerParameter::PostPleNorm)?,
            &[hidden],
            "post-PLE norm",
        )?;
        expect_shape(shape(LayerParameter::LayerScalar)?, &[1], "layer scalar")?;
        if geometry.kv_owner == layer {
            expect_shape(
                shape(LayerParameter::Key)?,
                &[kv_heads * width, hidden],
                "key",
            )?;
            expect_shape(
                shape(LayerParameter::Value)?,
                &[kv_heads * width, hidden],
                "value",
            )?;
            expect_shape(shape(LayerParameter::KeyNorm)?, &[width], "key norm")?;
        }
    }
    Ok(())
}

fn bf16_f32(value: f32) -> f32 {
    f32::from_bits(u32::from(bf16(value)) << 16)
}

fn load(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Index of the largest logit; ties keep the first, as the pinned argmax does.
#[must_use]
pub fn argmax(logits: &[u16]) -> Option<u32> {
    let mut best = 0_usize;
    for (index, &value) in logits.iter().enumerate().skip(1) {
        if load(value) > load(logits[best]) {
            best = index;
        }
    }
    if logits.is_empty() {
        return None;
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the vocabulary fits in u32"
    )]
    let token = best as u32;
    Some(token)
}

fn run_step(
    config: &Gemma4Config,
    weights: &Gemma4CpuWeights,
    caches: &mut [Option<OwnerCache>],
    position: usize,
    scales: Scales,
    tokens: &[u32],
) -> Result<Vec<u16>, Gemma4Error> {
    let sequence = tokens.len();
    let hidden_size = config.hidden_size();
    let ple = config.ple_dim();
    let packed = config.packed_ple_dim();
    let epsilon = config.rms_norm_eps();
    let embedding = weights
        .parameter(ParameterId::TokenEmbedding)
        .ok_or_else(|| Gemma4Error::new("CPU decoder", "missing token embedding"))?;
    let mut hidden = Vec::with_capacity(sequence * hidden_size);
    for &token in tokens {
        let row = token_row(embedding, token as usize)?;
        hidden.extend(scale(row, scales.embed)?);
    }
    let ple_embedding = weights
        .parameter(ParameterId::PleEmbedding)
        .ok_or_else(|| Gemma4Error::new("CPU decoder", "missing per-layer embedding"))?;
    let mut token_identity = Vec::with_capacity(sequence * packed);
    for &token in tokens {
        let row = token_row(ple_embedding, token as usize)?;
        token_identity.extend(scale(row, scales.ple_embed)?);
    }
    let projection = project(
        &hidden,
        sequence,
        weights
            .parameter(ParameterId::PleProjection)
            .ok_or_else(|| Gemma4Error::new("CPU decoder", "missing PLE projection"))?,
    )?;
    let projection = scale(&projection, scales.ple_projection)?;
    let ple_norm = weights
        .parameter(ParameterId::PleNorm)
        .ok_or_else(|| Gemma4Error::new("CPU decoder", "missing PLE norm"))?;
    let mut per_layer = Vec::with_capacity(config.layers().len());
    for layer in 0..config.layers().len() {
        let mut slice = Vec::with_capacity(sequence * ple);
        let mut identity = Vec::with_capacity(sequence * ple);
        for token in 0..sequence {
            let base = token * packed + layer * ple;
            slice.extend_from_slice(&projection[base..base + ple]);
            identity.extend_from_slice(&token_identity[base..base + ple]);
        }
        let context = rms_norm(&slice, sequence, ple, epsilon, ple_norm)?;
        per_layer.push(ple_combine(&context, &identity)?);
    }

    let positions: Vec<u32> = (position..position + sequence)
        .map(|value| {
            u32::try_from(value)
                .map_err(|_| Gemma4Error::new("CPU decoder", "position exceeds u32"))
        })
        .collect::<Result<_, _>>()?;
    for (layer, geometry) in config.layers().iter().enumerate() {
        hidden = layer_forward(
            config,
            weights,
            caches,
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
    let normalized = rms_norm(
        last,
        1,
        hidden_size,
        epsilon,
        weights
            .parameter(ParameterId::FinalNorm)
            .ok_or_else(|| Gemma4Error::new("CPU decoder", "missing final norm"))?,
    )?;
    let logits = project(&normalized, 1, embedding)?;
    softcap(&logits, config.final_logit_softcapping())
}

fn token_row(parameter: &crate::CpuParameter, token: usize) -> Result<&[u16], Gemma4Error> {
    let shape = parameter.shape();
    if shape.len() != 2 || token >= shape[0] {
        return Err(Gemma4Error::new(
            "CPU decoder",
            "token exceeds its embedding table",
        ));
    }
    Ok(&parameter.values()[token * shape[1]..(token + 1) * shape[1]])
}

#[expect(
    clippy::too_many_arguments,
    reason = "one decoder layer's semantic inputs, kept explicit"
)]
fn layer_forward(
    config: &Gemma4Config,
    weights: &Gemma4CpuWeights,
    caches: &mut [Option<OwnerCache>],
    geometry: &LayerGeometry,
    layer: usize,
    position: usize,
    sequence: usize,
    epsilon: f32,
    positions: &[u32],
    hidden: &[u16],
    per_layer: &[u16],
) -> Result<Vec<u16>, Gemma4Error> {
    let hidden = attention_forward(
        config, weights, caches, geometry, layer, position, sequence, epsilon, positions, hidden,
    )?;
    feedforward_forward(weights, layer, sequence, epsilon, &hidden, per_layer)
}

#[expect(
    clippy::too_many_arguments,
    reason = "one attention block's semantic inputs, kept explicit"
)]
fn attention_forward(
    config: &Gemma4Config,
    weights: &Gemma4CpuWeights,
    caches: &mut [Option<OwnerCache>],
    geometry: &LayerGeometry,
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
    let normed = rms_norm(
        hidden,
        sequence,
        hidden_size,
        epsilon,
        weights.layer(layer, LayerParameter::InputNorm)?,
    )?;
    let queries = project(
        &normed,
        sequence,
        weights.layer(layer, LayerParameter::Query)?,
    )?;
    let queries = rms_norm(
        &queries,
        sequence * heads,
        width,
        epsilon,
        weights.layer(layer, LayerParameter::QueryNorm)?,
    )?;
    let (cosine, sine) = factors(config, layer, positions)?;
    let queries = apply_rotary(&queries, sequence, heads, width, &cosine, &sine)?;
    if geometry.kv_owner == layer {
        let keys = project(
            &normed,
            sequence,
            weights.layer(layer, LayerParameter::Key)?,
        )?;
        let keys = rms_norm(
            &keys,
            sequence * kv_heads,
            width,
            epsilon,
            weights.layer(layer, LayerParameter::KeyNorm)?,
        )?;
        let keys = apply_rotary(&keys, sequence, kv_heads, width, &cosine, &sine)?;
        let values = project(
            &normed,
            sequence,
            weights.layer(layer, LayerParameter::Value)?,
        )?;
        let values = rms_norm_scale_free(&values, sequence * kv_heads, width, epsilon)?;
        let cache = caches[layer]
            .as_mut()
            .ok_or_else(|| Gemma4Error::new("CPU decoder", "owner cache missing"))?;
        cache.keys.extend_from_slice(&keys);
        cache.values.extend_from_slice(&values);
    }
    let cache = caches[geometry.kv_owner]
        .as_ref()
        .ok_or_else(|| Gemma4Error::new("CPU decoder", "key/value owner cache missing"))?;
    let window = match geometry.kind {
        AttentionKind::Sliding => Some(config.sliding_window()),
        AttentionKind::Full => None,
    };
    let config = AttentionConfig {
        heads,
        kv_heads,
        width,
        window,
    };
    let attended = attention(&queries, &cache.keys, &cache.values, config, position, 0)?;
    let projected = project(
        &attended,
        sequence,
        weights.layer(layer, LayerParameter::AttentionOutput)?,
    )?;
    let normed = rms_norm(
        &projected,
        sequence,
        hidden_size,
        epsilon,
        weights.layer(layer, LayerParameter::PostAttentionNorm)?,
    )?;
    add(hidden, &normed)
}

fn feedforward_forward(
    weights: &Gemma4CpuWeights,
    layer: usize,
    sequence: usize,
    epsilon: f32,
    hidden: &[u16],
    per_layer: &[u16],
) -> Result<Vec<u16>, Gemma4Error> {
    let hidden_size = weights
        .layer(layer, LayerParameter::PreFeedforwardNorm)?
        .shape()[0];
    let residual = hidden.to_vec();
    let normed = rms_norm(
        hidden,
        sequence,
        hidden_size,
        epsilon,
        weights.layer(layer, LayerParameter::PreFeedforwardNorm)?,
    )?;
    let gate = project(
        &normed,
        sequence,
        weights.layer(layer, LayerParameter::Gate)?,
    )?;
    let up = project(&normed, sequence, weights.layer(layer, LayerParameter::Up)?)?;
    let activated = gelu_tanh(&gate)?;
    let product = multiply(&activated, &up)?;
    let down = project(
        &product,
        sequence,
        weights.layer(layer, LayerParameter::Down)?,
    )?;
    let normed = rms_norm(
        &down,
        sequence,
        hidden_size,
        epsilon,
        weights.layer(layer, LayerParameter::PostFeedforwardNorm)?,
    )?;
    let hidden = add(&residual, &normed)?;

    let residual = hidden.clone();
    let gate = project(
        &hidden,
        sequence,
        weights.layer(layer, LayerParameter::PleGate)?,
    )?;
    let activated = gelu_tanh(&gate)?;
    let product = multiply(&activated, per_layer)?;
    let projected = project(
        &product,
        sequence,
        weights.layer(layer, LayerParameter::PleProjection)?,
    )?;
    let normed = rms_norm(
        &projected,
        sequence,
        hidden_size,
        epsilon,
        weights.layer(layer, LayerParameter::PostPleNorm)?,
    )?;
    let hidden = add(&residual, &normed)?;
    let scalar = weights.layer(layer, LayerParameter::LayerScalar)?;
    scale(&hidden, load(scalar.values()[0]))
}
