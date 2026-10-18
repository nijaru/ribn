//! Logical parameters and checkpoint names, independent of backend materialization.

use crate::{Gemma4Config, Gemma4Error};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum LayerParameter {
    Query,
    Key,
    Value,
    AttentionOutput,
    QueryNorm,
    KeyNorm,
    InputNorm,
    PostAttentionNorm,
    PreFeedforwardNorm,
    PostFeedforwardNorm,
    Gate,
    Up,
    Down,
    PleGate,
    PleProjection,
    PostPleNorm,
    LayerScalar,
}

impl LayerParameter {
    pub(crate) const ALL: [Self; 17] = [
        Self::Query,
        Self::Key,
        Self::Value,
        Self::AttentionOutput,
        Self::QueryNorm,
        Self::KeyNorm,
        Self::InputNorm,
        Self::PostAttentionNorm,
        Self::PreFeedforwardNorm,
        Self::PostFeedforwardNorm,
        Self::Gate,
        Self::Up,
        Self::Down,
        Self::PleGate,
        Self::PleProjection,
        Self::PostPleNorm,
        Self::LayerScalar,
    ];

    pub(crate) const fn needs_owner(self) -> bool {
        matches!(self, Self::Key | Self::Value | Self::KeyNorm)
    }
    const fn suffix(self) -> &'static str {
        match self {
            Self::Query => "self_attn.q_proj.weight",
            Self::Key => "self_attn.k_proj.weight",
            Self::Value => "self_attn.v_proj.weight",
            Self::AttentionOutput => "self_attn.o_proj.weight",
            Self::QueryNorm => "self_attn.q_norm.weight",
            Self::KeyNorm => "self_attn.k_norm.weight",
            Self::InputNorm => "input_layernorm.weight",
            Self::PostAttentionNorm => "post_attention_layernorm.weight",
            Self::PreFeedforwardNorm => "pre_feedforward_layernorm.weight",
            Self::PostFeedforwardNorm => "post_feedforward_layernorm.weight",
            Self::Gate => "mlp.gate_proj.weight",
            Self::Up => "mlp.up_proj.weight",
            Self::Down => "mlp.down_proj.weight",
            Self::PleGate => "per_layer_input_gate.weight",
            Self::PleProjection => "per_layer_projection.weight",
            Self::PostPleNorm => "post_per_layer_input_norm.weight",
            Self::LayerScalar => "layer_scalar",
        }
    }
}

/// The output projection aliases `TokenEmbedding`; it is not a second parameter.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ParameterId {
    TokenEmbedding,
    PleEmbedding,
    PleProjection,
    PleNorm,
    FinalNorm,
    Layer {
        layer: usize,
        parameter: LayerParameter,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParameterSpec {
    pub id: ParameterId,
    pub name: String,
    pub shape: Vec<usize>,
}

impl ParameterSpec {
    /// Encoded BF16 source bytes, not physical device allocation or workspace.
    /// # Errors
    /// Rejects products that cannot be represented on this host.
    pub fn byte_len(&self) -> Result<usize, Gemma4Error> {
        self.shape
            .iter()
            .try_fold(2_usize, |n, d| n.checked_mul(*d))
            .ok_or_else(|| Gemma4Error::new(&self.name, "BF16 parameter byte size overflows"))
    }
}

pub(crate) fn specifications(c: &Gemma4Config) -> Vec<ParameterSpec> {
    let mut specs = [
        (
            ParameterId::TokenEmbedding,
            "embed_tokens.weight",
            vec![c.vocab, c.hidden],
        ),
        (
            ParameterId::PleEmbedding,
            "embed_tokens_per_layer.weight",
            vec![c.ple_vocab, c.packed_ple],
        ),
        (
            ParameterId::PleProjection,
            "per_layer_model_projection.weight",
            vec![c.packed_ple, c.hidden],
        ),
        (
            ParameterId::PleNorm,
            "per_layer_projection_norm.weight",
            vec![c.ple],
        ),
        (ParameterId::FinalNorm, "norm.weight", vec![c.hidden]),
    ]
    .into_iter()
    .map(|(id, name, shape)| ParameterSpec {
        id,
        name: format!("model.language_model.{name}"),
        shape,
    })
    .collect::<Vec<_>>();
    for (layer, geometry) in c.layers.iter().enumerate() {
        for parameter in LayerParameter::ALL {
            let head = geometry.head_dim;
            let shape = match parameter {
                LayerParameter::Query => vec![c.heads * head, c.hidden],
                LayerParameter::Key | LayerParameter::Value => vec![c.kv_heads * head, c.hidden],
                LayerParameter::AttentionOutput => vec![c.hidden, c.heads * head],
                LayerParameter::QueryNorm | LayerParameter::KeyNorm => vec![head],
                LayerParameter::Gate | LayerParameter::Up => vec![c.intermediate, c.hidden],
                LayerParameter::Down => vec![c.hidden, c.intermediate],
                LayerParameter::PleGate => vec![c.ple, c.hidden],
                LayerParameter::PleProjection => vec![c.hidden, c.ple],
                LayerParameter::LayerScalar => vec![1],
                _ => vec![c.hidden],
            };
            specs.push(ParameterSpec {
                id: ParameterId::Layer { layer, parameter },
                name: format!("model.language_model.layers.{layer}.{}", parameter.suffix()),
                shape,
            });
        }
    }
    specs
}
