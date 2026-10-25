//! Vision identities deliberately do not carry text PLE or KV-owner roles.
use super::Gemma4VisionConfig;
use crate::Gemma4Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum VisionLinear {
    Query,
    Key,
    Value,
    AttentionOutput,
    Gate,
    Up,
    Down,
}
impl VisionLinear {
    pub(super) const ALL: [Self; 7] = [
        Self::Query,
        Self::Key,
        Self::Value,
        Self::AttentionOutput,
        Self::Gate,
        Self::Up,
        Self::Down,
    ];
    const fn path(self) -> &'static str {
        match self {
            Self::Query => "self_attn.q_proj",
            Self::Key => "self_attn.k_proj",
            Self::Value => "self_attn.v_proj",
            Self::AttentionOutput => "self_attn.o_proj",
            Self::Gate => "mlp.gate_proj",
            Self::Up => "mlp.up_proj",
            Self::Down => "mlp.down_proj",
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum VisionLinearPart {
    Weight,
    InputMin,
    InputMax,
    OutputMin,
    OutputMax,
}
impl VisionLinearPart {
    const ALL: [Self; 5] = [
        Self::Weight,
        Self::InputMin,
        Self::InputMax,
        Self::OutputMin,
        Self::OutputMax,
    ];
    const fn suffix(self) -> &'static str {
        match self {
            Self::Weight => "linear.weight",
            Self::InputMin => "input_min",
            Self::InputMax => "input_max",
            Self::OutputMin => "output_min",
            Self::OutputMax => "output_max",
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum VisionNorm {
    Input,
    PostAttention,
    PreFeedforward,
    PostFeedforward,
    Query,
    Key,
}
impl VisionNorm {
    const ALL: [Self; 6] = [
        Self::Input,
        Self::PostAttention,
        Self::PreFeedforward,
        Self::PostFeedforward,
        Self::Query,
        Self::Key,
    ];
    const fn path(self) -> &'static str {
        match self {
            Self::Input => "input_layernorm",
            Self::PostAttention => "post_attention_layernorm",
            Self::PreFeedforward => "pre_feedforward_layernorm",
            Self::PostFeedforward => "post_feedforward_layernorm",
            Self::Query => "self_attn.q_norm",
            Self::Key => "self_attn.k_norm",
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum VisionParameterId {
    PatchProjection,
    PositionEmbedding,
    Projection,
    Norm {
        layer: usize,
        norm: VisionNorm,
    },
    Linear {
        layer: usize,
        linear: VisionLinear,
        part: VisionLinearPart,
    },
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisionParameterSpec {
    pub id: VisionParameterId,
    pub name: String,
    pub shape: Vec<usize>,
}
impl VisionParameterSpec {
    /// Encoded BF16 size, including a single element for a rank-zero clip bound.
    /// # Errors
    /// Rejects shape/byte overflow.
    pub fn byte_len(&self) -> Result<usize, Gemma4Error> {
        self.shape
            .iter()
            .try_fold(2_usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| Gemma4Error::new(&self.name, "encoded size overflow"))
    }
}
impl Gemma4VisionConfig {
    /// Complete logical inventory, including persistent activation-clipping buffers.
    pub fn specifications(&self) -> impl Iterator<Item = VisionParameterSpec> + '_ {
        base_specifications(self)
            .into_iter()
            .chain((0..self.layers).flat_map(|layer| layer_specifications(self, layer)))
    }
}
pub(super) fn base_specifications(c: &Gemma4VisionConfig) -> [VisionParameterSpec; 3] {
    [
        (
            VisionParameterId::PatchProjection,
            "vision_tower.patch_embedder.input_proj.weight",
            vec![c.hidden, 3 * c.patch * c.patch],
        ),
        (
            VisionParameterId::PositionEmbedding,
            "vision_tower.patch_embedder.position_embedding_table",
            vec![2, c.positions, c.hidden],
        ),
        (
            VisionParameterId::Projection,
            "embed_vision.embedding_projection.weight",
            vec![c.output, c.hidden],
        ),
    ]
    .map(|(id, name, shape)| VisionParameterSpec {
        id,
        name: format!("model.{name}"),
        shape,
    })
}
pub(super) fn layer_specifications(
    c: &Gemma4VisionConfig,
    layer: usize,
) -> impl Iterator<Item = VisionParameterSpec> + '_ {
    let prefix = format!("model.vision_tower.encoder.layers.{layer}");
    let norms = VisionNorm::ALL.into_iter().map({
        let prefix = prefix.clone();
        move |norm| VisionParameterSpec {
            id: VisionParameterId::Norm { layer, norm },
            name: format!("{prefix}.{}.weight", norm.path()),
            shape: vec![if matches!(norm, VisionNorm::Query | VisionNorm::Key) {
                c.head_dim
            } else {
                c.hidden
            }],
        }
    });
    let linears = VisionLinear::ALL.into_iter().flat_map(move |linear| {
        let prefix = prefix.clone();
        VisionLinearPart::ALL
            .into_iter()
            .filter(move |part| c.clipped || *part == VisionLinearPart::Weight)
            .map(move |part| {
                let shape = if part == VisionLinearPart::Weight {
                    match linear {
                        VisionLinear::Query => vec![c.heads * c.head_dim, c.hidden],
                        VisionLinear::Key | VisionLinear::Value => {
                            vec![c.kv_heads * c.head_dim, c.hidden]
                        }
                        VisionLinear::AttentionOutput => vec![c.hidden, c.heads * c.head_dim],
                        VisionLinear::Gate | VisionLinear::Up => vec![c.intermediate, c.hidden],
                        VisionLinear::Down => vec![c.hidden, c.intermediate],
                    }
                } else {
                    vec![]
                };
                VisionParameterSpec {
                    id: VisionParameterId::Linear {
                        layer,
                        linear,
                        part,
                    },
                    name: format!("{prefix}.{}.{}", linear.path(), part.suffix()),
                    shape,
                }
            })
    });
    norms.chain(linears)
}
