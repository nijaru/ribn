//! Complete logical parameter inventory for Inkling 2 and Inkling 2 Loop GGUF artifacts.

use crate::config::{AttentionKind, Inkling2Config};

/// Non-layer global parameters.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum GlobalParameter {
    TokenEmbedding,
    TokenEmbeddingNorm,
    Output,
    OutputNorm,
    OutputResQuery,
}

/// Physical layer parameters (`blk.{layer}.*`).
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum LayerParameter {
    AttnQ,
    AttnK,
    AttnV,
    AttnOutput,
    AttnGate,
    AttnQClip,
    AttnSinkProj,
    AttnRel,
    AttnRelBias,
    AttnRelClip,
    AttnKConv,
    AttnVConv,
    AttnPostConv,
    FfnPostConv,
    AttnNorm,
    AttnQInpNorm,
    AttnKInpNorm,
    AttnVInpNorm,
    AttnKConvProj,
    AttnVConvProj,
    AttnResQuery,
    FfnResQuery,
    FfnNorm,
    FfnGateInp,
    ExpProbsBias,
    FfnGateUpExps,
    FfnDownExps,
}

/// Cross-pass loop controller target, matching `LOOP_CTRL_COUNT` order `[Q, K, V, FFN]`.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum LoopController {
    Query = 0,
    Key = 1,
    Value = 2,
    Ffn = 3,
}

impl LoopController {
    pub const ALL: [Self; 4] = [Self::Query, Self::Key, Self::Value, Self::Ffn];

    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Query => "q",
            Self::Key => "k",
            Self::Value => "v",
            Self::Ffn => "ffn",
        }
    }
}

/// Loop core layer parameters (`blk.{layer}.loop_*`, present only for `core_start..core_end`).
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum LoopParameter {
    MessageQuery,
    ControlDown(LoopController),
    ControlGain(LoopController),
    ControlForget(LoopController),
}

/// Logical parameter identifier.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ParameterId {
    Global(GlobalParameter),
    Layer {
        layer: usize,
        parameter: LayerParameter,
    },
    Loop {
        layer: usize,
        parameter: LoopParameter,
    },
}

/// Expected GGUF tensor specification for one logical parameter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParameterSpec {
    pub id: ParameterId,
    pub name: String,
    /// Canonical GGUF dimensions `[ne0, ne1, ...]`.
    pub dims: Vec<u64>,
    /// Optional equivalent GGUF shape (e.g. `[n_embd, 1]` vs `[n_embd]` for 1-query `attn_res_query`).
    pub alt_dims: Option<Vec<u64>>,
    pub required: bool,
}

impl ParameterSpec {
    #[must_use]
    pub fn matches_dimensions(&self, actual: &[u64]) -> bool {
        actual == self.dims.as_slice() || self.alt_dims.as_deref().is_some_and(|alt| actual == alt)
    }
}

/// Build the complete expected GGUF parameter inventory for `config`.
///
/// For Inkling 2 Mini (24 physical layers, core `[6, 18)`), this yields all 704
/// physical tensors; logical loop passes share the physical core tensors.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn specifications(config: &Inkling2Config) -> Vec<ParameterSpec> {
    let n_embd = config.hidden_size() as u64;
    let n_vocab = config.vocab_padded() as u64;
    let n_rel = config.relative_dim() as u64;
    let n_conv = config.conv_kernel() as u64;
    let n_exp = config.expert_count() as u64;

    let mut specs = vec![
        req(
            ParameterId::Global(GlobalParameter::TokenEmbedding),
            "token_embd.weight".into(),
            vec![n_embd, n_vocab],
        ),
        req(
            ParameterId::Global(GlobalParameter::TokenEmbeddingNorm),
            "token_embd_norm.weight".into(),
            vec![n_embd],
        ),
        req(
            ParameterId::Global(GlobalParameter::Output),
            "output.weight".into(),
            vec![n_embd, n_vocab],
        ),
        req(
            ParameterId::Global(GlobalParameter::OutputNorm),
            "output_norm.weight".into(),
            vec![n_embd],
        ),
        req(
            ParameterId::Global(GlobalParameter::OutputResQuery),
            "output_res_query.weight".into(),
            vec![n_embd],
        ),
    ];

    for layer in config.physical_layers() {
        let i = layer.index;
        let q_dim = layer.q_dim() as u64;
        let kv_dim = layer.kv_dim() as u64;
        let heads = layer.heads as u64;
        let kv_heads = layer.kv_heads as u64;
        let head_dim = layer.head_dim as u64;
        let ff_exp = layer.ff_expert as u64;
        let extent = layer.rel_extent as u64;

        // Attention projections and clipping/sinks/relative bias
        specs.push(layer_req(
            i,
            LayerParameter::AttnQ,
            "attn_q.weight",
            vec![n_embd, q_dim],
        ));
        specs.push(layer_req(
            i,
            LayerParameter::AttnK,
            "attn_k.weight",
            vec![n_embd, kv_dim],
        ));
        specs.push(layer_req(
            i,
            LayerParameter::AttnV,
            "attn_v.weight",
            vec![n_embd, kv_dim],
        ));
        specs.push(layer_req(
            i,
            LayerParameter::AttnOutput,
            "attn_output.weight",
            vec![q_dim, n_embd],
        ));
        specs.push(layer_req(
            i,
            LayerParameter::AttnGate,
            "attn_gate.weight",
            vec![n_embd, q_dim],
        ));
        specs.push(layer_req(
            i,
            LayerParameter::AttnQClip,
            "attn_q_clip.weight",
            vec![head_dim, heads],
        ));
        specs.push(layer_req(
            i,
            LayerParameter::AttnSinkProj,
            "attn_sink_proj.weight",
            vec![n_embd, heads],
        ));
        specs.push(layer_req(
            i,
            LayerParameter::AttnRel,
            "attn_rel.weight",
            vec![n_embd, n_rel * heads],
        ));
        specs.push(layer_req(
            i,
            LayerParameter::AttnRelBias,
            "attn_rel_bias.weight",
            vec![n_rel, extent],
        ));
        specs.push(layer_req(
            i,
            LayerParameter::AttnRelClip,
            "attn_rel_clip.weight",
            vec![n_rel, heads],
        ));

        // Short causal convolutions
        specs.push(layer_req(
            i,
            LayerParameter::AttnKConv,
            "attn_k_conv.weight",
            vec![n_conv, kv_dim],
        ));
        specs.push(layer_req(
            i,
            LayerParameter::AttnVConv,
            "attn_v_conv.weight",
            vec![n_conv, kv_dim],
        ));
        specs.push(layer_req(
            i,
            LayerParameter::AttnPostConv,
            "attn_post_conv.weight",
            vec![n_conv, n_embd],
        ));
        specs.push(layer_req(
            i,
            LayerParameter::FfnPostConv,
            "ffn_post_conv.weight",
            vec![n_conv, n_embd],
        ));

        // Attention input norms and dynamic conv projections
        match layer.kind {
            AttentionKind::Sliding => {
                specs.push(layer_req(
                    i,
                    LayerParameter::AttnNorm,
                    "attn_norm.weight",
                    vec![n_embd],
                ));
            }
            AttentionKind::Full => {
                specs.push(layer_req(
                    i,
                    LayerParameter::AttnQInpNorm,
                    "attn_q_inp_norm.weight",
                    vec![n_embd],
                ));
                specs.push(layer_req(
                    i,
                    LayerParameter::AttnKInpNorm,
                    "attn_k_inp_norm.weight",
                    vec![n_embd],
                ));
                specs.push(layer_req(
                    i,
                    LayerParameter::AttnVInpNorm,
                    "attn_v_inp_norm.weight",
                    vec![n_embd],
                ));
                specs.push(layer_req(
                    i,
                    LayerParameter::AttnKConvProj,
                    "attn_k_conv_proj.weight",
                    vec![n_embd, n_conv * kv_heads],
                ));
                specs.push(layer_req(
                    i,
                    LayerParameter::AttnVConvProj,
                    "attn_v_conv_proj.weight",
                    vec![n_embd, n_conv * kv_heads],
                ));
            }
        }

        // Depth residual queries
        if layer.has_attn_res_query {
            match layer.kind {
                AttentionKind::Sliding => specs.push(ParameterSpec {
                    id: ParameterId::Layer {
                        layer: i,
                        parameter: LayerParameter::AttnResQuery,
                    },
                    name: format!("blk.{i}.attn_res_query.weight"),
                    dims: vec![n_embd],
                    alt_dims: Some(vec![n_embd, 1]),
                    required: true,
                }),
                AttentionKind::Full => specs.push(layer_req(
                    i,
                    LayerParameter::AttnResQuery,
                    "attn_res_query.weight",
                    vec![n_embd, 3],
                )),
            }
        }
        specs.push(layer_req(
            i,
            LayerParameter::FfnResQuery,
            "ffn_res_query.weight",
            vec![n_embd],
        ));

        // MoE FFN
        specs.push(layer_req(
            i,
            LayerParameter::FfnNorm,
            "ffn_norm.weight",
            vec![n_embd],
        ));
        specs.push(layer_req(
            i,
            LayerParameter::FfnGateInp,
            "ffn_gate_inp.weight",
            vec![n_embd, n_exp],
        ));
        specs.push(ParameterSpec {
            id: ParameterId::Layer {
                layer: i,
                parameter: LayerParameter::ExpProbsBias,
            },
            name: format!("blk.{i}.exp_probs_b.bias"),
            dims: vec![n_exp],
            alt_dims: None,
            required: false,
        });
        specs.push(layer_req(
            i,
            LayerParameter::FfnGateUpExps,
            "ffn_gate_up_exps.weight",
            vec![n_embd, 2 * ff_exp, n_exp],
        ));
        specs.push(layer_req(
            i,
            LayerParameter::FfnDownExps,
            "ffn_down_exps.weight",
            vec![ff_exp, n_embd, n_exp],
        ));

        // Cross-pass loop controllers for core layers
        if let Some(lc) = config.loop_config()
            && i >= lc.core_start
            && i < lc.core_end
        {
            let rank = lc.gate_rank as u64;
            specs.push(req(
                ParameterId::Loop {
                    layer: i,
                    parameter: LoopParameter::MessageQuery,
                },
                format!("blk.{i}.loop_msg_query.weight"),
                vec![n_embd, 4],
            ));
            for ctrl in LoopController::ALL {
                let tag = ctrl.tag();
                specs.push(req(
                    ParameterId::Loop {
                        layer: i,
                        parameter: LoopParameter::ControlDown(ctrl),
                    },
                    format!("blk.{i}.loop_{tag}_ctrl_down.weight"),
                    vec![2 * n_embd, rank],
                ));
                specs.push(req(
                    ParameterId::Loop {
                        layer: i,
                        parameter: LoopParameter::ControlGain(ctrl),
                    },
                    format!("blk.{i}.loop_{tag}_ctrl_gain.weight"),
                    vec![rank, n_embd],
                ));
                specs.push(req(
                    ParameterId::Loop {
                        layer: i,
                        parameter: LoopParameter::ControlForget(ctrl),
                    },
                    format!("blk.{i}.loop_{tag}_ctrl_forget.weight"),
                    vec![rank, n_embd],
                ));
            }
        }
    }

    specs
}

fn layer_req(
    layer: usize,
    parameter: LayerParameter,
    suffix: &str,
    dims: Vec<u64>,
) -> ParameterSpec {
    req(
        ParameterId::Layer { layer, parameter },
        format!("blk.{layer}.{suffix}"),
        dims,
    )
}

fn req(id: ParameterId, name: String, dims: Vec<u64>) -> ParameterSpec {
    ParameterSpec {
        id,
        name,
        dims,
        alt_dims: None,
        required: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::tests::mini_loop_metadata;

    #[test]
    fn inkling2_mini_inventory_matches_exact_704_tensors() {
        let cfg = Inkling2Config::from_metadata(&mini_loop_metadata()).expect("mini config");
        let specs = specifications(&cfg);
        assert_eq!(specs.len(), 704);
    }
}
