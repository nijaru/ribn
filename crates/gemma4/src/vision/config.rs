//! Explicit semantics for the clipped, axial-RoPE Gemma 4 vision encoder.

use crate::config::{equal, positive, positive_float};
use crate::{Gemma4Config, Gemma4Error};
use serde_json::Value;

/// Validated BF16 vision geometry. Independent of text continuation/KV roles.
#[derive(Clone, Debug, PartialEq)]
pub struct Gemma4VisionConfig {
    pub(super) hidden: usize,
    pub(super) intermediate: usize,
    pub(super) layers: usize,
    pub(super) heads: usize,
    pub(super) kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) patch: usize,
    pub(super) pool: usize,
    pub(super) positions: usize,
    pub(super) output: usize,
    pub(super) epsilon: f32,
    pub(super) theta: f32,
    pub(super) clipped: bool,
}

impl Gemma4VisionConfig {
    /// Interpret vision semantics and the validated text projection width.
    /// # Errors
    /// Rejects unsupported algorithms, unknown fields, invalid geometry or overflow.
    pub fn from_json(root: &Value) -> Result<Self, Gemma4Error> {
        let text = Gemma4Config::from_json(root)?;
        Self::parse(
            root.get("vision_config").unwrap_or(&Value::Null),
            text.hidden_size(),
        )
    }

    pub(crate) fn parse(raw: &Value, output: usize) -> Result<Self, Gemma4Error> {
        let object = raw
            .as_object()
            .ok_or_else(|| Gemma4Error::new("vision_config", "expected object"))?;
        for key in object.keys() {
            if !FIELDS.contains(&key.as_str()) {
                return Err(Gemma4Error::new(
                    key,
                    "unsupported vision configuration field",
                ));
            }
        }
        for (key, value) in [
            ("model_type", Value::from("gemma4_vision")),
            ("dtype", Value::from("bfloat16")),
            ("hidden_activation", Value::from("gelu_pytorch_tanh")),
            ("attention_bias", Value::from(false)),
            ("standardize", Value::from(false)),
        ] {
            equal(raw, key, &value)?;
        }
        if raw["attention_dropout"].as_f64() != Some(0.0) {
            return Err(Gemma4Error::new(
                "attention_dropout",
                "expected numeric zero",
            ));
        }
        let rope = &raw["rope_parameters"];
        if rope.as_object().is_none_or(|o| {
            o.len() != 2 || !o.contains_key("rope_type") || !o.contains_key("rope_theta")
        }) || !matches!(rope["rope_type"].as_str(), Some("default" | "axial"))
        {
            return Err(Gemma4Error::new(
                "vision RoPE",
                "expected flat axial/default parameters",
            ));
        }
        // HF Gemma4VisionConfig.default_rope_type is axial: the checkpoint's
        // literal "default" is normalized to axial, never text's 1-D rotation.
        let config = Self {
            hidden: positive(raw, "hidden_size")?,
            intermediate: positive(raw, "intermediate_size")?,
            layers: positive(raw, "num_hidden_layers")?,
            heads: positive(raw, "num_attention_heads")?,
            kv_heads: positive(raw, "num_key_value_heads")?,
            head_dim: positive(raw, "head_dim")?,
            patch: positive(raw, "patch_size")?,
            pool: positive(raw, "pooling_kernel_size")?,
            positions: positive(raw, "position_embedding_size")?,
            output,
            epsilon: positive_float(raw, "rms_norm_eps")?,
            theta: positive_float(rope, "rope_theta")?,
            clipped: raw["use_clipped_linears"]
                .as_bool()
                .ok_or_else(|| Gemma4Error::new("use_clipped_linears", "expected boolean"))?,
        };
        if config.pool < 2 {
            return Err(Gemma4Error::new(
                "pooling_kernel_size",
                "only spatial average pooling with kernel greater than one is supported",
            ));
        }
        if !config.head_dim.is_multiple_of(4) || !config.heads.is_multiple_of(config.kv_heads) {
            return Err(Gemma4Error::new(
                "vision heads",
                "head dimension must be divisible by four and KV heads must divide query heads",
            ));
        }
        config
            .heads
            .checked_mul(config.head_dim)
            .and_then(|_| config.kv_heads.checked_mul(config.head_dim))
            .ok_or_else(|| Gemma4Error::new("vision heads", "head stride overflows"))?;
        config
            .patch
            .checked_mul(config.patch)
            .and_then(|n| n.checked_mul(3))
            .and_then(|_| config.pool.checked_mul(config.pool))
            .ok_or_else(|| {
                Gemma4Error::new("vision geometry", "patch/pooling dimensions overflow")
            })?;
        // Check fixed matrices and one layer before producing any potentially
        // large layer inventory. The iterator itself does not allocate per layer.
        let mut base = 0_usize;
        let mut layer = 0_usize;
        for spec in super::parameters::base_specifications(&config) {
            base = base.checked_add(spec.byte_len()?).ok_or_else(overflow)?;
        }
        for spec in super::parameters::layer_specifications(&config, 0) {
            layer = layer.checked_add(spec.byte_len()?).ok_or_else(overflow)?;
        }
        layer
            .checked_mul(config.layers)
            .and_then(|n| n.checked_add(base))
            .filter(|&n| isize::try_from(n).is_ok())
            .ok_or_else(overflow)?;
        Ok(config)
    }
    #[must_use]
    pub const fn hidden_size(&self) -> usize {
        self.hidden
    }
    #[must_use]
    pub const fn output_width(&self) -> usize {
        self.output
    }
    #[must_use]
    pub const fn layer_count(&self) -> usize {
        self.layers
    }
    #[must_use]
    pub const fn patch_size(&self) -> usize {
        self.patch
    }
    #[must_use]
    pub const fn pooling_kernel_size(&self) -> usize {
        self.pool
    }
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }
    #[must_use]
    pub const fn heads(&self) -> usize {
        self.heads
    }
    #[must_use]
    pub const fn kv_heads(&self) -> usize {
        self.kv_heads
    }
    #[must_use]
    pub const fn epsilon(&self) -> f32 {
        self.epsilon
    }
    #[must_use]
    pub const fn theta(&self) -> f32 {
        self.theta
    }
}
fn overflow() -> Gemma4Error {
    Gemma4Error::new("vision weights", "encoded geometry overflows")
}
const FIELDS: &[&str] = &[
    "model_type",
    "dtype",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "hidden_activation",
    "rms_norm_eps",
    "rope_parameters",
    "patch_size",
    "pooling_kernel_size",
    "position_embedding_size",
    "use_clipped_linears",
    "standardize",
    "attention_bias",
    "attention_dropout",
    // Inert HF serialization/training/diagnostic metadata; none chooses execution.
    "_name_or_path",
    "architectures",
    "chunk_size_feed_forward",
    "default_output_length",
    "global_head_dim",
    "id2label",
    "label2id",
    "initializer_range",
    "is_encoder_decoder",
    "max_position_embeddings",
    "output_attentions",
    "output_hidden_states",
    "problem_type",
    "return_dict",
];
