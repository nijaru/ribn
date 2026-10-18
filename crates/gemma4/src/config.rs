//! Model semantics from the modern Gemma 4 conditional-generation package config.

use crate::Gemma4Error;
use serde_json::Value;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionKind {
    Sliding,
    Full,
}

/// Split-half `RoPE`. Proportional `RoPE` pads the frequency vector with zeros;
/// it is not ordinary partial `RoPE` over a shorter contiguous head.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rope {
    Default { theta: f32 },
    Proportional { theta: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerGeometry {
    pub kind: AttentionKind,
    pub head_dim: usize,
    pub kv_owner: usize,
    /// Final decoder layer that reads this owner's current-step KV span.
    pub kv_last_consumer: usize,
    pub rope: Rope,
}

/// Validated dense causal text semantics. Fields are private so source selection
/// and continuation planning cannot observe geometry mutated after validation.
#[derive(Clone, Debug, PartialEq)]
pub struct Gemma4Config {
    pub(crate) vocab: usize,
    pub(crate) bos: u32,
    pub(crate) pad: u32,
    media: [u32; 3],
    pub(crate) hidden: usize,
    pub(crate) intermediate: usize,
    pub(crate) heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) ple: usize,
    pub(crate) ple_vocab: usize,
    pub(crate) packed_ple: usize,
    pub(crate) context: usize,
    pub(crate) window: usize,
    pub(crate) epsilon: f32,
    pub(crate) softcap: f32,
    pub(crate) layers: Vec<LayerGeometry>,
}

impl Gemma4Config {
    /// Interpret the supported dense/PLE/tied-BF16 subset. Unknown text semantic
    /// fields and unsupported overrides reject, rather than silently selecting a
    /// different architecture. Audio/vision config is not interpreted by this path.
    ///
    /// # Errors
    /// Rejects missing/invalid geometry, unsupported algorithms and overflow.
    pub fn from_json(root: &Value) -> Result<Self, Gemma4Error> {
        let text = text_config(root)?;
        let layers = layer_geometry(text)?;
        let ple = positive(text, "hidden_size_per_layer_input")?;
        let packed_ple = layers
            .len()
            .checked_mul(ple)
            .ok_or_else(|| Gemma4Error::new("PLE", "dimension overflow"))?;
        let config = Self {
            vocab: positive(text, "vocab_size")?,
            bos: token_id(text, "bos_token_id")?,
            pad: token_id(text, "pad_token_id")?,
            media: [
                token_id(root, "image_token_id")?,
                token_id(root, "audio_token_id")?,
                token_id(root, "video_token_id")?,
            ],
            hidden: positive(text, "hidden_size")?,
            intermediate: positive(text, "intermediate_size")?,
            heads: positive(text, "num_attention_heads")?,
            kv_heads: positive(text, "num_key_value_heads")?,
            ple,
            ple_vocab: positive(text, "vocab_size_per_layer_input")?,
            packed_ple,
            context: positive(text, "max_position_embeddings")?,
            window: positive(text, "sliding_window")?,
            epsilon: positive_float(text, "rms_norm_eps")?,
            softcap: positive_float(text, "final_logit_softcapping")?,
            layers,
        };
        if usize::try_from(config.bos).map_or(true, |n| n >= config.vocab)
            || usize::try_from(config.pad).map_or(true, |n| n >= config.vocab)
        {
            return Err(Gemma4Error::new(
                "special tokens",
                "BOS and PAD must be text vocabulary IDs",
            ));
        }
        if config.ple_vocab < config.vocab {
            return Err(Gemma4Error::new(
                "vocab_size_per_layer_input",
                "must cover every text token ID",
            ));
        }
        if !config.heads.is_multiple_of(config.kv_heads) {
            return Err(Gemma4Error::new(
                "num_key_value_heads",
                "must divide query head count",
            ));
        }
        for layer in &config.layers {
            config
                .heads
                .checked_mul(layer.head_dim)
                .and_then(|n| n.checked_mul(config.hidden))
                .and_then(|n| n.checked_mul(2))
                .ok_or_else(|| {
                    Gemma4Error::new("attention geometry", "projection byte size overflows")
                })?;
        }
        // Check every logical parameter product before callers open payloads.
        for spec in crate::parameters::specifications(&config) {
            spec.byte_len()?;
        }
        Ok(config)
    }

    /// Validate direct text-decoder input, including pre-tokenized callers.
    /// The conditional wrapper masks these three media placeholders to PAD before
    /// both main and per-layer embeddings; treating them as ordinary tokens is
    /// not equivalent. Media processing is outside this text-only implementation.
    /// # Errors
    /// Rejects out-of-vocabulary IDs and unsupported media placeholders.
    pub fn validate_text_tokens(&self, tokens: &[u32]) -> Result<(), Gemma4Error> {
        for &token in tokens {
            if self.media.contains(&token) {
                return Err(Gemma4Error::new(
                    "text input",
                    "media placeholders require the multimodal processor",
                ));
            }
            if usize::try_from(token).map_or(true, |n| n >= self.vocab) {
                return Err(Gemma4Error::new(
                    "text input",
                    "token ID outside model vocabulary",
                ));
            }
        }
        Ok(())
    }

    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocab
    }
    #[must_use]
    pub const fn hidden_size(&self) -> usize {
        self.hidden
    }
    #[must_use]
    pub const fn intermediate_size(&self) -> usize {
        self.intermediate
    }
    #[must_use]
    pub const fn num_attention_heads(&self) -> usize {
        self.heads
    }
    #[must_use]
    pub const fn num_key_value_heads(&self) -> usize {
        self.kv_heads
    }
    #[must_use]
    pub const fn ple_dim(&self) -> usize {
        self.ple
    }
    #[must_use]
    pub const fn packed_ple_dim(&self) -> usize {
        self.packed_ple
    }
    #[must_use]
    pub const fn max_position_embeddings(&self) -> usize {
        self.context
    }
    #[must_use]
    pub const fn sliding_window(&self) -> usize {
        self.window
    }
    #[must_use]
    pub const fn rms_norm_eps(&self) -> f32 {
        self.epsilon
    }
    #[must_use]
    pub const fn final_logit_softcapping(&self) -> f32 {
        self.softcap
    }
    #[must_use]
    pub fn layers(&self) -> &[LayerGeometry] {
        &self.layers
    }
}

fn text_config(root: &Value) -> Result<&Value, Gemma4Error> {
    equal(root, "model_type", &Value::from("gemma4"))?;
    equal(
        root,
        "architectures",
        &serde_json::json!(["Gemma4ForConditionalGeneration"]),
    )?;
    equal(root, "tie_word_embeddings", &Value::Bool(true))?;
    equal(root, "dtype", &Value::from("bfloat16"))?;
    let text = root
        .get("text_config")
        .ok_or_else(|| Gemma4Error::new("text_config", "missing"))?;
    let fields = text
        .as_object()
        .ok_or_else(|| Gemma4Error::new("text_config", "expected an object"))?;
    for key in fields.keys() {
        if !TEXT_FIELDS.contains(&key.as_str()) {
            return Err(Gemma4Error::new(
                key,
                "unsupported text configuration field",
            ));
        }
    }
    for (key, expected) in [
        ("model_type", Value::from("gemma4_text")),
        ("dtype", Value::from("bfloat16")),
        ("hidden_activation", Value::from("gelu_pytorch_tanh")),
        ("tie_word_embeddings", Value::Bool(true)),
        ("attention_bias", Value::Bool(false)),
        ("attention_k_eq_v", Value::Bool(false)),
        ("enable_moe_block", Value::Bool(false)),
        ("use_double_wide_mlp", Value::Bool(false)),
        ("use_cache", Value::Bool(true)),
        ("use_bidirectional_attention", Value::Null),
        ("num_global_key_value_heads", Value::Null),
    ] {
        equal(text, key, &expected)?;
    }
    if text.get("attention_dropout").and_then(Value::as_f64) != Some(0.0) {
        return Err(Gemma4Error::new(
            "attention_dropout",
            "only zero dropout is supported",
        ));
    }
    Ok(text)
}

fn layer_geometry(text: &Value) -> Result<Vec<LayerGeometry>, Gemma4Error> {
    let count = positive(text, "num_hidden_layers")?;
    let shared = unsigned(text, "num_kv_shared_layers")?;
    if shared >= count {
        return Err(Gemma4Error::new(
            "num_kv_shared_layers",
            "sharing needs a nonempty owner prefix",
        ));
    }
    let owners = count - shared;
    let local_dim = positive(text, "head_dim")?;
    let global_dim = positive(text, "global_head_dim")?;
    if !local_dim.is_multiple_of(2) || !global_dim.is_multiple_of(8) {
        return Err(Gemma4Error::new(
            "head dimensions",
            "local must be even; quarter-proportional global must be divisible by eight",
        ));
    }
    let kinds = text
        .get("layer_types")
        .and_then(Value::as_array)
        .filter(|a| a.len() == count)
        .ok_or_else(|| {
            Gemma4Error::new(
                "layer_types",
                "one explicit attention type per layer is required",
            )
        })?
        .iter()
        .map(|v| match v.as_str() {
            Some("sliding_attention") => Ok(AttentionKind::Sliding),
            Some("full_attention") => Ok(AttentionKind::Full),
            _ => Err(Gemma4Error::new(
                "layer_types",
                format!("unsupported attention: {v}"),
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if kinds.last() != Some(&AttentionKind::Full) {
        return Err(Gemma4Error::new(
            "layer_types",
            "last layer must explicitly be full attention (no silent rewrite)",
        ));
    }
    let rope = text
        .get("rope_parameters")
        .ok_or_else(|| Gemma4Error::new("rope_parameters", "missing"))?;
    let local_rope = parse_rope(rope, "sliding_attention", false)?;
    let global_rope = parse_rope(rope, "full_attention", true)?;
    if rope.as_object().is_none_or(|o| o.len() != 2) {
        return Err(Gemma4Error::new(
            "rope_parameters",
            "expected only sliding and full attention entries",
        ));
    }
    let mut layers = kinds
        .iter()
        .enumerate()
        .map(|(i, kind)| {
            let kv_owner = if i < owners {
                i
            } else {
                kinds[..owners]
                    .iter()
                    .rposition(|k| k == kind)
                    .ok_or_else(|| {
                        Gemma4Error::new(
                            "KV sharing",
                            "tail attention kind has no owner in the prefix",
                        )
                    })?
            };
            let (head_dim, rope) = match kind {
                AttentionKind::Sliding => (local_dim, local_rope),
                AttentionKind::Full => (global_dim, global_rope),
            };
            Ok(LayerGeometry {
                kind: *kind,
                head_dim,
                kv_owner,
                kv_last_consumer: 0,
                rope,
            })
        })
        .collect::<Result<Vec<_>, Gemma4Error>>()?;
    let mut last_consumers: Vec<_> = (0..layers.len()).collect();
    for (i, geometry) in layers.iter().enumerate() {
        last_consumers[geometry.kv_owner] = i;
    }
    for geometry in &mut layers {
        geometry.kv_last_consumer = last_consumers[geometry.kv_owner];
    }
    Ok(layers)
}

const TEXT_FIELDS: &[&str] = &[
    "model_type",
    "dtype",
    "attention_bias",
    "attention_dropout",
    "attention_k_eq_v",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "enable_moe_block",
    "expert_intermediate_size",
    "final_logit_softcapping",
    "global_head_dim",
    "head_dim",
    "hidden_activation",
    "hidden_size",
    "hidden_size_per_layer_input",
    "initializer_range",
    "intermediate_size",
    "layer_types",
    "max_position_embeddings",
    "num_attention_heads",
    "num_experts",
    "num_global_key_value_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "num_kv_shared_layers",
    "rms_norm_eps",
    "sliding_window",
    "rope_parameters",
    "tie_word_embeddings",
    "top_k_experts",
    "use_bidirectional_attention",
    "use_cache",
    "use_double_wide_mlp",
    "vocab_size",
    "vocab_size_per_layer_input",
];

fn parse_rope(root: &Value, kind: &str, proportional: bool) -> Result<Rope, Gemma4Error> {
    let value = root
        .get(kind)
        .ok_or_else(|| Gemma4Error::new(kind, "missing RoPE"))?;
    let expected_keys: &[&str] = if proportional {
        &["rope_type", "rope_theta", "partial_rotary_factor"]
    } else {
        &["rope_type", "rope_theta"]
    };
    if value.as_object().is_none_or(|o| {
        o.len() != expected_keys.len() || o.keys().any(|k| !expected_keys.contains(&k.as_str()))
    }) {
        return Err(Gemma4Error::new(kind, "unsupported RoPE fields"));
    }
    equal(
        value,
        "rope_type",
        &Value::from(if proportional {
            "proportional"
        } else {
            "default"
        }),
    )?;
    let theta = positive_float(value, "rope_theta")?;
    if proportional {
        if value.get("partial_rotary_factor").and_then(Value::as_f64) != Some(0.25) {
            return Err(Gemma4Error::new(
                kind,
                "only quarter-proportional RoPE is supported",
            ));
        }
        Ok(Rope::Proportional { theta })
    } else {
        Ok(Rope::Default { theta })
    }
}

fn equal(root: &Value, key: &str, expected: &Value) -> Result<(), Gemma4Error> {
    if root.get(key) == Some(expected) {
        Ok(())
    } else {
        Err(Gemma4Error::new(
            key,
            format!("expected {expected}, found {:?}", root.get(key)),
        ))
    }
}
fn unsigned(root: &Value, key: &str) -> Result<usize, Gemma4Error> {
    root.get(key)
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| Gemma4Error::new(key, "expected an unsigned platform-sized integer"))
}
pub(crate) fn token_id(root: &Value, key: &str) -> Result<u32, Gemma4Error> {
    root.get(key)
        .and_then(Value::as_u64)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| Gemma4Error::new(key, "expected a nonnegative u32 token ID"))
}

fn positive(root: &Value, key: &str) -> Result<usize, Gemma4Error> {
    let n = unsigned(root, key)?;
    if n == 0 {
        Err(Gemma4Error::new(key, "must be positive"))
    } else {
        Ok(n)
    }
}
fn positive_float(root: &Value, key: &str) -> Result<f32, Gemma4Error> {
    let n = root
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| Gemma4Error::new(key, "expected a number"))?;
    #[allow(
        clippy::cast_possible_truncation,
        reason = "model constants are represented in execution's FP32 scalar policy"
    )]
    let n = n as f32;
    if !n.is_finite() || n <= 0.0 {
        Err(Gemma4Error::new(key, "must be finite and positive in FP32"))
    } else {
        Ok(n)
    }
}
