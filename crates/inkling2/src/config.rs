//! Validated Inkling 2 / Inkling 2 Loop GGUF metadata and logical-slot mapping.

use std::collections::BTreeMap;

use engine_gguf::{GgufFile, MetadataValue};

use crate::Inkling2Error;

/// Supported Inkling 2 text GGUF architecture variants.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArchitectureKind {
    /// Non-looped `inkling2` (1-to-1 physical layers and state slots).
    Plain,
    /// Looped `inkling2_loop` (shared physical core weights, independent per-pass state slots).
    Looped,
}

impl ArchitectureKind {
    #[must_use]
    pub const fn gguf_prefix(self) -> &'static str {
        match self {
            Self::Plain => "inkling2",
            Self::Looped => "inkling2_loop",
        }
    }
}

/// Per-layer attention span kind.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionKind {
    Sliding,
    Full,
}

/// Loop topology for `inkling2_loop` checkpoints (fixed-loop or variable-loop).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoopConfig {
    pub core_start: usize,
    pub core_end: usize,
    pub passes: usize,
    pub gate_rank: usize,
    pub varying: bool,
}

impl LoopConfig {
    #[must_use]
    pub const fn core_layer_count(self) -> usize {
        self.core_end - self.core_start
    }
}

/// Validated geometry for one physical transformer layer `blk.{index}`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhysicalLayerGeometry {
    pub index: usize,
    pub kind: AttentionKind,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub ff_expert: usize,
    pub experts_used: usize,
    /// Maximum backward distance covered by `attn_rel_bias` (`sliding_window` for SWA, `relative_extent` for Full).
    pub rel_extent: usize,
    pub is_block_boundary: bool,
    pub has_attn_res_query: bool,
    pub loop_core_index: Option<usize>,
}

impl PhysicalLayerGeometry {
    #[must_use]
    pub const fn q_dim(self) -> usize {
        self.heads * self.head_dim
    }

    #[must_use]
    pub const fn kv_dim(self) -> usize {
        self.kv_heads * self.head_dim
    }

    #[must_use]
    pub const fn gqa_group_size(self) -> usize {
        self.heads / self.kv_heads
    }
}

/// One logical execution step in forward order.
///
/// Physical weights are indexed by `physical_layer`, whereas persistent KV and
/// short-convolution history are indexed independently by `state_slot`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutionStep {
    pub step_index: usize,
    pub state_slot: usize,
    pub physical_layer: usize,
    pub loop_pass: Option<usize>,
    pub uses_cross_pass_control: bool,
    pub starts_loop_pass: bool,
}

/// Validated Inkling 2 / Inkling 2 Loop text configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct Inkling2Config {
    architecture: ArchitectureKind,
    context_length: usize,
    hidden_size: usize,
    vocab_padded: usize,
    vocab_unpadded: usize,
    rms_norm_eps: f32,
    qk_clip: f32,
    relative_dim: usize,
    relative_extent: usize,
    relative_clip: f32,
    log_scale_floor: usize,
    log_scale_alpha: f32,
    sliding_window: usize,
    attn_res_block_size: usize,
    conv_kernel: usize,
    expert_count: usize,
    expert_weights_scale: f32,
    situ_beta: f32,
    situ_linear_beta: f32,
    final_logit_softcapping: f32,
    loop_config: Option<LoopConfig>,
    physical_layers: Vec<PhysicalLayerGeometry>,
    schedule: Vec<ExecutionStep>,
    slot_to_physical: Vec<usize>,
    max_conv_channels: usize,
}

impl Inkling2Config {
    /// Parse and validate Inkling 2 configuration from an opened GGUF file.
    ///
    /// # Errors
    /// Returns [`Inkling2Error`] for unsupported architectures, missing keys,
    /// mismatched metadata types, invalid dimensions, or arithmetic overflow.
    pub fn from_gguf(file: &GgufFile) -> Result<Self, Inkling2Error> {
        Self::from_metadata(file.metadata_map())
    }

    /// Parse and validate Inkling 2 configuration from a GGUF metadata map.
    ///
    /// # Errors
    /// Returns [`Inkling2Error`] for unsupported architectures, missing keys,
    /// mismatched metadata types, invalid dimensions, or arithmetic overflow.
    #[allow(clippy::too_many_lines)]
    pub fn from_metadata(
        metadata: &BTreeMap<String, MetadataValue>,
    ) -> Result<Self, Inkling2Error> {
        let arch_str = required_str(metadata, "general.architecture")?;
        let architecture = match arch_str {
            "inkling2" => ArchitectureKind::Plain,
            "inkling2_loop" => ArchitectureKind::Looped,
            other => {
                return Err(Inkling2Error::new(
                    "general.architecture",
                    format!(
                        "unsupported architecture {other:?}; expected \"inkling2\" or \"inkling2_loop\""
                    ),
                ));
            }
        };
        let prefix = architecture.gguf_prefix();
        let key = |suffix: &str| format!("{prefix}.{suffix}");

        if let Some(rope_dim) = optional_usize(metadata, &key("rope.dimension_count"))?
            && rope_dim != 0
        {
            return Err(Inkling2Error::new(
                key("rope.dimension_count"),
                format!("Inkling 2 uses relative position bias, not RoPE (got {rope_dim})"),
            ));
        }

        let n_phys = required_positive_usize(metadata, &key("block_count"))?;
        let context_length = required_positive_usize(metadata, &key("context_length"))?;
        let hidden_size = required_positive_usize(metadata, &key("embedding_length"))?;
        let vocab_padded = required_positive_usize(metadata, &key("vocab_size"))?;
        let vocab_unpadded = required_positive_usize(metadata, &key("vocab_size_unpadded"))?;
        if vocab_unpadded > vocab_padded {
            return Err(Inkling2Error::new(
                key("vocab_size_unpadded"),
                format!("{vocab_unpadded} exceeds padded vocab_size {vocab_padded}"),
            ));
        }

        let rms_norm_eps =
            required_positive_f32(metadata, &key("attention.layer_norm_rms_epsilon"))?;
        let qk_clip = required_positive_f32(metadata, &key("attention.qk_clip"))?;
        let relative_dim = required_positive_usize(metadata, &key("attention.relative_length"))?;
        let relative_extent = required_positive_usize(metadata, &key("attention.relative_extent"))?;
        let relative_clip = required_positive_f32(metadata, &key("attention.relative_clip"))?;
        let log_scale_floor = required_positive_usize(metadata, &key("attention.log_scale_floor"))?;
        let log_scale_alpha =
            required_nonnegative_f32(metadata, &key("attention.log_scale_alpha"))?;
        let sliding_window = required_positive_usize(metadata, &key("attention.sliding_window"))?;
        let attn_res_block_size = required_positive_usize(metadata, &key("attn_res.block_size"))?;
        let conv_kernel = required_positive_usize(metadata, &key("shortconv.l_cache"))?;
        if !(2..=64).contains(&conv_kernel) {
            return Err(Inkling2Error::new(
                key("shortconv.l_cache"),
                format!("must be within 2..=64, got {conv_kernel}"),
            ));
        }
        if i32::try_from(relative_extent).is_err() || i32::try_from(sliding_window).is_err() {
            return Err(Inkling2Error::new(
                "attention extents",
                "relative_extent and sliding_window must fit in i32",
            ));
        }

        let expert_count = required_positive_usize(metadata, &key("expert_count"))?;
        let expert_weights_scale =
            optional_positive_f32(metadata, &key("expert_weights_scale"))?.unwrap_or(1.0);
        let situ_beta = required_positive_f32(metadata, &key("activation.situ_beta"))?;
        let situ_linear_beta =
            required_positive_f32(metadata, &key("activation.situ_linear_beta"))?;
        let final_logit_softcapping =
            optional_nonnegative_f32(metadata, &key("final_logit_softcapping"))?.unwrap_or(0.0);

        let key_length = required_positive_usize(metadata, &key("attention.key_length"))?;
        let value_length = required_positive_usize(metadata, &key("attention.value_length"))?;
        if key_length != value_length {
            return Err(Inkling2Error::new(
                "attention head_dim",
                format!("key_length ({key_length}) must equal value_length ({value_length})"),
            ));
        }
        let key_length_swa = optional_positive_usize(metadata, &key("attention.key_length_swa"))?
            .unwrap_or(key_length);
        let value_length_swa =
            optional_positive_usize(metadata, &key("attention.value_length_swa"))?
                .unwrap_or(value_length);
        if key_length_swa != value_length_swa {
            return Err(Inkling2Error::new(
                "SWA attention head_dim",
                format!(
                    "key_length_swa ({key_length_swa}) must equal value_length_swa ({value_length_swa})"
                ),
            ));
        }

        let swa_pattern =
            required_bool_array(metadata, &key("attention.sliding_window_pattern"), n_phys)?;
        let head_counts =
            required_positive_usize_or_array(metadata, &key("attention.head_count"), n_phys)?;
        let kv_head_counts =
            required_positive_usize_or_array(metadata, &key("attention.head_count_kv"), n_phys)?;
        let ff_experts =
            required_positive_usize_or_array(metadata, &key("expert_feed_forward_length"), n_phys)?;
        let experts_used =
            required_positive_usize_or_array(metadata, &key("expert_used_count"), n_phys)?;

        let loop_config = match architecture {
            ArchitectureKind::Plain => None,
            ArchitectureKind::Looped => {
                let core_start = required_positive_usize(metadata, &key("loop.start"))?;
                let core_end = required_positive_usize(metadata, &key("loop.end"))?;
                let passes = required_positive_usize(metadata, &key("loop.count"))?;
                let gate_rank = required_positive_usize(metadata, &key("loop.gate_rank"))?;
                let varying = optional_bool(metadata, &key("loop.varying"))?.unwrap_or(false);
                let min_passes = if varying { 1 } else { 2 };
                if passes < min_passes
                    || core_start >= core_end
                    || core_end > n_phys
                    || !core_start.is_multiple_of(attn_res_block_size)
                    || !core_end.is_multiple_of(attn_res_block_size)
                {
                    return Err(Inkling2Error::new(
                        "loop configuration",
                        format!(
                            "invalid loop core [{core_start}, {core_end}) with block_size={attn_res_block_size}, passes={passes}, varying={varying}, layers={n_phys}"
                        ),
                    ));
                }
                Some(LoopConfig {
                    core_start,
                    core_end,
                    passes,
                    gate_rank,
                    varying,
                })
            }
        };

        let mut physical_layers = Vec::with_capacity(n_phys);
        let mut max_conv_channels: usize = 0;
        for i in 0..n_phys {
            let is_swa = swa_pattern[i];
            let heads = head_counts[i];
            let kv_heads = kv_head_counts[i];
            let head_dim = if is_swa { key_length_swa } else { key_length };
            let ff_expert = ff_experts[i];
            let used = experts_used[i];
            if !heads.is_multiple_of(kv_heads) {
                return Err(Inkling2Error::new(
                    format!("layer {i} attention"),
                    format!("head_count ({heads}) is not divisible by head_count_kv ({kv_heads})"),
                ));
            }
            if used > expert_count {
                return Err(Inkling2Error::new(
                    format!("layer {i} experts"),
                    format!("expert_used_count ({used}) exceeds expert_count ({expert_count})"),
                ));
            }
            let kv_dim = kv_heads
                .checked_mul(head_dim)
                .ok_or_else(|| Inkling2Error::new("geometry", "kv_dim overflow"))?;
            let _ = heads
                .checked_mul(head_dim)
                .ok_or_else(|| Inkling2Error::new("geometry", "q_dim overflow"))?;
            let conv_channels = kv_dim
                .checked_mul(2)
                .and_then(|v| v.checked_add(hidden_size.checked_mul(2)?))
                .ok_or_else(|| Inkling2Error::new("geometry", "conv_channels overflow"))?;
            max_conv_channels = max_conv_channels.max(conv_channels);
            let loop_core_index = loop_config
                .and_then(|lc| (i >= lc.core_start && i < lc.core_end).then(|| i - lc.core_start));
            physical_layers.push(PhysicalLayerGeometry {
                index: i,
                kind: if is_swa {
                    AttentionKind::Sliding
                } else {
                    AttentionKind::Full
                },
                heads,
                kv_heads,
                head_dim,
                ff_expert,
                experts_used: used,
                rel_extent: if is_swa {
                    sliding_window
                } else {
                    relative_extent
                },
                is_block_boundary: i.is_multiple_of(attn_res_block_size),
                has_attn_res_query: i > 0,
                loop_core_index,
            });
        }

        let conv_state_elems = max_conv_channels
            .checked_mul(conv_kernel - 1)
            .ok_or_else(|| Inkling2Error::new("shortconv", "convolution state size overflow"))?;
        if u32::try_from(conv_state_elems).is_err() {
            return Err(Inkling2Error::new(
                "shortconv",
                "convolution state exceeds u32 element count",
            ));
        }

        let (schedule, slot_to_physical) = build_schedule(n_phys, loop_config)?;

        Ok(Self {
            architecture,
            context_length,
            hidden_size,
            vocab_padded,
            vocab_unpadded,
            rms_norm_eps,
            qk_clip,
            relative_dim,
            relative_extent,
            relative_clip,
            log_scale_floor,
            log_scale_alpha,
            sliding_window,
            attn_res_block_size,
            conv_kernel,
            expert_count,
            expert_weights_scale,
            situ_beta,
            situ_linear_beta,
            final_logit_softcapping,
            loop_config,
            physical_layers,
            schedule,
            slot_to_physical,
            max_conv_channels,
        })
    }

    #[must_use]
    pub const fn architecture(&self) -> ArchitectureKind {
        self.architecture
    }

    #[must_use]
    pub const fn context_length(&self) -> usize {
        self.context_length
    }

    #[must_use]
    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    #[must_use]
    pub const fn vocab_padded(&self) -> usize {
        self.vocab_padded
    }

    #[must_use]
    pub const fn vocab_unpadded(&self) -> usize {
        self.vocab_unpadded
    }

    #[must_use]
    pub const fn rms_norm_eps(&self) -> f32 {
        self.rms_norm_eps
    }

    #[must_use]
    pub const fn qk_clip(&self) -> f32 {
        self.qk_clip
    }

    #[must_use]
    pub const fn relative_dim(&self) -> usize {
        self.relative_dim
    }

    #[must_use]
    pub const fn relative_extent(&self) -> usize {
        self.relative_extent
    }

    #[must_use]
    pub const fn relative_clip(&self) -> f32 {
        self.relative_clip
    }

    #[must_use]
    pub const fn log_scale_floor(&self) -> usize {
        self.log_scale_floor
    }

    #[must_use]
    pub const fn log_scale_alpha(&self) -> f32 {
        self.log_scale_alpha
    }

    #[must_use]
    pub const fn sliding_window(&self) -> usize {
        self.sliding_window
    }

    #[must_use]
    pub const fn attn_res_block_size(&self) -> usize {
        self.attn_res_block_size
    }

    #[must_use]
    pub const fn conv_kernel(&self) -> usize {
        self.conv_kernel
    }

    #[must_use]
    pub const fn expert_count(&self) -> usize {
        self.expert_count
    }

    #[must_use]
    pub const fn expert_weights_scale(&self) -> f32 {
        self.expert_weights_scale
    }

    #[must_use]
    pub const fn situ_beta(&self) -> f32 {
        self.situ_beta
    }

    #[must_use]
    pub const fn situ_linear_beta(&self) -> f32 {
        self.situ_linear_beta
    }

    #[must_use]
    pub const fn final_logit_softcapping(&self) -> f32 {
        self.final_logit_softcapping
    }

    #[must_use]
    pub const fn loop_config(&self) -> Option<LoopConfig> {
        self.loop_config
    }

    #[must_use]
    pub fn physical_layers(&self) -> &[PhysicalLayerGeometry] {
        &self.physical_layers
    }

    #[must_use]
    pub fn physical_layer_count(&self) -> usize {
        self.physical_layers.len()
    }

    /// Logical execution schedule in forward order.
    #[must_use]
    pub fn execution_schedule(&self) -> &[ExecutionStep] {
        &self.schedule
    }

    /// Number of independent persistent KV + short-convolution state slots.
    #[must_use]
    pub fn state_slot_count(&self) -> usize {
        self.slot_to_physical.len()
    }

    /// Physical layer index providing shared weights for logical `state_slot`.
    #[must_use]
    pub fn slot_physical_layer(&self, state_slot: usize) -> Option<usize> {
        self.slot_to_physical.get(state_slot).copied()
    }

    /// Physical geometry for logical `state_slot`.
    #[must_use]
    pub fn slot_geometry(&self, state_slot: usize) -> Option<&PhysicalLayerGeometry> {
        self.slot_physical_layer(state_slot)
            .and_then(|ip| self.physical_layers.get(ip))
    }

    /// Total convolution channels stored for physical layer `ip` (`K + V + attn_post + ffn_post`).
    #[must_use]
    pub fn conv_channels_for_layer(&self, ip: usize) -> Option<usize> {
        self.physical_layers
            .get(ip)
            .map(|layer| 2 * layer.kv_dim() + 2 * self.hidden_size)
    }

    #[must_use]
    pub const fn max_conv_channels(&self) -> usize {
        self.max_conv_channels
    }
}

fn build_schedule(
    n_phys: usize,
    loop_config: Option<LoopConfig>,
) -> Result<(Vec<ExecutionStep>, Vec<usize>), Inkling2Error> {
    let Some(lc) = loop_config else {
        let schedule = (0..n_phys)
            .map(|ip| ExecutionStep {
                step_index: ip,
                state_slot: ip,
                physical_layer: ip,
                loop_pass: None,
                uses_cross_pass_control: false,
                starts_loop_pass: false,
            })
            .collect();
        let slot_to_physical = (0..n_phys).collect();
        return Ok((schedule, slot_to_physical));
    };

    let core_len = lc.core_layer_count();
    let extra_slots = (lc.passes - 1)
        .checked_mul(core_len)
        .ok_or_else(|| Inkling2Error::new("loop schedule", "state slot count overflow"))?;
    let n_slots = n_phys
        .checked_add(extra_slots)
        .ok_or_else(|| Inkling2Error::new("loop schedule", "state slot count overflow"))?;

    let mut slot_to_physical = Vec::with_capacity(n_slots);
    for il in 0..n_slots {
        let ip = if il < n_phys {
            il
        } else {
            lc.core_start + (il - n_phys) % core_len
        };
        slot_to_physical.push(ip);
    }

    let mut schedule = Vec::with_capacity(n_slots);
    let mut step_index = 0;

    // Prelude: physical layers [0, core_start)
    for ip in 0..lc.core_start {
        schedule.push(ExecutionStep {
            step_index,
            state_slot: ip,
            physical_layer: ip,
            loop_pass: None,
            uses_cross_pass_control: false,
            starts_loop_pass: false,
        });
        step_index += 1;
    }

    // Core loop passes: [core_start, core_end) repeated `passes` times with distinct state slots
    for pass in 0..lc.passes {
        for ip in lc.core_start..lc.core_end {
            let state_slot = if pass == 0 {
                ip
            } else {
                n_phys + (pass - 1) * core_len + (ip - lc.core_start)
            };
            debug_assert_eq!(slot_to_physical[state_slot], ip);
            schedule.push(ExecutionStep {
                step_index,
                state_slot,
                physical_layer: ip,
                loop_pass: Some(pass),
                uses_cross_pass_control: pass > 0,
                starts_loop_pass: ip == lc.core_start,
            });
            step_index += 1;
        }
    }

    // Epilogue: physical layers [core_end, n_phys)
    for ip in lc.core_end..n_phys {
        schedule.push(ExecutionStep {
            step_index,
            state_slot: ip,
            physical_layer: ip,
            loop_pass: None,
            uses_cross_pass_control: false,
            starts_loop_pass: false,
        });
        step_index += 1;
    }

    Ok((schedule, slot_to_physical))
}

fn required_str<'a>(
    metadata: &'a BTreeMap<String, MetadataValue>,
    key: &str,
) -> Result<&'a str, Inkling2Error> {
    metadata
        .get(key)
        .ok_or_else(|| Inkling2Error::new(key, "missing required GGUF metadata key"))?
        .as_str()
        .ok_or_else(|| Inkling2Error::new(key, "expected string metadata value"))
}

fn value_as_usize(value: &MetadataValue, key: &str) -> Result<usize, Inkling2Error> {
    if let Some(v) = value.as_u64() {
        return usize::try_from(v)
            .map_err(|_| Inkling2Error::new(key, "integer exceeds host usize"));
    }
    if let Some(v) = value.as_i32()
        && v >= 0
    {
        return usize::try_from(v)
            .map_err(|_| Inkling2Error::new(key, "integer exceeds host usize"));
    }
    Err(Inkling2Error::new(
        key,
        "expected non-negative integer metadata value",
    ))
}

fn optional_usize(
    metadata: &BTreeMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<usize>, Inkling2Error> {
    metadata
        .get(key)
        .map(|value| value_as_usize(value, key))
        .transpose()
}

fn required_positive_usize(
    metadata: &BTreeMap<String, MetadataValue>,
    key: &str,
) -> Result<usize, Inkling2Error> {
    let value = metadata
        .get(key)
        .ok_or_else(|| Inkling2Error::new(key, "missing required GGUF metadata key"))?;
    let parsed = value_as_usize(value, key)?;
    if parsed == 0 {
        return Err(Inkling2Error::new(key, "must be positive"));
    }
    Ok(parsed)
}

fn optional_positive_usize(
    metadata: &BTreeMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<usize>, Inkling2Error> {
    let Some(parsed) = optional_usize(metadata, key)? else {
        return Ok(None);
    };
    if parsed == 0 {
        return Err(Inkling2Error::new(key, "must be positive"));
    }
    Ok(Some(parsed))
}

fn required_positive_usize_or_array(
    metadata: &BTreeMap<String, MetadataValue>,
    key: &str,
    expected_len: usize,
) -> Result<Vec<usize>, Inkling2Error> {
    let value = metadata
        .get(key)
        .ok_or_else(|| Inkling2Error::new(key, "missing required GGUF metadata key"))?;
    if let MetadataValue::Array(items) = value {
        if items.len() != expected_len {
            return Err(Inkling2Error::new(
                key,
                format!(
                    "expected array of length {expected_len}, got {}",
                    items.len()
                ),
            ));
        }
        items
            .iter()
            .map(|item| {
                let parsed = value_as_usize(item, key)?;
                if parsed == 0 {
                    return Err(Inkling2Error::new(key, "array elements must be positive"));
                }
                Ok(parsed)
            })
            .collect()
    } else {
        let parsed = value_as_usize(value, key)?;
        if parsed == 0 {
            return Err(Inkling2Error::new(key, "must be positive"));
        }
        Ok(vec![parsed; expected_len])
    }
}

fn required_bool_array(
    metadata: &BTreeMap<String, MetadataValue>,
    key: &str,
    expected_len: usize,
) -> Result<Vec<bool>, Inkling2Error> {
    let value = metadata
        .get(key)
        .ok_or_else(|| Inkling2Error::new(key, "missing required GGUF metadata key"))?;
    let MetadataValue::Array(items) = value else {
        return Err(Inkling2Error::new(key, "expected boolean array"));
    };
    if items.len() != expected_len {
        return Err(Inkling2Error::new(
            key,
            format!(
                "expected array of length {expected_len}, got {}",
                items.len()
            ),
        ));
    }
    items
        .iter()
        .map(|item| {
            item.as_bool()
                .ok_or_else(|| Inkling2Error::new(key, "expected boolean element"))
        })
        .collect()
}

fn optional_bool(
    metadata: &BTreeMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<bool>, Inkling2Error> {
    metadata
        .get(key)
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| Inkling2Error::new(key, "expected boolean metadata value"))
        })
        .transpose()
}

fn required_positive_f32(
    metadata: &BTreeMap<String, MetadataValue>,
    key: &str,
) -> Result<f32, Inkling2Error> {
    let value = metadata
        .get(key)
        .ok_or_else(|| Inkling2Error::new(key, "missing required GGUF metadata key"))?
        .as_f32()
        .ok_or_else(|| Inkling2Error::new(key, "expected f32 metadata value"))?;
    if !value.is_finite() || value <= 0.0 {
        return Err(Inkling2Error::new(
            key,
            format!("must be finite and positive, got {value}"),
        ));
    }
    Ok(value)
}

fn optional_positive_f32(
    metadata: &BTreeMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<f32>, Inkling2Error> {
    let Some(value) = metadata.get(key) else {
        return Ok(None);
    };
    let parsed = value
        .as_f32()
        .ok_or_else(|| Inkling2Error::new(key, "expected f32 metadata value"))?;
    if !parsed.is_finite() || parsed <= 0.0 {
        return Err(Inkling2Error::new(
            key,
            format!("must be finite and positive, got {parsed}"),
        ));
    }
    Ok(Some(parsed))
}

fn required_nonnegative_f32(
    metadata: &BTreeMap<String, MetadataValue>,
    key: &str,
) -> Result<f32, Inkling2Error> {
    let value = metadata
        .get(key)
        .ok_or_else(|| Inkling2Error::new(key, "missing required GGUF metadata key"))?
        .as_f32()
        .ok_or_else(|| Inkling2Error::new(key, "expected f32 metadata value"))?;
    if !value.is_finite() || value < 0.0 {
        return Err(Inkling2Error::new(
            key,
            format!("must be finite and non-negative, got {value}"),
        ));
    }
    Ok(value)
}

fn optional_nonnegative_f32(
    metadata: &BTreeMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<f32>, Inkling2Error> {
    let Some(value) = metadata.get(key) else {
        return Ok(None);
    };
    let parsed = value
        .as_f32()
        .ok_or_else(|| Inkling2Error::new(key, "expected f32 metadata value"))?;
    if !parsed.is_finite() || parsed < 0.0 {
        return Err(Inkling2Error::new(
            key,
            format!("must be finite and non-negative, got {parsed}"),
        ));
    }
    Ok(Some(parsed))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[allow(clippy::too_many_lines)]
    pub(crate) fn mini_loop_metadata() -> BTreeMap<String, MetadataValue> {
        let mut m = BTreeMap::new();
        m.insert(
            "general.architecture".into(),
            MetadataValue::String("inkling2_loop".into()),
        );
        m.insert("inkling2_loop.block_count".into(), MetadataValue::U32(24));
        m.insert(
            "inkling2_loop.context_length".into(),
            MetadataValue::U32(1_048_576),
        );
        m.insert(
            "inkling2_loop.embedding_length".into(),
            MetadataValue::U32(2304),
        );
        m.insert(
            "inkling2_loop.attention.head_count".into(),
            MetadataValue::Array(vec![MetadataValue::I32(24); 24]),
        );
        m.insert(
            "inkling2_loop.attention.head_count_kv".into(),
            MetadataValue::Array(vec![MetadataValue::I32(8); 24]),
        );
        m.insert(
            "inkling2_loop.attention.key_length".into(),
            MetadataValue::U32(128),
        );
        m.insert(
            "inkling2_loop.attention.value_length".into(),
            MetadataValue::U32(128),
        );
        m.insert(
            "inkling2_loop.attention.key_length_swa".into(),
            MetadataValue::U32(128),
        );
        m.insert(
            "inkling2_loop.attention.value_length_swa".into(),
            MetadataValue::U32(128),
        );
        m.insert(
            "inkling2_loop.attention.layer_norm_rms_epsilon".into(),
            MetadataValue::F32(1e-6),
        );
        m.insert(
            "inkling2_loop.attention.qk_clip".into(),
            MetadataValue::F32(22.627_417),
        );
        m.insert(
            "inkling2_loop.attention.relative_length".into(),
            MetadataValue::U32(16),
        );
        m.insert(
            "inkling2_loop.attention.relative_extent".into(),
            MetadataValue::U32(1024),
        );
        m.insert(
            "inkling2_loop.attention.relative_clip".into(),
            MetadataValue::F32(8.0),
        );
        m.insert(
            "inkling2_loop.attention.log_scale_floor".into(),
            MetadataValue::U32(8192),
        );
        m.insert(
            "inkling2_loop.attention.log_scale_alpha".into(),
            MetadataValue::F32(0.1),
        );
        m.insert(
            "inkling2_loop.attention.sliding_window".into(),
            MetadataValue::U32(1024),
        );
        let pattern: Vec<MetadataValue> =
            (0..24).map(|i| MetadataValue::Bool(i % 6 != 5)).collect();
        m.insert(
            "inkling2_loop.attention.sliding_window_pattern".into(),
            MetadataValue::Array(pattern),
        );
        m.insert(
            "inkling2_loop.attn_res.block_size".into(),
            MetadataValue::U32(6),
        );
        m.insert(
            "inkling2_loop.shortconv.l_cache".into(),
            MetadataValue::U32(4),
        );
        m.insert(
            "inkling2_loop.vocab_size".into(),
            MetadataValue::U32(201_024),
        );
        m.insert(
            "inkling2_loop.vocab_size_unpadded".into(),
            MetadataValue::U32(200_058),
        );
        m.insert("inkling2_loop.expert_count".into(), MetadataValue::U32(128));
        m.insert(
            "inkling2_loop.expert_used_count".into(),
            MetadataValue::U32(8),
        );
        m.insert(
            "inkling2_loop.expert_feed_forward_length".into(),
            MetadataValue::U32(1152),
        );
        m.insert(
            "inkling2_loop.expert_weights_scale".into(),
            MetadataValue::F32(8.0),
        );
        m.insert(
            "inkling2_loop.activation.situ_beta".into(),
            MetadataValue::F32(4.0),
        );
        m.insert(
            "inkling2_loop.activation.situ_linear_beta".into(),
            MetadataValue::F32(25.0),
        );
        m.insert(
            "inkling2_loop.final_logit_softcapping".into(),
            MetadataValue::F32(30.0),
        );
        m.insert(
            "inkling2_loop.rope.dimension_count".into(),
            MetadataValue::U32(0),
        );
        m.insert("inkling2_loop.loop.start".into(), MetadataValue::U32(6));
        m.insert("inkling2_loop.loop.end".into(), MetadataValue::U32(18));
        m.insert("inkling2_loop.loop.count".into(), MetadataValue::U32(3));
        m.insert(
            "inkling2_loop.loop.gate_rank".into(),
            MetadataValue::U32(64),
        );
        m.insert(
            "inkling2_loop.loop.varying".into(),
            MetadataValue::Bool(false),
        );
        m
    }

    #[test]
    fn parses_inkling2_mini_loop_schedule_and_slot_mapping() {
        let cfg = Inkling2Config::from_metadata(&mini_loop_metadata()).expect("mini config");
        assert_eq!(cfg.architecture(), ArchitectureKind::Looped);
        assert_eq!(cfg.physical_layer_count(), 24);
        assert_eq!(cfg.state_slot_count(), 48);
        assert_eq!(cfg.execution_schedule().len(), 48);

        // Prelude: steps 0..6 -> slots 0..6, phys 0..6
        for i in 0..6 {
            let s = cfg.execution_schedule()[i];
            assert_eq!((s.state_slot, s.physical_layer, s.loop_pass), (i, i, None));
        }
        // Pass 0: steps 6..18 -> slots 6..18, phys 6..18, no cross-pass control
        for i in 0..12 {
            let s = cfg.execution_schedule()[6 + i];
            assert_eq!(
                (
                    s.state_slot,
                    s.physical_layer,
                    s.loop_pass,
                    s.uses_cross_pass_control
                ),
                (6 + i, 6 + i, Some(0), false)
            );
        }
        // Pass 1: steps 18..30 -> slots 24..36, phys 6..18, cross-pass control enabled
        for i in 0..12 {
            let s = cfg.execution_schedule()[18 + i];
            assert_eq!(
                (
                    s.state_slot,
                    s.physical_layer,
                    s.loop_pass,
                    s.uses_cross_pass_control
                ),
                (24 + i, 6 + i, Some(1), true)
            );
            assert_eq!(cfg.slot_physical_layer(24 + i), Some(6 + i));
        }
        // Pass 2: steps 30..42 -> slots 36..48, phys 6..18
        for i in 0..12 {
            let s = cfg.execution_schedule()[30 + i];
            assert_eq!(
                (
                    s.state_slot,
                    s.physical_layer,
                    s.loop_pass,
                    s.uses_cross_pass_control
                ),
                (36 + i, 6 + i, Some(2), true)
            );
            assert_eq!(cfg.slot_physical_layer(36 + i), Some(6 + i));
        }
        // Epilogue: steps 42..48 -> slots 18..24, phys 18..24
        for i in 0..6 {
            let s = cfg.execution_schedule()[42 + i];
            assert_eq!(
                (s.state_slot, s.physical_layer, s.loop_pass),
                (18 + i, 18 + i, None)
            );
        }

        // 40 SWA slots, 8 Full slots across the 48 logical slots
        let swa_slots = (0..cfg.state_slot_count())
            .filter(|&slot| cfg.slot_geometry(slot).unwrap().kind == AttentionKind::Sliding)
            .count();
        let full_slots = cfg.state_slot_count() - swa_slots;
        assert_eq!((swa_slots, full_slots), (40, 8));
    }

    #[test]
    fn rejects_fixed_loop_single_pass_but_allows_varloop_single_pass() {
        let mut fixed = mini_loop_metadata();
        fixed.insert("inkling2_loop.loop.count".into(), MetadataValue::U32(1));
        assert!(Inkling2Config::from_metadata(&fixed).is_err());

        fixed.insert(
            "inkling2_loop.loop.varying".into(),
            MetadataValue::Bool(true),
        );
        let cfg = Inkling2Config::from_metadata(&fixed).expect("varloop 1 pass");
        assert_eq!(cfg.state_slot_count(), 24);
    }

    #[test]
    fn parses_non_loop_inkling2_variant() {
        let mut m = BTreeMap::new();
        for (k, v) in mini_loop_metadata() {
            if k == "general.architecture" {
                m.insert(k, MetadataValue::String("inkling2".into()));
            } else if !k.starts_with("inkling2_loop.loop.")
                && let Some(rest) = k.strip_prefix("inkling2_loop.")
            {
                m.insert(format!("inkling2.{rest}"), v);
            }
        }
        let cfg = Inkling2Config::from_metadata(&m).expect("plain inkling2");
        assert_eq!(cfg.architecture(), ArchitectureKind::Plain);
        assert_eq!(cfg.loop_config(), None);
        assert_eq!(cfg.state_slot_count(), 24);
    }
}
