//! Normalize GGUF metadata into the currently executable text-model semantics.
//!
//! Defaults/precedence follow llama.cpp b81c99b479d4c24e5eeca10de99032ebd343ef8f,
//! src/llama-model.cpp:1298-1340 and src/models/qwen35.cpp:4-23. Scaled `RoPE` and
//! nonperiodic layer catalogs need a different model representation, not ignored keys.
use super::{GgufError, GgufFile, MetadataValue};
use crate::QwenConfig;

pub(super) fn read(file: &GgufFile) -> Result<QwenConfig, GgufError> {
    let architecture = file
        .metadata("general.architecture")
        .and_then(MetadataValue::as_str)
        .ok_or_else(|| GgufError::MissingMetadata("general.architecture".to_owned()))?;
    if architecture != "qwen35" {
        return Err(GgufError::UnsupportedArchitecture(architecture.to_owned()));
    }
    let block_count = required_u32(file, "qwen35.block_count")?;
    let nextn_predict_layers = required_u32(file, "qwen35.nextn_predict_layers")?;
    let key_length = required_u32(file, "qwen35.attention.key_length")?;
    let config = QwenConfig {
        context_length: required_u64(file, "qwen35.context_length")?,
        embedding_length: required_u32(file, "qwen35.embedding_length")?,
        feed_forward_length: required_u32(file, "qwen35.feed_forward_length")?,
        block_count,
        attention_heads: required_u32(file, "qwen35.attention.head_count")?,
        kv_heads: required_u32(file, "qwen35.attention.head_count_kv")?,
        key_length,
        value_length: required_u32(file, "qwen35.attention.value_length")?,
        rms_norm_epsilon: float(file, "qwen35.attention.layer_norm_rms_epsilon", None)?,
        rope_dimensions: optional_u32(file, "qwen35.rope.dimension_count", key_length)?,
        rope_frequency_base: float(file, "qwen35.rope.freq_base", Some(10_000.0))?,
        full_attention_interval: layer_interval(file, block_count, nextn_predict_layers)?,
        ssm_group_count: required_u32(file, "qwen35.ssm.group_count")?,
        ssm_inner_size: required_u32(file, "qwen35.ssm.inner_size")?,
        ssm_state_size: required_u32(file, "qwen35.ssm.state_size")?,
        ssm_time_step_rank: required_u32(file, "qwen35.ssm.time_step_rank")?,
        ssm_conv_kernel: required_u32(file, "qwen35.ssm.conv_kernel")?,
        nextn_predict_layers,
    };
    config
        .validate()
        .map_err(|error| GgufError::InvalidModelConfiguration(error.reason()))?;
    validate_text_rope(file, config.rope_dimensions)?;
    Ok(config)
}

fn required_u64(file: &GgufFile, key: &str) -> Result<u64, GgufError> {
    file.metadata(key)
        .ok_or_else(|| GgufError::MissingMetadata(key.to_owned()))?
        .as_u64()
        .ok_or_else(|| type_mismatch(key))
}

fn required_u32(file: &GgufFile, key: &str) -> Result<u32, GgufError> {
    u32::try_from(required_u64(file, key)?).map_err(|_| type_mismatch(key))
}

fn optional_u32(file: &GgufFile, key: &str, default: u32) -> Result<u32, GgufError> {
    if file.metadata(key).is_none() {
        Ok(default)
    } else {
        required_u32(file, key)
    }
}

fn float(file: &GgufFile, key: &str, default: Option<f32>) -> Result<f32, GgufError> {
    match file.metadata(key) {
        Some(MetadataValue::F32(value)) => Ok(*value),
        Some(_) => Err(type_mismatch(key)),
        None => default.ok_or_else(|| GgufError::MissingMetadata(key.to_owned())),
    }
}

fn type_mismatch(key: &str) -> GgufError {
    GgufError::MetadataTypeMismatch {
        key: key.to_owned(),
    }
}

fn unsupported(reason: &'static str) -> GgufError {
    GgufError::InvalidModelConfiguration(reason)
}

fn recurrent_flag(value: &MetadataValue) -> Option<bool> {
    match value {
        MetadataValue::Bool(value) => Some(*value),
        MetadataValue::U32(value) => Some(*value != 0),
        MetadataValue::I32(value) => Some(*value != 0),
        _ => None,
    }
}

fn layer_interval(file: &GgufFile, blocks: u32, mtp: u32) -> Result<u32, GgufError> {
    const KEY: &str = "qwen35.attention.recurrent_layers";
    let Some(catalog) = file.metadata(KEY) else {
        return optional_u32(file, "qwen35.full_attention_interval", 4);
    };
    let language = blocks
        .checked_sub(mtp)
        .ok_or_else(|| unsupported("MTP layer count exceeds total block count"))?;
    let MetadataValue::Array(values) = catalog else {
        return match recurrent_flag(catalog) {
            Some(false) => Ok(1),
            Some(true) => Err(unsupported(
                "all-recurrent Qwen layer catalogs are unsupported",
            )),
            _ => Err(type_mismatch(KEY)),
        };
    };
    if values.len() != usize::try_from(blocks).map_err(|_| type_mismatch(KEY))? {
        return Err(unsupported(
            "Qwen recurrent-layer catalog length differs from block count",
        ));
    }
    let mut interval = None;
    for (index, value) in (0..blocks).zip(values) {
        let flag = recurrent_flag(value).ok_or_else(|| type_mismatch(KEY))?;
        if index < language && !flag && interval.is_none() {
            interval = Some(index + 1);
        }
        if let Some(interval) = interval {
            if flag != (index < language && !(index + 1).is_multiple_of(interval)) {
                return Err(unsupported(
                    "nonperiodic Qwen recurrent-layer catalogs are unsupported",
                ));
            }
        } else if index >= language && flag {
            return Err(unsupported("Qwen MTP layers must not be recurrent"));
        }
    }
    // This model representation requires regular hybrid groups; the catalog is
    // authoritative and any serialized interval is intentionally not consulted.
    interval.ok_or_else(|| unsupported("all-recurrent Qwen layer catalogs are unsupported"))
}

fn section_size(value: &MetadataValue) -> Option<u64> {
    let value = match value {
        MetadataValue::Bool(value) => Some(u64::from(*value)),
        MetadataValue::I64(value) => u64::try_from(*value).ok(),
        _ => value
            .as_u64()
            .or_else(|| value.as_i32().and_then(|v| u64::try_from(v).ok())),
    }?;
    u32::try_from(value).ok().map(u64::from)
}

fn validate_text_rope(file: &GgufFile, dimensions: u32) -> Result<(), GgufError> {
    const MODE: &str = "qwen35.rope.scaling.type";
    const SECTIONS: &str = "qwen35.rope.dimension_sections";
    let mode = match file.metadata(MODE) {
        None => "linear",
        Some(value) => value.as_str().ok_or_else(|| type_mismatch(MODE))?,
    };
    match mode {
        "none" => {} // The reference explicitly overrides frequency scaling to one.
        "linear" => {
            let key = if file.metadata("qwen35.rope.scaling.factor").is_some() {
                "qwen35.rope.scaling.factor"
            } else {
                "qwen35.rope.scale_linear"
            };
            let factor = float(file, key, Some(0.0))?;
            // Zero is the reference's identity sentinel, not a scaling factor.
            if factor.abs().to_bits() != 0.0_f32.to_bits() && factor.to_bits() != 1.0_f32.to_bits()
            {
                return Err(unsupported("scaled Qwen text RoPE is unsupported"));
            }
        }
        _ => return Err(unsupported("unsupported Qwen text RoPE scaling mode")),
    }
    if float(file, "qwen35.rope.scaling.attn_factor", Some(1.0))?.to_bits() != 1.0_f32.to_bits() {
        return Err(unsupported("Qwen RoPE attention scaling is unsupported"));
    }
    let value = file
        .metadata(SECTIONS)
        .ok_or_else(|| GgufError::MissingMetadata(SECTIONS.to_owned()))?;
    let sections = match value {
        MetadataValue::Array(values) if values.len() == 4 => {
            let mut sections = [0_u64; 4];
            for (section, value) in sections.iter_mut().zip(values) {
                *section = section_size(value).ok_or_else(|| type_mismatch(SECTIONS))?;
            }
            sections
        }
        _ => [section_size(value).ok_or_else(|| type_mismatch(SECTIONS))?; 4],
    };
    let total: u64 = sections.iter().sum();
    if total == 0
        || i32::try_from(total).is_err()
        || sections[..3]
            .iter()
            .any(|section| i32::try_from(*section * 3).is_err())
    {
        return Err(unsupported(
            "Qwen RoPE sections have an empty or overflowing selector",
        ));
    }
    // IMROPE text positions are [p,p,p,0]. For every visited sector, the
    // residue's section must cover it; otherwise the reference selects zero.
    // The visited sectors are exactly 0..min(rotated_pairs,total). This closed
    // form avoids work proportional to an untrusted declared dimension count.
    let visited = (u64::from(dimensions) / 2).min(total);
    if (0..3_u64)
        .zip(sections)
        .any(|(residue, section)| section < visited.saturating_sub(residue).div_ceil(3))
    {
        return Err(unsupported(
            "Qwen IMROPE sections do not reduce to ordinary text RoPE",
        ));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests;
