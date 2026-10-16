use std::collections::BTreeMap;

use super::*;
use crate::QwenLayerKind;

fn string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend((value.len() as u64).to_le_bytes());
    bytes.extend(value.as_bytes());
}

pub(crate) fn metadata() -> BTreeMap<String, MetadataValue> {
    let mut metadata = BTreeMap::from([
        (
            "general.architecture".to_owned(),
            MetadataValue::String("qwen35".to_owned()),
        ),
        (
            "qwen35.attention.layer_norm_rms_epsilon".to_owned(),
            MetadataValue::F32(1.0e-6),
        ),
        (
            "qwen35.rope.freq_base".to_owned(),
            MetadataValue::F32(1.0e7),
        ),
        (
            "qwen35.rope.dimension_sections".to_owned(),
            MetadataValue::Array([11, 11, 10, 0].map(MetadataValue::I32).to_vec()),
        ),
    ]);
    for (key, value) in [
        ("context_length", 262_144_u64),
        ("embedding_length", 5120),
        ("feed_forward_length", 17_408),
        ("block_count", 65),
        ("attention.head_count", 24),
        ("attention.head_count_kv", 4),
        ("attention.key_length", 256),
        ("attention.value_length", 256),
        ("rope.dimension_count", 64),
        ("full_attention_interval", 4),
        ("ssm.group_count", 16),
        ("ssm.inner_size", 6144),
        ("ssm.state_size", 128),
        ("ssm.time_step_rank", 48),
        ("ssm.conv_kernel", 4),
        ("nextn_predict_layers", 1),
        ("unused_field", 0),
    ] {
        metadata.insert(format!("qwen35.{key}"), MetadataValue::U64(value));
    }
    metadata
}

/// Metadata-only GGUF: any attempt to proceed to tokenizer/weights is an error.
pub(crate) fn encode(metadata: &BTreeMap<String, MetadataValue>) -> Vec<u8> {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3_u32.to_le_bytes());
    bytes.extend(0_u64.to_le_bytes());
    bytes.extend((metadata.len() as u64).to_le_bytes());
    for (key, value) in metadata {
        string(&mut bytes, key);
        match value {
            MetadataValue::String(value) => {
                bytes.extend(8_u32.to_le_bytes());
                string(&mut bytes, value);
            }
            MetadataValue::U64(value) => {
                bytes.extend(10_u32.to_le_bytes());
                bytes.extend(value.to_le_bytes());
            }
            MetadataValue::F32(value) => {
                bytes.extend(6_u32.to_le_bytes());
                bytes.extend(value.to_le_bytes());
            }
            MetadataValue::Bool(value) => {
                bytes.extend(7_u32.to_le_bytes());
                bytes.push(u8::from(*value));
            }
            MetadataValue::U32(value) => {
                bytes.extend(4_u32.to_le_bytes());
                bytes.extend(value.to_le_bytes());
            }
            MetadataValue::I32(value) => {
                bytes.extend(5_u32.to_le_bytes());
                bytes.extend(value.to_le_bytes());
            }
            MetadataValue::Array(values) => {
                bytes.extend(9_u32.to_le_bytes());
                let element_type = match values.first() {
                    Some(MetadataValue::Bool(_)) => 7_u32,
                    Some(MetadataValue::I32(_)) => 5_u32,
                    Some(MetadataValue::U32(_)) => 4_u32,
                    _ => panic!("unsupported fixture array type"),
                };
                bytes.extend(element_type.to_le_bytes());
                bytes.extend((values.len() as u64).to_le_bytes());
                for value in values {
                    match value {
                        MetadataValue::Bool(value) if element_type == 7 => {
                            bytes.push(u8::from(*value));
                        }
                        MetadataValue::U32(value) if element_type == 4 => {
                            bytes.extend(value.to_le_bytes());
                        }
                        MetadataValue::I32(value) if element_type == 5 => {
                            bytes.extend(value.to_le_bytes());
                        }
                        _ => panic!("nonuniform fixture array"),
                    }
                }
            }
            _ => panic!("unsupported fixture metadata type"),
        }
    }
    bytes.resize(bytes.len().next_multiple_of(32), 0);
    bytes
}

fn decode(metadata: &BTreeMap<String, MetadataValue>) -> Result<QwenConfig, GgufError> {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path =
        std::env::temp_dir().join(format!("ribn-qwen-config-{}-{id}.gguf", std::process::id()));
    std::fs::write(&path, encode(metadata)).unwrap();
    let file = GgufFile::open(&path).unwrap();
    let result = read(&file);
    drop(file);
    std::fs::remove_file(path).unwrap();
    result
}

#[test]
fn gguf_keys_normalize_into_the_model_definition() {
    let config = decode(&metadata()).unwrap();
    assert_eq!(config.embedding_length, 5120);
    assert_eq!(config.context_length, 262_144);
    assert_eq!(config.rms_norm_epsilon.to_bits(), 1.0e-6_f32.to_bits());
    assert_eq!(config.rope_frequency_base.to_bits(), 1.0e7_f32.to_bits());
    assert_eq!(config.rope_dimensions, 64);
    assert_eq!(config.layer_kind(3), Ok(QwenLayerKind::FullAttention));
}

#[test]
fn required_numerics_and_reference_defaults_do_not_guess_the_cuda_profile() {
    let mut values = metadata();
    values.remove("qwen35.rope.dimension_count");
    values.remove("qwen35.rope.freq_base");
    let config = decode(&values).unwrap();
    assert_eq!(config.rope_dimensions, 256);
    assert_eq!(config.rope_frequency_base.to_bits(), 10_000.0_f32.to_bits());
    let key = "qwen35.attention.layer_norm_rms_epsilon";
    values.remove(key);
    assert!(matches!(decode(&values), Err(GgufError::MissingMetadata(field)) if field == key));
    values.insert(key.to_owned(), MetadataValue::U64(1));
    assert!(
        matches!(decode(&values), Err(GgufError::MetadataTypeMismatch { key: field }) if field == key)
    );
}

#[test]
fn rope_scaling_normalizes_identity_and_rejects_unimplemented_transforms() {
    for (mode, modern, legacy, magnitude, accepted) in [
        ("linear", None, None, 1.0, true),
        ("linear", Some(0.0), Some(2.0), 1.0, true),
        ("linear", Some(-0.0), None, 1.0, true),
        ("linear", Some(1.0), Some(2.0), 1.0, true),
        ("none", Some(2.0), None, 1.0, true),
        ("linear", Some(2.0), None, 1.0, false),
        ("linear", None, Some(2.0), 1.0, false),
        ("none", Some(2.0), None, 2.0, false),
        ("yarn", Some(1.0), None, 1.0, false),
    ] {
        let mut values = metadata();
        values.insert(
            "qwen35.rope.scaling.type".to_owned(),
            MetadataValue::String(mode.to_owned()),
        );
        for (key, value) in [
            ("qwen35.rope.scaling.factor", modern),
            ("qwen35.rope.scale_linear", legacy),
        ] {
            if let Some(value) = value {
                values.insert(key.to_owned(), MetadataValue::F32(value));
            }
        }
        values.insert(
            "qwen35.rope.scaling.attn_factor".to_owned(),
            MetadataValue::F32(magnitude),
        );
        assert_eq!(
            decode(&values).is_ok(),
            accepted,
            "{mode}, modern={modern:?}, legacy={legacy:?}, magnitude={magnitude}"
        );
    }
}

#[test]
fn imrope_sections_must_select_text_positions_for_every_rotated_pair() {
    for (sections, accepted) in [
        ([11, 11, 10, 0], true),
        ([11, 11, 10, 1], true), // A nonzero fourth section need not be reached.
        ([1, 1, 30, 0], false),  // Fourth is zero but interleaving still has holes.
        ([0, 0, 0, 32], false),
        ([0, 0, 0, 0], false),
        ([-1, 11, 10, 0], false),
        ([i32::MAX, 11, 10, 0], false),
    ] {
        let mut values = metadata();
        values.insert(
            "qwen35.rope.dimension_sections".to_owned(),
            MetadataValue::Array(sections.map(MetadataValue::I32).to_vec()),
        );
        assert_eq!(decode(&values).is_ok(), accepted, "{sections:?}");
    }
    let mut values = metadata();
    values.insert(
        "qwen35.rope.dimension_sections".to_owned(),
        MetadataValue::Array([true, true, true, false].map(MetadataValue::Bool).to_vec()),
    );
    assert!(decode(&values).is_ok()); // The reference normalizes Boolean arrays to integers.
    values.insert(
        "qwen35.rope.dimension_sections".to_owned(),
        MetadataValue::I32(16),
    );
    assert!(decode(&values).is_ok()); // Reference scalar-to-four-section normalization.
    values.remove("qwen35.rope.dimension_sections");
    assert!(matches!(
        decode(&values),
        Err(GgufError::MissingMetadata(_))
    ));
}

#[test]
fn an_explicit_catalog_overrides_the_interval_but_must_fit_the_model_representation() {
    let encodings: [fn(bool) -> MetadataValue; 3] = [
        MetadataValue::Bool,
        |flag| MetadataValue::U32(if flag { 2 } else { 0 }),
        |flag| MetadataValue::I32(if flag { -1 } else { 0 }),
    ];
    for encode_flag in encodings {
        let mut values = metadata();
        let catalog = (0..65)
            .map(|index| encode_flag(index < 64 && (index + 1) % 4 != 0))
            .collect();
        let key = "qwen35.attention.recurrent_layers";
        values.insert(key.to_owned(), MetadataValue::Array(catalog));
        values.insert(
            "qwen35.full_attention_interval".to_owned(),
            MetadataValue::U64(2),
        );
        let config = decode(&values).unwrap();
        assert_eq!(config.full_attention_interval, 4);
        assert_eq!(config.layer_kind(3), Ok(QwenLayerKind::FullAttention));
        let MetadataValue::Array(catalog) = values.get_mut(key).unwrap() else {
            unreachable!()
        };
        catalog[6] = encode_flag(false);
        assert!(decode(&values).is_err());
        let MetadataValue::Array(catalog) = values.get_mut(key).unwrap() else {
            unreachable!()
        };
        catalog[6] = encode_flag(true);
        catalog[64] = encode_flag(true);
        assert!(decode(&values).is_err());
    }
}
