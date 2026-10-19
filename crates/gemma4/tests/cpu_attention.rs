//! Host CPU attention arithmetic, not a decoder or a performance claim.

use engine_gemma4::{AttentionConfig, attention, attention_with_weights, bf16};
use ribn_foundation::ScalarType;
use ribn_safetensors::SafeTensorArtifact;
use serde_json::Value;
use std::io::Read;
use std::path::PathBuf;

fn value(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn ulp(bits: u16) -> f32 {
    let difference = (value(bits.wrapping_add(1)) - value(bits)).abs();
    if difference.is_finite() && difference > 0.0 {
        difference
    } else {
        value(bits).abs() * 2.0 * f32::EPSILON
    }
}

#[test]
fn causal_and_sliding_masks_expose_only_visible_keys() {
    // One head, one channel, zero scores: outputs are means of visible values.
    let query = [bf16(1.0); 4];
    let keys = [bf16(0.0); 4];
    let values = [bf16(1.0), bf16(2.0), bf16(3.0), bf16(4.0)];
    let config = |window| AttentionConfig {
        heads: 1,
        kv_heads: 1,
        width: 1,
        window,
    };
    let (output, weights) = attention_with_weights(&query, &keys, &values, config(None), 0, 0)
        .expect("causal attention");
    assert_eq!(output, [bf16(1.0), bf16(1.5), bf16(2.0), bf16(2.5)]);
    for position in 0..4 {
        for key in 0..4 {
            #[expect(clippy::cast_precision_loss, reason = "test positions are small")]
            let uniform = 1.0 / (position + 1) as f32;
            let expected = if key <= position { bf16(uniform) } else { 0 };
            assert_eq!(
                weights[position * 4 + key],
                expected,
                "causal weight {position}:{key}"
            );
        }
    }
    let (output, weights) = attention_with_weights(&query, &keys, &values, config(Some(2)), 0, 0)
        .expect("sliding attention");
    assert_eq!(output, [bf16(1.0), bf16(1.5), bf16(2.5), bf16(3.5)]);
    assert_eq!(weights[2 * 4], 0, "sliding window excludes key zero");
    assert_eq!(weights[2 * 4 + 1], bf16(0.5));
    assert_eq!(weights[2 * 4 + 2], bf16(0.5));
}

#[test]
fn grouped_heads_share_key_value_heads_and_softmax_normalizes() {
    let query = [bf16(1.0); 4];
    let keys = [bf16(0.0); 2];
    let values = [bf16(10.0), bf16(20.0)];
    let grouped = AttentionConfig {
        heads: 4,
        kv_heads: 2,
        width: 1,
        window: None,
    };
    let output = attention(&query, &keys, &values, grouped, 0, 0).expect("grouped");
    assert_eq!(output, [bf16(10.0), bf16(10.0), bf16(20.0), bf16(20.0)]);
    // One query over two visible keys with scores 1 and 0.
    let query = [bf16(1.0)];
    let keys = [bf16(1.0), bf16(0.0)];
    let values = [bf16(1.0), bf16(2.0)];
    let single = AttentionConfig {
        heads: 1,
        kv_heads: 1,
        width: 1,
        window: None,
    };
    let output = attention(&query, &keys, &values, single, 1, 0).expect("two keys");
    let total = 1.0_f32.exp() + 0.0_f32.exp();
    let expected = bf16((1.0_f32.exp() / total) * 1.0 + (0.0_f32.exp() / total) * 2.0);
    assert_eq!(output, [expected]);
}

#[test]
fn invalid_attention_spans_reject() {
    let query = [bf16(0.0); 2];
    let keys = [bf16(0.0); 2];
    let values = [bf16(0.0); 2];
    let plain = AttentionConfig {
        heads: 1,
        kv_heads: 1,
        width: 1,
        window: None,
    };
    // Future query beyond the key span.
    assert!(attention(&query, &keys, &values, plain, 1, 0).is_err());
    // Queries before the key span.
    assert!(attention(&query, &keys, &values, plain, 0, 1).is_err());
    // Non-whole head groups and zero window.
    let uneven = AttentionConfig {
        heads: 3,
        kv_heads: 2,
        width: 1,
        window: None,
    };
    assert!(attention(&query, &keys, &values, uneven, 0, 0).is_err());
    let closed = AttentionConfig {
        window: Some(0),
        ..plain
    };
    assert!(attention(&query, &keys, &values, closed, 0, 0).is_err());
}

fn environment(name: &str) -> PathBuf {
    PathBuf::from(
        std::env::var_os(name)
            .unwrap_or_else(|| panic!("{name} must name the archived attention reference")),
    )
}

fn tap(artifact: &SafeTensorArtifact, name: &str) -> Vec<u16> {
    let tensor = artifact.tensor(name).unwrap();
    assert_eq!(*tensor.scalar_type(), ScalarType::I32, "{name}");
    let mut bytes = vec![0_u8; tensor.len()];
    tensor.reader().read_exact(&mut bytes).unwrap();
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u16::try_from(i32::from_le_bytes(*word)).unwrap())
        .collect()
}

#[test]
#[ignore = "requires the archived pinned attention reference"]
fn host_attention_matches_the_pinned_cpu_operands() {
    let path = environment("RIBN_ATTENTION_REFERENCE");
    let report: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(report["schema_version"], 1);
    assert_eq!(report["kind"], "cpu_bf16_attention_operands");
    assert_eq!(report["complete"], true);
    assert_eq!(report["precision"], "bf16");
    let artifact = SafeTensorArtifact::open(path.with_file_name("tensors.safetensors")).unwrap();
    let cases = report["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 4);
    let mut weight_usage = 0.0_f32;
    let mut output_usage = 0.0_f32;
    for case in cases {
        let layer = case["layer"].as_u64().unwrap();
        let heads = usize::try_from(case["heads"].as_u64().unwrap()).unwrap();
        let kv_heads = usize::try_from(case["kv_heads"].as_u64().unwrap()).unwrap();
        let width = usize::try_from(case["head_dim"].as_u64().unwrap()).unwrap();
        let sequence = usize::try_from(case["sequence"].as_u64().unwrap()).unwrap();
        let window = case["window"].as_u64().map(|w| usize::try_from(w).unwrap());
        let prefix = format!("layer.{layer}");
        let query = tap(&artifact, &format!("{prefix}.query"));
        let keys = tap(&artifact, &format!("{prefix}.key"));
        let values = tap(&artifact, &format!("{prefix}.value"));
        let expected_scores = tap(&artifact, &format!("{prefix}.scores"));
        let expected_weights = tap(&artifact, &format!("{prefix}.weights"));
        let expected_output = tap(&artifact, &format!("{prefix}.output"));
        assert_eq!(query.len(), sequence * heads * width);
        assert_eq!(keys.len(), sequence * kv_heads * width);
        let config = AttentionConfig {
            heads,
            kv_heads,
            width,
            window,
        };
        let (output, weights) =
            attention_with_weights(&query, &keys, &values, config, 0, 0).unwrap();
        let groups = heads / kv_heads;
        for head in 0..heads {
            let kv_head = head / groups;
            for position in 0..sequence {
                // Query position p attends keys p-window+1..=p, or 0..=p.
                let first = window.map_or(0, |length| position.saturating_sub(length - 1));
                // Score quantization can move a BF16 softmax value by at most
                // twice the largest visible score's encoding step.
                let mut slack = 0.0_f32;
                for key in first..=position {
                    slack = slack.max(ulp(
                        expected_scores[(head * sequence + position) * sequence + key]
                    ));
                }
                let weight_base = head * sequence * sequence + position * sequence;
                for key in 0..sequence {
                    let (actual, expected) = (
                        weights[weight_base + key],
                        expected_weights[weight_base + key],
                    );
                    if key < first || key > position {
                        assert_eq!(actual, 0, "{prefix} weight {head}:{position}:{key} masked");
                        assert_eq!(expected, 0, "{prefix} reference weight not masked");
                        continue;
                    }
                    let bound = 2.0 * slack * value(expected).abs() + ulp(expected);
                    let error = (value(actual) - value(expected)).abs();
                    weight_usage = weight_usage.max(error / bound);
                    assert!(
                        error <= bound,
                        "{prefix} weight {head}:{position}:{key} error {error:e}"
                    );
                }
                #[expect(clippy::cast_precision_loss, reason = "visible key counts are small")]
                let visible = (position - first + 1) as f32;
                let gamma = visible * f32::EPSILON / (1.0 - visible * f32::EPSILON);
                for column in 0..width {
                    let mut magnitude = 0.0_f32;
                    for key in first..=position {
                        let weight = value(expected_weights[weight_base + key]);
                        let stored =
                            value(values[key * kv_heads * width + kv_head * width + column]);
                        magnitude += (weight * stored).abs();
                    }
                    let index = position * heads * width + head * width + column;
                    let reference = expected_output[index];
                    let bound = (gamma + 2.0 * slack) * magnitude + ulp(reference);
                    let error = (value(output[index]) - value(reference)).abs();
                    output_usage = output_usage.max(error / bound);
                    assert!(
                        error <= bound,
                        "{prefix} output {index} error {error:e} exceeds {bound:e}"
                    );
                }
            }
        }
    }
    assert!(weight_usage <= 1.0, "weight bound usage {weight_usage}");
    assert!(output_usage <= 1.0, "output bound usage {output_usage}");
    println!("weight usage {weight_usage}, output usage {output_usage}");
}
