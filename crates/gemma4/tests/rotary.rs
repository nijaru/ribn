use engine_gemma4::{AttentionKind, Gemma4Config, Rope};
use serde_json::Value;
use sha2::{Digest, Sha256};

const CONFIG: &str = include_str!("../../../benchmarks/gemma4-reference/config.json");
const REFERENCE: &str = include_str!("fixtures/rotary.json");

#[test]
fn frequencies_match_pinned_transformers_geometry_and_fp32_policy() {
    let reference: Value = serde_json::from_str(REFERENCE).unwrap();
    assert_eq!(reference["schema_version"], 1);
    assert_eq!(
        reference["config_sha256"],
        format!("{:x}", Sha256::digest(CONFIG.as_bytes()))
    );
    for case in reference["cases"].as_array().unwrap() {
        let mut root: Value = serde_json::from_str(CONFIG).unwrap();
        for (key, value) in case["text_overrides"].as_object().unwrap() {
            root["text_config"][key] = value.clone();
        }
        let config = Gemma4Config::from_json(&root).unwrap();
        for (layer, geometry) in config.layers().iter().enumerate() {
            let kind = match geometry.kind {
                AttentionKind::Sliding => "sliding_attention",
                AttentionKind::Full => "full_attention",
            };
            let expected = &case["kinds"][kind];
            assert_eq!(
                expected["head_dim"].as_u64().unwrap(),
                geometry.head_dim as u64
            );
            let values = expected["inverse_frequency_bits"].as_array().unwrap();
            assert_eq!(values.len(), geometry.head_dim / 2);
            for (pair, bits) in values.iter().enumerate() {
                let bits = u32::try_from(bits.as_u64().unwrap()).unwrap();
                let got = config.rotary_inverse_frequency(layer, pair).unwrap();
                if bits == 0 {
                    assert_eq!(got.to_bits(), 0, "padding must be positive zero");
                    assert!(matches!(geometry.rope, Rope::Proportional { .. }));
                    assert!(pair >= geometry.head_dim / 8);
                    continue;
                }
                assert!(got.is_finite() && got > 0.0);
                // Independent, predeclared component budget: four FP32 ULPs
                // across platform power/reciprocal implementations. This is NOT a
                // tolerance for rotated tensors, accumulated phase or model logits.
                assert!(
                    got.to_bits().abs_diff(bits) <= 4,
                    "{} layer={layer} pair={pair}",
                    case["name"]
                );
                let theta = match geometry.rope {
                    Rope::Default { theta } | Rope::Proportional { theta } => theta,
                };
                // Independent FP64 equation, retaining the reference's explicit
                // FP32 index/divisor conversion and division before exponentiation.
                let exponent = f32::from(u16::try_from(2 * pair).unwrap())
                    / f32::from(u16::try_from(geometry.head_dim).unwrap());
                let exact = f64::from(theta).powf(-f64::from(exponent));
                let relative = (f64::from(got) - exact).abs() / exact;
                assert!(relative <= 3.0 * f64::from(f32::EPSILON));
            }
            assert_eq!(
                config.rotary_inverse_frequency(layer, 0).unwrap().to_bits(),
                1_f32.to_bits()
            );
        }
    }
}

#[test]
fn bounds_and_unrepresentable_frequencies_fail_without_materialization() {
    let mut root: Value = serde_json::from_str(CONFIG).unwrap();
    let config = Gemma4Config::from_json(&root).unwrap();
    assert!(
        config
            .rotary_inverse_frequency(config.layers().len(), 0)
            .is_err()
    );
    assert!(config.rotary_inverse_frequency(0, usize::MAX).is_err());
    assert!(config.rotary_inverse_frequency(5, 256).is_err());
    assert!(config.rotary_inverse_frequency(5, 63).unwrap().is_normal());
    assert_eq!(config.rotary_inverse_frequency(5, 64).unwrap().to_bits(), 0);
    assert_eq!(
        config.rotary_inverse_frequency(5, 255).unwrap().to_bits(),
        0
    );

    root["text_config"]["rope_parameters"]["sliding_attention"]["rope_theta"] = Value::from(1e-44);
    let tiny_theta = Gemma4Config::from_json(&root).unwrap();
    assert!(tiny_theta.rotary_inverse_frequency(0, 127).is_err());
}
