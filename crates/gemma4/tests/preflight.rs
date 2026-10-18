use std::collections::BTreeMap;
use std::io::Read;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use engine_gemma4::{AttentionKind, Exclusion, Gemma4Config, Gemma4Source, ParameterId};
use ribn_hf::LocalModelPackage;
use serde_json::{Value, json};

const E4B: &str = include_str!("../../../benchmarks/gemma4-reference/config.json");
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Dir(PathBuf);
impl Dir {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "ribn-gemma4-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn tiny() -> Value {
    let mut value: Value = serde_json::from_str(E4B).unwrap();
    let t = &mut value["text_config"];
    for (name, n) in [
        ("vocab_size", 7),
        ("vocab_size_per_layer_input", 9),
        ("hidden_size", 7),
        ("intermediate_size", 11),
        ("num_hidden_layers", 4),
        ("num_attention_heads", 2),
        ("num_key_value_heads", 1),
        ("num_kv_shared_layers", 2),
        ("head_dim", 4),
        ("global_head_dim", 8),
        ("hidden_size_per_layer_input", 2),
        ("sliding_window", 4),
        ("max_position_embeddings", 2048),
    ] {
        t[name] = json!(n);
    }
    t["layer_types"] = json!([
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention"
    ]);
    value
}

// Independent small package schema: do not generate this using production specs.
fn inventory() -> BTreeMap<String, (Vec<usize>, &'static str)> {
    let mut tensors = BTreeMap::new();
    for (name, shape) in [
        ("embed_tokens.weight", vec![7, 7]),
        ("embed_tokens_per_layer.weight", vec![9, 8]),
        ("per_layer_model_projection.weight", vec![8, 7]),
        ("per_layer_projection_norm.weight", vec![2]),
        ("norm.weight", vec![7]),
    ] {
        tensors.insert(format!("model.language_model.{name}"), (shape, "BF16"));
    }
    for i in 0..4 {
        let head = if i % 2 == 0 { 4 } else { 8 };
        for (name, shape) in [
            ("self_attn.q_proj.weight", vec![2 * head, 7]),
            ("self_attn.k_proj.weight", vec![head, 7]),
            ("self_attn.v_proj.weight", vec![head, 7]),
            ("self_attn.o_proj.weight", vec![7, 2 * head]),
            ("self_attn.q_norm.weight", vec![head]),
            ("self_attn.k_norm.weight", vec![head]),
            ("input_layernorm.weight", vec![7]),
            ("post_attention_layernorm.weight", vec![7]),
            ("pre_feedforward_layernorm.weight", vec![7]),
            ("post_feedforward_layernorm.weight", vec![7]),
            ("mlp.gate_proj.weight", vec![11, 7]),
            ("mlp.up_proj.weight", vec![11, 7]),
            ("mlp.down_proj.weight", vec![7, 11]),
            ("per_layer_input_gate.weight", vec![2, 7]),
            ("per_layer_projection.weight", vec![7, 2]),
            ("post_per_layer_input_norm.weight", vec![7]),
            ("layer_scalar", vec![1]),
        ] {
            tensors.insert(
                format!("model.language_model.layers.{i}.{name}"),
                (shape, "BF16"),
            );
        }
    }
    tensors
}

fn artifact(tensors: &BTreeMap<String, (Vec<usize>, &str)>) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (i, (name, (shape, dtype))) in tensors.iter().enumerate() {
        let start = data.len();
        let width = if *dtype == "BF16" { 2 } else { 4 };
        let bits = 0x3f80_u16 + u16::try_from(i).unwrap();
        for _ in 0..shape.iter().product::<usize>() {
            if width == 2 {
                data.extend(bits.to_le_bytes());
            } else {
                data.extend(1_f32.to_le_bytes());
            }
        }
        header.insert(
            name.clone(),
            json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let mut encoded = serde_json::to_vec(&header).unwrap();
    while !encoded.len().is_multiple_of(8) {
        encoded.push(b' ');
    }
    let mut output = u64::try_from(encoded.len()).unwrap().to_le_bytes().to_vec();
    output.extend(encoded);
    output.extend(data);
    output
}

fn package(
    dir: &Dir,
    tensors: &BTreeMap<String, (Vec<usize>, &str)>,
    sharded: bool,
) -> LocalModelPackage {
    std::fs::write(
        dir.0.join("config.json"),
        serde_json::to_vec(&tiny()).unwrap(),
    )
    .unwrap();
    if sharded {
        let mut parts = [BTreeMap::new(), BTreeMap::new()];
        let mut index = BTreeMap::new();
        for (i, (name, item)) in tensors.iter().enumerate() {
            parts[i % 2].insert(name.clone(), item.clone());
            index.insert(name, format!("part-{}.safetensors", i % 2));
        }
        for (i, part) in parts.iter().enumerate() {
            std::fs::write(dir.0.join(format!("part-{i}.safetensors")), artifact(part)).unwrap();
        }
        std::fs::write(
            dir.0.join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({"weight_map":index})).unwrap(),
        )
        .unwrap();
    } else {
        std::fs::write(dir.0.join("model.safetensors"), artifact(tensors)).unwrap();
    }
    LocalModelPackage::open(&dir.0).unwrap()
}

#[test]
fn real_config_resolves_heterogeneous_widths_and_shared_owners() {
    let c = Gemma4Config::from_json(&serde_json::from_str(E4B).unwrap()).unwrap();
    assert_eq!(
        (c.hidden_size(), c.packed_ple_dim(), c.layers().len()),
        (2560, 10752, 42)
    );
    for (i, layer) in c.layers().iter().enumerate() {
        let full = i % 6 == 5;
        assert_eq!(
            layer.kind,
            if full {
                AttentionKind::Full
            } else {
                AttentionKind::Sliding
            }
        );
        assert_eq!(layer.head_dim, if full { 512 } else { 256 });
        assert_eq!(
            layer.kv_owner,
            if i < 24 {
                i
            } else if full {
                23
            } else {
                22
            }
        );
    }
    let small = Gemma4Config::from_json(&tiny()).unwrap();
    assert_eq!(small.hidden_size(), 7); // deliberately not divisible by query heads
    assert_eq!(small.layers()[2].kv_owner, 0);
    assert_eq!(small.layers()[3].kv_owner, 1);
}

#[test]
fn invalid_or_unimplemented_semantics_fail_before_inventory() {
    for (field, value) in [
        ("hidden_size", json!(0)),
        ("num_key_value_heads", json!(3)),
        ("num_kv_shared_layers", json!(4)),
        ("global_head_dim", json!(6)),
        ("attention_k_eq_v", json!(true)),
        ("attention_bias", json!(true)),
        ("enable_moe_block", json!(true)),
        ("use_double_wide_mlp", json!(true)),
        ("use_bidirectional_attention", json!("all")),
        ("tie_word_embeddings", json!(false)),
        ("dtype", json!("float32")),
        ("hidden_activation", json!("relu")),
        ("per_layer_config", json!({"0":{"head_dim":8}})),
        ("intermediate_size", json!(u64::MAX)),
    ] {
        let mut c = tiny();
        c["text_config"][field] = value;
        assert!(Gemma4Config::from_json(&c).is_err(), "{field}");
    }
    let mut c = tiny();
    c["text_config"]["layer_types"] = json!([
        "full_attention",
        "full_attention",
        "sliding_attention",
        "full_attention"
    ]);
    assert!(Gemma4Config::from_json(&c).is_err(), "missing local donor");
    let mut c = tiny();
    c["text_config"]["rope_parameters"]["full_attention"]["partial_rotary_factor"] = json!(0.5);
    assert!(Gemma4Config::from_json(&c).is_err());
}

#[test]
fn single_and_sharded_sources_preserve_ties_encoding_and_explicit_exclusions() {
    for sharded in [false, true] {
        let dir = Dir::new();
        let mut tensors = inventory();
        tensors.insert("model.vision_tower.test.weight".into(), (vec![2], "BF16"));
        let p = package(&dir, &tensors, sharded);
        let source = Gemma4Source::open(&p).unwrap();
        assert_eq!(source.parameters().len(), 67);
        assert_eq!(
            source
                .excluded()
                .iter()
                .filter(|p| p.reason == Exclusion::SharedTailProjection)
                .count(),
            6
        );
        assert_eq!(
            source
                .excluded()
                .iter()
                .filter(|p| p.reason == Exclusion::NonText)
                .count(),
            1
        );
        assert!(std::ptr::eq(
            source.output_projection(),
            source.parameter(ParameterId::TokenEmbedding).unwrap()
        ));
        let mut total = 0;
        for parameter in source.parameters().values() {
            let spec = parameter.spec();
            let ordinal = tensors.keys().position(|name| name == &spec.name).unwrap();
            // The fixture numbers payloads independently within each shard.
            let ordinal = if sharded { ordinal / 2 } else { ordinal };
            let mut data = Vec::new();
            parameter.reader().read_to_end(&mut data).unwrap();
            let expected = (0x3f80_u16 + u16::try_from(ordinal).unwrap()).to_le_bytes();
            let (words, tail) = data.as_chunks::<2>();
            assert!(tail.is_empty());
            assert!(words.iter().all(|b| b == &expected));
            assert_eq!(data.len(), spec.byte_len().unwrap());
            total += data.len();
            let mut again = Vec::new();
            parameter.reader().read_to_end(&mut again).unwrap();
            assert_eq!(data, again, "prototype cursors never advance");
        }
        assert_eq!(source.encoded_bytes(), total);
    }
}

#[test]
fn malformed_missing_and_unexpected_parameters_never_publish_a_source() {
    for mode in 0..5 {
        let dir = Dir::new();
        let mut tensors = inventory();
        let name = "model.language_model.layers.3.layer_scalar";
        match mode {
            0 => {
                tensors.remove(name);
            }
            1 => {
                tensors.get_mut(name).unwrap().0 = vec![2];
            }
            2 => {
                tensors.get_mut(name).unwrap().1 = "F32";
            }
            3 => {
                tensors.insert(
                    "model.language_model.layers.3.unimplemented.weight".into(),
                    (vec![1], "BF16"),
                );
            }
            _ => {
                tensors.insert("lm_head.weight".into(), (vec![7, 7], "BF16"));
            }
        }
        let p = package(&dir, &tensors, false);
        assert!(Gemma4Source::open(&p).is_err(), "mode {mode}");
    }
}

#[test]
fn source_readers_survive_package_drop_and_pathname_replacement() {
    let dir = Dir::new();
    let p = package(&dir, &inventory(), false);
    let source = Gemma4Source::open(&p).unwrap();
    drop(p);
    let selected = source.parameter(ParameterId::TokenEmbedding).unwrap();
    let mut expected = Vec::new();
    selected.reader().read_to_end(&mut expected).unwrap();
    std::fs::rename(
        dir.0.join("model.safetensors"),
        dir.0.join("old.safetensors"),
    )
    .unwrap();
    std::fs::write(dir.0.join("model.safetensors"), b"not the selected source").unwrap();
    let mut reader = selected.reader();
    drop(source);
    let mut actual = Vec::new();
    reader.read_to_end(&mut actual).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn unused_shared_tail_weights_are_optional_but_index_routing_is_not() {
    let dir = Dir::new();
    let mut tensors = inventory();
    tensors.retain(|name, _| {
        !(name.starts_with("model.language_model.layers.2.")
            || name.starts_with("model.language_model.layers.3."))
            || ![
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.k_norm.weight",
            ]
            .iter()
            .any(|suffix| name.ends_with(suffix))
    });
    let p = package(&dir, &tensors, true);
    assert_eq!(Gemma4Source::open(&p).unwrap().parameters().len(), 67);
    drop(p);
    let index_path = dir.0.join("model.safetensors.index.json");
    let original: Value = serde_json::from_slice(&std::fs::read(&index_path).unwrap()).unwrap();
    for dangling in [false, true] {
        let mut index = original.clone();
        if dangling {
            index["weight_map"]["model.language_model.absent.weight"] = json!("part-0.safetensors");
        } else {
            let name = "model.language_model.embed_tokens.weight";
            let old = index["weight_map"][name].as_str().unwrap();
            index["weight_map"][name] = json!(if old == "part-0.safetensors" {
                "part-1.safetensors"
            } else {
                "part-0.safetensors"
            });
        }
        std::fs::write(&index_path, serde_json::to_vec(&index).unwrap()).unwrap();
        let p = LocalModelPackage::open(&dir.0).unwrap();
        assert!(Gemma4Source::open(&p).is_err());
    }
}
