use engine_gemma4::{AttentionKind, Exclusion, Gemma4Config, Gemma4Source, ParameterId};
use ribn_hf::LocalModelPackage;
use serde_json::{Value, json};
use std::io::Read;
#[path = "common/weights.rs"]
mod weights;
use weights::{Dir, E4B, inventory, package, tiny};

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
