//! Independent small package schema for source and CUDA materialization gates.
use ribn_hf::LocalModelPackage;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

pub(crate) const E4B: &str = include_str!("../../../../benchmarks/gemma4-reference/config.json");
static NEXT: AtomicU64 = AtomicU64::new(0);
pub(crate) struct Dir(pub(crate) PathBuf);
impl Dir {
    pub(crate) fn new() -> Self {
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

pub(crate) fn tiny() -> Value {
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
pub(crate) fn inventory() -> BTreeMap<String, (Vec<usize>, &'static str)> {
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

pub(crate) fn artifact(tensors: &BTreeMap<String, (Vec<usize>, &str)>) -> Vec<u8> {
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

pub(crate) fn package(
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
