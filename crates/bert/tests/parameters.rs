//! Host-side checks that the fixture package maps onto this encoder's parameters.

use std::path::PathBuf;

use engine_bert::{BertConfig, ConfigError, HostWeights, WeightError};
use ribn_hf::LocalModelPackage;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../batch/tests/fixtures/bert-tiny")
}

#[test]
fn fixture_package_supplies_every_parameter() {
    let package = LocalModelPackage::open(fixture()).expect("fixture package");
    let config = BertConfig::from_json(package.config()).expect("fixture config");
    let weights = HostWeights::load(&package, &config).expect("fixture parameters");

    assert_eq!(weights.word_embeddings.len(), 13 * 12);
    assert_eq!(weights.position_embeddings.len(), 16 * 12);
    assert_eq!(weights.token_type_embeddings.len(), 3 * 12);
    assert_eq!(weights.layers.len(), 2);
    assert_eq!(weights.embedding_norm.weight.len(), 12);
    assert_eq!(weights.pooler.weight.len(), 12 * 12);
    for layer in &weights.layers {
        assert_eq!(layer.query.output, 12);
        assert_eq!(layer.query.offset(11, 11), 143);
        assert_eq!(layer.intermediate.input, 12);
        assert_eq!(layer.intermediate.output, 20);
        assert_eq!(layer.output.input, 20);
        assert_eq!(layer.output.output, 12);
    }
    assert!(weights.word_embeddings.iter().any(|value| *value != 0.0));
}

#[test]
fn mutable_host_parameters_must_match_the_complete_model_geometry() {
    type Mutation = fn(&mut HostWeights);

    let package = LocalModelPackage::open(fixture()).unwrap();
    let config = BertConfig::from_json(package.config()).unwrap();
    let weights = HostWeights::load(&package, &config).unwrap();
    weights.validate(&config).unwrap();

    let mutations: &[(&str, Mutation)] = &[
        ("word embeddings", |w| {
            w.word_embeddings.pop();
        }),
        ("position embeddings", |w| {
            w.position_embeddings.pop();
        }),
        ("type embeddings", |w| {
            w.token_type_embeddings.pop();
        }),
        ("embedding norm", |w| {
            w.embedding_norm.bias.pop();
        }),
        ("layer count", |w| {
            w.layers.pop();
        }),
        ("projection buffer", |w| {
            w.layers[0].query.weight.truncate(1);
        }),
        ("projection dimensions", |w| {
            w.layers[0].key.input += 1;
        }),
        ("projection bias", |w| {
            w.layers[0].value.bias.pop();
        }),
        ("attention output", |w| {
            w.layers[0].attention_output.output += 1;
        }),
        ("attention norm", |w| {
            w.layers[0].attention_norm.weight.pop();
        }),
        ("intermediate", |w| {
            w.layers[0].intermediate.input += 1;
        }),
        ("layer output", |w| {
            w.layers[0].output.weight.pop();
        }),
        ("output norm", |w| {
            w.layers[0].output_norm.bias.pop();
        }),
        ("pooler", |w| {
            w.pooler.output += 1;
        }),
    ];
    for (name, mutate) in mutations {
        let mut invalid = weights.clone();
        mutate(&mut invalid);
        assert!(
            invalid.validate(&config).is_err(),
            "accepted invalid {name}"
        );
    }
    let overflow = BertConfig {
        vocab_size: usize::MAX,
        ..config
    };
    assert!(matches!(
        weights.validate(&overflow),
        Err(WeightError::SizeOverflow { .. })
    ));
    let invalid = BertConfig {
        hidden_size: 0,
        ..config
    };
    assert!(matches!(
        weights.validate(&invalid),
        Err(WeightError::Config(_))
    ));
}

#[test]
fn sharded_package_preserves_every_decoded_parameter() {
    struct Cleanup(PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let package = LocalModelPackage::open(fixture()).unwrap();
    let mut config = BertConfig::from_json(package.config()).unwrap();
    let mut expected = HostWeights::load(&package, &config).unwrap();
    // Exercise a complete 16-KiB decoder buffer and a partial tail, unlike the
    // tiny fixture. Preserve signed zero and a subnormal's exact encoded bits.
    config.vocab_size = 343;
    expected.word_embeddings = (0..config.vocab_size * config.hidden_size)
        .map(|i| f32::from_bits(0x3f80_0000 + u32::try_from(i).unwrap()))
        .collect();
    expected.word_embeddings[0] = -0.0;
    expected.word_embeddings[1] = f32::from_bits(1);
    let artifact = package
        .open_weights_for("embeddings.word_embeddings.weight")
        .unwrap();
    let root = std::env::temp_dir().join(format!(
        "ribn-bert-sharded-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root).unwrap();
    let _cleanup = Cleanup(root.clone());
    let mut config_json = package.config().clone();
    config_json["vocab_size"] = serde_json::json!(config.vocab_size);
    std::fs::write(root.join("config.json"), config_json.to_string()).unwrap();
    let mut shards: [(serde_json::Map<String, serde_json::Value>, Vec<u8>); 3] =
        std::array::from_fn(|_| (serde_json::Map::new(), Vec::new()));
    let mut weight_map = serde_json::Map::new();
    // Interleave parameters rather than assuming that model read order matches
    // shard order. Evicted artifacts must be reopened without losing values.
    for (index, name) in artifact.names().into_iter().enumerate() {
        let shard = index % shards.len();
        let tensor = artifact.tensor(&name).unwrap();
        let (header, payload) = &mut shards[shard];
        let start = payload.len();
        let shape = if name == "embeddings.word_embeddings.weight" {
            payload.extend(
                expected
                    .word_embeddings
                    .iter()
                    .flat_map(|v| v.to_le_bytes()),
            );
            vec![config.vocab_size, config.hidden_size]
        } else {
            std::io::Read::read_to_end(&mut tensor.reader(), payload).unwrap();
            tensor.shape().to_vec()
        };
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, payload.len()]}),
        );
        weight_map.insert(
            name,
            serde_json::json!(format!("model-{shard}.safetensors")),
        );
    }
    for (index, (header, payload)) in shards.into_iter().enumerate() {
        let mut header = serde_json::to_vec(&header).unwrap();
        header.resize(header.len().next_multiple_of(8), b' ');
        let mut bytes = u64::try_from(header.len()).unwrap().to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(payload);
        std::fs::write(root.join(format!("model-{index}.safetensors")), bytes).unwrap();
    }
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({"weight_map": weight_map}).to_string(),
    )
    .unwrap();
    let sharded = LocalModelPackage::open(root).unwrap();
    let actual = HostWeights::load(&sharded, &config).unwrap();
    assert_eq!(
        actual
            .word_embeddings
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        expected
            .word_embeddings
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(actual, expected);
}

#[test]
fn unsupported_architecture_is_named_rather_than_inferred() {
    let package = LocalModelPackage::open(fixture()).expect("fixture package");
    let mut config = package.config().clone();
    config["model_type"] = serde_json::json!("roberta");
    assert!(matches!(
        BertConfig::from_json(&config).unwrap_err(),
        ConfigError::UnsupportedModelType(_)
    ));
}
