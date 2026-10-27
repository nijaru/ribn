//! Independent stock-HF tiny vision fixture plus retained-source failure cases.
use engine_gemma4::vision::{
    Gemma4CpuVisionEncoder, Gemma4ImagePatches, Gemma4VisionConfig, Gemma4VisionCpuWeights,
    VisionImage,
};
use engine_gemma4::{Gemma4Source, bf16};
use ribn_hf::LocalModelPackage;
use serde_json::{Value, json};
#[path = "common/weights.rs"]
#[expect(
    dead_code,
    reason = "shared text fixture also provides artifact writers unused by this vision fixture"
)]
mod weights;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/vision.json")).unwrap()
}
fn package(dir: &weights::Dir, fixture: &Value) -> LocalModelPackage {
    let mut config = weights::tiny();
    config["vision_config"] = fixture["vision_config"].clone();
    std::fs::write(
        dir.0.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let mut tensors = serde_json::Map::new();
    for (name, (shape, _)) in weights::inventory() {
        let bits = vec![0x3f80_u16; shape.iter().product()];
        tensors.insert(name, json!({"shape":shape,"bits":bits}));
    }
    tensors.extend(fixture["weights"].as_object().unwrap().clone());
    let mut index = serde_json::Map::new();
    // A media-only shard has no text reader keeping it alive. Selection after
    // unlink therefore specifically proves retention of the unused inventory.
    for shard in ["text.safetensors", "vision.safetensors"] {
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for (name, tensor) in &tensors {
            if name.starts_with("model.language_model.") != (shard == "text.safetensors") {
                continue;
            }
            let start = payload.len();
            for bits in tensor["bits"].as_array().unwrap() {
                payload.extend(u16::try_from(bits.as_u64().unwrap()).unwrap().to_le_bytes());
            }
            index.insert(name.clone(), json!(shard));
            header.insert(name.clone(), json!({"dtype":tensor.get("dtype").unwrap_or(&json!("BF16")),"shape":tensor["shape"],"data_offsets":[start,payload.len()]}));
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = u64::try_from(header.len()).unwrap().to_le_bytes().to_vec();
        file.extend(header);
        file.extend(payload);
        std::fs::write(dir.0.join(shard), file).unwrap();
    }
    std::fs::write(
        dir.0.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({"weight_map":index})).unwrap(),
    )
    .unwrap();
    LocalModelPackage::open(&dir.0).unwrap()
}
fn image_data(f: &Value) -> (Vec<f32>, Vec<[i64; 2]>) {
    (
        serde_json::from_value(f["pixels"].clone()).unwrap(),
        serde_json::from_value(f["positions"].clone()).unwrap(),
    )
}

#[test]
fn cpu_encoder_matches_independent_stock_fixture_at_every_stage() {
    let dir = weights::Dir::new();
    let f = fixture();
    let source = Gemma4Source::open(&package(&dir, &f)).unwrap();
    assert_eq!(source.parameters().len(), 67);
    let vision = source.vision_source().unwrap();
    assert_eq!(vision.parameters().len(), 85);
    assert_eq!(
        vision
            .config()
            .specifications()
            .map(|s| s.byte_len().unwrap())
            .sum::<usize>(),
        vision.encoded_bytes()
    );
    let weights = Gemma4VisionCpuWeights::load(&vision).unwrap();
    assert_eq!(weights.bytes(), vision.encoded_bytes());
    let encoder = Gemma4CpuVisionEncoder::new(weights);
    let (pixels, positions) = image_data(&f);
    let mut seen = Vec::new();
    let result = encoder
        .encode_observed(
            VisionImage {
                pixels: &pixels,
                positions: &positions,
            },
            |stage, values| {
                let name = format!("{stage:?}");
                let expected: Vec<u16> =
                    serde_json::from_value(f["expected"][&name].clone()).unwrap();
                assert_eq!(values, expected, "{name}");
                seen.push(name);
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(seen.len(), f["expected"].as_object().unwrap().len());
    assert_eq!(result.len(), 7);
}

#[test]
fn resized_rgb_owner_feeds_existing_cpu_encoder() {
    let dir = weights::Dir::new();
    let source = Gemma4Source::open(&package(&dir, &fixture())).unwrap();
    let vision = source.vision_source().unwrap();
    let encoder = Gemma4CpuVisionEncoder::new(Gemma4VisionCpuWeights::load(&vision).unwrap());
    let patches = Gemma4ImagePatches::from_resized_rgb(
        encoder.config(),
        8,
        4,
        &(0_u8..96).collect::<Vec<_>>(),
        70,
    )
    .unwrap();
    let result = encoder.encode(patches.as_image()).unwrap();
    assert_eq!(
        result.len(),
        patches.soft_tokens() * encoder.config().output_width()
    );
    assert!(
        result
            .iter()
            .all(|&v| f32::from_bits(u32::from(v) << 16).is_finite())
    );
    // This checks the real producer/consumer geometry and ownership boundary;
    // the existing stock fixture, not these new pixels, owns encoder parity.
}

#[test]
fn vision_selection_is_opt_in_complete_and_uses_retained_files() {
    let dir = weights::Dir::new();
    let f = fixture();
    let p = package(&dir, &f);
    let source = Gemma4Source::open(&p).unwrap();
    std::fs::remove_file(dir.0.join("text.safetensors")).unwrap();
    std::fs::remove_file(dir.0.join("vision.safetensors")).unwrap();
    std::fs::remove_file(dir.0.join("config.json")).unwrap();
    drop(p);
    let vision = source.vision_source().unwrap();
    drop(source);
    let weights = Gemma4VisionCpuWeights::load(&vision).unwrap();
    drop(vision);
    let (pixels, positions) = image_data(&f);
    let output = Gemma4CpuVisionEncoder::new(weights)
        .encode(VisionImage {
            pixels: &pixels,
            positions: &positions,
        })
        .unwrap();
    assert_eq!(
        output,
        serde_json::from_value::<Vec<u16>>(f["expected"]["Projected"].clone()).unwrap()
    );

    let dir = weights::Dir::new();
    let mut f = fixture();
    f["vision_config"]["hidden_activation"] = json!("unsupported");
    let source = Gemma4Source::open(&package(&dir, &f)).unwrap();
    assert_eq!(source.parameters().len(), 67);
    assert!(source.vision_source().is_err());
}

#[test]
fn vision_rejects_missing_wrong_rank_dtype_unknown_weights_and_bad_bounds() {
    let clip = "model.vision_tower.encoder.layers.0.self_attn.q_proj.input_min";
    for mode in 0..6 {
        let dir = weights::Dir::new();
        let mut f = fixture();
        match mode {
            0 => {
                f["weights"].as_object_mut().unwrap().remove(clip);
            }
            1 => f["weights"][clip]["shape"] = json!([1]),
            2 => f["weights"][clip]["dtype"] = json!("F16"),
            3 => f["weights"]["model.vision_tower.unknown"] = json!({"shape":[1],"bits":[0]}),
            4 => f["weights"][clip]["bits"] = json!([bf16(f32::NAN)]),
            _ => f["weights"][clip]["bits"] = json!([bf16(1.0)]),
        }
        let source = Gemma4Source::open(&package(&dir, &f)).unwrap();
        let selected = source.vision_source();
        if mode < 4 {
            assert!(selected.is_err(), "mode {mode}");
        } else {
            assert!(
                Gemma4VisionCpuWeights::load(&selected.unwrap()).is_err(),
                "mode {mode}"
            );
        }
    }
    // Infinite bounds are the producer's legitimate no-clipping encoding.
    let dir = weights::Dir::new();
    let mut f = fixture();
    for (name, tensor) in f["weights"].as_object_mut().unwrap() {
        if name.ends_with("_min") {
            tensor["bits"] = json!([bf16(f32::NEG_INFINITY)]);
        }
        if name.ends_with("_max") {
            tensor["bits"] = json!([bf16(f32::INFINITY)]);
        }
    }
    let encode = |f: &Value| {
        let weights = Gemma4VisionCpuWeights::load(
            &Gemma4Source::open(&package(&dir, f))
                .unwrap()
                .vision_source()
                .unwrap(),
        )
        .unwrap();
        let (pixels, positions) = image_data(f);
        Gemma4CpuVisionEncoder::new(weights)
            .encode(VisionImage {
                pixels: &pixels,
                positions: &positions,
            })
            .unwrap()
    };
    let unclipped = encode(&f);
    f["vision_config"]["use_clipped_linears"] = json!(false);
    f["weights"]
        .as_object_mut()
        .unwrap()
        .retain(|name, _| !name.ends_with("_min") && !name.ends_with("_max"));
    assert_eq!(encode(&f), unclipped);
}

#[test]
fn config_and_canonical_input_fail_before_observation() {
    let dir = weights::Dir::new();
    let f = fixture();
    let p = package(&dir, &f);
    for (key, value) in [
        ("head_dim", json!(6)),
        ("num_key_value_heads", json!(3)),
        ("standardize", json!(true)),
        ("attention_dropout", json!(0.1)),
        ("pooling_kernel_size", json!(1)),
        ("rms_norm_eps", json!(0.0)),
        ("patch_size", json!(u64::MAX)),
        ("new_algorithm", json!(true)),
    ] {
        let mut root = p.config().clone();
        root["vision_config"][key] = value;
        assert!(Gemma4VisionConfig::from_json(&root).is_err(), "{key}");
    }
    let source = Gemma4Source::open(&p).unwrap();
    let encoder = Gemma4CpuVisionEncoder::new(
        Gemma4VisionCpuWeights::load(&source.vision_source().unwrap()).unwrap(),
    );
    for mode in 0..6 {
        let (mut pixels, mut positions) = image_data(&f);
        match mode {
            0 => pixels[0] = f32::NAN,
            1 => pixels[12] = 0.1,
            2 => {
                pixels.pop();
            }
            3 => positions.swap(1, 2),
            4 => positions[0] = [8, 0],
            _ => positions[4] = [-1, 0],
        }
        assert!(
            encoder
                .encode_observed(
                    VisionImage {
                        pixels: &pixels,
                        positions: &positions
                    },
                    |_, _| panic!("invalid input reached observer")
                )
                .is_err(),
            "mode {mode}"
        );
    }
}
