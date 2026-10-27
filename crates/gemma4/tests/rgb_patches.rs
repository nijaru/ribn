//! Stock processor RGB layout/rounding references, independent of encoder numerics.
use engine_gemma4::vision::{Gemma4ImagePatches, Gemma4VisionConfig};
use ribn_foundation::ScalarType;
use ribn_safetensors::SafeTensorArtifact;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{io::Read, path::PathBuf};

fn root() -> Value {
    serde_json::from_str(include_str!(
        "../../../benchmarks/gemma4-reference/config.json"
    ))
    .unwrap()
}
fn golden() -> Value {
    serde_json::from_str(include_str!("fixtures/vision-patches.json")).unwrap()
}
fn small(golden: &Value) -> Gemma4VisionConfig {
    let mut config = root();
    config["vision_config"] = golden["vision_config"].clone();
    Gemma4VisionConfig::from_json(&config).unwrap()
}
fn number(value: &Value, key: &str) -> usize {
    usize::try_from(value[key].as_u64().unwrap()).unwrap()
}
fn rgb(width: usize, height: usize) -> Vec<u8> {
    (0..width * height * 3)
        .map(|i| u8::try_from(i % 256).unwrap())
        .collect()
}

#[test]
fn resized_rgb_matches_stock_patch_order_rescale_and_padding() {
    let golden = golden();
    let config = small(&golden);
    for case in golden["cases"].as_array().unwrap() {
        let (width, height) = (number(case, "width"), number(case, "height"));
        let patches = {
            let input = rgb(width, height);
            Gemma4ImagePatches::from_resized_rgb(&config, width, height, &input, 70).unwrap()
        }; // Output owns its arrays; the caller's RGB storage is no longer retained.
        let image = patches.as_image();
        let expected: Vec<u32> = serde_json::from_value(case["valid_pixel_bits"].clone()).unwrap();
        assert_eq!(
            image.pixels[..expected.len()]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected
        );
        assert!(
            image.pixels[expected.len()..]
                .iter()
                .all(|v| v.to_bits() == 0)
        );
        let positions: Vec<[i64; 2]> =
            serde_json::from_value(case["valid_positions"].clone()).unwrap();
        assert_eq!(&image.positions[..positions.len()], positions);
        assert!(
            image.positions[positions.len()..]
                .iter()
                .all(|&v| v == [-1, -1])
        );
        assert_eq!(image.positions.len(), 70 * 4);
        assert_eq!(patches.soft_tokens(), number(case, "soft_tokens"));
    }
}

#[test]
fn largest_budget_accepts_a_full_grid_without_padding() {
    let config = Gemma4VisionConfig::from_json(&root()).unwrap();
    let side = config.patch_size() * config.pooling_kernel_size();
    let (width, height) = (35 * side, 32 * side);
    let patches = Gemma4ImagePatches::from_resized_rgb(
        &config,
        width,
        height,
        &vec![255; width * height * 3],
        1120,
    )
    .unwrap();
    let image = patches.as_image();
    assert_eq!(patches.soft_tokens(), 1120);
    assert_eq!(image.positions.len(), 1120 * 9);
    assert_eq!(image.positions.first(), Some(&[0, 0]));
    assert_eq!(image.positions.last(), Some(&[104, 95]));
    assert!(image.positions.iter().all(|&[x, y]| x >= 0 && y >= 0));
    assert!(
        image
            .pixels
            .iter()
            .all(|&v| v.to_bits() == 1.0_f32.to_bits())
    );
}

#[test]
fn rejects_bad_rgb_geometry_budgets_and_byte_counts() {
    let config = small(&golden());
    for (width, height, length, budget) in [
        (0, 4, 0, 70),
        (4, 0, 0, 70),
        (5, 4, 60, 70),
        (4, 2, 24, 70),
        (4, 4, 47, 70),
        (4, 4, 49, 70),
        (4, 4, 48, 0),
        (4, 4, 48, 1121),
        (40, 40, 4800, 70),         // budget exceeded
        (20, 20, 1200, 70),         // budget fits, learned coordinate table does not
        (usize::MAX - 3, 4, 0, 70), // aligned dimensions whose RGB byte product overflows
    ] {
        assert!(
            Gemma4ImagePatches::from_resized_rgb(&config, width, height, &vec![0; length], budget)
                .is_err(),
            "{width}x{height}, {length}, {budget}"
        );
    }
}

fn bytes(artifact: &SafeTensorArtifact, name: &str) -> Vec<u8> {
    let mut data = Vec::new();
    artifact
        .tensor(name)
        .unwrap()
        .reader()
        .read_to_end(&mut data)
        .unwrap();
    data
}

#[test]
#[ignore = "requires CPU corpus from benchmarks/gemma4_rgb_reference.py"]
fn full_resized_rgb_reference_corpus() {
    let dir = PathBuf::from(
        std::env::var_os("RIBN_GEMMA_RGB_REFERENCE").expect("CPU RGB reference directory"),
    );
    let report: Value =
        serde_json::from_slice(&std::fs::read(dir.join("report.json")).unwrap()).unwrap();
    assert_eq!(report["complete"], true);
    let cases = report["cases"].as_array().unwrap();
    assert_eq!(
        cases.len(),
        12,
        "two reduced geometries plus both E4B orientations at all five budgets"
    );
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let config: Value = serde_json::from_slice(
            &std::fs::read(dir.join(case["config"].as_str().unwrap())).unwrap(),
        )
        .unwrap();
        let config = Gemma4VisionConfig::from_json(&config).unwrap();
        let path = dir.join(format!("{name}.safetensors"));
        assert_eq!(
            format!("{:x}", Sha256::digest(std::fs::read(&path).unwrap())),
            case["file_sha256"].as_str().unwrap()
        );
        let artifact = SafeTensorArtifact::open(path).unwrap();
        let (width, height) = (number(case, "width"), number(case, "height"));
        assert_eq!(
            artifact.tensor("rgb").unwrap().scalar_type(),
            &ScalarType::U8
        );
        assert_eq!(artifact.tensor("rgb").unwrap().shape(), &[height, width, 3]);
        let patches = Gemma4ImagePatches::from_resized_rgb(
            &config,
            width,
            height,
            &bytes(&artifact, "rgb"),
            number(case, "budget"),
        )
        .unwrap();
        let image = patches.as_image();
        assert_eq!(patches.soft_tokens(), number(case, "soft_tokens"));
        assert_eq!(
            artifact.tensor("pixels").unwrap().scalar_type(),
            &ScalarType::F32
        );
        assert_eq!(
            artifact.tensor("pixels").unwrap().shape(),
            &[
                1,
                image.positions.len(),
                config.patch_size() * config.patch_size() * 3
            ]
        );
        let expected = bytes(&artifact, "pixels");
        assert_eq!(expected.len(), image.pixels.len() * 4);
        for (index, (&actual, expected)) in image
            .pixels
            .iter()
            .zip(expected.as_chunks::<4>().0)
            .enumerate()
        {
            assert_eq!(
                actual.to_bits(),
                u32::from_le_bytes(*expected),
                "{name} pixel {index}"
            );
        }
        assert_eq!(
            artifact.tensor("positions").unwrap().scalar_type(),
            &ScalarType::I64
        );
        assert_eq!(
            artifact.tensor("positions").unwrap().shape(),
            &[1, image.positions.len(), 2]
        );
        let expected = bytes(&artifact, "positions");
        assert_eq!(expected.len(), image.positions.len() * 16);
        for (&actual, expected) in image.positions.iter().zip(expected.as_chunks::<16>().0) {
            assert_eq!(
                actual,
                [
                    i64::from_le_bytes(expected[..8].try_into().unwrap()),
                    i64::from_le_bytes(expected[8..].try_into().unwrap())
                ],
                "{name}"
            );
        }
        println!(
            "{name}: {} FP32 words, {} XY positions exact; {} soft tokens",
            image.pixels.len(),
            image.positions.len(),
            patches.soft_tokens()
        );
    }
}
