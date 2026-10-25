//! Strict full-checkpoint fidelity gate; disagreement is not a fitted tolerance.
use engine_gemma4::Gemma4Source;
use engine_gemma4::vision::{
    Gemma4CpuVisionEncoder, Gemma4VisionCpuWeights, VisionImage, VisionStage,
};
use ribn_hf::LocalModelPackage;
use ribn_safetensors::SafeTensorArtifact;
use serde_json::{Value, json};
use std::{
    fs::File,
    io::{Read, Seek, Write},
    path::PathBuf,
};

fn bytes(artifact: &SafeTensorArtifact, name: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    artifact
        .tensor(name)
        .unwrap()
        .reader()
        .read_to_end(&mut bytes)
        .unwrap();
    bytes
}
fn difference(actual: &[u16], expected: &[u8]) -> Value {
    assert_eq!(actual.len() * 2, expected.len());
    let mut differing = 0;
    let mut maximum = 0.0_f64;
    let mut squared = 0.0_f64;
    for (&actual, expected) in actual.iter().zip(expected.as_chunks::<2>().0) {
        let expected = u16::from_le_bytes(*expected);
        differing += usize::from(actual != expected);
        let expected = f32::from_bits(u32::from(expected) << 16);
        let actual = f32::from_bits(u32::from(actual) << 16);
        assert!(expected.is_finite() && actual.is_finite());
        let error = (f64::from(actual) - f64::from(expected)).abs();
        maximum = maximum.max(error);
        squared += error * error;
    }
    json!({"elements":actual.len(),"differing":differing,"maximum_absolute_error":maximum,"squared_error_sum":squared})
}

fn stage_record(
    oracle: &SafeTensorArtifact,
    case: &str,
    image: usize,
    feature_offset: usize,
    stage: VisionStage,
    values: &[u16],
) -> Value {
    let (name, offset) = match stage {
        VisionStage::PatchEmbedding => ("vision.patch_embeddings".into(), image * 2520 * 768),
        VisionStage::Layer(layer) => (format!("vision.layer.{layer}"), image * 2520 * 768),
        VisionStage::Pooled => ("vision.valid".into(), feature_offset * 768),
        VisionStage::ProjectionNorm => ("vision.projection_norm".into(), feature_offset * 768),
        VisionStage::Projected => ("vision.projected".into(), feature_offset * 2560),
    };
    let raw = bytes(oracle, &name);
    assert_eq!(
        oracle.tensor(&name).unwrap().scalar_type(),
        &ribn_foundation::ScalarType::Bf16
    );
    let mut record = difference(values, &raw[offset * 2..(offset + values.len()) * 2]);
    record["case"] = json!(case);
    record["image"] = json!(image);
    record["stage"] = json!(name);
    eprintln!(
        "{case} image{image} {stage:?}: {}/{} differ, max={}",
        record["differing"],
        values.len(),
        record["maximum_absolute_error"]
    );

    record
}

fn checkpoint(output: &mut File, records: &[Value], complete: bool) -> bool {
    let exact = records.iter().all(|r| r["differing"] == 0);
    let data =
        serde_json::to_vec_pretty(&json!({"complete":complete,"exact":exact,"records":records}))
            .unwrap();
    output.rewind().unwrap();
    output.write_all(&data).unwrap();
    output.set_len(u64::try_from(data.len()).unwrap()).unwrap();
    output.flush().unwrap();
    exact
}

#[test]
#[ignore = "requires the full local package and frozen vision CPU corpus; release only"]
#[expect(
    clippy::assertions_on_constants,
    reason = "reject expensive debug execution without making debug all-target compilation fail"
)]
fn real_vision_corpus_exact_gate() {
    assert!(
        !cfg!(debug_assertions),
        "full vision corpus must run in release mode"
    );
    let package = LocalModelPackage::open(std::env::var("RIBN_GEMMA_PACKAGE").unwrap()).unwrap();
    let reference = PathBuf::from(std::env::var("RIBN_GEMMA_VISION_REFERENCE").unwrap());
    let output = PathBuf::from(std::env::var("RIBN_GEMMA_VISION_REPORT").unwrap());
    let mut output = File::options()
        .write(true)
        .create_new(true)
        .open(output)
        .expect("choose a fresh vision report path");
    let mut records = Vec::new();
    checkpoint(&mut output, &records, false);
    let source = Gemma4Source::open(&package).unwrap();
    assert_eq!(source.parameters().len(), 665);
    assert_eq!(
        source
            .excluded()
            .iter()
            .filter(|p| p.reason == engine_gemma4::Exclusion::NonText)
            .count(),
        1411
    );
    let vision = source.vision_source().unwrap();
    assert_eq!(
        (vision.parameters().len(), vision.encoded_bytes()),
        (659, 338_662_272)
    );
    let encoder = Gemma4CpuVisionEncoder::new(Gemma4VisionCpuWeights::load(&vision).unwrap());
    let inputs =
        SafeTensorArtifact::open(reference.join("prepared-v1/inputs.safetensors")).unwrap();
    for case in ["red", "blue", "landscape", "two_images"] {
        let oracle =
            SafeTensorArtifact::open(reference.join(format!("bf16/capture/{case}.safetensors")))
                .unwrap();
        let pixels = bytes(&inputs, &format!("{case}.pixel_values"));
        let pixels: Vec<_> = pixels
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let positions = bytes(&inputs, &format!("{case}.image_position_ids"));
        let positions: Vec<_> = positions
            .as_chunks::<16>()
            .0
            .iter()
            .map(|b| {
                [
                    i64::from_le_bytes(b[..8].try_into().unwrap()),
                    i64::from_le_bytes(b[8..].try_into().unwrap()),
                ]
            })
            .collect();
        assert_eq!(positions.len() % 2520, 0);
        assert_eq!(pixels.len(), positions.len() * 768);
        let mut feature_offset = 0;
        for image in 0..positions.len() / 2520 {
            let pos = &positions[image * 2520..(image + 1) * 2520];
            let valid = pos.iter().take_while(|&&p| p != [-1, -1]).count() / 9;
            encoder
                .encode_observed(
                    VisionImage {
                        pixels: &pixels[image * 2520 * 768..(image + 1) * 2520 * 768],
                        positions: pos,
                    },
                    |stage, values| {
                        let record =
                            stage_record(&oracle, case, image, feature_offset, stage, values);
                        records.push(record);
                        checkpoint(&mut output, &records, false);
                        Ok(())
                    },
                )
                .unwrap();
            feature_offset += valid;
        }
    }
    let exact = checkpoint(&mut output, &records, true);
    assert!(
        exact,
        "strict full-checkpoint vision gate failed; report preserves every case and stage"
    );
}
