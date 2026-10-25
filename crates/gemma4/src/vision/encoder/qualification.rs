//! Ignored real-operand diagnostics, using production private operations unchanged.
use super::*;
use crate::Gemma4Source;
use ribn_foundation::ScalarType;
use ribn_hf::LocalModelPackage;
use ribn_safetensors::SafeTensorArtifact;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{Read, Seek, Write},
    path::PathBuf,
};

#[path = "qualification/rms.rs"]
mod rms;

fn read_tensor(
    source: &SafeTensorArtifact,
    image: usize,
    name: &str,
    kind: &ScalarType,
) -> Vec<u8> {
    let tensor = source.tensor(name).unwrap();
    assert_eq!(tensor.scalar_type(), kind);
    let batch = tensor.shape()[0];
    assert!(image < batch);
    let mut bytes = Vec::new();
    tensor.reader().read_to_end(&mut bytes).unwrap();
    let size = bytes.len() / batch;
    bytes[image * size..(image + 1) * size].to_vec()
}

struct Probe<'a> {
    source: &'a SafeTensorArtifact,
    case: &'a str,
    image: usize,
    records: &'a mut Vec<Value>,
}
impl Probe<'_> {
    fn words(&self, name: &str) -> Vec<u16> {
        read_tensor(self.source, self.image, name, &ScalarType::Bf16)
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_le_bytes(*b))
            .collect()
    }
    fn positions(&self) -> Vec<[i64; 2]> {
        read_tensor(self.source, self.image, "positions", &ScalarType::I64)
            .as_chunks::<16>()
            .0
            .iter()
            .map(|b| {
                [
                    i64::from_le_bytes(b[..8].try_into().unwrap()),
                    i64::from_le_bytes(b[8..].try_into().unwrap()),
                ]
            })
            .collect()
    }
    fn compare(&mut self, operation: &str, actual: &[u16], expected: &str, exact: bool) {
        let expected = self.words(expected);
        assert_eq!(actual.len(), expected.len());
        let mut differing = 0;
        let mut maximum = 0.0_f64;
        let mut first = None;
        for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
            assert!(load(a).is_finite() && load(b).is_finite());
            maximum = maximum.max((f64::from(load(a)) - f64::from(load(b))).abs());
            if a != b {
                differing += 1;
                first.get_or_insert(json!({"index":i,"actual":a,"reference":b}));
            }
        }
        eprintln!(
            "{} image{} {operation}: {differing}/{} differ, max={maximum}",
            self.case,
            self.image,
            actual.len()
        );
        self.records.push(json!({"case":self.case,"image":self.image,"operation":operation,"elements":actual.len(),"differing":differing,"maximum_absolute_error":maximum,"first":first,"requires_exact":exact}));
    }
    fn norms(&mut self, encoder: &Gemma4CpuVisionEncoder) {
        let c = encoder.config();
        for (norm, input, expected, groups) in [
            (VisionNorm::Input, "hidden", "input_norm", 1),
            (VisionNorm::Query, "q", "q_norm", c.heads),
            (VisionNorm::Key, "k", "k_norm", c.kv_heads),
            (VisionNorm::PostAttention, "o", "post_attention_norm", 1),
            (
                VisionNorm::PreFeedforward,
                "residual_attention",
                "pre_ffn_norm",
                1,
            ),
            (VisionNorm::PostFeedforward, "down", "post_ffn_norm", 1),
        ] {
            let output = encoder
                .norm(&self.words(input), 2520 * groups, 0, norm)
                .unwrap();
            self.compare(expected, &output, expected, false);
        }
        let output =
            rms_norm_scale_free(&self.words("v"), 2520 * c.kv_heads, c.head_dim, c.epsilon)
                .unwrap();
        self.compare("v_norm", &output, "v_norm", false);
    }
    fn linears(&mut self, encoder: &Gemma4CpuVisionEncoder) {
        for (linear, input, name) in [
            (VisionLinear::Query, "input_norm", "q"),
            (VisionLinear::Key, "input_norm", "k"),
            (VisionLinear::Value, "input_norm", "v"),
            (VisionLinear::AttentionOutput, "attention", "o"),
            (VisionLinear::Gate, "pre_ffn_norm", "gate"),
            (VisionLinear::Up, "pre_ffn_norm", "up"),
            (VisionLinear::Down, "gated", "down"),
        ] {
            let output = encoder.linear(&self.words(input), 2520, 0, linear).unwrap();
            self.compare(name, &output, name, false);
            let weight = encoder
                .weights
                .get(VisionParameterId::Linear {
                    layer: 0,
                    linear,
                    part: VisionLinearPart::Weight,
                })
                .unwrap();
            let output =
                project(&self.words(&format!("{name}.linear_input")), 2520, weight).unwrap();
            let expected = format!("{name}.linear_output");
            self.compare(&expected, &output, &expected, false);
        }
    }
    fn remaining(&mut self, encoder: &Gemma4CpuVisionEncoder) {
        let c = encoder.config();
        let positions = self.positions();
        let grid = Grid::positions(&positions, c).unwrap();
        assert_eq!(grid.rows, 2520);
        for (name, input, heads) in [
            ("q_rope", "q_norm", c.heads),
            ("k_rope", "k_norm", c.kv_heads),
        ] {
            let output = axial(&self.words(input), &positions, heads, c).unwrap();
            self.compare(name, &output, name, false);
        }
        let output = vision_attention(
            &self.words("q_rope"),
            &self.words("k_rope"),
            &self.words("v_norm"),
            AttentionConfig {
                heads: c.heads,
                kv_heads: c.kv_heads,
                width: c.head_dim,
                window: None,
            },
            grid.valid,
        )
        .unwrap();
        self.compare("attention", &output, "attention", false);
        let output = gelu_tanh(&self.words("gate")).unwrap();
        self.compare("gelu", &output, "gelu", false);
        let output = multiply(&self.words("gelu"), &self.words("up")).unwrap();
        self.compare("gated", &output, "gated", true);
        for (name, left, right, expected) in [
            (
                "residual_attention",
                "hidden",
                "post_attention_norm",
                "residual_attention",
            ),
            (
                "residual_ffn",
                "residual_attention",
                "post_ffn_norm",
                "output",
            ),
        ] {
            let output = add(&self.words(left), &self.words(right)).unwrap();
            self.compare(name, &output, expected, true);
        }
    }
}
fn checkpoint(file: &mut File, records: &[Value], complete: bool) {
    let data = serde_json::to_vec_pretty(&json!({"complete":complete,"scope":"common-operand diagnostics, not full-model acceptance","records":records})).unwrap();
    file.rewind().unwrap();
    file.write_all(&data).unwrap();
    file.set_len(u64::try_from(data.len()).unwrap()).unwrap();
    file.flush().unwrap();
}

fn hash_matches(mut reader: impl Read, expected: &str) -> bool {
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        let count = reader.read(&mut buffer).unwrap();
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    format!("{:x}", digest.finalize()) == expected
}

fn inputs() -> (LocalModelPackage, PathBuf, Value) {
    let package_path = PathBuf::from(std::env::var("RIBN_GEMMA_PACKAGE").unwrap());
    let corpus = PathBuf::from(std::env::var("RIBN_GEMMA_VISION_COMPONENTS").unwrap());
    let report: Value =
        serde_json::from_slice(&std::fs::read(corpus.join("reference.json")).unwrap()).unwrap();
    assert_eq!(report["complete"], true);
    assert_eq!(report["kind"], "gemma4_vision_layer0_components");
    let files = report["files"].as_object().unwrap();
    assert!(files.contains_key("model.safetensors") && files.contains_key("config.json"));
    for (name, identity) in files {
        assert_eq!(std::path::Path::new(name).components().count(), 1);
        let path = package_path.join(name);
        assert!(
            hash_matches(
                File::open(path).unwrap(),
                identity["sha256"].as_str().unwrap()
            ),
            "checkpoint identity changed: {name}"
        );
    }
    let cases = report["cases"].as_array().unwrap();
    assert_eq!(
        cases
            .iter()
            .map(|c| c["name"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["red", "blue", "landscape", "two_images"]
    );
    for case in cases {
        assert_eq!(case["layer0_exact"], true);
        let name = case["name"].as_str().unwrap();
        let file = case["archive"]["file"].as_str().unwrap();
        assert_eq!(file, format!("{name}.safetensors"));
        assert!(
            hash_matches(
                File::open(corpus.join(file)).unwrap(),
                case["archive"]["sha256"].as_str().unwrap()
            ),
            "component archive changed: {name}"
        );
    }
    (
        LocalModelPackage::open(package_path).unwrap(),
        corpus,
        report,
    )
}

#[test]
fn identity_guard_reads_past_the_first_chunk_and_rejects_same_size_mutation() {
    let mut bytes = vec![0_u8; 65537];
    let expected = format!("{:x}", Sha256::digest(&bytes));
    assert!(hash_matches(bytes.as_slice(), &expected));
    bytes[65536] = 1;
    assert!(!hash_matches(bytes.as_slice(), &expected));
}

#[test]
#[ignore = "requires the real package and component corpus, hashes only"]
fn component_input_identities() {
    let _ = inputs();
}

#[test]
#[ignore = "requires frozen real vision layer0 component corpus; release only"]
#[expect(
    clippy::assertions_on_constants,
    reason = "reject expensive debug execution but permit debug all-target compilation"
)]
fn real_common_operand_layer0() {
    assert!(!cfg!(debug_assertions), "release only");
    let (package, corpus, report) = inputs();
    let mut file = File::options()
        .create_new(true)
        .write(true)
        .open(std::env::var("RIBN_GEMMA_VISION_COMPONENT_REPORT").unwrap())
        .unwrap();
    let mut records = Vec::new();
    checkpoint(&mut file, &records, false);
    let source = Gemma4Source::open(&package)
        .unwrap()
        .vision_source()
        .unwrap();
    let encoder = Gemma4CpuVisionEncoder::new(Gemma4VisionCpuWeights::load(&source).unwrap());
    let cases = report["cases"].as_array().unwrap();
    for case in cases {
        assert_eq!(case["layer0_exact"], true);
        let name = case["name"].as_str().unwrap();
        let artifact =
            SafeTensorArtifact::open(corpus.join(case["archive"]["file"].as_str().unwrap()))
                .unwrap();
        let images = usize::try_from(case["images"].as_u64().unwrap()).unwrap();
        assert_eq!(images, if name == "two_images" { 2 } else { 1 });
        for image in 0..images {
            let mut probe = Probe {
                source: &artifact,
                case: name,
                image,
                records: &mut records,
            };
            probe.norms(&encoder);
            probe.linears(&encoder);
            probe.remaining(&encoder);
            checkpoint(&mut file, &records, false);
        }
    }
    checkpoint(&mut file, &records, true);
    assert_eq!(records.len(), 140);
    assert!(
        records
            .iter()
            .all(|r| r["requires_exact"] != true || r["differing"] == 0),
        "arithmetic-preserving common-operand gate failed"
    );
}
