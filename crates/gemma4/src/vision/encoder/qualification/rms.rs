//! Source-replica RMS diagnostics, not hidden-intermediate tracing or acceptance.
use super::*;

#[path = "rms_math.rs"]
mod math;

const GROUPS: [(&str, &str, Option<VisionNorm>); 7] = [
    ("input_norm", "hidden", Some(VisionNorm::Input)),
    ("q_norm", "q", Some(VisionNorm::Query)),
    ("k_norm", "k", Some(VisionNorm::Key)),
    ("v_norm", "v", None),
    ("post_attention_norm", "o", Some(VisionNorm::PostAttention)),
    (
        "pre_ffn_norm",
        "residual_attention",
        Some(VisionNorm::PreFeedforward),
    ),
    ("post_ffn_norm", "down", Some(VisionNorm::PostFeedforward)),
];

#[derive(Default)]
struct Difference {
    elements: usize,
    differing: usize,
    maximum: f64,
}
impl Difference {
    fn add(&mut self, a: f32, b: f32) {
        assert!(a.is_finite() && b.is_finite());
        self.elements += 1;
        self.differing += usize::from(a.to_bits() != b.to_bits());
        self.maximum = self.maximum.max((f64::from(a) - f64::from(b)).abs());
    }
    fn json(&self) -> Value {
        json!({"elements":self.elements,"differing":self.differing,"maximum_absolute_error":self.maximum})
    }
}
fn words(source: &SafeTensorArtifact, image: usize, name: &str) -> Vec<u16> {
    read_tensor(source, image, name, &ScalarType::Bf16)
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect()
}
fn floats(source: &SafeTensorArtifact, image: usize, name: &str) -> Vec<f32> {
    read_tensor(source, image, name, &ScalarType::F32)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn source_statistics(values: &[u16], epsilon: f32) -> (f32, f32, f32) {
    let mut energy = 0.0_f32;
    for &bits in values {
        let v = load(bits);
        assert!(v.is_finite());
        energy += v * v;
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "validated model width is exactly representable in F32"
    )]
    let mean = energy / values.len() as f32;
    let statistic = mean + epsilon;
    (mean, statistic, statistic.powf(-0.5))
}

struct Audit<'a> {
    encoder: &'a Gemma4CpuVisionEncoder,
    common: &'a SafeTensorArtifact,
    stages: &'a SafeTensorArtifact,
    case: &'a str,
    image: usize,
}
impl Audit<'_> {
    fn working_stages(&self, name: &str, rows: usize, elements: usize) -> [Vec<f32>; 5] {
        let values = ["mean", "statistic", "inverse", "unscaled", "scaled"]
            .map(|label| floats(self.stages, self.image, &format!("{name}.{label}")));
        for (value, length) in values.iter().zip([rows, rows, rows, elements, elements]) {
            assert_eq!(value.len(), length);
        }
        values
    }
    fn operation(&self, group: (&str, &str, Option<VisionNorm>)) -> Value {
        let (name, input, role) = group;
        let c = self.encoder.config();
        let x = words(self.common, self.image, input);
        let expected = words(self.common, self.image, name);
        let weight = role.map(|norm| {
            self.encoder
                .weights
                .get(VisionParameterId::Norm { layer: 0, norm })
                .unwrap()
        });
        let width = weight.map_or(c.head_dim, |w| w.shape()[0]);
        assert_eq!(x.len(), expected.len());
        assert_eq!(x.len() % width, 0);
        let rows = x.len() / width;
        let actual = if let Some(norm) = role {
            self.encoder.norm(&x, rows, 0, norm).unwrap()
        } else {
            rms_norm_scale_free(&x, rows, width, c.epsilon).unwrap()
        };
        let [mean, statistic, inverse, unscaled, scaled] = self.working_stages(name, rows, x.len());
        assert_eq!(actual.len(), x.len());
        let mut differences: [Difference; 7] = std::array::from_fn(|_| Difference::default());
        let mut output_counts = [0_usize; 3];
        let mut equation_counts = [0_usize; 5];
        let mut disagreeing_equation_counts = [0_usize; 5];
        for (row, values) in x.chunks_exact(width).enumerate() {
            let (m, stat, inv) = source_statistics(values, c.epsilon);
            let common_inv = statistic[row].powf(-0.5);
            differences[0].add(m, mean[row]);
            differences[1].add(stat, statistic[row]);
            differences[2].add(inv, inverse[row]);
            differences[5].add(common_inv, inverse[row]);
            assert_eq!(
                (mean[row] + c.epsilon).to_bits(),
                statistic[row].to_bits(),
                "common epsilon addition changed"
            );
            let root = math::denominator(values, c.epsilon);
            for (column, &bits) in values.iter().enumerate() {
                let index = row * width + column;
                let scale = weight.map_or(0x3f80, |w| w.values()[column]);
                assert!(load(scale).is_finite());
                let apply_scale = |v: f32| if weight.is_some() { v * load(scale) } else { v };
                let u = load(bits) * inv;
                let result = apply_scale(u);
                assert_eq!(
                    bf16(result),
                    actual[index],
                    "source replica did not reproduce production BF16"
                );
                assert_eq!(
                    bf16(scaled[index]),
                    expected[index],
                    "HF unrounded output did not reproduce BF16"
                );
                differences[3].add(u, unscaled[index]);
                differences[4].add(result, scaled[index]);
                let on_hf_inverse = apply_scale(load(bits) * inverse[row]);
                differences[6].add(on_hf_inverse, scaled[index]);
                output_counts[0] += usize::from(actual[index] != expected[index]);
                output_counts[1] +=
                    usize::from(bf16(apply_scale(load(bits) * common_inv)) != expected[index]);
                output_counts[2] += usize::from(bf16(on_hf_inverse) != expected[index]);
                let category = match math::equation(bits, scale, root) {
                    None => 4,
                    Some(word) => match (word == actual[index], word == expected[index]) {
                        (true, true) => 0,
                        (true, false) => 1,
                        (false, true) => 2,
                        (false, false) => 3,
                    },
                };
                equation_counts[category] += 1;
                if actual[index] != expected[index] {
                    disagreeing_equation_counts[category] += 1;
                }
            }
        }
        let stages: serde_json::Map<String, Value> = [
            "mean",
            "statistic",
            "inverse",
            "unscaled",
            "scaled",
            "inverse_on_hf_statistic",
            "scaled_on_hf_inverse",
        ]
        .into_iter()
        .zip(differences.iter())
        .map(|(name, d)| (name.into(), d.json()))
        .collect();
        eprintln!(
            "{} image{} {name}: BF16 differences native/HFstat/HFinverse={output_counts:?}; equation both/native/HF/neither/ambiguous={equation_counts:?}",
            self.case, self.image
        );
        json!({"case":self.case,"image":self.image,"operation":name,"rows":rows,"width":width,"elements":x.len(),"epsilon_bits":c.epsilon.to_bits(),"production_and_replica_bf16_exact":true,"f32_source_replica_differences":stages,"bf16_differing":output_counts,"bf16_counterfactual_order":["production","HF_statistic_native_pow","HF_inverse_native_products"],"real_equation_rounding_counts":equation_counts,"disagreeing_real_equation_rounding_counts":disagreeing_equation_counts,"equation_count_order":["both","native_only","HF_only","neither","ambiguous"],"equation_policy":"real equation with one final BF16 round, NOT FP32 working-policy truth/acceptance"})
    }
}

#[test]
#[ignore = "requires pinned common RMS stage corpus and real package; release only"]
#[expect(
    clippy::assertions_on_constants,
    reason = "reject expensive debug execution"
)]
fn real_rms_stage_audit() {
    assert!(!cfg!(debug_assertions));
    let (package, common_root, common_report) = inputs();
    let root = PathBuf::from(std::env::var("RIBN_GEMMA_VISION_RMS").unwrap());
    let report: Value =
        serde_json::from_slice(&std::fs::read(root.join("reference.json")).unwrap()).unwrap();
    assert_eq!(report["complete"], true);
    assert_eq!(report["kind"], "gemma4_vision_rms_stages");
    assert_eq!(report["files"], common_report["files"]);
    assert!(hash_matches(
        File::open(common_root.join("reference.json")).unwrap(),
        report["common_reference_sha256"].as_str().unwrap()
    ));
    let source = Gemma4Source::open(&package)
        .unwrap()
        .vision_source()
        .unwrap();
    let encoder = Gemma4CpuVisionEncoder::new(Gemma4VisionCpuWeights::load(&source).unwrap());
    let cases = report["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 4);
    let mut file = File::options()
        .create_new(true)
        .write(true)
        .open(std::env::var("RIBN_GEMMA_VISION_RMS_REPORT").unwrap())
        .unwrap();
    let mut records = Vec::new();
    checkpoint(&mut file, &records, false);
    for (case, common_case) in cases.iter().zip(common_report["cases"].as_array().unwrap()) {
        assert_eq!(case["name"], common_case["name"]);
        assert_eq!(case["images"], common_case["images"]);
        let name = case["name"].as_str().unwrap();
        let filename = format!("{name}.safetensors");
        assert_eq!(case["archive"]["file"], filename);
        assert!(hash_matches(
            File::open(root.join(&filename)).unwrap(),
            case["archive"]["sha256"].as_str().unwrap()
        ));
        let common = SafeTensorArtifact::open(common_root.join(&filename)).unwrap();
        let stages = SafeTensorArtifact::open(root.join(filename)).unwrap();
        let controls = case["controls"].as_array().unwrap();
        assert_eq!(controls.len(), 7);
        for (control, group) in controls.iter().zip(GROUPS) {
            assert_eq!(control["name"], group.0);
            for field in ["bf16_exact", "fp32_return_exact", "staged_fp32_exact"] {
                assert_eq!(control[field], true);
            }
            assert_eq!(control["epsilon"], 1e-6);
        }
        for image in 0..usize::try_from(case["images"].as_u64().unwrap()).unwrap() {
            let audit = Audit {
                encoder: &encoder,
                common: &common,
                stages: &stages,
                case: name,
                image,
            };
            for group in GROUPS {
                records.push(audit.operation(group));
                checkpoint(&mut file, &records, false);
            }
        }
    }
    assert_eq!(records.len(), 35);
    checkpoint(&mut file, &records, true);
}
