//! Executable chunking oracle: full-history CPU caches, not device eviction evidence.

use engine_gemma4::{Gemma4CpuDecoder, Gemma4CpuWeights, Gemma4Source, bf16};
use ribn_hf::LocalModelPackage;
use serde_json::Value;
use std::sync::Arc;

#[path = "common/weights.rs"]
mod weights;
use weights::{Dir, inventory, package, tiny};

fn model(window: usize) -> Arc<Gemma4CpuWeights> {
    let dir = Dir::new();
    let package = package(&dir, &inventory(), false);
    let mut config = tiny();
    config["text_config"]["sliding_window"] = window.into();
    std::fs::write(
        dir.0.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();

    // Preserve the independent small schema, but replace constant tensor payloads:
    // distinct tokens, channels and layers must exercise real attention dependencies.
    let path = dir.0.join("model.safetensors");
    let mut artifact = std::fs::read(&path).unwrap();
    let length = usize::try_from(u64::from_le_bytes(artifact[..8].try_into().unwrap())).unwrap();
    let header: Value = serde_json::from_slice(&artifact[8..8 + length]).unwrap();
    let data = &mut artifact[8 + length..];
    for (name, tensor) in header.as_object().unwrap() {
        assert_eq!(tensor["dtype"], "BF16");
        let offsets = tensor["data_offsets"].as_array().unwrap();
        let start = usize::try_from(offsets[0].as_u64().unwrap()).unwrap();
        let end = usize::try_from(offsets[1].as_u64().unwrap()).unwrap();
        let scale = tensor["shape"].as_array().unwrap().len() == 1;
        let mut state = name.bytes().fold(17_u32, |seed, byte| {
            seed.wrapping_mul(31).wrapping_add(u32::from(byte))
        });
        for bytes in data[start..end].as_chunks_mut::<2>().0 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let signed = i16::try_from((state >> 24) % 17).unwrap() - 8;
            let value = f32::from(signed) / 16.0;
            let value = if scale { 1.0 + value / 4.0 } else { value };
            bytes.copy_from_slice(&bf16(value).to_le_bytes());
        }
    }
    std::fs::write(path, artifact).unwrap();
    // A package retains its parsed config; reopen after changing the window.
    let package = LocalModelPackage::open(package.root()).unwrap();
    let weights = Gemma4CpuWeights::load(&Gemma4Source::open(&package).unwrap()).unwrap();
    assert_eq!(weights.config().sliding_window(), window);
    Arc::new(weights)
}

fn assert_logits(actual: &[u16], expected: &[u16], context: &str) {
    assert_eq!(actual.len(), 7);
    assert!(
        actual
            .iter()
            .all(|&bits| f32::from_bits(u32::from(bits) << 16).is_finite()),
        "{context}: finite logits"
    );
    assert_eq!(actual, expected, "{context}");
}

#[test]
fn multirow_prefill_matches_serial_across_window_and_shared_tail_boundaries() {
    // Owners 0/1 have different widths and serve shared layers 2/3. Keep the
    // reduced geometry while checking both a small window and E4B's 512 positions.
    for window in [4, 512] {
        let weights = model(window);
        let mut serial = Gemma4CpuDecoder::shared(weights.clone()).unwrap();
        let mut chunked = Gemma4CpuDecoder::shared(weights.clone()).unwrap();

        // Equal last tokens with different prefixes must not produce a degenerate
        // fixture that could pass even if attention ignored continuation state.
        let left = serial.prefill(&[0, 3, 1]).unwrap();
        let right = chunked.prefill(&[6, 5, 1]).unwrap();
        assert_ne!(left, right, "fixture must distinguish prior histories");
        assert!(left.windows(2).any(|pair| pair[0] != pair[1]));
        serial.reset();
        chunked.reset();

        let prompt: Vec<_> = [0, 3, 1, 6, 2, 5, 4]
            .into_iter()
            .cycle()
            .take(window + 5)
            .collect();
        let continuation = [6, 0, 2];
        let reference: Vec<_> = prompt
            .iter()
            .chain(&continuation)
            .map(|&token| serial.decode(token).unwrap())
            .collect();
        let schedules = [
            vec![window - 1, 1, 1, 4],
            vec![window + 1, 4],
            vec![3; prompt.len().div_ceil(3)],
            vec![prompt.len()],
        ];
        for chunks in schedules {
            let mut position = 0;
            for count in chunks {
                let end = (position + count).min(prompt.len());
                let logits = chunked.prefill(&prompt[position..end]).unwrap();
                assert_eq!(chunked.position(), end);
                assert_logits(
                    &logits,
                    &reference[end - 1],
                    &format!("window {window}, chunk {position}..{end}"),
                );
                position = end;
            }
            assert_eq!(position, prompt.len());
            // Equal final prompt logits alone do not prove equivalent continuation.
            // Compare complete rows on a common teacher-forced history, not argmaxes.
            for (offset, &token) in continuation.iter().enumerate() {
                let logits = chunked.decode(token).unwrap();
                assert_logits(
                    &logits,
                    &reference[prompt.len() + offset],
                    &format!("window {window}, continuation {offset}"),
                );
            }
            assert_eq!(chunked.position(), prompt.len() + continuation.len());
            chunked.reset();
            assert_eq!(chunked.position(), 0);
        }
    }
}

#[test]
fn rejected_chunks_preserve_the_next_continuation() {
    let weights = model(4);
    let mut reference = Gemma4CpuDecoder::shared(weights.clone()).unwrap();
    let mut rejected = Gemma4CpuDecoder::shared(weights.clone()).unwrap();
    let prompt = [0, 3, 1];
    let expected = reference.prefill(&prompt).unwrap();
    let actual = rejected.prefill(&prompt).unwrap();
    assert_logits(&actual, &expected, "initial prefix");
    let overlong = vec![0; weights.config().max_position_embeddings() - prompt.len() + 1];
    for chunk in [&[][..], &[2, 7], &overlong] {
        assert!(rejected.prefill(chunk).is_err());
        assert_eq!(rejected.position(), prompt.len());
    }
    for token in [6, 0, 2] {
        let expected = reference.decode(token).unwrap();
        let actual = rejected.decode(token).unwrap();
        assert_logits(&actual, &expected, "continuation after rejection");
    }
}
