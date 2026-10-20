//! Prepared Gemma weights on the local Metal device; macOS-only.
#![cfg(all(target_os = "macos", feature = "metal"))]

use engine_gemma4::{
    Gemma4CpuWeights, Gemma4MetalWeights, Gemma4Source, LayerParameter, ParameterId,
};
use engine_metal::{MetalBf16Error, MetalBf16Executor};

#[path = "common/weights.rs"]
mod weights;
use weights::{Dir, inventory, package};

fn executor() -> Option<MetalBf16Executor> {
    match MetalBf16Executor::new() {
        Ok(executor) => Some(executor),
        Err(MetalBf16Error::Device) => {
            eprintln!("no Metal device; skipping");
            None
        }
        Err(error) => panic!("Metal preparation failed: {error}"),
    }
}

#[test]
fn materializes_every_parameter_bitwise_with_a_tied_head() {
    let Some(executor) = executor() else { return };
    let dir = Dir::new();
    let package = package(&dir, &inventory(), false);
    let source = Gemma4Source::open(&package).unwrap();
    let host = Gemma4CpuWeights::load(&source).unwrap();
    // Readers retain the preflighted file, not a pathname reopened during load.
    std::fs::rename(
        dir.0.join("model.safetensors"),
        dir.0.join("retained.safetensors"),
    )
    .unwrap();
    drop(package);
    let metal = Gemma4MetalWeights::load(&source, &executor).unwrap();
    drop(source);
    assert_eq!(metal.len(), host.len());
    assert_eq!(metal.bytes(), host.bytes());
    assert_eq!(metal.config(), host.config());
    for (id, parameter) in host.parameters() {
        let buffer = metal.parameter(id).expect("every active parameter uploads");
        assert_eq!(buffer.elements(), parameter.values().len());
        assert_eq!(buffer.bits(), parameter.values(), "{id:?}");
    }
    assert!(metal.parameter(ParameterId::TokenEmbedding).is_some());

    let tail = host
        .config()
        .layers()
        .iter()
        .enumerate()
        .find(|(index, geometry)| geometry.kv_owner != *index)
        .map(|(index, _)| index);
    if let Some(tail) = tail {
        assert!(metal.layer(tail, LayerParameter::Key).is_err());
        assert!(metal.layer(tail, LayerParameter::Value).is_err());
    }
    assert!(
        metal
            .layer(host.config().layers().len(), LayerParameter::Query)
            .is_err()
    );
}

#[test]
fn truncated_source_rejects_the_partial_set_and_a_fresh_load_recovers() {
    let Some(executor) = executor() else { return };
    let dir = Dir::new();
    let tensors = inventory();
    let package = package(&dir, &tensors, false);
    let source = Gemma4Source::open(&package).unwrap();
    let path = dir.0.join("model.safetensors");
    // Fault injection: an external writer violates the no-in-place-mutation
    // contract after preflight. Some native parameters prepare before this EOF.
    let length = std::fs::metadata(&path).unwrap().len();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(length - 1)
        .unwrap();
    assert!(Gemma4MetalWeights::load(&source, &executor).is_err());
    drop(source);
    std::fs::write(&path, weights::artifact(&tensors)).unwrap();
    let source = Gemma4Source::open(&package).unwrap();
    let weights = Gemma4MetalWeights::load(&source, &executor).unwrap();
    assert_eq!(weights.len(), source.parameters().len());
    assert_eq!(
        weights.bytes(),
        u64::try_from(source.encoded_bytes()).unwrap()
    );
}
