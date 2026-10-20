//! Prepared Gemma weights on the local Metal device; macOS-only.
#![cfg(all(target_os = "macos", feature = "metal"))]

use engine_gemma4::{
    Gemma4CpuWeights, Gemma4MetalWeights, Gemma4Source, LayerParameter, ParameterId,
};
use engine_metal::MetalBf16Executor;

#[path = "common/weights.rs"]
mod weights;
use weights::{Dir, inventory, package};

#[test]
fn materializes_every_parameter_bitwise_with_a_tied_head() {
    let Ok(executor) = MetalBf16Executor::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let dir = Dir::new();
    let package = package(&dir, &inventory(), false);
    let source = Gemma4Source::open(&package).unwrap();
    let host = Gemma4CpuWeights::load(&source).unwrap();
    let metal = Gemma4MetalWeights::load(&host, &executor).unwrap();
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
