//! The experimental selector must preserve native outputs and every KV owner.

use super::*;
use ribn_hf::LocalModelPackage;

fn same_state(native: &Gemma4MetalDecoder, graph: &Gemma4MetalDecoder, a: &[u16], b: &[u16]) {
    assert_eq!(a.len(), 262_144);
    assert!(
        a.iter()
            .chain(b)
            .all(|&bits| f32::from_bits(u32::from(bits) << 16).is_finite())
    );
    assert!(
        a == b,
        "complete common-history logits must be bit-identical"
    );
    assert_eq!(native.position(), graph.position());
    for (layer, (a, b)) in native.caches.iter().zip(&graph.caches).enumerate() {
        match (a, b) {
            (Some(a), Some(b)) => {
                assert!(
                    a.keys == b.keys,
                    "layer {layer} complete key history differs"
                );
                assert!(
                    a.values == b.values,
                    "layer {layer} complete value history differs"
                );
            }
            (None, None) => {}
            _ => panic!("layer {layer} KV ownership differs"),
        }
    }
}

#[test]
#[ignore = "loads two 14.9 GB models and executes Metal; use --release"]
fn graph_selection_preserves_native_logits_and_all_owner_histories() {
    let package =
        LocalModelPackage::open(std::env::var_os("RIBN_GEMMA_PACKAGE").expect("package")).unwrap();
    let source = Gemma4Source::open(&package).unwrap();
    let reference: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var_os("RIBN_GEMMA_REFERENCE").expect("reference")).unwrap(),
    )
    .unwrap();
    let prompt: Vec<u32> = reference["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "chat")
        .unwrap()["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| u32::try_from(value.as_u64().unwrap()).unwrap())
        .collect();
    let native = Arc::new(Gemma4MetalModel::load(&source, Gemma4MetalFfn::Native).unwrap());
    let graph =
        Arc::new(Gemma4MetalModel::load(&source, Gemma4MetalFfn::ExperimentalGraph).unwrap());
    let mut a = Gemma4MetalDecoder::shared(native).unwrap();
    let mut b = Gemma4MetalDecoder::shared(graph).unwrap();
    // Selected pair16/pair64/down64 and native tails/fallbacks coexist in different
    // owners. No global switch can accidentally change the native control.
    for length in [7, 16, 32, 64, 65] {
        a.reset();
        b.reset();
        let tokens: Vec<_> = prompt.iter().copied().cycle().take(length).collect();
        let x = a.prefill(&tokens).unwrap();
        let y = b.prefill(&tokens).unwrap();
        same_state(&a, &b, &x, &y);
        for &token in prompt.iter().take(2) {
            let x = a.decode(token).unwrap();
            let y = b.decode(token).unwrap();
            same_state(&a, &b, &x, &y);
        }
    }
    a.reset();
    b.reset();
    let tokens: Vec<_> = prompt.iter().copied().cycle().take(511).collect();
    for chunk in tokens.chunks(64) {
        let x = a.prefill(chunk).unwrap();
        let y = b.prefill(chunk).unwrap();
        same_state(&a, &b, &x, &y);
    }
    // Selected FFN shapes also cross the sliding-window and attention-chain
    // fallback boundaries. Compare full retained histories, not just visible KV.
    for length in [16, 64, 1] {
        let tokens: Vec<_> = prompt.iter().copied().cycle().take(length).collect();
        let x = a.prefill(&tokens).unwrap();
        let y = b.prefill(&tokens).unwrap();
        same_state(&a, &b, &x, &y);
    }
}
