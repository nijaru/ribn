//! Exercise ordinary Metal package loading, not a hand-assembled test facade.
#![cfg(all(target_os = "macos", feature = "metal"))]

use ribn::GenerationOptions;
use ribn_text::{Gemma4MetalFfn, TextInput, TextOwner};
use serde_json::Value;
use std::path::PathBuf;

fn environment(name: &str) -> PathBuf {
    PathBuf::from(
        std::env::var_os(name)
            .unwrap_or_else(|| panic!("{name} must name the archived evidence file")),
    )
}

#[test]
#[ignore = "loads the 14.9 GB package and runs the Metal decoder; use --release"]
fn gemma_text_facade_reproduces_the_archived_greedy_tokens() {
    let reference: Value =
        serde_json::from_slice(&std::fs::read(environment("RIBN_GEMMA_REFERENCE")).unwrap())
            .unwrap();
    let case = reference["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "chat")
        .expect("archived chat case");
    let prompt: Vec<u32> = case["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| u32::try_from(value.as_u64().unwrap()).unwrap())
        .collect();
    let expected: Vec<u32> = case["steps"]
        .as_array()
        .unwrap()
        .iter()
        .map(|step| u32::try_from(step["chosen"].as_u64().unwrap()).unwrap())
        .collect();

    let mut native_selected_tokens = None;
    for ffn in [Gemma4MetalFfn::Native, Gemma4MetalFfn::ExperimentalGraph] {
        // This ignored qualification gate targets the M3 Max Graph lane.
        // Unsupported preparation is a failure here, never fallback evidence.
        let (mut owner, _) = TextOwner::load_gemma(environment("RIBN_GEMMA_PACKAGE"), ffn)
            .expect("ordinary package/text-owner assembly");
        for _ in 0..2 {
            let response = owner
                .model()
                .generate_blocking(
                    TextInput::Tokens(prompt.clone()),
                    GenerationOptions {
                        max_output_tokens: 4,
                        ..GenerationOptions::default()
                    },
                )
                .expect("text generation");
            assert_eq!(
                response.tokens, expected,
                "{ffn:?}: text facade greedy tokens"
            );
        }
        // Under Graph, default scheduling splits the archived prompt into a
        // selected 16-row prefill and a native tail. Also compare a 16-token
        // request, without a tail, against the native policy.
        let selected = owner
            .model()
            .generate_blocking(
                TextInput::Tokens(prompt[..16].to_vec()),
                GenerationOptions {
                    max_output_tokens: 4,
                    ..GenerationOptions::default()
                },
            )
            .expect("selected-shape text generation");
        match ffn {
            Gemma4MetalFfn::Native => native_selected_tokens = Some(selected.tokens),
            Gemma4MetalFfn::ExperimentalGraph => {
                assert_eq!(Some(selected.tokens), native_selected_tokens);
            }
        }
        owner
            .shutdown()
            .expect("physical shutdown through the shared model owner");
    }
}
