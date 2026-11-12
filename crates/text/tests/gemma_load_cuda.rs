//! Real ordinary CUDA package-loader lifecycle, not a manually assembled executor.
#![cfg(feature = "cuda")]

use ribn_text::{FinishReason, GenerationOptions, TextEvent, TextInput, TextOwner};
use serde_json::Value;
use std::path::PathBuf;

fn environment(name: &str) -> PathBuf {
    PathBuf::from(
        std::env::var_os(name).unwrap_or_else(|| panic!("{name} must name pinned evidence")),
    )
}

fn archived_chat() -> (Vec<u32>, Vec<u32>) {
    let reference: Value =
        serde_json::from_slice(&std::fs::read(environment("RIBN_GEMMA_REFERENCE")).unwrap())
            .unwrap();
    assert_eq!(reference["complete"], true);
    assert_eq!(reference["precision"], "bf16");
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

    assert_eq!(prompt.len(), 18);
    assert_eq!(expected.len(), 4);
    (prompt, expected)
}

fn qualified_package() -> PathBuf {
    let path = environment("RIBN_GEMMA_PACKAGE");
    let package = ribn_hf::LocalModelPackage::open(path.clone()).unwrap();
    let source = engine_gemma4::Gemma4Source::open(&package).unwrap();
    let context = cudarc::driver::CudaContext::new(0).unwrap();
    assert!(
        engine_gemma4::Gemma4CudaModelWide::matches_qualification_scope(
            source.config(),
            &context.name().unwrap(),
            4096
        )
    );
    path
}

#[test]
#[expect(
    clippy::assertions_on_constants,
    reason = "compile in debug but require release device execution"
)]
#[ignore = "real ordinary package loader; release-only serialized CUDA qualification"]
fn ordinary_cuda_loader_preserves_concurrent_delivery_and_recovers_after_abandonment() {
    assert!(!cfg!(debug_assertions), "device gate requires --release");
    let (prompt, expected) = archived_chat();
    let (mut owner, report) = TextOwner::load_gemma_cuda(qualified_package())
        .expect("actual ordinary loader, not manually assembled executor");
    assert_eq!(report.model_bytes, 14_926_026_836);
    let model = owner.model().clone();
    let options = GenerationOptions {
        max_output_tokens: 4,
        ..GenerationOptions::default()
    };
    let mut abandoned = model
        .stream_blocking(
            TextInput::Tokens(prompt.clone()),
            GenerationOptions {
                max_output_tokens: 128,
                ..GenerationOptions::default()
            },
        )
        .unwrap();
    assert!(matches!(
        abandoned.next_blocking().unwrap().unwrap(),
        TextEvent::Delta { token: Some(token), .. } if token == expected[0]
    ));
    drop(abandoned);

    // Four fresh callers exercise the ordinary default owner at its active
    // population bound after a positive-progress stream was abandoned. This
    // observable path does not itself attest a particular physical M2 schedule.
    let barrier = std::sync::Barrier::new(4);
    std::thread::scope(|scope| {
        let mut callers = Vec::new();
        for _ in 0..4 {
            let model = model.clone();
            let (prompt, expected, options, barrier) = (&prompt, &expected, &options, &barrier);
            callers.push(scope.spawn(move || {
                barrier.wait();
                let mut stream = model
                    .stream_blocking(TextInput::Tokens(prompt.clone()), options.clone())
                    .unwrap();
                let mut tokens = Vec::new();
                let mut text = String::new();
                let mut finished = false;
                while let Some(event) = stream.next_blocking() {
                    assert!(!finished, "no output after the terminal event");
                    match event.unwrap() {
                        TextEvent::Delta { token, text: part } => {
                            tokens.extend(token);
                            text.push_str(&part);
                        }
                        TextEvent::Finished { reason, usage } => {
                            assert_eq!(reason, FinishReason::Length);
                            assert_eq!(usage.prompt_tokens as usize, prompt.len());
                            assert_eq!(usage.completion_tokens, 4);
                            assert_eq!(usage.total_tokens(), 22);
                            finished = true;
                        }
                    }
                }
                assert!(finished, "terminal event followed by full EOF");
                assert_eq!(&tokens, expected);
                assert_eq!(text, "A mutex (short");
            }));
        }
        for caller in callers {
            caller.join().unwrap();
        }
    });

    // This envelope cannot fit the public loader's 4K continuation limit. Its
    // rejection must not poison unrelated later work on the same loaded owner.
    assert!(
        model
            .generate_blocking(
                TextInput::Tokens(vec![2; 4096]),
                GenerationOptions {
                    max_output_tokens: 1,
                    ..GenerationOptions::default()
                },
            )
            .is_err()
    );
    let response = model
        .generate_blocking(TextInput::Tokens(prompt.clone()), options.clone())
        .unwrap();
    assert_eq!(response.tokens, expected);
    assert_eq!(response.text, "A mutex (short");
    assert_eq!(response.reason, FinishReason::Length);
    assert_eq!(response.usage.prompt_tokens as usize, prompt.len());
    assert_eq!(response.usage.completion_tokens, 4);
    owner.shutdown().unwrap();
    owner.shutdown().unwrap();
    assert!(
        model
            .generate_blocking(TextInput::Tokens(prompt), options)
            .is_err()
    );
}
