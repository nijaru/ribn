//! Runtime executor over the host CPU decoder, using the independent tiny package.

use engine_gemma4::{Gemma4CpuDecoder, Gemma4CpuWeights, Gemma4Executor, Gemma4Source, argmax};
use ribn::{Engine, Event, FinishReason, GenerationOptions, TokenRequest};
use std::sync::Arc;

#[path = "common/weights.rs"]
mod weights;
use weights::{Dir, inventory, package};

fn greedy(decoder: &mut Gemma4CpuDecoder, prompt: &[u32], steps: usize) -> Vec<u32> {
    let mut logits = decoder.prefill(prompt).unwrap();
    let mut tokens = Vec::new();
    for _ in 0..steps {
        let token = argmax(&logits).unwrap();
        tokens.push(token);
        logits = decoder.decode(token).unwrap();
    }
    tokens
}

#[test]
fn cpu_executor_matches_the_direct_decoder_through_the_engine() {
    let dir = Dir::new();
    let package = package(&dir, &inventory(), false);
    let source = Gemma4Source::open(&package).unwrap();
    let host = Arc::new(Gemma4CpuWeights::load(&source).unwrap());
    let prompt = [1_u32, 2, 3];
    let mut direct = Gemma4CpuDecoder::shared(host.clone()).unwrap();
    let expected = greedy(&mut direct, &prompt, 3);

    let executor = Gemma4Executor::cpu(host).unwrap();
    let mut engine = Engine::with_defaults(executor).unwrap();
    let request = engine
        .enqueue(TokenRequest::new(
            prompt.to_vec(),
            GenerationOptions {
                max_output_tokens: 3,
                ..GenerationOptions::default()
            },
        ))
        .unwrap();
    let mut tokens = Vec::new();
    let mut reason = None;
    for _ in 0..16 {
        engine.step().unwrap();
        while let Some(event) = engine.pop_event_for(request) {
            match event {
                Event::Token { token, .. } => tokens.push(token),
                Event::Finished { reason: end, .. } => reason = Some(end),
            }
        }
        if reason.is_some() {
            break;
        }
    }
    assert_eq!(tokens, expected, "engine tokens match the direct decoder");
    assert_eq!(reason, Some(FinishReason::Length));
    assert_eq!(engine.status().requests, 0);
    engine.shutdown().unwrap();
}
