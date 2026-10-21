//! Runtime executor over the host CPU decoder, using the independent tiny package.

use engine_gemma4::{Gemma4CpuDecoder, Gemma4CpuWeights, Gemma4Executor, Gemma4Source, argmax};
use ribn::{Engine, Event, FinishReason, GenerationOptions, RequestId, Sampling, TokenRequest};
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

fn request(prompt: &[u32], steps: u32) -> TokenRequest {
    TokenRequest::new(
        prompt.to_vec(),
        GenerationOptions {
            max_output_tokens: steps,
            sampling: Sampling::default(),
            stop_tokens: Vec::new(),
        },
    )
}

/// Drive the engine until every request finishes, keeping each token stream apart.
fn drive(
    engine: &mut Engine,
    requests: &[(RequestId, u32)],
) -> Vec<(Vec<u32>, Option<FinishReason>)> {
    let mut streams = vec![(Vec::new(), None); requests.len()];
    for _ in 0..64 {
        engine.step().unwrap();
        for (index, (request, _)) in requests.iter().enumerate() {
            while let Some(event) = engine.pop_event_for(*request) {
                match event {
                    Event::Token { token, .. } => streams[index].0.push(token),
                    Event::Finished { reason, .. } => streams[index].1 = Some(reason),
                }
            }
        }
        if streams.iter().all(|(_, reason)| reason.is_some()) {
            return streams;
        }
    }
    panic!("requests never finished");
}

fn engine_over(host: Arc<Gemma4CpuWeights>) -> Engine {
    let executor = Gemma4Executor::cpu(host).unwrap();
    Engine::with_defaults(executor).unwrap()
}

#[test]
fn cpu_executor_matches_the_direct_decoder_through_the_engine() {
    let dir = Dir::new();
    let package = package(&dir, &inventory(), false);
    let source = Gemma4Source::open(&package).unwrap();
    let host = Arc::new(Gemma4CpuWeights::load(&source).unwrap());
    let prompt = [1_u32, 2, 3];
    let expected = greedy(
        &mut Gemma4CpuDecoder::shared(host.clone()).unwrap(),
        &prompt,
        3,
    );

    let mut engine = engine_over(host);
    let id = engine.enqueue(request(&prompt, 3)).unwrap();
    let streams = drive(&mut engine, &[(id, 3)]);
    assert_eq!(streams[0].0, expected);
    assert_eq!(streams[0].1, Some(FinishReason::Length));
    assert_eq!(engine.status().requests, 0);
    engine.shutdown().unwrap();
}

#[test]
fn interleaved_sequences_keep_independent_continuations() {
    let dir = Dir::new();
    let package = package(&dir, &inventory(), false);
    let source = Gemma4Source::open(&package).unwrap();
    let host = Arc::new(Gemma4CpuWeights::load(&source).unwrap());
    let prompts: [&[u32]; 2] = [&[1, 2], &[3, 4, 5, 6]];
    let expected: Vec<_> = prompts
        .iter()
        .map(|prompt| {
            greedy(
                &mut Gemma4CpuDecoder::shared(host.clone()).unwrap(),
                prompt,
                4,
            )
        })
        .collect();

    let mut engine = engine_over(host.clone());
    let requests: Vec<_> = prompts
        .iter()
        .map(|prompt| (engine.enqueue(request(prompt, 4)).unwrap(), 4))
        .collect();
    let streams = drive(&mut engine, &requests);
    for (stream, expected) in streams.iter().zip(&expected) {
        assert_eq!(&stream.0, expected);
        assert_eq!(stream.1, Some(FinishReason::Length));
    }
    assert_eq!(engine.status().requests, 0);

    // A released sequence frees its continuation state for a later request.
    let id = engine.enqueue(request(&[6_u32, 0], 2)).unwrap();
    let streams = drive(&mut engine, &[(id, 2)]);
    let expected = greedy(&mut Gemma4CpuDecoder::shared(host).unwrap(), &[6, 0], 2);
    assert_eq!(streams[0].0, expected);
    engine.shutdown().unwrap();
}

#[test]
fn invalid_input_and_unsupported_sampling_leave_a_healthy_peer_running() {
    let dir = Dir::new();
    let package = package(&dir, &inventory(), false);
    let source = Gemma4Source::open(&package).unwrap();
    let host = Arc::new(Gemma4CpuWeights::load(&source).unwrap());
    let invalid = u32::try_from(host.config().vocab_size()).unwrap();
    let expected = greedy(
        &mut Gemma4CpuDecoder::shared(host.clone()).unwrap(),
        &[1, 2],
        3,
    );
    let mut engine = engine_over(host);
    let bad = engine.enqueue(request(&[1, invalid], 3)).unwrap();
    let mut sampling = request(&[1, 2], 3);
    sampling.options.sampling.temperature = 0.5;
    let unsupported = engine.enqueue(sampling).unwrap();
    let good = engine.enqueue(request(&[1, 2], 3)).unwrap();
    let streams = drive(&mut engine, &[(bad, 3), (unsupported, 3), (good, 3)]);
    for (tokens, reason) in &streams[..2] {
        assert!(tokens.is_empty());
        assert!(matches!(reason, Some(FinishReason::Failed(_))));
    }
    assert_eq!(streams[2].0, expected);
    assert_eq!(streams[2].1, Some(FinishReason::Length));
    assert_eq!(engine.status().requests, 0);
    engine.shutdown().unwrap();
}

#[test]
fn engine_stop_token_ends_generation_on_the_real_executor() {
    let dir = Dir::new();
    let package = package(&dir, &inventory(), false);
    let source = Gemma4Source::open(&package).unwrap();
    let host = Arc::new(Gemma4CpuWeights::load(&source).unwrap());
    let prompt = [0_u32, 2];
    let expected = greedy(
        &mut Gemma4CpuDecoder::shared(host.clone()).unwrap(),
        &prompt,
        1,
    );
    let stop = expected[0];

    let mut engine = engine_over(host);
    let mut options = request(&prompt, 4);
    options.options.stop_tokens = vec![stop];
    let id = engine.enqueue(options).unwrap();
    let streams = drive(&mut engine, &[(id, 4)]);
    assert_eq!(streams[0].1, Some(FinishReason::Stop));
    assert!(streams[0].0.is_empty(), "the stop token is suppressed");
    assert_eq!(engine.status().requests, 0);
    engine.shutdown().unwrap();
}

#[test]
fn cancelled_request_releases_its_capacity() {
    let dir = Dir::new();
    let package = package(&dir, &inventory(), false);
    let source = Gemma4Source::open(&package).unwrap();
    let host = Arc::new(Gemma4CpuWeights::load(&source).unwrap());
    let mut engine = engine_over(host.clone());
    let id = engine.enqueue(request(&[1, 2, 3], 8)).unwrap();
    engine.step().unwrap();
    while engine.pop_event_for(id).is_some() {}
    assert_eq!(engine.status().requests, 1);
    engine.cancel(id).unwrap();
    for _ in 0..8 {
        if engine.status().requests == 0 {
            break;
        }
        engine.step().unwrap();
        while engine.pop_event_for(id).is_some() {}
    }
    assert_eq!(engine.status().requests, 0);

    // The released capacity accepts a later request with a clean continuation.
    let expected = greedy(&mut Gemma4CpuDecoder::shared(host).unwrap(), &[6, 0], 2);
    let id = engine.enqueue(request(&[6, 0], 2)).unwrap();
    let streams = drive(&mut engine, &[(id, 2)]);
    assert_eq!(streams[0].0, expected);
    engine.shutdown().unwrap();
}

#[test]
fn shutdown_settles_a_live_request() {
    let dir = Dir::new();
    let package = package(&dir, &inventory(), false);
    let source = Gemma4Source::open(&package).unwrap();
    let host = Arc::new(Gemma4CpuWeights::load(&source).unwrap());
    let mut engine = engine_over(host);
    let id = engine.enqueue(request(&[1, 2, 3], 8)).unwrap();
    engine.step().unwrap();
    while engine.pop_event_for(id).is_some() {}
    engine.shutdown().unwrap();
    assert!(engine.status().closed);
    assert_eq!(engine.status().requests, 0);
    assert!(engine.enqueue(request(&[1], 1)).is_err());
}
