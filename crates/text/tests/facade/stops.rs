//! Decoded-text stop boundaries over the real facade and processors.
use super::*;

#[test]
fn stop_strings_trim_decoded_text_and_report_stop() {
    let harness = Harness::start(limits(), TextConfig::default());
    harness.plan.script(
        b'A',
        &[
            u32::from(b'a'),
            u32::from(b'b'),
            u32::from(b'E'),
            u32::from(b'N'),
            u32::from(b'D'),
            u32::from(b'c'),
            EOS,
        ],
    );
    let request =
        ribn_text::TextRequest::new(TextInput::prompt("A"), options()).with_stops(["END"]);
    let response = harness
        .model
        .generate_request_blocking(request)
        .expect("stopped generation");
    assert_eq!(response.text, "ab");
    assert!(
        matches!(response.reason, FinishReason::Stop),
        "{:?}",
        response.reason
    );
}

#[test]
fn a_text_stop_ignores_buffered_invalid_utf8_but_preserves_committed_token_ids() {
    let harness = Harness::default_start();
    harness.plan.script(b'A', &[u32::from(b'X'), 255, EOS]);
    let request = ribn_text::TextRequest::new(TextInput::prompt("A"), options()).with_stops(["X"]);
    let mut stream = harness.model.stream_request_blocking(request).unwrap();
    wait_until(|| harness.plan.polls() >= 3);
    let result = drain_until_error(&mut stream);
    assert!(result.error.is_none(), "{:?}", result.error);
    assert!(result.text.is_empty());
    assert_eq!(result.tokens, [u32::from(b'X'), 255]);
    assert_eq!(result.terminals, [FinishReason::Stop]);
}

#[test]
fn stop_prefix_release_obeys_the_delta_bound_on_push_and_terminal() {
    for suffix in [Some(u32::from(b'X')), None] {
        let mut script = vec![u32::from(b'B'); 5];
        script.extend(suffix);
        script.push(EOS);
        let harness = Harness::start(
            ProcessorLimits {
                max_decoded_token_bytes: 1,
                max_decoded_delta_bytes: 4,
                ..limits()
            },
            TextConfig::default(),
        );
        harness.plan.script(b'A', &script);
        harness.plan.script(b'B', &[u32::from(b'!'), EOS]);
        let request =
            ribn_text::TextRequest::new(TextInput::prompt("A"), options()).with_stops(["BBBBBD"]);
        let error = harness
            .model
            .generate_request_blocking(request)
            .expect_err("released stop prefix plus decoded text must fit the delta bound");
        assert!(
            matches!(
                error,
                TextError::LimitExceeded {
                    field: "decoded delta bytes",
                    allowed: 4,
                    actual,
                } if actual == 5 + usize::from(suffix.is_some())
            ),
            "{error:?}"
        );
        assert_eq!(
            harness
                .model
                .generate_blocking(TextInput::prompt("B"), options())
                .unwrap()
                .text,
            "!"
        );
    }
}

#[test]
fn stop_strings_flush_an_incomplete_prefix_at_the_terminal_boundary() {
    let harness = Harness::start(limits(), TextConfig::default());
    harness.plan.script(
        b'A',
        &[
            u32::from(b'a'),
            u32::from(b'b'),
            u32::from(b'E'),
            u32::from(b'N'),
            EOS,
        ],
    );
    let request =
        ribn_text::TextRequest::new(TextInput::prompt("A"), options()).with_stops(["END"]);
    let response = harness
        .model
        .generate_request_blocking(request)
        .expect("terminal flush");
    assert_eq!(response.text, "abEN");
    assert!(
        matches!(response.reason, FinishReason::Stop),
        "{:?}",
        response.reason
    );
}

#[test]
fn multiple_stops_use_the_earliest_then_longest_match() {
    let harness = Harness::start(limits(), TextConfig::default());
    harness.plan.script(
        b'A',
        &[
            u32::from(b'a'),
            u32::from(b'x'),
            u32::from(b'b'),
            u32::from(b'y'),
            EOS,
        ],
    );
    let request =
        ribn_text::TextRequest::new(TextInput::prompt("A"), options()).with_stops(["by", "y"]);
    let response = harness
        .model
        .generate_request_blocking(request)
        .expect("earliest stop");
    assert_eq!(response.text, "ax");
}

#[test]
fn streamed_stops_never_publish_a_partial_match() {
    let harness = Harness::start(limits(), TextConfig::default());
    harness.plan.script(
        b'A',
        &[
            u32::from(b'a'),
            u32::from(b'E'),
            u32::from(b'N'),
            u32::from(b'D'),
            u32::from(b'z'),
            EOS,
        ],
    );
    let request =
        ribn_text::TextRequest::new(TextInput::prompt("A"), options()).with_stops(["END"]);
    let mut stream = harness
        .model
        .stream_request_blocking(request)
        .expect("stream with stop");
    let mut text = String::new();
    let mut reason = None;
    while let Some(event) = stream.next_blocking() {
        match event.expect("stream event") {
            ribn_text::TextEvent::Delta { text: delta, .. } => {
                assert!(!delta.contains("EN"), "partial stop leaked: {delta:?}");
                text.push_str(&delta);
            }
            ribn_text::TextEvent::Finished {
                reason: finished, ..
            } => {
                reason = Some(finished);
            }
        }
    }
    assert_eq!(text, "a");
    assert!(matches!(reason, Some(FinishReason::Stop)), "{reason:?}");
}

#[test]
fn ignore_eos_continues_past_model_stop_tokens() {
    let harness = Harness::start(limits(), TextConfig::default());
    harness
        .plan
        .script(b'A', &[u32::from(b'a'), EOS, u32::from(b'b'), EOS]);
    let stopping = harness
        .model
        .generate_request_blocking(ribn_text::TextRequest::new(
            TextInput::prompt("A"),
            options(),
        ))
        .expect("model stop");
    assert_eq!(stopping.text, "a");
    assert!(
        matches!(stopping.reason, FinishReason::Stop),
        "{:?}",
        stopping.reason
    );

    harness
        .plan
        .script(b'A', &[u32::from(b'a'), EOS, u32::from(b'b'), EOS]);
    let continuing = harness
        .model
        .generate_request_blocking(
            ribn_text::TextRequest::new(TextInput::prompt("A"), options()).with_ignore_eos(true),
        )
        .expect("ignored stop");
    assert!(continuing.text.starts_with('a') && continuing.text.contains('b'));
    assert!(
        matches!(continuing.reason, FinishReason::Length),
        "{:?}",
        continuing.reason
    );
}

#[test]
fn invalid_stop_bounds_reject_before_enqueue() {
    let harness = Harness::start(limits(), TextConfig::default());
    for stops in [
        vec![String::new()],
        vec!["a".to_owned(); 5],
        vec!["x".repeat(ribn_text::MAX_STOP_BYTES + 1)],
    ] {
        let request =
            ribn_text::TextRequest::new(TextInput::prompt("A"), options()).with_stops(stops);
        assert!(matches!(
            harness.model.stream_request_blocking(request),
            Err(TextError::InvalidInput(_))
        ));
    }
}
