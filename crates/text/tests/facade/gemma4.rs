//! Real Gemma codec, production facade/driver, scripted token execution only.
use super::*;
use ribn_hf::LocalModelPackage;
use ribn_text::{GemmaProcessor, TextEvent};
#[path = "../../../gemma4/tests/common/processor.rs"]
mod assets;
fn byte(value: u8) -> u32 {
    12 + u32::from(value)
}
fn gemma_limits() -> ProcessorLimits {
    ProcessorLimits {
        max_prompt_tokens: 1024,
        ..limits()
    }
}
fn processor(limits: ProcessorLimits) -> Arc<dyn TextProcessor> {
    let fixture = assets::Fixture::new();
    Arc::new(GemmaProcessor::open(&LocalModelPackage::open(&fixture.0).unwrap(), limits).unwrap())
}
fn harness(processor: Arc<dyn TextProcessor>) -> Harness {
    Harness::with_processor(
        processor,
        TextConfig {
            preprocessing_workers: 2,
            batch_window: 2,
        },
        8,
    )
}
fn input(id: u32) -> TextInput {
    TextInput::Tokens(vec![id])
}

#[test]
fn package_stops_and_special_visibility_do_not_use_gguf_prompt_policy() {
    let fixture = assets::Fixture::new();
    // Root EOS includes 4; generation metadata deliberately does not. Preserve 4.
    fixture.change("generation_config.json", |v| {
        v["eos_token_id"] = serde_json::json!([1, 5]);
    });
    let processor: Arc<dyn TextProcessor> = Arc::new(
        GemmaProcessor::open(
            &LocalModelPackage::open(&fixture.0).unwrap(),
            gemma_limits(),
        )
        .unwrap(),
    );
    for request in [
        TextInput::prompt("X"),
        TextInput::chat(vec![Message::user("X")]),
        input(10),
    ] {
        let encoded = processor.encode(request).unwrap();
        assert_eq!(encoded.stop_tokens(), [1, 5]);
    }
    for (id, marker) in [(7, "<|image|>"), (8, "<|audio|>"), (9, "<|video|>")] {
        assert!(processor.encode(input(id)).unwrap_err().is_request_local());
        assert!(
            processor
                .encode(TextInput::prompt(marker))
                .unwrap_err()
                .is_request_local()
        );
    }
    let harness = harness(processor);
    harness
        .plan
        .script(10, &[2, 4, byte(b'a'), byte(0xff), 10, 5]);
    let result = harness
        .model
        .generate_blocking(input(10), options())
        .unwrap();
    assert_eq!(result.text, "<bos><turn|>��X");
    assert_eq!(result.tokens, [2, 4, byte(b'a'), byte(0xff), 10]);
    assert_eq!(result.reason, FinishReason::Stop);
    assert_eq!(result.usage.completion_tokens, 6);
}

#[test]
fn interleaved_requests_withhold_whole_runs_and_flush_before_terminal() {
    let harness = harness(processor(gemma_limits()));
    harness
        .plan
        .script(10, &[byte(b'a'), byte(0xff), 10, byte(0xc3), byte(0xa9), 1]);
    harness.plan.script(11, &[byte(b'e'), 1]);
    let mut a = harness.model.stream_blocking(input(10), options()).unwrap();
    for id in [byte(b'a'), byte(0xff)] {
        assert_eq!(
            a.next_blocking().unwrap().unwrap(),
            TextEvent::Delta {
                token: Some(id),
                text: String::new()
            }
        );
    }
    assert_eq!(
        harness
            .model
            .generate_blocking(input(11), options())
            .unwrap()
            .text,
        "e"
    );
    assert_eq!(
        a.next_blocking().unwrap().unwrap(),
        TextEvent::Delta {
            token: Some(10),
            text: "��X".into()
        }
    );
    for id in [byte(0xc3), byte(0xa9)] {
        assert_eq!(
            a.next_blocking().unwrap().unwrap(),
            TextEvent::Delta {
                token: Some(id),
                text: String::new()
            }
        );
    }
    assert_eq!(
        a.next_blocking().unwrap().unwrap(),
        TextEvent::Delta {
            token: None,
            text: "é".into()
        }
    );
    assert!(matches!(
        a.next_blocking().unwrap().unwrap(),
        TextEvent::Finished {
            reason: FinishReason::Stop,
            ..
        }
    ));
    assert!(a.next_blocking().is_none());
}

#[test]
fn output_budget_rejects_before_enqueue_and_returns_preparation_permits() {
    let harness = harness(processor(ProcessorLimits {
        max_decode_pending_bytes: 4,
        ..gemma_limits()
    }));
    harness.plan.script(10, &[10, 1]);
    for _ in 0..3 {
        let error = expect_stream_err(
            harness.model.stream_blocking(input(10), options()),
            "eight tokens exceed four pending bytes",
        );
        assert!(matches!(
            error,
            TextError::LimitExceeded {
                field: "decoder pending bytes",
                allowed: 4,
                actual: 8
            }
        ));
    }
    assert_eq!(harness.plan.polls(), 0);
    let result = harness
        .model
        .generate_blocking(
            input(10),
            GenerationOptions {
                max_output_tokens: 4,
                ..options()
            },
        )
        .unwrap();
    assert_eq!(result.text, "X");
}

#[test]
fn push_and_finalization_bounds_settle_only_the_failing_batch_item() {
    for script in [
        vec![byte(b'a'), byte(0xff), 10, 1], // push would emit seven bytes
        vec![byte(b'a'), byte(0xff), 1],     // terminal would emit six bytes
    ] {
        let harness = harness(processor(ProcessorLimits {
            max_decoded_delta_bytes: 4,
            ..gemma_limits()
        }));
        harness.plan.script(10, &script);
        harness.plan.script(11, &[10, 1]);
        let results = harness.model.generate_batch(vec![
            ribn_text::TextRequest::new(input(10), options()),
            ribn_text::TextRequest::new(input(11), options()),
        ]);
        assert!(results[0].as_ref().unwrap_err().is_request_local());
        assert_eq!(results[1].as_ref().unwrap().text, "X");
    }
}

type Gate = (flume::Sender<()>, flume::Receiver<()>);

struct TrackedProcessor {
    inner: Arc<dyn TextProcessor>,
    live: Arc<AtomicUsize>,
    finalized: Arc<AtomicUsize>,
    finish_gate: Option<Gate>,
    constructor_gate: Mutex<Option<Gate>>,
    drop_gate: Mutex<Option<Gate>>,
}
impl TextProcessor for TrackedProcessor {
    fn limits(&self) -> ProcessorLimits {
        self.inner.limits()
    }
    fn encode(&self, input: TextInput) -> Result<EncodedInput, TextError> {
        self.inner.encode(input)
    }
    fn decoder(self: Arc<Self>, tokens: usize) -> Result<Box<dyn TextDecoder>, TextError> {
        let inner = self.inner.clone().decoder(tokens)?;
        self.live.fetch_add(1, Ordering::SeqCst);
        let gate = self.constructor_gate.lock().unwrap().take();
        if let Some((entered, resume)) = gate {
            entered.send(()).unwrap();
            resume.recv().unwrap();
        }
        Ok(Box::new(TrackedDecoder {
            inner: Some(inner),
            live: self.live.clone(),
            finalized: self.finalized.clone(),
            finish_gate: self.finish_gate.clone(),
            drop_gate: self.drop_gate.lock().unwrap().take(),
        }))
    }
}
struct TrackedDecoder {
    inner: Option<Box<dyn TextDecoder>>,
    live: Arc<AtomicUsize>,
    finalized: Arc<AtomicUsize>,
    finish_gate: Option<Gate>,
    drop_gate: Option<Gate>,
}
impl TextDecoder for TrackedDecoder {
    fn push(&mut self, token: u32) -> Result<String, TextError> {
        self.inner.as_mut().unwrap().push(token)
    }
    fn finish(mut self: Box<Self>) -> Result<String, TextError> {
        self.finalized.fetch_add(1, Ordering::SeqCst);
        if let Some((entered, resume)) = &self.finish_gate {
            entered.send(()).unwrap();
            resume.recv().unwrap();
        }
        self.inner.take().unwrap().finish()
    }
}
impl Drop for TrackedDecoder {
    fn drop(&mut self) {
        if let Some((entered, resume)) = &self.drop_gate {
            let _ = entered.send(());
            let _ = resume.recv();
        }
        self.live.fetch_sub(1, Ordering::SeqCst);
    }
}

#[test]
fn a_blocked_finalizer_keeps_its_admission_lease() {
    let (entered, observed) = flume::bounded(1);
    let (resume, blocked) = flume::bounded(1);
    let processor = Arc::new(TrackedProcessor {
        inner: processor(gemma_limits()),
        live: Arc::new(AtomicUsize::new(0)),
        finalized: Arc::new(AtomicUsize::new(0)),
        finish_gate: Some((entered, blocked)),
        constructor_gate: Mutex::new(None),
        drop_gate: Mutex::new(None),
    });
    let harness = Harness::with_processor(processor, TextConfig::default(), 4);
    assert_eq!(harness.model.request_capacity(), 1);
    harness.plan.script(10, &[byte(b'a'), 1]);
    harness.plan.script(11, &[10, 1]);
    let model = harness.model.clone();
    let finishing = std::thread::spawn(move || model.generate_blocking(input(10), options()));
    observed.recv_timeout(Duration::from_secs(3)).unwrap();
    let attempted = harness.model.stream_blocking(input(11), options());
    // Always release/join before asserting, including the failing-before version.
    resume.send(()).unwrap();
    assert_eq!(finishing.join().unwrap().unwrap().text, "a");
    assert!(
        matches!(
            attempted,
            Err(TextError::Admission(DriverError::Overloaded))
        ),
        "decoder still owns its byte run until finalization finishes"
    );
}

#[test]
fn abandoned_prepared_reply_keeps_capacity_until_decoder_drop_finishes() {
    let (constructing, constructed) = flume::bounded(1);
    let (resume_construction, construction_gate) = flume::bounded(1);
    let (dropping, dropped) = flume::bounded(1);
    let (resume_drop, drop_gate) = flume::bounded(1);
    let processor = Arc::new(TrackedProcessor {
        inner: processor(gemma_limits()),
        live: Arc::new(AtomicUsize::new(0)),
        finalized: Arc::new(AtomicUsize::new(0)),
        finish_gate: None,
        constructor_gate: Mutex::new(Some((constructing, construction_gate))),
        drop_gate: Mutex::new(Some((dropping, drop_gate))),
    });
    let harness = Harness::with_processor(processor, TextConfig::default(), 4);
    assert_eq!(harness.model.request_capacity(), 1);
    let mut context = Context::from_waker(Waker::noop());
    let mut preparing = Box::pin(harness.model.stream(input(10), options()));
    assert!(preparing.as_mut().poll(&mut context).is_pending());
    constructed.recv_timeout(Duration::from_secs(3)).unwrap();
    drop(preparing); // The worker's prepared reply now has no recipient.
    resume_construction.send(()).unwrap();
    dropped.recv_timeout(Duration::from_secs(3)).unwrap();
    let mut attempted = Box::pin(harness.model.stream(input(11), options()));
    let status = attempted.as_mut().poll(&mut context);
    drop(attempted);
    resume_drop.send(()).unwrap();
    assert!(
        matches!(
            status,
            Poll::Ready(Err(TextError::Admission(DriverError::Overloaded)))
        ),
        "a dropped reply must free its decoder before returning capacity"
    );
}

#[test]
fn every_terminal_releases_decoder_and_owner_failure_never_flushes() {
    for mode in ["stop", "cancel", "owner", "decode", "finalize"] {
        let live = Arc::new(AtomicUsize::new(0));
        let finalized = Arc::new(AtomicUsize::new(0));
        let processor = processor(ProcessorLimits {
            max_decoded_delta_bytes: if mode == "finalize" { 1 } else { 128 },
            ..gemma_limits()
        });
        let harness = harness(Arc::new(TrackedProcessor {
            inner: processor,
            live: live.clone(),
            finalized: finalized.clone(),
            finish_gate: None,
            constructor_gate: Mutex::new(None),
            drop_gate: Mutex::new(None),
        }));
        let first = if mode == "finalize" {
            byte(0xff)
        } else {
            byte(b'a')
        };
        harness
            .plan
            .script(10, &[first, if mode == "decode" { 268 } else { 1 }]);
        harness.plan.holding(true);
        let mut stream = harness.model.stream_blocking(input(10), options()).unwrap();
        assert!(
            matches!(stream.next_blocking().unwrap().unwrap(), TextEvent::Delta { token: Some(id), ref text } if id == first && text.is_empty())
        );
        assert_eq!(live.load(Ordering::SeqCst), 1);
        if mode == "cancel" {
            stream.cancel();
        }
        if mode == "owner" {
            harness.plan.failing_polls(true);
        }
        harness.plan.holding(false);
        let output = drain_until_error(&mut stream);
        match mode {
            "stop" | "cancel" => {
                assert!(output.error.is_none());
                assert_eq!(output.text, "a");
                assert_eq!(
                    output.terminals,
                    [if mode == "stop" {
                        FinishReason::Stop
                    } else {
                        FinishReason::Cancelled
                    }]
                );
            }
            "owner" => {
                assert!(matches!(output.error, Some(TextError::Owner(_))));
                assert!(output.text.is_empty());
                assert!(output.terminals.is_empty());
            }
            _ => {
                assert!(output.error.unwrap().is_request_local());
                assert!(output.text.is_empty());
                assert!(output.terminals.is_empty());
            }
        }
        // Keep the completed/error stream alive: cleanup must not wait for its Drop.
        assert_eq!(live.load(Ordering::SeqCst), 0, "{mode}");
        assert_eq!(
            finalized.load(Ordering::SeqCst),
            usize::from(matches!(mode, "stop" | "cancel" | "finalize")),
            "{mode}"
        );
        assert!(stream.next_blocking().is_none());
    }
}
