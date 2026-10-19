//! Text preprocessing above the token runtime.
//!
//! Formatting, tokenization and per-token decoding are model-specific; the
//! runtime only sees encoded tokens. Preparation therefore runs on a bounded
//! worker pool, after the caller has reserved a request permit, so tokenization
//! cannot outgrow the permit pool or occupy an async executor thread.

use std::sync::{Arc, Mutex};
use std::thread;

use flume::{Receiver, Sender};
use ribn::GenerationOptions;
use ribn::TokenRequest;
use ribn::driver::{DeliveryLease, RequestPermit};

use crate::error::TextError;
use crate::input::{Message, TextInput};
use crate::model::TextRequest;

/// Separate bounds for input payload, owned allocations, rendering and decode output.
///
/// These are text-layer limits. The token driver keeps its own encoded-input
/// envelope, and a request must satisfy both.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessorLimits {
    /// Raw prompt text, summed message role/content, or token payload bytes.
    pub max_input_bytes: usize,
    /// Owned input and generation-option heap capacity before preprocessing.
    /// Includes chat-container storage, string/vector spare capacity and stop IDs;
    /// fixed job objects and allocator overhead are not part of this envelope.
    pub max_retained_bytes: usize,
    /// Rendered chat-template bytes per request.
    pub max_rendered_bytes: usize,
    /// Encoded prompt tokens per request.
    pub max_prompt_tokens: usize,
    /// Decoded vocabulary-piece bytes, before delayed-run expansion/fusion.
    pub max_decoded_token_bytes: usize,
    /// Bytes retained across token events by one decoder.
    pub max_decode_pending_bytes: usize,
    /// Bytes emitted in one delta, including a delayed run or terminal flush.
    pub max_decoded_delta_bytes: usize,
}

impl Default for ProcessorLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: 256 * 1024,
            max_retained_bytes: 512 * 1024,
            max_rendered_bytes: 512 * 1024,
            max_prompt_tokens: 65_536,
            max_decoded_token_bytes: 4 * 1024,
            max_decode_pending_bytes: 65_536,
            max_decoded_delta_bytes: 65_536,
        }
    }
}

impl ProcessorLimits {
    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.max_input_bytes > 0
            && self.max_retained_bytes > 0
            && self.max_rendered_bytes > 0
            && self.max_prompt_tokens > 0
            && self.max_decoded_token_bytes > 0
            && self.max_decode_pending_bytes > 0
            && self.max_decoded_delta_bytes > 0
    }

    /// Heap capacity retained by an owned input, including chat-container storage.
    /// This excludes fixed job objects and allocator overhead, not spare capacity.
    #[must_use]
    pub fn retained_bytes(input: &TextInput) -> usize {
        match input {
            TextInput::Prompt(text) => text.capacity(),
            TextInput::Chat(messages) => messages.iter().fold(
                messages.capacity().saturating_mul(size_of::<Message>()),
                |total, message| {
                    total
                        .saturating_add(message.role.capacity())
                        .saturating_add(message.content.capacity())
                },
            ),
            TextInput::Tokens(tokens) => tokens.capacity().saturating_mul(size_of::<u32>()),
        }
    }

    /// Validate payload and owned allocation bounds without copying or shrinking.
    /// Excess capacity rejects rather than introducing a normalization peak.
    ///
    /// # Errors
    /// Returns a request-local limit error.
    pub fn check_input(self, input: &TextInput) -> Result<(), TextError> {
        // Reject container-only overflow before walking arbitrarily many empty
        // messages. Once this fits, the number of messages inspected is bounded.
        if let TextInput::Chat(messages) = input {
            let bytes = messages.capacity().saturating_mul(size_of::<Message>());
            self.check_retained("retained input bytes", bytes)?;
        }
        let bytes = match input {
            TextInput::Prompt(text) => text.len(),
            TextInput::Chat(messages) => messages.iter().fold(0usize, |total, message| {
                total
                    .saturating_add(message.role.len())
                    .saturating_add(message.content.len())
            }),
            TextInput::Tokens(tokens) => tokens.len().saturating_mul(size_of::<u32>()),
        };
        if bytes > self.max_input_bytes {
            return Err(TextError::limit("input bytes", self.max_input_bytes, bytes));
        }
        self.check_retained("retained input bytes", Self::retained_bytes(input))
    }

    fn check_retained(self, field: &'static str, bytes: usize) -> Result<(), TextError> {
        if bytes > self.max_retained_bytes {
            return Err(TextError::limit(field, self.max_retained_bytes, bytes));
        }
        Ok(())
    }

    fn check_request(
        self,
        input: &TextInput,
        options: &GenerationOptions,
    ) -> Result<(), TextError> {
        self.check_input(input)?;
        let bytes = Self::retained_bytes(input).checked_add(
            options
                .stop_tokens
                .capacity()
                .saturating_mul(size_of::<u32>()),
        );
        match bytes {
            Some(bytes) => self.check_retained("retained request bytes", bytes),
            None => Err(TextError::limit(
                "retained request bytes",
                self.max_retained_bytes,
                usize::MAX,
            )),
        }
    }
}

/// Tokenized input plus the stop IDs the request must honor.
///
/// The processor reports both together because the stop set depends on the
/// model and input. For example, GGUF chat stops at the conversation delimiters
/// its own rendered prompt contains; Gemma uses generation metadata. Caller stop
/// tokens are added to these, never replaced by them.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedInput {
    tokens: Vec<u32>,
    stop_tokens: Vec<u32>,
}

impl EncodedInput {
    #[must_use]
    pub const fn new(tokens: Vec<u32>, stop_tokens: Vec<u32>) -> Self {
        Self {
            tokens,
            stop_tokens,
        }
    }

    #[must_use]
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    #[must_use]
    pub fn stop_tokens(&self) -> &[u32] {
        &self.stop_tokens
    }

    #[must_use]
    pub fn into_parts(self) -> (Vec<u32>, Vec<u32>) {
        (self.tokens, self.stop_tokens)
    }
}

/// Model-specific text preparation and incremental decoding.
///
/// Implementations are shared across preprocessing workers and consuming
/// streams. Request-local state belongs in a separately owned decoder.
pub trait TextProcessor: Send + Sync + 'static {
    #[must_use]
    fn limits(&self) -> ProcessorLimits;

    /// Format and tokenize one input, reporting the stop IDs the request must
    /// honor. Called only after admission.
    ///
    /// # Errors
    /// Returns a request-local error for an invalid, oversized, unrenderable or
    /// untokenizable input.
    fn encode(&self, input: TextInput) -> Result<EncodedInput, TextError>;

    /// Construct request-local decoding under the permit, before runtime enqueue.
    /// Immutable vocabulary may be shared; enforce the declared pending/delta bounds.
    /// # Errors
    /// Returns a request-local error for unsupported output budgets or limits.
    fn decoder(self: Arc<Self>, output_tokens: usize) -> Result<Box<dyn TextDecoder>, TextError>;
}

/// Request-owned model semantics. Fragments must be irrevocable; every normal or
/// cancelled terminal consumes the finalizer, while owner failure drops it unflushed.
pub trait TextDecoder: Send {
    /// # Errors
    /// A request-local token, encoding or output-bound failure.
    fn push(&mut self, token: u32) -> Result<String, TextError>;
    /// # Errors
    /// A request-local finalization or output-bound failure.
    fn finish(self: Box<Self>) -> Result<String, TextError>;
}

/// Decoder state and its existing grant travel together, including cancelled
/// acknowledgement waits. Field order frees state before capacity on implicit Drop.
pub(crate) struct RequestDecoder {
    inner: Box<dyn TextDecoder>,
    delivery: DeliveryLease,
}
impl RequestDecoder {
    pub(crate) fn push(&mut self, token: u32) -> Result<String, TextError> {
        self.inner.push(token)
    }
    pub(crate) fn finish(self) -> Result<String, TextError> {
        let result = self.inner.finish();
        drop(self.delivery);
        result
    }
}

/// Encoded input that still owns its admission permit.
pub(crate) struct Prepared {
    pub(crate) request: TokenRequest,
    pub(crate) permit: RequestPermit,
    pub(crate) decoder: RequestDecoder,
    pub(crate) stops: Vec<String>,
}

struct Job {
    request: TextRequest,
    permit: RequestPermit,
    reply: Sender<Result<Prepared, TextError>>,
}

/// Serializes queue insertion with closure and the final worker's drain.
struct Serviceability {
    closed: bool,
    live: usize,
}

struct PoolState {
    jobs: Sender<Job>,
    serviceability: Arc<Mutex<Serviceability>>,
    limits: ProcessorLimits,
    #[cfg(test)]
    hooks: Arc<tests::Hooks>,
}

/// Owner of the preprocessing workers. Workers exit once every client and this
/// owner have released the queue.
pub(crate) struct Pool {
    state: Arc<PoolState>,
}

/// Cloneable preprocessing submission handle.
#[derive(Clone)]
pub(crate) struct PoolClient {
    state: Arc<PoolState>,
}

/// Reports one worker leaving, including through a panic. The last worker to
/// leave makes the pool observably dead: it refuses new work and drops queued
/// jobs, so no caller waits on a queue nothing will serve.
struct Worker<'a> {
    jobs: &'a Receiver<Job>,
    serviceability: &'a Mutex<Serviceability>,
    #[cfg(test)]
    hooks: &'a tests::Hooks,
}

impl Drop for Worker<'_> {
    fn drop(&mut self) {
        let mut serviceability = self
            .serviceability
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        serviceability.live -= 1;
        if serviceability.live != 0 {
            return;
        }
        serviceability.closed = true;
        // Closure excludes insertion even after releasing this lock. Dropping
        // replies can invoke caller wakers, so never drain under the state lock.
        drop(serviceability);
        while self.jobs.try_recv().is_ok() {}
        #[cfg(test)]
        tests::pause(&self.hooks.after_drain);
    }
}

impl Pool {
    /// Start `workers` preprocessing threads with a queue bounded by `queue`.
    pub(crate) fn start(processor: &Arc<dyn TextProcessor>, workers: usize, queue: usize) -> Self {
        let (jobs, receiver) = flume::bounded(queue);
        let serviceability = Arc::new(Mutex::new(Serviceability {
            closed: false,
            live: workers,
        }));
        #[cfg(test)]
        let hooks = Arc::new(tests::Hooks::default());
        for index in 0..workers {
            let processor = Arc::clone(processor);
            let worker_jobs = receiver.clone();
            let worker_serviceability = serviceability.clone();
            #[cfg(test)]
            let worker_hooks = hooks.clone();
            let started = thread::Builder::new()
                .name(format!("ribn-preprocess-{index}"))
                .spawn(move || {
                    let _worker = Worker {
                        jobs: &worker_jobs,
                        serviceability: &worker_serviceability,
                        #[cfg(test)]
                        hooks: &worker_hooks,
                    };
                    run_worker(&processor, &worker_jobs, &worker_serviceability);
                });
            if started.is_err() {
                // A worker that never starts must still be accounted, or a lost
                // thread would keep the pool looking alive forever.
                drop(Worker {
                    jobs: &receiver,
                    serviceability: &serviceability,
                    #[cfg(test)]
                    hooks: &hooks,
                });
            }
        }
        drop(receiver);
        Self {
            state: Arc::new(PoolState {
                jobs,
                serviceability,
                limits: processor.limits(),
                #[cfg(test)]
                hooks,
            }),
        }
    }

    pub(crate) fn client(&self) -> PoolClient {
        PoolClient {
            state: self.state.clone(),
        }
    }

    /// Reject new preprocessing; in-flight jobs finish and return their permits.
    pub(crate) fn close(&self) {
        self.state
            .serviceability
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .closed = true;
    }
}

impl PoolClient {
    pub(crate) fn is_closed(&self) -> bool {
        self.state
            .serviceability
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .closed
    }

    /// Queue preparation under an existing permit.
    ///
    /// Input and options must satisfy their joint retained-allocation envelope
    /// before queuing; aggregate heap capacity is permits times that envelope. The queue
    /// length cannot exceed the permit pool either, because every queued job owns
    /// one permit.
    pub(crate) fn submit(
        &self,
        request: TextRequest,
        permit: RequestPermit,
    ) -> Result<Receiver<Result<Prepared, TextError>>, TextError> {
        self.state
            .limits
            .check_request(&request.input, &request.options)?;
        crate::stream::validate_stops(&request.stops)?;
        // Even without prompt tokens, these options must fit the driver's stop
        // storage plus runtime-copy allowance. Do not queue an impossible job.
        permit
            .check_input(0, &request.options)
            .map_err(TextError::Admission)?;
        #[cfg(test)]
        tests::pause(&self.state.hooks.before_submit);
        let serviceability = self
            .state
            .serviceability
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if serviceability.closed {
            return Err(TextError::Closed);
        }
        let (reply, receiver) = flume::bounded(1);
        self.state
            .jobs
            .try_send(Job {
                request,
                permit,
                reply,
            })
            .map_err(|_| TextError::Closed)?;
        Ok(receiver)
    }
}

fn run_worker(
    processor: &Arc<dyn TextProcessor>,
    jobs: &Receiver<Job>,
    serviceability: &Mutex<Serviceability>,
) {
    while let Ok(job) = jobs.recv() {
        if serviceability
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .closed
        {
            // Shutdown or pool failure already began. Do not start queued work:
            // dropping the job reports a closed pool and returns its permit.
            continue;
        }
        let result = prepare(processor, job.request, job.permit);
        // A dropped caller means the result, and its permit, are released here.
        let _ = job.reply.try_send(result);
    }
}

fn prepare(
    processor: &Arc<dyn TextProcessor>,
    request: TextRequest,
    permit: RequestPermit,
) -> Result<Prepared, TextError> {
    let TextRequest {
        input,
        mut options,
        stops,
        ignore_eos,
    } = request;
    let (tokens, stop_tokens) = processor.encode(input)?.into_parts();
    if !ignore_eos {
        for stop in stop_tokens {
            if !options.stop_tokens.contains(&stop) {
                options.stop_tokens.push(stop);
            }
        }
    }
    // Model-added stop IDs can grow the vector. Validate before retaining a
    // prepared reply, not only when its consumer eventually submits to the driver.
    permit
        .check_input(tokens.len(), &options)
        .map_err(TextError::Admission)?;
    let output_tokens = usize::try_from(options.max_output_tokens)
        .map_err(|_| TextError::InvalidInput("output token budget exceeds address space"))?;
    let decoder = RequestDecoder {
        inner: processor.clone().decoder(output_tokens)?,
        delivery: permit.retain_delivery(),
    };
    Ok(Prepared {
        request: TokenRequest::new(tokens, options),
        permit,
        decoder,
        stops,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::time::Duration;

    use ribn::driver::{Driver, DriverConfig};
    use ribn::{
        Admission, BatchItem, BatchPreparation, Engine, EngineConfig, ExecutionError, ExecutorInfo,
        GenerationExecutor, GenerationLimits, RequestId, SchedulePolicy, SequenceId,
        StepCompletion, SubmissionId,
    };

    use super::*;

    const TIMEOUT: Duration = Duration::from_secs(5);
    type Pause = Mutex<Option<(Sender<()>, Receiver<()>)>>;

    #[derive(Default)]
    pub(super) struct Hooks {
        pub(super) before_submit: Pause,
        pub(super) after_drain: Pause,
    }

    pub(super) fn pause(hook: &Pause) {
        let gate = hook.lock().unwrap().take();
        if let Some((entered, resume)) = gate {
            // A failed test drops its endpoints. Never panic during worker unwind.
            let _ = entered.send(());
            let _ = resume.recv();
        }
    }

    fn gate(hook: &Pause) -> (Receiver<()>, Sender<()>) {
        let (entered, observed) = flume::bounded(1);
        let (release, resume) = flume::bounded(1);
        *hook.lock().unwrap() = Some((entered, resume));
        (observed, release)
    }

    struct PanickingProcessor {
        entered: Sender<()>,
        release: Receiver<()>,
    }

    impl TextProcessor for PanickingProcessor {
        fn limits(&self) -> ProcessorLimits {
            ProcessorLimits::default()
        }

        fn encode(&self, _: TextInput) -> Result<EncodedInput, TextError> {
            self.entered.send(()).unwrap();
            self.release.recv().unwrap();
            panic!("fixture processor panic");
        }

        fn decoder(self: Arc<Self>, _: usize) -> Result<Box<dyn TextDecoder>, TextError> {
            unreachable!("fixture encoding panics before decoder creation")
        }
    }

    /// Supplies real admission permits; no request reaches token execution.
    struct IdleExecutor(ExecutorInfo);

    impl GenerationExecutor for IdleExecutor {
        fn info(&self) -> &ExecutorInfo {
            &self.0
        }

        fn admit(
            &mut self,
            _: RequestId,
            _: SequenceId,
            _: &TokenRequest,
        ) -> Result<Admission, ExecutionError> {
            unreachable!("no token request submitted")
        }

        fn prepare(&mut self, _: &[BatchItem]) -> Result<BatchPreparation, ExecutionError> {
            unreachable!("no token request submitted")
        }

        fn abandon_preparation(&mut self) -> Result<(), ExecutionError> {
            Ok(())
        }

        fn submit(&mut self, _: &[BatchItem]) -> Result<SubmissionId, ExecutionError> {
            unreachable!("no token request submitted")
        }

        fn poll(&mut self, _: SubmissionId) -> Result<Option<Vec<StepCompletion>>, ExecutionError> {
            unreachable!("no token request submitted")
        }

        fn release(&mut self, _: SequenceId) -> Result<(), ExecutionError> {
            unreachable!("no token request submitted")
        }

        fn synchronize(&mut self) -> Result<(), ExecutionError> {
            Ok(())
        }
    }

    #[test]
    fn final_worker_drain_cannot_strand_a_racing_submission() {
        let engine = Engine::new(
            IdleExecutor(ExecutorInfo {
                name: "permit fixture".into(),
                limits: GenerationLimits {
                    context_tokens: 4096,
                    max_sequences: 8,
                    max_batch_tokens: 128,
                    max_decode_tokens: 1,
                },
            }),
            EngineConfig::default(),
            SchedulePolicy::default(),
        )
        .unwrap();
        let (mut owner, handle) = Driver::spawn(
            engine,
            DriverConfig {
                max_requests: 3,
                ..DriverConfig::default()
            },
        )
        .unwrap();
        let (entered, encoding) = flume::bounded(1);
        let (release, resume) = flume::bounded(1);
        let processor: Arc<dyn TextProcessor> = Arc::new(PanickingProcessor {
            entered,
            release: resume,
        });
        let pool = Pool::start(&processor, 1, 3);
        let client = pool.client();
        let submit = |client: &PoolClient| {
            client.submit(
                TextRequest::new(TextInput::prompt("input"), GenerationOptions::default()),
                handle.try_reserve().unwrap(),
            )
        };
        let panicking = submit(&client).unwrap();
        encoding.recv_timeout(TIMEOUT).unwrap();
        let queued = submit(&client).unwrap();
        let (submitting, send) = gate(&pool.state.hooks.before_submit);
        let (drained, exit) = gate(&pool.state.hooks.after_drain);
        let peer = client.clone();
        let permit = handle.try_reserve().unwrap();
        let racing = thread::spawn(move || {
            peer.submit(
                TextRequest::new(TextInput::prompt("racing"), GenerationOptions::default()),
                permit,
            )
        });
        submitting.recv_timeout(TIMEOUT).unwrap();

        // Finish the last worker's drain while retaining its receiver. A stale
        // open check used to allow insertion in this exact window.
        release.send(()).unwrap();
        drained.recv_timeout(TIMEOUT).unwrap();
        assert!(client.is_closed());
        assert!(matches!(
            panicking.recv_timeout(TIMEOUT),
            Err(flume::RecvTimeoutError::Disconnected)
        ));
        assert!(matches!(
            queued.recv_timeout(TIMEOUT),
            Err(flume::RecvTimeoutError::Disconnected)
        ));
        send.send(()).unwrap();
        let result = racing.join().unwrap();
        exit.send(()).unwrap();
        match result {
            Err(TextError::Closed) => {}
            Ok(reply) => assert!(
                matches!(
                    reply.recv_timeout(TIMEOUT),
                    Err(flume::RecvTimeoutError::Disconnected)
                ),
                "submission after final drain stranded its reply and permit"
            ),
            Err(error) => panic!("unexpected submission error: {error:?}"),
        }
        // Keep clients alive: dropping senders must not be necessary for permits
        // to return. Together these reservations exhaust the original bound.
        assert!(matches!(submit(&client), Err(TextError::Closed)));
        let _permits = (0..3)
            .map(|_| {
                handle
                    .try_reserve()
                    .expect("preprocessing returned its permit")
            })
            .collect::<Vec<_>>();
        owner.shutdown().unwrap();
    }
}
