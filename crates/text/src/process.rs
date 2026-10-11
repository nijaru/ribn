//! Text preprocessing above the token runtime.
//!
//! Formatting, tokenization and per-token decoding are model-specific; the
//! runtime only sees encoded tokens. Preparation therefore runs on a bounded
//! worker pool, after the caller has reserved a request permit, so tokenization
//! cannot outgrow the permit pool or occupy an async executor thread.

use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;

use engine_gguf::{ChatMessage, ChatTemplateOptions, GgufTokenizer};
use flume::{Receiver, Sender};
use ribn::GenerationOptions;
use ribn::TokenRequest;
use ribn::driver::RequestPermit;

use crate::error::TextError;
use crate::input::{Message, TextInput};

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
    /// Decoded bytes a single token may contribute.
    pub max_decoded_token_bytes: usize,
}

impl Default for ProcessorLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: 256 * 1024,
            max_retained_bytes: 512 * 1024,
            max_rendered_bytes: 512 * 1024,
            max_prompt_tokens: 65_536,
            max_decoded_token_bytes: 4 * 1024,
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
/// encoded prompt: a chat request stops at the conversation delimiters its own
/// rendered prompt contains. The caller's explicit stop tokens are added to
/// these, never replaced by them.
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
/// streams, so they must be stateless per call or internally synchronized.
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

    /// Replace `out` with one token's decoded bytes.
    ///
    /// # Errors
    /// Returns a request-local error for an unknown token, an unsupported
    /// vocabulary symbol, or a token over the decoded byte bound.
    fn decode_token_into(&self, token: u32, out: &mut Vec<u8>) -> Result<(), TextError>;
}

/// GGUF tokenizer and chat template with enforced text bounds.
pub struct GgufProcessor {
    tokenizer: GgufTokenizer,
    limits: ProcessorLimits,
}

impl GgufProcessor {
    #[must_use]
    pub const fn new(tokenizer: GgufTokenizer, limits: ProcessorLimits) -> Self {
        Self { tokenizer, limits }
    }

    fn render_chat(&self, messages: &[ChatMessage]) -> Result<String, TextError> {
        let mut rendered = LimitedText::new(self.limits.max_rendered_bytes);
        // Preserve the existing Qwen CLI behavior: generation prompt on,
        // thinking off until reasoning controls have an explicit public API.
        let options = ChatTemplateOptions::new(true, false);
        match self
            .tokenizer
            .render_chat_to(messages, options, &mut rendered)
        {
            Ok(()) => Ok(rendered.text),
            Err(_) if rendered.exceeded.is_some() => Err(rendered.exceeded_error()),
            Err(error) => Err(TextError::Processor(error)),
        }
    }
}

impl TextProcessor for GgufProcessor {
    fn limits(&self) -> ProcessorLimits {
        self.limits
    }

    fn encode(&self, input: TextInput) -> Result<EncodedInput, TextError> {
        // The retention boundary checks the same rule before queuing; this keeps
        // the processor authoritative for direct callers.
        self.limits.check_input(&input)?;
        let chat = matches!(&input, TextInput::Chat(_));
        let tokens = match input {
            TextInput::Tokens(tokens) => tokens,
            TextInput::Prompt(text) => {
                self.tokenizer.encode(&text).map_err(TextError::Processor)?
            }
            TextInput::Chat(messages) => {
                if messages.is_empty() {
                    return Err(TextError::InvalidInput(
                        "chat input must contain at least one message",
                    ));
                }
                let messages = messages
                    .into_iter()
                    .map(|message| ChatMessage::new(message.role, message.content))
                    .collect::<Vec<_>>();
                let rendered = self.render_chat(&messages)?;
                self.tokenizer
                    .encode_rendered(&rendered)
                    .map_err(TextError::Processor)?
            }
        };
        if tokens.len() > self.limits.max_prompt_tokens {
            return Err(TextError::limit(
                "prompt tokens",
                self.limits.max_prompt_tokens,
                tokens.len(),
            ));
        }
        // A chat turn ends at the artifact's declared EOS IDs and at the
        // delimiters its own prompt used; raw prompts and token inputs carry no
        // template structure to delimit.
        let stop_tokens = if chat {
            self.tokenizer.chat_stop_token_ids(&tokens)
        } else {
            self.tokenizer.stop_token_ids()
        };
        Ok(EncodedInput::new(tokens, stop_tokens))
    }

    fn decode_token_into(&self, token: u32, out: &mut Vec<u8>) -> Result<(), TextError> {
        self.tokenizer
            .decode_token_into(token, out, self.limits.max_decoded_token_bytes)
            .map_err(|source| TextError::Decode { token, source })
    }
}

/// A writer that fails instead of retaining more than its byte bound.
struct LimitedText {
    text: String,
    limit: usize,
    /// Attempted size at the point the bound was crossed, for diagnostics.
    exceeded: Option<usize>,
}

impl LimitedText {
    const fn new(limit: usize) -> Self {
        Self {
            text: String::new(),
            limit,
            exceeded: None,
        }
    }

    fn exceeded_error(&self) -> TextError {
        TextError::limit(
            "rendered prompt bytes",
            self.limit,
            self.exceeded.unwrap_or(self.limit),
        )
    }
}

impl io::Write for LimitedText {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.limit.saturating_sub(self.text.len()) {
            self.exceeded = Some(self.text.len().saturating_add(bytes.len()));
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "rendered prompt exceeds its byte bound",
            ));
        }
        // Templates render text; a non-UTF-8 write is a template defect and must
        // not be silently lossy.
        let text = std::str::from_utf8(bytes).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "template emitted invalid UTF-8")
        })?;
        self.text.push_str(text);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Encoded input that still owns its admission permit.
pub(crate) struct Prepared {
    pub(crate) request: TokenRequest,
    pub(crate) permit: RequestPermit,
}

struct Job {
    input: TextInput,
    options: GenerationOptions,
    permit: RequestPermit,
    reply: Sender<Result<Prepared, TextError>>,
}

struct PoolState {
    jobs: Sender<Job>,
    closed: Arc<AtomicBool>,
    limits: ProcessorLimits,
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
    closed: &'a AtomicBool,
    live: &'a AtomicUsize,
}

impl Drop for Worker<'_> {
    fn drop(&mut self) {
        if self.live.fetch_sub(1, Ordering::SeqCst) != 1 {
            return;
        }
        self.closed.store(true, Ordering::SeqCst);
        // Dropping a queued job drops its reply sender, which reports a closed
        // pool to its caller and returns that permit.
        while self.jobs.try_recv().is_ok() {}
    }
}

impl Pool {
    /// Start `workers` preprocessing threads with a queue bounded by `queue`.
    pub(crate) fn start(processor: &Arc<dyn TextProcessor>, workers: usize, queue: usize) -> Self {
        let (jobs, receiver) = flume::bounded(queue);
        let closed = Arc::new(AtomicBool::new(false));
        let live = Arc::new(AtomicUsize::new(workers));
        for index in 0..workers {
            let processor = Arc::clone(processor);
            let worker_jobs = receiver.clone();
            let worker_closed = closed.clone();
            let worker_live = live.clone();
            let started = thread::Builder::new()
                .name(format!("ribn-preprocess-{index}"))
                .spawn(move || {
                    let _worker = Worker {
                        jobs: &worker_jobs,
                        closed: &worker_closed,
                        live: &worker_live,
                    };
                    run_worker(&*processor, &worker_jobs, &worker_closed);
                });
            if started.is_err() {
                // A worker that never starts must still be accounted, or a lost
                // thread would keep the pool looking alive forever.
                drop(Worker {
                    jobs: &receiver,
                    closed: &closed,
                    live: &live,
                });
            }
        }
        drop(receiver);
        Self {
            state: Arc::new(PoolState {
                jobs,
                closed,
                limits: processor.limits(),
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
        self.state.closed.store(true, Ordering::SeqCst);
    }
}

impl PoolClient {
    pub(crate) fn is_closed(&self) -> bool {
        self.state.closed.load(Ordering::SeqCst)
    }

    /// Queue preparation under an existing permit.
    ///
    /// Input and options must satisfy their joint retained-allocation envelope
    /// before queuing; aggregate heap capacity is permits times that envelope. The queue
    /// length cannot exceed the permit pool either, because every queued job owns
    /// one permit.
    pub(crate) fn submit(
        &self,
        input: TextInput,
        options: GenerationOptions,
        permit: RequestPermit,
    ) -> Result<Receiver<Result<Prepared, TextError>>, TextError> {
        self.state.limits.check_request(&input, &options)?;
        // Even without prompt tokens, these options must fit the driver's stop
        // storage plus runtime-copy allowance. Do not queue an impossible job.
        permit
            .check_input(0, &options)
            .map_err(TextError::Admission)?;
        if self.is_closed() {
            return Err(TextError::Closed);
        }
        let (reply, receiver) = flume::bounded(1);
        self.state
            .jobs
            .try_send(Job {
                input,
                options,
                permit,
                reply,
            })
            .map_err(|_| TextError::Closed)?;
        Ok(receiver)
    }
}

fn run_worker(processor: &dyn TextProcessor, jobs: &Receiver<Job>, closed: &AtomicBool) {
    while let Ok(job) = jobs.recv() {
        if closed.load(Ordering::SeqCst) {
            // Shutdown or pool failure already began. Do not start queued work:
            // dropping the job reports a closed pool and returns its permit.
            continue;
        }
        let result = prepare(processor, job.input, job.options, job.permit);
        // A dropped caller means the result, and its permit, are released here.
        let _ = job.reply.try_send(result);
    }
}

fn prepare(
    processor: &dyn TextProcessor,
    input: TextInput,
    mut options: GenerationOptions,
    permit: RequestPermit,
) -> Result<Prepared, TextError> {
    let (tokens, stop_tokens) = processor.encode(input)?.into_parts();
    for stop in stop_tokens {
        if !options.stop_tokens.contains(&stop) {
            options.stop_tokens.push(stop);
        }
    }
    // Model-added stop IDs can grow the vector. Validate before retaining a
    // prepared reply, not only when its consumer eventually submits to the driver.
    permit
        .check_input(tokens.len(), &options)
        .map_err(TextError::Admission)?;
    Ok(Prepared {
        request: TokenRequest::new(tokens, options),
        permit,
    })
}
