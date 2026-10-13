use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use engine_gguf::{GgufTokenizer, MetadataValue, byte_token_symbol};
use ribn::driver::{Driver, DriverConfig};
use ribn::{
    Admission, BatchItem, BatchPreparation, Engine, EngineConfig, ExecutionError, ExecutorInfo,
    GenerationExecutor, GenerationLimits, RequestId, SchedulePolicy, SequenceId, StepCompletion,
    StepKind, SubmissionId, TokenRequest,
};
use ribn_server::{Server, ServerConfig};
use ribn_text::{GgufProcessor, ProcessorLimits, TextConfig, TextOwner};
mod wire;
use wire::{Reply, post};
pub use wire::{exchange, first_delta, wait_until};

use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

#[derive(Default)]
pub struct Plan {
    pub hold: AtomicBool,
    pub fail: AtomicBool,
    pub fail_sync: AtomicBool,
    pub live: AtomicUsize,
    pub produced: AtomicUsize,
}

struct Fixture {
    info: ExecutorInfo,
    plan: Arc<Plan>,
    rows: HashMap<SequenceId, (u32, u32)>,
    batch: Vec<BatchItem>,
    next: u64,
}

impl GenerationExecutor for Fixture {
    fn info(&self) -> &ExecutorInfo {
        &self.info
    }
    fn admit(
        &mut self,
        _: RequestId,
        sequence: SequenceId,
        request: &TokenRequest,
    ) -> Result<Admission, ExecutionError> {
        self.rows.insert(
            sequence,
            (
                u32::try_from(request.tokens.len()).unwrap(),
                if request.tokens[0] == u32::from(b'b') {
                    256
                } else {
                    u32::from(b'x')
                },
            ),
        );
        self.plan.live.fetch_add(1, Ordering::SeqCst);
        Ok(Admission::Ready)
    }
    fn prepare(&mut self, _: &[BatchItem]) -> Result<BatchPreparation, ExecutionError> {
        Ok(BatchPreparation::Ready)
    }
    fn abandon_preparation(&mut self) -> Result<(), ExecutionError> {
        Ok(())
    }
    fn submit(&mut self, batch: &[BatchItem]) -> Result<SubmissionId, ExecutionError> {
        self.batch = batch.to_vec();
        self.next += 1;
        Ok(SubmissionId::new(self.next))
    }
    fn poll(&mut self, _: SubmissionId) -> Result<Option<Vec<StepCompletion>>, ExecutionError> {
        if self.plan.fail.load(Ordering::SeqCst) {
            return Err(ExecutionError::new("fixture fault"));
        }
        if self.plan.hold.load(Ordering::SeqCst)
            && self.batch.iter().any(|b| b.kind == StepKind::Decode)
        {
            return Ok(None);
        }
        let rows = std::mem::take(&mut self.batch)
            .into_iter()
            .map(|item| {
                let (prompt, token) = self.rows[&item.sequence];
                let prefix = item.prefix + item.token_budget;
                let tokens = if item.kind == StepKind::Decode || prefix == prompt {
                    self.plan.produced.fetch_add(1, Ordering::SeqCst);
                    vec![token]
                } else {
                    Vec::new()
                };
                StepCompletion {
                    sequence: item.sequence,
                    prefix,
                    tokens,
                }
            })
            .collect();
        Ok(Some(rows))
    }
    fn release(&mut self, sequence: SequenceId) -> Result<(), ExecutionError> {
        if self.rows.remove(&sequence).is_some() {
            self.plan.live.fetch_sub(1, Ordering::SeqCst);
        }
        Ok(())
    }
    fn synchronize(&mut self) -> Result<(), ExecutionError> {
        if self.plan.fail_sync.load(Ordering::SeqCst) {
            return Err(ExecutionError::new("fixture drain failure"));
        }
        self.batch.clear();
        self.rows.clear();
        self.plan.live.store(0, Ordering::SeqCst);
        Ok(())
    }
}

pub fn owner(plan: Arc<Plan>, requests: usize) -> TextOwner {
    let mut metadata = BTreeMap::new();
    let mut tokens = (0..=255_u8)
        .map(|b| MetadataValue::String(byte_token_symbol(b).to_string()))
        .collect::<Vec<_>>();
    tokens.push(MetadataValue::String("x".repeat(4096)));
    for (key, value) in [
        ("tokenizer.ggml.model", MetadataValue::String("gpt2".into())),
        ("tokenizer.ggml.pre", MetadataValue::String("qwen35".into())),
        ("tokenizer.ggml.tokens", MetadataValue::Array(tokens)),
        ("tokenizer.ggml.merges", MetadataValue::Array(Vec::new())),
        ("tokenizer.ggml.eos_token_id", MetadataValue::U32(0)),
        ("tokenizer.ggml.bos_token_id", MetadataValue::U32(254)),
        ("tokenizer.ggml.padding_token_id", MetadataValue::U32(253)),
        (
            "tokenizer.ggml.token_type",
            MetadataValue::Array((0..257).map(|_| MetadataValue::I32(1)).collect()),
        ),
        (
            "tokenizer.chat_template",
            MetadataValue::String("{% for m in messages %}{{ m.content }}{% endfor %}".into()),
        ),
    ] {
        metadata.insert(key.to_owned(), value);
    }
    let processor = Arc::new(GgufProcessor::new(
        GgufTokenizer::from_metadata(&metadata).unwrap(),
        ProcessorLimits::default(),
    ));
    let fixture = Fixture {
        info: ExecutorInfo {
            name: "HTTP fixture".into(),
            limits: GenerationLimits {
                context_tokens: 8192,
                max_sequences: requests,
                max_batch_tokens: 8,
                max_decode_tokens: 1,
            },
        },
        plan,
        rows: HashMap::new(),
        batch: Vec::new(),
        next: 0,
    };
    let engine = Engine::new(
        fixture,
        EngineConfig {
            max_active_requests: requests,
            max_queued_requests: requests,
            max_queued_input_tokens: 4096,
            max_buffered_events: requests * 2,
            max_events_per_request: 2,
        },
        SchedulePolicy {
            max_batch_tokens: 8,
            ..SchedulePolicy::default()
        },
    )
    .unwrap();
    let config = DriverConfig {
        max_requests: requests,
        events_per_request: 2,
        ..DriverConfig::for_engine(&engine)
    };
    let (driver, handle) = Driver::spawn(engine, config).unwrap();
    TextOwner::new(
        processor,
        driver,
        handle,
        TextConfig {
            preprocessing_workers: 1,
            batch_window: 2,
        },
    )
    .unwrap()
}

pub struct Running {
    pub address: std::net::SocketAddr,
    stop: oneshot::Sender<()>,
    task: JoinHandle<(Server, Result<(), ribn_server::ServerError>)>,
}
impl Running {
    pub async fn start(mut config: ServerConfig, plan: Arc<Plan>, requests: usize) -> Self {
        config.bind.set_port(0);
        let mut server = Server::bind(config).await.unwrap();
        let address = server.local_addr();
        let owner = owner(plan, requests);
        let (stop, stopped) = oneshot::channel();
        let task = tokio::spawn(async move {
            let result = server
                .run(async { Ok(owner) }, async {
                    let _ = stopped.await;
                })
                .await;
            (server, result)
        });
        let running = Self {
            address,
            stop,
            task,
        };
        wait_until(|| async { running.get("/ready").await.status == 200 }).await;
        running
    }
    pub async fn stop(self) -> (Server, Result<(), ribn_server::ServerError>) {
        self.stop.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(3), self.task)
            .await
            .unwrap()
            .unwrap()
    }
    pub async fn get(&self, path: &str) -> Reply {
        exchange(
            self.address,
            &format!("GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n"),
        )
        .await
    }
    pub async fn post(&self, path: &str, body: &str) -> Reply {
        exchange(self.address, &post(path, body)).await
    }
    pub async fn stream(&self, path: &str, body: &str) -> TcpStream {
        self.stream_with_tail(path, body, "").await
    }
    pub async fn stream_with_tail(&self, path: &str, body: &str, tail: &str) -> TcpStream {
        let mut socket = TcpStream::connect(self.address).await.unwrap();
        socket
            .write_all(format!("{}{tail}", post(path, body)).as_bytes())
            .await
            .unwrap();
        socket
    }
}
