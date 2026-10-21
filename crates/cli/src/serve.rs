use std::time::Duration;

use ribn_server::{Server, ServerConfig};
#[cfg(feature = "cuda")]
use ribn_text::LoadOptions;
use ribn_text::{TextError, TextOwner};

use crate::serve_cli::{self, ModelLoad, ServeModel};

pub(crate) fn run(arguments: &[String], kind: ServeModel) -> Result<(), String> {
    if matches!(arguments, [help] if matches!(help.as_str(), "-h" | "--help")) {
        let model = match kind {
            ServeModel::Qwen => "Qwen GGUF/CUDA",
            ServeModel::Gemma => {
                "Gemma 4 package, using the same Metal/CUDA selection as run-gemma"
            }
        };
        println!(
            "{}\n\nExperimental {model} HTTP server; loopback only, greedy decoding.\nSee docs/serving.md for the supported completion/chat subset and bounds.",
            kind.usage()
        );
        return Ok(());
    }
    let options = serve_cli::parse(arguments, kind)?;
    if kind == ServeModel::Qwen && !cfg!(feature = "cuda") {
        return Err("Qwen serving requires building ribn with --features cuda".to_owned());
    }
    let defaults = ServerConfig::default();
    let config = ServerConfig {
        bind: options.bind,
        model_name: options.model_name,
        max_connections: options.max_connections.unwrap_or(defaults.max_connections),
        max_requests: options.max_requests.unwrap_or(defaults.max_requests),
        max_body_bytes: options.max_body_bytes.unwrap_or(defaults.max_body_bytes),
        max_response_bytes: options
            .max_response_bytes
            .unwrap_or(defaults.max_response_bytes),
        max_output_tokens: options
            .max_output_tokens
            .unwrap_or(defaults.max_output_tokens),
        connection_deadline: options
            .deadline_secs
            .map_or(defaults.connection_deadline, |n| {
                Duration::from_secs(u64::from(n))
            }),
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    runtime.block_on(async move {
        // Register signal handlers before loading so shutdown during a blocking
        // device load settles that load into an owner and then cleans it up.
        let shutdown = shutdown_signal()?;
        let http_slots = config.max_requests;
        let mut server = Server::bind(config).await.map_err(|e| e.to_string())?;
        eprintln!(
            "listening on http://{}; HTTP request slots={http_slots}; loading model (experimental Ribn runtime)",
            server.local_addr()
        );
        let loading = async move {
            tokio::task::spawn_blocking(move || {
                load_model(options.model, options.load)
            })
            .await
            .map_err(|_| TextError::InvalidInput("model loading task panicked"))?
        };
        let result = server.run(loading, shutdown).await;
        if let Err(error) = result {
            // Retain the same owner and make one explicit cleanup retry. Neither
            // a failure nor dropping the owner authorizes early device reuse.
            if let Err(cleanup) = server.shutdown_async().await {
                return Err(format!("{error}; shutdown retry also failed: {cleanup}"));
            }
            return Err(error.to_string());
        }
        Ok(())
    })
}

fn load_model(path: std::path::PathBuf, load: ModelLoad) -> Result<TextOwner, TextError> {
    match load {
        #[cfg(feature = "cuda")]
        ModelLoad::Qwen {
            device,
            context_length,
            max_sequences,
        } => {
            let load = LoadOptions {
                device,
                context_tokens: context_length,
                max_sequences: max_sequences.unwrap_or(LoadOptions::default().max_sequences),
                ..LoadOptions::default()
            };
            let (owner, memory) = TextOwner::load(path, load)?;
            eprintln!(
                "ready: active sequence limit={}; application request capacity={}; sequence state reservation={} bytes",
                load.max_sequences,
                owner.model().request_capacity(),
                memory.reserved_sequence_bytes
            );
            Ok(owner)
        }
        #[cfg(not(feature = "cuda"))]
        ModelLoad::Qwen { .. } => Err(TextError::InvalidInput(
            "Qwen serving requires --features cuda",
        )),
        ModelLoad::Gemma => {
            let (owner, report) = crate::gemma::load_model(path)?;
            eprintln!(
                "ready: application request capacity={}; encoded model bytes={} (not resident or peak bytes)",
                owner.model().request_capacity(),
                report.model_bytes
            );
            Ok(owner)
        }
    }
}

#[cfg(unix)]
fn shutdown_signal() -> Result<impl std::future::Future<Output = ()>, String> {
    use tokio::signal::unix::{SignalKind, signal};
    let mut interrupt = signal(SignalKind::interrupt()).map_err(|e| e.to_string())?;
    let mut terminate = signal(SignalKind::terminate()).map_err(|e| e.to_string())?;
    Ok(async move {
        tokio::select! { _ = interrupt.recv() => {}, _ = terminate.recv() => {} }
    })
}

#[cfg(not(unix))]
fn shutdown_signal() -> Result<impl std::future::Future<Output = ()>, String> {
    // Constructing/polling ctrl_c registers the platform handler.
    Ok(async {
        if let Err(error) = tokio::signal::ctrl_c().await {
            eprintln!("signal handler failed: {error}");
        }
    })
}
