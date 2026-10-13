use std::time::Duration;

use ribn_server::{Server, ServerConfig};
use ribn_text::{LoadOptions, TextError, TextOwner};

use crate::serve_cli::{self, USAGE};

pub(crate) fn run(arguments: &[String]) -> Result<(), String> {
    if matches!(arguments, [help] if matches!(help.as_str(), "-h" | "--help")) {
        println!(
            "{USAGE}\n\nExperimental Qwen GGUF/CUDA HTTP server; loopback only, greedy decoding.\nSee docs/serving.md for the supported completion/chat subset and bounds."
        );
        return Ok(());
    }
    let options = serve_cli::parse(arguments)?;
    let load = LoadOptions {
        device: options.device,
        context_tokens: options.context_length,
        max_sequences: options
            .max_sequences
            .unwrap_or(LoadOptions::default().max_sequences),
        ..LoadOptions::default()
    };
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
        // CUDA load settles that load into an owner and then cleans it up.
        let shutdown = shutdown_signal()?;
        let mut server = Server::bind(config).await.map_err(|e| e.to_string())?;
        eprintln!(
            "listening on http://{}; loading model (experimental Qwen/CUDA)",
            server.local_addr()
        );
        let loading = async move {
            tokio::task::spawn_blocking(move || {
                let (owner, memory) = TextOwner::load(options.model, load)?;
                eprintln!(
                    "ready: {} bytes reserved for sequence state",
                    memory.reserved_sequence_bytes
                );
                Ok(owner)
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
