//! Actual HTTP entry point over CUDA; direct facade comparisons qualify transport
//! parity/ownership, not independent numerical accuracy or competitive throughput.
#![cfg(feature = "cuda")]

#[path = "support/wire.rs"]
mod wire;

use std::net::SocketAddr;

use ribn_server::{Readiness, Server, ServerConfig};
use ribn_text::{GenerationOptions, LoadOptions, Message, TextInput, TextOwner, TextResponse};
use serde_json::{Value, json};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use tokio::sync::oneshot;
use wire::{exchange, first_delta, post, wait_until};

const PROMPT: &str = "Write a Rust function that adds two integers.";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires an idle CUDA device and ENGINE_QWEN_GGUF"]
async fn http_matches_direct_text_and_preserves_disconnect_peer_progress() {
    let path = std::env::var("ENGINE_QWEN_GGUF").expect("pinned artifact path");
    let (owner, _) = tokio::task::spawn_blocking(move || {
        TextOwner::load(
            path,
            LoadOptions {
                context_tokens: 512,
                max_sequences: 2,
                ..LoadOptions::default()
            },
        )
    })
    .await
    .unwrap()
    .unwrap();
    let model = owner.model().clone();
    let options = GenerationOptions {
        max_output_tokens: 8,
        ..GenerationOptions::default()
    };
    let raw = model
        .generate(TextInput::prompt(PROMPT), options.clone())
        .await
        .unwrap();
    let chat = model
        .generate(TextInput::chat(vec![Message::user(PROMPT)]), options)
        .await
        .unwrap();
    assert!(!raw.text.is_empty());
    assert!(!chat.text.is_empty());

    let mut server = Server::bind(ServerConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        max_requests: 2,
        ..ServerConfig::default()
    })
    .await
    .unwrap();
    let address = server.local_addr();
    let (stop, stopped) = oneshot::channel();
    let task = tokio::spawn(async move {
        let result = server
            .run(async { Ok(owner) }, async {
                let _ = stopped.await;
            })
            .await;
        (server, result)
    });
    wait_until(|| async {
        exchange(address, "GET /ready HTTP/1.1\r\nHost: local\r\n\r\n")
            .await
            .status
            == 200
    })
    .await;
    nonstream_parity(address, &raw, &chat).await;
    disconnect_with_peer(address).await;
    stream_parity(address, &raw).await;
    stop.send(()).unwrap();
    let (server, result) = task.await.unwrap();
    result.unwrap();
    assert_eq!(server.readiness(), Readiness::Closing);
    drop(model);
}

async fn nonstream_parity(address: SocketAddr, raw: &TextResponse, chat: &TextResponse) {
    for (path, body, expected) in [
        (
            "/v1/completions",
            json!({"model":"ribn", "prompt":PROMPT, "max_tokens":8}),
            raw,
        ),
        (
            "/v1/chat/completions",
            json!({"model":"ribn", "messages":[{"role":"user", "content":PROMPT}], "max_tokens":8}),
            chat,
        ),
    ] {
        let reply = exchange(address, &post(path, &body.to_string())).await;
        assert_eq!(reply.status, 200);
        let value: Value = serde_json::from_str(&reply.body).unwrap();
        let choice = &value["choices"][0];
        assert_eq!(
            choice["text"]
                .as_str()
                .or_else(|| choice["message"]["content"].as_str()),
            Some(expected.text.as_str())
        );
        assert_eq!(
            value["usage"]["completion_tokens"],
            expected.usage.completion_tokens
        );
    }
}

async fn disconnect_with_peer(address: SocketAddr) {
    // Abandon after a real delta with trailing input, then prove healthy progress
    // and capacity recovery. Device retirement remains owned by the text driver.
    let mut abandoned = TcpStream::connect(address).await.unwrap();
    let body = json!({"model":"ribn", "prompt":PROMPT, "max_tokens":64, "stream":true}).to_string();
    abandoned
        .write_all(format!("{}x", post("/v1/completions", &body)).as_bytes())
        .await
        .unwrap();
    first_delta(&mut abandoned).await;
    let peer = tokio::spawn(async move {
        exchange(
            address,
            &post(
                "/v1/completions",
                &json!({"model":"ribn","prompt":PROMPT,"max_tokens":8}).to_string(),
            ),
        )
        .await
    });
    drop(abandoned);
    assert_eq!(peer.await.unwrap().status, 200);
    wait_until(|| async {
        exchange(address, "GET /metrics HTTP/1.1\r\nHost: local\r\n\r\n")
            .await
            .body
            .contains("ribn_requests_abandoned_total 1\n")
    })
    .await;
}

async fn stream_parity(address: SocketAddr, raw: &TextResponse) {
    let reply = exchange(
        address,
        &post(
            "/v1/completions",
            &json!({"model":"ribn","prompt":PROMPT,"max_tokens":8,"stream":true,
        "stream_options":{"include_usage":true}})
            .to_string(),
        ),
    )
    .await;
    assert_eq!(reply.status, 200);
    let mut streamed = String::new();
    for frame in reply
        .body
        .split("\n\n")
        .filter_map(|f| f.strip_prefix("data: "))
    {
        if frame == "[DONE]" {
            continue;
        }
        let value: Value = serde_json::from_str(frame).unwrap();
        assert!(value.get("error").is_none(), "{value}");
        if let Some(delta) = value["choices"][0]["text"].as_str() {
            streamed.push_str(delta);
        }
    }
    assert_eq!(streamed, raw.text);
    assert_eq!(reply.body.matches("[DONE]").count(), 1);
}
