//! Exercise the actual TCP/HTTP entry point, real tokenizer, facade and driver.
//! Only device execution is substituted; these are not GPU qualification tests.
mod support;

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use ribn_server::{Readiness, Server, ServerConfig};
use ribn_text::TextError;
use serde_json::{Value, json};
use support::{Plan, Running, exchange, first_delta, owner, wait_until};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::oneshot;

fn prompt(stream: bool, tokens: u32) -> String {
    json!({"model": "ribn", "prompt": "a", "stream": stream, "max_tokens": tokens}).to_string()
}

#[tokio::test]
async fn completion_chat_streaming_and_usage_have_one_ordered_terminal() {
    let running = Running::start(ServerConfig::default(), Arc::new(Plan::default()), 2).await;
    for (path, body, text_key) in [
        ("/v1/completions", prompt(true, 3), "text"),
        (
            "/v1/chat/completions",
            json!({"model":"ribn", "messages":[{"role":"user", "content":"a"}],
            "stream":true, "max_tokens":3, "stream_options":{"include_usage":true}})
            .to_string(),
            "content",
        ),
    ] {
        let reply = running.post(path, &body).await;
        assert_eq!(reply.status, 200);
        let mut chunks = reply
            .body
            .split("\n\n")
            .filter_map(|s| s.strip_prefix("data: "))
            .collect::<Vec<_>>();
        assert_eq!(chunks.pop(), Some("[DONE]"));
        let values = chunks
            .iter()
            .map(|s| serde_json::from_str::<Value>(s).unwrap())
            .collect::<Vec<_>>();
        let finishes = values
            .iter()
            .filter(|v| v["choices"][0]["finish_reason"] == "length")
            .collect::<Vec<_>>();
        assert_eq!(finishes.len(), 1);
        let mut text = String::new();
        for value in &values {
            let choice = &value["choices"][0];
            if let Some(delta) = choice[text_key]
                .as_str()
                .or_else(|| choice["delta"][text_key].as_str())
            {
                text.push_str(delta);
            }
        }
        assert_eq!(text, "xxx");
        if text_key == "content" {
            assert_eq!(values[0]["choices"][0]["delta"]["role"], "assistant");
            assert_eq!(values.last().unwrap()["usage"]["completion_tokens"], 3);
            assert_eq!(values.last().unwrap()["choices"], json!([]));
        }
    }
    let reply = running.post("/v1/completions", &prompt(false, 2)).await;
    let value: Value = serde_json::from_str(&reply.body).unwrap();
    assert_eq!(reply.status, 200);
    assert_eq!(value["choices"][0]["text"], "xx");
    assert_eq!(value["usage"]["total_tokens"], 3);
    assert!(
        running
            .get("/metrics")
            .await
            .body
            .contains("ribn_requests_completed_total 3\n")
    );
    assert!(running.stop().await.1.is_ok());
}

#[tokio::test]
async fn malformed_unsupported_and_oversized_bodies_do_not_poison_peers() {
    let config = ServerConfig {
        max_body_bytes: 512,
        ..ServerConfig::default()
    };
    let running = Running::start(config, Arc::new(Plan::default()), 2).await;
    for body in [
        "{",
        r#"{"model":"ribn","prompt":"a","tools":[]}"#,
        r#"{"model":"ribn","prompt":"a","temperature":1}"#,
        r#"{"model":"other","prompt":"a"}"#,
        r#"{"model":"ribn","messages":[],"prompt":"a"}"#,
        r#"{"model":"ribn","prompt":"a","max_tokens":0}"#,
        r#"{"model":"ribn","prompt":"a","stream":false,"stream_options":{"include_usage":true}}"#,
    ] {
        assert_eq!(
            running.post("/v1/completions", body).await.status,
            400,
            "{body}"
        );
    }
    assert_eq!(
        running
            .post("/v1/completions", &"a".repeat(513))
            .await
            .status,
        413
    );
    let chunked = format!(
        "POST /v1/completions HTTP/1.1\r\nHost: local\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n201\r\n{}\r\n0\r\n\r\n",
        "a".repeat(513)
    );
    assert_eq!(exchange(running.address, &chunked).await.status, 413);
    let body =
        json!({"model":"ribn","messages":[{"role":"user","content":[{"type":"image_url"}]}]})
            .to_string();
    assert_eq!(
        running.post("/v1/chat/completions", &body).await.status,
        400
    );
    assert_eq!(
        running
            .post("/v1/completions", &prompt(false, 1))
            .await
            .status,
        200
    );
    assert!(running.stop().await.1.is_ok());
}

#[tokio::test]
async fn disconnect_cancels_with_saturated_transport_and_runtime_admission() {
    for (stream, tail) in [(true, ""), (false, ""), (true, "x"), (false, "x")] {
        let plan = Arc::new(Plan::default());
        plan.hold.store(true, Ordering::SeqCst);
        let running = Running::start(
            ServerConfig {
                max_requests: 1,
                ..ServerConfig::default()
            },
            plan.clone(),
            1,
        )
        .await;
        let mut socket = running
            .stream_with_tail("/v1/completions", &prompt(stream, 50), tail)
            .await;
        if stream {
            first_delta(&mut socket).await;
        }
        wait_until(|| async { plan.produced.load(Ordering::SeqCst) > 0 }).await;
        assert_eq!(
            running
                .post("/v1/completions", &prompt(false, 1))
                .await
                .status,
            503
        );
        assert_eq!(running.get("/ready").await.status, 200);
        drop(socket);
        // The transport must abandon interest before pending device work settles.
        wait_until(|| async {
            running
                .get("/metrics")
                .await
                .body
                .contains("ribn_requests_abandoned_total 1\n")
        })
        .await;
        assert_eq!(plan.live.load(Ordering::SeqCst), 1);
        plan.hold.store(false, Ordering::SeqCst);
        wait_until(|| async { plan.live.load(Ordering::SeqCst) == 0 }).await;
        assert!(
            plan.produced.load(Ordering::SeqCst) <= 2,
            "disconnect must not finish 50 tokens"
        );
        assert_eq!(
            running
                .post("/v1/completions", &prompt(false, 1))
                .await
                .status,
            200
        );
        assert!(running.stop().await.1.is_ok());
    }
}

#[tokio::test]
async fn stalled_client_preserves_healthy_progress_and_deadline_reclaims_interest() {
    let plan = Arc::new(Plan::default());
    let config = ServerConfig {
        connection_deadline: Duration::from_secs(2),
        ..ServerConfig::default()
    };
    let running = Running::start(config, plan.clone(), 2).await;
    let slow = running
        .stream(
            "/v1/completions",
            &json!({"model":"ribn","prompt":"b","stream":true,"max_tokens":4096}).to_string(),
        )
        .await;
    // Each token decodes to 4 KiB. Leave a response much larger than socket
    // buffers unread; the real body stream must eventually stop pulling events.
    wait_until(|| async { plan.produced.load(Ordering::SeqCst) >= 100 }).await;
    assert_eq!(
        running
            .post("/v1/completions", &prompt(false, 2))
            .await
            .status,
        200
    );
    wait_until(|| async {
        running
            .get("/metrics")
            .await
            .body
            .contains("ribn_connection_deadlines_total 1\n")
    })
    .await;
    wait_until(|| async { plan.live.load(Ordering::SeqCst) == 0 }).await;
    assert!(
        plan.produced.load(Ordering::SeqCst) < 4096,
        "stalled response should not collect all output"
    );
    drop(slow);
    assert!(running.stop().await.1.is_ok());
}

#[tokio::test]
async fn connection_limit_and_body_deadline_have_no_waiter_queue() {
    let config = ServerConfig {
        max_connections: 1,
        connection_deadline: Duration::from_millis(200),
        ..ServerConfig::default()
    };
    let running = Running::start(config, Arc::new(Plan::default()), 1).await;
    let mut slow = tokio::net::TcpStream::connect(running.address)
        .await
        .unwrap();
    slow.write_all(b"POST /v1/completions HTTP/1.1\r\nHost: local\r\nContent-Type: application/json\r\nContent-Length: 50\r\n\r\n{").await.unwrap();
    tokio::time::sleep(Duration::from_millis(20)).await;
    let mut rejected = tokio::net::TcpStream::connect(running.address)
        .await
        .unwrap();
    let mut bytes = Vec::new();
    assert_eq!(rejected.read_to_end(&mut bytes).await.unwrap(), 0);
    assert_eq!(slow.read_to_end(&mut bytes).await.unwrap(), 0);
    assert_eq!(running.get("/ready").await.status, 200);
    assert!(running.stop().await.1.is_ok());
}

#[tokio::test]
async fn owner_fault_emits_error_not_success_and_shutdown_keeps_retry_owner() {
    let plan = Arc::new(Plan::default());
    plan.hold.store(true, Ordering::SeqCst);
    let running = Running::start(ServerConfig::default(), plan.clone(), 1).await;
    let mut socket = running.stream("/v1/completions", &prompt(true, 50)).await;
    first_delta(&mut socket).await;
    plan.fail.store(true, Ordering::SeqCst);
    let mut rest = String::new();
    socket.read_to_string(&mut rest).await.unwrap();
    assert_eq!(rest.matches("\"error\"").count(), 1);
    assert_eq!(rest.matches("[DONE]").count(), 1);
    assert!(!rest.contains("\"usage\""));
    let ready = running.get("/ready").await;
    assert_eq!(ready.status, 503);
    assert!(ready.body.contains("faulted"));
    assert_eq!(
        running
            .post("/v1/completions", &prompt(false, 1))
            .await
            .status,
        503
    );
    plan.fail_sync.store(true, Ordering::SeqCst);
    let (mut server, result) = running.stop().await;
    assert!(result.is_err());
    assert_eq!(server.readiness(), Readiness::Faulted);
    plan.fail_sync.store(false, Ordering::SeqCst);
    server.shutdown_async().await.unwrap();
    assert_eq!(server.readiness(), Readiness::Closing);
}

#[tokio::test]
async fn response_limit_is_request_local() {
    let running = Running::start(
        ServerConfig {
            max_response_bytes: 1024,
            ..ServerConfig::default()
        },
        Arc::new(Plan::default()),
        2,
    )
    .await;
    let reply = running
        .post(
            "/v1/completions",
            &json!({"model":"ribn","prompt":"b","max_tokens":1}).to_string(),
        )
        .await;
    assert_eq!(reply.status, 413);
    let reply = running
        .post(
            "/v1/completions",
            &json!({"model":"ribn","prompt":"b","max_tokens":1,"stream":true}).to_string(),
        )
        .await;
    assert_eq!(reply.status, 200);
    assert!(reply.body.contains("\"error\""));
    assert!(reply.body.contains("[DONE]"));
    assert_eq!(running.get("/ready").await.status, 200);
    assert_eq!(
        running
            .post("/v1/completions", &prompt(false, 1))
            .await
            .status,
        200
    );
    assert!(running.stop().await.1.is_ok());
}

#[tokio::test]
async fn terminal_usage_overflow_never_publishes_success() {
    let name = "\u{1}".repeat(144);
    let running = Running::start(
        ServerConfig {
            model_name: name.clone(),
            max_response_bytes: 1024,
            ..ServerConfig::default()
        },
        Arc::new(Plan::default()),
        1,
    )
    .await;
    let reply = running
        .post(
            "/v1/completions",
            &json!({"model":name,"prompt":"a","max_tokens":1,
        "stream":true,"stream_options":{"include_usage":true}})
            .to_string(),
        )
        .await;
    assert_eq!(reply.status, 200);
    assert!(reply.body.contains("\"error\""));
    assert!(!reply.body.contains("\"finish_reason\":\"length\""));
    assert!(!reply.body.contains("\"usage\""));
    let metrics = running.get("/metrics").await.body;
    assert!(metrics.contains("ribn_requests_completed_total 0\n"));
    assert!(metrics.contains("ribn_requests_failed_total 1\n"));
    assert!(metrics.contains("ribn_completion_tokens_total 0\n"));
    assert!(running.stop().await.1.is_ok());
}

#[tokio::test]
async fn facade_byte_and_message_limits_reject_locally_with_capacity_recovery() {
    let running = Running::start(
        ServerConfig {
            max_body_bytes: 512 * 1024,
            ..ServerConfig::default()
        },
        Arc::new(Plan::default()),
        2,
    )
    .await;
    // First exceeds the text payload limit; second fits that payload but encoded
    // tokens plus artifact/runtime stop copies exceed the driver's envelope.
    for length in [256 * 1024 + 1, 65_536] {
        let body = json!({"model":"ribn", "prompt":"a".repeat(length), "max_tokens":1}).to_string();
        assert_eq!(running.post("/v1/completions", &body).await.status, 413);
    }
    for (count, status) in [(128, 200), (129, 400)] {
        let messages = vec![json!({"role":"user", "content":"a"}); count];
        let body = json!({"model":"ribn", "messages":messages,"max_tokens":1}).to_string();
        assert_eq!(
            running.post("/v1/chat/completions", &body).await.status,
            status
        );
    }
    assert_eq!(
        running
            .post("/v1/completions", &prompt(false, 1))
            .await
            .status,
        200
    );
    assert!(running.stop().await.1.is_ok());
}

#[tokio::test]
async fn invalid_transport_configuration_is_rejected_without_panicking() {
    for config in [
        ServerConfig {
            bind: "0.0.0.0:0".parse().unwrap(),
            ..ServerConfig::default()
        },
        ServerConfig {
            max_requests: tokio::sync::Semaphore::MAX_PERMITS + 1,
            max_body_bytes: 1,
            max_connections: 1,
            max_response_bytes: 1024,
            ..ServerConfig::default()
        },
        ServerConfig {
            max_response_bytes: usize::MAX,
            ..ServerConfig::default()
        },
    ] {
        assert!(Server::bind(config).await.is_err());
    }
}

#[tokio::test]
async fn cancelled_run_and_cleanup_retain_pending_load_for_retry() {
    let mut server = Server::bind(ServerConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        ..ServerConfig::default()
    })
    .await
    .unwrap();
    let (loaded, load) = oneshot::channel();
    assert!(
        tokio::time::timeout(
            Duration::from_millis(20),
            server.run(async { load.await.unwrap() }, std::future::pending())
        )
        .await
        .is_err()
    );
    assert!(
        tokio::time::timeout(Duration::from_millis(20), server.shutdown_async())
            .await
            .is_err()
    );
    assert_eq!(server.readiness(), Readiness::Closing);
    loaded
        .send(Ok(owner(Arc::new(Plan::default()), 1)))
        .ok()
        .unwrap();
    server.shutdown_async().await.unwrap();
    assert_eq!(server.readiness(), Readiness::Closing);
}

#[tokio::test]
async fn loading_failed_load_and_shutdown_during_load_are_observable() {
    for fail in [false, true] {
        let mut server = Server::bind(ServerConfig {
            bind: "127.0.0.1:0".parse().unwrap(),
            ..ServerConfig::default()
        })
        .await
        .unwrap();
        let address = server.local_addr();
        let (loaded, load) = oneshot::channel();
        let (stop, shutdown) = oneshot::channel();
        let task = tokio::spawn(async move {
            let result = server
                .run(async { load.await.unwrap() }, async {
                    let _ = shutdown.await;
                })
                .await;
            (server, result)
        });
        let ready = exchange(address, "GET /ready HTTP/1.1\r\nHost: local\r\n\r\n").await;
        assert_eq!(ready.status, 503);
        assert!(ready.body.contains("loading"));
        assert_eq!(
            exchange(address, "GET /health HTTP/1.1\r\nHost: local\r\n\r\n")
                .await
                .status,
            200
        );
        if fail {
            loaded
                .send(Err(TextError::InvalidInput("fixture load failure")))
                .ok()
                .unwrap();
            wait_until(|| async {
                exchange(address, "GET /ready HTTP/1.1\r\nHost: local\r\n\r\n")
                    .await
                    .body
                    .contains("faulted")
            })
            .await;
            stop.send(()).unwrap();
            assert!(task.await.unwrap().1.is_err());
        } else {
            stop.send(()).unwrap();
            tokio::time::sleep(Duration::from_millis(20)).await;
            assert!(
                !task.is_finished(),
                "cleanup must wait for pending load ownership"
            );
            loaded
                .send(Ok(owner(Arc::new(Plan::default()), 1)))
                .ok()
                .unwrap();
            let (server, result) = task.await.unwrap();
            assert!(result.is_ok());
            assert_eq!(server.readiness(), Readiness::Closing);
        }
    }
}
