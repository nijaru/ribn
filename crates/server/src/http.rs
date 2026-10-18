use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::{Method, Request, Response, StatusCode, body::Incoming};
use serde_json::json;

use crate::ServerConfig;
use crate::protocol::{Operation, Output, Request as GenerationRequest};
use crate::response::{self, ApiError, Body};
use crate::state::{Readiness, RequestGuard, State};

pub(crate) async fn route(
    request: Request<Incoming>,
    state: Arc<State>,
    config: Arc<ServerConfig>,
    body_done: Arc<tokio::sync::Notify>,
) -> Result<Response<Body>, Infallible> {
    Ok(match (request.method(), request.uri().path()) {
        (&Method::GET, "/health" | "/ready") => {
            let phase = state.readiness();
            let healthy = if request.uri().path() == "/ready" {
                phase == Readiness::Serving
            } else {
                matches!(phase, Readiness::Loading | Readiness::Serving)
            };
            response::json_response(
                if healthy {
                    StatusCode::OK
                } else {
                    StatusCode::SERVICE_UNAVAILABLE
                },
                &json!({"status": phase.name()}),
            )
        }
        (&Method::GET, "/metrics") => response::response(
            StatusCode::OK,
            "text/plain; version=0.0.4",
            Full::new(Bytes::from(state.metrics_text())).boxed_unsync(),
        ),
        (&Method::POST, "/v1/completions" | "/v1/chat/completions") => {
            let operation = if request.uri().path() == "/v1/completions" {
                Operation::Completion
            } else {
                Operation::Chat
            };
            state.metrics.requests.fetch_add(1, Ordering::Relaxed);
            generation(request, operation, &state, &config, &body_done)
                .await
                .unwrap_or_else(|e| {
                    state.metrics.rejected.fetch_add(1, Ordering::Relaxed);
                    e.response()
                })
        }
        _ => response::json_response(
            StatusCode::NOT_FOUND,
            &json!({"error": {"message": "unsupported route"}}),
        ),
    })
}

async fn generation(
    request: Request<Incoming>,
    operation: Operation,
    state: &Arc<State>,
    config: &ServerConfig,
    body_done: &tokio::sync::Notify,
) -> Result<Response<Body>, ApiError> {
    if state.readiness() != Readiness::Serving {
        return Err(ApiError::unavailable());
    }
    // Reserve before reading/parsing owned bodies; never queue transport waiters.
    let permit = state
        .slots
        .clone()
        .try_acquire_owned()
        .map_err(|_| ApiError::unavailable())?;
    let mut guard = RequestGuard::new(state.clone(), permit);
    if request.uri().query().is_some() {
        return Err(ApiError::bad("query parameters are unsupported"));
    }
    if request.headers().get("content-encoding").is_some() {
        return Err(ApiError::bad("content encoding is unsupported"));
    }
    let content_type = request
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok());
    if !content_type.is_some_and(|v| {
        v.split(';')
            .next()
            .is_some_and(|v| v.trim() == "application/json")
    }) {
        return Err(ApiError::bad("content-type must be application/json"));
    }
    if request.headers().get("content-length").is_some_and(|v| {
        v.to_str()
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .is_some_and(|n| n > config.max_body_bytes as u64)
    }) {
        return Err(ApiError::large());
    }
    let mut body = request.into_body();
    let mut bytes = Vec::with_capacity(config.max_body_bytes);
    while let Some(frame) = body.frame().await {
        let frame = frame.map_err(|_| ApiError::bad("invalid request body"))?;
        let data = frame
            .into_data()
            .map_err(|_| ApiError::bad("request trailers are unsupported"))?;
        if data.len() > config.max_body_bytes - bytes.len() {
            return Err(ApiError::large());
        }
        bytes.extend_from_slice(&data);
    }
    body_done.notify_one();
    let request: GenerationRequest = serde_json::from_slice(&bytes)
        .map_err(|_| ApiError::bad("malformed or unsupported request fields"))?;
    drop(bytes);
    let stream = request.stream;
    let (input, options, include_usage) = request.prepare(operation, config)?;
    let model = state.model().ok_or_else(ApiError::unavailable)?;
    guard.started = true;
    let events = model.stream(input, options).await.map_err(|e| {
        guard.observe_owner();
        guard.fail();
        ApiError::text(&e)
    })?;
    let output = Output {
        operation,
        id: format!("ribn-{}", events.request_id().get()),
        model: config.model_name.clone(),
        created: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs()),
    };
    if stream {
        // At most six JSON bytes per decoded byte, plus bounded envelope fields.
        let frame_limit = model
            .limits()
            .max_decoded_delta_bytes
            .checked_mul(6)
            .and_then(|n| n.checked_add(4096))
            .unwrap_or(config.max_response_bytes)
            .min(config.max_response_bytes);
        Ok(response::streaming(
            events,
            output,
            include_usage,
            frame_limit,
            guard,
        ))
    } else {
        response::collected(events, output, config.max_response_bytes, guard).await
    }
}
