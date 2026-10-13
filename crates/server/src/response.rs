use std::convert::Infallible;
use std::io::{self, Write};

use bytes::Bytes;
use http_body_util::{BodyExt, Full, StreamBody, combinators::UnsyncBoxBody};
use hyper::{Response, StatusCode, body::Frame};
use ribn_text::{FinishReason, TextError, TextEvent, TextStream};
use serde_json::{Value, json};

use crate::protocol::{Operation, Output};
use crate::state::RequestGuard;

pub(crate) type Body = UnsyncBoxBody<Bytes, Infallible>;

pub(crate) struct ApiError {
    pub status: StatusCode,
    pub code: &'static str,
    pub message: &'static str,
}

impl ApiError {
    pub(crate) const fn bad(message: &'static str) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "invalid_request",
            message,
        }
    }
    pub(crate) const fn large() -> Self {
        Self {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            code: "limit_exceeded",
            message: "HTTP byte limit exceeded",
        }
    }
    pub(crate) const fn unavailable() -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "unavailable",
            message: "model unavailable or request capacity exhausted",
        }
    }
    pub(crate) const fn failed() -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "generation_failed",
            message: "generation failed; no successful usage is available",
        }
    }
    pub(crate) fn text(error: &TextError) -> Self {
        if error.is_size_limit() {
            Self::large()
        } else if error.is_request_local() {
            Self::bad("request rejected by text processor")
        } else {
            Self::unavailable()
        }
    }
    pub(crate) fn value(&self) -> Value {
        json!({"error": {"message": self.message, "type": self.code, "code": self.code, "param": null}})
    }
    pub(crate) fn response(&self) -> Response<Body> {
        let mut response = json_response(self.status, &self.value());
        if self.status == StatusCode::SERVICE_UNAVAILABLE {
            response
                .headers_mut()
                .insert("retry-after", hyper::header::HeaderValue::from_static("1"));
        }
        response
    }
}

pub(crate) fn json_response(status: StatusCode, value: &Value) -> Response<Body> {
    response(
        status,
        "application/json",
        Full::new(Bytes::from(value.to_string())).boxed_unsync(),
    )
}

pub(crate) fn response(
    status: StatusCode,
    content_type: &'static str,
    body: Body,
) -> Response<Body> {
    let mut response = Response::new(body);
    *response.status_mut() = status;
    response.headers_mut().insert(
        "content-type",
        hyper::header::HeaderValue::from_static(content_type),
    );
    response.headers_mut().insert(
        "connection",
        hyper::header::HeaderValue::from_static("close"),
    );
    response
}

/// Serializing through this writer rejects before growing beyond the envelope.
struct LimitedWriter {
    bytes: Vec<u8>,
    limit: usize,
}

impl Write for LimitedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.limit - self.bytes.len() {
            return Err(io::Error::other("response byte limit exceeded"));
        }
        let required = self.bytes.len() + bytes.len();
        if required > self.bytes.capacity() {
            let capacity = self
                .bytes
                .capacity()
                .saturating_mul(2)
                .max(required)
                .min(self.limit);
            self.bytes.reserve_exact(capacity - self.bytes.len());
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn serialize(value: &Value, limit: usize, sse: bool) -> Result<Bytes, ApiError> {
    let mut writer = LimitedWriter {
        bytes: Vec::with_capacity(limit.min(1024)),
        limit,
    };
    if sse {
        writer.write_all(b"data: ").map_err(|_| ApiError::large())?;
    }
    serde_json::to_writer(&mut writer, value).map_err(|_| ApiError::large())?;
    if sse {
        writer.write_all(b"\n\n").map_err(|_| ApiError::large())?;
    }
    Ok(Bytes::from(writer.bytes))
}

fn finish_reason(reason: &FinishReason) -> Result<&'static str, ApiError> {
    match reason {
        FinishReason::Length => Ok("length"),
        FinishReason::Stop => Ok("stop"),
        FinishReason::Cancelled | FinishReason::Failed(_) => Err(ApiError::failed()),
    }
}

pub(crate) fn streaming(
    mut events: TextStream,
    output: Output,
    include_usage: bool,
    frame_limit: usize,
    mut guard: RequestGuard,
) -> Response<Body> {
    let frames = async_stream::stream! {
        let mut error = None;
        if matches!(output.operation, Operation::Chat) {
            match serialize(&output.role(), frame_limit, true) {
                Ok(bytes) => yield Ok(Frame::data(bytes)),
                Err(e) => error = Some(e),
            }
        }
        while error.is_none() {
            match events.next_async().await {
                Some(Ok(TextEvent::Delta { text, .. })) => {
                    if text.is_empty() { continue; }
                    match serialize(&output.chunk(&text, None), frame_limit, true) {
                        Ok(bytes) => yield Ok(Frame::data(bytes)),
                        Err(e) => error = Some(e),
                    }
                }
                Some(Ok(TextEvent::Finished { reason, usage })) => {
                    let terminal = finish_reason(&reason).and_then(|reason| {
                        let finish = serialize(&output.chunk("", Some(reason)), frame_limit, true)?;
                        let usage = include_usage.then(|| serialize(&output.usage(usage), frame_limit, true)).transpose()?;
                        Ok((finish, usage))
                    });
                    match terminal {
                        Ok((finish, usage_frame)) => {
                            guard.complete(usage);
                            yield Ok(Frame::data(finish));
                            if let Some(bytes) = usage_frame { yield Ok(Frame::data(bytes)); }
                        }
                        Err(e) => error = Some(e),
                    }
                    break;
                }
                Some(Err(e)) => {
                    guard.observe_owner();
                    error = Some(ApiError::text(&e));
                }
                None => { error = Some(ApiError::failed()); }
            }
        }
        if let Some(error) = error {
            guard.fail();
            // Error payloads are fixed, smaller than the minimum frame envelope.
            yield Ok(Frame::data(Bytes::from(format!("data: {}\n\n", error.value()))));
        }
        yield Ok(Frame::data(Bytes::from_static(b"data: [DONE]\n\n")));
    };
    let mut response = response(
        StatusCode::OK,
        "text/event-stream",
        StreamBody::new(frames).boxed_unsync(),
    );
    response.headers_mut().insert(
        "cache-control",
        hyper::header::HeaderValue::from_static("no-cache"),
    );
    response
}

pub(crate) async fn collected(
    mut events: TextStream,
    output: Output,
    limit: usize,
    mut guard: RequestGuard,
) -> Result<Response<Body>, ApiError> {
    let mut text = String::with_capacity(limit);
    while let Some(event) = events.next_async().await {
        match event {
            Ok(TextEvent::Delta { text: delta, .. }) => {
                if delta.len() > limit - text.len() {
                    guard.fail();
                    return Err(ApiError::large());
                }
                text.push_str(&delta);
            }
            Ok(TextEvent::Finished { reason, usage }) => {
                let bytes = finish_reason(&reason)
                    .and_then(|reason| {
                        serialize(&output.complete(&text, reason, usage), limit, false)
                    })
                    .inspect_err(|_| guard.fail())?;
                guard.complete(usage);
                let frames = async_stream::stream! {
                    yield Ok(Frame::data(bytes));
                    drop(guard);
                };
                return Ok(response(
                    StatusCode::OK,
                    "application/json",
                    StreamBody::new(frames).boxed_unsync(),
                ));
            }
            Err(e) => {
                guard.observe_owner();
                guard.fail();
                return Err(ApiError::text(&e));
            }
        }
    }
    guard.fail();
    Err(ApiError::failed())
}
