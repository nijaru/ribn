use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;

/// Transport envelopes, separate from the text/runtime admission envelopes.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub bind: SocketAddr,
    /// Exact public model identifier accepted in request bodies.
    pub model_name: String,
    pub max_connections: usize,
    pub max_requests: usize,
    pub max_body_bytes: usize,
    pub max_response_bytes: usize,
    pub max_output_tokens: u32,
    /// Absolute lifetime, including headers, body, generation and writes.
    pub connection_deadline: Duration,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8000),
            model_name: "ribn".to_owned(),
            max_connections: 64,
            max_requests: 32,
            max_body_bytes: 256 * 1024,
            max_response_bytes: 1024 * 1024,
            max_output_tokens: 4096,
            connection_deadline: Duration::from_secs(120),
        }
    }
}

impl ServerConfig {
    pub(crate) fn validate(&self) -> Result<(), &'static str> {
        if !self.bind.ip().is_loopback() {
            return Err(
                "only loopback binding is supported; non-local exposure needs a security policy",
            );
        }
        if self.model_name.is_empty() || self.model_name.len() > 256 {
            return Err("model identifier must contain 1..=256 bytes");
        }
        if self.max_requests > tokio::sync::Semaphore::MAX_PERMITS {
            return Err("HTTP request slots exceed semaphore capacity");
        }
        if self.max_connections == 0
            || self.max_requests == 0
            || self.max_body_bytes == 0
            || self.max_response_bytes < 1024
            || self.max_output_tokens == 0
            || self.connection_deadline.is_zero()
        {
            return Err("server bounds must be nonzero and response bytes at least 1024");
        }
        // Include JSON escaping and bounded serialization/collection overlap.
        if self.max_body_bytes.checked_mul(self.max_requests).is_none()
            || self
                .max_response_bytes
                .checked_mul(8)
                .and_then(|n| n.checked_mul(self.max_connections))
                .is_none()
        {
            return Err("aggregate HTTP byte envelope overflows");
        }
        Ok(())
    }
}
