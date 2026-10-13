//! Bounded loopback text serving over the owned facade, without another scheduler.

mod config;
mod http;
mod protocol;
mod response;
mod state;
mod transport;

use std::fmt;
use std::future::Future;
use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::{TokioIo, TokioTimer};
use ribn_text::{TextError, TextOwner};
use tokio::net::TcpListener;
use tokio::task::JoinSet;

pub use config::ServerConfig;
pub use state::Readiness;
use state::{ConnectionGuard, State};

#[derive(Debug)]
pub enum ServerError {
    Config(&'static str),
    Io(io::Error),
    Model(TextError),
    Task(tokio::task::JoinError),
}

impl fmt::Display for ServerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(e) => f.write_str(e),
            Self::Io(e) => e.fmt(f),
            Self::Model(e) => e.fmt(f),
            Self::Task(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for ServerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Config(_) => None,
            Self::Io(e) => Some(e),
            Self::Model(e) => Some(e),
            Self::Task(e) => Some(e),
        }
    }
}

type LoadFuture = Pin<Box<dyn Future<Output = Result<TextOwner, TextError>> + Send>>;

/// Owns network tasks and the same text shutdown owner across cleanup retries.
pub struct Server {
    listener: Option<TcpListener>,
    address: SocketAddr,
    config: Arc<ServerConfig>,
    state: Arc<State>,
    connections: JoinSet<()>,
    owner: Option<TextOwner>,
    load_error: Option<TextError>,
    loading: Option<LoadFuture>,
    started: bool,
}

impl Server {
    /// Bind while loading; health endpoints become available once `run` is polled.
    ///
    /// # Errors
    /// Invalid/overflowing bounds, unsupported non-local binding or socket failure.
    pub async fn bind(config: ServerConfig) -> Result<Self, ServerError> {
        config.validate().map_err(ServerError::Config)?;
        let listener = TcpListener::bind(config.bind)
            .await
            .map_err(ServerError::Io)?;
        let address = listener.local_addr().map_err(ServerError::Io)?;
        let state = Arc::new(State::new(config.max_requests));
        Ok(Self {
            listener: Some(listener),
            address,
            config: Arc::new(config),
            state,
            connections: JoinSet::new(),
            owner: None,
            load_error: None,
            loading: None,
            started: false,
        })
    }

    #[must_use]
    pub const fn local_addr(&self) -> SocketAddr {
        self.address
    }

    #[must_use]
    pub fn readiness(&self) -> Readiness {
        self.state.readiness()
    }

    /// Serve through one load attempt until shutdown. A failed load stays faulted
    /// with health/metrics available until shutdown. Cancelling this future stops
    /// its caller, not device cleanup: call `shutdown_async` on the retained server.
    ///
    /// # Errors
    /// Socket/task failure, loading failure, or retryable model cleanup failure.
    pub async fn run(
        &mut self,
        load: impl Future<Output = Result<TextOwner, TextError>> + Send + 'static,
        shutdown: impl Future<Output = ()>,
    ) -> Result<(), ServerError> {
        if self.listener.is_none() || self.started {
            return Err(ServerError::Config("a server accepts one run/load attempt"));
        }
        self.started = true;
        self.loading = Some(Box::pin(load));
        tokio::pin!(shutdown);
        let mut failure = None;
        loop {
            tokio::select! {
                () = &mut shutdown => break,
                result = async {
                    match &mut self.loading {
                        Some(load) => load.await,
                        None => std::future::pending().await,
                    }
                }, if self.loading.is_some() => {
                    self.loading = None;
                    self.loaded(result);
                }
                joined = self.connections.join_next(), if !self.connections.is_empty() => {
                    if let Some(Err(e)) = joined {
                        failure = Some(ServerError::Task(e));
                        break;
                    }
                }
                accepted = async {
                    match &self.listener {
                        Some(listener) => listener.accept().await,
                        None => std::future::pending().await,
                    }
                } => {
                    match accepted {
                        Ok((socket, _)) => {
                            if self.connections.len() >= self.config.max_connections {
                                self.state.metrics.connections_rejected.fetch_add(1, Ordering::Relaxed);
                                // No additional task/waiter/body allocation when saturated.
                                drop(socket);
                                continue;
                            }
                            if let Err(e) = socket.set_nodelay(true) {
                                failure = Some(ServerError::Io(e));
                                break;
                            }
                            self.spawn_connection(socket);
                        }
                        Err(e) => { failure = Some(ServerError::Io(e)); break; }
                    }
                }
            }
        }
        self.close_transport().await;
        let cleanup = self.shutdown_async().await;
        if let Some(failure) = failure {
            cleanup?;
            return Err(failure);
        }
        cleanup
    }

    fn loaded(&mut self, result: Result<TextOwner, TextError>) {
        match result {
            Ok(owner) => {
                if self.state.readiness() == Readiness::Loading {
                    self.state.install(owner.model().clone());
                }
                self.owner = Some(owner);
            }
            Err(error) => {
                self.load_error = Some(error);
                self.state.set_phase(Readiness::Faulted);
            }
        }
    }

    fn spawn_connection(&mut self, socket: tokio::net::TcpStream) {
        self.state
            .metrics
            .connections
            .fetch_add(1, Ordering::Relaxed);
        let guard = ConnectionGuard(self.state.clone());
        let state = self.state.clone();
        let config = self.config.clone();
        self.connections.spawn(async move {
            let _guard = guard;
            let service_state = state.clone();
            let service_config = config.clone();
            let body_done = Arc::new(tokio::sync::Notify::new());
            let (io, disconnected) = transport::observe(socket, body_done.clone());
            let service = service_fn(move |request| {
                http::route(
                    request,
                    service_state.clone(),
                    service_config.clone(),
                    body_done.clone(),
                )
            });
            let mut builder = http1::Builder::new();
            builder
                .keep_alive(false)
                .max_headers(32)
                .max_buf_size(16 * 1024)
                .timer(TokioTimer::new())
                .header_read_timeout(std::time::Duration::from_secs(5));
            let connection = builder.serve_connection(TokioIo::new(io), service);
            tokio::select! {
                () = disconnected => {},
                result = tokio::time::timeout(config.connection_deadline, connection) => {
                    if result.is_err() { state.metrics.deadlines.fetch_add(1, Ordering::Relaxed); }
                }
            }
            // HTTP parse/disconnect/write errors end this connection only. Drop
            // abandons request interest; the token owner retains retirement.
        });
    }

    async fn close_transport(&mut self) {
        self.state.set_phase(Readiness::Closing);
        self.listener = None;
        self.state.slots.close();
        self.connections.abort_all();
        while self.connections.join_next().await.is_some() {}
    }

    /// Close transport and synchronize text execution. Failure keeps the owner
    /// available for another call. No timeout can assert device completion.
    ///
    /// # Errors
    /// Load failure or cleanup failure with the same model owner retained.
    pub async fn shutdown_async(&mut self) -> Result<(), ServerError> {
        self.close_transport().await;
        // Keep the pending future in self across cancellation of cleanup.
        if let Some(load) = &mut self.loading {
            let result = load.await;
            self.loading = None;
            self.loaded(result);
        }
        if let Some(owner) = &mut self.owner {
            if let Err(error) = owner.shutdown_async().await {
                self.state.set_phase(Readiness::Faulted);
                return Err(ServerError::Model(error));
            }
            self.owner = None;
        }
        if let Some(error) = &self.load_error {
            return Err(ServerError::Model(error.clone()));
        }
        Ok(())
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.state.set_phase(Readiness::Closing);
        self.state.slots.close();
        self.connections.abort_all();
        // TextOwner Drop requests cleanup; it never proves device completion.
    }
}
