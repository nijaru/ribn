//! Observe EOF independently once the sole request body has been consumed.
//!
//! Hyper stops reading when trailing/pipelined bytes remain in its read buffer.
//! With one request per connection, those bytes have no further protocol owner;
//! drain them without retention so they cannot conceal a disconnect indefinitely.

use std::future::{Future, poll_fn};
use std::io;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::{
    TcpStream,
    tcp::{OwnedReadHalf, OwnedWriteHalf},
};
use tokio::sync::Notify;

pub(crate) struct Io {
    read: Arc<Mutex<OwnedReadHalf>>,
    write: OwnedWriteHalf,
}

pub(crate) fn observe(socket: TcpStream, body_done: Arc<Notify>) -> (Io, impl Future<Output = ()>) {
    let (read, write) = socket.into_split();
    let read = Arc::new(Mutex::new(read));
    let io = Io {
        read: read.clone(),
        write,
    };
    let disconnected = async move {
        body_done.notified().await;
        let mut scratch = [0; 4096];
        loop {
            let result = poll_fn(|cx| {
                let mut buf = ReadBuf::new(&mut scratch);
                let mut read = read
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                match Pin::new(&mut *read).poll_read(cx, &mut buf) {
                    Poll::Ready(Ok(())) => Poll::Ready(Ok(buf.filled().len())),
                    Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
                    Poll::Pending => Poll::Pending,
                }
            })
            .await;
            match result {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
        }
    };
    (io, disconnected)
}

impl AsyncRead for Io {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        // Both readers are polled by the same connection task; the lock is held
        // only during poll, never across await. After body_done, neither needs
        // trailing bytes for another request (keep-alive is disabled).
        let mut read = self
            .read
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Pin::new(&mut *read).poll_read(cx, buf)
    }
}

impl AsyncWrite for Io {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bytes: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.write).poll_write(cx, bytes)
    }
    fn is_write_vectored(&self) -> bool {
        self.write.is_write_vectored()
    }
    fn poll_write_vectored(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bytes: &[io::IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.write).poll_write_vectored(cx, bytes)
    }
    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.write).poll_flush(cx)
    }
    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.write).poll_shutdown(cx)
    }
}
