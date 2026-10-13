use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

pub fn post(path: &str, body: &str) -> String {
    format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
}

pub struct Reply {
    pub status: u16,
    pub body: String,
}
pub async fn exchange(address: std::net::SocketAddr, request: &str) -> Reply {
    let mut socket = TcpStream::connect(address).await.unwrap();
    socket.write_all(request.as_bytes()).await.unwrap();
    let mut bytes = Vec::new();
    tokio::time::timeout(Duration::from_secs(3), socket.read_to_end(&mut bytes))
        .await
        .unwrap()
        .unwrap();
    let raw = String::from_utf8(bytes).unwrap();
    let (headers, body) = raw.split_once("\r\n\r\n").unwrap();
    let status = headers.split_whitespace().nth(1).unwrap().parse().unwrap();
    let body = if headers
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked")
    {
        dechunk(body)
    } else {
        body.to_owned()
    };
    Reply { status, body }
}
fn dechunk(mut body: &str) -> String {
    let mut result = String::new();
    loop {
        let (size, tail) = body.split_once("\r\n").unwrap();
        let size = usize::from_str_radix(size, 16).unwrap();
        if size == 0 {
            break;
        }
        result.push_str(&tail[..size]);
        body = &tail[size + 2..];
    }
    result
}

pub async fn first_delta(socket: &mut TcpStream) {
    let mut bytes = Vec::new();
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let mut buf = [0; 4096];
            let n = socket.read(&mut buf).await.unwrap();
            assert!(n > 0);
            bytes.extend_from_slice(&buf[..n]);
            if String::from_utf8_lossy(&bytes).contains("\"text\":") {
                break;
            }
        }
    })
    .await
    .unwrap();
}

pub async fn wait_until<F, Fut>(mut condition: F)
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    tokio::time::timeout(Duration::from_secs(3), async {
        while !condition().await {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
}
