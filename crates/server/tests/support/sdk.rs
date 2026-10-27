//! Optional official-client qualification, not a second server or execution loop.
use std::io;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::{Output, Stdio};
use std::time::Duration;

use serde_json::Value;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;

pub struct Client {
    node: PathBuf,
    module: PathBuf,
}

impl Client {
    pub fn configured() -> Self {
        let path = |name| {
            let value = PathBuf::from(std::env::var_os(name).expect(name));
            assert!(
                value.is_absolute() && value.is_file(),
                "{name}: absolute file required"
            );
            value
        };
        Self {
            node: path("RIBN_OPENAI_NODE"),
            module: path("RIBN_OPENAI_SDK"),
        }
    }

    pub async fn run(&self, address: SocketAddr, mode: &str) -> Result<Value, String> {
        let mut child = Command::new(&self.node)
            .arg(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/support/openai_client.mjs"
            ))
            .arg(&self.module)
            .arg(format!("http://{address}/v1"))
            .arg(mode)
            // The SDK must not inherit real credentials, organization/project
            // headers, proxy settings or debugging that could retain secrets.
            .env_clear()
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|error| format!("spawn SDK client: {error}"))?;
        let stdout = child.stdout.take().expect("stdout configured as a pipe");
        let stderr = child.stderr.take().expect("stderr configured as a pipe");
        let result = tokio::time::timeout(Duration::from_secs(12), async {
            let (status, stdout, stderr) =
                tokio::try_join!(child.wait(), bounded_output(stdout), bounded_output(stderr))?;
            Ok::<_, io::Error>(Output {
                status,
                stdout,
                stderr,
            })
        })
        .await;
        let output = match result {
            Ok(Ok(output)) => output,
            other => {
                // The wait borrows the child: timeout/read failure does not
                // consume its owner. Kill and reap before returning failure.
                let cleanup = child.kill().await;
                return Err(format!(
                    "SDK communication failed: {other:?}; cleanup: {cleanup:?}"
                ));
            }
        };
        println!(
            "SDK {mode}: {}\nstdout: {}\nstderr: {}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        if !output.status.success() {
            return Err(format!(
                "SDK {mode} failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        let report: Value = serde_json::from_slice(&output.stdout)
            .map_err(|error| format!("SDK result: {error}"))?;
        if report["mode"] != mode {
            return Err("SDK result mode differs".into());
        }
        Ok(report)
    }
}

async fn bounded_output(pipe: impl AsyncRead + Unpin) -> io::Result<Vec<u8>> {
    const LIMIT: usize = 64 * 1024;
    let mut bytes = Vec::new();
    pipe.take((LIMIT + 1) as u64)
        .read_to_end(&mut bytes)
        .await?;
    if bytes.len() > LIMIT {
        return Err(io::Error::other("SDK output exceeded 64 KiB"));
    }
    Ok(bytes)
}
