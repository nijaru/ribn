//! Model-specific load controls must reject before transport or device preparation.
#![cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]

use std::net::TcpListener;
use std::process::Command;

#[test]
fn gemma_rejects_unsupported_load_controls_before_binding() {
    // A regression that ignores a flag fails to bind, rather than loading a model
    // or hanging. The diagnostic distinguishes that failure from proper rejection.
    let occupied = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = occupied.local_addr().unwrap().to_string();
    for (name, value) in [
        ("--device", "0"),
        ("--context-length", "4096"),
        ("--max-sequences", "4"),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_ribn"))
            .args([
                "serve-gemma",
                "missing-package",
                "--bind",
                &address,
                name,
                value,
            ])
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
        let error = String::from_utf8(output.stderr).unwrap();
        assert!(
            error.contains(&format!("{name} is not supported by serve-gemma")),
            "{error}"
        );
    }
}
