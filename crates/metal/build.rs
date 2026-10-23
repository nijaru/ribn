//! Build the small Objective-C boundary without changing the Rust deployment floor.

use std::{env, path::PathBuf, process::Command};

fn output(command: &mut Command) -> String {
    let result = command.output().expect("start Apple build tool");
    assert!(
        result.status.success(),
        "{command:?} failed ({}):\n{}",
        result.status,
        String::from_utf8_lossy(&result.stderr)
    );
    String::from_utf8(result.stdout)
        .expect("Apple build tool output is UTF-8")
        .trim()
        .to_owned()
}

fn main() {
    for name in [
        "MACOSX_DEPLOYMENT_TARGET",
        "SDKROOT",
        "DEVELOPER_DIR",
        "RUSTC",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    println!("cargo:rerun-if-changed=src/ffn_graph.m");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    let arch = match env::var("CARGO_CFG_TARGET_ARCH").as_deref() {
        Ok("aarch64") => "arm64",
        Ok("x86_64") => "x86_64",
        _ => panic!("unsupported macOS architecture"),
    };
    // Ask the pinned compiler, not the SDK or host, for the existing target floor.
    // This also honors MACOSX_DEPLOYMENT_TARGET when explicitly configured.
    let rustc = env::var_os("RUSTC").expect("Cargo RUSTC");
    let deployment = output(
        Command::new(rustc)
            .arg("--target")
            .arg(env::var_os("TARGET").expect("Cargo TARGET"))
            .arg("--print=deployment-target"),
    );
    let deployment = deployment
        .strip_prefix("MACOSX_DEPLOYMENT_TARGET=")
        .expect("rustc macOS deployment-target output");
    let sdk = env::var("SDKROOT").unwrap_or_else(|_| {
        output(Command::new("xcrun").args(["--sdk", "macosx", "--show-sdk-path"]))
    });
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo OUT_DIR"));
    let object = out.join("ffn_graph.o");
    let archive = out.join("libribn_ffn_graph.a");
    output(
        Command::new("xcrun")
            .args(["--sdk", "macosx", "clang"])
            .arg("-target")
            .arg(format!("{arch}-apple-macosx{deployment}"))
            .arg("-isysroot")
            .arg(sdk)
            .args([
                "-O3",
                "-fobjc-arc",
                "-fobjc-arc-exceptions",
                "-fblocks",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-Wunguarded-availability",
                "-c",
                "src/ffn_graph.m",
                "-o",
            ])
            .arg(&object),
    );
    output(
        Command::new("xcrun")
            .args(["ar", "rcs"])
            .arg(&archive)
            .arg(&object),
    );
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=ribn_ffn_graph");
    println!("cargo:rustc-link-lib=framework=Foundation");
    // @available lowers to compiler-rt's OS version check. Rust links with
    // -nodefaultlibs, so propagate the support archive rather than relying on
    // a host linker default (or accidentally raising the deployment target).
    let runtime = PathBuf::from(output(Command::new("xcrun").args([
        "--sdk",
        "macosx",
        "clang",
        "--print-file-name=libclang_rt.osx.a",
    ])));
    assert!(runtime.is_file(), "Clang macOS runtime archive unavailable");
    println!(
        "cargo:rustc-link-search=native={}",
        runtime.parent().expect("runtime directory").display()
    );
    println!("cargo:rustc-link-lib=static=clang_rt.osx");
    // MPS/Graph are loaded only for explicit Graph preparation. Ordinary
    // framework= metadata would introduce mandatory transitive load commands.
}
