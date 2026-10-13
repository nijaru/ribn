#[cfg(any(feature = "cuda", test))]
mod cli;
#[cfg(any(feature = "cuda", test))]
mod input;
mod inspect;
#[cfg(feature = "cuda")]
mod run;
#[cfg(feature = "cuda")]
mod serve;
#[cfg(any(feature = "cuda", test))]
mod serve_cli;

const USAGE: &str = "ribn <command>\n\ncommands:\n  inspect  inspect a GGUF artifact without GPU initialization\n  run      stream one Qwen GGUF request through the experimental Ribn runtime\n  serve    serve a bounded loopback completion/chat HTTP endpoint";

fn main() {
    let mut arguments = std::env::args().skip(1);
    let command = arguments.next();
    let rest = arguments.collect::<Vec<_>>();
    let result = match command.as_deref() {
        None | Some("-h" | "--help") => {
            println!("{USAGE}");
            Ok(())
        }
        Some("inspect") => inspect::run(&rest),
        Some("run") => run_prepared(&rest),
        Some("serve") => serve_prepared(&rest),
        Some(other) => Err(format!("unknown command {other:?}\n\n{USAGE}")),
    };
    if let Err(error) = result {
        eprintln!("ribn: {error}");
        std::process::exit(2);
    }
}

#[cfg(feature = "cuda")]
fn run_prepared(arguments: &[String]) -> Result<(), String> {
    run::run(arguments)
}

#[cfg(feature = "cuda")]
fn serve_prepared(arguments: &[String]) -> Result<(), String> {
    serve::run(arguments)
}

#[cfg(not(feature = "cuda"))]
fn serve_prepared(_arguments: &[String]) -> Result<(), String> {
    Err("Qwen serving requires building ribn with --features cuda".to_owned())
}

#[cfg(not(feature = "cuda"))]
fn run_prepared(_arguments: &[String]) -> Result<(), String> {
    Err("Qwen inference requires building ribn with --features cuda".to_owned())
}
