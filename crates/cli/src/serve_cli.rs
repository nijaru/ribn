use std::collections::HashSet;
use std::net::SocketAddr;
use std::path::PathBuf;

pub(crate) const USAGE: &str = "ribn serve <model.gguf> [--bind <loopback-ip:port>] [--model-name <name>] [--device <ordinal>] [--context-length <n>] [--max-sequences <n>] [--max-connections <n>] [--max-requests <n>] [--max-body-bytes <n>] [--max-response-bytes <n>] [--max-output-tokens <n>] [--connection-deadline-secs <n>] [--sampled-prefill-final]";

#[derive(Debug)]
pub(crate) struct ServeOptions {
    pub model: PathBuf,
    pub bind: SocketAddr,
    pub model_name: String,
    pub device: u16,
    pub context_length: u32,
    pub max_sequences: Option<usize>,
    pub sampled_prefill_final: bool,
    pub max_connections: Option<usize>,
    pub max_requests: Option<usize>,
    pub max_body_bytes: Option<usize>,
    pub max_response_bytes: Option<usize>,
    pub max_output_tokens: Option<u32>,
    pub deadline_secs: Option<u32>,
}

pub(crate) fn parse(arguments: &[String]) -> Result<ServeOptions, String> {
    let mut options = ServeOptions {
        model: PathBuf::new(),
        bind: "127.0.0.1:8000".parse().expect("constant loopback address"),
        model_name: "ribn".to_owned(),
        device: 0,
        context_length: 4096,
        max_sequences: None,
        sampled_prefill_final: false,
        max_connections: None,
        max_requests: None,
        max_body_bytes: None,
        max_response_bytes: None,
        max_output_tokens: None,
        deadline_secs: None,
    };
    let mut model = None;
    let mut seen = HashSet::new();
    let mut args = arguments.iter();
    while let Some(name) = args.next() {
        if !name.starts_with('-') && model.is_none() {
            model = Some(PathBuf::from(name));
            continue;
        }
        if !seen.insert(name) {
            return Err(format!("duplicate option {name}"));
        }
        if name == "--sampled-prefill-final" {
            options.sampled_prefill_final = true;
            continue;
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {name}; {USAGE}"))?;
        match name.as_str() {
            "--bind" => {
                options.bind = value
                    .parse()
                    .map_err(|_| "--bind expects an IP address and port".to_owned())?;
                if !options.bind.ip().is_loopback() {
                    return Err("only loopback binding is supported".to_owned());
                }
            }
            "--model-name" => options.model_name.clone_from(value),
            "--device" => {
                options.device = value
                    .parse()
                    .map_err(|_| "--device expects a non-negative ordinal".to_owned())?;
            }
            "--context-length" => options.context_length = positive(value, name)?,
            "--max-sequences" => options.max_sequences = Some(positive(value, name)? as usize),
            "--max-connections" => options.max_connections = Some(positive(value, name)? as usize),
            "--max-requests" => options.max_requests = Some(positive(value, name)? as usize),
            "--max-body-bytes" => options.max_body_bytes = Some(positive(value, name)? as usize),
            "--max-response-bytes" => {
                options.max_response_bytes = Some(positive(value, name)? as usize);
            }
            "--max-output-tokens" => options.max_output_tokens = Some(positive(value, name)?),
            "--connection-deadline-secs" => options.deadline_secs = Some(positive(value, name)?),
            _ => return Err(format!("unknown serve option {name:?}; {USAGE}")),
        }
    }
    options.model = model.ok_or_else(|| format!("model is required; {USAGE}"))?;
    if options.model_name.is_empty() || options.model_name.len() > 256 {
        return Err("model name must contain 1..=256 bytes".to_owned());
    }
    Ok(options)
}

fn positive(value: &str, name: &str) -> Result<u32, String> {
    let value = value
        .parse::<u32>()
        .map_err(|_| format!("{name} expects a positive integer"))?;
    if value == 0 {
        return Err(format!("{name} must be positive"));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|s| (*s).to_owned()).collect()
    }
    #[test]
    fn parses_serving_and_load_settings_separately() {
        let options = parse(&args(&[
            "m.gguf",
            "--bind",
            "[::1]:9000",
            "--max-sequences",
            "2",
            "--max-requests",
            "8",
            "--model-name",
            "qwen",
        ]))
        .unwrap();
        assert_eq!(options.model, PathBuf::from("m.gguf"));
        assert_eq!(options.bind.port(), 9000);
        assert_eq!(options.max_sequences, Some(2));
        assert_eq!(options.max_requests, Some(8));
        assert_eq!(options.model_name, "qwen");
        assert_eq!(options.context_length, 4096);
        assert!(!options.sampled_prefill_final);
        let experimental = parse(&args(&["m.gguf", "--sampled-prefill-final"])).unwrap();
        assert!(experimental.sampled_prefill_final);
    }
    #[test]
    fn rejects_unresolved_exposure_and_invalid_flags() {
        for values in [
            vec![],
            vec!["m", "--bind", "0.0.0.0:8000"],
            vec!["m", "--max-requests", "0"],
            vec!["m", "--device", "-1"],
            vec!["m", "--raw"],
            vec!["m", "--sampled-prefill-final", "--sampled-prefill-final"],
            vec!["m", "other"],
            vec!["m", "--model-name", "a", "--model-name", "b"],
        ] {
            assert!(parse(&args(&values)).is_err(), "{values:?}");
        }
    }
}
