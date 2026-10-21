use std::collections::HashSet;
use std::net::SocketAddr;
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ServeModel {
    Qwen,
    Gemma,
}

impl ServeModel {
    pub(crate) fn usage(self) -> String {
        let (command, artifact, load) = match self {
            Self::Qwen => (
                "serve",
                "model.gguf",
                " [--device <ordinal>] [--context-length <n>] [--max-sequences <n>]",
            ),
            Self::Gemma => ("serve-gemma", "package-dir", ""),
        };
        format!(
            "ribn {command} <{artifact}>{load} [--bind <loopback-ip:port>] [--model-name <name>] [--max-connections <n>] [--max-requests <n>] [--max-body-bytes <n>] [--max-response-bytes <n>] [--max-output-tokens <n>] [--connection-deadline-secs <n>]"
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum ModelLoad {
    Qwen {
        device: u16,
        context_length: u32,
        max_sequences: Option<usize>,
    },
    Gemma,
}

#[derive(Debug)]
pub(crate) struct ServeOptions {
    pub model: PathBuf,
    pub load: ModelLoad,
    pub bind: SocketAddr,
    pub model_name: String,
    pub max_connections: Option<usize>,
    pub max_requests: Option<usize>,
    pub max_body_bytes: Option<usize>,
    pub max_response_bytes: Option<usize>,
    pub max_output_tokens: Option<u32>,
    pub deadline_secs: Option<u32>,
}

pub(crate) fn parse(arguments: &[String], kind: ServeModel) -> Result<ServeOptions, String> {
    let usage = kind.usage();
    let mut options = ServeOptions {
        model: PathBuf::new(),
        load: match kind {
            ServeModel::Qwen => ModelLoad::Qwen {
                device: 0,
                context_length: 4096,
                max_sequences: None,
            },
            ServeModel::Gemma => ModelLoad::Gemma,
        },
        bind: "127.0.0.1:8000".parse().expect("constant loopback address"),
        model_name: "ribn".to_owned(),
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
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {name}; {usage}"))?;
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
            "--device" | "--context-length" | "--max-sequences" => {
                let ModelLoad::Qwen {
                    device,
                    context_length,
                    max_sequences,
                } = &mut options.load
                else {
                    return Err(format!("{name} is not supported by serve-gemma; {usage}"));
                };
                match name.as_str() {
                    "--device" => {
                        *device = value
                            .parse()
                            .map_err(|_| "--device expects a non-negative ordinal".to_owned())?;
                    }
                    "--context-length" => *context_length = positive(value, name)?,
                    _ => *max_sequences = Some(positive(value, name)? as usize),
                }
            }
            "--max-connections" => options.max_connections = Some(positive(value, name)? as usize),
            "--max-requests" => options.max_requests = Some(positive(value, name)? as usize),
            "--max-body-bytes" => options.max_body_bytes = Some(positive(value, name)? as usize),
            "--max-response-bytes" => {
                options.max_response_bytes = Some(positive(value, name)? as usize);
            }
            "--max-output-tokens" => options.max_output_tokens = Some(positive(value, name)?),
            "--connection-deadline-secs" => options.deadline_secs = Some(positive(value, name)?),
            _ => return Err(format!("unknown serving option {name:?}; {usage}")),
        }
    }
    options.model = model.ok_or_else(|| format!("model is required; {usage}"))?;
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
        for kind in [ServeModel::Qwen, ServeModel::Gemma] {
            let options = parse(
                &args(&[
                    "model",
                    "--bind",
                    "[::1]:9000",
                    "--max-requests",
                    "8",
                    "--model-name",
                    "named",
                ]),
                kind,
            )
            .unwrap();
            assert_eq!(options.model, PathBuf::from("model"));
            assert_eq!(options.bind.port(), 9000);
            assert_eq!(options.max_requests, Some(8));
            assert_eq!(options.model_name, "named");
            assert!(matches!(
                (kind, options.load),
                (
                    ServeModel::Qwen,
                    ModelLoad::Qwen {
                        device: 0,
                        context_length: 4096,
                        max_sequences: None
                    }
                ) | (ServeModel::Gemma, ModelLoad::Gemma)
            ));
        }
        let options = parse(
            &args(&[
                "m.gguf",
                "--max-sequences",
                "2",
                "--device",
                "1",
                "--context-length",
                "512",
            ]),
            ServeModel::Qwen,
        )
        .unwrap();
        assert!(matches!(
            options.load,
            ModelLoad::Qwen {
                device: 1,
                context_length: 512,
                max_sequences: Some(2)
            }
        ));
    }
    #[test]
    fn gemma_rejects_unimplemented_load_controls_instead_of_ignoring_them() {
        for (name, value) in [
            ("--device", "0"),
            ("--context-length", "4096"),
            ("--max-sequences", "4"),
        ] {
            let error = parse(&args(&["package", name, value]), ServeModel::Gemma).unwrap_err();
            assert!(
                error.contains(&format!("{name} is not supported")),
                "{error}"
            );
        }
    }
    #[test]
    fn rejects_unresolved_exposure_and_invalid_flags() {
        for kind in [ServeModel::Qwen, ServeModel::Gemma] {
            for values in [
                vec![],
                vec!["m", "--bind", "0.0.0.0:8000"],
                vec!["m", "--max-requests", "0"],
                vec!["m", "--device", "-1"],
                vec!["m", "--raw"],
                vec!["m", "other"],
                vec!["m", "--model-name", "a", "--model-name", "b"],
            ] {
                assert!(parse(&args(&values), kind).is_err(), "{kind:?} {values:?}");
            }
        }
    }
}
