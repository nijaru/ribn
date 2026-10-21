use std::io;
use std::path::PathBuf;

use ribn_text::{GenerationOptions, Message, ProcessorLimits, TextInput, TextOwner};

use crate::cli::RunOptions;
use crate::output::{display, stream};

const USAGE: &str =
    "ribn run-gemma <package-dir> [--prompt <text> | --file <path>] [--raw] [--max-tokens <n>]";

pub(crate) fn run(arguments: &[String]) -> Result<(), String> {
    if matches!(arguments, [help] if matches!(help.as_str(), "-h" | "--help")) {
        println!(
            "{USAGE}\n\nWithout --raw, text input is sent as one user chat message through the package's embedded chat template. If neither --prompt nor --file is given, piped stdin is used.\n\nExperimental Gemma 4 SafeTensors path on the build's Metal or CUDA backend; the package directory must contain config.json, tokenizer and chat assets. Greedy decoding only. --device and --context-length are not accepted here."
        );
        return Ok(());
    }
    let mut options = parse(arguments)?;
    let mut message = Message::user("");
    let role_bytes = if options.raw { 0 } else { message.role.len() };
    let max_text_bytes = ProcessorLimits::default()
        .max_input_bytes
        .checked_sub(role_bytes)
        .ok_or_else(|| "input byte limit cannot hold the chat role".to_owned())?;
    let input_text = crate::input::read(&mut options, max_text_bytes)?;
    let input = if options.raw {
        TextInput::prompt(input_text)
    } else {
        message.content = input_text;
        TextInput::chat(vec![message])
    };

    eprintln!("preparing the Gemma 4 package on the device (experimental Ribn runtime)");
    let (mut owner, report) = load_model(options.model).map_err(display)?;
    eprintln!("ready: {} encoded model bytes", report.model_bytes);
    let generation = GenerationOptions {
        max_output_tokens: options.max_tokens,
        ..GenerationOptions::default()
    };
    let result = stream(owner.model(), input, generation, &mut io::stdout().lock());
    let shutdown = owner.shutdown().map_err(display);
    match (result, shutdown) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
        (Err(error), Err(cleanup)) => Err(format!("{error}; shutdown also failed: {cleanup}")),
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) fn load_model(
    path: PathBuf,
) -> Result<(TextOwner, ribn_text::GemmaLoadReport), ribn_text::TextError> {
    TextOwner::load_gemma(path)
}

#[cfg(all(feature = "cuda", not(all(feature = "metal", target_os = "macos"))))]
pub(crate) fn load_model(
    path: PathBuf,
) -> Result<(TextOwner, ribn_text::GemmaLoadReport), ribn_text::TextError> {
    TextOwner::load_gemma_cuda(path)
}

fn parse(arguments: &[String]) -> Result<RunOptions, String> {
    let mut model = None;
    let mut prompt = None;
    let mut file = None;
    let mut raw = false;
    let mut max_tokens = 32_u32;
    let mut index = 0;
    while index < arguments.len() {
        let name = &arguments[index];
        if name == "--raw" {
            raw = true;
            index += 1;
        } else if matches!(name.as_str(), "--prompt" | "--file" | "--max-tokens") {
            let value = arguments
                .get(index + 1)
                .ok_or_else(|| format!("{name} requires a value; {USAGE}"))?;
            match name.as_str() {
                "--prompt" => prompt = Some(value.clone()),
                "--file" => file = Some(PathBuf::from(value)),
                _ => {
                    max_tokens = value
                        .parse()
                        .map_err(|_| format!("--max-tokens expects a positive integer; {USAGE}"))?;
                }
            }
            index += 2;
        } else if let Some(value) = name.strip_prefix("--") {
            return Err(format!("unknown run-gemma option {value:?}; {USAGE}"));
        } else if model.replace(PathBuf::from(name)).is_some() {
            return Err(format!(
                "exactly one package directory is required; {USAGE}"
            ));
        } else {
            index += 1;
        }
    }
    if max_tokens == 0 {
        return Err(format!("--max-tokens must be positive; {USAGE}"));
    }
    let model = model.ok_or_else(|| USAGE.to_owned())?;
    if prompt.is_some() && file.is_some() {
        return Err(format!("use either --prompt or --file, not both; {USAGE}"));
    }
    Ok(RunOptions {
        model,
        prompt,
        file,
        raw,
        max_tokens,
        context_length: 4096,
        device: 0,
    })
}
