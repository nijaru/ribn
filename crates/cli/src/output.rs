use std::io::Write;

use ribn_text::{FinishReason, GenerationOptions, TextEvent, TextInput, TextModel};

pub(crate) fn stream(
    model: &TextModel,
    input: TextInput,
    options: GenerationOptions,
    output: &mut impl Write,
) -> Result<(), String> {
    let events = model.stream_blocking(input, options).map_err(display)?;
    for event in events {
        match event.map_err(display)? {
            TextEvent::Delta { text, .. } => {
                output.write_all(text.as_bytes()).map_err(display)?;
                output.flush().map_err(display)?;
            }
            TextEvent::Finished { reason, .. } => {
                return match reason {
                    FinishReason::Length | FinishReason::Stop => Ok(()),
                    FinishReason::Cancelled => Err("generation was cancelled".to_owned()),
                    FinishReason::Failed(error) => Err(display(error)),
                };
            }
        }
    }
    Err("generation ended without a terminal event".to_owned())
}

pub(crate) fn display(error: impl std::fmt::Display) -> String {
    error.to_string()
}
