//! Owned text delivery with request-local model decoding.

use ribn::driver::GenerationStream;
use ribn::{Event, FinishReason, RequestId, Usage};

use crate::error::TextError;
use crate::process::RequestDecoder;

/// Incremental text event. Token IDs remain available even when one token does
/// not independently form valid UTF-8 and therefore has an empty text delta.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TextEvent {
    /// Irrevocable text. `token` is absent only for a decoder's terminal flush;
    /// that flush may contain text withheld from many earlier token events.
    Delta {
        token: Option<u32>,
        text: String,
    },
    Finished {
        reason: FinishReason,
        usage: Usage,
    },
}

/// One owned text request. Dropping an unfinished stream abandons delivery
/// without waiting for device completion; the execution owner retains retirement.
pub struct TextStream {
    request: RequestId,
    stream: Option<GenerationStream>,
    decoder: Option<RequestDecoder>,
    pending: Option<(FinishReason, Usage)>,
    done: bool,
}

impl TextStream {
    pub(crate) fn new(decoder: RequestDecoder, stream: GenerationStream) -> Self {
        let request = stream.request_id();
        Self {
            request,
            stream: Some(stream),
            decoder: Some(decoder),
            pending: None,
            done: false,
        }
    }

    /// Runtime identity allocated before this stream was returned.
    #[must_use]
    pub const fn request_id(&self) -> RequestId {
        self.request
    }

    /// Record cancellation intent. Buffered output is preserved and an ordinary
    /// `Cancelled` terminal still arrives; this does not wait for the device.
    pub fn cancel(&self) {
        if let Some(stream) = &self.stream {
            stream.cancel();
        }
    }

    /// Await one event. Cancelling this future loses no event.
    pub async fn next_async(&mut self) -> Option<Result<TextEvent, TextError>> {
        if self.done {
            return None;
        }
        if let Some((reason, usage)) = self.pending.take() {
            self.done = true;
            return Some(Ok(TextEvent::Finished { reason, usage }));
        }
        let event = self.stream.as_mut()?.next().await;
        Some(self.accept_next(event))
    }

    /// Blocking counterpart of [`Self::next_async`]. Not for async executor tasks.
    pub fn next_blocking(&mut self) -> Option<Result<TextEvent, TextError>> {
        if self.done {
            return None;
        }
        if let Some((reason, usage)) = self.pending.take() {
            self.done = true;
            return Some(Ok(TextEvent::Finished { reason, usage }));
        }
        let event = self.stream.as_mut()?.next_blocking();
        Some(self.accept_next(event))
    }

    fn accept_next(
        &mut self,
        event: Option<Result<Event, ribn::driver::DriverError>>,
    ) -> Result<TextEvent, TextError> {
        match event {
            Some(event) => self.accept(event),
            None => Err(self.abandon(TextError::MissingTerminal)),
        }
    }

    fn accept(
        &mut self,
        event: Result<Event, ribn::driver::DriverError>,
    ) -> Result<TextEvent, TextError> {
        match event {
            Ok(Event::Token { token, .. }) => {
                let text = self
                    .decoder
                    .as_mut()
                    .expect("active stream owns its decoder")
                    .push(token)
                    .map_err(|error| self.abandon(error))?;
                Ok(TextEvent::Delta {
                    token: Some(token),
                    text,
                })
            }
            Ok(Event::Finished { reason, usage, .. }) => {
                let trailing = self
                    .decoder
                    .take()
                    .expect("active stream owns its decoder")
                    .finish()
                    .map_err(|error| self.abandon(error))?;
                // No decoder/assets or driver delivery remain in a terminated stream.
                self.stream = None;
                if trailing.is_empty() {
                    self.done = true;
                    return Ok(TextEvent::Finished { reason, usage });
                }
                self.pending = Some((reason, usage));
                Ok(TextEvent::Delta {
                    token: None,
                    text: trailing,
                })
            }
            // Owner failure preserves already-delivered events and does not
            // invent a successful terminal flush or usage.
            Err(error) => Err(self.abandon(TextError::Owner(error))),
        }
    }

    /// A request-local decode failure abandons only this request.
    fn abandon(&mut self, error: TextError) -> TextError {
        self.done = true;
        self.decoder = None;
        // Dropping the driver stream records discard intent without joining.
        self.stream = None;
        error
    }
}

impl Iterator for TextStream {
    type Item = Result<TextEvent, TextError>;

    /// Blocking consumption. Async clients call [`TextStream::next_async`].
    fn next(&mut self) -> Option<Self::Item> {
        self.next_blocking()
    }
}
