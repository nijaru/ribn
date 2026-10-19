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

/// Largest supported number of stop strings in one request.
pub const MAX_STOP_SEQUENCES: usize = 4;
/// Largest supported total stop-string bytes in one request.
pub const MAX_STOP_BYTES: usize = 256;

/// Bounded stop-string matching over incrementally decoded text.
///
/// A suffix that could still complete a configured stop is withheld, so a
/// streamed delta never publishes a stop sequence before it is detected. The
/// first match (earliest start, then longest) suppresses the stop text and
/// everything after it.
struct StopMatcher {
    sequences: Vec<String>,
    held: String,
    hit: bool,
}

pub(crate) fn validate_stops(sequences: &[String]) -> Result<(), TextError> {
    if sequences.len() > MAX_STOP_SEQUENCES {
        return Err(TextError::InvalidInput(
            "at most four stop sequences are supported",
        ));
    }
    let mut bytes = 0_usize;
    for stop in sequences {
        if stop.is_empty() {
            return Err(TextError::InvalidInput("stop sequences must not be empty"));
        }
        bytes = bytes
            .checked_add(stop.len())
            .ok_or(TextError::InvalidInput("stop sequence bytes overflow"))?;
    }
    if bytes > MAX_STOP_BYTES {
        return Err(TextError::InvalidInput(
            "stop sequences exceed the byte bound",
        ));
    }
    Ok(())
}

impl StopMatcher {
    fn new(sequences: Vec<String>) -> Result<Option<Self>, TextError> {
        if sequences.is_empty() {
            return Ok(None);
        }
        validate_stops(&sequences)?;
        Ok(Some(Self {
            sequences,
            held: String::new(),
            hit: false,
        }))
    }

    /// Text that is irrevocably outside every stop; the remainder stays held.
    fn push(&mut self, text: &str) -> String {
        if self.hit {
            return String::new();
        }
        self.held.push_str(text);
        if let Some(start) = self.earliest_match() {
            self.hit = true;
            let mut matched = std::mem::take(&mut self.held);
            matched.truncate(start);
            return matched;
        }
        let keep = self.pending_prefix_len();
        let safe = self.held.len() - keep;
        let pending = self.held.split_off(safe);
        std::mem::replace(&mut self.held, pending)
    }

    /// Remaining text at the terminal boundary: a complete stop still truncates,
    /// but an incomplete prefix is ordinary text and must be delivered.
    fn finish(&mut self) -> String {
        if self.hit {
            self.held.clear();
            return String::new();
        }
        if let Some(start) = self.earliest_match() {
            self.hit = true;
            let mut matched = std::mem::take(&mut self.held);
            matched.truncate(start);
            return matched;
        }
        std::mem::take(&mut self.held)
    }

    fn earliest_match(&self) -> Option<usize> {
        let mut best: Option<(usize, usize)> = None;
        for stop in &self.sequences {
            if let Some(start) = self.held.find(stop.as_str()) {
                let better = match best {
                    None => true,
                    Some((current, length)) => {
                        start < current || (start == current && stop.len() > length)
                    }
                };
                if better {
                    best = Some((start, stop.len()));
                }
            }
        }
        best.map(|(start, _)| start)
    }

    /// Longest suffix of held text that is a proper prefix of any stop sequence.
    fn pending_prefix_len(&self) -> usize {
        let mut keep = 0;
        let bytes = self.held.as_bytes();
        for stop in &self.sequences {
            let limit = bytes.len().min(stop.len() - 1);
            for length in (1..=limit).rev() {
                if bytes.ends_with(&stop.as_bytes()[..length]) {
                    keep = keep.max(length);
                    break;
                }
            }
        }
        keep
    }
}

/// One owned text request. Dropping an unfinished stream abandons delivery
/// without waiting for device completion; the execution owner retains retirement.
pub struct TextStream {
    request: RequestId,
    stream: Option<GenerationStream>,
    decoder: Option<RequestDecoder>,
    pending: Option<(FinishReason, Usage)>,
    stops: Option<StopMatcher>,
    stopping: bool,
    done: bool,
}

impl TextStream {
    pub(crate) fn new(
        decoder: RequestDecoder,
        stream: GenerationStream,
        stops: Vec<String>,
    ) -> Result<Self, TextError> {
        let request = stream.request_id();
        Ok(Self {
            request,
            stream: Some(stream),
            decoder: Some(decoder),
            pending: None,
            stops: StopMatcher::new(stops)?,
            stopping: false,
            done: false,
        })
    }

    fn stopped(&self) -> bool {
        self.stops.as_ref().is_some_and(|stops| stops.hit)
    }

    /// Match one decoded token, cancelling the runtime once a stop is complete.
    fn delta(&mut self, token: Option<u32>, text: String) -> TextEvent {
        let text = match self.stops.as_mut() {
            None => text,
            Some(stops) => {
                let emitted = stops.push(&text);
                self.cancel_after_stop();
                emitted
            }
        };
        TextEvent::Delta { token, text }
    }

    fn cancel_after_stop(&mut self) {
        if self.stopping || !self.stopped() {
            return;
        }
        self.stopping = true;
        if let Some(stream) = &self.stream {
            stream.cancel();
        }
    }

    /// Decoder terminal flush; a stop may complete exactly on this boundary.
    fn flush(&mut self, trailing: String) -> String {
        match self.stops.as_mut() {
            None => trailing,
            Some(stops) => {
                let mut text = stops.push(&trailing);
                text.push_str(&stops.finish());
                text
            }
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
                Ok(self.delta(Some(token), text))
            }
            Ok(Event::Finished { reason, usage, .. }) => {
                let trailing = self
                    .decoder
                    .take()
                    .expect("active stream owns its decoder")
                    .finish()
                    .map_err(|error| self.abandon(error))?;
                let trailing = self.flush(trailing);
                let reason = if self.stopped() {
                    FinishReason::Stop
                } else {
                    reason
                };
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

#[cfg(test)]
mod tests {
    use super::*;

    fn matcher(stops: &[&str]) -> StopMatcher {
        StopMatcher::new(stops.iter().map(|stop| (*stop).to_owned()).collect())
            .unwrap()
            .unwrap()
    }

    #[test]
    fn stop_matching_withholds_possible_prefixes_and_cross_token_matches() {
        let mut stops = matcher(&["END"]);
        assert_eq!(stops.push("hel"), "hel");
        assert_eq!(stops.push("lo E"), "lo ");
        assert!(!stops.hit);
        assert_eq!(stops.push("ND after"), "");
        assert!(stops.hit);
        assert_eq!(stops.push("ignored"), "");
        assert_eq!(stops.finish(), "");
    }

    #[test]
    fn stop_matching_prefers_the_earliest_then_longest_sequence() {
        let mut stops = matcher(&["bc", "abc", "c"]);
        assert_eq!(stops.push("zabc"), "z");
        assert!(stops.hit);
        let mut trailing = matcher(&["xx", "x"]);
        assert_eq!(trailing.push("yxx"), "y");
        assert!(trailing.hit);
    }

    #[test]
    fn terminal_flush_delivers_incomplete_prefixes_but_truncates_full_stops() {
        let mut incomplete = matcher(&["END"]);
        assert_eq!(incomplete.push("done EN"), "done ");
        assert_eq!(incomplete.finish(), "EN");
        assert!(!incomplete.hit);
        let mut complete = matcher(&["END"]);
        assert_eq!(complete.push("done END"), "done ");
        assert!(complete.hit);
        assert_eq!(complete.finish(), "");
    }

    #[test]
    fn stop_bounds_and_empty_sequences_reject() {
        assert!(StopMatcher::new(Vec::new()).unwrap().is_none());
        for stops in [
            vec![String::new()],
            (0..=MAX_STOP_SEQUENCES).map(|_| "a".to_owned()).collect(),
            vec!["a".repeat(MAX_STOP_BYTES + 1)],
        ] {
            assert!(StopMatcher::new(stops).is_err());
        }
    }
}
