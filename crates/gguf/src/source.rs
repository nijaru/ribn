//! Shared, cursor-independent access to the one opened artifact.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::sync::{Arc, Mutex};

#[derive(Debug)]
pub(super) struct Source {
    pub file: File,
    pub position: Option<u64>,
}

/// Bounded reader retaining the selected GGUF source, not reopening its pathname.
/// Readers share one descriptor and have independent logical cursors. The source
/// file must not be modified in place while any owner or reader remains alive.
pub struct GgufReader {
    source: Arc<Mutex<Source>>,
    offset: u64,
    pub(super) remaining: u64,
}

impl GgufReader {
    pub(super) fn new(source: Arc<Mutex<Source>>, range: std::ops::Range<u64>) -> Self {
        Self {
            source,
            offset: range.start,
            remaining: range.end - range.start,
        }
    }
}

impl Read for GgufReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if self.remaining == 0 || buffer.is_empty() {
            return Ok(0);
        }
        let requested = usize::try_from(self.remaining)
            .unwrap_or(usize::MAX)
            .min(buffer.len());
        let mut source = self
            .source
            .lock()
            .map_err(|_| io::Error::other("GGUF source lock poisoned"))?;
        if source.position != Some(self.offset) {
            source.position = None;
            source.file.seek(SeekFrom::Start(self.offset))?;
        }
        // An I/O error need not establish the descriptor's resulting position.
        // The next attempt must seek again; sequential successful reads needn't.
        source.position = None;
        let read = source.file.read(&mut buffer[..requested])?;
        self.offset += read as u64;
        self.remaining -= read as u64;
        source.position = Some(self.offset);
        Ok(read)
    }
}
