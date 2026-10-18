use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::ops::Range;
use std::sync::{Arc, Mutex};

use super::SafeTensorArtifact;

pub(super) enum Storage {
    Bytes(Arc<[u8]>),
    File(Mutex<FileCursor>),
}

pub(super) struct FileCursor {
    pub file: File,
    pub position: Option<u64>,
}

/// Reader owning the selected artifact and a bounded independent cursor.
/// Reads borrow caller storage; there is no internal payload buffering/prefetch.
/// Dropping an artifact or evicting its cache entry does not invalidate a reader.
pub struct ArtifactReader {
    artifact: SafeTensorArtifact,
    position: usize,
    end: usize,
}

impl ArtifactReader {
    pub(super) fn new(artifact: SafeTensorArtifact, range: Range<usize>) -> Self {
        Self {
            artifact,
            position: range.start,
            end: range.end,
        }
    }
}

impl Read for ArtifactReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let requested = buffer.len().min(self.end - self.position);
        if requested == 0 {
            return Ok(0);
        }
        let read = match &self.artifact.inner.storage {
            Storage::Bytes(bytes) => {
                buffer[..requested]
                    .copy_from_slice(&bytes[self.position..self.position + requested]);
                requested
            }
            Storage::File(source) => {
                let mut source = source
                    .lock()
                    .map_err(|_| io::Error::other("SafeTensors source lock poisoned"))?;
                let offset = u64::try_from(self.position).map_err(io::Error::other)?;
                if source.position != Some(offset) {
                    source.position = None;
                    source.file.seek(SeekFrom::Start(offset))?;
                }
                // A failed read/seek may leave the descriptor at an unknown
                // position. The next attempt must seek, even at the same offset.
                source.position = None;
                let read = source.file.read(&mut buffer[..requested])?;
                source.position = Some(offset + u64::try_from(read).map_err(io::Error::other)?);
                if read == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "selected SafeTensors file was truncated",
                    ));
                }
                read
            }
        };
        self.position += read;
        Ok(read)
    }
}
