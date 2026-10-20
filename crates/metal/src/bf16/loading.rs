//! CPU initialization of private Shared allocations before immutable publication.

use super::{MetalBf16Buffer, MetalBf16Error, MetalBf16Executor};
use metal::foreign_types::ForeignType;
use metal::objc::{Message, runtime::Sel};
use metal::{Buffer, MTLBuffer, MTLResourceOptions};
use std::io::{ErrorKind, Read};

impl MetalBf16Executor {
    /// Copy native BF16 encodings into one immutable shared-storage buffer.
    /// # Errors
    /// Rejects empty/oversized input or a failed allocation.
    pub fn upload(&self, values: &[u16]) -> Result<MetalBf16Buffer, MetalBf16Error> {
        let length = bf16_bytes(values.len())?;
        let buffer = self.allocate_shared(length)?;
        // SAFETY: the new Shared allocation owns `length` writable bytes and
        // cannot alias the live input slice. No view or GPU access exists yet.
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr().cast::<u8>(),
                buffer.contents().cast::<u8>(),
                length,
            );
        }
        Ok(MetalBf16Buffer {
            buffer,
            elements: values.len(),
        })
    }

    /// Read exactly `elements` little-endian BF16 encodings, followed by EOF.
    ///
    /// Supported macOS hosts are little-endian. Reads use at most 64 KiB of
    /// scratch, copying into final Shared storage without a second weight buffer.
    /// No mutable view or GPU command sees the allocation before completion.
    /// # Errors
    /// Rejects empty/oversized shapes, allocation failure, short/trailing data or
    /// reader errors. An incomplete allocation is released, never published.
    pub fn read_bf16(
        &self,
        elements: usize,
        reader: &mut impl Read,
    ) -> Result<MetalBf16Buffer, MetalBf16Error> {
        let length = bf16_bytes(elements)?;
        let buffer = self.allocate_shared(length)?;
        let mut scratch = Vec::new();
        let scratch_len = length.min(64 * 1024);
        scratch
            .try_reserve_exact(scratch_len)
            .map_err(|error| MetalBf16Error::Allocation(format!("BF16 read scratch: {error}")))?;
        scratch.resize(scratch_len, 0_u8);
        let mut offset = 0;
        while offset < length {
            let count = (length - offset).min(scratch.len());
            reader
                .read_exact(&mut scratch[..count])
                .map_err(MetalBf16Error::Read)?;
            // SAFETY: this private allocation owns `length` writable bytes;
            // offset + count <= length. Scratch is initialized and disjoint.
            // Raw writes avoid forming references to uninitialized GPU storage.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    scratch.as_ptr(),
                    buffer.contents().cast::<u8>().add(offset),
                    count,
                );
            }
            offset += count;
        }
        let mut trailing = [0];
        loop {
            match reader.read(&mut trailing) {
                Ok(0) => break,
                Ok(_) => return Err(MetalBf16Error::Invalid("trailing BF16 source bytes")),
                Err(error) if error.kind() == ErrorKind::Interrupted => {}
                Err(error) => return Err(MetalBf16Error::Read(error)),
            }
        }
        Ok(MetalBf16Buffer { buffer, elements })
    }

    pub(super) fn allocate_shared(&self, length: usize) -> Result<Buffer, MetalBf16Error> {
        let bytes = u64::try_from(length)
            .map_err(|_| MetalBf16Error::Invalid("Metal buffer length exceeds u64"))?;
        if length == 0 || length > isize::MAX as usize || bytes > self.device.max_buffer_length() {
            return Err(MetalBf16Error::Invalid(
                "Metal buffer length exceeds device or host limits",
            ));
        }
        // SAFETY: matches MTLDevice's nullable newBufferWithLength:options: ABI.
        // `new` returns a +1 owned reference. Check nil before constructing the
        // Rust owner; metal-rs's non-null-returning wrapper cannot express OOM.
        let pointer: *mut MTLBuffer = unsafe {
            self.device.send_message(
                Sel::register("newBufferWithLength:options:"),
                (bytes, MTLResourceOptions::StorageModeShared),
            )
        }
        .map_err(|error| MetalBf16Error::Allocation(format!("{bytes} bytes: {error}")))?;
        if pointer.is_null() {
            return Err(MetalBf16Error::Allocation(format!(
                "{bytes} bytes: device returned nil"
            )));
        }
        // SAFETY: the checked non-null pointer is the new +1 MTLBuffer reference.
        Ok(unsafe { Buffer::from_ptr(pointer) })
    }
}

fn bf16_bytes(elements: usize) -> Result<usize, MetalBf16Error> {
    elements
        .checked_mul(2)
        .filter(|&bytes| bytes > 0)
        .ok_or(MetalBf16Error::Invalid("empty or overflowing BF16 buffer"))
}
