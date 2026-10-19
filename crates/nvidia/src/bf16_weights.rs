//! Native BF16 storage, independent of GGUF/F32 materialization.

mod storage;

#[cfg(feature = "cuda")]
pub use storage::CudaBf16WeightsError;

#[cfg(feature = "cuda")]
mod cuda {
    use std::{io::Read, ops::Range, sync::Arc};

    use cudarc::driver::{CudaStream, result};
    use engine_core::WeightTensorSpec;
    use ribn_foundation::{AllocationId, BytePool};

    use super::CudaBf16WeightsError;
    use super::storage::{Storage, Transport};

    /// One immutable, native-BF16 weight set on one CUDA stream.
    ///
    /// Slots preserve source shape, order and bytes; each occupies a 16-byte-rounded
    /// region of one synchronous allocation. Metadata is available while loading,
    /// but no data can be read until every slot uploads successfully. No mutable
    /// device views or raw pointers are exposed publicly. Future backend operations
    /// must use this same stream and retain this owner through their completion.
    ///
    /// The pool charges requested allocation bytes including padding, not driver,
    /// context or allocation-rounding overhead. Staging is a separate host envelope.
    /// Call `close` explicitly for cleanup errors; failed drains retain ownership for
    /// retry. An uncertain free is permanently quarantined, including on Drop.
    pub struct CudaBf16Weights {
        storage: Storage<CudaTransport>,
    }

    impl CudaBf16Weights {
        /// Validate the complete inventory before reserving or allocating anything.
        /// Own one pageable staging buffer before any transfer can be submitted.
        ///
        /// # Errors
        /// Rejects empty/duplicate/non-BF16 inventories, byte/offset/host-size overflow,
        /// zero staging capacity, pool refusal, host allocation or CUDA errors.
        pub fn allocate(
            specs: Vec<WeightTensorSpec>,
            stream: Arc<CudaStream>,
            pool: &Arc<BytePool>,
            staging_bytes: usize,
        ) -> Result<Self, CudaBf16WeightsError> {
            Ok(Self {
                storage: Storage::allocate(specs, CudaTransport { stream }, pool, staging_bytes)?,
            })
        }

        /// Upload exactly the slot's declared bytes, without conversion/repacking.
        /// Each chunk completes on the associated stream before staging is reused.
        ///
        /// # Errors
        /// Rejects invalid/already uploaded slots and poisoned/closed sets. Source
        /// errors (including short/long input) or transfer errors poison the whole
        /// set: retain it and call `close`, never retry an upload.
        pub fn upload(
            &mut self,
            slot: usize,
            reader: &mut impl Read,
        ) -> Result<(), CudaBf16WeightsError> {
            self.storage.upload(slot, reader)
        }

        #[must_use]
        pub fn ready(&self) -> bool {
            self.storage.ready()
        }

        /// Read at most the staging capacity after complete publication. Byte offsets
        /// may split BF16 elements. The returned allocation is caller-owned, outside
        /// this owner's retained staging envelope; there is no prefetch.
        ///
        /// # Errors
        /// Rejects nonready sets, invalid ranges and oversized chunks. Transfer/drain
        /// errors poison the set and retain the DMA destination in this owner.
        pub fn read_chunk(
            &mut self,
            slot: usize,
            byte_offset: u64,
            len: usize,
        ) -> Result<Vec<u8>, CudaBf16WeightsError> {
            self.storage.read_chunk(slot, byte_offset, len)
        }

        /// Drain the associated stream, synchronously free, then refund the pool.
        /// Idempotent on success. A failed drain can be retried; a failed free cannot
        /// (the pointer might already have been released/reused).
        ///
        /// # Errors
        /// Returns CUDA drain/free errors, or permanent uncertain-free quarantine.
        pub fn close(&mut self) -> Result<(), CudaBf16WeightsError> {
            self.storage.close()
        }

        /// Requested CUDA allocation size, including each slot's trailing padding.
        #[must_use]
        pub fn allocated_bytes(&self) -> u64 {
            self.storage.allocated_bytes()
        }

        /// Identity of the single pool grant, retained as metadata after close.
        #[must_use]
        pub fn allocation_id(&self) -> AllocationId {
            self.storage.allocation_id()
        }

        #[must_use]
        pub fn spec(&self, slot: usize) -> Option<&WeightTensorSpec> {
            self.storage.spec(slot)
        }

        /// Canonical byte range in the allocation, excluding padding; metadata only.
        #[must_use]
        pub fn tensor_range(&self, slot: usize) -> Option<Range<u64>> {
            self.storage.tensor_range(slot)
        }
    }

    struct CudaTransport {
        // Retains the context as well. No CudaSlice or event-based destructor owns
        // this raw allocation; Storage alone controls free and lease retirement.
        stream: Arc<CudaStream>,
    }

    fn driver_error(operation: &'static str, source: result::DriverError) -> CudaBf16WeightsError {
        CudaBf16WeightsError::Transport {
            operation,
            source: Box::new(source),
        }
    }

    impl Transport for CudaTransport {
        fn allocate(&mut self, bytes: usize) -> Result<u64, CudaBf16WeightsError> {
            self.stream
                .context()
                .bind_to_thread()
                .map_err(|e| driver_error("bind", e))?;
            // SAFETY: positive, validated byte count; uninitialized storage is used
            // only as bytes, and never published until all slots are initialized.
            // Return directly: no fallible tracking-event creation after malloc.
            unsafe { result::malloc_sync(bytes) }.map_err(|e| driver_error("allocate", e))
        }

        unsafe fn upload(&mut self, dst: u64, src: &[u8]) -> Result<(), CudaBf16WeightsError> {
            self.stream
                .context()
                .bind_to_thread()
                .map_err(|e| driver_error("bind", e))?;
            // SAFETY: Storage supplies a live in-bounds allocation range and retains
            // its boxed staging unchanged until a successful same-stream drain.
            unsafe { result::memcpy_htod_async(dst, src, self.stream.cu_stream()) }
                .map_err(|e| driver_error("upload", e))
        }

        unsafe fn read(&mut self, src: u64, dst: &mut [u8]) -> Result<(), CudaBf16WeightsError> {
            self.stream
                .context()
                .bind_to_thread()
                .map_err(|e| driver_error("bind", e))?;
            // SAFETY: same ownership/range guarantees as upload, with exclusive
            // access to the owner-held DMA destination until successful drain.
            unsafe { result::memcpy_dtoh_async(dst, src, self.stream.cu_stream()) }
                .map_err(|e| driver_error("read", e))
        }

        fn drain(&mut self) -> Result<(), CudaBf16WeightsError> {
            self.stream
                .synchronize()
                .map_err(|e| driver_error("drain", e))
        }

        unsafe fn free(&mut self, ptr: u64) -> Result<(), CudaBf16WeightsError> {
            self.stream
                .context()
                .bind_to_thread()
                .map_err(|e| driver_error("bind", e))?;
            // SAFETY: Storage has drained this stream and will never retry this
            // invocation, even on error. The pointer came from malloc_sync.
            unsafe { result::free_sync(ptr) }.map_err(|e| driver_error("free", e))
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda::CudaBf16Weights;
