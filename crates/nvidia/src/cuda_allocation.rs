//! Shared raw CUDA allocation ownership; logical initialization belongs to callers.

use std::{error::Error, fmt, io::Read, mem::ManuallyDrop, sync::Arc};

use ribn_foundation::{AllocationId, BytePool, PoolLease, ReserveError};

#[derive(Debug)]
pub enum CudaBf16Error {
    Invalid(&'static str),
    InvalidSlot(usize),
    AlreadyUploaded(usize),
    NotReady,
    Poisoned,
    Closed,
    InUse,
    FreeUncertain,
    Source(std::io::Error),
    SourceTooLong,
    Reserve(ReserveError),
    Staging(std::collections::TryReserveError),
    Transport {
        operation: &'static str,
        source: Box<dyn Error + Send + Sync>,
    },
}

impl fmt::Display for CudaBf16Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(reason) => write!(f, "invalid BF16 storage: {reason}"),
            Self::InvalidSlot(slot) => write!(f, "invalid BF16 slot {slot}"),
            Self::AlreadyUploaded(slot) => write!(f, "BF16 slot {slot} already uploaded"),
            Self::NotReady => f.write_str("BF16 set is not fully uploaded"),
            Self::Poisoned => f.write_str("BF16 set is poisoned; close it"),
            Self::Closed => f.write_str("BF16 storage is closed"),
            Self::InUse => f.write_str("BF16 allocation is in use"),
            Self::FreeUncertain => {
                f.write_str("BF16 resource release uncertain; permanently quarantined")
            }
            Self::Source(e) => write!(f, "BF16 source: {e}"),
            Self::SourceTooLong => f.write_str("BF16 source exceeds declared bytes"),
            Self::Reserve(e) => write!(f, "BF16 reservation: {e}"),
            Self::Staging(e) => write!(f, "BF16 staging: {e}"),
            Self::Transport { operation, source } => write!(f, "BF16 {operation}: {source}"),
        }
    }
}

impl Error for CudaBf16Error {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Source(e) => Some(e),
            Self::Reserve(e) => Some(e),
            Self::Staging(e) => Some(e),
            Self::Transport { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}

type Result<T> = std::result::Result<T, CudaBf16Error>;

// Private FFI boundary, not a pluggable storage/allocator API. Successful drain
// proves all accesses on this transport's one stream have completed. Failed copy
// calls may have enqueued work; failed free calls may have released the pointer.
pub(crate) trait Transport {
    fn allocate(&mut self, bytes: usize) -> Result<u64>;
    // Caller retains host storage unchanged, and the live device range, through
    // successful drain (including after errors). Read additionally needs exclusive
    // host access; its device bytes must be initialized.
    unsafe fn upload(&mut self, dst: u64, src: &[u8]) -> Result<()>;
    unsafe fn read(&mut self, src: u64, dst: &mut [u8]) -> Result<()>;
    fn drain(&mut self) -> Result<()>;
    // Caller has drained all accesses and must not invoke free twice, even on error.
    unsafe fn free(&mut self, ptr: u64) -> Result<()>;
}

struct Resources<T> {
    transport: T,
    pointer: Option<u64>,
    _lease: PoolLease,
    staging: Box<[u8]>,
    free_uncertain: bool,
}

// Logical owners must prohibit IO after any IO error/unwind: DMA may still hold
// staging. Only retirement may proceed until completion is established. Shared
// operation scopes likewise retain their Arc through successful stream drain.
pub(crate) struct Allocation<T: Transport> {
    // No implicit refund/staging destruction if an FFI wrapper panics during
    // defensive cleanup. Only acknowledged retirement unwraps this envelope.
    resources: Option<ManuallyDrop<Resources<T>>>,
    bytes: u64,
    staging_bytes: usize,
    id: AllocationId,
}

impl<T: Transport> Allocation<T> {
    pub(crate) fn allocate(
        bytes: u64,
        transport: T,
        pool: &Arc<BytePool>,
        staging_bytes: usize,
    ) -> Result<Arc<Self>> {
        if bytes == 0 || staging_bytes == 0 {
            return Err(CudaBf16Error::Invalid(
                "zero allocation or staging capacity",
            ));
        }
        let host_bytes = usize::try_from(bytes)
            .map_err(|_| CudaBf16Error::Invalid("allocation exceeds usize"))?;
        let mut staging = Vec::new();
        staging
            .try_reserve_exact(staging_bytes)
            .map_err(CudaBf16Error::Staging)?;
        staging.resize(staging_bytes, 0);
        let lease = pool.reserve(bytes).map_err(CudaBf16Error::Reserve)?;
        let allocation = lease.allocation();
        let mut owner = Self {
            resources: Some(ManuallyDrop::new(Resources {
                transport,
                pointer: None,
                _lease: lease,
                staging: staging.into_boxed_slice(),
                free_uncertain: false,
            })),
            bytes,
            staging_bytes,
            id: allocation,
        };
        let resources = owner
            .resources
            .as_deref_mut()
            .expect("new owner has resources");
        // Capture immediately on success: no fallible event creation or wrapping
        // after malloc. The complete retirement guard already exists on failure.
        resources.pointer = Some(resources.transport.allocate(host_bytes)?);
        Ok(Arc::new(owner))
    }

    pub(crate) fn allocated_bytes(&self) -> u64 {
        self.bytes
    }

    pub(crate) fn allocation_id(&self) -> AllocationId {
        self.id
    }

    pub(crate) fn staging_bytes(&self) -> usize {
        self.staging_bytes
    }

    // Access scopes must retain this Arc through successful same-stream drain;
    // Arc ownership alone is not a completion dependency.
    pub(crate) fn transport(&self) -> &T {
        &self
            .resources
            .as_deref()
            .expect("live allocation")
            .transport
    }

    pub(crate) fn pointer(&self) -> Result<u64> {
        let resources = self.resources.as_deref().ok_or(CudaBf16Error::Closed)?;
        if resources.free_uncertain {
            return Err(CudaBf16Error::FreeUncertain);
        }
        resources.pointer.ok_or(CudaBf16Error::Closed)
    }

    fn checked_address(&self, offset: u64, len: u64) -> Result<u64> {
        let end = offset
            .checked_add(len)
            .ok_or(CudaBf16Error::Invalid("offset overflow"))?;
        if end > self.bytes {
            return Err(CudaBf16Error::Invalid("range exceeds allocation"));
        }
        let base = self.pointer()?;
        base.checked_add(end)
            .ok_or(CudaBf16Error::Invalid("device address overflow"))?;
        Ok(base + offset)
    }

    pub(crate) fn write_exact(
        &mut self,
        offset: u64,
        len: u64,
        reader: &mut impl Read,
    ) -> Result<()> {
        let mut dst = self.checked_address(offset, len)?;
        let resources = self.resources.as_deref_mut().expect("live allocation");
        let mut remaining = len;
        while remaining != 0 {
            let chunk = usize::try_from(remaining.min(resources.staging.len() as u64))
                .expect("bounded chunk");
            reader
                .read_exact(&mut resources.staging[..chunk])
                .map_err(CudaBf16Error::Source)?;
            // SAFETY: validated live range; boxed staging is owner-held and not
            // accessed/reused until successful drain, even if enqueue fails.
            unsafe { resources.transport.upload(dst, &resources.staging[..chunk]) }?;
            resources.transport.drain()?;
            dst += chunk as u64;
            remaining -= chunk as u64;
        }
        // A bounded overflow probe only after the final chunk has drained. No
        // separate tail allocation; Interrupted does not report long/short input.
        loop {
            match reader.read(&mut resources.staging[..1]) {
                Ok(0) => return Ok(()),
                Ok(_) => return Err(CudaBf16Error::SourceTooLong),
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(CudaBf16Error::Source(e)),
            }
        }
    }

    // Let logical owners reject invalid reads without poisoning publication.
    pub(crate) fn check_read(&self, offset: u64, len: usize) -> Result<()> {
        self.checked_address(offset, len as u64)?;
        if len > self.staging_bytes() {
            return Err(CudaBf16Error::Invalid("read exceeds staging capacity"));
        }
        Ok(())
    }

    // Caller ensures the requested bytes are initialized. This owner provides
    // range and DMA lifetime safety, not a logical initialization/publication state.
    pub(crate) fn read_chunk(&mut self, offset: u64, len: usize) -> Result<Vec<u8>> {
        self.check_read(offset, len)?;
        if len == 0 {
            return Ok(Vec::new());
        }
        let src = self.checked_address(offset, len as u64)?;
        let resources = self.resources.as_deref_mut().expect("live allocation");
        // SAFETY: in-bounds live range. Destination is owner-held, never a caller's
        // Vec which could drop on failed drain; caller guarantees initialization.
        unsafe { resources.transport.read(src, &mut resources.staging[..len]) }?;
        resources.transport.drain()?;
        Ok(resources.staging[..len].to_vec())
    }

    pub(crate) fn close(&mut self) -> Result<()> {
        let Some(resources) = self.resources.as_deref_mut() else {
            return Ok(());
        };
        if let Some(ptr) = resources.pointer {
            if resources.free_uncertain {
                return Err(CudaBf16Error::FreeUncertain);
            }
            resources.transport.drain()?;
            // Failed free can have released/reused the pointer. Mark BEFORE the
            // first and only invocation, including possible wrapper unwinding.
            resources.free_uncertain = true;
            // SAFETY: all uses are on the drained stream; allocation is still owned,
            // and this is its first and only free invocation.
            unsafe { resources.transport.free(ptr) }?;
            resources.pointer = None;
        }
        // Sole refund point: acknowledged release, or clean malloc refusal before
        // a pointer existed. A failed drain retains the whole retry envelope.
        drop(self.resources.take().map(ManuallyDrop::into_inner));
        Ok(())
    }
}

impl<T: Transport> Drop for Allocation<T> {
    fn drop(&mut self) {
        if self.close().is_err() {
            // Safety backstop, not the primary cleanup protocol. Retain allocation,
            // charge, DMA staging AND stream/context together. ManuallyDrop also
            // retains the whole envelope if close unwinds during defensive Drop.
            let _retained = self.resources.take();
        }
    }
}
