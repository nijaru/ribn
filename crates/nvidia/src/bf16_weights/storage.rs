use std::{
    collections::HashSet, error::Error, fmt, io::Read, mem::ManuallyDrop, ops::Range, sync::Arc,
};

use engine_core::{DataType, WeightTensorSpec};
use ribn_foundation::{AllocationId, BytePool, PoolLease, ReserveError};

#[derive(Debug)]
pub enum CudaBf16WeightsError {
    Invalid(&'static str),
    InvalidSlot(usize),
    AlreadyUploaded(usize),
    NotReady,
    Poisoned,
    Closed,
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

impl fmt::Display for CudaBf16WeightsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(reason) => write!(f, "invalid BF16 storage: {reason}"),
            Self::InvalidSlot(slot) => write!(f, "invalid BF16 slot {slot}"),
            Self::AlreadyUploaded(slot) => write!(f, "BF16 slot {slot} already uploaded"),
            Self::NotReady => f.write_str("BF16 set is not fully uploaded"),
            Self::Poisoned => f.write_str("BF16 set is poisoned; close it"),
            Self::Closed => f.write_str("BF16 set is closed"),
            Self::FreeUncertain => f.write_str("BF16 free was uncertain; permanently quarantined"),
            Self::Source(e) => write!(f, "BF16 source: {e}"),
            Self::SourceTooLong => f.write_str("BF16 source exceeds declared tensor bytes"),
            Self::Reserve(e) => write!(f, "BF16 reservation: {e}"),
            Self::Staging(e) => write!(f, "BF16 staging: {e}"),
            Self::Transport { operation, source } => write!(f, "BF16 {operation}: {source}"),
        }
    }
}

impl Error for CudaBf16WeightsError {
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

type Result<T> = std::result::Result<T, CudaBf16WeightsError>;

// Private FFI boundary, not a pluggable storage/allocator API. Successful drain
// proves all accesses on this transport's one stream have completed. Failed copy
// calls may have enqueued work; failed free calls may have released the pointer.
pub(super) trait Transport {
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

struct Slot {
    spec: WeightTensorSpec,
    range: Range<u64>,
    uploaded: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Loading,
    Ready,
    Poisoned,
    Closed,
}

struct Resources<T> {
    transport: T,
    pointer: Option<u64>,
    _lease: PoolLease,
    staging: Box<[u8]>,
    free_uncertain: bool,
}

pub(super) struct Storage<T: Transport> {
    // No implicit refund/staging destruction if an FFI wrapper panics during
    // defensive cleanup. Only acknowledged retirement unwraps this envelope.
    resources: Option<ManuallyDrop<Resources<T>>>,
    slots: Vec<Slot>,
    bytes: u64,
    allocation: AllocationId,
    state: State,
}

fn layout(specs: Vec<WeightTensorSpec>, staging_bytes: usize) -> Result<(Vec<Slot>, u64)> {
    if specs.is_empty() || staging_bytes == 0 {
        return Err(CudaBf16WeightsError::Invalid(
            "empty set or zero staging capacity",
        ));
    }
    let mut names = HashSet::with_capacity(specs.len());
    let mut offset = 0_u64;
    let mut slots = Vec::with_capacity(specs.len());
    for spec in specs {
        if spec.dtype() != DataType::BF16 {
            return Err(CudaBf16WeightsError::Invalid(
                "only native BF16 is accepted",
            ));
        }
        if !names.insert(spec.name().to_owned()) {
            return Err(CudaBf16WeightsError::Invalid("duplicate tensor name"));
        }
        let bytes = spec
            .byte_size()
            .ok_or(CudaBf16WeightsError::Invalid("tensor byte overflow"))?;
        let end = offset
            .checked_add(bytes)
            .ok_or(CudaBf16WeightsError::Invalid("byte sum overflow"))?;
        // Include trailing padding on EVERY tensor (including the last). Logical
        // ranges exclude padding, which is never read or published as tensor data.
        let padded_end = end
            .checked_add(15)
            .ok_or(CudaBf16WeightsError::Invalid("alignment overflow"))?
            & !15;
        usize::try_from(padded_end)
            .map_err(|_| CudaBf16WeightsError::Invalid("allocation exceeds usize"))?;
        slots.push(Slot {
            spec,
            range: offset..end,
            uploaded: false,
        });
        offset = padded_end;
    }
    Ok((slots, offset))
}

impl<T: Transport> Storage<T> {
    pub(super) fn allocate(
        specs: Vec<WeightTensorSpec>,
        transport: T,
        pool: &Arc<BytePool>,
        staging_bytes: usize,
    ) -> Result<Self> {
        let (slots, bytes) = layout(specs, staging_bytes)?;
        let mut staging = Vec::new();
        staging
            .try_reserve_exact(staging_bytes)
            .map_err(CudaBf16WeightsError::Staging)?;
        staging.resize(staging_bytes, 0);
        let lease = pool.reserve(bytes).map_err(CudaBf16WeightsError::Reserve)?;
        let allocation = lease.allocation();
        let mut owner = Self {
            resources: Some(ManuallyDrop::new(Resources {
                transport,
                pointer: None,
                _lease: lease,
                staging: staging.into_boxed_slice(),
                free_uncertain: false,
            })),
            slots,
            bytes,
            allocation,
            state: State::Loading,
        };
        let resources = owner
            .resources
            .as_deref_mut()
            .expect("new owner has resources");
        // layout already proved usize conversion. Capture immediately on success;
        // absolutely no fallible event creation/wrapping after the raw allocation.
        resources.pointer = Some(
            resources
                .transport
                .allocate(usize::try_from(bytes).expect("validated usize"))?,
        );
        Ok(owner)
    }

    pub(super) fn allocated_bytes(&self) -> u64 {
        self.bytes
    }
    pub(super) fn allocation_id(&self) -> AllocationId {
        self.allocation
    }
    pub(super) fn spec(&self, slot: usize) -> Option<&WeightTensorSpec> {
        self.slots.get(slot).map(|s| &s.spec)
    }
    pub(super) fn tensor_range(&self, slot: usize) -> Option<Range<u64>> {
        self.slots.get(slot).map(|s| s.range.clone())
    }
    pub(super) fn ready(&self) -> bool {
        self.state == State::Ready
    }

    fn check_open(&self) -> Result<()> {
        match self.state {
            State::Closed => Err(CudaBf16WeightsError::Closed),
            State::Poisoned => Err(CudaBf16WeightsError::Poisoned),
            State::Loading | State::Ready => Ok(()),
        }
    }

    pub(super) fn upload(&mut self, slot: usize, reader: &mut impl Read) -> Result<()> {
        self.check_open()?;
        let tensor = self
            .slots
            .get(slot)
            .ok_or(CudaBf16WeightsError::InvalidSlot(slot))?;
        if tensor.uploaded {
            return Err(CudaBf16WeightsError::AlreadyUploaded(slot));
        }
        let range = tensor.range.clone();
        // Poison BEFORE any user reader/FFI call: errors or unwinding cannot publish
        // a partial set. Only a complete successful upload restores loading/ready.
        self.state = State::Poisoned;
        let resources = self
            .resources
            .as_deref_mut()
            .expect("open owner has resources");
        let base = resources.pointer.expect("open owner has allocation");
        let mut offset = range.start;
        while offset < range.end {
            let len = usize::try_from((range.end - offset).min(resources.staging.len() as u64))
                .expect("bounded chunk");
            reader
                .read_exact(&mut resources.staging[..len])
                .map_err(CudaBf16WeightsError::Source)?;
            let dst = base
                .checked_add(offset)
                .ok_or(CudaBf16WeightsError::Invalid("device address overflow"))?;
            // SAFETY: validated live range; boxed staging is owner-held and not
            // accessed/reused until successful drain, even if enqueue fails.
            unsafe { resources.transport.upload(dst, &resources.staging[..len]) }?;
            resources.transport.drain()?;
            offset += len as u64;
        }
        // A bounded overflow probe after the final chunk has drained. No separate
        // tail allocation; Interrupted does not falsely report long/short input.
        loop {
            match reader.read(&mut resources.staging[..1]) {
                Ok(0) => break,
                Ok(_) => return Err(CudaBf16WeightsError::SourceTooLong),
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(CudaBf16WeightsError::Source(e)),
            }
        }
        self.slots[slot].uploaded = true;
        self.state = if self.slots.iter().all(|s| s.uploaded) {
            State::Ready
        } else {
            State::Loading
        };
        Ok(())
    }

    pub(super) fn read_chunk(
        &mut self,
        slot: usize,
        byte_offset: u64,
        len: usize,
    ) -> Result<Vec<u8>> {
        self.check_open()?;
        if !self.ready() {
            return Err(CudaBf16WeightsError::NotReady);
        }
        let tensor = self
            .slots
            .get(slot)
            .ok_or(CudaBf16WeightsError::InvalidSlot(slot))?;
        let resources = self
            .resources
            .as_deref_mut()
            .expect("open owner has resources");
        let end = byte_offset
            .checked_add(len as u64)
            .ok_or(CudaBf16WeightsError::Invalid("read offset overflow"))?;
        if len > resources.staging.len() || end > tensor.range.end - tensor.range.start {
            return Err(CudaBf16WeightsError::Invalid(
                "read exceeds tensor or staging capacity",
            ));
        }
        if len == 0 {
            return Ok(Vec::new());
        }
        let src = resources
            .pointer
            .expect("open owner has allocation")
            .checked_add(tensor.range.start)
            .and_then(|p| p.checked_add(byte_offset))
            .ok_or(CudaBf16WeightsError::Invalid("device address overflow"))?;
        self.state = State::Poisoned;
        // SAFETY: only initialized tensor bytes, within the allocation. Destination
        // is owner-held, never a caller's Vec which could drop on failed drain.
        unsafe { resources.transport.read(src, &mut resources.staging[..len]) }?;
        resources.transport.drain()?;
        self.state = State::Ready;
        Ok(resources.staging[..len].to_vec())
    }

    pub(super) fn close(&mut self) -> Result<()> {
        let Some(resources) = self.resources.as_deref_mut() else {
            return Ok(());
        };
        self.state = State::Poisoned;
        if let Some(ptr) = resources.pointer {
            if resources.free_uncertain {
                return Err(CudaBf16WeightsError::FreeUncertain);
            }
            resources.transport.drain()?;
            // Neither cudarc's raw wrapper nor the locally available driver contract
            // proves failed free leaves ownership unchanged (prior async errors may
            // surface here). Never blindly retry a potentially released/reused ptr.
            // Mark before invocation, including possible unwinding.
            resources.free_uncertain = true;
            // SAFETY: all uses are on the drained stream; allocation is still owned,
            // and this is its first and only free invocation.
            unsafe { resources.transport.free(ptr) }?;
            resources.pointer = None;
        }
        // The sole refund point after a successful physical release (or clean
        // malloc refusal, before any pointer existed).
        drop(self.resources.take().map(ManuallyDrop::into_inner));
        self.state = State::Closed;
        Ok(())
    }
}

impl<T: Transport> Drop for Storage<T> {
    fn drop(&mut self) {
        if self.close().is_err() {
            // Safety backstop, not the primary cleanup protocol. Retain the raw
            // allocation, charge, host DMA buffer AND stream/context together.
            // Leaking only the pointer/lease would free active host memory or allow
            // destruction of the CUDA context underneath outstanding accesses.
            // Dropping ManuallyDrop deliberately retains all of its contents.
            let _retained = self.resources.take();
        }
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
