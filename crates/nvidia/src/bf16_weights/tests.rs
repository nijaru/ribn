use super::*;
use std::{io, sync::Mutex};

const BASE: u64 = 0x1000;

#[derive(Clone, Copy)]
enum Direction {
    Upload,
    Read,
}

struct Pending {
    host: usize,
    offset: usize,
    len: usize,
    direction: Direction,
}

#[derive(Default)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "independent, combinable FFI failure injections"
)]
struct Probe {
    bytes: Vec<u8>,
    allocations: usize,
    frees: usize,
    drains: usize,
    copies: Vec<usize>,
    pending: Option<Pending>,
    fail_allocate: bool,
    fail_copy: bool,
    fail_drain: bool,
    fail_free: bool,
}

struct Fake(Arc<Mutex<Probe>>);

fn failure(operation: &'static str) -> CudaBf16WeightsError {
    CudaBf16WeightsError::Transport {
        operation,
        source: Box::new(io::Error::other("injected")),
    }
}

impl Fake {
    fn enqueue(&self, host: usize, ptr: u64, len: usize, direction: Direction) -> Result<()> {
        let mut p = self.0.lock().unwrap();
        assert!(
            p.pending.is_none(),
            "staging reused without a successful drain"
        );
        let offset = usize::try_from(ptr - BASE).unwrap();
        assert!(offset + len <= p.bytes.len());
        p.copies.push(len);
        p.pending = Some(Pending {
            host,
            offset,
            len,
            direction,
        });
        if p.fail_copy {
            // Simulate a partial enqueue which writes some bytes before reporting
            // failure, with the remaining DMA still referencing the host buffer.
            complete(&mut p, 1);
            return Err(failure("copy"));
        }
        Ok(())
    }
}

fn complete(p: &mut Probe, limit: usize) {
    let Some(mut pending) = p.pending.take() else {
        return;
    };
    let len = pending.len.min(limit);
    // SAFETY: the tested storage protocol must retain its staging box unchanged
    // until a successful drain. Fake models delayed driver DMA, not a host clone.
    unsafe {
        match pending.direction {
            Direction::Upload => std::ptr::copy_nonoverlapping(
                pending.host as *const u8,
                p.bytes.as_mut_ptr().add(pending.offset),
                len,
            ),
            Direction::Read => std::ptr::copy_nonoverlapping(
                p.bytes.as_ptr().add(pending.offset),
                pending.host as *mut u8,
                len,
            ),
        }
    }
    pending.host += len;
    pending.offset += len;
    pending.len -= len;
    if pending.len != 0 {
        p.pending = Some(pending);
    }
}

impl Transport for Fake {
    fn allocate(&mut self, bytes: usize) -> Result<u64> {
        let mut p = self.0.lock().unwrap();
        p.allocations += 1;
        if p.fail_allocate {
            return Err(failure("allocate"));
        }
        p.bytes = vec![0xcd; bytes];
        Ok(BASE)
    }
    unsafe fn upload(&mut self, dst: u64, src: &[u8]) -> Result<()> {
        self.enqueue(src.as_ptr() as usize, dst, src.len(), Direction::Upload)
    }
    unsafe fn read(&mut self, src: u64, dst: &mut [u8]) -> Result<()> {
        self.enqueue(dst.as_mut_ptr() as usize, src, dst.len(), Direction::Read)
    }
    fn drain(&mut self) -> Result<()> {
        let mut p = self.0.lock().unwrap();
        p.drains += 1;
        if p.fail_drain {
            return Err(failure("drain"));
        }
        complete(&mut p, usize::MAX);
        Ok(())
    }
    unsafe fn free(&mut self, ptr: u64) -> Result<()> {
        let mut p = self.0.lock().unwrap();
        assert_eq!(ptr, BASE);
        assert!(p.pending.is_none());
        p.frees += 1;
        // An uncertain free may actually have succeeded. A retry must never occur.
        p.bytes.clear();
        if p.fail_free {
            return Err(failure("free"));
        }
        Ok(())
    }
}

fn spec(name: &str, elements: u64) -> WeightTensorSpec {
    WeightTensorSpec::new(name, vec![elements], DataType::BF16).unwrap()
}

fn setup(
    specs: Vec<WeightTensorSpec>,
    staging: usize,
) -> (Storage<Fake>, Arc<BytePool>, Arc<Mutex<Probe>>) {
    let pool = BytePool::new(128).shared();
    let probe = Arc::new(Mutex::new(Probe::default()));
    let owner = Storage::allocate(specs, Fake(probe.clone()), &pool, staging).unwrap();
    (owner, pool, probe)
}

#[test]
fn validates_the_entire_set_before_reservation_or_allocation() {
    let pool = BytePool::new(u64::MAX).shared();
    // Each spec itself is valid: these fail at BF16 bytes, aggregate sum,
    // alignment padding, or aggregate usize conversion rather than shape parsing.
    let inventories = vec![
        (vec![], 3),
        (vec![spec("a", 1)], 0),
        (vec![spec("a", 1), spec("a", 2)], 3),
        (
            vec![
                spec("a", 1),
                WeightTensorSpec::new("b", vec![1], DataType::F32).unwrap(),
            ],
            3,
        ),
        (vec![spec("a", u64::MAX)], 3),
        (vec![spec("a", u64::MAX / 2)], 3),
        (vec![spec("a", 1 << 62), spec("b", 1 << 62)], 3),
    ];
    let wait = pool.capacity_wait();
    for (specs, staging) in inventories {
        let probe = Arc::new(Mutex::new(Probe::default()));
        assert!(Storage::allocate(specs, Fake(probe.clone()), &pool, staging).is_err());
        assert_eq!(probe.lock().unwrap().allocations, 0);
        assert_eq!(pool.granted(), 0);
    }
    // No failed preflight acquired/refunded a lease.
    assert!(!wait.changed());
    #[cfg(target_pointer_width = "32")]
    assert!(layout(vec![spec("large", 1 << 32)], 3).is_err());
}

#[test]
fn host_capacity_and_pool_failures_never_reach_device_allocation() {
    let pool = BytePool::new(15).shared();
    for staging in [3, usize::MAX] {
        let probe = Arc::new(Mutex::new(Probe::default()));
        assert!(
            Storage::allocate(vec![spec("a", 1)], Fake(probe.clone()), &pool, staging).is_err()
        );
        assert_eq!(probe.lock().unwrap().allocations, 0);
        assert_eq!(pool.granted(), 0);
    }
    let pool = BytePool::new(16).shared();
    let probe = Arc::new(Mutex::new(Probe {
        fail_allocate: true,
        ..Probe::default()
    }));
    assert!(Storage::allocate(vec![spec("a", 1)], Fake(probe.clone()), &pool, 3).is_err());
    let p = probe.lock().unwrap();
    assert_eq!((p.allocations, p.frees, p.drains), (1, 0, 0));
    assert_eq!(pool.available(), 16);
}

#[test]
fn preserves_bytes_odd_chunks_alignment_and_exactly_once_refund() {
    let (mut owner, pool, probe) = setup(vec![spec("a", 5), spec("b", 1)], 3);
    assert_eq!(owner.allocated_bytes(), 32);
    assert_eq!(owner.tensor_range(0), Some(0..10));
    assert_eq!(owner.tensor_range(1), Some(16..18));
    assert_eq!(owner.spec(0).unwrap().dimensions(), &[5]);
    assert!(owner.spec(2).is_none());
    assert!(!owner.ready());
    assert!(matches!(
        owner.read_chunk(0, 0, 1),
        Err(CudaBf16WeightsError::NotReady)
    ));
    let a = [0, 0, 0, 128, 128, 63, 129, 127, 255, 255];
    owner.upload(0, &mut a.as_slice()).unwrap();
    assert!(!owner.ready());
    assert!(matches!(
        owner.upload(0, &mut [].as_slice()),
        Err(CudaBf16WeightsError::AlreadyUploaded(0))
    ));
    assert!(matches!(
        owner.upload(2, &mut [].as_slice()),
        Err(CudaBf16WeightsError::InvalidSlot(2))
    ));
    owner.upload(1, &mut [0x45, 0x23].as_slice()).unwrap();
    assert!(owner.ready());
    {
        let p = probe.lock().unwrap();
        assert_eq!(p.copies, [3, 3, 3, 1, 2]);
        assert_eq!(&p.bytes[..10], &a);
        assert_eq!(&p.bytes[10..16], &[0xcd; 6]);
        assert_eq!(&p.bytes[16..18], &[0x45, 0x23]);
    }
    for start in 0..a.len() {
        let len = 3.min(a.len() - start);
        assert_eq!(
            owner.read_chunk(0, start as u64, len).unwrap(),
            a[start..start + len]
        );
    }
    for (offset, len) in [(0, 4), (9, 2), (u64::MAX, 1)] {
        assert!(owner.read_chunk(0, offset, len).is_err());
        assert!(owner.ready(), "invalid read cannot corrupt publication");
    }
    assert!(owner.read_chunk(0, 10, 0).unwrap().is_empty());
    let id = owner.allocation_id();
    let wait = pool.capacity_wait();
    assert_eq!(pool.granted(), 32);
    owner.close().unwrap();
    assert!(wait.changed());
    let refunded = pool.capacity_wait();
    owner.close().unwrap();
    assert_eq!(owner.allocation_id(), id);
    assert!(!owner.ready());
    assert!(matches!(
        owner.read_chunk(0, 0, 1),
        Err(CudaBf16WeightsError::Closed)
    ));
    drop(owner);
    assert!(!refunded.changed(), "close/Drop must not refund twice");
    assert_eq!(pool.granted(), 0);
    assert_eq!(probe.lock().unwrap().frees, 1);
}

#[test]
fn short_long_and_io_error_sources_poison_the_complete_set() {
    struct Broken;
    impl Read for Broken {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::other("source"))
        }
    }
    for bytes in [vec![], vec![1], vec![1, 2, 3], vec![1, 2, 3, 4, 5]] {
        let (mut owner, pool, probe) = setup(vec![spec("a", 2), spec("b", 1)], 3);
        assert!(owner.upload(0, &mut bytes.as_slice()).is_err());
        assert!(!owner.ready());
        assert!(matches!(
            owner.upload(1, &mut [0, 0].as_slice()),
            Err(CudaBf16WeightsError::Poisoned)
        ));
        assert_eq!(pool.granted(), 32);
        owner.close().unwrap();
        assert_eq!(pool.granted(), 0);
        assert_eq!(probe.lock().unwrap().frees, 1);
    }
    let (mut owner, _, _) = setup(vec![spec("a", 1)], 3);
    assert!(matches!(
        owner.upload(0, &mut Broken),
        Err(CudaBf16WeightsError::Source(_))
    ));
    assert!(!owner.ready());
}

#[test]
fn partial_copy_and_failed_drain_retain_dma_buffer_until_retry() {
    for direction in [Direction::Upload, Direction::Read] {
        for fail_copy in [false, true] {
            let (mut owner, pool, probe) = setup(vec![spec("a", 2)], 3);
            if matches!(direction, Direction::Read) {
                owner.upload(0, &mut [1, 2, 3, 4].as_slice()).unwrap();
            }
            let host = owner.resources.as_ref().unwrap().staging.as_ptr();
            {
                let mut p = probe.lock().unwrap();
                p.fail_copy = fail_copy;
                p.fail_drain = true;
            }
            let result = match direction {
                Direction::Upload => owner.upload(0, &mut [1, 2, 3, 4].as_slice()),
                Direction::Read => owner.read_chunk(0, 0, 3).map(|_| ()),
            };
            assert!(result.is_err());
            assert!(!owner.ready());
            assert!(matches!(
                owner.read_chunk(0, 0, 1),
                Err(CudaBf16WeightsError::Poisoned)
            ));
            assert!(matches!(
                owner.upload(0, &mut [0; 4].as_slice()),
                Err(CudaBf16WeightsError::Poisoned)
            ));
            assert!(owner.close().is_err());
            assert_eq!(pool.granted(), 16);
            assert_eq!(owner.resources.as_ref().unwrap().staging.as_ptr(), host);
            assert!(probe.lock().unwrap().pending.is_some());
            assert_eq!(probe.lock().unwrap().frees, 0);
            probe.lock().unwrap().fail_drain = false;
            // This actually completes delayed raw-pointer access to retained staging.
            owner.close().unwrap();
            assert_eq!(pool.granted(), 0);
            assert_eq!(probe.lock().unwrap().frees, 1);
            assert!(probe.lock().unwrap().pending.is_none());
        }
    }
}

#[test]
fn uncertain_free_is_never_retried_and_drop_keeps_the_whole_envelope() {
    for fail_free in [false, true] {
        let (mut owner, pool, probe) = setup(vec![spec("a", 2)], 3);
        owner.upload(0, &mut [1, 2, 3, 4].as_slice()).unwrap();
        {
            let mut p = probe.lock().unwrap();
            p.fail_free = fail_free;
            p.fail_drain = !fail_free;
        }
        let host = owner.resources.as_ref().unwrap().staging.as_ptr();
        if !fail_free {
            // Leave a D2H DMA pending across BOTH explicit failed cleanup and Drop.
            assert!(owner.read_chunk(0, 0, 3).is_err());
            assert!(probe.lock().unwrap().pending.is_some());
        }
        assert!(owner.close().is_err());
        assert_eq!(owner.resources.as_ref().unwrap().staging.as_ptr(), host);
        assert_eq!(pool.granted(), 16);
        assert_eq!(Arc::strong_count(&probe), 2);
        if fail_free {
            assert!(matches!(
                owner.close(),
                Err(CudaBf16WeightsError::FreeUncertain)
            ));
            assert_eq!(probe.lock().unwrap().frees, 1);
        }
        drop(owner);
        assert_eq!(pool.granted(), 16);
        assert_eq!(
            Arc::strong_count(&probe),
            2,
            "transport/context must stay alive too"
        );
        assert_eq!(probe.lock().unwrap().frees, usize::from(fail_free));
        if !fail_free {
            let mut p = probe.lock().unwrap();
            // The owner is gone, but its host allocation must still be writable by
            // delayed DMA. The whole ManuallyDrop envelope deliberately survives.
            complete(&mut p, usize::MAX);
            assert!(p.pending.is_none());
        }
        // Intentional tiny safety-backstop leaks; production uses explicit retry.
    }
}

#[test]
fn ordinary_drop_drains_and_refunds_without_publication() {
    let (owner, pool, probe) = setup(vec![spec("a", 2)], 3);
    drop(owner);
    assert_eq!(pool.granted(), 0);
    assert_eq!(probe.lock().unwrap().frees, 1);
    assert_eq!(Arc::strong_count(&probe), 1);
}
