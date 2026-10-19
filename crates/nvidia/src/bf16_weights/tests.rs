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
    base: Option<u64>,
    allocations: usize,
    frees: usize,
    drains: usize,
    copies: Vec<usize>,
    pending: Option<Pending>,
    staging: Option<usize>,
    panic_drain: bool,
    panic_free: bool,
    fail_allocate: bool,
    fail_copy: bool,
    fail_drain: bool,
    fail_free: bool,
}

struct Fake(Arc<Mutex<Probe>>);

fn failure(operation: &'static str) -> CudaBf16Error {
    CudaBf16Error::Transport {
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
        if let Some(staging) = p.staging {
            assert_eq!(host, staging, "owner must retain the same DMA staging");
        } else {
            p.staging = Some(host);
        }
        let offset = usize::try_from(ptr - p.base.unwrap_or(BASE)).unwrap();
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
        Ok(p.base.unwrap_or(BASE))
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
        if p.panic_drain {
            drop(p); // A wrapper unwind must not poison the probe's mutex.
            panic!("injected drain unwind");
        }
        if p.fail_drain {
            return Err(failure("drain"));
        }
        complete(&mut p, usize::MAX);
        Ok(())
    }
    unsafe fn free(&mut self, ptr: u64) -> Result<()> {
        let mut p = self.0.lock().unwrap();
        assert_eq!(ptr, p.base.unwrap_or(BASE));
        assert!(p.pending.is_none());
        p.frees += 1;
        // An uncertain free may actually have succeeded. A retry must never occur.
        p.bytes.clear();
        if p.panic_free {
            drop(p);
            panic!("injected free unwind after physical release");
        }
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
        Err(CudaBf16Error::NotReady)
    ));
    let a = [0, 0, 0, 128, 128, 63, 129, 127, 255, 255];
    owner.upload(0, &mut a.as_slice()).unwrap();
    assert!(!owner.ready());
    assert!(matches!(
        owner.upload(0, &mut [].as_slice()),
        Err(CudaBf16Error::AlreadyUploaded(0))
    ));
    assert!(matches!(
        owner.upload(2, &mut [].as_slice()),
        Err(CudaBf16Error::InvalidSlot(2))
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
        Err(CudaBf16Error::Closed)
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
            Err(CudaBf16Error::Poisoned)
        ));
        assert_eq!(pool.granted(), 32);
        owner.close().unwrap();
        assert_eq!(pool.granted(), 0);
        assert_eq!(probe.lock().unwrap().frees, 1);
    }
    let (mut owner, _, _) = setup(vec![spec("a", 1)], 3);
    assert!(matches!(
        owner.upload(0, &mut Broken),
        Err(CudaBf16Error::Source(_))
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
            let host = probe.lock().unwrap().staging;
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
                Err(CudaBf16Error::Poisoned)
            ));
            assert!(matches!(
                owner.upload(0, &mut [0; 4].as_slice()),
                Err(CudaBf16Error::Poisoned)
            ));
            assert!(owner.close().is_err());
            assert_eq!(pool.granted(), 16);
            if let Some(host) = host {
                assert_eq!(probe.lock().unwrap().staging, Some(host));
            }
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
        let host = probe.lock().unwrap().staging.unwrap();
        if !fail_free {
            // Leave a D2H DMA pending across BOTH explicit failed cleanup and Drop.
            assert!(owner.read_chunk(0, 0, 3).is_err());
            assert!(probe.lock().unwrap().pending.is_some());
        }
        assert!(owner.close().is_err());
        assert_eq!(probe.lock().unwrap().staging, Some(host));
        assert_eq!(pool.granted(), 16);
        assert_eq!(Arc::strong_count(&probe), 2);
        if fail_free {
            assert!(matches!(owner.close(), Err(CudaBf16Error::FreeUncertain)));
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
fn shared_access_requires_publication_and_excludes_mutation_without_poisoning() {
    let (mut owner, pool, probe) = setup(vec![spec("a", 2), spec("b", 1)], 3);
    assert!(matches!(owner.access(0), Err(CudaBf16Error::NotReady)));
    owner.upload(0, &mut [1, 2, 3, 4].as_slice()).unwrap();
    assert!(matches!(owner.access(0), Err(CudaBf16Error::NotReady)));
    owner.upload(1, &mut [5, 6].as_slice()).unwrap();
    assert!(matches!(
        owner.access(2),
        Err(CudaBf16Error::InvalidSlot(2))
    ));
    let (access, range) = owner.access(0).unwrap();
    assert_eq!(range, 0..4);
    assert!(matches!(owner.access(0), Err(CudaBf16Error::InUse)));
    assert!(matches!(owner.access(1), Err(CudaBf16Error::InUse)));
    assert_eq!(access.allocation_id(), owner.allocation_id());
    assert_eq!(access.allocated_bytes(), 32);
    assert_eq!(access.staging_bytes(), 3);
    assert_eq!(access.pointer().unwrap(), BASE);
    assert!(Arc::ptr_eq(&access.transport().0, &probe));
    let before = {
        let p = probe.lock().unwrap();
        (p.copies.len(), p.drains, p.frees)
    };
    assert!(matches!(owner.close(), Err(CudaBf16Error::InUse)));
    assert!(matches!(
        owner.read_chunk(0, 0, 1),
        Err(CudaBf16Error::InUse)
    ));
    assert!(matches!(
        owner.read_chunk(0, 0, 0),
        Err(CudaBf16Error::InUse)
    ));
    assert!(matches!(
        owner.upload(0, &mut [].as_slice()),
        Err(CudaBf16Error::InUse)
    ));
    assert!(owner.ready());
    assert_eq!(pool.granted(), 32);
    {
        let p = probe.lock().unwrap();
        assert_eq!((p.copies.len(), p.drains, p.frees), before);
    }
    drop(access);
    let (access, range) = owner.access(1).unwrap();
    assert_eq!(range, 16..18);
    drop(access);
    assert_eq!(owner.read_chunk(0, 1, 3).unwrap(), [2, 3, 4]);
    owner.close().unwrap();
    assert!(matches!(owner.access(0), Err(CudaBf16Error::Closed)));
    assert_eq!(owner.allocated_bytes(), 32);
    assert_eq!(owner.tensor_range(1), Some(16..18));
    assert_eq!(pool.granted(), 0);
}

#[test]
fn last_access_retires_after_root_drop_without_leaking_the_root_reference() {
    let (mut owner, pool, probe) = setup(vec![spec("a", 2)], 3);
    owner.upload(0, &mut [1, 2, 3, 4].as_slice()).unwrap();
    let (access, _) = owner.access(0).unwrap();
    let other = Arc::clone(&access);
    let wait = pool.capacity_wait();
    let drains = probe.lock().unwrap().drains;
    drop(owner);
    assert_eq!(
        Arc::strong_count(&access),
        2,
        "root must relinquish its Arc"
    );
    assert_eq!(pool.granted(), 16);
    assert!(!wait.changed());
    assert_eq!(probe.lock().unwrap().frees, 0);
    drop(access);
    assert_eq!(pool.granted(), 16);
    assert_eq!(probe.lock().unwrap().drains, drains);
    drop(other);
    assert!(wait.changed());
    assert_eq!(pool.granted(), 0);
    let p = probe.lock().unwrap();
    assert_eq!(p.drains, drains + 1);
    assert_eq!(p.frees, 1);
    assert_eq!(Arc::strong_count(&probe), 1);
}

#[test]
fn last_access_drop_quarantines_failed_retirement() {
    for fail_free in [false, true] {
        let (mut owner, pool, probe) = setup(vec![spec("a", 2)], 3);
        owner.upload(0, &mut [1, 2, 3, 4].as_slice()).unwrap();
        let (access, _) = owner.access(0).unwrap();
        drop(owner);
        {
            let mut p = probe.lock().unwrap();
            p.fail_free = fail_free;
            p.fail_drain = !fail_free;
        }
        drop(access);
        assert_eq!(pool.granted(), 16);
        assert_eq!(probe.lock().unwrap().frees, usize::from(fail_free));
        assert_eq!(Arc::strong_count(&probe), 2, "quarantine retains transport");
    }
}

#[test]
fn reader_and_cleanup_unwinds_preserve_retirement_ownership() {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    struct Panics;
    impl Read for Panics {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            panic!("injected source unwind");
        }
    }
    let (mut owner, pool, _) = setup(vec![spec("a", 2)], 3);
    assert!(catch_unwind(AssertUnwindSafe(|| owner.upload(0, &mut Panics))).is_err());
    assert!(matches!(owner.access(0), Err(CudaBf16Error::Poisoned)));
    owner.close().unwrap();
    assert_eq!(pool.granted(), 0);

    let (mut owner, pool, probe) = setup(vec![spec("a", 2)], 3);
    probe.lock().unwrap().panic_free = true;
    assert!(catch_unwind(AssertUnwindSafe(|| owner.close())).is_err());
    assert!(matches!(owner.close(), Err(CudaBf16Error::FreeUncertain)));
    drop(owner);
    assert_eq!(
        probe.lock().unwrap().frees,
        1,
        "unwound free is never retried"
    );
    assert_eq!(pool.granted(), 16);
    assert_eq!(Arc::strong_count(&probe), 2);

    let (mut owner, pool, probe) = setup(vec![spec("a", 2)], 3);
    owner.upload(0, &mut [1, 2, 3, 4].as_slice()).unwrap();
    probe.lock().unwrap().fail_drain = true;
    assert!(owner.read_chunk(0, 0, 3).is_err());
    probe.lock().unwrap().panic_drain = true;
    assert!(catch_unwind(AssertUnwindSafe(|| drop(owner))).is_err());
    assert_eq!(pool.granted(), 16);
    assert_eq!(Arc::strong_count(&probe), 2);
    let mut p = probe.lock().unwrap();
    assert_eq!(p.frees, 0);
    // Actual delayed DMA still has a live destination after defensive Drop unwinds.
    complete(&mut p, usize::MAX);
    assert!(p.pending.is_none());
}

#[test]
fn shared_allocation_checks_byte_spans_before_reader_or_transport_actions() {
    struct Untouched;
    impl Read for Untouched {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            panic!("invalid range reached source");
        }
    }
    let pool = BytePool::new(16).shared();
    let probe = Arc::new(Mutex::new(Probe::default()));
    for (bytes, staging) in [(0, 3), (16, 0)] {
        assert!(Allocation::allocate(bytes, Fake(probe.clone()), &pool, staging).is_err());
    }
    assert_eq!(probe.lock().unwrap().allocations, 0);
    let mut shared = Allocation::allocate(16, Fake(probe.clone()), &pool, 3).unwrap();
    let owner = Arc::get_mut(&mut shared).unwrap();
    for (offset, len) in [(17, 0), (15, 2), (u64::MAX, 1), (1, u64::MAX)] {
        assert!(owner.write_exact(offset, len, &mut Untouched).is_err());
    }
    for (offset, len) in [(0, 4), (15, 2), (u64::MAX, 1)] {
        assert!(owner.read_chunk(offset, len).is_err());
    }
    assert!(probe.lock().unwrap().copies.is_empty());
    // Neither the span nor staging chunks need BF16 element alignment.
    owner
        .write_exact(1, 5, &mut [1, 2, 3, 4, 5].as_slice())
        .unwrap();
    assert_eq!(owner.read_chunk(2, 3).unwrap(), [2, 3, 4]);
    assert_eq!(probe.lock().unwrap().copies, [3, 2, 3]);
    owner.close().unwrap();
    assert!(matches!(owner.pointer(), Err(CudaBf16Error::Closed)));
    assert_eq!(owner.staging_bytes(), 3);
    assert_eq!(pool.granted(), 0);

    let probe = Arc::new(Mutex::new(Probe {
        base: Some(u64::MAX - 8),
        ..Probe::default()
    }));
    let mut shared = Allocation::allocate(16, Fake(probe.clone()), &pool, 3).unwrap();
    let owner = Arc::get_mut(&mut shared).unwrap();
    // The first chunk would fit; the entire span must validate BEFORE reading it.
    assert!(owner.write_exact(0, 9, &mut Untouched).is_err());
    assert!(owner.read_chunk(7, 2).is_err());
    let p = probe.lock().unwrap();
    assert!(p.copies.is_empty());
    assert_eq!(p.drains, 0);
    drop(p);
    owner.close().unwrap();
}

#[test]
fn ordinary_drop_drains_and_refunds_without_publication() {
    let (owner, pool, probe) = setup(vec![spec("a", 2)], 3);
    drop(owner);
    assert_eq!(pool.granted(), 0);
    assert_eq!(probe.lock().unwrap().frees, 1);
    assert_eq!(Arc::strong_count(&probe), 1);
}
