use std::mem::ManuallyDrop;

use crate::cuda_allocation::CudaBf16Error;

/// Retains every accessed resource until the concrete stream's successful drain.
/// The callback is the narrow completion boundary, also exercised by host faults.
/// `ManuallyDrop` protects the envelope even if a drain wrapper unwinds.
pub(super) struct Pending<T> {
    resources: Option<ManuallyDrop<T>>,
    drain: fn(&T) -> Result<(), CudaBf16Error>,
}
impl<T> Pending<T> {
    pub(super) fn new(resources: T, drain: fn(&T) -> Result<(), CudaBf16Error>) -> Self {
        Self {
            resources: Some(ManuallyDrop::new(resources)),
            drain,
        }
    }

    pub(super) fn retire(&mut self) -> Result<(), CudaBf16Error> {
        if let Some(resources) = self.resources.as_deref() {
            (self.drain)(resources)?;
            drop(self.resources.take().map(ManuallyDrop::into_inner));
        }
        Ok(())
    }
}
impl<T> Drop for Pending<T> {
    fn drop(&mut self) {
        if self.retire().is_err() {
            // Retain the whole access set, not just addresses or pool charges.
            let _quarantined = self.resources.take();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    };

    struct Access {
        owner: Arc<AtomicUsize>,
        complete: Arc<AtomicBool>,
        panic: bool,
    }
    impl Drop for Access {
        fn drop(&mut self) {
            self.owner.fetch_add(1, Ordering::SeqCst);
        }
    }
    fn drain(access: &Access) -> Result<(), CudaBf16Error> {
        assert!(!access.panic, "injected completion unwind");
        if access.complete.load(Ordering::SeqCst) {
            Ok(())
        } else {
            Err(CudaBf16Error::Invalid("injected incomplete stream"))
        }
    }

    #[test]
    fn failed_completion_retains_all_access_then_success_releases_once() {
        let drops = Arc::new(AtomicUsize::new(0));
        let complete = Arc::new(AtomicBool::new(false));
        let mut guard = Pending::new(
            Access {
                owner: Arc::clone(&drops),
                complete: Arc::clone(&complete),
                panic: false,
            },
            drain,
        );
        assert!(guard.retire().is_err());
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        complete.store(true, Ordering::SeqCst);
        guard.retire().unwrap();
        guard.retire().unwrap();
        drop(guard);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn failed_or_unwinding_defensive_drain_quarantines_the_access_envelope() {
        for panic in [false, true] {
            let drops = Arc::new(AtomicUsize::new(0));
            let owner = Arc::clone(&drops);
            let result = std::panic::catch_unwind(move || {
                drop(Pending::new(
                    Access {
                        owner,
                        complete: Arc::new(AtomicBool::new(false)),
                        panic,
                    },
                    drain,
                ));
            });
            assert_eq!(result.is_err(), panic);
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            assert_eq!(Arc::strong_count(&drops), 2, "the actual owner is retained");
        }
    }
}
