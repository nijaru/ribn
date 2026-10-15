//! Ownership of accepted submissions across terminal device failures.
use engine_core::BackendSubmissionId;
use std::collections::HashMap;

pub(crate) struct Submissions<T> {
    pub(crate) pending: HashMap<BackendSubmissionId, T>,
    quarantined: Vec<T>,
    fault: Option<String>,
    undrained: bool,
}

impl<T> Default for Submissions<T> {
    fn default() -> Self {
        Self {
            pending: HashMap::new(),
            quarantined: Vec::new(),
            fault: None,
            undrained: false,
        }
    }
}

impl<T> Submissions<T> {
    pub(crate) fn fault(&self) -> Option<&str> {
        self.fault.as_deref()
    }
    pub(crate) fn release_is_safe(&self) -> bool {
        !self.undrained
    }
    pub(crate) fn quarantine(&mut self, value: T) {
        self.quarantined.push(value);
    }

    /// Stop polling every affected submission. A successful stream drain is
    /// the only evidence permitting retirement; otherwise ownership stays here.
    pub(crate) fn fail(&mut self, error: String, drained: bool) -> Vec<T> {
        self.fault = Some(error);
        self.undrained = !drained;
        self.quarantined
            .extend(self.pending.drain().map(|(_, value)| value));
        if drained {
            self.retire_after_barrier()
        } else {
            Vec::new()
        }
    }

    /// The stream owner must establish a successful barrier before calling.
    /// Normal pending submissions retain their outcomes; only fault quarantine
    /// retires here. Cleanup never reopens a faulted dispatcher.
    pub(crate) fn retire_after_barrier(&mut self) -> Vec<T> {
        self.undrained = false;
        std::mem::take(&mut self.quarantined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    struct Resource(Arc<AtomicUsize>);
    impl Drop for Resource {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn failed_drain_retains_all_resources_until_teardown() {
        let dropped = Arc::new(AtomicUsize::new(0));
        let mut submissions = Submissions::default();
        submissions.pending.insert(
            BackendSubmissionId::new(1).unwrap(),
            Resource(dropped.clone()),
        );
        submissions.quarantine(Resource(dropped.clone()));
        assert!(
            submissions
                .fail("event query failed".into(), false)
                .is_empty()
        );
        assert!(submissions.pending.is_empty());
        assert!(submissions.fault().is_some());
        assert!(!submissions.release_is_safe());
        assert_eq!(dropped.load(Ordering::SeqCst), 0);
        drop(submissions);
        assert_eq!(dropped.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn later_barrier_retires_failed_drain_quarantine_exactly_once() {
        let dropped = Arc::new(AtomicUsize::new(0));
        let mut submissions = Submissions::default();
        submissions.pending.insert(
            BackendSubmissionId::new(1).unwrap(),
            Resource(dropped.clone()),
        );
        submissions.quarantine(Resource(dropped.clone()));
        assert!(
            submissions
                .fail("event query failed".into(), false)
                .is_empty()
        );
        assert!(submissions.fail("retry failed".into(), false).is_empty());
        assert_eq!(dropped.load(Ordering::SeqCst), 0);
        assert!(!submissions.release_is_safe());
        drop(submissions.retire_after_barrier());
        assert_eq!(dropped.load(Ordering::SeqCst), 2);
        assert!(submissions.release_is_safe());
        assert_eq!(submissions.fault(), Some("retry failed"));
        assert!(submissions.retire_after_barrier().is_empty());
        drop(submissions);
        assert_eq!(dropped.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn shutdown_barrier_preserves_healthy_pending_outcomes() {
        let mut submissions = Submissions::default();
        let id = BackendSubmissionId::new(1).unwrap();
        submissions.pending.insert(id, 7);
        assert!(submissions.retire_after_barrier().is_empty());
        assert_eq!(submissions.pending.remove(&id), Some(7));
        assert!(submissions.fault().is_none());
    }

    #[test]
    fn successful_drain_retires_resources_but_keeps_dispatcher_poisoned() {
        let mut submissions = Submissions::default();
        submissions
            .pending
            .insert(BackendSubmissionId::new(1).unwrap(), 7);
        submissions.quarantine(8);
        let mut retired = submissions.fail("pinned read failed".into(), true);
        retired.sort_unstable();
        assert_eq!(retired, vec![7, 8]);
        assert!(submissions.pending.is_empty());
        assert!(submissions.release_is_safe());
        assert!(submissions.fault().is_some());
    }
}
