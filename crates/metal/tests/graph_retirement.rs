//! Public-SDK fault injection lives in this isolated test process, never production.
#![cfg(target_os = "macos")]

use engine_metal::{MetalBf16Error, MetalBf16Executor, MetalLinearShape};
use std::{
    ffi::{CString, c_void},
    os::unix::ffi::OsStrExt,
    process::Command,
    time::{Duration, Instant, SystemTime},
};

unsafe extern "C" {
    fn dlopen(path: *const std::ffi::c_char, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const std::ffi::c_char) -> *mut c_void;
}

struct Hook {
    arm: unsafe extern "C" fn(i32) -> *mut c_void,
    signal: unsafe extern "C" fn(*mut c_void),
    live: unsafe extern "C" fn(*mut c_void) -> usize,
    finished: unsafe extern "C" fn(*mut c_void) -> i32,
    output: unsafe extern "C" fn(*mut c_void, usize, *mut u16, usize) -> usize,
    destroy: unsafe extern "C" fn(*mut c_void),
}

impl Hook {
    fn load() -> Self {
        let stamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory =
            std::env::temp_dir().join(format!("ribn-graph-faults-{}-{stamp}", std::process::id()));
        std::fs::create_dir(&directory).unwrap();
        let source = directory.join("graph_faults.m");
        let library = directory.join("libgraph_faults.dylib");
        std::fs::write(&source, include_str!("fixtures/graph_faults.m")).unwrap();
        let result = Command::new("xcrun")
            .args([
                "--sdk",
                "macosx",
                "clang",
                "-dynamiclib",
                "-O3",
                "-fobjc-arc",
                "-fobjc-arc-exceptions",
                "-fblocks",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-mmacosx-version-min=26.0",
            ])
            .arg(&source)
            .args(["-framework", "Foundation", "-framework", "Metal", "-o"])
            .arg(&library)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "test shim compilation: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        // Preserve source/binary and print their identities BEFORE device launch.
        let hashes = Command::new("shasum")
            .args(["-a", "256"])
            .arg(&source)
            .arg(&library)
            .output()
            .unwrap();
        assert!(hashes.status.success());
        eprintln!("{}", String::from_utf8_lossy(&hashes.stdout));
        let path = CString::new(library.as_os_str().as_bytes()).unwrap();
        // SAFETY: trusted test-built dylib. Intentionally never dlclose: SDK
        // methods/callbacks retain its IMPs through process-scoped quarantine.
        let handle = unsafe { dlopen(path.as_ptr(), 2) };
        assert!(!handle.is_null(), "test shim loading");
        macro_rules! symbol {
            ($name:literal, $ty:ty) => {{
                // SAFETY: the embedded C fixture defines exactly this C ABI;
                // symbols remain loaded for the entire isolated test process.
                let pointer = unsafe { dlsym(handle, concat!($name, "\0").as_ptr().cast()) };
                assert!(!pointer.is_null(), "test symbol {}", $name);
                unsafe { std::mem::transmute::<*mut c_void, $ty>(pointer) }
            }};
        }
        Self {
            arm: symbol!("ribn_qual_arm", unsafe extern "C" fn(i32) -> *mut c_void),
            signal: symbol!("ribn_qual_signal", unsafe extern "C" fn(*mut c_void)),
            live: symbol!("ribn_qual_live", unsafe extern "C" fn(*mut c_void) -> usize),
            finished: symbol!(
                "ribn_qual_finished",
                unsafe extern "C" fn(*mut c_void) -> i32
            ),
            output: symbol!(
                "ribn_qual_output",
                unsafe extern "C" fn(*mut c_void, usize, *mut u16, usize) -> usize
            ),
            destroy: symbol!("ribn_qual_destroy", unsafe extern "C" fn(*mut c_void)),
        }
    }

    fn arm(&self, mode: i32) -> Control<'_> {
        // SAFETY: Graph classes prepared; modes 1..4 are the fixture's one-shot
        // injection protocol. The returned +1 never strongly retains buffers.
        let pointer = unsafe { (self.arm)(mode) };
        assert!(!pointer.is_null(), "test hook installation");
        Control {
            hook: self,
            pointer,
        }
    }
}

struct Control<'a> {
    hook: &'a Hook,
    pointer: *mut c_void,
}
impl Control<'_> {
    fn live(&self) -> usize {
        // SAFETY: this control owns its +1; weak observations are fixture-atomic.
        unsafe { (self.hook.live)(self.pointer) }
    }
    fn finished(&self) -> bool {
        // SAFETY: acquire observation of the actual SDK terminal callback.
        unsafe { (self.hook.finished)(self.pointer) == 1 }
    }
    fn signal(&self) {
        // SAFETY: independent live controller; the fixture owns its GPU gate.
        unsafe { (self.hook.signal)(self.pointer) }
    }
    fn wait(&self) {
        let start = Instant::now();
        while !self.finished() {
            assert!(
                start.elapsed() < Duration::from_secs(10),
                "late callback deadline"
            );
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    fn output(&self, index: usize) -> Vec<u16> {
        let mut bits = vec![0; 32];
        // SAFETY: writable span; fixture requires actual terminal completion and
        // weak-loads an output strongly only during this bounded copy.
        let count =
            unsafe { (self.hook.output)(self.pointer, index, bits.as_mut_ptr(), bits.len()) };
        assert!(count <= bits.len());
        bits.truncate(count);
        bits
    }
}
impl Drop for Control<'_> {
    fn drop(&mut self) {
        // SAFETY: consume exactly our +1; an SDK callback independently owns the
        // controller when pending. Neither destructor asserts GPU completion.
        unsafe {
            (self.hook.destroy)(self.pointer);
        }
    }
}

#[test]
#[ignore = "real Graph faults, including its 30-second callback deadline; isolated process"]
fn graph_faults_retain_access_until_callback_and_never_reopen() {
    let first = match MetalBf16Executor::with_ffn_graphs(4, 8) {
        Ok(executor) => executor,
        Err(MetalBf16Error::Device) => return,
        Err(MetalBf16Error::Unsupported(_)) => {
            // Explicit unsupported selection rejects rather than falling back;
            // this branch is not Graph device qualification.
            assert!(MetalBf16Executor::new().is_ok());
            return;
        }
        Err(error) => panic!("Graph preparation: {error}"),
    };
    let hook = Hook::load();
    drop(first);
    for (mode, drop_pending) in [(1, false), (3, false), (2, false), (4, false), (4, true)] {
        let executor = MetalBf16Executor::with_ffn_graphs(4, 8).unwrap();
        let input = executor.upload(&[0x3f80, 0, 0, 0].repeat(16)).unwrap();
        let left = executor.upload(&[0x3f80, 0, 0, 0].repeat(8)).unwrap();
        let right = executor.upload(&[0x4000, 0, 0, 0].repeat(8)).unwrap();
        let shape = MetalLinearShape {
            rows: 16,
            columns: 8,
            inner: 4,
        };
        // Compile the independent peer before gating actual GPU execution, so
        // this test never relies on outrunning the OS watchdog during setup.
        let peer = MetalBf16Executor::new().unwrap();
        let control = hook.arm(mode);
        let failed = executor.rms_linear_pair(&input, &left, &right, shape, 0.75, None);
        eprintln!(
            "mode {mode}: error={:?}, callback_finished={}",
            failed.as_ref().err(),
            control.finished()
        );
        assert!(failed.is_err());
        if mode == 1 {
            executor.check_healthy().unwrap();
            let (a, b) = executor
                .rms_linear_pair(&input, &left, &right, shape, 0.75, None)
                .unwrap();
            assert_eq!(a.as_slice(), &[0x3f80; 128]);
            assert_eq!(b.as_slice(), &[0x4000; 128]);
            continue;
        }
        drop((input, left, right));
        assert!(executor.check_healthy().is_err());
        assert_eq!(
            control.live(),
            5,
            "error must retain every accessible allocation"
        );
        if mode == 3 {
            executor.retire_failure().unwrap();
            assert_eq!(control.live(), 0);
            assert!(executor.check_healthy().is_err());
            continue;
        }
        let start = Instant::now();
        assert!(!control.finished());
        assert!(executor.retire_failure().is_err());
        assert!(executor.retire_failure().is_err());
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "failure settlement must not wait"
        );
        let value = executor.upload(&[0x3f80]).unwrap();
        assert!(
            executor.scale(&value, 2.0).is_err(),
            "native work must also close"
        );
        assert_eq!(peer.scale(&value, 3.0).unwrap().as_slice(), &[0x4040]);
        if drop_pending {
            drop(executor);
            assert_eq!(control.live(), 5);
            control.signal();
            control.wait();
            assert_eq!(control.output(0), vec![0x3f80; 32]);
            assert_eq!(control.output(1), vec![0x4000; 32]);
            assert_eq!(control.live(), 5, "unresolved Drop permanently quarantines");
        } else {
            control.signal();
            control.wait();
            assert_eq!(control.output(0), vec![0x3f80; 32]);
            assert_eq!(control.output(1), vec![0x4000; 32]);
            executor.retire_failure().unwrap();
            executor.retire_failure().unwrap();
            assert_eq!(
                control.live(),
                0,
                "explicit terminal retirement releases buffers"
            );
            assert!(executor.check_healthy().is_err());
        }
        eprintln!("mode {mode}, drop={drop_pending}: callback/retirement contract passed");
    }
}
