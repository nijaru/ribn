use engine_metal::{MetalBf16Error, MetalBf16Executor};

/// Missing hardware skips; any compilation or pipeline failure must fail the test.
pub fn executor() -> Option<MetalBf16Executor> {
    match MetalBf16Executor::new() {
        Ok(executor) => Some(executor),
        Err(MetalBf16Error::Device) => {
            eprintln!("no Metal device; skipping");
            None
        }
        Err(error) => panic!("Metal preparation failed: {error}"),
    }
}
