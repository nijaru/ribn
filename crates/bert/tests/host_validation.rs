//! The safe host-to-device boundary rejects malformed parameters before CUDA.
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use engine_bert::{BertConfig, CudaBertEncoder, EncoderError, HostWeights, WeightError};
use ribn_hf::LocalModelPackage;

#[test]
fn malformed_host_weights_reject_before_device_initialization() {
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../batch/tests/fixtures/bert-tiny");
    let package = LocalModelPackage::open(fixture).unwrap();
    let config = BertConfig::from_json(package.config()).unwrap();
    let mut host = HostWeights::load(&package, &config).unwrap();
    host.layers[0].query.weight.truncate(1);
    // An invalid device ordinal distinguishes host rejection from attempting
    // backend initialization. No actual device allocation is needed by this test.
    match CudaBertEncoder::from_host(config, &host, usize::MAX) {
        Err(EncoderError::Weights(WeightError::Length { .. })) => {}
        Err(error) => panic!("malformed host weights reached CUDA initialization: {error}"),
        Ok(_) => panic!("malformed host weights were accepted"),
    }
}
