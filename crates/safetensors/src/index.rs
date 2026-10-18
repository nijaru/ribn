//! Use the upstream header/layout validator without materializing the payload.

use std::collections::BTreeMap;

use safetensors::tensor::Metadata;

use super::{ArtifactError, TensorEntry, scalar_type};

// Same ceiling as safetensors 0.8.0's SafeTensors::read_metadata. Check before
// allocating or reading a header claimed by an untrusted length prefix.
const MAX_HEADER_BYTES: usize = 100_000_000;

pub(super) fn header_len(prefix: &[u8], file_len: usize) -> Result<usize, ArtifactError> {
    let prefix: [u8; 8] = prefix.try_into().map_err(ArtifactError::format)?;
    let len = usize::try_from(u64::from_le_bytes(prefix)).map_err(ArtifactError::format)?;
    if len > MAX_HEADER_BYTES || len.checked_add(8).is_none_or(|end| end > file_len) {
        return Err(ArtifactError::format("invalid header length"));
    }
    Ok(len)
}

pub(super) fn parse(
    header: &[u8],
    file_len: usize,
) -> Result<BTreeMap<String, TensorEntry>, ArtifactError> {
    // Metadata's Deserialize calls Metadata::new, which validates shape/dtype
    // sizes, checked arithmetic and contiguous offsets. No second format parser.
    let metadata: Metadata = serde_json::from_slice(header).map_err(ArtifactError::format)?;
    let data_start = 8 + header.len();
    if metadata.data_len().checked_add(data_start) != Some(file_len) {
        return Err(ArtifactError::format(
            "payload length does not match metadata",
        ));
    }
    Ok(metadata
        .tensors()
        .into_iter()
        .map(|(name, info)| {
            // All relative ranges are <= data_len; the checked total above also
            // proves these absolute additions cannot overflow or escape the file.
            (
                name,
                TensorEntry {
                    scalar_type: scalar_type(info.dtype),
                    shape: info.shape.clone(),
                    range: data_start + info.data_offsets.0..data_start + info.data_offsets.1,
                },
            )
        })
        .collect())
}
