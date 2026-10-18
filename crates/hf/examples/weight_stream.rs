//! Host-only probe: hash a package-selected artifact and every encoded tensor
//! through the ordinary loader with one caller-owned payload buffer. No execution.

use std::error::Error;
use std::io::Read;

use ribn_hf::LocalModelPackage;
use sha2::{Digest, Sha256};

fn checksum(mut reader: impl Read, buffer: &mut [u8]) -> Result<(u64, String), Box<dyn Error>> {
    let mut digest = Sha256::new();
    let mut bytes = 0_u64;
    loop {
        let read = reader.read(buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
        bytes = bytes
            .checked_add(u64::try_from(read)?)
            .ok_or("byte count overflow")?;
    }
    Ok((bytes, format!("{:x}", digest.finalize())))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: weight_stream PACKAGE PARAMETER_IN_SELECTED_SHARD".into());
    }
    let parameter = args[1].to_str().ok_or("parameter must be UTF-8")?;
    let package = LocalModelPackage::open(&args[0])?;
    let mut weights = package.weight_set_resident(1)?;
    let artifact = weights.artifact(parameter)?;
    let mut buffer = vec![0_u8; 1024 * 1024];
    let (source_bytes, source_sha256) = checksum(artifact.reader(), &mut buffer)?;
    if source_bytes != u64::try_from(artifact.len())? {
        return Err("incomplete artifact stream".into());
    }
    let mut tensors = serde_json::Map::new();
    for name in artifact.names() {
        let tensor = artifact.tensor(&name)?;
        let (bytes, sha256) = checksum(tensor.reader(), &mut buffer)?;
        if bytes != u64::try_from(tensor.len())? {
            return Err(format!("incomplete tensor: {name}").into());
        }
        tensors.insert(
            name,
            serde_json::json!({
                "dtype": format!("{:?}", tensor.scalar_type()),
                "shape": tensor.shape(), "bytes": bytes, "sha256": sha256,
            }),
        );
    }
    println!(
        "{}",
        serde_json::json!({
            "source_bytes": source_bytes, "source_sha256": source_sha256,
            "resident_payload_bytes": artifact.resident_payload_bytes(),
            "buffer_bytes": buffer.len(), "tensors": tensors,
        })
    );
    Ok(())
}
