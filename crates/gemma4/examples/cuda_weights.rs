//! Native BF16 residency/readback gate. This does not execute a Gemma decoder.
use cudarc::driver::CudaContext;
use engine_gemma4::{Gemma4CudaWeights, Gemma4Source};
use ribn_foundation::BytePool;
use ribn_hf::LocalModelPackage;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{io::Read, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    if !(2..=3).contains(&args.len()) {
        return Err("usage: cuda_weights PACKAGE WEIGHT_BUDGET_BYTES [STAGING_BYTES]".into());
    }
    let budget = args[1].to_str().ok_or("invalid budget")?.parse::<u64>()?;
    let staging = args.get(2).map_or(Ok(1024 * 1024), |v| {
        v.to_str()
            .ok_or("invalid staging size")?
            .parse::<usize>()
            .map_err(|_| "invalid staging size")
    })?;
    let started = Instant::now();
    let package = LocalModelPackage::open(&args[0])?;
    let source = Gemma4Source::open(&package)?;
    let context = CudaContext::new(0)?;
    let stream = context.new_stream()?;
    let pool = BytePool::new(budget).shared();
    let (free_before, total) = context.mem_get_info()?;
    let mut weights = match Gemma4CudaWeights::load(&source, stream, &pool, staging) {
        Ok(weights) => weights,
        Err(mut failure) => {
            if let Err(retirement) = failure.retire() {
                eprintln!("load cleanup remains unresolved: {retirement}");
            }
            return Err(failure.into());
        }
    };
    let (free_resident, _) = context.mem_get_info()?;
    let allocation = weights.allocation_id().get();
    let requested = weights.allocated_bytes();
    let tied = weights.token_embedding_slot() == weights.output_projection_slot();
    if !tied || pool.granted() != requested {
        return Err("tied allocation or charge mismatch".into());
    }
    let mut records = Vec::with_capacity(source.parameters().len());
    let mut expected = vec![0; staging];
    for (&id, parameter) in source.parameters() {
        let mut reader = parameter.reader();
        let mut source_hash = Sha256::new();
        let mut device_hash = Sha256::new();
        let mut offset = 0;
        let length = parameter.spec().byte_len()?;
        while offset < length {
            let len = staging.min(length - offset);
            reader.read_exact(&mut expected[..len])?;
            let actual = weights.read_chunk(id, u64::try_from(offset)?, len)?;
            if actual != expected[..len] {
                return Err(
                    format!("BF16 bit mismatch: {} at {offset}", parameter.spec().name).into(),
                );
            }
            source_hash.update(&expected[..len]);
            device_hash.update(&actual);
            offset += len;
        }
        records.push(json!({"name":parameter.spec().name,"shape":parameter.spec().shape,"dtype":"BF16","bytes":length,"source_sha256":format!("{:x}",source_hash.finalize()),"device_sha256":format!("{:x}",device_hash.finalize())}));
    }
    weights.close()?;
    let (free_after, _) = context.mem_get_info()?;
    if pool.granted() != 0 {
        return Err("retired weights retained a pool charge".into());
    }
    println!(
        "{}",
        json!({"complete":true,"scope":"native BF16 weight materialization only; no model execution or full-inference peak bound","active_parameters":records.len(),"unique_encoded_bytes":source.encoded_bytes(),"requested_allocation_bytes":requested,"staging_bytes":staging,"allocation_id":allocation,"tied_embedding_head":tied,"pool_granted_after_close":pool.granted(),"device_memory":{"total":total,"free_before":free_before,"free_resident":free_resident,"free_after_close":free_after},"elapsed_seconds":started.elapsed().as_secs_f64(),"parameters":records})
    );
    Ok(())
}
