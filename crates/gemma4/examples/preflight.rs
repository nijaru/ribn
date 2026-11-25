//! Actual local-package preflight, optionally hashing every active encoded source.
//! No model execution, device allocation or whole-weight host materialization.

use engine_gemma4::{Exclusion, Gemma4Source, ParameterPayload};
use ribn_hf::LocalModelPackage;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::io::{self, Read};

fn hash_reader(
    reader: &mut dyn Read,
    buffer: &mut [u8],
    hash: &mut Sha256,
) -> Result<usize, io::Error> {
    let mut total = 0;
    loop {
        let n = reader.read(buffer)?;
        if n == 0 {
            return Ok(total);
        }
        hash.update(&buffer[..n]);
        total += n;
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.is_empty() || args.len() > 2 || args.get(1).is_some_and(|s| s != "--hash-parameters") {
        return Err("usage: preflight PACKAGE [--hash-parameters]".into());
    }
    let package = LocalModelPackage::open(&args[0])?;
    let source = Gemma4Source::open(&package)?;
    let mut parameters = serde_json::Map::new();
    let mut buffer = vec![0; 1024 * 1024];
    for parameter in source.parameters().values() {
        let spec = parameter.spec();
        let dtype = match parameter.payload() {
            ParameterPayload::Bf16(_) => "BF16".to_owned(),
            ParameterPayload::AffineGroup64 { bits, .. } => format!("AFFINE_Q{bits}_G64"),
        };
        let digest = if args.len() == 2 {
            let mut hash = Sha256::new();
            let total = match parameter.payload() {
                ParameterPayload::Bf16(r) => hash_reader(&mut r.clone(), &mut buffer, &mut hash)?,
                ParameterPayload::AffineGroup64 {
                    packed,
                    scales,
                    biases,
                    ..
                } => {
                    hash_reader(&mut packed.clone(), &mut buffer, &mut hash)?
                        + hash_reader(&mut scales.clone(), &mut buffer, &mut hash)?
                        + hash_reader(&mut biases.clone(), &mut buffer, &mut hash)?
                }
            };
            if total != parameter.encoded_bytes() {
                return Err("incomplete parameter read".into());
            }
            Some(format!("{:x}", hash.finalize()))
        } else {
            None
        };
        parameters.insert(
            spec.name.clone(),
            json!({"role": format!("{:?}",spec.id),
            "shape": spec.shape, "dtype": dtype, "bytes": parameter.encoded_bytes(), "sha256": digest}),
        );
    }
    let layers: Vec<_> = source.config().layers().iter().enumerate().map(|(i,g)| json!({
        "layer":i,"attention":format!("{:?}",g.kind),"head_dim":g.head_dim,"kv_owner":g.kv_owner,
    })).collect();
    let report = json!({
        "scope": "host source preflight only; not executable model support",
        "parameters": parameters, "active_encoded_bytes": source.encoded_bytes(), "layers":layers,
        "excluded_tail_parameters": source.excluded().iter().filter(|p|p.reason==Exclusion::SharedTailProjection).count(),
        "excluded_non_text_parameters": source.excluded().iter().filter(|p|p.reason==Exclusion::NonText).count(),
    });
    serde_json::to_writer_pretty(io::stdout().lock(), &report)?;
    Ok(())
}
