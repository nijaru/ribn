//! Host-only continuation geometry probe. No state or device storage is created.

use engine_gemma4::Gemma4Config;
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        return Err("usage: kv_ranges CONFIG_JSON PREFIX CHUNK REQUEST_CAPACITY".into());
    }
    let value = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    let config = Gemma4Config::from_json(&value)?;
    let step = config.kv_step(args[1].parse()?, args[2].parse()?, args[3].parse()?)?;
    let mut owners = serde_json::Map::new();
    for owner in step.owners() {
        let before = owner.before().range();
        let attention = owner.attention().range();
        let after = owner.after().range();
        owners.insert(
            owner.layer().to_string(),
            json!({
                "head_dim":owner.head_dim(), "kv_heads":config.num_key_value_heads(),
                "last_consumer":owner.last_consumer(), "before":[before.start,before.end],
                "attention":[attention.start,attention.end], "after":[after.start,after.end],
            }),
        );
    }
    let input = step.input();
    serde_json::to_writer_pretty(
        std::io::stdout().lock(),
        &json!({
            "input":[input.start,input.end],"owners":owners,
            "logical_retained_bytes":step.retained_payload_bytes(),
            "logical_donor_bytes":step.donor_payload_bytes(),
            "scope":"range arithmetic only; no state validity, reservation or physical allocation proof",
        }),
    )?;
    Ok(())
}
