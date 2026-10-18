//! Qualify the real text processor against a frozen independent CPU corpus.
use engine_gemma4::{Gemma4TextProcessor, TextLimits, TextMessage};
use ribn_hf::LocalModelPackage;
use serde_json::{Value, json};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: processor PACKAGE CORPUS_JSON".into());
    }
    let package = LocalModelPackage::open(&args[0])?;
    let processor = Gemma4TextProcessor::open(
        &package,
        TextLimits {
            input_bytes: 65_536,
            rendered_bytes: 131_072,
            tokens: 8192,
            messages: 128,
        },
    )?;
    let corpus: Value = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let mut checks = 0;
    for row in corpus["raw"].as_array().ok_or("missing raw corpus")? {
        let ids = processor.encode(row["text"].as_str().ok_or("missing text")?)?;
        require(json!(ids) == row["ids"], "raw token IDs")?;
        require(
            processor.decode(&ids)? == row["decoded"],
            "raw decoded text",
        )?;
        checks += 1;
    }
    for row in corpus["chat"].as_array().ok_or("missing chat corpus")? {
        let messages = row["messages"]
            .as_array()
            .ok_or("missing messages")?
            .iter()
            .map(|m| {
                Ok(TextMessage {
                    role: m["role"].as_str().ok_or("missing role")?,
                    content: m["content"].as_str().ok_or("missing content")?,
                })
            })
            .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
        require(
            processor.render_chat(&messages)? == row["rendered"],
            "rendered chat",
        )?;
        let ids = processor.encode_chat(&messages)?;
        require(json!(ids) == row["ids"], "chat token IDs")?;
        require(
            processor.decode(&ids)? == row["decoded"],
            "chat decoded text",
        )?;
        checks += 1;
    }
    for row in corpus["decode"].as_array().ok_or("missing decode corpus")? {
        let ids: Vec<u32> = serde_json::from_value(row["ids"].clone())?;
        require(processor.decode(&ids)? == row["text"], "whole decode")?;
        let mut stream = processor.decoder();
        let mut text = String::new();
        for id in ids {
            text.push_str(&stream.push(id)?);
        }
        text.push_str(&stream.finish()?);
        require(text == row["text"], "finalized stream decode")?;
        checks += 1;
    }
    require(
        json!(processor.stop_token_ids()) == corpus["stops"],
        "generation stops",
    )?;
    println!(
        "{}",
        json!({"passed_cases":checks,"stops":processor.stop_token_ids(),"scope":"host processor only; no model/device execution"})
    );
    Ok(())
}
fn require(ok: bool, name: &str) -> Result<(), Box<dyn std::error::Error>> {
    if !ok {
        return Err(format!("processor reference mismatch: {name}").into());
    }
    Ok(())
}
