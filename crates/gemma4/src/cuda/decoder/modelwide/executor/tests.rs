//! Real backend contract gates. These use runtime-issued identities, actual
//! cached continuations and positive CUDA work, not simulated model forwards.
use super::*;
use crate::{Gemma4Source, ParameterId, cuda_test_support::issued};
use cudarc::driver::CudaContext;
use ribn::{GenerationOptions, StepKind};
use ribn_foundation::BytePool;
use ribn_hf::LocalModelPackage;
use serde_json::Value;
use std::{
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

fn chat() -> Vec<u32> {
    let value: Value = serde_json::from_slice(
        &std::fs::read(std::env::var_os("RIBN_GEMMA_REFERENCE").expect("pinned reference"))
            .unwrap(),
    )
    .unwrap();
    value["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "chat")
        .unwrap()["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|id| u32::try_from(id.as_u64().unwrap()).unwrap())
        .collect()
}
fn loaded() -> (Gemma4CudaExecutor, Arc<BytePool>) {
    let package = LocalModelPackage::open(PathBuf::from(
        std::env::var_os("RIBN_GEMMA_PACKAGE").expect("pinned package"),
    ))
    .unwrap();
    let source = Gemma4Source::open(&package).unwrap();
    let pool = BytePool::new(20_u64 << 30).shared();
    let scratch = 64 * source.config().num_attention_heads() * 16;
    let model =
        Gemma4CudaModelWide::load(&source, &CudaContext::new(0).unwrap(), &pool, 64, scratch)
            .unwrap();
    (Gemma4CudaExecutor::new(model).unwrap(), pool)
}
fn row(
    sequence: SequenceId,
    kind: StepKind,
    prefix: u32,
    token_budget: u32,
    output_budget: u32,
) -> BatchItem {
    BatchItem {
        sequence,
        kind,
        prefix,
        token_budget,
        output_budget,
    }
}
fn completed(executor: &mut Gemma4CudaExecutor, id: SubmissionId) -> Vec<StepCompletion> {
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        if let Some(rows) = executor.poll(id).unwrap() {
            return rows;
        }
        assert!(Instant::now() < deadline, "CUDA completion deadline");
        std::thread::sleep(Duration::from_micros(100));
    }
}
fn execute(executor: &mut Gemma4CudaExecutor, rows: &[BatchItem]) -> Vec<StepCompletion> {
    assert!(matches!(
        executor.prepare(rows).unwrap(),
        BatchPreparation::Ready
    ));
    let id = executor.submit(rows).unwrap();
    completed(executor, id)
}
fn owner(executor: &Gemma4CudaExecutor) -> &Execution {
    executor.execution.as_ref().unwrap()
}

#[test]
#[ignore = "real 15 GB CUDA model; release-only serialized device gate"]
#[expect(
    clippy::too_many_lines,
    reason = "one mixed/reordered/deferred-cancel/partial-enqueue owner lifecycle"
)]
fn mixed_offers_defer_cancelled_cache_refunds_and_never_commit_a_later_failed_offer() {
    let _device = crate::cuda_test_support::DEVICE.lock().unwrap();
    let (mut executor, pool) = loaded();
    let identities = issued::<4>();
    let ids = identities.map(|(_, id)| id);
    let prompt = chat();
    assert_eq!(prompt.len(), 18);
    let prompts = [
        prompt.clone(),
        prompt[..17].to_vec(),
        prompt.clone(),
        prompt.repeat(3),
    ];
    let mut cancelled_charge = 0;
    for (index, ((request, sequence), prompt)) in identities.into_iter().zip(prompts).enumerate() {
        let before = pool.granted();
        assert!(matches!(
            executor
                .admit(
                    request,
                    sequence,
                    &TokenRequest::new(
                        prompt,
                        GenerationOptions {
                            max_output_tokens: 4,
                            ..GenerationOptions::default()
                        }
                    )
                )
                .unwrap(),
            Admission::Ready
        ));
        if index == 1 {
            cancelled_charge = pool.granted() - before;
        }
    }
    let first = ids[..3]
        .iter()
        .map(|&id| row(id, StepKind::Prefill, 0, 16, 0))
        .collect::<Vec<_>>();
    let partial = execute(&mut executor, &first);
    assert!(
        partial
            .iter()
            .all(|result| result.prefix == 16 && result.tokens.is_empty())
    );
    let tail = [
        row(ids[2], StepKind::Prefill, 16, 2, 1),
        row(ids[1], StepKind::Prefill, 16, 1, 1),
        row(ids[0], StepKind::Prefill, 16, 2, 1),
    ];
    let tail_results = execute(&mut executor, &tail);
    assert_eq!(tail_results[0].tokens, vec![236_776]);
    assert_eq!(tail_results[2].tokens, vec![236_776]);
    assert!(
        tail_results
            .iter()
            .zip(tail)
            .all(|(result, work)| result.sequence == work.sequence)
    );

    let mixed = [
        row(ids[0], StepKind::Decode, 18, 1, 1),
        row(ids[3], StepKind::Prefill, 0, 16, 0),
        row(ids[2], StepKind::Decode, 18, 1, 1),
    ];
    executor.prepare(&mixed).unwrap();
    let submission = executor.submit(&mixed).unwrap();
    assert_eq!(
        owner(&executor).active,
        Some([0, 2]),
        "nonadjacent physical members"
    );
    assert_eq!(owner(&executor).states[&ids[0]].cursor, 18);
    assert_eq!(owner(&executor).model.position(ids[0]), Some(18));
    assert_eq!(owner(&executor).model.position(ids[3]), Some(0));
    let charged = pool.granted();
    executor.release(ids[1]).unwrap();
    executor.release(ids[1]).unwrap();
    assert_eq!(
        pool.granted(),
        charged,
        "logical cancellation does not refund pending physical backing"
    );
    assert_eq!(owner(&executor).model.position(ids[1]), None);
    assert!(
        executor.release(ids[0]).is_err(),
        "a submitted member stays owned"
    );
    let results = completed(&mut executor, submission);
    assert_eq!(
        results
            .iter()
            .map(|result| result.sequence)
            .collect::<Vec<_>>(),
        mixed.map(|work| work.sequence)
    );
    assert_eq!(results[0].tokens, vec![126_641]);
    assert!(results[1].tokens.is_empty());
    assert_eq!(results[2].tokens, vec![126_641]);
    assert_eq!(
        pool.granted(),
        charged - cancelled_charge,
        "acknowledgement reclaims exactly the retired cache"
    );
    assert_eq!(owner(&executor).states[&ids[3]].cursor, 16);

    let failing = [
        row(ids[3], StepKind::Prefill, 16, 16, 0),
        row(ids[0], StepKind::Decode, 19, 1, 1),
        row(ids[2], StepKind::Decode, 19, 1, 1),
    ];
    executor.prepare(&failing).unwrap();
    // An internal late-binding failure after real positive work, without a raw
    // stream escape, invalid memory access or production fault-injection hook.
    assert!(
        executor
            .execution
            .as_mut()
            .unwrap()
            .model
            .owner_mut()
            .model
            .cuda
            .slots
            .remove(&ParameterId::TokenEmbedding)
            .is_some()
    );
    let charged = pool.granted();
    assert!(executor.submit(&failing).is_err());
    let execution = owner(&executor);
    assert!(execution.phase == Phase::Failed);
    assert_eq!(
        execution.model.position(ids[3]),
        Some(32),
        "earlier qualified fallback really ran"
    );
    assert_eq!(
        execution.states[&ids[3]].cursor, 16,
        "no earlier logical commit"
    );
    assert_eq!(execution.states[&ids[3]].remaining, 4);
    assert_eq!(execution.states[&ids[3]].input, None);
    assert_eq!(execution.model.position(ids[0]), Some(19));
    assert_eq!(execution.states[&ids[0]].cursor, 19);
    assert_eq!(execution.states[&ids[0]].remaining, 2);
    assert!(
        execution.model.owner().commands.is_some(),
        "recording queue retains the actual layer accesses"
    );
    assert_eq!(pool.granted(), charged);
    assert!(executor.prepare(&failing).is_err());
    executor.synchronize().unwrap();
    executor.synchronize().unwrap();
    executor.release(ids[0]).unwrap();
    assert_eq!(pool.granted(), 0);
}

#[test]
#[ignore = "real 15 GB CUDA model; release-only serialized device gate"]
fn two_pairs_commit_together_and_later_malformed_rows_never_advance() {
    let _device = crate::cuda_test_support::DEVICE.lock().unwrap();
    let (mut executor, pool) = loaded();
    let identities = issued::<4>();
    let ids = identities.map(|(_, id)| id);
    for (request, sequence) in identities {
        assert!(matches!(
            executor
                .admit(
                    request,
                    sequence,
                    &TokenRequest::new(
                        chat(),
                        GenerationOptions {
                            max_output_tokens: 4,
                            ..GenerationOptions::default()
                        }
                    )
                )
                .unwrap(),
            Admission::Ready
        ));
    }
    let prefill = ids.map(|id| row(id, StepKind::Prefill, 0, 18, 1));
    assert!(
        execute(&mut executor, &prefill)
            .iter()
            .all(|result| result.tokens == [236_776])
    );
    let offered = [ids[3], ids[0], ids[2], ids[1]].map(|id| row(id, StepKind::Decode, 18, 1, 1));
    executor.prepare(&offered).unwrap();
    let submission = executor.submit(&offered).unwrap();
    assert_eq!(owner(&executor).active, Some([0, 1]));
    let deadline = Instant::now() + Duration::from_secs(120);
    while owner(&executor).active != Some([2, 3]) {
        assert!(
            executor.poll(submission).unwrap().is_none(),
            "no partial whole-offer result"
        );
        assert!(
            Instant::now() < deadline,
            "first lane acknowledgement deadline"
        );
        std::thread::sleep(Duration::from_micros(100));
    }
    for (index, work) in offered.iter().enumerate() {
        assert_eq!(owner(&executor).states[&work.sequence].cursor, 18);
        assert_eq!(
            owner(&executor).model.position(work.sequence),
            Some(if index < 2 { 19 } else { 18 })
        );
    }
    let charged = pool.granted();
    assert!(
        executor.release(offered[0].sequence).is_err(),
        "even acknowledged members stay owned until whole-offer completion"
    );
    assert_eq!(pool.granted(), charged);
    let completed = completed(&mut executor, submission);
    for (result, work) in completed.iter().zip(offered) {
        assert_eq!(result.sequence, work.sequence);
        assert_eq!(result.prefix, 19);
        assert_eq!(result.tokens, vec![126_641]);
    }
    assert!(
        executor
            .prepare(&[
                row(ids[0], StepKind::Decode, 19, 1, 1),
                row(ids[1], StepKind::Decode, 20, 1, 1),
            ])
            .is_err()
    );
    for id in ids {
        assert_eq!(owner(&executor).model.position(id), Some(19));
        assert_eq!(owner(&executor).states[&id].cursor, 19);
        assert_eq!(owner(&executor).states[&id].remaining, 2);
    }
    assert!(owner(&executor).rows.is_empty());
    assert_eq!(pool.granted(), charged);
    assert!(owner(&executor).phase == Phase::Failed);
    executor.synchronize().unwrap();
    assert_eq!(pool.granted(), 0);
}
