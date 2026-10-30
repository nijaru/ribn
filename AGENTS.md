# Ribn agent guidance

Ribn (pronounced “ribbon”) targets a **state-of-the-art, idiomatic Rust, server-first
inference engine**, initially competing with vLLM, SGLang and similar serving engines.
Broad support for common latest models and hardware is the product goal; Qwen GGUF/CUDA
and the RTX 4090 are qualification workloads, not the product boundary. Training is a
possible later execution system, not inference policy.

## Product direction

- Pursue competitive throughput, latency, memory efficiency, model-support velocity
  and serving reliability together. Correct ownership and Rust ergonomics enable that
  goal; do not substitute a correctness-only, local-only or niche engine strategy.
- Distinguish the destination from implemented and qualified support. Current gaps
  determine staged work, not a smaller product ambition. Do not claim SOTA, universal
  hardware support or automatic day-zero model support without evidence.
- Idiomatic Rust means clear ownership, typed interfaces and errors, and cohesive
  execution—not Rust-only kernels or abstraction for its own sake. Use qualified
  vendor/external kernels when appropriate; retain backend-native optimization.
- Follow `docs/inference-engine-design.md#scope-and-success-criterion` for success
  criteria and `docs/roadmap.md` for sequencing. Do not invent a new competitive niche,
  kernel strategy or hardware priority from the prototype's limitations.
- Select work by `docs/roadmap.md#decision-milestones`, not by whichever host-only
  repair is easiest to finish. When device access is unavailable, advance concrete
  baseline or real-model integration prerequisites; do not accumulate unqualified
  device changes or substitute unrelated subsystem completion for competitive evidence.

## Authority and working reference

- Work on `main` unless asked otherwise. All v0 APIs are unstable: replace flawed
  contracts and remove obsolete paths rather than add compatibility shims.
- Read the owners relevant to the requested change, then trace its source/tests:
  `README.md` for setup and interfaces; `docs/inference-engine-design.md` for target
  scope and architecture; `docs/resource-protocol.md` for execution/resource contracts;
  `docs/roadmap.md` for sequencing and qualification status; `docs/architecture.md`
  for current implementation structure. Do not load every document for every task.
- `docs/execution-foundation.md` and `docs/pipeline-composition.md` retain experimental
  counterexamples and limitations, not provisional APIs to preserve. Historical evidence
  and private notes do not override current design owners.
- Resolve observable ownership, failure scope, cancellation, bounds, readiness and
  shutdown before implementing a dependent architectural slice. Record open decisions
  in the roadmap. A counterexample may justify changing the design; update the owning
  contract and tests rather than silently weaken it.
- Choose local data structures, helper names and implementation details within those
  contracts. Do not invent a universal framework to avoid making concrete decisions.
- Keep commits focused and the tree green. Preserve unrelated work and device artifacts.

- Keep `README.md` stable and low-maintenance: identity, setup, current interfaces and
  documentation links. Do not turn it into a status log or a next-steps list.

Architecture and execution invariants belong in the design and resource-protocol
owners above, not a second specification here. Current queue topology, reservation
strategy and in-flight batch limits are prototype mechanics; change them with
correctness and workload evidence, not merely to imitate another engine.

## Qualification and performance

Read `.agents/skills/model-integration/SKILL.md` for model integration, support review
or optimized-variant promotion. Read `.agents/skills/execution-contracts/SKILL.md` when
changing lifecycle, preparation, resource ownership or concurrent application access.

- Compilation and host tests are not GPU evidence. Qualify optimized variants over
  their entire automatic selection scope; preserve a qualified fallback.
- Numerical acceptance is independent of observed failure. Check shape and finiteness;
  bit equality applies when arithmetic preservation is intended. Reordered algorithms
  require independently justified metrics and tolerances. Compare common histories.
- GDN chunk scan remains opt-in while its full-model gate fails. Do not call it harmless
  or widen tolerances based on existing reference error.
- Before repeating rejected kernel approaches, read the relevant Qwen results in
  `benchmarks/qwen-prefill-qualification.md` or Gemma Metal results in `docs/roadmap.md`.
  Reconsider them only with new evidence; preserve the workload, numerical and timing
  conditions that made them lose.
- Serialize full-model device tests on `ssh desktop`, one model process at a time, and
  check GPU utilization before launching. Its login shell is fish, so send multi-line
  scripts with `ssh desktop bash -s`. Preserve its untracked `cuda-rust-probe/simt/*`.
  Profiler counters currently require unavailable privilege; do not infer measured
  occupancy/stall causes from speculation.
- Benchmark server workloads with mixed lengths/arrivals, memory pressure and latency
  objectives. One prompt's prefixes are not representative throughput evidence.

## Verification

```sh
python3 tools/check-boundaries.py
cargo fmt --all -- --check
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo clippy --workspace --all-targets --locked --features cuda -- -D warnings
# macOS Metal second decoder (ignored tests are release-only; a missing device is the
# only permitted skip, never a kernel/executor build failure):
cargo clippy -p engine-gemma4 -p ribn-text -p ribn-cli --all-targets --locked --features metal -- -D warnings
RIBN_GEMMA_PACKAGE=<package> RIBN_GEMMA_REFERENCE=<reference.json> \
  cargo test -p engine-gemma4 --features metal --release --test metal_decoder -- --ignored --test-threads=1
```

Capture actual exit codes; pipes must not mask failures. Text behavior is host-testable;
CUDA assembly and device-specific CLI paths still require CUDA-feature checks.
Device commands/evidence live in
`benchmarks/runtime-contract.md`; kernel-language qualification lives in
`docs/cuda-rust-migration.md`. Report failed or unrun checks explicitly.
