# Engineering roadmap

[Target design](inference-engine-design.md) owns product direction and architecture;
[resource protocol](resource-protocol.md) owns execution/resource ownership;
[architecture](architecture.md) maps current code. This document owns delivery order,
open implementation choices and acceptance gates. All v0 interfaces may change.

## Current execution order

**Next: establish a matched competitive endpoint baseline on the qualified contiguous
path.** The loopback text adapter is host-qualified, with a bounded CUDA HTTP gate and
neutral/native client compatibility smokes passed. Owned-input retention is
host-qualified. Prove model-local integration with a real dense second decoder alongside
baseline work; do not postpone it until hybrid caching or scheduling is complete.
Choose physical execution and constrained-memory experiments from the measured deficit.
Prefix-cache benefit, a second decoder/VLM, CUDA Rust migration and distributed execution
are not prerequisites for the first baseline. They remain important product work.

This replaces the serial 4b/4c → 4d → 5 dependency. The historical slice identifiers below
still locate evidence; they no longer require finishing prefix reuse before serving or
measuring scheduling. Safety prerequisites apply to the path being enabled, not every
future capability. No redesigned capability is implemented or qualified by this document.

| Delivery | Required outcome | Dependency |
| --- | --- | --- |
| Owned-input repair — host-qualified | Accepted heap capacity/options checked before preprocessing queue insertion; excess rejects without normalization | Completed prerequisite for bounded text admission; not a process-RSS bound |
| Minimal text serving (5) | Thin owned-facade adapter, protocol/overload/cancellation gates, matched endpoint measurements | Repaired text retention and current path's applicable lifecycle gates; no paging dependency |
| Physical execution and scheduling (4d) | Useful heterogeneous lanes, measured scheduling costs/goodput, bounded metadata | Numerical/retirement qualification for each selected lane; can start on contiguous storage |
| Dynamic continuation (4b/4c) | Paged writes/growth, resident accounting and recomputation progress | Physical retirement, replay history and all-active-pressure proof before partial envelopes |
| Exact prefix reuse (4b reuse) | Valid hybrid restoration with measured net workload benefit | Qualified dynamic storage and bounded checkpoint ownership; not a serving prerequisite |
| Model-local breadth (4e) — alongside baseline | Real dense second decoder through ordinary package loading, then VLM, proving integration and loading boundaries | Relevant operation/backend contracts, not every runtime optimization; not a prerequisite for measuring current Qwen |

Desktop unavailability permits host implementation and CUDA-feature compilation, not
device qualification. Preserve contiguous/full-reachable Qwen and qualified small-row
kernels until their replacements pass. The [CUDA Rust gate-2 candidate](cuda-rust-migration.md#gate-2-candidate-deferred-2026-09-22)
remains deferred: arithmetic/sanitizer results do not offset its single-row projection
and GDN regressions. Do not start gate 3 or build a compiler project around it.

## Decision milestones

Work should produce evidence that changes the next engineering decision, or remove a
concrete prerequisite for that evidence. Fix correctness/security failures on affected
paths; do not turn every unfinished subsystem into a dependency of text measurement.

1. **First competitive baseline.** Establish low-concurrency latency, saturation and SLO
   goodput, heterogeneous arrivals, and capacity/recovery under a tight grant. Start with
   the smallest credible matched suite, not the entire eventual acceptance matrix below.
   Compare identical artifacts where supported; keep independently quality-qualified
   different-artifact comparisons separate. Cross-check client timing/token semantics and
   preserve failed offered work. Protocol validation alone is not a performance result.
2. **First measured improvement.** Identify the dominant loss, select one bounded
   experiment and demonstrate repeatable end-to-end benefit without hiding quality,
   memory, fairness or latency regressions. Scalable prefill, heterogeneous physical
   batching, persistent metadata/graphs, overlap and paging are candidates, not a fixed
   implementation staircase. Compare logical-policy changes with the same physical lanes.
3. **Decoder generality.** Deliver the real second decoder early enough to change the
   boundaries. Measure integration effort and cross-cutting edits as well as numerical
   correctness and loading peaks. A registry entry or host-only parameter mapping does
   not establish executable support. A real VLM follows this test.
4. **Broader qualification.** Select subsequent model/workload and hardware targets from
   observed gaps. Another NVIDIA GPU class tests performance portability; a materially
   different backend tests backend portability. Neither follows from the other.

Choose workload objectives before the baseline; choose the incumbent gap to address,
acceptance criteria and an effort budget before the optimization experiment. None is
numerically selected by this roadmap. Record the
result and whether it warrants promotion, another experiment or redesign in the existing
benchmark evidence. Do not count a completed abstraction as a competitive milestone.

While device access is unavailable, close specific host-testable baseline gaps or prepare
independent references and real integration boundaries for the second decoder. Do not
build a general benchmarking framework, fill time with unrelated repairs, or accumulate
untested GPU kernels. Physical accounting gates apply before enabling their dependent
features; they do not require completing the entire non-AR lifecycle before text timing.

## Completed foundations

| Slice | Result and retained evidence |
| --- | --- |
| 1. Ownership baseline | Positive partial progress, whole-completion validation, runtime-owned discard and retryable retirement. Completion-time blocking removed. [Runtime evidence](../benchmarks/runtime-contract.md) |
| 2. Concurrent application access | One token worker, bounded request/event counts, handles/streams, cancellation and shutdown retry; text preprocessing/decoding and ordered offline batching. Accepted allocation/options retention is now host-qualified; non-AR/physical gaps remain below. [Contract](resource-protocol.md#owned-ar-driver-contract) |
| 3. Encoder preparation | BERT CUDA fixture through `ribn-batch`, result leases/dependencies and post-enqueue retirement gates. Not a complete encoder application lifecycle or hard physical-memory bound. [Qualification](../benchmarks/encoder-qualification.md) |
| 4a. Request-reachable continuation | Prompt plus output reach, not maximum context. Three reference-matching requests fit a 500 MiB authority that cannot hold two context-sized charges. [Evidence](../benchmarks/runtime-contract.md) |
| 4b host preparation | Aggregate accepted ranges, credit refunds, readiness/local rejection and uncertain ownership through the AR engine. Qwen validates/retains a fully reserved backend batch; no physical growth. [Host gate](../benchmarks/runtime-contract.md#ar-pre-submit-preparation-host-gate) |
| 4b host authority | Supplied `BytePool` authorities, atomic hybrid grants and sibling readiness/closure. Production Qwen remains private. [Host gate](../benchmarks/runtime-contract.md#ar-reservation-authority-host-gate) |
| 4b prerequisite | Block-table attention **reads** qualified at `ee689d7`; no paged executor follows from that primitive. [Addressing evidence](../benchmarks/runtime-contract.md#block-table-kv-addressing-2026-10-12-ee689d7) |

Admission notification (`3e4dbec`) and shared batch/AR readiness (`e8e3b48`) are
host-qualified; their device requalification remains pending. Only pending device
completion retains timed polling; idle resource waits arm one-shot notifications.

The duplicate core scheduler/runtime and `ribn local` are removed. The Qwen benchmark
now uses the real AR engine. Migrated linear/GGUF/8–9-row retirement fixtures and the
nine-row AR cancellation case passed device re-execution; see the [current Qwen gate](../benchmarks/runtime-contract.md#mixed-arrival-native-qwen-gate).
Historical timings do not qualify the migrated harness.

## Immediate bounds and lifecycle repairs

Source audits at `093cd28` and `70e5586` found gaps that passing fixtures do not cover:

1. **Text accepted-allocation retention — repaired, host-qualified.**
   `ProcessorLimits::max_retained_bytes` bounds input/options heap capacity separately
   from payload length. Before queue insertion, reject spare capacity, chat containers,
   joint input/options overflow and stop storage that cannot fit the driver envelope.
   No normalization copy/peak is introduced. Validate model-added stops/encoded counts
   before prepared replies. Real-facade regressions failed on the original spare-capacity
   and blocked-worker options defects; host gates now cover these, exact joint bounds,
   runtime stop copies, local errors, permit return and invalid aggregate configuration.
   This does not qualify allocator overhead, arbitrary processor workspace or process RSS.
2. **Non-AR application access.** `BatchRuntime` in `crates/batch/src/lib.rs` bounds counts,
   not submitted input bytes, selects FIFO prefixes and lacks local close/shutdown.
   Executor `Blocked(C)` has no general readiness registration. Add retained-input
   admission, bounded feasible-peer selection/aging, cancellation settlement behind a
   blocked head and local retryable shutdown without closing sibling pools. The BERT
   sequence-limit preparation defect is repaired and [host-qualified](../benchmarks/encoder-qualification.md#host-sequence-limit-repair):
   one immutable preparation binds model geometry to the backend-owned 4096-token
   attention limit, rejecting oversized requests locally before accepting a range.
   Larger model position tables remain valid for shorter requests. Other non-AR
   bounds/readiness/shutdown work above remains open; preserve request identity/result
   ownership. This gates encoder application exposure, not the text-only server. See the [non-AR contract](resource-protocol.md#non-ar-application-lifecycle).
3. **Physical release/accounting.** BERT partial construction and normal result/quarantine
   teardown need charge ownership until safe allocator reuse or backing release. Locked
   cudarc 0.19.9 can lose an allocation pointer if tracking-event creation fails; a drain
   cannot release an unreachable pointer. NVIDIA's deferred-release acknowledgement and
   Qwen's following logical refund must be reconciled before overlapping submissions make
   that an ordinary path. Pinned allocation construction also needs an owner before
   fallible tracking-event creation. BERT's default quarantine/result teardown can refund
   charges without a successful drain; an explicit retry method alone does not make Drop
   safe. Gate partial allocation/upload, teardown, failed drain and sibling-stream
   pressure. No hard shared-device peak claim until these owners qualify.

The 2026-10-15 audit repaired preprocessing insertion racing final-worker exit,
removed the unused charge-dropping `Completed::into_outcome` extraction, validated all
mutable BERT host parameters before CUDA initialization, and added the missing Qwen
quarantine settlement after a successful shutdown drain. Host regressions and bounded
CUDA success-path requalification passed; real failed-drain device injection and the
physical-accounting repairs above remain open. See [audit qualification](../benchmarks/runtime-contract.md#audit-boundary-repairs-2026-10-15).

Loading still needs coherent executable semantics: Qwen geometry/template/weights are
not resolved from one immutable source owner, and same-shaped artifacts with different
normalization/RoPE metadata are not fully validated against the pinned backend. F16
sources materialized as F32 must be charged at materialized size, not encoded size.
Resolve those boundaries before expanding artifact support or claiming loader bounds.
The known long-prompt reference divergence after the eighteen-token agreed prefix is an
active baseline fidelity question, not indefinite conditional work. Classify it with
common-history distributions/state and independent task-quality evidence; do not infer
broad quality from short token equality or widen tolerances to hide disagreement. Timing
within an explicitly qualified scope can proceed; broader performance comparisons need
that scope independently justified.

The remaining defects/limitations do not weaken the owning contracts or block unrelated
qualified paths. The next concrete delivery is matched endpoint measurement below; non-AR and
physical-accounting repairs gate their affected paths. Do not relabel count/payload or
post-enqueue tests as physical-release evidence.

## 4. Dynamic AR, physical execution and model-local integration

### 4b. Block storage, preparation and prefix reuse

Deliver storage/progress separately from cache policy:

1. **Bounded resident backing.** Use a bounded resident K/V arena per required backend
   layout, matching the qualified common-base reader. Charge physical arenas until safe
   retirement; maintain separate block-slot occupancy. Include private recurrent state,
   scratch, table/copy staging and retirement overlap. Qualify paged writes and table
   lifetime across all selected lanes, not just reads. Initial arena capacity is fixed
   within the load-time grant; online growth is deferred until peak/retirement evidence.
2. **Growth preparation.** Integrate physical block grants into the real AR preparation
   transaction. Reserve whole accepted ranges, settle only valid committed growth, and
   retain old continuation plus new growth after partial enqueue. Gate abandoned/invalid
   preparation, positive shortening, waits with healthy-peer progress and local rejection.
3. **Replay ownership and pressure progress.** Retain prompt plus committed generation
   under one bounded sequence-history owner. Keep replay cursor separate from delivered
   progress. Implement completion-safe recomputation preemption, with a protected survivor
   and growth/workspace priority over victim readmission. Show the survivor can finish
   after reclaiming safe victim/cache occupancy; reject individually impossible requests.
   Select output-ready survivors. If delivery credits exhaust, completion-safely preempt
   the stalled survivor, park it on output readiness and reselect a healthy peer; memory
   feasibility alone is not eventual completion. Bound/age victims so fresh arrivals
   cannot starve readmission. Readiness is not a solution to all-active-growth deadlock.
   This gate precedes partial-envelope admission.
4. **Storage qualification with reuse disabled.** Compare real Qwen common histories with
   contiguous versus paged execution, mixed reaches/positions, tails, pressure, cancellation,
   failed enqueue and failed retirement. Demonstrate progress and capacity recovery and
   measure physical peaks, throughput and latency. Preserve the fallback outside scope.
5. **Exact reuse as a separate experiment.** Add snapshot-scoped immutable content blocks
   and selected sparse recurrent checkpoints. Require model-specific retained KV coverage
   and every hybrid component at the restore boundary; copy private recurrence and replay
   a suffix ending at the prompt's logits-producing token. Gate exact hits, partial tails,
   missing components, cancellation during restore and uncertain retirement. Compare reuse
   off/on with equal grants, quality and traffic, including copy/retention peaks and replay
   cost. Enable only where repeated net TTFT/goodput or capacity benefit justifies retention;
   a nonbeneficial policy remains disabled rather than manufacturing a favorable workload.

Open implementation measurements: arena/block geometry, bounded history envelope,
workspace grant, victim ordering and checkpoint budget/placement. The decisions are
bounded resident backing, recomputation first and optional exact reuse—not a universal
continuation framework or cost vector.

Recorded hybrid geometry: 149.6 MiB recurrence per sequence, 16.8 MiB KV at 257-token
reach and 275 MiB KV at 4096. Three private recurrent copies plus one checkpoint exceed
500 MiB before KV. Dense snapshots and blanket short-request concurrency claims are
unjustified. Start with no reuse and a few selected completed boundaries; only compare
more elaborate overlap-aware retention when traces expose useful partial matches.

### 4c. Eviction and preemption

Implement live recomputation with 4b's progress gate, not as optional cleanup after
partial admission. Evict unreferenced cached occupancy first where present; preempt live
state only after completion-safe retirement. Resident arena bytes remain charged.

Gate all-active pressure, a stalled survivor beside healthy victims, preempted cancellation,
survivor completion, aged readmission, replay without duplicated output/usage/sampling,
uncertainty without early reuse and
capacity recovery. Keep cancellation, cache eviction and live preemption distinct.
Host swap/tiered caching follows measured replay/retention costs, not paging by default.

### 4d. Scheduling with real costs

This work starts alongside serving and paging, not after prefix reuse.

1. **Observe actual execution.** Record phase costs, physically batched rows/tokens,
   fallback frequency, host/device gaps, queue age and allocated/resident/occupied bytes.
   Keep instrumentation bounded; kernel timelines establish hypotheses before expensive
   profiler-counter work. Current 13.74 tok/s / zero-goodput smoke evidence does not isolate
   a bottleneck. Historical traces and same-prompt sweeps are not matched endpoint baselines.
   The native benchmark now has an opt-in bounded dispatch snapshot: submitted backend
   rows/tokens, fully enqueued decode lane widths, separate prefill chunks, fallback
   reasons and explicitly host-only submit timing. [Host gates](../benchmarks/runtime-contract.md#native-dispatch-diagnostics-host-gate)
   cover selection/accounting, not actual CUDA counter mapping or overhead. Device
   timelines, queue age, memory attribution and endpoint exposure remain open.
2. **Qualify useful physical lanes.** Compare packed/ragged heterogeneous-capacity work,
   scalable prefill GEMM and tiled attention against qualified small-row kernels and native
   vendor options. Preserve independent nonadjacent histories/completion mapping. Do not
   remove uniform-capacity guards until descriptors/scratch/addressing qualify, or restore
   maximum-context reservation just to batch. Measure conversion/repacking/workspace costs.
3. **Evaluate unified logical policy.** Select bounded prefill/decode/replay frontiers using
   age/priority, output credits, token bounds and aggregate backend feasibility. Protect
   decode latency and prefill/replay progress. Compare against current queues with the same
   physical lanes, then compare lane changes separately. Tune chunk sizes with actual
   TTFT/ITL/E2E objectives, mixed arrivals and pressure; submission counts are not time bounds.
4. **Persistent metadata and bounded overlap.** Separate stable request rows from kernel
   ordering. Start with independent-sequence preparation/execution overlap only after
   per-submission owners and physical release qualify. Gate immutable staging/table versions,
   out-of-order completion, stalled delivery, shutdown and failed-drain recycling. Graph
   capture needs qualified stable storage and measured eager-versus-replay benefit.

Promote policies/variants only over their qualified scope, with repeatable benefit beyond
run variability and no hidden quality, memory, fairness or latency regressions. Retain
losing cases. Dependent work ahead of host commitment and speculation need their own
[contracts](resource-protocol.md#bounded-overlap-and-submission-ownership) and gates.

### 4e. Model-local integration and composition

Integrate a real dense second decoder through ordinary package/SafeTensors loading,
model/processor/backend, central registration and tests alongside baseline work, then a
real VLM/processor. Pin a concrete checkpoint and independent reference before coding;
exercise another configuration/checkpoint without another architectural rewrite. Use
these to replace hard-coded geometry and the
transitional core bridge where appropriate, not to add family branches in lifecycle/routing
or create a universal operator layer. Start after applicable baseline contracts; it need
not wait for prefix-cache or overlap promotion. Pin independent numerical references and
measure integration effort and loading/physical peaks, not just compilation.

A byte-aware streaming loader must bound concurrent reads, decoded weights, retained shard
clones and upload staging before large-model memory claims. Remote resolution and package
processors follow a concrete integration need; a registry alone is not day-zero support.

Then qualify a sequential encoder-decoder and a small iterative model. Actual models decide
staged versus coupled execution. Prove downstream workspace progress under a shared grant;
separate AR/BERT fixtures do not establish it. Non-AR access repairs precede encoder serving.

## 5. Minimal serving, then wider systems

This is the first user-facing milestone, not the final step of slice 4.

The initial loopback adapter is implemented in `ribn-server` with `ribn serve` CUDA
assembly; [serving](serving.md) owns the protocol subset and operational bounds.
Host TCP tests exercise the real tokenizer/facade/driver with device substitution:
completion/chat order and usage, ordinary and trailing-byte disconnect under saturation,
malformed/oversized/unsupported input, message and encoded envelopes, stalled-peer
progress, connection/body deadlines, response and atomic terminal/usage limits,
loading/faulted readiness, metrics and retryable/cancelled-load shutdown.
The trailing-byte disconnect regression failed before its EOF observer was added.
These host gates do not establish GPU serving behavior. The separate short-context
CUDA HTTP gate, actual CLI shutdown and AIPerf/native-client protocol smokes passed;
see [HTTP qualification](../benchmarks/runtime-contract.md#initial-text-http-qualification-2026-10-13).
No matched competitive measurement or physical-peak claim follows. The remaining
work is the representative acceptance matrix below, not another execution-loop adapter.

Deliver a documented text protocol subset over `TextModel`/`TextOwner`, preserving the
existing execution owner. Start with conventional completion/chat streaming semantics;
state supported parameters and explicitly reject unsupported ones. Bound connections,
body parsing, messages/options, preprocessing and streaming buffers; default to local
binding and require an explicit authentication/exposure policy for non-local deployment.
No remote URL/media fetching is needed for this first text path.

Gate streaming/terminal order, disconnect and queue-saturated cancellation, overload,
malformed/oversized requests, stalled clients with healthy-peer progress, readiness,
metrics and retryable shutdown. Health/readiness must distinguish loading, serving,
closing and faulted states. Python access follows the same owned semantics later.

### First competitive acceptance matrix

Use AIPerf plus the incumbent-native cross-check described in [benchmark methodology](../benchmarks/README.md).
Pin revisions, artifacts/tokenizer, numerical policy, device/software, protocol fields,
warmup and objectives. Compare identical artifacts where supported; otherwise report
quality-matched configurations as a separate comparison, never an equal-artifact ratio.

| Workload | Required evidence |
| --- | --- |
| Mixed short/balanced/prefill-heavy/decode-heavy/long-context arrivals and load sweeps | TTFT/ITL/E2E tails, SLO goodput, queue age/fairness, overload/failure/cancellation, host/device cost |
| Tight grant with cancellations, long reaches and stalled consumers | Positive progress, safe retirement, physical peak and recovery; no logical-byte proxy for device residency |
| Cold/warm shared-prefix, multi-turn continuation and branching/tool-delay traces | Actual computed/cached tokens, cache/copy peaks, replay, completed-job latency/throughput per device; client owns orchestration |
| Lane/policy/graph/overlap variants off versus on | Common-history numerical evidence, matched repeats, capacity/quality/SLO costs and qualified fallback |

Structural synthetic traces test load and prefix patterns; content-real cases are required
for speculative acceptance, MoE routing and output quality. Separate steady-state execution
from loading/warmup and network/tokenization from native diagnostics. Current eight-request
smoke traces do not satisfy this matrix. Report unsupported scope and competitor startup
failures explicitly rather than interpreting them as a Ribn win. No SOTA claim until matched
representative results establish it.

Complement open-loop sweeps with qualified closed-loop agent cohorts. Preserve
inter-turn delays, shared versus unique prefixes, output-length skew and
branching while recording which input tokens were actually computed versus
reused. Report both session progress and per-replica interactivity/throughput;
prerecorded future turns do not demonstrate agent task success. See
[agent-session methodology](../benchmarks/README.md#agent-session-workload-qualification).
This expands evidence, not the dependencies for the first matched endpoint baseline.

A materially different backend, real collectives/sharding and disaggregated transfer have
independent gates. External systems own fleet allocation. Do not make single-device use
traverse serialization or distributed-worker layers.

### Conditional multi-replica placement

This is **not** on the current serving, paging or single-device optimization
critical path. Revisit once matched single-replica measurements and a real
multi-replica demand expose routing/cache losses. External infrastructure owns
resource leases and fleet scaling; Ribn's optional inference-aware deployment
layer can own model-specific placement.

1. Establish backend/replica-local cache validity, exact snapshot/tenant isolation,
   bounded load/cache telemetry, fault and stale-report behavior. Advisory client
   session IDs never authorize reuse; actual valid prefix coverage decides hits.
2. On fixed multi-replica hardware compare random, load-only, fixed affinity and
   cache-plus-load policies under identical closed-loop and open-loop traces.
   Include long/short context skew, hot/concurrently branched sessions, cache
   pressure, cold replicas, scale-out and restart. Distinguish recomputation
   avoidance from queue-induced tail regression and relocation cost.
3. Qualify placement by p95/p99 TTFT/ITL/E2E, session and replica fairness,
   completed jobs/device-time, memory/transfer overhead, overload recovery and
   consistent correctness on cache misses/evictions/restarts. Keep stale metadata
   advisory, not authoritative for continuation. Show repeated net gain beyond
   run variability and no extra local-path dependency before promotion.

No router API, multi-node KV transfer, global cache index or general scheduler
framework is selected before this evidence. Consult the
[inference placement boundary](inference-engine-design.md#multi-replica-inference-placement-conditional).

## 6. Training integration, then execution

Start with coherent trainer-to-rollout publication and Python tensor/result interchange.
Drain-and-replace precedes overlapping snapshots; overlap requires explicit extra capacity
and cache compatibility. Never expose optimizer-mutating weights to inference.

A reference-backed forward/backward/update experiment decides useful shared mathematics.
Training owns gradients, activations, autograd and optimizer scheduling; checkpoint recovery
and distributed training need separate qualification. Training is not an AR scheduling mode.

## Conditional work and known non-promotions

- **GDN scan:** opt-in while full-model error `1.10e-2` exceeds `5.0e-3`. Compare captured
  common-history state against higher precision; no tolerance widening from baseline error.
- **Decode subgroups:** bounded device cases passed, but matched mixed traces showed no
  benefit and zero SLO goodput. Promotion also needs failed-drain evidence: no uncertain
  pinned-slot reuse and exactly-once recycling after proven drain. See [qualification](../benchmarks/runtime-contract.md#compatible-decode-subgroup-candidate).
- **Kernel losers:** do not repeat four rows per warp, shared IQ4 codebook or pre-elimination
  decayed keys without new evidence. Keep qualified small-M fallback while scaling work runs.
- **Approximate hybrid replay:** recent suffix-replay research changes numerical/quality
  semantics. Outside initial production work; any experiment needs explicit policy/cache
  identity and independent task-level acceptance, not an exact-cache label.
- **Speculation:** select a concrete draft/verify backend only after target commitment,
  hybrid rollback, output/usage and sampling-state contracts qualify. Compare
  target-conditioned, learned and simpler drafters on content-real cohorts before
  selecting a training strategy. Measure whole-round net goodput with weights,
  intermediate/state/scratch charged; allow drafting to be disabled. Exact token-ID
  training traces require explicit content authority and snapshot/processor provenance.
- **Lossy state and host tiers:** try FP8/other supported continuation encodings only
  after exact-cache and physical accounting gates, with independently qualified
  full-model quality. CPU/remote cache tiers are optional, evaluated against
  recompute/retention for tail benefit under pressure including transfer/retirement
  overhead; no universal cache hierarchy is required.
- **Text access:** thinking controls, thread-affine construction and offline-window throughput
  remain conditional. They do not justify replacing the owned facade or raising buffers
  without boundedness/progress evidence.
- **Research/reference backend:** evaluate when a concrete integration needs it, not as a
  parallel universal implementation backlog. Research sources/limits live in the
  [target design](inference-engine-design.md#research-basis-reviewed-2026-10-13).

## Required checks

```sh
python3 tools/check-boundaries.py
cargo fmt --all -- --check
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo clippy --workspace --all-targets --locked --features cuda -- -D warnings
```

Capture actual exit codes. CUDA compilation does not qualify device execution. Follow
[model integration](../.agents/skills/model-integration/SKILL.md) for numerical gates and
[execution contracts](../.agents/skills/execution-contracts/SKILL.md) for lifecycle changes.
Keep unrun gates explicit; documentation redesign does not complete implementation gates.
