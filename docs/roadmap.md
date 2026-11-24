# Engineering roadmap

[Target design](inference-engine-design.md) owns product direction and architecture;
[resource protocol](resource-protocol.md) owns execution/resource ownership;
[architecture](architecture.md) maps current code. This document owns delivery order,
open implementation choices and acceptance gates. All v0 interfaces may change.

## Current execution order

**2026-11-05 decision: redesign the optimized GPU backend as model-wide
physical execution while retaining the Rust AR serving authority.**
The [model-wide design](inference-engine-design.md#model-wide-physical-execution-selected-2026-11-05)
and [physical resource protocol](resource-protocol.md#model-wide-physical-preparation-and-result-mapping)
are the implementation contracts. The depth-one Gemma CUDA owner now passes its
model-wide device, exact-row and fresh six-process serving gates. Ordinary CUDA
loading selects it within the qualified RTX 4090/E4B/prepared-4K scope; other
configurations retain serial execution. Competitive acceptance remains open.

The serial fallback in Gemma's `executor/batch.rs` executes each offered
sequence with a synchronous decoder call. Its CUDA sixteen/eight-row
prefill groups tokens from *one* request and still reads PLE to host,
uploads contexts each layer and reads full logits to host. The promoted
loopback c4/c8 cohort reached roughly 41 output tokens/s versus fresh
SGLang 281–291; the models were **not independently quality-equivalent**.
In the [Qwen equal-artifact follow-up](../benchmarks/runtime-contract.md#qwen-equal-artifact-diagnostic-without-http-slot-rejection-2026-11-21),
both engines complete 32/32 after the HTTP slot cap increases. Ribn still
misses the joint latency objective for eleven burst requests versus none
for llama.cpp, and first content is slower even outside the burst. Trace
short-prompt prefill on the current device path and resolve output
quality differences before choosing a kernel or scheduler change.
Neither workload establishes a clean competitive ranking.
[Gemma evidence](../benchmarks/gemma4-cuda-baseline.md).

### Active deliveries

1. **Matched quality/serving baseline, in parallel.** Preserve all offered,
   admitted/rejected, cancelled and completed arrivals with actual usage
   and terminal semantics. Match the admission envelope, artifact and
   quality policy; measure TTFT, ITL, E2E, SLO goodput, fairness, pressure,
   physical memory and recovery. Repair benchmark accounting rather
   than ranking only successful request subsets. Use the current single-turn
   runner for this first baseline; causal closed-loop agent replay and
   job-level metrics need separate client qualification, not an engine
   orchestrator or a new prerequisite for the first comparison. The
   [2026-11-11 frozen HTTP QA](../benchmarks/gemma4-cuda-baseline.md#frozen-corpus-http-qa-2026-11-11)
   completes 128/128 at c1/c4 for both engines, but retains two native lexical
   losses against SGLang. This is not matched-quality competitive acceptance;
   preserve those cases and the fixed scoring policy for the model-wide path.
2. **Competitive packed device execution.** Qualify multi-request
   decode/prefill and compatible attention/state lanes. Measure
   physical rows/tokens, fallback rates, GPU work, transfers, host
   gaps, memory and actual HTTP goodput. Then choose indexed-row
   device PLE, sampled-token readout, capture or fusion from measured
   losses, not the launch count. Do not repeat unchanged negative
   M4, descriptors, PLE-only, attention and FFN graph candidates.
   On Metal, the single-command-buffer model-wide engine (`Gemma4MetalModelWide`
   + `Gemma4MetalModelWideExecutor`) is now the default `Native` loader path for
   `head_dim <= 512` with 16-token paged KV (16K context, 2,048 shared blocks),
   exact content-addressed radix prefix caching, mixed-precision BF16 MMA prefill
   (`512`-token chunking), exact shared-KV tail pruning, fused post-norm/residual
   kernels, parallel softcap/argmax, capability-gated HTTP stochastic sampling,
   and opt-in affine group-64 weight quantization (`--quantize bf16|q8|q4`,
   default `bf16`). On M3 Max E4B, BF16 matches 12/12 archived CPU BF16 oracle
   winners (`2,114 tok/s` direct 512-token prefill, `364.3 tok/s` `M=16` decode)
   and beats `mlx-lm==0.32.2` across all 10 tested BF16 HTTP SSE cohorts on TTFT
   and batched throughput; opt-in `q8` (`7.39 GiB` weights / `9.58 GiB` RSS) and
   `q4` (`3.91 GiB` weights / `6.11 GiB` RSS) also match 12/12 BF16 oracle
   winners with bit-exact 512-token prefix caching (`11.9×`/`16.9×`) and reach
   `431.9` / `459.1 tok/s` live HTTP SSE throughput at `c=16` (`466.2` / `492.5`
   tok/s direct; see
   [Metal baseline](../benchmarks/gemma4-metal-baseline.md#affine-group-64-metal-quantization-q8--q4-fused-norms-and-parallel-softcap-2026-11-24)).
   Formal multi-process matched-quality external serving cohorts, quantized task
   QA, pre-quantized checkpoint loading, memory-pressure recovery under
   constrained RAM, and CUDA/4090 packed-batch parity remain open.
3. **Native versus externally prepared executor, concurrently.**
   Test a pinned Gemma model and at least two contexts/shapes through
   a truly Rust-invocable prepared program; compare full numerical/
   task results, mixed-request goodput, cold compile/load, memory,
   state/retirement fidelity and integration/maintenance effort.
   Stock PyTorch is a reference, not the native production candidate.
   Choose the superior qualified method without prematurely building
   a universal IR/compiler.
4. **Capacity and safe overlap.** Close deferred-free accounting,
   qualify physical resident grants, recomputation progress and
   stalled-output recovery, then overlap independent submissions
   with immutable per-submission storage and real fences. Exact hybrid
   prefix reuse remains a separate workload-dependent experiment.
5. **Broader architecture/hardware.** Inkling 2 Mini is the next new text
   port, alongside continued Qwen and Gemma qualification and optimization.
   The fixed-loop checkpoint in the pinned [reference replay](../benchmarks/inkling-reference/README.md)
   is a starting point, not a model-support boundary. Qualify continuation across
   loop passes and the sliding-window boundary before exposing support. Continue Gemma VLM's failed
   full-checkpoint numerical and feature-handoff gates, then test
   a different architecture and backend. Bind encoder features to processor/model/input
   identity and exact prompt layout, with bounded structural metadata and producer
   completion independent of sampled tokens. Gate mismatched grids/lengths/placeholders,
   delayed production and cancellation before consumer access. Track support velocity
   and ongoing maintenance, not just model registry entries.

### Model-wide CUDA cutover (complete for the qualified scope)

Ordinary loading selects the depth-one model-wide executor only for the qualified
RTX 4090/E4B/prepared-4K scope. The model-local scope check is independent of the
selected scratch lanes. Outside that scope, keep serial execution; a failure after
selecting the owning loader propagates without a second upload or serial retry.

The [device gate](../benchmarks/runtime-contract.md#model-wide-gemma-cuda-device-gate-2026-11-11)
proves cross-request weight sharing, separate KV/window routing and whole-offer
ownership, including reordered continuation at the 4K boundary. The
[ragged prefill gate](../benchmarks/runtime-contract.md#genuine-ragged-prefill-qualification-2026-11-12)
qualifies widths 2–16. Only the
[fresh six-process serving gate](../benchmarks/gemma4-cuda-baseline.md#genuine-ragged-prefill-serving-qualification-2026-11-12)
supports promotion; previous negative populations stay closed. Keep the serial
8/16/scalar serving control distinct from the all-width numerical reference.
Full-model failed-driver-drain remains composed native/owner evidence, not a directly
injected model-wide fault. Internal promotion does not establish external parity.

For later capture/replay or restore, delay the final cache/feature load and late
metadata reads, compare common-history results against eager, and test cancellation
and failed drain with dependencies pending. A passing undelayed replay does not
prove ordering; an early shared-read fence must not release model state.

### Decision milestones

If multi-sequence work still launches serial model forwards, fix
physical lanes before retuning scheduler policy. If executable
external preparation passes full state/lifetime/quality requirements
and wins overall serving *and* integration economics, adopt it for
its qualified scope; otherwise use model-wide native execution.
Select kernels/layout for measured device losses, graph/overlap for
actual launch gaps, and compatible packing for excessive fallback.
No SOTA claim without matched admission, quality, SLO goodput,
representative traffic and physical pressure recovery per the
[competitive matrix](#first-competitive-acceptance-matrix).

## Completed foundations

| Slice | Result and retained evidence |
| --- | --- |
| 1. Ownership baseline | Positive partial progress, whole-completion validation, runtime-owned discard and retryable retirement. Completion-time blocking removed. [Runtime evidence](../benchmarks/runtime-contract.md) |
| 2. Concurrent application access | One token worker, bounded request/event counts, handles/streams, cancellation and shutdown retry; text preprocessing/decoding and ordered offline batching. Accepted allocation/options retention is now host-qualified; non-AR/physical gaps remain below. [Contract](resource-protocol.md#owned-ar-driver-contract) |
| 3. Encoder preparation | BERT CUDA fixture through `ribn-batch`, result leases/dependencies and post-enqueue retirement gates. Not a complete encoder application lifecycle or hard physical-memory bound. [Qualification](../benchmarks/encoder-qualification.md) |
| 4a. Request-reachable continuation | Prompt plus output reach, not maximum context. Three reference-matching requests fit a 500 MiB authority that cannot hold two context-sized charges. [Evidence](../benchmarks/runtime-contract.md) |
| 4b host preparation | Aggregate accepted ranges, credit refunds, readiness/local rejection and uncertain ownership through the AR engine. Qwen validates/retains a fully reserved backend batch; no physical growth. [Host gate](../benchmarks/runtime-contract.md#ar-pre-submit-preparation-host-gate) |
| 4b host authority | Supplied `BytePool` authorities, atomic hybrid grants and sibling readiness/closure. Production Qwen remains private. [Host gate](../benchmarks/runtime-contract.md#ar-reservation-authority-host-gate) |
| 4b page addressing | Host continuation is page-granular and production Qwen decode appends and attends through a device page table (`e82f58d`, `e36e097`). Tables are identity-bound from a per-sequence run, so a shared arena, non-identity pages and page growth remain. [Addressing evidence](../benchmarks/runtime-contract.md#page-table-continuation-e36e097) |

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
   Decoded-text stop containers/string capacities now join that same pre-queue envelope.
   Their matcher and codec share the delivery grant; a matched stop cannot decode or
   finalize a discarded suffix, and released stop prefixes obey the delta bound.
   [Stop-boundary regressions](../benchmarks/runtime-contract.md#text-stop-ownership-repair-2026-10-22)
   exercise these gaps on the actual facade with GGUF/Gemma processors.
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
   cannot release an unreachable pointer. Cudarc 0.19.10 includes a pinned-allocation
   error cleanup change. Qualify the patch against the locked 0.19.9 path, including
   failed construction and retirement, before editing the lockfile. NVIDIA's deferred-release acknowledgement and
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

Qwen now normalizes model-owned numerical metadata and rejects mismatches against the
backend's declared execution profile before tokenizer/device access. `QwenCudaProfile`
owns that geometry and its numerical bits as data, and a mirror test fails if it drifts
from the compiled kernel constants; a second executable geometry still needs its own
kernel specialization. [Host preflight evidence](../benchmarks/runtime-contract.md#qwen-model-metadata-preflight-host-gate)
covers normalization/RoPE and same-width head decompositions. The actual pinned header
and affected Qwen/HTTP/BERT device gates [passed at `dc4aed0`](../benchmarks/runtime-contract.md#device-re-execution-2026-10-16-dc4aed0),
within their existing bounded reference scope—not broad-context qualification.
Qwen preparation now binds geometry, tokenizer/template, complete-artifact identity and
weight streams to one pinned GGUF source, and returns the tokenizer with its prepared
executor instead of reopening the text frontend's pathname. Host regressions cover
replacement/unlink, retained readers and independent cursors; the path-replacement test
fails on the old implementation. The selected file must remain unmodified in place
through every alias while artifact owners/readers live: this is descriptor retention,
not a copied or filesystem-enforced immutable snapshot. The staging budget now charges
every F32-materialized source at its four-bytes-per-element device footprint
instead of its encoded length, with host cases covering the mixed set, quantization and
overflow; host arithmetic is not device evidence, so no loader byte bound is claimed yet.
Device loading/retirement qualification and physical byte bounds remain separate gates.
The known long-prompt reference divergence after the eighteen-token agreed prefix is an
active baseline fidelity question, not indefinite conditional work. Classify it with
common-history distributions/state and independent task-quality evidence; do not infer
broad quality from short token equality or widen tolerances to hide disagreement. Timing
within an explicitly qualified scope can proceed; broader performance comparisons need
that scope independently justified.

Gemma executor preparation now validates the whole offered batch before decoder mutation,
retains the exact offer through submission and checks completion identity. Malformed later
rows no longer follow earlier continuation execution; logical state commits only after
all rows succeed. The initial blanket quarantine in `7abb830` incorrectly treated every
execution failure as unretired device access, permanently retaining even completed CPU
work. That policy and its retention assertions are replaced by backend-owned teardown:
execution stays faulted, but CPU and terminal-Metal resources can be destroyed. Failed
completion or sequence close retains the owner and its charges for another attempt;
defensive Drop retains the complete envelope only if cleanup fails or unwinds.
[Executor transaction and teardown qualification](../benchmarks/runtime-contract.md#gemma-executor-transaction-and-teardown-2026-10-22)
distinguishes the reproduced leak, internal-contract hardening and injected cleanup cases.

CUDA retirement now drains the associated stream, including transfer-only failure, retires
retained compute operations and explicitly closes sequence caches. Partial cache construction
is model-owned before allocating keys/values, so failed construction cleanup remains reachable
even without an admitted decoder; only one such bundle can remain pending. Retirement does
not reopen a poisoned model. The [CUDA device gate](../benchmarks/runtime-contract.md#cuda-device-requalification-2026-10-22)
now exercises partial cache construction under a restricted pool budget, idempotent cache
close and a real driver synchronization refusal during thread-local stream capture. Charges
remain held until capture ends and shutdown retry succeeds; execution stays closed. Captured
work is not executed: this is not running-kernel failure, driver OOM or failed-free evidence.
Uncertain free may correctly require permanent quarantine rather than an unsafe second free.
This is safe teardown plumbing, not an in-process device/model recovery framework.

The remaining defects/limitations do not weaken the owning contracts or block unrelated
qualified paths. The first bounded execution improvement is promoted; cross-engine
numerical qualification and remaining prefill/projection investigation remain open. Non-AR
and physical-accounting repairs gate their affected paths. Do not relabel count/payload
or post-enqueue tests as physical-release evidence.

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
   missing components, cancellation during restore and uncertain retirement. Size backing
   for live state plus all retained checkpoint copies and restore/copy peaks; test
   coincident boundaries, multiple retained turns and a budget that fits live sequences
   but not their checkpoints. Free checkpoint slots do not refund resident backing. Compare reuse
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

**Make physical batches real before rewriting scheduling policy.**
The AR engine offers aggregate work. The qualified Gemma CUDA executor
shares physical model work; the fallback still serializes sequences.
Keep backend-owned physical lanes with exact sequence/state mapping and
whole-offer resource grants, and preserve the CPU/single-row oracles.

Compare physically packed/ragged decode, prefill and separated mixed
phases under identical logical offers. Track true batched rows/tokens,
fallback, transfer and kernel/host costs, pressure, queue age and
latency-objective goodput. Preserve continuation state and failure retirement.
Use the [numerical contract](inference-engine-design.md#qualification-and-current-evidence)
to distinguish arithmetic-preserving changes from reordered execution.
Do not pad to maximum context merely to batch.

Only once physical costs stabilize compare chunked prefill, decode
protection, replay aging and measured-time versus token budgets
on identical lanes. Depth-two initially overlaps independent
sequences with separate immutable staging, metadata, workspace,
outputs and completion owners. Same-sequence work-ahead, graphs
and speculation need their own qualified state dependencies.

### 4e. Model-local integration and composition

Gemma 4 E4B text has an experimental CPU/Metal/CUDA greedy decoder.
The scoped CUDA promotion does not establish competitive concurrent execution.
Keep its model-local processor/configuration,
independent CPU oracle and qualified fallbacks. Retain historical
experiment data in [Gemma CUDA](../benchmarks/gemma4-cuda-baseline.md),
[Metal](../benchmarks/gemma4-metal-baseline.md) and
[model reference](../benchmarks/gemma4-reference/README.md).

Native and externally prepared physical executors must both preserve
the same Rust owner for admission, mutable state, submission, outputs,
cancellation and fallible device retirement. Pin compiler/artifact,
dynamic shapes, numerical scope and cold startup. Judge compiled
programs on quality, serving economics, integration speed and
maintenance, not just stock Python forward timing. Keep model
semantics/model-specific kernels local; no universal graph ABI.

The Gemma VLM reference and CPU vision code are not serving support:
the strict full-checkpoint numerical gate remains unresolved,
including normalization differences. Qualify numerical policy,
feature lifetime, media processor, shared workspace and complete
handoff before multimodal serving. The specialized BERT batch
runtime retains separate input-byte, readiness and shutdown gates.

## 5. Minimal serving, then wider systems

The bounded loopback HTTP adapter, text facade, shared AR owner,
cancellation and shutdown paths already exist and have host/
bounded CUDA qualification. [Serving](serving.md) owns the current
protocol and limits. Broader exposure, sampling/model breadth and
competitive SLO results remain unqualified. Do not add a second
frontend scheduler; instead measure real model-wide execution
through the existing text/HTTP owner alongside matched quality/
pressure baselines. Native Python bindings must reuse that same
execution lifecycle. Non-local use needs a separate security policy.

### First competitive acceptance matrix

Use the [audited endpoint runner](../benchmarks/README.md#audited-endpoint-runner)
and incumbent-native cross-checks.
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
- **Synchronous CUDA PLE:** replacing the projection readback and per-layer uploads with
  device combines [failed its predeclared 3% prefill gate](../benchmarks/gemma4-cuda-baseline.md#synchronous-device-ple-experiment-2026-10-22).
  The candidate preserved all 12 captured full rows, but 64-token prefill improved only
  0.16% and decode was 0.29% slower. All candidate code was reverted; this does not qualify
  a device-resident variant or isolate the original round trip's standalone cost.
- **Metal intermediate-prefill heads:** eliding discarded heads preserved all captured
  logits but [failed its 3% runtime prefill gate](../benchmarks/gemma4-metal-baseline.md#intermediate-prefill-output-head-elision-2026-10-22):
  64/129-token TTFT improved only 1.82%/0.94%. Candidate APIs, code and test changes were
  reverted. This does not measure CUDA head cost or justify changing the chunk policy.
- **Kernel losers:** do not repeat four rows per warp, shared IQ4 codebook or pre-elimination
  decayed keys without new evidence. Keep qualified small-M fallback while scaling work runs.
- **Approximate hybrid replay:** recent suffix-replay research changes numerical/quality
  semantics. Outside initial production work; any experiment needs explicit policy/cache
  identity and independent task-level acceptance, not an exact-cache label.
- **Speculation:** select a concrete draft/verify backend only after target commitment,
  hybrid rollback, output/usage and sampling-state contracts qualify. Compare
  target-conditioned, learned and simpler drafters on content-real cohorts before
  selecting a training strategy. Measure whole-round net goodput with weights,
  intermediate/state/scratch charged; allow drafting to be disabled. Once draft/verify
  exists, compare fixed-length, adaptive-length and confidence-ranked GPU allocation
  within a CPU-reserved verification budget, including confidence calibration, accepted
  tokens, total work, latency and SLO goodput. This is a later candidate, not an immediate
  milestone or evidence of distribution-preserving sampling. Exact token-ID
  training traces require explicit content authority and snapshot/processor provenance.
- **Lossy state and host tiers:** try FP8/other supported continuation encodings only
  after exact-cache and physical accounting gates, with independently qualified
  full-model quality. CPU/remote cache tiers are optional, evaluated against
  recompute/retention for tail benefit under pressure including transfer/retirement
  overhead; no universal cache hierarchy is required.
- **Reproducibility:** qualify optional batch-invariant execution independently of ordinary
  numerical correctness; keyed reproducible sampling additionally needs a usable sampling
  path. Specify supported batch shapes/orderings and speculation settings and report cost.
- **Text access:** thinking controls, thread-affine construction and offline-window throughput
  remain conditional. For richer tool/protocol operations, qualify structural token
  provenance versus literal markers and incremental parsing across token, UTF-8 and JSON
  argument boundaries. Keep this in model-owned processors above token scheduling; Recipe
  is a reference, not a processor replacement. These features do not justify replacing the
  owned facade or raising buffers without boundedness/progress evidence.
- **Research/reference backend:** evaluate when a concrete integration needs it, not as a
  parallel universal implementation backlog. Research sources/limits live in the
  [target design](inference-engine-design.md#research-basis).

## Required checks

Use the [maintainer verification guide](README.md#verification) and the affected
qualification gates above.
