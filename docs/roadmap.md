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
Qwen's 2026-11-04 equal-artifact diagnostic completed 23/32 requests
versus llama.cpp's 32/32 under differing admission policies. Neither
establishes a clean competitive ranking.
[Gemma evidence](../benchmarks/gemma4-cuda-baseline.md);
[Qwen evidence](../benchmarks/runtime-contract.md#current-source-equal-artifact-endpoint-diagnostic-2026-11-04).

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
2. **Model-wide Gemma CUDA execution.** Replace optimized per-sequence
   `GemmaDecode` dispatch with a shared model/backend prepared batch.
   Bind stable sequence incarnations, physical lane membership, exact
   prefixes, caches, accepted ranges and full resource grants.
   Start depth one; retain CPU/qualified CUDA reference fallbacks.
   Gate malformed later rows, mixed reaches, row reorder, state routing,
   cancellation, partial enqueue and failure retirement.
   **Exit:** at least two sequences actually share physical GPU
   weight/model work. A renamed serial loop does not pass.
3. **Competitive packed device execution.** Qualify multi-request
   decode/prefill and compatible attention/state lanes. Measure
   physical rows/tokens, fallback rates, GPU work, transfers, host
   gaps, memory and actual HTTP goodput. Then choose indexed-row
   device PLE, sampled-token readout, capture or fusion from measured
   losses, not the launch count. Do not repeat unchanged negative
   M4, descriptors, PLE-only, attention and FFN graph candidates.
4. **Native versus externally prepared executor, concurrently.**
   Test a pinned Gemma model and at least two contexts/shapes through
   a truly Rust-invocable prepared program; compare full numerical/
   task results, mixed-request goodput, cold compile/load, memory,
   state/retirement fidelity and integration/maintenance effort.
   Stock PyTorch is a reference, not the native production candidate.
   Choose the superior qualified method without prematurely building
   a universal IR/compiler.
5. **Capacity and safe overlap.** Close deferred-free accounting,
   qualify physical resident grants, recomputation progress and
   stalled-output recovery, then overlap independent submissions
   with immutable per-submission storage and real fences. Exact hybrid
   prefix reuse remains a separate workload-dependent experiment.
6. **Broader architecture/hardware.** Complete Gemma VLM's failed
   full-checkpoint numerical and feature-handoff gates, then test
   a different architecture and backend. Bind encoder features to processor/model/input
   identity and exact prompt layout, with bounded structural metadata and producer
   completion independent of sampled tokens. Gate mismatched grids/lengths/placeholders,
   delayed production and cancellation before consumer access. Track support velocity
   and ongoing maintenance, not just model registry entries.

### First implementation cutover

Keep the change under the optimized Gemma CUDA backend; do not rewrite
the common AR scheduler or the CPU oracle.

1. In `crates/gemma4/src/executor.rs` and `executor/backend.rs`,
   retain synchronous `Gemma4Executor<D: GemmaDecode>` as CPU/scalar
   reference and fallback. Add a distinct model-wide CUDA executor
   implementing the existing `GenerationExecutor`; it owns shared
   weights, sequence physical state and prepared submissions. Share
   pure validation, not a per-request forward interface.
2. In `crates/gemma4/src/cuda/decoder.rs` and `decoder/buffers.rs`,
   move mutable cache handles and positions under the model-wide
   owner and separate them from submission workspace. Stage packed
   rows with original sequence mapping. Initially prove actual
   multi-sequence matrix work with correct per-member KV/attention;
   keep the qualified 8/16-token single-request and scalar fallbacks.
3. In `crates/text/src/gemma_load.rs`, change only the CUDA
   constructor behind the existing `Engine`, model processor and
   owned `TextModel`. Preserve CPU/Metal loading and existing
   run/serve behavior; do not add a frontend scheduler.
4. Preserve the depth-one scheduling and completion model in
   `crates/runtime/src/executor.rs` and `engine.rs`. Keep the
   repaired admission failure boundary: clean refusal is `Rejected`,
   while uncertain partial construction faults the execution owner
   and retains its retirement envelope. This is not a scheduler rewrite.
   A physical lane may reorder work,
   but the current `poll` contract returns completions in
   originally submitted row order. Any later depth-two extension
   needs its own qualified runtime/resource change.

Host prerequisites now separate decoder-independent logical metadata and accepted-row
binding in `gemma4/src/generation.rs` from serial physical execution. The ordinary Gemma
loader accepts `GenerationExecutor`; the initial refactor did not change backend
selection. Admission regressions cover clean refusal, fatal partial construction,
stopped later admission and
retained peer/storage charges through failed drain and successful retry. The
[2026-11-11 CUDA recheck](../benchmarks/runtime-contract.md#current-source-cuda-resource-recheck-2026-11-11)
passes six existing decoder/executor/pressure cases, including real partial-cache
retirement and failed execution-drain retry. Uncertain partial-construction cleanup
before any decoder exists remains host-injected, not device-qualified. None of these
prerequisites establishes multi-request GPU weight work.

The [native command-stream prerequisite](../benchmarks/runtime-contract.md#native-bf16-command-stream-foundation-2026-11-11)
now passes its component device gate: a privately owned, uncapturable stream domain,
bounded dependent BF16 operations, nonblocking event query and explicit retirement.
Its two-row graph shares projections while preserving unequal member KV histories;
memcheck/initcheck report zero errors. Existing blocking primitives and Gemma gates
also pass after the dispatch refactor. The subsequent
[model-wide device gate](../benchmarks/runtime-contract.md#model-wide-gemma-cuda-device-gate-2026-11-11)
now establishes actual two-sequence shared projections through all 42 layers, with a
separate physical owner and `GenerationExecutor`, independent KV/window routing,
whole-offer commitment, deferred cancellation retirement and partial construction/
enqueue ownership. Common-history archived winners, unchanged prefill bits, M2
projection bounds and full-model memcheck/initcheck pass. Two-pair offers also
acknowledge workspace reuse before whole-offer commitment. This meets the first
physical-sharing proof, not loader promotion or competitive serving acceptance.
The same gate now also passes the service's 4K boundary: unchanged full prefill
rows, reordered paired continuation to positions 4096/4095, request-reachable overrun
rejection and complete retirement. Full-case memcheck/initcheck report zero errors.
Those long synthetic histories use a fresh qualified native control, not publisher
labels. The subsequent [fresh HTTP QA screen](../benchmarks/gemma4-cuda-baseline.md#model-wide-http-qa-2026-11-11)
passes all 128 questions at c1/c4 for both native control and model-wide candidate:
full answers, usage and per-item EM/F1 are unchanged. This permits only private
performance measurement; the earlier native lexical gap against SGLang remains.
The subsequent [private mixed-arrival screen](../benchmarks/gemma4-cuda-baseline.md#model-wide-mixed-arrival-screen-2026-11-11)
is closed after one matched pair and the second candidate: one mixed8 answer chooses
a shorter approved gold alias, leaving EM/F1 unchanged but failing the frozen
same-output/work condition. The other three model processes are unrun; no paired
median or qualified performance result is available. Do not extend or relax that population.
Controlled common-history scalar/M2 mixtures reproduce the continuation/EOS BF16
score tie, with bit-exact scalar control and complete retirement. This is arithmetic-
path sensitivity, not an observed score loss; the timed dispatch mask was not traced.
At that stopped population, ordinary loading remained serial. The
[owning numerical contract](inference-engine-design.md#qualification-and-current-evidence)
already makes batch invariance optional, not a compulsory fast path. Do not invent
that requirement to repair the stopped benchmark. A separate prospective serving
comparison may accept different EOS outcomes only with unchanged per-question EM/F1
floors, explicit actual-work accounting and repeated token/request-rate, latency and
residency evidence; it cannot turn the closed same-output population into a pass.
That separate [six-process serving screen](../benchmarks/gemma4-cuda-baseline.md#quality-preserving-model-wide-serving-screen-2026-11-12)
is now complete and **negative**: all quality/integrity checks and median rate/p95
gates pass, but mixed8 joint diagnostic deadline counts fall from 4/3/3 to 2/2/2.
Keep it closed. The repeatable drain-tail cohort differs under closed-loop feedback;
these observations do not establish decode-priority starvation. The executor now
moves blocking physical advancement out of submission and into polling, avoiding
next-offer work in the previous offer's delivery path. The failing-before GPU
regression and existing ownership/failure cases pass. Its new source-specific
six-process comparison also closes **negative**, with unchanged criteria: quality,
integrity and all median rate/p95 gates pass, but mixed8 joint counts are
4 → 3, 4 → 4 and 3 → 4. The prior drain-tail question passes in every new pair;
one initial-burst question misses first content in pair 0 only. A separate single
host trace captures the fast phenotype with prompt post-settlement publication;
its next output waits for an expensive fifteen-token fallback in the same offer.
An existing-path common-history diagnostic confirms the large scalar-tail cost.
The [genuine ragged-prefill prerequisite](../benchmarks/runtime-contract.md#genuine-ragged-prefill-qualification-2026-11-12)
now passes for exact model-wide widths 2–16: independent native and captured-real
numerical checks, all-width continuation/4K boundaries, resource retirement and all
16 memcheck/initcheck requirements. The prospective ordinary-loader lifecycle also
passes. Preserve serial 8/16/scalar as the serving control; its all-width numerical
reference sampler is not a new control. The separate [fresh six-process serving
qualification](../benchmarks/gemma4-cuda-baseline.md#genuine-ragged-prefill-serving-qualification-2026-11-12)
now passes the original per-item quality, actual-work rate, p95 and every-pair joint
latency gates. Ordinary loading selects model-wide execution within the qualified
RTX 4090/E4B/prepared-4K scope, retaining serial execution outside it. Earlier closed
negative populations remain negative; no criterion or whole-offer guarantee changed.
The next competitive gate is matched external-baseline mixed-arrival serving evidence,
not a claim that this internal cutover establishes vLLM/SGLang parity.
The model-local `matches_qualification_scope` query now checks normalized topology,
rotary and vocabulary scope, device, projection geometry and prepared context,
independently of scratch-selected lanes. Host near-miss tests protect that boundary
without narrowing existing serial prefill selection. The ordinary loader uses the
query before construction; selected owning load failures propagate without a serial
retry or second weight upload. The [captured-real-M2 arithmetic and public-entry-point gates](../benchmarks/runtime-contract.md#model-wide-gemma-cuda-device-gate-2026-11-11)
now pass on device: 11 actual non-head geometries satisfy the preregistered per-output
bound and range checks; the ordinary loader lifecycle passes a private prospective
wide-selection patch with deferred submission. Those earlier prerequisites alone did
not enable automatic loading; the later ragged and fresh serving gates establish the
scoped cutover above. Full-model failed-driver-drain remains composed native/owner
evidence rather than a directly injected model-wide fault. Preserve that scope distinction; do not expose the
qualified serial model's raw stream to force adoption or fault injection.

Host acceptance: whole-offer prevalidation, exact range/cursor
and row mapping, state-only intermediate prefill, fallback,
cancellation, partial enqueue, failed drain and retryable release.
4090 device acceptance: at least two sequences sharing real
weight work, correct independent KV histories, full-row numerical
checks, access sanitizers, memory grants and complete retirement.
Then measure actual text/HTTP mixed-arrival SLO goodput.

For later capture/replay or restore variants, deliberately delay the final cache/feature
load and late metadata reads. Compare common-history outputs with the qualified eager
path, and test cancellation/failed drain while dependencies remain pending. A passing
host bookkeeping test or an undelayed replay is not device-ordering evidence; an early
shared-read fence must not release model state. Keep this gate with the variant rather
than making graph capture a prerequisite for depth-one model-wide batching.

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
The AR engine already offers aggregate work; Gemma serializes it
under that contract. First implement backend-owned prepared physical
lanes with exact sequence/state mapping, whole-offer resource grants
and model-wide shared-weight execution. Keep CPU/single-row oracles.

Compare physically packed/ragged decode, prefill and separated mixed
phases under identical logical offers. Track true batched rows/tokens,
fallback, transfer and kernel/host costs, pressure, queue age and
latency-objective goodput. Preserve output identity, complete
continuation state, numerical acceptance and failure retirement.
Do not pad to maximum context merely to batch.

Only once physical costs stabilize compare chunked prefill, decode
protection, replay aging and measured-time versus token budgets
on identical lanes. Depth-two initially overlaps independent
sequences with separate immutable staging, metadata, workspace,
outputs and completion owners. Same-sequence work-ahead, graphs
and speculation need their own qualified state dependencies.

### 4e. Model-local integration and composition

Gemma 4 E4B text already has a real CPU/Metal/CUDA experimental
greedy decoder and bounded within-request prefill, not competitive
concurrent execution. Keep its model-local processor/configuration,
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
