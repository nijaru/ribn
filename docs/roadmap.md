# Engineering roadmap

[Target design](inference-engine-design.md) owns product direction and architecture;
[resource protocol](resource-protocol.md) owns execution/resource ownership;
[architecture](architecture.md) maps current code. This document owns delivery order,
open implementation choices and acceptance gates. All v0 interfaces may change.

## Current execution order

**Next: measured Gemma/Qwen execution improvements and remaining quality/capacity gates.**
The real second decoder runs on CPU, Metal and CUDA through the synchronous greedy
executor, token runtime, text facade and CLI. The CPU full-history decoder remains the
independent arithmetic oracle. CUDA now selects [qualified sixteen/eight-row prefill](../benchmarks/gemma4-cuda-baseline.md#sixteen-row-full-reach-and-integration-qualification-2026-10-25)
for RTX 4090/E4B geometry, up to 4096 positions and sufficient prepared scratch. Eight-row
and scalar fallbacks, complete owner histories and completion-owned device K/V copies remain.

The [matched diagnostic](../benchmarks/runtime-contract.md#bounded-eight-row-cuda-prefill-2026-10-21-utc)
reduces 18/64-token prefill from 271.93/1004.29 to 68.44/146.96 ms, with +1.35% pooled
decode latency and 3.94 MiB additional charged workspace. It meets independently declared
GEMM/greedy, ownership, sanitizer and performance gates, including no-tail heads and a
window crossing inside a group. All 12 archived winners agree across 19 evaluations,
but full distributions differ more from stock CPU than scalar CUDA; no distributional,
sampling, task-quality or competitive serving acceptance follows. Executor admission now
uses actual prepared capacity and rejects invalid vocabulary/media/non-greedy input
locally. Keep the scalar exact partition test scoped to its unchanged row-one arithmetic.

A larger row count is not the next milestone by default. Compare a matched incumbent,
preserve numerical-policy distinctions and measure mixed arrivals/capacity before claiming
competitive progress. Cache eviction, sampling and mixed-retention lifetime remain separate
work. The [host multi-row oracle](../benchmarks/gemma4-reference/README.md#host-multi-row-prefill-oracle)
and [CUDA row-count preflight](../benchmarks/gemma4-reference/README.md#cuda-projection-row-count-preflight-2026-10-21-utc)
retain the evidence that CPU bitwise batching cannot be assumed for cuBLAS.

**Preparation-boundary comparison — direct-call evidence:** the earlier [matched CUDA diagnostic](../benchmarks/gemma4-cuda-baseline.md)
uses the same active BF16 parameters and common histories. Eight-row native 18/64-token
prefill was 2.81x/5.81x slower than stock Transformers eager; pooled native/eager decode was
19.33/21.18 ms. Stock whole-forward compilation works, but `dynamic=False` produced eight
specializations for nine advancing decodes; `dynamic=True` reduced that to two, with
63/53 s first-forward costs and changed logits. Fixed-position warm calls near 14 ms
are not a general serving result. Full rows, cold costs, allocator/RSS observations and
scope limits are retained; no numerical budget or automatic promotion follows.

This supports further investigation of model/compiler reuse, not a selected Rust importer,
package format or universal graph ABI. Model-support velocity, Rust-owned preparation,
recovery and backend portability remain unqualified. Preserve the independent CPU oracle.
The [SGLang engine-level direct-call comparison](../benchmarks/gemma4-cuda-baseline.md#owned-fp32-host-output-engine-comparison-2026-10-22)
now runs on the RTX 4090 with the full pinned Linux stack and fresh native/stock controls.
All clocks include a completed owned FP32 host row. Across three fresh processes per
configuration, eight-row native 64-token prefill measured 146.844 ms versus 18.280 ms for graph-configured
SGLang; pooled decode is 19.271 versus 12.037 ms—about 8.0x/1.60x slower. SGLang's full
one-request graph runs every decode, not prefill. All 12 common-history winners agree,
but native/SGLang KL and TV maxima exceed stock; this is not quality-equivalent acceptance.
Old BF16-host timings are not reused. The [eight-row loopback serving study](../benchmarks/gemma4-cuda-baseline.md#matched-loopback-serving-diagnostic-2026-10-22)
measured 48 common variable-length prompts, c1/c4/c8 and rate-controlled admission overload,
with three fresh server processes per engine. Native stayed near 38 output tokens/s while
SGLang reached 281–291 at c4/c8; c8 TTFT p95 was 4.68 s versus 0.49 s. Raw client clocks,
actual usage, phase-window rates and joint diagnostic deadlines reconcile across all 24
phases; six servers exit cleanly. At c1 only 34/48 continuations match, so these are not
quality-equivalent ratios. AIPerf can suppress concurrency-gated arrival ticks; its issued
cohort is not a planned-arrival ledger. Physical pressure/recovery, independent quality,
representative SLO acceptance and incumbent-native client cross-checks remain open.
Target the measured concurrency/prefill deficit without selecting a universal compiler/importer
or widening rows by default.

The [fresh CUDA execution-cost trace](../benchmarks/gemma4-cuda-baseline.md#native-cuda-execution-cost-trace-2026-10-25)
records 86.6 ms of collected matrix/reduction work in a 161.7 ms instrumented 64-token
prefill. Collection warnings prevent a complete GPU/host partition; synchronization
spans overlap device execution and are not pure host overhead. This positive device-cost
evidence selects one bounded, model-local **16-row prefill candidate versus the qualified
eight-row lane**, not a row-count sweep or another synchronous PLE-only patch. The
[fresh private 16-row cohort](../benchmarks/gemma4-cuda-baseline.md#private-16-row-repeatability-rescreen-2026-10-25)
now passes the original three-pair advancement screen after an exclusive GPU handoff
and isolation-monitor repair. Median 64-token prefill falls 147.038→76.743 ms (47.8%)
and 18-token prefill 68.523→51.570 ms (24.7%); every pair improves both shapes, other
prefix/decode aggregates stay within their gates, and charged memory rises 7.0 MiB.
All 372 matched full-row comparisons are bit-exact on this cohort. The same frozen
candidate retains eight-row/scalar fallback and the existing independent component,
archived greedy/window and constrained-pool cache-retirement evidence. The earlier
interrupted cohort and zero-row monitor failure remain preserved, not pooled into
these results. The [full-reach and integration follow-up](../benchmarks/gemma4-cuda-baseline.md#sixteen-row-full-reach-and-integration-qualification-2026-10-25)
now adds 40 bit-exact full-reach comparisons, focused whole-model and maximum-span
attention memchecks, actual-sixteen-row executor/text checks and an ordinary CUDA CLI
smoke. Automatic selection is enabled only for the qualified device/geometry/reach and
prepared ceiling, retaining eight/scalar fallbacks and unchanged decode. This preserves
the existing GPU numerical policy; it does not establish new long-context CPU parity
or quality-equivalent acceptance. The [promoted-lane serving comparison](../benchmarks/gemma4-cuda-baseline.md#promoted-sixteen-row-serving-comparison-2026-10-26)
now measures three fresh eight/sixteen-row process pairs on that mixed-length loopback
cohort. Median paired c4/c8 throughput improves 9.9%/9.8%, and all 501 shared successful
native prompt comparisons preserve text, usage and finish. The internal advancement
screen passes, but native throughput still flattens near 41 tokens/s versus fresh SGLang
281–291, without quality equivalence. Native c8 TTFT p95 remains 4.10 s and only 4/48
requests pass the diagnostic deadlines. Retain sixteen rows; select subsequent model-local
cross-sequence physical execution from this remaining deficit, not wider prefill by default.
The first [four-sequence decode prototype](../benchmarks/gemma4-cuda-baseline.md#four-sequence-decode-numerical-gate-2026-10-26)
fails full-logit bit equality on its first common-history decode, despite finite outputs
and four unchanged argmaxes. It remains private and unpromoted; the new advancement
contract below must pass before performance measurement. Later candidate cohorts and
lifecycle/serving gates are unexecuted, not passed. A host-only [first-projection reference](../benchmarks/gemma4-reference/README.md#first-ple-projection-reference)
now supplies expected BF16 operands and complete binary64 diagnostic sums for the
first rejected decode inputs. A bounded [native first-projection replay](../benchmarks/gemma4-cuda-baseline.md#first-ple-projection-native-capture-2026-10-28)
now matches every consumed input/weight bit to that reference, but serial and four-row
BF16 outputs differ at 12 of 10,752 coordinates per row (maximum difference 0.0625).
An [actual decoder trace](../benchmarks/gemma4-cuda-baseline.md#actual-decoder-projection-trace-2026-10-28)
confirms those PLE differences on the frozen histories and finds layer-zero Q differs
at 5 of 2,048 coordinates per row (maximum 0.25), also with matching consumed operands;
K and V match in this scope. All same-mode prefills and final logits remain bit-exact
to the archived witnesses. PLE-only parity would not establish projection-wide bit
invariance. Keep the candidate unpromoted; these observations supply no tolerance or
precision-policy change. The accepted [reordered-decode advancement contract](resource-protocol.md#reordered-cross-sequence-cuda-decode)
now permits a new, separately frozen experiment with unchanged BF16/FP32 policy,
independent GEMM bounds including the batched vocabulary head, common-history and
continuation-routing checks, and a per-question no-loss labeled generation screen.
The [128-question cohort](../benchmarks/gemma4-cuda-baseline.md#reordered-four-sequence-decode-advancement-contract-2026-10-29)
is selected and tokenized without model execution. The fixed-corpus scorer is
host-tested, including per-item losses and incomplete records; the model collector,
component/full-model and quality runs remain unimplemented/unexecuted. No old failure
becomes a pass. Next implement and validate the remaining collector/bound/state checks
before another model run, rather than trace another projection or choose thresholds
from the failed logits. A numerical/quality pass would authorize private performance
measurement, not automatic selection or
quality-equivalent serving claims.
The [shared-grant recovery gate](../benchmarks/gemma4-cuda-baseline.md#shared-grant-allocation-and-recovery-gate-2026-10-26)
now qualifies real partial cache-allocation failure, healthy-peer preservation, charge
retention for cancelled accepted submissions and subsequent reuse through the native Engine.
It changes no production behavior and proves logical-grant recovery, not driver OOM,
physical peaks or already-enqueued asynchronous cancellation. Endpoint pressure/recovery
and independent common-history quality remain measurement priorities; representative
arrivals and product SLO acceptance are still open.

vLLM remains source-reviewed only, and unmodified Flare excludes SM89. Use their mechanisms
and the [Flare findings](inference-engine-design.md#nvidia-flare-reviewed-2026-10-21) without
narrowing hardware breadth or replacing ownership gates.

The same-artifact llama.cpp comparison (2026-10-18) measured 32.7 versus 39.9 tok/s at
c1—about 18% lower Ribn throughput, with better TTFT—and 2.2x/3.3x throughput gaps at c4/c8. This is not
single-stream parity or independently quality-qualified competitive acceptance.
[Baseline](../benchmarks/runtime-contract.md#matched-cross-engine-decode-baseline-2026-10-18)

Fresh [Qwen descriptor A/B evidence](../benchmarks/runtime-contract.md#member-descriptor-native-ab-2026-10-23)
measures 1,764→1,252 decode launches per eight-row step, superseding the historical
3,862 count for this scope. Three paired native repetitions improve burst throughput
4.26%/6.91% at c4/c8, but only 0.31% on the mixed-arrival trace. The candidate misses
its predeclared >=5% gain at both widths; joint diagnostic-deadline goodput stays zero.
Keep it default-off and defer further lifetime qualification without new bottleneck
evidence. Its parity and five-kernel memcheck pass, not whole-engine sanitizer or
serving-owner lifetime qualification. The qualified per-member path remains the control.
Historical prefill-host-cost observations and the modest mixed benefit motivate measuring
remaining prefill/projection/device costs, not attributing the entire gap to the host.
The Gemma prefill result does not qualify Qwen descriptors, graphs or hybrid-state lifetimes.

Qualify the cross-engine numerical comparison and target remaining prefill/projection
costs on the contiguous path. The [first endpoint diagnostic](../benchmarks/runtime-contract.md#first-equal-artifact-endpoint-diagnostic-2026-10-16)
exposed flat mixed c4/c8 throughput at about 22 tok/s. The [first measured lane improvement](../benchmarks/runtime-contract.md#heterogeneous-capacity-decode-2026-10-16)
now batches compatible decode rows across different KV capacities: the same cohort
improved to about 32/35 tok/s (+44%/+59%), with unchanged sampled outputs and short-c1
latency. Its c8 TTFT p95 remains near 5 s. Keep logical policy as the control when
selecting the next bounded execution experiment.
The earlier llama.cpp observations were 48/97/187 tok/s, but cross-engine histories
differ and the incumbent's mixed outputs vary. Quality-equivalent competitive acceptance
remains open; classify differences on common histories rather than chase greedy equality.
The bounded CUDA HTTP/reference gates and host input-retention gates passed.
The real second decoder is now the integration test bed. Continue toward a real VLM
alongside baseline work; do not wait for hybrid caching or scheduling completion.
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

### Planned slices: batch-width launch reduction

The original order followed the 2026-10-18 host-cost diagnostic. Fresh decode-only
profiles now measure 1,764 control launches at eight rows, not the historical 3,862.
The default-off candidate reaches 1,252, misses its native advancement gate, and leaves
mixed throughput nearly unchanged. These are mechanisms to reconsider from measured
costs, not a mandatory implementation staircase; unpassed acceptance below stays open.

**Slices 1–2 have a default-off decode candidate, not a promotion.**
`QwenLoadOptions::member_descriptors` and the native benchmark's `--member-descriptors`
select a single uploaded packet and member-dimension convolution, recurrent update,
KV append and attention kernels. A gather kernel replaces the per-member q/k copies.
Packet assembly/bounds are host-tested; Rust CUDA-feature checks cover the binding and
access-guard lifetimes, not device completion. Publication explicitly waits for the H2D
copy before reusing its host packet; uncertain copies retain or quarantine that packet.
The qualified per-member path remains the default; prefill, arenas and graphs are unchanged.
Numerical parity and a complete five-kernel memcheck pass on the RTX 4090.
The [native timing screen](../benchmarks/runtime-contract.md#member-descriptor-native-ab-2026-10-23)
does not meet its advancement gate. Asynchronous-owner lifetime and broader performance
qualification remain open if new evidence warrants revisiting it.
The launch target below remains unproven: projections, norms and residual operations
already batch members, so removing state loops alone cannot imply a batch-width reduction
of the entire step. Profile the remaining fixed layer chain before choosing further fusion.

1. **Member descriptor.** One device descriptor per step: per-member KV layer pointers and
   page-table pointers, per-member recurrent matrix and convolution-history pointers, and
   the member count, published in a single host-to-device copy. Host-tested assembly,
   bounds and lifetime; replaces per-member view arguments for state-access operations.
2. **Member-dimension kernels.** KV append, paged attention, GDN convolution and state
   update, and per-member copies consume the descriptor. Acceptance (device): launches per
   eight-row step fall below ~600 (fresh control 1,764; candidate 1,252), with bit-identical
   hidden state, KV and sampled tokens against the current path on `cuda_reference` and the
   Qwen reference gate.
   Keep the per-member path until that passes.
3. **Prefill descriptor.** The same descriptor at chunk granularity. Acceptance (device):
   eight-row 30-token prefill host enqueue below ~100 ms (from 0.94-1.5 s) with unchanged
   outputs.
4. **Resident KV arena.** Pages granted by the one authority and materialized from a
   resident arena, so descriptor entries and cache pointers stop moving; non-identity page
   tables exercised by the heterogeneous decode suite. Host-tested accounting; device
   evidence for the mapping.
5. **CUDA graph capture.** Decode step captured per (member count, position bucket) with
   device-resident positions and counts. Acceptance (device): host enqueue under 5 ms per
   step, bit-identical outputs, and cancellation/retirement semantics preserved through
   replay. This is the slice vLLM and llama.cpp both rely on.

Slice 2 precedes arena integration; graph capture requires stable captured storage and
completion-safe descriptor updates. A resident arena is the planned storage increment,
not evidence that descriptors alone eliminate the fixed per-layer launch chain. Do not claim any
throughput figure from this sequence without the device suites; the c1 path stays the
correctness oracle throughout.

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
   not establish executable support. Use the real second decoder to compare native and
   externally prepared execution before expanding handwritten integrations; model-support
   velocity and maintenance burden are part of the result, not inferred from host language.
   A real VLM follows this test.
4. **Broader qualification.** Select subsequent model/workload and hardware targets from
   observed gaps. Another NVIDIA GPU class tests performance portability; a materially
   different backend tests backend portability. Neither follows from the other.

Choose workload objectives before the baseline; choose the incumbent gap to address,
acceptance criteria and an effort budget before the optimization experiment. Individual
experiment thresholds below are not product SLOs or a general effort commitment. Record
the result and whether it warrants promotion, another experiment or redesign in the existing
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
   profiler-counter work. The historical 13.74 tok/s / zero-goodput smoke is not a matched
   baseline. The new endpoint cohort exposes flat mixed c4/c8 throughput and rising latency.
   [Bounded device probes and a native CUDA timeline](../benchmarks/runtime-contract.md#bounded-device-mapping-and-trace-2026-10-16-dc4aed0)
   map the opt-in snapshot to actual one/four/nine-row and prefill-chunk work. The mixed
   native trace has 128 scalar decode calls and no whole-decode groups; quantized warp
   GEMV takes 70.2% of summed kernel time and kernels occupy 96.1% of its single-stream
   first-to-last-kernel span. These are instrumented native-workload observations, not
   endpoint phase attribution, occupancy or an overhead bound. Investigate physical
   projection/prefill throughput and heterogeneous decode before a scheduler-only rewrite;
   compare qualified native/vendor options as well as custom kernels. Queue age, physical
   memory attribution, endpoint exposure and broad diagnostics overhead remain open.
   **First experiment promoted:** the existing whole-decode lane now accepts different
   positive KV capacities with matching model layout. Stream-ordered score reuse needs
   no new kernel, policy or continuation padding. The one-cycle experiment passed its
   predeclared >=10% mixed-c4/c8 median throughput and <=5% short-c1 p95 E2E-regression
   criteria, unchanged cohort outputs/usage, capacity-invariance/attention memcheck and
   independent reference/cancellation/retirement gates. These are scoped experimental
   results, not broad numerical equivalence or product SLOs. The native replay now has
   23 whole-decode groups instead of zero, while retaining identical work-token totals.
   Mixed phases still fall back; investigate prefill/tail and projection costs next.
   **Second experiment promoted:** the eight-token prefill lane now samples its final chunk
   row by default instead of re-running that token on the serial path. It applies only when
   the chunk already advanced every persistent component for the token, the lane runs warp
   projections without the chunk scan, and the prompt is an exact whole number of chunks;
   every other shape or mode keeps the serial final step, and no kernel or grant changed.
   All predeclared criteria passed over three alternating-order endpoint repeats: median
   short-c1 p95 TTFT fell 57.3% (272.68 → 116.56 ms, ≥20% required), throughput and p95
   content E2E stayed within 5% in every cohort (short-c1 gained 16.0%; mixed c4/c8 gained
   2.1%/2.4%), and all 144 paired visible outputs and core usage counts matched. Chunk rows
   remain bit-identical to the serial final residual, with a clean memcheck and a
   re-executed independent reference/cancellation/retirement suite. See the
   [endpoint measurement](../benchmarks/runtime-contract.md#sampled-final-prefill-chunk-2026-10-17-promoted).
   These are scoped experimental results, not product SLOs. Short prompts now spend about
   half their previous prefill latency in the remaining chunks and serial remainder; long
   prompts move less in proportion. Sampling remains greedy-only.
2. **Qualify useful physical lanes.** Compare packed/ragged heterogeneous-capacity work,
   scalable prefill GEMM and tiled attention against qualified small-row kernels and native
   vendor options. Preserve independent nonadjacent histories/completion mapping. Do not
   weaken geometry/addressing checks without qualification, or restore maximum-context
   reservation just to batch. The capacity-only restriction above is removed after its
   addressing gate; other lanes need their own evidence. Measure conversion/repacking/
   workspace costs.
3. **Evaluate unified logical policy.** Select bounded prefill/decode/replay frontiers using
   age/priority, output credits, token bounds and aggregate backend feasibility. Protect
   decode latency and prefill/replay progress. Compare against current queues with the same
   physical lanes, then compare lane changes separately. Tune chunk sizes with actual
   TTFT/ITL/E2E objectives, mixed arrivals and pressure; submission counts are not time bounds.
   Once lane/dispatch costs stabilize, compare calibrated prefill-time quotas with token
   budgeting on those same lanes. Report prediction error/headroom, context/length skew,
   fairness and SLO goodput; retain the token baseline and losing cases. DeepInfer's fitted
   quadratic is a production reference, not a selected Ribn formula.
4. **Persistent metadata and bounded overlap.** Separate stable request rows from kernel
   ordering. Start with independent-sequence preparation/execution overlap only after
   per-submission owners and physical release qualify. Gate immutable staging/table versions,
   out-of-order completion, stalled delivery, shutdown and failed-drain recycling. Graph
   capture needs qualified stable storage and measured eager-versus-replay benefit.
   After launch reduction, use timelines to measure recoverable CPU scheduling, metadata,
   enqueue and device gaps. DeepInfer's result reconciliation informs later dependent
   overlap; it does not bypass projected/committed cursor or EOS/cancellation gates.

Promote policies/variants only over their qualified scope, with repeatable benefit beyond
run variability and no hidden quality, memory, fairness or latency regressions. Retain
losing cases. Dependent work ahead of host commitment and speculation need their own
[contracts](resource-protocol.md#bounded-overlap-and-submission-ownership) and gates.

### 4e. Model-local integration and composition

Integrate a real dense second decoder through ordinary package/SafeTensors loading,
model/processor/backend, central registration and tests alongside baseline work, then a
real VLM/processor. Pin a concrete checkpoint and independent reference before coding;
exercise another configuration/checkpoint without another architectural rewrite. Use
these to replace hard-coded kernel geometry where the second model needs a different shape,
not to add family branches in lifecycle/routing or create a universal operator layer. The
plan/description/capability layer between the model and its backend is gone; what remains
shared is the prepared batch, continuation state declarations and artifact weight
descriptions. Start after applicable baseline contracts; it need
not wait for prefix-cache or overlap promotion. Pin independent numerical references and
measure integration effort and loading/physical peaks, not just compilation.

The selected second-decoder workload is **Gemma 4 E4B instruction-tuned, BF16,
text-first**. Its
[manifest](../benchmarks/gemma4-reference/model.toml) pins the real package and independent
reference; [CPU reference collection](../benchmarks/gemma4-reference/README.md) covers
FP32/BF16 operands and the 511/512/513-position local-window transition. These are
integration prerequisites, not executable Ribn support, multimodal qualification or
GPU-fit evidence.

The [canonical image reference](../benchmarks/gemma4-reference/README.md#canonical-image-reference)
now adds the real checkpoint's image encoder/projector and full image+text CPU forwards:
four frozen RGB cases in each precision, ten exact archived text controls and 230 finite
captured tensors. The two-image answer remains capped at eight tokens; this is not
vision-quality acceptance. E4B uses causal text masks, not the larger-Gemma
vision-bidirectional branch. A model-local CPU vision component now selects its 659
BF16 encoder/projector tensors from the text source's retained artifact inventory,
without changing text-only selection. An independent small stock-HF fixture checks
all encoder/projection stages exactly, including grouped heads and padding. The
[strict full-checkpoint vision gate](../benchmarks/gemma4-reference/README.md#native-cpu-vision-component)
failed across all four cases/five images: patch embeddings match exactly, but
layer 0 and projected features disagree. Do not use the small fixture to claim
full-model fidelity. Shared-operand layer-0 comparisons now reproduce all four
stock layer outputs exactly before checking 140 Native operations: axial RoPE,
GELU, gated products and residual adds match exactly, while RMS norms, projections
and attention differ on identical inputs. The first local difference is input RMS
normalization. The [RMS working-precision audit](../benchmarks/gemma4-reference/README.md#vision-rms-working-precision-audit)
now covers all seven norms/five images: 1,669 of 67,737,600 BF16 words differ, while
using the stock inverse makes every unrounded FP32 product output exact. Statistics
and inverse evaluation differ; this does not identify a unique faulty operator.
A separate real-equation enclosure diagnoses final rounding, not FP32-policy
correctness or a precision winner. Production arithmetic and acceptance remain
unchanged. Establish independently justified numerical acceptance or repair a
demonstrated defect before binding processor features to AR execution; these
local diagnostics do not waive the failed full-model gate. Bounded feature
ownership, snapshot compatibility and admission through the existing executor/text
path remain open. Ribn still rejects media; this component does not establish a general
staged/coupled execution contract or waive the separate text-serving fidelity gate.

The file-backed SafeTensors path now reads its approximately 16-GB artifact without
whole-file payload allocations. The [host loading gate](../benchmarks/loader-memory/README.md)
checks the complete artifact and every encoded tensor against the independent source,
plus BERT F32 bit preservation and a matched host-memory comparison. This is artifact
I/O evidence, not Gemma execution, whole-loader byte accounting or device preparation.

The model-local `engine-gemma4` host preflight now validates this real package's
665 active BF16 parameters, preserves their encoded bytes and represents the output
head as the same logical embedding parameter. The 54 inactive shared-tail tensors
and 1,411 non-text tensors are explicitly excluded from text selection. Host fixtures
also exercise a second, reduced geometry and sharded loading without new family branches. This is
source selection alone, not the second-decoder exit gate. The optional CUDA loader now
prepares those native BF16 weights in one charged allocation with a tied head and owned
partial-load retirement. Two real-package runs matched all 665 source/device hashes and
restored free VRAM on unload; storage and partial-source cleanup gates passed memcheck. The
[device weight gate](../benchmarks/gemma4-reference/README.md#native-bf16-device-weight-gate)
is not numerical forward execution.

Model-local KV range semantics now separate persistent suffixes from full current
attention spans and resolve shared-owner/last-consumer routing at configuration load.
All 24 archived prediction rows (FP32/BF16) match the 24 persistent owner and two donor
shapes. The [mixed-retention contract](resource-protocol.md#mixed-retention-and-shared-kv)
requires executor-bound validity, reservation and completion-safe donor retention;
pure range arithmetic establishes none of those physical properties.

The canonical host text processor now matches 291 independent cases: exact raw/chat
token IDs, rendered chat, whole/finalized decoding and all generation stops. It rejects
unsupported media placeholders in both encoded and direct-token input. Its stateful
decoder retains whole byte-fallback runs because a later malformed byte invalidates
the entire run in HF. The text facade now creates an owned, bounded decoder during
permit-backed preprocessing, before runtime enqueue. Gemma's adapter preserves its
metadata stop policy; GGUF retains strict UTF-8 and its existing control policy.
Host lifecycle tests cover interleaved streams, cancellation, owner failure, and
request-local push/finalization failures, including immediate scratch release.
These are host delivery checks, not Gemma execution or CUDA lifecycle qualification.

An independent stock Transformers BF16 eager run now fits a 4096-token prompt plus
32 greedy predictions on the 4090: 17.10 GiB peak allocator reservation, 17.62 GiB sampled
whole-device use. No quantization/offload was needed for that single-request workload;
this does not establish Ribn's eventual workspace/serving peak.

The first native BF16 linear component now binds model projections to cuBLAS with
preallocated output, charged workspace and completion-owned failure guards. Its
shared-allocation refactor and arithmetic now pass the
[device component gates](../benchmarks/gemma4-reference/README.md#native-bf16-linear-component-gate)
at `53c8aed`: six tests normally and under memcheck, plus two real-package
665-parameter byte/readback/unload regressions against the independent source audit.
These qualify the linear building block, not a second executable decoder or model support.
Native grouped BF16 RMS now also passes its
[device component gate](../benchmarks/gemma4-reference/README.md#native-bf16-rms-component-gate),
including direct learned scales, per-head/packed-PLE grouping, owner-only scale-free V,
memcheck, racecheck and synccheck. Native BF16 rotary execution now passes its
[device component gate](../benchmarks/gemma4-reference/README.md#native-bf16-rotary-execution-gate):
the backend prepares factors from the model's FP32 inverse frequencies and positions
and applies them with exact per-product rounding, reproducing stock-CUDA cosine, sine
and rotated output bits at positions up to 131071. Frequency derivation from theta
remains host-side. Shared completion ownership and full weight readback/unload were
requalified; none of these components establishes full-model numerical acceptance. Separately, the
model-owned [host rotary-frequency gate](../benchmarks/gemma4-reference/README.md#host-rotary-frequency-gate)
now checks the padded proportional/global and ordinary/local factors against pinned
Transformers and an independent FP64 equation. This is preparation evidence, not
large-position phase, device rotation or whole-model acceptance. Independent
[recomputed component operands](../benchmarks/gemma4-reference/README.md#recomputed-attention-components)
now cover input/Q/K/V RMS, projections and RoPE for local/global owners 0/5/22/23,
using the archived real-model hidden inputs under both CPU precision policies.
Their one-token processing schedule is distinct from original prefill taps; no new
Ribn device slice or full-model qualification follows.

The remaining elementwise layer math is now device-qualified against those stage
references: both residual adds, PLE combine, layer scaling, the gated product, GELU
and the final soft cap. Exact transformations match the archived encodings; GELU stays
inside the host audit's FP32 bound and soft-capped logits matched every one of 262,144
encodings. See the [elementwise gate](../benchmarks/gemma4-reference/README.md#elementwise-layer-math-component-gate).
Native attention now also matches the archived stock operands bit-for-bit for
local/global layers, including sliding windows and decode-against-history
([gate](../benchmarks/gemma4-reference/README.md#native-bf16-attention-component-gate)).
A first CUDA decoder now reproduces the archived tokens with explicit KV owners,
shared-tail reads, the runtime executor and the text facade; `ribn run-gemma` selects
it when built with `--features cuda`. See the
[decoder gate](../benchmarks/gemma4-reference/README.md#native-cuda-gemma-decoder-gate).
Device-side K/V append now passes byte-preservation, initialized-prefix and memcheck
gates, with exact short-history logits against host append. The subsequent
[eight-row lane](../benchmarks/gemma4-reference/README.md#bounded-eight-row-cuda-prefill-2026-10-21-utc)
binds whole-group attention/donor spans and prepared workspace while retaining full
histories and scalar fallback. Its [sixteen-row extension](../benchmarks/gemma4-cuda-baseline.md#sixteen-row-full-reach-and-integration-qualification-2026-10-25)
retains the same numerical policy and eight/scalar fallbacks. The greedy contract permits
independently bounded reordered GEMMs, not presumed scalar bit equality.
The shared `ribn serve-gemma` HTTP entry
point now passes its [Metal](../benchmarks/runtime-contract.md#gemma-http-cli-qualification-2026-10-21)
and [CUDA functional gates](../benchmarks/runtime-contract.md#cuda-device-requalification-2026-10-22),
including JSON/SSE output, decoded stops, disconnect peer progress and shutdown during loading.
Trimmed/mixed-retention cache lifetime, sampling and eviction remain unqualified; no
competitive serving acceptance follows.
The host CPU path now composes a full Gemma decoder step — embeddings, PLE, all 42
hybrid layers with owner/shared-tail KV, attention, the remaining layer math and the
tied head — and reproduces all four greedy tokens of the archived 18-token chat row.
It keeps complete per-owner history and masks local spans rather than evicting them.
A synchronous greedy executor now runs the host decoder through the token runtime and
the text facade for admitted sequences sharing one prepared weight set; device executors
are exported by `ribn run-gemma` on Metal and CUDA. Cache eviction, donor trimming,
sampling and long-context behaviour remain open; the Metal and CUDA decode timings are
diagnostic, not performance or serving claims. On Apple silicon, a first
Metal (GPU, not ANE) slice now runs BF16 linear, RMS, causal/sliding attention, rotary
and elementwise operations on real hardware with independent-reference tests, and the
validated Gemma inventory uploads bitwise to one Metal device, and a host-orchestrated
Metal decoder reproduces the archived four greedy chat tokens through the runtime
engine and the text facade, and a synchronous greedy executor shares weights across
admitted sequences. Splitting attention into ordered score, reduction and value
dispatches cut single-sequence decode from ~5.5 s to ~0.4-0.5 s per token on the M3 Max in
the diagnostic benchmark; that is not a competitive result. The package now also runs
through the ordinary CLI path (`ribn run-gemma`, macOS Metal, greedy). Two
reordered-accumulation GEMM variants (transposed weight layout and warp-cooperative
reduction) were numerically qualified but measured slower than the four-wide exact
kernel, so both were removed and recorded. Four-output-column register blocking also
lost and was removed. Pairing input rows with wide weight loads is now selected on
M3 Max for even 8–64-row batches: the diagnostic median prefill fell from 474 to 365 ms
at 18 tokens and 694 to 527 ms at 32, with exact common-history logits and continued
execution. Other GPUs and shapes retain the original kernel. The
[qualification](../benchmarks/runtime-contract.md#paired-row-linear-qualification-2026-10-20)
records substantial timing variance and does not establish serving performance.
Metal CPU embedding reads borrow the shared weight inventory. Direct source loading
now also removes the full CPU inventory and parameter-sized upload copy from preparation,
using at most 64 KiB of read/copy scratch
([measurement](../benchmarks/runtime-contract.md#direct-metal-source-loading-2026-10-20)).
This is one synchronous load, not a global concurrent-loader grant.
A [command-buffer profile](../benchmarks/runtime-contract.md#metal-command-buffer-profile-and-ple-batching-experiment-2026-10-20)
now distinguishes repeated short-op synchronization from GPU intervals and model-side
CPU work. The initial bounded PLE normalization batching candidate missed its
predeclared promotion gate and was removed. A later rescreen after the other chains
and GELU changes is qualified below; neither study attributes CUDA costs.
The subsequent [fixed linear → RMS chain](../benchmarks/runtime-contract.md#metal-linear-rms-chain-qualification-2026-10-20)
keeps the BF16 projection on device and removes its intermediate readback/upload and
216 synchronous waits per step. The [FFN input chain](../benchmarks/runtime-contract.md#metal-ffn-input-chain-qualification-2026-10-20)
now removes another 84 waits by sharing normalization across the gate/up projections.
The [query/key rotary chain](../benchmarks/runtime-contract.md#metal-rotary-projection-chain-qualification-2026-10-20)
removes another 66 waits while preserving both intermediate BF16 boundaries and sharing
read-only factors. A refreshed profile identified CPU GELU as a large prefill cost;
an [exact prepared BF16 table](../benchmarks/runtime-contract.md#metal-exact-gelu-table-qualification-2026-10-20)
now replaces repeated host tanh evaluation without changing the CPU oracle. A separate
[four-input-row linear kernel](../benchmarks/runtime-contract.md#four-row-metal-linear-qualification-2026-10-20)
further reduces 32/64-token prefill by 30/31% with exact logits, selected only for M3 Max
32–64-row chunks divisible by four and vector-width inner dimensions; qualified two-row
and original kernels remain the fallbacks. Other activation transfers and CPU elementwise
work remain; no universal graph or CUDA policy was introduced.
The 2026-10-20 [same-artifact MLX calibration](../benchmarks/gemma4-metal-baseline.md)
measured 148.5 ms versus 31.1 ms pooled warmed direct decode. Both backends match the 12 archived
winners but differ from stock CPU full logits; this is not a quality-equivalent serving
baseline. Large Ribn timing swings remain unexplained. After the bounded command/transfer chains below, a
[fresh execution-cost profile](../benchmarks/gemma4-metal-baseline.md#execution-cost-refresh-2026-10-23)
puts 71–76% of prefill GPU-command intervals in the compound FFN gate/up and down chains.
The [native main-FFN chain](../benchmarks/gemma4-metal-baseline.md#native-main-ffn-chain-2026-10-27)
now keeps exact activation/products on device for M3 Max/E4B chunks through 64 rows,
with separate-call fallback elsewhere. Three fresh processes pass the declared paired
screen: median process reductions are 6.93% for direct decode after 18 tokens and
9.37/10.98% for 16/64-row prefill. Full logits/KV remain bit-exact; this is not a new
MLX comparison, serving-quality acceptance or a promotion of MPSGraph. A subsequent
[HTTP screen](../benchmarks/gemma4-metal-baseline.md#native-ffn-http-screen-2026-10-28)
retains exact completed responses and observed paired gains, but hits its fixed cutoff
before the three-pair cohort finishes. No c4/c8 request in its completed phases meets
the joint latency objective; serving-performance acceptance remains open. A later
[matched-history entrypoint diagnostic](../benchmarks/gemma4-metal-baseline.md#matched-history-entrypoint-cost-2026-10-28)
places about 99.7% of observed wall time inside synchronous decoder calls in its three
complete matched processes. Its four-process gate remains incomplete after a private
harness readiness-path error. A separate [native-command profile](../benchmarks/gemma4-metal-baseline.md#native-command-cost-2026-10-28)
completes 36 generations: completion waits dominate wall time, command preparation
is below 0.7%, and nested GPU-command intervals cover roughly 68–73% across process
medians. Only the 113-token prefill passes its frozen GPU-dominance criterion. This
does not isolate driver/queue/scheduling causes or attribute the older AIPerf gap.
A subsequent [single-row FFN fusion screen](../benchmarks/gemma4-metal-baseline.md#single-row-ffn-fusion-screen-2026-10-28)
preserves BF16 rounding boundaries and matches full logits/KV and selected post-product
words across 46 paired states, but all nine process/history decode cells fail the
frozen 5% gain requirement. Even unchanged prefill paths show large timing swings.
The candidate stays private; neither
a causal slowdown nor a new automatic-selection policy is established. Do not repeat
it unchanged without new implementation or measurement evidence.
A [real-input FFN component pre-screen](../benchmarks/gemma4-metal-baseline.md#ffn-projection-component-pre-screen-2026-10-23)
now separates RMS from projections and compares a native FP32 MPS candidate including
conversion and per-call resource release. Prefill component calls improve, but decode
gate/up regress; retained FP32 shadows add 300 MiB per layer and 86 BF16 elements differ
across the 27 sampled tuples. No production variant is promoted. Resolve independent
full-model numerical acceptance and a bounded materialization/memory envelope before
prefill-lane integration. A [BF16-storage vendor follow-up](../benchmarks/gemma4-metal-baseline.md#bf16-storage-vendor-pre-screen-2026-10-23)
now runs MPSGraph directly on the same BF16 operands without caller FP32 weight shadows;
all 27 sampled outputs match the earlier FP32 MPS results. Direct MPSMatrix BF16 inputs
abort at encoding on the tested runtime. The private native Rust full-model pilot now
passes 117 exact paired logit/cache states and observes **14.61/28.11%** lower paired
direct-prefill wall time at 16/64 rows over 48 pairs/three processes. It reuses BF16 weights,
with native RMS around prepared pair/Down graphs. A subsequent
[private owned-retirement proof](../benchmarks/gemma4-metal-baseline.md#completion-owned-graph-retirement-proof)
now replaces fail-stop with callback-owned retention, model-local closure, retryable
settlement and defensive quarantine. Nine real device/owner cases pass, including delayed
writes, exception-after-enqueue, full-model shutdown and a nonblocking native-retirement
regression. Another 117 exact logit/cache pairs preserve numerical behavior; 48 new prefill
pairs observe 15.29/27.32% lower paired wall time. That proof remains historical evidence,
not an automatic selection policy.

The [integrated model-local option](../benchmarks/gemma4-metal-baseline.md#model-local-graph-integration-2026-10-23)
now exposes `Gemma4MetalFfn::ExperimentalGraph` through Rust model/text loading, with
SDK/runtime checks and the native deployment floor preserved. Native remains the
default and CLI/HTTP policy. Actual selection passes 117 exact full-logit pairs,
26 complete KV-state comparisons, public text loading and real runtime retry/Drop
qualification. Forty-eight new paired prefills observe **14.24/27.44%** reductions at
16/64 rows; a separate frozen-native comparison passes its ≤3% regression gate.
This is an experimental opt-in, not automatic promotion. A matched
[memory screen](../benchmarks/gemma4-metal-baseline.md#model-local-graph-memory-screen-2026-10-23)
now observes equal sampled Metal allocation maxima and flat post-Drop Metal totals
across three model reloads per process; all 195 full-row pairs remain exact. The
reviewed SDK exposes no Graph allocator hook or worst-case workspace query. Sampling
and process high-water counters do not resolve internal allocation authority or hard
physical bounds. Resolve that external-pool accounting/envelope policy before widening
selection. A [public text-driver diagnostic](../benchmarks/gemma4-metal-baseline.md#public-text-driver-graph-diagnostic-2026-10-24)
now exercises real default scheduling and mixed arrivals: 390 requests match exactly;
paired total burst/staggered trace times fall 14.43/18.92%. Default 16-token chunks select
Gate/Up pair16, not Down64. Most mixed requests miss the predeclared latency targets,
and burst target-meeting requests fall from 4/48 to 1/48 despite faster total draining.
Short pooled median E2E also rises 5.25%, though its p95 regression screen passes.
This is not HTTP, sustained-load, physical-pressure or competitive SLO acceptance.
A separate [Native HTTP versus stock MLX diagnostic](../benchmarks/gemma4-metal-baseline.md#native-http-versus-stock-mlx-diagnostic-2026-10-24)
now passes reference-text, protocol and accounting checks on both real servers.
All nine paired traffic groups have identical inputs/counts, but 12 of 72 paired
profiling outcomes differ in visible text across three synthetic prompts. The
predeclared matched-result gate stops the remaining process pairs; no quality-equivalent
serving speed ratio is accepted. A follow-up at 20 common prediction histories reproduces
raw-logit batch-width effects without HTTP; independent CPU BF16/FP32 references also
disagree, and distribution errors are not merely argmax ties. All 250 captured MLX raw
and normalized argmax pairs agree. Matched-chunk layer traces locate the first difference
at PLE projection and independently reproduce Native's documented FP32 addition order;
CPU BF16 prediction rows remain unchanged from whole-prompt prefill. Controlled raw-PLE
substitutions now leave all four forced-history predictions unchanged in each engine,
even with identical first-projection matrices. That operation is not the sole explanation;
a precision rewrite based only on its first bit difference is not an established quality
fix. A head-only replay with exact raw/post-softcap self-controls then makes the disputed
winner follow the normalized hidden input, not the head backend: both heads choose 3872
on Native's vector and 3823 on CPU's. All ten shared hidden inputs preserve raw and
post-softcap winner agreement. Replaying the pre-RMS decoder states then gives identical
BF16 normalized vectors from both RMS implementations on all ten shared inputs. All four
RMS/head combinations keep the disputed winner with the pre-RMS state. Captured-upstream
shared-KV replays first check layer 41, then the full sharing suffix; their winners follow
the joint input stream, but distribution gaps remain. A fuller eight-cell decoder-body
factorial then crosses the actual prefix backend, suffix backend and all 42 captured PLE
inputs, carrying the prefix's real KV with it. All original layer outputs and tail rows
are reproduced exactly before interpretation. The mixed Native-prefix/CPU-PLE stream
selects 3872 through the Native suffix but 3823 through the CPU suffix. A conditional
RMS/head replay makes that choice follow the suffix decoder output state, not its tail
implementation. Thus the observed effect is not a universal prefix-only cause: prefix,
suffix and PLE inputs interact. Across factorial contrasts, post-softcap total variation
reaches about 0.475 for the prefix axis, 0.095 for the suffix and 0.115 for PLE. No error
budget, faulty operation or accuracy winner is established. The 21-token history does not
qualify sliding-window eviction. A separate
[57-question labeled diagnostic](../benchmarks/gemma4-metal-baseline.md#independent-labeled-multiple-choice-diagnostic)
then freezes one MMLU test item per subject, raw zero-shot prompts and four-choice scoring
before model runs, retaining every selected item. Native and diagnostic CPU FP32 choose
identically on all 57 and score 30/57; CPU BF16 scores 28/57, differing on three choices.
All five original full prediction rows per backend reproduce exactly before evaluation.
This small sample shows no Native task-score deficit against these references, not broad
accuracy acceptance; FP32 is not ground truth. A subsequent
[32-question generated-QA diagnostic](../benchmarks/gemma4-metal-baseline.md#generated-extractive-qa-diagnostic)
uses independent SQuAD v1.1 labels through the real default Native and stock MLX HTTP
servers. Each passes 13 archived controls before its questions. Both score 17/32
whole-answer exact match and 79.1127429 token F1 on a 0–100 scale; 30 literal answers
and all 32 normalized answers agree. The same 15 EM failures remain included, without
answer extraction or explanation stripping. This is small-sample lexical generation
evidence, not broad task-quality, distributional or quality-equivalent serving acceptance.
Keep the original fidelity counterexample and failed serving gate open. Broader generated
task acceptance still needs a justified scope, and serving comparisons still need their
matched-result gate. Neither diagnostic supplies an equivalence budget or justifies
further causal bisection without a concrete decision it would resolve. Production
arithmetic stays unchanged; do not loosen tolerances or omit disagreeing prompts.
Opaque Graph submissions are not a single-command chain. Broader residency/synchronization
gains remain possible; no universal Graph ABI is selected.
This does not change CUDA sequencing or qualify capacity, recovery or SLO goodput.
A [lifetime ablation](../benchmarks/runtime-contract.md#metal-native-handle-lifetime-ablation-2026-10-21-utc)
now bounds autoreleased command/encoder handles at their backend factories instead of
leaving them to a caller pool. Repeated-call host retention falls, but the latency swings
were not consistently removed. A [bounded PLE RMS rescreen](../benchmarks/gemma4-metal-baseline.md#bounded-packed-ple-rms-rescreen-2026-10-23)
now uses randomized old/new trials within each warmed process: 126 runtime pairs show
5.23% lower c1 decode interval, with bit-identical full logits and unchanged generated
tokens. Metal groups the existing RMS kernel only for chunks up to 64 tokens and packed
payloads up to 2 MiB each; larger chunks/geometries keep the per-layer path. Actual
64/65-token decoder-boundary, executor and text-facade tests pass. This is M3 Max/E4B
execution evidence, not HTTP/SLO, physical-pressure or CUDA qualification.
The [attention-input projection chain](../benchmarks/gemma4-metal-baseline.md#attention-input-projection-chain-2026-10-23)
now keeps normalized inputs private through Q/K/V preparation in one serial command,
removing another 90 synchronous waits per selected E4B step. Across 126 matched runtime
pairs, median c1 decode interval fell 13.29% (process gains 7.44–13.48%) with exact full
rows and unchanged runtime tokens. A separate 96-pair whole-chunk screen passed at
7/8/32/64 tokens. Selection is bounded to 64 tokens and 4 MiB of total native chain
buffers, with the old calls retained above either bound. CPU/CUDA and scheduling are
unchanged; this is not a new incumbent, SLO or physical-pressure qualification.
The [attention-output chain](../benchmarks/gemma4-metal-baseline.md#attention-output-projection-chain-2026-10-23)
also removes 42 waits per selected step (298 → 256), preserving the attended BF16
boundary without its host roundtrip. It passes exact full-row and matched runtime gates:
4.42% lower primary decode interval over 126 pairs (process gains 4.38–6.28%), plus
96 whole-chunk pairs and the 576/577-key continuation boundary. Selection is limited
to 64 input rows, 576 keys and 4 MiB total native chain-buffer payload. Some individual
process cohorts regressed; the linked owner records variability and the predeclared
aggregate diagnostic gates. This is not serving/SLO or physical-pressure acceptance.
Sampling and eviction remain open. Shader/pipeline failures now fail both primitive
and materialization tests rather than being mistaken for missing hardware.
Then mixed-width local/global KV with donor spans pinned through shared-tail device reads. Use strongly typed model/backend state
behind `GenerationExecutor`, not the current first-`kv()`/homogeneous-cache convenience
or family branches in routing/scheduling. Qualified physical allocation/retirement and
numerical execution remain device-gated.

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
Those smokes alone establish no competitive measurement or physical-peak claim. The
[first equal-artifact diagnostic](../benchmarks/runtime-contract.md#first-equal-artifact-endpoint-diagnostic-2026-10-16)
adds measured deficits but leaves numerical equivalence and the representative acceptance
matrix below open—not another execution-loop adapter.

Deliver a documented text protocol subset over `TextModel`/`TextOwner`, preserving the
existing execution owner. Start with conventional completion/chat streaming semantics;
state supported parameters and explicitly reject unsupported ones. Bound connections,
body parsing, messages/options, preprocessing and streaming buffers; default to local
binding and require an explicit authentication/exposure policy for non-local deployment.
No remote URL/media fetching is needed for this first text path.

Gate streaming/terminal order, disconnect and queue-saturated cancellation, overload,
malformed/oversized requests, stalled clients with healthy-peer progress, readiness,
metrics and retryable shutdown. Health/readiness must distinguish loading, serving,
closing and faulted states. Networked Python access uses the existing protocol subset.
Native Python bindings follow a usable shared Qwen/Gemma execution path, without
waiting for every engine feature; they reuse its owned semantics as specified in the
[application API design](inference-engine-design.md#application-api-and-serving).

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
