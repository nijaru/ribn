# Current implementation

This is a source map, not a second target design. [Inference engine design](inference-engine-design.md)
owns direction; [resource protocol](resource-protocol.md) owns execution contracts;
[roadmap](roadmap.md) owns sequencing and open gates.

## Text generation

```text
CLI run / HTTP (CLI serve → ribn-server)
  → ribn-text: bounded preprocessing, chat/tokenization, incremental decoding
  → ribn::driver: one worker, request permits, owned streams, shutdown
  → ribn::Engine: AR lifecycle, scheduling, bounded output mailboxes
  → engine-qwen: QwenCuda / QwenExecution
  → engine-core: prepared batch, continuation state and weight contracts
  → engine-nvidia: physical state, dispatch and kernels
```

`TextOwner::load` assembles the Qwen GGUF/CUDA path. `TextOwner` owns shutdown;
cloneable `TextModel` handles share its token worker. Preprocessing uses a permit-bounded
pool, not a second execution loop. Separate payload and retained-heap envelopes reject
excess string/token capacity, chat/stop-container storage and generation options before
queue insertion, including each decoded-text stop string's spare capacity. Stop-token
capacity plus the runtime copy must also fit the driver's encoded envelope;
model-added stops and actual encoded tokens are checked before publishing a prepared reply.
Prepared replies own one request-local output decoder: model codec, decoded-text stop
matcher and their delivery lease together, constructed under the permit before runtime
enqueue. Pending-byte and emitted-delta bounds are separate from vocabulary-piece bytes;
post-matching deltas obey the same output bound. A matched text stop drops codec/matcher
scratch without finalizing irrelevant buffered text, while retaining the grant through
runtime settlement/error. Other normal/cancelled termination finalizes the codec once;
all termination/error paths free matching state before returning its grant. GGUF's strict
UTF-8 policy and Gemma's whole-run byte fallback are model-local, not branches in delivery
or scheduling.
Fixed job objects are count-bounded; allocator overhead and processor workspace are not
included in the retained-input envelope. These host checks are not a process-RSS bound.
Offline batching consumes a bounded window and
yields ordered per-item outcomes. A slow head bounds lookahead; decoding failure
abandons only its request. See [text semantics](resource-protocol.md#text-application-facade).

`ribn-server` uses Hyper HTTP/1 and Tokio for a loopback-only completion/chat subset.
It bounds connection tasks, body parsing, HTTP admission and response collection/frames;
streaming pulls directly from owned text delivery. Each connection serves one request.
A post-body EOF observer drains unused trailing bytes without retention, so they cannot
hide disconnect from Hyper. Closure abandons transport interest before retryable text
shutdown; pending loading remains owned across cancelled run/cleanup futures.
Owner availability is observable without reserving a permit. Host TCP fixtures exercise
protocol, disconnect/saturation, stalled-peer progress, deadlines, failure and cleanup
retry. A bounded CUDA HTTP parity/cancellation gate and actual CLI shutdown also
passed; client compatibility smokes do not establish competitive performance.
See [serving](serving.md) for external semantics and qualification limits.

The token driver provides fail-fast application preparation permits, encoded-input bounds,
nonblocking bounded delivery, cancellation intent and retryable shutdown. Permits
cover preprocessing, execution and retained delivery, including retirement after
stream abandonment. Admission and AR preparation waits arm one-shot notifications on the capacity-one
wake channel before parking; only device completion retains timed polling.

Direct `Engine` use remains channel-free. It owns one in-flight batch and negotiates
aggregate work before launch. It validates the whole preparation report, refunds omitted
or shortened output credits and parks deferred rows outside runnable queues. All-omitted
offers rebuild for healthy unoffered peers before sleeping. Completion rows still validate
before any logical prefix or output commitment.
Cancellation is intent, not completion. Executor uncertainty faults the owner and
retains retirement ownership. Mailboxes can outlive execution slots; runtime-owned
discard works across that boundary, without frontend orphan queues.

Qwen's GGUF adapter normalizes required normalization epsilon, reference RoPE defaults,
identity scaling, text-equivalent MRoPE sections and supported regular layer catalogs.
Model configuration owns the resulting numerical semantics. GGUF metadata, full-artifact
hashing and bounded tensor readers retain one shared opened source, with independent
reader cursors; pathname replacement/unlink cannot redirect it. The source file must not
be modified in place through any alias while artifact owners/readers live. Qwen CUDA
preparation returns its tokenizer with the executor, so text assembly no longer reopens
the pathname or builds a second tokenizer. Loading rejects disagreement
with the CUDA executor's declared geometry/numerical profile before tokenizer or device
access; the profile is data in `engine-nvidia`, not a list of comparisons here. The pinned artifact's real header and existing bounded CUDA gates passed
re-execution; this does not establish broader artifact or context support.
See [metadata preflight evidence](../benchmarks/runtime-contract.md#qwen-model-metadata-preflight-host-gate).

Qwen reserves continuation for prompt plus output reach at admission. Deferred
admission retains no allocation and retries only after authority publication. It
checks whole-bundle capacity before allocation so a rolled-back partial attempt does
not wake itself. The adapter still translates scalar AR rows into transitional core
batch/state types; it has no second scheduler. Preparation retains one validated backend batch without taking
continuation leases; submit consumes its exact rows, and abandonment preserves old state.

The host continuation contract is page-granular — a model declares a fixed KV page size,
a sequence holds whole pages, and its capacity is `pages * page_tokens` — and production
decode addresses state through a device page table, so appends and attention both resolve
logical tokens to physical ones. Each sequence still owns its own run of pages, so the
table is identity-bound: sharing a resident arena, non-identity pages and page growth are
not implemented. There is no shared-arena page-grant/growth consumer, prefix cache, restore operation
or recomputation preemption. The surviving `engine-core` contracts are the prepared batch
a backend executes, continuation state declarations and accounting, and artifact weight
descriptions; they are not a model or execution-plan layer. A plan/description/capability
layer that nothing read was removed, so a model now describes only what its backend and its
own state accounting consume.
The former duplicate scheduler, serving runtime and `ribn local` frontend have been removed.
Whole decode offers of two through eight rows can now batch across different positive
KV capacities/positions when model geometry/dtype match. Attention remains per-member
on one stream, sharing executor-owned score scratch without padding caches; mixed
prefill/decode and larger offers retain their fallbacks. See the
[qualification](../benchmarks/runtime-contract.md#heterogeneous-capacity-decode-2026-10-16).
The default-off member-descriptor candidate lowers state access through
`nvidia/src/decode/member_state.rs`, with host packet layout in `member_descriptor.rs`
and CUDA binding/guards in `state/descriptor.rs`. NVRTC, device bit-equality and
five-member-kernel memcheck gates pass. The bounded native A/B misses its advancement
gate; asynchronous lifetime and broader performance qualification remain open;
the [pending gates](../benchmarks/runtime-contract.md#member-descriptor-candidate-device-gates-pending)
own numerical, retirement and performance acceptance.
The eight-token same-sequence prefill lane samples its final chunk row by default, so a
prompt that is an exact whole number of chunks never re-runs its last token; partial
prompts, the scalar projection mode and the chunk scan keep the serial final step. See the
[sampled-final measurement](../benchmarks/runtime-contract.md#sampled-final-prefill-chunk-2026-10-17-promoted).
The Qwen-owned serving benchmark exercises the real AR engine and preserves explicit
GEMV variants; its timings are not the legacy baseline's qualification. Compatible decode
subgroups within mixed offers are an experimental, default-off backend option. They
retain original completion order and whole-batch completion. Bounded device gates
passed, but the mixed trace showed no measurable benefit and failed-drain ownership
qualification remains open; ordinary text/CLI loading retains whole-batch selection.

## Model sources, encoder and artifact paths

`ribn-safetensors` validates bounded headers/layouts with the upstream metadata
validator. File-backed artifacts retain one opened file, not encoded payloads in heap;
in-memory artifacts explicitly retain supplied bytes. Independent bounded readers
retain the source after artifact drop or cache eviction. Files must remain unmodified
in place; pathname replacement cannot redirect an opened reader.

`ribn-hf` resolves local config and unsharded/sharded weights without choosing model
semantics. It caches metadata/open files and recognizes legacy repository blobs plus
Hub shared-blob layout v1. Only canonical, well-formed shared blob files beneath a
version-marked cache store are authorized, not arbitrary hub files. This is path
confinement, not repository membership, content authentication or tenant isolation.
Cache config/index and unopened members must stay unchanged throughout loading;
individual descriptor retention is not an atomic package snapshot.

BERT decodes through a fixed 16-KiB byte buffer directly into each final F32 parameter.
Source-cache counts do not bound metadata, caller-retained readers, decoded host weights,
OS page cache or upload staging. The [host comparison](../benchmarks/loader-memory/README.md)
and full E4B artifact probe qualify this I/O path, not Gemma execution or device peaks.

`engine-gemma4` owns dense causal Gemma 4 text configuration, logical parameter
mapping and canonical HF text processing. Its host source preflight validates the whole HF-resolved inventory, retains
BF16 parameter readers, aliases the output head to the input embedding and explicitly
excludes inactive shared-tail projections and non-text weights. Its optional CUDA loader
materializes native BF16 weights through one charged NVIDIA allocation, preserving tied
head identity and retaining partial-load cleanup ownership. All 665 real parameters pass
full byte readback and explicit unload at `1d4a7fb`; this is not an executable decoder.
The subsequent shared allocation owner and BF16 cuBLAS linear binding pass their
[device component gates](../benchmarks/gemma4-reference/README.md#native-bf16-linear-component-gate),
including exact dyadic arithmetic, memcheck and repeated real-package weight readback/unload.
Native grouped BF16 RMS also passes its
[component gate](../benchmarks/gemma4-reference/README.md#native-bf16-rms-component-gate).
NVIDIA owns physical storage/completion; Gemma binds logical projections, epsilon,
learned normalization scales, grouping and owner-only scale-free V. Native BF16
rotary execution passes its
[component gate](../benchmarks/gemma4-reference/README.md#native-bf16-rotary-execution-gate):
the backend prepares cosine/sine factors from the model's FP32 inverse frequencies and
positions, then applies them with eager per-product rounding. A completion-owned
elementwise component now covers residual adds, PLE combine, layer scaling, the gated
product, tanh-GELU and the final soft cap, matching the archived stage encodings
exactly where the operation is exact and within the host audit's bounds for GELU/soft
cap. A three-kernel causal grouped-query attention component matches the archived stock
operands bit-for-bit for local and sliding layers, including decode-against-history.
A host-orchestrated CUDA decoder now binds the components, keeps KV owner caches
device-resident and reproduces the archived greedy chat tokens through the engine, the
text facade and `ribn run-gemma` (CUDA or Metal). Completion-owned device copies append
K/V without a host round trip; initialized-row prefixes bound cache reads and attention.
The decoder materializes only host token/PLE lookup tables and layer scalars from that
same validated source, not a full CPU inventory alongside device projections/norms.
See the [host-capacity qualification](../benchmarks/gemma4-cuda-baseline.md#cuda-host-lookup-capacity-qualification-2026-10-29).
Decode remains synchronous and one token at a time. Qualified RTX 4090/E4B preparation
selects the largest feasible sixteen/eight-row prefill width through 4096 positions,
with eight/scalar fallbacks, preallocated working sets and maximum-reach scratch.
Insufficient scratch for sixteen retains eight or scalar; unqualified devices/geometries
and reaches outside 8–4096 retain scalar execution. Each
owner appends the whole group before attention; per-query masks and full-history shared
donors preserve causal/window semantics. Only the final token reaches the output head,
with device-side last-row extraction for a full group. Executor admission binds to the
prepared cache capacity and rejects invalid token/media/sampling requests locally.
See the [eight-row numerical policy](../benchmarks/gemma4-reference/README.md#bounded-eight-row-cuda-prefill-2026-10-21-utc)
and [sixteen-row qualification](../benchmarks/gemma4-cuda-baseline.md#sixteen-row-full-reach-and-integration-qualification-2026-10-25).
`ribn serve-gemma` loads the same text owner behind the existing bounded HTTP adapter;
its [Metal](../benchmarks/runtime-contract.md#gemma-http-cli-qualification-2026-10-21) and
[CUDA CLI qualification](../benchmarks/runtime-contract.md#cuda-device-requalification-2026-10-22)
cover transport and shutdown, not competitive serving. Sampling and trimmed/mixed-retention
cache lifetime remain unqualified.
A host CPU path now materializes the same 665-parameter BF16 inventory into ordinary
heap buffers and composes a complete decoder step: scaled embeddings, the per-layer
embedding pipeline, all 42 hybrid layers with owner/shared-tail KV, causal and
sliding-window grouped-query attention, the residual/norm/MLP/PLE sequence with the
layer scalar, and the tied head with the final soft cap. Reference tests against real
CPU captures keep linear, head and normalization inside FP32 bounds, GELU inside its
evaluation bound, residual/gated/combine cases bit-exact, rotary bit-exact and
attention matching with zero error; the archived 18-token chat row reproduces all four
greedy tokens. Intermediate prefill uses `GemmaDecode::advance_prefill`, which
certifies completed continuation without returning logits. CPU execution omits final
normalization, the tied projection and soft cap; final prefill/decode retain readout.
CUDA also omits readout for intermediate prefill, using the final native layer
scale's existing completion owner before acknowledging progress. Its
[common-history, uninstrumented and CLI HTTP qualification](../benchmarks/gemma4-cuda-baseline.md#state-only-cuda-uninstrumented-and-http-qualification-2026-10-30)
is bounded to the pinned E4B/RTX 4090 path. Metal retains its qualified full-head
completion fallback; head-free Metal execution is unqualified. The CPU cache keeps
complete per-owner history and masks local spans instead of evicting them. Cache eviction, donor trimming, long-context behaviour,
sampling and competitive performance remain unqualified; this
is a correctness-first host path, not a portable tensor layer or a selected hardware
priority.
An Apple GPU path has started in `engine-metal`: macOS-only Metal compute pipelines
for BF16 linear, grouped RMS, split-half rotary and the decoder's elementwise
operations (add, multiply, scale, tanh-GELU and soft cap), compiled at runtime from
checked-in MSL and driven by one device plus command queue. Linear blocks input rows
on qualified M3 Max even 8–64-row shapes whose inner dimension is divisible by four:
four rows for 32–64-row chunks divisible by four, otherwise two. Separate kernels reuse
weights without changing each output's accumulation order; other GPUs/shapes retain
the original kernel. Causal/sliding grouped-query attention runs as three ordered
kernel dispatches in one command buffer: query-key scores, a per-row softmax reduction,
and weighted values, each preserving the
score/weight rounding and per-output accumulation order. The Gemma model crate reads
its preflighted BF16 source directly into owned Shared Metal buffers using at most
64 KiB of read/copy scratch, preserving encodings and the tied head. Immutable buffers
also provide borrowed CPU embedding rows; preparation constructs neither a full CPU
inventory nor a parameter-sized encoded upload copy. A host-orchestrated
decoder runs the full step: matmuls, normalization, rotary and attention execute on
the GPU while embeddings, remaining elementwise algebra and the soft cap reuse the
validated host primitives. GELU uses a model-owned 128 KiB table prepared from the unchanged
CPU reference, exact over every finite BF16 encoding and still rejecting non-finite
inputs; the CPU decoder remains the independent arithmetic oracle. PLE normalization
can group token/layer rows in one bounded RMS call; larger chunks or packed payloads
retain per-layer normalization. See the [matched qualification](../benchmarks/gemma4-metal-baseline.md#bounded-packed-ple-rms-rescreen-2026-10-23).
Attention input RMS and Q/K/V preparation can also share one bounded serial command,
with the normalized input kept private and no K/V work for shared-tail consumers;
larger chunks/geometries retain separate calls. See the
[attention-chain qualification](../benchmarks/gemma4-metal-baseline.md#attention-input-projection-chain-2026-10-23).
Attention can similarly feed its output projection and RMS in one bounded serial
command; larger chunks, histories and scratch payloads retain separate calls. See the
[attention-output qualification](../benchmarks/gemma4-metal-baseline.md#attention-output-projection-chain-2026-10-23).
Native normalized projections share one serial command buffer and private BF16
intermediates. Query/key projections also rotate in that command, sharing immutable
per-position factors without changing kernels or rounding boundaries. Native FFN input
normalization and gate/up projections likewise share one command. Qualified native
M3 Max/E4B chunks of at most 64 rows now retain the entire main FFN through exact
GELU lookup/product and Down/RMS in that command, borrowing an additional model-owned
128 KiB device table. Integer product encoding preserves CPU subnormal behavior;
other shapes/devices and PLE retain the existing host path. See the
[main-FFN qualification](../benchmarks/gemma4-metal-baseline.md#native-main-ffn-chain-2026-10-27).
An explicit Rust
`Gemma4MetalFfn::ExperimentalGraph` policy instead prepares BF16 Gate/Up pairs at
16/64 rows and Down at 64 rows for the qualified M3 Max/E4B scope, keeping native RMS
and native fallback elsewhere. It reuses immutable BF16 weights; CLI/HTTP and ordinary
decoder loading remain native. SDK/runtime rejection precedes weight loading without
raising the native deployment floor. See the [qualification and limits](../benchmarks/gemma4-metal-baseline.md#model-local-graph-integration-2026-10-23).
One execution mutex owns native and Graph submissions. Before possible enqueue it
retains the complete access envelope; only a successful terminal result publishes outputs.
Possible-enqueue or cleanup uncertainty closes that owner, while explicit retirement
queries completion and unresolved Drop quarantines its resources. Graph's callback,
not an unrelated fence or reference count, establishes physical completion.
Output publication requires successful completion;
remaining activation transfers and CPU elementwise work stay host-orchestrated.
The archived 18-token chat row reproduces all four greedy tokens on
the M3 Max, and again through the token runtime's engine and the text facade with the
real Gemma tokenizer and chat assets. `Gemma4Executor` preflights the whole offered batch,
retains one exact preparation/submission and checks its completion identity. It shares
loaded weights across sequences and commits logical cursor/input/output state only after
all synchronous rows succeed. Failed continuation cannot resume, but teardown can succeed:
paired model creation/completion ownership and explicit sequence close establish when
resources may be destroyed. CPU calls need no device drain; Metal settles the same shared
model owner even if decoder construction failed, and CUDA settles its associated stream
and retained operations before closing caches. Partial
CUDA cache construction stays model-owned, including an unfinished key when value allocation
fails. Failed completion or close preserves the same owner and charges for retry; defensive
Drop quarantines only unresolved cleanup. This does not reset a poisoned model. The
[roadmap](roadmap.md#immediate-bounds-and-lifecycle-repairs) records device qualification
limits. A host test checks successful CPU execution against the decoder's direct path. The decoders keep
complete per-owner history; cache eviction and sampling remain absent, and the
single-sequence Metal decode timing is diagnostic only. Tests on the
M3 Max compare against an independent host reference, including exact dyadic products
and attention spans plus bounded general operands; rotary and elementwise results are
exact against their host references (GELU/soft cap inside declared FP32 bounds). This
is real device evidence for those primitives. Metal, not the Neural
Engine, is the Apple decode target: batch decode is bandwidth-bound and needs dynamic
shapes and custom kernels, while the ANE is reachable only through CoreML. The CPU
decoder remains the correctness oracle; sampling, eviction and competitive performance
remain absent.

Model-local KV range calculations distinguish
persistent suffixes from untrimmed donor spans, with routing/last consumers resolved at
configuration load; they do not prove live state, grants or safe physical lifetimes.
Its bounded text processor uses the package's tokenizer/template/generation metadata;
request-local decoding retains complete byte-fallback runs until a boundary or finalization.
`ribn-text::GemmaProcessor` binds it to owned text delivery, including bounded delayed
byte runs and fallible terminal finalization. Host fixtures substitute token execution
only; host CPU, macOS Metal and CUDA are the executable Gemma backends. A separate stock Transformers CUDA
probe establishes BF16 fit for one 4096-token prompt and 32 predictions, not Ribn fit.
Unknown text semantics or parameters reject;
see the [real-package gate](../benchmarks/gemma4-reference/README.md#ribn-text-source-preflight).

`engine_gemma4::vision` separately selects encoder/projector parameters from the
same retained artifact inventory and runs a synchronous CPU component over canonical
processed patches. Vision configuration is interpreted only on demand; text-only
loading keeps its own parameter set and rejection behavior. This component does not
admit media or hand features to AR execution: snapshot binding, byte admission and
retirement belong to that later integration. Full-checkpoint vision fidelity is not
established; see the
[CPU vision gate](../benchmarks/gemma4-reference/README.md#native-cpu-vision-component).

`engine-bert` owns configuration, parameter mapping, masked bidirectional attention,
embedding/LayerNorm/GELU/pooler semantics and a CUDA executor through `ribn-batch`.
The executor chooses a feasible FIFO prefix and concrete byte envelope. Results own
device storage, a producer completion dependency and the lease covering them together.
Dequeue does not refund the charge. Cancellation and malformed handoff return storage
through the executor's retirement path; uncertain enqueue quarantines storage and its
charge until a drain proves completion. Partial construction and normal result/quarantine
teardown still lack qualified charge retention through allocator-safe reuse/backing release;
post-enqueue tests do not prove a hard physical-memory bound. `ribn-batch` bounds waiting
and result counts, not accepted input bytes, selects FIFO prefixes and lacks model-local
close/shutdown. Pool readiness exists; generic executor blocking has no corresponding
registration. Temporary head pressure can stall otherwise feasible peers. These limitations
are owned by the [repair gates](roadmap.md#immediate-bounds-and-lifecycle-repairs).

`ribn-foundation` owns `BytePool`/`PoolLease`/`AllocationId`, shared readiness,
parameter version labels and scalar metadata. Unused topology/materialization/placement
metadata and the fixture-only operator registry are removed. AR logical state and
encoder reservations use this same byte authority; host tests cover supplied pools,
atomic hybrid grants and sibling readiness. Production Qwen retains a private pool.
Shared CUDA physical retirement and downstream-workspace progress remain unproven.

## Source owners

| Path/package | Responsibility |
| --- | --- |
| `crates/runtime` / `ribn` | AR lifecycle, scheduling, mailboxes and owned token driver |
| `crates/text` / `ribn-text` | Text processor, bounded preparation/decoding, owned facade and offline batching |
| `crates/qwen` / `engine-qwen` | Qwen config, GGUF mapping, CUDA assembly and AR adapter; serving benchmark |
| `crates/nvidia` / `engine-nvidia` | NVIDIA storage, completion, dispatch and kernels |
| `crates/bert` / `engine-bert` | BERT semantics, preparation, executor binding and device retirement |
| `crates/gemma4` / `engine-gemma4` | Gemma 4 text processing, config, source/KV preflight, CPU text decoder and separate vision component, native BF16 CUDA weight/compute bindings |
| `crates/metal` / `engine-metal` | Apple Metal BF16 linear/RMS/rotary/attention/elementwise primitives; macOS-gated, no decoder |
| `crates/core` / `engine-core` | Prepared batch, backend submission, continuation state and weight contracts; no scheduler, plan or model description |
| `crates/gguf` / `engine-gguf` | GGUF artifact access and current tokenizer/chat-template implementation |
| `crates/safetensors` / `ribn-safetensors` | Validated SafeTensors metadata and bounded payload readers |
| `crates/hf` / `ribn-hf` | Local HF-style config/shard resolution and residency |
| `crates/foundation` / `ribn-foundation` | Byte accounting, readiness and consumed version/scalar types |
| `crates/batch` / `ribn-batch` | Non-AR batching, leased results and cancellation |
| `crates/server` / `ribn-server` | Bounded loopback text HTTP transport, health/metrics and lifecycle adapter |
| `crates/cli` / `ribn-cli` | `inspect`, experimental `run`/`serve` (CUDA) and `run-gemma` (Metal or CUDA) |

`tools/check-boundaries.py` checks production dependency direction and retired-path
resurrection. It does not freeze the remaining shared crates against an accepted migration.

## Qualification and missing support

Current Qwen/text and BERT fixture evidence lives in [runtime qualification](../benchmarks/runtime-contract.md),
[encoder qualification](../benchmarks/encoder-qualification.md) and
[prefill qualification](../benchmarks/qwen-prefill-qualification.md). Host fixtures
establish lifecycle contracts, not GPU memory safety or model support. Qwen preparation,
owned-driver, constrained admission, reference/cancellation and migrated linear/GGUF/8–9-row
retirement gates have been re-executed; see the runtime evidence. Broader untested paths
do not inherit that qualification.
GDN chunk scan remains opt-in while its full-model numerical gate fails.

[Foundation evidence](execution-foundation.md) and [composition counterexamples](pipeline-composition.md)
retain useful limitations. They do not qualify a universal execution API.

Not implemented: non-local/authenticated serving, general architecture/source resolution, remote HF
loading, real media processors, dynamic hybrid continuation, executable snapshot
replacement, a qualified non-NVIDIA backend beyond the experimental macOS Metal path,
or distributed execution.
