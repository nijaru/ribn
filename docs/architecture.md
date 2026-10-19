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
excess string/token capacity, chat-container storage and generation options before queue
insertion. Stop capacity plus the runtime copy must also fit the driver's encoded envelope;
model-added stops and actual encoded tokens are checked before publishing a prepared reply.
Prepared replies also own one request-local model decoder, constructed under the
permit before runtime enqueue. Pending-byte and emitted-delta bounds are separate
from vocabulary-piece bytes; the stream consumes its finalizer on normal/cancelled
termination and drops all decoder state on failure. GGUF's strict UTF-8 policy and
Gemma's whole-run byte fallback are model-local, not branches in delivery or scheduling.
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
and CUDA binding/guards in `state/descriptor.rs`. NVRTC and device bit-equality gates
pass, but sanitizer, asynchronous lifetime and performance qualification remain pending;
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
The NVIDIA operation owns physical storage/completion; Gemma binds logical projections.
Model-local KV range calculations distinguish
persistent suffixes from untrimmed donor spans, with routing/last consumers resolved at
configuration load; they do not prove live state, grants or safe physical lifetimes.
Its bounded text processor uses the package's tokenizer/template/generation metadata;
request-local decoding retains complete byte-fallback runs until a boundary or finalization.
`ribn-text::GemmaProcessor` binds it to owned text delivery, including bounded delayed
byte runs and fallible terminal finalization. Host fixtures substitute token execution
only; there is still no executable Gemma model. A separate stock Transformers CUDA
probe establishes BF16 fit for one 4096-token prompt and 32 predictions, not Ribn fit.
Unknown text semantics or parameters reject;
see the [real-package gate](../benchmarks/gemma4-reference/README.md#ribn-text-source-preflight).

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
| `crates/gemma4` / `engine-gemma4` | Gemma 4 text processing, config, source/KV preflight and native BF16 CUDA weights; no model execution |
| `crates/core` / `engine-core` | Prepared batch, backend submission, continuation state and weight contracts; no scheduler, plan or model description |
| `crates/gguf` / `engine-gguf` | GGUF artifact access and current tokenizer/chat-template implementation |
| `crates/safetensors` / `ribn-safetensors` | Validated SafeTensors metadata and bounded payload readers |
| `crates/hf` / `ribn-hf` | Local HF-style config/shard resolution and residency |
| `crates/foundation` / `ribn-foundation` | Byte accounting, readiness and consumed version/scalar types |
| `crates/batch` / `ribn-batch` | Non-AR batching, leased results and cancellation |
| `crates/server` / `ribn-server` | Bounded loopback text HTTP transport, health/metrics and lifecycle adapter |
| `crates/cli` / `ribn-cli` | `inspect`, experimental `run` and `serve` |

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
replacement, a second hardware backend or distributed execution.
