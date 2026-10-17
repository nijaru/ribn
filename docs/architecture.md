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
  → engine-core: transitional batch/plan/logical-state contracts
  → engine-nvidia: physical state, dispatch and kernels
```

`TextOwner::load` assembles the Qwen GGUF/CUDA path. `TextOwner` owns shutdown;
cloneable `TextModel` handles share its token worker. Preprocessing uses a permit-bounded
pool, not a second execution loop. Separate payload and retained-heap envelopes reject
excess string/token capacity, chat-container storage and generation options before queue
insertion. Stop capacity plus the runtime copy must also fit the driver's encoded envelope;
model-added stops and actual encoded tokens are checked before publishing a prepared reply.
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
Model configuration owns the resulting numerical semantics. Loading rejects disagreement
with the CUDA executor's fixed geometry/numerical profile before tokenizer or device
access. The pinned artifact's real header and existing bounded CUDA gates passed
re-execution; this does not establish broader artifact or context support.
See [metadata preflight evidence](../benchmarks/runtime-contract.md#qwen-model-metadata-preflight-host-gate).

Qwen reserves continuation for prompt plus output reach at admission. Deferred
admission retains no allocation and retries only after authority publication. It
checks whole-bundle capacity before allocation so a rolled-back partial attempt does
not wake itself. The adapter still translates scalar AR rows into transitional core
batch/state types; it has no second scheduler. Preparation retains one validated backend batch without taking
continuation leases; submit consumes its exact rows, and abandonment preserves old state.

Block-table attention reads exist in the backend, but production Qwen remains
contiguous. There is no paged write/growth consumer, prefix cache, restore operation
or recomputation preemption. The surviving `engine-core` contracts are transitional;
its duplicate scheduler, serving runtime and `ribn local` frontend have been removed.
Whole decode offers of two through eight rows can now batch across different positive
KV capacities/positions when model geometry/dtype match. Attention remains per-member
on one stream, sharing executor-owned score scratch without padding caches; mixed
prefill/decode and larger offers retain their fallbacks. See the
[qualification](../benchmarks/runtime-contract.md#heterogeneous-capacity-decode-2026-10-16).
The Qwen-owned serving benchmark exercises the real AR engine and preserves explicit
GEMV variants; its timings are not the legacy baseline's qualification. Compatible decode
subgroups within mixed offers are an experimental, default-off backend option. They
retain original completion order and whole-batch completion. Bounded device gates
passed, but the mixed trace showed no measurable benefit and failed-drain ownership
qualification remains open; ordinary text/CLI loading retains whole-batch selection.

## Encoder and artifact paths

`ribn-safetensors` validates artifact bytes once and exposes indexed tensor views.
`ribn-hf` resolves local config and unsharded/sharded weights without choosing model
semantics. It lazily caches whole owned shards, evicting before opening incoming
shards under a resident policy. Entry limits do not bound total peak bytes: caller
clones and transient read/validation copies can retain storage. BERT loads with one
resident source shard and moves each decoded parameter into its host owner without
a duplicate tensor cache; host weights and device-upload staging still consume memory.
See the [loader-memory comparison](../benchmarks/loader-memory/README.md).

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
| `crates/core` / `engine-core` | Transitional execution/batch/plan, backend, state and weight contracts; no scheduler |
| `crates/gguf` / `engine-gguf` | GGUF artifact access and current tokenizer/chat-template implementation |
| `crates/safetensors` / `ribn-safetensors` | Generic validated SafeTensors views |
| `crates/hf` / `ribn-hf` | Local HF-style config/shard resolution and residency |
| `crates/foundation` / `ribn-foundation` | Byte accounting, readiness and consumed version/scalar types |
| `crates/batch` / `ribn-batch` | Non-AR batching, leased results and cancellation |
| `crates/server` / `ribn-server` | Bounded loopback text HTTP transport, health/metrics and lifecycle adapter |
| `crates/cli` / `ribn-cli` | `inspect`, experimental `run` and `serve` |

`tools/check-boundaries.py` checks production dependency direction and retired-path
resurrection. It does not freeze transitional crates against an accepted migration.

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
