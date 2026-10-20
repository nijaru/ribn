# Inference engine design

Status: accepted target, revised 2026-10-20; not implemented support or a SOTA claim.
[Resource protocol](resource-protocol.md) owns execution/resource transitions;
[roadmap](roadmap.md) owns delivery order, open implementation choices and gates;
[architecture](architecture.md) maps current source. Public v0 interfaces are unstable.

## Scope and success criterion

Ribn targets a state-of-the-art, idiomatic Rust, **server-first** inference engine,
initially competitive with vLLM, SGLang and similar engines. Throughput within latency
objectives, memory efficiency, model-support velocity, hardware breadth and reliability
are joint goals. Correct ownership enables that destination; it is not a substitute
correctness-only niche. Rust alone is neither a performance advantage nor a kernel strategy.

Qwen GGUF/CUDA and the development RTX 4090 are qualification workloads, not the
product boundary. Establish a matched competitive baseline now, expand qualified
workloads and support, and report gaps. Pin hardware, artifacts, quality policy,
competitor versions, protocol semantics and arrival distributions. Neither internal
speedups nor a feature checklist establishes competitiveness or optimality.

Server-first means sustained concurrency, fairness, overload, cancellation and bounded
memory. It does not require IPC. CLI, Rust, Python and HTTP reuse model behavior and
execution ownership rather than implementing separate schedulers. Minimal serving and
competitive measurement must not wait for prefix caching, broad model coverage or a
kernel-language migration. The qualified contiguous path is a valid initial baseline.

All v0 APIs may be replaced. Remove superseded paths rather than adding compatibility
shims; retain correctness evidence and explicitly scoped qualified fallbacks.

## Four boundaries

1. **Artifacts and model definition:** validated configuration, processors, parameter
   interpretation, operations and continuation semantics; no serving lifecycle.
2. **Prepared execution:** owning backend-specific weights, storage, kernels, workspaces
   and completion dependencies; lower accepted logical work into actual device work.
3. **Execution policy:** specialized AR, batch/encoder and later iterative/training
   runtimes; own admission, progress, fairness and logical commitment.
4. **Application access:** operation-specific owned requests/results, bounded submission,
   cancellation, diagnostics and frontend adapters.

```text
CLI / Rust / Python / HTTP
            |
operation-specific model handle + processor
            |
   +--------+----------------+----------------+
   |                         |                |
AR token runtime       batch / encoder    iterative runtime
continuation           pooling / scores    diffusion / other
   |                         |                |
   +------ model/backend prepared execution --+
                         |
             storage / devices / kernels
```

A single-runtime model crosses one application boundary to one mutable execution owner.
Do not insert pass-through orchestrator/worker/executor layers. A shallow composition
owner is justified by actual cross-runtime dependencies, placement, bounded handoff,
cancellation or output ordering, not merely by a model having multiple components.

`crates/runtime` (`ribn`) remains the AR runtime, not the universal inference API.
Encoders, speech encoder-decoder models and iterative media generation have different
units of progress. Share mechanisms where concrete consumers demonstrate reuse, not
scheduling policy or fictional token requests.

## Application API and serving

Users load or serve a model and invoke a supported operation. Internal runtime selection
is not a mandatory task-default ceremony.

- CLI: `ribn run <model>`, inspection and benchmark utilities.
- Server: `ribn serve <model>` with a documented protocol subset, streaming,
  health/readiness, metrics, bounded overload, disconnect cancellation and shutdown.
- Rust/Python: cloneable loaded handles and owned results/streams; operation-specific
  generate/chat/embed/score/transcribe methods as support exists. Preserve direct,
  channel-free runtime access.
- Use operation-appropriate protocols. Chat, embeddings, reranking, audio and media
  generation are not interchangeable payloads. Reject unsupported operations/options.

The initial HTTP adapter now serves the documented loopback text subset; wider
operations and general loading remain targets. Keep serving a thin adapter over the
owned text facade, not a second execution loop.

Python access has two roles: existing protocol clients for a separately served model,
and eventual native bindings for offline batches, evaluation, data generation and
custom Python servers. Native bindings must wrap the same Rust model handles,
admission, streams, cancellation and shutdown owners, not add another scheduler.
Python remains optional for Rust/CLI use. A Python API does not imply execution of
arbitrary Python model code or automatic support for every Transformers architecture.

Bound connections, request-body bytes, retained input/options, preprocessing, application
requests and buffered responses separately. Payload lengths do not bound spare capacity,
message containers or stop lists. Acquire admission before expensive processing and
validate retained allocations before queue insertion. Slow clients
cannot occupy unbounded transport buffers or prevent cancellation and healthy-peer
progress. Network/authentication/fetching policy stays above model execution.

Separate load/resource configuration, scheduler policy, request generation settings and
serving settings. Prefer ordinary typed structs/errors to typestate machinery or
`infer(Any) -> Any`. Async submission must not block an async executor with tokenization
or GPU polling. An owned stream yields ordered deltas and one terminal outcome;
abandoning it never asserts device completion. Direct access need not create a worker.

## Execution and scheduling

### One owner, bounded overlap

One mutable owner serializes authoritative state transitions, not all CPU and GPU work.
The target separates persistent sequence state from each submission's input ordering.
Backend request rows remain stable while live; batch indirection gathers the rows needed
by a prepared kernel lane. Row identity is not allocation lifetime. Preemption/reuse needs
completion-safe reclamation, and derived metadata follows its executable snapshot.

First remove avoidable per-step allocation, metadata rebuilding and synchronization.
Then qualify bounded preparation/execution overlap when timelines show a benefit.
Each outstanding submission owns immutable transfer staging, table versions, output
credits, workspace and completion dependencies. A later submission cannot overwrite an
older buffer. Same-sequence dependencies, ordered commitment, EOS/cancellation and
failed-drain retirement must be explicit before allowing dependent work ahead of host
commitment. Start at depth one; deeper execution is a qualified backend capability,
not an async API guarantee. See the [overlap contract](resource-protocol.md#bounded-overlap-and-submission-ownership).

### Logical policy and physical batching

Use one logical progress frontier per sequence and a bounded token budget across prefill,
decode and replay. Phase remains an execution distinction, not a reason for unrelated
ownership paths. Give decoding latency protection while aging prefills/replays so neither
starves. Select policy and chunk sizes from measured TTFT/ITL/goodput tradeoffs; a token
budget is not a universal estimate of bytes or execution time. Once physical costs are
stable enough to calibrate, compare measured-time chunk budgeting against token budgeting
on identical lanes and latency/fairness objectives; no cost formula is selected here.

The model/backend negotiates aggregate feasible ranges and lowers them into a bounded
set of compatible physical lanes. Packed/ragged work must amortize weight reads across
heterogeneous requests where qualified kernels permit it. Request lengths, positions and
reachable capacities are data, not inherent model incompatibility. Actual layout, dtype,
state shape and kernel constraints still determine compatibility. Never remove a check
without qualifying the replacement addressing path or pad every request to maximum
context merely to preserve a dense kernel.

A mixed logical batch is not evidence of mixed physical batching. Measure scheduled versus
actually batched rows/tokens, per-phase execution time, host dispatch gaps and fallback
frequency. Compare policy changes independently of lane changes, including workloads
where homogeneous batching wins. Keep the current scheduler as a baseline, not an API
to preserve or a policy to replace without evidence.

### Backend-native optimized execution

Prioritize scalable prefill GEMM, tiled attention, packed/ragged work and persistent
metadata/workspace before micro-optimization. Evaluate qualified vendor/external kernels
alongside custom kernels; compare end-to-end costs, including format conversion,
repacking, workspace, startup and native Rust integration. A Python wrapper's existence
is not proof of a usable native dependency. GGUF-specific encodings need their own gates.

Resolve static compatibility at preparation where practical. Choose shape/device-specific
lanes without a universal tensor IR, operator registry, plugin ABI or compiler project.
Graph capture and fusion are backend-owned variants: stable storage, capture/replay
lifetimes and numerical scope are part of qualification. Tune in the actual eager or
captured execution mode; isolated kernel winners may lose in a dependent serving loop.

For non-AR batching, separate execution selection from result ordering. Bound owned input
bytes before acceptance, permit bounded bypass/bucketing of temporarily infeasible heads
with aging, and attach real readiness to every temporary wait. Model-local closure and
retryable cleanup must not close sibling pool authorities. The current BERT FIFO/count-only
runtime does not yet meet these application-access and healthy-peer-progress targets.

Speculation is a later bounded draft/verify/commit operation, not a flag on ordinary
decode. It needs target sampling semantics, rollback for every continuation component,
charged draft/verification storage and accepted-prefix-only delivery/usage. Permit zero
drafting; optimize committed tokens per complete round and SLO goodput, not acceptance
rate alone. Deterministic token replay and distribution-preserving sampling are distinct
contracts. No drafter or algorithm is selected by this design.

Learned target-conditioned drafters and workload-specific fine-tuning are candidate
experiments, not the default. Preserve exact generated token IDs (rather than
decoded/re-tokenized text), tokenizer/template identity and the target snapshot
in *authorized* training/evaluation traces. Compare full-round latency and net SLO
goodput on real workload cohorts with draft weights, intermediate buffers,
tentative state and verification charged. Draft quantization may reduce cost,
but correct target verification and full-state settlement remain mandatory.

## Continuation resources and prefix reuse

Dynamic storage, pressure progress and prefix-cache policy are separate deliveries.
Full-reachable reservation remains the current qualified Qwen fallback, not the destination.

- Initial Qwen paging uses bounded resident K/V arenas, matching the qualified common-base
  addressing primitive. Charge backing while resident; separately grant reusable slots.
  Reserve recurrence, workspace, staging and retirement overlap as well as KV occupancy.
  Do not grow arenas online until allocation/retirement and transient peaks qualify.
- Negotiate growth for the whole accepted range before launch. Abandonment releases only
  new unsubmitted ownership; partial enqueue retains old state and new growth until safe
  settlement. Declarations describe shape/bounds, never reservations.
- Use completion-safe recomputation as the initial live-pressure mechanism, with survivor
  growth ahead of victim readmission. Prove a survivor can finish within the granted
  continuation/workspace envelope. Waiting alone cannot break all-active-growth deadlock.
  Protection requires output readiness: completion-safely preempt/reselect a survivor
  whose consumer stalls, rather than letting it strand healthy victims. Retain one
  bounded replay history; do not duplicate output, usage or sampling.
- Prefix reuse initially uses snapshot-scoped immutable content blocks and sparse exact
  recurrent checkpoints. The index identifies content; live allocation owners establish
  residency. Full attention needs retained prefix coverage, sliding attention its required
  window, and recurrence its matching state. Reuse advances only to a boundary valid for
  every model-required component; physical page sizes need not be universal.
- Restore recurrence into private writable state. Stop before the final prompt sampling
  token and execute the suffix for logits; cached output/logit state is deferred.
- Cache admission has a byte budget and a measured benefit requirement. Initially retain
  selected completed boundaries, not periodic snapshots everywhere. Compare no reuse,
  simple bounded retention and overlap-aware placement. Cache eviction reclaims safe
  occupancy; it does not refund resident arena bytes or preempt a live sequence.

Treat transient device-memory waste, state encoding and host cache tiers as
separate candidate optimizations. Measure intermediate/copy/repacking peaks and
physical backing, not just nominal bytes per cached token. Lossy KV or recurrent
state quantization needs independent full-model numerical and task-quality gates,
with its numerical policy represented in cache compatibility. Host restoration
must beat recomputation or improve overload tails after transfer, memory and
retirement costs. Neither a CPU tier nor compressed state is a prerequisite
for the first qualified serving path.

The pinned hybrid's 149.6 MiB recurrent state per sequence versus 16.8 MiB KV at
257-token reach makes dense checkpoints unattractive. It does not prove that hybrid
reuse improves short-request concurrency. Measure checkpoint/copy peaks, retained
capacity, computed tokens, replay cost and net goodput on branching and continuation
traffic. A cache may decline a hit or retain nothing.

Approximate recurrent reconstruction, such as recent suffix-replay research, is not exact
prefix reuse. Keep it outside the initial production path. Any later experiment requires
an explicit numerical/quality policy, independent task-level acceptance, separate cache
identity and opt-in selection; existing numerical error grants no additional error budget.

## Model support and loading

For a supported architecture and backend, a compatible new checkpoint should load
without Rust changes. A new architecture or custom model should normally add model
semantics, processor/parameter mappings, backend bindings, registration and qualification,
not edit scheduling, cancellation or frontend lifecycle. A new execution regime or
unsupported device operation may require a shared-contract change; do not promise an
arbitrary-model plugin interface or hide missing support behind a registry entry.

Model families own configuration, parameter interpretation, processors, special tokens,
operations, continuation semantics and backend execution. Registration is central selection,
not family branches throughout lifecycle/routing. Prove this boundary with a real dense
second decoder, then a real VLM/processor; do not wait for all continuation optimizations.
The surviving `engine-core` contracts are the prepared batch, continuation state declarations
and weight descriptions its model and backend actually consume; the plan/description/capability
layer that nothing read has been removed. What remains is not the universal model layer.
Replace hard-coded geometry and obsolete bridge contracts with their actual model/backend
owners during those integrations, without creating another catch-all crate.

GGUF and SafeTensors expose artifact data. HF package resolution owns source/revision/files,
not model semantics. Target byte-aware streaming materialization with bounded concurrent
reads/conversion/upload staging; a shard-count LRU does not bound clones or transient peak
bytes. Loading and device preparation need failure owners before advertising memory bounds.
Tokenizer/processor metadata is part of the executable configuration, not frontend guesswork.

Raw media and fetching/preprocessing stay above token scheduling. Actual models determine
staged versus coupled execution; prompt-positioned VLM features may need just-in-time
encoder/AR preparation. Network types do not expose backend tensors. Outputs carry their
own representation and bounds. An optional external reference path can aid bring-up and
numerical comparison, but Python is not a mandatory production runtime.

## Composition, hardware and later training

[Composition counterexamples](pipeline-composition.md) do not establish a universal stage
graph. Concrete pipelines must prove device dependencies and shared-pool downstream
workspace progress before claiming composition support.

Physical layouts and kernels remain backend-specific across NVIDIA, AMD, Apple/Metal and
CPU. A second real backend must prove any portability claim; the design selects no new
hardware priority. External systems allocate fleet resources; Ribn executes within its
grant. Collectives, sharding, expert/pipeline/context parallelism and disaggregated transfer
require implementations and qualification, not topology metadata. Local use needs no
synthetic distributed layers. Host cache tiers and remote transfer follow measured demand,
not the assumption that every cache should be hierarchical.

An accepted request pins one coherent executable snapshot: weights/adapters, processor
and numerical policy. Start drain-and-replace; overlapping versions need explicit extra
capacity and derived-state compatibility. Exact snapshot identity is the initial cache
policy, not a numeric label pretending to own storage.

### Multi-replica inference placement (conditional)

Model-aware request placement is a deployment concern distinct from fleet/GPU
allocation and a single replica's AR scheduler. When real multi-replica workload
justifies it, an optional inference router can balance **valid reusable-prefix
coverage** against projected uncached prefill, decode/queue work, cache pressure
and replica health. Keep useful affinity for sequential turns, spread highly
concurrent branches or hot sessions when queueing overwhelms reuse, and prefer
new sessions on newly started replicas rather than needlessly relocating warm
sessions. Compare random, load-only, fixed-affinity and combined routing instead
of assuming a fixed scoring formula wins.

Affinity depends on cache topology: locally resident reuse may avoid prefill, while a
shared accessible cache can reduce that advantage. Include retrieval/transfer cost and
queue imbalance rather than assuming every cache hit requires a particular replica.

Client session/parent identifiers are advisory locality hints, not proof of equal
content or authorization. Reuse requires exact compatible tokenized prefix,
processor/model/adapter/numerical snapshot, all required hybrid components and
tenant/trust namespace. Replica cache reports are bounded and eventually stale;
verify residency before use, expire state on eviction/restart/replacement and
fall back to safe recomputation on a miss. Do not retain raw prompts just to route.

Replicas remain authorities for admission, continuation, cache validity,
cancellation and retirement. An optional deployment router owns its placement,
retries and bounded queues; external infrastructure, possibly Archon, owns
provisioning and resource leases. No distributed coordinator, global event bus
or IPC is mandatory for the single-model local fast path. Cross-replica cache
transfer and placement protocols need independent workloads and qualification.

Training is a later execution system. Share artifact/device/storage/completion/collective
mechanisms where real implementations align. Gradients, autograd, activations, optimizer
scheduling and recovery remain outside inference policy. Trainer-to-rollout publication
and Python tensor/result interchange precede a real forward/backward/update experiment;
that experiment decides useful shared mathematics rather than prebuilding a tensor framework.

## Qualification and current evidence

Measure throughput within explicit latency objectives, TTFT/ITL/E2E distributions, fairness,
overload/cancellation, host cost, loading and physical peaks. Count cached versus freshly
computed input separately. Add multi-turn branching/tool-delay traces and job latency /
completed jobs per device where appropriate; the client owns agent orchestration. Structural
synthetic replay is insufficient for speculative acceptance, MoE routing or quality claims.
Use content-real cases for those risks and common input histories for numerical comparisons.
For agentic workloads distinguish per-request latency from closed-loop session
progress, actual compute from cache hits, and individual interactivity from
aggregate throughput. Compare deep growing-context coding and shared-prompt
workflow traces with tool delays, branches, modified prefixes and bursty arrivals.
Prerecorded future-turn replay measures serving, not autonomous task success.
Use real content and independent outcome checks when evaluating model quality;
do not extrapolate a vendor's 100k-token profile to an unqualified device/context.

Ordinary numerical correctness does not promise batch-invariant results or repeatable
stochastic sampling. A future optional reproducibility policy needs separate qualification
across batch shapes, sequence order and supported execution variants, with stable per-sequence
sampling identity and a measured performance cost. It is not the compulsory fast path.

Arithmetic-preserving changes require exact checks; reordered algorithms need independently
justified metrics/tolerances. Current long-prompt reference disagreement is a fidelity item,
not permission to widen optimized-variant gates. Follow [model qualification](../.agents/skills/model-integration/SKILL.md).
Expand to encoder-decoder speech, iterative/non-AR text/media, structured outputs, logprobs,
adapters, realtime sessions and wider hardware through concrete operation contracts, not a
universal execution API. The [roadmap](roadmap.md) defines the next delivery gates.

### Research basis

These sources justify mechanisms and experiments, not Ribn performance or universal support.
Development-head features may differ from releases; pin deployed versions for comparisons.
Refresh this evidence when implementing the corresponding slice or when upstream contracts change.

- [vLLM MRV2 at `3e8e136`](https://github.com/vllm-project/vllm/blob/3e8e1364ff41330fb667b2d76d6645176d61fdd0/docs/design/model_runner_v2.md): persistent rows separate from step ordering, owned async staging and explicit graph lifecycle. The document explicitly remains incomplete and not rigorously qualified.
- [vLLM hybrid manager at the same revision](https://github.com/vllm-project/vllm/blob/3e8e1364ff41330fb667b2d76d6645176d61fdd0/vllm/v1/core/kv_cache_manager.py): group-valid cache hits, last-token replay, growth/lookahead reservations and progress headroom. One byte authority does not require one page geometry.
- [SGLang unified cache at `46a97e5`](https://github.com/sgl-project/sglang/blob/46a97e55400d871ba05537f21ed603321801da2d/python/sglang/srt/mem_cache/unified_radix_cache.py): full/sliding/recurrent component validity and transfer owners. Its component framework is not a Ribn requirement.
- [TensorRT-LLM overlap at `55279c1`](https://github.com/NVIDIA/TensorRT-LLM/blob/55279c109fe6115979dbf3f99102b513711f99f8/docs/source/features/overlap-scheduler.md) and [agentic trace study](https://github.com/NVIDIA/TensorRT-LLM/blob/55279c109fe6115979dbf3f99102b513711f99f8/docs/source/blogs/tech_blog/blog27_Evaluating_Agentic_Serving_with_Trace_Replay_and_Job_Level_Metrics.md): bounded work ahead of CPU result processing and job-level measurement. Reported GPU/workload results do not transfer to Ribn.
- [FlashInfer v0.7, 2026-09-22](https://flashinfer.ai/2026/09/22/flashinfer-v07.html): native kernel evaluation and execution-mode-aware tuning; neither Rust integration nor GGUF encoding compatibility is established.
- [Sarathi-Serve, v3](https://arxiv.org/abs/2403.02310v3): chunked prefill and decode batching under latency constraints, not evidence that one policy wins every workload.
- [Marconi, MLSys 2025](https://arxiv.org/abs/2411.19379v3) and [sparse exact checkpoints, v1](https://arxiv.org/abs/2605.05219v1): reuse-value/memory economics and overlap-aware placement. The latter's layer-group prototype is not full-serving qualification.
- [SuffixReplay, 2026-09-27 v1](https://arxiv.org/abs/2609.33477v1): approximate state reconstruction with reported quality retention below 100%; an experimental numerical policy, not transparent exact caching.
- [Modal, 2026-09-23](https://modal.com/blog/trillion-tokens-trillion-parameters): long-prefix coding-agent traffic; reducing temporary HBM, draft and KV quantization, host overflow caching, speculation and load-aware placement. Its Kimi/Blackwell results are workload-specific evidence, not default policies or transferable Ribn speedups.
- [NVIDIA Dynamo KV-aware router, v1.5.1](https://docs.dynamo.nvidia.com/dynamo/v1.5.1/knowledge-base/concepts/system-architecture/kv-aware-routing) and [session-aware inference, 2026-10-08](https://pytorch.org/blog/session-aware-agentic-inference-with-nvidia-dynamo/): cache overlap plus projected load; agent IDs inform locality, but do not establish resident state. A shared event bus is not automatically required.
- [MLPerf Agentic Inference, 2026-07-08](https://mlcommons.org/2026/07/agentic-inference-for-mlperf-inference/): closed-loop multi-turn pacing, coding and shared-prompt cohorts, interactivity/throughput frontiers and accuracy gates. Pin actual reference rules and version before comparison.
- [vLLM prefix-caching design](https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md): prefix hashes include model-dependent extras and tenant salts; this evolving design informs identity/isolation, not a substitute for Ribn's hybrid continuation proof.

#### DeepInfer and Recipe (reviewed 2026-10-20)

DeepInfer at [`774ce244`](https://github.com/deepseek-ai/DeepInfer/tree/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01)
is a specialized, offline DeepSeek-V4.1-Flash release, not its complete production serving
system or a matched cross-engine performance result. The following findings inform existing
slices; they do not select a new engine architecture or reorder delivery:

- [Worker](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/worker/worker.py) and [sequence remapping](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/csrc/engine/zos.cpp): device-resident sampled tokens, stable sequence mapping, position correction and explicit events permit dependent work ahead of host result processing. This informs the later dependent-overlap contract, not just the first independent-sequence increment; “zero overhead” is not synchronization-free execution.
- [CPU verification policy](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/csrc/engine/mtp_varlen.hpp) and [GPU confidence allocator](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/layers/tilelang_ops/cuda/schedule_mtp_varlen.py): choose a bounded verification budget on the CPU, then rank cumulative acceptance estimates using newer GPU-resident confidence within that budget. Reported workload-derived throughput estimates include a low-concurrency loss; they are not Ribn gains or an online-serving comparison.
- [Deployment Q&A](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/doc/inference_deployment_qa.md): fitted prefill-time quotas and shared 3FS block KV plus host SWA storage describe production, not the released scheduler/cache service. The [released scheduler](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/csrc/engine/scheduler.hpp) uses token/context limits. Approximate bounded SWA replay is not exact restoration; the Q&A's 128-token replay and the README's 140-token decoder compact setting describe different boundaries. Cache topology informs routing economics, not a mandatory storage hierarchy.
- [Reproducibility scope](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/README.md): deterministic sampling requires NVIDIA, batch-invariant execution, fixed sequence seeds and variable-length verification disabled. Reproducibility and sampling correctness remain distinct qualification policies.
- DeepSeek Recipe at [`935817c`](https://github.com/deepseek-ai/deepseek-recipe/tree/935817c66f5d6e64838f419692545f06787e485f): [encoding](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/deepseek-recipe-encoding/src/tokenizer.rs) distinguishes structural token IDs from ordinary text; the [incremental parser](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/deepseek-recipe/src/stream/output_parser.rs) retains parser state and emits argument fragments. Token provenance matters: text-only parsing cannot distinguish a literal marker from a structural token. This informs future model-owned protocol/tool parsing, not a replacement for Ribn's processor boundary or a defense against semantic prompt injection.
