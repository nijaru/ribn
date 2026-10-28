# Ribn benchmark methodology

Benchmarking is part of the architecture, not a release-time exercise. Headline Ribn claims must be reproducible against current incumbents on matched hardware, model, precision, workload, and endpoint semantics.

## Load generators

Use three complementary surfaces:

1. **AIPerf** as the primary backend-neutral online load generator for OpenAI-compatible endpoints. It supports request/concurrency control, arrival-pattern sweeps, latency/throughput/goodput analysis, raw exports, server/GPU telemetry, custom datasets, trace replay, and multimodal endpoints.
2. **vLLM `bench serve`** as an incumbent-native cross-check for vLLM and OpenAI-compatible serving behavior.
3. **SGLang `sglang.benchmark.serving`** as an incumbent-native cross-check for SGLang and OpenAI-compatible serving behavior.

Do not rely on one benchmark client for all conclusions. If the neutral and incumbent-native tools materially disagree, investigate before publishing a comparison.
Include [mistral.rs](https://github.com/EricLBuehler/mistral.rs) when evaluating the
Rust-native engine proposition; it is an existing engine/server, not merely a tensor
framework. Confirm support and pin its deployed configuration like other incumbents.

Current references:

- https://docs.nvidia.com/aiperf/
- https://docs.vllm.ai/en/latest/cli/bench/serve/
- https://github.com/sgl-project/sglang/blob/main/docs/developer_guide/benchmark_and_profiling.md

## Reproducibility rules

Every recorded run must capture at least:

- Ribn/incumbent name, version, and commit/container digest;
- exact model and revision;
- tokenizer and revision;
- dtype/quantization and any model transformation;
- device model, VRAM, driver, CUDA/runtime versions, clocks/power policy when relevant;
- server command/config and environment variables;
- benchmark-client version and exact command/config;
- endpoint type and streaming behavior;
- warmup method;
- input/output length distribution or dataset identity;
- request-rate/concurrency/arrival pattern and seed;
- sampling parameters and EOS behavior;
- benchmark duration/request count;
- any SLO/goodput thresholds;
- raw result artifacts and server logs sufficient to reproduce anomalies.

Do not compare results produced with materially different semantic settings merely because the nominal model name is the same.

## Metrics

Minimum online serving metrics:

- time to first token (TTFT), including p50/p95/p99;
- inter-token latency / time per output token (ITL/TPOT), including tails;
- end-to-end request latency;
- request throughput;
- output-token throughput;
- total-token throughput where meaningful;
- goodput under explicit TTFT/ITL/E2E SLOs;
- error/cancellation rate;
- server CPU utilization and RSS;
- GPU utilization, memory use, and relevant device telemetry;
- scheduler/host overhead when Ribn instrumentation can separate it;
- cold and warm startup time for startup comparisons.

Throughput without latency/SLO context is not sufficient for a serving claim.
Report freshly computed input tokens separately from cache hits; reused tokens are not
new device work. For multi-turn/branching workloads, also measure completed jobs per
device and job-latency distributions under declared objectives. Keep tool delays,
prefix identity, compaction and branch/join structure in client-side replay; the engine
is not an agent orchestrator. Structural synthetic replay does not establish speculative
acceptance, MoE routing or output quality; use content-real cases for those comparisons.
The [target design's research basis](../docs/inference-engine-design.md#research-basis)
records the primary sources and their scope.

## Qualification evidence

- [Runtime contract](runtime-contract.md) — runtime, driver and text-facade device gates.
- [Encoder qualification](encoder-qualification.md) — BERT encoder numerical parity with an
  independent Hugging Face reference.
- [Qwen prefill qualification](qwen-prefill-qualification.md) — chunked same-sequence prefill.
- [Runtime alignment](runtime-alignment/README.md) — host and CUDA frontend overhead.
- [Loader memory](loader-memory/README.md) — sharded BERT host-loading peak and bit preservation.
- [Gemma 4 reference](gemma4-reference/README.md) — pinned E4B CPU numerical inputs/tensors
  and staged native decoder qualification.
- [Gemma 4 CUDA baseline](gemma4-cuda-baseline.md) — same-artifact native/stock/prepared
  direct-call and bounded loopback serving diagnostics; not quality-equivalent competitive
  acceptance or a task-quality result.

## Unqualified historical tools

Do not use `incumbent-serving-bench.py` or the custom `incumbent-qwen38/*stream_bench.py`
clients for new competitive claims. They can count requested tokens or content events
rather than authoritative generated tokens, and do not consistently reject incomplete
or error-only streams. Their historical outputs are not retroactively validated by a
successful process exit. Use the neutral/native clients above and inspect terminal
outcomes and actual usage.

`run-qwen-serving-sweep.sh` resolves the executable from Cargo's build messages,
including custom `CARGO_TARGET_DIR` and target configuration. Build failure prevents
execution. Native diagnostics remain separate from matched HTTP/EOS-enabled comparisons.

**Do not run `quality-probes/probe.py` on an ordinary host.** It evaluates/executes
model-produced Python without isolation; subprocess timeouts are not a sandbox. Its
marker-based code checks are not an independent task-quality gate. It is retained as
historical experiment source, not a qualified evaluation tool.

## Initial workload matrix

Start with deterministic synthetic workloads because they isolate engine behavior, then add realistic traces/datasets.

| Workload | Purpose |
|---|---|
| batch-1 / low concurrency | decode latency and host overhead floor |
| short 256 -> 128 | interactive short chat |
| balanced 1k -> 256 | general serving baseline |
| prefill-heavy 4k -> 128 | prompt processing pressure |
| decode-heavy 256 -> 1k | long generation / reasoning pressure |
| long-context 8k+ -> 256 | KV/state and chunked-prefill behavior |
| mixed lengths | scheduler robustness under heterogeneity |
| repeated shared prefix | prefix-cache/reuse behavior |
| multi-turn continuation / branching with tool delays | retained-cache economics and job-level latency/throughput |
| request-rate sweep | saturation curve and goodput frontier |
| concurrency sweep | queueing and scheduling behavior |

Add modality-, MoE-, speculation- and distribution-specific cases when the corresponding
capability exists. Client-side text trace replay does not require an engine stage graph,
agent runtime or prefix cache; with reuse disabled it establishes the no-cache baseline.

### Agent-session workload qualification

Keep open-loop arrivals for saturation/fairness and add independent **closed-loop**
replays for agent progress. A sequential session's next request follows the prior
terminal response plus the recorded tool/user delay; preserve planned arrivals
rather than hiding generator lag. Include concurrent sibling/subagent branches,
cache-unfriendly prompt edits or compaction, cancellation, restart and failures.
Shared session identity does not imply equal prompts or serialized requests.

Use at least two cohorts: long, growing-context coding (state capacity and
tool-delay retention) and shared-system-prompt workflows (cross-session reuse).
The ~100k input/~500 output token Modal Kimi workload is a vendor example, not
a Ribn default or a reason to exceed the qualified model context. Scale controlled
traces to available hardware and add real long-context traces only after device
qualification. Treat MLPerf Agentic Inference as a version-pinned methodology
reference, not an automatic certification of current results.

A replay manifest should pin model/processor/tokenizer/adapter snapshot,
sampling/stop behavior, optional session/parent IDs, request dependencies and
timing, length distributions, prefix identity/isolation and whether future turns
come from recorded text or actual prior model output. Where justified and
authorized, retain exact output token IDs for speculation studies; decoded and
re-tokenized strings are not equivalent. Do not collect private prompts,
responses or token IDs by default. Content-free hashed/structural replay may
test scheduling and cache patterns but cannot establish speculative acceptance,
model quality, or actual agent task success.

Report both:

- **Request/replica:** TTFT/ITL/E2E tails, interactive decode rate, computed
  versus valid reused prompt tokens by tier, admission/queue delay, physical
  device/host peak and residency, transfer/replay costs, rejections/cancellation.
- **Session/deployment:** inference wait separately from tool idle, completed
  job latency and jobs/device-time, per-session fairness, interactivity versus
  aggregate SLO goodput, cache-miss penalties and restart/scale-out tails.

When distributed serving exists, compare random, load-only, session-affinity
and cache-plus-load routing on identical inputs/hardware. Include overlapping
requests sharing a session hint, skewed context sizes, cache churn, scale-up and
stale/unavailable cache metadata. Require task-level quality checks for lossy
state or draft changes; trace-level replay alone is not an accuracy gate.
See the [conditional target](../docs/inference-engine-design.md#multi-replica-inference-placement-conditional)
and [delivery criteria](../docs/roadmap.md#conditional-multi-replica-placement).

## Native Qwen serving sweep

The Qwen-owned `qwen_serving_bench` example exercises the production AR engine with
the pinned Qwen GGUF path. It replaces the legacy core-runtime harness; historical
measurements do not qualify the migrated harness or imply directly comparable timing:

```text
ENGINE_QWEN_GGUF=/path/to/Qwen3.8-27B-UD-Q4_K_M.gguf \
  bash benchmarks/run-qwen-serving-sweep.sh
```

Defaults are 32 output tokens at concurrency 1, 2, 4, and 8. Override them with `TOKENS` and `CONCURRENCIES`, for example:

```text
TOKENS=64 CONCURRENCIES="1 4 8" \
  ENGINE_QWEN_GGUF=/path/to/model.gguf \
  bash benchmarks/run-qwen-serving-sweep.sh
```

The script builds `engine-qwen`'s example and records revision, environment and one
raw log per concurrency under `benchmarks/results/`. Compatible multi-row decode uses
backend batch lanes; mixed/single rows stay per-row. Explicit `--gemv=scalar|warp|int-dot`,
`--prefill-chunk`, fixture prompts and six divergence probes remain available.
Preparation timing now includes loading/staging/kernels; execution timing includes
AR admission and event observation. The effective batch-token limit is printed.
This fixed-arrival sweep is not representative online serving evidence. Only the
separately documented [device cases](runtime-contract.md#mixed-arrival-native-qwen-gate)
are qualified; neither old timings nor a smoke trace qualify the complete sweep matrix.

### Mixed-arrival native traces

Replay a finite open-loop workload through the same AR engine:

```sh
WORKLOAD=benchmarks/workloads/qwen-mixed-arrivals.json \
QUEUED_REQUESTS=4 CONCURRENCIES="1 2 4" \
CONTINUATION_CAPACITY_BYTES=524288000 \
TTFT_SLO_MS=5000 ITL_SLO_MS=200 E2E_SLO_MS=10000 \
ENGINE_QWEN_GGUF=/path/to/Qwen3.8-27B-UD-Q4_K_M.gguf \
bash benchmarks/run-qwen-serving-sweep.sh
```

Trace mode defaults to the qualified warp GEMV and production same-sequence prefill
chunking; the old burst sweep retains its scalar-script/serial-prefill baseline.
The example accepts `--prefill-chunk=off` for a serial comparison.
`--grouped-decode` opts into the experimental compatible-row subgroup candidate;
bounded device gates passed, but the mixed trace showed no measurable benefit and
failed-drain qualification remains open. Omit it for the default whole-batch policy.
JSON records the choice; it does not enable paging or change admission.

A workload is a JSON array. Each row specifies a nondecreasing `arrival_ms`,
positive `output_tokens`, a `prompt_fixture` path relative to the workload file,
and optional `prompt_tokens` selecting a nonempty fixture prefix. Fixture files use
the three-line artifact/prompt/reference format in
[`crates/qwen/tests/fixtures`](../crates/qwen/tests/fixtures/README.md).
The trace supplies request lengths; do not combine it with burst prompt/token flags.

`--concurrency` bounds active requests; `--queue-capacity` bounds additional requests
(default zero). `--continuation-capacity-bytes` optionally constrains the model's private
continuation authority. Rejected arrivals are counted, never delayed or retried.
TTFT and E2E start at the **planned arrival**, so host dispatch lateness is included
rather than erased; submit-lag statistics expose that lateness. A blocking engine step
can delay actual submission, so this is not an independent network load generator.
ITL measures successive host-observed token events. Successful-request distributions
report nearest-rank p50/p95/p99; failures and rejections are separate. SLO goodput counts
completed requests satisfying every supplied TTFT, maximum-ITL and E2E threshold.
A one-token request has no ITL interval.

The script writes JSON with configuration, workload and per-request outcomes alongside
raw logs and environment metadata. Direct invocation uses `--workload=PATH`,
`--result-json=PATH` (a new file, never overwriting inputs/results) and optional
`--ttft-slo-ms`, `--itl-slo-ms`, `--e2e-slo-ms`.
Inputs are preloaded; model load and client-bookkeeping initialization are outside the
arrival timeline. There is no execution warmup. Reservation capacity and device-free
snapshots are **not** a measured physical peak.

Direct example invocation also accepts `--dispatch-diagnostics` (off by default).
It prints a bounded snapshot to stderr, including on measurement/shutdown failure;
trace JSON includes it under `dispatch_diagnostics`, or `null` when disabled. Rust
callers can enable the same observer with `QwenCuda::observe_dispatch()` before
moving the executor into an engine. The observer retains counters, not device state.

The snapshot separates submitted backend rows/tokens from fully enqueued model-step
work: decode lane calls by width (index 1 is scalar decode), same-sequence prefill
chunks and row-wise fallback reasons. Grouped-policy selection does not imply a
physical group was formed. Work/route totals exclude failed submissions, which may
have launched partial work; `enqueue_failures` preserves that limitation. Successful
enqueue is not device completion, logical output usage or numerical acceptance.

`host_submit_nanos` covers routing, enqueue and completion-event recording, including
failed cleanup, but excludes initial validation and diagnostic publication. It is
not device time, host/device idle time or a sum of per-row durations. Counters saturate;
publication uses one fixed-size snapshot per observed attempt, with no per-step log.
[Bounded device probes](runtime-contract.md#bounded-device-mapping-and-trace-2026-10-16-dc4aed0)
now cover specific lane/chunk histories; broad instrumentation overhead remains unqualified.
Keep observation settings matched across performance comparisons. Device phase costs,
queue age and physical-memory peaks still need separate instrumentation/qualification;
these native diagnostics are not exposed by the HTTP endpoint.

The checked-in eight-request trace is a heterogeneous smoke workload, not steady-state
traffic or competitive evidence. It excludes HTTP and tokenization. Use matched endpoint
load generators above for serving comparisons. Its initial device evidence and the
heterogeneous-capacity regression are in the [runtime contract](runtime-contract.md#mixed-arrival-native-qwen-gate).
The [qualified capacity-independent decode lane](runtime-contract.md#heterogeneous-capacity-decode-2026-10-16)
now batches whole decode offers of two through eight rows with matching KV model layout,
even when capacities/positions differ. Each request retains its own allocation; no
continuation padding is introduced. Mixed prefill/decode and larger offers still use
their existing fallbacks. The v0 diagnostics reason `incompatible_kv` replaces
`nonuniform_kv`; historical snapshots retain their original keys.

The 2026-10-07 subgroup comparison used three `8427f63` baseline runs followed by three
`c0c0242` runs with `--grouped-decode`, otherwise identical flags including
`--print-tokens`. At four active/four queued requests and 500 MiB continuation capacity,
median throughput was 13.735 versus 13.736 tok/s; TTFT/ITL/E2E p95 were effectively
unchanged and both earned zero goodput at 5000/200/10000 ms objectives. All runs completed
8 requests/136 tokens without rejection or failure; every request's tokens matched.
This is **no measurable improvement**, not competitive evidence or an isolated bottleneck.
The [six-run CSV](qwen-trace/2026-10-07-grouped-comparison.csv) and median-throughput
[baseline](qwen-trace/2026-10-07-grouped-baseline.json)/[candidate](qwen-trace/2026-10-07-grouped-candidate.json)
JSON preserve outcomes and configuration. Runs were process-cold, not filesystem-cache-cold,
without warmup, on RTX 4090/driver 615.71.09/Rust 1.98.0. Pre/post-load free-memory
snapshots were identical; they are not peak-memory evidence. Use separate Cargo target
directories for compared worktrees: shared example paths can be overwritten while
another worktree's fingerprints still report fresh.

## Qwen multi-token prefill gate

Before selecting the experimental same-sequence Qwen prefill path in serving, follow
[`qwen-prefill-qualification.md`](qwen-prefill-qualification.md). It records the
four-layer diagnostic and full 64-layer ignored parity gates plus a matched serial
versus chunk-size 2/4/8 timing matrix for the RTX 4090. The qualified scope and selected default are recorded there; new geometry or variants
need their own device gates. CPU CI and CUDA-feature compilation are not performance
or numerical evidence.

## Audited endpoint runner

`endpoint.py` runs **AIPerf 0.13.0** against an already running
`/v1/completions` endpoint, then audits its native aggregate, per-request and raw
exports. It supports synthetic fixed or variable nominal prompt lengths with fixed
output caps, greedy sampling with EOS enabled, concurrency-limited traffic and
constant/Poisson rate control. It also supports the finite single-turn text traces
below. Output caps are not expected generated-token counts. Variable output caps,
media, multi-turn traces and chat are not implemented here.

Set `workload.input_tokens_stddev` to a nonnegative integer: zero keeps the fixed
nominal length; positive values use AIPerf's seeded positive-normal length generator.
For example, mean 256 and standard deviation 128 exercise heterogeneous input reaches
without changing generation caps. These are distribution parameters, not a context
ceiling or exact token-count promise: inspect the generated prompts and server usage,
and stay within the independently qualified context scope. The manifest seed, pinned
client/tokenizer and exported effective distribution preserve the run configuration;
they do not establish identical wall-clock arrivals or a representative production mix.
`output_tokens_stddev` must remain integer zero.

The runner needs Python 3.11+. AIPerf 0.13.0 needs Python 3.11–3.13; use a separate
environment rather than installing it into an unsupported interpreter:

```sh
uv venv /tmp/ribn-aiperf --python 3.13
uv pip install --python /tmp/ribn-aiperf/bin/python 'aiperf==0.13.0'
cp benchmarks/manifest.example.toml /tmp/ribn-baseline.toml
# Fill in the manifest, including model/tokenizer revisions, artifact SHA-256,
# server provenance, quality scope and independently chosen objectives.
python3 benchmarks/endpoint.py run /tmp/ribn-baseline.toml \
  --aiperf /tmp/ribn-aiperf/bin/aiperf \
  --output benchmarks/results/my-baseline
uv pip freeze --python /tmp/ribn-aiperf/bin/python \
  > benchmarks/results/my-baseline/client-environment.txt

# Re-audit the preserved evidence without contacting a server:
python3 benchmarks/endpoint.py validate benchmarks/results/my-baseline
```

With `HF_HUB_OFFLINE=1` or `TRANSFORMERS_OFFLINE=1`, AIPerf 0.13.0's tokenizer
loader resolves `model.tokenizer` as a cached Hub repository ID, not a local directory.
Use the repository ID and exact `model.tokenizer_revision`; verify that they resolve
to the intended byte-pinned snapshot before issuing load.

The output directory must be new; existing evidence is never replaced by `run`.
It retains the manifest/hash, exact client command, process exit code, log, raw
exports and, when auditing completes, `validation.json`. Early failures may leave
only partial evidence; a missing validation report never counts as success.
Enable `run.gpu_telemetry` on the measured device
host when collecting GPU telemetry; sampled memory is not a physical-peak proof.
Server command, engine/model revisions, artifact hash and hardware are caller
assertions, not discovered or verified against the remote server. Preserve server
logs/configuration independently. The runner does not launch or stop the server.

Validation reconciles the effective client configuration, record identities,
profiling/warmup counts and usage totals. `attempted_requests` counts issued profiling
records, including failures; it is not a ledger of every planned arrival. A successful stream must have a finish
reason, coherent integer usage and `[DONE]`, without late errors or extra data.
Cleanly truncated streams can pass AIPerf itself, so ordinary summaries and HTTP
trace metadata alone are insufficient. Failed requests remain in issued-work
accounting; warmup errors are separate and invalidate the audit. Missing exports,
nonzero client exit, incomplete runs and zero validated successes cannot pass.
The audit also accepts MLX-LM 0.32.2's `chat.completion` label on an empty-choice,
usage-only frame after the completion finish. This is a vendor wire-format
exception, not support for chat content on this endpoint; usage and terminal-order
checks still apply.
AIPerf treats replies without visible content, including immediate EOS, as errors;
the audit preserves that classification rather than silently changing it.

`protocol_valid` validates protocol/accounting and the per-request clocks below.
`audited_request_timing_metrics` lists the reconciled native metrics; general
`performance_metrics_audited` remains false. This is not numerical acceptance, a
matched comparison, a SOTA result or proof of physical-memory bounds. Server usage remains a server
assertion; normalized raw SSE exports are not byte-exact wire captures. Readiness
probes are explicitly disabled to avoid unrecorded inference; start with a ready
endpoint. Zero warmup requests does not establish cold execution.

`prompt_cohort` fingerprints the actual profiling prompt multiset, including failed
requests and excluding warmup. Identical seeds and nominal lengths alone do not
establish identical inputs. Compare non-null fingerprints to check prompt identity;
request IDs and record order do not affect the digest, but repeated prompts retain
their multiplicity. A missing, non-string or non-UTF-8-encodable prompt makes `sha256`
null and lists its request ID in `unavailable_request_ids`, never silently dropping it.
An otherwise properly recorded failure can still pass protocol accounting with an
unavailable cohort; `protocol_valid` is not a matched-workload certificate.

The versioned `sha256-utf8-prompt-multiset-v1` algorithm hashes each decoded prompt's
UTF-8 bytes without Unicode normalization, counts the resulting lowercase hex
digests, and sorts `(digest, count)` pairs. It hashes their compact ASCII JSON
(`separators=(",", ":")`), prefixed by `ribn:profiling-prompt-multiset:v1` and a NUL byte.
The report contains no prompt text, but hashes do not anonymize private prompts.
Matching prompts does not establish matching artifacts, tokenizers, tokenized lengths,
generation policies, arrival order/schedules or output quality; check those separately.

Optional `[slo]` values must be positive; omit unset objectives. Their boundaries are:

- `ttft_ms`: request start to first nonempty content chunk, not first headers or an empty event.
- `last_content_ms`: request start to last nonempty content chunk (AIPerf's `request_latency`).
  It excludes later empty finish, usage and `[DONE]` frames, so it is **not stream-completion
  E2E**. The misleading `e2e_ms` manifest key is no longer accepted.
- `itl_ms`: the first-to-last content window divided by server-reported output tokens
  minus one. Without per-chunk token counts, that assumes one token in the first chunk;
  bundled/withheld output and suppressed stop tokens can change the interpretation.
  It is undefined below two output tokens: the per-request metric must be absent,
  not zero. It is not an individual-token or chunk-gap tail. Inspect
  `inter_chunk_latency` separately.

These clocks start at actual client request initiation, not planned arrival. With a
positive request rate, AIPerf skips scheduled ticks when its concurrency ceiling is full
and can reanchor a late schedule. The count target covers issued requests, not suppressed
ticks; exported credit times are stamped after admission to the concurrency gate. Report
nominal rate separately from actual starts and achieved throughput. `planned_arrivals_audited`
remains false: neither a complete planned-arrival ledger nor planned-arrival SLO goodput
can be recovered from these exports.

The audit checks integer, nonnegative raw `start_perf_ns` and response `perf_ns`
clocks: responses cannot precede the request or regress, but frames received together
may share a timestamp. It reconciles each successful request's native TTFT and
`request_latency` against first/last nonempty content. It also checks per-request
`inter_token_latency` against that content window and actual terminal usage, including
absence below two output tokens. Exact effective settings disable per-chunk usage,
so the client's assumed first-chunk count is one. Arithmetic occurs in nanoseconds
before the pinned nanosecond-to-millisecond conversion, without a fitted tolerance.
This audits the client's average-ratio formula, not actual token delivery intervals.
`request_timing_ns` retains first-content, last-content and `[DONE]`-receipt
durations from actual request start for validated profiling successes. DONE receipt
is not HTTP EOF.

`issued_load` reports the configured rate/concurrency separately from actual profiling
start span and minimum/median/maximum start intervals, in nanoseconds. It includes
failed issued requests and excludes warmup; a single start has zero span and null
interval statistics. These clocks belong to the single-host client, not a distributed
clock synchronization protocol. They do not reveal suppressed arrivals.

For configured SLOs, `slo_attainment` reconciles each successful request's native
`good_request_count`, the profiling aggregate count and `good_request_fraction`.
All configured thresholds must pass, inclusively. The comparison uses internal
nanoseconds after converting manifest thresholds from milliseconds; comparing
rounded displayed metrics can change a boundary decision. An ITL requirement below
two output tokens is undefined, not a zero-latency pass: the native counter must be
absent and the request does not qualify. Failed attempts remain in the fraction's
denominator; warmup does not. The report separates met, violated and undefined
profiling outcomes, and returns no attainment audit without configured SLOs or
when protocol/metric validation fails. These are issued-request attainment counts,
not planned-arrival or product SLO acceptance.

For finite-phase throughput, divide actual successful request/token totals by the
aggregate `end_time - start_time`, including drain—not `benchmark_duration`, which
ends at content. Aggregate rates (including SLO goodput) and percentiles remain
unaudited; a study must cross-check those separately. `performance_metrics_audited`
stays false. The per-request ITL arithmetic audit does not establish its aggregate
tails or the first chunk's real token count.

Zero warmup omits AIPerf's positive-only warmup option; it does not establish cold execution.
The [client timing/Metal gate](runtime-contract.md#gemma-endpoint-client-qualification-2026-10-21)
checks these bounded semantics, not aggregate metric accuracy or load-generator capacity.
See the pinned [metrics definitions](https://docs.nvidia.com/aiperf/v0.13.0/reference/ai-perf-metrics-reference).

**Raw exports contain prompts and responses.** Use public/synthetic data, keep
artifacts outside version control, and do not enable private-content collection
merely to obtain a benchmark.

Host verification (the full unit suite needs NumPy; integration also needs the
pinned AIPerf environment):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s benchmarks -p 'test_*.py'
# Explicit integration gate: real AIPerf plus a CPU-only HTTP fixture, no model/GPU.
PYTHONDONTWRITEBYTECODE=1 /tmp/ribn-aiperf/bin/python -m unittest discover -s benchmarks -p '*_integration.py'
```

Leave `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` unset for this integration gate:
the pinned client's offline resolver treats its generated local tokenizer directory
as a Hub repository ID and fails during setup. Normal local-path loading needs no
model or tokenizer download.

The integration gate exercises early EOS, silent truncation, late JSON errors,
missing usage and rate-controlled partial overload through the actual client and
CLI. A seeded variable-input case verifies that different tokenized prompt lengths
reach the endpoint and that actual usage excludes warmup rather than substituting
nominal lengths or output caps. Prompt fingerprints are independently reconciled
against actual profiling payloads, retaining failed requests and excluding warmup.
Delayed empty/terminal frames check content versus DONE clocks and the usage-based
ITL divisor rather than visible chunk count or output cap. A single-token response
checks ITL absence and undefined SLO attainment; without an ITL requirement, its
other latency SLOs remain measurable. Joint SLO rejection and partial overload check
attainment counts and the all-attempt fraction. A concurrency-capped rate case checks
actual start intervals against raw timestamps. These checks do not establish device
qualification, aggregate rate/percentile accuracy or load-generator capacity.

## Timestamped text replay

For a finite declared arrival cohort, use `kind = "fixed_schedule"` rather than
a rate target whose ticks can be suppressed at the concurrency ceiling. Keep the
other manifest sections and replace its workload section with:

```toml
[workload]
kind = "fixed_schedule"
trace_file = "schedule.jsonl"
output_tokens_mean = 8
output_tokens_stddev = 0
request_count = 4
warmup_requests = 0
max_concurrency = 1
request_timeout_seconds = 10
```

Place `schedule.jsonl` beside that manifest:

```jsonl
{"session_id":"req-0","timestamp":0,"text_input":"hello","output_length":8}
{"session_id":"req-1","timestamp":50,"text_input":"hello hello","output_length":8}
{"session_id":"req-2","timestamp":100,"text_input":"hello hello hello","output_length":8}
{"session_id":"req-3","timestamp":150,"text_input":"hello hello hello hello","output_length":8}
```

Run the same entry point with a new output directory:

```sh
python3 benchmarks/endpoint.py run /path/to/manifest.toml \
  --output /tmp/ribn-fixed-replay --aiperf /tmp/ribn-aiperf/bin/aiperf
```

This small example is a smoke, not a representative serving workload. The initial
trace subset accepts exactly these four row fields: unique nonempty session IDs
using ASCII letters, digits, `_.:-`; sorted nonnegative integer millisecond offsets
starting at zero; nonempty UTF-8 text; and a uniform output cap matching the manifest.
The row count must match `request_count`. The sidecar must have a simple ASCII
`.jsonl` filename, not a path. Warmup must be zero, and synthetic length, rate and
arrival-pattern controls are rejected rather than ignored. Cache hashes, payloads,
messages, media, delays and multi-turn sessions are outside this subset.

The runner validates the whole trace before launching the client, copies its bytes
beside the copied manifest, records its SHA-256, and binds the effective native file
configuration to that snapshot. Revalidation rejects changed bytes or mismatched
session IDs, turns, prompts and caps—including failed attempts. `trace_cohort`
reports the checked input offsets and attempt coverage; `issued_load` still reports
actual starts, with no configured request rate. Keep the snapshot and exports at
their recorded location when revalidating.

The CPU replay gate exercises the actual runner at concurrency four and one, proves
that the single-slot streams do not overlap, and retains a partial-overload cohort.
It reconciles native relative send-lag scalars using the client's float64 epoch-offset
column, including its low-bit loss before centering—not integer-exact quantiles.
The client's 500 ms `replay_sched_degraded` threshold is a diagnostic, not Ribn's SLO.

The runner now sets `TZ=UTC` only in its client subprocess environment and records
`client_timezone` in `invocation.json`; it does not change the caller's timezone.
AIPerf's summary exports the profiling lifecycle's wall-clock start as `start_time`,
converted through floating-point seconds to a timezone-naive, microsecond-resolution
string. This is not the exact monotonic schedule origin.

For UTC-pinned runs, `trace_cohort.reported_phase_clock` reports an epoch-origin
interval with a conservative ±2,000 ns serialization uncertainty, restricted to
1970 through epochs below 2³² seconds. `dispatch_offset_ns_bounds` subtracts that
interval and each declared offset from the recorded request-start epoch, including
failed attempts. Signed results are retained, not clamped. Snapshots without a
recorded client timezone return a null diagnostic rather than guessing local time;
malformed UTC origins or unsupported recorded timezones reject revalidation.

These bounds cover **origin serialization only**, not alignment between worker and
controller clock anchors. `monotonic_alignment_audited` remains false. Do not add
these epoch-coordinate offsets to monotonic `request_timing_ns` or treat them as
qualified planned-arrival latency/SLO goodput. Native relative lag still cannot
detect uniform lateness, and dispatch-based TTFT excludes queued time.
`planned_arrivals_audited` and `performance_metrics_audited` remain false.

A private CPU fixture in `gemma4-phase-origin/20261028T085747Z` observes the native
lifecycle anchors and shifts scheduler targets by 2³⁰ ns without changing trace
offsets. Both retained four-request delay cases place the captured epoch origin
inside the reported interval, including one HTTP-overload failure. Actual monotonic
send lateness exceeds 1.07 s, yet native centered-lag p99 is 5.87/0.99 ms and its
500 ms degradation flag stays zero. Subtracting the common delay leaves those
centered scalars unchanged. Epoch-derived offsets differ from the directly observed
monotonic offsets by 12.58–23.79 µs, beyond the serialization interval: clock alignment
is a separate prerequisite, not a tolerance to widen.

The intended three-case private cohort remains incomplete. Its control exits 1
before issuing requests because the private harness incorrectly enabled the pinned
client's offline resolver for a generated local tokenizer. That failure is retained;
only the two unstarted delay cases use the documented local-path environment, with
child-only outbound Python connections blocked. No control is repeated or filled.
Installed dependency files remain unchanged; the observation/delay hooks stay private.
The separate stock-client integration suite exercises the ordinary runner without
these hooks. This is CPU client qualification, not device-performance evidence.

## Ribn endpoint client configuration

The initial [protocol subset](../docs/serving.md) has no model-list endpoint or EOS
bypass. Use AIPerf inference readiness, request terminal usage, and keep model stops
enabled. These commands reproduced the small [compatibility smokes](runtime-contract.md#initial-text-http-qualification-2026-10-13);
increase workload scope and add repeats/SLOs for a representative comparison.

```sh
# The pinned tokenizer snapshot must already be cached for offline operation.
HF_HUB_OFFLINE=1 aiperf profile \
  --model ribn --tokenizer Qwen/Qwen3.8-27B \
  --tokenizer-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --endpoint-type completions --endpoint /v1/completions --streaming \
  --url http://127.0.0.1:18000 --isl 16 --isl-stddev 0 --osl 8 --osl-stddev 0 \
  --concurrency 1 --request-count 4 --num-prompts 4 --warmup-request-count 1 \
  --random-seed 42 --use-server-token-count --extra-inputs '{"temperature":0}' \
  --connection-reuse-strategy never --wait-for-model-mode inference \
  --wait-for-model-timeout 30 --output-artifact-dir /tmp/ribn-aiperf-smoke \
  --no-auto-plot --ui none --export-http-trace

# TOKENIZER_PATH is that pinned local snapshot. The random dataset sets
# ignore_eos=true; keep the explicit override so this smoke matches the
# recorded policy.
HF_HUB_OFFLINE=1 vllm bench serve \
  --backend openai --base-url http://127.0.0.1:18000 --endpoint /v1/completions \
  --model ribn --tokenizer "$TOKENIZER_PATH" --dataset-name random \
  --random-input-len 16 --random-output-len 8 --num-prompts 2 \
  --max-concurrency 1 --request-rate 1 --seed 42 \
  --extra-body '{"temperature":0,"ignore_eos":false}' --disable-tqdm \
  --save-result --result-dir /tmp/ribn-vllm-client-smoke
```

A native client exit of zero can still mean every request failed. Check exported
completion/failure counts before using timing fields. These commands measure Ribn
through two clients, not a vLLM engine or a matched incumbent result.

## Baseline run shape

For an OpenAI-compatible completion endpoint, a current AIPerf synthetic example is conceptually:

```text
aiperf profile \
  --model <model> \
  --endpoint-type completions \
  --endpoint /v1/completions \
  --streaming \
  --synthetic-input-tokens-mean <isl> \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean <osl> \
  --output-tokens-stddev 0 \
  --url localhost:8000 \
  --request-count <n>
```

Use the installed AIPerf version's `--help` as the authority for exact flags before a run; preserve the version and full command with the result.

Current vLLM cross-check shape:

```text
vllm bench serve \
  --backend openai \
  --base-url http://127.0.0.1:8000 \
  --model <model> \
  --dataset-name random \
  --input-len <isl> \
  --output-len <osl> \
  --request-rate <rps> \
  --num-prompts <n>
```

Current SGLang cross-check shape:

```text
python -m sglang.benchmark.serving \
  --backend <backend> \
  --base-url http://127.0.0.1:8000 \
  --model <model> \
  --dataset-name random \
  --max-concurrency <n> \
  --num-prompts <n>
```

This SGLang shape is **not a qualified Ribn invocation**. In the pinned
[v0.5.22 client](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/benchmark/serving.py),
source review of its `sglang-oai` and `vllm` completion backends (not the vLLM CLI)
found:

- They send `best_of=1`, outside Ribn's supported request fields. Ribn rejects it.
- They index `choices[0]` on each non-DONE frame, including usage frames. Ribn's
  terminal usage frame has `choices: []` and cannot pass that parser.
- Output length starts at the requested cap and updates only from usage on
  nonempty content frames. Without those counts, early EOS can inflate reported
  output tokens and distort TPOT. Separate retokenized counts do not establish
  the actual generated token count.
- Latency ends at the last nonblank frame, including `[DONE]`, rather than the
  last-content clock audited for AIPerf. HTTP stream EOF can mark a request
  successful without checking for `[DONE]`.

For Ribn, `--disable-ignore-eos`, an explicit `--model`, and
`--ready-check-timeout-sec 0` avoid the default EOS bypass and unsupported
`/v1/models` readiness probe, but **do not resolve the gaps above**. Qualify a
compatible native client before using its results; do not omit terminal usage or
force generation to the cap to make a cross-check pass. These are source-reviewed
client limitations, not an executed cross-check or a claim about SGLang's engine.

Preserve exact installed versions and commands rather than treating these
examples as frozen APIs.

## Result layout

Store benchmark outputs outside version control by default. A run should be reproducible from a small checked-in manifest plus external/raw artifacts:

```text
results/
  <date>-<engine>-<model>-<workload>/
    manifest.toml
    client/
    server/
    telemetry/
    summary.json
```

`results/` should remain ignored unless a small curated result is intentionally committed for a regression or published benchmark.

## First success criterion

Before adding broad Ribn features, establish a repeatable baseline suite that can answer:

> On one model and one RTX 4090, where do current vLLM and SGLang spend time and host resources, and what performance envelope must Ribn match or beat to be credible?
