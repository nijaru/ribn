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
The [target design's research basis](../docs/inference-engine-design.md#research-basis-reviewed-2026-10-13)
records the primary sources and their scope.

## Qualification evidence

- [Runtime contract](runtime-contract.md) — runtime, driver and text-facade device gates.
- [Encoder qualification](encoder-qualification.md) — BERT encoder numerical parity with an
  independent Hugging Face reference.
- [Qwen prefill qualification](qwen-prefill-qualification.md) — chunked same-sequence prefill.
- [Runtime alignment](runtime-alignment/README.md) — host and CUDA frontend overhead.
- [Loader memory](loader-memory/README.md) — sharded BERT host-loading peak and bit preservation.

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

The checked-in eight-request trace is a heterogeneous smoke workload, not steady-state
traffic or competitive evidence. It excludes HTTP and tokenization. Use matched endpoint
load generators above for serving comparisons. Its initial device evidence and the
heterogeneous-capacity regression are in the [runtime contract](runtime-contract.md#mixed-arrival-native-qwen-gate).

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
exports. It supports synthetic fixed nominal prompt lengths/output caps, greedy sampling
with EOS enabled, concurrency-limited traffic and constant/Poisson rate control.
Output caps are not expected generated-token counts. Mixed-length/trace datasets
and chat are not implemented in this runner yet.

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
profiling/warmup counts and usage totals. A successful stream must have a finish
reason, coherent integer usage and `[DONE]`, without late errors or extra data.
Cleanly truncated streams can pass AIPerf itself, so ordinary summaries and HTTP
trace metadata alone are insufficient. Failed requests remain in offered-work
accounting; warmup errors are separate and invalidate the audit. Missing exports,
nonzero client exit, incomplete runs and zero validated successes cannot pass.
AIPerf treats replies without visible content, including immediate EOS, as errors;
the audit preserves that classification rather than silently changing it.

`protocol_valid` validates protocol/accounting only. Performance metrics are
explicitly marked unaudited: it is not numerical acceptance, a matched comparison,
a SOTA result or proof of physical-memory bounds. Server usage remains a server
assertion; normalized raw SSE exports are not byte-exact wire captures. Readiness
probes are explicitly disabled to avoid unrecorded inference; start with a ready
endpoint. Zero warmup requests does not establish cold execution.

Optional `[slo]` values must be positive; omit unset objectives. AIPerf's
`inter_token_latency` is per-request average TPOT, not an individual-gap tail.
Inspect `inter_chunk_latency` separately; chunks may contain multiple tokens.
See the pinned [metrics definitions](https://docs.nvidia.com/aiperf/v0.13.0/reference/ai-perf-metrics-reference).

**Raw exports contain prompts and responses.** Use public/synthetic data, keep
artifacts outside version control, and do not enable private-content collection
merely to obtain a benchmark.

Host verification:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s benchmarks -p 'test_*.py'
# Explicit integration gate: real AIPerf plus a CPU-only HTTP fixture, no model/GPU.
PYTHONDONTWRITEBYTECODE=1 /tmp/ribn-aiperf/bin/python benchmarks/endpoint_integration.py
```

The integration gate exercises early EOS, silent truncation, late JSON errors,
missing usage and rate-controlled partial overload through the actual client and
CLI. It does not establish device qualification or load-generator capacity.

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

# TOKENIZER_PATH is that pinned local snapshot. The random dataset otherwise
# forces ignore_eos=true; explicitly override it to match the server's policy.
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

Again, preserve exact installed versions and commands rather than treating these examples as frozen APIs.

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
