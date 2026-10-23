# Gemma 4 Metal: baseline and execution experiments

Measured 2026-10-20 on the 40-core M3 Max, 128 GiB memory, macOS 26.6.2 (25G83).
This initial calibration compares **direct decoder calls**, not HTTP endpoints,
request scheduling or SLO goodput. It identifies a remaining latency gap; it does not
establish equal task quality or complete the roadmap's competitive serving baseline.

## Pinned inputs and implementations

- Ribn `ff01225`, Rust 1.98.0 release build: current qualified Metal kernels, exact
  prepared GELU table, synchronous execution and full host K/V histories.
- Published wheels: `mlx-lm==0.32.2`, `mlx==0.32.4`, `mlx-metal==0.32.4`,
  `numpy==2.5.4`, `transformers==5.20.1`, Python 3.13.11. No upstream source patches.
  MLX-LM's [v0.32.2 source](https://github.com/ml-explore/mlx-lm/tree/93e6d4a6447b3080861985947f0820e76fcb957d)
  includes [Gemma 4 text execution and cache selection](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/models/gemma4_text.py).
  The installed text-model file matches that release's SHA256
  `73ffa75e9069ce7b2f864ccd8f0267bb80c5b5d4e2b73102b8a2a93fe03f936d`.
- The identical, unconverted `google/gemma-4-E4B-it` BF16 snapshot
  `ee0ef6023621cff504d758262d4e04895a5af4a2`, owned by the
  [reference manifest](gemma4-reference/model.toml). Both implementations materialize
  **665 active BF16 parameters, 14,926,026,836 bytes**, with the embedding/head tied.
  No weight or K/V quantization, adapter, speculative decoder or media input is enabled.
- Archived chat token IDs, cycled/truncated to lengths 1/8/18/32/64, followed by the
  same three teacher-forced tokens. Neither tokenizer nor chat-template execution is
  timed. Model loading is excluded; each trial starts with empty continuation state.

MLX uses its native model and cache classes, fused primitives and graph evaluation.
Ribn uses its own native kernels and host orchestration. The arithmetic, retention and
submission strategies are not identical. MLX's model constructs all-row output logits
before the caller selects the last row; backend pruning of that graph was not profiled.
Ribn explicitly projects only the last prompt row. This is an API-level comparison, not
an isolated matmul or equal-dispatch-count experiment.

The MLX clock covers model construction of the forward graph, evaluation of the last
logit row **and cache state**, and a copied host array containing all 262,144 BF16 logits.
The Ribn clock likewise ends after its owned host logit vector returns. Validation,
argmax, reference-file reads and trace printing are outside both clocks. The Ribn probe
uses an enclosing autorelease pool; MLX retains its own runtime's lifetime behavior.
The timed model processes ran serially, without this session's builds or tests
overlapping them. Other host activity was not controlled.

## Numerical calibration is not an equivalence budget

Both backends were separately checked against all **12** archived stock CPU BF16 rows:
the 18-token chat and actual 511/513-token text prefixes, each with three continuations.
Every call used the reference's exact submitted IDs, preserving common histories even
if a prediction could differ. Shape and finiteness were checked before reducing errors.
The reference tensor SHA256 was
`46878e19c907f6cb4d2e88ae6077238465612d3bc1aab8441f57bfb4b83772f3`.

| Against stock CPU BF16 | Ribn Metal | MLX |
| --- | ---: | ---: |
| Finite full-vocabulary rows | 12/12 | 12/12 |
| Matching prediction winners | 12/12 | 12/12 |
| Bit-identical full rows | 0/12 | 0/12 |
| Largest absolute logit error | 0.8125 | 1.033203125 |
| Per-row RMSE range | 0.06838–0.42449 | 0.06936–0.27837 |
| Per-row relative L2 range | 0.00508–0.03507 | 0.00327–0.02614 |
| Largest `KL(reference || backend)` | 0.001195 | 0.003444 |

Metrics use reconstructed BF16 values and FP64 reductions. Relative L2 means
`||observed-reference||₂ / ||reference||₂`; KL uses full-vocabulary softmax distributions
and natural logarithms. These observations define **no new tolerance** and do not prove
distributional or task-quality equivalence. Internal Ribn optimization gates preserve
its previous kernel outputs exactly; that must not be confused with bit identity to
stock CPU arithmetic. MLX is not a replacement numerical oracle.

## Warmed direct-call timing

Three interleaved processes per backend used one warmup plus three measured trials
per prefix, with three common continuation calls per trial. All measured samples,
including large timing swings, were retained. Each reported entry is a pooled median:

| Prompt tokens | Ribn prefill | MLX prefill | Ribn decode | MLX decode |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 335.68 ms | 31.48 ms | 260.76 ms | 30.92 ms |
| 8 | 161.70 ms | 32.66 ms | 143.84 ms | 31.18 ms |
| 18 | 229.62 ms | 63.16 ms | 148.07 ms | 30.96 ms |
| 32 | 732.17 ms | 64.28 ms | 364.88 ms | 31.05 ms |
| 64 | 362.31 ms | 71.23 ms | 148.95 ms | 31.12 ms |

Pooled warmed decode was **148.549 ms Ribn versus 31.087 ms MLX**, a 4.78× latency ratio
on these observations, not a quality-qualified serving speedup. Ribn's per-process
medians were 147.10/282.08/146.54 ms; MLX's were 30.81/30.68/32.92 ms. Ribn's 32-token
prefill process medians were 232.55/741.38/737.17 ms. The large swings materially affect
the 1/32-token entries above; their cause was not established and those cohorts must
not be presented as stable speed ratios. Maximum measured decode was 393.54 ms Ribn
versus 44.57 ms MLX. No occupancy, scheduling or thermal cause is inferred.

External maximum RSS was 14.020–14.025 GiB Ribn versus 14.992–15.010 GiB MLX across the
timed processes. Equal encoded parameter bytes do not imply equal resident or peak
memory. These measurements do not establish admission capacity, pressure behavior,
recovery or a hard shared-memory grant.

## Reproduce the MLX call contract

Use the unchanged snapshot and completed reference described in
[reference preparation](gemma4-reference/README.md). Install the pinned wheel versions
above in a separate environment, set `RIBN_GEMMA_PACKAGE` and `RIBN_GEMMA_REFERENCE`,
and disable Hub networking (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`). The core timed
loop is below; run three separate processes, retaining trial 0 but excluding it from
warmed statistics. The calibration also saves full rows and checks repeated-trial bit
stability, not merely the printed latency.

```python
import json, os, time
from itertools import cycle, islice
import mlx.core as mx
from mlx_lm import load
import numpy as np

reference = json.load(open(os.environ["RIBN_GEMMA_REFERENCE"]))
chat = next(c for c in reference["cases"] if c["name"] == "chat")
model, _ = load(os.environ["RIBN_GEMMA_PACKAGE"], lazy=False)
forced = [s["chosen"] for s in chat["steps"][:3]]
for length in [1, 8, 18, 32, 64]:
    prompt = list(islice(cycle(chat["prompt_ids"]), length))
    for trial in range(4):
        cache = model.make_cache()
        for step in range(4):
            ids = prompt if step == 0 else [forced[step - 1]]
            start = time.perf_counter_ns()
            row = model(mx.array([ids], dtype=mx.int32), cache=cache)[0, -1]
            assert row.dtype == mx.bfloat16
            mx.eval(row, [c.state for c in cache])
            bits = np.array(row.view(mx.uint16), copy=True).reshape(-1)
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            assert bits.shape == (262144,)
            assert np.isfinite((bits.astype(np.uint32) << 16).view(np.float32)).all()
            print(length, trial, step, elapsed_ms)
```

For Ribn, the matching calls are `Gemma4MetalDecoder::load(&source)`, `reset()` before
each trial, `prefill(&prompt)` for step 0 and `decode(forced[step-1])` afterward. Start
its clock immediately before each decoder call and stop after the host vector returns.
Use the same lengths, partitions and histories; do not substitute the existing cold
`metal_latency` example's one-request timing for this warmed comparison.

## Engineering consequence

The comparison supports measuring **native command consolidation and resident
activations** before another larger row block. Ribn still performs 429 synchronous
submissions per step and many activation readback/uploads; existing profiling separated
substantial commit/wait time from GPU intervals. Both measurements bypass the HTTP/text
frontend, so this gap cannot be blamed on endpoint overhead. It is not an attribution
of the entire gap to synchronization: kernel arithmetic, graph fusion, cache placement,
host copies and lifetime behavior also differ. A bounded ablation must separate them.
The subsequent [autorelease-cadence experiment](runtime-contract.md#metal-native-handle-lifetime-ablation-2026-10-21-utc)
bounded known native-handle temporaries, but did not consistently eliminate the timing
swings; it does not explain away the observed baseline gap.

Any new Ribn path must retain its independently justified numerical contract, whole-step
validation, private intermediate ownership and completion-before-publication. No quality
budget was widened, implementation changed, competitive milestone declared or desktop
CUDA qualification attempted as part of this calibration.

## Intermediate-prefill output-head elision (2026-10-22)

**Rejected:** removing discarded heads did not meet the predeclared 3% prefill
improvement gate. All candidate code, API additions and candidate-only test changes
were reverted. No new execution variant is enabled.

The default token-runtime policy accepts at most 16 prompt tokens per sequence per
submission. Only the final prompt chunk reserves output, but the Gemma executor
calls the decoder's full `prefill` for every chunk.
An 18/64/129-token prompt therefore computes 1/3/8 discarded heads. A four-request
burst with prompt lengths 18/32/64/129 computes 13. This was confirmed through the
real `Engine::with_defaults` path, not inferred solely from the scheduler source.

The candidate added intermediate-prompt advancement to the model-local decoder
contract. CPU and Metal ran the unchanged layer/cache computation, then skipped
final RMS normalization, the tied vocabulary projection, host logit materialization
and soft cap when no output was requested. CUDA kept its existing compute-and-discard
execution; no CUDA kernel or device experiment was involved. Scheduling, chunk sizes,
row arithmetic, cache retention and sampling were unchanged.

### Method and acceptance

The control was `58f2fbe`, with a detached baseline worktree and separate release
target directories for control and candidate. Both probes used Rust 1.98.0 and the
unchanged package/reference manifest above on the M3 Max, 128 GiB, macOS 26.6.2.
All nine package files were hash-checked before each model load.

Three fresh processes per variant ran in order B1/C1/C2/B2/B3/C3, one model process
at a time, with no concurrent builds or tests from this session. Other host activity
was not controlled. Each process ran one retained warmup and three measured trials
per workload, with four generated tokens per request. Prompt IDs were the archived
18-token chat cycled to the declared lengths; timing histories were not the actual
511/513-token quality cases. Every trial created a fresh runtime and continuation
state over the process's shared loaded model.

Clocks start before runtime stepping and include admission, scheduling, execution,
argmax and event observation; enqueue/model loading precede them, shutdown follows.
TTFT ends at observation of the first token; the burst's last-terminal clock ends
at the final request's terminal event. Neither is an HTTP/content/EOF measurement.
The probe also retained per-poll submitted prefixes, input counts, output budgets
and completed-call durations. These reconcile to the declared chunk plans and
discard counts.

Before implementing the candidate, the screen required at least 3% lower
64- and 129-token TTFT, no more than 2% decode-interval regression, bit-identical
complete logits on common histories and unchanged generated tokens. Entries below
are medians of three process medians, not pooled individual samples.

| Runtime metric | Control ms | Candidate ms | Change |
| --- | ---: | ---: | ---: |
| c1 TTFT, 18 input tokens | 367.376 | 363.914 | −0.94% |
| c1 TTFT, 64 input tokens | 867.916 | 852.148 | −1.82% |
| c1 TTFT, 129 input tokens | 1897.099 | 1879.224 | −0.94% |
| c4 mixed, median request TTFT | 2383.349 | 2365.962 | −0.73% |
| c4 mixed, last terminal | 5402.905 | 5336.669 | −1.23% |

The 64-token process medians ranged from 851.192–870.093 ms for the control and
850.951–852.189 ms for the candidate; 129-token ranges were 1859.794–1905.051 ms and
1869.054–1896.340 ms. All three c1 decode-interval cohorts stayed within the 2%
regression limit: the largest increase was 0.42%. The failed prefill gate, not a
numerical failure, rejected the candidate. These small changes do not isolate the
head's standalone cost or establish a repeatable benefit under other policies or
backends.

All 12 full-vocabulary BF16 rows from each of the six processes were finite and
bit-identical across variants/processes. Inputs were the stock CPU reference's
18/511/513-token histories plus three teacher-forced continuations, with identical
16-token prefill partitions. All 72 winners matched stock CPU BF16, and all runtime
token sequences matched across the six processes. No numerical tolerance was
widened; token agreement is not task-quality acceptance.

All six model processes exited zero. Raw evidence is retained under
`~/ribn-qualification/gemma4-prefill-head/20261022T212720Z/`: input/acceptance
`plan.json`, `candidate.patch`, isolated worktree/builds, probe and analyzer sources,
invocations and exit codes, full rows, runtime traces and `analysis.json`.
This is a local token-runtime diagnostic, not a new MLX comparison, HTTP study,
capacity/recovery gate or CUDA qualification.

## Bounded packed PLE RMS rescreen (2026-10-23)

**Promoted:** the bounded Metal path normalizes independent token/layer PLE groups
in one existing RMS call instead of one call per layer. On E4B this removes 41
synchronous RMS calls per selected decoder step. The kernel, each group's reduction
order, BF16 boundaries and shared normalization scale are unchanged. CPU and CUDA
execution are unchanged.

Selection requires at most 64 input tokens and at most 1,048,576 BF16 projection
elements. Each packed upload, native result and copied host result is therefore
at most 2 MiB of encoded payload; these three temporary payloads can overlap.
Larger chunks or PLE geometries retain layer-at-a-time normalization. This bounds
the added packed scratch, not process RSS, driver allocation overhead or a shared
device-memory grant. The current E4B geometry uses 21 KiB per packed buffer for
decode, 336 KiB for a 16-token runtime chunk and 1,344 KiB at the 64-token boundary.

### Why revisit the earlier rejection?

The [2026-10-20 candidate](runtime-contract.md#metal-command-buffer-profile-and-ple-batching-experiment-2026-10-20)
missed its 3% decode gate on the older 795-command decoder. The subsequent linear/RMS,
FFN-input and rotary chains plus the prepared GELU table changed that execution
baseline. This rescreen uses the later decoder rather than rounding the old 2.995%
result into a pass.

The first fresh-process screen was not eligible for promotion: its candidate omitted
the workspace bound, and the control's pooled c1 process medians
336.019/152.739/235.844 ms versus 145.291/145.060/223.680 ms for the candidate were
confounded by the known timing swings. The apparent 38.4% difference is **not a
speedup claim**. Those six captures remain archived and excluded from the promotion
measurement. Read-only review found no arithmetic/cache defect but identified both
the workspace and measurement problems; the bounded implementation and matched
experiment below resolve them.

### Randomized within-process ablation

Each of three fresh processes loaded the same hash-verified package once. An isolated
diagnostic worktree added a per-model control solely to choose old per-layer or new
bounded packed PLE normalization between completed trials; both modes shared weights,
pipelines, queue and host-buffer handling. No runtime flag, mutable model control or
instrumentation from that worktree is retained in production.

The workload and clocks follow the preceding runtime experiment: default 16-token
prefill chunks, four generated tokens, c1 prompt lengths 18/64/129 and a c4 burst
with lengths 18/32/64/129. Every trial creates a fresh runtime and continuations.
Each process warms both modes once per workload, then runs 12 old/new pairs per c1
length and six c4 pairs. Pair order is balanced within each workload, randomized,
and shuffled across workloads with recorded seeds. This yields **126 measured pairs**
over three processes. Model processes ran serially without this session's builds or
tests overlapping them; other host activity was not controlled.

For each pair, candidate/control ratios use matched TTFT, median decode interval and
last-terminal clocks. The primary estimator pools the 36 c1 decode ratios per process,
takes their median, then takes the median of the three process medians. It is not a
ratio of separately pooled process timings. The predeclared gate requires at least
3% lower decode interval, a hierarchical-bootstrap 95% upper ratio below 1, and at
most 2% regression in each c1 TTFT/decode cohort and c4 last-terminal time.
The bootstrap resamples processes and then pairs within each c1 workload, using
10,000 draws and the recorded seed; it describes these captures, not universal
hardware or workload confidence.

The primary process median ratios were **0.950063/0.946107/0.947691**: a
**5.23% lower decode interval**, with a 95% bootstrap reduction interval of
**4.88–5.40%**. All non-regression gates passed.

| Workload | Paired TTFT change | Paired decode-interval change | Paired last-terminal change |
| --- | ---: | ---: | ---: |
| c1, 18 input tokens | −4.48% | −5.26% | −4.89% |
| c1, 64 input tokens | −3.51% | −5.33% | −3.95% |
| c1, 129 input tokens | −3.71% | −5.12% | −4.07% |
| c4 mixed burst | −3.88% | −4.61% | −4.23% |

Table entries are medians of process median paired ratios. For c4, TTFT and decode
intervals summarize the four requests; last-terminal time ends after every request
finishes. These are token-runtime clocks, not HTTP/content/EOF metrics or SLO goodput.
Absolute timings still vary; no occupancy, thermal or scheduling cause is inferred.

### Numerical, boundary and application checks

All **72** captured 262,144-element BF16 rows—12 common-history rows in each mode
and process—were finite and bit-identical across modes/processes. All winners matched
the independent stock CPU BF16 reference, with unchanged full-distribution diagnostics.
Every measured pair generated identical runtime tokens in both modes. No tolerance
was widened and the CPU oracle was not changed.

The existing RMS host-reference gate now covers packed token/layer group counts
through the 2 MiB payload boundary, retaining its original one-BF16-ULP limit.
The real Metal decoder test compares 16/32/64-token partitions and a whole 65-token
fallback chunk, including exact full rows and continued cache use. The actual Metal
decoder, runtime executor and text-facade release tests pass. Pipeline/kernel errors
remain failures, not missing-device skips. Existing synchronous terminal waits still
precede output publication and native-buffer retirement.

All three paired model processes exited zero. The evidence root is
`~/ribn-qualification/gemma4-metal-ple-norm/20261022T235759Z/`.
The initial six-process screen is at that root; authoritative promotion evidence is
under `paired/`: the predeclared `plan.json`, isolated ablation patch/source, probe,
randomized schedules, invocations/exits, full rows, per-poll work traces, analyzer,
absolute clocks, paired ratios and bootstrap output in `analysis.json`.
The shipped source retains only the bounded grouping and its qualified fallback.
This is an internal M3 Max/E4B execution improvement, not a new MLX comparison,
competitive serving acceptance, pressure/recovery qualification or CUDA evidence.
