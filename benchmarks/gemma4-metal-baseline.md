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
activations** before another larger row block. During this calibration Ribn performed
429 synchronous submissions per step and many activation readback/uploads; profiling separated
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

## Attention-input projection chain (2026-10-23)

**Promoted:** one input RMS and the query/key/value preparation chains now share a
serial Metal command. The normalized input stays private on device rather than being
copied to a host vector and uploaded separately for each projection. Shared-KV tail
layers prepare only the query. Every existing kernel, BF16 boundary, accumulation
order, per-head scale and rotary factor is unchanged; CPU and CUDA are unchanged.

E4B has 24 KV owners and 18 shared-tail layers. Owner preparation falls from four
commands to one, and shared-tail preparation from two to one: **90 fewer synchronous
commands per selected step**, from 388 after packed PLE to **298**. GPU dispatch count
is unchanged. Source inspection also counts 90 removed native input uploads and 42
removed host input-norm copies, about 10.3 MiB of copying per 16-token step. These are
source counts, not a profile attributing latency to waits, copies or GPU occupancy.

### Ownership and selection envelope

The complete input-RMS and all Q/K/V geometry, scales and rotary extents validate
before any intermediate allocation or encoding. One serial encoder orders producers
before consumers. All private buffers and borrowed inputs/factors survive the terminal
wait; no query or partial K/V pair publishes unless the whole command succeeds.
Input width comes explicitly from model configuration, not division of a possibly
wrong input length.

Gemma selects the chain only for **1–64 input tokens** and at most **4 MiB total
encoded native chain buffers**. Checked sizing sums the normalized input, three
Q buffers, three K buffers and two V buffers; shared-tail consumers omit K/V.
E4B uses 352/624 KiB for sliding/full layers at 16 tokens, and 2,496 KiB for a
full layer at 64 tokens. Larger chunks or geometries keep the separate qualified
calls. Input uploads, immutable weights/scales/factors, host result copies and
driver overhead are separate; this is not a physical peak or concurrent memory grant.

A shared geometry-only preflight and owned-buffer helper also serve the standalone
normalized projection, avoiding two implementations of the same validation/encoding.
There is no operator list, graph runtime, new shader or production ablation flag.

### Matched runtime result

The control and candidate use the same model, weights, pipelines and queue within
each process, with only the attention-chain choice changed between completed trials.
Both retain the promoted packed PLE path. The control shares the new host preparation
structure and lifetime cleanup; it is not a separately pooled old executable.

The [preceding paired method](#randomized-within-process-ablation) is reused:
three fresh processes, eight warmup trials per process, 12 balanced randomized pairs
per c1 length 18/64/129 and six c4 burst pairs at lengths 18/32/64/129; four output
tokens and default 16-token chunks. This gives **126 measured runtime pairs**.
The same predeclared 3% primary decode, hierarchical-bootstrap upper-bound and
2% cohort non-regression gates all pass. No build, test or other model process from
this session overlapped timing; other host activity was uncontrolled.

Process median pooled c1 paired ratios were **0.925633/0.867114/0.865238**:
reductions of **7.44/13.29/13.48%**. The primary median reduction is **13.29%**,
with a 10,000-draw hierarchical-bootstrap 95% reduction interval of **7.24–13.58%**.
This interval describes the captures, not universal hardware/workload confidence.

| Workload | Paired TTFT change | Paired decode-interval change | Paired last-terminal change |
| --- | ---: | ---: | ---: |
| c1, 18 input tokens | −11.06% | −13.54% | −12.32% |
| c1, 64 input tokens | −9.70% | −13.18% | −10.78% |
| c1, 129 input tokens | −9.99% | −13.06% | −10.51% |
| c4 mixed burst | −9.47% | −11.77% | −9.63% |

These are medians of process median within-pair ratios, not ratios of pooled clocks.
Large absolute swings remain: c1/18 process median decode clocks were
223.068 → 207.134, 144.126 → 124.249 and 143.389 → 123.962 ms. These absolute
medians are descriptive, not the paired estimator. No thermal, occupancy or scheduler
cause is inferred, and the result does not establish a new MLX or HTTP comparison.

### Remaining shapes and numerical gates

After the primary captures, a separate shape screen was declared before observing its
data. It strengthens selection-scope qualification without changing the primary gate.
Three fresh processes each ran eight balanced randomized pairs per whole-prompt length
7/8/32/64. The clock covers a complete direct prefill returning an owned full BF16 row;
decoder construction, validation and hashing are outside it. All **96 additional pairs**
pass the per-shape 2% non-regression gate. Paired prefill reductions were
**10.29/12.30/9.00/6.24%**, respectively. These direct clocks are not pooled with
runtime TTFT/decode clocks. They cover odd/scalar, two-row and four-row kernel classes
and the largest selected chunk; the primary runtime already exercises 1/2/16-row calls.

All **72** primary full-vocabulary rows were finite and bit-identical across both modes,
all three processes and the archived promoted `b6717a9` captures. All 72 winners match
stock CPU; full-distribution diagnostics are unchanged. Every runtime pair generated
identical tokens. All 192 supplemental paired rows also compare exactly and are finite;
24 representative raw rows are retained. No oracle or tolerance changed.

Primitive tests compare every Q/K/V result with independently invoked existing kernels,
including scaled/unscaled input RMS, unequal Q/K head counts, no-K/V consumers, column
tails and 64/65 rows. Invalid late branches reject and subsequent calls recover.
The existing real Metal decoder's 64/65-token partition/fallback and continuation gates,
plus executor and text-facade release tests, pass on the final source.

Read-only review found the new valid-model chain sound, but exposed a pre-existing
backend boundary bug: attention did not check value length against keys. The consumer
now checks both extents, and the new producer rejects unequal K/V columns before
allocation. Regression assertions failed before these fixes using oversized, safely
readable value buffers; final primitive/model gates pass. The timing captures precede
these final host-only extent guards; valid inputs, buffers, kernels and dispatches
are unchanged.

Evidence: `~/ribn-qualification/gemma4-metal-attention-chain/20261023T023356Z/`.
`paired/` owns the primary plan, isolated ablation/source diffs, probe, raw rows,
work traces, exits, analyzer and `analysis.json`; `shapes/` owns the independently
declared supplemental plan, captures and analyzer. All six model processes exited zero.
CPU/CUDA, scheduling and cache retention are unchanged. This remains an M3 Max/E4B
execution improvement, not serving/SLO, physical-pressure/recovery, independent
task-quality or other-device performance acceptance.

## Attention-output projection chain (2026-10-23)

**Promoted:** attention's BF16 output now feeds its output projection and grouped RMS
in the same serial command, using the unchanged score/reduce/values, linear and RMS
kernels. The attended value remains a real BF16 intermediate, but no longer needs a
host copy/upload between operations. Gemma's CPU residual add still consumes the
completed normalized output. Input Q/K/V chaining and packed PLE remain enabled.

Each selected layer removes one synchronous boundary: E4B's 42 layers reduce the
selected step from **298 to 256 commands**, with unchanged GPU dispatch counts.
It also removes 42 host attended-vector copies and 42 native attended-input uploads
per selected step. These are source counts, not a measured latency attribution.

### Bound and ownership

Selection requires **1–64 input rows**, **1–576 cached key positions**, and a checked
total of at most **4 MiB** for all simultaneously live native chain buffers:
BF16 scores, two FP32 summaries per query/head row, BF16 attended values and the
projected/normalized BF16 outputs. The exact payload is
`rows × (heads × (2 × keys + 8 + 2 × head_width) + 4 × hidden)`.
Gemma's preflighted output weight has `columns == hidden`, with full-width RMS and
no rotation. At 64 rows/576 keys the E4B full layer uses **1,773,568 bytes**.
Larger chunks, histories or payloads retain the separate calls.

This merges lifetimes that were previously disjoint: attention scratch and Q/K/V
inputs remain live through projection. Input uploads, persistent parameters, host
result copies, allocator rounding and driver storage are outside the payload cap.
It is not an additional-allocation estimate, physical peak or concurrent memory grant.

Standalone attention and this chain share geometry, allocation and encoding in one
cohesive backend module. Both complete attention spans/actual K/V extents and the entire
projection/normalization validate before any scratch or output allocation. All private
buffers survive the terminal wait; only the final normalized result publishes after
success. No shader, arithmetic policy, scheduler, cache retention or CPU/CUDA path changed.

### Matched runtime screen

The predeclared [paired method](#randomized-within-process-ablation) and 3% primary,
bootstrap-upper-bound and 2% cohort gates are unchanged: three fresh model processes,
eight retained warmups per process, 12 balanced randomized pairs per c1 length
18/64/129 and six mixed c4 burst pairs per process. Four generated tokens and default
16-token chunks give **126 measured pairs**. Control/candidate share the same loaded
weights, pipelines, queue and host structure; only this chain changes between fully
completed/shut-down trials. No production ablation flag ships.

Process median pooled c1 paired ratios were **0.955815/0.937168/0.956216**,
or **4.42/6.28/4.38%** lower decode interval. The primary reduction is **4.42%**,
with a 10,000-draw hierarchical-bootstrap 95% interval of **3.40–6.33%**
(seed `2026102303`). All declared gates pass.

| Workload | Paired TTFT change | Paired decode-interval change | Paired last-terminal change |
| --- | ---: | ---: | ---: |
| c1, 18 input tokens | −4.18% | −5.94% | −5.84% |
| c1, 64 input tokens | −3.13% | −4.05% | −3.09% |
| c1, 129 input tokens | −5.65% | −4.38% | −5.14% |
| c4 mixed burst | −2.99% | −1.93% | −2.56% |

These are median process medians of within-pair ratios, not ratios of pooled clocks.
Uncontrolled host variability is substantial: c1/18 absolute process median decode
clocks were 157.308 → 199.327, 108.712 → 103.166 and 126.432 → 143.809 ms.
The within-pair estimator avoids substituting those confounded marginal medians.
Variation remains even within cohorts: process-one c4 TTFT/last-terminal paired ratios
regressed **11.84/4.70%**, while the other processes improved. Promotion follows the
predeclared aggregate diagnostic gates, not a claim that every repetition improves.
No thermal, occupancy or scheduler cause is inferred. Model process wall times were
313.888/254.947/442.089 seconds, all exit zero.

### Shapes, history transition and correctness

The independently declared direct-prefill screen ran another **96 balanced pairs**
across three fresh processes at whole lengths 7/8/32/64. All per-shape aggregate
non-regression gates pass; paired reductions are **4.90/5.86/3.25/2.99%**.
The third process's 32-row median paired ratio regressed **4.04%**; the aggregate
3.25% improvement is not universal per-process performance. These clocks return an
owned full BF16 row and are not pooled with runtime TTFT/decode.

All **72 full 262,144-element primary BF16 rows** are finite and bit-identical across
modes/processes and the archived promoted `4822a99` captures. All stock CPU winners
and full-distribution diagnostics are unchanged; every runtime pair generated identical
tokens. All 192 supplemental paired rows compare exactly and are finite, with 24
representative raw rows retained. A real 512-token prefix followed by a whole 64-token
chunk exercises 576 keys; the next decode crosses to the 577-key fallback. All 12
boundary/continuation rows across modes/processes are finite and bit-identical.

Primitive tests compare the combined path with actual standalone attention plus host
roundtrip and normalized projection, covering positive key offsets, causal/sliding
masks, odd/two-row/four-row classes, 64/65 rows, 576 keys, tails and optional scales/
rotation. Invalid late projection parameters and unequal K/V lengths reject; a
subsequent valid call recovers. Pure selector tests cover exact byte/history/row bounds
and overflow. Actual release decoder/executor/text-facade gates and repository checks
pass. Independent read-only review found no blocking issue and confirmed the exact
allocation formula for the model-owned descriptor.

Evidence: `~/ribn-qualification/gemma4-metal-attention-output/20261023T052027Z/`.
The primary `plan.json`, `candidate.patch`, isolated source/control diffs and native
probes, `paired/analysis.json`, `shapes/analysis.json`, full rows, traces, process/GPU
snapshots and actual exits retain the experiment. All six timed model processes
exited zero, without another model/build/test from this session overlapping timing.
This is an M3 Max/E4B execution diagnostic, not a fresh incumbent/HTTP/SLO comparison,
physical-pressure/recovery, independent task-quality or other-device acceptance.

## Execution-cost refresh (2026-10-23)

A temporary instrumented `ba9e119` refreshes the cost breakdown after command
consolidation. **No production code, profiling switch or execution variant ships.**
The result selects a bounded FFN component investigation, not another optimization
promotion or a new incumbent comparison.

### Capture and accounting

Three fresh processes each loaded the pinned BF16 E4B once, then ran four balanced
randomized profile-on/off pairs for each of five cases: whole 16/64-token prefill,
and one teacher-forced decode after 18/129/513-token prefixes. Prefix preparation uses
16-token partitions outside the measured call. There are **60 pairs/120 measured calls**,
plus retained warmups. Both modes use the same instrumented binary; off mode disables
collection but is not an untouched production executable.

In-memory thread-local spans record native dispatch APIs, shared-buffer allocation,
upload/readback, selected CPU operations and model phases. Every command records host
commit/wait/status duration and reads `GPUStartTime`/`GPUEndTime` only after successful
terminal completion. The installed SDK documents those properties as host seconds,
available since macOS 10.15. No trace formatting or file output occurs during a call;
collection is bounded to 8,192 records and draining/serialization happens afterward.

All **60 traced steps** reconcile **256 commands each**, or **15,360** retained command
intervals. GPU intervals are positive and serial/nonoverlapping; none exceeds its
corresponding host submission duration. A command interval is not individual-kernel
busy time, occupancy or a measurement of the entire GPU's utilization.

The analyzer checks interval containment/disjointness before adding leaf costs.
Native API spans include preparation, allocation, encoding, submission and release;
command and allocation spans are nested within them. Upload spans include their
allocation, and CPU leaf spans cover only the explicitly instrumented operations.
These categories must not all be added together. Model phases are another enclosing
view, not extra cost. The nonoverlapping API/upload/readback/CPU leaves fit within each
measured step, with the remainder retained rather than labeled as CPU computation.

### Current observations

Each column takes the median of four traced calls per process, then the median of
those three process values. Separate column medians need not add.

| Call | Traced wall ms | Commit/wait/status ms | GPU-command intervals ms | Selected CPU leaves ms | Host upload/readback ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Whole prefill, 16 rows | 306.964 | 276.945 | 214.309 | 8.381 | 5.507 / 3.516 |
| Whole prefill, 64 rows | 541.566 | 480.714 | 417.169 | 28.478 | 12.825 / 12.028 |
| Decode after 18 tokens | 156.196 | 145.309 | 97.344 | 2.306 | 2.604 / 0.427 |
| Decode after 129 tokens | 215.672 | 202.840 | 151.084 | 2.453 | 3.889 / 0.507 |
| Decode after 513 tokens | 200.018 | 186.074 | 137.817 | 3.088 | 6.039 / 0.407 |

The compound FFN gate/up and down commands account for **70.86/76.09%** of summed
GPU-command intervals at 16/64-row prefill; process-median ranges are
**70.14–72.17% / 75.82–76.63%**. Their shares for decode after 18/129/513 tokens are
**54.44/50.67/42.86%**, with process ranges **53.17–54.85 / 49.51–51.54 /
42.69–44.16%**. Shares are computed within each call before process aggregation,
not from ratios of independently summarized table columns. These commands include
RMS; the data does not isolate GEMM from normalization or internal GPU gaps.

Prepared GELU-table lookup totals **3.134/12.657 ms** at 16/64-row prefill and
**0.247–0.312 ms** in the three decode cohorts. Host multiply totals **1.531/6.182 ms**
at prefill. Soft cap accounts for **1.566–2.518 ms** across the five cohorts.
Uploads/readbacks move **29.07/41.52 MiB** at 16 rows and **116.26/164.57 MiB** at 64;
decode upload volume grows from **3.55 to 50.92 MiB** over the measured histories.
These are copied payloads, not resident-memory or physical peak measurements.

### Interpretation and limits

Profile-on/off median paired wall ratios are **0.8942/1.0618/0.9996/1.0110/0.9896**
in table order. Individual process ratios range from **0.8020 to 1.2625**. Collection
perturbs execution and uncontrolled host/device state remains substantial; these
differences are neither pure profiler overhead nor a speedup. Absolute traced clocks
must not replace the earlier untraced paired performance gates.

The next experiment should isolate the actual FFN projection components and compare
complete costs before choosing a backend-native lane. Small GELU/multiply host times
do not rule out broader residency gains from removing transfers and synchronization.
This evidence prioritizes the FFN region; it does not prove a particular kernel cause,
select a new algorithm/row width, or justify abandoning command/transfer optimization.

All **72 full common-history quality rows** are finite and bit-identical across on/off
and processes and to the archived promoted `ba9e119` rows. All 120 measured-call rows
also pass shape/finiteness and paired equality, with hashes agreeing across processes;
30 representative timing rows are retained. These timing rows are not all independent
prior-reference comparisons. No numerical tolerance or CPU oracle changed.

Evidence: `~/ribn-qualification/gemma4-metal-cost-profile/20261023T075312Z/`.
It retains the predeclared plan, isolated instrumented worktree/patch, pinned probe,
SDK header, source/binary hashes, raw rows/traces, GPU/process snapshots, invocation/
exit records and `analysis.json`. All three model processes exited zero
(216.156/251.776/213.114 seconds); this session ran no overlapping model/build/test.
Independent read-only review reproduced the accounting and summary values. Production
CPU/CUDA/Metal code remains unchanged. This is a bounded direct-call diagnosis, not
HTTP/SLO, physical-pressure/recovery, task-quality or other-device acceptance.
