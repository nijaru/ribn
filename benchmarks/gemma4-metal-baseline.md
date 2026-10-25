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

## FFN projection component pre-screen (2026-10-23)

**Projection work, rather than RMS, warrants the next bounded backend comparison.**
An FP32 MPS pre-screen reduces measured prefill projection-call cost, but is **not
promoted**: it adds prepared weight storage, regresses decode gate/up calls and changes
some BF16 results. No production source, dependency, profiler or automatic selector changes.

### Native decomposition

A temporary `a88486d` worktree captures real FFN operands at layers **0/20/41** for
whole 16/64-token prefill and one decode after the 18-token chat prefix. The three full
262,144-element final logit rows match the previous cost-profile captures bit-for-bit.
Standalone input RMS, gate/up projections, down projection and output RMS reproduce the
captured model gate/up/normalized-down outputs. Combined commands reproduce the
standalone outputs too. All checks include shapes and finiteness.

Three fresh Rust component processes run two warmup and eight randomized measured
rounds per layer/case: **1,512 measured commands**, plus 378 warmups. The table gives
median GPU-command intervals across three layers/eight rounds within each process,
then the median across processes. These are separate commands, not timestamps inserted
inside the actual chains: this M3 Max reports dispatch-boundary counter sampling
unsupported. They do not measure kernel busy time or preserve the full model's cache state.

| Rows | Input RMS ms | Gate ms | Up ms | Down ms | Output RMS ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1, decode after 18 | 0.0116 | 0.1563 | 0.1552 | 0.3453 | 0.0117 |
| 16, whole prefill | 0.0112 | 0.5901 | 0.5909 | 0.4016 | 0.0112 |
| 64, whole prefill | 0.0114 | 1.3558 | 1.3763 | 0.9617 | 0.0115 |

### Native vendor candidate and measured scope

The candidate uses Apple
[`MPSMatrixMultiplication`](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication)
with FP32 operands/results, `MPSKernelOptionsNone`, unit alpha, zero beta and logical
weight transpose. BF16 weights expand exactly into prepared FP32 buffers; each call
expands BF16 activations on the GPU, performs GEMM and packs with Ribn's unchanged
BF16-RNE helper. The requested MPS pitches equal tight row-major pitches for these shapes.
Direct BF16 vendor GEMM with an FP32-accumulation contract remains unverified; an SDK
BF16 enum or Graph dtype support alone does not establish that contract.

A standalone Objective-C harness compares this path with the exact production MSL and
qualified 1/2/4-row dispatch geometry. Its native outputs must match the archived Rust
components. Both modes start with resident BF16 input and prepared weights; the host
clock includes per-call output/scratch allocation, MPS matrix wrapping, encoding,
conversion, terminal wait/status, owned BF16 CPU copy and per-call GPU-resource release/
autorelease-pool drain. Only the owned CPU result escapes. One command buffer completes
each call; MPS is checked not to commit it while encoding. This is not Rust integration,
a full FFN-chain comparison, or a model/runtime/HTTP speedup.

Three fresh processes run **648 measured pairs/1,296 calls**, plus 54 warmup pairs
and 27 separately retained initial pairs per process. Each measured cohort has four native-first and four MPS-first pairs per process/layer. The table uses
within-pair MPS/native wall ratios, median over eight rounds/three layers per process,
then median over processes; the range is the three process medians.

| Call | Gate ratio (range) | Up ratio (range) | Down ratio (range) |
| --- | ---: | ---: | ---: |
| Decode after 18 | 1.0866 (1.0763–1.1195) | 1.1626 (1.0946–1.3272) | 0.6191 (0.5867–0.8020) |
| Whole prefill 16 | 0.5844 (0.5670–0.6114) | 0.6065 (0.5764–0.6742) | 0.6972 (0.6876–0.7258) |
| Whole prefill 64 | 0.5640 (0.5613–0.5642) | 0.5692 (0.5657–0.5892) | 0.7372 (0.6668–0.7560) |

Ratios below one favor MPS. These conditional paired estimates are not ratios of
marginal medians, which can disagree or reverse under the observed state swings.
Hot component loops, the Objective-C host path, MPS cache effects and per-call
conversions differ from the production model's 15 GB parameter walk. The host was not
isolated: pre-run GPU-utilization snapshots were nonzero and unrelated desktop/browser
GPU clients existed. Only owned model processes were serialized; fixed clocks or an
otherwise idle GPU are not established. Do not extrapolate the table into precise
end-to-end latency or SLO goodput.

An earlier retained screen stopped its clock before GPU-resource reclamation.
The table above uses a separately frozen cleanup-inclusive follow-up, not that screen.
The original source/binary/numeric captures remain available. The first screen observed
a 130.128 ms first-use MPS call and 55.961 ms library compile; later fresh processes
reuse system compiler caches. Other initial calls reuse already-executed MPS shapes,
so the retained initial-call range is not a cold-start distribution. Follow-up weight expansion costs **5.403–5.706 ms host /
2.198–2.203 ms GPU** per matrix (ranges of process medians), separate from steady calls.

### Numerical and memory limits

Every measured result is finite; native results match Rust component captures, and MPS
results are stable within/across processes. The candidate changes **86 of 5,598,720**
BF16 elements across **9 of 27** distinct case/layer/projection tuples, with maximum
absolute change **0.125**. All nine 64-row tuples are bit-identical in this sample;
that does not qualify other layers, prompts, rows or automatic selection. Differences
are not evidence of worse accuracy; prior bit-preserving-chain qualification simply
cannot establish acceptance for this reordered path.

Independent NumPy FP64 dots use the exact captured BF16 operands. Error reporting
compares complete outputs to those dots and directly rounds FP64 to BF16 for an extra
mismatch diagnostic, avoiding FP32 double-rounding. Both paths have relative L2 error
roughly **0.00134–0.00179** against the FP64 dots. These are post-BF16-rounding
output errors, not raw FP32 accumulator errors. These observations grant **no extra
error budget** and do not establish full-model/task quality; no new numerical tolerance
is selected.

Each prepared matrix expands **50 MiB BF16 → 100 MiB FP32**. Keeping all three FP32
copies alongside their BF16 fallback adds **300 MiB per tested layer**; doing so for
all 42 layers would add **12.30 GiB**, an unmeasured extrapolation, not a memory grant.
Per-projection FP32 activation/result scratch reaches **3.125 MiB** at 64 rows,
excluding BF16 buffers, CPU copies and opaque library storage. The cleanup processes'
peak RSS is about **600 MiB**, not a full-model or whole-device physical peak.

**Decision:** investigate a prefill-oriented projection lane, not blanket FP32
replacement. Independent full-model numerical acceptance and a bounded materialization/
memory envelope remain prerequisites. Compare BF16-storage vendor/native alternatives
before committing to full-model FP32 weight shadows; retain the qualified BF16 path.
The [BF16-storage follow-up](#bf16-storage-vendor-pre-screen-2026-10-23) below now
establishes an MPSGraph component candidate without caller-created FP32 shadows.

Evidence: `~/ribn-qualification/gemma4-metal-ffn-components/20261023T083429Z/`.
It retains predeclared plans/schedules, isolated source worktrees, pinned capture and
component binaries, raw operands/results, FP64 dots, both vendor probes/analyses and
actual exit records. One cleanup launcher aborted before spawning; its correction record
identifies a replaced first-screen GPU idle snapshot. Numerical captures were unaffected.
All capture, native component and both vendor-screen GPU processes exited zero.

## BF16-storage vendor pre-screen (2026-10-23)

**MPSGraph is a viable BF16-storage component candidate on this host; direct
MPSMatrix BF16 input is not.** This resolves a preparation choice, not numerical,
full-model, memory-envelope or production Rust acceptance. Production code remains unchanged.

### Actual datatype capability

Two isolated `MPSMatrixMultiplication` probes use real 16-row layer-zero gate operands,
BF16 A/W, `MPSKernelOptionsNone` and either BF16 or FP32 output. BF16 descriptors and
recommended pitches construct successfully, but both calls abort at encoding with
**SIGABRT (-6)** before the caller commits. The runtime assertion allows FP32, FP16,
Int8 or Int16 inputs, not BF16. The failures are retained; they are not missing-device
skips. This is scoped to M3 Max/macOS 26.6.2, not a claim about future Apple runtimes.

A separate MPSGraph probe compiles **BF16 A/W/output placeholders**, weight transpose
and matmul for the same 1/16/64-row gate/up/down shapes. Compilation targets the explicit
Metal device with optimization level 0 and `reducedPrecisionFastMath=None`; no caller
FP32 weight buffer or activation-conversion kernel is created. Buffer-backed tensor
data keeps the prepared BF16 inventory, rather than baking weights into graph constants.
BF16 dtype and this precision setting still do not establish an accumulator/reduction-order
contract over untested inputs.

### Matched component result

The previous real E4B operands, native Rust component captures and independent FP64
dots are reused unchanged. Three fresh processes execute **648 measured pairs/1,296 calls**,
plus 54 warmup and 27 initial pairs per process, across layers 0/20/41 and the three
row counts. Eight measured rounds per instance balance native-first/Graph-first order.
All native results match the qualified MSL/Rust captures.

The host clock starts with resident BF16 inputs/weights and includes BF16 output
allocation, tensor wrapping, blocking executable execution, completion-callback
acknowledgment, owned BF16 CPU copy and per-call ARC/autorelease-pool cleanup. Graph's
internal command count and execution intervals are opaque here. Its API may commit
internally, so this does not preserve or qualify the existing single-command chain.

| Call | Gate Graph/native wall ratio | Up ratio | Down ratio |
| --- | ---: | ---: | ---: |
| Decode after 18 | 0.8370 (0.8122–0.8440) | 0.8362 (0.8245–0.8386) | 0.4374 (0.4302–0.4473) |
| Whole prefill 16 | 0.5096 (0.4821–0.5448) | 0.5453 (0.5063–0.6154) | 0.5976 (0.5847–0.6656) |
| Whole prefill 64 | 0.4653 (0.4578–0.4678) | 0.4688 (0.4537–0.4752) | 0.5139 (0.5121–0.5341) |

Values are median paired ratios over eight rounds/three layers per process, then the
median across processes; parentheses span the three process medians, not a confidence
interval. This screen uses a new matched native control, not the earlier FP32 screen's
clocks. Background GPU clients/state remain uncontrolled. Hot Objective-C component
loops do not establish Rust, whole-model or serving speedups.

All complete Graph outputs are finite, stable across calls/processes and **bit-identical
to the previous FP32 MPS outputs** in all 27 sampled case/layer/projection instances.
Thus the same **86 of 5,598,720** elements differ from native BF16 across nine instances;
full-array FP64 error comparisons are retained. This neither proves FP32 accumulation
generally nor grants a new numerical error budget.

### Preparation, memory and next gate

The nine compiled shape/role graphs take **17.154–22.844 ms total** across the three
processes. The first process observes a **149.768 ms** initial execution; later processes
reuse system caches. Initial calls for later instances can also reuse shapes and are
not a cold-start distribution.

The caller retains **150 MiB BF16 weights per tested layer** and allocates **zero
FP32 weight shadows**, avoiding the explicit extra 300 MiB from the earlier candidate.
Process peak RSS is **301–348 MiB**; Metal allocated size after scoped weights/results
retire is about **2.45 MiB**. Neither snapshots nor source inspection prove the absence
of internal FP32 conversion/repacking/workspace or qualify full-model physical peaks.

### Native Rust full-model pilot

A private worktree at `1379b51` now integrates three prepared BF16 executables with the
real Rust E4B decoder: a **Gate/Up pair at exactly 16 or 64 rows**, and a **Down projection
at exactly 64 rows**. The existing native RMS runs before the pair and after Down, with
terminal completion at each handoff. Graph borrows the existing immutable BF16 weight
buffers; it does not build a second weight inventory. Per-context mutexes serialize
execution. Other shapes keep native execution. These are concrete FFN graphs, not an
operator registry or universal Graph ABI.

The native control and candidate share one release binary, weight inventory and all
three prepared graphs; only the private selection switch changes. Three fresh processes
pass **117 paired full-logit/cache states**: **234 saved 262,144-element BF16 rows**
are finite and bit-exact, and complete cached K/V hashes match after every step.
The histories include 511/513-token prefixes in 64-token chunks, continued teacher-forced
decode, whole 16/64-row prefills and 7/32-row fallbacks. **72 rows also match the previously
qualified cost-profile captures exactly.** No tolerance was widened. Successful Graph
execution counts confirm 42 pair calls per 16-row step, 84 pair/Down calls per 64-row step,
and zero calls for fallback steps.

After that numerical gate, **48 matched prefill pairs** measure eight balanced
native-first/Graph-first rounds per row count/process, with warmup outside the clock.
Mode order alternates deterministically; only row-count order is seeded-shuffled:

| Direct decoder prefill | Median paired Graph/native ratio | Three process medians | Observed reduction |
| --- | ---: | --- | ---: |
| 16 rows | 0.8539 | 0.8581 / 0.7914 / 0.8539 | 14.61% |
| 64 rows | 0.7189 | 0.7189 / 0.7038 / 0.7217 | 28.11% |

Each headline is the median of the three within-process paired medians. All 96 timed
rows are finite and paired bit-exact. The clock covers `decoder.prefill` through its
owned CPU logits, excluding decoder construction/drop, load/compilation and output/cache
validation. Process two has large within-process timing swings: 64-row native calls
range about **314–826 ms**, versus **220–505 ms** for Graph. Ratios of marginal medians
can therefore disagree with the paired statistic. No replicate is excluded post hoc;
three fresh processes do not imply three isolated timing samples. Background GPU
clients/state remain uncontrolled; these are conditional direct-prefill observations, not HTTP/SLO, cold-start,
memory-pressure, decode or general-device speedups. Graph's internal repacks, workspace,
command count and physical full-model peak remain unqualified.

**No production promotion.** The diagnostic bridge returns only before launch or after
both blocking execution and its terminal completion callback. An execution exception
with uncertain submission, or an absent callback after the synchronous call, deliberately
fail-stops the private process rather than returning and freeing potentially live buffers.
A stuck synchronous call has a separate 600-second process supervisor. Successful runs
do not qualify recoverable failure/retirement behavior; fault recovery was not injected.
The bridge also uses a private macOS 26 deployment target, not a new supported floor.

### Completion-owned Graph retirement proof

A subsequent **private** worktree at `5bdad2a` replaces the fail-stop bridge with
an explicit owner. One mutex serializes native commands and all three FFN Graphs.
Before possible enqueue, the owner retains buffers, tensor wrappers, compiled
execution, device and queue. Graph runs asynchronously inside the synchronous backend
call; only its terminal callback authorizes output publication or physical retirement.

A possible-enqueue failure closes that model owner to both native and Graph work.
Explicit settlement queries completion without reopening it; defensive Drop permanently
quarantines an unresolved envelope. There is at most one unresolved submission per
owner, not a global device poison. This is not a new shared-pool or physical-byte grant.

**Nine actual device/owner cases pass** on the final archived executables:

- Exact sparse-dot success; clean prelaunch rejection followed by successful reuse;
  terminal failed-result rejection without partial output publication.
- Event-gated queued failure and an Objective-C exception after enqueue: all five
  accessible buffers survive caller-handle drops and repeated failed settlement.
  Native work on that owner rejects; an independent executor/queue progresses.
  After signaling the gate, the real callback permits reading correct late GPU
  writes, then explicit retirement releases the buffers without reopening execution.
- Drop before completion preserves the buffers and allows those late GPU writes;
  this defensive quarantine is intentionally permanent until process exit.
- Two real E4B runtime cases fail both owned requests without token publication.
  Failed shutdown retains the model/factory/state envelope; retry after completion
  releases it. Dropping unresolved runtime ownership quarantines the whole model.
- A native event-gated command also has nonblocking failure settlement. Review found
  an unbounded native wait in the first prototype; restoring it in a separate source
  reproduces a two-second supervisor timeout, while the corrected query passes.

The final source also repeats **117 exact logit/cache pairs, 234 finite saved rows**
and **48 paired prefill trials** across three fresh processes. The 72 prior-reference
rows remain exact. Median paired Graph/native ratios are **0.8471 / 0.7268**
at 16/64 rows (observed reductions **15.29% / 27.32%**); process medians are
0.8471/0.8619/0.8430 and 0.7284/0.7268/0.7206. These are new matched controls,
not an overhead comparison against the earlier prototype. Timing remains nonisolated;
no serving, hardware-fault recovery, physical-memory or general-device claim follows.

This private proof led to the [model-local experimental integration](#model-local-graph-integration-2026-10-23)
below. It does not itself qualify that implementation, its availability or its physical
memory envelope. Native BF16 remains the default/control/fallback; opaque Graph
submission is not a single-command chain.

Retirement evidence:
`~/ribn-qualification/gemma4-metal-graph-retirement/20261023T102653Z/`.
Only `runs/final-*`, `final-source/`, prearchived `binaries/<sha>/` and
`final-analysis.json` are final qualification evidence. Earlier prototype executables
were overwritten by rebuilds; an aborted launcher also replaced two superseded
prelaunch snapshots before spawning any child. Those cohorts are not primary evidence;
`capture-corrections.json` records the limits without reconstructing provenance.

Rust-pilot evidence:
`~/ribn-qualification/gemma4-metal-graph-pilot/20261023T094336Z/` retains the frozen plan,
separate source/bridge worktrees, compiler/static-analysis output, actual three-process
exits, source/binary/lock hashes, full rows/cache hashes, balanced schedules and analyses.
The pilot builds with the prior registry dependency versions and Rust 1.98.0 unchanged.

Evidence: `~/ribn-qualification/gemma4-metal-bf16-vendor/20261023T092014Z/`.
It retains plans/schedules, both Matrix abort logs/checkpoints/actual exits, Graph source/
binary/hashes, three successful process invocations, full result matrices and analysis.
No MAX internals, dependencies, compiler pins or production interfaces changed.

## Model-local Graph integration (2026-10-23)

The backend now implements the owned protocol as an **experimental Rust opt-in**.
`Gemma4MetalFfn::Native` remains the default; CLI/HTTP and ordinary decoder loading
choose native execution. Advanced callers prepare `Gemma4MetalModel::load(&source, ffn)`
and share it with decoders, or use the actual text loader:

```rust
use ribn_text::{Gemma4MetalFfn, TextOwner};

let (owner, report) =
    TextOwner::load_gemma(package_path, Gemma4MetalFfn::ExperimentalGraph)?;
```

This option prepares Gate/Up pairs at 16/64 rows and Down at 64 rows, with native RMS
and native fallbacks. The scope is E4B 2560/10240 FFN geometry on the M3 Max. Unsupported
SDK/runtime/device/geometry rejects before weights load; it does not silently select
native while reporting Graph. There is no process-global execution switch or CLI flag.

### Qualification

The same pinned BF16 package, Rust 1.98.0, M3 Max and macOS 26.6.2 were used with SDK
27.0/Clang 21; dependencies and environment pins were unchanged. Each model retains
one existing BF16 inventory, without caller-authored FP32 shadows. Unlike the private
global-switch pilot, the matched comparison prepares **two independent model inventories
and queues**, one native and one Graph. This is not a physical-memory comparison.

- Three fresh processes pass **117 paired full-logit states**: all **234 saved
  262,144-element BF16 rows** are finite and bit-exact. The 511/513-token histories,
  teacher-forced continuation, selected 16/64-row shapes and native 7/32-row fallbacks
  are preserved. **72 saved rows** also match previously qualified cost-profile captures.
  External SDK observation records 4,158 successful quality Graph calls and zero for native.
- A committed full-model unit compares **all retained owner K/V bytes**, not only logits
  or hashes, across **26 paired states**. It adds 65-row fallback and selected shapes
  crossing the sliding-window boundary. These are separate protections; the private
  numerical probe does not export or fabricate cache hashes.
- Native decoder/executor gates and the public text-loader gate pass. The latter now
  lives in `ribn-text`, not a hand-assembled Gemma test facade: both policies reproduce
  the four archived greedy tokens twice, compare a selected 16-row prefix and shut down
  explicitly. A separate public-loader probe observes **42 actual SDK calls per 16-row
  request**, zero for native, and equal four-token responses across two requests each.
- Isolated public-SDK fault injection passes clean prelaunch reuse, terminal-result
  rejection, the real 30-second missing-callback deadline, exception after enqueue,
  late GPU writes after owner Drop, retry and permanent quarantine. No injection or
  observation hook is present in production.
- Two fresh **full E4B runtime** processes fail two owned requests without publishing
  tokens. Failed shutdown retains the same model/factory/state envelope; callback then
  retry releases it, while unresolved runtime Drop permanently quarantines it.

**48 paired prefills**, with eight balanced mode-order rounds per shape/process and
warmup outside the clock, pass the predeclared ≥5% improvement gate at both shapes:

| Direct decoder prefill | Median paired Graph/native ratio | Three process medians | Observed reduction |
| --- | ---: | --- | ---: |
| 16 rows | 0.8576 | 0.8206 / 0.8662 / 0.8576 | 14.24% |
| 64 rows | 0.7256 | 0.7208 / 0.7297 / 0.7256 | 27.44% |

All 96 timed rows remain paired bit-exact. The clock covers prefill through owned host
logits, excluding preparation, decoder construction/drop and validation. It is not GPU
busy time. Timing remains nonisolated: 16-row native calls span 172.5–448.2 ms and Graph
142.2–512.2 ms; no process or trial was excluded.

A separate six-process comparison against pristine `b3bb3c1` uses one native inventory
per process and eight trials after warmup at decode/16/64 rows. The median of three
process-paired ratios of trial medians is **0.9932 / 0.9947 / 0.9983**, passing the
predeclared ≤3% regression gate. The first pair has large timing drift; later pairs
are near unity. This is evidence against a material observed native regression, not
a claim that the ownership machinery speeds native execution.

### Availability and failure limits

Warnings-as-errors C compilation and linkage pass at ARM macOS **11.0** and x86 macOS
**10.12**, preserving the Rust target floors. Load commands contain no mandatory
MPS/Graph framework; class lookup and framework loading occur only for explicit Graph
preparation. A forced older-SDK stub and Clang static analysis pass. Actual execution
on an older OS was **not** tested. A fresh release link exposed the Clang availability
helper omitted by Rust's `-nodefaultlibs`; linking the compiler's own support archive
fixes it without raising deployment targets.

Review also found that failure retirement could wait on a concurrently held execution
mutex. A focused test fails on that implementation and passes with `try_lock`: a busy
owner returns unresolved ownership, while existing event-gated native status queries
remain nonblocking. Settlement never reopens the failed owner.

An early fault-test assumption was wrong: holding a GPU event for 30 seconds produced
a real SDK GPU-timeout error before the bridge deadline. Final mode 2 instead withholds
the original SDK callback acknowledgment on a **host** semaphore; it proves the actual
deadline/retention path, not late GPU writes. Mode 4 gates real GPU work and throws
immediately after enqueue, allowing the gate to be released before the OS watchdog;
this separately proves late writes after caller/owner drops. The bridge's 30-second
deadline begins only after the asynchronous SDK CPU method returns and does not bound
that method itself.

Boundaries, fmt, locked workspace tests, default/CUDA clippy and targeted Metal clippy
pass with `CARGO_BUILD_JOBS=2`. Run the Graph gates explicitly: the native
`--test metal_decoder` target does not include the full-KV library unit.

```sh
# M3 Max, capable SDK/runtime; set the pinned package and reference environment.
CARGO_BUILD_JOBS=2 cargo test -p engine-metal --release --locked -- --test-threads=1
CARGO_BUILD_JOBS=2 cargo test -p engine-metal --release --locked \
  --test graph_retirement -- --ignored --test-threads=1
CARGO_BUILD_JOBS=2 cargo test -p engine-gemma4 --features metal --release --locked \
  --lib graph_tests -- --ignored --test-threads=1
CARGO_BUILD_JOBS=2 cargo test -p ribn-text --features metal --release --locked \
  --test gemma_metal -- --ignored --test-threads=1
```

Internal Graph precision, repacking/workspace, full-model physical peaks, hardware-fault
recovery and mixed-arrival serving/SLO acceptance remain unqualified. These results do
not promote automatic Graph selection or establish competitive serving performance.

Evidence: `~/ribn-qualification/gemma4-metal-graph-integration/20261023T200555Z/`.
Plans, source manifests/snapshots, prearchived binaries, actual exits, complete rows and
`analysis.json` retain the numerical/timing cohort. `native-overhead/frozen/` and
`runs-1/` retain its separate native control; `availability-final/` retains build/link
checks. The final mutex-retirement fix changes no measured compute path; its focused
red/green proof and renewed ownership gates are retained separately. Failed and superseded
captures remain preserved, including the release-link error, watchdog assumption,
test-fixture compile failure and a runtime test corrected to distinguish still-owned
model weights from private Graph intermediates.

## Model-local Graph memory screen (2026-10-23)

**Native and Graph reach the same sampled Metal allocation maximum in this cohort;
this is not a hard physical-memory bound.** Graph remains an experimental Rust opt-in.
No production source, dependency, environment pin or selection policy changes.

### Method and SDK boundary

Six fresh processes use one frozen private executable linked to unchanged `8bbf0a2`,
with Native/Graph order N/G, G/N, N/G. Each process loads, executes and drops **three models sequentially**;
only one weight inventory is live at a time. This avoids the two-inventory comparison
used for the earlier timing integration. The BF16 package, Rust 1.98.0 and M3 Max remain
unchanged: macOS 26.6.2, Xcode 27.1, **SDK 27.0** and Clang 21.

Cycle one covers 53 common-history states at 7/16/32/64/65 rows, archived chat and
511/513-token histories, including selected 16/64-row continuations through 592 tokens.
Four further alternating 16/64-row rounds reuse the prepared model with fresh histories.
Cycles two and three each execute a fresh 16/64-row prefix. All **390 saved full BF16
rows**, or **195 matched pairs including repetitions**, are finite and bit-exact across
policies and processes. The analyzer derives exact unique call names, positions and
token counts from the frozen plan; equal record counts alone are insufficient.

The reviewed SDK's Graph compilation/execution descriptors, executable and device APIs
expose **no internal allocator/heap hook, worst-case workspace query or aggregate cap**.
Caller-supplied results and compile-completion control do not constrain intermediates.
Related MPS temporary-memory hints and NDArray-kernel allocators are not Graph allocation
authority. This is a finding within the reviewed public SDK surface, not a universal
negative or a claim about private APIs.

### Observed envelope

The private observer samples `MTLDevice.currentAllocatedSize` and revision-checked
`TASK_VM_INFO` fields, including current footprint and its process-lifetime high-water
ledger. A 32 MiB host allocation and 64 MiB shared Metal buffer calibrate the queries;
both default-device handles see the buffer, and its weak reference clears before
model loading. Fresh-load counter growth covers the **14,926,026,836-byte** active BF16
payload. Later reload deltas can include reclamation of earlier resources and are not
weight-inventory measurements.

A fixed **81,600,000-byte** sampler buffer is allocated and touched before baseline.
Across **148,268 samples**, median intervals are 1.256–1.264 ms, p99 intervals
1.267–1.518 ms and the largest gap is **62.744 ms**. Sampling can miss short-lived
allocations; these are maxima of the API counter, not exact whole-device physical peaks
or an inventory of Graph intermediates. Queries are sequential, not atomic
cross-domain snapshots.
Process lifetime peaks include instrumentation and calibration; they are not cycle-local.
CPU/Metal counters overlap on unified memory and are **not added together**.

All sizes below are MiB. Every process has the same sampled Metal maximum:
**14,239.078 MiB (14,930,755,584 bytes)**. Loaded-model snapshots are 14,234.953 MiB.

| Repetition / policy | Process lifetime footprint peak | Footprint 1 s after third model Drop |
| --- | ---: | ---: |
| 1 Native | 14,639.495 | 225.766 |
| 1 Graph | 14,651.620 | 208.813 |
| 2 Native | 14,662.698 | 241.141 |
| 2 Graph | 14,635.417 | 223.720 |
| 3 Native | 14,658.276 | 227.376 |
| 3 Graph | 14,622.214 | 206.141 |

One second after **every** model Drop, the observed Metal total is **475,136 bytes
(0.453 MiB)** in both policies: no growing Metal residual appears over these three
reloads. Process footprint residuals vary and some grow modestly; they are not proof
of a leak or a causal Graph memory saving. Cycle workloads differ. The observer retains
a device reference during these measurements, which can prolong context/cache lifetime;
this is not device teardown. Final task-only queries after observer release also succeed.
Neither equal sampled maxima nor post-Drop residuals prove zero internal Graph workspace.

### Qualification limits and evidence

All six final processes and the corrected analyzer exit zero. Private release build
and clippy pass with `CARGO_BUILD_JOBS=2`; registry pins are an audited subset of the
unchanged workspace lock. Existing production/model ownership evidence remains tied
to unchanged `8bbf0a2`; this docs-only follow-up does not rerun every workspace/device gate.

Independent review exposed analyzer gaps, corrected before accepting these results:
validate the exact state manifest, separate instantaneous maxima from lifetime high-water
readings, exclude boundary-crossing samples from phase attribution, include endpoints
and reject failed final queries.
The final observer-release sleeps are incremental, nominally 0/100/1100 ms. Abrupt exit
or timeout can lose the RAM sampler trace; such runs cannot establish peaks. No final
process timed out, hit the experimental 48 GiB observed stop or reported a query error.

Evidence: `~/ribn-qualification/gemma4-metal-graph-memory/20261023T222343Z/` retains
`plan-v3.json`, `frozen-v3/`, `manifest-v3.json`, immutable binary
`2a209abf797141a356b0825d6b9e449111876cb6bcfc497b79bc411b63586b46`,
`runs-v3/`, complete rows/samples, SDK headers, `analysis-v3.json` and `verification.json`.
Earlier captures remain preserved: one stopped on an invalid reload-delta assertion;
another caught an autoreleased calibration buffer. A standalone pool-drain probe and
local factory pool fix validate the measuring fixture, not production behavior.

**Decision:** no weight-scale regression was observed, but sampling does not settle
opaque SDK storage, reservation authority or worst-case full-model bounds. Resolve the
external-pool accounting/envelope policy before widening selection. Memory-pressure,
failed-retirement memory envelopes, full-context capacity and mixed-arrival serving/SLO
qualification remain open; this screen supplies no new latency or competitive claim.

## Public text-driver Graph diagnostic (2026-10-24)

**Graph reduces paired mixed-trace time, but this is not service acceptance.** Most
mixed requests miss the predeclared latency targets, and the burst has fewer
target-meeting requests under Graph. Native remains the default, including CLI/HTTP;
Graph remains an explicit Rust opt-in. Production behavior and environment pins do not
change. A text-test comment is corrected: default scheduling chunks the archived
18-token prompt, rather than executing one 18-row native fallback.

### Entry point and offered work

Three fresh M3 Max processes use one frozen release executable against `819d1c6`
(production implementation unchanged from `8bbf0a2`). Each process loads two independent
`TextOwner::load_gemma` owners, Native and ExperimentalGraph, with separate weight
inventories and queues. Only one owner receives requests at a time. The combined
active BF16 parameter payload is **29,852,053,672 bytes**, not a physical-memory bound.
This differs from the preceding one-inventory memory screen. The package, Rust 1.98.0,
macOS 26.6.2 and SDK 27.0 remain pinned.

The actual public streaming facade and default scheduler are used—not a manually
assembled engine. Default prefill chunks are **16 tokens**, with at most four active
Gemma sequences. Members execute separately; four requests are not a physical 64-row
matmul. Consequently this diagnostic exercises **Gate/Up pair16 only**, not pair64 or
Down64. Decode and short tails remain native.

Prompts cycle the archived 18-token chat IDs to these synthetic lengths. Inputs are
pre-encoded token IDs, so request tokenization/chat formatting is not measured.
Generation is greedy, with no text stops and EOS ignored to fix equal output work.

| Trace | Prompt lengths | Outputs per request | Planned arrivals (ms) |
| --- | --- | --- | --- |
| Short c1 | 16 | 4 | 0 |
| Mixed burst | 16, 65, 129, 513 | 4 each | all 0 |
| Staggered, four active slots | 513, 16, 65, 129, 16, 513, 65, 129 | 4, 8, 4, 8, 4, 8, 8, 4 | 0, 25, 100, 175, 400, 650, 900, 1200 |

One consumer thread per request offers work against absolute deadlines and drains
public stream events continuously. Storage and threads are prepared before the clock.
TTFT measures first token-ID receipt, E2E measures terminal receipt, and both start at
**planned** arrival; actual-offer timestamps and public stream-open latency are also
retained. Stream opening is not executor admission. First nonempty text is recorded
separately; it coincides with first-token receipt in this cohort. Observed ITL includes
client scheduling and channel buffering, not just GPU execution. Maximum offer lateness
is **10.047 ms**.

Each process warms every trace once per policy, then runs four paired rounds per
trace, balanced two Native-first and two Graph-first. All **90 trials / 390 offered
requests**, including warmup, complete successfully with exact token IDs, decoded text,
usage and Length outcomes across policies/processes: **2,040 output tokens**. This is
cross-policy agreement, not a new independent quality oracle; the existing full-logit
and complete-KV qualification remains the numerical anchor. The external SDK observer
counts **85,680 actual Graph runs** including warmup. Per Graph trial the counts are
42 / 1,890 / 3,780 respectively; Native counts zero. Both owners explicitly shut down.

### Timing and latency objectives

The predeclared timing screen requires at least 5% lower paired total trace time for
both mixed traces and no more than 5% regression in pooled short-c1 p95 E2E. Ratios
below take the median of four within-process paired Graph/Native ratios, then the
median across three processes. Absolute medians instead pool the 12 measured trials
per policy; these are different statistics. Warmups are excluded, not discarded.
Trace time ends at the last client terminal, not at thread joining/JSON construction.

| Trace | Native / Graph pooled median trace (s) | Three process median ratios | Median ratio / reduction |
| --- | ---: | --- | ---: |
| Short c1 | 0.702 / 0.738 | 0.9095, 1.0525, 0.9678 | 0.9678 / 3.22% |
| Mixed burst | 19.425 / 16.711 | 0.8557, 0.8987, 0.8272 | 0.8557 / 14.43% |
| Staggered | 39.931 / 28.223 | 0.8925, 0.8108, 0.6149 | 0.8108 / 18.92% |

The timing screen passes: short pooled p95 E2E is **1.259 / 1.123 s**, a ratio of
0.8918. That does not establish universal short-request improvement: its pooled median
is **5.25% higher** under Graph, and one process's paired median ratio is 1.0525.

Diagnostic request targets, chosen before clocks, are TTFT ≤2 s, E2E ≤8 s and
per-request empirical p95 observed ITL ≤1 s. A request must meet all three to count.
These are experiment objectives, not product SLOs. The ITL target is evaluated per
request, not against a token-pooled percentile that overweights longer outputs.

| Trace | Native / Graph target-meeting requests | Native / Graph target-meeting requests/s |
| --- | ---: | ---: |
| Short c1 | 12/12 / 12/12 | 1.2277 / 1.2919 |
| Mixed burst | 4/48 / 1/48 | 0.01842 / 0.00560 |
| Staggered | 2/96 / 4/96 | 0.00415 / 0.01115 |

Rates divide qualifying requests by the sum of complete measured trace durations,
including all offered work. There are no failed/rejected requests to omit; any such
failure would invalidate the cohort rather than produce survivor-only acceptance.
Mixed-burst request p95 TTFT/E2E is **23.174/24.085 s Native** versus
**18.241/18.886 s Graph**; staggered values are **48.903/51.596 s** versus
**35.552/37.258 s**. Faster draining therefore does not imply acceptable request latency.
The burst's lower target-meeting count is retained, not hidden by the trace-time win;
these small counts do not establish a causal fairness regression.

### Limits and retained evidence

Owned model/GPU processes are serialized, with no overlapping builds/tests. Other host
clients, clocks and thermal state are not controlled. Measured Native mixed traces span
**9.879–27.017 s** and staggered traces **22.224–65.390 s**; Graph spans
**8.856–22.039 s** and **19.236–46.434 s**. Preserve the process variation rather than
claiming an isolated kernel cause. These short finite traces are neither sustained-load
capacity estimates nor precise tail guarantees. No HTTP, incumbent, cancellation,
recovery, memory-pressure or physical-grant acceptance is added.

Private locked/offline release build and clippy pass with `CARGO_BUILD_JOBS=2`; registry
pins are an audited subset of the unchanged workspace lock. This docs/test-comment-only
delivery reuses the existing production gates; the full workspace/CUDA/device suites
are not rerun. Independent review found no probe blocker and prompted explicit
stream-open terminology, run/package/shim identity checks, unique balanced-pair
validation and separate token-pooled/per-request ITL
reporting. No clocks, workloads or decision thresholds were changed after the cohort.

Evidence: `~/ribn-qualification/gemma4-metal-text-arrivals/20261024T003442Z/` retains
`plan.json`, `frozen/`, `manifest.json`, immutable executable
`ff20746485ee48412fa5aec952e7bf9024c9cf34ddcb5785d3d660976b197bb3`,
`runs/`, `analysis.json`, `analysis-reviewed.json` and `verification.json`. All three
processes, both-owner shutdowns, the frozen analyzer and reviewed verification exit zero.
Earlier preliminary build artifacts remain preserved and are not used as timing evidence.

**Decision:** pair16 gains survive this public-driver diagnostic, but most mixed
requests miss the latency objectives and serving qualification remains open. Keep Native
default; resolve opaque-pool accounting before widening Graph selection and measure the actual
serving entry point against a quality-equivalent incumbent before competitive claims.

## Native HTTP versus stock MLX diagnostic (2026-10-24)

**Actual HTTP protocol and accounting pass; the predeclared matched-result gate
fails. No quality-equivalent serving speed ratio is accepted.** Identical synthetic
prompts produce different greedy continuations on three distinct inputs. This does
not establish which decoder, if either, is wrong.

### Serving contract and offered work

The same 40-core M3 Max, 128 GiB host runs macOS 26.6.2, Xcode 27.1 with SDK 27.0,
and Rust 1.98.0. The archived Ribn release CLI is built from `8ebe5a8` (production
implementation unchanged from `8bbf0a2`); its SHA256 is
`10925f0f91adf665ca7942c0e8c7ab71f95e74c06912e20411fc9944dec2ac84`.
The stock MLX-LM/MLX wheels and unconverted BF16 snapshot remain those in
[Pinned inputs and implementations](#pinned-inputs-and-implementations).
AIPerf **0.13.0**, Python 3.13.11, Transformers 5.20.1 and tokenizers 0.23.3 drive
both endpoints through the unchanged `benchmarks/endpoint.py` runner. Installed
sources, native libraries, executable and all nine package assets are byte-pinned
and verified after the run; no environment, dependency or model pin changes.

Run these servers separately, never concurrently, using the pinned MLX environment
and archived native Metal executable:

```sh
"$RIBN_NATIVE_BINARY" serve-gemma "$RIBN_GEMMA_PACKAGE" \
  --bind 127.0.0.1:9000 --model-name default_model \
  --max-connections 16 --max-requests 8 --max-output-tokens 8 \
  --connection-deadline-secs 120

"$MLX_PYTHON" -m mlx_lm.server --model "$RIBN_GEMMA_PACKAGE" \
  --host 127.0.0.1 --port 9001 --prompt-cache-size 0 --temp 0
```

Both use raw streaming `/v1/completions`, model alias `default_model`, temperature
zero and EOS enabled. No request seed is sent: a seed would disable stock MLX
batching even for greedy generation. AIPerf's dataset seed is 42. There is no adapter,
speculative decoder or weight/KV quantization. Each server owns one model inventory;
Native remains Ribn's CLI/HTTP policy. Ribn keeps its four-active-sequence/16-token
prefill defaults; MLX keeps its 32-decode/8-prompt batch and 2,048-token prefill defaults.
These are different execution policies, not equal physical work. Native has no retained
prefix reuse; MLX's cache entry limit is zero and every reported cached-token count is
zero. This disables retained prompt reuse, not active-KV memory allocation.

Before load testing, each server passes **13 HTTP checks**: twelve separate one-token
continuations at the archived common histories, including the 511/513 sliding-window
cases, plus the four-token greedy chat response. Raw prompt strings round-trip to the
exact submitted IDs in both tokenizers. Window continuations preserve teacher-forced
histories; they are not incorrectly treated as a four-token greedy rollout. Expected
visible text, usage and terminal behavior agree with the CPU archive. HTTP does not
expose generated token IDs or full logits, so these checks add neither full-row
numerical acceptance nor a distribution/task-quality equivalence budget.

The plan calls for three fresh paired server processes, with reversed engine order
in the second pair. Each server runs three repetitions of each workload below, with
balanced workload ordering, **eight profiling requests plus one retained warmup** per
trial. All requests have a four-token cap, not an assumed output count.

| Workload | Nominal input mean / stddev | Concurrency | Nominal requests/s |
| --- | ---: | ---: | ---: |
| Short c1 | 18 / 0 | 1 | unlimited |
| Short c4 | 18 / 0 | 4 | unlimited |
| Variable input | 64 / 32 | 4 | 1, constant |

### Observed gate failure

The first pair completes **18 valid trials: 144 profiling requests and 18 warmups**.
All nine paired trial groups have identical issued prompt multisets, caps, temperatures
and input-token counts, including warmups. Profiling usage totals **5,268 input and
576 output tokens** across both engines. All 144 profiling requests actually generate
four tokens and finish with Length. There are no failed or invalid benchmark streams.

Nevertheless, **12 of 72 paired profiling outcomes differ in visible text**, across
three distinct prompts; warmup outcomes match. One 18-token prompt yields
` horum MIST` from Native, versus ` horum harum hor` from MLX at c1 and
` quickly quickly` at c4, each reporting four tokens. Count agreement is not output
agreement. These HTTP observations alone do not distinguish numerical sensitivity,
batching behavior or a decoder defect. MLX is not substituted for an independent oracle.

The runner stops before the remaining two server pairs, as declared before clocks.
Both servers exit zero after SIGINT, with their owned process groups retired; all
18 client groups also retire successfully. The cohort wrapper exits **1** at the
matching assertion, not because of a client/protocol failure. No paired latency,
throughput or target-meeting goodput ratios are published from this incomplete cohort.
Exact visible-output agreement was a conservative eligibility screen, not a claim
that independently reordered BF16 implementations must be bit-identical. Resolve
independent common-history numerical/margin and task-quality acceptance on the divergent
inputs, including batching scope, before accepting an incumbent serving comparison.
Do not choose a new tolerance or remove these prompts merely to pass this screen.

### Common-history numerical follow-up

A separate diagnostic retains all three divergent prompts and adds two nondiffering
controls from the same HTTP cohort. Native's public decoder API, with default 16-token
prefill chunks, reproduces all five archived four-token HTTP responses exactly. Its
actual greedy IDs then define **20 shared prediction histories**, rather than comparing
free-running outputs after divergence. No prompt, BOS policy or output cap is changed.

The independent CPU reference uses its existing pinned Torch 2.14.0 / Transformers
5.20.0 environment, eager attention and two threads, in separate BF16 and FP32 model
processes. Both consume the exact Native histories. FP32 changes arithmetic, including
embedding-scale rounding; it is a higher-precision diagnostic, not ground truth.
Stock MLX-LM uses its unchanged pinned environment and `BatchGenerator`, with scalar
or four identical prompt/history rows. Native chunks its initial prompt, CPU forwards
the whole prompt, and MLX separates the last prompt token from prefill. These are
common logical histories, not identical physical reductions or cache construction.

A private observer records MLX's post-softcap raw logits, original BF16 normalized
logprobs and stock sampled IDs. Ten fixed-batch observer-off/on pairs match delivered
IDs and normalized-row hashes exactly. A separate teacher-forced mode records the stock
prediction **before** replacing the next input with Native's predetermined ID; its
forced delivery is not reported as stock greedy output. All **60 Native/CPU logit rows**
and **500 MLX raw/normalized rows** have 262,144 finite elements. The MLX count includes
identical four-row replicas and the stock generator's one-token lookahead, not 500
independent prediction points.

MLX's raw and normalized argmax agree at **all 250 captured sampling rows**. Thus the
normalization/sampling stage does not explain these captured winner differences.
Batch width does change raw model logits: scalar/four-identical-row greedy runs
reproduce the 18-token prompt's ` horum harum hor` / ` quickly quickly` split without
HTTP. This does not reconstruct the original server's heterogeneous batch membership.
Across the five implementations/policies, winners differ at four shared prediction
points; the two controls' eight points agree. The independent CPU BF16 and FP32 winners
also differ at three points, so neither stock MLX nor precision voting identifies a
defective decoder.

The following are **descriptive maxima over the 20 shared histories**, computed after
shape/finiteness checks with FP64 reductions and unit-temperature softmax of post-softcap
logits, not the greedy requests' degenerate sampling distributions. KL is
`KL(CPU FP32 || candidate)` in nats; each column's maximum is taken independently.

| Candidate | Maximum KL | Maximum total variation |
| --- | ---: | ---: |
| Native Metal BF16 | 0.3844 | 0.3526 |
| Stock CPU BF16 | 0.8505 | 0.5319 |
| Stock MLX BF16, scalar | 0.1669 | 0.2325 |
| Stock MLX BF16, four identical rows | 0.2599 | 0.3506 |

These are not merely harmless argmax ties: CPU FP32 top-two gaps at the four differing
points range from 0.0874 to 0.7113 logits, and distribution differences can be substantial.
This selected diagnostic set establishes neither model accuracy nor an acceptable
numerical/task-quality budget. The matched-result gate remains failed. Further
localization and independent task-quality acceptance are still needed before accepting
a quality-equivalent serving comparison; no tolerance is fitted to these errors,
production arithmetic is unchanged, and no additional performance cohort is run.

A further **layer-boundary trace of the unchanged 18-token prompt** matches Native and
independent CPU prefill chunks at 16 tokens, followed by the same forced continuation.
Private host-buffer snapshots add no GPU operations. Native trace-off/on reproduces
all five call-level logit rows exactly, including the first 16-token prefix, and both
modes reproduce the four archived Native prediction rows. CPU BF16 and FP32 each pass
the same five-row observer-neutrality check. All **1,395 stage tensors** across Native,
CPU BF16 and CPU FP32 are finite and shape-checked; off/call/prediction copies add 28
verified tensors. These include every token row at all 42 layer outputs, not just scalars.

Native and CPU BF16 embedding and PLE lookup tensors are bit-identical at all five
calls. The first unequal boundary is the raw PLE projection: **41 of 172,032 values**
in the first 16-token block differ, and 59 coordinates differ across all five calls.
An independent replay uses the actual identical BF16 inputs and pinned weight rows.
Exact integer dyadic summation, rounded directly to BF16 ties-even, agrees with Native
at 8 of those already-disagreeing coordinates, CPU at 48, and neither at 3; 70
predeclared matching controls agree with both. These selected counts are not an
accuracy rate over the matrix or model.

A separate explicit ascending-`k` FP32 sum, with each selected BF16 product checked
exactly representable in FP32, reproduces **all 129 selected Native outputs**. The
first differences are therefore consistent with the documented Native accumulation
order, rather than establishing an operand/layout or rounding defect. Correctly rounded
real dot products are not a promise of sequential FP32 accumulation. At the final
prompt chunk, last-token layer-output relative L2 against CPU BF16 grows from about
0.000481 at layer 0 to 0.02739 at layer 41; this does **not** prove the first PLE
perturbations alone cause the final divergence or rule out a later defect.

Matching CPU chunk sizes does not remove this counterexample: all four CPU BF16
prediction rows are bit-identical to its prior whole-18-token prefill results. CPU FP32
rows change by at most 0.00006104 logits without changing winners.

A **causal boundary substitution** then replaces only the raw PLE projection, before
scaling/RMS, in Native and stock CPU BF16. Each engine's identity substitution reproduces
all five archived full logit rows exactly. Every call also checks the actual embedding
and computed projection against its original bits before replacement. Continuation IDs
remain fixed to Native's original history regardless of the counterfactual prediction;
these are joint interventions across all five calls, with prior changes propagated through
KV, not isolated-coordinate effects or free-running greedy responses.

Swapping the engines' projection matrices changes 59 values but leaves all four
predictions unchanged in each engine: Native still starts with ID 3872 and CPU with
3823. Correcting only those previously disagreeing coordinates to the exact-real-dot
BF16 reference changes 51 Native values and 11 CPU values, producing identical raw
projection matrices for both engines. This is **not** an exact reference for the entire
projection or model. All five Native logit rows then match its CPU-matrix substitution
bit-for-bit; all five CPU rows match its original identity control bit-for-bit.

The first projection is therefore not the sole explanation for the cross-engine gap.
At the 18-token prompt's prediction, equalizing Native's projection to CPU's still leaves
`KL(CPU BF16 || Native)` at 0.2584 and total variation at 0.3386, versus 0.2056 / 0.2921
without substitution. These are descriptive unit-temperature distributions on that one
history, not a quality threshold. The differences can affect logits without changing
these winners; neither this result nor shared first-projection bits establishes later
correctness. A precision rewrite of that first operation is not justified as a quality
fix by this counterexample. Independent task-quality acceptance remains unresolved;
no production arithmetic, input policy or error budget changes.

A **head-only replay** next feeds each engine the same captured final-normalized BF16
hidden vectors, without loading either full decoder. Only the tied vocabulary weight
is loaded. Metal uses the existing native projection and model softcap; CPU uses its
original-rank BF16 linear and divide/tanh/multiply sequence. Each backend first
reproduces all five of its own archived raw and post-softcap rows bit-for-bit before
running the other engine's hidden vectors. No normalization is repeated.

Both head implementations agree on the raw and post-softcap winner for **all ten
shared hidden inputs**. At the disputed 18-token prediction, the winner follows the
hidden input, not the head backend:

| Captured normalized hidden | Native Metal head | Stock CPU BF16 head |
| --- | ---: | ---: |
| Native | 3872 | 3872 |
| CPU BF16 | 3823 | 3823 |

The heads are not bit-identical: 9–29 raw coordinates and 8–25 post-softcap coordinates
per row differ, out of 262,144, with maximum absolute difference 0.125. Across these ten
shared inputs, post-softcap unit-temperature `KL(CPU head || Metal head)` is at most
7.436e-10 and total variation at most 1.593e-8; these maxima occur at different inputs.
The FP64 KL/TV calculations use stable relative normalization and agree to nine significant
digits with an independent 80-digit full-vocabulary calculation, including the original
different-hidden baselines. These are descriptive comparisons, not an accepted error budget.

Thus the normalized hidden state carries this counterexample; vocabulary-head arithmetic
alone does not explain it. That replay alone does not distinguish final RMS from earlier
decoder operations, establish either engine's accuracy, or qualify any other head geometry.
The inputs retain the same forced Native histories, not counterfactual greedy rollouts.
No same-input winner differs, so the conditional selected-dot replay is unnecessary.
Production behavior and the failed serving-quality gate remain unchanged.

A **final-RMS replay** then starts from the captured last decoder-layer outputs. Their
last rows match the independently captured pre-RMS inputs exactly. Native retains its
last-row-only RMS call; CPU retains its original full chunk and rank before selecting
the last normalized row. Only the learned final-norm scale and tied vocabulary weight
are loaded. Each backend first reproduces all five of its own archived normalized
vectors and raw/post-softcap head rows bit-for-bit before crossing inputs. A third
Metal process checks five original head controls before completing the crossed heads.

Both RMS implementations return **bit-identical BF16 normalized vectors on all ten
shared pre-RMS inputs**. Crossing both RMS and both head implementations gives 40 head
cells across the five call positions, including the prefix diagnostic. Every cell retains
its input source's original raw and post-softcap winners. At the disputed prediction,
all four paths choose 3872 from Native's pre-RMS state and 3823 from CPU's. Changing only
the RMS implementation leaves every full raw/post row bit-identical for a fixed input
and head. The head-backend differences remain those of the preceding shared-hidden replay.

This counterexample is therefore already carried by the different decoder states before
final RMS, rather than explained by choosing between these RMS or head implementations.
Equal BF16 normalized outputs do not establish identical internal FP32 statistics, general
RMS correctness, or either decoder's accuracy. Earlier decoder operations and independent
task-quality acceptance remain unresolved. Histories, production arithmetic and the failed
serving-quality gate are unchanged; no conditional exact-dot replay is needed.

Earlier **shared-KV replays** check provider 23/layer 41, then providers 22/23 and the
entire sharing suffix, **layers 24–41**, through RMS/head. The latter takes captured
`layer21.output`, `layer22.output`, `layer23.output` and all 18 suffix PLE inputs as a
joint boundary; provider projections run on the suffix backend. Every own-source layer
output and tail row is reproduced exactly before crosses. All ten shared joint-input
winners agree, following the captured source stream. But post-softcap unit-temperature
`KL(CPU unit || Native unit)` reaches 0.027954 and total variation 0.117059. This result
is conditional on those joint inputs, not proof that the prefix alone determines choice.

A **decoder-body factorial** then recomputes layers 0–23 from identical captured
embeddings and crosses three independent axes: prefix backend, suffix backend and the
source of all 42 captured PLE inputs. The prefix's actual complete owner-22/23 KV states
travel with it into either suffix backend; they are not recomputed by the consumer.
Only 408 prefix or 254 suffix-plus-tail weight tensors are loaded per process. Actual
operators and original chunk ranks are retained. CPU masks size against real prior
owner-22/23 state before importing each new prefix increment; both backends require the
entire imported span to match the producer's bits. No dummy cache owners are populated.

Each backend first passes its original-input controls before its own crosses: all 24
prefix outputs and four KV tensors, then all 18 suffix outputs and normalized/raw/post
rows, across all five calls. There are **490 exact archive controls and 160 exact KV
import comparisons**. All 40 body predictions complete. At the disputed position, raw
and post-softcap choices are:

| Prefix backend | Suffix backend, including RMS/head | Native PLE inputs | CPU PLE inputs |
| --- | --- | ---: | ---: |
| Native | Native | 3872 | 3872 |
| Native | CPU | 3872 | **3823** |
| CPU | Native | 3823 | 3823 |
| CPU | CPU | 3823 | 3823 |

The other four sampled positions agree across all eight cells; position zero remains a
prefix diagnostic. Winner effects depend on the other axes. In particular, the same
Native-prefix hidden/KV stream with CPU PLE selects differently between suffix units.
A conditional RMS/head replay checks both resulting complete pre-RMS streams. All ten
shared inputs produce bit-identical BF16 normalized vectors across RMS implementations,
and both heads agree on their winners. Each tail also reproduces its parent mixed
normalized/raw/post rows exactly. The new disputed choice therefore follows the suffix
**decoder output state**, not RMS/head selection. This does not identify a faulty layer,
attention/FFN operation or PLE calculation, or make either backend ground truth.

Holding the other two axes fixed gives 20 contrasts per axis. Independent descriptive
post-softcap maxima at unit temperature are below; KL runs from the CPU-valued axis to
the Native-valued axis. The maxima need not occur on the same contrast.

| Changed axis | Winner disagreements / 20 | Maximum KL | Maximum total variation |
| --- | ---: | ---: | ---: |
| Prefix backend, including its KV | 3 | 0.517570 | 0.475084 |
| Suffix backend, including RMS/head | 1 | 0.0184105 | 0.0950499 |
| Captured PLE input source | 1 | 0.0331448 | 0.114966 |

These are not error budgets, task accuracy or sampling-equivalence evidence. Histories
remain forced Native histories, not greedy rollouts. All contexts reach only 21 tokens,
below the 512-token sliding window: eviction and longer contexts remain unqualified.
Production arithmetic, independent task-quality acceptance and the failed serving-quality
gate are unchanged.

### Independent labeled multiple-choice diagnostic

A separate **57-question, subject-balanced diagnostic** adds task labels rather than
another causal substitution. It uses the original [MMLU test archive](https://people.eecs.berkeley.edu/~hendrycks/data.tar)
linked by the [pinned upstream repository](https://github.com/hendrycks/test/tree/4450500f923c49f1fb1dd3d99108a0bd9717b660).
The archive SHA256 is
`bec563ba4bac1d6aaf04141cd7d1605d7a5ca833e38f994051e818489592989b`.
Before any model runs, one row per subject is selected by the minimum SHA256 of
`seed + NUL + subject + NUL + decimal zero-based row index`, with seed
`ribn-gemma4-labeled-diagnostic-v1`. Selection does not inspect labels, prompt lengths
or model results; all 57 selected questions are retained.

The raw zero-shot prompt contains the subject header, question, four labeled options
and `Answer:`, with no examples, chat template, BOS or special-token insertion.
For every question, appending each of ` A`, ` B`, ` C` and ` D` preserves the entire
prompt encoding and adds exactly one token: IDs 562, 603, 565 and 622 respectively.
The frozen prompts total 6,336 tokens; the longest has 471. Each backend uses identical
16-token chunks and fresh continuation state per question. Each process first
reproduces all five archived full post-softcap rows of the original 18-token
counterexample exactly, in its own precision, before evaluating any labeled question.

Scores restrict the actual final post-softcap prediction row to those four candidate
tokens. Ties choose the first letter in A/B/C/D order and are recorded separately.
Four-choice NLL normalizes only those candidates at unit temperature; it is not
full-vocabulary perplexity. These are **restricted-choice predictions, not generated
answers or instruction-following measurements**.

| Backend | Correct / 57 | Accuracy | Tied option maxima | Mean four-choice correct-label NLL |
| --- | ---: | ---: | ---: | ---: |
| Native Metal BF16 | 30 | 52.6% | 1 | 1.63259249 |
| Eager CPU BF16 | 28 | 49.1% | 3 | 1.65945911 |
| Eager CPU FP32, diagnostic | 30 | 52.6% | 0 | 1.67363764 |

Native and CPU FP32 choose the same option on all 57 questions. CPU BF16 differs on
three: `human_aging` and `logical_fallacies` are correct only for Native/FP32, while
`public_relations` is wrong for all three. The CPU BF16 `human_aging` choice ties B/C;
its `public_relations` choice ties A/B. All backends are wrong on the selected
`global_facts` item; both BF16 paths tie B/C there.

This sample shows no Native task-score deficit relative to either CPU reference,
but it is **not a full MMLU result, an accuracy ranking, or quality acceptance**.
The original labels are independent of the engines, not infallible; training
contamination is unknown. FP32 remains diagnostic, not ground truth or a precision
vote. The unresolved fixed-history divergence and failed matched-result serving gate
remain intact. Broader generation/task-quality acceptance needs its own justified
scope; neither these scores nor the causal replays supply an equivalence budget.

### Generated extractive-QA diagnostic

A separate **32-question SQuAD v1.1 development diagnostic** measures free-form
answers through both real HTTP servers, not restricted-choice logits. The
[publisher dataset](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
is byte-identical to its copy at publisher commit
`e0c66cfec263165fb3c15cb76a22175e371b71d7`; its SHA256 is
`95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9`.
Before model execution, select the 32 smallest SHA256 values of
`seed + NUL + question_id` across all 10,570 questions, breaking ties by ID, with
seed `ribn-gemma4-generated-qa-v1`. Selection ignores labels, content, lengths and
model results; all 32 questions and every original answer alias remain included.

Each user message asks for only the exact answer span from its supplied context,
without explanation. The publisher chat template renders that single message with
`add_generation_prompt=true` and `enable_thinking=false`. Both installed tokenizers
produce identical strings and IDs: 7,001 prompt tokens in total, at most 479 per
question. The frozen strings go through nonstreaming `/v1/completions`, with no
second template. Each question owns a fresh continuation; greedy temperature zero,
EOS enabled, 64 new tokens maximum, and disabled prefix reuse apply to both servers.
This distinct chat-formatted task does not replace or repair the raw counterexample.

Score the **complete returned text**, maximizing over all human aliases, using
standard SQuAD v1.1 lowercase/ASCII-punctuation/article/whitespace normalization,
exact match and token F1. There is no answer extraction or explanation stripping.
The unchanged [Python evaluator mirror](https://github.com/allenai/bi-att-flow/blob/49004549e9a88b78c359b31481afa7792dbb3f4a/squad/evaluate-v1.1.py)
is pinned before selection and models. The author-hosted Python URL is unavailable;
this is an AllenAI mirror, not a successful author-hosted download. The retained
publisher JavaScript has different empty-answer/whitespace behavior and is not
claimed equivalent. A separate normalizer and exact rational token-overlap calculation
cross-check every alias score and aggregate against the pinned Python evaluator.

| Real server | Whole-answer EM / 32 | EM | Mean token F1, 0–100 |
| --- | ---: | ---: | ---: |
| Default Native Metal BF16 | 17 | 53.125% | 79.1127429 |
| Stock MLX BF16 | 17 | 53.125% | 79.1127429 |

Literal answers agree on 30/32 questions; normalized answers and per-question EM/F1
agree on all 32. The two differences are `12–4 record` versus `a 12–4 record`, and
`Surveyor 3 unmanned lunar probe` versus `the Surveyor 3 unmanned lunar probe`
(Native first). Both pairs fail exact match; their token F1 scores are respectively
0 and 4/7. The same 15 questions fail EM in each engine. Longer-than-labeled answers
and explanations remain scored whole, not cleaned into correct answers. All 64
responses report `stop`; none is empty or reaches the 64-token cap.

Each fresh server first passes the same 12 archived one-token reference checks and
one four-token greedy chat check. Both processes then complete all questions and
exit zero, non-forced, with positive process-group absence. This adds **small-sample
lexical generation evidence**, not a full SQuAD result, general instruction-following
or broad quality acceptance. Labels can be imperfect and contamination is unknown.
It does not newly qualify full-row finiteness, distributional equivalence, sliding-window
eviction, SSE, pressure, performance or SLOs. The earlier 12/72 serving text mismatches
and original fidelity counterexample remain open; equal sample scores do not supply
an equivalence budget or authorize a precision change.

### Metrology, compatibility and evidence

A separate analyzer cross-checks selected AIPerf per-request clocks against raw SSE
receipts for all 18 trials, including warmup: actual start to first/last nonempty
content, finish, usage and `[DONE]`, plus content-chunk gaps. These checks pass. The
first-to-last content interval divided by `completion_tokens - 1` is a TPOT proxy,
not individual-token ITL; bundled or suppressed text matters. `[DONE]` receipt is not
HTTP EOF. Aggregate rate denominators include draining, not just the client's
content-only benchmark duration. The general runner still marks performance metrics
unaudited; these study-specific checks do not change that contract. AIPerf's positive
rate is concurrency-gated and can skip nominal ticks, so it supplies no independent
planned-arrival ledger or exogenous open-loop SLO proof. The predeclared 2 s TTFT,
8 s last-content/DONE and 1 s proxy/chunk-gap objectives are diagnostic, not product SLOs.

Stock MLX labels its empty-choice terminal usage frame `chat.completion`, even on
this completion endpoint. The benchmark auditor now accepts only that narrow variant,
not chat content, while retaining finish/usage/DONE ordering and coherent integer
counts. A regression fails the original auditor; focused unit and actual pinned-client
CPU fixture tests pass the correction. Neither server nor AIPerf is patched.

An earlier attempt passes Native's 13 reference checks but fails AIPerf tokenizer
setup before benchmark requests. In offline mode this pinned client's loader expects
a cached repository ID and revision, not an absolute snapshot path. Using
`google/gemma-4-E4B-it` at the same exact revision resolves the identical byte-pinned
assets. A transient denied process-group probe during that attempt is not treated as
retirement proof; later ESRCH and process inspection establish absence. The corrected
runner retains uncertainty until an absence probe succeeds. The failed attempt remains
preserved separately.

Owned model processes are serialized, with no overlapping builds or tests. Other host
GPU clients, clocks and thermals are uncontrolled. Five-second CPU/RSS/GPU/VM samples
are neither physical peaks nor memory grants. No cancellation, overload, pressure,
recovery, sustained capacity, competitive SLO or Graph promotion is qualified here.

The preceding benchmark-code verification passes boundary checks, formatting, locked
workspace tests, default/CUDA/Metal-feature clippy, 40 Python unit tests and the five-test
real-AIPerf CPU integration suite. This numerical follow-up changes only private probes
and the evidence documentation; those code gates remain applicable. Every Cargo
invocation uses `CARGO_BUILD_JOBS=2`; dependencies and toolchain remain pinned. Existing
Graph and retirement qualification is reused; ignored Metal decoder/executor suites and
CUDA device tests are not rerun for this follow-up.

Evidence: `~/ribn-qualification/gemma4-metal-http/20261024T015229Z/` retains the plans,
`frozen/` and `frozen-v1/` source/environment snapshots, manifests, immutable executable,
all raw client/server captures, `matching-diagnostic.json`, `outcome-summary.json`,
`analysis-v1.json` and `verification.json`. The analyzer and post-run verification exit
zero while explicitly recording the stopped, unqualified comparative cohort. Earlier
failed captures and verifier revisions remain intact; no accepted capture is replaced.

Common-history evidence: `~/ribn-qualification/gemma4-metal-http-quality/20261024T031051Z/`
retains the predeclared plan, exact inputs, frozen `80e0aa1` source and installed-file
manifests, original-dtype tensors, observers, `analysis-v1.json` and
`verification-v1.json`. The Cargo-artifact-discovered Native probe SHA256 is
`3a6d0cf08b8a7b251bd601483b65f7a92f67bc9cae649963be07eed3deb7bf41`;
its registry versions and checksums are an exact subset of the workspace lockfile.
All nine package assets and the byte-pinned installed files in both reference
environments remain unchanged after the runs. Four successful model-policy processes
exit zero with their groups retired. One earlier CPU collector stops after four rows because loader diagnostics
contain non-JSON sets; its failed process retires, and its partial rows/source/logs
remain excluded and preserved. The corrected collector uses the reference generator's
existing sorted-list metadata serialization without changing model arithmetic.
Unused preliminary build artifacts and failed build captures are also retained, not
substituted for the qualified executable. Analysis and post-run provenance verification
exit zero; these checks do not turn the failed serving-quality gate into acceptance.

Layer-localization evidence: `~/ribn-qualification/gemma4-metal-layer-trace/20261024T041113Z/`
retains the frozen `b578d0d` source plus private trace-only patch, source/installed-file
manifests, `layer-analysis-v1.json`, exact/ascending replay sources and reports, and
`verification-v1.json`. All four model processes exit zero, serially within the original
30-minute model-phase deadline, with owned groups retired. The qualified Native probe
SHA256 is `aef3237fa72888ac99cd1a1059ea8097c58ae7bdee3ef0c03773b110d643dbc4`.
Post-run source, artifact, environment, finite-tensor and neutrality verification exits
zero. An independent mathematical review finds no material replay defect in this scope;
the exact rounder rejects overflow and is not an unrestricted BF16 implementation.
The earlier unused build and all prior captures remain preserved.

Causal-substitution evidence: `~/ribn-qualification/gemma4-metal-ple-causality/20261024T045515Z/`
retains the plan, frozen `defc167` source plus private boundary-only patch, replay/input
and installed-file manifests, `analysis-v1.json` and `verification-v1.json`. The qualified
Native probe SHA256 is `42bbeca49a1240c355b42f9d7c33625049ba8d6365822d4331a2fcdbd18f6875`.
Three Native processes and one CPU process with three fresh-cache policies exit zero,
serially within the original model-phase deadline, with their groups retired. Post-run
verification checks 42 finite full-head files, 20 finite replay matrices, all 30 call-level
intervention guards, unchanged pins and both five-row identity controls. An independent
collector review finds no material defect within this fixed-history scope. Build,
analysis and post-run verification exit zero; no production hook or kernel is added.

Head-replay evidence: `~/ribn-qualification/gemma4-metal-head-causality/20261024T080854Z/`
retains the plan, pristine frozen `9c0e125` source, private component probes, exact input
provenance, installed-file manifests, `analysis-v3.json`, `decimal-check-v3.json` and
`verification-v3.json`. The Cargo-artifact-discovered executable SHA256 is
`69b7b4196934fd40f4fbc7faa741ca7532c8215b98f8f263b26b7f49397ec33b`.
Both serialized head-only processes exit zero within their fresh component-phase budget,
with their groups positively retired. Verification checks all 70 finite tensor files,
20 exact self-control rows, 1,224 frozen/live study files, 5,669 installed files and
unchanged asset/toolchain/dependency pins. Earlier cancellation-prone analyzers and the
failed high-precision check remain preserved; no component capture is replaced or rerun.
The corrected analyzer passes all 30 independent full-vocabulary distribution checks.
Build, accepted analysis and post-run verification exit zero. This is neither serving
performance nor physical-memory qualification.

Final-RMS evidence: `~/ribn-qualification/gemma4-metal-final-rms/20261024T181052Z/`
retains the plan, pristine frozen `05d819d` source, private probes, exact full-chunk/tail
provenance, installed-file manifests, `analysis-v1.json`, `decimal-check-v1.json` and
`verification-v2.json`. The Cargo-artifact-discovered executable SHA256 is
`994dc8876d758123db2855c93dbad7ea03f054abef7f051db28f3c2dbdd650f0`.
All three component processes exit zero, serially within their fresh 30-minute budget,
with positive group retirement. Verification checks 150 finite tensor files, 40 exact
self-control tensor comparisons, 1,241 frozen/live study files, 5,669 installed files and
unchanged asset/toolchain/dependency pins. All 90 full-vocabulary distribution comparisons
agree to nine significant digits with the independent 80-digit check; identical distributions
are handled as exact zero. The first verifier incorrectly compares JSON epsilon decimals
rather than their identical effective FP32 bits; that failed inspector and its correction
are preserved, without rerunning components or changing arithmetic. An initial toolchain
guard also stops before Cargo; the actual build explicitly pins installed Rust 1.98.0.
Build, analysis, independent distribution checks and corrected post-run verification exit
zero. This is neither a serving-performance nor a general decoder-quality qualification.

Last-layer evidence: `~/ribn-qualification/gemma4-metal-last-layer/20261024T201341Z/`
retains the frozen `32891ef` base and private replay changes, subset-weight inventory,
70 copied input/expected tensors, `analysis-v1.json`, `decimal-check-v1.json` and
`verification-v1.json`. The Cargo-artifact-discovered executable SHA256 is
`b677c98341e5e9c4ac4fbba8b84ee19407d0e1b0efec28b605586eb09d7859b6`.
Both serialized processes exit zero, non-forced, within fresh 15-minute process and
30-minute aggregate bounds, with positive group retirement. Verification checks 210
finite tensor files, 40 exact self-control tensor comparisons, 1,390 frozen/live study
files, 3,918 registry source files and 20,530 installed environment files, plus assets,
interpreter, executable and Rust pins. All 30 full-vocabulary distribution comparisons
agree to nine significant digits with the independent 80-digit calculation.

The deleted CPU cache environment is restored separately at
`~/ribn-qualification/gemma4-reference-environment/20261024T195933Z/`, not recreated as
historical provenance. Python, the five pinned package versions and all seven reference
source hashes match; 5,666 of 5,669 archived package-file hashes match exactly. Only three
new installation `RECORD` files differ, and all 20,496 hash-bearing entries across the
34 installed distributions validate. Their actual new hashes are pinned for this replay.
The offline Rust build uses 102 checksum-verified archives copied from preserved caches,
without downloads or dependency changes. Earlier formatting, offline-cache, type-inference
and preflight-inspector failures remain preserved; the accepted build, preflight, analysis,
independent distribution check and post-run verification exit zero. No component is rerun
or production hook merged.

Shared-suffix evidence: `~/ribn-qualification/gemma4-metal-shared-suffix/20261024T210356Z/`
retains the frozen `642277b` base and private replay changes, exact subset-weight inventory,
420 copied input/expected tensors, `analysis-v1.json`, `decimal-check-v1.json` and
`verification-v1.json`. The Cargo-artifact-discovered executable SHA256 is
`eb3eea5a741fff3e12e5613b924cd6cd62fc563fe639f338bd33a7e96d972065`.
Both serialized processes exit zero, non-forced, within fresh 15-minute process and
30-minute aggregate bounds, with positive group retirement. Verification checks 1,280
finite tensor files, 210 exact self-control tensor comparisons, 1,728 frozen/live study
files, 3,918 registry source files, 20,530 installed environment files and unchanged
asset/toolchain/dependency pins. All 30 full-vocabulary distribution comparisons agree
to nine significant digits with the independent 80-digit calculation. An independent
read-only collector review finds no material defect within this fixed-history scope.
The offline build, preflight, analysis, independent check and post-run verification all
exit zero; no component rerun or production change is needed. This study runs locally
on the M3 Max, without desktop/CUDA access.

Body-factorial evidence: `~/ribn-qualification/gemma4-metal-body-factorial/20261024T214059Z/`
retains the frozen `999f62c` base, private actual-layer replay, 920 copied input/expected
files, `analysis-v1.json`, `decimal-check-v1.json` and `verification-v1.json`. The actual
Cargo-discovered executable SHA256 is
`275bd6078e48bb744386548050551aa4c3a808c01ef1640acd617ceb0341477b`.
The prefix outputs are frozen before the suffix phase. Verification covers 2,320 finite
tensor files, 490 archive controls, 160 KV imports, 2,233 frozen/live source files, 569
second-phase files, 3,918 registry files and 20,530 installed environment files. All 120
full-vocabulary contrasts agree to nine significant digits with independent 80-digit
calculations. Read-only review finds no material prefix/import/suffix collector defect.

Its `tail-check/` supplement reuses the archived `994dc887…650f0` RMS/head executable,
with unchanged production source, rather than rebuilding it or rerunning a full decoder.
Two fresh Metal processes each check five original Native triplets before one mixed
stream; one CPU process checks five original CPU triplets before both mixed streams and
Metal-RMS inputs. `analysis-v2.json`, `decimal-check-v2.json` and `verification-v2.json`
check 175 additional finite files, 45 original-reference and 30 mixed-parent tensor
controls, ten equal RMS vectors and 20 independently checked distributions. The Native
head is not separately rerun on CPU-RMS outputs: those inputs are bit-identical to its
measured Metal-RMS inputs. The combined closure covers **2,495 finite tensor files**.
All seven actual components exit zero, non-forced, with positive process-group retirement
within fresh bounded phases. No desktop/CUDA access or production change is involved.

The first launch guard exits one **before any component or budget starts**, after
matching unrelated Cargo tests. That failure and both actual launcher versions remain
frozen. The corrected guard records unrelated project build directories while retaining
model-process checks; accepted build, preflights, components, analyses and verification
exit zero. Exploratory tail-analysis V1 is retained; accepted V2 freezes the same analysis
method before its calculations, without model reruns or tolerance changes.

Labeled-quality evidence:
`~/ribn-qualification/gemma4-metal-labeled-quality/20261024T231839Z/` retains the
predeclared protocol, original archive and 57 CSVs, deterministic sample, pristine
`018e9ec` source, public Native-decoder probe and actual pinned CPU collectors.
The Cargo-discovered executable SHA256 is
`19a9c0301189c86810cf54e693d008ff1407462e74992b6bab185f5dfad32a5d`.
`analysis-v1.json`, `decimal-check-v1.json` and `verification-v1.json` pass:
201 finite tensor files, 15 exact original-reference controls, all 171 task predictions,
1,389 frozen/live source files, 3,918 registry files, 20,530 installed CPU files and
nine pinned assets. All 171 four-choice probability/log-probability/NLL rows agree
to nine significant digits with independent 80-digit Decimal calculations.
Read-only review finds no material sampling, token-boundary or collector defect;
numerical/provenance checks remain separately verified. All three serialized model
processes exit zero, non-forced, with positive group retirement under fresh 5,400-second
global and 1,800-second per-process bounds. Build, preflight, analysis and verification
exit zero. No production source, precision policy, environment or dependency is changed,
and no desktop/CUDA access is used. The unchanged-code gates above are reused; the
ignored device suites are not rerun.

Generated-QA evidence:
`~/ribn-qualification/gemma4-metal-generated-qa/20261025T000551Z/` retains the
predeclared protocol, actual dataset/evaluator bytes and source-resolution record,
deterministic sample, both rendered input sets, pristine `d0f7290` source and raw
requests/responses. It reuses the actual Cargo-discovered `10925f0f…2ac84` CLI above,
without a new Rust build: all 343 tracked files outside `docs/` and `benchmarks/`
are byte-identical to the original build's frozen source. `analysis.json`,
`verification.json` and an independent primary-label/raw-body cross-check pass:
1,311 frozen/live source files, 20,530 CPU preparation/scoring files, 5,824 MLX
installed files, nine assets, 26 exact controls and all 64 answers. Both real servers
retire serially within fresh 3,600-second global and 1,800-second per-server bounds,
including shutdown. Requests have absolute 120-second deadlines and bounded bodies;
real-socket fixtures reject an oversized response and a trickling peer.

Read-only review identifies shutdown-bound, failure-reporting and score-qualification
gaps in the new private harness; all three are corrected before models. Shutdown
gets a reserved budget with the watchdog active, failures retain retirement status,
and scoring rejects an unqualified cohort. Source-lookup failures and the first
renderer’s incorrect list-versus-`BatchEncoding` assertion remain preserved; the
corrected renderer checks exact IDs and attention masks, without changing prompts,
selection or metrics. Accepted preparation, preflight, supervisor, analysis and
post-run verification exit zero. No production arithmetic, dependency, environment
or default changes; no desktop access, new CPU model run or performance cohort.
The unchanged-code gates above are reused; ignored device suites are not rerun.
