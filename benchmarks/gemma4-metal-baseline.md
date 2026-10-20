# Gemma 4 Metal: same-artifact MLX calibration

Measured 2026-10-20 on the 40-core M3 Max, 128 GiB memory, macOS 26.6.2 (25G83).
This compares **direct decoder calls**, not HTTP endpoints, request scheduling or SLO
goodput. It identifies a remaining latency gap; it does not establish equal task quality
or complete the roadmap's competitive serving baseline.

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

Any new Ribn path must retain its independently justified numerical contract, whole-step
validation, private intermediate ownership and completion-before-publication. No quality
budget was widened, implementation changed, competitive milestone declared or desktop
CUDA qualification attempted as part of this calibration.
