# Gemma 4 CUDA: native, stock and prepared execution

Measured 2026-10-21 on the RTX 4090, 24,564 MiB, driver 615.71.10.
**Native prefill remains 2.81×/5.81× slower than stock Transformers at 18/64 tokens.**
Native warmed decode is modestly faster than stock eager. Stock compilation demonstrates
reusable whole-forward execution, but specialization, numerical differences and ownership
still matter. This is a direct-call diagnostic, not competitive serving or quality acceptance.

## Pinned implementations and call contract

- Ribn `6b1d333`, execution unchanged from `59b9f9d`, Rust 1.98.0 release build;
  qualified eight-row groups and scalar tails, synchronous full-history CUDA decoder.
  The immutable probe executable SHA256 was
  `aeb8130a66241d723d0e1badaeb9b4593ff897f0183ec47784b85561ebf6e176`.
- Published `transformers==5.20.2`, `torch==2.14.0`, `triton==3.8.0`,
  `safetensors==0.8.0`, `tokenizers==0.23.3`, Python 3.13.11. Transformers release
  [commit `18329f1`](https://github.com/huggingface/transformers/tree/18329f1065382e54db698e242cf98bd3f3e5a766).
  Its Gemma model/configuration, cache and RoPE source hashes match the pinned CPU
  reference; masking code differs. The reference environment was not modified.
- Both consume the unchanged BF16 [snapshot and checksums](gemma4-reference/model.toml):
  **665 unique active checkpoint tensors, 14,926,026,836 encoded bytes**, tied embedding/head.
  In PyTorch these are 623 parameters plus 42 persistent layer-scalar buffers. No weight
  conversion, quantization, adapters, media or speculation is enabled.
- Stock loading uses `Gemma4ForConditionalGeneration.from_pretrained` on CPU with clean
  loader diagnostics. Its original language/head modules are attached to a meta-created
  stock `Gemma4ForCausalLM`; media modules are released before GPU upload. No upstream
  source or forward mathematics is patched. Active names/shapes/dtypes/bytes and tying
  were checked. All 12 dynamic-eager rows are bit-identical to the earlier full conditional
  CUDA capture, independently checking this text-only extraction on common histories.
- Stock numerical policy: BF16 weights/cache, eager attention, deterministic algorithms,
  TF32 and BF16/FP16 reduced-precision reductions disabled. Native uses cuBLAS 13.6.0.2;
  Torch's CUDA 13.0 bundle uses cuBLAS 13.1.1.3. This is not an isolated compiler or
  synchronization ablation.

Each clock starts before API input preparation and ends with completion and the complete
262,144-element **BF16 host logit row**. Checks, argmax, tokenization, file I/O and loading
are outside the warmed clock. Stock uses `logits_to_keep=1`, explicit absolute positions,
no padding and native cache classes. Native partitions prompts into eight-row groups;
stock eager submits the whole prompt. Native retains full preallocated histories; stock
DynamicCache retains sliding/full state, while StaticCache preallocates local/full
512/4096-token storage. These are intentional execution differences, not equal kernels.

## Warmed native versus dynamic eager

Three fresh processes per implementation ran sequentially in order N1/T1/T2/N2/N3/T3.
A fixed-position compiler probe ran between T1 and T2, with no model-process overlap.
Each prefix has four trials: retain trial zero as warmup, summarize the other three.
Each trial has three common forced continuation calls. Timing inputs cycle the archived
18-token chat IDs; they are not the actual 511/513-token quality inputs. The GPU had no
compute applications before launches; this session's builds/tests did not overlap them.
Other host activity, clocks and thermal state were not controlled or causally attributed.

Pooled medians, milliseconds:

| Prompt tokens | Native prefill | Stock eager prefill | Native decode | Stock eager decode |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 18.126 | 24.092 | 18.168 | 21.013 |
| 8 | 20.187 | 23.595 | 18.253 | 21.197 |
| 18 | 68.375 | 24.296 | 18.509 | 21.161 |
| 32 | 73.513 | 24.333 | 19.368 | 21.042 |
| 64 | 146.840 | 25.273 | 19.557 | 21.193 |
| 65 | 163.438 | 27.690 | 19.432 | 21.257 |
| 128 | 297.329 | 28.596 | 19.986 | 21.243 |

Pooled decode is **19.331 ms native versus 21.179 ms eager**. The 18-token prefill
process medians were 68.15/68.82/68.38 ms native and 24.45/24.30/24.15 ms stock;
64-token medians were 146.60/147.69/146.84 and 25.41/25.27/25.19 ms. All samples were
retained. This reproduces the native control and identifies a substantial prefill gap;
it does not attribute that gap solely to host launches or establish serving throughput.

## Prepared decode: specialization is observable

The unmodified stock forward used a preinitialized `StaticCache(max_cache_len=4096)`
and `torch.compile(..., fullgraph=True, mode="reduce-overhead")`. Prefills remained
**eager**. One cache owner was reset/reused across the three real archived prefixes,
then nine teacher-forced advancing decodes. Compiler threads were limited to two.
Each configuration used a fresh private Inductor disk cache; other compiler/driver
caches were not flushed. Deadlines were declared before launching, not extended.

- **`dynamic=False`:** nine decodes succeeded, but eight distinct Dynamo/Inductor
  specializations were compiled. New sampled positions incurred approximately 27–30 s
  first-call wall time; the repeated position at length 514 reused a specialization.
  The bounded subprocess completed in 235.3 s under its 240 s limit.
- **`dynamic=True`:** the same calls used **two** compiled specializations: one before
  the local cache became full, another at the window transition. Their first forwards
  took 63.33 s and 52.77 s, including tracing/compilation/first execution. The process
  completed in 123.6 s under its 150 s limit. Other advancing calls took 16.87–39.01 ms;
  these single observations include further capture/warmup effects, not a latency distribution.
- Four separately replayed calls at the fixed 18-token chat position retained trial zero
  as warmup. The remaining three-call medians were **14.007 ms static-shape** and
  **14.337 ms dynamic-shape**. These are one-process, fixed-position diagnostics, not
  the three-process native/eager decode sweep or a general warmed decode claim.

Neither configuration reported Dynamo graph breaks. A full Dynamo graph is **not** one
universal CUDA graph: dynamic mode reported four Inductor CUDA-graph partitions. The
sampled positions do not qualify every position, batch shape, cache reach or recovery
path. No compiled-prefill performance, Rust export/import, portable package, or Python-free
serving integration was tested. A static-shape result alone would have misleadingly
suggested that compilation necessarily requires one executable per token position.

## Numerical diagnostics, not a new budget

Shape and finiteness were checked before FP64 reductions over all vocabulary logits.
The independent stock CPU BF16 reference remains Transformers 5.20.0 with tensor SHA256
`46878e19c907f6cb4d2e88ae6077238465612d3bc1aab8441f57bfb4b83772f3`.
Common IDs are teacher-forced, not independently generated histories. Native rows reuse
the unchanged qualified control capture. Maxima below are per-row diagnostics:

| Against CPU BF16 | Rows | Winners agree | Bit-equal rows | Max absolute | Max centered RMSE | Max relative L2 | Max KL | Max TV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native eight-row | 12 | 12 | 0 | 1.0000 | 0.167896 | 0.047227 | 0.004639 | 0.042821 |
| Stock dynamic eager | 12 | 12 | 0 | 1.2656 | 0.120362 | 0.027394 | 0.000966 | 0.013898 |
| Stock static eager | 12 | 12 | 0 | 0.8125 | 0.145470 | 0.034696 | 0.005148 | 0.041094 |
| Compiled, `dynamic=False` | 9 | 9 | 0 | 0.7188 | 0.134284 | 0.033608 | 0.007926 | 0.044716 |
| Compiled, `dynamic=True` | 9 | 9 | 0 | 0.6250 | 0.118889 | 0.025457 | 0.007970 | 0.046285 |

The compiled rows are only the **nine continuations after eager prefills**, not all 12
rows. Do not rank nine-row maxima against twelve-row maxima as equal scopes. Centered
RMSE removes the mean logit delta; relative L2 is `||observed-reference||₂/||reference||₂`;
KL is `KL(reference || observed)` using full-vocabulary softmax and natural logarithms;
TV is half the L1 probability distance. No observed error defines a tolerance.

Static versus dynamic eager rows agree bitwise in only 3/12 cases. Neither compiled
configuration matches its static eager control bitwise in any of nine cases, nor do the
two compiler configurations match each other bitwise. All sampled winners agree, but
**distributional, sampling and task-quality equivalence remain unqualified**. Graph capture,
batch invariance and arithmetic preservation are separate claims.

## Preparation and memory observations

These are process-cold observations, not filesystem-cold startup comparisons. Stock CPU
loading took 0.37/0.42/1.57 s; extraction plus GPU upload took 1.07/1.09/1.26 s across the
three eager processes. Native executable start through prepared model/cache took
9.85/7.69/8.08 s, including different startup stages. Do not turn those clocks into a
loader speed ratio. Compiler first-forward costs above are separate from warm API timing.

| Accounting surface | Observed bytes |
| --- | ---: |
| Native model charge | 14,939,560,928 |
| Native model plus one full cache charge | 15,174,441,952 |
| Torch after upload: allocated / reserved | 14,926,052,352 / 14,944,305,152 |
| Dynamic eager 12-row run peak: allocated / reserved | 15,050,937,856 / 15,101,591,552 |
| Static eager run peak: allocated / reserved | 15,313,897,984 / 15,521,021,952 |
| Either advancing compiled run peak: allocated / reserved | 15,313,897,984 / 15,772,680,192 |

Native charges and Torch allocator statistics have different boundaries; neither equals
whole-device physical residency or proves equal capacity. The timing-only eager runs had
lower peaks than the run that also captured 12 quality rows. Native prepared RSS was
about 14.30 GiB in two sampled processes; stock post-upload RSS was about 0.88–0.89 GiB,
with CPU peak RSS near 14.8 GiB. Native retains its CPU text inventory; stock releases
CPU text parameters after upload. Compiled-run RSS reached about 2.75–2.78 GiB after
execution. RSS is not CUDA allocation or a hard memory grant.

Dropping only `model` while retaining the compiled callable left its captured model
allocations live in the initial probe. Dropping the callable and resetting Dynamo before
model cleanup returned both later compiled runs to the same 32 MiB allocator remainder
as eager execution. All native runs returned charged bytes to zero. Every selected
process completed; afterward there were no compute applications and device use was
33 MiB. This does not inject OOM, failed enqueue/drain or cancellation recovery.

## Reproduce the preparation seam

Use the pinned package, environment and [reference histories](gemma4-reference/README.md).
Set offline Hub flags and `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing Torch; apply
the numerical flags above and run under `torch.inference_mode()`. The core stock setup is:

```python
full, loading = Gemma4ForConditionalGeneration.from_pretrained(
    package, dtype=torch.bfloat16, attn_implementation="eager",
    local_files_only=True, trust_remote_code=False, output_loading_info=True,
)
assert not any(loading.values())
with torch.device("meta"):
    text = Gemma4ForCausalLM(full.config.text_config)
text.model, text.lm_head = full.model.language_model, full.lm_head
assert text.lm_head.weight.data_ptr() == text.model.embed_tokens.weight.data_ptr()
del full
text.eval().to("cuda")
cache = StaticCache(config=text.config, max_cache_len=4096)
geometry = text.config.per_layer_config[:len(cache.layers)]
cache.early_initialization(
    1, [g.num_key_value_heads for g in geometry], [g.head_dim for g in geometry],
    torch.bfloat16, torch.device("cuda"),
)
prepared = torch.compile(text.forward, fullgraph=True, dynamic=True, mode="reduce-overhead")
# Eager prefill into cache, then prepared single-token forwards with explicit
# position_ids, use_cache=True and logits_to_keep=1. Complete the BF16 .cpu()
# row and synchronize inside the clock; validate and write captures outside it.
```

Retain active-tensor checks, exact common IDs, full rows, source/version hashes, process
exits, first-call costs, all warmup/timing samples and allocator/RSS observations. Raw
probe source, manifests, logs and comparisons are retained outside Git; the first eager
attempt's loader-set JSON serialization failure is retained and excluded from timings.
No production engine code changed, so its existing Cargo/device qualification was not
rerun; the new evidence consists of the actual native, stock and compiled CUDA probes.

## Decision and incumbent scope

Whole-forward model/compiler reuse is concrete enough to keep investigating at the
model/backend boundary. It does **not** yet establish Rust-loadable execution, model-support
velocity or a maintained cross-backend contract. Do not start a universal exporter or
replace the execution owner. Preserve the independent CPU oracle and numerical gates.
The measured native prefill loss, not merely a plausible launch-overhead story, should
drive the next bounded execution experiment.

An engine-level incumbent comparison remains open. Source review found real Gemma 4
loaders/decoders in released [vLLM v0.32.0](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/model_executor/models/gemma4.py)
and [SGLang v0.5.22](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/models/gemma4_causal.py),
not just registry entries. Neither was installed or run here; source eligibility is not
4090 qualification. SGLang's released [one-batch runner](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/benchmark/one_batch.py)
is a concrete starting point for direct engine execution, with real runner/cache/graph
setup. Match histories and the host-output clock before comparing its numbers; then
measure actual serving/SLO behavior separately. Flare's mechanisms remain relevant,
but its unmodified hardware gate excludes SM89.

### SGLang capture adapter: host-checked, CUDA pending

[`gemma4_sglang.py`](gemma4_sglang.py) prepares the pinned SGLang 0.5.22 one-batch
comparison. It has **not run on CUDA**. Its host tests protect history continuity,
forced-token selection, malformed input rejection and preservation of failure evidence;
they do not qualify model loading, kernels, graph replay or GPU cleanup.

Validate the archived CPU BF16 input corpus without importing Torch or SGLang:

```sh
python3.13 benchmarks/gemma4_sglang.py \
  --snapshot <local-pinned-package> --reference <cpu-bf16/reference.json> \
  --output <new-plan-directory> --validate-only
```

The plan contains the three actual reference cases and the seven cycled-chat timing
prefixes. Validation leaves `complete: false` and creates no logits. In a separately
resolved Linux/CUDA environment, omit `--validate-only` to attempt capture. Use a new
output directory for every process; existing evidence, including failures, is never
replaced. Completed forwards are journaled before the next launch, so a later failure
does not erase earlier samples. Run three fresh processes per configuration for the
eventual comparison.

The adapter follows the released runner's publication/bootstrap, model loading, pool,
attention and graph initialization. It uses the original checkpoint's native
`Gemma4ForConditionalGeneration`: the release's
[`--language-model-only` allow-list](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/server_args.py#L334-L339)
excludes Gemma, and its conditional constructor retains vision/audio modules. No config
flattening, class replacement or media-weight deletion is applied. Active text inference
uses the same checkpoint, but retained media changes preparation and memory boundaries.

This first diagnostic uses one request, Triton attention, a requested 4096-token pool,
no prefix reuse/overlap/speculation, disabled prefill graphs and a one-request full decode
graph. `--eager` disables decode graphs too. It records resolved graph settings and each
forward result's `can_run_graph`; configuration alone is not replay evidence. These are
bounded bring-up settings, not a claim about SGLang's best/default serving configuration.
The sampler is omitted; every decode consumes the next archived `input_ids`, even if
an engine prediction differs. Pools are cleared only after successful synchronization;
a failed forward/synchronization aborts the process rather than reusing its state.

SGLang's native logits processor returns **FP32**, including FP32 final softcapping.
The clock covers request/batch input preparation through the complete, synchronized
262,144-element FP32 host row; checks, argmax and files stay outside it. Quality rows
are stored as little-endian `.f32le` (1,048,576 bytes), without BF16 rounding. All four
trials and three forced continuations per prefix are retained; trial zero is warmup.
`complete: true` means capture finished, not numerical acceptance. **Do not compare these
clocks directly with the earlier BF16-host table.** Rerun native/stock controls with
FP32 host output, including their BF16-to-FP32 conversion inside the clock, and retain
full-row diagnostics against the unchanged independent CPU reference.

The released [dependency metadata](https://pypi.org/pypi/sglang/0.5.22/json) matches the
[source pins](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/pyproject.toml):
Torch 2.14.1, Transformers 5.19.0, tokenizers 0.23.2, sglang-kernel 0.4.9 and FlashInfer
Python 0.7.1. Do not modify the earlier canonical/reference environments. The tool checks
these versions and hashes the pinned runner, batch, model and logits source files before
importing SGLang. It records all installed package versions and checks checkpoint files.

### Isolated Linux dependency recipe

The 2026-10-21 metadata review found Linux CPython 3.13 wheels for the engine/core
packages; the SGLang wheel requires glibc 2.34. PyPI's
[`cuda-tile==1.6.0rc5`](https://pypi.org/pypi/cuda-tile/1.6.0rc5/json) sdist is a
`wheel_stub` downloader, **not compiler source to build**. Its `pyproject.toml` points
to [NVIDIA's package index](https://pypi.nvidia.com/cuda-tile/), which has the matching
CPython 3.13/x86_64 wheel. The [requirements input](gemma4-sglang-requirements.in)
pins that same-version artifact by URL and SHA256 without adding an unrestricted extra
index or changing SGLang's dependency versions.

With that wheel, binary-only resolution next fails on `outlines-core==0.1.26`, required
by `outlines==0.1.11`: no compatible CPython 3.13 wheel is available. Allowing **only
that CPU Rust extension** to use a source distribution closes the full dependency graph.
This was metadata resolution on the Mac, not a Linux build or a CUDA import test.
Generate and retain a hash-locked environment outside the repository:

```sh
CARGO_BUILD_JOBS=2 uv pip compile benchmarks/gemma4-sglang-requirements.in \
  --default-index https://pypi.org/simple --exclude-newer 2026-10-22 \
  --python-platform x86_64-manylinux_2_34 --python-version 3.13.11 \
  --only-binary :all: --no-binary outlines-core --generate-hashes \
  --output-file <new-evidence-directory>/requirements.lock
```

On **Linux**, with the existing Rust toolchain and Python development headers available:

```sh
uv venv --python 3.13.11 <new-engine-environment>
CARGO_BUILD_JOBS=2 uv pip sync --python <new-engine-environment>/bin/python \
  --only-binary :all: --no-binary outlines-core --require-hashes \
  <new-evidence-directory>/requirements.lock
uv pip check --python <new-engine-environment>/bin/python
```

Do not run this GPU environment installation on macOS, modify the old reference/canonical
environments, bypass dependencies with `--no-deps`, or silently change pins/Python versions.
The CPU extension build, actual installation, CUDA imports, driver/library compatibility,
model loading and SM89 execution remain pending. Preserve the resolved lock, build/import
logs and installed-package identities. After installation, qualify actual 18/511/513
histories and graph/eager outputs before interpreting timings; HTTP scheduling, mixed
arrivals, capacity and SLO goodput remain separate gates.
