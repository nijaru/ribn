# Gemma 4 CUDA: prepared execution and serving diagnostics

Measured 2026-10-21 on the RTX 4090, 24,564 MiB, driver 615.71.10.
**Native prefill remains 2.81×/5.81× slower than stock Transformers at 18/64 tokens.**
Native warmed decode is modestly faster than stock eager. Stock compilation demonstrates
reusable whole-forward execution, but specialization, numerical differences and ownership
still matter. That preparation experiment is a direct-call diagnostic, not competitive
serving or quality acceptance. Later sections add [owned-FP32 direct calls](#owned-fp32-host-output-engine-comparison-2026-10-22)
and a [loopback serving diagnostic](#matched-loopback-serving-diagnostic-2026-10-22).

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
execution. RSS is not CUDA allocation or a hard memory grant. These measurements
predate the [host-lookup capacity change](#cuda-host-lookup-capacity-qualification-2026-10-29);
the old timing/RSS comparison is not rerun or relabeled.

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
not just registry entries. SGLang now runs on the 4090 in the bounded direct-call
comparison below; vLLM remains source-reviewed, not device-qualified. SGLang's released [one-batch runner](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/benchmark/one_batch.py)
is a concrete starting point for direct engine execution, with real runner/cache/graph
setup. Match histories and the host-output clock before comparing its numbers; then
measure actual serving/SLO behavior separately. Flare's mechanisms remain relevant,
but its unmodified hardware gate excludes SM89.

### SGLang capture adapter

[`gemma4_sglang.py`](gemma4_sglang.py) runs the pinned SGLang 0.5.22 one-batch
comparison. Its host tests protect history continuity, forced-token selection, malformed
input rejection and preservation of failure evidence. The CUDA capture below separately
exercises model loading, eager execution and one-request decode graphs; it does not qualify
arbitrary GPU failure recovery or serving.

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
does not erase earlier samples. Run three fresh processes per configuration; the comparison below retains all three.

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
Resolution was initially metadata-only. The separate CPU build/import check is scoped
below; the subsequent full Linux installation and CUDA capture have their own evidence.
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
  --exclude-newer 2026-10-22 \
  --only-binary :all: --no-binary outlines-core --require-hashes \
  <new-evidence-directory>/requirements.lock
uv pip check --python <new-engine-environment>/bin/python
```

Do not run this GPU environment installation on macOS, modify the old reference/canonical
environments, bypass dependencies with `--no-deps`, or silently change pins/Python versions.
The original resolution cutoff is explicit during sync: the desktop's relative default
cutoff otherwise filtered out an exact locked package. No global configuration changed.
On this Fedora host, CUDA 13.3 rejects the default GCC 16. The already-installed GCC 15
passes a CUDA compile probe and is selected only for these SGLang processes:

```sh
CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15 NVCC_CCBIN=/usr/bin/g++-15 \
  <new-engine-environment>/bin/python benchmarks/gemma4_sglang.py \
  --snapshot <local-pinned-package> --reference <cpu-bf16/reference.json> \
  --output <new-capture-directory>
```

No unsupported-compiler flag, package pin, model class or upstream source was changed.
The full 206-package hash-checked installation, `uv pip check`, CUDA imports, model loading
and SM89 eager/graph capture now pass. Preserve failed attempts, the effective Outlines
Cargo lock, installed identities and all samples. HTTP scheduling, mixed arrivals, capacity
and SLO goodput remain separate gates.

### CPU-only source-build qualification

The unpatched [`outlines-core==0.1.26` sdist](https://pypi.org/pypi/outlines-core/0.1.26/json)
(SHA256 `481c4301341e77cc8f1832d616784adb4d461b4fec65878e7c0d2cba7163a189`)
built and imported on Linux x86_64/glibc 2.36 with CPython 3.13.11, Rust 1.98.0 and
uv 0.13.4. This ran in an emulated Linux container on the Mac, capped at two CPUs and
4 GiB, not on the desktop. Only Outlines Core and its six runtime dependencies were
installed, using the same versions and hashes as the full SGLang lock. No CUDA stack,
model weights or modified package pins were used.

The source build, `uv pip check`, native extension import and Rust `FSMInfo` construction
passed; a small regex FSM accepted/rejected the expected strings. This closes that CPU
build/import prerequisite, not complete grammar behavior, SGLang integration, a glibc
2.34 portability claim or GPU qualification. The generated wheel is tagged
`cp313-cp313-linux_x86_64`, not a qualified portable manylinux release.

Retain the effective `Cargo.lock` and Python build-package identities with each build:
the sdist ships no Cargo lock, and the Python requirement lock alone does not freeze
Rust compilation. The successful resolution used PyO3 0.22.6 and bincode 2.0.1; build
packages included setuptools 84.2.1, setuptools-scm 10.3.4 and setuptools-rust 1.13.0.
Recheck the complete environment on the actual CUDA host before model bring-up.

### Compare completed FP32 captures

[gemma4_compare.py](gemma4_compare.py) checks one completed capture against the archived
CPU BF16 oracle without importing Torch or SGLang. Run it with NumPy (host-checked with
2.5.4) and the exact producer source archived with that capture:

```sh
python3.13 benchmarks/gemma4_compare.py \
  --report <capture-directory>/report.json --source <archived-producer-source> \
  --reference <cpu-bf16/reference.json> --tensors <cpu-bf16/tensors.safetensors> \
  --output <new-comparison.json>
```

The reader checks checkpoint, manifest, reference implementation/package and producer
identities, plus hashes, byte counts, encodings and finiteness of the 12 stored quality
rows. It validates every consumed input, position and full-row metadata record in the
three quality sequences and 28 timing sequences; timing rows have producer finiteness
checks, not separately archived full vectors. It rejects validation-only, failed,
partial or reordered captures. Source hashes establish
artifact consistency, not proof of execution: inspect the original report's hardware,
packages and settings before comparing engines. Existing output is never overwritten.

CPU BF16 rows are promoted exactly to FP32, including signed zero. Observed FP32 values
are **not** rounded to BF16. Full-distribution diagnostics use FP64: KL is CPU-to-observed
with natural logs, TV is half L1, centered RMSE removes the mean logit delta, and relative
L2 is null when the reference norm is zero. Bit equality and winner agreement are
reported separately; no acceptance budget is selected.

The comparison keeps the 12 common-history rows separate from the old nine-row compiled
scope. Per-prefix medians use three prefill and nine decode samples from one process;
trial-zero samples remain in the source capture but are excluded from those medians.
Do not treat these as multi-process distributions or serving/SLO evidence.

The prepared native/stock experiment controls preserve BF16 model/head arithmetic and
promote their completed host rows inside the clock. Native uses an owned FP32 vector
with `f32::from_bits(u32::from(bits) << 16)`; stock converts the synchronized CPU BF16
tensor to FP32. SGLang instead transfers its genuine FP32 output. This matches the host
API contract, not internal transfer widths or kernels. Native host compilation and an
exhaustive 65,536-encoding conversion test, real CPU-corpus reads, and analyzer data/CLI
tests pass. The completed native/stock/SGLang captures below use this boundary. They do not select
a numerical acceptance budget or promote a new execution variant.

## Owned FP32 host-output engine comparison (2026-10-22)

The restored RTX 4090 now has three completed fresh processes for each of four configurations:
Ribn's qualified eight-row native path, stock Transformers eager, SGLang eager and SGLang
with a full one-request decode graph. Production Ribn source is `ff7ed2d`; only qualification
tests/documentation changed during capture. All use the pinned BF16 package and exactly the
same archived inputs/continuations. Each process retains 12 complete quality rows and four
trials at every timing prefix; trial zero remains recorded but is excluded below.

The clock ends with an owned, synchronized **FP32 host row**. Native and stock retain BF16
model/head arithmetic and promote the completed CPU row inside the clock; SGLang transfers
its genuine FP32 logits, including FP32 final softcapping. Equal host interfaces do not imply
equal internal arithmetic or transfer widths. The earlier BF16-host table is a separate
experiment, not a source of samples for this one.

### Direct-call clocks

Milliseconds below are medians of three process medians. Each process contributes three
non-warmup prefill samples per prefix; its pooled decode median uses 63 samples across the
seven prefixes. All samples, warmups, process exits and first-call/preparation costs are
retained. These cycled-chat prefixes are a shape diagnostic, **not representative serving
traffic, tail latency or throughput/capacity evidence**.

| Prompt tokens | Ribn native 8 | Stock eager | SGLang eager | SGLang full decode graph |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 17.990 | 24.772 | 16.911 | 16.842 |
| 8 | 20.283 | 23.726 | 17.879 | 17.762 |
| 18 | 68.377 | 24.245 | 18.174 | 18.163 |
| 32 | 73.625 | 24.303 | 18.165 | 18.318 |
| 64 | 146.844 | 25.474 | 18.152 | 18.280 |
| 65 | 163.449 | 27.816 | 19.018 | 18.324 |
| 128 | 297.069 | 28.783 | 18.567 | 18.434 |
| Decode, pooled prefixes | 19.271 | 21.235 | 16.533 | 12.037 |

Across processes, native 64-token prefill medians span 146.799–147.890 ms versus
17.881–19.174 ms for graph-configured SGLang; pooled decode spans 19.265–19.430 versus
12.037–12.044 ms. Native is about **8.0x slower at 64-token prefill and 1.60x slower at
decode** in this diagnostic. This is not a quality-equivalent serving comparison or a
Rust-versus-Python attribution.

Each SGLang graph process reports its real `DecodeCudaGraphRunner` and takes the graph
branch for all 93 decode forwards, never for its 31 prefills. Eager processes take neither
graph branch. Both resolve to Triton attention, one running request, context/pool 4096,
page size 1, static-memory fraction 0.85 and no Torch compilation. This is the explicitly
chosen bring-up configuration, not SGLang's best/default serving configuration.

### Common-history numerical diagnostics

All 12 archived histories (18/511/513-token cases and their forced continuations) have finite
262,144-element rows and matching CPU-reference winners in every process. Full-row bits
are stable across the three processes within each configuration. SGLang eager/graph also
produce identical FP32 bits for these 12 rows; none of the configurations equals CPU bits.
Maxima below are over these histories, not an independently selected acceptance budget:

| Configuration | Max absolute delta | Max KL, CPU → observed | Max TV |
| --- | ---: | ---: | ---: |
| Native 8 | 1.000000 | 0.004639 | 0.042821 |
| Stock eager | 1.265625 | 0.000966 | 0.013898 |
| SGLang eager/full graph | 0.915292 | 0.004033 | 0.031622 |

Winner agreement is not distributional, sampling or task-quality equivalence. Native and
SGLang have larger KL/TV maxima than stock despite their smaller maximum absolute deltas.
No tolerance is widened, CPU oracle changed or optimized variant promoted from these results.

### Environment and limits

Hardware is RTX 4090 24,564 MiB, driver 615.71.10; Fedora 44/glibc 2.43. Native uses project
Rust 1.98.0 and CUDA Toolkit 13.3.73. Stock reuses the unchanged isolated Transformers
5.20.2 overlay with Torch 2.14.0/tokenizers 0.23.3. SGLang uses the separate hash-locked
0.5.22 environment with Torch 2.14.1, Transformers 5.19.0, tokenizers 0.23.2,
sglang-kernel 0.4.9 and FlashInfer 0.7.1. Its CPU Outlines extension built with the existing
host Rust 1.99.1 and the same effective Cargo-lock SHA256
`f2524293a52b45872f4205c600099c16053c6123e27736ba55e1357c74dacb7e` as the prior CPU build.
No canonical/reference environment, package pin or upstream source changed.

The initial sync failure from the host's relative cutoff and first CUDA forward's unsupported
GCC 16 JIT failure are retained. The successful run uses the original lock cutoff and
existing GCC 15 as shown above, not an unsupported-compiler override. Native controls use
an isolated Cargo target, the current borrowed-model binding and a freshly hashed binary.

SGLang retains its conditional model's media modules; native and stock time text-only
execution. Preparation boundaries therefore differ and are not compared as matched cold
loads. Fresh processes do not imply cold filesystem/JIT caches. Pools are explicitly prepared;
allocator counters or successful process cleanup do not establish physical peaks or hard
capacity. The [CUDA lifecycle/HTTP gate](runtime-contract.md#cuda-device-requalification-2026-10-22)
is separate functional evidence, not a serving-performance result.

The measured prefill scaling and decode gap justify investigation of repeated native work
and backend-owned reuse/fusion, not a selected compiler/importer or automatic wider-row
promotion. The following serving diagnostic adds mixed inputs, concurrency and admission
overload; representative quality, physical-pressure and recovery acceptance remain open.

## Matched loopback serving diagnostic (2026-10-22)

**Native throughput stays near 38 output tokens/s as concurrency rises.** SGLang reaches
281–291 at c4/c8; native c8 TTFT p95 rises to 4.68 s versus 0.49 s. These are measured
serving deficits, **not quality-equivalent ratios**: generated continuations differ.

### Workload and controls

- Same pinned BF16 package, RTX 4090 and driver as above. All nine package files were
  freshly hash-checked, including the weights. Ribn uses the qualified release CUDA HTTP
  binary SHA256 `4821c97c19bf64ec51a11a00a1bd79ba9008b612e61a94441d401d87729a56a6`,
  production code `ff7ed2d` (unchanged by qualifier `fd1c5b1`). SGLang uses the unchanged
  full hash-locked 0.5.22 environment above; no upstream source, model class or pins changed.
- AIPerf 0.13.0 runs on desktop loopback, in a separate Python 3.13.11 environment with
  Transformers 5.20.1/tokenizers 0.23.3. Full dependency pins/hashes, installed identities,
  commands, native exports and server logs are retained. Neither the SGLang nor reference
  environment was modified. The client's tokenizer files match the package byte-for-byte.
- Every phase sends the same 48 synthetic prompts, seed 42, nominal input 64±48 tokens,
  fixed output cap 32, temperature zero and EOS enabled. Actual inputs are 1–179 tokens,
  median 67, p95 145.95, totaling **3,382**. Raw prompt hashes/multiplicities match across
  all 24 phases; successful server prompt usage also matches independent tokenization.
  There are no decoded-text stops, minimum-output constraints or EOS bypasses.
- Three fresh server processes per engine, one full-model GPU process at a time. Order
  alternates across repetitions. Each process runs c1, c4, c8 and constant-rate 4 requests/s
  with a c32 ceiling. Server startup/warmup and one canonical four-token HTTP preflight
  precede measurement; each phase additionally excludes four client warmup requests.
  All six servers and all 24 clients exit zero; owned GPU processes/groups settle before
  the next launch. Fresh processes are not cold filesystem/JIT caches.
- Both engines allow four active sequences and context 4096, with prefix reuse disabled.
  Ribn retains its default 16 application permits and 20 GiB model pool, allocating a full
  4096-position cache per admitted sequence. SGLang uses TP1, Triton attention, page size 1,
  total KV pool 4096, at most 12 queued requests, static memory fraction 0.85, full decode
  graphs through batch size four, and disabled prefill graphs. Other serving defaults,
  including overlap, are retained. Its logs show four-row running/prefill batches and
  graph decode. These are explicit bounded configurations, not identical physical-memory
  grants or a claim about the best/default SGLang configuration.

### Independently reconciled results

For each phase, raw monotonic timestamps reproduce actual-init TTFT and last-nonempty-content
latency. `[DONE]` receipt is checked separately, not called HTTP EOF. Finite-phase throughput
uses actual terminal usage divided by aggregate `end_time - start_time`, including drain;
using the content-window `benchmark_duration` instead would change the denominator.
The reported clocks, rates and joint deadline counts reproduce the native exports.
General `performance_metrics_audited` remains false; this does not audit every AIPerf metric
or qualify token-gap tails, exact EOF timing or load-generator capacity.

Before measurement, diagnostic objectives were set to **TTFT ≤1,000 ms and last content
≤5,000 ms together**. They are not product SLOs. Failed attempts never pass; diagnostic
goodput is passing requests divided by the finite phase duration. Latency percentiles below
cover successful requests only. Each cell is the median of three process results; p95 is
computed within each 48-attempt phase before taking that median.

| Engine / client load | Successes / 48 | Output tokens/s | TTFT p95 ms | Last-content p95 ms | Diagnostic goodput requests/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ribn c1 | 48 | 37.63 | 397.6 | 1027.9 | 1.225 |
| SGLang c1 | 48 | 85.30 | 30.5 | 381.0 | 2.724 |
| Ribn c4 | 48 | 37.73 | 1055.2 | 3811.7 | 1.100 |
| SGLang c4 | 48 | 281.16 | 53.2 | 466.0 | 8.985 |
| Ribn c8 | 48 | 37.75 | 4680.3 | 7665.3 | 0.102 |
| SGLang c8 | 48 | 290.98 | 487.5 | 906.6 | 9.299 |
| Ribn nominal 4/s, c32 | 29 | 38.25 | 9292.1 | 12057.4 | 0.176 |
| SGLang nominal 4/s, c32 | 48 | 121.14 | 44.7 | 441.5 | 3.869 |

Native diagnostic passes are 48/43/4/4 in those load shapes; SGLang passes all 48.
Native c4 output throughput spans 37.69–37.76 tokens/s across processes; SGLang spans
269.72–281.57. Native c8 spans 37.73–37.78 versus 290.96–291.09. Preserve all repetitions,
not just their medians. The rate-controlled native phase has **19 HTTP 503 failures** in
every repetition; all attempts remain accounted for, without crediting failed token work.
Actual start windows span 11.748–11.750 s for 48 rate-controlled attempts on both engines.
AIPerf can suppress scheduled ticks at its concurrency ceiling and reanchor late schedules;
this is bounded rate-controlled traffic, not an independently exogenous arrival ledger.
See [client measurement contracts](README.md#audited-endpoint-runner).

### Quality limits and next decision

EOS changes realized work. Native closed-loop phases each generate 1,475 committed tokens,
including four stop finishes; SGLang c1 generates 1,503 with three stop finishes. SGLang c4
varies from 1,477 to 1,502 tokens, and c8 generates 1,502. Output counts are not substituted
with `48 * 32`. At c1, visible text, usage and finish agree between engines on **34/48**
prompts in each repetition. Native successful outputs remain identical across its load
shapes, while SGLang's batched outputs can change with scheduling. This diagnostic neither
classifies those differences against the independent CPU oracle nor establishes task-quality,
distributional or sampling equivalence. The earlier 12 common-history winner checks do not
extend to every generated continuation here. Do not turn these throughput ratios into a
quality-qualified competitive claim.

The flat native c1/c4/c8 throughput and growing queue-visible latency make cross-sequence
physical execution and prefill scaling concrete optimization candidates. Four logical
sequences currently execute decoder calls sequentially; that source fact is not a measured
attribution of the whole deficit. Choose a bounded experiment with end-to-end repeats and
numerical gates, rather than widening rows or selecting a universal compiler by default.
Physical peak/tight-grant recovery, long contexts, realistic content/arrivals, independent
quality acceptance and an incumbent-native client cross-check remain open. This is not the
complete competitive acceptance matrix.

### Reproduce

Use the isolated Linux dependency recipe above and existing supported GCC 15 for SGLang.
The server commands are:

```sh
ribn serve-gemma <pinned-package> --bind 127.0.0.1:18080 --model-name gemma
CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15 NVCC_CCBIN=/usr/bin/g++-15 MAX_JOBS=2 \
  sglang serve --model-path <pinned-package> --served-model-name gemma \
  --host 127.0.0.1 --port 18081 --dtype bfloat16 --context-length 4096 \
  --max-total-tokens 4096 --max-running-requests 4 --max-queued-requests 12 \
  --mem-fraction-static 0.85 --attention-backend triton --page-size 1 \
  --cuda-graph-max-bs-decode 4 --cuda-graph-backend-prefill disabled --disable-radix-cache
```

Run only one full-model server at a time. Fill the existing endpoint manifest with the
cohort, server identity and diagnostic objectives above; vary concurrency/rate, preserve
new output directories, and repeat with fresh servers. In AIPerf 0.13 offline mode, use
`google/gemma-4-E4B-it` plus its exact revision and a populated cache, not an absolute
package path: that code path calls the Hub cache resolver even for a directory. This study
uses a private `HF_HUB_CACHE` with byte-checked `config.json`, `tokenizer.json` and
`tokenizer_config.json` under `models--google--gemma-4-E4B-it/snapshots/<revision>`.

The two failed absolute-path client attempts are retained and excluded from results.
A separate startup pilot used process-group SIGTERM, killing SGLang workers before its
parent could drain; its resulting -9 exit is retained, not reported as graceful shutdown.
The actual study signals only each parent, verifies settled owned processes and records
six zero exits. Neither fix changes library code or canonical caches/environments.

## Synchronous device PLE experiment (2026-10-22)

**Rejected; all candidate code and candidate-only tests reverted.** Removing the PLE
projection readback and per-layer uploads did not meet the predeclared performance gate.
This does not measure the round trip's standalone cost: the replacement still completed
one device operation per layer. It does not rule out amortizing completion across a
larger device-resident execution unit.

The baseline is `aa75df1`, whose production decoder is unchanged from `ff7ed2d`. The
candidate reused the raw projection allocation after scaling and RMS normalization had
completed, uploading the packed token identity there once. A strided kernel then read
one layer's columns from that identity and the resident normalized projection. It
preserved the CPU operation's two separate BF16 roundings:
`bf16(bf16(context + identity) * 0.70710677f)`. Each physical group replaced one packed
D2H read and 42 H2D uploads with one packed H2D upload and 42 synchronous kernel calls.
No additional device allocation was introduced. Selection was limited to the existing
RTX 4090/E4B eight-row preparation scope, including its scalar tails; other preparation
kept the host route.

Before implementation, the screen required at least **3% lower median prefill latency
at both 18 and 64 tokens**, with no more than **2% higher pooled decode latency**.
Serving, full-KV and sanitizer qualification were conditional on passing that screen.
The experiment allowed one candidate and at most one correctness repair, within a
90-minute device-investigation budget; no arithmetic repair or variant sweep was needed.

The same pinned package, RTX 4090, driver 615.71.10, CUDA 13.3.73, Rust 1.98.0 and
[owned-FP32 control plan](#owned-fp32-host-output-engine-comparison-2026-10-22) were used.
Three fresh processes per variant ran in baseline/candidate, candidate/baseline,
baseline/candidate order, serialized with idle GPU checks. Each process retained trial
zero as warmup and excluded it from medians. Each prefill median uses three nonwarmup
samples; decode pools 63 nonwarmup samples over the seven prefix lengths. These are
completed full-host-row direct calls, not HTTP clocks.

| Prefix tokens | Baseline ms | Device PLE ms |
| ---: | ---: | ---: |
| 1 | 17.910 | 17.979 |
| 8 | 20.200 | 20.297 |
| 18 | 68.375 | 68.454 |
| 32 | 73.383 | 73.427 |
| 64 | 146.905 | 146.663 |
| 65 | 163.329 | 163.262 |
| 128 | 297.144 | 296.926 |
| Pooled decode | 19.234 | 19.290 |

Values are medians of the three process medians. At 18 tokens the candidate was 0.12%
slower; at 64 tokens it was 0.16% faster; decode was 0.29% slower. Process-median ranges
were 68.202–68.923 versus 68.313–68.859 ms at 18 tokens, 146.454–147.843 versus
146.170–147.831 ms at 64 tokens, and 19.225–19.443 versus 19.271–19.603 ms for decode.
The observed differences do not establish a repeatable benefit and fail the 3% gate.

All six accepted model processes and the three component tests exited zero. The component
gate checked exact two-round arithmetic, packed row strides, column tails and rejection
before enqueue. Each model process captured all 262,144 logits for the same 12 common
histories; those rows were bit-identical across all six processes. Independent comparison
against the unchanged CPU oracle retained all 12 winners and the previous distributional
errors, not CPU bit equality or new task-quality acceptance. The one-cache, 4096-position
BytePool charge was identical at 15,174,441,952 bytes; this is not a physical-peak or
pressure/recovery result. A narrow independent source review found no material arithmetic,
bounds or ownership defect.

One earlier harness attempt stopped because a rolling GPU-utilization sample remained
6% after the model process had exited, despite no live compute process. Its captures are
retained and excluded; the accepted harness waits for an idle sample before proceeding.
The failing performance screen ended the experiment: no serving run, full-KV comparison
or sanitizer run followed, and no new execution variant is enabled. The qualified host
PLE route remains unchanged.

## Native CUDA execution-cost trace (2026-10-25)

**Collected activity points to repeated matrix work, not a demonstrated host-only gap.**
The trace carries a collection-completeness warning, so it does not establish a complete
GPU/host time partition or an idle-time budget. No execution variant was implemented
or promoted in this study.

The RTX 4090 and driver 615.71.10 were unchanged. Native source was `252b140`, with no
Gemma CUDA or NVIDIA BF16-compute source changes since `58f2fbe`. An isolated Rust 1.98.0
build reused the [owned-FP32 control plan](#owned-fp32-host-output-engine-comparison-2026-10-22):
all 12 common quality rows, seven prefix lengths, four trials per prefix and three forced
continuations per trial. Trial zero remained warmup. The prepared lane was eight rows,
with a 4096-position cache and sufficient scratch. These are direct decoder calls ending
in a completed owned FP32 host row, not HTTP, greedy-continuation or traffic measurements.

Three fresh processes ran untraced, traced, untraced, serialized with idle GPU checks.
All three exited zero without forced cleanup; launcher/model process groups were
positively absent within their original deadlines. The frozen limits were 90 minutes
overall, 600 seconds per process including 60 seconds reserved for cleanup, and 3 GiB
of artifacts. Each worker verified the nine checkpoint assets and reference identity
before loading; all 12 full quality rows were bit-identical to the earlier qualified
native capture and across these three processes. Those rows ran **outside** collection.
The traced timing calls retained finiteness and winners, not full rows, so this does not
prove full-row parity under tracing or new agreement with the CPU oracle.

Nsight Systems 2026.3.2 collected 24 separate ranges: prefix 18 and 64, trials 1–3,
prefill and each of three continuation calls. Loading, quality controls and warmup were
excluded. The retained probe calls cudarc's checked `profiler_start()` before the
forward timer and `profiler_stop()` after owned-row completion and the timer read.
Token ID vectors were prepared before the timer. With paths supplied by the frozen
run manifest, its capture command was:

```sh
PROBE_PROFILE=1 PROBE_ROWS=8 PROBE_PLAN="$PLAN" PROBE_OUTPUT="$OUTPUT" \
RIBN_GEMMA_PACKAGE="$PACKAGE" RIBN_GEMMA_REFERENCE="$REFERENCE" \
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --cuda-memory-usage=false --capture-range=cudaProfilerApi \
  --capture-range-end=repeat:24:defer --output="$TRACE_PREFIX" \
  "$OWNED_ROW_PROBE"
```

`OWNED_ROW_PROBE` names the frozen qualification harness, not a repository CLI command;
`PLAN` is the existing validated input-only plan and both output paths must be fresh.
`defer` delays report generation; it is not evidence that CUDA-buffer flushing was disabled.
The probe source SHA256 is
`73ddfdfa02e03582c415c5c1e2eb2ab00b84a894725af1f848f4d474550bf036`.

The two untraced process medians were **68.166/68.232 ms** at 18 tokens and
**146.342/146.488 ms** at 64 tokens. Their pooled decode medians, each over 63 nonwarmup
calls, were 19.259/19.342 ms. Tracing perturbed latency; the following traced clocks
are not replacement production timings or new cross-engine ratios.

| Traced call | Samples | Call ms | Collected GPU union ms | Collected matrix/reduction ms | Kernel records | Stream-sync calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prefill 18 | 3 | 77.663 | 50.873 | 40.842 | 5,026 | 5,063 |
| Prefill 64 | 3 | 161.678 | 113.411 | 86.572 | 10,378 | 10,124 |
| Decode after 18 | 9 | 21.617 | 12.944 | 10.507 | 1,217 | 1,268 |
| Decode after 64 | 9 | 22.308 | 13.945 | 10.496 | 1,217 | 1,268 |

Each column is a median, not an additive decomposition. GPU intervals union the collected
kernels and copies, rather than adding overlapping durations. API intervals are clipped
between the end of `cuProfilerStart` and the start of `cuProfilerStop`; profiler-triggered
`cuCtxSynchronize` after stop is excluded. The collected stream-synchronization intervals
alone span medians of 59.971/132.254 ms for prefill 18/64, **overlapping device execution**:
they are not an additional host cost. Collected matrix/reduction kernels account for
median shares of 80.9%/76.9% of collected prefill kernel time. Their names include cuBLAS
WMMA GEMMs, GEMV kernels and split-K reductions; kernel names do not establish occupancy,
bandwidth saturation or stall causes.

**Collection limit:** all 24 exports warn, “Not all CUDA events might have been collected.”
Every collected kernel has a matching launch correlation; independent read-only checks
also reconcile memcpy correlations, one device/context/stream and valid timestamps.
Those checks cannot exclude an omitted API/activity pair. Produced-versus-collected event
counters cover different recorded categories and are not a demonstrated dropped-kernel
count. The tool reports software-instrumented tracing; hardware tracing is unavailable.
Collected duration unions therefore supply lower bounds within these instrumented calls,
not complete utilization. Their complements are not established idle or host overhead.
The warning's cause remains unresolved; metadata alone does not justify a flush-setting
repair or an unchanged rerun.

**Selected experiment:** the [private 16-row screen](#private-16-row-prefill-screen-2026-10-25)
targets repeated matrix traversals rather than another PLE-only patch. This device evidence,
not a row-count default, motivates the candidate. Its independent numerical and ownership
gates were frozen before implementation; changing cuBLAS row count is already known not
to guarantee BF16-bit preservation. The trace alone neither supplies those gates nor
justifies automatic wider-row selection, a compiler ABI or quality-equivalent acceptance.

Frozen probe sources/locks/binary, protocol, all three reports and quality rows, 24 raw
traces and SQLite exports, `analysis-v2.json` and post-run identity audits remain under
`gemma4-cuda-profile/20261025T082600Z` in the qualification store. Earlier analysis is
retained, with its coverage label corrected in v2 rather than treated as GPU utilization.

## Private 16-row prefill screen (2026-10-25)

**Initial cohort: numerical checks passed; the performance screen is incomplete.**
The [fresh repeatability cohort](#private-16-row-repeatability-rescreen-2026-10-25) below
records the later advancement result without reclassifying this interrupted run.
Two complete pairs
show substantially lower prefill latency, but an unexpected GPU PID interrupted the
last candidate process. The original three-pair gate was not met. No candidate code or
automatic-selection change was merged in that cohort; eight-row/scalar selection remained.

Both detached source trees start at `a6c47ee`, on the same RTX 4090, driver 615.71.10
and pinned E4B package. The private candidate prepares 16- **and eight-row** activation/PLE
working sets before publication, one maximum-reach attention workspace and per-layer
rotary factors with capacity 16. It consumes 16-row groups, then eight-row groups, then
scalar tails through the existing layer implementation. A ceiling sufficient only for
eight rows retains that lane. Weights, completion-owned K/V, whole-chunk validation,
synchronous execution and failure-retirement owners are unchanged. No kernel, precision,
cuBLAS mode, decoder head or frontend changed.

The frozen numerical/ownership gates retain the existing
[eight-row qualification policy](gemma4-reference/README.md#bounded-eight-row-cuda-prefill-2026-10-21-utc):

- The independently derived exact-integer GEMM forward-error bound passes all 12 selected
  E4B projection geometries at rows 1/8/16, without a fitted tolerance. Existing exact-dot
  and stream-identity checks also pass.
- Eight- and sixteen-row attention bit-matches scalar queries at key spans equal to the
  row count, 513 and 4096, for width-256/sliding-512 and width-512/full attention. The
  independent host-accuracy and invalid-geometry gates remain intact and pass.
- All 19 window/chunk evaluations agree with the 12 distinct archived CPU greedy predictions,
  including coalesced 512-token and shifted `[1,512]` inputs. Full rows are finite;
  positions, reset and whole invalid-tail/capacity rejection pass. The existing scalar
  partition test also passes. These are common histories, not new free-running quality data.
- Budget refusal after a key allocation and after a complete layer retains the existing
  retirement/refund behavior with an actual 16-row-prepared model. This is not driver OOM
  or injected device-enqueue failure qualification.

All four gate processes exit zero, nonforced, with positive process-group absence inside
600-second deadlines including cleanup. Independent read-only source review found no
material arithmetic, bounds or ownership issue. It was static review, not device evidence.

The latency protocol keeps the owned-FP32 clock and existing input-only plan: seven
prefix lengths, four trials per prefix, trial zero warmup, and three **forced** continuations
per trial. ID preparation and row writes are outside the clock. Three fresh processes
per variant were required in order 8/16, 16/8, 8/16. The frozen screen requires at least
3% lower 18/64-token process-median aggregates, improvements in all three pairs, no more
than 2% higher other-prefix or pooled-decode aggregates, and at most 16 MiB additional
charged memory. Limits are 90 minutes overall, 600 seconds per process with 60 seconds
reserved for cleanup, and 6 GiB of artifacts; no row-count sweep or precision changes.

The first five model processes finish zero, nonforced and retired within their original
bounds. Their per-process medians are below; each prefill median has three nonwarmup
samples and each pooled decode median has 63. These **two complete pairs** are descriptive,
not a reduced replacement for the predeclared three-pair screen.

| Call, ms | Pair 1: eight | Pair 1: sixteen | Pair 2: eight | Pair 2: sixteen |
| --- | ---: | ---: | ---: | ---: |
| Prefill 1 | 17.854 | 17.894 | 17.943 | 18.261 |
| Prefill 8 | 20.134 | 20.177 | 20.300 | 20.355 |
| Prefill 18 | 68.319 | 51.493 | 68.425 | 51.965 |
| Prefill 32 | 73.274 | 39.390 | 73.784 | 39.723 |
| Prefill 64 | 146.373 | 76.980 | 146.956 | 77.348 |
| Prefill 65 | 162.925 | 93.225 | 163.552 | 94.311 |
| Prefill 128 | 296.166 | 154.137 | 297.487 | 155.228 |
| Decode, pooled | 19.198 | 19.359 | 19.460 | 19.515 |

The observed reductions are 24.63%/24.06% at 18 tokens and 47.41%/47.37% at 64 tokens.
The unpaired third eight-row control is retained: 68.461/147.024 ms at 18/64 tokens and
19.294 ms pooled decode. Its missing candidate is not imputed. Charged model-plus-cache
memory is 15,174,441,952 bytes for eight rows and 15,181,768,992 for the candidate:
**7,327,040 bytes (7.0 MiB) more**, not a physical-peak, RSS or pressure/recovery result.
Each of the five completed workers returns its allocation charge to zero after teardown.

All **620 full host rows** are retained (124 per completed process, including repeated
warmup/trial inputs). They repeat bitwise across the available workers, and all 248
matched row comparisons across the two pairs are bit-exact between variants, including
scalar/eight-row fallbacks. This is observed equality on the frozen cohort, not a proof
for every selected input. The 12 quality rows still differ from CPU BF16; their maximum
CPU-to-native KL/TV remain 0.004639/0.042821. Those full-distribution diagnostics do not
define an acceptance tolerance or qualify sampling, task quality or serving equivalence.

**Interruption:** the final candidate worker was stopped with SIGINT (actual exit `-2`)
before any quality/timing row, when the guard observed GPU PID 162154 that did not match
the worker's identity. Only the owned worker group was signalled and positively retired.
The unexpected PID had disappeared by investigation; no matching journal metadata was
found. The guard preserved its PID but not the raw NVML row, command identity or process
group, so the cause cannot be established or dismissed as harmless. No replacement run
was started. The screen remains **inconclusive**, not numerically rejected or accepted.
Before a new frozen cohort, establish GPU isolation and retain that unexpected-process
metadata. Full long-context, sanitizer, executor and serving qualification were not run.

The frozen protocol, baseline/candidate source trees and patch, separate build targets,
identical probe sources, input/binary hashes, all six process records, interrupted report,
620 rows and `analysis-partial-v2.json` remain under
`gemma4-cuda-16row/20261025T174000Z` in the qualification store. Earlier analysis is retained.
The probe source SHA256 is
`88f0e7f6292cf8e7e7176bc50bbbfa1254f33fc5a11372f27d7e26d0e2d868f4`;
`candidate.patch` SHA256 is
`acd2be4ddcb803134389dba36dc34f0f3b6afd022fb8bfe9285e78ccfdcfcc33`.
The probe is a retained qualification harness, not a repository CLI; each variant runs
with its own frozen binary, a fresh output path and the same plan/reference:

```sh
PROBE_ROWS="$ROWS" PROBE_PLAN="$PLAN" PROBE_OUTPUT="$OUTPUT" \
RIBN_GEMMA_PACKAGE="$PACKAGE" RIBN_GEMMA_REFERENCE="$REFERENCE" \
"$OWNED_ROW_PROBE"
```

## Private 16-row repeatability rescreen (2026-10-25)

**The complete three-pair advancement screen passes.** This is repeatable benefit for
one private candidate, not product promotion, a quality-equivalent comparison or a
serving/SLO result. At this stage, selection remained eight-row/scalar pending the
[full-reach and integration qualification](#sixteen-row-full-reach-and-integration-qualification-2026-10-25).

An explicit exclusive-GPU handoff preceded a new frozen campaign on the same RTX 4090
and driver 615.71.10. It reused the exact baseline/candidate binaries and source patch
above, the owned-FP32 probe, all inputs and the existing independent numerical/ownership
prescreen evidence. All 2,392 frozen source/binary hashes and 12 input hashes were
reverified before and after collection. No kernel, arithmetic, precision, tuning or
candidate-selection change was made; no earlier measurements were pooled into this cohort.

The six fresh processes run 8/16, 16/8, 8/16, with the original seven prefix lengths,
four trials and forced continuation inputs. The gate remains at least 3% lower
18/64-token aggregates with improvement in every pair, at most 2% higher other-prefix
and pooled-decode aggregates, and at most 16 MiB additional charge. All gates pass.
The new campaign has a 30-minute overall deadline, 600 seconds per process including
60 seconds reserved for cleanup, and a 1 GiB output cap.

| Call | Eight-row ms | Sixteen-row ms | Latency reduction |
| --- | ---: | ---: | ---: |
| Prefill 1 | 17.975 | 17.915 | 0.3% |
| Prefill 8 | 20.321 | 20.165 | 0.8% |
| Prefill 18 | 68.523 | 51.570 | 24.7% |
| Prefill 32 | 73.597 | 39.218 | 46.7% |
| Prefill 64 | 147.038 | 76.743 | 47.8% |
| Prefill 65 | 163.842 | 93.448 | 43.0% |
| Prefill 128 | 297.268 | 154.525 | 48.0% |
| Decode, pooled | 19.400 | 19.220 | 0.9% |

Each figure is the median of three **process medians**, not a pooled nine-sample
prefill median. Each process contributes three nonwarmup prefill clocks per shape
and 63 pooled decode clocks. Individual pairs reduce prefill by 24.74%/24.91%/24.01%
at 18 tokens and 47.81%/47.97%/47.29% at 64. Decode arithmetic is unchanged; the small
observed decode difference is not evidence of a new decode optimization.

All six measured workers exit zero without forced cleanup. Their groups and both
supervisors are positively absent within the original campaign deadline; the successful
supervisor's actual exit zero is separately captured. All 744 full host rows pass an
independent post-run byte-hash, finiteness and argmax audit. All 372 matched comparisons
across three pairs are bit-exact between variants, as are repeated rows within each
variant. These are repeated frozen inputs, not 744 independent quality cases. All
72 quality captures also match the earlier eight-row captures exactly. CPU BF16
full-distribution differences remain unchanged: maximum KL/TV 0.004639/0.042821,
diagnostic only. Extra charge is still 7,327,040 bytes (7.0 MiB), not physical peak.
All completed workers return their allocation charge to zero. The GPU is empty before
the pre-existing model endpoint is restored after qualification; restoration is outside
all measured clocks.

**Monitor repair:** the first initialization attempt was stopped before any quality
or timing row. The new diagnostic log showed that `/proc` initially returned an empty
command line for the **owned** process, then a populated one; PID, start ticks, executable,
process group and session were identical. Gating lifetime identity on the mutable argv
hash caused a false alarm. One permitted collection-harness repair retains argv/hash
as diagnostics while enforcing those stable ownership checks; foreign or unobservable
identities still fail closed. CPU tests and replay of the actual saved event reproduce
the old rejection and the corrected classification. No foreign process was signalled.

Original source, the failed initialization and its actual `-2` exit remain intact;
`cohort-v2` holds all six new workers. The repair does not reset the original campaign
deadline, change thresholds or replace a measured failing case. Independent read-only
review found no consequential issue in this ownership correction or repair scope.
Raw NVML rows, process metadata and actual supervisor exit status are now retained.
The older campaign's unidentified PID is **not** retrospectively explained by this event.

Protocol, repair record, control sources, regression checks, full logs/rows, `analysis.json`
and post-run identity/retirement audits remain under
`gemma4-cuda-16row/20261025T195000Z` in the qualification store. Full-reach model,
Compute Sanitizer and executor/text integration were not part of this screen; the
follow-up below owns their advancement result. Competitive serving remains open.

## Sixteen-row full-reach and integration qualification (2026-10-25)

**Sixteen-row selection is enabled for the qualified RTX 4090/E4B lane.** Preparation
chooses sixteen only when the declared reach (16–4096 positions) and scratch ceiling
permit it, retaining eight-row and scalar fallbacks. Both batched working sets are
prepared before publication; one maximum-reach attention workspace is reused. The
ordinary CUDA loader's existing ceiling is sufficient. No kernel, model equation,
precision policy, decode arithmetic or serving policy changes.

The unchanged private binaries first run three archived CPU-control cases and three
long common histories: 4093 tokens plus three fixed continuations, 4095 plus one, and
a no-tail 4096-token prefill. Each long case has four repetitions. All **40 full-row
comparisons** between eight and sixteen are bit-exact, including signed zero; both
variants retain the 12 archived CPU greedy winners. These are 19 distinct history
states, not 40 independent cases. All 80 actual FP32 row files pass SHA, finiteness and
argmax audits. Each worker exits zero, nonforced, with zero final allocation charge,
positive group absence and idle GPU within its 600-second bound.

Memory checking separates expensive whole-model coverage from maximum-span primitive
coverage, without changing the numerical comparison:

- The sixteen-row full model prefills 4093 tokens and performs three fixed single-token
  continuations to reach 4096. This exercises sixteen/eight/scalar dispatch. All four
  instrumented full rows are bit-exact against the corresponding normal capture.
- The unchanged batched-attention fixture checks eight and sixteen queries, eight query
  heads/two KV heads, width 256 with a 512-token window and width 512 without a window,
  at key spans equal to the row count, 513 and 4096. It compares all 144 scalar-query
  outputs bitwise with the batched outputs and checks finiteness. This reference is the
  scalar CUDA path, not a newly collected CPU oracle; the independent host/operator
  qualification above still applies.

Both run under installed CUDA 13.3 Compute Sanitizer 2026.2.1.0 with `memcheck`, child
tracking, full leak checking and error exit 86. Each tool **and separately recorded
native process** exits zero, nonforced; both report **0 errors and 0 leaked bytes in
0 allocations**. Known native GPU identities are observed. Owned groups are positively
absent and the GPU idle before restoring the default endpoint. The full-model and
primitive bounds are 900 and 240 seconds, including 60 seconds reserved for cleanup.
This is focused coverage, not sanitizer execution of every numerical repetition,
driver-OOM recovery, arbitrary device failure or competitive serving qualification.

**Retained incomplete attempts:** an initially planned stock BF16 CPU corpus reproduced
all 12 archived controls exactly, then completed the 4093-plus-three case. The 30-minute
CPU cap forced exit `-9` during the next long prefill; the corpus remains incomplete and
is not accepted. Repeating three nearly identical full CPU prefills was unnecessarily
costly. A later instrumented four-repeat GPU driver took about eight minutes per 4K
prefill and was stopped rather than repeat identical coverage. Neither interrupted
scope is marked passed. The focused driver changes only its timing-loop repetition
count from four to one; model/kernel source is unchanged. The original 90-minute
campaign deadline is not reset. No new long-context CPU parity, distribution tolerance
or precision winner is inferred from GPU baseline preservation.

A separate, frozen 30-minute integration phase then exercises **actual sixteen-row
preparation**, not an unchanged eight-row fixture:

- Engine execution reproduces all four archived chat tokens, `Length` and 18/4 usage,
  while oversized input remains request-local rejection.
- A sixteen-token captured submission induces the existing synchronous-drain failure.
  Owners/charges remain retained until capture ends; shutdown retry retires them,
  execution stays closed and final charge is zero. This is not running-kernel failure
  injection or permission to enable Graph execution.
- The text facade drops a stream after its first archived token, then generates through
  a cloned handle using the same model. Tokens, decoded text, `Length`, usage and explicit
  shutdown/zero charge pass. The test does not assert an in-kernel cancellation race.
- A scratch-limited model still selects eight and passes the archived window continuations.
  The ordinary Linux CUDA `ribn run-gemma` loader also emits the exact independently
  decoded four-token text, `A mutex (short`, without a BOS workaround.

All four integration processes exit zero, nonforced, and retire inside their original
240-second bounds. An independent post-run audit rechecks 1,209 source/control/binary
pins and endpoint restoration. Charged model/cache memory remains **7.0 MiB above eight
rows**, not a physical-peak result. The previously measured 24.7%/47.8% reductions at
18/64 tokens support this selection; integration and sanitizer clocks are not new
performance samples. Sampling, task-quality equivalence, mixed-arrival capacity/pressure
and SLO goodput remain open.

Protocols, all failed/stopped attempts, identity snapshots, actual exits, raw rows,
loop-count equivalence and post-run audits are retained under
`gemma4-cuda-16row/20261025T204609Z`; the separate executor/text/CLI integration source,
binaries, logs and audit are under `gemma4-cuda-16row/20261025T221200Z`. The same package
and archived CPU identities apply. Required host gates remain separate from this device
evidence.

## Promoted sixteen-row serving comparison (2026-10-26)

**The promoted lane improves paired c4/c8 output throughput by 9.9%/9.8%, but native
throughput still flattens near 41 tokens/s.** This is an end-to-end improvement over
an eight-row control, not quality-equivalent competitive acceptance against SGLang.

### Controls and acceptance

The [loopback method above](#matched-loopback-serving-diagnostic-2026-10-22) is repeated
with three fresh processes for each of eight-row Ribn, sixteen-row Ribn and SGLang:
36 phases, each with 48 profiling requests and four excluded warmups. Process orders
are eight/sixteen/SGLang, SGLang/sixteen/eight, then sixteen/eight/SGLang. All phases
retain the same 48 raw prompts, including rejected requests: 1–179 input tokens,
3,382 total, seed 42, nominal 64±48 input, output cap 32, temperature zero and EOS
enabled. Successful prompt usage matches independent tokenization. The canonical
18-input/four-output-token preflight precedes each server's measured phases.

Both native binaries use source `74e4112`, Rust 1.98.0 and the same release CUDA build.
The private eight-row control changes only `PREFILL_LANES` in CUDA decoder preparation
from `[8, 16]` to `[8]`; the ordinary sixteen-row CLI is unmodified. Binary SHA256s are
`d7a17e9f1cb98359ef4f888bbbf3072fda04da2fd4c717391543916bbf61a4f4` and
`ac70983cdb6fa58427e5a8f8a43737879ffcd3ce310de98c28d879102b801bb0`, respectively.
No kernel implementation, precision, logical policy or Graph selection changes. The
RTX 4090, driver 615.71.10, CUDA 13.3.73 and pinned client/SGLang environments are retained.
All nine package-file identities and selected installed-source/metadata hashes are checked
before/after; neither dependency environment is modified. Native still uses its default
20 GiB pool and full per-sequence 4096-position storage; SGLang retains its 4096-total-token
arena, four active requests and existing decode graphs. These are not equal physical
memory grants. Reproduce with the preceding server/client commands and these two native
builds; do not substitute the older eight-row serving binary.

Before measurement, the internal advancement screen required at least 5% median paired
throughput gain at both c4 and c8; no paired shape could lose more than 5%. Per-shape
median paired TTFT/last-content p95 could not regress more than 5%, and no paired phase
could add failed requests or lose joint diagnostic-deadline passes. Shared successful
native outputs had to match exactly. **All these gates pass.** No unchanged model cohort
is retried, and all nine servers and 36 clients exit zero within the original one-hour
campaign bound. Model identities/groups and GPU contexts settle before each subsequent
launch; a final executable-identity scan also finds no native or shared client/reference
Python processes. The GPU returns to its initial empty/idle state, not a restarted service.

### Results and remaining deficit

Each cell is the median of three process results. Latency p95 covers successful requests;
diagnostic goodput uses joint TTFT ≤1,000 ms and last content ≤5,000 ms, with failures
never passing. Raw content clocks, actual usage, finite phase-window rates including
drain and joint counts are independently reconciled. Rate checks account only for the
exporter's epoch-float/microsecond timestamp serialization error, not a fitted tolerance.
Generic `performance_metrics_audited` and `planned_arrivals_audited` remain false.

| Configuration / client load | Successes / 48 | Output tokens/s | TTFT p95 ms | Last-content p95 ms | Diagnostic passes / 48 | Diagnostic goodput requests/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Eight c1 | 48 | 37.57 | 398.8 | 1029.5 | 48 | 1.223 |
| Sixteen c1 | 48 | 41.16 | 254.5 | 882.2 | 48 | 1.339 |
| SGLang c1 | 48 | 85.04 | 30.6 | 381.8 | 48 | 2.716 |
| Eight c4 | 48 | 37.61 | 1059.1 | 3826.9 | 43 | 1.096 |
| Sixteen c4 | 48 | 41.27 | 860.9 | 3435.1 | 46 | 1.287 |
| SGLang c4 | 48 | 280.83 | 52.9 | 465.8 | 48 | 8.998 |
| Eight c8 | 48 | 37.65 | 4705.9 | 7700.8 | 4 | 0.102 |
| Sixteen c8 | 48 | 41.28 | 4098.2 | 6754.6 | 4 | 0.112 |
| SGLang c8 | 48 | 290.64 | 487.9 | 907.6 | 48 | 9.288 |
| Eight nominal 4/s, c32 | 29 | 38.16 | 9340.6 | 12109.6 | 4 | 0.176 |
| Sixteen nominal 4/s, c32 | 31 | 41.38 | 8735.7 | 11078.2 | 4 | 0.177 |
| SGLang nominal 4/s, c32 | 48 | 121.18 | 44.0 | 441.7 | 48 | 3.870 |

Every c4 pair improves throughput by 9.23–10.13%; every c8 pair improves 9.20–10.06%.
Eight-row c4 throughput spans 37.47–37.75 tokens/s versus sixteen-row 41.24–41.32;
c8 spans 37.50–37.78 versus 41.26–41.32. Sixteen-row c1 TTFT p95 falls 36.3% by the
median paired comparison. Native closed-loop phases all produce exactly 1,475 tokens.
Rate-controlled phases produce 869 versus 935: admission outcomes change, not just timing.
Each rate-controlled eight-row phase has 19 HTTP 503 failures versus sixteen-row 17;
all attempts remain accounted for. The successful rate-load prompt sets are not nested. Across all loads and
repetitions, all **501 shared successful native prompt comparisons** have identical text,
usage and finish. This does not assert an output for a rejected request.

SGLang c1 produces 1,503 tokens and agrees with sixteen-row native on text, core usage
counts and finish for **34/48** prompts in each repetition. Batched SGLang outputs still
vary with scheduling; its c4 phases produce 1,477–1,502 tokens. No independent oracle here
classifies those disagreements. The SGLang/native rates therefore remain diagnostics,
not quality-equivalent speed ratios. Largest one-second device-used memory samples are
15,590 MiB for eight and 15,598 MiB for sixteen, versus 17,899 MiB for SGLang; these are
neither allocation charges nor physical peaks. Largest summed owned-process RSS samples
are 14.89 GiB for either native configuration and 6.74 GiB for SGLang; summing RSS can
double-count shared pages and is not unique/private resident memory. The separately
qualified charged-memory increment remains 7.0 MiB.

Retain the qualified sixteen-row selection. Its direct-call gains now have bounded
serving evidence, but higher logical concurrency still adds queue-visible latency without
material native throughput scaling. Native c8 still passes only 4/48 diagnostic deadlines.
Choose subsequent model-local cross-sequence physical-execution work from this deficit,
with numerical and retirement gates, rather than widening prefill rows by default.
End-to-end tight-grant serving recovery, representative arrivals/product SLOs, long-context
CPU parity, independent quality and an incumbent-native client cross-check remain open.

The protocol, both builds/patches, all 36 raw exports, installed-source/package identities,
one-second RSS/GPU samples, deadline observer and final absence proof are retained under
`gemma4-serving/20261026T045225Z`. A zero-context patch application and a misspelled
installed-distribution lookup failed during setup; their logs and corrected setup are
retained. Neither repeated a model cohort, changed dependencies or reset the deadline.

## Shared-grant allocation and recovery gate (2026-10-26)

The native CUDA executor and token Engine now pass a real allocation-pressure test on
RTX 4090/E4B, with the qualified sixteen-row preparation and full 4096-position caches.
This adds coverage without changing production behavior. The existing direct-decoder
partial-construction test did not protect healthy Engine peers or subsequent admission.

The model retains **14,946,887,968 charged bytes** from a 20 GiB `BytePool`. Cache size is
derived from BF16 K/V geometry for every KV-owning layer, excluding donor aliases:
**234,881,024 bytes (224 MiB) per sequence**. A competing owning lease leaves room for
two complete caches and one first-layer key, **4,194,304 bytes (4 MiB)**. That competing
lease consumes logical grant capacity; it does not allocate dummy physical VRAM.

Two-credit mailboxes hold each original request after its first generated token, without
sleeps or consuming those tokens. Under that stable pressure:

- A third request emits no token and terminates with request-local `Failed`, retaining
  prompt usage and zero completion usage. Its diagnostic identifies the exact refused
  4 MiB value reservation with zero bytes available. Partial-key retirement publishes
  capacity readiness and restores the exact pre-attempt charge; both peers remain active
  and the Engine is not faulted.
- Consuming one victim token permits a decode submission. Cancellation before executor
  polling retains both complete cache charges. Completing that accepted submission then
  retires the victim, suppresses its new output and delivers `Cancelled` with completion
  usage one. Only then does one full cache charge become reusable.
- A replacement finishes while the other original consumer remains stalled. That peer
  subsequently finishes, as does another request after dropping the competing lease.
  All three full continuations reproduce the four archived greedy token IDs, `Length`
  and exact 18-prompt/4-completion usage. Shutdown and model drop return the pool charge
  to zero.

The new pressure test and the existing text-facade abandonment/reuse test each pass in
one serialized release invocation, with native exit zero. The latter uses its existing
32-position cache; it is a separate facade check, not text-layer pressure qualification.
The build and all six required host gates also exit zero. Package sizes/hashes and the
archived reference are unchanged before/after. Both process groups are absent after
exit; GPU samples identify only the owned test process, and the final device is empty,
0% utilized and 33 MiB used. Source, binaries, protocol, raw logs and samples remain under
`gemma4-pressure/20261026T063624Z`.

Reproduce the pressure transition with the pinned package/reference and CUDA toolchain:

```sh
CARGO_BUILD_JOBS=2 RUSTUP_TOOLCHAIN=1.98.0 \
  RIBN_GEMMA_PACKAGE=/path/to/package RIBN_GEMMA_REFERENCE=/path/to/cpu-bf16-reference.json \
  cargo test -p engine-gemma4 --features cuda --release --locked --test cuda_pressure \
  -- --ignored --test-threads=1 --nocapture
```

This qualifies logical-grant exhaustion with real CUDA allocations, not driver OOM,
a physical-memory peak bound, endpoint pressure/SLO recovery or other hardware.
Cancellation precedes the synchronous executor's device work in `poll`; it does not
qualify cancellation of already-enqueued asynchronous CUDA work. Capacity refusal is
terminal request-local failure, not deferred admission or a capacity-wait/reactivation
gate. Dynamic growth, retraction and broad fairness remain separate gates. Static review
strengthened the failure assertion before the first device build/run; neither a model attempt nor the
original campaign deadline was reset.

## Four-sequence decode numerical gate (2026-10-26)

A private RTX 4090/E4B prototype batches the dense operations of four independent
continuations through existing BF16 primitives, while gathering queries and running
attention against each row's own cache and position. This is cross-sequence decode,
not a return to four-row prefill. It adds 3,689,472 charged bytes (3.52 MiB), with no
kernel, precision, logical-policy or graph change. **It fails the numerical gate and
is not promoted.** The public sixteen-row-prefill/serial-decode path remains unchanged.

The control executable uses public source `6d5035f`; the candidate uses a separate
patched build. They do not compare two paths through the same refactored implementation.
Both receive the same archived-token-derived histories, full 4096-position caches and
fixed teacher-forced continuation IDs. These synthetic arithmetic probes are not
EOS-enabled greedy-serving or task-quality evidence. A separate archived chat-greedy
check is planned in the same input deck.

All four initial prefill logits match bit-for-bit. On the first decode at processed
prefixes 1, 17, 64 and 513, every row has the expected 262,144 finite BF16 logits, but
all four fail full-logit bit equality:

| Processed prefix | Differing logits | Maximum absolute difference | Argmax unchanged |
| ---: | ---: | ---: | :---: |
| 1 | 188,187 | 0.375 | yes |
| 17 | 240,860 | 0.59375 | yes |
| 64 | 247,764 | 0.5625 | yes |
| 513 | 231,525 | 0.4375 | yes |

The differences are descriptive, not fitted tolerances. Four unchanged argmaxes do not
establish distributional equivalence or make the discrepancy harmless. The numerical
source is not isolated; changed vendor-GEMM reduction order is a hypothesis, not a
finding. A later variant needs common-history component comparisons and an independently
justified numerical contract before performance advancement. Existing reference error
supplies no extra error budget.

The candidate captures every row of that failed step, then stops. Its remaining steps
and later window, maximum-reach, permutation, chat-greedy and serial-fallback cohorts
are **unexecuted**.
The control alone completes all eight cohorts, including the archived four-token chat
check; that does not qualify the candidate. No serving comparison or four-row lifecycle
promotion gate runs. Invalid-token prevalidation does reject the last row without
changing any cursor or charge. Both probes close their caches and return the pool charge
to zero, but this is not failed-drain, cancellation or physical-OOM qualification.

The control exits zero; the candidate exits one for the explicit numerical rejection.
Both native process groups disappear normally, with no forced cleanup. Package/reference
hashes remain unchanged, sampled compute processes belong only to the active probe, and
the final GPU is empty, 0% utilized and 33 MiB used. The source patches, both binaries,
all 76 control and eight candidate raw logit vectors, input deck, protocol and failed
setup attempts remain under `gemma4-batch4/20261026T202039Z`. Two lock-resolution setup
failures preceded all model execution; native-only offline builds then preserved every
workspace dependency pin. No dependency installation, model retry, tolerance change or
campaign-clock reset occurred.

## First PLE-projection native capture (2026-10-28)

A bounded RTX 4090 component replay now compares the first PLE projection used by
that prototype. **Its consumed operand bits match the independent reference, but
its serial and four-row BF16 outputs are not invariant.** The four-sequence variant
remains private and unpromoted.

Separate fresh processes load the original `6d5035f` control and private candidate.
The control executes four scalar projections; the candidate executes one projection
through its prepared four-row decode buffers. All inputs are token 236776, the first
teacher-forced decode input of each rejected row. Both use the model's existing
scaled-embedding recipe, role binding, 8 MiB cuBLAS workspace and unchanged native
GEMM. BF16 operands and output, FP32 accumulation, disallowed reduced-precision
reductions and the default algorithm selector remain unchanged. This is a component
replay, **not a decoder forward**: it recreates neither prefix/cache histories nor
the failed full-model step, and does not fill any unexecuted prior cohort.

After each native call acknowledges completion, checked readback captures the actual
input, weight and output before PLE scaling or normalization. Only private host
readback staging grows; device matrix geometry and backend code do not change.
Every input and weight bit, including signed zero, matches the independent
[first-projection pack](gemma4-reference/README.md#first-ple-projection-reference).
All 43,008 output values are finite. Duplicate output rows match within each mode,
but the modes differ at **12 of 10,752 coordinates in every row**: 48 differences
altogether, with maximum absolute difference **0.0625**. For example, column 5466
has scalar bits `0x4120` and four-row bits `0x411f`.

The comparator preserves the real BF16 output dtype rather than changing the GEMM to
produce FP32 or relabelling a cast. Against the unrounded binary64 reference, maximum
absolute error is 0.086465673 in both modes; RMSE is 0.0051707224 scalar and
0.0051707268 four-row. An independent standard-library reduction agrees with these
whole-output metrics. They are descriptive, not a rounding oracle, error budget or
precision vote. The replay establishes non-invariance at this component with matched
operands; it does not identify cuBLAS's internal reduction order, prove which later
operations amplify differences, or explain the full-model discrepancy by itself.

Both native processes exit zero, explicitly retire their models, return logical pool
charges to zero and leave no owned process groups or compute processes. The device
is driver 615.71.10, CUDA toolkit 13.3.73; loaded cuBLAS/cuBLASLt 13.6.0.2 and NVRTC
13.3.33 paths/hashes are retained. Source, binary, package and protocol identities
are frozen before execution and unchanged afterward. Builds use locked offline
Rust 1.98 with two jobs and separate fresh targets. One initial private baseline
helper fails compilation before any model execution; its corrected build and the
candidate build pass, with no device retry or deadline extension. Comparator tests,
Python benchmark tests, Ruff, boundaries and formatting pass. Full production Rust
and device suites are not rerun for this Python/docs-only change; unchanged production
Rust retains the earlier qualification, not a new CUDA variant gate.

Raw BF16 captures, typed SafeTensors packs, comparisons, process monitors, binaries,
source patches and failed setup evidence remain under
`gemma4-cuda-first-projection/20261028T113835Z`. The fixed task window begins at
11:38:35 UTC, with device cutoff 12:33:35 and total deadline 12:38:35. No kernel,
precision policy, numerical tolerance, performance selection or production Rust changes.

## Actual decoder projection trace (2026-10-28)

The component result now has a bounded actual-path check: the original control and
private four-sequence decoder each execute one step after the frozen prefixes
1, 17, 64 and 513, using teacher-forced token 236776 in every row. Four serial calls
are compared with one physical four-sequence call. This collects new projection
traces; it does not retry the original acceptance gate or execute its later cohorts.

Collection is disabled during prefills. After each selected native linear call
acknowledges completion, checked readback retains its BF16 input, weight and output.
The selection is global PLE projection and layer-zero Q/K/V only: sixteen records
in the control and four in the candidate. Every paired projection has bit-identical
consumed inputs and weights, including signed zero. PLE operands also match the
independent reference pack. All captured values have valid shapes and are finite.

| Native projection | Output width | Differing coordinates per row | Maximum absolute output difference |
| --- | ---: | ---: | ---: |
| Global PLE | 10,752 | 12 | 0.0625 |
| Layer-zero Q | 2,048 | 5 | 0.25 |
| Layer-zero K | 512 | 0 | 0 |
| Layer-zero V | 512 | 0 | 0 |

Each count holds for all four rows. Thus the first-PLE discrepancy occurs on the
actual decoder path, but it is not the only early projection with non-invariant
native output. Q differs despite matching its own inputs and weights; PLE-only
parity would not establish projection-wide bit equality. K/V equality is limited to
these operands, not a guarantee for other layers or histories. No vendor reduction
order, downstream amplification mechanism or acceptable full-model error is inferred.

All eight new prefill vectors match the previously retained independent control
bit-for-bit. All eight final logit vectors also match their archived **same-mode**
witnesses bit-for-bit. The original control/candidate logit discrepancies remain
188,187 / 240,860 / 247,764 / 231,525 elements. These checks support observation of
the known failed step; they do not turn it into a numerical pass, qualify cache-word
equivalence, change precision policy or justify a performance experiment.

Both native collectors exit zero for completed data collection, explicitly retire
models/caches, return logical pool charges to zero and leave no owned groups or
compute processes. The prior candidate's numerical rejection remains unchanged.
Source, binary, package, input-deck and original-witness hashes are frozen before
execution and verified afterward. Both locked offline release builds pass in fresh
separate targets; native backend code, device geometry and arithmetic remain unchanged.
Only private host readback staging and observation change. Independent standard-library
analysis reproduces the output counts/maxima. Boundaries, formatting and diff checks
pass; unchanged public Python/Rust/device gates are reused rather than rerun. This
is neither performance, physical-memory-pressure nor lifetime-failure qualification.

Raw captures, typed PLE packs, source patches, binaries, monitors and comparisons
remain under `gemma4-cuda-decoder-trace/20261028T121633Z`. The task starts at
12:16:33 UTC, with device cutoff 13:11:33 and deadline 13:16:33. Exactly one native
process per mode runs, with no retry, arithmetic intervention or deadline extension.

## Reordered four-sequence decode advancement contract (2026-10-29)

**Accepted gate, not automatic-selection approval.** Reordered decode is the selected
investigation; the [execution contract](../docs/resource-protocol.md#reordered-cross-sequence-cuda-decode)
permits a new experiment without scalar-logit bit equality. The original four-row
rejection and its unexecuted cohorts remain unchanged. This gate supplies no fitted
logit tolerance, precision vote or automatic-selection permission.

### Scope and arithmetic

Use the E4B BF16 package and reference pins in [model.toml](gemma4-reference/model.toml),
RTX 4090, four independent full-history continuations and positions below 4096.
Keep qualified sixteen/eight-row prefill, scalar decode for other batch sizes, BF16
operands/results, FP32 accumulation, reduced-precision reductions disabled and the
8 MiB cuBLAS workspace. No eviction, graph, overlap, sampling or policy change is
part of this experiment. Freeze the actual toolkit/library/device, source, binary,
processor and input identities before execution.

The original prefill component gate is insufficient: its matrix rows are 1/8/16,
and its vocabulary head stays scalar. The decode gate covers rows **1/4** for all
12 existing E4B projection geometries **plus the 2560→262144 tied vocabulary head**.
Retain exact dyadic checks and the independently derived FP32/BF16 forward-error
bound from the [prefill gate](gemma4-reference/README.md#numerical-acceptance-fixed-before-measurement).
Use the same predeclared operand family, not inputs selected to make this candidate
pass. Check all outputs, shape and finiteness. The exact-integer fixture's safe range
is part of its proof; do not apply that proof blindly to arbitrary real operands.

On the first archived four-row decode (rotation zero, prediction step one), capture
actual consumed operands/results for PLE, layer-zero Q/K/V and the vocabulary head,
after checked completion. Evaluate each mode against its **own** operands, since later inputs can
legitimately differ. Any acceptance comparison needs exact encoded-operand dots or
certified enclosures and validated underflow/overflow assumptions; the existing
`math.fsum` diagnostic pack alone cannot grant acceptance. Shared inputs/weights must
match where the paths promise equality. Unchanged RMS, rotary, elementwise and
attention operations retain their exact shared-operand checks. These are operation
checks, not an accumulated full-logit error bound.

### Encoded-operand component certificate

`gemma4_linear_bound.py` supplies the certified enclosure for stored BF16 operands;
`gemma4_linear_compare.py` remains diagnostic. It decodes BF16 directly into binary64,
without a float32 intermediate that could flush subnormals on the host. For `K<=10240`,
BF16 products are exact normal binary64 values and all possible partial sums lie on
a grid no finer than `2^-266`, far above binary64 underflow. Explicit products and
[`sum(dtype=float64)`](https://numpy.org/doc/2.5/reference/generated/numpy.sum.html)
use binary64 accumulation, regardless of reduction order; no BLAS precision assumption
is needed. With `g=gamma_K(2^-52)` and computed positive-product sum `S_hat`, enclose
`S` by `S_hat/(1+g)` and `S_hat/(1-g)`, then enclose the dot by
`d_hat ± g*S_upper`. Every bound operation rounds outward with `nextafter`; the
rational gamma itself is enclosed before use.

For the FP32/BF16 bound, conservatively require normal-or-zero BF16 operands, a common
product quantum at least `2^-126`, and `S_upper<2^127`. BF16 products are then exact
FP32 values; every nonzero partial sum remains on a normal FP32 grid, and rounding
preserves that grid. The magnitude ceiling and `gamma_K(2^-23)<1/2` exclude accumulation
and BF16 conversion overflow. If every product is zero, exact `S=d=0` is checked
separately. These are sufficient proof conditions, not a claim that all other inputs
are invalid. Unproven domains fail closed as **unsupported**.

Acceptance requires the **largest** possible error over the dot interval to fit the
**lower** allowed-error bound. A definite violation is separate from an indeterminate
interval; neither passes. Host tests use independent exact rational BF16 dots,
including cancellation exactly on the theoretical bound where interval uncertainty
must not produce an optimistic pass. A real host-FP32-flush probe reproduces the
pre-fix conversion's false zero-product acceptance and verifies rejection after direct
binary64 decoding. No device arithmetic or inference tolerance changes.

```sh
python3 benchmarks/gemma4_linear_bound.py \
  --capture /path/to/trace/00-ple.json --output /path/to/new-bound-directory
```

The raw trace descriptor names BF16 input/weight/output shapes, index, role and declared
completion. The reader accepts only rows 1/4 and the 13 scoped geometries, checks byte
extents and stable hashes, and memory-maps weights while processing at most 64 output
columns at a time. Actual-model head capture and certification are recorded
[separately below](#actual-model-prerequisite-results-2026-10-29). Declared completion
is not execution attestation; exit zero grants only the stored component bound,
never full-model acceptance.

Host-only validation of the retained decoder traces certifies **110592 outputs**
across 16 serial and four batched PLE/layer-zero Q/K/V records, each against its own
operands. There are no violations, indeterminate intervals or unsupported domains in
that sample. The corrected source, fixed protocol, hashes and reports are retained at
`gemma4-encoded-bound/20261029T044119Z`; this reads old operands, not a new model run
or a reversal of the old bit-equality rejection.

A fresh release-only RTX 4090 component run passes the expanded exact-integer test:
**50 GEMMs, 2520832 outputs**, all 13 geometries, rows 1/4 including the tied head,
and retained rows 8/16 on the 12 non-head geometries. Explicit close succeeds and
logical pool charges return to zero after each geometry. The test uses a 2 GiB device
pool and 3.75 GiB peak host weight payload; no model is loaded. The binary exits zero,
its owned process group is absent, and compute processes are empty before and after.
CUDA 13.3, driver 615.71.10 and cuBLAS 13.6.0.2 are recorded with actual loaded-library
hashes. Source, binary, fixed deadline, controller and raw receipts are retained at
`gemma4-cuda-component-bound/20261029T045730Z` (binary SHA256
`5268b269798a12ce1503f456fb566e3d1f8483b5755537600d1fb67d70851285`).
This qualifies the synthetic component scope, not actual head operands, continuation
routing, model quality or four-row automatic selection.

### Native routing component gate (2026-10-29)

A separate no-model-weight device gate exercises the private prototype's actual
`CacheRouting::Independent`, not a host simulation. It uses the pinned E4B configuration,
local/global head widths 256/512, eight query heads, two KV heads and 42 cache slots.
Owner/tail pairs 22→24 and 23→29 use the model's real sharing boundaries; distinct
populated donor slots 0/5 expose wrong-donor reads. The fixed prefix groups and reversed
row order from the continuation deck below run against 4096-position native caches.

All **48 routing calls and 192 same-operand scalar replays pass bit-exactly** on RTX 4090.
Each initialized K/V prefix matches the declared history plus that sequence's own new
rows. The alternate donor stays unchanged and uninitialized tails stay unreadable.
Every attended/scattered row matches qualified scalar attention on its own query,
cache and absolute span; the final gathered query is also read back exactly. Poisoned
unused K/V inputs on the shared-tail call must not be appended. Readback chunks are
bounded to 1 MiB; the positive run returns its 256 MiB logical pool to zero.

Four separately built, single-change counterexamples are detected on device: copying
the wrong sequence's appended value, reading the alternate shared-tail donor, gathering
row zero for every query, and scattering every result to row zero. Each exits 101 at
the predeclared cache-value or scalar-replay/scatter assertion, not a compiler/driver
error. The original route exits zero. All five owned process groups disappear without
forced cleanup; no compute process remains. CUDA 13.3, NVRTC 13.3.33 and driver 615.71.10
libraries are frozen and checked. Sources, configuration, mutations, targets, binaries,
fixed deadline and raw positive/negative receipts are retained under
`gemma4-cuda-routing/20261029T051810Z`; positive binary SHA256 is
`e048c1c53ae5cfdebe1c6c3578092779e0a6fbdd3c5b783fe7e185b58f9382ef`.

A separately frozen retained-row run adds explicit shape/finiteness evidence rather
than relying on byte equality alone. It preserves all **192 actual/scalar row pairs**;
an independent reader verifies complete ordered headers, exact shapes, bit equality
and finiteness of all **589824 BF16 elements per side**. The 2363912-byte capture
and source/binary/host-verifier receipts are retained at
`gemma4-cuda-routing-capture/20261029T055507Z` (capture SHA256
`24af4be8bca492b46803e75c38be8c6d003246cc33611b95c22ea75add4ac454`).
The raw-output bound is 3 MiB. This changes only the private fixture's observation and
validation, not the routing algorithm or prior experiment's recorded scope.

This qualifies these native routing fixtures and their failure detection, **not**
actual-model continuation state, invalid-row atomicity, executor/text selection,
request retirement, numerical quality or serving performance. It does not promote the
private prototype or replace the full-model collector and gates below.

### Full-model and continuation gate

Use a fresh public serial control, not two paths through a refactored candidate.
Require exact common input IDs and positions, complete finite 262144-element logits,
and all **12 independent archived predictions** from the chat/window CPU BF16
reference. Use all four cyclic row rotations of
`[chat, window_511, window_513, chat]`, each from fresh state through all four
prediction steps. Feed archived inputs, not a backend's chosen continuation, to keep
histories identical. The first prompt
runs through unchanged prefill; subsequent steps must execute physical four-row decode.

Add synthetic common-history groups starting at processed prefixes `[1,17,64,513]`,
`[511,512,513,514]` and `[4092,4093,4094,4095]`, with respectively 3/2/1 decode steps.
For length `p`, form the prompt from the archived chat's original first ID followed
by `p-1` IDs cycling its remaining prompt. For sequence index `i` and step `s`, force
`[236776,126641,568][(i+s) % 3]`; sequence identity and input do not change under
permutation. Repeat each group with reversed row order. These synthetic rows have
no independently labeled expected winner; do not invent one from the serial output.
Keep complete paired logits and report differences, centered RMSE, relative L2, KL
and TV descriptively. Neither greedy agreement nor a smaller distance grants a
numerical budget.

Check every KV owner's initialized prefix and committed position. Old cache rows
must remain unchanged; appended rows must equal that sequence's own produced K/V
encodings, and shared-tail reads must use its own full donor. Gather/scatter and
per-row attention must match the qualified scalar operation on the **same** captured
query, cache and span. Serial/candidate caches need not match after changed projections.
All unchanged prefills and one/two/three-row fallbacks must remain bit-exact to the
independent serial control. Invalid last-row input must leave every row's position,
cache and charge unchanged. Missing state/routing checks cannot be replaced by logits
or by logical pool charges returning to zero.

### Actual-model prerequisite results (2026-10-29)

The separately frozen direct-model campaign passes its component, continuation,
atomicity and noninterference prerequisites. It does **not** establish the separate
generated-QA or Engine/text gates; their later results are recorded below. The original
full-logit bit-equality rejection is unchanged.

Four fresh controls use public serial source, public serial plus observations, the
original unobserved candidate patch, and that candidate plus the same observations.
Each executes the complete 13-cohort, 160-row deck above, returning finite full
262144-element BF16 logits. Each passes all 12 independent archived CPU prediction
conditions, with 88 repeated checks. Histories are reconstructed from archived
`input_ids` and checked against `history_ids`: the windows force **562**, not their
prefill winners 6178/815. The first capture has IDs `[11,12,13,14]`, positions
`[18,511,513,18]` and inputs `[236776,562,562,236776]`. No reference labels changed.

Independent validation establishes:

- All 160 public plain/observed rows and all 160 original candidate plain/observed
  rows are bit-exact. Candidate prefill (46 rows) and width-one/two/three fallback
  (18 rows) are bit-exact to public serial. The 96 reordered decode rows have complete
  finite logits; their distances remain descriptive, not an acceptance budget.
- The actual captured PLE, layer-zero Q/K/V and tied-head operands, weights and
  outputs pass the unchanged encoded-operand certificate: 20 serial and five
  four-row GEMMs, **2207744 outputs**, including **2097152 head outputs**. Each mode
  is certified against its own operands. Actual learned weights and scales are
  streamed bytewise against the pinned checkpoint; no full-head host vector is added.
- Separate no-model processes replay every unchanged native RMS, rotary,
  elementwise and scalar-attention operation on its own captured operands and span:
  **2988 scalar rows per mode**, all bit-exact. Separate CPU processes check the
  unchanged public embedding, scaling, PLE combination and softcap: **184 checks per
  mode**, on desktop Linux/libm. Replay outputs are retained, native row scratch
  returns to baseline, final pool grants are zero and cleanup errors are empty.
- The captured graph binds actual attention outputs to projection-consumed scatter
  rows, checks 336 same-logical-position rotary factor pairs and 192 initialized
  prefill K/V prefix files across modes. Source-built ledgers check each observed
  owner's old prefix and own produced append, including shared donors.
- Eight actual before/after snapshots prove invalid-last-row atomicity for all four
  continuations: initialized cache bytes, positions, charges, geometry, allocation
  consistency tags and healthy model state. The next valid decode succeeds. Tags
  are not treated as allocation-lifetime proofs. Internal capped rows also match
  the actual caller-returned rows bytewise.

The RTX 4090 run pins Rust 1.98.0, driver 615.71.10, CUDA 13.3, actual loaded
NVRTC 13.3.33 and cuBLAS 13.6.0.2. Actual pre/post process library mappings match
pre-frozen real-path hashes, not merely `ldd` discovery. All four release builds,
four release all-target clippy checks and twelve execution/analysis stages exit zero;
whole owned process groups and their independent watchdog groups are absent, with
no forced cleanup. Model/replay processes are serialized. These are qualification
runs, not serving-performance measurements.

Sources, isolated targets, binaries, raw captures, validation reports, replay outputs
and stage receipts are retained under `gemma4-model-campaign/20261029T103000Z`.
Its source manifest SHA256 is
`ab119df33a68fb33f2d346b385edab4964ae39369340758bfe377d9d1a5164a5`.
`prerequisites.json` closes that scope at 11:15:13 UTC, before its frozen 12:20 UTC
cutoff. The earlier controller failures and the rejected build packet
`gemma4-model-campaign/20261029T102323Z` remain intact; that packet launched no model.
No campaign is extended or missing sample filled. Further executor/QA work requires
new frozen sources, targets, deck, budget and absolute cutoff.

### Independent generated-QA screen

The task gate is **no per-question EM or token-F1 loss** relative to a fresh native
serial control, on every frozen question. Score candidate and control independently
against human aliases; gains on another question cannot compensate for a loss.
Literal generation equality is not required. This strict finite-corpus rule needs
no fitted non-inferiority margin, but proves no population non-inferiority, statistical
equivalence, general instruction-following or distributional fidelity.

Select 128 new SQuAD v1.1 development questions without consulting labels, lengths or
model outputs. Use the [publisher dataset](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
at publisher revision `e0c66cfec263165fb3c15cb76a22175e371b71d7`, SHA256
`95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9`.
Exclude the original 32-question [Metal diagnostic](gemma4-metal-baseline.md#generated-extractive-qa-diagnostic)
by reproducing its selection, not by inspecting its scores. On the remaining
questions, take the 128 smallest SHA256 values of
`ribn-gemma4-cuda-reordered-decode-qa-v1 + NUL + question_id`, breaking ties by ID.
Retain that order and all original answer aliases. No truncation, resampling or
replacement is permitted if a selected item cannot fit or fails.

Construct each user message exactly as:

```text
Answer the question using only the exact answer span from the context. Do not explain.

Context:
{original context}

Question: {original question}
```

The message has no trailing newline. Render the pinned publisher template with
`add_generation_prompt=true` and `enable_thinking=false`; encode without a second
BOS or template. Match the actual Rust processor's rendered tokens before model
work. Consecutive groups of four use fresh continuation state and unchanged 16-token
prefill chunks. The direct executor harness prepares all four prompts before decode;
this controls the arithmetic experiment, not production scheduling. Use greedy
generation, package EOS stops and a maximum of 64 new tokens. Decode through the
real candidate executor and canonical processor, not a mock or a quality-only kernel.
Within each group, remove a finished request without replacement and retain scalar
fallback below four rows. Record actual physical calls per request; each question
must reach at least one four-row decode (possibly producing its stop), or the screen
is inconclusive. Do not force an ended request to continue to preserve batch width.

Separately exercise physical four-row selection through the actual Engine and text
facade on the archived chat group, checking all four archived token IDs, delivery,
usage and completion. Default scheduling may select fallback: record that as missing
integration evidence, not a pass inferred from four submitted requests. This gate
precedes performance advancement; HTTP and mixed-arrival qualification follow it.

Retain exact token IDs, complete text, actual usage and terminal outcome. Validate
full-row finiteness before selecting each token. Score **whole returned text**, with
no answer extraction or explanation stripping, by the pinned
[Python evaluator mirror](https://github.com/allenai/bi-att-flow/blob/49004549e9a88b78c359b31481afa7792dbb3f4a/squad/evaluate-v1.1.py),
SHA256 `f5a673dbbd173e29e9ea38f1b2091d883583b77b3a4c17144b223fb0f2f9bd09`.
Use its SQuAD normalization and maximum over aliases; independently check token F1
with exact rational overlap arithmetic. A `length` outcome stays scored whole and
must report 64 generated tokens. A timeout, error, missing/duplicate ID, malformed
usage, unretired owner or incomplete process is a failed/incomplete run, not a dropped
question or a zero score hidden in the aggregate. Free-running histories may diverge;
never aggregate numerical differences after divergence. Task scoring still includes
both complete answers.

The model-free freeze verifies all answer offsets and both tokenizer entry points:
128 questions, **26421 prompt tokens**, range **75–581**, all within 4096 positions
including the output cap. The ordered `[question_id, prompt_ids]` list has SHA256
`d7048e2e7172913b0246031f59544b0920e5213eecabd73ba7f33a9a2f186ad5`, using UTF-8
`json.dumps(pairs, ensure_ascii=False, separators=(',', ':'))` with no trailing newline.
Selection, complete cases, tokenizer/package metadata hashes and the freeze source
are retained under `gemma4-cuda-reordered-contract/20261029T034758Z`.
That model-free freeze loaded no weights and collected no predictions or quality scores.

### Scoring retained results

`gemma4_decode_quality.py` implements the fixed-corpus scoring boundary without Torch,
model loading or code execution. Given the frozen cases and retained run records:

```sh
python3 benchmarks/gemma4_decode_quality.py \
  --dataset /path/to/dev-v1.1.json --cases /path/to/cases.json \
  --baseline /path/to/serial-results.json --candidate /path/to/four-row-results.json \
  --output /path/to/new-score-directory
```

Each run is a JSON object with `schema_version: 1`, `complete: true` and a `results`
list. Each result contains `id`, `prompt_ids`, `token_ids`, `text`, `finish_reason`,
`usage` and `prediction_widths`. Generated IDs include the final EOS when present;
usage counts every retained generated ID. Prediction widths align with those IDs:
zero for the initial prefill prediction, then one or four for each decode prediction.
The serial control uses only one; candidate widths must follow the fixed groups'
actual active counts after EOS. Results may arrive out of order but not lose,
duplicate or add IDs. The collector must establish canonical text/ID correspondence,
EOS semantics, actual physical calls, numerical checks and successful retirement.
The scorer cannot attest those facts from JSON.

The scorer verifies the source dataset hash, independently selected IDs, every human
alias, frozen prompt digest, terminal/usage consistency and recorded row exposure.
F1 comparisons use exact fractions. Inputs are limited to 8 MiB each; duplicate JSON
keys and nonfinite numbers reject. Outputs use a new directory; errors retain an
incomplete report rather than replacing old evidence. A completed no-loss sample
returns zero; an observed per-question loss returns one with a complete failure
verdict. Neither exit zero nor a `complete` record grants numerical, lifecycle,
execution-provenance or population-quality acceptance. Host tests cover compensating
aggregate scores, F1-only losses, incomplete/invalid records, EOS shrinking, capped
answers, corpus mutation and non-overwrite behavior.

### Executor QA and default text selection results (2026-10-29)

**The frozen QA screen passes; the first, sequential-caller Engine/text scope selects
no physical-four calls.** The later concurrent-caller scope below supplies that missing
selection evidence without changing scheduling. These private sources reuse the original
candidate patch's existing executor dispatcher, rather than adding another execution
loop. Both controls validate every complete 262144-element BF16 logit row for shape
and finiteness before argmax, including intermediate prefill returns. Bounded native
receipts record actual prepared hidden/head matrix rows, returned rows and completed
calls, correlated with accepted continuations, positions and input tokens—not a
requested batch width. The CUDA-feature library tests and example clippy checks pass
for both controls, as do both release builds and release example clippy checks.

The first campaign exercises four archived chat requests through the real
Engine/Driver/TextOwner/TextStream with **no prefill barrier or scheduling change**.
All four deliver `[236776,126641,568,15721]`, canonical text `A mutex (short`,
`Length` and 18/4/22 prompt/completion/total usage. Their admissions preserve the
canonical 18-token prompts and separate request/continuation identities. However,
all twelve decode predictions use scalar fallback: **zero actual native M4 calls**.
The trace contains staggered prefill/decode; four submitted requests do not establish
four physically batched decode rows. The integration assertion exits one, with all
four continuations released, no cleanup errors and final device grants zero. The
campaign stops there, leaving QA unrun in that scope. This is missing integration
evidence, not a numerical rejection of an executed M4 call.

A separately frozen **QA-only** campaign then executes the previously unrun arithmetic
screen, without retrying the failed facade case. The prescribed QA prefill barrier
prepares all four prompts before decode; it does not qualify production scheduling.
The actual Rust processor matches every frozen rendered prompt and both encoding
entry points before weights load. All 32 fresh groups finish without refill, using
unchanged 16-token prefill chunks, package EOS and greedy cap 64:

| Retained result | Public serial | Candidate |
| --- | ---: | ---: |
| Questions / generated IDs including EOS | 128 / 1227 | 128 / 1227 |
| Stop / length outcomes | 127 / 1 | 127 / 1 |
| Complete vocabulary rows checked before selection | 2808 | 2808 |
| Actual native M4 calls | 0 | 105 |
| Exact-match answers | 67 / 128 | 67 / 128 |
| Token F1, descriptive percent | 77.882668834 | 77.882668834 |

Every question receives actual physical-four exposure: 105 calls produce 420 M4
predictions; 679 later decode predictions use scalar fallback as groups shrink.
All 128 whole returned texts and generated-ID sequences match the fresh serial
control. The strict scorer finds **zero per-question EM or exact-rational F1 losses**.
It scores the capped answer whole, without explanation stripping. Independent checks
cover all **866 retained answer/alias pairs** against the pinned official evaluator;
the predeclared binary64 evaluator-roundoff check is not an inference tolerance.
This is a passed finite-corpus screen, not population-quality or full-logit equivalence.

Independent artifact validation binds admissions, actual native matrices/inputs,
completed ranges, raw predictions, EOS suppression in actual Engine delivery, usage,
releases and final zero grants. Each QA process's actual pre/post library maps verify
13 loaded DSOs against the new real-path freeze. Source, package and binary inputs
remain unchanged; all owned worker groups and independent watchdog groups are absent,
with no forced cleanup and no remaining compute PID. The earlier host-check packet
`gemma4-executor-check/20261029T114307Z` retains its Clippy and watchdog failures and
launched no model. The stricter ownership reader and independently supervised
read-only queries pass fresh Linux boundary, churn, orphan and deadline probes before
these device campaigns. No specific cause is asserted for the old unverified-live
`/proc` observation.

All sources, targets, binaries, partial outputs, raw traces and receipts remain intact:

- `gemma4-executor-campaign/20261029T121800Z`: failed default text selection;
  closed 12:24:34 UTC before its frozen 13:50 UTC cutoff. Source manifest SHA256:
  `856562f80a9aaacb1318be50bd7fe6ce17b1755b3351dc68f935e94d351e107b`.
- `gemma4-executor-qa-campaign/20261029T122500Z`: passed QA-only scope;
  closed 12:39:32 UTC before its frozen 13:45 UTC cutoff. Source manifest SHA256:
  `d77a7f9f8d5165a7b4b21203c4f3c7d5672fda317a84ffa387e86535d0fdaf66`.

The two campaigns produce identical respective binary hashes: serial
`6101fc4c89f81635c00a57c377ccc96268bff0d2938b2c72c8574f30b9f6cc3c`, candidate
`403a81a7b5c66e0a6b4affc096859664b6e207fccbd142d2ea1aa633136a5c94`.
Fresh source copies and isolated targets are used in each scope. Fixed process-group
budgets are 900 seconds per QA run and 300 seconds for the facade; neither campaign
is extended, retried unchanged or filled with replacement samples. These are
qualification runs, not serving-performance measurements.

### Concurrent-caller default text selection (2026-10-29)

**The changed four-concurrent-caller workload passes actual Engine/text M4 selection.**
The earlier sequential-caller failure remains closed and unchanged. Source inspection
shows that `stream_request_blocking` waits for preprocessing and runtime-enqueue
acknowledgement before returning. The driver can step between those sequential starts;
`TextBatch::fill` also starts streams sequentially. An admission window of four does
not promise four simultaneous decode rows. This does not establish a missing dispatcher
or justify holding early decode behind a production prefill barrier.

A fresh private scope changes only caller concurrency and consumption: four bounded
scoped threads clone the same `TextModel`, each starting and draining its own stream.
They share one real Engine/Driver/TextOwner, with unchanged default Engine scheduling,
one preprocessing worker and batch window four. There is no start barrier, prefill
barrier, sleep, output-cap increase or scheduling-policy change. Each request retains
the same archived chat, canonical 18-token prompt, greedy cap four and independent CPU
labels. The only source differences from the host-qualified candidate snapshot are the
caller harness and observation fields binding rows to actual submission IDs, native
receipt ordinals and admission frontiers. Model arithmetic, dispatch, dependencies and
lockfile are unchanged; the passed 128-question QA evidence is reused, not rerun.

The six actual submissions include staggered prefill, mixed scalar-decode/prefill,
two homogeneous four-owner decode submissions, and a final two-owner scalar fallback.
The two native calls report actual hidden/head/returned row counts four and completion
acknowledgement. Their own-history inputs and absolute positions are:

| Native call | Positions in submitted order | Input token IDs |
| --- | --- | --- |
| 1 | `[19,19,18,18]` | `[126641,126641,236776,236776]` |
| 2 | `[20,20,19,19]` | `[568,568,126641,126641]` |

Every request receives **two actual M4 predictions**. All four deliver exactly
`[236776,126641,568,15721]`, canonical text `A mutex (short`, `Length` and 18/4/22
usage. All 20 complete vocabulary rows pass shape/finiteness checks before selection:
eight prefill returns, eight M4 decode returns and four scalar decode returns.
Independent artifact validation binds each actual submission, owner, positive completed
range, native input/position, raw prediction and text delivery; all four continuations
are released and final device grants are zero. Actual pre/post mappings verify all
13 loaded DSOs against the fresh real-path freeze. Package, source and binary hashes
remain stable.

The candidate release library tests pass **15 tests, with three corpus-dependent vision
tests ignored**; CUDA/private-feature release example clippy and build pass. The device
stage and independent analysis exit zero. All owned worker and watchdog groups are
absent without forced cleanup; independently supervised read-only probes also pass,
and the GPU has no compute PID afterward. No main binary or old artifact is changed.

Sources, isolated target, binary, raw outputs and receipts remain under
`gemma4-concurrent-facade-campaign/20261029T131400Z`. The scope closes at 13:17:14 UTC,
before its frozen 14:10 UTC cutoff, with 600-second host-stage and 300-second
library/model/analysis process-group budgets. Source manifest SHA256:
`f9fbf25eceae3ad9dcf3b730e10cd3a99ffa2f7b333409440e5ea80bd8ef865a`.
Candidate example binary SHA256:
`8309cca7a0adb88f4766b51b6394da8d47398faa39d0b6fb61a4b77d5dbd0cc9`.
Independent verification SHA256:
`df6516267ab55869156c0e924bdae118a5455e5bfd8f116050ad26c87eb1a3f7`.
This is one declared, observed concurrent-caller qualification workload, not a claim
of universal batching, uninstrumented timing or serving performance. No failed scope
is retried or relabeled.

### Private reordered-decode performance interruption (2026-10-29)

**The first three-pair campaign stops on a host OOM during pair two. Performance
advancement remains unqualified; no missing repetition is replaced or inferred.**
Both sources reuse the qualified model arithmetic and observations: public serial
versus the original private M4 candidate, with complete vocabulary-row validation
before argmax. The new private harness uses the real text facade, one execution owner
at a time, default Engine/Driver scheduling, one preprocessing worker and window four.
There is no execution or prefill barrier. Sources, complete deck, isolated targets,
budgets and a 17:00 UTC cutoff are frozen before execution. The planned process order
is serial/candidate, candidate/serial, serial/candidate, with a fresh loaded model
per process and fresh sequential owners for these profiles:

- Four archived chat requests warm the process; their timings are excluded.
- All 128 frozen QA inputs run at c1, c4 and eight closed-loop clients, unchanged
  greedy cap 64 and package EOS. No question is selected using known output.
- Eight-client initial delays are `[0,10,30,60,0,20,80,140]` ms; per-client think
  times are `[0,5,15,30,0,10,25,50]` ms. Requests are assigned by index modulo client
  count. Response feedback changes actual arrivals; this is not an identical open-loop trace.
- Four accepted synthetic prompts repeat the archived chat IDs to lengths 4092–4095,
  with caps 4/3/2/1 and initial delays 0/20/40/60 ms. They have no semantic labels.
  This exercises near-limit residency, not physical OOM or eviction recovery.

Throughput counts actual completion usage **including suppressed EOS**, divided by
client-thread launch through all stream drains and owner shutdown. It includes planned
think times, but excludes model/processor construction and JSON export. First/last
content clocks measure actual nonempty text from request start; empty content retains
null clocks and fails the joint 1000/5000 ms diagnostic deadlines. These are not product
SLOs. Percentiles use nearest rank `ceil(0.95*n)` over nonempty content, with empty
counts retained. The predeclared screen requires median paired throughput gains of
at least 5% at c4 and mixed8, c1 throughput regression at most 5%, and at most 5%
median paired p95 first/last-content regression with no paired deadline-pass loss
across the QA profiles. Actual M4 selection must occur in each c4/mixed8 repetition.

The **only complete pair**, pair zero, matches all 392 whole generated-ID sequences,
texts, usage and finish reasons. Each QA profile produces 1227 IDs including EOS and
checks 2808 complete vocabulary rows. Its instrumented results are descriptive only:

| QA profile | Output tokens/s, serial → candidate | First-content p95, seconds | Last-content p95, seconds | Joint passes, both / 128 | Candidate M4 calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| c1 | 17.728 → 17.737 | 0.573 → 0.573 | 0.941 → 0.941 | 127 | 0 |
| c4 | 17.681 → 17.761 | 2.267 → 2.352 | 3.526 → 3.489 | 19 | 7 |
| mixed8 | 17.693 → 17.942 | 4.670 → 4.673 | 6.444 → 6.446 | 2 | 23 |

The candidate executes 28/1099 c4 decode predictions and 92/1099 mixed8 decode
predictions through actual M4 calls—2.55% and 8.37%. The rest use genuine scalar
fallback. This pair's c4/mixed8 throughput gains are only 0.46%/1.41%, not a repeated
benefit or a passed 5% screen. The near-limit profile has four resident owners, six
raw generated IDs, two empty-content outcomes per mode, and no M4 calls. Observed
peak logical grants are 14.795/14.799 GiB for serial/candidate; both return to their
model baselines before final zero grants. These observations do not measure physical
VRAM peaks, infeasible admission or driver OOM recovery.

Pair one's candidate process also completes and passes an independent model audit,
but remains **unpaired**. Its c1/c4/mixed8 rates are 17.656/17.716/17.475 tokens/s;
c4/mixed8 execute 9/23 M4 calls. Its serial partner is killed after warmup and before
any c1 result is saved: the kernel records a **global host OOM killing actual model
PID 637234**, in owned group 637233,
at 15:06:24 UTC. Concurrent Python and Rust compilation processes appear in that
kernel snapshot; the evidence does not isolate their individual contributions to
memory pressure. This is not a numerical rejection or watchdog/RSS-limit cleanup.
The stage exits `-9` after 86.25 seconds, below its 900-second deadline, with observed
owned RSS 16.10 GB. Its output directory retains only processor, prepared-deck and
completed-warmup reports; pre-model maps are also retained. No c1 result, later
profile, post-model map or final grant report is fabricated.
Its group is absent and the GPU has no compute PID, but normal model retirement and
final zero grants were **not observed for the killed process**.

Pair-one comparison, both pair-two processes and the final three-pair summary remain
unrun. The scope closes at 15:09:44 UTC before its cutoff. All owned worker/watchdog
groups are absent without controller-forced cleanup; 186 independently supervised
query checks pass, and the GPU is idle at 33 MiB. The three completed model processes verify all
13 actual pre/post mapped DSOs and final zero grants. Source, both binaries and all
nine package pins remain stable. Host library tests pass 14/15 cases with three
vision tests ignored per mode; CUDA-feature release example clippy and both fresh
release builds pass. The earlier host packet's single Clippy failure and the first
performance packet's prelaunch environment-key error are preserved separately;
neither ran a model.

All raw failures, partial outputs, sources, binaries and targets remain intact under
`gemma4-decode-perf-check/{20261029T142500Z,20261029T142700Z}` and
`gemma4-decode-perf-campaign/{20261029T145000Z,20261029T145200Z}`. The latter campaign's
source manifest SHA256 is
`7f7b95743ce923d8c9e43bed044ed93c47c18352fb20ab799e6951a7ed2298c1`;
serial/candidate binary hashes are
`b2a75373cbb7ce9f6059e814c3a3ef4ddec2d1cf136669b7397c161f95970de3` /
`a35fa2caf07ab5179a1c23b89740c952ccaa9cfa0ee87e6483580b103c913cc1`.
The complete-pair audit SHA256 is
`84f61876c301373cb143d8cc46723ea79cb6ae7d761d3f9666ec86b4fb0db9cf`;
retained failure-investigation SHA256 is
`9a35656659c3655f1d052c294c18fbf68fbe9f3c9a97ad26fb772a94d0e6b216`.
No three-pair medians or variability estimate exist. A later performance experiment
needs a changed, evidenced host-capacity/isolation precondition and a new freeze,
not an unchanged retry. Instrumented facade timings do not qualify HTTP serving,
uninstrumented production performance, incumbent parity or automatic selection.

### CUDA host-lookup capacity qualification (2026-10-29)

**The CUDA decoder now retains only its two host lookup tables and layer scalars,
not a complete CPU weight inventory.** The model constructor materializes these from
its validated `Gemma4Source`; the ordinary text/CLI loader no longer loads
`Gemma4CpuWeights`. Device weights, kernels, arithmetic, prefill selection and runtime
policies are unchanged. This is a host-capacity prerequisite, not another M4 performance
campaign or a claim that unrelated host processes cannot cause an OOM.

For the pinned E4B package, independently summed SafeTensors extents agree with the
model's retained-payload report:

| Host BF16 payload | Bytes | GiB |
| --- | ---: | ---: |
| Former complete active CPU inventory | 14,926,026,836 | 13.90095 |
| Token table, PLE table and 42 scalars | 6,979,321,940 | 6.50000 |
| Removed projection/norm payload | 7,946,704,896 | 7.40095 |

That removes **53.24% of retained host weight payload**, not 53.24% of process RSS.
The token/PLE shapes remain `[262144,2560]` and `[262144,10752]`. Metadata, source
readers, staging, allocator overhead and device storage are outside this payload
count. `GemmaLoadReport.model_bytes` still reports the complete logical active payload;
`Gemma4CudaModel::host_lookup_bytes()` reports the retained host subset.

The separately frozen capacity scope passes these checks on the same RTX 4090 and
package, without keeping an external full CPU inventory alive:

- All **160 complete, finite 262144-element BF16 rows** across the existing 13-cohort
  [common-history deck](#actual-model-prerequisite-results-2026-10-29) are bit-identical
  to the archived public scalar control: 41,943,040 elements, no fitted tolerance.
  This includes window boundaries, 4092–4095-token histories, reordered continuation
  owners and widths one through four; it does not select private M4 execution.
- All seven existing ignored CUDA tests run and pass: three decoder cases cover
  scalar/eight/sixteen-row preparation and partial-cache retirement; two executor cases
  cover real Engine delivery and failed-drain ownership; text and allocation-pressure
  cases cover facade delivery, healthy peers, cancellation charges and reuse.
- The fresh `ribn run-gemma` binary exercises ordinary package loading and returns
  `A mutex (short` for the archived four-token chat. Logical device materialization
  remains 14,946,887,968 bytes in the direct-model control; all 13 cohorts return to
  baseline and final device grants are zero.
- Observed whole-owned-group RSS peaks range from 7.71 to 7.84 GB. The largest sampled
  child process high-water RSS is **8,038,342,656 bytes (7.48629 GiB)**. Both measurements
  stay below the predeclared absolute 10 GiB screen. These are sampled process
  observations, not a matched old/new RSS comparison, physical VRAM peaks or a new
  aggregate host-memory reservation guarantee.
- All 13 actual forward-time pre/post mapped DSOs match the archived control.
  Initialization/last-live child maps match their pre-frozen installed-library hashes;
  last-live samples are not post-exit maps. Source, actual binaries and all nine
  package pins remain stable through closure.

Host format, workspace tests, default/CUDA all-target clippy and CUDA-feature release
library tests pass; the latter run 15 tests with three corpus-dependent vision cases
ignored. The new host-only test checks every lookup/scalar bit and the exact 250-byte
fixture payload against a complete CPU inventory, including sharded/unsharded retained
readers after package paths are replaced. Existing executor tests share that fixture
module rather than loading it twice. Metal-feature clippy and Metal device qualification
are not rerun for this CUDA-only change.

The failed host packets remain under `gemma4-host-lookups-check/20261029T153500Z`
(prelaunch formatter-command contract error; no Cargo/model execution) and
`20261029T153800Z` (duplicate fixture-module clippy error; later stage unrun).
The corrected source passes all five host stages in `20261029T154000Z`.
The first device scope, `gemma4-host-lookups-device/20261029T160300Z`, stops when its
new observer rejects an initial mapped-library identity. It retains zero logit rows;
the observer failed before saving that map, so the specific path/hash cause cannot
be recovered. Its verified owned group is forcibly retired, the GPU is empty, and
all later stages remain unrun. No final model retirement/grant result is inferred.
A changed observer persists identities before checking and the new inventory also
pins installed system loader modules; this repairs the evidence gap without claiming
to identify the earlier mismatch.

The new frozen source/target scope `gemma4-host-lookups-device/20261029T161100Z`
passes all ten stages, closing at **16:21:15 UTC before the unchanged 18:00 cutoff**.
Build budgets are 600 seconds, library/audit budgets 300 and each serialized device
stage 900, all for whole owned groups with independent watchdogs. All worker/watchdog
groups are absent without forced cleanup in this successful scope; 196 independently
supervised query checks pass, and the GPU is idle at 33 MiB with no compute PID.
Every failed/partial scope, source, target, binary and raw result remains intact;
no missing sample is filled and neither main binary is changed. Source manifest SHA256:
`b41c16547cd1769a5bfb810b10af7044175353d7214ccba57d72591a97f9f0d0`.
Direct-model/ordinary CLI binary SHA256:
`f05399f45daa0c6a6718cdff9a9e0801d12ca98daa00388d8ff345243292f678` /
`ea5b95246c5ed142e732f7ccc3be006494868807589ee1d25be0e41b4d805704`.
Independent verification SHA256:
`ba7edad023afeccb9bbe36bc6fdfac476326a6f3cdba1388d5aa72266e46b237`.
The interrupted performance cohort remains closed and unqualified.

### Lower-host-memory reordered-decode performance result (2026-10-30)

**The complete three-pair instrumented screen is negative.** Median paired throughput
improves only **0.98% at c4 and 1.57% at mixed8**, below the unchanged 5% requirement.
Paired diagnostic deadline-pass losses also fail the screen. Keep the original M4
candidate private; do not retry it unchanged, increase exposure artificially or relax
thresholds. This is a completed negative result, unlike the earlier host-OOM interruption.

The candidate is mechanically rebased onto public `4f04a41`'s host-lookup loader.
Constructor and host-field access change; kernels, BF16/FP32 policy, device inventory,
prefill, dispatch and routing do not. Fresh serial/candidate library suites pass
15/16 tests, with three corpus-dependent vision tests ignored per mode; CUDA release
example clippy and both builds pass. Before performance collection:

- All **320 complete vocabulary rows** match their respective archived old-mode
  controls bit for bit: 160 serial and 160 candidate, all 83,886,080 elements finite.
- All 64 prefill/scalar-fallback rows remain cross-mode identical. The candidate
  executes the expected 24 actual physical-four calls on the complete common-history
  deck, with exact owner inputs, positions, native shapes and completion receipts.
- All 13 cohorts per mode return to baseline, final grants are zero and all 13 actual
  pre/post forward-time DSOs match the archived controls. Observed owned RSS and
  process high-water RSS stay below the predeclared 10 GiB screen.

This proves the loader rebase preserves each mode's arithmetic; it does **not** undo
M4-versus-scalar full-logit bit-equality rejection or grant a new numerical tolerance.
Qualification artifacts are under `gemma4-m4-lowhost-campaign/20261029T213000Z`.
Collection is deferred before any performance process starts. That window closes
with ten passed qualification stages and ten unrun performance/analysis stages,
without changing its expired execution guard or fabricating samples.

Collection resumes in `gemma4-m4-lowhost-performance/20261030T010000Z`, linking the
same immutable source worktrees and targets: **no rebuild, changed workload or failed
model retry**. All six model specifications and their serial/candidate,
candidate/serial, serial/candidate order are frozen before the first collection.
The [original performance protocol](#private-reordered-decode-performance-interruption-2026-10-29),
complete QA population, default scheduling, phase clocks and acceptance criteria are
unchanged. An additional pre-exec check requires at least 12 GiB `MemAvailable`;
owned-group RSS and actual Linux `VmHWM` must each remain at most 10 GiB. These checks
are not host isolation or a shared physical-memory reservation.

| QA profile | Paired throughput changes, runs 0 / 1 / 2 | Median paired change | Median paired p95 first / last-content ratio | Candidate M4 calls, runs 0 / 1 / 2 |
| --- | ---: | ---: | ---: | ---: |
| c1 | +0.20% / −0.05% / +0.30% | +0.20% | 0.99626 / 1.00058 | 0 / 0 / 0 |
| c4 | +1.19% / +0.67% / +0.98% | +0.98% | 0.99851 / 0.97200 | 9 / 9 / 10 |
| mixed8 | −1.04% / +1.57% / +1.77% | +1.57% | 0.99957 / 0.99882 | 23 / 23 / 23 |

These are medians of exact paired ratios, not ratios of median rates. All median p95
latency checks pass, but mixed8's first pair individually regresses first-content p95
by 8.69%, and its throughput does not improve. Joint 1 s / 5 s diagnostic passes are
127→126, 127→127, 127→127 at c1; 17→20, 20→20, 20→18 at c4; and 2→2 in every mixed8
pair, each out of 128 requests. The fixed no-paired-loss criterion therefore fails at
c1 and c4. These diagnostic deadlines are not product SLOs; the observations do not
isolate a causal latency effect of M4, which is never selected at c1.

Actual M4 execution remains sparse: c4 uses 36/36/40 of 1099 decode predictions
(3.28%/3.28%/3.64%), and mixed8 uses 92/1099 (8.37%) in every repetition. Remaining
decode predictions use genuine scalar fallback; each QA profile also executes 1709
prefill returns. These counts describe exposure, not measured GPU stall/occupancy causes.
Do not infer that higher exposure would pass the end-to-end gate.

All **1176 paired request outcomes** match complete IDs, text, usage and finish reasons.
Across six fresh model processes, all 2352 requests complete; each QA profile retains
1227 raw completion IDs, 127 stops and one length-64 result. All 18 QA profiles agree
on each question's delivered output. Instrumented source checks 56,820 complete finite
vocabulary rows across all profiles. Every profile returns to its model baseline and
every process ends at zero grants. The descriptive near-context profile retains four
owners, six raw generated IDs, two empty-content outcomes and no M4 calls in each mode;
logical peaks remain 14.795/14.799 GiB. Physical VRAM peaks, infeasible admission and
eviction/driver-OOM recovery are not qualified.

The smallest pre-exec `MemAvailable` is 21,340,499,968 bytes. Maximum observed owned RSS
is 8,231,776,256 bytes; maximum actual process `VmHWM` is 8,226,615,296 bytes. All six
processes pass the frozen memory screens without a host OOM. This is not a matched
old/new RSS comparison or a guarantee against unrelated host pressure. All 13 actual
forward-time DSOs are identical across modes/repetitions, and sources, binaries and
nine package pins remain stable. The scope closes at **01:27:24 UTC**, before its
predeclared 03:59:42 execution guard. All owned worker/watchdog groups are
absent, 74 supervised query checks pass and the GPU is idle at 33 MiB with no compute
PID. No stage is missing or filled. The summary's exit 1 is the expected **complete
negative screen**, with no controller error or forced cleanup, not another interrupted
model process.

Source manifest SHA256:
`fc4b0c862a0c996694484089c037e7922529db473a98b64593c3dd47587dbc0d`.
Serial/candidate performance binary SHA256:
`0cbcc2a88f98f28c55f95586d001fed8383176319591aca24c98229f8332c668` /
`e72cae31f3d67bda1e2bec644b1c2914399932b3cee5c1ef3587c472b94c3a7b`.
Own-mode rebase verification SHA256:
`848ac2b0b1eb791d1120933263e206155e9b9b5efeb61e9cc4a8478a7ee91666`.
Complete three-pair summary SHA256:
`34af7ac517034cd4d4ab2a5c46d4700086a290a7faa5e973ce25202a8f8ba7a9`.
Closure SHA256:
`58bfb069ff91250159c02f887d92233acb26951005320d31b33821d1e34a3037`.
Raw profiles, prepared IDs, clocks, traces, memory preconditions, maps, receipts and
analyses are also retained locally. Earlier failures, partial cohorts, sources,
binaries and targets remain intact and are not pooled into this population. Public
Rust is unchanged since `4f04a41`; its workspace/device qualification is reused, not
rerun. Metal and uninstrumented/HTTP performance checks are not run in this scope.

### Default c4 execution-cost observation (2026-10-30)

**Unused intermediate-prefill heads select the next bounded experiment, not a
speedup claim.** In the unchanged public-serial default c4 workload, their head
stream intervals total **2.256285 s** and their downstream CPU softcap totals
**2.480597 s**: **4.736882 s / 69.380802 s = 6.8274%** of the observed phase.
This exceeds the predeclared 5% cost-selection threshold. No head is omitted in
this measurement, and the rejected M4 candidate remains parked.

The private probe changes observation only. It retains the canonical 128-question
population, 26421 prompt tokens, greedy cap 64, EOS IDs, preprocessing worker count
one, text window four and default Engine scheduling. It prepares the original full
performance deck before loading weights, then executes its unchanged four-request
warmup and c4 profile only. There is no prefill barrier, artificial sleep, row-count
increase, kernel change or precision change.

Three timing-enabled CUDA events belong to the existing mutable model owner and
its original stream. They mark body start, head start and head enqueue completion,
with the last marker **before** the existing full-logit readback. No new barrier
splits body from head. All three events must already be complete after that owned
readback, before querying their elapsed times or reusing them. Failure uses the
existing model fault/retirement path. At most 4000 records are retained; each actual
decoder call contributes one record, correlated with accepted owner inputs and
positions rather than duplicated across submission rows. CPU clocks separately
measure readback/BF16 decoding and softcap.

[CUDA events](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-driver-api/group__CUDA__EVENT.html)
measure **stream intervals**, including host-enqueue gaps and possible
contention, not isolated kernel execution or occupancy. CPU head/readback wall time
can drain preceding body work; it is not head GPU cost. Body/head stream intervals
and enclosing host wall times overlap and must not be added. The selected sum uses
only the head stream interval and its downstream, nonoverlapping CPU softcap.

| Actual c4 work | Calls | Body stream interval, s | Head stream interval, s | CPU softcap, s |
| --- | ---: | ---: | ---: | ---: |
| Intermediate prefill, no returned token | 1581 | 31.015762 | 2.256285 | 2.480597 |
| Final prefill, token returned | 128 | 9.433150 | 0.181235 | 0.200584 |
| Decode | 1099 | 19.301426 | 1.554361 | 1.719388 |

Before workload collection, the observed decoder passes the entire 13-cohort
common-history control: **160 complete BF16 vocabulary rows are bit-identical** to
the qualified serial reference, with all 41,943,040 elements independently scanned
finite and every archived CPU winner condition preserved. All cohort grants return
to baseline. Both plain and observed c4 runs then preserve all 128 whole generated
ID sequences, texts, usage and finish reasons, including all 1227 raw completion
IDs, 127 stops and one length-64 result. Each run checks 2808 complete finite
vocabulary rows before selection, including intermediate-prefill returns. All four
warmup outputs also match the archived IDs.

The plain/observed c4 phases take **69.832034 / 69.380802 s**, ratio **0.993538**.
This passes the frozen single-pair observation screen (ratio at most 1.05), but an
ordered pair is not a statistical overhead bound, negative overhead or a measured
optimization gain. Loading, processor construction and JSON export remain outside
the phase clock. These are text-facade observations, not HTTP or uninstrumented
performance evidence.

The private source/target is retained under `gemma4-default-cost/20261030T020500Z`.
Its release CUDA library suite passes 15 tests, with three vision-corpus tests
ignored; both qualification examples pass release CUDA clippy and build. An earlier
host-only packet, `gemma4-default-cost/20261030T020000Z`, stops at Clippy's oversized
retained-harness functions; its build and all model stages remain unrun. The revised
packet splits those functions without removing validation. A GPU-occupied preflight
then defers device work before any model attempt. The fresh execution-only scope
`gemma4-default-cost-execution/20261030T024800Z` links the same qualified source and
binaries without copying, rebuilding or retrying a failed model.

All five numerical/model/analysis stages complete with exit zero. Sources, binaries
and nine package pins stay unchanged; actual pre/post maps match the same 13 DSOs.
Minimum pre-exec `MemAvailable` is 21,253,021,696 bytes; maximum sampled owned RSS
is 8,187,928,576 bytes and maximum actual process `VmHWM` is 8,183,955,456 bytes.
The 12 GiB availability floor and 10 GiB RSS/HWM screens pass, without proving host
isolation. Every process ends at zero grants. Closure at **02:57:48 UTC** precedes
the frozen execution guard; worker/watchdog groups are absent, all 38 supervised
queries complete and the GPU is idle at 33 MiB with no compute PID. Raw rows, inputs,
profiles, maps, receipts, analyses and an independent local recomputation are retained.

Execution source manifest SHA256:
`a65cfb1b22032c0c10ffa74c6f2b4432c58973a3ebc6353141d920ef07a9fa82`.
Cost analysis SHA256:
`2724bb987ff58a370a771128a8cf488dc2a745f3197813a469942a81d6c41e79`.
Independent verification SHA256:
`a3d6cedb882ae4a30132788a5329e0416965b17de3fca1ca7d6e0ccde695cf55`.
Closure SHA256:
`845b9c9c9f20019959e0b0d8b862e34860c549481384ff8f9cd23abbd2df3b3b`.

This selects an experiment that avoids heads only when no logits are requested.
It does not establish recoverable speedup: head-free progress needs explicit device
completion and unchanged failure retention before acknowledgement, plus a clear
state-only versus logits result contract. Common-history state/output qualification
and repeated end-to-end performance gates remain ahead. Public Rust is unchanged;
no full-workspace, Metal, HTTP or uninstrumented-performance checks run in this scope.

### State-only CUDA head-omission screen (2026-10-30)

**The private head-omission candidate passes the fixed three-pair screen:** median
paired throughput gains are **7.37% at c4** and **7.64% at mixed8**, with exact whole
outputs and no paired loss of diagnostic-deadline passes. This is a measured
text-facade improvement, not the earlier cost fraction treated as a speedup. The
candidate remains private at this screen; public CUDA still uses the full-head
advance fallback. The [subsequent qualification](#state-only-cuda-uninstrumented-and-http-qualification-2026-10-30)
below completes the remaining gates for integration.

Both sources start from public `6d9b17e` (production Rust `448a3a0`), rather than
reusing the older cost-observation executable. The only candidate/control production
source differences are in the CUDA decoder and its `GemmaDecode` bridge. A typed
readout callback shares whole-chunk validation, body execution, model-fault retention
and position commitment. Intermediate `advance_prefill` returns `Result<()>` without
last-row extraction, final normalization, tied-head projection, final-logit readback
or softcap. Final prefill and decode retain their original arithmetic.

The final layer's native scale already acknowledges completion: its successful
`Pending::retire` drains the same associated stream before publishing output. The
head-free operation uses that existing boundary, with no added synchronization,
event, kernel, scheduler or precision change. A failed body still faults the shared
model and retains its operation/allocation envelope; neither position acknowledgement
nor logical batch commitment bypasses retirement. The host PLE readback remains.

Both sources receive identical qualification-only requested-logit finiteness guards
and bounded executor traces. There are no CUDA timing events. Before performance,
**each source passes all 160 complete common-history BF16 vocabulary rows bit-for-bit
against the qualified serial reference**, with 41,943,040 finite elements, all archived
CPU-winner conditions and all 13 cohort grant returns preserved. Prompts now advance
in 16-token chunks without intermediate requested heads. Final rows and teacher-forced
continuations therefore check the state produced by omission, not just its token
choices. Separate real decoder tests compare headed/state-only partitions exactly at
scalar and eight-row/window-boundary preparation, including invalid whole chunks.

Each source also passes all seven ignored CUDA decoder/Engine/text/pressure tests.
The real capture-failure test submits a 17-token prompt, making its first 16-token
chunk intermediate: failed completion retains the admitted charges until capture
ends and shutdown settles, then refuses execution reuse. This is a real driver-error
and retirement check, not execution of a captured graph or arbitrary partial-kernel
fault injection. Workspace tests pass on Linux (**412 passed, 10 ignored per source**),
as do boundaries, formatting, CUDA-feature workspace all-target Clippy, release CUDA
tests/examples and CLI builds. Every Cargo invocation uses Rust 1.98.0, two jobs,
offline mode, locked dependencies where applicable and fresh isolated targets.

The frozen performance deck is unchanged: archived four-request warmup, all 128 QA
questions at c1/c4/mixed8, and four synthetic resident prompts of 4092–4095 tokens.
Greedy caps/EOS, worker one, window four and default scheduling remain fixed. Mixed8
uses the same closed-loop arrival/think delays, not an identical open-loop trace.
Each pair compares all **392 whole outcomes** (raw IDs, text, usage and finish reason);
all **1176 paired outcomes / 2352 requests** match, without retries or missing samples.
Each QA phase completes 1581 intermediate advances, 128 final-prefill heads and 1099
decode heads. Its **1227 requested complete vocabulary rows** are scanned finite;
the 1581 head-free advances do not pretend to return or validate discarded logits.
The control physically retains its full-head fallback on those advances.

Pairs run control/candidate, candidate/control, control/candidate, one model process
at a time. The predeclared screen requires median paired throughput gains of at least
5% at both c4 and mixed8; c1 throughput may regress at most 5%. Median paired p95
first/last-content latency may regress at most 5% in those three profiles, with no
paired loss of joint 1 s/5 s diagnostic passes and no failed requests. These diagnostic
thresholds are not product SLOs. Resident4 remains diagnostic rather than a selection
gate. Throughput counts completion IDs including EOS over client launch through full
text drain and explicit shutdown; loading, processor construction and JSON export
are outside that clock.

| Profile | Paired throughput gains, % | Median gain, % | Median p95 first / last-content ratios | Joint diagnostic passes, control → candidate |
| --- | --- | ---: | --- | --- |
| c1 | 7.22 / 8.50 / 3.38 | 7.22 | 0.927 / 0.916 | 127→128 / 127→128 / 127→128 |
| c4 | 7.68 / 7.37 / 6.72 | 7.37 | 0.959 / 0.916 | 20→31 / 19→31 / 20→31 |
| mixed8 | 7.67 / 7.64 / 6.52 | 7.64 | 0.940 / 0.930 | 2→3 / 2→3 / 2→3 |
| resident4, diagnostic | 7.84 / 9.42 / 9.60 | 9.42 | 0.913 / 0.913 | 0→0 / 0→0 / 0→0 |

The RTX 4090/driver 615.71.10, BF16 package and nine package pins are unchanged.
Actual numerical/performance process maps retain the same 13 DSOs. All profile
continuation grants return to the model baseline and every numerical/performance
process ends at zero grants. Minimum pre-exec `MemAvailable` is 27,773,296,640 bytes;
maximum sampled owned RSS is 8,213,192,704 bytes and maximum process `VmHWM` is
8,204,681,216 bytes. The 12 GiB availability floor and 10 GiB RSS/HWM screens pass;
these do not prove host isolation or a physical memory peak. Fresh launch preflights
check preserved desktop Git/binary state, idle GPU and no compute PID; all recorded
owned groups are absent at closure, with no forced cleanup.

The immutable execution scope is `gemma4-headless/20261030T065021Z`. All model stages
and three pair audits finish successfully. Its final aggregation exits 1 on a stale
M4 field name, `native_selected_each_repeat`, before writing a summary. The fresh
**analysis-only** scope `gemma4-headless-analysis/20261030T074522Z` corrects that key to
`state_only_requested_each_repeat`, re-audits all retained pairs, numerical rows,
source/binary/package/DSO identities and group absence, and passes the unchanged
thresholds. No model is rerun, population extended or tolerance changed. The original
aggregation failure remains intact. An earlier host-only Clippy failure on the
inherited non-Linux provenance stub and a packaging-only AppleDouble extraction
remain separate; neither launched a CUDA model.

Source manifest SHA256:
`0ab0581845f11bd069339d2c90b47b638a1903d67832e2674312840cd2321c44`.
Control/candidate performance binary SHA256:
`a1c202ff0fb7e8a9422b8f2d78199279114d24c005a962eb2006363fa7737de3` /
`38ae29365dda38536059fbc2450ae62e5179e6c47a30428ee07fdb1bf429859a`.
Repaired summary SHA256:
`6094909e999dc0a1fc474c53e93a892ad560d5b64114aeeb2ebee39669b4365f`.
Retained-output re-audit SHA256:
`80f390b01390c9fc28f34a8fe0a8a4aa9885a384638989a1ddcc039772c52d0b`.
Analysis closure SHA256:
`361f03ef7e88ff60b4191703b2f8d2090c417f52a06c2139a3ff3d6747ba4d70`.

This positive screen supports further qualification of this changed variant, not
automatic promotion. At its closure, HTTP selection/cancellation and **uninstrumented**
end-to-end performance remain unrun; the matched row guards and trace are observation
work. No broader device/model scope, isolation, competitive-engine parity or SOTA
claim follows. Metal head omission is untested. Public Rust and the qualified
full-head fallback remain unchanged at this point, and M4 stays parked.

### State-only CUDA uninstrumented and HTTP qualification (2026-10-30)

**Both remaining gates pass.** The uninstrumented fixed three-pair population measures
median throughput gains of **7.62% at c4** and **6.91% at mixed8**. The actual CUDA CLI
passes output/usage parity and cancellation with an already-enqueued healthy peer.
The public CUDA executor now uses the qualified state-only operation for intermediate
prefill. Final prefill/decode retain headed arithmetic; Metal retains its full-head
fallback. This is bounded experimental greedy E4B/RTX 4090 qualification, not broader
hardware/model support or competitive acceptance. M4 remains parked.

The uninstrumented control starts from `c4666a5` (production Rust `448a3a0`). Its
candidate changes only the CUDA decoder and `GemmaDecode` bridge described above.
Those two production files are byte-identical to the qualified numerical/lifecycle
candidate. Their 160 complete finite common-history rows and seven real CUDA tests
are reused by exact identity, not rerun. The committed decoder tests cover invalid
state-only chunks, headed/state-only full-row equality and teacher-forced continuation
through scalar/window-boundary partitions. The failed-drain test's 17-token prompt
forces its first 16 tokens through intermediate prefill.

The minimal private measurement entry point directly constructs
`Gemma4Executor::cuda` and `Engine::with_defaults`. It adds no executor/decoder row
scans, counters, traces, boundary wrappers or CUDA timing events. Pool/provenance
checks and JSON export remain outside the performance clock. Both sources pass
boundaries, formatting, release example/CLI Clippy and release example/CLI builds
in fresh isolated targets with Rust 1.98.0, two jobs, offline and locked dependencies.
The unchanged device prerequisites retain their earlier evidence. Public integration
also passes fresh macOS workspace tests (**438 passed, 11 ignored**), boundaries,
formatting, default/CUDA-feature workspace all-target Clippy and Gemma/text/CLI Metal
all-target Clippy. This adds compile/host evidence, not another device population.

The frozen deck, timing denominator and criteria are unchanged from the screen:
four warmups, all 128 QA questions at each of c1/c4/mixed8, then resident4 diagnostics;
orders control/candidate, candidate/control, control/candidate, serialized. All
**1176 paired observable outcomes / 2352 requests** match each other and the retained
qualified visible IDs, text, usage and finish reasons. Without trace, the suppressed
terminal EOS identity is not observed or inferred. No failed requests, retries,
missing-sample filling or adaptive extension occur. Mixed8 retains closed-loop
arrival/think delays, not an identical open-loop trace.

| Profile | Paired throughput gains, % | Median gain, % | Median p95 first / last-content ratios | Joint diagnostic passes, control → candidate |
| --- | --- | ---: | --- | --- |
| c1 | 7.69 / 8.23 / 6.77 | 7.69 | 0.896 / 0.916 | 127→128 / 127→128 / 127→128 |
| c4 | 7.62 / 7.94 / 6.53 | 7.62 | 0.919 / 0.917 | 20→32 / 20→30 / 20→31 |
| mixed8 | 9.08 / 6.91 / 6.60 | 6.91 | 0.942 / 0.936 | 2→3 / 2→3 / 2→3 |
| resident4, diagnostic | 9.60 / 9.26 / 9.00 | 9.26 | 0.915 / 0.915 | 0→0 / 0→0 / 0→0 |

All frozen throughput/latency criteria pass with no paired diagnostic-pass loss.
The 1 s/5 s thresholds remain diagnostics, not product SLOs; resident4 is not a
selection gate. Every profile returns to its 14,946,887,968-byte model grant baseline,
and each process ends at zero grants. Minimum pre-exec `MemAvailable` is
26,412,421,120 bytes; maximum sampled owned RSS is 8,219,688,960 bytes and maximum
process `VmHWM` is 8,215,543,808 bytes. The 12 GiB availability floor and 10 GiB
RSS/HWM screens pass, without establishing host isolation or physical peaks. Model,
package pins, driver/toolchain and actual mapped DSOs match the earlier screen.
All 22 stages pass; all 200 recorded groups are absent at closure, without forced
cleanup. The immutable scope is `gemma4-headless-uninstrumented/20261030T084306Z`.

The separate functional scope, `gemma4-headless-http/20261030T093548Z`, uses its exact
candidate CLI binary without rebuilding or repeating performance. It runs
`ribn serve-gemma <package> --bind 127.0.0.1:0 --model-name gemma --max-requests 4
--max-output-tokens 64` on the default CUDA path. Archived chat (18 prompt tokens,
cap four) returns `A mutex (short` and usage 18/4/22. The fixed longest canonical QA
has 581 prompt tokens and cap 64. Raw completions and chat, each JSON and SSE, match
its retained whole text, usage and finish. SSE checks one finish, one usage-only
final chunk and one `[DONE]`; chat also checks its assistant-role chunk. Wrong model
returns request-local HTTP 400.

Cancellation reads an actual nonempty delta, then receives a healthy QA peer's
HTTP 200 headers before disconnecting the first stream. The server constructs those
headers only after `TextModel::stream_request` has returned from runtime enqueue;
metrics also show both requests active before disconnect. The healthy peer finishes
exactly, abandonment increments by one, active requests settle to zero, readiness
stays 200 and a later archived chat completes exactly. `ignore_eos` is used only on
the cancellation request to keep it live, not as a numerical workaround. Final
metrics record nine requests, seven completed, one rejected, one abandoned, zero
failed and zero connection deadlines. Checked child SIGTERM drains the CLI to exit
zero; the child is reaped, all 24 recorded groups are absent and no compute PID
remains. CLI `VmHWM` is 8,050,618,368 bytes; maximum sampled owned-group RSS is
8,080,232,448 bytes. Actual mapped DSOs match before/after. Both supervised stages
pass without forced cleanup.

This proves functional CLI path coverage and recovery, not HTTP performance,
measured head-call counts or suppressed EOS identity. The CLI does not expose direct
BytePool grants; final-zero grant evidence comes from the separate numerical/text
facade scopes, not an HTTP observation. Long prompts plus the unchanged loader,
executor and chunk wiring establish selection without hot-path instrumentation.

Archive verification finds the original HTTP closure hashed `drive.stdout` before
its controller appended the two closure announcements. Its expected hash matches
the exact original 760-byte prefix; only those announcements extend it to 1090 bytes.
All other retained hashes match. The original manifest and log remain intact. The
fresh **analysis-only** scope `gemma4-headless-close-verification/20261030T094200Z`
verifies the exact append, all retained sources/outputs/receipts and group absence,
then records a full post-exit manifest after the controllers have exited. Its own
closure hashes verify in full; no models, criteria or execution verdicts change.

SHA256 identities:

- Production CUDA decoder / bridge:
  `44dd9482012658e7a77f1221241165702594bb35b682ec17c5f90f078ed05dcb` /
  `01d90423dc1002bb18624903c476f019a756b81506f17af1ed49ed418a251fba`.
- Uninstrumented control / candidate performance binaries:
  `db1f8bba9cc3cf45d08e23f117544c9f58e131222621e01d92493016a0bbc200` /
  `2c06dc5d2eb67f6b3089b0a5675669972fd3abb55f58eda3beaba0f197cd486d`.
- Actual candidate CLI: `9619050695d3c3628d86cc5d471a53c25915b715417370a9aa606276df3707f3`.
- Performance summary: `e257b4927ab05085e49ea0d4f36d077721b379841c7225f98612602a3ff4ce2e`.
- HTTP audit: `381c52f523bf2a3c7ada2f355bdd0650169dac45da7967b7ec88d37dcd692422`.
- Performance / HTTP evidence archives:
  `a6342f10e76d0992a5904be1057986c1dc66e5e2eabc585bc6d018b15b62b73d` /
  `5506d7aeba3fd8add9671b7034789221037c6ff979387300921ac4b7a3a7fb34`.
- Post-exit manifest: `a022c8c68fb26b890bb247fdcf1bb12211adc5203e7b7bcabe71198d9a32e7f3`;
  verification: `ab158c581076045077208131cabad8456bd9a59c54e77f1eb8c66e593e57869c`;
  analysis archive: `73558b81bc4a7fde5a34ff3f9d106081587f71afa5be3817a1d1d74907211ca6`.

### Post-head-omission native-call wall observation (2026-10-30)

**Attention is the largest observed call category, at 15.24% of c4 phase wall time.**
It selects the next bounded investigation after public state-only prefill (`5f51148`),
not a new optimized variant. Down, Gate and Up projections are close behind at
14.33%, 13.24% and 13.20%; this single observation does not establish a stable ranking
or attribute costs to GPU occupancy, stalls or host overhead.

The private observer times original synchronous native calls with host `Instant`.
Each call still executes exactly once with its original return value/error. No CUDA
kernel, timing event, wait, arithmetic, scheduler or executor change is added.
Nonoverlapping intervals include validation, driver work, device execution and
completion as encountered by the blocking call: **these are not GPU kernel times or
a GPU-versus-host partition**. The observer is model-owned and exports after explicit
execution-owner shutdown, outside the phase clock. Prepared widths 1/8/16 describe
one decoder working set, not concurrent-request counts. A normalized reverse-source
check removes the observation changes and reconstructs the public decoder, accounting
for rustfmt whitespace, closure blocks and trailing commas. The instrumented decoder
itself is not byte-identical to public source. All other tracked files are unchanged
apart from the private example manifest; no observer or harness is integrated publicly.

Precollection criteria fix **one OFF→ON pair using the same binary**, each with four
archived-chat warmups and the complete canonical c4 QA128 profile. The wrapper-off
control is not an untouched production binary. Both modes explicitly report private
instrumentation; OFF must have empty observations and zero counters. ON/OFF c4 elapsed
must be at most 1.05 before selecting any investigation; the largest original-call
category must account for at least 10% of observed c4 phase wall. Neither threshold is
fitted to these results. Auditor controls use retained valid outputs and explicitly
synthetic counter fixtures; corrupt outputs, clocks, populations, counts, timing sums,
observer modes and a failed ratio are rejected before collection.

| Observed c4 quantity | OFF | ON |
| --- | ---: | ---: |
| Complete phase wall, s | 64.232398 | 65.091219 |
| Requests / completion tokens including suppressed stops | 128 / 1227 | 128 / 1227 |
| p95 first / last content, s | 2.122860 / 3.225959 | 2.064243 / 3.266020 |
| Joint 1 s / 5 s diagnostic passes | 34 | 29 |

ON is **1.34% longer** in this ordered pair, passing the frozen elapsed-ratio screen.
It is not a statistical overhead bound or a causal estimate of instrumentation cost.
The diagnostic-pass count falls from 34 to 29; this observer is not a performance
promotion and has not passed the repeated latency/goodput advancement gates.

| Original blocking call category | Calls | Host wall, s | Share of ON c4 phase |
| --- | ---: | ---: | ---: |
| Layer attention | 135030 | 9.921687 | 15.24% |
| Layer Down projection | 135030 | 9.325162 | 14.33% |
| Layer Gate projection | 135030 | 8.616813 | 13.24% |
| Layer Up projection | 135030 | 8.595039 | 13.20% |
| Layer Query projection | 135030 | 2.925536 | 4.49% |
| Layer attention-output projection | 135030 | 2.816564 | 4.33% |
| Requested-head CPU softcap | 1227 | 1.923118 | 2.95% |
| Requested tied-head projection | 1227 | 1.723463 | 2.65% |

All measured categories sum to 63.734885 s within 64.242476 s of model-step wall and
65.091219 s of phase wall. The 0.507591 s unwrapped step residual and 0.848743 s outside
model steps are not assigned to a host bottleneck. The 2808 model steps consume
27520 inputs: 26421 prompt tokens plus 1099 decode inputs. There are 3215 body groups
(1590 sixteen-row, 65 eight-row, 1560 scalar); each complete layer role occurs 42 times
per group and KV-owner roles 24 times. Exactly 1227 requested heads remain, with
no intermediate-prefill head calls. Attention spans 4.824674 s at width one,
0.222579 s at width eight and 4.874434 s at width sixteen; scalar work includes prompt
tails as well as decode, so these bins do not partition prefill versus decode.

All **132 paired observable outcomes / 264 requests** match the retained qualified
visible IDs, text, usage and finish reasons, including warmups. Suppressed terminal
EOS identity is not observed. Every profile returns its model grant baseline and both
processes finish at zero grants. Actual pre/post mapped DSOs, all nine package pins,
preserved desktop Git state/binary, the 12 GiB disk and host-availability floors, and
the 10 GiB sampled RSS/HWM screens pass. All 15 execution/analysis stages pass and
all 127 recorded process groups are absent, with no forced cleanup. An external
closure runs only after every controller has exited; all 1980 retained file hashes
verify after copying locally. Targets remain preserved remotely; the archive includes
an exact copy of the model binary, not the build target tree.

Final private Mac and Linux prerequisites pass formatting, release CUDA example
Clippy and **18 library tests (3 ignored)**; Linux also passes boundaries and the release
example build. Earlier host-only preparation failures (an identifier lint, then a
Mac-only provenance stub) remain separate; fresh source/target scopes pass the corrected
checks. No failed model process is retried. The existing 160-row common-history
numerical evidence is retained, not rerun or relabeled as a new full-row collection.

The next bounded investigation is the native attention call's three-stage
score/softmax/value execution and completion path at this workload's widths and cache
spans. The category measurement does not distinguish those stages or establish that
fusion, launch reduction or another algorithm will win. Preserve the qualified path
and existing BF16/numerical and retirement contracts; any changed variant still needs
independent numerical qualification and repeated ≥5% median c4 **and** mixed8 throughput gains,
with c1, latency/goodput, HTTP, cancellation, pressure and fallback protections.
Do not substitute wider rows, unchanged M4, a compiler project or a softcap-only patch
for this evidence-selected investigation. No production speedup or competitive
acceptance is established by this diagnostic.

Retained scope: `gemma4-headless-body-cost/20261030T112431Z`; separate post-exit closure:
`gemma4-headless-body-cost-close/20261030T113333Z`. SHA256 identities:

- Single OFF/ON binary: `40e0f1c2256e0fef2c93920a0180f3996885ea4b32b411860576505883412f59`.
- Source manifest: `90416c51ff0542dd273527cd4f0c90b979bcea610848405c73d71799ba53b555`.
- Diagnostic summary: `2d1c009c12ef707987a3e304116893f2d9ddeadeebd307ae6f5dcab6c2a76a47`.
- Completed-file manifest: `a2764936d19091f127ada1906171d0f828dc713ece04f44c7a1c1776f3329e53`.
- Evidence archive: `f71430b7f354b43e0702383d354f3442057e5e6cde1812f9fd6ff6aed1d8aff7`.

### Attention-stage stream observation (2026-10-30)

**No stage-specific optimization is selected.** Scores are the largest measured stream
interval category, but account for only **44.74%** of the three-stage total, below the
frozen 50% majority criterion. Softmax accounts for 36.32% and values 18.94%. This
completes the attention-stage investigation selected by the preceding call-wall study;
it does not establish that attention has no optimization opportunity.

The private source starts from public `1e8f2f8`, with unchanged CUDA kernels, launch
arguments/order, BF16 arithmetic and original `Pending` retirement. Four reusable
same-stream timing events live in the kernel bundle retained by that pending owner,
including on partial enqueue/drain failure. They delimit scores, softmax and values.
Only after the original successful drain are all four queried for completion and
three elapsed intervals read. The observer uses the raw elapsed query, not cudarc
0.19.9's `elapsed_ms()` wrapper, which calls event synchronization internally. No new
synchronization or stream-wait call is added. Normalized reverse-source checks recover
the public Rust paths; the instrumented Rust files are not byte-identical to public
source. No observer is integrated publicly.

CUDA event intervals can include submission gaps and other scheduling effects; they
are **not pure kernel time or a GPU/host partition**. See the pinned
[CUDA 13.3 event API](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-driver-api/group__CUDA__EVENT.html).
The observer separately records host preflight, enqueue, original retirement and
elapsed-query intervals. Those host intervals overlap the stream intervals and must
not be added to them. Per-call event queries are inside the phase clock; only aggregate
export occurs after explicit execution-owner shutdown, outside that clock.

Precollection criteria fix one same-binary OFF→ON pair, each with four warmups and the
complete canonical c4 QA128 profile. Both modes use one private instrumented build;
OFF disables observations and creates none of this observer's timing markers. ON/OFF c4 elapsed must be at most 1.05.
A stage is selected only if its aggregate interval is at least 50% of the three-stage
sum **and** its duration is at least 5% of observed ON phase time. The latter is a
selection comparison across clocks, not a claim that GPU execution occupies that
fraction of the phase. The rules and auditor controls are frozen before collection;
no repetition, missing-sample filling or threshold adjustment follows the result.

| Observed c4 quantity | OFF | ON |
| --- | ---: | ---: |
| Complete phase wall, s | 64.304639 | 65.438715 |
| p95 first / last content, s | 2.172626 / 3.227469 | 2.208891 / 3.283660 |
| Joint 1 s / 5 s diagnostic passes | 31 | 31 |

ON is **1.76% longer**, passing the elapsed-ratio screen. One ordered pair is not a
statistical overhead bound, causal instrumentation-cost estimate or production gain.

| Same-stream interval | Aggregate, s | Share of three-stage intervals | Duration / ON phase |
| --- | ---: | ---: | ---: |
| Scores | 4.501416 | 44.74% | 6.88% |
| Softmax | 3.654138 | 36.32% | 5.58% |
| Values | 1.905516 | 18.94% | 2.91% |

The three intervals sum to 10.061070 s. Scores pass the 5% phase-duration screen but
fail the majority screen, so the result remains **no selection**. Ranking also varies
by geometry/cache bin; there is no qualified universal stage ranking. The 18 bounded
bins combine prepared rows 1/8/16, local width 256/window 512 or global width 512, and
key-span upper bounds 128/512/4096. The last bound is a bin label, not 4K qualification:
accepted histories in this population reach at most 593 positions. Scalar bins include
prompt tails as well as decode.

Separately, host preflight/enqueue/original-retirement/elapsed-query totals are
0.009833 / 0.643357 / 9.887204 / 0.061370 s, summing to 10.601765 s within 10.611899 s
of complete attention-call wall. Retirement includes waiting for the queued device
work: its 9.887204 s is **not measured host overhead**. Complete model-step wall is
64.586468 s within the 65.438715 s phase. Do not combine these nested host clocks with
the stream totals or infer occupancy/stalls from them.

All **132 paired observable outcomes / 264 requests** match retained visible IDs,
text, usage and finish reasons, including warmups; suppressed terminal EOS identity
is not observed. ON records all 135030 attention calls across 3215 body groups and
2808 model steps consuming 27520 inputs. Every geometry/cache-bin count independently
matches the canonical prompt/chunk and accepted-continuation schedule. All profile
grants return to baseline and both processes end at zero grants. Package pins, actual
mapped DSOs, preserved desktop Git/binary, the 12 GiB disk/host-availability floors and
10 GiB RSS/HWM screens pass. All 15 execution/analysis stages pass, all 127 recorded
groups are absent and no forced cleanup occurs. External post-controller closure and
local verification check all 2009 retained file hashes; targets stay preserved remotely.

Fresh private Mac and Linux checks pass boundaries, formatting, release CUDA library/
example Clippy and **74 library tests (4 ignored)**; Linux also builds the release
example. Earlier host-only Clippy failures remain preserved separately; corrected
sources use a fresh target, and no model run is retried. Existing common-history
numerical evidence is retained, not relabeled as a new full-row collection.

Keep the qualified attention path. Do not relax the majority criterion to select
scores, equate blocking retirement with CPU work, or infer a fusion/row-width winner.
A changed, separately frozen experiment is needed before choosing an optimization;
this population is complete and must not be extended. Future variants retain the
existing numerical, repeated-throughput, latency/goodput, retirement, HTTP, pressure
and fallback gates. M4 stays parked; no production speedup or broader qualification
follows from this diagnostic.

Retained scope: `gemma4-attention-cost/20261030T183714Z`; separate post-exit closure:
`gemma4-attention-cost-close/20261030T184530Z`. SHA256 identities:

- Single OFF/ON binary: `4091582dbb6c504ada9bcb65ad725e03324203014e9e75f9d27ecc681bb9ab96`.
- Source manifest: `ad4923bd416ab495fdba83baa094756664bf9db11fdfbd5aa8dc968b609d69e1`.
- Diagnostic summary: `0363b8ea9275cc202d89e35bec03798d1e62a4bf73a215b9a220c144521581d3`.
- Completed-file manifest: `2202796375c04765d6b6dc17fd0cbf278069d8b0545fc3ddd4046a36272ea966`.
- Evidence archive: `b9c5f2ca12df3f07e4a7f8be0a6a64a2a27a4faf820b8379169458a86ee428f9`.

### Advancement and remaining work

The scorer, component bounds, routing fixtures, actual-model prerequisites,
frozen-128 executor QA screen and concurrent-caller default Engine/text M4 selection
now pass in their stated scopes. Sequential starts still have their preserved
zero-M4 result; the concurrent-caller scope neither changes that outcome nor adds an
execution barrier.

Retain the validated scorer/routing controls and evaluator cross-checks; rerun them
when their affected inputs or implementation change. Preserve all old failures and
the original full-logit bit-equality rejection. New work requires frozen sources,
complete deck, isolated targets, finite budget and absolute cutoff. Do not restart
an unchanged failed cohort or fit a threshold to its output.

Passing these prerequisite gates permits **private performance experiments only**.
The first attempt above remains incomplete after a host OOM; its one complete pair
is not pooled with the new population. The host-lookup reduction and exact own-mode
rebase now pass, but the completed three-pair screen rejects the unchanged candidate:
c4/mixed8 miss 5% throughput gains and paired diagnostic-pass losses remain. Do not
advance this variant to further promotion gates without new bottleneck evidence and
a changed experiment. The [default c4 cost observation](#default-c4-execution-cost-observation-2026-10-30)
selected unused intermediate-prefill heads; the [changed head-omission screen](#state-only-cuda-head-omission-screen-2026-10-30)
passes exact common-history, runtime retirement and repeated text-facade gates. Its
[subsequent uninstrumented and actual CLI HTTP qualification](#state-only-cuda-uninstrumented-and-http-qualification-2026-10-30)
now passes the remaining bounded gates and supports public CUDA state-only integration.
Do not restart either completed three-pair population. The
[post-head-omission call-wall observation](#post-head-omission-native-call-wall-observation-2026-10-30)
selected native attention; the subsequent [stage-stream observation](#attention-stage-stream-observation-2026-10-30)
completes that investigation without selecting a stage-specific optimization.
Thin M4 exposure alone still does not select a scheduling or row-count change.
Host isolation remains unqualified. Further variants or broader selection scopes
still require evidence for their HTTP/mixed-arrival selection, cancellation, partial-enqueue/failed-drain
retirement, pressure, fallback and uninstrumented-performance scope.
The 128-question lexical screen does not qualify broad quality-equivalent serving,
sampling or a full-distribution error budget. The host-lookup change affects public
CUDA loading and retention only; neither it nor the passed prerequisites promotes M4,
changes precision policy or revises a prior full-model verdict.
