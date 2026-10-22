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
