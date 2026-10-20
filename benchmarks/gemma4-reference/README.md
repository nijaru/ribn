# Gemma 4 E4B independent reference

This prepares the numerical reference for Ribn's second decoder integration:
**Gemma 4 E4B instruction-tuned, BF16 SafeTensors, text-first**. It does not implement
Gemma execution in Ribn or qualify image/audio input or serving quality. The stock
CUDA fit measurement below is separate from Ribn execution.
The checkpoint, source checksums and reference versions are owned by
[`model.toml`](model.toml).

## Reproduce

Run from the repository root, with `hf` and `uv` installed. The full checkpoint is
about 16 GB on disk. FP32 collection loads the checkpoint's parameters in FP32 on
CPU and needs substantially more host RAM; the recorded host has 128 GiB. These are
not instructions to load FP32 weights onto the 24-GiB qualification GPU.

```bash
set -euo pipefail
MODEL=google/gemma-4-E4B-it
REV=ee0ef6023621cff504d758262d4e04895a5af4a2
SNAPSHOT=$(hf download "$MODEL" --revision "$REV" --format quiet)
hf cache verify "$MODEL" --revision "$REV" \
  --fail-on-missing-files --fail-on-extra-files

uv run --python 3.13 --script benchmarks/gemma4_reference.py \
  --snapshot "$SNAPSHOT" --output /tmp/gemma4-reference-fp32
uv run --python 3.13 --script benchmarks/gemma4_reference.py \
  --snapshot "$SNAPSHOT" --output /tmp/gemma4-reference-bf16 --precision bf16
```

Output directories must not already exist, including after a failed run. Keep the
entire package immutable throughout collection. The generator requires its exact flat
file set: extra sidecars such as `chat_templates/default.jinja` are rejected rather
than allowed to override verified inputs. It verifies every package file's size/SHA256,
checks pinned library versions and seven upstream implementation-file hashes, disables
network/model-code loading, uses CPU eager attention and checks tied embeddings and
loader diagnostics. The declared upstream commit is not a whole-installation attestation;
observed dependency versions/build metadata and the checked source-file scope are retained.
FP32 is a higher-precision diagnostic policy, not a bit-identical BF16 reproduction:
embedding-constant rounding and operation precision differ. BF16 is the stock CPU
reference policy, not evidence of identical NVIDIA arithmetic. No numerical tolerance
or task-quality acceptance follows from the difference between these references.

### Chunked prompt collection

Pass `--prefill-chunk-size N` to bound prompt input rows per forward, for example:

```sh
uv run --python 3.13 --script benchmarks/gemma4_reference.py \
  --snapshot "$SNAPSHOT" --output /tmp/gemma4-reference-chunked-bf16 \
  --precision bf16 --prefill-chunk-size 256 --stages
```

The default remains one whole-prompt forward. The final prompt chunk is prediction
`steps[0]`; earlier chunks appear in `prefill_intermediates`, with tensor prefixes
`<case>.prefill.<index>`. Every call records its actual input IDs, absolute positions,
history and cache/donor shapes. Intermediate predictions, including EOS, never stop
prompt ingestion. Tensor taps still retain the last row of each call, plus the full
persistent/donor spans described below; this is not an all-activation dump. Component
recomputation accepts these intermediate rows and preserves their tensor names.

The collector has a separate small stock-model gate:

```sh
uv run --python 3.13 --script benchmarks/gemma4-reference/test_capture.py
```

It verifies the pinned runtime/sources, uses synthetic BF16 weights with reduced widths
and vocabulary but all 42 layers and the real owner/shared-tail topology, and compares
recorded tensors with direct stock forwards. It covers default and staged taps,
whole/chunked prompts, partial tails, EOS during prompt ingestion and hook cleanup after
failure. This gate reproduced a default-mode missing-tap regression before its repair.
No new full-checkpoint chunked corpus or GPU qualification is claimed by these tests.

## Inputs and retained evidence

- One public non-thinking chat prompt, four greedy prediction rows unless a declared
  stop is selected. The generation configuration's EOS IDs are **1, 106 and 50**;
  using only the nested text config's EOS ID loses two stops.
- Two raw public-text prefixes of **511 and 513 tokens**, each followed by three
  teacher-forced tokens. These cross the 512-position local-attention boundary.
  Forced cases preserve the exact supplied history even if a predicted winner differs;
  they are numerical diagnostics, not serving/EOS or task-quality traces.
- Exact prompt/history/submitted IDs, position IDs, masks and top-five logprobs in
  `reference.json`; the source tensor shapes/dtypes in `catalog.json`.
- Full vocabulary logits before/after softcapping, embedding/PLE taps and every layer's
  last-position hidden state. Shapes and finiteness are checked before archiving.
- Persistent K/V samples for every cache owner, full persistent donor caches at layers
  **22/23**, and the actual **pre-trim donor spans** consumed by shared-tail layers.
  Last-position snapshots are not a whole-activation or whole-continuation-state proof.
- Runtime/build metadata, implementation-file hashes and tensor-artifact SHA256.
  A failure records `complete=false` with its error; an existing run is never replaced.

The distinction between persistent local storage and the donor attention span matters:
stock decoding retains at most **511** previous positions, while a donor returns those
positions plus the entire current chunk. Prefilling 513 tokens therefore supplies a
513-position span to late consumers even though its persistent local cache has already
trimmed to 511. Reusing only the trimmed cache changes the computation.

## Recorded CPU collection, 2026-10-18

Both pinned arithmetic policies completed all three cases: **12 prediction rows and
1,200 archived tensors per policy**. Loading reported no missing/unexpected active
weights, the output embedding stayed tied, and all archived tensors passed a separate
finiteness audit. The audit also checked exact input histories, donor-span shapes and
bit equality of the recorded pre/post-softcap transform in each policy.

| Policy | Tensor artifact bytes | SHA256 |
| --- | ---: | --- |
| FP32 CPU | 135687704 | `78fcdb0522c5dd2836cc75d571bb77eeb6d0a94653fce96bb7085bb1bd7b27de` |
| BF16 CPU | 67899336 | `46878e19c907f6cb4d2e88ae6077238465612d3bc1aab8441f57bfb4b83772f3` |

These CPU checks validate reference collection and its instrumentation, not numerical
agreement with an unimplemented Ribn decoder. They do not qualify image/audio,
device lifecycle, GPU memory peaks or serving performance.

## Recomputed attention components

Prepare operands for the next RMS/QKV/RoPE device gates without another full-model
forward. Use a completed corpus of the **same precision** from above. Keep both the
package and input corpus unchanged throughout collection:

```sh
uv run --python 3.13 --script benchmarks/gemma4_reference.py \
  --snapshot "$SNAPSHOT" --precision bf16 \
  --components-from /tmp/gemma4-reference-bf16 \
  --output /tmp/gemma4-components-bf16
```

The mode checks input provenance, contiguous greedy/forced histories and the input
archive's size/hash, then re-verifies the full package and pinned runtime. It loads
only selected layer parameters. All package bytes are still hashed. Existing output
directories are refused; failures retain an incomplete report.

For layers **0, 5, 22 and 23** (local/global, including both donors), it feeds each
archived last-position hidden vector through the pinned input RMS and attention
module. Hooks retain raw Q/K/V projections and their norms; the rotary module supplies
cos/sin at the actual history position. A capture-only attention boundary retains
rotated Q/K and normalized V, then raises an internal stop before attention, cache
updates or output projection. No fabricated attention output is returned.

**This recomputes one token, even for an original multi-token prefill.** Projection
rounding can depend on processing width. Match exact histories when comparing
corpora; identical case/step names do not establish identical input tokens. These
are component reference operands, not original full-model taps, a replayed
continuation, or evidence of Ribn parity.
The report links the input report/archive hashes; it does not independently re-run
or attest the earlier full-model execution. No numerical acceptance threshold follows.

On 2026-10-18 both policies completed **12 rows × four layers × 13 stages = 624
tensors**. A separate audit checked every tensor's shape/dtype/finiteness, bit-equal
archived hidden inputs, direct learned RMS and scale-free V RMS equations, split-half
rotation, transposition and global positive-zero phase padding. The rotation audit
explicitly rounds each multiply and the final add to the operand dtype; a fused FP32
implementation must not assume this is identical to eager BF16 arithmetic.

| Policy | Tensor artifact bytes | SHA256 |
| --- | ---: | --- |
| FP32 CPU components | 3845656 | `320c5f4eb03027aa2f0d31a33ac302dc4a7231c81f286f53eeeaa75a3de4e73a` |
| BF16 CPU components | 1953560 | `48debd4ed42f7a2081f632ab62a6ab8d0829e9f8138002ea5daed024d730dc8c` |

This qualifies collection/instrumentation only. Native BF16 linear device acceptance
still precedes dependent device math; mixed-KV lifetimes and full-model acceptance
remain separate gates.

## Layer-math stage taps

`benchmarks/gemma4_reference.py --stages` records the remaining decoder-layer
arithmetic from a real full-model CPU forward: the residual boundaries and four
norms, MLP gate/up/activation/product/down, per-layer PLE gate/activation/product/
projection/norm, each layer's per-layer input and scalar, and the final
normalization. Taps come from layers 0, 5, 22 and 23 at the last token of every row.
Nothing is recomputed from synthetic hidden inputs, so attention and cache behavior
stay the model's own.

```sh
python benchmarks/gemma4_reference.py --snapshot SNAPSHOT --output /new/path \
    --precision bf16 --stages
```

The committed fixture remains the small rotary/config evidence; these stage tensors
are large checkpoint-derived evidence kept outside the repository. An independent
audit re-derives the relations from the captured taps:

- exact BF16 equality for the residual additions, the layer-scalar multiply, the PLE
  `(context + token) × 2^-1/2` combine, the gated products, the tap identities
  (`mlp_output = mlp_down`, `ple_projection_raw = ple_projection`) and the final
  normalization state;
- GELU-tanh, RMS normalization and the tanh soft cap within eight FP32 epsilons of
  the prescribed formula applied to the captured input.

On 2026-10-19 this covered 444 exact and 168 bounded checks over the twelve rows, and
all 1200 taps shared with the earlier reference reproduced bit-identically. This is
reference evidence for the remaining layer-math gate: it does not execute Ribn
kernels, relax a device tolerance or establish model quality. Matrix products still
need their own numerical acceptance; only the elementwise and normalization
relations are checked here.

## Attention operand taps

`benchmarks/gemma4_attention_reference.py` captures the attention boundary of the
pinned CPU model on one ten-token prompt for layers 0/5/22/23: per-position queries,
keys and values (after Q/K/V normalization and rotary), the BF16 scores before the
additive mask, the post-softmax weights and the pre-`o_proj` output. It wraps the
module-level eager function while keeping `attn_implementation="eager"`, because
selecting a custom implementation name makes the model skip mask construction and
silently produce unmasked attention.

```sh
python benchmarks/gemma4_attention_reference.py --snapshot SNAPSHOT --output /new/path
```

The host CPU attention primitive reproduces the pinned policy: BF16 dot-product
scores, FP32 softmax cast to BF16 weights, one FP32 output accumulation and one final
BF16 rounding, causal masks with an optional sliding window and grouped-query head
mapping. Its ignored audit checks masked weights are exactly zero, visible weights
stay inside twice the visible score encoding step plus BF16 rounding slack, and
outputs stay inside the visible-key FP32 reduction bound plus the propagated weight
shift. On 2026-10-19 all four captures matched with zero error. The primitive checks
query containment, not history coverage; callers must pass the complete span their
cache policy requires. This is host operand evidence, not a decoder, KV lifetime or
Ribn device claim.

## Stock CUDA fit, 2026-10-18

The pinned stock Transformers model completed BF16 eager text execution on an
RTX 4090 (driver 615.71.09, PyTorch CUDA runtime 13.0). All model parameters,
including media weights unused by these inputs, were resident in BF16; the output
head remained tied. No quantization or CPU offload was used.

| Single-request case | Prediction rows | Peak PyTorch reserved, GiB |
| --- | ---: | ---: |
| Archived chat prefix, 18 tokens | 4 | 14.852 |
| Archived 511-token prefix | 4 | 14.963 |
| Archived 513-token prefix | 4 | 14.969 |
| 4096-token full prefill, then greedy continuation | 32 | 17.098 |

The first three cases replayed the exact CPU BF16 histories. All 12 greedy winners
matched; logits were finite and all 24 persistent KV-owner shapes matched. This is
not distribution-level numerical acceptance. The larger prompt was BOS followed by
`FILLER * 110` from `gemma4_reference.py`, tokenized without added specials and
truncated to 4096 tokens. Generation honored the metadata stops and reached all 32
requested predictions. It used one 4096-token prefill, not chunked prefill.

The probe used the manifest's exact package/library/source pins, offline loading,
deterministic algorithms, eight CPU threads, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
TF32 disabled and BF16/FP16 reduced-precision reductions disabled. Peak allocator
counters reset before each case; caches were released between cases. Full-vocabulary
last-position logits were copied to CPU for recording. The one-off probe's SHA256
was `e794a728e6026a1f26748a5e352e0e82a2f56ef5ab7061ae1642793355522d46`;
its 44-row logits artifact was 46,141,320 bytes, SHA256
`106c4319524df77a9fff3fc72e820536139286a1f5f5bd1cffd9933c6bd15e6b`.

Whole-device sampling every 200 ms peaked at **18,042 MiB used (17.62 GiB)**,
distinct from PyTorch's 17.10-GiB reserved peak. Before the run, the device reported
24,564 MiB total, 404 MiB reserved, 33 MiB used and 24,128 MiB free: existing non-free
use was about 437 MiB, not an assumed GUI workload or a fixed OS charge. The sampler
can miss shorter peaks. After model/cache release, PyTorch still reported 32 MiB
allocated/reserved; this is not a zero-allocation teardown claim.

**BF16 fits this measured workload without quantization.** This does not establish
Ribn forward execution, concurrent serving capacity, arbitrary-context fit, task
quality or performance. The CPU collector above remains CPU-only.

### Offline CPU/CUDA logit diagnostic

A 2026-10-18 offline audit matched the exact history and submitted token IDs for all
12 archived rows across CPU FP32, CPU BF16 and CUDA BF16. Input report/archive hashes,
package/source pins and the CUDA probe's link to its CPU input report were checked.
The CUDA JSON does not independently record positions: the hash-verified probe passes
the linked CPU row's exact positions and all-ones history mask. No model was rerun.

Every pair agreed on all 12 greedy winners, but **no full logit vector was bit-identical**.
The table reports maxima over those 12 rows, using complete 262,144-element
**post-softcap** vectors and FP64 metric arithmetic:

| Reference → candidate | Max absolute logit delta | Max centered logit RMSE | Max KL(ref ∥ candidate), nats | Max total variation |
| --- | ---: | ---: | ---: | ---: |
| CPU BF16 → CUDA BF16 | 1.265625 | 0.120362 | 0.000965690 | 0.0138979 |
| CPU FP32 → CUDA BF16 | 0.855922 | 0.107663 | 0.00188854 | 0.0239722 |
| CPU FP32 → CPU BF16 | 0.750395 | 0.106215 | 0.00474907 | 0.0348529 |

Probabilities are full-vocabulary softmax values. KL is `sum(p_ref * (log p_ref -
log p_candidate))`; total variation is half their L1 distance. Centered RMSE removes
the mean logit delta, since a uniform shift leaves softmax unchanged. A separate
stdlib `math.fsum` implementation checked the metric calculations. The 32 CUDA-only
4096-token-workload predictions have no CPU counterpart and were excluded.

These observations do not rank accuracy against an ideal model, establish task quality,
explain the differences' causes, or define a Ribn error budget. CPU references remain
useful diagnostics, not bitwise CUDA targets. Native device/model acceptance still needs
an independently specified reference scope and numerical/quality policy; existing
stock-reference differences grant no extra tolerance. Hidden states, cache contents
and raw pre-softcap logits were not retained by this CUDA probe.

## Integration constraints established by the real package

The verified artifact has **2,130 BF16 tensors**. Its stored text namespace has 719;
the stock implementation intentionally ignores 54 tail K/V/K-normalization tensors
because layers 24–41 reuse earlier K/V. `lm_head.weight` is absent and tied to the main
embedding. Loader diagnostics must otherwise remain clean, not silently initialize
missing active parameters.

The decoder has 42 layers, with local/global head widths **256/512**, 20 local and
four global persistent KV owners, and a required **5.25-GiB BF16 per-layer embedding
table**. Do not infer head width from hidden width divided by query-head count.
The current Transformers configuration requires per-layer geometry access for this
heterogeneous layout.

Header-derived payload sizes are about **14.9 GiB for the full checkpoint** and
**13.90 GiB for active text tensors**, excluding unused tail projections and media.
These are encoded-byte sums, not measured device allocations, repacking peaks or a
GPU-fit qualification. The stock CUDA measurement above replaces the provisional
0.5-GiB idle-use estimate for that run; future capacity planning must remeasure free
VRAM. Converting active text weights wholesale to FP32 still exceeds 24 GiB before
any KV or scratch allocation.

Ribn's file-backed SafeTensors path now uses retained metadata/files and caller-buffered
payload readers. The [host I/O gate](../loader-memory/README.md) checks the full artifact
and every tensor's encoded bytes without loading the checkpoint into application heap.
The device weight gate below qualifies native BF16 materialization and tied allocation
ownership separately. Mixed-window KV, donor-span lifetimes and numerical forward
execution remain prerequisites for a Ribn decoder.

## Ribn text-source preflight

`engine-gemma4` now owns the dense causal text configuration and parameter schema.
The checked-in [config fixture](config.json) is the exact publisher file pinned by
`model.toml`, not a reduced handwritten E4B approximation. Unknown text configuration
fields, unsupported attention/MLP/precision variants, missing active tensors, wrong
shapes/dtypes, inconsistent shard indices and untied output tensors reject.

```sh
cargo run --release -p engine-gemma4 --example preflight -- \
  /path/to/pinned/snapshot --hash-parameters > /tmp/gemma4-source.json
```

On the same M3 Max host as the [artifact gate](../loader-memory/README.md), the real
package yielded **665 active BF16 parameters**, **54 excluded shared-tail parameters**
and **1,411 excluded non-text parameters**. Every active shape/dtype matched the stock
catalog; every payload SHA256 matched the independently audited artifact reader.
Unique active encoded bytes were **14,926,026,836**. The output projection is the exact
same logical source as the unscaled input embedding, not another allocation.

The hash probe used a 1-MiB caller buffer and peaked at **7,028,736 bytes process RSS**.
This measures retained metadata/readers, I/O and the diagnostic report, not a loaded
model, GPU preparation, physical VRAM or a process-wide byte guarantee. Omit
`--hash-parameters` for header-only preflight. Keep package members unchanged while
opening and never mutate selected files in place through aliases while readers live.

Host tests cover a second geometry with hidden width not divisible by query-head count,
single/sharded inventories, optional inactive tail weights, wrong/dangling shard routes,
missing/malformed/unknown active parameters, exact BF16 payload preservation and retained
readers after package drop/path replacement. These checks do not qualify execution.

## Native BF16 device weight gate

At `1d4a7fb`, the model-owned CUDA loader materialized all **665 active parameters**
without F32 conversion or a second output-head allocation. On the same RTX 4090,
two complete load/readback/unload runs matched every parameter's shape, dtype,
encoded length and SHA256 against the independently audited source. Readback compared
every byte, not samples. Both explicit unloads returned the pool charge to zero and
restored the exact pre-allocation CUDA free-memory reading.

```sh
cargo run --release -p engine-gemma4 --features cuda --example cuda_weights -- \
  /path/to/pinned/snapshot 17179869184
```

The final argument supplies a 16-GiB weight pool, not a whole-inference budget.
The default staging bound is 1 MiB; the diagnostic also owns bounded source/readback
comparison buffers. The JSON report includes all source/device parameter hashes,
tied-slot identity, logical charges and device free-memory readings.

| Quantity | Bytes |
| --- | ---: |
| Unique encoded text weights | 14,926,026,836 |
| Requested allocation including 16-byte tensor padding | 14,926,027,424 |
| Observed CUDA free-memory reduction | 14,927,527,936 |

The difference between requested and observed allocation illustrates driver rounding;
the logical pool is not a hard physical-VRAM bound. Whole-device 200-ms sampling
peaked at **14,664 MiB used** for each weight-only run. Neither activations, KV,
attention workspace nor a decoder forward pass was allocated/executed by this probe.

The NVIDIA storage gate and both model-level device gates also passed under Compute
Sanitizer memcheck with zero reported errors. They protect arbitrary BF16 bits, odd
staging/tail sizes, alignment, unload/reload identity, retained readers after pathname
replacement, budget refusal before allocation, and explicit cleanup of a partially
uploaded set after injected reader EOF. Failed device drain/free and unwind ownership
are exercised by deterministic host transport tests, not injected hardware failures.

```sh
cargo test --release -p engine-nvidia --features cuda --test cuda_bf16_weights --locked -- --ignored --test-threads=1
cargo test --release -p engine-gemma4 --features cuda --test cuda_weights --locked -- --ignored --test-threads=1
```

This qualifies the weight preparation slice, **not Gemma model execution**, mixed-KV
retirement, concurrent loading/serving capacity or task quality.

## Native BF16 linear component gate

`CudaBf16Gemm` now binds prepared BF16 weights and row-major `CudaBf16Matrix`
inputs/outputs to cuBLAS; `Gemma4CudaWeights::linear_into` owns the logical-role
mapping, including the raw tied head. Output and explicit library workspace are
charged before launch. Success includes stream completion; failed/partial enqueue
retains every accessed allocation and the handle until an acknowledged drain.
CUDA's per-thread default pseudo-stream is rejected because these owners/errors can
move between threads. This is an execution building block, not a Gemma decoder.

At `53c8aed` on 2026-10-19, the shared-allocation/access-owner refactor and linear
operations passed all six device tests on the RTX 4090 (driver 615.71.09), both
normally and under unfiltered Compute Sanitizer memcheck, with zero reported errors:
one storage test, two model-weight tests, two linear tests and one logical-binding
test. These rerun the weight byte/readback/unload and partial-source cleanup protection
rather than inheriting it from the earlier `1d4a7fb` result.

Two fresh real-package diagnostics also matched every byte, shape, dtype and source
SHA256 for all 665 parameters against the independent host audit. Both explicit
unloads refunded the pool to zero. CUDA free memory was 24,885,723,136 bytes before
allocation and after unload, and 9,958,195,200 bytes while weights were resident.
Requested weight allocation, driver rounding and 14,664-MiB sampled weight-only peaks
were unchanged from the table above. The 18.5/15.6-second runs include loading,
hashing and readback; they are not inference timings. Linux workspace tests passed.

Reproduce with the weight gates and real-package diagnostic above, then:

```sh
cargo test --release -p engine-nvidia --features cuda --test cuda_bf16_gemm --locked -- --ignored --test-threads=1
cargo test --release -p engine-gemma4 --features cuda --test cuda_linear --locked -- --ignored --test-threads=1
```

Run these device binaries under Compute Sanitizer memcheck as well, serialized
with other full-model work. The recorded runs used a 180-second process-group timeout
per small gate and completed without interruption. The component reference uses exact integer dot products
scaled by 1/64 and independent BF16 round-to-nearest-even, with widths 7, 63, 256,
512 and 2560, odd output widths, multiple rows and output reuse. The model gate uses
selected-source dyadic operands for four logical projections. Both require exact
finite results; neither sets a whole-model tolerance. A separate gate rejects
per-thread pseudo-streams on their originating thread and after a cross-thread move.
Failed-completion/unwind retention remains deterministic host-injection evidence.
No full forward pass, mixed-KV lifetime, serving or speedup is qualified by this slice.

## Native BF16 RMS component gate

On 2026-10-19 (UTC), native grouped RMS passed its backend and model-binding gates
on the RTX 4090 with driver 615.71.09 / CUDA 13.3. The backend normalizes contiguous
BF16 groups using FP32 variance and normalization, optional direct learned scale,
and final BF16 RNE. Gemma binds its epsilon, hidden/per-head/packed-PLE grouping
and scale-free V at KV owners; shared tails reject fresh V normalization.

```sh
cargo test --release -p engine-nvidia --features cuda --test cuda_bf16_rms --locked -- --ignored --test-threads=1
cargo test --release -p engine-gemma4 --features cuda --test cuda_norm --locked -- --ignored --test-threads=1
```

The backend gate covers widths 7/63/256/512/2560, multiple rows/groups, mixed input
exponents, negative/zero learned scales, scale-free RMS, signed zeros and reused
output. It rejects uninitialized input, invalid groups/epsilon/scale and foreign
streams without changing output. Model fixtures cover local/global Q/K, hidden
and packed-PLE norms, inactive scales, missing query heads and shared-tail V rejection.
The per-thread pseudo-stream rejection gate now covers the normalizer too.

Acceptance was declared before device execution: finite, correctly shaped output,
exact signed zeros, otherwise at most **one adjacent BF16 encoding** from an independent
FP64 RMS equation with direct nearest-BF16 selection. For these bounded finite
vectors, the FP32 positive-energy sum bound at width 2560 is approximately 1.53e-4;
inverse square root reduces its relative effect, far below BF16 spacing. FP32
multiplications and the documented `powf` error are also finer than BF16 spacing.
The [CUDA 13.3 math guide](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-programming-guide/05-appendices/mathematical-functions.html)
states observed, not guaranteed, transcendental error bounds. This policy permits a
rounding-boundary difference for these component cases, not arbitrary-input accuracy,
real-model RMS parity or a whole-model/logit budget. No fast-math mode is enabled.

Both new tests passed normally and under unfiltered memcheck; the backend also
passed racecheck (zero hazards/warnings) and synccheck (zero errors). All six prior
native BF16 tests and their four memcheck binaries passed again after the shared
completion-error/access-owner changes. Two further real-package diagnostics matched
all 665 source/device hashes against the independent host audit and again restored
free VRAM exactly on unload with zero remaining pool charge. These are weight
regressions, not real-package forward or RMS numerical execution. Linux workspace
and CUDA-feature NVIDIA unit tests passed; failed-completion/unwind retention remains
host fault-injection evidence. Device gates were serialized and bounded, with no
interrupted test counted as a pass. No complete decoder, mixed-KV lifetime, serving
or performance is qualified here.

## Native BF16 rotary execution gate

Rotary execution is split into two completion-owned device operations.
`CudaBf16RotaryFactors` consumes the model's padded FP32 inverse frequencies and
integer positions, evaluates `float(position) × frequency` in FP32, computes cosine
and sine on the device without fast-math, duplicates the half-head vector across both
halves and stores BF16 round-to-nearest-even. Scaling is exactly 1.0 in the pinned
path, so it is not applied. `CudaBf16RotaryApply` then rotates prepared BF16 inputs
`[rows, heads × head_dim]` with those factors, rounding each product to BF16 before
the addition. A fused FP32 rotation is not equivalent, and both factor halves are
read rather than assuming duplication or padding.

A call validates every matrix, actual allocation range, whole head, positive even
width, row count and one exact stable stream before invalidating output; positions
must convert to FP32 exactly and fit the declared row capacity. Frequency and position
buffers are charged at construction, positions are re-uploaded and drained per call,
and each pending completion retains exactly the buffers that operation reads or writes
together with its compiled code. Failed or uncertain enqueue/completion
quarantines that envelope instead of refunding a guess. CUDA's per-thread default
pseudo-stream is rejected before module loading because these owners and errors can
move between threads. Kernel code/context overhead is not an application buffer grant.

```sh
cargo test --release -p engine-nvidia --features cuda --test cuda_bf16_rotary --locked -- --ignored --test-threads=1
cargo test --release -p engine-nvidia --features cuda --test cuda_bf16_factors --locked -- --ignored --test-threads=1
RIBN_ROTARY_REFERENCE=/path/reference.json \
  cargo test --release -p engine-gemma4 --features cuda --test cuda_rotary_apply --locked -- --ignored --test-threads=1
```

The application gate uses exact dyadic BF16 operands at widths 2/10/126/256/512 and
one to eight heads, including negative zero, sign flips, non-identical factor halves
and output reuse. Acceptance is bit equality with an independent FP64 nearest-BF16
computation that rounds each product before addition; the fixture fails if a single
final rounding is substituted. The factor gate checks the analytic identities that do
not need a trig oracle: zero position gives exactly one, zero frequency gives exactly
one and positive zero, a negative frequency keeps the product's negative zero at
position zero, opposite frequencies share cosine and negate sine, both halves are
identical, and capacity/position/stream violations reject without changing factors.

The model gate replays a stock-CUDA fixture from `--apply-cuda`, which applies pinned
Transformers' own CUDA cosine/sine to two geometries, two attention kinds and two head
counts — eight applications at positions including 0, 511, 512, 513, 4095, 65535 and
131071, repeated and non-monotonic. Frequencies are prepared in CPU FP32 and moved to
CUDA without a dtype cast, matching the pinned stock path. It compares cosine, sine
and rotated output bits exactly, and requires the fixture's frequency vector to be the
same bit pattern the host frequency gate qualified. Frequency derivation from theta on
the device is not implemented; the model-owned FP32 vector remains host-prepared.

```sh
python benchmarks/gemma4_rotary_reference.py --apply-cuda --output /new/path/reference.json
```

On 2026-10-19 (UTC), the RTX 4090 with driver 615.71.10 (CUDA 13.4 UMD, toolkit and
NVRTC 13.3.33) passed twelve storage/weight/linear/binding/RMS/rotary/factor cases
across nine device binaries normally and under nine unfiltered memcheck binaries with
zero reported errors. The stock fixture recorded 580/2370/346/1259 single-rounding
mismatches for E4B and 12/39/7/13 for the alternate geometry, so exact comparison is
not vacuous. The two 665-parameter readback/unload probes requalified at `309ac25`
remain valid because weight storage and allocation are unchanged. Linux workspace
tests (354) and Python tests (25) passed.

This qualifies rotary execution from a model-owned FP32 inverse-frequency vector with
supplied integer positions. Native frequency derivation, Gemma layer binding, a full
forward pass, mixed-KV lifetime and performance remain unqualified.

## Elementwise layer-math component gate

`CudaBf16Elementwise` prepares add, multiply, scale, tanh-GELU, soft cap and PLE
combine on one stable stream with the same completion-owned retention as the other
BF16 components. Each call validates whole equal-shaped ready operands, one associated
stream and finite scalar parameters before invalidating output; failed preflight
preserves it. Add, multiply, scale and the PLE multiply round once to BF16; PLE rounds
its add separately. Soft cap reproduces the host policy's three roundings (divide,
tanh, multiply). GELU evaluates the pinned algebraic form in FP32 and rounds once.
Invalid calls do not enqueue or refund storage, and uncertain completion retains the
program, stream and every accessed matrix.

The component gate uses exact dyadic operands at 2×8, 1×17 and 3×5, including an
element count that is not a multiple of the 256-thread block, and requires bit
equality for add/multiply/scale. PLE, GELU and soft cap are compared with an
independent host FP32 evaluation and RNE, allowing at most one adjacent BF16 encoding.
Validation cases reject shape mismatch, non-finite/zero parameters, a foreign stream
and reuse after a successful call without changing output.

The model gate replays the archived stage taps for layers 0/5/22/23: 20 bit-exact
comparisons (both residual adds, PLE combine, layer-scalar scale and gated product),
four GELU comparisons inside the host audit's bound (`16·eps·|x|` plus one reference
encoding), and the final soft cap over all 262,144 logits (`8·eps·cap` plus two
encodings).

```sh
cargo test --release -p engine-nvidia --features cuda --test cuda_bf16_elementwise --locked -- --ignored --test-threads=1
RIBN_GEMMA_STAGES=/path/tensors.safetensors \
  cargo test --release -p engine-gemma4 --features cuda --test cuda_elementwise --locked -- --ignored --test-threads=1
```

On 2026-10-20 the RTX 4090 (driver 615.71.10, toolkit 13.3) passed both device tests
normally and under memcheck with zero reported errors; racecheck and synccheck were
clean for the component. Worst GELU bound usage was 1/64; the soft-capped logits
matched all 262,144 archived encodings exactly. This qualifies the remaining
elementwise layer math at `24f1ac2`, not attention, MLP projections, a decoder,
mixed-KV lifetime or performance.

## Native BF16 attention component gate

`CudaBf16Attention` runs three completion-owned kernels over one charged, reusable
BF16 score/weight scratch matrix: query-key scores, softmax weights and the weighted
value sum. Scores and weights round to BF16; softmax and value accumulation stay FP32
and walk visible keys in ascending order like the host decoder. Grouped-query heads,
causal masking, an optional sliding window and non-zero absolute query/key starts are
explicit; the caller supplies every key the cache policy requires, and containment is
checked rather than coverage. A call validates geometry, whole head groups, scratch
capacity and one associated stream before invalidating output. The scratch bound is
fixed at construction; this is not flash attention, a fused prefill kernel or a KV
cache owner.

The host-contract gate compares six cases against the CPU implementation: GQA at width
8/256/512, full and sliding windows (2, 3 and 512), a single decode query against
history, and non-zero query starts. Every output was bit-identical (worst encoding
distance 0), including a window larger than the position count. That case found and
fixed an unsigned window-start underflow that had skipped the earliest key.

The model gate replays the archived stock attention operands for layers 0/5/22/23
(ten positions, local and global head widths). The host decoder first matches the
independent stock capture bit-for-bit; the device output is then bit-identical to that
capture for all four layers (worst encoding distance 0).

```sh
cargo test --release -p engine-nvidia --features cuda --test cuda_bf16_attention --locked -- --ignored --test-threads=1
RIBN_GEMMA_ATTENTION=/path/to/attention/corpus \
  cargo test --release -p engine-gemma4 --features cuda --test cuda_attention --locked -- --ignored --test-threads=1
```

On 2026-10-20 the RTX 4090 (driver 615.71.10, toolkit 13.3) passed both device tests
normally and under memcheck with zero reported errors; racecheck and synccheck were
clean for the component. This qualifies the attention component at `7a8293e`, not KV
cache ownership, decoder wiring, donor-span lifetime, prefill/softmax performance or a
full forward pass. The serial per-head softmax and three-pass key traversal are
correctness choices, not optimized execution.

## Native CUDA Gemma decoder gate

`Gemma4CudaModel` binds the qualified components into a host-orchestrated,
one-token decoder: scaled embedding and PLE lookups on the host, device projections,
grouped RMS, rotary factors/application, causal grouped-query attention, the
residual/PLE/MLP elementwise algebra, final normalization, the tied head and a host
soft cap. KV owners keep device-resident BF16 caches; shared-tail layers read their
owner's cache. K/V rows now append through completion-owned device copies (see below).
Cache capacity is explicit
and attention distinguishes storage rows from the visible key span. Any failed
component retains its retirement owner on the model for an explicit retry.

The archived chat row reproduces all four greedy tokens through this decoder. On
2026-10-20 the RTX 4090 (driver 615.71.10) loaded the 16 GB package in **2.25 s**,
processed the 18-token prompt in **1.57 s**, and decoded at **82.9/83.0/83.1 ms per
token**. The full test, including weight load, passed under memcheck with zero
reported errors. These are diagnostic timings for a correctness-first path with
per-operation synchronization, host K/V round trips and no batching; they are not a
competitive result or a serving measurement.

```sh
RIBN_GEMMA_PACKAGE=/path/to/package RIBN_GEMMA_REFERENCE=/path/to/reference.json \
  cargo test --release -p engine-gemma4 --features cuda --test cuda_decoder --locked -- --ignored --test-threads=1
```

At `88b5da5` this establishes executable CUDA Gemma arithmetic end to end. At
`6bfdf87` the same decoder is bound to `Gemma4Executor`, the text facade and the
`ribn run-gemma` CLI: the executor and text-path tests reproduce the archived chat
tokens (text path 9.24 s including load), and the CLI streams coherent text on the
4090. This still does not qualify sampling, eviction, mixed-width cache lifetime,
batched prefill, HTTP serving or competitive performance. Device append is qualified
separately below.

## Device-side K/V append

At `a8e5d23` (2026-10-20), `CudaBf16Matrix::copy_rows_from` replaces the decoder's K/V D2H/H2D
round trip with a same-stream device copy. It preserves encodings without arithmetic,
retains both allocations and their charges before possible enqueue, and drains before
returning success. Failed enqueue/completion retains the existing retry/quarantine
owner. No asynchronous execution or new kernel is introduced.

Matrices now track their contiguous initialized row prefix instead of declaring the
whole cache ready after a partial write. Copies cannot leave holes; readback and
attention reject uninitialized rows, while whole-matrix operations require full
initialization. The obsolete host `write_chunk` interface is removed.

Acceptance was fixed before measurement: exact copied bytes, unchanged untouched rows,
exact common-history full-model logits and the archived greedy tokens. On the same
RTX 4090, driver 615.71.10 and CUDA 13.3:

- Both `cuda_bf16_copy` tests passed normally and under memcheck: one/three-row copies,
  widths 7/256/512, first/interior/final offsets, raw BF16 encodings including NaN payloads,
  invalid ranges/foreign streams, initialized-prefix extension and attention rejection.
  Existing linear/RMS/rotary/factor/elementwise/attention component gates also passed.
- A temporary before/after probe compared all **262,144 finite BF16 logits at each of
  four predictions**, with the same 18-token prompt and three common continuation tokens.
  Every encoding matched exactly. This compares append implementations, not a new
  independent model-quality reference. The committed decoder/executor/text gates each
  reproduced the four independent archived greedy tokens. The real CUDA `run-gemma`
  CLI also completed a raw/piped-stdin four-token smoke; that is input-path evidence,
  not a quality check for its unconstrained output.
- Copy-component and full-decoder memcheck reported **zero errors**. Failed-drain
  ownership relies on the shared host fault-injected `Pending` envelope plus source
  review; hardware failure was not injected. Mixed-retention lifetime remains unqualified.

A temporary timer around each owner's paired K/V append measured 504 append regions
(24 owners × 21 processed tokens). The baseline spent **635.8 ms** there during prefill
and **35.18–35.26 ms** per decode step; device append spent **3.46 ms** and
**0.188–0.189 ms** respectively. These wall intervals include synchronization; baseline
cache writes split through its five-byte staging buffer, so this is not a general
CPU/GPU transfer-cost estimate.

The instrumented pair measured **1.593 → 0.921 s prefill**, **83.1–83.2 → 47.3–47.4 ms/token**.
Three subsequent uninstrumented device-append runs measured **0.921–0.989 s prefill**
and **47.2–48.5 ms/token**; the initial uninstrumented host-append baseline was
**1.601 s** and **83.0–83.2 ms/token**. Loading varied from 2.28 to 4.25 s and is not
part of these execution intervals. These are single-sequence, short-prompt diagnostics,
not mixed-arrival serving, long-context quality or competitive acceptance. Prefill
still executes one token at a time with per-operation synchronization.

```sh
cargo test --release -p engine-nvidia --features cuda --locked \
  --test cuda_bf16_copy -- --ignored --test-threads=1
# Set RIBN_GEMMA_PACKAGE and RIBN_GEMMA_REFERENCE as above.
cargo test --release -p engine-gemma4 --features cuda --locked \
  --test cuda_decoder --test executor_cuda --test text_path_cuda \
  -- --ignored --test-threads=1
# Run the built copy/decoder test executables under compute-sanitizer:
# compute-sanitizer --tool memcheck --error-exitcode 86 <test-executable> \
#   --ignored --test-threads=1
```

## CUDA prompt-head and upload cleanup

The cleanup at `2ae340f` (2026-10-20) removes host-orchestration waste without changing
kernels or physical batching. A temporary timer at `3c60043` measured about **16 ms**
per token for embedding uploads (including an unused device PLE identity), **13 ms**
for all per-layer PLE-context uploads, and **3.1 ms** for final normalization, the tied
vocabulary head, logit readback and host soft cap. Uploads used five-byte staging and
drained after each fragment.

The cleanup removes the unused device PLE-identity allocation/upload, stages each
actual embedding/PLE-context upload as one complete row, and computes the output head
only for the last token of each accepted prefill chunk. Other tokens still execute all
layers and cache writes in the same order. Device-only matrices keep minimal staging;
readback buffers retain their explicit capacities. This adds neither overlap nor batched
prefill and makes no whole-loader/physical-peak claim.

Before device work, prefill now validates the entire chunk's token/media semantics and
reachable capacity. A regression test fails on the original code: a valid token followed
by an out-of-vocabulary token leaves position 19 instead of 18 after rejection. Revised
code preserves the old position and continuation. Possible device failure still faults
the shared model and retains its retirement owner; validation is not device rollback.

### Matched sequential diagnostic

Same RTX 4090, driver 615.71.10, CUDA 13.3, pinned BF16 package and archived chat token IDs.
A temporary probe repeats those IDs to lengths 1/18/64/129, warms three input tokens, then
runs each length three times with reset and three common teacher-forced continuation
tokens. Cache capacity is 160 and attention scratch is `160 × attention_heads`. Timings
exclude loading and evidence-file comparison. Variants run sequentially in baseline,
upload-cleanup and combined order, not alternating order; these are bounded diagnostics,
not mixed-arrival serving, long-context quality or competitive acceptance.

Median prefill wall time (ms):

| Prompt tokens | Device-append baseline | Upload cleanup only | Upload + last-token head |
| --- | ---: | ---: | ---: |
| 1 | 46.87 | 17.88 | 17.99 |
| 18 | 845.88 | 324.87 | 274.76 |
| 64 | 3058.09 | 1200.12 | 1014.12 |
| 129 | 6208.06 | 2468.54 | 2091.13 |

Baseline prefill ranges were 46.80–47.68, 845.73–846.01, 3054.09–3060.01 and
6204.74–6209.18 ms respectively; combined ranges were 17.99–18.27, 274.56–277.31,
1013.32–1014.47 and 2090.82–2091.60 ms. At the 18-token history, median decode fell
**47.46 → 18.51 ms/token**; across all four histories combined decode medians were
18.06–19.90 ms/token. Upload-only versus combined decode differed by less than 1%;
last-token head selection benefits prefill, not single-token decode.

Every variant/repetition checked the shape, finiteness and exact BF16 encodings of all
262,144 logits after prefill and each continuation step against the baseline. This is
transformation equivalence, not an independent model-quality reference. The committed
`cuda_decoder` gate retains the archived four-token check and adds exact full-logit
comparisons for one-token and seven-token prompt chunks, continued decoding and reset,
plus rejection without prefix mutation. Decoder, executor and text gates passed; the
expanded full decoder test passed memcheck with **zero errors**. Host boundaries, fmt,
387 workspace tests and default/CUDA/targeted Metal clippy passed. Failed-drain hardware
injection and Metal device reruns were not performed for this CUDA-only slice.

The existing `cuda_decoder` command above reproduces the short archived gate and prints
cold execution timings; it does not reproduce the warmed length sweep. Raw probe code,
source variants and logit captures are retained with qualification artifacts, not as a
new public benchmark framework. Next is a separately qualified batched-prefill lane,
including window/chunk boundaries and common-history numerical acceptance.

## Host multi-row prefill oracle

```sh
cargo test -p engine-gemma4 --locked --test cpu_prefill
```

The CPU decoder already executes multiple input rows layer-by-layer. This gate uses
the independent reduced package schema with deterministic, nonuniform BF16 weights:
two different head widths, local/full KV owners and two shared-tail consumers. It
compares complete, finite logit rows with single-token execution at every submitted
chunk boundary and on three common teacher-forced continuation tokens. Windows 4 and
512 exercise chunks ending before, at and after the window boundary, chunks larger
than the window, nonzero prefixes and partial tails. Reset and whole-chunk rejection
must preserve subsequent results. The fixture distinguishes equal-last-token histories;
constant-valued source/materialization fixtures alone cannot protect these dependencies.

Acceptance is bit equality because this CPU path preserves each row's arithmetic and
visible-key accumulation order across chunk shapes. A temporary mutation that applied
the last query's mask to every row made the gate fail; restoring per-query masks passed.
This is an execution-transformation oracle, not an independent check of model semantics,
physical donor retirement or GPU numerical accuracy. The CPU path deliberately retains
complete owner histories. CUDA batching can keep that baseline without also introducing
eviction; reordered device math still needs independently justified numerical bounds,
common-history reference comparisons and device lifetime/memory qualification.

## Host rotary-frequency gate

`Gemma4Config::rotary_inverse_frequency` prepares individual FP32 factors without
allocating a table or touching a device. The committed small fixture comes from the
pinned Transformers rotary module, using the checkpoint config and a second geometry
with different widths/thetas. It loads **no checkpoint weights**; only config bytes,
package versions and the seven pinned implementation files are verified.

```sh
# Use the reference environment pinned in model.toml; output must not exist.
python benchmarks/gemma4_rotary_reference.py --output /new/path/rotary.json
cargo test -p engine-gemma4 --test rotary --locked
```

For E4B, local attention has 128 active frequency pairs. Global attention has 64
active pairs followed by **192 positive-zero entries**. Its exponents use the full
512-channel head; cosine/sine repeat the 256-entry vector across both halves, so
channel `j` pairs with `j + 256`. Ordinary partial rotation over a 128-channel prefix
is not equivalent. The host test fails when that shorter exponent denominator is
substituted, and passes after restoring the full-head formula.

Acceptance is exact shape, first frequency and padding, finite positive active
values, at most four FP32 ULPs from the pinned CPU reference, and relative error at
most three FP32 epsilons against an independent FP64 equation using the prescribed
FP32 exponent. These are scalar frequency budgets across power/reciprocal
implementations, **not tolerances for accumulated phase, rotated tensors or logits**.
Layer/pair bounds and unrepresentable FP32 factors reject. Device rotation, large-position
phase accuracy and whole-model numerical acceptance remain separate pending gates.

## Host continuation-range gate

`Gemma4Config::kv_step` calculates absolute half-open old/current/retained ranges
without claiming a valid continuation, granting memory or committing execution.
Model configuration resolves owner routing and final consumers once: local donor
22 is read through layer 40, global donor 23 through layer 41. Separate retained-span
and attention-span types prevent substituting a trimmed suffix for a donor input.

```sh
cargo run --release -p engine-gemma4 --example kv_ranges -- \
  benchmarks/gemma4-reference/config.json 511 513 4096
```

For that chunk, the local attention/donor span is **[0,1024)**, not merely 512 positions;
the post-step retained suffix is **[513,1024)**. Per-query causal/window masks still
restrict each query. Tests include single-token and larger-than-window chunks,
partition-independent final coverage (not arithmetic identity), window one, invalid
progress/reach and byte overflow.

The probe matched all **24 prediction rows** in the existing FP32/BF16 CPU records:
**576 persistent-owner shapes and 48 donor-span shapes**, using their exact submitted
positions and lengths. This validates range interpretation, not tensor contents,
physical state validity or asynchronous lifetime safety.

Logical BF16 K+V payload is
`40960 * min(prefix,511) + 16384 * prefix` bytes across the 24 owners: **88,039,424 bytes**
at prefix 4096. Donor spans may share backing with persistent state; do not add their
logical bytes blindly or call either formula a GPU allocation/peak bound. Device
storage, append/trim layout, staging/workspace and uncertain retirement remain unqualified.
The [owning contract](../../docs/resource-protocol.md#mixed-retention-and-shared-kv)
defines preparation, last-consumer retention, commitment and failure scope.

## Canonical text-processor gate

The model-owned processor loads `tokenizer.json`, `tokenizer_config.json`,
`chat_template.jinja` and `generation_config.json` through the local package.
It uses the package's retained config, not a second config open. Assets must remain
unchanged while loading; afterward the processor owns them in memory. Alternate
`chat_templates/` directories reject rather than override the selected template.

The frozen [canonical template](chat_template.jinja) matches the publisher checksum
in `model.toml`. The [processor corpus](processor-cases.json) was collected with the
pinned Transformers/tokenizers versions and records the Jinja version. It covers
12 raw inputs, nine chat histories and 270 decode cases, including all 256 individual
fallback bytes, malformed/truncated runs, whitespace, Unicode, literal special tokens,
system/developer roles, consecutive assistant turns, channel stripping and Python's
extra control-whitespace trimming.

```sh
cargo run --release -p engine-gemma4 --example processor -- \
  /path/to/pinned/snapshot benchmarks/gemma4-reference/processor-cases.json
```

**All 291 cases passed**, using Rust `tokenizers 0.23.2` (pure-Rust regex, no Hub/network
features) against Python `tokenizers 0.23.3`. Raw prompts add no BOS/EOS; chat uses the
actual non-thinking template with its BOS and generation prompt. Special markers are
preserved on decode, without whitespace cleanup. Stops come from generation metadata:
**1, 106, 50**, not only the model config's EOS. The chat prompt and first 511/513 raw
prefix IDs also match the numerical reference inputs above.

A request-local decoder retains each consecutive fallback run until a non-fallback
token or terminal finalization. This is necessary for exact HF behavior:
`<0x61><0xFF>` decodes to two replacement characters, not `a�`. Emitting the valid
prefix early would require retracting already-delivered text. Pending bytes and tokens
are bounded; errors poison the decoder, and consuming finalization flushes at most once.
`ribn-text::GemmaProcessor` now adapts this model-local API to owned text delivery.
Its decoder shares immutable vocabulary and has explicit output-token, pending-byte,
vocabulary-piece and delta-byte limits independent of chat/input limits. Preparation
couples decoder state to the existing request permit before submission; the charge
survives abandoned replies, acknowledgement waits and fallible finalization. Host
fixtures exercise the real codec/facade with scripted token execution, including
interleaved runs, stop/cancel flushes, owner failure without flush, and request-local
push/finalization errors. Barrier tests protect permit retention through finalization
and abandoned-reply cleanup. None of this implements or qualifies Gemma model execution.

Input bytes/message count, rendered/decoded bytes and token counts have per-call bounds.
Rendering has a fuel limit; tokenizer assets are limited to 64 MiB and other processor
assets to 1 MiB each. These do not bound Jinja/tokenizer intermediate allocations, process RSS or aggregate
frontend retention.
The three root media-placeholder IDs reject after encoding and in direct-token input;
the conditional reference masks them to PAD before main/PLE embedding, so ordinary
text embedding would be incorrect. Other media boundary markers are not blanket-rejected.

To audit the corpus independently, load the pinned tokenizer with
`AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)`.
For each raw row compare `encode(text, add_special_tokens=False)` and
`decode(ids, skip_special_tokens=False)`; for each chat row compare
`apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
enable_thinking=False)` and encode the result without added specials. Decode rows
compare the same whole-sequence decode. No weights or model execution are needed.

Primary sources:

- [Pinned model config](https://huggingface.co/google/gemma-4-E4B-it/blob/ee0ef6023621cff504d758262d4e04895a5af4a2/config.json)
- [Pinned checkpoint files](https://huggingface.co/google/gemma-4-E4B-it/tree/ee0ef6023621cff504d758262d4e04895a5af4a2)
- [Reference model code](https://github.com/huggingface/transformers/blob/48e4bcd72362ff4887651112dbfa8227211a393a/src/transformers/models/gemma4/modeling_gemma4.py)
- [Reference cache behavior](https://github.com/huggingface/transformers/blob/48e4bcd72362ff4887651112dbfa8227211a393a/src/transformers/cache_utils.py)
