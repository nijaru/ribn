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
