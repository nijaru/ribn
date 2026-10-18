# Gemma 4 E4B independent reference

This prepares the numerical reference for Ribn's second decoder integration:
**Gemma 4 E4B instruction-tuned, BF16 SafeTensors, text-first**. It does not implement
Gemma execution in Ribn or qualify image/audio input, GPU memory fit or serving quality.
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

These checks validate reference collection and its instrumentation, not numerical
agreement with an unimplemented Ribn decoder. No image/audio, device-lifecycle, GPU
memory-peak or serving-performance gates ran.

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
GPU-fit qualification. Capacity planning provisionally reserves roughly 0.5 GiB for
existing qualification-GPU use; replace that estimate with measured free VRAM rather
than treating it as a fixed OS charge. In particular, converting the active text weights wholesale
to FP32 exceeds 24 GiB before any KV or scratch allocation.

Ribn's file-backed SafeTensors path now uses retained metadata/files and caller-buffered
payload readers. The [host I/O gate](../loader-memory/README.md) checks the full artifact
and every tensor's encoded bytes without loading the checkpoint into application heap.
This does not implement Gemma: native BF16 device preparation, tied allocation ownership,
mixed-window KV ownership and donor-span lifetimes remain prerequisites for execution.

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

Primary sources:

- [Pinned model config](https://huggingface.co/google/gemma-4-E4B-it/blob/ee0ef6023621cff504d758262d4e04895a5af4a2/config.json)
- [Pinned checkpoint files](https://huggingface.co/google/gemma-4-E4B-it/tree/ee0ef6023621cff504d758262d4e04895a5af4a2)
- [Reference model code](https://github.com/huggingface/transformers/blob/48e4bcd72362ff4887651112dbfa8227211a393a/src/transformers/models/gemma4/modeling_gemma4.py)
- [Reference cache behavior](https://github.com/huggingface/transformers/blob/48e4bcd72362ff4887651112dbfa8227211a393a/src/transformers/cache_utils.py)
