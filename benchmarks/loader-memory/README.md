# Host loading memory

## Streaming encoded weights (2026-10-18)

File-backed SafeTensors artifacts now retain a validated header/index and one opened
file, not the entire encoded payload. Owning readers use caller-provided buffers and
survive cache eviction, artifact drop and pathname replacement. In-memory artifacts
still explicitly retain their supplied bytes. No unsafe mapping or GPU work is involved.

### Matched BERT comparison

Measured in release mode on Apple M3 Max, 128-GiB host RAM, macOS 26.6.2 (25G83),
Rust 1.98.0. The baseline is `527bebf`; the candidate uses streamed artifact readers
and a fixed 16-KiB F32 decode buffer feeding each final BERT parameter owner. Both load
and retain the same 437,950,960-byte, four-shard package described below. Three
alternating baseline/candidate pairs ran directly under `/usr/bin/time -l`; input
creation was outside measurement and filesystem cache state was not controlled.
[Raw CSV](2026-10-18-macos.csv) retains every measured run and internal load time.

| Median peak process RSS | Baseline | Streamed |
| --- | ---: | ---: |
| Bytes | 997,081,088 | 441,679,872 |
| MiB | 950.89 | 421.22 |

Peak process RSS was **55.7% lower** for this loading workload. All six runs returned
109,482,240 parameters and the same F32-bit digest `4ae09df98cbc9b25`. Timing is not a
serving-throughput result or a general loading-speed claim. The expanded sharded host
test also checks a full decode-buffer boundary, partial tail, signed zero and subnormal
bits; the tiny original fixture did not exercise the new chunk loop's tail.

```sh
python3 benchmarks/loader-memory/generate.py /tmp/ribn-loader-fixture
cargo build --release -p engine-bert --example bert_loading_bench --locked
/usr/bin/time -l target/release/examples/bert_loading_bench /tmp/ribn-loader-fixture
```

Build the unchanged probe in a detached `527bebf` worktree for the matched baseline;
run both binaries against the same generated directory.

### Real E4B artifact gate

The ordinary HF resolver and `weight_stream` probe read the pinned
[Gemma 4 E4B package](../gemma4-reference/model.toml): **15,992,595,884 source bytes**,
with a single **1-MiB caller buffer** and **9,306,112 bytes (8.88 MiB) peak process RSS**.
It hashes the full artifact and then every tensor: these are two payload passes, not
model execution. The full SHA256 matched the pin. An independent Python header/range
reader checked all **2,130** per-tensor SHA256 values and their BF16 shapes/dtypes
against the previously captured stock-reference catalog. The probe reported zero
file-payload bytes resident in application heap; metadata, buffer and report storage
are included in the measured process RSS.

```sh
cargo build --release -p ribn-hf --example weight_stream --locked
/usr/bin/time -l target/release/examples/weight_stream /path/to/pinned/snapshot \
  model.language_model.embed_tokens.weight > /tmp/gemma4-stream.json
```

The real package first exposed a pre-existing resolver rejection: Hub 2.2's repository
blob links lead into a version-marked, cache-wide shared store. Support now accepts only
that v1 store's canonical, well-formed blob files, not arbitrary hub paths. ETag/LFS and
Xet identifiers need not match. This is path confinement, not repository membership,
content authentication or tenant isolation. The new single/sharded shared-cache test
failed before the fix; legacy confinement and malformed/escaping shared-store cases pass.
See the [provider's layout](https://huggingface.co/docs/huggingface_hub/v2.2.0/en/guides/manage-cache#shared-blobs-across-repos).

These are encoded-artifact I/O and host-decoding gates, **not Gemma execution, GPU fit,
BF16 device materialization or serving-quality evidence**. No device tests ran.

## Historical decoded-cache repair (2026-10-06)

Measured on Apple M3 Max, macOS 26.6.2 (25G83), Rust 1.98.0, release mode,
2026-10-06. The baseline is `1dadac1`; candidate `dd10c72` removes BERT's duplicate decoded
parameter cache, retains one source shard and evicts old cache entries before opening
incoming shards. No CUDA execution or upload runs.

The synthetic F32 package has 109,482,240 parameters: hidden width 768, 12 layers,
12 heads, intermediate width 3072, vocabulary 30,522 and 512 position embeddings.
Four shards contain embeddings, layers 0–3, layers 4–7 and layers 8–11 plus pooler.
Their files total 437,950,960 bytes. Values are deterministic constants per parameter;
this is a loading-memory workload, not a numerical model reference.

Three baseline processes preceded three candidate processes. Inputs were generated
before measurement; filesystem cache state was not controlled. `/usr/bin/time -l`
measured each process directly, not Cargo. [Raw CSV](2026-10-06-macos.csv) records
peak process RSS and the loader's internal elapsed time.

| Median peak process RSS | Baseline | Candidate |
| --- | ---: | ---: |
| Bytes | 1,549,664,256 | 997,031,936 |
| MiB | 1477.88 | 950.84 |

The candidate's median peak was 35.7% lower in this workload. Every run produced
parameter count 109,482,240 and digest `4ae09df98cbc9b25` over all decoded F32 bits.
The measured process includes a post-load digest pass and retains the returned host
weights. Neither timing nor peak RSS establishes a general speedup, a hard byte bound,
large-model support or GPU/serving performance.

### Historical reproduction

Generate the fixture once, then build/run the probe from each historical worktree
listed below, not from current main:

```sh
python3 benchmarks/loader-memory/generate.py /tmp/ribn-loader-fixture
cargo build --release -p engine-bert --example bert_loading_bench --locked
/usr/bin/time -l target/release/examples/bert_loading_bench /tmp/ribn-loader-fixture
```

For this historical comparison, use detached `1dadac1` and `dd10c72` worktrees rather
than current main. Copy the probe into the baseline, build both and use the same fixture.
The FNV-1a digest checks decoded bits, not cryptographic artifact identity.

## Current limits

- Each decoded parameter remains in the complete `HostWeights` owner. Device upload
  is still a separate operation over that owner, not incremental streaming.
- File-backed source payloads are no longer retained in heap, but metadata/header
  parsing still allocates (with the upstream 100,000,000-byte header ceiling).
- Source-cache counts exclude metadata byte bounds, caller-held descriptors/readers,
  read buffers, allocator overhead and OS page cache. No whole-loader byte authority
  or process-wide bound follows from removing encoded payload copies.
- Files must remain unmodified through all aliases while readers/artifacts live.
  Config/index and unopened package members must remain unchanged during loading;
  descriptor retention does not make the whole package an atomic snapshot.
- Interleaved shards can require reopening/revalidating headers under one-source
  residency. Owning readers remain valid across eviction; workload-specific costs
  and decoder/upload concurrency still need separate budgets and measurements.
