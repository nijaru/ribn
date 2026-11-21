# Maintaining Ribn

Read the owner for the change at hand:

| Question | Owner |
| --- | --- |
| Where is the implementation? | [Source owners](architecture.md#source-owners) and [current paths](architecture.md) |
| What architecture and product trade-offs are intentional? | [Target design](inference-engine-design.md) |
| Who owns admission, preparation, completion, memory and shutdown? | [Resource protocol](resource-protocol.md) |
| What should ship next, and what evidence would change course? | [Roadmap](roadmap.md#current-execution-order) |
| What does the local endpoint accept? | [Serving](serving.md) |
| How do I collect and audit evidence? | [Benchmark methodology](../benchmarks/README.md), [runtime/device gates](../benchmarks/runtime-contract.md) |
| Is CUDA Rust a qualified replacement? | [Migration gates](cuda-rust-migration.md) |

[Foundation](execution-foundation.md) and [composition](pipeline-composition.md)
are counterexamples, not API specifications. Keep qualification data tied to its
artifact, workload, source revision and device; the roadmap records decisions,
not a second copy of experiment timelines.

## Verification

CI commands and the supported Python matrix are in
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml). For Rust changes, run the
affected tests during iteration and the host gate before delivery:

```sh
python3 tools/check-boundaries.py
cargo fmt --all -- --check
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
```

For CUDA assembly, backend or CLI changes, also run:

```sh
cargo clippy --workspace --all-targets --locked --features cuda -- -D warnings
```

For Metal decoder, text assembly or CLI changes on macOS:

```sh
cargo clippy -p engine-gemma4 -p ribn-text -p ribn-cli --all-targets --locked --features metal -- -D warnings
RIBN_GEMMA_PACKAGE=<package> RIBN_GEMMA_REFERENCE=<reference.json> \
  cargo test -p engine-gemma4 --features metal --release --test metal_decoder -- --ignored --test-threads=1
```

The ignored Metal gate is release-only; a missing device is a permitted skip,
a kernel or executor failure is not. For other affected device paths, use the
[runtime qualification commands](../benchmarks/runtime-contract.md). Host tests and
feature compilation are not numerical or lifecycle device evidence.

For benchmark changes, use the [Python environment](../benchmarks/README.md#python-environments)
and run `python3 -m unittest discover -s benchmarks -p 'test_*.py'`. For docs-only
changes, check affected links, commands and `git diff --check`; rerun runtime gates
only if their behavior or procedure changed.
