# Ribn

A Rust-first, server-first model inference engine.

Ribn targets competitive throughput, latency, memory efficiency and serving reliability,
with broad support for common models and hardware. The current implementation is an
experimental Qwen GGUF/CUDA autoregressive path, an experimental Gemma 4 SafeTensors
decoder on host CPU and Apple Metal, and a reference-qualified BERT encoder fixture. Those are qualification workloads, not the product boundary. There is no
general model loader or claim of state-of-the-art performance yet. The initial HTTP
adapter exposes a bounded loopback completion/chat subset.

## Usage

Use the Rust toolchain pinned in `rust-toolchain.toml`. Metadata inspection needs no GPU:

```sh
cargo run -p ribn-cli -- inspect /path/model.gguf
```

The experimental NVIDIA path generates text from a Qwen GGUF artifact. Input is a user
chat message by default; `--raw` selects plain completion. The model can be positional
or supplied with `--model`.

```sh
cargo run -p ribn-cli --features cuda -- run /path/model.gguf \
  --prompt 'Explain a mutex.' --max-tokens 128

cargo run -p ribn-cli --features cuda -- run /path/model.gguf \
  --raw --prompt 'The answer is'
```

`--file` and piped stdin are supported. Input is checked before model loading: up to
256 KiB of UTF-8 for raw completion, or 256 KiB minus the four-byte `user` role for chat.
Oversized input rejects rather than truncates; file/stdin reads use a one-byte overflow
probe. Interactive terminal chat is not implemented. The obsolete `local` command is
removed; `run` is the shared text entry point.

A Gemma 4 SafeTensors package runs through the same text layer on the Apple GPU
(macOS, experimental, greedy decoding). The positional argument is the package
directory rather than a single file:

```sh
cargo run -p ribn-cli --features metal -- run-gemma /path/to/gemma-package \
  --prompt 'Explain a mutex.' --max-tokens 128
```

The same owned text path also serves a loopback HTTP endpoint:

```sh
cargo run -p ribn-cli --features cuda -- serve /path/model.gguf
```

See [serving](docs/serving.md) for the supported protocol, greedy decoding, limits,
health/metrics and shutdown. Non-local exposure is unsupported.

## Rust interfaces

- `ribn` is the low-level AR token runtime. Direct `Engine` use is channel-free;
  `ribn::driver` provides one execution worker, cloneable generation handles, bounded
  owned streams, cancellation and explicit shutdown/retry. It accepts encoded input.
- `ribn-text` adds raw/chat/token input, bounded preprocessing, incremental UTF-8
  decoding and ordered bounded-window offline batching. `TextModel` is cloneable;
  `TextOwner` owns shutdown. CUDA assembly still selects Qwen GGUF.
- `engine-gemma4` is the Gemma 4 second decoder: package preflight, a host CPU BF16
  decoder and a synchronous greedy executor; `engine-metal` supplies its macOS Metal
  primitives. Experimental and greedy-only, not a competitive backend.
- `ribn-server` is the bounded HTTP adapter over the text facade; it owns transport
  tasks and retains the text shutdown owner for cleanup retry, not a second scheduler.
- `ribn-batch` is the non-AR batching runtime used by the BERT encoder. Its results
  retain owning byte leases and completion dependencies, not fake token sequences.
- `ribn-hf` and `ribn-safetensors` resolve local packages and validated artifact views;
  model-specific interpretation stays in model code. Remote HF loading is not implemented.

The application API is intended to grow operation-specific generation, embedding,
scoring, speech and media capabilities above specialized runtimes. It will not force
every model into token-generation contracts or require a stage graph for a single model.
See [current source owners](docs/architecture.md) for the crate map and limitations.

## Development

```sh
python3 tools/check-boundaries.py
cargo fmt --all -- --check
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo clippy --workspace --all-targets --locked --features cuda -- -D warnings
```

CUDA-feature compilation can run without a GPU. Numerical and device-lifecycle
qualification requires compatible NVIDIA hardware; ignored device tests are not
exercised by the host suite.

## Documentation

1. [Target design](docs/inference-engine-design.md) — scope, architecture and API direction.
2. [Execution contracts](docs/resource-protocol.md) — ownership, preparation, bounds and readiness.
3. [Roadmap](docs/roadmap.md) — sequencing, unresolved choices and acceptance gates.

Supporting references:

- [Current implementation](docs/architecture.md)
- [Foundation evidence](docs/execution-foundation.md)
- [Composition counterexamples](docs/pipeline-composition.md)
- [Runtime qualification](benchmarks/runtime-contract.md)
- [Benchmarks](benchmarks/README.md)
- [CUDA Rust migration](docs/cuda-rust-migration.md)

## License

[Apache-2.0](LICENSE)
