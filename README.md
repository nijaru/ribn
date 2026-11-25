# Ribn

Ribn is an early Rust inference engine for serving models. The experimental text
paths are Qwen GGUF on CUDA and Gemma 4 SafeTensors packages on Metal or CUDA.
These are specific model paths, not a general GGUF or SafeTensors loader.

## Try it

Use the Rust toolchain pinned in `rust-toolchain.toml`. If you already have a
GGUF file, inspect its metadata without a GPU. This does not check whether
Ribn can run the model.

```sh
cargo run --locked -p ribn-cli -- inspect /path/model.gguf
```

### Qwen on NVIDIA CUDA

Use `Qwen3.8-27B-UD-Q4_K_M.gguf` from `unsloth/Qwen3.8-27B-GGUF` (16.46 GB).
The [device qualification](benchmarks/runtime-contract.md#device-re-execution-2026-10-16-dc4aed0)
records the artifact hash and tested RTX 4090 setup. You need a CUDA-capable
NVIDIA GPU with enough memory for the model and request state.

```sh
cargo run --locked --release -p ribn-cli --features cuda -- \
  run /path/Qwen3.8-27B-UD-Q4_K_M.gguf --prompt 'Explain a mutex.' --max-tokens 128
```

### Gemma 4 on Apple silicon or NVIDIA CUDA

Use the `google/gemma-4-E4B-it` BF16 SafeTensors snapshot specified in the
[artifact manifest](benchmarks/gemma4-reference/model.toml). Pass the local
snapshot directory, not the individual weights file. The BF16 weights file
alone is about 16 GB; allow additional memory for execution.

```sh
cargo run --locked --release -p ribn-cli --features metal -- \
  run-gemma /path/to/gemma-4-E4B-it --prompt 'Explain a mutex.' --max-tokens 128
```

Use `--features cuda` instead of `--features metal` on NVIDIA. `--raw` selects
plain completion instead of chat; `--file` and piped stdin also work. Input over
the 256 KiB envelope (including the role in chat mode) is rejected, not truncated.

See [serving](docs/serving.md) for the loopback HTTP quickstart and supported
completion/chat subset. Sampling support depends on the executor. The server
is not for public exposure.

[Maintainer guide](docs/README.md) · [Apache-2.0](LICENSE)
