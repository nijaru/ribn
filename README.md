# Ribn

Ribn is an early Rust inference engine for serving models. The experimental text
paths are Qwen GGUF on CUDA and Gemma 4 SafeTensors packages on Metal or CUDA.
These are specific model paths, not a general GGUF or SafeTensors loader.

## Try it

Use the Rust toolchain pinned in `rust-toolchain.toml`. Inspect a GGUF artifact
without a GPU:

```sh
cargo run -p ribn-cli -- inspect /path/model.gguf
```

Generate text from a supported Qwen artifact:

```sh
cargo run --release -p ribn-cli --features cuda -- run /path/qwen.gguf \
  --prompt 'Explain a mutex.' --max-tokens 128
```

For a supported Gemma 4 package on macOS, use
`--features metal -- run-gemma /path/to/gemma-package` with the same prompt flags.
Use `--features cuda` for NVIDIA. `--raw` selects plain completion instead of chat;
`--file` and piped stdin also work. Input over the 256 KiB envelope (including the
role in chat mode) is rejected, not truncated.

See [serving](docs/serving.md) for the loopback HTTP quickstart and supported
completion/chat subset. The server is greedy-only and not for public exposure.

[Maintainer guide](docs/README.md) · [Apache-2.0](LICENSE)
