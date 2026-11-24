# Inkling 2 Mini reference replay

`engine-inkling2` now implements model-local config, 704-tensor GGUF preflight,
logical 48-slot continuation accounting, a pure-Rust CPU reference forward, and
a greedy `ribn::GenerationExecutor` (host-verified on synthetic mixed-encoding
GGUFs plus exact 1,036-token oracle tokenization and real chat-template rendering);
full-checkpoint GGUF numerical qualification against this oracle and GPU serving
remain pending. This oracle pins the text execution of
`ggml-org/Inkling-2-Mini-GGUF` snapshot `db61c5b492a6a034d986328d28274af0a439aa6c`,
file `Inkling-2-Mini-IQ4_NL.gguf`, SHA-256
`cbdf3fa9dc50a4124977980f37da4f1fa764b0cfe800d275cd43d26ce2a81af4`.
Use llama.cpp `e6d624b9a4b95c6b52f5ca384d70b75581f9821a` built with CUDA
and shared libraries. The oracle has no runtime dependency edge into Ribn.

```sh
# LLAMA points at the pinned source; MODEL is the exact local artifact above.
c++ -std=c++17 -O2 -Wall -Wextra -Werror -I"$LLAMA/include" -I"$LLAMA/ggml/include" \
  benchmarks/inkling-reference/replay.cpp -L"$LLAMA/build/bin" \
  -Wl,-rpath,"$LLAMA/build/bin" -lllama -o /tmp/inkling-replay
/tmp/inkling-replay "$MODEL" /tmp/inkling-logits  # output directory must not exist
```

The runner stages one model on the RTX 4090, uses fresh contexts for widths
64 and 127, and keeps fixed token histories independent of predicted tokens.
Context length is 2048, KV is F16, and Flash Attention is enabled. It records
token IDs and full 200,058-element F32 logit rows (native little endian on the
qualified host), including checkpoints around the 1024-token sliding window
and four single-token continuation steps. The fixture is for numerical replay,
not a task-quality or throughput benchmark.

On 2026-11-24 both schedules produced finite rows and identical argmax at all
ten checkpoints. The first two histories (8 and 32 tokens) use identical batch
shapes and matched bit for bit. At later checkpoints centered logit RMSE between
schedules was 0.209–0.299, and maximum absolute element difference reached 1.828.
These are reference observations, not acceptance tolerances. Match the exact
chunk schedule and cache precision first; use layer taps to localize drift
before attributing differences to chunk arithmetic or state. Raw qualification:
`benchmarks/results/qualification/inkling-oracle-e6d624b/`.

## Native port boundary

The checkpoint has 24 physical layers with core `[6,18)` run three times:
48 attention and convolution state slots, but shared core weights. Each pass
starts from the prelude output; the previous pass's depth residual bank feeds
learned Q/K/V and FFN controls. That bank is forward scratch, whereas each
logical slot's KV and four-tap causal convolution history survive into the next
token. Keep the mapping in the model, not in request scheduling. The artifact
has 478 F32, 186 Q8_0 and 40 IQ4_NL tensors. Attention uses learned relative bias
and a sink, not RoPE. The FFN routes eight of 128 experts with sigmoid scoring
and SiTU. The input uses the Inkling GPT-2-style pretokenizer. Do not substitute
Qwen's attention, routing, or tokenizer.

Before exposing support, qualify a model-local forward against this reference
and targeted layer/state fixtures, then test continuation and cancellation via
Ribn's HTTP path. A first-class serving claim also needs matched quality and
latency/goodput against the same artifact and cache policy on representative
traffic. A loader, an external-only replay, or token agreement on this fixture
alone is not that result.
