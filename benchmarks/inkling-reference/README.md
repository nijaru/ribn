# Inkling 2 Mini reference replay

`engine-inkling2` now implements model-local config, 704-tensor GGUF preflight,
logical 48-slot continuation accounting, a pure-Rust CPU reference forward, and
a greedy `ribn::GenerationExecutor` (host-verified on synthetic mixed-encoding
GGUFs plus exact 1,036-token oracle tokenization and real chat-template rendering);
the full-checkpoint GGUF numerical gate below did not pass. GPU serving remains
unqualified. This oracle pins the text execution of
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

## Rust CPU numerical gate (2026-11-24)

On Fedora at `fbb8c9a`, the real-artifact GGUF preflight found 704 tensors and
48 state slots and matched all 1,036 oracle token IDs. The CPU forward used F16
KV and the width-64 checkpoint schedule from `replay.cpp`. The 8- and 32-token
checkpoints passed (centered logit RMSE 0.13521 and 0.22549). At 1,023 tokens,
argmax still matched (368), but centered RMSE was 0.40711 against the existing
`< 0.25` bound; max absolute logit error was 2.17214. The test stopped there,
so the 1,024-token boundary and later checkpoints remain untested. Do not
raise the bound on this evidence. Compare matched-history layer/state taps
before attributing the drift to a specific operation.

A follow-up compared last-token layer rows at lengths 8 and 32 against the
same pinned llama.cpp revision on **CPU**, using the same tokens and F16 KV.
All 48 attention-output and layer-output rows per checkpoint were finite and
shape-matched. At 8 tokens, relative L2 was already 0.48% at layer 0's attention
output, then reached 20.73% at the attention output of logical layer 32 on the
second core pass. At 32 tokens, it was 0.56% at layer 0's attention output and
17.05% at logical layer 44's residual output. These CPU-to-CPU taps show
short-history drift before the window boundary; they do not separate
quantized projection arithmetic from model semantics or explain the failed
1,023-token CUDA-oracle comparison.

A disposable CPU intervention rounded only the inputs of `IQ4_NL` and `Q8_0`
matvecs to decoded Q8_0 (32 values per block, F16 scale), while retaining the
existing Rust accumulation. Compared with the same llama.cpp CPU taps, layer 0
attention relative L2 fell from 0.480% to 0.141% at 8 tokens and from 0.555% to
0.054% at 32 tokens. Centered logit RMSE fell from 0.17692 to 0.13736 and from
0.22287 to 0.15173. Activation rounding accounts for part of the observed drift,
not all of it. This is not a bit-exact substitute for GGML's integer dot or a
production change.

With the same intervention, the long replay against the width-64 CUDA oracle
improved from 0.40711 to 0.23070 centered logit RMSE at 1,023 tokens and passed
at 1,024 (0.18948). It failed at the first token that drops the oldest key,
1,025 (0.38878 vs `< 0.25`); all three argmaxes matched. The Rust window-start
expression agrees with the pinned llama.cpp standard mask (`q - k >= window`
excludes a key), so this does not establish an off-by-one error. A synthetic
F16 cache test independently perturbs the oldest stored value in each logical
slot: it affects attention on both layer types at the last in-window query,
then only on full-attention slots at the next query. This checks Rust mask
behavior, not real-artifact parity. The two stock
CUDA schedules themselves differ by 0.20854 at 1,025; that is context for
investigation, not a new tolerance. Capture matched operands and attention
state around the transition before changing runtime arithmetic. Raw rows and
disposable probes:
`benchmarks/results/qualification/inkling-layer-taps/` (local).

```sh
RIBN_INKLING_GGUF="$MODEL" RIBN_INKLING_ORACLE=/tmp/inkling-logits \
  cargo test -p engine-inkling2 --locked --release --test real_artifact \
  cpu_forward_matches_all_ten_pinned_oracle_checkpoints_across_swa_boundary \
  -- --ignored --exact --nocapture --test-threads=1
```

### CUDA boundary capture (2026-11-25)

The pinned CUDA replay captured the last-token Q/K/V and relative-query rows,
attention outputs and residual outputs for all 48 logical layers at 1,023,
1,024 and 1,025 tokens. The instrumented run's logits were bit-identical to the
archived width-64 reference at all five checkpoints through 1,025, and its
1,025 input token IDs matched the original history. This validates the tapped
reference, not Rust parity. The taps contain current-token projections before
cache storage, not historical KV rows. Keep that distinction when localizing
any difference. The matching Rust CPU capture with the disposable Q8_0
activation intervention showed layer-0 attention relative L2 of 0.108%, 0.099%
and 0.116%, respectively. The largest layer-output relative L2 was 16.62%,
15.37% and 86.71%, respectively; at 1,025 it occurred in logical layer 32
(physical layer 14, second core pass). The increase is already present by
that pass, not confined to the head or layer-0 mask. These are end-to-end taps
with diverging activations, not shared-operand evidence of a fault in layer 32.
A separate CUDA capture recorded biased router scores and selected expert IDs
for the same 144 layer/checkpoint pairs. Its logits also matched the archived
replay bit-for-bit. Re-ranking the captured scores reproduced each selected
set; the smallest eighth-to-ninth score gap was 8.87e-5. This establishes the
reference for the Rust router comparison. The Rust replay reproduced its prior
logits bit-for-bit. The two top-eight sets agree in 37/48, 38/48 and 9/48 logical
layers at 1,023, 1,024 and 1,025 tokens, respectively. At 1,025 the first set
change is in layer 8 (7/8 overlap); its output relative L2 rises to 11.94%,
from 1.69% at the preceding layer. That makes layer 8 a useful intervention
point, not proof of a router defect: the router operands already differ.
Logs, tensor rows and the disposable driver are in the local
`benchmarks/results/qualification/inkling-layer-taps/` directory.

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
