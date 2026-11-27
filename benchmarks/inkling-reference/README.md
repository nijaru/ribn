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

With the captured CUDA layer-8 input at 1,025 tokens, the Rust router recovered
all eight CUDA expert IDs in order (router logit max absolute error 1.43e-6).
The Rust MoE output with decoded Q8_0 activation rounding differed from CUDA by
0.0186% relative L2 on that same input. Replacing only expert 33 with 37, as in
the end-to-end Rust run, changed the MoE output by 36.15% relative L2 against
the same Rust computation with the CUDA route. This shows the swap can amplify
upstream drift. It does not establish why the inputs diverged or justify
forcing routes in production.

A full-model counterfactual branched from the same committed Rust cache at
1,024 tokens and replaced only expert 37 with 33 in logical layer 8 on the next
token. The unmodified branch reproduced previous Rust logits bit-for-bit;
layer 7 output and layer 8 attention output were identical across branches.
The swap reduced layer 8 output error from 11.94% to 2.43% relative L2, but
final centered logit RMSE only moved from 0.38878 to 0.38134 (bound `< 0.25`,
argmax unchanged). Correcting that single route is not sufficient to recover
end-to-end parity.

A pinned 4090 A/B held width 64, F16 KV, artifact, token histories and library
revision fixed and changed only llama.cpp's Flash Attention setting. The
Flash-enabled control was bit-identical to the archived oracle at all ten
checkpoints. Flash-disabled preserved all ten argmaxes, but the two CUDA
paths differed by 0.17239–0.38463 centered logit RMSE on the eight long
checkpoints. At 1,025 the Rust packed-activation probe was closer to
Flash-disabled than Flash-enabled (0.22269 vs 0.38878); at 1,023 the ordering
reversed (0.31139 vs 0.23070). Switching or selecting the closer reference
per checkpoint would not qualify the Rust implementation. The observed
sensitivity includes downstream expert routing; it is not a direct attention
kernel error bound. No acceptance threshold changed.

### Shared-operand attention check (2026-11-26)

The pinned Flash-enabled replay captured layer 0's actual fused input tensors
at 1,024 and 1,025 tokens: F32 queries, historical F16 K/V, the combined F16
bias/mask (including the sink-logit shift), and the zero-logit sink. The capture
reproduced archived width-64 logits bit-for-bit at all five checked positions.
Both sides of the transition had exactly 1,024 visible keys per query head even
though the physical KV span grew to 1,280 rows at 1,025 tokens.

An independent FP64 dot/softmax/value calculation used the saved operands. A
standalone Rust replay mirrored the CPU attention loop but consumed the same
already-combined bias/mask instead of recomputing it. Against FP64, CUDA
relative L2 was 8.34e-8 and 7.22e-8; scalar Rust F32 was 5.06e-7 and 6.51e-7.
This isolates the final attention calculation on shared operands: it is not
large enough to account for the end-to-end gap in this layer. It does not test
how the two paths produce their queries, caches, or relative-bias operands.
A second capture took the F32 bias/mask immediately before its F16 cast.
The cast output matched host IEEE F16 rounding for every element, and the
previous Q/K/V, mask and output captures reproduced bit-for-bit. Replaying the
same Q/K/V and sink with only the mask precision changed perturbed the FP64
attention output by 7.07e-5 and 7.22e-5 relative L2, respectively. This measures
the direct layer-0 contribution of that cast, not how it accumulates across
layers. This capture only verifies the fused path.

The cache comparison verified that all 1,023 retained CUDA rows were bitwise
unchanged across the boundary; Rust also kept its prefix unchanged. In a
layer-0-only Rust replay, the rounded-activation probe reproduced all 15 prior
layer-0 taps bit-for-bit across 1,023–1,025. In the same FP64 attention
calculation, with CUDA bias and sink held fixed, replacing only the historical
K/V rows with stock Rust's changed the output by 1.81e-3 and 2.14e-3 relative
L2 at 1,024 and 1,025. With the disposable rounded-activation path, those
changes fell to 8.17e-5 and 1.02e-4. Swapping just the rounded path's current
K/V changed the output by less than 1e-6 at either position. The retained rows
carry more error than the current row, but neither path showed a cache boundary
shift in this layer.

### Layer-0 output projection on shared operands (2026-11-26)

The post-attention capture also reproduced all five archived logit rows
bit-for-bit. With the same CUDA inputs, the stock CPU gate and output
projections (both Q8_0 weights) agreed with an FP64 dot product on decoded
weights within 2.2e-7 relative L2, but differed from CUDA by 0.30–0.35% and
0.71–0.86%, respectively. The disposable activation-rounding path matched
CUDA within 2.0e-7 for both projections. An independent FP64 replay of the
CUDA sigmoid/multiply and depthwise convolution also agreed within 6e-8.

With each backend's own inputs, the rounded CPU path still differed just
before the output projection by 2.67e-4 and 2.77e-4 at 1,024 and 1,025. Feeding
those two pairs of gated rows through a single output projection without
activation rounding gave differences of 2.73e-4 and 2.55e-4. With rounding,
the same pairs changed 36 and 23 of 3,072 activation codes and the projection
outputs differed by 2.07e-3 and 2.03e-3. Convolution replays on captured inputs
reproduced the post-attention taps and showed the current projected row
contributed more error than the three stored rows at both positions. This
identifies an early amplification mechanism, not a full-model fix.

A layer-0 follow-up applied the sink shift and F16 bias/mask cast to Rust's
own attention operands. It lowered the pre-gate error at both positions, but
post-attention error moved in opposite directions: 9.94e-4 → 1.08e-3 at 1,024
and 1.16e-3 → 1.13e-3 at 1,025. We did not promote the cast or rerun the full
numerical gate for it.

### Layer-0 K/V projection scales by chunk width (2026-11-26)

We captured the layer-0 Q8_0 K and V projections before convolution and
cache conversion at widths 1, 8, 24, 31, and 64. The tapped CUDA replay matched
all five archived logit rows bit-for-bit. On identical captured inputs, an
FP64 host projection matched CUDA with F16 activation scales at widths 1 and 8,
and F32 scales at widths 24, 31, and 64. The matching precision was within
1.1e-7 relative L2 per row; the other precision differed by 1.09e-4 to 1.67e-4
in aggregate. Holding the activation codes fixed and changing scale precision
was sufficient to distinguish the paths. This agrees with the pinned source's
F16-scale MMVQ and F32-scale MMQ `D4` layouts; it does not establish a global
cutoff for other weight shapes, expert dispatches, or devices.

Applying only the measured scale choice to layer-0 K and V in the disposable
rounded Rust replay cut the historical K/V discrepancy from 2.84e-4 to 1.34e-4
for K and from 1.99e-4 to 9.51e-5 for V at 1,025. The layer-0 post-attention
error fell from 9.94e-4 to 5.04e-4 at 1,024 and from 1.16e-3 to 5.10e-4 at
1,025. The improvement did not carry through the model: the unchanged gate
passed 8 and 32 tokens, then failed at 1,023 with centered RMSE **0.56313**
(previous all-F16-scale probe: **0.23070**). The argmax still matched, but
later checkpoints were not reached. Layer-0 post-attention at that same
checkpoint improved from 1.08e-3 to 9.18e-4 relative L2. Do not promote the
K/V-only intervention or infer end-to-end improvement from the early layer.

### Layer-0 expert projections on shared operands (2026-11-26)

Another capture matched all five archived logit rows and saved the inputs,
selected expert IDs and outputs of the layer-0 IQ4_NL gate/up and down
projections. The CPU replay used the recorded routes rather than rerouting.
For each projection, it checked all eight selected experts on three positions
per multi-token chunk and one position at width 1. With the captured inputs,
F16 activation scales matched at widths 1 and 8, and F32 scales matched at
24, 31 and 64. The matching replay was within 1.9e-7 relative L2 on every
sampled row. At width 8, changing only the normalization used to form integer
codes (`x / d` versus `x * (127 / peak)`) left one down row above 2.8e-4,
even though both arms rounded their scale to F16. Match the packing path as
well as its scale; these samples do not establish dispatch for other shapes.

On the captured gate/up rows, Rust's SiTU and an independent FP64 evaluation
both differed from CUDA by up to about 9e-6 relative L2. At width 64, using
Rust's SiTU flipped 56 of 589,824 activation codes before the down projection.
Swapping only SiTU on rows with changed codes moved a down output by up to
1.39e-3 relative L2. One width-64 row was excluded from that swap because the
baseline still differed from CUDA by 7.84e-4 on its original input; the
packing replay is not exact on all rows.

In the layer-0 replay, correcting expert packing on top of the K/V-only probe
left the final layer output error nearly unchanged (for example, 3.8808e-3 →
3.8815e-3 at 1,025). Neither intervention was promoted.

### Layer-0 FFN and history on shared operands (2026-11-26)

A convolution-only tap reproduced the five archived logit rows bit-for-bit.
A broader router and reduction tap changed the long rows and was discarded.
Ribn's FFN convolution matched CUDA within 4.2e-8 relative L2 on identical
current MoE values and history at 1,023–1,025.

At 1,023 and 1,025, substituting only CUDA's post-attention residual into
Ribn's depth mix and FFN RMSNorm reduced the normalized-input error from
1.73e-3 and 8.30e-4 to 5.06e-7 and 3.14e-7, respectively. On the shared
normalized input, the router chose the same eight experts in order. At
1,023, the router logits still differed by 3.22e-5; reusing CUDA's logits
instead of Rust's reduced the error of the weighted sum of captured down
outputs from 2.11e-5 to 1.02e-7. This attributes that difference to the
logits, not to the aggregation; it does not identify the CUDA matmul's
arithmetic mode. The width-matched CPU MoE on shared normalized inputs
matched within 2.13e-4 at both positions.

With each backend's own inputs, swapping just the current MoE value in an
otherwise CUDA convolution and residual reproduced most of the layer-output
error at all three positions. At 1,025 the current value alone changed the
layer output by 3.84e-3 relative L2, versus 6.06e-4 from swapping history and
3.88e-3 from all differences together. These effects are not additive.
The next candidate should target the remaining post-attention input error
rather than change FFN history or rerun the full gate on expert packing alone.

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
