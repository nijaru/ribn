# Local text serving

```sh
cargo run --release -p ribn-cli --features cuda -- serve /path/qwen.gguf \
  --bind 127.0.0.1:8000 --model-name ribn --context-length 4096

curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ribn","messages":[{"role":"user","content":"Explain a mutex."}],"max_tokens":64,"stream":true}'
```

For a Gemma 4 package, use `serve-gemma /path/to/gemma-package` with
`--features metal` on macOS or `--features cuda` on NVIDIA. Gemma uses its
prepared context (16K paged KV with automatic exact prefix caching on Metal;
4K on CUDA) and default device; omit `--device`, `--context-length` and
`--max-sequences` for `serve-gemma`. On Metal, pre-quantized MLX affine
group-64 4-bit/8-bit packages (`mlx_lm convert -q --q-bits 4|8 --q-group-size 64`)
load directly at their native precision without `--quantize` (`~0.60 s` / `~0.90 s`
weight load on Gemma 4 E4B). For BF16 packages on Metal, optional
`--quantize bf16|q8|q4` (default `bf16`; also available on `run-gemma`) streams
weights directly into affine group-64 quantized GPU storage on load (`q8`: ~7.4 GiB
weights / ~9.6 GiB RSS; `q4`: ~3.9 GiB weights / ~6.1 GiB RSS on Gemma 4 E4B).
Mismatched `--quantize` overrides on pre-quantized packages and non-`bf16` values
on non-Metal builds are rejected explicitly.

The adapter binds only to loopback. It has no authentication, TLS or authorization
boundary; do not expose it through a proxy. This is an experimental OpenAI-style
subset, not a general OpenAI-compatible server.

## Requests

POST JSON to `/v1/completions` or `/v1/chat/completions`, or discover the loaded
model via `GET /v1/models` and `GET /v1/models/{model}` (`503` while loading;
`404` for unknown model IDs).

| Field | Accepted input |
| --- | --- |
| `model` | Exact `--model-name` value (default `ribn`) |
| `prompt` | One string, completions only; no template |
| `messages` | 1–128 text messages; roles `system`, `user`, `assistant`; `content` accepts a string or up to 64 `{"type":"text","text":"..."}` parts; artifact template applies |
| `max_tokens` / `max_completion_tokens` | Positive integer, default 32; limited by server cap and remaining context. `max_completion_tokens` is accepted on `/v1/chat/completions` (specify one, not both) |
| `stream`, `stream_options` | Default false; streaming optionally accepts `{"include_usage":true/false}` |
| `temperature`, `top_p`, `top_k`, `n` | `n` must be `1`. On sampling-capable executors (Gemma 4 Metal model-wide, Inkling 2), accepts finite `temperature >= 0`, `top_p` in `(0, 1]`, and optional `top_k`; on greedy-only executors, only `temperature = 0`, `top_p = 1`, and omitted/`0` `top_k` are accepted |
| `repetition_penalty`, `presence_penalty`, `frequency_penalty`, `logprobs` | Only `1`, `0`, `0`, and `null` respectively |
| `stop` | One string or up to four strings, 256 bytes total; matched against decoded text and excluded from output |
| `seed` | Deterministic sampling seed when stochastic sampling is active; accepted with no effect on greedy decoding |
| `ignore_eos` | Boolean; disables artifact/chat token stops, not explicit strings |

Unknown fields, query parameters, content encodings and request trailers reject.
Multimodal input, tools, token-ID prompts and structured output are unsupported.

## Responses and limits

JSON responses contain one choice, finish reason and usage. Streams are SSE:
ordered deltas (starting with assistant role for chat), a `stop` or `length`
finish chunk, an optional usage-only chunk with empty choices, then `data: [DONE]`.
Usage counts committed tokens, including suppressed stop tokens. A failure after
headers sends an error frame then `[DONE]` without successful finish or usage;
a broken connection or deadline can end delivery without a final frame.

Before streaming headers: 400 for invalid input (including prompt plus output
beyond context), 413 for byte limits, 503 with `Retry-After: 1` for unavailable
or overloaded admission, 500 for generation failure, and 404 for unknown routes.

Each HTTP/1 connection serves one request. Defaults include a 256 KiB JSON
body, 4,096 output tokens, 32 generation slots, 64 connections and a 120-second
absolute connection deadline. See `serve --help` or `serve-gemma --help` for options.
HTTP slots include body reading and response consumption; they are separate from
application admission and active sequences. Qwen defaults to one active sequence;
increasing that limit can increase state reservation without increasing physical
batch width. Connection saturation closes excess sockets; generation-slot
saturation returns 503 without reading the body.

- `GET /health`: 200 while loading or serving; otherwise 503.
- `GET /ready`: 200 only while serving. It does not promise admission capacity.
- `GET /v1/models` and `GET /v1/models/{model}`: OpenAI model discovery (`503` while loading; `404` for unknown IDs).
- `GET /metrics`: Prometheus counters, gauges, and latency sums/counts (`ribn_ttft_seconds_*`, `ribn_request_duration_seconds_*`); successful terminal counts do
  not imply that the client received the response.

SIGINT/SIGTERM stop admission and abandon connections before model cleanup.
Disconnect or timeout cancels client interest, not pending device work.

See the [HTTP adapter contract](resource-protocol.md#text-http-adapter)
for ownership and bounds, and the [real-device qualification](../benchmarks/runtime-contract.md#initial-text-http-qualification-2026-10-13)
and [SDK gate](../benchmarks/README.md#http-sdk-compatibility-gate) for tested scope.
