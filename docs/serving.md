# Text serving

Run the experimental Qwen GGUF/CUDA path:

```sh
cargo run -p ribn-cli --features cuda -- serve /path/model.gguf \
  --bind 127.0.0.1:8000 --model-name ribn --context-length 4096

curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ribn","messages":[{"role":"user","content":"Explain a mutex."}],"max_tokens":64,"stream":true}'
```

This is a documented OpenAI-style **subset**, not general OpenAI API compatibility.
The server uses the same text facade and execution owner as CLI/Rust generation.
It accepts only loopback IP addresses. There is no authentication, TLS, CORS policy,
remote fetching or authorization boundary; do not expose it through a public proxy.

## Request subset

POST JSON to `/v1/completions` or `/v1/chat/completions`.

| Field | Supported values |
| --- | --- |
| `model` | Required exact identifier from `--model-name` (default `ribn`) |
| `prompt` | One string, completions only; no template applied |
| `messages` | 1–128 objects with string `role` and `content`, chat only |
| Message `role` | `system`, `user`, `assistant`; artifact template determines rendering |
| `max_tokens` | Positive integer, default 32, at most `--max-output-tokens` |
| `stream` | Boolean, default false |
| `stream_options` | Only `{"include_usage":true/false}`, with `stream:true` |
| `temperature`, `top_p`, `n` | Only 0, 1 and 1 respectively; decoding is greedy |
| `repetition_penalty`, `logprobs` | Only 1 and null respectively; neutral defaults, no penalty or logprob output |
| `ignore_eos` | Only false; artifact and chat control-token stops remain enabled |

Encoded prompt plus `max_tokens` must also fit the loaded model's context. An
impossible combined reach returns 400, not retryable overload.

Unknown fields, query parameters, content encodings and request trailers reject.
Arrays of prompts, token-ID prompts, multimodal content, tools, stop strings, logprob output,
structured output, stochastic sampling and alternate operations are unsupported.
Artifact EOS and rendered chat control-token stops remain model-owned; this subset
does not override them.

Non-streaming responses contain one choice, finish reason and token usage.
Streaming uses SSE `data: <JSON>\n\n` frames: ordered text/content deltas, one finish
chunk (`stop` or `length`), optional usage-only chunk with empty choices, then
`data: [DONE]\n\n`. Chat starts with an assistant-role chunk. Usage counts committed
tokens, including suppressed generated stop tokens. A stream failure after headers
emits one `{"error":...}` frame and `[DONE]`, without successful finish/usage.
A broken connection or deadline may end delivery without a terminal frame.

Before streaming headers, invalid/unsupported requests return 400, byte-envelope
violations 413, unavailable/loading/faulted or overloaded admission 503 with
`Retry-After: 1`, and generation failure 500. Protocol-parser errors may use Hyper's
HTTP error response instead. Unsupported routes return 404.

## Execution capacity

`--max-sequences` selects prepared sequence capacity and the AR active-sequence
limit (default 1). It is not capped by the runtime's conservative eight-request
constructor default. Assembly keeps the default global event grant or increases
it enough to cover every requested active sequence's mailbox. Overflow in the
active-plus-queued count or global event grant rejects before model/device access.

`--max-requests` separately limits HTTP generation slots. Application request
permits cover preprocessing, queued work, execution and retained output/retirement;
`TextModel::request_capacity()` reports this fixed bound, not currently free permits.
Startup reports HTTP slots, the active-sequence limit and application capacity
separately. Other admission/memory bounds can prevent all execution slots being used.
Increasing sequence capacity can increase state reservation and accepted host/output
storage; it does not increase the physical kernel batch limit or prove a memory peak.

## Bounds and transport

Each HTTP/1 connection serves **one** request and closes. Keep-alive, HTTP/2 and
pipelined execution are not supported. Extra bytes after the first generation body
are discarded without retention while observing disconnect; they cannot hide EOF.

| Envelope | Default |
| --- | --- |
| Accepted connection tasks | 64 |
| Generation HTTP slots (body reading through response-body handoff) | 32 |
| Header/parser buffer; header count | 16 KiB; 32 |
| Header read timeout | 5 seconds |
| Body bytes (including JSON syntax) | 256 KiB |
| Chat messages | 128 |
| Output tokens | 4096 |
| Non-streaming serialized response; individual SSE frame | 1 MiB maximum |
| Absolute connection lifetime | 120 seconds |

`serve --help` lists configurable bounds. The absolute deadline includes body reading,
preparation, generation and writing; it also limits a healthy long generation. Adjust
it for the workload, rather than interpreting it as an idle timeout.

Connection saturation closes excess sockets before parsing or creating another task.
Generation-slot saturation returns 503 without reading its body. Health/metrics do not
consume generation slots, but still require a connection slot. There is no transport
waiter queue. Text/runtime permits can independently reject admission.

Streaming pulls bounded frames directly from the owned text stream, with no detached
producer. Its frame envelope is the smaller of the response limit and six times the
processor's decoded-token limit plus 4096 bytes of metadata. Terminal/usage frames are
validated together before publishing success. Hyper's write buffering can overshoot
its 16 KiB threshold by one frame; those frames and bounded serialization scratch remain
connection-owned even after Hyper consumes the response body and its HTTP slot returns.
Non-streaming text collection and serialization are separately
bounded, including escaping/temporary overlap, rather than using an unbounded collector.

Body parsing is limited by bytes and message count; the text facade additionally checks
retained string/vector capacity before preprocessing. These are host allocation envelopes,
not process-RSS or GPU peak-memory guarantees. Allocator overhead, kernel socket buffers,
loaded weights, processor workspace and device storage have separate owners and limits.

Disconnect, deadline and shutdown abandon interest, not device completion. Pending
device work and retirement remain with the existing execution owner.

## Health, metrics and shutdown

- `GET /health`: 200 while loading or serving; 503 if closing or faulted.
- `GET /ready`: 200 only while serving; otherwise 503.
- Both return `{"status":"loading|serving|closing|faulted"}`. Readiness is an owner-state
  snapshot, not an admission-capacity promise.
- `GET /metrics`: Prometheus text with fixed names and no request/model labels:
  readiness; active connections/requests; connection rejection/deadline counts;
  generation attempts, rejected/failed/completed/abandoned counts; and successful
  terminal prompt/completion tokens.

Completed/usage counters mean a valid model terminal was observed, not that the client
received all bytes. Failed counters cover generation/preparation/response failure;
rejected counts cover generation calls returning non-200 before streaming headers.
These populations can overlap. Abandoned means accepted preparation/generation lost
interest before a terminal outcome; it does not assert physical reclamation.

SIGINT/SIGTERM stop admission and abandon connections before model cleanup.
Shutdown during loading waits for that load's owner, then cleans it up.
`ribn-server::Server` retains a failed cleanup owner for `shutdown_async` retry;
the CLI makes one explicit retry and reports unresolved failure. No cleanup timeout
releases device-visible storage. A failed load leaves health/metrics available in the
faulted state until shutdown.

## Qualification

Host TCP tests exercise the real HTTP entry point, tokenizer, text facade and driver
with only the device substituted. The bounded CUDA HTTP gate and actual CLI streaming/
SIGTERM shutdown passed, along with AIPerf and native vLLM client compatibility smokes.
The [load-capacity host gate](../benchmarks/runtime-contract.md#text-load-capacity-host-gate)
covers configured active limits beyond eight; higher-concurrency device
requalification remains pending and physical batching is unchanged.
See [qualification](../benchmarks/runtime-contract.md#initial-text-http-qualification-2026-10-13)
for revisions, artifact, conditions and limitations. These are not matched competitive
measurements, broad-context numerical qualification or peak-memory evidence. The
[roadmap](roadmap.md#5-minimal-serving-then-wider-systems) and
[benchmark methodology](../benchmarks/README.md) own those remaining gates.
