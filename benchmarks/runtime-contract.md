# Runtime contract verification

## Host-only gates

The `ribn` crate has no CUDA dependency. It uses foundation readiness and Flume for
bounded blocking/async driver channels. External fixture model
implementations exercise its real public trait with different private state
layouts. Qwen bridge tests inject a host backend into the actual translation and
lease code. These tests do not evaluate model numerics or execute GPU kernels.

```sh
python3 tools/check-boundaries.py
cargo test -p engine-qwen --no-default-features --test model_config
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy -p engine-nvidia --features cuda --all-targets -- -D warnings
cargo clippy -p engine-qwen --features cuda --all-targets -- -D warnings
cargo clippy -p ribn-cli --features cuda --all-targets -- -D warnings
cargo test -p engine-qwen --features cuda
cargo test -p ribn-cli --features cuda
```

CUDA-feature compilation and the nonignored adapter tests run on a CPU CI host.
Device tests remain explicitly ignored. The redesign session ran these checks
with Rust 1.98.0; it did not have an NVIDIA device.

Coverage includes atomic batch commitment, delayed completion, in-flight
cancellation with peer progress, multi-token completion, live-policy changes,
queue/input limits, output credit reservation, admission rollback, deferred
admission, invalid output, release retries, partial submission and poll failures,
shutdown retries, and conservative retention after uncertain teardown. New tests also cover
per-request mailbox isolation, draining after execution-slot reuse, bounded ready
list membership, source-format-independent model geometry, and small-executor defaults.

`crates/runtime/tests/progress_negotiation.rs` pins positive partial completion:
shortened prefill commits exactly the reported range, output belongs only to final
prefill, and decode output matches advancement. A two-credit output pool tests refunds
when an offered final-prefill range is shortened. Zero-prefill success is rejected
with a failing-before regression. Mixed malformed rows commit no healthy peer output.
`crates/runtime/tests/multimodal_admission.rs` checks shared submission budgets and
admission-time rejection of impossible items. It deliberately cannot express temporary
mid-request waiting; no completion-time `Blocked` API remains.

`crates/runtime/tests/abandoned_requests.rs` separates execution and mailbox
lifetimes: an execution slot can be reclaimed before its terminal event is consumed.
`Engine::discard` cancels and suppresses present/future delivery, even after that
slot is gone, while retaining in-flight retirement ownership. Tests cover repeated
discard without draining, healthy peers and in-flight cancellation. Output unit tests
pin reservation retention, ready-list unlinking and discarded terminal publication
when a peer holds all buffer capacity. Request-scoped callers use `pop_event_for`;
the aggregate drain intentionally includes other clients' events.

## Qwen model-metadata preflight host gate

```sh
cargo test -p engine-qwen --no-default-features --test model_config --locked
cargo test -p engine-qwen --features cuda --locked
```

The model definition now owns normalization epsilon and rotary dimensions/frequency.
GGUF normalization follows the independent llama.cpp reference pin
[`b81c99b`](https://github.com/ggml-org/llama.cpp/tree/b81c99b479d4c24e5eeca10de99032ebd343ef8f):
normalization epsilon is required, missing RoPE frequency defaults to 10000, and missing
rotary dimensions default to the key head width—not the pinned CUDA values. Identity
linear scaling honors modern/legacy factor precedence and the zero sentinel; `none`
overrides inactive frequency factors, but not attention magnitude. IMRoPE sections must
map every rotated text pair to a text position, not the fourth zero-position channel.
A nonzero fourth section alone does not imply incompatibility, and zero fourth alone
does not establish equivalence. Section arithmetic must fit the reference selector.
Explicit Boolean/UINT32/INT32 recurrent catalogs override the interval and normalize only
when they describe the regular layer groups and nonrecurrent MTP tail this model supports.
Other scaling modes, nonidentity transforms and nonperiodic catalogs reject explicitly.

CUDA loading compares normalized metadata with the executor's actual geometry constants
and current numerical profile before tokenizer/device access. The real public load
entrypoint regression first failed with missing tokenizer metadata instead of rejecting
an unsupported epsilon (exit 101). It now rejects incompatible epsilon, rotary settings,
SSM groups and a different Q/K/V head decomposition with identical aggregate projection
widths. Metadata-only GGUF fixtures and an invalid device ordinal make this host-testable;
matching profile validation does not initialize CUDA. Device execution, lease settlement
and kernels are unchanged. The constructor receives the validated model epsilon rather
than an independently substituted literal.

Host gates pass. A temporary exhaustive check compared the closed-form IMRoPE text
predicate with the pinned selector for 82,880 small configurations; committed boundary
cases protect holes, unused fourth sections and overflow. These are parser/preflight
results, not GPU arithmetic or broad model qualification. Subsequent device
re-execution is recorded below; neither gate resolves the separate long-prompt
divergence or lazy-source/loading peaks.

### Device re-execution (2026-10-16, `dc4aed0`)

The pinned artifact SHA-256 `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`
was verified on the RTX 4090 (driver 615.71.09, reported CUDA 13.4, Rust 1.98.0).
Its actual header declares epsilon `1e-6` as F32, rotary dimensions 64, frequency
base `1e7`, sections `[11,11,10,0]` and interval 4, with no overriding recurrent
catalog or scaling declaration. The normalized profile is accepted without a bypass.

Serialized with no competing compute process, all commands exited 0:

```sh
export RIBN_MODEL=/absolute/path/Qwen3.8-27B-UD-Q4_K_M.gguf
export ENGINE_QWEN_GGUF="$RIBN_MODEL"
export RIBN_REFERENCE=$(pwd)/crates/qwen/tests/fixtures/qwen38-code-fill4096-257.tokens
cargo test --release -p engine-qwen --features cuda --test cuda_runtime --locked \
  -- --ignored --test-threads=1
cargo test --release -p ribn-server --features cuda --test cuda_http --locked \
  -- --ignored --test-threads=1
cargo test --release -p engine-bert --features cuda --test device_reference --locked \
  -- --ignored --test-threads=1
```

Qwen passed **5/5** (554.41 s test time; 559 s including build): independent reference,
all widths 1–9, cancellation, mixed/nonadjacent reaches and histories, owned-driver
stalled/abandoned peers and constrained continuation. The reference remains the bounded
257-token prompt / eighteen-token agreed prefix, not broad context or distributional
acceptance. HTTP passed **1/1** (25 s including build); BERT passed **3/3** (0.27 s test
time). No competing compute process or residual compute allocation remained after teardown.
Raw logs/header/environment are retained outside Git at
`desktop:~/ribn-qualification/dc4aed0-device/`.

## AR pre-submit preparation host gate

The real `Engine` now negotiates aggregate work before submission through required
executor preparation/abandonment methods. Qwen retains a validated backend batch and
consumes its exact ready rows, preserving full-reachable reservation and current kernels.
There is no paged Qwen growth or inherited device qualification.

`crates/runtime/tests/preparation.rs` uses the real byte authority and owning host storage
to cover shortened prefill/decode, credit refunds before completion, aggregate pressure,
healthy peer progress, parked cancellation, local rejection, whole-report refusal,
abandonment preserving old continuation, and failed preparation/abandonment/enqueue plus
failed-barrier retry. These are control/ownership tests, not numerical or GPU evidence.
Driver notification tests now exercise both admission and preparation waits, including
publication in the check-to-park gap and re-registration without idle polling.

A new scheduling regression was reproduced before its fix: with a one-row offer budget,
two deferred rows could hide a third ready row and leave the worker sleeping indefinitely.
A bounded rebuild now offers unoffered peers in the same step; parked rows are not retried.

```sh
cargo test -p ribn --test preparation --locked
cargo test -p ribn --lib resource --locked
cargo test -p engine-qwen --features cuda --lib --locked
```

The synthetic [dispatch comparison](runtime-alignment/README.md#ar-preparation-relative-to-71c326f)
is separately bounded evidence. Subsequent [Qwen/reference and HTTP re-execution](#device-re-execution-2026-10-16-dc4aed0)
and [capacity-independent decode qualification](#heterogeneous-capacity-decode-2026-10-16)
cover bounded real-device success/cancellation paths. Injected preparation/readiness
interleavings remain host evidence; no physical continuation growth follows.

## Mixed-arrival native Qwen gate

The production AR engine's finite trace harness varies prompt prefixes, output budgets
and planned arrival times. It retains the original clock through dispatch lag and overload,
counts rejected/failed requests, and reports latency-objective goodput. See
[trace usage and semantics](README.md#mixed-arrival-native-traces).

The initial device run exposed a real selection defect: heterogeneous request-reachable
KV capacities were routed to a dense batched attention lane requiring uniform capacity.
The run faulted with `batched attention requires positive uniform KV capacity`.
The initial repair required equal concrete KV specifications and kept heterogeneous
valid rows on the qualified per-row lane, without padding requests to maximum context.
The later [capacity-independent lane qualification](#heterogeneous-capacity-decode-2026-10-16)
replaces that temporary capacity restriction with stream-ordered score reuse; model-layout
checks remain. The historical measurements below describe the earlier row-wise fallback.

The host selection regression failed before the fix and passes afterward.
`mixed_request_reaches_preserve_reference_and_cancelled_peers` exercises three identical
prompts with output budgets 4/8/8, including cancellation leaving compatible survivors.
Both cases passed on RTX 4090, driver 615.71.09, Rust 1.98.0, using the pinned GGUF and
independent llama.cpp fixture in [fixture provenance](../crates/qwen/tests/fixtures/README.md).
The gate compares eight tokens, not a complete 32-token reference: the fixture's known
near-tie divergence after its eighteen-token agreed prefix remains unresolved.
The existing full-model Qwen gates also passed again: reference/cancellation at
1/2/8/9 rows (including nine-to-eight), owned-driver stalled/abandoned peers and
constrained request-reachable admission. Those three tests took 267 seconds serialized;
there were no competing compute processes and 33 MiB remained after teardown.
The migrated linear backend, bounded GGUF materialization and physical 8/9-row
retirement gates also passed (0.44, 0.19 and 21.48 seconds respectively).

One cold-execution smoke run on 2026-10-06, base `9a78aa8` plus this selection fix
and trace harness, used the checked-in workload, warp GEMV, prefill chunks of eight,
active/queue limits 4/4 and a 524,288,000-byte continuation authority:

| Offered/completed/rejected/failed | Output tokens | Elapsed | Completed output tok/s | TTFT p95 | ITL p95 | E2E p95 | SLO goodput |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 / 8 / 0 / 0 | 136 | 9.900 s | 13.74 | 8994.78 ms | 288.85 ms | 9449.68 ms | 0 requests/s |

SLOs were TTFT <= 5000 ms, maximum per-request ITL <= 200 ms and E2E <= 10000 ms.
Mean/max submit lag was 75.45/180.53 ms. The sample is one finite eight-request run,
not a throughput distribution, optimization comparison or SOTA claim. No physical-peak
or bottleneck-cause claim follows from it. [Raw structured results](qwen-trace/2026-10-06-mixed-4.json) retain per-request outcomes,
configuration and token streams. Reproduce with the checked-in workload and flags above.

```sh
RIBN_MODEL=/path/to/Qwen3.8-27B-UD-Q4_K_M.gguf \
RIBN_REFERENCE=$(pwd)/crates/qwen/tests/fixtures/qwen38-code-fill4096-257.tokens \
cargo test --release -p engine-qwen --features cuda --test cuda_runtime --locked \
  mixed_request_reaches_preserve_reference_and_cancelled_peers \
  -- --ignored --exact --test-threads=1
```

### Native dispatch diagnostics host gate

```sh
cargo test -p engine-nvidia --features cuda --lib --locked
cargo test -p engine-qwen --features cuda --example qwen_serving_bench --locked
```

Host tests cover the actual whole-decode eligibility predicate at one/eight/nine
rows, unequal KV capacities, mixed phases, unsupported decode shape and missing
positions; subgroup order/compatibility coverage remains. Accounting tests separate
backend offers from fully enqueued work, merge each group once, distinguish prefill
chunks from decode lanes, retain failed-attempt counts and saturate host durations.
These are classifier/accounting tests, not injected real CUDA launches or a
failing-before device regression. The benchmark flag stays off by default and does
not select grouped decode or change prefill policy. CI runs both host gates.

The observer owns fixed-size data only; enqueue publication follows the existing
completion-event/pending-record transaction. Host tests alone establish no CUDA mapping,
retirement, physical-memory or performance qualification.
See [usage and counter scope](README.md#mixed-arrival-native-traces).

#### Bounded device mapping and trace (2026-10-16, `dc4aed0`)

[Structured evidence](qwen-trace/2026-10-16-dispatch-profile.json) retains nine native
probes and a separate Nsight Systems trace on the same idle RTX 4090/pinned artifact.
Eight generated tokens per member produced exactly eight successful submissions:

| Members / prompt | Prefill rows / tokens | Decode model-step calls |
| --- | --- | --- |
| 1 / 5 tokens, serial | 1 / 5 | 7 scalar |
| 4 / 5 tokens, serial | 4 / 20 | 7 width-four groups, not 28 groups |
| 9 / 5 tokens, serial | 9 / 45 | 63 scalar; row-limit fallback |
| 4 / 13 tokens, chunk size eight | 4 / 52 | 7 width-four groups; four eight-token prefill chunks |

All attempts enqueued successfully; offered tokens equal prefill tokens plus decode
calls times width. The thirteen-token case executes one eight-token chunk and a
five-token serial tail per member, including the sampling token. Its three alternating
off/on pairs preserved the common-history token outputs and routing counts. Observed
throughput ranges were 31.00–31.14 tok/s off and 31.11–31.18 on. These cold, eight-token
runs have no execution warmup and millisecond-rounded elapsed reports: they establish
no detectable regression in this sample, **not** zero overhead or an all-shape bound.
Failed enqueue/drain device injection and broader observation scope remain unqualified.

The separate existing eight-request native mixed-arrival workload used four active/four
queued slots, 500 MiB continuation grant and production prefill chunks. It is not the
endpoint cohort below, and native requests do not apply text EOS filtering. Nsight
Systems 2026.3.2.476 captured CUDA activity without privileged counters or CPU sampling:

```sh
nsys profile --trace=cuda --sample=none --cpuctxsw=none --cuda-memory-usage=false \
  --output=/tmp/ribn-mixed \
  target/release/examples/qwen_serving_bench --model=/path/model.gguf \
  --workload=benchmarks/workloads/qwen-mixed-arrivals.json \
  --concurrency=4 --queue-capacity=4 --continuation-capacity-bytes=524288000 \
  --ttft-slo-ms=5000 --itl-slo-ms=200 --e2e-slo-ms=10000 \
  --dispatch-diagnostics --result-json=/tmp/ribn-mixed.json
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv /tmp/ribn-mixed.nsys-rep
```

All 95 submissions succeeded. The snapshot counted 832 prefill tokens (816 in 102
same-sequence chunks), 128 scalar decode calls, **zero whole-decode groups** and 44
nonuniform-KV fallbacks. `_batch` kernel names here therefore do not prove cross-request
decode batching. The trace contains 375,822 kernels on one stream: 9.610 s summed kernel
duration over a 9.999 s first-to-last-kernel span. That 96.1% is an interval calculation,
not occupancy or an unprofiled utilization counter. Quantized warp GEMV kernels account
for 70.2% of summed kernel time; `gdn_state_update` for 11.7% and `attn_score_gqa` for 6.8%.
The first kernel is embedding, after preparation. Profiler overhead is not quantified;
do not use this run's latency as a competitive measurement.

This supports investigating physical projection/prefill throughput and heterogeneous
decode lanes before a scheduler-only rewrite or a speculative host-gap fix. It does not
select a kernel implementation or explain endpoint phase costs by itself. Queue age,
physical-memory attribution and endpoint-level dispatch exposure remain open. Raw
reports are retained at `desktop:~/ribn-qualification/dc4aed0-{dispatch,profile}/`.

## AR reservation authority host gate

AR's logical state manager now grants from supplied `BytePool` authorities.
A hybrid bundle validates and receives one aggregate lease, shared by its component
records until the last retires. Qwen no longer checks a free-byte snapshot before
per-component allocation and rollback; the authority makes the atomic decision.

Core host tests cover failed feasibility/duplicate validation without charge or wake,
conservative bundle charging across partial release, invalid whole-set release,
sibling grants and readiness, closure without returning live bytes, and charge retention
when an unreleased logical owner disappears. Existing allocation-provenance and atomic
prefix-commit coverage remains. Qwen tests cover a refused hybrid grant, waiting peer
progress, and authority closure rejecting a waiter while its admitted peer completes.

```sh
cargo test -p engine-core --locked
cargo test -p engine-qwen --locked
```

These are reservation/control tests, not a shared CUDA peak-memory qualification.
Production Qwen retains its private pool, full-reachable reservation and contiguous
storage. See [physical accounting and retirement](../docs/resource-protocol.md#encoder-preparation-and-prepared-resources)
for partial-construction, event-creation failure and cross-stream release gaps.

## Shared text frontend

`ribn-text` reuses the low-level runtime for raw prompt, chat-message, and token-ID
input. Since the owned text facade landed it is a thin layer over `ribn::driver`:
bounded preprocessing, incremental decoding and ordered batching above owned request
streams. Its behavior is host-qualified; only `TextOwner::load` (Qwen GGUF/CUDA
assembly) needs a device.

CLI behavior uses the same frontend: default text input is one user chat message,
`--raw` bypasses the chat template, and prompt/file/piped-stdin input share the same
generation implementation. GPU numerical qualification remains the separate hardware
gate below.

`TextResponse.text` and the `generate_batch` collect helper hold caller-collected
results and are deliberately outside buffered-application accounting. `TextBatch` is
the bounded incremental interface. CLI file/stdin ingestion now reads at most the
payload limit plus one overflow-probe byte before model loading, rejecting oversize or
invalid UTF-8 without truncation. Caller-provided argument storage and allocator spare
capacity are not a process-RSS bound.

### Runtime-owned discard qualification (2026-09-14)

Revision `4477a0e` removes the text facade's discard/retry list. Batch collection
relinquishes all submitted requests on both success and error; stream engine/decoder
errors relinquish delivery immediately. Physical retirement remains runtime-owned.

- Host workspace tests, formatting, boundaries, default and CUDA-feature all-target
  clippy passed with actual zero exit statuses.
- At that revision, `discard_survives_a_blocked_in_flight_completion` failed before
  the cancellation branch fix and passed after it. The old branch erased cancellation
  intent. Removal of blocked outcomes later replaced this with partial-completion
  cancellation coverage.
- CUDA text lifecycle: **4/4, 77.32 s**, serialized on the RTX 4090 with the pinned
  Qwen artifact. Log: `desktop:~/ribn-lifecycle-2026-09-14/01-runtime-discard.log`.
- Other newly added discard tests exercise a new API, not failing-before tests against
  an API that previously existed. Host fixture tests do not exercise text decoding.
- Full CUDA kernel/reference and exact-token gates were not rerun for this slice;
  kernel arithmetic and the Qwen executor were unchanged. This is not a new numerical
  qualification claim. Concurrent handles and event-driven readiness remain undone.

### Positive completion and combined gates (2026-09-14)

Revision `14d0290` removes runtime `StepOutcome`/`StepStatus::blocked`. Qwen and all
runtime adapters return `StepCompletion` directly. Successful prefill advancement
must be positive; final-prefill output is exactly one token, intermediate output is
empty, and decode output equals advancement. Whole-batch validation borrows rows
without allocating/cloning a row plan; commit moves rows and retains batch capacity.
This is a source-level allocation removal, not an allocation-free runtime or measured
model-speedup claim.

The zero-prefill regression failed with the old predicate (exit 101) and passed with
positive advancement required. The partial-prefill test uses a two-event global pool
to require refund of unused final-prefill credits. Discard retirement tests also cover
failed release with mailbox reuse and uncertain completion through shutdown.

All required host checks passed with actual zero exit codes. Serialized device gates
against the pinned Qwen artifact on the RTX 4090:

| Gate | Result | Time | Log under `desktop:~/ribn-lifecycle-2026-09-14/` |
| --- | --- | --- | --- |
| Text lifecycle | 4/4, exit 0 | 76.91 s | `02-completion-text.log` |
| Exact-token runtime | 1/1, exit 0 | 174.22 s | `03-completion-runtime.log` |
| CUDA reference | 61/61, exit 0 | 408.13 s | `04-completion-reference.log` |

`completion-revision.txt` pins the tested commit; `completion-exits.txt` records each
process exit. Kernel arithmetic was unchanged. These gates do not qualify concurrent
handles, readiness, a real VLM, or the still-opt-in GDN scan.

### Owned token driver host gate (`7005fad`)

The first driver increment adds owned token streams, not a concurrent text/model API.
The 19 tests in `crates/runtime/src/driver/tests.rs` cover independent callers, stalled
consumers, retained terminal permits, stop-vector capacity, invalid configuration,
enqueue rejection, cancellation under saturated admission, cancellation without a
device batch, dropped submission/next futures, failed sync/release, shutdown retry,
cancelled shutdown futures, owner drop and worker panic.

Mutation checks observed exit 101, then pass with the mechanism restored:

- `abandoned_in_flight_requests_keep_their_admission_charge_until_retirement`: remove
  the runtime's private retention guard; a discarded route returns its permit while
  its sequence still retains input/options storage.
- `output_credit_return_between_check_and_park_is_not_lost`: omit consumption's wakeup;
  the test pauses the worker after its condition check and before sleep, returns a
  credit, then stalls at its three-second deadline (poll fallback is 60 seconds).
- `queued_submission_and_shutdown_acknowledgements_are_dropped_on_worker_panic`: omit
  exit's explicit queue drain; queued acknowledgement senders survive disconnection
  and the awaiting client hits its three-second deadline.

Required host checks, including CUDA-feature clippy, exited 0. The 19 driver tests
passed 100 consecutive suite runs. These tests exercise the actual driver, not a
separate scheduling simulation. No model arithmetic changed.
[Matched direct/driver measurements](runtime-alignment/README.md#owned-token-driver-7005fad)
record the observed host cost and raw CSV, not inference throughput.

**Device gate passed (2026-09-14, `dc511bd` + later):** desktop was reachable and idle
(RTX 4090, driver 615.71.09, GPU 0% / 33 MiB before the run). The checkout was pulled to
the tested revision and the pinned artifact verified by SHA-256 before running. Both
runtime tests passed serially, exit 0: **2/2 in 199.11 s**, including
`owned_driver_preserves_reference_with_stalled_and_abandoned_peers`. Private log:
`desktop:/tmp/gate1.log`. This qualifies the owned driver for the tested device path.

Superseded blocker (kept for history): desktop qualification and checkout synchronization
could not run because Tailscale was stopped and `ssh desktop` failed hostname resolution.
`owned_driver_preserves_reference_with_stalled_and_abandoned_peers` in the existing
Qwen CUDA runtime test is compiled but unrun. Do not infer GPU qualification from host
success. Once desktop is idle/reachable, run both runtime gates serialized:

```sh
RIBN_MODEL=/path/to/pinned/model.gguf \
RIBN_REFERENCE=$(pwd)/crates/qwen/tests/fixtures/qwen38-code-fill4096-257.tokens \
cargo test --release -p engine-qwen --features cuda --test cuda_runtime --locked \
  -- --ignored --test-threads=1
```

Text preprocessing/decoder injection, a bounded concurrent text facade and ordered
incremental offline batches are now implemented and host-qualified (see the owned text
facade gate below). Subsequent bounded CUDA gates now cover the owned driver, facade,
abandoned/stalled peers and HTTP recovery; see [device re-execution](#device-re-execution-2026-10-16-dc4aed0).
Those passes do not establish arbitrary concurrency, context or failed-drain behavior.

### Owned text facade host gate (2026-09-14)

`crates/text/tests/facade.rs` drives the real `GgufProcessor`, a real GGUF tokenizer
built from synthetic byte-level metadata, the real token driver and the real facade
over a scripted fixture `GenerationExecutor`. Only the device is substituted, so the
preprocessing, delivery, batching and shutdown paths under test are production ones.

Twenty-two tests pass with no failures across consecutive suite runs:

- concurrent callers on cloned handles, plus the async submission path;
- a decode failure settling exactly one request while its peer completes;
- invalid UTF-8 reported with its offending token;
- a terminal flush replacing an incomplete trailing code point exactly once;
- cancellation returning a `Cancelled` terminal for its own request while the owner
  keeps serving (buffered-event preservation is proven by the driver tests, not here);
- oversized prompt/message/prompt-token inputs rejected before admission, with the
  same handle still serving afterwards (permit refund on the failure path);
- ordered per-item batch outcomes with a rejected input between healthy ones, and a
  decode failure inside a batch settling only that item;
- chat stop IDs following the artifact and the rendered prompt, with raw prompts adding
  only the declared IDs;
- shared-handle overload: with every permit retained by stalled consumers, a third
  request and each batch item report `DriverError::Overloaded` for themselves, and
  the permits return once the consumers release;
- a pull-counting iterator proving batch lookahead stays inside the window;
- dropping a batch abandoning delivery without stranding the owner;
- shutdown closing admission while reporting success;
- zero preprocessing workers rejected at assembly;
- a stalled consumer not blocking a peer request;
- the decoded-payload bound rejecting one oversized token for its own request only;
- the rendered-prompt bound during template expansion, with the raw input inside its
  own bound;
- owner failure preserving delivered events, inventing no terminal, and reporting one
  error after which the stream stays exhausted;
- a failed text shutdown reported through its owner, with the same owner retrying and
  succeeding;
- a preprocessing pool whose only worker panics reporting failures to its caller and
  to later submissions instead of queueing them forever;
- permit refund under a two-permit pool where three rejected requests would exhaust
  the pool if any leaked;
- every `drain` assertion requiring exactly one terminal event, so a duplicated or
  missing terminal cannot pass unnoticed.

Required checks at this revision, each with captured exit 0:

```sh
python3 tools/check-boundaries.py
cargo fmt --all -- --check
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo clippy --workspace --all-targets --locked --features cuda -- -D warnings
```

**Device gate passed (2026-09-14):** the migrated lifecycle tests ran serially on the
RTX 4090 with the pinned artifact, exit 0: **5/5 in 95.83 s**
(`an_invalid_token_fails_only_its_own_batch_member`,
`an_unpreparable_batch_input_settles_only_its_own_item`,
`cancellation_is_request_local`, `dropping_a_stream_does_not_poison_a_later_batch`,
`repeated_abandonment_stays_bounded`). Private log: `desktop:/tmp/gate2.log`.

A further device test, `repeated_requests_are_deterministic_and_coherent`, covers what
those gates could not: four identical requests to one owner must reproduce identical
text (greedy sampling has no legitimate reason to change), which catches cross-request
delivery, shared state or a stale continuation. It passed, **1/1 in 19.52 s**, and the
example ran four concurrent callers, an abandoned stream and explicit shutdown
successfully. Private log: `desktop:/tmp/det2.log`.

Observed model behavior under the **superseded** stop policy, **not** a pipeline defect:
with greedy sampling and only `eos_token_id` in the stop list, this artifact can emit
chat special markers (`<|endoftext|>`, `<|im_start|>`) as ordinary text, leak that marker
text into output and continue instead of ending the turn. Repeated suffix prompts
("... Caller 3.") also degenerate into a repeated-token loop. Both are prompt/policy
behavior: identical requests stayed deterministic across indices, and the same prompt at
request index 0 and request index 6 produced identical coherent output. An earlier read
of this as index-dependent corruption was wrong. The stop policy below replaces it.

### Text load capacity host gate

```sh
cargo test -p ribn-text --features cuda --locked cuda::tests
```

The CUDA text assembly's production configuration/engine helpers run on the host
with a pending device-executor substitute. Requested sequence limits 1, 8, 9, 16,
17 and 32 admit that many live sequences, keep one excess request queued, respect
the executor's three-token batch limit and retain enough driver permits for all
active slots. The original `Engine::with_defaults` assembly behavior failed at
requested 9 versus actual 8 (exit 101); the corrected gate passes (exit 0).
The 17/32 cases also protect against the default event grant imposing a separate
sixteen-permit cap. Zero and overflowing request/event envelopes reject through
`TextOwner::load` before accessing the nonexistent fixture artifact or a device.

`TextModel::request_capacity()` is exercised alongside the facade's real
stalled-stream overload test. It describes total permits, not free capacity or
physical batching. No CUDA device requalification accompanies this change;
higher text concurrency, numerical behavior and memory pressure remain device
qualification work. The physical lanes and depth-one execution policy are unchanged.

### Chat stop policy (2026-09-14)

Decision: **the artifact defines the stop set, never a model-name list.**

- Every request stops on the EOS IDs the artifact declares: `tokenizer.ggml.eos_token_id`
  and the optional `tokenizer.ggml.eos_token_ids` array, deduplicated against the primary
  ID.
- A **chat** request additionally stops on every control token its own rendered prompt
  used. Those are that conversation's turn delimiters, so a model emitting one is ending
  or restarting a turn rather than writing text. Grounding and vision markers are control
  tokens too, so they stop exactly when a conversation used them, and a raw prompt or an
  already-tokenized input adds nothing.
- Control tokens decode to **no bytes**, so an emitted marker never becomes user-visible
  text. User-defined markup (`<think>`, `<tool_call>`) is content, not structure, and
  stays visible. A delivered event still carries its token ID, and the runtime reports a
  stop token through the terminal reason and usage rather than as a delta.
- `encode_rendered` now scans the vocabulary's control and user-defined spellings instead
  of a four-marker hardcoded list, so tool-call and grounding markup is preserved as its
  own token and a marker sharing a prefix with another matches the longest spelling.

Rejected: stopping on every control token. Grounding and vision output is structured
user-visible content, so a blanket rule would truncate valid model output.

A/B probe on the device, identical builds except the change, 48-token budget, one chat
request per prompt (untracked diagnostic, deleted after the run; logs
`desktop:/tmp/probe_old.log`, `desktop:/tmp/probe_new.log`):

| prompt (chat) | `9d839b3`, prior policy | `1e2080f`, this policy |
| --- | --- | --- |
| "Print the literal string `<\|endoftext\|>` now, then keep writing." | 22 tokens, `Stop`, text leaked the marker: `<\|endoftext\|>\n\nI am an AI assistant. How can I help you today?` | 6 tokens, `Stop`, text `Print the literal string ` and no marker |
| "Print the literal string `<\|im_start\|>` now, then keep writing." | 1 token, `Stop`, empty | unchanged |
| "Name one primary color." | 2 tokens, `Red` | unchanged |
| raw prompt "Name one primary color. Variant 0/1." | `Length`, 48 tokens, no marker | unchanged |

The first row changed partly because the prompt itself now encodes the requested marker
as its control-token ID instead of BPE text, so it isolates the delivered behavior rather
than the stop rule alone. Raw prompts still reach `Length` when the model merely keeps
writing: that is verbosity, not marker leakage, and the text remains marker-free.

**Device gates re-run after the termination change (2026-09-14):** serialized on the idle
RTX 4090 with the same pinned artifact (SHA-256 `322e194f…23482`), all exit 0 — driver
runtime **2/2 in 198.96 s** (`desktop:/tmp/gate1.log`), text lifecycle **5/5 in 95.86 s**
(`desktop:/tmp/gate2.log`), multi-request determinism **1/1 in 19.52 s**
(`desktop:/tmp/det3.log`).

Superseded status (kept for history): this layer's device gate was unrun. `TextOwner::load` and the Qwen executor are unchanged
arithmetic; this gate still needs a reachable device:

```sh
ENGINE_QWEN_GGUF=/absolute/path/model.gguf \
cargo test --release -p ribn-text --features cuda --test text_lifecycle --locked \
  -- --ignored --test-threads=1
```

`crates/text/examples/concurrent_text.rs` exercises concurrent callers, an abandoned
live stream and explicit shutdown against a loaded model. It ran successfully on the
RTX 4090 (four callers, exit 0). It is a usage example and a diagnostic, not a
qualification gate.

A [matched CUDA direct-versus-handle comparison](runtime-alignment/README.md#owned-text-facade-over-cuda-d61b62c)
now records five alternating pairs per workload at concurrency one. Higher-concurrency
and mixed-arrival frontend-overhead comparisons remain unmeasured; endpoint throughput
measurements do not isolate frontend cost.


## Gemma 4 Metal second-decoder timing (2026-10-20)

Reproducible diagnostic only: one M3 Max, one sequence, the archived 18-token chat
prompt, greedy decoding, the correctness-first host-orchestrated Metal decoder
(`crates/gemma4/examples/metal_latency.rs`). It is not a competitive comparison or an
endpoint measurement. Later internal qualifications below preserve their previous
kernel outputs, not necessarily stock CPU bit patterns. The separate
[same-artifact MLX calibration](gemma4-metal-baseline.md) records current external
model-call timings, full-logit differences and the remaining quality/serving limits.

| Stage | First pass | Split attention |
| --- | --- | --- |
| Host inventory load | 1.81 s | 1.88 s |
| Device upload | 1.69 s | 1.65 s |
| Prefill, 18 tokens | 5.92 s | 0.98 s |
| Decode, four tokens | 5.14 / 5.45 / 5.72 / 5.92 s | 0.32 / 0.40 / 0.70 / 0.55 s |

Per-operation instrumentation of the first pass attributed 25.0 s of 28.1 s to
attention, 2.5 s to linear, 0.47 s to RMS and 0.09 s to rotary. An earlier draft of
this section blamed the naive GEMM; that attribution was wrong. Attention dispatched
one thread per query head (eight threads) and recomputed the query-key dot product
inside the output-channel loop, costing about 119 ms per layer. Splitting it into
score, per-row softmax-reduction and weighted-value kernels with unchanged per-output
arithmetic removed the dominant cost: the archived greedy tokens and the text-facade
path stay identical. A repeat run measured a 0.41 s decode mean, so run-to-run variance
is material at this scale.

With attention fixed, the remaining ~0.4 s/token is consistent with the ~0.5 s/step
linear cost measured in the same run, so a tiled BF16 GEMM is the next candidate.
Command-buffer batching, cache eviction and graph capture remain secondary until a
representative workload is measured.

Four-wide BF16 loads in the linear kernel, with the same ascending-`k` additions,
reduced decode to 0.30 s/token and 18-token prefill to 0.81 s; timing variance also
collapsed to 0.300-0.310 s per token across four steps. The archived tokens and the
engine-metal tests stayed bit-identical.

A first GEMM experiment is a measured loser and was reverted: uploading every matrix
transposed (so threads read adjacent output columns) cost about 30 s of host transposition
for the 15 GB inventory and raised decode to 0.95-1.1 s/token across two runs, versus
0.4-0.5 s with the original row-major weight layout. The per-thread sequential read of
one weight row is faster on this GPU than the coalesced column walk. Keep the original
layout; a tiled kernel must preserve the ascending-`k` accumulation order and beat this
baseline, not merely change the layout.

A second GEMM experiment was qualified but not promoted. Criteria were declared before
running: each output within one BF16 encoding of the exact kernel on random BF16 operands
(five shapes, inner 16-1024), bit-identical dyadic cases, and identical archived greedy
tokens. All three held: at most one ulp, dyadic outputs exact, and the four tokens
identical. Performance did not: 0.35 s versus 0.30 s per decode token and 0.94 s versus
0.81 s prefill under the same conditions, so the warp kernel and its qualification test
were removed. One threadgroup per output issues scalar two-byte loads and one FMA per
element; the four-wide exact kernel needs fewer instructions per element, and better
coalescing did not offset that. Register blocking with wide loads was the next
experiment, rather than another layout or reduction split.

### Paired-row linear qualification (2026-10-20)

On the same 40-core M3 Max, macOS 26.6.2, 128 GiB host memory and Rust 1.98.0 release
build, a four-output-column register block also lost: the 18-token prefill median rose
from 473 to 540 ms and decode from 218 to 284 ms. All checked full logit rows matched,
but restoring the baseline restored its timing. That candidate was removed.

Pairing **input rows** instead reuses each four-wide weight load for two accumulators.
Each output keeps the same ascending-inner-index additions and final BF16 rounding.
At this promotion, pairing was limited to M3 Max, even row counts **8–64**, and inner
widths divisible by four. The later [four-row variant](#four-row-metal-linear-qualification-2026-10-20)
replaces pairs on its qualified sub-scope. Other GPUs, single-token decode, small/odd
chunks, larger batches and non-vector widths retain the original kernel. Linear preparation also rejects
input/weight element products beyond the kernel's u32 address range before allocating
output; checking dimensions alone did not establish safe offsets.

The warmed diagnostic used the pinned BF16 package and archived 18-token chat IDs,
repeated/truncated to lengths 1/2/4/8/18/32/64, with three repetitions per length and
three common teacher-forced continuation tokens. Each process warmed three input tokens
before the sweep and reset continuation for every repetition. Host load/upload are
excluded from the measurements. All 262,144 finite logits after prefill and each
continuation matched baseline captures exactly at every length/repetition; the chat
case also retained its four independent reference winners.

The last baseline sweep and final-kernel sweep, including every repetition:

| Prompt rows | Baseline median ms | Paired median ms | Paired range ms | Median reduction |
| --- | ---: | ---: | ---: | ---: |
| 8 | 294.02 | 256.81 | 253.00–532.22 | 12.7% |
| 18 | 473.66 | 365.44 | 363.69–522.72 | 22.8% |
| 32 | 694.26 | 526.69 | 524.79–531.06 | 24.1% |
| 64 | 1254.59 | 1002.79 | 823.64–1046.75 | 20.1% |

Earlier paired sweeps also improved the 18/32/64-token cases, but timing variance was
substantial in both baseline and candidate rounds, including some unchanged decode
samples above 400 ms. All rounds and outliers were retained; their cause was not
established. These sequential full-model diagnostics justify the bounded selection,
not a serving/SLO result, a decode speedup or a general claim for other Apple GPUs.
Raw source variants, probe and exact logit captures are qualification artifacts. The
existing `metal_latency` example measures one cold request, not this warmed sweep.

The component gate now checks exact source-order results for paired/unpaired rows,
selection boundaries, tiny grids, column tails, non-vector widths and dyadic operands.
The real-package decoder gate compares whole-prompt execution with one-token and
seven-token chunks across all four full logit rows on common histories. Decoder,
executor, text facade and `ribn run-gemma` device paths passed; the materialization test
now skips only a missing device, not shader/pipeline failures. An injected MSL syntax
error falsely passed the old harness and failed the repaired one; restored-source gates
passed. Boundaries, fmt, 389 workspace tests and default/CUDA/targeted Metal clippy passed.

```sh
CARGO_BUILD_JOBS=2 cargo test -p engine-metal --release --locked --test bf16 -- --test-threads=1
# Set RIBN_GEMMA_PACKAGE and RIBN_GEMMA_REFERENCE as for the decoder gate.
CARGO_BUILD_JOBS=2 cargo test -p engine-gemma4 --features metal --release --locked \
  --test metal_decoder --test executor -- --ignored --test-threads=1
# The public-loader gate lives with ribn-text and exercises Native plus the M3 Max
# experimental Graph policy. Its separate Graph ownership/KV gates are documented
# in gemma4-metal-baseline.md#model-local-graph-integration-2026-10-23.
CARGO_BUILD_JOBS=2 cargo test -p ribn-text --features metal --release --locked \
  --test gemma_metal -- --ignored --test-threads=1
```

CUDA device requalification and Metal command-failure/retirement injection were not run;
this changes neither the CUDA path nor the current Metal ownership protocol.

### Shared Metal weight inventory (2026-10-20)

The Metal model no longer retains the complete `Gemma4CpuWeights` loading mirror after
upload. Its immutable Shared buffers now supply CPU token/PLE embedding rows as well
as GPU parameters. The token embedding and tied output head still use one native
buffer. `MetalBf16Buffer::as_slice` borrows that buffer's lifetime; no mutable view or
raw Metal handle escapes. Producers finish initialization or wait for GPU completion
before publication. At `3655ab4`, `newBufferWithBytes:length:options:` copied each
initializer, allowing CPU loading arrays to retire after preparation.

Matched full-package probes on the same M3 Max compared `9188e50` with `3655ab4`:

| Metric | CPU mirror retained | Shared inventory |
| --- | ---: | ---: |
| Owned BF16 weight payload, host + native | 27.80 GiB | 13.90 GiB |
| Process RSS immediately after preparation | 29.16 GiB | 16.23 GiB |
| Maximum RSS during loading/execution | 29.20 GiB | 29.17 GiB |
| Native Metal allocated bytes after preparation | 14,926,430,208 | 14,926,430,208 |

A second shared-inventory run reproduced 16.23 GiB prepared RSS and 29.17 GiB maximum
RSS. That revision removed a retained representation, **not the loading peak**:
preparation still held the entire CPU inventory while constructing Metal buffers. The
[direct-source follow-up](#direct-metal-source-loading-2026-10-20) below removes this overlap.
Native
padding, allocator/driver storage, caches and request workspace are not the encoded
weight payload. After model drop, native allocated bytes returned to about 475 KB even
though process RSS remained nonzero; RSS alone is not evidence of live Metal buffers.

All finite, full logit rows at prompt lengths 18 and 64 and three common teacher-forced
continuations matched the earlier baseline captures exactly. The upload test now mutates
and drops its source before checking both borrowed and copied results. These are
successful-load, shared-view and execution checks, not OOM or failed-command recovery
qualification. CPU oracle and CUDA storage are unchanged. Boundaries, fmt, 389 workspace
tests, default/CUDA/targeted Metal clippy, Metal device/runtime/facade gates and the CLI
passed.

The diagnostic prints phase RSS in KiB, sampled outside its phase timers. macOS `time -l` supplies the maximum RSS; run the built executable so
compiler processes are not part of the measurement:

```sh
cargo build -p engine-gemma4 --features metal --release --locked --example metal_latency
# Set RIBN_GEMMA_PACKAGE and RIBN_GEMMA_REFERENCE as for the decoder gate.
/usr/bin/time -l target/release/examples/metal_latency
```

### Direct Metal source loading (2026-10-20)

After whole-source config/catalog/shape/dtype preflight, Metal reads each parameter
through bounded retained artifact readers into its final Shared buffer. At most 64 KiB
of fallibly allocated scratch stages each synchronous read/copy. No CPU inventory or
parameter-sized encoded upload copy is constructed. EOF is checked before immutable
publication; a failed read or Rust unwind releases the private allocation, and a later
parameter failure releases the entire incomplete set. CPU-slice uploads also copy
directly into Shared storage. All buffer allocations now check host/device limits and
nullable native allocation results before constructing their Rust owners.

Same artifact, M3 Max and toolchain as above; baseline is `3655ab4`. The built
`metal_latency` executable runs the archived 18-token prompt and four greedy decode
steps, with `/usr/bin/time -l` measuring the process rather than Cargo. Two direct-load
runs gave:

| Metric | CPU inventory then upload | Direct source loading |
| --- | ---: | ---: |
| Prepared process RSS | 16.23 GiB | 13.92 GiB |
| Maximum process RSS | 29.17 GiB | 13.93–13.94 GiB |
| Owned encoded weight payload | 13.90 GiB | 13.90 GiB |
| Native Metal allocated bytes after preparation | 14,926,430,208 | 14,926,430,208 |

The peak reduction is about 52%, not a smaller model representation. A separate
18/64-token probe, repeated twice, compared all 262,144 finite BF16 logits at four
prediction rows per length against the earlier captures: every bit matched, including
three common teacher-forced continuations. It dropped the source/package before
inference. Its 64-token maximum RSS was 13.95–13.96 GiB; after model drop, native
allocated bytes returned to about 475 KB while residual RSS remained nonzero.

Device tests cover fragmented reads across the scratch boundary, source independence,
short/trailing input, read errors and Rust panics after a copied prefix, and invalid
lengths before reads. The tiny full-inventory gate compares every parameter against an
independently loaded CPU inventory after pathname removal and source drop; a source
truncated after preflight rejects the partial set and a fresh load recovers. Workspace
tests, boundaries/fmt, default/CUDA/Metal clippy, real decoder/executor/text-facade gates
and the ordinary CLI passed.

This establishes the synchronous Metal loading path, not a global concurrent loading
grant, a hard RSS bound, or serving throughput. Native allocation refusal/OOM and
failed-command hardware recovery were not injected; CPU oracle arithmetic and CUDA
loading/execution were not changed. The diagnostic's phases are now source preflight,
direct preparation and execution, rather than CPU materialization followed by upload.

### Metal command-buffer profile and PLE batching experiment (2026-10-20)

On `42df49a`, the direct Metal decoder (not an endpoint or runtime-scheduler benchmark)
ran prompt prefixes 1/8/18/32/64 on the same M3 Max/package/toolchain, followed by three
common teacher-forced continuations. Each cohort had one warmup and
three measured repetitions; no measured samples were discarded. Temporary tracing
read `GPUStartTime`/`GPUEndTime` after `waitUntilCompleted` and required terminal
`Completed` status, finite timestamps and a positive start time. The installed Metal
SDK documents these as host-time seconds bounding command-buffer GPU execution, not
occupancy, kernel-only busy time or a stall-counter attribution.

Every step issued **795 synchronous command buffers**. At 18-token history, median
per-group sums for the first decode step were:

| Operation | Commands | GPU interval sum | Method wall sum |
| --- | ---: | ---: | ---: |
| Linear | 344 | 49.17 ms | 113.63 ms |
| RMS | 343 | 3.25 ms | 62.53 ms |
| Rotary | 66 | 0.36 ms | 11.48 ms |
| Attention (three kernels per buffer) | 42 | 2.11 ms | 10.23 ms |

Group medians are not additive. Method clocks include allocation, encoding, commit
and wait, but exclude model-side input uploads and output copies. Warmed decode at
that history was about 209.75 ms untraced versus 224.22 ms traced. Timestamp reads and
per-command logging perturb the measurement, so only **untraced** controls determined
promotion. Longer prefill also has substantial CPU work outside these method clocks;
this does not attribute the entire deficit to submission or transfer these proportions
to CUDA.

The bounded candidate normalized all independent token/layer PLE rows in one RMS
call for chunks up to 64 tokens, preserving each row's reduction and BF16 arithmetic.
It reduced command count to 754. Longer chunks retained layer-at-a-time normalization
rather than multiplying normalization workspace by the layer count. The initial
unbounded grouping was not eligible for promotion because of that workspace growth.
Every full 262,144-element row stayed finite and bit-identical to earlier captures,
including continuations and the independent archived greedy winners at 18 tokens.

The predeclared gate was at least 3% warmed median decode improvement with no material
prefill regression. Two untraced baseline runs yielded a pooled decode median of
209.081 ms. The final bounded candidate's two run medians
were 201.290 and 203.771 ms; its pooled median was 202.819 ms, **2.995% lower**, just
below the gate and with meaningful run-to-run variation. Prefill medians did not regress.
The candidate was **removed**, not promoted by rounding or relaxing the criterion.
All tracing was removed too; no Rust implementation or test changes remain from this
experiment. The next useful experiment is a concrete larger command chain with declared
numerical and workspace bounds, not an assumption that another kernel variant fixes
synchronous orchestration. No serving/SLO gain or hardware-failure recovery is claimed.
The [2026-10-23 bounded rescreen](gemma4-metal-baseline.md#bounded-packed-ple-rms-rescreen-2026-10-23)
revisits this grouping after the later chains and GELU changes, using matched randomized
trials within each warmed process; it does not revise this older experiment's result.

### Metal linear-RMS chain qualification (2026-10-20)

Baseline `153153f` and the fixed chain ran the same direct decoder, M3 Max, package,
reference and toolchain as the preceding profile. Each candidate call encodes the
existing linear and RMS kernels in one serial command buffer, preserving the private
BF16 projection and every per-output arithmetic step. Query/key/value normalization,
attention-output normalization, FFN down-projection normalization and PLE output
normalization use the chain. This removes 216 intermediate readback/upload pairs and
synchronous waits from the measured 795-command schedule: **579 command buffers per
step**, with no reduction in kernel count or new kernel-selection policy. Whole-chain
preflight, private buffers and checked terminal publication follow
[the resource contract](../docs/resource-protocol.md#synchronous-metal-projection-chains).

The predeclared promotion gate was exact finite full logits, at least 10% warmed
median decode improvement, and no material prefill or memory regression. Three
untraced baseline runs and three candidate runs each used one warmup plus three
measured repetitions per prefix, followed by three common teacher-forced decode steps.
The baseline was rebuilt in a detached `153153f` worktree; controls before, between
and after candidate runs retained all timing variation. Medians pool the measured
repetitions; no measured sample was removed:

| Prompt tokens | Baseline prefill | Chain prefill | Baseline decode | Chain decode |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 207.99 ms | 169.18 ms | 208.50 ms | 169.19 ms |
| 8 | 251.66 ms | 209.76 ms | 206.65 ms | 168.89 ms |
| 18 | 377.69 ms | 338.64 ms | 208.94 ms | 173.44 ms |
| 32 | 533.87 ms | 510.63 ms | 209.95 ms | 174.15 ms |
| 64 | 915.27 ms | 860.75 ms | 212.13 ms | 173.10 ms |

The pooled decode median across all five histories was **209.286 → 172.273 ms**,
**17.69% lower**. Individual run medians were 208.90/211.35/208.96 ms for baseline and
164.00/174.14/174.15 ms for the chain; the first candidate run was appreciably faster
than the other two and remains included. These are diagnostic direct-decoder timings,
not serving goodput, latency SLOs or evidence about CUDA. No measured prefill cohort's
pooled median regressed.

Every 262,144-element full-logit row was finite and bit-identical to earlier native
captures for 1/8/18/32/64-token prompts and all three common continuations. All four
independent archived greedy winners at 18 tokens remained unchanged. Additional
baseline/candidate comparisons covered **65/511/512/513**-token prompts and three
continuations apiece, including the sliding-window boundary and larger original-kernel
linear grids. The release decoder gate also compared whole-prompt execution with
chunks 1/7 through reset and common-history continuation. Primitive tests independently
retain the host arithmetic references and now check compound-versus-standalone exact
bits across paired/unpaired shapes, multiple norm groups, column/inner tails, scaled
and unscaled RMS, invalid whole-chain rejection and recovery.

External `/usr/bin/time -l` measurements included loading and the complete probe.
Short-prefix baseline maximum RSS was 14.075/14.067 GiB versus 14.032/14.030 GiB for
the chain. The 65–513-token probes peaked at 14.257 versus 14.233 GiB. This supports
no observed peak-RSS regression under these conditions, not a hard device-memory or
concurrent-loader bound. Each chain owns two output-sized native buffers through the
wait and avoids the former projection CPU copy and upload.

All 393 workspace tests, 13 release Metal primitive tests, materialization tests,
ignored real decoder/executor/text gates and the four-token `run-gemma` CLI passed.
Production boundaries, formatting, default/CUDA/targeted-Metal clippy also passed.
Independent source/SDK ownership review found no introduced issue. Hardware command
failure, null-handle/OOM injection and recovery were **not** exercised; healthy-device
results do not qualify them. Only macOS Metal was device-tested; desktop CUDA remains
unavailable. The CPU oracle and CUDA implementation are unchanged.

### Metal FFN input-chain qualification (2026-10-20)

The next fixed chain normalizes FFN input once and projects it through the gate and
up weights in one serial command. It reuses the existing RMS and linear kernels,
paired-row selection and BF16 intermediate boundary. Both weights validate before any
allocation/submission; both outputs publish only after successful completion. The
private workspace is one input-sized normalized buffer and two output-sized buffers.
The model drops each native output owner after its CPU copy rather than deliberately
holding both owners alongside both CPU copies. CPU GELU/multiply and subsequent
linear/RMS chains are unchanged. This removes another **84 waits per step**, taking
the command schedule from 579 to **495**, without changing kernel count.

The predeclared gate was exact finite full logits, at least 5% warmed median decode
improvement, and no material prefill or memory regression. The baseline was the saved
`260cbd2` executable from the preceding qualification; the same probe/package/reference
and M3 Max/toolchain ran three baseline and three candidate processes, with controls
before, between and after candidate measurements. Each prefix had one warmup and three
measured trials plus three common continuation steps. All measured samples remain:

| Prompt tokens | Baseline prefill | Chain prefill | Baseline decode | Chain decode |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 174.26 ms | 157.25 ms | 173.78 ms | 158.89 ms |
| 8 | 215.28 ms | 201.61 ms | 172.98 ms | 159.26 ms |
| 18 | 340.24 ms | 320.60 ms | 175.09 ms | 159.81 ms |
| 32 | 509.11 ms | 489.18 ms | 176.55 ms | 160.10 ms |
| 64 | 889.29 ms | 864.56 ms | 178.74 ms | 160.94 ms |

Pooled warmed decode median: **175.379 → 159.809 ms**, **8.88% lower**. Individual
run medians were 173.57/175.77/177.29 ms baseline and 159.89/159.78/159.81 ms candidate.
No prefix's pooled prefill median regressed. These remain direct-decoder diagnostics,
not serving throughput, SLOs or a result transferable to CUDA.

All 262,144-element rows were finite and bit-identical to the prior captures for
1/8/18/32/64-token prefixes and three common continuations. Fresh baseline/candidate
65/511/512/513-token comparisons also stayed exact. Archived four-token greedy winners,
reset/chunks 1/7 through the real decoder gate, executor/text delivery and the actual
four-token CLI remained unchanged. New primitive cases compare **both** distinct-weight
outputs with three standalone calls across scaled/unscaled RMS, paired/unpaired rows
and tails; invalid right-weight/scale/epsilon cases cannot publish a partial pair and
are followed by a valid recovery, including aliased immutable weights.

External maximum RSS across short-prefix runs was 14.035–14.053 GiB baseline versus
14.021–14.031 GiB candidate. The larger-prefix probes peaked at 14.245 versus 14.233
GiB. This is no observed peak-RSS regression, not a global device-memory grant.
All 395 workspace tests, 15 release Metal primitive tests, materialization and ignored
real-model gates passed, as did boundaries, formatting and default/CUDA/targeted-Metal
clippy. No new unsafe interop or asynchronous ownership was introduced. Physical OOM,
null-handle and hardware-fault recovery remain unqualified; CPU oracle and CUDA code
are unchanged, and the shut-down desktop was not contacted.

### Metal rotary projection-chain qualification (2026-10-20)

Baseline `a6e9fb4` and the normalized projection with optional rotary used the same
M3 Max, package, reference and toolchain. Query/key projections now encode linear,
RMS and split-half rotary in one serial command. The kernels and both BF16 boundaries
are unchanged. Full factor/head/row geometry validates before any output allocation;
rotation owns a third output-sized private buffer through terminal completion. The
model prepares immutable factors once per layer and shares them across query/key
consumers. Standalone and chained rotation share one validator/encoder; the obsolete
`linear_rms` API/callers were replaced, not retained as compatibility wrappers.
This removes another **66 waits**, leaving **429 command buffers per step**. It is
not a new kernel-selection policy or universal graph.

The predeclared gate was exact finite full logits, at least 5% warmed median decode
improvement, and no material prefill or memory regression. Three baseline and three
candidate processes used one warmup plus three measured trials per prefix and three
common continuation steps, with bracketed controls and all measured samples retained:

| Prompt tokens | Baseline prefill | Chain prefill | Baseline decode | Chain decode |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 157.71 ms | 144.12 ms | 158.93 ms | 143.92 ms |
| 8 | 195.60 ms | 185.57 ms | 157.95 ms | 145.37 ms |
| 18 | 298.74 ms | 286.61 ms | 156.71 ms | 146.39 ms |
| 32 | 460.29 ms | 445.87 ms | 157.28 ms | 146.50 ms |
| 64 | 764.82 ms | 771.32 ms | 159.48 ms | 151.19 ms |

Pooled warmed decode median: **157.907 → 145.906 ms**, **7.60% lower**. Run medians
were 159.87/157.70/157.87 ms baseline versus 149.03/144.88/145.61 ms candidate.
The first baseline and candidate processes had large, unexplained timing swings:
measured decode samples reached about 396 ms in each, and candidate 64-token prefill
included 854/854/1762 ms. These were not discarded or attributed to occupancy,
stalls, thermal state or another application without evidence. The 64-token pooled
prefill median rose **0.85%**; that is a nonmaterial observed trade-off for promotion,
not an established prefill improvement. Later controls reproduce the decode direction,
but these noisy diagnostics are not serving/SLO evidence or a CUDA result.

All full 262,144-element rows remained finite and bit-identical for
1/8/18/32/64/65/511/512/513-token prefixes and three common continuations. Archived
four-token winners and real decoder reset/chunks 1/7, executor/text and CLI gates
remained unchanged. New primitive cases compare the complete chain against three
standalone calls, including both non-symmetric factor halves, multiple heads,
scaled/unscaled RMS, paired/unpaired rows and inner/column tails. Bad head/factor
geometry rejects without publication and a subsequent valid call recovers.

Short-probe maximum RSS was 14.020–14.024 GiB baseline versus 14.013–14.028 GiB
candidate; larger-prefix probes peaked at 14.239 versus 14.226 GiB. No material
peak-RSS regression was observed; this is not a concurrent memory grant. All 397
workspace tests, 17 release Metal primitive tests, materialization and ignored
real-model gates passed, along with boundaries, formatting and default/CUDA/targeted-
Metal clippy. Independent source review found no introduced issue. Hardware failure,
null-handle/OOM injection and recovery remain unqualified. CPU oracle/CUDA code are
unchanged, and the shut-down desktop was not contacted.

### Metal attention-input projection chain (2026-10-23)

The [matched chain qualification](gemma4-metal-baseline.md#attention-input-projection-chain-2026-10-23)
keeps input RMS and Q/K/V preparation in one serial command using unchanged kernels.
The selected E4B path removes 90 more waits after packed PLE: 388 → 298 commands per
step. It passes exact full-row, bounded-workspace, application and matched runtime/shape
gates; the linked owner records timing variability and the lack of serving/pressure
acceptance. Attention now also rejects mismatched K/V extents before allocation.

### Metal attention-output projection chain (2026-10-23)

The [bounded output-chain screen](gemma4-metal-baseline.md#attention-output-projection-chain-2026-10-23)
keeps attention, output projection and RMS in one serial command with unchanged kernels
and BF16 boundaries. Selected E4B steps remove 42 more waits: 298 → 256. Exact full-row,
64/65-row and 576/577-key continuation/fallback gates pass. Matched runtime and whole-chunk
aggregate gates pass, with substantial timing variability and some per-process cohort
regressions retained by the linked evidence owner. No serving, physical-pressure or
other-device acceptance follows.

### Metal native main-FFN chain (2026-10-27)

The [matched main-FFN screen](gemma4-metal-baseline.md#native-main-ffn-chain-2026-10-27)
retains RMS, Gate/Up, exact table/product and Down/RMS in one serial command for native
M3 Max/E4B chunks through 64 rows. Selected short-history steps remove 42 waits,
256 → 214; larger chunks retain the prior path. Integer BF16 product encoding is required
for CPU subnormal parity. Complete logits/KV, actual decoder/executor/text gates and
three-process paired timings pass. The linked owner records the failed naive multiply
preflight, corrected control-counter harness, native byte envelope and limits.

### Metal exact GELU table qualification (2026-10-20)

A refreshed temporary profile on `ba1906a` observed **429** synchronous commands
per step at that revision and timed host GELU separately. GPU timestamps were read after
successful completion; commit/wait clocks excluded allocation/encoding, unlike the
earlier whole-method clocks. Timestamp reads and trace output perturb model wall time,
so only untraced executables determined promotion:

| Step | Traced wall | Commit/wait sum | GPU interval sum | CPU GELU sum |
| --- | ---: | ---: | ---: | ---: |
| 18-token prefill | 298.01 ms | 202.24 ms | 111.88 ms | 59.03 ms |
| 64-token prefill | 751.36 ms | 477.69 ms | 365.29 ms | 206.26 ms |
| Decode after 18 tokens | 158.42 ms | 133.66 ms | 55.68 ms | 3.39 ms |

These are medians of per-step sums and need not add to the wall median. GPU intervals
are not kernel-only busy time or occupancy/stall attribution. All profiling changes
were saved privately and reverted before implementation.

The prepared Metal model now owns a **128 KiB GELU table**, generated once through the
unchanged CPU reference. Lookup replaces only Metal's repeated host GELU evaluation,
not the CPU oracle, GPU kernels, multiply, soft cap or command schedule. An exhaustive
committed host test compares every **65,280 finite BF16 encoding** against a separate
single-value reference call, including signed zeros and subnormals, and verifies all
**256 non-finite encodings** still reject. Runtime checks reject those encodings before
table indexing; no approximate polynomial, tolerance relaxation or silent fallback
was introduced. The shared model owns the table; sequences borrow it.

The predeclared promotion gate was exact exhaustive/full-model parity, at least 15%
median improvement at 64-token prefill, and no material decode/RSS regression. Three
untraced baseline and three candidate processes used the same M3 Max, package,
reference/toolchain and one warmup plus three measured trials per prefix, with bracketed
controls and all measured samples retained:

| Prompt tokens | Baseline prefill | Table prefill | Baseline decode | Table decode |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 261.23 ms | 144.41 ms | 301.82 ms | 142.66 ms |
| 8 | 185.39 ms | 163.44 ms | 146.55 ms | 144.23 ms |
| 18 | 287.18 ms | 227.69 ms | 147.29 ms | 144.61 ms |
| 32 | 447.04 ms | 334.65 ms | 148.60 ms | 145.37 ms |
| 64 | 748.42 ms | 528.40 ms | 149.74 ms | 147.32 ms |

The primary 64-token prefill median fell **29.40%**, 748.423 → 528.402 ms. Its baseline
run medians were 772.72/742.52/748.42 ms versus 528.17/528.51/527.62 ms for the table.
Pooled warmed decode was 147.970 → 144.759 ms. The 1-token baseline cohort had large
process-start timing swings; its apparent gain is not a clean startup-speed result.
These remain direct-decoder diagnostics, not serving/SLO or CUDA evidence.

All 262,144-element full-logit rows stayed finite and bit-identical for
1/8/18/32/64/65/511/512/513-token prefixes and three common continuations. Archived
greedy winners, real decoder reset/chunks 1/7, executor/text and actual CLI gates passed.
Short-probe maximum RSS ranges overlapped: 14.013–14.023 GiB baseline versus
14.014–14.023 GiB candidate; larger-prefix probes peaked at 14.225 versus 14.226 GiB.
The table adds 128 KiB of mathematical state and its preparation uses a separate
128 KiB input vector, not additional weight read/copy staging or a global memory grant.

All 397 workspace tests, the release exhaustive table test, materialization and ignored
real-model gates passed, along with boundaries, formatting and default/CUDA/targeted-
Metal clippy. Native Metal code is unchanged from the preceding qualified delivery.
Hardware failure/OOM recovery remains unqualified, and the shut-down desktop was not
contacted.

### Four-row Metal linear qualification (2026-10-20)

After exact GELU tabulation, the existing profile identified matrix multiplication as
most of the remaining 64-token prefill cost. This experiment reused each four-wide BF16
weight load across **four input rows** rather than two, with independent FP32 accumulators
and unchanged ascending-inner-index additions and final BF16 rounding. It is a separate
MSL entry/pipeline, leaving the qualified original/pair kernel unchanged. It is not the
previously rejected four-output-column block or CUDA quantized four-rows-per-warp path.

Automatic selection is limited to **M3 Max, rows 32–64 divisible by four, inner widths
divisible by four**. Other eligible even 8–64-row chunks retain two-row blocking; other
GPUs, single-token decode, odd/small/large chunks and non-vector widths retain the
original kernel. Whole input/weight/output u32 geometry still validates before output
allocation or encoding. The new kernel needs no additional buffers or command submissions.

The predeclared gate was exact full logits, at least **10% faster 32- and 64-token
prefill**, and no material decode/RSS regression; the experiment would stop after two
matched sweeps if it clearly lost. On the same 40-core M3 Max, pinned package, reference,
macOS and Rust release configuration, three interleaved baseline (`84993ab`) and candidate
processes each used one warmup plus three measured trials per prefix and three common
teacher-forced continuations. All measured samples were retained:

| Prompt tokens | Baseline prefill | Four-row prefill | Baseline decode | Four-row decode |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 142.67 ms | 142.41 ms | 143.46 ms | 142.79 ms |
| 8 | 162.99 ms | 162.16 ms | 144.82 ms | 144.38 ms |
| 18 | 227.79 ms | 228.71 ms | 146.02 ms | 145.53 ms |
| 32 | 331.61 ms | 231.74 ms | 142.79 ms | 145.53 ms |
| 64 | 525.35 ms | 361.82 ms | 145.91 ms | 147.01 ms |

The primary prefill medians fell **30.12%** at 32 tokens and **31.13%** at 64. Per-process
32-token baseline medians were 331.43/333.90/330.63 ms versus 231.74/229.61/269.64 ms;
64-token medians were 537.24/522.66/523.23 versus 361.82/362.74/360.47 ms. The slower third
32-token candidate run was retained. Pooled decode changed 144.823 → 145.176 ms (+0.24%);
the 32-token-history cohort rose 1.92%, while the decode kernel itself is unchanged.
No occupancy/stall or timing-variance cause was established. These are direct-decoder
latency diagnostics, not serving/SLO evidence or a CUDA result.

A further matched warmed sweep covers every intermediate newly selected row count:
36/40/44/48/52/56/60-token prefill medians were respectively
358.37/375.52/409.43/425.98/456.43/480.16/505.47 ms baseline versus
253.18/260.67/278.62/290.00/317.10/328.16/343.79 ms candidate. Full 262,144-element rows
were finite and bit-identical at all these sizes, 1/8/18/32/64 and unblocked
65/511/512/513-token prefixes, through three common continuations each. Component
coverage now checks every selected row count against independent host source-order
math, tiny grids, column tails, long inner loops and non-vector fallbacks. A host
selection test protects device/row/vector boundaries. The real-model decoder gate now
also compares 32/64-token chunks against qualified 16-token chunks through continuation;
the archived 18-token case alone could not exercise the new kernel.

Short-probe peak RSS remained overlapping, 14.009–14.017 GiB baseline versus
14.006–14.013 GiB candidate; intermediate-row sweeps peaked at 14.038 versus 14.031 GiB.
All 398 workspace tests, 17 release Metal primitive tests (with expanded cases), the
release selector test, exhaustive GELU gate, materialization, ignored decoder/executor/text
and actual CLI gates passed, alongside boundaries, formatting and default/CUDA/targeted-
Metal clippy. Independent read-only review found no actionable issues in arithmetic,
addressing, selection or synchronous ownership; parent device qualification supplies
the numerical evidence. Hardware fault/OOM recovery remains unqualified. No desktop
access, cache eviction, scheduling change or universal graph was introduced.

### Metal native-handle lifetime ablation (2026-10-21 UTC)

The [external MLX calibration](gemma4-metal-baseline.md) left large Ribn latency swings
unexplained. A controlled caller-lifetime experiment changed only the probe: retain
one outer autorelease pool, but optionally drain a nested pool after each completed
decoder call, inside that call's clock. Kernels, math, command count and model stayed
unchanged. Two interleaved sweeps retained all measured samples:

| Probe cadence | Decode median, process 1 / 2 | Peak RSS range |
| --- | ---: | ---: |
| One pool around the sweep | 331.87 / 316.24 ms | 14.021–14.027 GiB |
| Nested pool per completed call | 331.77 / 145.88 ms | 13.9665–13.9666 GiB |

Per-call draining reduced host retention but did **not** consistently eliminate the
latency swings. A later unchanged control measured 144.29 ms median decode. This does
not establish pool cadence as their cause or justify a latency-improvement claim.

Source inspection did expose a narrower ownership problem: the backend's nullable
command-buffer/encoder factories returned autoreleased `+0` handles, retained them for
Rust, but left their autoreleases dependent on an external pool. Rust execution workers
do not install such a pool. A small real-device probe isolated the effect without the
large model: 10,000 synchronous 1×1 BF16 linear calls inside one outer pool, checking
`0.5 × 0.5 == 0.25` every time. Two runs per implementation sampled current RSS:

| Completed calls | Old factory RSS, KiB | Local factory-pool RSS, KiB |
| ---: | ---: | ---: |
| 1 | 15,216–15,248 | 15,136–15,200 |
| 1,000 | 17,152–17,184 | 15,296–15,328 |
| 5,000 | 24,480–24,512 | 15,328–15,424 |
| 10,000 | 33,776–33,792 | 15,328–15,440 |

Native device allocation stayed **475,136 bytes** at every sample in both variants.
This is growing host-side retention, not a growing GPU allocation; no allocator/object
type profiler was used. RSS is supporting evidence, not a formal memory bound.

The backend now scopes only those two known autoreleased factories with local pools,
retaining owned Rust wrappers **before** each pool drains. Returned command/encoder
owners remain live through the same encoding and terminal wait. Null checks, retained
resource mode, serial ordering, output publication and failure scope are unchanged;
no global pool framework or frontend cleanup obligation was added.

Two full-model runs with the fix peaked at 13.958–13.964 GiB. Their decode medians were
143.27/143.91 ms, but the unchanged 144.29 ms control precludes attributing the earlier
large timing swings to this fix. Full logit rows remained finite and bit-identical at
1/8/18/32/64/65/511/512/513-token prefixes through three common continuations.

All 398 workspace tests, 17 release Metal primitive tests, selector/GELU gates,
materialization, ignored decoder/executor/text and the actual CLI passed, as did
boundaries, formatting and default/CUDA/targeted-Metal clippy. Existing device gates
now exercise owned native handles after each creation pool has drained; no flaky RSS
threshold was added to the permanent suite. Independent source/SDK review found no
soundness or lifecycle findings. Hardware failure/OOM recovery remains unqualified;
this changes neither kernels nor CUDA execution, and the desktop was not contacted.

### Bounded eight-row CUDA prefill (2026-10-21 UTC)

The Gemma CUDA lane following `d481274` uses fixed eight-row groups and scalar tails,
with model-owned bounded workspace, full-history KV and last-row-only heads. Selection
is prepared once for the qualified RTX 4090/E4B geometry, 8–4096 cache reach and a
sufficient scratch ceiling. Other scopes retain scalar execution. It does not batch
separate requests, add overlap or alter scheduling policy.

[Independent numerical and ownership qualification](gemma4-reference/README.md#bounded-eight-row-cuda-prefill-2026-10-21-utc)
precedes promotion: a mathematically derived GEMM error bound, exact same-operand
attention checks, 12 distinct archived histories across 19 prediction evaluations,
continued/reset/rejected work, actual facade/CLI execution, and sanitizer gates. Full
logit distributions differ; this remains experimental greedy execution, not sampling or
quality-equivalent acceptance.

Predeclared performance/memory gates were **at least 25% lower prefill latency at both
18 and 64 tokens**, **no more than 5% decode regression**, and **at most 32 MiB additional
charged workspace**. Same checkpoint, RTX 4090, driver 615.71.10, toolkit 13.3.73 and
cuBLAS 13.6.0.2. One immutable probe binary toggles only the scratch ceiling between
one-row and eight-row preparation and asserts the selected width. Both configurations
reserve 4096-position caches. Source/package handles drop before inference.

Three fresh processes per mode run sequentially in order scalar, batch, batch, scalar,
scalar, batch. Each prefix has one warmup trial and three measured trials, reset before
each trial and three common teacher-forced continuations. Prefixes cycle the archived
18-token chat IDs to lengths 1/8/18/32/64/65/128. They are **not** real long-prompt quality
cases or serving arrivals. The clock includes the decoder call and complete owned
262,144-entry host logit row; checks, argmax and file writes occur outside it. No sample
was removed except the declared warmups. This session serialized its model processes,
builds and tests; other host activity was not controlled. Immediately subsequent GPU
utilization samples can include a completed prior process; no model processes overlap.

Pooled medians over nine measured trials per prefix (ms):

| Prefix tokens | Scalar prefill | Eight-row prefill | Prefill change | Scalar decode | Eight-row decode |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 17.872 | 18.143 | +1.52% | 17.891 | 18.203 |
| 8 | 121.650 | 20.166 | −83.42% | 17.970 | 18.239 |
| 18 | 271.934 | 68.436 | **−74.83%** | 18.356 | 18.487 |
| 32 | 489.859 | 73.452 | −85.01% | 19.076 | 19.343 |
| 64 | 1004.294 | 146.959 | **−85.37%** | 19.290 | 19.548 |
| 65 | 1020.323 | 163.552 | −83.97% | 19.277 | 19.380 |
| 128 | 2053.070 | 297.035 | −85.53% | 19.691 | 19.952 |

All three process medians retain the primary improvement. At 18 tokens scalar medians
were 271.98/271.61/271.85 ms versus 68.17/68.90/68.44 ms; at 64 they were
1005.15/1003.94/1004.29 versus 146.39/147.70/146.96 ms. Pooled decode is
**19.076 → 19.334 ms (+1.35%)**, with every prefix cohort below +1.75%. No decode gain
is claimed; one-token prefill is also slightly slower.

Prepared-model pool charges are **14,935,433,032 → 14,939,560,928 bytes**, an increment
of **4,127,896 bytes (3.94 MiB)**. Including one full cache set gives
15,170,314,056 → 15,174,441,952 bytes; charges return to zero after dropping the model
and continuation in every probe process. These are charged physical allocations, not
nominal token declarations or total device/driver memory. Process peak RSS ranges are
14.5436–14.5438 GiB scalar and 14.5683–14.5701 GiB batch; RSS is not a hard bound or an
attribution of the difference. All declared timing/charged-workspace gates pass.

Three GEMM and three attention tests pass, including the new complete-output encoded-
operand bound and maximum-reach eight-row cases. GEMM/attention and the full batched
window decoder pass memcheck with zero errors; attention also passes initcheck. The
prepared-capacity/vocabulary admission repairs have failing-before regressions, and a
whole-chunk-validation mutation fails the new automatic-eight-row gate. The independent
reviews' concrete findings are resolved. Required host gates and the unaffected Metal
model/facade regressions pass; hardware drain/OOM failures were not injected.

These are **single-sequence warmed diagnostics**, not SLO goodput, concurrency scaling,
end-to-end serving throughput or incumbent parity. Raw sources/locks, all samples,
complete rows, independent-reference/scalar comparisons and immutable binaries remain
qualification artifacts. The next competitive decision needs a matched incumbent and
serving/capacity evidence, not an extrapolation from this prefill speedup.

### Gemma CUDA native/stock/prepared diagnostic (2026-10-21 UTC)

The [same-artifact comparison](gemma4-cuda-baseline.md) reproduces the unchanged native
control against released Transformers eager execution, then probes stock compiled
advancing decode with explicit static cache ownership. It records full rows, preparation
costs, compilation specializations, allocator/RSS statistics and direct-call latency.
Native prefill remains substantially behind stock eager; compiled execution is neither
bit-identical nor a tested Rust-loadable backend. This is not another promotion gate,
engine-level incumbent baseline, or serving/SLO/quality-equivalence claim. Existing native
qualification remains unchanged; no production engine code changed for this diagnostic.

## Synthetic CPU benchmark

```sh
cargo run -p ribn --release --example host_overhead
```

This reports median/p99 nanoseconds per warm scheduling iteration and median
cost per row at concurrency 1/8/32/128. Each run warms 128 iterations and measures
10,000. The mock immediately completes work and allocates its result vectors;
those costs are included. The aggregate event bound has room for both completed output
and the next batch's reserved credits. Admission, preparation, and teardown are
outside the samples.

It is not inference throughput, a GPU benchmark, or an old/new speed comparison.
Do not turn inverse iteration time into advertised model tokens/second. Run
multiple samples on a controlled CPU and record the revision, compiler, CPU,
load, and configuration. Allocation instrumentation and old/new matched traces
remain required before an allocation-free or performance-improvement claim.

## Hardware qualification

Only run on an authorized, idle qualification GPU. Follow the existing device
ownership checks in the CUDA probe/benchmark scripts. Do not interrupt another
GPU process to make this test pass.

Obtain a token fixture from the pinned llama.cpp reference or the independently
qualified old path, with the exact artifact and greedy/input semantics. The
fixture has exactly three lines:

```text
gguf:sha256:<64 lowercase hexadecimal digits of the complete GGUF>
<prompt token IDs separated by spaces>
<expected generated token IDs separated by spaces>
```

Use real token IDs and hash, not the placeholders. At least four output tokens
are required by the harness; use diverse 256/1024-token fixtures for the cutover
gate. Keep the reference engine revision, command/settings, tokenizer/template
semantics, and artifact metadata beside each fixture. The candidate implementation
must not produce its own expected output.

```sh
RIBN_MODEL=/absolute/path/model.gguf \
RIBN_REFERENCE=/absolute/path/reference.tokens \
cargo test -p engine-qwen --features cuda --test cuda_runtime \
  prepared_qwen_matches_reference_and_preserves_cancelled_peers \
  -- --ignored --exact --test-threads=1
```

The harness verifies artifact identity and exact tokens at concurrency 1/2/8/9,
then in-flight decode cancellation with peer progress at 2/8. It uses no stop
filter so reference tokens, including any stop ID, are compared directly. It
checks terminal ownership and shutdown. This is a starting harness, not all
cutover evidence: add mixed inputs, constrained memory, repeated admissions,
real cancellation/fallback combinations, and fault-path device evidence as
specified in the roadmap.

### Block-table KV addressing (2026-10-12, `ee689d7`)

Slice 4b needs continuation that lives in fixed token blocks rather than one
contiguous per-sequence cache, so attention must resolve each logical token
through a block table. That addressing change is qualified separately from the
cache, reuse policy and recurrent checkpoints it exists to enable, because a wrong
token address corrupts output silently.

The kernel now has one templated device body and two entry points. The contiguous
entry point folds the block lookup away at compile time; the paged entry point
resolves `block_table[token / block_tokens] * block_tokens + token % block_tokens`
and runs the identical per-row arithmetic. Nothing else changed: no arithmetic was
reordered, so results are compared by bit equality rather than a tolerance.

Serialized on an idle RTX 4090 (driver 615.71.09), all exit 0:

```sh
cargo test --release -p engine-nvidia --features cuda --test paged_attention --locked \
  -- --ignored --test-threads=1
cargo test --release -p engine-nvidia --features cuda --test cuda_reference --locked \
  -- --ignored --test-threads=1
RIBN_MODEL=/absolute/path/model.gguf \
RIBN_REFERENCE=$(pwd)/crates/qwen/tests/fixtures/qwen38-code-fill4096-257.tokens \
cargo test --release -p engine-qwen --features cuda --test cuda_runtime --locked \
  -- --ignored --test-threads=1
```

- **paged attention 2/2** (0.98 s): scores and outputs are bit-identical to the
  contiguous path for a shuffled block table and for the identity table, at tokens
  {1, 4} (aligned block), 5 (partial tail block), 8 (exact block multiple), 7 with 3
  causal prefill rows, and 10 with 10 rows. Geometry that cannot be satisfied is
  rejected before launch: a table too short for the requested extent, and a zero
  block size.
- **CUDA reference 61/61** (412.80 s): the restructure of the contiguous entry point
  into a templated body changed no observed behavior on the existing kernel suite.
- **Qwen CUDA runtime 3/3** (231.54 s): the pinned Qwen3.8-27B UD-Q4_K_M artifact
  (SHA-256 `322e194f…23482`) still matches its reference tokens, including the
  constrained-authority and stalled/abandoned-peer gates, through the modified kernel.

This qualifies addressing only. The runtime contract is now page-granular — a model
declares a fixed KV page size (16 tokens for Qwen), a sequence holds whole pages, and its
capacity is `pages * page_tokens` — but materialization is still one contiguous run per
sequence and the paged kernel entry point has no runtime consumer. Page tables, growth,
shared ownership and hybrid validity rules remain unimplemented.

### Page-table continuation (`e36e097`)

The host continuation contract is page-granular (`e82f58d`): a model declares a fixed
KV page size — 16 tokens for Qwen — a sequence holds whole pages, its capacity is
`pages * page_tokens`, and reservation charges whole pages. Production decode then
addresses state through a page table: appends map the logical token to a physical one
and attention runs the paged entry point with the table as a device buffer. With the
table currently identity-bound, the expected result is bit equality with the previous
contiguous addressing, not a tolerance.

Serialized on an idle RTX 4090, all exit 0: full `cuda_reference` 63/63 (444 s),
`paged_attention` 2/2, Qwen CUDA runtime 5/5 (556 s), `cuda_http` 1/1, in-flight
retirement 1/1, and `sampled_final_chunk_row_matches_serial_hidden_bits` under
compute-sanitizer memcheck (205 s, 0 errors).

One caveat is recorded rather than resolved: a first full `cuda_reference` run failed
`heterogeneous_decode::different_capacities_preserve_hidden_and_all_continuation_bits`
and the same commit then passed that test in isolation (36 s) and in a second full run
(63/63). That test builds eight states of up to 4096 tokens, so device-memory pressure
across a long binary is the likely cause, but the failing assertion was not captured
and the cause is unattributed.

This qualifies addressing, not sharing. A sequence still owns its own run of pages, so
page indices are consecutive and no cache, reuse policy or growth path exists yet: a
shared arena with non-identity tables and page growth is the next slice, and the
per-request device allocation it removes is still on the admission path.

### Matched cross-engine decode baseline (2026-10-18)

The first comparison against an incumbent on the same artifact, same client and same
sampling. `llama-server` 0.4.1-dev (build 11138) and `ribn serve` at `ec47bc5` both served
the pinned Qwen3.8-27B UD-Q4_K_M GGUF (SHA-256 `322e194f...23482`), greedy, streaming
`/v1/completions`, 16 requests of 32 output tokens, AIPerf 0.13.0 with identical
manifests. Ribn ran `--context-length 512 --max-sequences 8`; llama.cpp ran `-c 4096 -np
8 -fa on` (the same 512 tokens per slot). Serialized on an idle RTX 4090; raw runs,
manifests and server logs live in `desktop:~/ribn-qualification/cross-engine-2026-10-18/`.

| case | ribn tok/s | llama.cpp tok/s | ribn TTFT p95 | llama TTFT p95 | ribn ITL | llama ITL |
| --- | --- | --- | --- | --- | --- | --- |
| short-c1 (isl 16) | 32.68 | 39.89 | 116 ms | 182 ms | 27.8 ms | 20.1 ms |
| mixed-c4 (isl 64) | 32.16 | 70.89 | 2283 ms | 743 ms | 76.0 ms | 36.8 ms |
| mixed-c8 (isl 64) | 35.46 | 118.58 | 4804 ms | 1455 ms | 116.8 ms | 32.9 ms |

Single-stream latency is at parity: ribn's c1 TTFT is 1.6x better and its ITL 1.4x worse.
Everything else is concurrency scaling, and the two engines diverge there by 2.2x at c4
and 3.3x at c8. This is a decode/prefill *batch* gap, not a per-token kernel gap.

#### Where the batch gap lives

The historical 2026-10-18 direct-engine diagnostic used
`qwen_serving_bench --concurrency=N --tokens=32 --dispatch-diagnostics`, skipping HTTP:

| concurrency | engine tok/s | host enqueue per step | batch width used |
| --- | --- | --- | --- |
| 1 | 32.3 | 8.8 ms | 1 (single row) |
| 2 | 47.4 | 18.9 ms | 2 |
| 4 | 70.7 | 35.9 ms | 4 |
| 8 | 94.7 | 32.5 ms | 8 |

Batched lanes were in use (`decode_lane_calls_by_width[8]`). These observations
motivated reducing host enqueue and state-access overhead; they do not establish that
launch count alone explains the endpoint deficit.

**Launch-count correction:** the earlier **3,862 launches/step** statement is superseded
for this workload by the [fresh decode-only decomposition](#member-descriptor-native-ab-2026-10-23):
**1,764 control / 1,252 candidate** at eight rows. Every one of 31 decode steps has
that count, with scalar prefill separated by embedding boundaries and totals reconciled
to kernel events. The historical count is not reconciled to this split; do not use
it, its per-kernel breakdown or a derived per-launch cost as a current baseline.

Historical `RIBN_DIAG` instrumentation (not committed) reported 0.94 s and 1.50 s
of host enqueue for an eight-row 30-token prefill, against 34 ms for an eight-row decode
step. That prefill was one submission regardless of the `--prefill-chunk` setting.
These scoped observations motivate prefill investigation; they neither separate all
current endpoint costs nor establish the cause of the whole competitive gap.

`ec47bc5` reduced decode host cost by normalizing each batched GDN q/k buffer in one launch instead of
two per member: decode host enqueue fell from 39.1 ms to 32.5 ms per eight-row step
(-17%) and engine throughput rose from 91.2 to 94.7 tok/s. Device gates on that commit:
`cuda_reference` 63/63, Qwen CUDA runtime 5/5, `cuda_http`, in-flight retirement, and the
sampled-final bit-identity memcheck, all exit 0.

Member-dimension state kernels and persistent metadata/graphs were candidate mechanisms,
not established fixes for the whole gap. The default-off descriptor candidate below
changes decode state access without a resident arena. Prefill descriptors, arenas and
graphs remain unimplemented. Most projection/norm/residual operations already batch
members, so removing state loops cannot imply an eightfold reduction in all launches.
Graph replay still needs stable captured addresses and completion-safe updates; that
does not establish a universal arena prerequisite.

#### Member-descriptor candidate (device gates pending)

`QwenLoadOptions::member_descriptors` (default `false`) and native benchmark flag
`--member-descriptors` select the candidate only for cross-request decode lanes.
The text/HTTP defaults, single-row dispatcher fallback and same-sequence prefill path
are unchanged. One member-major u64 packet holds layer pointer pairs and page metadata;
convolution, recurrent update, KV append and attention consume it, with disjoint score
scratch for concurrently executing members. Q/k gathering changes copies, not normalization.
The eager candidate explicitly waits for metadata publication, retaining its host packet
through failed copies/drains. The native screen includes this wait; it is not hidden overhead.

Host evidence: packet assembly, missing/misaligned addresses, page reach and count bounds
pass without CUDA. Rust CUDA-feature checks compile the binding and optional load/benchmark
path. A local C++ syntax probe with CUDA built-ins stubbed also passes; it does **not**
compile GPU code or substitute for NVRTC, numerical or lifetime qualification.
Actual allocation lengths/dtypes now validate for every row before any model launch,
including allocations replaced through a public mutable state accessor.

The new ignored gate
`heterogeneous_decode::member_descriptors_preserve_hidden_and_all_continuation_bits`
compares the existing batched path on common histories at widths 1/3/8, nonadjacent reaches,
page crossings and score growth/reuse. It checks finite, bit-identical hidden state, all KV,
recurrent matrices and convolution history, plus greedy selections. It also replaces a
late member's matrix with an undersized allocation and requires rejection without peer
mutation. The existing heterogeneous-capacity gate now uses one-token pages to exercise
multiple table entries without rounding its exact-capacity cases.

On 2026-10-18 at `b0c7c2a`, serialized qualification on the RTX 4090 (driver 615.71.09,
CUDA toolkit 13.3) passed the isolated descriptor gate, all **64 CUDA reference tests**,
all **five Qwen runtime tests**, all **six text-lifecycle tests** and the HTTP device
gate. This includes NVRTC compilation, the changed capacity/page gate and in-flight
retirement coverage. The pinned Qwen artifact above was verified before running.

Two descriptor memcheck attempts were operator-interrupted for excessive qualification
cost: the unfiltered run exited 15 after 636 seconds; the five-member-kernel-filtered
attempt was also interrupted. Neither is a pass, regardless of the tool's zero-error
summary on termination. The cause of sanitizer overhead was not measured.

On **2026-10-23**, a fresh release/locked build at `5bdad2a` passed the isolated
descriptor gate again on the RTX 4090, now driver **615.71.10** (previously 615.71.09),
CUDA toolkit **13.3.73**, Compute Sanitizer **2026.2.1.0**, and Rust **1.98.0**.
The normal run passed **1/1 in 34.31 s, exit 0**, without a missing-device skip.
The source worktree was clean; the same pinned model and fixture were hash-verified.
The test executable came from Cargo's compiler-artifact JSON, with SHA-256
`afe91ea1f6851e081ec87d6bc6a5df771cadeca38ebbb693442afad679be8010`.

**One filtered memcheck completed: 1/1 passed in 1,156.12 s, exit 0,
`ERROR SUMMARY: 0 errors`.** It used that same executable and isolated test,
`--tool memcheck --error-exitcode 99`, and additive `--kernel-name kernel_name=`
filters for `gdn_conv_members`, `gdn_gather_qk_members`, `gdn_state_members`,
`kv_append_members` and `attn_members`. The predeclared 2,400-second deadline
was not reached. Each device launch began idle, with no other compute process.
This completes memcheck for those five member kernels on the tested cases, not for
the control-path kernels or the whole engine; it is not racecheck or synccheck.

Evidence: `desktop:~/ribn-qualification/member-descriptor-memcheck/20261023T103624Z/`.
The plan, source, executable, hashes, prelaunch snapshots, exact invocations, logs and
actual build/test/memcheck exit codes are retained there.

Before promotion, exercise descriptor reuse/disable, partial enqueue and failed drain
through the asynchronous serving owner; host tests and this eager numerical gate do not
qualify those lifetimes.
The matched native c1/c4/c8 screen below now measures the unchanged candidate, but
does not meet its advancement gate. Broader performance and endpoint comparison remain
unqualified if that scope is revisited. Keep prompt/chunk/token/diagnostic settings
matched and count launches, host enqueue, device/total time, descriptor/score payloads
and output agreement. NVRTC acceptance and bounded bit equality do not establish
competitive improvement; the candidate remains default-off.

##### Member-descriptor native A/B (2026-10-23)

**Decision: keep default-off and defer further lifetime qualification without new
bottleneck evidence.** The candidate misses the predeclared advancement gate: at least
5% median paired throughput gain at **both c4 and c8**, exact outputs, at most 3%
c1 throughput regression, and at most 3% mixed throughput/TTFT-p95/ITL-p95 regression.
Only the c4 gain condition fails. These are experiment thresholds, not product SLOs.

[Structured results](qwen-trace/2026-10-23-member-descriptor-ab.json) record a clean
`8b7294f` worktree, pinned artifact/fixture, Rust 1.98.0 and fresh release/locked build
with `CARGO_BUILD_JOBS=2`. RTX 4090, driver 615.71.10, toolkit 13.3.73.
One executable, archived by SHA-256 before execution, serves both modes.

Three fresh-process pairs per width, with off/on order varied, use the existing
five-token timing prompt, 32 outputs/request, warp GEMV, serial prefill, diagnostics
and token logging.
Only `--member-descriptors` changes selection. No execution warmup; load/compilation
are outside the native measurement. Gains below are medians of paired ratios, not
ratios of marginal medians.

| Concurrency | Control tok/s range | Candidate tok/s range | Paired gain |
| --- | --- | --- | --- |
| 1 | 32.25–32.32 | 32.29–32.35 | +0.12% |
| 4 | 72.00–72.09 | 75.05–75.19 | +4.26% |
| 8 | 94.72–94.78 | 101.28–101.37 | +6.91% |

Median total host submit wall time falls **1,042.57→711.32 ms** at c4 and
**1,983.98→1,286.85 ms** at c8. These totals include prefill and 31 decode submissions,
routing, enqueue/event recording and the candidate's publication wait—not CPU-only
or decode-only time. TTFT is essentially unchanged; mean ITL improves about 5.82%/10.62%.

Three paired replays of the existing eight-request mixed-arrival trace use four active
slots, four queued slots, a 500 MiB continuation grant and eight-token prefill chunks.
All runs complete **8/8, 136 tokens, no failures/rejections**; actual decode widths
are 1/2/3, not four throughout. Paired throughput improves only **0.31%**
(marginal medians 14.894→14.941 tok/s). TTFT p95 is 8,238.7→8,207.4 ms and ITL p95
291.636→291.538 ms. Joint diagnostic-deadline completions remain **0/8 in both modes**.
All 2,496 unprofiled burst tokens and 816 mixed tokens agree on common inputs;
this is not new full-logit/state or independent task-quality qualification.

Six separate Nsight Systems profiles—not throughput repetitions—split scalar prefill
from 31 decode steps. Decode launches/step are **1,204→1,204 / 1,444→1,252 /
1,764→1,252** at c1/c4/c8. Median summed kernel durations are respectively
26.946→26.924 / 39.110→37.281 / 51.403→47.055 ms. The candidate still misses the
roadmap's below-600 launch target. Profiled clocks do not establish native timing,
occupancy or stall causes.

Source-derived burst payloads at the page-rounded 48-token reach: c4/c8 selected packets
are 4,248/8,472 bytes **each on host and device**; all prepared lane packets retain
9,576/37,128 bytes each. Selected batched score payload grows from 4,608 bytes to
18,432/36,864 bytes. These are requested payloads, not measured live/backing/allocator
peaks, physical-byte grants or a memory-pressure gate.

Evidence: `desktop:~/ribn-qualification/member-descriptor-ab/20261023T185422Z/`.
All **30 model processes, six profile exports and both analyzers exit 0**.
A cached utilization sample falsely blocked the first candidate before launch; a frozen
idle-settle amendment resumed only unexecuted cases, retaining the first control and
original order, sample count, thresholds and deadline. No capture was overwritten or
excluded based on performance. Results were verified before desktop SSH became
unreachable; a subsequent artifact-copy/additional cross-check attempt was blocked.

The modest burst benefit does not close the mixed workload deficit. Do not promote,
widen scope or repeat this unchanged screen. Remaining serving-owner lifetime gates
still apply if new evidence justifies revisiting the candidate. Select the next bounded
prefill/projection experiment from the measured remaining costs.

## Experimental CLI

```sh
cargo run -p ribn-cli --features cuda -- run \
  --model /absolute/path/model.gguf --prompt 'Explain a mutex.' --max-tokens 128
```

`run` uses the shared text facade and reports preparation/memory diagnostics on stderr.
It targets the existing Qwen text artifact path, not arbitrary GGUF architectures.
The duplicate `local` command is retired. Compare token fixtures rather than raw stdout:
`run` streams without an added newline. A token limit cutting a UTF-8 code point ends
with one replacement character when the facade flushes its terminal.

## Initial text HTTP qualification (2026-10-13)

The [loopback text adapter](../docs/serving.md) at `5451cff`, with neutral request-default
validation completed at `efbffdb`, retains the existing contiguous/full-reachable Qwen
path. Host TCP tests pass on macOS and Fedora: real HTTP parsing, tokenizer, facade and
driver, with only device execution substituted. The trailing-byte disconnect test failed
before the post-body EOF observer; ordinary Hyper disconnect coverage alone missed it.
A separate invalid-reach regression failed with 503 before the text error classifier
recognized permanent enqueue rejection; it now returns 400 with peer/capacity recovery.

The CUDA HTTP gate passed **1/1, exit 0** on an idle RTX 4090, driver 615.71.09,
Rust 1.98.0 and the pinned `Qwen3.8-27B-UD-Q4_K_M.gguf`, SHA-256
`322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`:

```sh
ENGINE_QWEN_GGUF=/absolute/path/Qwen3.8-27B-UD-Q4_K_M.gguf \
  cargo test --release -p ribn-server --features cuda --test cuda_http --locked \
  -- --ignored --exact http_matches_direct_text_and_preserves_disconnect_peer_progress \
  --test-threads=1
```

It uses a 512-token context and two sequences. Raw completion and templated chat text/usage
match direct facade generation over the same eight-token histories. An abandoned
64-token stream with trailing input preserves a healthy eight-token peer, later stream
capacity and explicit shutdown. This is transport/ownership parity, not an independent
numerical reference, broad context qualification or a device peak-memory measurement.
The initial release build exited 101 because Cargo reused obsolete NVIDIA dependency
metadata after the checkout rename; rebuilding `engine-nvidia` resolved it without a
source change. The successful gate took 24.19 seconds including model loading.

The actual CLI also passed readiness, chat SSE terminal/usage and SIGTERM shutdown
(exit 0), using:

```sh
ribn serve /absolute/path/model.gguf --bind 127.0.0.1:18000 --model-name ribn \
  --context-length 512 --max-sequences 2 --max-requests 2
```

Two **client compatibility smokes**, not matched performance comparisons, then passed:

| Client | Conditions | Observed outcome |
| --- | --- | --- |
| AIPerf 0.13.0 at `5451cff` | Completion SSE, concurrency 1, four measured requests, one warmup, seed 42, 16→8 tokens, server usage | 4/4 completed; 64 input / 32 output tokens |
| vLLM 0.31.0 `bench serve` at `efbffdb` | OpenAI completion SSE, random target 16→8, two measured requests, no execution warmup, concurrency 1, Poisson 1 RPS, seed 42 | 2/2 completed, zero failures; 31 input / 16 output tokens |

Both use cached `Qwen/Qwen3.8-27B` tokenizer revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, greedy decoding and enabled model stops.
AIPerf's `--use-server-token-count` exercises usage-only frames. With
`HF_HUB_OFFLINE=1`, its local-directory argument initially failed before requests;
`--tokenizer Qwen/Qwen3.8-27B --tokenizer-revision <revision>` resolved the cached snapshot.
The vLLM random dataset sets `ignore_eos=true`, which Ribn now supports; the explicit
`--extra-body '{"temperature":0,"ignore_eos":false}'` keeps the served policy identical
to the recorded run. Earlier native attempts exited 0 despite **zero successes and two
HTTP 400 failures**; verify exported success/failure counts, not just the client's
process exit. Non-neutral penalties and requested logprobs remain explicit rejections.

Commands, exports and logs are retained outside Git by source revision. No incumbent
engine was measured in these smokes; there is no representative mixed-load, SLO-goodput,
quality-matched competitive or RSS/device-peak result. Those remain the
[serving acceptance matrix](../docs/roadmap.md#first-competitive-acceptance-matrix).
The desktop's four untracked SIMT artifacts retained their content fingerprints.

## Gemma HTTP CLI qualification (2026-10-21)

`ribn serve-gemma` reuses `run-gemma`'s package loader/backend selection and the existing
bounded HTTP server. There is no new scheduler, stream producer or cleanup owner.
Gemma rejects `--device`, `--context-length` and `--max-sequences` before binding;
transport limits remain configurable. The existing Qwen `serve` configuration is retained.

The actual release CLI passed on the M3 Max, macOS 26.6.2, Rust 1.98.0 and the pinned
[Gemma 4 E4B BF16 package](gemma4-reference/model.toml), with 665 active tensors and
14,926,026,836 logical encoded bytes:

```sh
CARGO_BUILD_JOBS=2 cargo build -p ribn-cli --features metal --release --locked
ribn serve-gemma /absolute/path/gemma-package --bind 127.0.0.1:0 \
  --model-name gemma --max-requests 4 --max-output-tokens 64
```

The functional capture retained each response, process log and actual exit code:

- Raw completion and chat each returned JSON and SSE with matching visible text,
  four committed output tokens, one finish and one `[DONE]` per stream. The 18-token
  chat's text matched the independent CPU corpus's first four predicted tokens
  `[236776, 126641, 568, 15721]`, decoded with the checkpoint tokenizer. Raw/SSE parity
  alone is not an independent numerical comparison.
- A real image-token placeholder returned 400 without faulting the owner. Abandoning a
  64-token raw stream after a generated delta preserved an already-submitted healthy
  four-token chat peer, incremented the abandoned counter and left readiness at 200.
- A missing package retained faulted health until SIGTERM, then exited 2 as expected.
  A separate real-package process observed in `loading` settled its pending load and
  exited 0 after SIGTERM. The normal serving process also exited 0 on SIGTERM.

Startup reported four HTTP slots and 16 application permits; those are different bounds,
not physical batch width or a memory guarantee. Default/Metal/CUDA CLI host tests cover
model-specific option rejection, including the real binary before bind/device access.
The full host gates and feature checks passed. No CUDA device was contacted or CUDA HTTP
request executed. This is a bounded Metal transport/lifecycle gate, not new full-logit,
long-context, failed-device-drain, physical-memory, latency or SLO-goodput qualification.

## Gemma endpoint client qualification (2026-10-21)

The existing [endpoint runner](README.md#audited-endpoint-runner) now passes actual
AIPerf 0.13.0 / Python 3.13.11 captures against the release Metal Gemma CLI at `ccf624f`
on the M3 Max and pinned BF16 package above. One owned server process served two
sequential client runs and exited 0 on SIGTERM. Telemetry was disabled; unified-memory
capacity and physical peaks were not measured.

Both runs used seed 42, nominal input mean/stddev 18/6, eight profiling requests, an
eight-token output cap and greedy decoding with model stops enabled. The c1 run had
zero warmups; c2 had two excluded warmups. Each audit accepted all eight profiling
requests, with actual input lengths 6, 11, 14, 18, 19, 23, 25 and 29 (145 tokens total).
Each counted 62 output tokens, not the 64-token cap total: seven length finishes and
one six-token stop. Prompt hashes, per-prompt usage and finish reasons matched across
runs. That is accounting evidence, not independent output-quality acceptance.

A separate CPU-only delayed-stream producer and the real pinned client check the clock
boundaries. It emits an empty event, two nonempty content chunks, a separate empty
finish, usage and `[DONE]`, retaining independent producer monotonic timestamps and
client exports. TTFT starts at actual client initiation and ends at first content;
`request_latency` ends at last content, excluding the later terminal frames. The
reported ITL divides the content window by reported output tokens minus one: five
reported tokens and two visible chunks in the fixture do not imply five observed token
timestamps. The integration regression checks these boundaries without a device.

Two runner controls were corrected from this evidence: zero warmup now omits the
positive-only AIPerf CLI option instead of passing an invalid zero, and the misleading
`e2e_ms` manifest key is replaced by `last_content_ms`. The older endpoint tables below
called this same clock “content E2E”; their labels are clarified without changing the
stored measurements. This does not establish stream-completion E2E. Initial failures and
successful captures are retained. All protocol/accounting audits still report
`performance_metrics_audited = false`: aggregate metric accuracy, offered-arrival
lateness, load-generator capacity, competitive latency/SLO goodput, CUDA execution,
physical-memory and numerical quality remain separate gates.

## Text stop ownership repair (2026-10-22)

The shared text facade now treats codec state, decoded-text stop matching and their
delivery lease as one request-owned output state. Three defects were reproduced:

- Stop-container and string spare capacity bypassed the retained-request envelope.
  Those allocations now join input and stop-token capacity before preprocessing queue
  insertion. A blocked-worker case rejects a one-character string with 1 MiB capacity
  and an empty stop vector with 65,536 slots; a joint-bound case accepts exactly 256
  bytes and rejects 257 without normalization or copying.
- A matched stop still decoded/finalized its discarded suffix. Buffered invalid GGUF
  UTF-8 or an oversized Gemma fallback expansion could turn successful stopping into
  an error. Matching now drops codec/matcher scratch unflushed, preserves committed
  token IDs/usage, and retains the grant while observing actual runtime settlement.
  Owner failure after a match still reports an owner error, not fabricated success.
- Withheld stop prefixes could exceed the declared delta bound when released on a later
  chunk or at termination. The bound now applies after matching as well as inside the
  model codec; overflow remains request-local and a healthy peer still progresses.

The initial updated facade tests ran against unchanged production source at `fbe43fc` in
an isolated worktree: six regressions failed, while the other 37 passed. A further assembly
case failed before the bounded matcher workspace was included in aggregate overflow
validation. The repaired facade passes all 43. The tests use actual GGUF/Gemma processors
and the real driver; only token execution is scripted. Existing owner-failure fixture races were fixed by
keeping decode held until polling observes the injected fault, without relaxing their
error assertions. Normal/cancelled finalizers, cancelled prepared replies, early scratch
release and terminal exhaustion retain their protection.

A newly built release Metal CLI also passed the pinned full-package HTTP gate: ordinary
raw/chat JSON and SSE still produce the archived four-token chat text, while
`stop=["mutex"]` yields exactly `A ` with one stop terminal in both JSON and SSE.
Each stop response reports 18 input / 3 committed output tokens; the third token was
already committed, not made visible or retroactively refunded. Media rejection, abandoned
stream/healthy-peer progress, readiness, faulted loading and shutdown during loading also
passed. The three serial server exits were 2 (expected missing-package failure), 0 and 0.
The binary SHA-256 is `cd884c8275b62f738eafc5d6c3c67524585ff2047426cb0e4e1abe470af1615a`.

This is text ownership/protocol qualification, not new decoder arithmetic, physical peak
memory, competitive serving or CUDA execution evidence. CUDA-feature compilation does not
close the separately recorded Gemma executor preflight/failed-retirement gaps.

## Gemma executor transaction and teardown (2026-10-22)

Whole-batch validation and exact preparation/submission identity remain in place.
The malformed-offer cases reproduce internal-contract violations, not a normal HTTP
request failure: a bad later prefix, duplicate sequence, zero inputs, missing final
output, premature decode or aggregate limit excess previously followed one or two real
CPU decoder calls. Preparation now rejects those offers before any decoder call.
Logical cursor, input and remaining output commit only after all rows succeed.

The original cleanup policy in `7abb830` was overbroad. Its two retention tests required
permanent ownership after a completed CPU error; those expectations did not establish a
need for quarantine. Corrected tests instead require safe teardown without resuming the
invalid continuation. Both fail against `7abb830`; the preflight and identity tests still
pass there. This corrects the earlier characterization of all four as necessary regressions.

Creation and completion now capture the same concrete backend model. CPU errors leave no
asynchronous accesses; Metal waits for terminal command completion before reporting status.
CUDA settlement drains its associated stream, including transfer-only failure, retires
the retained compute operation and explicitly closes sequence caches. An unfinished CUDA
key and completed cache layers remain model-owned if value/later allocation fails.
Failed construction cleanup stays reachable even without an admitted decoder; at most one
such bundle may remain pending. CUDA binding borrows the caller's shared model handle so
failure does not consume its retirement owner; the unused combined decoder-load constructor
was removed rather than retaining that ownership trap.

Execution remains closed after failure. Successful completion/close permits destruction;
failed completion or partial close retains the same resources and charges for retry.
Defensive Drop attempts the same teardown and retains its complete envelope only if cleanup
fails or unwinds. Poisoned execution is never reopened, and uncertain free is never blindly
retried. This is teardown, not an in-process model/device recovery framework.

Nine host cases cover the retained transaction checks, completed-error/unwind teardown,
pending completion, partial close, close-unwind fencing, partial construction without a
decoder and unresolved cleanup during Drop. Cleanup injection uses real pool leases, including a lease acquired
before construction fails, and verifies retention followed by refund after successful
retirement. The six existing CPU executor cases also pass. Independent source review found
the partial-construction retry-owner gap during implementation; the model-owned construction
guard and borrowed binding interfaces address it. These host cases do not inject physical
CUDA driver failure.

The local release Metal executor and existing full-row chunk/continuation gates pass.
The CLI HTTP gate also passes with binary SHA-256
`7a3d0fa2d8ae5e5349402bfc9b4ce46229a456a20eef41a8ca268baff6e93a88`: raw/chat JSON and SSE,
decoded stops, media rejection, abandonment/healthy-peer progress, readiness, faulted loading
and shutdown while loading. Its serial process exits are 2 (expected missing-package fault),
0 and 0. Workspace tests, boundaries, formatting, default/CUDA/Metal Clippy and Metal-feature
host tests pass. No kernel arithmetic or numerical acceptance policy changes.
### CUDA device requalification (2026-10-22)

On the restored RTX 4090, driver 615.71.10/CUDA Toolkit 13.3.73, glibc 2.43 and project
Rust 1.98.0, six release-only device cases pass, serialized with two Cargo build jobs.
All nine package files match the pinned E4B revision
`ee0ef6023621cff504d758262d4e04895a5af4a2`. The existing scalar partition/full-window
decoder, eight-row executor and text-facade cases still pass against the unchanged CPU
reference. Two new cases cover device ownership:

- Restrict pool grants to one complete key, then to a complete first layer plus the next
  key. The following value cannot reserve. Partial construction retires the actual CUDA
  allocations, publishes readiness and returns their full charges. Subsequent construction
  and a finite full-row forward succeed; repeated cache close does not refund twice, and
  dropping the last model owner
  returns total granted bytes to zero. This is pool-budget refusal, **not driver OOM**.
- Admit a real sequence, then begin thread-local capture on its own stream. The synchronous
  step and shutdown encounter a real driver capture/synchronization error. All admitted
  charges remain held while capture is invalidated. End capture, retry shutdown, and verify
  sequence charges return, execution cannot resume and final destruction refunds the model.
  No driver result is mocked, illegal memory accessed or GPU reset attempted. Captured
  operations are **not executed**; this does not qualify a failed running kernel or uncertain
  physical free. See NVIDIA's [stream-capture contract](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__STREAM.html).

The real CUDA `serve-gemma` HTTP gate also passes with release binary SHA-256
`4821c97c19bf64ec51a11a00a1bd79ba9008b612e61a94441d401d87729a56a6`.
It checks raw/chat JSON and SSE, the archived four-token chat continuation, decoded stop
`mutex` yielding `A ` with 18 input/3 committed output/21 total usage, media rejection,
abandonment with healthy-peer progress, readiness, loading/faulted health and SIGTERM cleanup.
Its three serial processes exit 2 (expected missing-package fault), 0 and 0.

Reproduce the device cases with the pinned local package and archived CPU reference:

```sh
CARGO_BUILD_JOBS=2 RIBN_GEMMA_PACKAGE=<package> RIBN_GEMMA_REFERENCE=<reference.json> \
  cargo test -p engine-gemma4 --features cuda --locked --release \
  --test cuda_decoder --test executor_cuda --test text_path_cuda \
  -- --ignored --test-threads=1
```

These functional/ownership checks do not establish physical peak memory, arbitrary CUDA
fault recovery, sampling quality or competitive serving acceptance.

### State-only prefill Metal fallback (2026-10-30)

The required state-only operation added at `448a3a0` omits the CPU readout, but
Metal still calls its existing full-head/readback path before acknowledging progress.
A first local decoder observation passes its assertions but fails supervision on an
unverifiable `vmmap` observer cookie. That scope stays closed and unqualified; its
planned executor stage never starts. No unchanged decoder retry follows.

A fresh execution-only scope runs that never-started executor test using the exact
previously built binary, SHA-256
`0c507e10a30fdd144e05b92955262e34f3c4338e2ec8efc36c0854064bcb7428`.
Source hashes still match `448a3a0`; no rebuild, kernel, precision or scheduler change
is involved. On M3 Max/macOS 26.6.2 (25G83), native BF16 Metal execution through the
real Engine processes the archived 18-token chat prompt with default 16-token chunks,
returns all four archived greedy tokens and shuts down successfully. This exercises
one intermediate state-only chunk through the retained full-head completion path.
The test and independent watchdog both exit zero; all recorded worker/guard groups
are absent, without forced model cleanup. Maximum sampled group RSS is
14,952,071,168 B, below the frozen 28 GiB screen, not a physical-memory peak bound.

The replacement observer is dyld's own `DYLD_PRINT_LIBRARIES` logging inside the
owned test process, not another `vmmap` subprocess or weaker ownership checks. It
records 602 loaded-image paths/UUIDs; the smoke-test images keep the same identities.
All nine package pins and the archived CPU reference match before and after execution.
The execution manifest SHA-256 is
`6b143bc28e233b15211df924845174db20421946e7c603677ff0f55af57a290b`;
the independent audit is
`80a58993e9e922cae0af5b2b9d45c37052c50cba5bda95e772815bb9e9b8c0d3`.

This qualifies the existing Metal fallback's Engine integration, not head-free device
execution, full-logit distributional equivalence, HTTP, exclusive host/GPU access or
performance. The failed decoder observation remains separate. CUDA integration still
needs its own device gate.

## Heterogeneous-capacity decode (2026-10-16)

**Promoted for the existing whole-decode lane.** Compatible KV model geometry/dtype no
longer requires equal token capacities. The kernels already attend each member's cache
sequentially on one stream; the old member-major score slicing imposed an unnecessary
uniform stride. Decode now reuses executor-owned `[q_heads][max_capacity]` scratch,
viewed with each member's own score stride. Same-sequence prefill retains its separate
query-row dimension. Each request keeps its original continuation allocation/grant;
there is no KV padding, new kernel, scheduler change or additional execution depth.
Single rows, more than eight rows, mixed phases and incompatible model layouts retain
their existing fallbacks. Experimental mixed-offer grouping remains off by default.
The v0 diagnostics key `incompatible_kv` replaces `nonuniform_kv`; historical artifacts
retain their original schema keys.

The predeclared budget was one implementation/qualification cycle. Promotion required
at least 10% median throughput gain on both mixed c4/c8 cohorts, no more than 5%
short-c1 p95 last-content latency regression, unchanged cohort outputs/core usage, and the numerical and
lifecycle gates below. These are experiment criteria, not product SLOs.
[Structured results](qwen-trace/2026-10-16-heterogeneous-decode.json) retain the source
patch/binary identities, all run results, numerical scope and native dispatch counts.
The control is the retained `bf76a6a` release CLI; candidate and control ran on the same
RTX 4090 with three alternating-order A/B, B/A, A/B repeats of the
[previous endpoint cohort](#first-equal-artifact-endpoint-diagnostic-2026-10-16).

Medians of three run-level measurements; 16 profiling requests plus two warmups/run,
seed 42, output cap 32, actual usage and the same client/tokenizer/settings:

| Cohort | Control output tok/s | Candidate output tok/s | Gain | Control → candidate p95 last-content latency ms |
| --- | ---: | ---: | ---: | ---: |
| Short c1 | 28.23 | 28.26 | Within run variation | 1134.71 → 1132.95 |
| Mixed c4 | 21.99 | 31.72 | 44.2% | 6423.94 → 4648.60 |
| Mixed c8 | 21.98 | 34.92 | 58.9% | 13342.88 → 9067.41 |

All 18 protocol/usage audits and raw-clock checks passed. All **144 paired profiling
visible outputs and core usage counts matched**. This is not an invertible token-ID
comparison or distributional-quality proof. At the existing illustrative 5 s TTFT /
200 ms average TPOT / 10 s last-content latency objectives, c8 improved from 1/16 to 15/16
passing requests in every repeat. Its TTFT p95 barely changed (5006.06 → 4958.60 ms),
while average-TPOT p95 fell from 297.11 to 165.86 ms. Prefill/projection throughput and
remaining mixed-offer fallback therefore remain investigation targets, not a reason to
rewrite logical policy. This finite cohort does not establish production saturation,
fairness, population tails or quality equivalence to llama.cpp.

### Numerical, addressing and retirement evidence

The new capability test first failed on the original uniform-capacity guard (exit 101);
the host selection test likewise rejected different capacities. After the change:

- Full-model capacity-invariance test passed for widths 1/2/4/8, small capacities 2–9,
  mixed capacities through 4096, and smaller capacities on reused lanes. Real distinct
  prefixes plus two teacher-forced steps gave finite, bit-identical hidden values,
  selected tokens, KV contents and all recurrent matrices/convolution histories against
  a uniform-capacity control. Both batch instances use the candidate implementation:
  this is a metamorphic addressing check, **not an independent model oracle**.
- Shared-score attention at active extents 1, 2, 4096 and 262144 matched the unchanged
  owned-slice per-member operator bitwise, with overlapping score views queued before
  readback. Compute Sanitizer memcheck passed with **zero errors**. These are sparse
  primitive inputs, not a 262144-token model-history or long-context-quality gate.
- The independent Qwen runtime reference/cancellation/admission tests passed **5/5**
  (553.66 s), HTTP parity/disconnect recovery **1/1**, and low-level in-flight retirement
  at eight/nine rows **1/1**. The reference still covers the 257-token prompt and
  eighteen-token agreed continuation, not broad numerical acceptance.

```sh
ENGINE_QWEN_GGUF=/absolute/path/model.gguf cargo test --release -p engine-nvidia --features cuda --test cuda_reference --locked heterogeneous_decode -- --ignored --test-threads=1
cargo test --release -p engine-nvidia --features cuda --test cuda_reference --no-run --locked --message-format=json-render-diagnostics
# Set EXE to that target's compiler-artifact.executable, not a guessed target path.
EXE=/absolute/path/from-cargo-compiler-artifact
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 "$EXE" heterogeneous_decode::shared_scores_preserve_attention_bits_through_model_context_limit --ignored --exact --test-threads=1
```

The same native eight-request workload then preserved all 960 model-step tokens
(832 prefill / 128 decode), completed all eight requests and emitted 136 tokens without
failure. Previously all 128 decode calls were scalar; the candidate used 67 scalar,
eight width-two and fifteen width-three calls: **23 whole-decode groups**. The changed
first-failure classification also moves mixed offers from `nonuniform_kv` to
`non_decode_phase`; it does not mean the logical scheduler changed. This replay confirms
physical lane use; the endpoint A/B runs, not one native sample, determine promotion.

All required host gates, CUDA-feature checks and the focused review passed. Kernels and
reduction orders within each lane are unchanged. The broader cross-engine fidelity
question, failed-drain device injection and physical-accounting gaps remain open; no
hard shared-memory bound, broader model/hardware qualification or SOTA claim follows.
Raw evidence/scripts remain outside Git in `desktop:~/ribn-qualification/hetero-{experiment,candidate}/`
and `desktop:~/ribn-qualification/bf76a6a-hetero-before/`.

## Sampled final prefill chunk (2026-10-17, promoted)

**Promoted: the eight-token same-sequence prefill lane now samples its final chunk row by
default.** The lane can include the final prompt sampling token and hand that chunk's last
residual row to the unchanged batch-1 output head. The chunk already advanced every
persistent component for the token, so this runs only `RMSNorm` + vocabulary projection +
argmax; it never consumes the token twice. Selection requires the warp projection mode with
the chunk-parallel recurrent scan disabled, whose full-model rows are bit-identical to the
batch-1 warp rows, and a prompt that is an exact whole number of chunks. Partial prompts,
the scalar projection mode and the chunk scan keep the previous serial final step, so
`QwenLoadOptions::sampled_prefill_final` is a preference rather than a capability request.
No kernel changed and no state layout or grant changed.

### Device evidence

| Gate | Result |
| --- | --- |
| Chunk row versus serial final residual, 8 tokens, full 64-layer model | 5120/5120 floats bit-identical |
| Sampled token from that row vs serial `decode_step`, fresh states | Equal |
| Serving seam, 16-token anchored prompt vs serial dispatcher | Equal sampled token, eight equal decodes |
| Diagnostics at 16 tokens | 2 chunk calls, 16 chunked tokens, 0 serial prompt tokens |
| Existing 21-token fallback (2 chunks + 5 serial) | Unchanged, still matches llama-server |
| Multi-chunk (3×8) chunk-versus-serial continuation | Still within its existing 5e-3 gate |
| Compute Sanitizer memcheck over the new path | 0 errors |
| Re-execution of the above on the promoted source | Bit identity, serving seam and memcheck passed |

Raw logs: `desktop:~/ribn-qualification/sampled-prefill/`
(`sampled-bits.log`, `memcheck-sampled-bits.log`, `serves-chunked.log`,
`multi-chunk.log`, `exits.txt`) and `desktop:~/ribn-qualification/sampled-prefill/promoted/`
for the promotion re-run. Commands:

```sh
ENGINE_QWEN_GGUF=/absolute/path/model.gguf \
cargo test --release -p engine-nvidia --features cuda --test cuda_reference --locked \
  sampled_final_chunk_row_matches_serial_hidden_bits -- --ignored --exact --test-threads=1

ENGINE_QWEN_GGUF=/absolute/path/model.gguf \
cargo test --release -p engine-nvidia --features cuda --test cuda_reference --locked \
  serves_chunked_prefill_matching_the_serial_path -- --ignored --exact --test-threads=1
```

### Endpoint measurement

Three alternating-order A/B repeats (`control`, `candidate`, `control` order) of the
[previous endpoint cohort](#first-equal-artifact-endpoint-diagnostic-2026-10-16): AIPerf
0.13.0, seed 42, greedy with EOS, 32-token cap, 16 profiling requests plus two warmups per
arm. The control is the retained `d1af02a` release CLI; only the candidate requested the
new selection. [Structured results](qwen-trace/2026-10-17-sampled-final-prefill.json) retain
source and binary identities, all 18 run summaries and the promotion criteria.

Medians of three run-level measurements:

| Cohort | Control output tok/s | Candidate output tok/s | Gain | p95 TTFT ms | Control → candidate p95 last-content latency ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short c1 | 28.22 | 32.72 | 16.0% | 272.68 → 116.56 (−57.3%) | 1135.52 → 979.07 |
| Mixed c4 | 31.71 | 32.39 | 2.1% | 2309.11 → 2257.75 | 4649.63 → 4645.47 |
| Mixed c8 | 34.91 | 35.74 | 2.4% | 4960.08 → 4789.07 | 9090.90 → 8876.23 |

All predeclared criteria passed: short-c1 median p95 TTFT fell 57.3% (≥20% required), no
cohort regressed throughput or p95 last-content latency by more than 5%, and all **144 paired
profiling visible outputs and core usage counts matched**. Every cohort kept its SLO pass
count (16/16/16, 16/16/16, 15/15/15). The independent Qwen reference/cancellation/admission
suite, the bounded CUDA HTTP parity/disconnect gate (which loads the promoted defaults
through `TextOwner::load`) and low-level in-flight retirement were re-executed on the
promoted source and passed. This is not an invertible token-ID comparison,
distributional-quality proof, or product SLO claim.

Limits: one artifact, one device, one warp lane. Only prompts that are an exact whole number
of eight-token chunks benefit; the retained serial final step still covers every other
shape, and the measurement says nothing about its latency. There is no failed-enqueue/drain
injection on this path. Sampling remains greedy-only. Bit identity is a property of the
current warp lane; changing projection mode, scan selection or chunk size voids it and needs
its own gate.

## First equal-artifact endpoint diagnostic (2026-10-16)

Ribn `dc4aed0` and llama.cpp `4e416ee7308dd6b581796f1a6241276cd5982691`
(`0.4.1-dev`, build 11138) served the exact pinned GGUF on the RTX 4090. The host was
an i9-13900KF, 33,377,128,448 bytes RAM, Fedora kernel 7.2.10-200.fc44.x86_64,
driver 615.71.09 (CUDA UMD 13.4 reported). The default installed nvcc is 13.3.73;
this does not assert which compiler built either binary.
[Structured evidence](qwen-trace/2026-10-16-endpoint-diagnostic.json) contains the base
manifest, exact server commands, per-run results, input-order checks and visible-output
comparisons. Raw manifests, exports, logs, environment and independent audit script are
retained outside Git at `desktop:~/ribn-qualification/dc4aed0-baseline/`.

Both servers used eight sequence slots, 512 tokens per slot, F16 KV and loopback
`/v1/completions` under alias `ribn`. Ribn retained its default execution policy;
llama used full GPU offload and disabled prompt reuse, RAM cache, recurrent checkpoints
and context shift (`--no-cache-prompt --cache-reuse 0 --cache-ram 0 --ctx-checkpoints 0
--no-context-shift`). Its ordinary startup warmup remained enabled; startup is outside
the comparison. Client settings: AIPerf 0.13.0 on Python 3.12.12 (frozen packages in
structured evidence), cached tokenizer revision
`Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, seed 42, greedy,
EOS enabled, 16 profiling requests and two warmups per run, output cap 32, connection
reuse disabled, GPU telemetry enabled. Three repeats alternated engine order A/B, B/A,
A/B. These are concurrency-limited cohorts, not open-loop Poisson arrivals.

All **18 protocol/accounting audits passed**: 288 measured and 36 warmup requests,
zero failures, and all measured replies reached 32 tokens. Input lengths/counts and
phase-separated prompt multisets matched. Concurrent request-start order did not always
match; a fixed seed is not a fixed arrival trace. The ordinary prompts contain no special
marker spellings. The artifact declares no BOS/EOS insertion flags; this llama BPE
reader defaults both off, matching Ribn raw input. Vocabulary-derived versus declared
stop-set equivalence remains unqualified; no early EOS occurred in this cohort.

An independent check recomputed all 324 requests' first-visible-content TTFT,
last-content latency and average TPOT from raw monotonic timestamps and usage,
matching native metrics within 1e-6 ms absolute / 1e-9 relative tolerance. Terminal
`[DONE]` latency was recorded separately. Active-window throughput uses actual reported
output tokens over first profiling request start to last request end, excluding warmup
and startup; it differs from native throughput by at most 0.064%. This checks named
metrics for these runs, not the runner's general `performance_metrics_audited: false`
contract. Average TPOT is not a maximum token/chunk gap or device time.

Medians of the three **run-level** values (not pooled population-tail estimates):

| Cohort, input distribution → output cap | Engine | Output tok/s | TTFT p95 ms | Average TPOT p95 ms | Illustrative SLO passes/run |
| --- | --- | ---: | ---: | ---: | ---: |
| c1, 16 fixed → 32 | Ribn | 28.17 | 272.99 | 27.87 | 16/16 |
| c1, 16 fixed → 32 | llama.cpp | 48.36 | 33.54 | 20.29 | 16/16 |
| c4, mean 64/stddev 32 → 32 | Ribn | 21.94 | 2351.65 | 153.03 | 16/16 |
| c4, mean 64/stddev 32 → 32 | llama.cpp | 97.11 | 200.83 | 50.48 | 16/16 |
| c8, same mixed cohort → 32 | Ribn | 21.95 | 5009.53 | 295.49 | 1/16 |
| c8, same mixed cohort → 32 | llama.cpp | 186.57 | 373.05 | 40.47 | 16/16 |

Actual mixed inputs ranged 22–119 tokens. Objectives were chosen before execution:
TTFT <= 5000 ms, **average TPOT** <= 200 ms and last-content latency <= 10000 ms. These are
illustrative diagnostic thresholds, not approved product SLOs. Ribn's c8 active-window
goodput was about 0.043 requests/s versus llama's 5.83; the small finite cohort is not a
steady-state saturation, fairness, physical-peak or robust-tail qualification.

**Quality-equivalent competitive acceptance remains open.** Visible text plus core
usage matched 7/16 short requests, 3/16 c4 requests and 4–8/16 c8 requests. Ribn's
visible texts repeated across all three runs and c4/c8 for the same mixed inputs;
llama's mixed outputs varied across repeats and concurrency. This is not proof that
either implementation is numerically correct or incorrect. Text is not invertible to
token IDs, and diverged histories cannot be used for distributional error comparisons.
Classify differences on common histories and independently qualified numerical/task
criteria; do not change tolerances or treat one engine's greedy continuation as universal
truth. The existing eighteen-token reference gate remains bounded.

The measured performance deficit is large and repeatable, but not a quality-equivalent
speed ratio. Next, pair that fidelity investigation with physical lane/phase attribution
and one bounded measured execution experiment. Keep logical policy as the control;
this evidence does not justify prefix caching, a language migration or a scheduler rewrite.

## Audit boundary repairs (2026-10-15)

Production checkpoint `f8c69c3` repairs four boundaries without changing model arithmetic:

- Preprocessing queue insertion and final-worker closure/drain share one authority.
  The deterministic real-pool regression failed on the original interleaving (exit 101)
  and passes after repair; replies and permits cannot remain in an unserviceable queue.
- `Completed::into_outcome` is removed: it returned output while silently dropping its
  lease. Existing explicit handoffs retain storage and charge through `into_parts`.
- BERT validates all mutable host parameter buffers, projection dimensions and layer
  counts before CUDA initialization. Checked shape products reject overflow. On the
  original source, the malformed-query regression reached invalid-device initialization
  instead of rejecting its parameters (exit 101); repaired CUDA-feature execution passes.
- Qwen shutdown asks the dispatcher to synchronize and settle fault quarantine, rather
  than synchronizing a separate stream handle without clearing the undrained state.
  Host state-machine tests cover failed drain followed by successful settlement,
  exactly-once retirement, preserved poison and untouched healthy pending outcomes.

All five repository gates passed on this production source, exit 0 each. On the idle
RTX 4090, driver 615.71.09 and Rust 1.98.0, these serialized checks also exited 0:

```sh
cargo test -p engine-bert --locked --features cuda --test host_validation
cargo test -p engine-nvidia --locked --features cuda --lib submissions::tests
cargo test --release -p engine-bert --locked --features cuda --test device_reference \
  -- --ignored --test-threads=1
ENGINE_QWEN_GGUF=/absolute/path/Qwen3.8-27B-UD-Q4_K_M.gguf \
  cargo test --release -p ribn-server --features cuda --test cuda_http --locked \
  -- --ignored --exact http_matches_direct_text_and_preserves_disconnect_peer_progress \
  --test-threads=1
```

Results: host validation 1/1, quarantine bookkeeping 4/4, independent tiny-BERT CUDA
reference/lifecycle cases 3/3 (0.32 seconds), Qwen HTTP parity/cancellation/shutdown 1/1
(24.13 seconds), excluding build times. The Qwen artifact and short-context scope are
unchanged from [initial HTTP qualification](#initial-text-http-qualification-2026-10-13).
No real device fault/failed synchronization was injected: the quarantine transition is
host-tested, and the CUDA gate requalifies the healthy/cancellation/shutdown path only.
Physical free completion, allocator-failure ownership, BERT final quarantine Drop and
hard shared-memory peaks remain unqualified. This is not competitive measurement.
Raw desktop evidence is retained under `~/ribn-qualification/f8c69c3-audit/`; all four
untracked SIMT artifacts retain their fingerprints.

## Legacy-runtime retirement: device re-execution passed

The duplicate core serving scheduler/runtime is removed. Its wrapper-only host tests
retired; current AR/Qwen contract tests retain lifecycle protection. The two stateless
CUDA reference tests now use the backend directly and retain exact `[16.0, 14.0]` checks.
The 8/9-row cancellation fixtures are consolidated into
`retires_in_flight_members_without_losing_peers_or_state`: it checks deferred physical
release, charge retention, peer progress, the nine-to-eight lane transition and an empty
registry after teardown. Actual cancellation/output suppression remains in Qwen's AR
gate, now also at concurrency nine. No numerical kernels changed.

These migrated fixtures and the Qwen nine-row case passed serialized device
re-execution on 2026-10-06. See the [current Qwen gate](#mixed-arrival-native-qwen-gate)
for hardware, artifact, scope and timing. This qualifies the changed test plumbing;
earlier performance measurements remain historical.

## Compatible decode subgroup candidate

Device gates passed on 2026-10-07 for production `c0c0242`, RTX 4090/driver 615.71.09,
Rust 1.98.0 and the same pinned artifact/reference as the mixed-arrival gate. The
updated test source is committed with this evidence; production arithmetic is unchanged.
Run serially with `RIBN_GROUPED_DECODE=1`, `RIBN_MODEL` and `RIBN_REFERENCE`:

- `prepared_qwen_matches_reference_and_preserves_cancelled_peers`: 1–9 rows, plus
  cancellation at 2/8/9, matching all 18 independently agreed outputs. The known
  near-tie at index 18 remains excluded; no independent 32-token parity claim.
- `mixed_request_reaches_preserve_reference_and_cancelled_peers`: `[4,8,8]` and
  nonadjacent `[8,4,8]` budgets, normal/cancelled; staggered prefill beside decode peers.
- `nonadjacent_reaches_preserve_distinct_reference_histories`: `[8,4,6]` budgets with
  reference offsets `[0,0,2]`. The last prompt includes two independently recorded
  teacher-forced outputs, preserving equal reach for rows 0/2 while their decode outputs
  differ. This detects token routing swaps that same-prompt rows cannot.
- `owned_driver_preserves_reference_with_stalled_and_abandoned_peers`: grouped execution
  with healthy, stalled and abandoned consumers.
- NVIDIA `retires_in_flight_members_without_losing_peers_or_state`: physical 8/9-row
  retirement, retained charges, nine-to-eight transition and clean teardown. In grouped
  mode, a launched decode subgroup followed by unsupported sampling on a later prefill
  row drains successfully, restores all taken states and retains logical charges until
  explicit release. This does **not** test a failed synchronization or prove pinned-slot
  exactly-once recycling after every composed failure.

All five gates exited zero. The three-request constrained-capacity gate also passed,
with default whole-batch selection, not grouped execution. Host gates passed separately;
compilation is not this device evidence. The [matched trace comparison](README.md#mixed-arrival-native-traces)
found no measurable benefit, so the candidate remains default-off. Paging, growth and
cross-runtime physical capacity are not enabled or qualified by these tests.

## Ground-up alignment cost and isolation

The current mailbox implementation reserves both global and per-request event
capacity. Request-specific draining does not need an unbounded frontend event
buffer. Aggregate draining is round-robin across clients, preserving each client's
order rather than a global arrival order. Stalled-consumer and mailbox/slot reuse
tests run in the ordinary host suite.

[Host comparison](runtime-alignment/README.md) records the additional synthetic
CPU cost relative to the original global-queue runtime. Stable internal mailbox
slots avoid repeated hashing, but the result is not an inference speedup claim.
