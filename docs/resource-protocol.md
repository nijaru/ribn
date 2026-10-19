# Resource, submission and snapshot protocol

Status: accepted execution contract, revised 2026-10-11. Implemented behavior and
qualification limits are identified below; targets do not retroactively qualify source.
AR preparation control is host-qualified. Dynamic Qwen growth, bounded overlap,
non-AR application lifecycle and snapshot replacement remain unimplemented. Accepted text
allocation/options retention is host-qualified; CUDA physical-accounting gaps remain open. The
[roadmap](roadmap.md) owns delivery order and acceptance gates.

## Current boundary

`GenerationExecutor` combines admission, aggregate preparation, submission and completion. Since roadmap 4a,
Qwen reserves the continuation capacity its own request can reach instead of the
model's whole context (see [AR continuation resources](#ar-continuation-resources));
it still reserves that capacity once, at admission, and executes the offered range.
`poll` returns one positive `StepCompletion` per row. `None` means submitted work
remains pending, not a settled request waiting for a resource.

The incomplete completion-time `Blocked` enum was removed: it had no readiness source
or parking and could immediately resubmit unchanged work. Positive partial-prefill
completion remains: it reports a contiguous consumed range, not permission to exceed
physical capacity and not a pre-submit reservation. Ordinary resource waiting belongs to pre-submit preparation and its readiness source,
not another isolated completion enum. Admission and preparation waits carry an
authority-owned readiness registration. Direct callers recheck registrations when stepping; the driver arms
one-shot notifications on its capacity-one wake channel before parking. Device
completion alone retains the bounded polling fallback.

Keep these invariants throughout replacement:

- validate every submitted row before logical commitment;
- reserve output capacity before launch;
- cancellation records intent; it never proves completion;
- malformed completion or uncertain executor state faults the execution owner,
  including its other live requests; ordinary request rejection is different;
- retain ownership of anything the device may access until completion is known.

## One authority, four ownership states

For each shared physical pool, one allocator/accountant grants reservations. The
model/backend calculates concrete demand and asks that authority during preparation.
The scheduler does not charge the same reservation again.

1. **Candidate:** scheduler policy offers work; no new allocation is owned by the
   candidate. Token counts are policy bounds, not a universal memory cost.
2. **Prepared:** the backend owns newly granted reservations and an accepted range
   for each ready request. Existing continuation remains sequence-owned. Preparation
   may shorten work, report temporary waiting, or reject an impossible request.
3. **Submitted:** the executor consumes the prepared work and owns every resource
   visible to partially or fully enqueued device work, even if enqueue fails.
4. **Settled:** after completion, temporary storage can return to its pool; accepted
   persistent growth transfers to continuation or result ownership. Logical progress
   reflects the actual valid continuation boundary, not speculative physical work.

Demand estimates are plain values. Reservations are owning, non-duplicable leases.
Moving a lease between owners does not reserve again. Physical storage and its
charge live together; popping a result from a queue does not release its charge.
Reference-counted sharing is valid where necessary, but must not duplicate accounting.

A resident arena/slab owns its physical byte charge until it is actually retired.
A block allocator separately tracks which slots may be reused. Releasing or evicting
a slot returns occupancy to that allocator, not physical bytes to sibling runtimes:
the backing allocation still exists. Accounting a live slab once and then charging
its blocks again is equally wrong. Preparation must identify both the physical
resident owner and the authority granting safe slot reuse.

Abandoning prepared, unsubmitted work releases **all new** reservations, including
uncommitted persistent growth. It does not release pre-existing continuation. Ordinary
RAII can handle host-only reservation release; fallible device retirement needs an
explicit executor-owned retry/quarantine path. Drop cannot assert device completion.

An enqueue error must distinguish a clean rejection before device access from
uncertain partial submission. Never return ownership to the caller as though nothing
happened after device-visible mutations. A separate `prepare` method alone cannot
make enqueue infallible; the implementation must retain a concrete completion owner.

Keep prepared ownership concrete to the model/backend. Add shared types only where
real consumers demonstrate the same lifetime and settlement contract; no public
universal claim/lease interface follows from these states.

## Execution observation

Diagnostics do not grant resources, commit progress or authorize retirement. Keep
aggregate dispatch observation opt-in and bounded: one execution owner publishes fixed-size
snapshots without retaining requests, device owners, callbacks or per-step traces.
Counters saturate rather than wrap. Reading a snapshot must not expose its write lock.

Distinguish offered work, fully enqueued work and completed work. Count each physical
model-step group once, not once per member; same-sequence prefill chunks are not
cross-request decode batches. A failed enqueue can launch partial work: record the
failure and identify any work excluded from the counters. State clock boundaries;
host routing/enqueue/event-recording time is not GPU execution time. Per-row backend
metrics may repeat a group-wide interval and are not implicitly additive.

## Feasible batches and waiting

Preparation considers aggregate submission resources, including shared encoder work,
cache residency, workspace, continuation growth and output capacity. A per-row budget
reset is not a batch bound. Choose the accepted ranges before executing their work.

A prepared report distinguishes:

- **Ready:** positive feasible work with owned reservations;
- **Waiting:** no committed work; a concrete condition can change;
- **Rejected:** permanent request-local infeasibility, with a useful diagnostic.

One oversized indivisible item must not wait forever for a per-step budget that
cannot grow. Do not infer impossibility from an arbitrary retry count.

Waiting requests park outside runnable queues. Readiness uses a generation/epoch or
registration-and-recheck protocol so notification before parking cannot be lost.
Reactivation works with no device batch in flight. Cancellation and shutdown wake
parked owners; output consumption wakes output-blocked work. A new compute-budget
epoch differs from an external encoder dependency becoming ready.

Device completion may initially require timed event polling. A worker should sleep
on commands/readiness while idle and use bounded timed waits while polling such a
backend. Document that fallback rather than claiming fully event-driven execution.
Do not stall healthy requests merely because another request is waiting.

## Output ownership and boundedness

The execution owner owns cancellation and discarded mailboxes. A frontend dropping
a request relinquishes interest once; it does not maintain a second cleanup queue.
Discard must work after an execution slot is freed but its terminal mailbox remains.
It suppresses future delivery without releasing in-flight storage early.

Bound submission count, owned input/options bytes, retained event count/bytes and
physical storage separately. Payload length is not retained allocation size: include
string/vector capacity, message-container storage, stop lists and retained clones.
At application acceptance, validate or normalize owned allocations before queue
insertion, including bounded temporary overlap during normalization. Define shared
immutable storage accounting explicitly: conservative per-request envelopes may
count shared input in each request's allowance, but a physical allocation has only one
charged owner. Caller-owned storage before acceptance is outside the application budget,
not after acceptance.
Allocator overhead and external library/device pools need separately stated margins;
do not describe payload accounting as a process-RSS or device-wide peak bound.

A bounded channel containing arbitrarily large vectors is not a memory bound. Public
collect helpers intentionally allocate the requested result; document their limit
and offer incremental consumption for large workloads.

Cancellation must remain deliverable when the ordinary submission queue is full.
Use per-request cancellation intent plus a wakeup, or a separately bounded control
path whose capacity follows admitted requests—not an unbounded emergency queue. The
concrete non-AR form (2026-09-15) is intent recorded on the waiting request itself: a
cancelled request keeps its FIFO place and reports cancellation instead of executing,
so the intent occupies no additional structure and cannot be lost to a full terminal
queue. Cancelling a request whose result is already retained gives that result — and
the charge covering it — back to the executor through the same retirement path a
malformed completion uses; the runtime never releases device storage or its charge by
itself.

## Encoder preparation and prepared resources

Status: implemented for the first real encoder (2026-09-15, roadmap slices 3b.2–3b.3).
The concrete representation is deliberately not generic: `ribn-foundation`'s
`BytePool`/`PoolLease`/`AllocationId` is the authority, `ribn-batch` owns reservation,
accepted ranges, capacity registration, result retention and the enqueue-failure
contract, and `engine-bert` supplies the concrete shape decisions
(`crates/bert/src/request.rs`), its device result and its retirement list
(`crates/bert/src/executor.rs`, `crates/bert/src/cuda.rs`). Roadmap 3c qualified the failure
paths on device (constrained pool, cancellation before and after enqueue, a lagging
consumer beside a healthy peer, permanent rejection with peer progress, charges
surviving dequeue) except for one criterion whose precondition does not exist yet:
downstream workspace headroom needs a real downstream stage sharing the pool; it
is not established by the separate AR/encoder fixtures. Two limits are recorded:
a request's whole envelope remains with its result rather than releasing temporary
storage early; an explicit failed retirement attempt retains storage and charge for
retry. Default result/quarantine Drop is not a completion-safe retirement protocol:
it can refund the charge without proving physical reuse/release. The unused extraction
that silently discarded a result's lease is removed; explicit transfers still must move
storage and lease together. These logical lease rules do not prove queued frees completed.
Partial allocation/upload failure during submission construction is an additional
source-traced gap: already-created buffers can drop into stream-ordered frees while
no retained submission carries their charge. Post-enqueue qualification does not
cover that path or prove global physical-byte availability after normal result Drop.
These paths need completion-safe charge retirement before a hard cross-stream peak
bound is claimed; safe ordering of device accesses alone is not physical release.
Locked [cudarc 0.19.9](https://github.com/coreylowman/cudarc/blob/08ae53f143b635e77df5f68f273b7e277632c449/src/driver/safe/core.rs#L1534-L1560)
also calls `malloc_async`/`malloc_sync` before creating tracking
events in `CudaStream::alloc`. If event creation fails, it returns without a `CudaSlice`
owner for the pointer. A stream drain alone cannot free that unreachable allocation;
allocation failure cannot uniformly be classified as a clean refusal. Resolve the
allocator failure owner as well as stream-ordered free completion before claiming a
hard shared CUDA bound. The pinned-host allocator has the same allocate-before-event
failure gap; it is used for Qwen output slots. Failed pinned-event synchronization must
not authorize host-buffer reclamation while a device copy can still access it.

1. **One authority, owning byte leases.** The first concrete authority is a shared byte
   pool that grants an owning, non-duplicable lease per reservation. Moving a lease
   between owners does not reserve again. Physical storage and its charge live in one
   owner, so a backend that materializes device memory for a lease keeps both together,
   and popping a result from a queue does not release its charge. A lease carries the
   allocation identity that a later derived-state compatibility check needs; a bare byte
   count is not an allocation identity.
2. **Accepted work.** Preparation reports the concrete work the executor accepted and
   its reservations. Current BERT selects a bounded FIFO prefix. The target permits
   bounded indexed selection/bucketing with age protection so a temporarily infeasible
   head cannot strand feasible peers. Selection order is not caller-visible result order;
   retain request identity and reserve bounded delivery/reorder storage. Aggregate
   resources are budgeted across the whole accepted work, never reset per row.
   BERT host preparation binds an immutable model configuration to the backend's
   sequence limit. The usable range is their intersection; a larger position table
   remains valid for shorter requests rather than being rewritten or rejected.
   Shape/envelope validation and batch selection must apply this same prepared
   limit before reservation or enqueue. The CUDA attention backend owns its current
   4096-token limit; exceeding it is request-local rejection, not a peer-wide launch
   failure.
3. **Waiting versus rejection.** Waiting names both a condition and a readiness source.
   Byte-pool and AR capacity waits share `ReadinessWait`: register before checking,
   then arm the caller's wake transport before parking. Arming rechecks the
   source epoch so a release between the capacity check and parking cannot be lost.
   Pool closure also publishes, allowing a waiter to observe terminal refusal without
   waiting for a live lease to release; closure does not refund those leases.
   Callbacks run after the accounting lock is released. Rejection is permanent
   request-local infeasibility — an indivisible input
   larger than the pool can ever grant, or an unsupported shape — and is delivered
   without waiting. A retry count never establishes impossibility. Waiting work parks
   outside runnable queues and does not stall healthy requests; while a backend provides
   no completion notification, a bounded timed fallback is explicit, not implied.
4. **Pre-submit abandonment.** Abandoning prepared, unsubmitted work releases all new
   reservations, including uncommitted persistent growth, and never releases
   pre-existing continuation. Host-only reservations may rely on ordinary RAII; fallible
   device retirement needs an executor-owned retry/quarantine path, and Drop cannot
   assert device completion.
5. **Partial enqueue.** An enqueue failure distinguishes clean rejection before device
   access from uncertain partial submission. After any device-visible mutation the
   executor retains completion ownership of everything that may have been enqueued; a
   caller is never given ownership back as though nothing happened. Concretely, a
   granted reservation travels into the submission with the request it covers, so a
   failure returns either "nothing was created, every reservation released" or "the
   executor keeps the storage and the charge". The distinction is part of the reported
   error, an enqueue error carries no reservation, and output the runtime cannot commit
   is handed back to the executor rather than released by the runtime.
6. **Producer/consumer completion ownership.** A device-resident result carries its
   representation, model identity, pool charge and a producer completion dependency. The
   consumer waits on that dependency before reading. [Cross-runtime device
   ownership](#cross-runtime-device-ownership) already owns the reuse, cancellation and
   quarantine rules; preparation adds only the lease and dependency that make them
   checkable.

### Non-AR application lifecycle

Status: accepted target, not qualified by the current direct BERT fixture. `BatchRuntime`
currently bounds waiting/result counts, accepts executor-defined inputs without an
input-byte budget, selects a FIFO prefix and has no model-local close/shutdown surface.
Pool waits have `capacity_wait`; `BatchSelection::Blocked(C)` supplies no general
executor readiness registration. Temporary head pressure can strand feasible later
requests and postpone a cancelled peer's terminal delivery.

Before exposing owned encoder access, establish these contracts through the real runtime:

- Bound retained input/options before acceptance, independently of device/result leases.
- Select bounded feasible peers with aging; keep result ordering in the chosen application
  contract, not as a compulsory execution FIFO. Cancelled queued work must settle without
  waiting for an unrelated head to become feasible, subject to delivery backpressure.
- Every temporary wait carries registration/recheck or an explicit completion dependency.
  A reason enum alone is insufficient. Wake local cancellation, closure and output release.
- Close this model's admission without closing a shared pool used by siblings. The execution
  owner retains submitted work, abandoned results, charges and retry/quarantine cleanup.
  Dequeued device results retain their own safe-retirement owner after model shutdown.
- Output dequeue returns delivery capacity, not physical bytes. Charges release only when
  backing is safely reusable within its charged arena or proven released to the byte pool.

## AR continuation resources

Status: the declared-shape/concrete-capacity split and capacity backpressure are
implemented for the Qwen path (2026-09-15, roadmap 4a, device-qualified at `323f258`).
The backend block-table reader was qualified at `ee689d7`; production Qwen append and
attention now use table addressing (`e36e097`). The host contract is page-granular —
fixed page size per model, whole pages per sequence, capacity `pages * page_tokens` —
while materialization is still one contiguous run per sequence with an identity-bound table. Paged block reuse, eviction and preemption are accepted design and remain
unimplemented; this section states the contract they must satisfy.

Continuation is the state a sequence must keep to continue: full-attention KV,
sliding-window KV, recurrent matrices and convolution history, and hybrid combinations
of them. It is owned per sequence until a cache owns it. The target is one physical
byte authority per shared device pool. AR and encoder use the same `BytePool` mechanism,
but production Qwen still owns a private authority; neither common types nor readiness
establish shared physical capacity. Resident backing and slot occupancy remain distinct.
The first paged Qwen materialization uses bounded resident K/V arenas matching the
qualified common-base reader. Account recurrent storage, workspace, immutable table/copy
staging and retirement overlap separately. Online arena growth is outside the initial
path until its allocation, peak overlap and physical retirement are qualified.

### Mixed-retention and shared KV

Status: model-local Gemma 4 range semantics can be checked on the host; physical
storage, charged preparation and execution remain unimplemented. Do not encode this
as two homogeneous full-attention requirements and then select only the first KV.

The model owns the layer-to-KV-owner mapping, head widths, retention and last consumers.
The AR runtime sees sequence progress through `GenerationExecutor`, not model-family
branches or layer scheduling. A valid committed prefix requires every owner to cover
its model-required range. Shared consumers create no independent persistent KV.

Preparation binds one positive accepted input range to the actual sequence incarnation,
current committed prefix and request reach. Reject stale/mismatched coverage, overflow,
over-capacity work or a second outstanding transaction before launch. A pure range plan
contains no storage, reservation, completion proof or restoration capability.

For a sliding window `w`, prefix `p` and accepted end `e`, retained input is
`[max(0,p-(w-1)),p)`; the current attention span includes that suffix **plus the entire
accepted chunk**, ending at `e`. The next persistent suffix begins at
`max(0,e-(w-1))`. Each query still uses its own causal/window mask. Full attention keeps
`[0,p)` and supplies `[0,e)`. Never substitute the post-trim suffix for a donor span:
a 513-token Gemma prefill needs 513 donor positions while retaining only 511.

Producer and shared-consumer layers of the same accepted chunk are one model/backend
transaction. Pin donor storage until its last consumer's device read completes, not
until host dispatch or persistent-cache trimming. E4B owner 22 serves through layer 40;
owner 23 serves through layer 41. Segmented views, concatenation and overlapping backing
are backend choices, not changes to these ranges. Count shared backing once, but charge
actual allocation/page sizes, copy/append overlap, workspace, staging and retirement;
logical payload sums are not physical byte bounds.

Start with one in-flight batch and no dependent next chunk before settlement. Validate
all required owner completions and all rows before advancing any committed metadata.
Initially execute exactly the range selected during preparation, rather than executing
farther and reporting a shorter prefix without rollback. Abandonment leaves committed
state unchanged; this is not device rollback. A partial in-place enqueue may destroy
resumability: fault the owner and drain/quarantine old/new storage with its charges.
Cancellation intent and failed drain never release donor storage early.

### AR reservation authority

AR's transitional state manager uses supplied `BytePool` authorities, not another
counter/epoch implementation. Bundle allocation validates all components and grants
one aggregate lease before committing any allocation records. Failed feasibility or
validation retains no grant and publishes no artificial rollback wake. The lease is
shared only among that bundle's manager records: it charges once and stays reserved
until the last component retires. Individual allocations have independent grants.

Pool usage/readiness includes every owner of that authority, not just one manager.
Closure wakes waiters and refuses new allocation while live charges remain. A logical
handle does not prove physical retirement. Explicit release is allowed only after its
backend establishes safe reuse; dropping a manager with unreleased records retains
those charges rather than silently refunding memory whose lifetime it cannot prove.

Production Qwen still constructs a private continuation pool. No shared CUDA deployment
or resident-byte bound follows from this common mechanism: backend teardown and
partial-construction retirement must qualify before sharing it with the encoder.

1. **A declaration is a shape and a bound.** A model declares each continuation
   component's shape and maximum capacity. A concrete state materializes a capacity
   within that bound, and validation accepts any concrete capacity whose shape matches
   the declaration. Reserving less is the point: capacity that a request cannot reach is
   not reserved. The bound is a hard limit, not a suggestion.
2. **Capacity is a reservation request.** Requested capacity is derived from the request
   (prompt plus output budget, or the accepted range for growth), granted by the
   authority, and charged as granted. Growing continuation is a new reservation
   transaction; it never edits the declaration and never borrows unreserved capacity.
   Insufficient capacity is ordinary backpressure and an indivisible demand larger than
   the authority can ever grant is request-local rejection, delivered without waiting.
   Admission returns `Deferred(ReadinessWait)` registered before checking the blocking
   condition. The runtime retains this registration on the waiting request, outside
   runnable queues, and re-offers it only after the source changes. The capacity authority
   publishes after a successful release, never after commit, failed release or allocation.
   A change permits retry, not allocation: a peer may already have consumed the capacity.
   A retry that still cannot fit registers again. Cancellation and shutdown remove the
   wait with its request; no executor-owned waiter list is needed. Before parking,
   the driver arms each retained wait with its existing capacity-one wake channel.
   Arming rechecks the epoch: publication before subscription wakes immediately;
   publication afterward sends a one-shot notification. Sources retain only weak
   subscriptions, pruned on arming/publication; dropping a wait releases its waker.
   A notification already racing cancellation may arrive but grants no resource and
   cannot restore the cancelled request. Wake callbacks run outside source locks.
   With no device work in flight the driver parks without a resource polling timer;
   device completion retains its explicit bounded polling fallback.
3. **One owner holds storage and its charge.** A backend materializes device storage for
   the lease it was granted, and the same owner releases both together. A lease is never
   duplicated to share storage: storage shared between sequences is owned once (by the
   cache or another owner) and handed out as a reference, so the charge is counted once
   and reference counting does not duplicate accounting.
4. **A sequence's continuation is a bundle with per-component validity.** A prefix is
   reusable only when every component the model requires at that boundary is present and
   valid — for a hybrid model, KV without the matching recurrent checkpoint is not a
   continuation. A partial match is not partially usable; it advances only to the last
   boundary where all required components agree.
5. **Commit reports the valid boundary.** After completion, the runtime advances the
   sequence's committed prefix to what the model actually validated, not to what was
   enqueued. Persistent growth that was charged as prepared work settles into
   continuation ownership at that boundary; temporary work returns to its pool.
6. **Release follows proven completion.** Dropping continuation is logical until the
   device may have finished reading it. A release the backend cannot prove keeps its
   storage and its charge, on the same retirement path as any other uncertain device
   work. Cancellation is intent and does not by itself release continuation.
7. **Eviction and preemption are different owners.** Cache eviction returns
   unreferenced slots for reuse, not a resident arena's bytes. Preemption takes
   continuation from a live sequence only after completion-safe retirement. Cancellation,
   preemption and eviction remain distinct outcomes because they imply different work
   and accounting. Recompute requires one bounded owner of the prompt and committed
   generated history. Its physical replay cursor is separate from externally committed
   progress; replay must not duplicate output, usage or sampling. Survivor growth takes
   precedence over victim readmission. Initial partial-envelope admission uses recomputation
   preemption, not resource waiting alone or a host-swap tier. Before admitting partial
   envelopes, demonstrate that a selected survivor can finish within the available grant
   after completion-safe victim/cache reclamation, including workspace and replay storage.
   Capacity feasibility is not consumer-dependent liveness: choose an output-ready survivor.
   If its consumer stalls and credits run out, relinquish protection through completion-safe
   preemption and reselect a healthy peer. Park its replay/readmission until output readiness
   changes; do not require client cancellation or release in-flight storage early.
   While a survivor remains runnable, do not readmit victims/newcomers at the expense of
   its required growth. Keep victims bounded and age their readmission; new arrivals cannot
   repeatedly displace them. Individually impossible requests reject locally, never through
   endless self-preemption.
   Failure to prove safe reclamation faults the owner; it is not a reason to refund early.
8. **Restore includes execution semantics.** A matching hash is not proof of model-required
   retained KV coverage or hybrid checkpoints. Full attention requires prefix coverage;
   sliding attention requires its valid window, and recurrent/other components must agree
   at the same boundary. Restore only a complete valid boundary and reserve a private
   writable recurrent copy. Initially that boundary must
   be strictly before the final prompt sampling token; execute the remaining suffix to
   produce logits. Re-consuming a token after restoring its post-token recurrent state
   would advance state twice. Cached logits/output state is a separate deferred policy.
9. **Growth negotiates before launch.** An AR preparation transaction accepts positive
   ranges within the offered budgets, parks waiting rows with registered readiness, and
   rejects permanently impossible rows locally. Selection is not submission: do not mark
   omitted rows as in flight or encode shortage as a submit fault/zero-progress completion.
   Refund omitted/shortened output reservations. Abandoning unsubmitted preparation
   releases only new growth; uncertain enqueue retains it together with old continuation
   until executor-owned settlement or quarantine. Aggregate demand, not per-row resets,
   determines the accepted batch.

### AR preparation transaction

`GenerationExecutor::prepare` negotiates one offered batch at a time. The executor,
not the returned report, owns newly granted resources. `BatchPreparation::Ready`
accepts the entire offer without allocating a per-row report; `Selected` returns exactly
one ordered `PreparedRow` per offered row: a ready `BatchItem`, a registered deferred
wait, or a permanent request-local error. Ready rows preserve sequence, kind and prefix;
only positive token budgets and their corresponding output budgets may shrink.

The engine reserves output before offering work, validates the **whole** report before
changing any row, refunds omitted/shortened credits, and marks only accepted rows in
flight. Deferred rows retain their old continuation but no new preparation reservation.
They park outside runnable queues and share the admission wake transport. Rejected rows
finish locally; peers remain eligible. An all-omitted offer triggers a bounded rebuild
for unoffered runnable peers before parking, without reactivating unchanged waits.
A source change permits another attempt, not a guaranteed allocation.

`submit` consumes the exact ready subset in its original order. Current source supports
at most one outstanding preparation and one submission; another preparation cannot
overwrite either. This is the depth-one implementation, not the target limit.
`abandon_preparation` is idempotent and releases only new, unsubmitted reservations.
The engine abandons an invalid report or an offer with no accepted rows. Preparation,
abandonment or enqueue errors fault the owner and keep executor retirement ownership;
`synchronize` must also settle any outstanding preparation. Fallible retirement cannot
turn into implicit Drop-based proof of device completion.

In current source preparation and submission are synchronous within one `Engine::step`;
driver intent cannot interrupt that call. Cancellation observed after launch waits for
completion.
Cancellation of a parked row drops its registration and retires its old continuation
without restoring it to a runnable queue. Logical prefixes still advance only at valid
completion. This control seam does not enable partial-envelope Qwen admission or prove
all-active growth progress.

The engine owns which sequence holds which committed prefix and the policy decisions
(admission, eviction, preemption, priority). The model/backend owns layout, block or
page granularity, checkpoint materialization and kernel-facing indexing. The authority
owns bytes. None of these three charges the same reservation again.

### Bounded overlap and submission ownership

Status: accepted target; current AR/Qwen remains depth one. One mutable execution owner
remains authoritative even when CPU preparation overlaps device execution. Do not widen
`prepare`/`submit` until pending records, completion mapping and release semantics own
multiple submissions without singleton overwrite.

- Configure a finite preparation/submission depth and aggregate byte/output-credit bounds.
  Each concrete prepared record is uniquely identifiable and consumed once; abandoned
  records release only their new unsubmitted resources. Completion refers to its actual
  submission and sequence generation, not a reused row index or current batch ordering.
- The first overlap increment permits independent sequences only, with at most one
  unresolved state-mutating transaction per sequence. Logical prefixes advance on validated
  completion. Persistent request rows are independent of gathered kernel input order.
- Each submission owns immutable transfer staging, block-table versions, descriptors,
  output slots, workspaces and referenced storage until every device dependency completes.
  Safe shared immutable backing can be charged once; scratch cannot be shared concurrently
  without an execution dependency. Depth-one buffer reuse is not proof of overlap safety.
- Dependent same-sequence work ahead of host commitment needs an explicit device dependency
  and a separate projected versus committed cursor. The backend must own the input token
  and tentative state; guessed progress cannot enter ordinary scheduling or cache validity.
  EOS/cancellation may leave extra physical work to drain but never extra delivery or usage.
  This later capability is separate from independent-sequence overlap and speculation.
- Validate an entire completion report before committing its rows. Deliver per-request
  output in order even if independent submissions complete out of order. Reserve delivery
  and temporary reorder capacity before launch; a stalled consumer cannot hold an
  unbounded reorder list. Unknown completion faults the owner across all dependent work.
- A backend release acknowledgement must prove storage safely reusable/released, or
  transfer storage **and its charge** to an executor-owned retirement record. Returning
  success while deferring only the physical buffer cannot authorize a logical refund.
  The current NVIDIA deferred-release/Qwen-refund seam must change before overlap enables
  it as an ordinary path. Kernel completion alone does not prove queued frees completed.
- Shutdown closes admission, abandons new unsubmitted preparation and drains/quarantines
  every submitted owner. Failed drain retains table/staging/output slots and charges for
  retry; no slot generation or pool charge recycles on cancellation, Drop or timeout.

### Speculative settlement

Status: prerequisites for a future concrete implementation, not a current capability.
Draft/verification work owns tentative KV, recurrent/convolution state, scratch and
sampling state separately from committed continuation. Reserve its entire envelope and
potential output before launch. Report physical work and the accepted logical boundary
separately; cache publication, usage and delivery follow only accepted commitment.

Rejected drafts must restore **every** model-required component to that boundary,
including recurrent state and the chosen RNG/sampling semantics. KV truncation alone is
not hybrid rollback. Early stop/cancellation settles or retires all later tentative work
without delivering it. A zero-acceptance draft round must either commit a valid correction
or terminate/park for a named external condition; it cannot spin on zero progress.

Qualify the actual contract: reproducible greedy/keyed token equality versus ordinary
decode, or independently justified target-distribution preservation for stochastic
sampling. Those are different guarantees. Multi-token positive-completion fixtures do not
establish either, nor do they provide rejection rollback. Choose a concrete representation
only with a real drafter/target backend; no public speculative framework is selected here.

## Owned AR driver contract

The first owned-access increment drives the existing token runtime; it is not the
loaded-model or multimodal application API. `Driver::spawn(engine, config)` consumes
an idle, open engine and returns a shutdown owner and cloneable submission handle.
Direct `Engine` use remains channel-free. One worker owns execution, including failed
retirement. Moving an engine requires its existing `GenerationExecutor: Send` contract;
this increment does not support non-Send/thread-affine executor construction.

### Admission and bounds

- Admission is fail-fast. `try_reserve` acquires one request permit or returns typed
  overload; there is no hidden, potentially unbounded queue of waiting producers.
  A frontend must acquire this permit **before** expensive preprocessing. The token
  driver itself accepts already-encoded input, not raw media or preprocessing tasks.
- Permits cover unsubmitted preparation, queued commands, execution and undelivered
  stream output. A permit returns only after runtime retirement and stream delivery
  both relinquish it. A private runtime lifetime guard pins the admission charge even
  after the worker drops a discarded route; there is no frontend retirement retry list.
  Runtime execution-slot counts are not application request counts.
- Each permit has a configured encoded-input byte envelope covering token storage and
  stop-token vector capacity plus the runtime's cloned stop list. The aggregate bound
  is request permits times this envelope. Overflow is rejected. Caller-owned inputs
  before acceptance, allocator overhead and immutable loaded model storage are not
  included. Raw-input/tokenizer scratch bounds remain a frontend obligation.
- Each stream has a bounded channel of token events. Runtime mailboxes remain bounded
  separately: at most the runtime event limit plus permit count times stream capacity
  are buffered. Runtime global event capacity must cover every permitted request's
  per-request mailbox limit, so a stalled peer cannot monopolize delivery credits.
  This conservative configuration can be revisited with shared-credit evidence.
- Tokens/usage are copied values, not shared device storage. Runtime execution-error
  diagnostic strings are bounded to 4096 UTF-8 bytes at construction; overflow ends
  with an explicit truncation marker. This bounds retained diagnostic payload, not
  backend formatting scratch. Collected caller-owned results are deliberately outside
  buffered-delivery accounting; physical output leases belong to a later real encoder.

### Delivery, wakeups and failure

Use Flume 0.12 bounded channels for both blocking and runtime-independent async waits.
One capacity-one wake channel coalesces notifications; commands and persistent atomic
cancellation/discard flags are the authoritative state. The worker drains old wake
notifications **before** inspecting state, never after checking a condition and before
sleeping. Mutators publish state before notifying. This covers arrivals, stream drops,
explicit cancel and output-credit return without a second cancellation queue.

- `stream(...).await` or its blocking counterpart returns after runtime enqueue, not
  model admission or device completion. Rejection before enqueue returns a typed error.
  Dropping the submission future relinquishes its request even across enqueue/ack races.
- An owned stream yields ordered token events and one terminal event or owner error,
  then remains exhausted. Dropping an individual `next` future loses no event.
  Explicit cancel preserves already-buffered output and requests a `Cancelled` terminal;
  it can race with an already-settled terminal. Dropping the stream suppresses delivery.
- The worker never blocks sending output. Only it sends to each stream; it checks channel
  space before draining that runtime mailbox. Consumption wakes the execution owner.
- An idle/output-blocked owner sleeps on the wake channel. Registered admission/preparation waits
  arm one-shot notifications before parking and recheck the source epoch; unchanged
  registrations do not rerun model admission. Only pending device completion uses the
  configured nonzero timed-poll fallback. Synchronous backend calls remain non-preemptible;
  wakeups cannot interrupt a blocked driver operation.
- Request-local runtime admission failures remain `FinishReason::Failed`. Driver-level
  enqueue rejection retains its `EngineError` source. Execution-owner failure stops
  admission, preserves events already handed to stream channels, then exposes an
  owner-scoped error. Undelivered runtime-mailbox events are abandoned; no final
  request usage is promised after owner failure. It does not relabel healthy peers
  as individually invalid.

### Shutdown

The non-cloneable shutdown owner has blocking and async shutdown methods. Shutdown
closes admission first, abandons outstanding delivery and synchronizes/releases the
engine. Previously queued stream events remain readable, followed by an owner-closed
error unless a terminal was already delivered. Shutdown is not per-request cancellation.
A failed shutdown reports its source and leaves the same worker owning the engine for
explicit retry. Cancelling the shutdown future does not reopen admission or discard
that retry owner: its next shutdown call observes the same attempt's result before
another retry starts. Successful shutdown reports cleanup success, not recovery from
an earlier execution fault; handles retain that fault as their diagnostic.
Dropping the owner requests final shutdown without joining; the
worker retains resources through cleanup, with the runtime's conservative quarantine
on unresolved completion. Explicit shutdown is required for error reporting.

Worker exit closes queue insertion before explicitly draining queued acknowledgements.
Flume receiver disconnection alone does not drop queued values while senders remain;
otherwise a queued reply sender can keep a waiting client alive forever.
Worker unwind is reported distinctly, never as successful exhaustion. Defensive engine
Drop still establishes completion or retains device-visible ownership. Process abort,
OOM abort and a backend that never returns cannot promise recoverable notification.
No force-kill timeout may release device-visible memory.

## Text application facade

Status: implemented; host-qualified, with the CUDA lifecycle, cancellation and
multi-request determinism gates passing on the RTX 4090. Retained input/options bounds
are separately host-qualified: spare capacity, chat-container storage, joint allocations
and stop-list copies reject before preprocessing queue insertion, even with a blocked
worker. These heap-capacity checks are not process-RSS or device-memory qualification.
The one accepted deviation from the first draft is that this layer does **not** own an
architecture registry or a second worker: it owns text preprocessing, incremental
decoding and offline batching above the owned token driver. It holds no scheduler,
no second execution loop and no device state.

`TextOwner` is the non-cloneable lifecycle owner; it owns the token driver's
shutdown owner and the preprocessing pool. `TextModel` is a cloneable generation
handle over the same worker. Assembly rejects a shutdown owner and generation handle
that came from different workers, because that pair would shut down one worker while
another served requests. `TextConfig` carries preprocessing worker count and batch
window; `ProcessorLimits` carries the text byte bounds. CUDA/GGUF assembly is
`TextOwner::load`, and it is the only part that needs a device.

For CUDA text loading, `LoadOptions::max_sequences` selects both prepared sequence
capacity and the AR active-sequence limit; assembly must not silently apply the
runtime's default eight-request cap. Keep at least the existing default global
event grant, increasing it when necessary to cover the requested active sequences'
per-request mailboxes. Validate the active-plus-queued count and this global event
grant's arithmetic before device preparation.
Application request permits and HTTP slots remain separate bounded admission limits;
report them separately. None of these counts claims a physical kernel batch size or
proves requests feasible within a continuation-memory grant.

Stop IDs and control-token visibility are model-owned, never a model-name list in
routing or scheduling. Gemma 4 uses its generation metadata's stop IDs and otherwise
preserves special markers. The GGUF processor uses the following policy: every request carries
the artifact's declared EOS IDs, and a chat request also carries each control token its
**own** rendered prompt used: those are the delimiters that conversation's template
rendered, so a model emitting one is ending or restarting a turn rather than writing
text. Grounding and vision markers are control tokens too, which is why they stop
exactly when a conversation used them and a raw prompt adds nothing. Control tokens
decode to no bytes, so an emitted marker never reaches user-visible output, while
user-defined content markup such as a thinking or tool-call tag stays visible. Every
delivered event carries its token ID, so a structured consumer loses nothing to the
text policy. Caller-supplied stop tokens are added to this set, not replaced by it.

Implementation status and host evidence: [architecture](architecture.md#text-generation)
and [runtime evidence](../benchmarks/runtime-contract.md#owned-text-facade-host-gate-2026-09-14).

### Preprocessing ownership and bounds

The token driver accepts encoded input only. Text preprocessing therefore sits
above it and must be bounded independently:

- A caller acquires a request permit **before** preprocessing. The permit covers
  retained raw/message input, template rendering, encoded tokens and driver
  delivery until retirement. Preprocessing may not run before that reservation.
- Raw prompt/message content, rendered chat-template bytes, encoded prompt tokens,
  one vocabulary piece's bytes, decoder pending bytes and emitted delta bytes are
  separate limits. A delta can contain text withheld from many earlier tokens. Raw content and rendered
  bytes are rejected while the payload is produced, and decoded bytes are bounded as
  they are decoded. Encoded prompt tokens are bounded by those byte bounds and by an
  explicit token limit; a caller-supplied token vector is checked before admission.
  Exceeding any limit is a request-local rejection with a typed error; it is never a
  device fault.
- `ProcessorLimits::max_retained_bytes` bounds owned input and generation-option heap
  capacity **before** queue insertion, separately from `max_input_bytes` payload length.
  Include string/token capacity, `Vec<Message>` capacity and role/content allocations,
  plus caller stop-vector capacity. Reject excess capacity without copying/shrinking;
  many empty messages cannot bypass container accounting. Fixed jobs are count-bounded,
  and allocator overhead is outside this envelope. Assembly rejects aggregate-bound
  overflow; queued input/options capacity is at most permits times this envelope.
- Stop capacity plus its future runtime copy must fit the permit's encoded envelope
  before queuing. After encoding and model-stop merging, check the complete encoded
  token count and stop storage before publishing a prepared reply. Submission rechecks
  the same driver rule. The text layer does not enlarge that envelope; rejection is
  request-local and returns the unused permit. Processor workspace/rendering have their
  own bounds, not a hard process-RSS guarantee derived from retained input.
- Preprocessing runs on a fixed, bounded worker pool with a bounded queue. Jobs
  own their permit, so outstanding preprocessing cannot exceed the permit pool
  and the queue is bounded by construction. A dropped submission may let its job
  finish; the permit returns only when that work releases its input.
- Preprocessing must not occupy the execution worker or an async executor thread.
  Cancelling an async submission does not strand state: the permit is released by
  the job, not by the cancelled future.
- A pool that loses its last worker — including through a processor panic — becomes
  observably closed: queued jobs are released and later submissions are refused, so
  no caller waits on a queue nothing will serve. Once closure begins, a worker that
  is about to start a queued job drops it instead of beginning new work.
- Preparation creates one owned model-specific decoder, under the request permit,
  before publishing its prepared reply or submitting to execution. Constructor errors
  return the unused permit. Cancelled preparation drops the decoder with its reply.
  Preparation couples the decoder to a delivery lease on the same permit, before
  moving the permit into submission. That owned pair survives acknowledgement waits,
  failed/cancelled submissions and finalization/drop: consuming the underlying token
  terminal must not return capacity while decoder state is still live. This extends the existing grant, not a second
  accounting authority or device reservation. Raw token streams keep their existing
  terminal-delivery release semantics. Shared immutable vocabulary is not cloned per
  request. Pending bytes and delta bytes have explicit per-request bounds; assembly checks aggregate arithmetic with
  the permit count. These are payload/state envelopes, not allocator or process-RSS
  guarantees. Immutable vocabulary and caller-collected results are outside them.
- Gemma's worst-case pending run is one byte per generated token. Its decoder factory
  rejects output budgets exceeding the pending-byte envelope before enqueue, and
  reserves that run once. A malformed run can expand to three bytes per byte;
  exceeding the delta bound during push or finalization is request-local.

### Delivery and terminal semantics

- Each request owns model-specific incremental decoder state. Every runtime token
  retains its ID, including an empty text delta while text is withheld. Emitted
  fragments are irrevocable. GGUF retains incomplete UTF-8 code points and rejects
  invalid UTF-8. Gemma retains entire consecutive byte-fallback runs: one invalid
  byte makes every byte in that run a replacement character, including a formerly
  valid-looking prefix. Neither policy belongs in the token runtime.
- A normal or explicitly cancelled terminal consumes the decoder's finalizer once,
  including when the stop token itself was not ingested. Any trailing text is a
  token-less delta before the text terminal. GGUF replaces an incomplete final code
  point once; Gemma flushes the final fallback run. Finalization may fail.
- A push or finalization error abandons only its own request. Peers continue, and
  the failing request yields one typed error instead of a text terminal.
- A terminated stream releases its decode scratch and keeps only its terminal
  outcome until its consumer reads it, so a retained finished stream does not
  accumulate request payload.
- Owner failure preserves events already handed to stream channels, then yields
  one owner error. It does not invent successful usage or perform an ordinary
  terminal flush, and it does not relabel healthy peers as invalid.
- Text cancellation forwards to the driver's intent flag. It does not wait for
  device completion and does not discard buffered output.

### Offline batching

- Batching consumes a caller iterator lazily through a bounded admission window.
  It never collects or prepares an arbitrary iterator before admission. Lookahead
  is bounded by the window; a stalled earliest item bounds how far later items
  advance rather than admitting replacements ahead of the ordered head.
- Results are yielded in input order. Each item settles independently: an
  unencodable input, an admission failure, a decode error or an item-local
  terminal failure affects only that item. There is no batch-wide error variant;
  the collect helper returns ordered per-item outcomes.
- Collect helpers intentionally allocate the requested result. The incremental
  iterator is the bounded interface for large workloads. Dropping either abandons
  live streams without waiting for device completion; the driver retains
  retirement ownership.
- Admission is fail-fast. An item that cannot reserve because other handles hold
  permits reports overload for that item. There is no unbounded waiter queue and
  no silent spin-retry.

### Shutdown

Closing the preprocessing pool precedes driver shutdown; no new text work starts
while cleanup runs. `TextOwner` keeps the driver's retry ownership, so a failed or
cancelled shutdown leaves the same worker available and reports the earlier fault
through the handle. Successful shutdown reports cleanup success, not recovery from
an execution fault.

## Text HTTP adapter

Status: implemented; host TCP gates and a bounded CUDA HTTP gate passed.
[Qualification](../benchmarks/runtime-contract.md#initial-text-http-qualification-2026-10-13)
scopes the evidence; matched competitive and physical-peak measurement remain open.

The initial server is a thin adapter over `TextModel` and `TextOwner`, with no scheduler
or device state of its own. Its public protocol subset is documented in
[serving](serving.md). Only loopback binding is accepted; authentication, TLS and
non-local exposure remain unsupported rather than silently insecure defaults.

Bound accepted connections before spawning tasks, and reserve a separate fail-fast
HTTP request slot before reading generation bodies. Health/readiness and metrics do
not consume generation slots. Bound header bytes/count, body bytes, message count,
output tokens and response bytes. Parsing allocations are bounded by the body and
message limits; accepted text allocations also pass the facade's retained-capacity
checks before preprocessing. These envelopes exclude allocator overhead, kernel socket
buffers, immutable weights and processor workspace; they are not process-RSS bounds.

Use one HTTP/1 request per connection initially. A finite absolute connection deadline
covers slow headers/bodies, preparation, generation and stalled writes. This deliberately
limits long generations too; it is not an idle timeout or proof of device completion.
Streaming pulls one bounded SSE frame at a time directly from the owned text stream;
there is no producer task or unbounded response channel. Disconnect, deadline and server
closure drop request interest, leaving retirement with the execution owner. Non-streaming
collection has a separate byte limit, not the unbounded facade collect helper.

Normal streaming delivers ordered deltas, one finish chunk and `[DONE]`. A failure after
headers yields one error frame then `[DONE]`, never fabricated successful usage. Before
headers, malformed/unsupported input is 400, oversize is 413 and temporary overload is
503. Owner failure is server-scoped and makes readiness faulted; request-local failure
leaves peers serving. Readiness distinguishes loading, serving, closing and faulted.
Metrics use fixed names and no request/model label cardinality.

Shutdown closes transport admission and abandons connections before invoking text
shutdown. A pending load must settle into an owner before cleanup, not be detached by
cancelling a blocking load. Failed cleanup retains that same `TextOwner` for explicit
retry; a network deadline never releases device-visible ownership.

## Cross-runtime device ownership

A device-resident result needs its representation, model identity, pool charge and
producer completion dependency. The consumer must wait on that dependency before
reading. Storage cannot be overwritten until all consumer device accesses finish;
an `Arc` alone proves neither completion nor exclusive reuse.

Cancellation before downstream admission leaves the producer/transport owner
responsible for retirement. Successful admission transfers that responsibility to
the consumer. Failed handoffs retain an owner. Unknown completion quarantines the
storage and its charge rather than returning it to a free list.

Async host-to-device transfers likewise own an immutable staging buffer until the
copy completes. Reusable pinned metadata is not automatically safe transfer storage.

Pool boundedness is not progress: upstream results can fill a pool and starve the
consumer workspace needed to release them. Reserve downstream headroom or admit
whole pipeline resource envelopes for the concrete pipeline; prove this with a
constrained-pool test before generalizing it.

## Owning executable snapshot

A snapshot is the owned executable configuration: prepared weights/materializations,
adapters, processor configuration and execution resources. A numeric version is not
an owner. Mutable request state stays in the runtime, not inside a shared mutable
snapshot object protected by one global lock.

Local GGUF preparation selects one open source for metadata, processor/template,
complete-artifact identity and streamed weights. Pathnames are diagnostics, not
source identity: replacing or unlinking the pathname must not redirect existing
readers. Readers retain the source independently and share its descriptor without
sharing logical cursors. The selected file must not be overwritten or truncated
through any alias while artifact owners/readers live. Descriptor retention is not
a copied or filesystem-enforced immutable snapshot; publish updates by replacing
the pathname, not by mutating the selected file in place. Prepare the executor and
processor together before exposing the application handle.

SafeTensors file access validates a bounded header and the complete payload layout
against the selected file's length, then retains that opened file for independent,
bounded readers. File-backed artifacts retain no encoded payload in application heap;
reads use caller-provided buffers with no hidden prefetch or background queue. An explicit
in-memory artifact instead owns its supplied bytes. Readers and clones retain their
source after cache eviction or artifact drop. The same no-in-place-mutation requirement
applies to file sources. Package config/index and shards not yet opened must remain
unchanged throughout loading; retaining individual files is not an atomic package snapshot.
Shard-count limits bound cache entries, not caller-retained descriptors, metadata bytes,
decoded model weights, OS page cache or upload staging. Model/backend owners must budget
those separately before claiming a whole-loader or device-memory bound.

### Native BF16 weight preparation

The Gemma 4 loading slice materializes the already-validated text inventory,
not an executable model. Keep its BF16 bytes and row ordering unchanged; the raw
input embedding and output head borrow the same physical region. No whole-model
F32 conversion, duplicate tied allocation or GGUF value-type tag represents this path.

- Validate the complete shape/dtype/name inventory and checked aligned layout before
  allocation. One supplied `BytePool` grants the entire allocation request, including
  inter-tensor padding. A single immutable weight allocation avoids per-tensor partial
  construction and grants no permission to omit workspace or continuation charges.
- The NVIDIA owner captures a synchronous raw CUDA allocation immediately on success,
  without a fallible tracking-event step that can lose the pointer. All access belongs
  to its associated stream. Publish immutable weights only after every parameter's
  exact-length upload and completion; a failed set cannot be partly used or repaired
  by mixing another source into it.
- Stream one parameter at a time through one explicitly bounded owner-held host buffer.
  Establish copy completion before buffer reuse. Failed/partial copy or failed drain
  retains allocation, charge, stream/context and staging together. Readback uses the
  same ownership rule: do not return an unowned borrowed destination after uncertain DMA.
- Cleanup drains the associated stream, acknowledges synchronous deallocation, then
  releases the charge. Failed drain retains a retry owner. If a failed free leaves
  allocation identity uncertain, quarantine rather than blindly retry a possibly
  released/reused pointer. Defensive Drop must not refund an unresolved allocation or
  free staging the device may still access; explicit cleanup is the observable path.
- The model's load failure carries that cleanup ownership until settlement. Loader
  publication is separate from request submission; cancellation of a load future does
  not establish completion. A loaded weight set alone is not Gemma execution support.

The [native BF16 device gate](../benchmarks/gemma4-reference/README.md#native-bf16-device-weight-gate)
qualifies real-package byte preservation, tied ownership and successful load/unload,
plus bounded partial-source failure cleanup. Failed drain/free and unwind retention
remain host fault-injection evidence, not injected hardware failures. Requested bytes
plus padding do not measure driver rounding, context/library storage or full inference
peaks; keep those margins separate from encoded payload totals and logical grants.

### Native BF16 linear execution

The linear execution slice computes row-major `output = input × weightᵀ` through
cuBLAS with BF16 operands/results, FP32 accumulation and reduced-precision reductions
disabled. It is a backend operation, not a model or universal tensor interface.
Matrices and the explicit cuBLAS workspace are prepared and charged before launch;
immutable weight slots retain their source layout. Model owners bind logical roles.

A call validates the complete dimensions, initialized input, actual allocation ranges
and common stable stream before invalidating output or entering cuBLAS. Reject CUDA's
per-thread default pseudo-stream: moving a Send owner/error to another thread would
drain a different queue despite retaining the same stream wrapper. A successful call
includes stream completion and publishes initialized output. Error or unwind after
possible enqueue retains weights, both matrices, handle/workspace and stream together.
Those retained references forbid close, staging access or reuse until an explicit
successful drain. Reference release alone is not device completion; defensive Drop
retries drain or quarantines the entire access envelope. Uncertain handle destruction,
like uncertain allocation free, is never blindly retried. No asynchronous application
submission, cancellation protocol or model-level failure recovery is added here.

Numerical qualification starts with exactly representable dot products and independent
BF16 round-to-nearest-even results, including odd dimensions and multi-row transposition.
That component gate is not whole-model parity or a performance claim. Real model
execution still requires independently justified numerical acceptance and mixed-KV
lifetime qualification.

### Native BF16 normalization

Grouped RMS uses the same completion-owned matrix storage as linear execution.
The model selects epsilon, direct learned scales and contiguous group widths;
NVIDIA validates equal input/output shapes, whole groups and live BF16 scale ranges
before invalidating output. Scale-free RMS does not allocate an all-ones parameter.
Gemma normalizes Q/K per head and PLE per layer slice. Fresh V is normalized without
scale at its KV owner; shared-tail consumers must not normalize cached V again.

Possible kernel enqueue retains compiled code/module, associated stable stream,
optional scale allocation and both matrices together. Successful stream completion
publishes output; failure or unwind follows the same retirement/quarantine rule as
linear execution. Kernel code/context overhead is not an application buffer grant;
this operation adds no global device workspace or application submission protocol.
The [RMS component gate](../benchmarks/gemma4-reference/README.md#native-bf16-rms-component-gate)
owns its bounded numerical policy, not a whole-model error budget.

### Snapshot replacement

Start with one loaded execution owner's lifetime pinning one snapshot. Implement
**drain-and-replace** first: stop new admission, finish/cancel old work, establish
completion, release old resources, then publish the replacement. Concurrent old/new
snapshots are a distinct double-buffered policy requiring measured extra capacity,
not another name for draining.

An accepted request uses one coherent snapshot. Derived-state reuse requires semantic
compatibility (parameters, adapters, processor, numerical policy and trust domain)
and physical compatibility (backend, encoding, layout, addresses where captured).
Do not partially reuse hybrid continuation unless all required components form a
valid boundary. Exact snapshot identity is a safe initial compatibility policy;
relax it only with evidence.

Trainer-to-rollout updates eventually prepare a serving materialization and publish
it at an explicit version boundary. Inference must never observe tensors while an
optimizer mutates them. Sharing logical parameter identity does not require training
master weights and quantized serving weights to share an allocation.

## Acceptance evidence

- Positive partial prefill, final-prefill output rules, credit refunds, malformed
  mixed rows with no logical commit, cancellation during delayed completion.
- Discard before admission, during execution, after terminal publication, on decoder
  errors and on batch failure; stalled and healthy consumers together; bounded memory.
- Multiple callers, queue saturation, cancellation under saturation, owner shutdown,
  lost-wakeup interleavings and worker failure propagation.
- A real asynchronous encoder: aggregate resource pressure, partial enqueue failure,
  abandonment of new persistent growth, consumer stalls, failed handoffs and delayed
  producer/consumer completion without premature reuse or leaked reservations.
- AR continuation: per-request capacity within the declared bound, capacity exhaustion
  as waitable backpressure rather than a fault or an unbounded deferral, reuse only at a
  boundary whose every component is valid, eviction of unreferenced storage, preemption
  by recomputation, and cancellation that leaves no continuation charge behind.
- Small payloads with large capacities, many empty messages, oversized options behind
  blocked preprocessing, non-AR input-byte saturation and local shutdown with sibling progress.
- Multiple independent submissions, out-of-order completion, immutable staging/table ownership,
  EOS/cancellation with extra physical work, and failed-drain retention without slot reuse.
- Speculative zero/partial acceptance, correction, early stop, all-component rollback and
  sampling consistency; no cache publication of tentative state.
- Snapshot draining and replacement, incompatible adapter/processor state, and later
  explicit overlap accounting before supporting concurrent versions.

Host fixtures prove lifecycle transitions, not GPU memory safety or model support.
Run real device qualification for the affected execution path.
