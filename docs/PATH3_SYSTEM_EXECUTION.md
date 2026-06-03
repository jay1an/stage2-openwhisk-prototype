# Path 3 System Execution Architecture

This document specifies how the path 3 components (entry prewarm, JIT
downstream prewarming, dynamic plan adjustment) execute and cooperate at
runtime. It is the result of extended design alignment (2026-05-29/30).

Companion documents:
- `ARCHITECTURE_DECISIONS.md` - high-level decisions
- `PATH3_PLANNER_DESIGN.md` - planner / search algorithm
- `ANALYTICAL_RISK_MODEL.md` - closed-form risk model

---

## 0. Components Overview

```
[Entry Prewarm Component]  (independent process/service)
   observes workflow arrival times
   -> predicts next-window entry arrivals (oracle first, then advanced predictor)
   -> issues entry warmup invocations to keep entry warm pool stocked

[Replay Client]  (open-loop workflow launcher)
   reads schedule, launches workflows at their arrival times
   tags each request with an SLO class (premium/free, random by ratio)
   selects that class's precomputed plan
   runs run_one_workflow(plan, slo_class)

[run_one_workflow]  (per-workflow executor, orchestrator)
   for each stage, at stage START:
     - dynamic plan adjustment (decide downstream tiers)
     - schedule JIT warmup for downstream (precise timing)
   executes stages in DAG order

[JIT Warmup Scheduler]  (single background thread, shared)
   priority queue of (fire_time, stage, tier) tasks
   pops due tasks, issues {__warmup: true} invocations
   supports upsert (update a queued task)

[OpenWhisk]  (unchanged platform)
   actions have a __warmup fast-return branch (the only action-side change)
```

Key principle: ALL scheduling intelligence (when to warm, whom, which
tier) lives in the orchestrator. The action's only concession is a
3-line `__warmup` fast-return branch.

---

## 1. SLO Class Assignment

- Each workflow request is tagged premium or free at launch, randomly
  by a configured ratio (e.g., 3:7).
- The planner precomputes ONE plan per SLO class (offline, beam search).
- At launch, the request uses its class's plan as the initial plan.
- Plans are per-stage heterogeneous tier maps: {stage_name: tier_mb}.

---

## 2. The __warmup Action Branch (only action-side change)

OpenWhisk's native prewarm (stemcell) only pre-creates EMPTY containers
at the (runtime, memory) level; the action's code is loaded only when a
real request arrives (/init). It cannot pre-warm a SPECIFIC action to
the WarmedData state.

Therefore, to make a specific action variant (e.g.,
wf_civic_match_face_2048) warm, we must invoke it once. To avoid wasting
a full execution (a 512 MB workflow stage takes tens of seconds), the
action gets a fast-return branch:

```python
def main(args):
    if args.get("__warmup"):
        return {"warmed": True}   # container created + code loaded; no work
    # ... normal action body ...
```

This is the standard FaaS warmup idiom (AWS Lambda does the same). The
scheduling logic stays entirely in the orchestrator; the action only
returns early when asked.

---

## 3. Per-Stage Execution Model (the unified trigger point)

There is ONE trigger point: when a stage STARTS executing. Both dynamic
plan adjustment and JIT scheduling hang off this point.

```
when stage X starts (record T_X_start):

  # (a) Dynamic plan adjustment — runs IMMEDIATELY at stage start
  recompute downstream tiers using:
    - upstream stages: ACTUAL measured completion times
    - stage X itself: PREDICTED completion (latency model; X just started,
      but warm execution variance is small so the estimate is good)
    - downstream stages: PREDICTED (latency model)
  via incremental beam search (only pending downstream stages are free
  variables; upstream is fixed/done)
  -> may upgrade downstream stage tiers to meet the SLO constraint

  # (b) JIT scheduling — schedules warmup for downstream, fired LATER
  for B in X.downstream:
    if all of B's upstreams have already STARTED:   # only the last upstream schedules B
      B_needed = max over B's upstreams of (upstream_start + D_upstream)
      C_B = cold overhead of B at its (dynamic-decided) tier
      JIT_fire = B_needed - C_B - safety_margin
      upsert(scheduler_queue, key=B, fire_time=JIT_fire, tier=B_tier)
    else:
      pass   # a later-starting upstream will schedule B

  # (c) invoke stage X
  invoke X's tier variant (X should already be warm from the previous
  round's JIT, or from entry prewarm if X is the entry stage)

when stage X completes:
  record actual completion time (feeds the next stages' dynamic step)
```

### Why "dynamic computed early, warmup fired late"

- Dynamic must decide downstream tier ASAP (so we know WHICH tier variant
  to warm). It runs immediately at stage start.
- JIT warmup must NOT fire at stage start — that would warm the container
  too early, wasting paused time and risking keepalive expiry. It fires
  at the precise computed JIT_fire time.

### Why only the LAST upstream schedules B (race fix)

If every upstream scheduled B at its own start, an early upstream (X1)
would compute B_needed using incomplete info (X2 not started yet), and
the scheduler might pop and fire B's warmup too early — before X2 even
starts.

Fix: B's warmup is scheduled ONLY when ALL of B's upstreams have started.
At that moment every upstream's start_time is known, so
B_needed = max(upstream_start + D_upstream) is accurate.

This still leaves enough lead time as long as
D_last_upstream > C_B + safety_margin (for civic_alert: upstream warm
~3s > cold ~2s + margin 0.5s, OK).

### Container return-to-pool delay (P3.A finding)

P3.A verification revealed: a container only becomes reusable after it
enters the Paused state (Running -> Ready -> Pausing -> Paused). This
takes a sub-second delay (pauseGrace ~50ms + transition overhead) after
the warmup invocation's code finishes.

Observed: invoking normal IMMEDIATELY (0.14s) after warmup did NOT reuse
the container (it was still transitioning, not yet in the pool), so
OpenWhisk spun up a second container. Waiting ~5s allowed reuse.

Implication for the JIT lead-time condition: the precise requirement is
  D_last_upstream > C_B + safety_margin + return_to_pool_delay
where return_to_pool_delay is sub-second. For civic_alert this still
holds comfortably (upstream ~3s >> cold ~2s + margin 0.5s + pool ~0.1s).

This is NOT a problem for our JIT design because the real invoke of B
happens only after the upstream completes (~3s after warmup fired), far
longer than the return-to-pool delay. The concern only arises if an
upstream stage executes faster than C_B + return_to_pool_delay, which
does not occur for civic_alert. Future workflows with very fast upstream
stages would need an alternative mechanism for those stages.

Implementation: maintain the set of started stages (run_one_workflow
already has a `running`/`completed` set). When stage X starts, for each
downstream B, check if all B.parents are in started-set; only then
schedule B.

---

## 4. JIT Warmup Scheduler (method B: priority queue)

A single background thread shared across all in-flight workflows.

```
queue: min-heap of (fire_time, task_id, stage_variant, ...)
index: dict task_id -> current queued entry (for upsert)

loop:
  if queue empty: wait for new task
  else:
    peek earliest fire_time
    sleep until fire_time (or wake early if a new/updated task arrives)
    pop due task
    issue {__warmup: true} invocation to the stage's tier variant
```

### Upsert support

When a later upstream starts and recomputes B_needed (changing B's
JIT_fire), the scheduler must UPDATE B's queued entry, not add a
duplicate. Keyed by (request_id, downstream_stage). Per Q2 fix above,
in the common case B is only scheduled once (by its last upstream), but
upsert is kept for safety against re-scheduling.

### Why method B over threading.Timer

- Concurrency may reach ~200 queued warmups; 200 Timer threads wasteful
- Priority queue: one thread, O(log n) insert, cheap
- Upsert / cancel is natural in a queue, awkward with Timers

---

## 5. Dynamic Plan Adjustment (rolling, incremental beam)

At each stage start, recompute downstream tiers. Inputs:
- upstream completion: ACTUAL (measured)
- current stage: PREDICTED completion (latency model)
- downstream: PREDICTED

Algorithm: incremental beam search starting from the current plan,
varying only the PENDING downstream stages' tiers. Since upstream is
fixed and the number of pending stages shrinks as the workflow
progresses, the search gets cheaper at each step:
- at entry: 5 pending stages
- at last stage: 0 pending (no search needed)

Objective: keep P(remaining E2E > remaining SLO budget) <= class target,
at minimum added cost. Uses the closed-form risk model (Mode 2: JIT
assumed, downstream cold hidden) with the remaining budget.

Rolling benefit: each step has more measured info and less prediction,
so decisions get progressively more accurate. This is stronger than
"only recover when entry is cold" — every stage self-adapts.

The same beam search code is reused offline (initial plan) and online
(this rolling adjustment), which is a paper contribution.

---

## 6. Entry Prewarm Component (independent)

A separate process/service, decoupled from the replay client.

```
observe: workflow arrival timestamps (the entry-stage launch stream)
predict: next-window entry arrival count
   - v1: oracle (read actual next-window count from the schedule) to
     validate the prewarm mechanism in isolation
   - v2: advanced predictor (LSTM / online selector from Stage 2;
     NOT the simple EWMA/burst-aware) for realistic accuracy
prewarm: issue N entry warmup invocations to maintain the warm pool
   N = ceil(safety_factor * predicted_arrivals) - surviving_warm
```

Decoupling rationale: entry prewarm is logically a standalone platform
service that observes the arrival stream and feeds predictions to the
warming mechanism. Keeping it separate from the replay client mirrors a
real deployment (prewarm is a resident service, not embedded in a client).

Imperfect prediction is acceptable: when prediction misses a burst and
entry goes cold, the dynamic plan adjustment recovers by upgrading
downstream tiers. Entry prewarm need not be perfect, only mostly right.

---

## 7. Race Condition (accepted in v1)

During the safety_margin window (between a warmed container becoming
ready and this workflow's real invoke of it), another workflow's request
for the same (action, tier) may consume the warm container, forcing this
workflow to cold start.

Decision (v1): ACCEPT the race, do NOT modify OpenWhisk.

Mitigations:
- Small margin (e.g., 1σ) shrinks the exposure window
- Self-compensation: a stolen warm container means some OTHER workflow's
  cold start was avoided; aggregate warm container count is preserved,
  so race does not increase total cold starts, only shuffles which
  request pays
- Multi-node dispersion (ρ≈0) further reduces same-node contention

Measurement: during end-to-end replay, count how many requests are
affected by the race. If the impact is large (e.g., >5% of requests),
revisit by modifying the OpenWhisk invoker.

This limitation is logged; we will intrude into the invoker code later
only if measured impact warrants it.

### Options considered and their disposition (2026-05-30)

We discussed three approaches to the race; all DEFERRED in favor of
first fixing JIT timing and measuring actual race impact.

**Option A: orchestrator warm-pool ledger (REJECTED)**
- Idea: orchestrator keeps a local ledger of "which actions have warm
  containers alive (within keepalive)" by recording every invocation it
  issues, and predicts cold/warm before dispatching.
- Rejected by owner: the ledger has too many accuracy problems (race it
  can't see, OpenWhisk internal reclamation, multi-node placement), so
  its predictions would be unreliable.

**Option B: OpenWhisk container reservation (DEFERRED, heavy)**
- Idea: warmup tags a container with a token; the real request with the
  same token only matches that container. True reservation.
- Cost: ~2-3 weeks. Changes ContainerPool.scala (token-filtered
  matching), ContainerProxy.scala (token state), ActivationMessage
  schema (token field), Controller routing (same-invoker affinity), plus
  rebuild + redeploy OpenWhisk. High risk (deadlock, container leak,
  perf regression) and ongoing merge burden on OW upgrades.
- Problem: a reserved-but-unused container is locked from others — worse
  waste than the race if prediction is wrong (needs timeout release).

**Option C: warmup does not reuse free-pool containers (DEFERRED, lighter)**
- Idea: when a request carries __warmup, skip the freePool lookup and
  force a new container; normal requests reuse as usual. Only one branch
  in ContainerPool.scala.
- Cost: ~3-5 days (lighter than B; no token matching/binding/routing).
  Still requires Scala change + OW rebuild + redeploy.
- Problem: "always new container" OVER-CREATES. If the freePool already
  has M warm containers, warmup blindly adds more, exceeding the needed
  count; the surplus idles and is reclaimed. The correct behavior is
  "top up to the deficit (N - M)", which again needs to know M (the
  rejected ledger or an OpenWhisk state query).

### Decision: fix timing first, measure, then decide

Root cause of the P3.C race exposure was NOT free-pool reuse — it was
JIT timing being computed with the WARM predicted duration for a COLD
entry, so warmups fired far too early and sat paused for seconds
(large exposure window).

The fix (P3.C-fix) is to compute JIT timing with the CORRECT predicted
completion time:
- At this stage (no entry prewarm yet), the entry stage is ALWAYS cold,
  so the orchestrator hard-assumes entry is cold and uses entry's
  cold-start completion time (warm exec + cold overhead) when computing
  downstream needed_at. No ledger needed — the answer is deterministic.
- This shrinks the warmup exposure window to ~margin (e.g., 0.6s), which
  makes the race small.

Plan:
1. P3.C-fix: correct timing (entry assumed cold) + speculative scheduling
2. Measure race impact in end-to-end replay
3. If race still > ~5%: implement Option C (warmup-no-reuse, lighter) as
   the preferred invoker change; Option B (full reservation) only if C
   proves insufficient

When entry prewarm (P3.E) is added, the entry will mostly be warm; the
hard "entry is cold" assumption is then replaced by "entry is warm
unless prewarm missed", with dynamic plan recovering the misses.

---

## 8. Full Request Lifecycle (putting it together)

```
1. Request arrives; replay client tags it premium/free (random by ratio)
2. Select that class's precomputed initial plan
3. [entry prewarm component, in background] has been warming the entry
   pool by window prediction
4. invoke entry stage (its tier variant)
   - hits warm if entry prewarm covered it
   - cold starts if prediction missed (recovered later by dynamic)
5. entry starts -> dynamic decides downstream tiers; JIT scheduler is
   updated for downstream warmups
6. entry completes (record actual time)
7. next stage starts -> its real invoke should hit the JIT-warmed
   container; dynamic re-evaluates remaining downstream; JIT schedules
   further downstream
8. repeat until workflow completes
```

The entry stage is special: it has no upstream to hide its cold start,
so it relies on the entry prewarm component. All downstream stages rely
on JIT (cold hidden by upstream execution).

---

## 8.5 P3.C-fix: Speculative Absolute-Time JIT Scheduling (2026-05-30)

P3.C (first version) scheduled a downstream's warmup when its upstream
was SUBMITTED. Since a stage is submitted only after ALL its upstreams
COMPLETED, and it used the WARM predicted duration even for a cold entry,
short-upstream stages had fire_at in the past (late_jit) and the cold
entry made downstream warmups fire far too early.

P3.C results (N=10, every workflow forced cold baseline):
- estimate_pose cold rate 100% -> 30% (entry cold gave it long cover)
- match_face 100% -> 70%, classify_scene 100% -> 60%,
  translate_alert 100% -> 50%
- E2E 38.4s -> 18.9s (JIT clearly works)
- late_jit_count 30/40 (timing too tight for short stages)

### The fix: schedule all warmups up front (speculative), absolute time

Instead of scheduling at upstream-submit time, at WORKFLOW START predict
each stage's absolute needed_at via the DAG + latency model, and enqueue
ALL warmups at their absolute fire_time. Then, as stages actually
complete, upsert (correct) the still-pending warmups using the measured
completion times.

```
at workflow start:
  cold/warm pre-judgement:
    - entry stage: assumed COLD (no entry prewarm yet) -> use cold-start
      completion time (warm exec + cold overhead)
    - downstream stages: assumed WARM (JIT will warm them) -> use warm
      completion time
  recursively predict each stage's absolute needed_at down the DAG
  for each stage B (non-entry):
    needed_at(B) = max over parents p of predicted_completion(p)
    fire_at(B)   = needed_at(B) - C_B(cold overhead at B's tier) - margin
    enqueue warmup(B) at fire_at(B)   (absolute monotonic time)

as each stage completes (real time + real cold/warm known):
  recompute predicted completion of pending downstream stages using the
  measured upstream completion
  upsert their warmup fire_times in the scheduler
```

Why this removes late_jit: warmups are enqueued at workflow start (well
before any short upstream submits), so there is enough lead time. The
absolute needed_at uses the correct (cold for entry) completion estimate,
so downstream warmups are neither too early nor too late.

This is still JIT-ONLY (no dynamic tier change). Tier comes from the
fixed initial plan. Dynamic tier change + warmup cancel/upsert is P3.D.

### Interface left for P3.D (dynamic)

When P3.D later changes a downstream tier mid-execution, it must
cancel the old-tier warmup and schedule the new-tier one. The JIT
scheduler (P3.B) already supports cancel + upsert for this.

---

## 8.6 Cold-start facts and the JIT-coverage / dynamic-upgrade tension (2026-05-30)

### Cold start is a stable ~2s; no "first cold start is much longer"

Verified from the 9-tier multi-node sweep (per stage, per tier):
- Cold overhead (cold dispatch - warm action) is STABLE at ~1.6-2.3s
  across ALL tiers and stages (mean ~2.0s). It does NOT scale with tier.
- Components: ow_init_ms mean 202ms / p95 416ms (code load — small,
  image is local), ow_wait_ms mean 1805ms / p50 2176ms (scheduling +
  container create — the bulk).
- Changing memory/cpu does NOT re-pull the image (it is cached locally);
  ow_init ~200ms confirms code/image load is fast. The "seconds to tens
  of seconds image pull" in the literature is the REMOTE-registry
  first-pull case, which does not apply to this cluster.

Correction: an earlier hypothesis that "first cold start after redeploy
is 8-10s" was speculation with no data support and is WITHDRAWN. Cold
start is ~2s steady-state here.

### Per-tier warm execution times (sweep, action_duration_ms)

| tier | detect | estimate | match | classify | translate |
|---|---|---|---|---|---|
| 512  | 8305 | 6132 | 9291 | 6978 | 6722 |
| 768  | 4069 | 4105 | 5327 | 5111 | 4561 |
| 1024 | 3595 | 3090 | 4193 | 3888 | 2986 |
| 1280 | 2836 | 2533 | 3441 | 3223 | 2874 |
| 1536 | 2358 | 2410 | 3466 | 3130 | 2791 |
| 2048 | 2489 | 2046 | 2417 | 2440 | 2458 |
| 2560 | 2245 | 1722 | 2517 | 2184 | 2178 |
| 3072 | 2011 | 1986 | 2379 | 1694 | 2179 |
| 3840 | 1896 | 1843 | 2079 | 1717 | 2064 |

### The JIT-coverage vs dynamic-upgrade tension (核心设计约束 for P3.D)

At the FASTEST tier (3840), warm execution is ~1700-2100ms, which is
≈ the cold overhead (~2000ms). So at high tiers, a single upstream's
execution time CANNOT cover a downstream's cold start:
  single-hop needs D_upstream > C_downstream + margin
  at 3840: ~1900ms < ~2000ms + 600ms = 2600ms  -> FAILS

Implication: dynamic plan upgrading a stage to a high tier (to meet SLO
faster) SHORTENS its execution, WEAKENING its ability to hide the
downstream's cold start via JIT. The two mechanisms share execution time
as a resource: it is both a latency cost AND a cold-start-hiding budget.

Decisions (owner-aligned 2026-05-30):
1. Keep CROSS-LEVEL speculative warmup (already in P3.C-fix). Do NOT
   rely on single-hop coverage. Warmups are enqueued at workflow start
   using the cumulative lead time of all preceding stages, so a fast
   upstream alone need not cover the downstream cold start.
2. Do NOT inflate action workload to force single-hop coverage
   (rejected: requires re-sweep/re-model/re-deploy, slows E2E, less
   natural). Cross-level speculative is preferred.
3. Cross-level speculative has a physical bound:
     C_downstream + margin  <  lead_time  <  keepalive
   Lower bound: enough to cover cold start. Upper bound: a warmup fired
   too early sits paused past keepalive and is reclaimed (wasted).
   When a workflow is short or all stages are at high tiers, total lead
   time may be < C_downstream for early stages — those stages then
   cannot be JIT-covered.
4. P3.D dynamic MUST model JIT coverage: when deciding to upgrade a
   stage, account for the loss of its cold-hiding capability for
   downstream. The closed-form risk model's Mode-2 assumption
   (p_downstream_cold ≈ 0) breaks when an upstream is too fast; those
   stages' cold must then be counted. A counter-intuitive but correct
   strategy: dynamic may DELIBERATELY keep an upstream at a lower tier
   (slower) to preserve its ability to hide a downstream cold start.

This "execution time as a dual-purpose budget (latency + cold-hiding)"
is a novel modeling angle vs ORION/Jolteon (static sizing, no JIT).

### Open: estimate_pose anomaly (to diagnose before P3.E)

P3.C-fix gave estimate_pose 90% cold (worse than P3.C's 30%), while
match/classify/translate improved greatly. By the data, estimate's JIT
timing should make it warm:
  detect@1536 predicted completion = warm(2358) + cold(1815) = 4173ms
    (matches measured cold dispatch 4174ms — prediction is near-perfect)
  estimate fire_at = 4173 - C_est(1823) - margin(600) = 1750ms
  estimate warmup container ~3750ms ready -> paused ~3800ms
  detect completes 4174ms -> estimate needed 4174ms > 3800ms -> SHOULD be warm
Yet measured 90% cold. There is an unexplained factor. Diagnose with a
--skip-reset run and per-request timing trace (warmup fire time,
container ready time, real invoke time, cold_like) BEFORE proceeding to
P3.E. Do not carry an unexplained anomaly forward.

---

## 8.7 P3.C-diag findings + P3.C-sync design (2026-05-31)

### Diagnosis verdict (per-request timing traces)

JIT IS effective for far stages, and the estimate_pose anomaly is a
return-to-pool settle issue, not a timing miscalculation.

Evidence (Experiment B, per-run redeploy):
| stage | cold | same_container | gap(warmup_ready->real_invoke) |
|---|---|---|---|
| detect_object | 10/10 | 0/10 | (entry, no warmup — expected cold) |
| estimate_pose | 7/10 | 3/10 | gap_mean 1581ms, gap_MIN 317ms |
| match_face | 1/10 | 9/10 | gap_mean 2640ms |
| classify_scene | 4/10 | 6/10 | gap_min -533ms |
| translate_alert | 0/10 | 10/10 | gap_mean 3619ms |

- Cold rows are exactly the same_container=False rows; warm rows are
  same_container=True. So when the real invoke hits the warmup's
  container, it is warm; when it does not, it cold-starts a new one.
- match/translate have LARGE gaps (2.6-3.6s) -> warmup container fully
  settled into the Paused pool -> real invoke reliably hits it.
- estimate (first hop) has a TINY gap (0.3s) because the warmup
  invocation itself costs a full cold start (~2.3s = ow_wait 2.16s +
  init 0.14s), consuming most of the first hop's lead time. The real
  invoke arrives before the warmup container has settled into Paused ->
  OpenWhisk creates a new (cold) container.
- Experiment A (--skip-reset, near-real deployment): estimate cold only
  on run 1, warm runs 2-10 (hits the warm container left by the previous
  run). So the anomaly is amplified by per-run redeploy.

Literature check: in real serverless, >50% of functions execute <100ms
while cold starts are 0.1-2s. So single-hop execution generally CANNOT
hide cold starts; this is exactly why prewarming is researched. Our mock
actions (2-9s) are already far longer than real functions, so inflating
action workload is the wrong direction.

### The first-hop problem and why prewarm is NOT the answer

The first hop (entry's direct downstream) is the hardest for JIT because:
- Its lead time is just the entry stage's execution time.
- A warmup invocation itself takes ~2.3s (a full cold start) to create
  the container.
- So the first hop needs entry_exec > 2.3s + settle, which may not hold,
  especially once entry prewarm makes the entry stage fast.

Owner's better idea (adopted): instead of handing the first hop to
prewarm, the orchestrator should NOT blindly fire the real invoke and
eat a cold start. It should wait (bounded) for its own warmup container
to be ready, then dispatch -> hit warm.

### P3.C-sync: warmup-synchronized dispatch

Before issuing a stage's real invoke, the orchestrator checks the status
of ITS OWN warmup for that stage (a known fact — the warmup is a
blocking call it issued, so it knows when it returns):

```
before real invoke of stage X:
  if X's warmup has COMPLETED (blocking call returned):
      container is ready/warm -> dispatch immediately (no wait)
  elif X's warmup is IN FLIGHT (issued, not yet returned):
      ready_time = warmup_completion + pauseGrace
      wait_needed = ready_time - now
      if wait_needed < cold_overhead:   # waiting beats cold-starting
          sleep(wait_needed); then dispatch -> hit warm
      else:
          dispatch now (waiting not worthwhile)
  else (no warmup issued, e.g. entry):
      dispatch now
```

Why bounded wait always beats blind cold start: the warmup was fired
earlier (JIT lead), so the warmup container's creation started BEFORE a
fresh cold start would. Waiting for it is always <= cold-starting anew.

Why NOT query the global container pool: we considered letting the
orchestrator query "does this action currently have a warm container"
to skip waiting when already warm. Rejected as infeasible/not worth it:
- Prometheus scrape interval is 15-30s; far too stale for ms-level
  dispatch decisions (literature: Prometheus unsuitable for ms
  granularity). OpenWhisk's pool metrics are also mostly aggregate, not
  per-action warm count.
- Adding a real-time OpenWhisk query interface means invoker source
  changes + a network round trip per dispatch (slows dispatch) + still
  a staleness gap.
- The orchestrator's OWN warmup-completion status (a deterministic local
  fact, NOT the rejected ledger's global inference) already answers
  "is it warm?" for the common case. The only thing global-pool query
  would add is avoiding an over-wait when ANOTHER request left a warm
  container — and that over-wait costs only a little extra wait, never a
  cold start. Not worth the cost.

This is distinct from the rejected ledger: it uses only the certain
return of a blocking warmup call this orchestrator issued, not a global
inference subject to race/reclamation/multi-node blind spots.

Note: bounded wait adds a little latency to that stage's start, but
since wait < cold_overhead, net E2E is lower than blind cold start.
Multi-workflow race can still let another request grab the container
in the instant before dispatch; that is the accepted v1 race, measured
later.

### Division of labor (revised, owner-aligned)

- entry stage (detect): has no upstream, cannot JIT -> entry prewarm
  (P3.E) keeps it warm.
- first hop and beyond (estimate, match, ...): JIT + warmup-synchronized
  bounded-wait dispatch. NO prewarm coverage needed for the first hop;
  the bounded wait makes first-hop JIT effective.

---

## 8.8 P3.C-sync results + residual analysis (2026-05-31)

Implemented warmup-synchronized bounded-wait dispatch. Results (N=10,
per-run redeploy cold-stress):

| stage | off | no-sync | sync | sync waited mean |
|---|---|---|---|---|
| detect_object | 100% | 100% | 100% | 0ms (entry) |
| estimate_pose | 100% | 80% | 30% | 1251ms |
| match_face | 100% | 30% | 0% | 52ms |
| classify_scene | 100% | 10% | 0% | 392ms |
| translate_alert | 100% | 20% | 0% | 112ms |

- match/classify/translate: cold-stress cold rate -> 0% with small wait.
- estimate_pose (first hop): 80% -> 30%, same_container 20% -> 70%.
- E2E: off 23.1s, no-sync 18.0s, sync 18.0s (sync trims tail, mean ~same).
- skip-reset sync E2E 14.4s.

### Residual estimate_pose 30% root cause

A diagnostic row showed: jit_sync_status=completed, waited 1772ms,
gap(warmup_ready->real)=757ms, yet same_container=False, cold_like=True,
real_ow_wait=2122ms. I.e. the orchestrator waited for its own warmup to
COMPLETE, but OpenWhisk STILL routed the real invoke to a NEW container.

So "wait for own warmup completion" does NOT guarantee the real invoke
reuses that container — OpenWhisk's scheduling does not guarantee reuse.
For far stages the warmup container settles into the Paused pool with a
large buffer (gap 2.6-3.6s) so reuse is reliable; for the first hop the
warmup completes only ~0.3-0.8s before the real invoke, and even a
completed warmup may not be in the schedulable Paused pool yet, OR
OpenWhisk independently chose a new container.

This is NOT a logic bug and NOT a cross-workflow race (the test runs
workflows strictly sequentially: run_one_workflow blocks, then
wait_for_scheduler_empty, before the next workflow). It is the inherent
"OpenWhisk does not guarantee warmup-container reuse" limitation, worst
in the first hop where timing is tightest.

### Concurrency / race status in current tests

- Cross-workflow race: NONE (sequential test, one workflow fully
  finishes before the next).
- Within-workflow same-stage multiple warmups: NONE (JitScheduler
  upsert dedups by task_key = request_id:stage_name; one warmup per
  stage, only its fire_time updates).
- Within-workflow different stages: different action variants, no shared
  container.
- TRUE multi-workflow concurrency race: NOT exercised yet; will appear
  in P3.G end-to-end multi-workflow replay. That is where the accepted
  v1 race gets measured (>5% -> reconsider invoker change).

### Next: settle grace experiment

Add a settle grace after warmup completion before dispatch, and test
300ms vs 500ms, to see how low estimate_pose can go. Expectation: it may
drop to ~15-20% but likely NOT zero, because part of the residual is
OpenWhisk not guaranteeing reuse (a scheduling decision, not a settle
delay). If added grace does not help beyond ~20-30%, accept the residual
as a worst-case (cold-stress) bound; real deployment (skip-reset) is
already much better, and entry prewarm (P3.E) will further help the
first hop.

---

## 9. Implementation Subtasks (revised order)

Given this architecture, the path 3 system subtasks:

- P3.3  Deploy action variants .......................... DONE
- P3.3b Per-stage tier routing in run_one_workflow ...... DONE
- P3.A  Add __warmup fast-return branch to workflow_action.py
- P3.B  JIT warmup scheduler (priority-queue background thread)
- P3.C  Integrate JIT scheduling into run_one_workflow
        (stage-start trigger, last-upstream scheduling, precise timing)
- P3.D  Dynamic plan adjustment (rolling incremental beam in executor)
- P3.E  Entry prewarm component (oracle predictor first)
- P3.F  Entry prewarm advanced predictor (LSTM / online selector)
- P3.G  End-to-end multi-SLO replay + measurement
        (violation rates per class, cost, race impact, vs baselines)

Each subtask is verified independently before integration.

### Suggested build order (incremental, each independently testable)

1. P3.A (__warmup) — trivial, foundational
2. P3.B (scheduler) — standalone, unit-testable
3. P3.C (JIT integration) — wire scheduler into executor; test cold-hiding
4. P3.D (dynamic) — add rolling adjustment; test recovery
5. P3.E (entry prewarm, oracle) — standalone component
6. P3.G (end-to-end) — integrate all, measure
7. P3.F (advanced predictor) — swap oracle for real predictor

---

## 10. Open Items / Deferred

- Race container reservation (invoker modification): deferred unless
  measured impact > ~5%
- safety_margin exact value: start at 1σ, tune from measurements
- Advanced entry predictor choice (LSTM vs online selector): decide
  after oracle validation
- Multi-workflow joint planning, more workflows, convex baseline: all
  deferred to after the single-workflow system runs end-to-end

---

## Changelog

- 2026-05-30: Initial document. Captures the full execution architecture
  aligned across several discussion rounds: unified stage-start trigger,
  dynamic-early/warmup-late separation, last-upstream JIT scheduling
  (race fix), priority-queue scheduler with upsert, independent entry
  prewarm component, accepted race condition.
