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

## 8.9 Container reuse settle delay — MEASURED (2026-06)

scripts/probe_container_reuse_delay.py ran a clean isolated probe:
warmup an action, wait a controlled delay, then real-invoke, record hit.

Result (hit rate vs delay, both actions identical):
```
delay_s | estimate_pose_1280 | match_face_2048
0.0     | 0%                 | 0%
0.5     | 0%                 | 0%
1.0     | 0%                 | 0%
1.5     | 0%                 | 12.5%
2.0     | 100%               | 100%
2.3-4.0 | 100%               | 100%
```
- miss: same_container=False, cold_like=True, real waitTime ~1.5-1.9s
- hit: same_container=True, cold_like=False, real waitTime ~10-14ms
- Threshold is ~2.0s, CONSISTENT across both actions -> it is a shared
  OpenWhisk container-pool settle delay, not a per-stage workload trait.

### Corrected total: warmup-to-reusable ≈ 4s

The 2.0s settle is measured FROM warmup return. But the warmup
invocation itself takes ~2.0s (its own ow_wait + init to create the
container) before it returns. So end to end:
```
warmup issued
  ├─ ~2.0s : warmup ow_wait + init (create container, load code) -> warmup returns
  ├─ ~2.0s : settle (container becomes reusable from free pool)
  └─ ~4.0s total : real invoke can hit this container
```

### Implication: first-hop JIT has limited value (owner's insight)

JIT's lead time for the first hop (estimate) = the single entry stage's
execution time. Compare to warmup-to-reusable ≈ 4s:
- detect cold (~4.17s): lead time barely covers 4s; JIT can just hide it
  (but margin ~0, fragile)
- detect warm (after entry prewarm, ~2.4s): lead time 2.4s < 4s; the
  warmup container is NOT ready when estimate is needed. "Waiting" for it
  (~1.6s) ≈ cold-starting (~2s). So first-hop JIT does NOT pay off, and
  entry prewarm making detect faster makes it WORSE.

Therefore (data-backed, matches the earlier deferred Q1/Q2):
- entry (detect) AND first hop (estimate) -> covered by PREWARM (kept warm)
- second hop onward (match/classify/translate) -> JIT, because their lead
  time = cumulative multiple upstream stages >> 4s (probe shows
  match/translate hit 100% in real workflow when gap > 2.4s).

### JIT fire-time correction (separate fix)

Current fire_time = needed_at - cold_overhead - margin MISSES the settle
term. Correct: fire_time = needed_at - (cold_overhead + settle) - margin
≈ needed_at - 4s - margin. This makes the warmup fire ~2s earlier so the
container is settled before the real invoke arrives.

IMPORTANT (addresses owner's E2E concern): fire-time change moves WHEN
THE WARMUP IS ISSUED, not when the real invoke is issued. The real invoke
still fires as soon as the upstream completes. Firing warmup earlier only
makes the container ready sooner -> fewer/shorter sync waits -> E2E only
improves or stays equal, never worsens. The thing that can ADD to E2E is
the warmup-sync WAIT (real invoke waiting for settle); firing earlier
REDUCES that wait. To be quantified by the per-stage wait experiment.

Owner's "sense whether a free container already exists" point: in steady
state a free warm container exists -> real invoke hits directly, no JIT
needed. That is exactly entry prewarm's job (keep entry + first hop warm).
No runtime pool query needed (ledger/Prometheus rejected earlier): entry
+ first hop guaranteed warm by prewarm; second hop+ guaranteed by JIT.

Risk noted: any sync WAIT adds to E2E; the per-stage wait experiment will
measure how much, and whether the fire-time fix removes the need to wait.

---

## 8.10 RESOLVED: the 2s was synchronous log collection (2026-06-05)

The ~2s container-reuse settle is now fully explained and fixed, with
source-level + measurement-level closure. Two earlier hypotheses of mine
were WRONG and are corrected here.

### Root cause (source + experiment)

ContainerProxy reuse chain (ContainerProxy.scala):
```
run action -> initializeAndRun (includes collectLogs) -> RunCompleted
  -> container transitions Running -> Ready -> NeedWork (now reusable)
```
collectLogs sits between "action finished" and "container reusable".

- DockerToActivationLogStore.collectLogs (and the File variant): calls
  `container.logs(logLimit, waitForSentinel=true)` — reads container
  stdout and WAITS for the sentinel marker. On docker json-file this
  read/flush takes ~2s. The container does not become reusable until
  this returns.
- LogDriverLogStore.collectLogs: `Future.successful(ActivationLogs())`
  — returns empty immediately, does NOT read logs. Container becomes
  reusable at once.

### Experiment confirmation (probe, LogDriver vs DockerToActivation)

| delay_s | DockerToActivation hit | LogDriver hit |
|---|---|---|
| 0.0-1.5 | 0% (cold, new container) | 100% |
| 2.0+    | 100% | 100% |

Under LogDriver, EVERY delay (including 0.0s) hits with wait ~13ms.
Under DockerToActivation, delay<2s misses (the container is still in
collectLogs, not yet in the free pool, so a new cold container is made).

### State machine + timing (the full picture)

```
Running -> [collectLogs] -> Ready -> [pause-grace] -> Paused -> [idle-container] -> Removed
```
- collectLogs: ~2s under DockerToActivation, ~0 under LogDriver
- pause-grace = 10 seconds (OpenWhisk DEFAULT; we did NOT override it)
- idle-container (keepalive) = 10 seconds (we set this)

Because pause-grace is 10s and the probe's max delay is 4s, the container
stays in Ready the whole time and NEVER pauses (invoker logs show zero
pause/resume during the probe). So a real invoke within 10s of the
warmup hits a Ready container directly, wait ~13ms — no resume involved.

### Corrections to my earlier wrong claims

1. "The 2s is a platform-fixed property, switching log provider won't
   help" — WRONG. It was DockerToActivation's synchronous collectLogs.
   LogDriver removes it (verified: 0s delay now hits). I over-extrapolated
   from the File-provider result (File still reads logs, so it also
   blocked ~2s; LogDriver does not read logs at all).
2. "pauseGrace defaults to 50ms" — WRONG. It defaults to 10 seconds.
   That is why the container stays Ready across the whole 0-4s probe
   range and the earlier 2s miss was NOT pause/resume but collectLogs
   blocking the container from entering Ready/free-pool.

### Decision: LogDriver is now PERMANENT

The cluster's invoker now runs
`-Dwhisk.spi.LogStoreProvider=...LogDriverLogStoreProvider` permanently
(owner switched it, will not revert). Consequences:
- Container reuse settle: ~2s -> ~13ms (delay 0 hits).
- `wsk activation logs` no longer returns logs (logs go to the node-level
  docker/k8s log driver). We do NOT depend on activation logs; our
  trace/probe read response.result + annotations (container_id,
  cold_like, ow_init_ms, ow_wait_ms), which are unaffected (verified).
- Also offloads invoker log processing, which is the recommended setup
  for high-concurrency load (matches the planned high-pressure tests).
- ENVIRONMENT CHANGE: all experiments from here on run under LogDriver.
  Prior data based on DockerToActivation + 2s settle belongs to the old
  environment.

### Implications for JIT (simplification)

With settle ~0 and a 10s Ready window:
- warmup container is reusable immediately after the warmup returns, and
  stays Ready for ~10s (pause-grace).
- A real invoke within 10s of the warmup hits directly (~13ms), no resume.
- Therefore:
  * JIT fire-time no longer needs the settle term (set jit_fire_settle_ms
    back to 0). fire_time = needed_at - cold_overhead - margin.
  * warmup-synchronized bounded WAIT is no longer needed (container is
    already reusable when the real invoke arrives). enable_jit_sync can
    default to off.
  * JIT reduces to the plain form: fire the warmup ~cold_overhead before
    the downstream is needed; the container builds and stays Ready; the
    real invoke hits it. Just keep warmup-to-real-invoke gap < 10s
    (pause-grace), which always holds for civic_alert stage spacing.

### Paper value

A concrete, source-grounded systems optimization: OpenWhisk's default
synchronous log collection (DockerToActivation) blocks container reuse
for ~2s after every activation; switching to LogDriver (offloading log
handling) cuts container-reuse latency from ~2s to ~13ms and removes a
hidden bottleneck for JIT prewarming and high-concurrency throughput.

---

## 8.11 P3.E entry prewarm result + value-boundary insight (2026-06-06)

scripts/entry_prewarm.py: standalone entry prewarm, oracle predictor,
isolated validation (no run_workflow/JIT changes). Reuses JitScheduler
for timed warmup firing. Simple v1: one warmup per predicted arrival,
no 10s Ready-window reuse modeling (TODO).

Validation (first 200 arrivals of the 60-min 2x schedule, window 3s,
lead 2s, redeploy entry action before each run for a cold pool):

| mode | arrivals | warmups | cold rate | mean ow_wait |
|---|---|---|---|---|
| no-prewarm | 200 | 0 | 5.0% | 81 ms |
| oracle | 200 | 200 | 0.5% | 17.6 ms |
| initial burst (first 20) | — | — | 40% -> 5% | — |

### Key insight: entry prewarm's value is at bursts / sparse arrivals, NOT dense steady state

Baseline was only 5% cold (not ~100%) because this arrival slice is dense
(mean gap 0.686s) and, under LogDriver, a container stays Ready ~10s
(pause-grace). So after the first wave, containers are naturally reused
and warm without any prewarm. Entry prewarm's marginal value in dense
steady state is therefore small.

Where entry prewarm DOES matter:
- The initial burst / cold-start (container pool empty): 40% -> 5%.
- Sparse arrivals (gap > 10s Ready window): containers expire between
  requests, every request cold -> prewarm prevents it.
- Burst edges after quiet periods in the real Azure trace (periodic).

So entry prewarm covers the entry stage's cold starts at burst edges and
sparse/cold-start conditions; in dense steady state the LogDriver Ready
window already keeps entry warm via natural reuse.

### Follow-ups (logged)
- 10s Ready-window reuse modeling: v1 fires 1 warmup per arrival (200:200,
  costly). Modeling reuse would cut warmups sharply in dense arrivals
  (a few reused containers suffice). OWNER asked to remember this.
- First-window lead time: 2s too short for an initial multi-container
  burst (window 0 missed); use 2.5-3s for the cold first window.
- Advanced predictor (LSTM / online selector) to replace oracle.

### Cold-start elimination chain now complete (component level)
- entry (detect): entry prewarm (P3.E) — covers burst/sparse cold starts
- first hop + downstream: plain JIT under LogDriver (P3.C-final) — 0% cold
Both validated in isolation. Real-trace value (especially entry prewarm
at burst edges) to be confirmed in P3.G end-to-end replay.

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

## 11. P3.G End-to-End Dynamic System — Final Alignment (2026-06-08)

This section locks the end-to-end design for P3.G. It BUILDS ON (does not
replace) the rolling dynamic design in Section 5, the JIT-coverage vs
dynamic-upgrade tension in Section 8.6, and the online-mode planner in
PATH3_PLANNER_DESIGN.md Section 3. Where this section and earlier drafts
differ, THIS section is authoritative.

### 11.1 Context update under LogDriver

The permanent switch to LogDriverLogStoreProvider (Section 8.10) changed
what "cold" means here:
- The ~2s container-reuse SETTLE (synchronous log collection) is GONE;
  warm containers are reusable ~13ms after RunCompleted.
- The cold-START overhead (a brand-new container from nothing, ~2s) is
  UNCHANGED — LogDriver does not touch container creation.
- Validated: with plain JIT under LogDriver, all downstream stages (incl.
  the first hop) reach 0% cold (Section 8.10/8.11).

Consequence for risk and plans: warm execution-time distributions are
unaffected by LogDriver (it changes reuse latency, not compute time), so
the offline plans do NOT need re-fitting. The only quantity that drops is
the entry natural cold rate (denser steady-state reuse), which only makes
existing plans safer. => Decision A: reuse offline plans as the dynamic
STARTING point; do not re-plan from scratch.

The Section 8.6 tension ("execution time is a dual-purpose budget:
latency + cold-hiding") STILL HOLDS — it is about cold-START overhead,
which LogDriver did not remove. It is mitigated, not eliminated, by
cross-level speculative warmup (cumulative lead time).

### 11.2 System boundary: control plane vs data plane

The replay client is EXTERNAL to our system: a workload generator that
stands in for "users" submitting requests. It is NOT a contribution. Our
system is the CONTROL PLANE (planner + prewarm + JIT orchestration)
layered on the standard OpenWhisk DATA PLANE via its REST API.

```
═══ EXTERNAL (simulated users — NOT our work) ═══
   Replay Client: emit requests per Azure schedule, tag SLO class
        │ workflow request
        ▼
══════════════ OUR SYSTEM (control plane) ══════════════
   ┌────────────────────────┐  runtime latency feedback
   │ Orchestrator + JIT      │──(completed-stage actuals)──┐
   │  route tier per plan     │                            ▼
   │  JIT-prewarm downstream   │              ┌──────────────────────┐
   │  execute dynamic reroute  │◀──push/update plan──│ Online Planner  │
   └──────────┬─────────────┘                │  holds per-class plan │
              │                              │  monitors slack       │
   ┌──────────┴──────────┐                   │  online conditional   │
   │ Prewarm Controller   │                   │    risk → UP-upgrade  │
   │  entry prewarm        │                   └──────────────────────┘
   └──────────┬──────────┘
              │ invoke (REST)
              ▼
═══ OpenWhisk DATA PLANE (standard, existing) ═══
   controller → invoker → containers (45 variants: 5 stages × 9 tiers)
```

"Online" does NOT mean "k8s pod". v1 is process-level (pod-ification was
considered and deferred 2026-06-08: too much engineering for now; the
control/data-plane split is logical, the architecture figure is what
matters). The Online Planner is a resident process/thread, not a pod.

### 11.3 Dynamic adjustment closed loop (final)

Reuses Section 5's rolling design and Section 8.6 decision 4:
1. On each stage completion, record its ACTUAL finish time (relative to
   workflow start).
2. Before scheduling the next stage(s), compute the online conditional
   risk: completed nodes = measured constants, remaining nodes = current-
   tier distributions, `conditional_risk = P(remaining path > SLO −
   t_elapsed)`. Math + generalized DAG aggregation: PLANNER Section 9.
3. If `conditional_risk > class target`, upgrade pending (not-yet-started)
   stages by marginal efficiency until back under target.
4. Cold accounting (Decision a): an upgrade candidate's risk INCLUDES the
   target tier's cold-START overhead when that variant is not warm, so an
   upgrade is taken only when remaining slack absorbs (cold + larger-tier
   execution) AND it is still safer than not upgrading. No backup
   pre-warming (option b rejected).

### 11.4 Final decisions locked (2026-06-08)

| Decision | Choice | Rationale / ref |
|----------|--------|-----------------|
| Initial plan | Reuse offline `multi_slo_planner` output, no re-fit | Decision A; LogDriver doesn't change warm exec time |
| Dynamic | Direct-to-dynamic (v1 is already dynamic, not static-first) | owner 2026-06-08 |
| Adjust direction | UP-only in v1 (downgrade deferred) | matches robustness intent; no oscillation; cost vs Always-Warm expected to still win (confirm in T6) |
| Eval cadence | Every stage completion | analytic aggregation is ~µs |
| Cold on upgrade | Decision a: count target-tier cold in conditional risk; no backup prewarm | owner 2026-06-08; Section 8.6 decision 4 |
| Risk scope | Online conditional risk only (no correlation / non-stationary in v1) | owner 2026-06-08 |
| Replay client | External workload generator, not a contribution | owner 2026-06-08 |

### 11.5 Build roadmap T1–T6 (LogDriver + dynamic-first)

Supersedes the Section 9 build order for the current context (most of
P3.A–P3.F are DONE; the LogDriver switch + direct-to-dynamic decision
re-shape the remaining work). Each task has an independent verify gate.

| # | Task | Depends | Verify |
|---|------|---------|--------|
| T1 | Generalized DAG aggregation (topological order, reuse Fenton/Clark; civic_alert as special case) | — | civic_alert output matches current `aggregate_civic_alert` bit-for-bit |
| T2 | Online conditional risk (completed=const, remaining=dist, include target-tier cold) | T1 | MC validation of conditional distribution; boundary cases |
| T3 | Online Planner (hold initial plan + slack monitor + UP-only marginal-efficiency upgrade API) | T2 + existing greedy | given execution state → correct upgrade decision |
| T4 | Orchestrator integration (per-stage hook: record actual → call planner → reroute pending tiers) | T3 + existing JIT | injected slowdown triggers upgrade |
| T5 | End-to-end replay driver (external replay tags SLO class + prewarm parallel + metrics) | T4 + existing entry_prewarm | full-trace run produces metrics |
| T6 | Baseline comparison (Scale-To-Zero / Always-Warm) | T5 | comparison table (baseline defs TBD) |

---

## 12. P3.G Measurement Issue + Tier Re-alignment Plan (2026-06-09)

T5.2b end-to-end showed measured warm execution did NOT match the offline spline
(actual 1.1-1.4x slower, worst at high tiers, premium SLO-satisfaction only 0.61). A
data-driven investigation (several hypotheses ruled out: cold start, wiring, our
concurrency, node heterogeneity) found TWO independent causes — neither a flaw in the
planning method.

### 12.1 Root cause A — worker/cpu-limit mismatch → CFS throttle
The action ProcessPool uses workers = ceil(cpu), but the k8s pod has a cpu LIMIT (= the
tier's cores). When worker processes > cpu quota the whole cgroup is CFS-throttled,
cores go intermittently idle, and the hardware never grants turbo → the tier's cores
aren't used at speed. Of 9 tiers only 1280/2560/3840 (integer cores 1/2/3, where
workers==cores) avoid throttle. Yet 4/5 stage tier-pairs still showed +18~30% speedup,
so tier sizing is fundamentally sound — just degraded by throttle.

### 12.2 Root cause B — all-core turbo suppression (shared node)
dell (Xeon 8352M, 64 physical / 128 logical cores, base 2.30GHz) is SHARED. A co-tenant
('ye') ran 9 training processes throughout. They use only ~7% of cores, but by raising
the active-core count they pushed the CPU from single-core turbo (~3.4GHz) to all-core
turbo (~2.3GHz), uniformly slowing every pod. Evidence: now-isolated (mi=1) cpu_process
≈ now-loaded (mi=32) ≈ 2x the 5/28 sweep — NOT our concurrency, but the node turbo
state. This scales absolute latency uniformly but preserves relative tier differences.

### 12.3 Re-aligned tier→CPU→worker design
Keep the cpu-scaling formula (200m×ceil(mem/256)) — it already yields clean values:
- sub-1 core (throttles, but differentiated by quota ratio; fits "low tier = slow"):
  512=0.4, 768=0.6, 1024=0.8
- integer cores (workers=cores → no throttle → boost → near-linear speedup):
  1280=1, 2560=2, 3840=3
Change ONLY the worker rule: `workers = max(1, round(cpu))`. Non-integer >1 tiers
(1536/2048/3072) stay deployed for completeness but are NOT used for planning (they
throttle: round(1.6)=2 > 1.6).

### 12.4 Re-measurement plan (decided 2026-06-09)
1. workers = max(1, round(cpu)) in workflow_action.py
2. bump keepalive to 600s (sweep warm samples need containers to survive)
3. redeploy all 9 tiers with the new action
4. full re-sweep (per-stage + e2e, cold + warm) → reports/sweep_realign/
5. re-fit the warm resource model from the new measurements
6. recompute JIT lead times / plans from the new model
7. switch to a SPARSER, more serverless-like trace (see 12.5)

Caveat: dell is busy now (turbo suppressed), so absolute times are biased. We record
now and RE-COMPARE when the node is idle. Idle==busy → fine; else revisit.

### 12.5 Trace density finding — current trace is too dense
cand2-2x: 4011 reqs/60min, mean 1.11/s, inter-arrival p50=0.57s, p99=5.21s. Under 10s
keepalive only 0.1% of gaps exceed keepalive → almost everything is warm. Real
serverless is intermittent/sparse (why it beats a dedicated server on cost). A dense
trace hides the cold-start problem the system targets. Plan: adopt a sparser/bursty
trace so cold start becomes a real cost and JIT/prewarm value shows.

---

## 13. Reservation invoker + warm-pool closed loop + realized SLO (2026-06)

### 13.1 Cross-workflow steal root cause + reservation invoker
The realized SLO fat tail was traced (isolated-vs-concurrent A/B) to cross-workflow
warm-pool **stealing**: a freshly warmed container can be grabbed by another workflow
during the warmup→dispatch gap, sending the owner cold (isolated downstream cold 0%
vs concurrent 6.25%). Fix: a custom-compiled invoker (`owlocal/invoker:res2`, from
apache/openwhisk) whose ContainerPool reserves a freshly warmed container for its
owning request, keyed on `__ow_reservation_key` read from message content
(content-passthrough — no Controller/Message change); lease TTL via env
`RESERVATION_TTL_MS` (default 5000). Result: **downstream cold 10.4% → 0%**.

### 13.2 /poolState read interface
The same invoker exposes `GET /poolState` (open route on the invoker HTTP server),
returning per-action@tier `{free, busy, warming, oldestFreeIdleMs, memoryMB}`. This
is the "eyes" for closed-loop pool control (query real state → top up the deficit)
instead of firing warmup invocations blind.

### 13.3 Critical-prefix warm pool — A/B/C verdict
A demand-sized always-warm pool for the critical prefix (detect+estimate), three
arms (OFF / blind-demand / closed-loop) over 120 workflows: blind demand already
reaches premium 91.67%; the closed-loop did **not** beat it (also 91.67%, higher
cost) because the blind-demand failure it was built to fix did not reproduce.
Verdict: the **pool lever is maxed at ~91.67%**; the residual gap to 95% was a
**model** (over-optimistic planner) problem, closed by Risk-Model §11
(sigma+rho+contention → realized premium 97.92%). Closed-loop kept behind a flag,
not default.

### 13.4 Effective capacity is CPU-bound (~160 GB), not 200 GB
Action containers are K8s pods (KubernetesContainerFactory). cpu-scaling sets pod
CPU = 200m per 256MB = **800 millicpu/GB**; the node is 128 cores / 251 GB =
**510 millicpu/GB**. Since 800 > 510, CPU saturates first: at 128 cores only ~160 GB
of containers are placed, under the 200 GB userMemory limit (~90 GB RAM left idle).
Transient "pending" pods at peak are `Insufficient cpu`, not memory. Implication:
size pools/plans against ~160 GB effective; pending is a CPU-headroom signal.

### 13.5 Next: online evaluation protocol (design locked, not built)
Prediction handles only **entry arrivals** (feeds entry-prewarm timing + pool K_t);
downstream JIT is plan/elapsed-driven, no forecast needed. Three layers
**static / forecaster / oracle** (F−static = prediction value, oracle−F =
imperfection cost); causal forecaster = CoV triage (forecast only regular functions,
CoV ≤ 2) + seasonal + lag GBM quantile regression for K_t + histogram method for
next-arrival. Trace: pick regular functions from the two-week Azure 2021 trace
(aggregate is unpredictable, held-out R² negative). Cost unified in USD (AWS Lambda
$1.6667e-5/GB-s + $2e-7/req). Three eval workloads (ML inference / video transcode /
ETL) share one trace; each needs its own resource sweep first. Optional drift-
adaptive layer per §11.5.

## Changelog

- 2026-06-25: Model alignment to realized concurrency (sigma+rho+contention,
  Risk-Model §11) closed the planner's over-optimism — corrected re-plan realized
  premium 97.92% (≥95%). Plus reservation invoker (steal fix, downstream cold
  10.4%→0%), `/poolState` closed-loop interface, prefix-pool A/B/C (pool lever maxed
  ~91.67%), CPU-bound effective capacity ~160 GB. See Section 13.
- 2026-06-09: T5.2b measurement investigation (Section 12). Warm-exec gap traced to
  (A) worker>cpu-limit CFS throttle and (B) shared-node all-core turbo suppression
  (co-tenant), NOT a planning flaw (tier sizing 4/5 sound). Re-alignment: workers=
  round(cpu), 6 aligned tiers (sub-1 + integer cores), keepalive 600s, full re-sweep +
  re-fit + recompute JIT, switch to sparser trace. dell busy now → record, re-compare idle.
- 2026-06-08: P3.G final alignment. Locked direct-to-dynamic end-to-end
  design: control/data-plane split with replay as EXTERNAL workload,
  online conditional risk (generalized DAG aggregation), per-stage UP-only
  dynamic upgrade with Decision-a cold accounting, offline plans reused as
  starting point. Build roadmap T1–T6. Pod-ification deferred. See Sec 11.
- 2026-05-30: Initial document. Captures the full execution architecture
  aligned across several discussion rounds: unified stage-start trigger,
  dynamic-early/warmup-late separation, last-upstream JIT scheduling
  (race fix), priority-queue scheduler with upsert, independent entry
  prewarm component, accepted race condition.
