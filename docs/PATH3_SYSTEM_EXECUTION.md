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
revisit by modifying the OpenWhisk invoker to add container reservation
(future work / limitation discussion).

This limitation is logged; we will intrude into the invoker code later
only if measured impact warrants it.

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
