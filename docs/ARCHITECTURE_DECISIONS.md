# Architecture Decisions and Aligned Thinking

This document captures the architectural decisions and reasoning aligned through
discussion between the project owner and the assistant. It is the single source
of truth for "what we agreed to do and why," superseding earlier informal
discussions.

Companion documents:
- `RELATED_WORK_AND_INNOVATION.md` - paper-by-paper analysis and innovation framing
- `STAGE_PIPELINE_AND_SERVER_MIGRATION.md` - operational/runtime notes
- `CODE_STRUCTURE.md` - code layout

---

## 1. Overall System Architecture

```
Slow layer (per-experiment fixed):
  - Global keepalive (OpenWhisk idleTimeout): 20s
  - Per-stage memory tier: 1280 MB (first round)
  - SLO classes (multi-SLO design)

Medium layer (per-window, decision frequency 3s):
  - Entry stage prewarm count
    Predicts arrivals in next horizon (15s)
    Subtracts surviving idle containers (exponential decay model)
    Prewarms the deficit
  - Client-side SLO-aware priority queue (slack-based)

Fast layer (per-request, continuous time):
  - JIT downstream prewarming
    When upstream stage starts, compute downstream prewarm timing
    Trigger warmup such that downstream container is ready just as
    upstream completes
    No window quantization

Ultra-fast layer (per-stage-completion, reactive):
  - Dynamic plan adjustment (added 2026-05)
    After each stage completes, observe actual completion time
    Recompute remaining workflow risk via closed-form model
    If risk exceeds budget, upgrade downstream stage resources
    or trigger extra prewarms
```

### Why this split

| Decision | Mechanism | Rate of change |
|---|---|---|
| Memory/CPU tier | Static config (first round) | Hours/days |
| Keepalive | Static OpenWhisk invoker config | Hours/days |
| SLO classes | Static experimental design | Hours/days |
| Entry prewarm count | Window-based prediction | Seconds (~3s) |
| Priority queue ordering | Slack-based per request | Real-time |
| JIT downstream prewarm | Per-request timing computation | Continuous |

---

## 2. OpenWhisk Container Lifecycle (Verified)

State machine (verified against `ContainerProxy.scala`):

```
Starting -> Started -> Running -> Ready -> Pausing -> Paused -> Removing
                          ^                              |
                          | resume (~ms)                 |
                          +-- (new request) -------------+
```

Key timing parameters:
- `pauseGrace`: 50 ms (Ready -> Paused transition; very short)
- `idleTimeout`: 20s (Paused -> Removing; this is the "keepalive" we tune)
- Total post-execution life: ~50 ms warm + ~20s paused = ~20.05s

**Important facts about Paused state:**
- Implemented via Linux cgroup freezer (`docker pause`)
- CPU usage = 0 (processes frozen, cannot observe their own pause)
- Memory still consumed (kernel keeps process state in RAM)
- Unpause is millisecond-scale
- Kubernetes has NO native pause concept; OpenWhisk does this at the
  container layer below K8s, K8s sees pod as Running throughout
- Paused containers consume memory in the cost model (correct as is)

**Source:** [ContainerProxy.scala](https://github.com/apache/openwhisk/blob/master/core/invoker/src/main/scala/org/apache/openwhisk/core/containerpool/ContainerProxy.scala)

---

## 3. JIT Prewarming Mechanism

### Design

For each in-flight workflow, when an upstream stage begins execution:

```
T_upstream_start: upstream begins
T_jit_trigger = T_upstream_start  (fire warmup immediately)
T_downstream_ready = T_jit_trigger + cold_overhead
T_downstream_needed = T_upstream_start + upstream_duration
JIT works iff T_downstream_ready <= T_downstream_needed
            iff cold_overhead <= upstream_duration
```

For civic_alert at 1280 MB:
- upstream_duration (warm): ~3.5s (detect_object)
- cold_overhead: ~2s
- Slack: 1.5s -> JIT works comfortably

### Implementation

Use a simple no-op warmup action invocation. OpenWhisk's scheduler:
1. If a free Warm/Paused container exists, runs warmup there (no new container)
2. Otherwise creates a new container (cold start, ~2s)
3. Either way, after warmup the container is in Paused state, ready for the
   real request

The "active sleep" approach (warmup action sleeps for hold duration) was
considered but rejected: during sleep the container is Busy, so it cannot
accept the real request anyway, leading to either queueing or new cold start.

### Race Condition Analysis

Real race exists when upstream execution time has variance σ_D:
- If upstream finishes faster than predicted, downstream container enters
  Paused state for σ_D milliseconds before the real request arrives
- During this window, another workflow's JIT can absorb the container
- Probability of theft per request ≈ σ_D × concurrent_arrival_rate
- For civic_alert: σ_D ≈ 150ms (warm), λ ≈ 2.2/s -> ~33% theft probability

**Consequence:** stolen containers cause that workflow to see ~cold_overhead
extra latency, BUT OpenWhisk creates a new container in response, so total
warm container count remains correct.

**Decision: do NOT implement race defense in v1.** Characterize the effect
quantitatively in evaluation. Frame as "race causes ~σ_D / D fraction of
requests to experience cold_overhead extra latency; container reservation
is future work."

---

## 4. Entry Prewarming Mechanism

### Window parameters (independent, not tied together)

| Parameter | Symbol | Value | Purpose |
|---|---|---|---|
| Update frequency | W | 3s | How often we re-predict and re-prewarm |
| Prediction horizon | H | 15s | How far into the future to predict |
| Keepalive | K | 20s | Container natural lifetime after idle |
| Cold overhead | C | ~2s | Lead time needed to ready a container |

Required ordering: W < C < H <= K

### Algorithm

At start of window n:
1. Predict total arrivals in [T_n + W, T_n + W + H]
2. Estimate surviving warm containers at T_n + W (using decay model below)
3. Fire `max(0, predicted - survivors)` warmup invocations now

### Container survival model (exponential decay)

Per-container state: residual idle lifetime tau, initially K, reset on each
reuse, decremented by elapsed wall time.

Aggregate model for planning:
```
survivors(t) = sum over recent prewarms m of:
                 prewarm_count(m) * exp(-(t - T_m) / tau_eff)
```
where tau_eff is fitted from trace data (accounts for both K and the
typical reuse pattern).

**v1 implementation:** start with two layers:
- Base layer: direct monitoring of current idle pool size (ground truth)
- Model layer: exponential decay prediction (for planning lookahead)
Use base layer for current state, model layer for projecting forward.

### Container reuse rate (justification for decay modeling)

For execution time D and keepalive K (no reuse):
```
reuse_rate ≈ 1 + K / D
```
For civic_alert detect_object: D=3.5s, K=20s -> reuse rate ≈ 6.7

One prewarm can serve ~6.7 requests over its lifecycle. Ignoring this leads
to severe over-prewarming. The decay model lets the planner exploit reuse.

---

## 5. Multi-SLO Architecture (Path Y)

### Decision: pursue multi-SLO directly, not as a v2 extension

Rationale: this is the strongest differentiation from ORION/Jolteon/AQUATOPE,
all of which assume a single global SLO. Multi-tenant DAG workflows with
heterogeneous SLOs are realistic and underserved by prior work.

### SLO class design (v1)

- Two SLO classes: premium and free
- Mixing ratios for ablation: 1:9, 3:7, 5:5
- Random assignment at the workload generator (replay-time labeling)
- SLO targets to be calibrated based on observed performance (see Section 7)

### Client-side priority queue

Architecture:
- Workload generator maintains a priority queue indexed by (slack, class)
- Slack = remaining_SLO_budget - estimated_remaining_workflow_time
- Higher priority = smaller slack
- Premium class gets prioritized when slack is tight
- Requests held in queue until either:
  - System has capacity (current in-flight < limit), OR
  - Slack < threshold (must dispatch even at risk of overload)

This is inspired by GrandSLAm but applied client-side (no OpenWhisk changes)
and combined with resource planning.

### How SLO classes affect resource planning

- Premium class: larger memory tier (faster execution) + aggressive prewarm
- Free class: smaller memory tier + lazy prewarm (accept some cold starts)
- Joint optimization across classes: minimize total cost subject to
  per-class SLO satisfaction constraints

---

## 6. Cost Model

### Decision: use Lambda-style billing as primary metric

```
cost = memory_GB * execution_seconds * price_per_GB_second
```

Idle (Paused) memory and CPU are NOT counted.

### Why

- Standard FaaS billing model (AWS Lambda, Google Cloud Functions, Azure)
- Comparable to commercial baselines
- Reflects user-facing cost

### Bug to fix

Current `cost_summary.csv` reports `idle_vcpu_seconds_total > 0`. This is
incorrect: during Paused state, CPU is frozen via cgroup, actual CPU usage
is 0. Should be fixed to:
- `idle_vcpu_seconds_total = 0` (always)
- `idle_gb_seconds_total` retained (memory IS still allocated in Paused)
  but reported as auxiliary metric, not primary cost

### Auxiliary metric: provider-cost (for ablation)

For provider-perspective analysis:
```
provider_cost = memory_GB * (execution + idle) * price_memory
              + cpu_cores * execution_seconds * price_cpu
```
Shown as secondary metric in evaluation.

---

## 7. Resource-Latency Model (T_action vs CPU)

### Decision: use Amdahl-style structured model on mock actions

Formula:
```
T_action(cpu) = S / min(cpu, 1) + P / min(cpu, W_eff) + C

where:
  S       = serial CPU work (core-ms)
  P       = parallel CPU work (core-ms)
  W_eff   = floor(cpu) or min(floor(cpu), max_workers)
  C       = IO + memory + overhead (constant across CPU tiers)
```

Validated against sweep data (768/1280/2048/2560 MB, 10 cycles each).
Key sweep observations:
- IO time constant across tiers (0.6 to 2.0 vCPU) - validates C structure
- Serial time roughly halves from 0.6 -> 1.0 vCPU then flattens (1-core cap)
- Parallel time has clear breakpoint at workers transition (1 -> 2 workers
  at ~1.0 -> 1.6 vCPU)

### For real (non-mock) functions

Fallback formula:
```
T_action(cpu) = a * cpu^(-alpha) + c     (3 parameters)
```

Paper strategy:
- Show Amdahl is more accurate when serial_fraction is known (mock actions)
- Show power-law works adequately when only T(cpu) measurements are available
- Frame as "model accuracy vs information availability tradeoff"

### Cold overhead model

`cold_overhead` is treated as **CPU-independent constant per stage** (~2s
for civic_alert at all memory tiers, verified from sweep). This is because
cold overhead is dominated by:
- OpenWhisk control-plane operations
- Container image pull / cgroup creation
- Runtime init
None of these scale with the action's allocated CPU.

---

## 8. Delay Kernel: Demoted Role

Initial belief: delay_kernel is the bridge between Stage 2 entry forecast
and Stage 5 per-stage prewarm decisions.

Revised after JIT discussion: with per-workflow JIT prewarming, downstream
prewarm timing is computed from the per-request latency model, NOT from
delay_kernel. The delay_kernel becomes an optional aggregate-demand
prediction tool, useful only as a sanity check.

**Current role:** retained as an output of Stage 3 for completeness, but
not on the critical path. May be used in offline capacity planning or as
alternative input to the planner for ablation.

---

## 9. SLO Definition (Pending)

Needs grounding in real measurements. The latest 45-min replay at 1280 MB
shows:
- all_warm E2E mean = 17.96s, max = 23.54s
- partial_cold E2E mean = 22.15s, max = 61.89s (tail!)
- all_cold E2E mean = 28.90s, max = 30.10s

Tentative SLO classes (to be validated):
- Premium (strict): 20s - requires all-warm path; cannot tolerate cold
- Free (lax): 30s - tolerates 1-2 partial cold stages

These will be refined after running paired experiments at different memory
tiers, since premium class likely needs larger memory.

---

## 10. Dynamic Plan Adjustment (Added 2026-05)

### Motivation

Static plans assume deterministic execution time. In reality:
- Network jitter can delay individual stage dispatch
- Platform contention causes queuing
- CPU throttling under load makes warm execution variable
- Preemption / resource pressure can extend execution

A workflow's plan made at entry time may become invalid mid-execution.
Without dynamic adjustment, the system fails to maintain SLO under
runtime variability.

### Design

After each stage completes, the controller:
1. Observes actual completion time vs predicted
2. Updates remaining workflow E2E estimate
3. Computes residual SLO violation risk via closed-form analytical model
4. If risk exceeds budget threshold, applies corrective actions:
   - Upgrade memory tier of downstream stages (faster execution)
   - Fire extra warmup invocations
   - Re-prioritize within the priority queue
5. If still over budget after corrections, accept the violation and log it

### Why closed-form risk, not Bayesian or MPC

Triggered per stage completion -> 5 times per workflow -> potentially
thousands per second under load. Must be microsecond-scale.

- Monte Carlo: too slow (milliseconds per evaluation)
- Bayesian state inference: overkill, adds complexity without proven
  benefit at this granularity
- MPC: powerful but complex; reserved for future work
- Closed-form: microsecond evaluation, interpretable, sufficient

If closed-form proves inadequate (e.g., poor accuracy on tail risk),
Bayesian state tracking can be added as a v2 mechanism. Initial design
uses closed-form only.

### Relationship to existing layers

This is an additional layer in the architecture, complementing (not
replacing) the existing layers:
- Slow layer: sets system parameters (unchanged)
- Medium layer: forecast-driven entry prewarming (unchanged)
- Fast layer: per-request JIT prewarming (unchanged)
- **Ultra-fast layer (new): reactive plan adjustment per stage completion**

JIT prewarming hides cold start under upstream execution; dynamic
adjustment recovers slack when execution itself deviates from prediction.
They address orthogonal sources of variability.

### Paper positioning

Strengthens differentiation from prior art:
- vs ORION/Jolteon: static, no dynamic adjustment
- vs AQUATOPE: dynamic prewarming but static resource allocation
- vs SMIless: re-plans at workflow start only, not mid-workflow
- vs GrandSLAm: reactive scheduling (re-ordering) but no resource changes
- vs Taming Cold Starts (MPC, 2024): entry-only adjustment, not DAG-internal

The paper narrative becomes "online adaptive resource control" rather
than just "joint planner with forecast".

### Implementation roadmap

Depends on path 2 (closed-form risk model) being completed first.
Then implementation has three parts:
1. Per-stage completion hook in the workflow executor
2. Closed-form risk re-computation with updated state
3. Action upgrade routing (since OpenWhisk doesn't support hot memory
   change, route to a different memory-tier action variant)

Estimated 2-3 weeks after path 2 completes.

---

## 11. Decisions Not Yet Made

- Exact SLO numbers for each class (calibrate against measurements)
- Premium/free workload mix in evaluation suite
- Whether to instrument GrandSLAm-style stage-level slack tracking, or
  only workflow-level slack at client
- Optimization method for the planner: convex (Jolteon-style chance
  constraint) vs heuristic (Pareto sweep, our current code) vs RL
- Risk budget threshold for triggering dynamic adjustment
- Whether to support more than 2 SLO classes in v1

---

## 12. P4 Findings: Cold Sample Structure (Added 2026-05-27)

### What we tried

Attempted to bucket the `cold_like` sample pool into:
- `clean_cold`: workflow has 1 cold stage AND ow_wait_ms < 200
- `cold_with_contention`: workflow has 1 cold stage AND ow_wait_ms >= 200
- `partial_cold_cascade`: workflow has >= 2 cold stages

### What we found

The `clean_cold` bucket is EMPTY. Reason: OpenWhisk's `ow_wait_ms` is not
pure queuing time; it includes container creation, image pull, and runtime
init. Even a no-contention cold start has ow_wait_ms in the range
1000-2000 ms. The 200 ms threshold was based on a wrong assumption.

The 14 real all-cold workflows (5 stages each = 70 samples) are all
classified as `partial_cold_cascade` per the workflow_cold_count rule.

### Implication

We do not need MC to estimate Scale-To-Zero and Always-Warm bounds:
- Scale-To-Zero bound = empirical distribution of the 14 all_cold
  workflows (mean=28.9s, p95=29.8s, max=30.1s)
- Always-Warm bound = empirical distribution of the 3632 all_warm
  workflows (mean=18.0s, p95=20.1s)

MC may still under- or over-estimate these because of the
independent-sampling assumption (sampling 5 stages independently
explodes the tail).

### Correct method for future MC bounds

If MC is needed (e.g., for plans not directly measured), use
WORKFLOW-LEVEL joint resampling, not stage-level independent sampling:
- Pick a real workflow at random, use its 5 stage latencies as a unit
- This preserves within-workflow correlation (which is large in practice)

### Lesson for path 2

Path 2 (closed-form risk model) requires fitting cold/warm latency
distributions per stage. When fitting cold distributions:
- Use samples from full-cascade workflows (workflow_cold_count == 5)
  to capture clean cold behavior, OR
- Use the full cold pool with explicit correlation modeling
- Do NOT fit a single distribution to mixed clean/contended samples
  and then sample independently — this is what MC was effectively doing
  in P3/P4 and it produces tail explosion

P4 deliverables:
- `scripts/run_p4_rebucketed_bounds.py`: bucketing wrapper (kept for
  reference even though clean_cold turned out empty)
- `reports/stage3_cold_bucketed/`: v2 sample pool, MC bounds attempt,
  comparison report

---

## Changelog

- 2026-05-26: Initial document created, aligning all decisions made in
  pre-implementation discussion.
- 2026-05-27: Added Section 10 on dynamic plan adjustment. P3 first-pass
  validation completed (see reports/stage4_p3_first_pass/); identified
  Stage 3 cold sample bucketing as next priority before path 2.
- 2026-05-27 (later): Added Section 12 documenting P4 findings. The
  `clean_cold` bucket is empty due to OpenWhisk's ow_wait_ms including
  container creation. Decision: skip P4b, use empirical bounds for
  Scale-To-Zero / Always-Warm, proceed to path 2 with proper joint
  sampling or parametric fitting.
