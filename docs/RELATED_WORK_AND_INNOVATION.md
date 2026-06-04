# Related Work Analysis and Innovation Planning

This document records our analysis of 6 key papers on serverless DAG execution
planning, their relevance to our project, and the innovation directions we
identified through discussion.

---

## Paper-by-Paper Analysis

### 1. GrandSLAm (EuroSys 2019)

**How it finds a plan:**
Reactive, per-request scheduling. At runtime, estimates each request's
time-to-completion through the microservice pipeline. Uses the "slack"
(SLO minus elapsed time) to decide priority: tight-slack requests get
dispatched first; loose-slack requests are batched or reordered.

**Relevance to our project:**
- The slack concept could theoretically apply to DAG stage dispatching.
- However, our OpenWhisk setup does not support request queuing or batching
  at the stage level — each invocation is one container, one request.
- Co-locating two stage invocations in one pod (serial execution) was
  considered but rejected: OpenWhisk's one-invocation-per-container model
  makes this impractical without modifying the invoker. The complexity is
  high and the benefit uncertain given our DAG structure.

**Verdict:** Not directly applicable. The slack-based priority idea is noted
but deferred — our DAG dispatching is currently parallel, not queue-based.

---

### 2. ORION (OSDI 2022)

**How it finds a plan:**
Offline profiling → empirical CDF per stage → CDF propagation through DAG
using CONV (series) and MAX (parallel) → greedy right-sizing along the
critical path → prewarming with DAG look-ahead timing.

Three optimizations:
1. Right-sizing: greedy memory allocation to critical-path stages.
2. Bundling: co-locate parallel invocations in one VM to reduce skew.
3. Right-prewarming: warm downstream VMs at the p50 completion time of
   their upstream stage.

**Relevance to our project:**
- The CDF CONV+MAX propagation is the standard approach for DAG latency
  estimation. Our project currently uses Monte Carlo for this; we should
  also implement the closed-form version.
- The JIT prewarming idea (warm downstream based on upstream completion
  estimate) is directly what we want. Our delay_kernel already captures the
  upstream-to-downstream propagation delay distribution.
- Bundling is not feasible in our OpenWhisk setup.

**What we adopt:** JIT prewarming using delay_kernel timing. CDF propagation
as a fast analytical risk model alternative to Monte Carlo.

**What we improve over ORION:**
- ORION is fully static (one-shot profiling). We add time-varying forecast.
- ORION does not optimize keepalive. We add keepalive as a decision variable.
- ORION treats cold starts simplistically (root=cold, rest=warm). We model
  cold probability per stage per window.

---

### 3. Jolteon (NSDI 2024)

**How it finds a plan:**
Parametric stochastic model + chance-constrained convex optimization.

1. Whitebox model: analytical T(memory) per stage.
2. Blackbox noise: stochastic component for platform variability.
3. DAG aggregation: CONV+MAX on parametric distributions.
4. Optimization: minimize cost subject to P(E2E > SLO) ≤ ε.
5. Solved via sampling approximation + convex solver.

Only optimizes memory/CPU per stage. No prewarm, no keepalive.

**Relevance to our project:**
- The mathematical formulation (chance-constrained optimization) is exactly
  the style we want for our planner. Currently our Stage 5 uses heuristic
  Pareto search — formalizing it as an optimization problem is better.
- The whitebox model concept aligns with our Amdahl-based CPU-time model,
  but see the caveat below.

**Caveat on Amdahl model vs real functions (our discussion):**
- Our mock actions report per-component timing (serial, parallel, IO,
  memory), so Amdahl decomposition is possible and accurate.
- Real serverless functions only report total action_duration_ms. For those,
  a Jolteon-style opaque T(memory) fit is more practical.
- Paper strategy: validate Amdahl on mock actions (show it outperforms
  opaque fit with fewer samples), then note that for real functions the
  model degrades gracefully to Jolteon-style fitting, but can exploit
  profiling hints (like serial_fraction) if available.

**What we adopt:** Chance-constrained optimization formulation.

**What we improve over Jolteon:**
- Jolteon is static (one-shot optimization for steady-state). We add
  time-varying workload forecasting and re-solve per window.
- Jolteon has no prewarm/keepalive dimensions. We jointly optimize
  (memory, prewarm, keepalive).
- Jolteon does not model cold starts. We explicitly model cold probability
  and its impact on latency.

---

### 4. IceBreaker (ASPLOS 2022)

**How it finds a plan:**
Per-function utility scoring with heterogeneous server placement.

1. Fourier-based arrival forecasting per function.
2. Utility = arrival_probability × execution_benefit / keepalive_cost.
3. High utility → warm on high-end node; medium → warm on low-end node;
   low → evict.

**Relevance to our project:**
- Fourier-based arrival forecasting is interesting and already implemented
  in our Stage 2 (fip-fourier method).
- Heterogeneous server placement does not apply — we have a homogeneous
  OpenWhisk cluster.
- The utility scoring concept could inform our keepalive decision, but
  IceBreaker is per-function (no DAG awareness).

**Verdict:** Fourier forecasting already incorporated. Rest not applicable.

---

### 5. AQUATOPE (ASPLOS 2023)

**How it finds a plan:**
Two separate subsystems:
1. Prewarming: Hybrid Bayesian Neural Network predicts next arrival +
   uncertainty → conservative prewarming when uncertain.
2. Resource allocation: Bayesian Optimization (BO) searches near-optimal
   memory config per stage.

**Relevance to our project:**
- BO is a black-box optimization method: tries real configs, builds a
  Gaussian Process surrogate, picks next config to try. Very general but
  slow (needs real invocations to evaluate each candidate).
- Our project already has analytical/physical models for latency-vs-resource.
  We do not need BO — we can compute the objective directly.
- The uncertainty-aware prewarming concept is interesting: when the forecast
  is uncertain, prewarm more. This could be incorporated into our risk
  model (wider confidence interval → higher prewarm target).

**Verdict:** BO is overkill for our setting. Uncertainty-aware prewarming is
a useful concept to incorporate.

---

### 6. SMIless (SC 2024) — VERIFIED FROM SOURCE (github blinkbear/smiless-ad)

Earlier notes about SMIless were based on the abstract and were WRONG on
two points (keepalive=0, knapsack). The following is corrected after
reading the actual optimizer source (optimizer-engine/optimizer/
smiless.py 593 lines + path_search.py 509 lines).

**How it actually finds a plan:**
A* search (priority queue with admissible heuristic + SLA pruning) over
the DAG, deciding PER NODE which device to use.

1. Nodes are ordered topologically (current_index 0..N).
2. Each node has only TWO candidates: device in {cpu, cuda}. NOT memory
   tiers. So the search space is 2^N, not tiers^N.
3. A* state = (prefix of decided nodes, current_index). Priority
   f = g + h:
   - g = actual accumulated cost of decided nodes
   - h = calc_heuristic_cost = remaining nodes' minimum execution time
     lower bound (gpu_inference_time sum); if current_time + remaining
     > SLA, prune (return inf)
4. Constraint: completion_time <= SLA. Objective: min total cost.
5. The repo also implements bfs / dfs / bfs_with_top_k(k=2) as variants.
   bfs_with_top_k IS essentially beam search (they call it aug_smiless),
   but the MAIN method is A*.

**Prewarming (the "adaptive pre-warming window"):**
```
prewarm_window = max(0, IT - execution_time)
```
where IT = inter-arrival time (online predicted). If the predicted gap
between requests exceeds execution time, prewarm that much ahead. This is
WINDOW-LEVEL / statistical prewarming based on arrival rate, NOT
per-request continuous-time JIT.

**Keepalive (CORRECTED — SMIless DOES use keepalive, not 0):**
SMIless chooses keepalive OR prewarm per function:
```
if prewarm_window == 0:  keep_alive_time = calc_keep_alive_time(entry)*unit
else:                    keep_alive_time = -1   # rely on prewarm instead
```
- Frequent arrivals (execution_time >= interval, prewarm_window=0) ->
  keep the container ALIVE.
- Sparse arrivals (prewarm_window > 0) -> use prewarm, no keepalive.
Keepalive duration is computed from the entry node's arrival statistics.

**Relevance / what we genuinely improve over SMIless:**
- Decision space: SMIless picks CPU-vs-GPU (2 choices/node); we pick
  among 9 memory tiers/stage (9^N, much larger and harder).
- Search: SMIless uses A* (admissible heuristic = remaining min exec
  time -> provably optimal). We use beam (near-optimal, empirically
  matches brute force). We can RE-EXPRESS our search as A* using our
  risk-aware marginal efficiency as a MORE INFORMED heuristic than their
  execution-time lower bound (see Innovation notes).
- Prewarming granularity: SMIless prewarms at the window/statistical
  level (IT - execution_time); ours is per-request continuous-time JIT
  triggered off the actual upstream completion, plus
  warmup-synchronized dispatch. Finer-grained.
- Container reuse settle delay: SMIless's prewarm model ASSUMES a warmed
  container is immediately reusable. It does NOT model the time from
  "warmup completes" to "container is reusable by the real invoke"
  (we measured ~2.3s on OpenWhisk). Their window-level prewarming is
  insensitive to this; our per-request JIT is not, so we discovered and
  must model it. This is a unique finding (pending probe confirmation).
- Keepalive: SMIless decides keepalive-vs-prewarm from arrival stats
  (single SLA). We treat keepalive as a static/per-stage config and add
  multi-SLO classes + dynamic plan adjustment.
- SMIless: single SLA, CPU/GPU heterogeneity. Ours: multi-SLO classes,
  memory-tier heterogeneity, time-varying arrival forecasting, dynamic
  mid-workflow replanning.

**Implication for our search method (beam -> A*):**
Re-expressing our planner as A* (g = decided-stage cost, h = lower bound
on remaining-stage cost + SLO feasibility pruning) gives provable
optimality and a stronger narrative (same "A* with admissible heuristic"
framing as SMIless), on a HARDER decision space (9 tiers vs 2 devices).
Our marginal-efficiency theory is NOT discarded: it becomes (a) the
risk-aware search-ordering heuristic inside A*, and (b) the fast
near-optimal fallback for online dynamic replanning when A* is too slow.
Keep beam as a comparison baseline (as SMIless keeps bfs/dfs/beam).

---

## Cross-Paper Comparison

| Capability | ORION | Jolteon | AQUATOPE | SMIless | IceBreaker | GrandSLAm | Ours |
|---|---|---|---|---|---|---|---|
| DAG-aware latency | CDF conv | parametric | per-stage | A* over DAG | none | per-stage | MC + analytical (FW+Clark) |
| Resource decision | mem greedy | mem convex | mem BO | CPU/GPU (2/node) A* | none | none | 9 mem tiers/stage, beam→A* |
| Prewarm | JIT lookahead | none | BNN | window (IT-exec) | yes | none | per-request continuous JIT |
| Container reuse settle | not modeled | n/a | not modeled | not modeled | n/a | n/a | **measured ~2.3s, modeled** |
| Keep-alive | none | none | none | keepalive-or-prewarm (arrival stats, 1 SLA) | utility | none | static/per-stage + multi-SLO |
| Time-varying load | static | static | BNN pred | online IT | Fourier | runtime | **multi-method forecast** |
| Joint optimization | partial | sizing only | separated | resource+prewarm | keepalive only | none | **resource+prewarm+dynamic** |
| Multi-SLO classes | no | no | no | no (single SLA) | no | no | **yes** |
| Online adaptive | none | none | partial | none | yes | yes | dynamic mid-workflow (planned) |

---

## Innovation Directions

### Innovation 1 (Core Contribution): Online Risk-Driven DAG Joint Planning

No existing work does the full chain:
```
time-varying forecast → DAG propagation → joint (resource + prewarm + keepalive) planning
```

- ORION/Jolteon: static, one-shot optimization.
- AQUATOPE: predicts but doesn't propagate through DAG.
- IceBreaker: predicts but no DAG awareness.
- GrandSLAm: DAG-aware but reactive only, no proactive planning.

Our pipeline (Stage 2 → delay_kernel → analytical risk → joint planner) is
the first to connect forecasting, DAG propagation, and joint multi-knob
planning in one online loop.

### Innovation 2 (Technical Tool): Physically-Informed Latency Model

Applicable when using mock/profiled actions (our experimental setup):
```
T(cpu) = T_serial/cpu + T_parallel/min(cpu, workers) + T_io + T_memory + T_overhead
```
- More interpretable than Jolteon's opaque fit.
- Fewer profiling samples needed (physical prior constrains the curve shape).
- Degrades to Jolteon-style fit for real functions without profiling hints.

**Caveat (from our discussion):** Real functions only report total
action_duration_ms, not per-component breakdown. The Amdahl decomposition
is valid for our mock actions and serves as an upper bound on what's
achievable with instrumented functions. For uninstrumented functions, we
fall back to parametric T(memory) fitting.

### Innovation 3 (Mechanism): keepalive handling

CORRECTION: SMIless DOES use keepalive (keepalive-or-prewarm per function
from arrival statistics, single SLA). So "nobody does keepalive" is wrong.
Our differentiation is: keepalive under MULTI-SLO classes + combined with
per-request JIT and dynamic mid-workflow replanning, not keepalive
existence per se.

### Innovation 3b (Measurement + Mechanism): Container Reuse Settle Delay

Discovered while building per-request JIT + warmup-synchronized dispatch:
on OpenWhisk, after a warmup invocation completes (container created, code
loaded), the container is NOT immediately reusable by a subsequent real
invoke. There is a settle delay (measured gap threshold ~2.3s, consistent
across stages: gap>2.4s -> reuse hit, gap<2.0s -> cold start) before the
container is schedulable from the free pool.

Why this is novel:
- Window/statistical prewarming (SMIless: prewarm_window = IT - exec_time)
  ASSUMES instant reusability and is insensitive to this delay.
- Only per-request, precise-timing JIT (ours) hits this wall, because it
  tries to make the container ready exactly when the downstream is needed.
- The first hop (entry's direct downstream) is hardest: its lead time is
  just the entry execution, and the warmup itself costs a full cold start,
  so the gap is tight and often < 2.3s.

Our mechanism: warmup-synchronized bounded-wait dispatch — wait (bounded
by cold_overhead) for the settle window before issuing the real invoke,
turning a blind cold start into a short wait + warm hit.

STATUS: pending confirmation by the container-reuse-delay probe experiment
(scripts/probe_container_reuse_delay.py, to be run). If confirmed, this is
a standalone measurement insight + mechanism contribution that prior DAG
prewarming work (SMIless/ORION/AQUATOPE) does not address.

### Innovation 3c (Search): A* with risk-aware heuristic (beam as baseline)

SMIless frames its planner as A* over the DAG with an admissible heuristic
(remaining min execution time). We can re-express our planner the same way
on a harder decision space (9 memory tiers/stage vs their 2 devices/node),
using our marginal efficiency (risk_reduction / cost_increase) as a MORE
INFORMED, risk-aware heuristic. The existing beam search becomes a
comparison baseline (SMIless similarly keeps bfs/dfs/beam variants). The
marginal-efficiency theory is reused as (a) A* search-ordering heuristic
and (b) fast near-optimal fallback for online dynamic replanning.

### Innovation 3d (original): keepalive as joint variable

(Original framing, partially superseded by 3 above.) Expose keepalive as a
decision variable jointly optimized with prewarm and memory tier, valuable
for bursty-then-quiet workloads. Note SMIless already does a
keepalive-or-prewarm choice; our addition is the multi-SLO + JIT + dynamic
combination.

### Innovation 4 (Modeling): DAG Delay Kernel for Demand Propagation

No prior work explicitly models how workflow entry arrivals propagate into
per-stage demand with a conditional (warm/cold) delay distribution.

Our delay_kernel captures:
- "Given entry at window t, stage S is triggered at window t+k with
  probability p(k|warm) or p(k|cold)."
- This drives both JIT prewarm timing (when to start warming) and
  per-stage demand forecasting (how many containers needed at t+k).

### Innovation 5 (Discussed but Deferred): Cold-Start Cascading

The observation that one stage's cold start delays downstream stages,
potentially causing their containers to time out and also cold-start.
JIT prewarming mitigates this, but the cascading model itself informs
the prewarm timing decisions. Currently folded into delay_kernel design
rather than a standalone contribution.

---

## Proposed Paper Framing

**Title direction:** "Online Risk-Budgeted DAG Scheduling with Joint
Resource, Prewarm, and Keep-alive Control for Serverless Workflows"

**Problem:** How to jointly plan resource allocation, container prewarming,
and keep-alive duration for serverless DAG workflows under time-varying
workloads to meet end-to-end SLO while minimizing cost.

**Gap in prior work:**
- Static methods (ORION, Jolteon) cannot adapt to workload variations.
- Online methods (AQUATOPE, IceBreaker) lack DAG-level joint optimization.
- No one jointly optimizes all three control knobs (resource + prewarm +
  keepalive) with DAG-aware risk estimation.

**Our approach:**
1. Time-varying entry forecast with online adaptive method selection (Stage 2).
2. DAG delay kernel for propagating entry demand to per-stage load (Stage 3).
3. Fast analytical risk model for evaluating plans, validated against MC
   ground truth (Stage 4).
4. Online joint planner: per-window re-solve (memory, prewarm, keepalive)
   using chance-constrained optimization (Stage 5).

**Contributions:**
- First forecast-driven, DAG-aware, online joint planner for serverless DAGs.
- Physically-informed latency model (Amdahl-based, for profiled functions).
- keepalive as explicit per-stage optimization variable.
- DAG delay kernel for demand propagation and JIT prewarm timing.

---

## Trace Status (as of 2026-05-25)

Latest replay: `reports/civic_azure_cand2_45min_1280mb_1cpu_keepalive20s_target20s_balanced_mi96/`
- 4011 workflows, 0 errors, all summary files present.
- 45 min duration (1.5x stretched Azure trace), max_inflight=96.
- Cold distribution: all_cold=14, partial_cold=365, all_warm=3632.
- Per-stage cold samples: detect_object=101, estimate_pose=95, match_face=123,
  classify_scene=109, translate_alert=111.
- Warm E2E mean=17963ms, Cold E2E mean=28903ms.
- Action config: 1280 MB / 1 vCPU, keepalive=20s.

This trace is the first clean, full-scale replay with the tuned action
configuration and can serve as input for Stage 3 profiling.

---

## Next Steps (Pending Alignment)

1. Run Stage 3 (profile_latency + build_delay_kernel) on the new trace.
2. Run Stage 4 Monte Carlo with a trial SLO to validate the pipeline.
3. Design memory/CPU sweep experiment for the Amdahl/parametric latency model.
4. Formalize the Stage 5 optimization problem (chance-constrained).
5. Implement analytical risk model as fast alternative to MC.

---

## Appendix: SMIless Source-Verified Facts (2026-06, github blinkbear/smiless-ad)

Read directly from optimizer-engine/optimizer/{smiless.py, path_search.py}.
Recorded so we never again rely on the (wrong) abstract-level summary.

Search (path_search.py):
- Main: A* / priority-queue search over DAG nodes in topological order.
- Per node, only 2 device candidates {cpu, cuda}. Search space 2^N.
- Priority f = g (accumulated actual cost of decided nodes) + h.
- Heuristic h (calc_heuristic_cost): sum of remaining nodes'
  gpu_inference_time (min exec lower bound); prune to inf if
  current_time + remaining > SLA.
- Constraint completion_time <= SLA; objective min total cost.
- Variants implemented: path_search (A*-like), A_star_search, bfs, dfs,
  bfs_with_top_k(k=2) == beam (called aug_smiless). Main is A*.

Prewarm (calc_prewarm_window):
- prewarm_window = max(0, IT - execution_time), IT = inter-arrival time.
- Window/statistical level, based on predicted arrival rate. Assumes a
  warmed container is immediately reusable (no settle-delay modeling).

Keepalive (smiless.py get_keep_alive_time*):
- Per function: if prewarm_window == 0 -> keep_alive_time =
  calc_keep_alive_time(entry)*unit (KEEP ALIVE); else -> -1 (rely on
  prewarm). So it is keepalive-OR-prewarm, single SLA.
- Variants: SMIlessFIP uses Fourier extrapolation for arrival prediction;
  SMIlessAzure uses distribution-based IT+keepalive prediction.

Forecasting:
- Online predictor estimates inter-arrival time (IT). SMIlessFIP uses
  Fourier extrapolation (n_harm harmonics) — similar in spirit to our
  fip-fourier Stage 2 method.

Built on OpenFaaS (not OpenWhisk). Reported up to 5.73x cost reduction
vs baselines while meeting SLA.

Corrections to earlier notes in this doc: NOT keepalive=0; NOT knapsack;
it is A* over a CPU/GPU device choice with arrival-driven keepalive-or-
prewarm.
