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

### 6. SMIless (SC 2024)

**How it finds a plan:**
Critical-path analysis + knapsack-style resource-cold-start co-optimization.

1. Analyze DAG, find critical path (considering dynamic invocations where
   different inputs activate different branches).
2. Allocate budget to critical-path stages first (knapsack: maximize latency
   reduction per dollar spent).
3. Co-optimize resource config and prewarm targets jointly.

keepalive is always 0 (scale-to-zero between requests).

**Relevance to our project:**
- Critical-path-first budget allocation is a good heuristic for our planner.
  Our DAG has one join point (classify_scene), so the critical path shifts
  between the two paths depending on cold/warm state.
- The knapsack formulation could inspire our resource allocation: given a
  cost budget, which stages benefit most from more memory?

**What we improve over SMIless:**
- SMIless has keepalive=0. We add keepalive as an optimization variable.
  This matters for bursty workloads with quiet periods — keepalive bridges
  short gaps between bursts without paying cold-start cost.
- SMIless targets ML inference (fixed DAG with conditional branches). We
  handle general serverless workflows with time-varying arrival rates.
- SMIless does not forecast arrivals. We use Stage 2 forecasting to
  proactively adjust plans.

**How keepalive helps (our discussion):**
keepalive is a cost-risk tradeoff knob:
- Long keepalive → containers survive quiet gaps → fewer cold starts → more
  idle cost.
- Short keepalive → containers released quickly → save idle cost → cold
  starts after gaps.
- Combined with prewarm: short keepalive + accurate JIT prewarm = cheapest
  way to avoid cold starts. Long keepalive = insurance when forecasting is
  inaccurate.
The planner can trade off between these strategies per-stage and per-window.

---

## Cross-Paper Comparison

| Capability | ORION | Jolteon | AQUATOPE | SMIless | IceBreaker | GrandSLAm | Ours |
|---|---|---|---|---|---|---|---|
| DAG-aware latency | CDF conv | parametric | per-stage | crit-path | none | per-stage | MC + analytical |
| Resource sizing | greedy | convex opt | BO | knapsack | none | none | sweep + opt |
| Prewarm | JIT | none | BNN | yes | yes | none | JIT per-window |
| Keep-alive | none | none | none | none | utility | none | **per-stage opt** |
| Time-varying load | static | static | BNN pred | static | Fourier | runtime | **multi-method forecast** |
| DAG propagation | none | none | none | none | none | none | **delay_kernel** |
| Joint optimization | partial | sizing only | separated | partial | keepalive only | none | **all three knobs** |
| Online adaptive | none | none | partial | none | yes | yes | planned |

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

### Innovation 3 (Mechanism): keepalive as Explicit Optimization Variable

All prior work either ignores keepalive (ORION, Jolteon, AQUATOPE, SMIless)
or treats it independently from DAG-level SLO (IceBreaker).

We expose keepalive_sec as a per-stage, per-window decision variable and
jointly optimize it with prewarm_count and memory_tier. This is especially
valuable for bursty-then-quiet workloads where:
- Short keepalive + accurate prewarm = minimal cost, no cold starts.
- Long keepalive = insurance against forecast errors during quiet periods.

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
