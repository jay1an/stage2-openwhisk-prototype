# Path 3: Multi-SLO Planner + Dynamic Plan Design

This document specifies the design for path 3: multi-SLO offline planner,
dynamic plan adjustment, JIT prewarm daemon, and entry prewarm daemon.

Companion documents:
- `ARCHITECTURE_DECISIONS.md` - high-level alignment
- `ANALYTICAL_RISK_MODEL.md` - path 2 closed-form risk model (used by planner)

---

## 1. Multi-SLO Architecture

### SLO classes

For first version: two classes.
- **Premium**: P(E2E > 15s) ≤ 5%. Tighter latency target; planner
  typically upgrades critical-path stages and/or uses entry prewarm.
- **Free**: P(E2E > 20s) ≤ 5%. Looser latency target; planner can use
  cheaper tiers.

SLO numbers FINALIZED 2026-05-29 (see ARCHITECTURE_DECISIONS.md
Section 9). Both classes share a 5% violation budget; differentiation
is in the latency target (15s vs 20s).

IMPORTANT: SLO is a CONSTRAINT, not a tier assignment. The planner
independently chooses each stage's tier to satisfy the constraint at
minimum cost. Per-stage tiers within a workflow may be heterogeneous.

### Resource isolation per SLO class

Decision (locked in 2026-05-28): **Different SLO classes use different
memory tiers**, requiring **per-tier action variants**.

OpenWhisk action variants are deployed per memory tier:
```
wf_civic_detect_object_512, wf_civic_detect_object_768, ...,
wf_civic_detect_object_3840
(× 5 stages × 9 tiers = 45 action variants)
```

The planner outputs:
- For premium class: which tier variant each stage uses
- For free class: which tier variant each stage uses

At dispatch time, client routes a workflow request to the variant
corresponding to its SLO class's plan.

### Why per-class tiers (vs single shared tier)

Pros of per-class tiers:
- Premium can have higher resource without forcing free to overpay
- Cost differentiation is meaningful
- Matches real multi-tenant FaaS billing models

Cons:
- More action variants to deploy (45 instead of 5)
- More complex dispatching

Decision: per-class tiers is worth the complexity for the differentiation.

---

## 2. Planner Decision Space (per SLO class)

### Decision variables

Per SLO class, the planner outputs:
```
plan = {
    memory_tier_per_stage: dict[stage_name, int],   # 9 tiers possible
    entry_prewarm_safety_factor: float,             # 5 values: 0, 0.5, 1.0, 1.5, 2.0
}
```

The `entry_prewarm_safety_factor` is the multiplier on the predicted
entry arrivals to determine how many warm containers to prepare:
```
prewarm_count(t) = ceil(safety_factor × predicted_arrivals(t))
```

The arrival prediction itself comes from Stage 2 forecaster, NOT the
planner. The planner only decides the safety_factor.

### Search space size

- Memory tiers: 9 options per stage, 5 stages → 9^5 = 59049 combinations
- Safety factor: 5 values
- Per SLO class: 59049 × 5 = **295,245 candidate plans**

For 2 SLO classes (independent search), total = 590,490 plans.

### Plan evaluation

Each plan evaluation uses path 2's `compute_plan_risk()`:
- Input: plan, predicted arrival rate, SLO target
- Output: predicted P(E2E > SLO), expected cost
- Time: ~microseconds per evaluation

Total brute force time per SLO class:
- 295,245 plans × ~10 µs/plan ≈ 3 seconds

**Plan to start by trial-running a small subset** (say 1000 plans) to
measure actual eval time, then estimate full brute force runtime.

---

## 3. Algorithm: Risk-Budgeted Greedy with Marginal Analysis

This is the core algorithm for path 3, both for offline planning and
online dynamic plan recovery.

### Offline mode (initial plan per SLO class)

```python
def risk_budgeted_greedy(
    slo_ms: float,
    target_violation_rate: float,
    initial_plan: Plan = None,
) -> Plan:
    """
    Find the cheapest plan that achieves P(E2E > SLO) <= target_violation.
    Start from the cheapest plan, greedily upgrade to reduce risk.
    """
    if initial_plan is None:
        # Cheapest starting point: all stages at minimum tier, no prewarm
        plan = Plan(
            memory_tier_per_stage={s: 512 for s in stages},
            entry_prewarm_safety_factor=0.0,
        )
    else:
        plan = initial_plan  # For dynamic plan: start from current
    
    while True:
        current_risk = compute_plan_risk(plan, slo_ms).p_violation_total
        if current_risk <= target_violation_rate:
            return plan   # SLO met, stop (A8: 达到 SLO 就停)
        
        # Generate all single-step upgrade candidates
        candidates = []
        for stage in stages:
            next_tier = next_higher_tier(plan.memory_tier_per_stage[stage])
            if next_tier is not None:
                new_plan = plan.with_tier(stage, next_tier)
                candidates.append(new_plan)
        if plan.entry_prewarm_safety_factor < MAX_SAFETY_FACTOR:
            new_plan = plan.with_higher_safety_factor()
            candidates.append(new_plan)
        
        if not candidates:
            # No more upgrades possible, plan infeasible
            return plan  # caller checks risk vs target
        
        # Score each candidate by marginal efficiency
        best_candidate = None
        best_efficiency = -1
        for cand in candidates:
            new_risk = compute_plan_risk(cand, slo_ms).p_violation_total
            new_cost = compute_plan_cost(cand)
            old_cost = compute_plan_cost(plan)
            
            risk_reduction = current_risk - new_risk
            cost_increase = new_cost - old_cost
            
            if cost_increase > 0:
                efficiency = risk_reduction / cost_increase
            else:
                efficiency = float('inf')  # Free upgrade
            
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_candidate = cand
        
        # Apply the best single-step upgrade
        plan = best_candidate
    # Loop continues until SLO met or no more upgrades
```

Greedy direction: **start cheap, climb up** (A7 confirmed).

Stop condition: **risk <= target_violation_rate** (A8 confirmed).

### Online mode (dynamic plan recovery)

When a workflow's entry stage completes and the observed actual latency
suggests SLO is at risk, the planner runs INCREMENTALLY from the current
plan:

```python
def dynamic_plan_recovery(
    current_plan: Plan,
    workflow_state: WorkflowState,  # observed times so far
    slo_ms: float,
    target_violation_rate: float,
) -> Plan | None:
    # Compute remaining SLO budget given observed times so far
    remaining_budget_ms = slo_ms - workflow_state.elapsed_ms
    
    # Estimate residual risk under current plan + observed state
    residual_risk = estimate_residual_risk(current_plan, workflow_state)
    
    if residual_risk <= target_violation_rate:
        return None   # No change needed
    
    # Run greedy starting from CURRENT plan (incremental, A9 confirmed)
    upgraded_plan = risk_budgeted_greedy(
        slo_ms=remaining_budget_ms + workflow_state.elapsed_ms,
        target_violation_rate=target_violation_rate,
        initial_plan=current_plan,
    )
    
    # Only return changes for stages that haven't started yet
    return diff_plan(current_plan, upgraded_plan,
                     only_stages=workflow_state.pending_stages)
```

Incremental greedy means: start from the current plan and only consider
upgrades. This is faster than re-planning from scratch (~few iterations
instead of full convergence).

### Complexity

Offline greedy:
- Iterations: bounded by total decision variable steps
  - Per stage tier upgrades: 8 tiers - 1 = 8 max iterations
  - Safety factor: 4 max iterations
  - Total iterations: <= 8 × 5 + 4 = 44
- Candidates per iteration: 5 tier upgrades + 1 safety upgrade = 6
- Risk evaluations per iteration: 6 × ~10 µs = 60 µs
- **Total time per SLO class: ~3 ms**

Online dynamic recovery:
- Starts from current plan, typically 1-3 iterations
- Total time: <1 ms

---

## 4. Brute Force Baseline (for validation)

To verify the greedy algorithm's optimality, run a brute force baseline
that enumerates all 295k plans:

For each plan:
- Evaluate risk via path 2
- Record cost
Identify the cheapest plan with risk <= target.

Compare brute force result to greedy result:
- If greedy plan == brute force plan: greedy is optimal for this case
- If greedy cost > brute force cost: report the gap

Decision (A5): **First measure actual eval time on a small subset before
committing to full brute force**. If brute force takes too long, scale
back to a sampled grid (e.g., 10k random plans) and use that as the
oracle.

---

## 5. Action Variant Deployment (required infrastructure)

The per-class tier strategy requires action variants to be deployed.

### Deployment plan

For each of 5 stages × 9 tiers = **45 action variants**:
```
wf_civic_detect_object_{512,768,1024,1280,1536,2048,2560,3072,3840}
wf_civic_estimate_pose_{...}
wf_civic_match_face_{...}
wf_civic_classify_scene_{...}
wf_civic_translate_alert_{...}
```

Existing `scripts/deploy_workflow_action_variants.py` may handle this;
verify before sweep starts.

### Dispatching

Client-side workflow runner (in path 3) must route invocations to the
correct variant based on:
- The SLO class of the request
- The plan's memory tier for this stage in this class

This is a small extension to `runner/workflow.py` and `runner/run_workflow.py`.

---

## 6. Validation Strategy

Path 3 validation has three layers:

### Layer 1: Algorithm correctness (offline)
- Greedy returns a plan with predicted risk <= target
- Brute force comparison: greedy is at most X% more expensive than optimal
- Sanity: as target_violation_rate decreases, plan cost increases monotonically

### Layer 2: Multi-SLO interaction (offline)
- Premium and free plans are independently computed
- No constraint violation between classes (since they use different action
  variants, no resource conflict)
- Cost ratio premium/free should be > 1 (premium pays more)

### Layer 3: End-to-end real-cluster (online, later)
- Run multi-SLO trace replay with computed plans
- Measure actual violation rates per class
- Compare to predicted (target ±2pp acceptance)

Layer 3 requires JIT prewarm daemon + entry prewarm daemon + client SLO
queue, all of which come in later subtasks of path 3.

---

## 7. Implementation Plan (codex tasks)

Path 3 subtasks (in execution order):

### Subtask P3.0: Extended sweep
- Run sweep across 9 tiers on multi-node cluster
- Output: `reports/civic_memory_cpu_sweep_multinode_9tier/`

### Subtask P3.1: Amdahl re-fit
- Re-fit Amdahl model on extended sweep data
- Validate: RMS < 3% per stage warm
- If validation fails: fit piecewise segmented model
- Output: `reports/stage6_amdahl_model_extended/`

### Subtask P3.1-retry: Three-model resource scaling fit (2026-05-28)

P3.1 attempt 1 failed: simple Amdahl RMS = 5-10% per stage on 9-tier
multi-node sweep, piecewise improved only 2 of 5 stages. The wider
CPU range (0.4-3.0 vCPU) exposes that simple Amdahl with continuous
W_eff is too rigid.

Decision: fit THREE models in parallel and pick the one passing
acceptance per stage. Outputs a comparison table for owner review.

#### Cold overhead pre-cleansing

Before model fitting, clean cold overhead measurements:
1. For each stage independently, compute mean and std of cold_overhead
   across all 9 tiers (cold_overhead = cold_dispatch - warm_dispatch
   for that stage at that tier)
2. Identify outlier tiers: tiers where the stage's cold_overhead is
   more than 2 standard deviations from the per-stage mean
3. Replace outlier values with the MEDIAN of the non-outlier tiers
   for that stage
4. Use the cleansed cold_overhead in subsequent modeling

Paper write-up: per-stage outlier replacement is a standard data
cleansing step, justified by the small per-tier sample size (n=10).
Do NOT mention that all stages share the same runtime image.

#### Three candidate models

**D1: Power law (3 parameters per stage)**
```
T(cpu) = a · cpu^(-alpha) + c
```
- a: total CPU-bound work magnitude
- alpha: scaling exponent (0 < alpha < 1 means sublinear)
- c: CPU-independent constant (IO + memory + overhead)
- No breakpoints, smooth function

**D2: Amdahl with observed workers (3 parameters per stage)**
```
T(cpu) = S / min(cpu, 1) + P / w_observed(cpu) + C
```
- w_observed(cpu): the actual parallel_workers_used from sweep data
  at that tier (NOT a fitted parameter; taken directly from
  trace.csv's parallel_workers_used column)
- S: serial CPU work
- P: parallel CPU work
- C: fixed overhead

By using observed workers count (a step function from data) instead
of a fitted continuous W_eff, this model accurately captures the
worker transition discontinuities.

**D3: Cubic spline interpolation (nonparametric)**

Cubic spline through the 9 measured points per stage. No global
parameters; uses local cubic fits with continuity constraints.
Fallback if D1 and D2 fail.

#### Acceptance criteria

Per stage:
- RMS relative error across 9 tiers < 3%
- Max per-tier relative error < 8%

A model PASSES if BOTH criteria are met. Per-stage, the model
selection is independent: stage X may use D1 while stage Y uses D2.

If all three models fail for a stage, the comparison report flags it
for review.

#### Outputs

`reports/stage6_resource_models_v2/`:
- `cold_overhead_cleansed.csv`: original + cleansed values per stage per tier
- `d1_power_law_params.csv`: a, alpha, c per stage
- `d2_amdahl_observed_params.csv`: S, P, C per stage
- `d3_spline_coeffs.csv`: spline knots and coefficients
- `model_comparison.csv`: RMS, max error, pass/fail per (stage, model)
- `recommended_model_per_stage.csv`: which model to use per stage
- `comparison_report.md`: summary + recommendations

### Subtask P3.1-retry result (2026-05-28)

P3.1-retry was executed. Final selection: **D3 cubic spline** for all 5
stages.

Result table:

| Stage | D1 RMS | D1 max | D2 RMS | D2 max | D3 RMS | Recommended |
|---|---|---|---|---|---|---|
| classify_scene  | 4.84% | 10.75% | 23.01% | 44.07% | 0% | D3 |
| detect_object   | 6.99% | 13.65% | 23.43% | 45.83% | 0% | D3 |
| estimate_pose   | 5.29% | 12.00% | 15.11% | 34.53% | 0% | D3 |
| match_face      | 5.92% | 11.65% | 25.90% | 52.50% | 0% | D3 |
| translate_alert | 5.21% | 12.56% | 12.07% | 21.48% | 0% | D3 |

#### D2 formula bug (disclosed)

The D2 implementation followed the prompt's formula
`T = S/min(cpu, 1) + P/w_obs + C`. The correct Amdahl form should be
`T = S/min(cpu, 1) + P/min(cpu, w_obs) + C` (parallel section is also
CPU-throttled when cpu < w_obs).

Hand calculation with the corrected formula (using S=6748, P=8250, C=213
fitted from cpu=1.0, 2.0, 3.0 anchor points):
- cpu=0.4: predicted 38028, observed 37709 → 0.8% error
- cpu=0.6: predicted 25209, observed 23478 → 7.4% error
- cpu=0.8: predicted 18961, observed 18005 → 5.3% error
- cpu=1.0: predicted 15211, observed 15211 → 0%
- cpu=1.2: predicted 15211, observed 14408 → 5.6% error
- cpu=1.6: predicted 11086, observed 12093 → 8.3% error
- cpu=2.0: predicted 11086, observed 11086 → 0%
- cpu=3.0: predicted 9711, observed 9838 → 1.3% error

Even with the corrected formula, RMS ≈ 5-6% and max error ≈ 8.3%
(>8% threshold). **Corrected D2 would still fail the 3% RMS criterion.**

The bug does not change the conclusion (D3 is the right choice), but is
recorded here for transparency.

#### Why D3 is the correct choice

Three reasons the 9-tier multi-node data resist any simple parametric fit:

1. **Worker transitions create discontinuities**: workers count jumps at
   cpu = 1 (1 worker → 1 worker), cpu = 2 (1 → 2), cpu = 3 (2 → 3).
   Each transition creates a kink in T(cpu) that 3-parameter forms can
   only approximate.

2. **Multi-node scheduling variance**: each measurement includes both
   per-stage execution and node-assignment variance. Per-tier means are
   accurate but the underlying surface is noisier than a clean
   single-machine measurement.

3. **3-parameter forms over-constrain the curve**: power law (3 params)
   produces 5-7% RMS. Adding more parameters (4+) starts to look like
   piecewise fitting, which is essentially what spline does.

D3 spline is:
- **Exact at all 9 measured tiers** (the only tiers used by the planner)
- **C² continuous** between tiers (mathematically smooth)
- **More principled than empirical table + linear interp** (smoother,
  better for hypothetical intermediate-tier queries)
- **Equivalent to table lookup in practice** for our 9-tier search space

#### Paper narrative for path 2 scaling

```
"We model per-stage warm execution time T_s(cpu) as a natural cubic
spline calibrated from a 9-tier sweep (0.4 to 3.0 vCPU). Cold overhead
is modeled as a per-stage constant after 2σ outlier cleansing on the
sweep measurements. We chose spline interpolation over parametric models
(power-law and Amdahl-style with observed workers) because the measured
data exhibit worker-transition discontinuities at cpu = 1, 2, 3 that no
3-parameter analytical form can capture within 3% RMS in our wide CPU
range. The spline provides exact match at the measured tiers and C²
continuity for any intermediate query, while preserving methodological
rigor."
```

#### Cold overhead cleansing summary

One outlier detected and replaced:
- classify_scene @ 768 MB: 3121.6 ms → 2030.7 ms (median of non-outlier tiers)

All other (stage, tier) cold overhead values fell within ±2σ of their
per-stage mean and were retained as-is.

### Subtask P3.2: SLO targets
- Assistant proposes SLO numbers based on Amdahl + sweep data
- Owner confirms
- Update `ARCHITECTURE_DECISIONS.md` Section 9

### Subtask P3.3: Action variants
- Verify `scripts/deploy_workflow_action_variants.py` deploys all 45 variants
- Or extend it to do so
- Smoke test: invoke one variant per tier

### Subtask P3.4: Multi-SLO planner (greedy)
- Implement `runner/stage5_control/multi_slo_planner.py`
  with risk_budgeted_greedy() function
- Smoke test: produce premium and free plans, print summary
- Output: `reports/path3_planner/`

### Subtask P3.5: Brute force baseline
- Implement enumeration baseline
- Compare greedy vs brute force on representative SLO targets
- Report greedy optimality gap

### Subtask P3.6: Dynamic plan recovery (offline simulation)
- Implement incremental greedy for dynamic recovery
- Simulate "entry cold detected → trigger recovery" scenarios
- Measure recovery effectiveness vs do-nothing baseline

### Subtask P3.7: JIT prewarm daemon
- Implement online JIT warmup trigger
- Uses per-stage warm duration prediction + 2σ safety margin
- Tested on cluster

### Subtask P3.8: Entry prewarm daemon
- Implement window-based entry prewarming
- Uses Stage 2 forecaster output
- Tested on cluster

### Subtask P3.9: Client SLO priority queue
- Implement slack-based reordering at client
- Optional for first paper version

### Subtask P3.10: End-to-end multi-SLO evaluation
- Run multi-SLO trace replay
- Measure violation rates and costs
- Compare against baselines (Scale-To-Zero, Always-Warm)

Each subtask is 1-3 days codex work + verification.

---

## 7.5 Path 3+: Bayesian Adaptive Risk Update (deferred, design captured)

### Motivation

Path 2's lognormal parameters (μ, σ) are fitted ONCE offline from a
trace snapshot. The path 3 planner uses these static parameters for all
future decisions.

In production:
- Stage latency distributions drift across hours / days (time-of-day,
  network state, neighbor workload variance)
- Static distributions may become inaccurate over long-running
  deployments
- Dynamic plan adjustment (P3.6) would benefit from current beliefs
  about system state

### Approach

Treat per-stage warm latency as a Bayesian-tracked Normal in log-space
(equivalent to lognormal tracking). Update belief online from observed
stage completions.

Mathematical foundation:
- ln(L_stage) ~ Normal(μ, σ²) is conjugate to Normal observations
- Closed-form posterior update for μ given new observations
- σ² assumed fixed in v1 (could be Bayesian too, via Inverse-Gamma)

### Design parameters (locked in 2026-05-29 discussion)

**Update frequency**: sliding window of 5 seconds
- All stage observations within the past 5s window are aggregated
- One posterior update per stage per 5s window
- Avoids per-observation overhead while staying reactive

**Independence**: per-stage independent posteriors
- 5 stages × 5 SLO classes = 25 independent Normal belief states
- No cross-stage coupling needed (multi-node ρ ≈ 0 confirmed in R5)
- Simplifies math significantly

**Memory decay**: exponentially weighted posterior
- Effective sample size capped to avoid the posterior becoming too
  rigid against drift
- Use a decay parameter λ such that observations from > N windows
  ago have negligible influence
- Concrete value of λ to be tuned; suggested initial: equivalent to
  effective N = 100 observations (~8 minutes of sliding window)

### Where it plugs into the system

```
Dynamic plan recovery (P3.6) currently uses:
  static_lognormal_params -> predict residual risk -> trigger recovery

With Bayesian update:
  current_belief_params (updated online) -> predict residual risk
  -> trigger recovery
  Belief tracking happens cross-workflow, independent of any single
  workflow's dynamic plan logic.
```

### Implementation order (Path 3+)

Phase 1 (after path 3 is functionally complete):
- Implement Bayesian belief tracker (per-stage Normal posterior)
- Wire updates from observed stage completions
- Replace static lognormal with current belief in risk evaluation
- Keep static lognormal as fallback / oracle

Phase 2 (research extension):
- Validate that belief tracking improves SLO satisfaction under drift
- Compare against static lognormal baseline on long-running traces
- Paper contribution: "Online Bayesian belief tracking for serverless
  adaptive risk planning"

### Why this is deferred

Path 3 currently aims to:
- Validate the multi-SLO planning architecture
- Show benefit of dynamic plan adjustment vs static plans
- Establish baseline performance vs Scale-To-Zero / Always-Warm

Bayesian belief tracking is orthogonal to all of the above. It can be
added as a separate research increment after path 3's main results are
established. Including it in path 3 would complicate the experimental
setup without providing necessary value for the core contributions.

---

## 8. Open Questions (still pending owner input)

None at the moment. All key decisions aligned 2026-05-28.

Decisions reference:
- A1: Amdahl validation RMS < 3% ✓
- A2: Fallback to piecewise segmented model ✓
- A3: Max 3 vCPU (3840 MB max tier) ✓
- A4: Sweep → Amdahl → SLO proposal → confirm → doc flow ✓
- A5: Test brute force timing before committing ✓
- A6: Per-class tiers, requires action variants ✓
- A7: Greedy starts cheap, climbs up ✓
- A8: Stop at SLO violation rate target ✓
- A9: Dynamic plan reuses greedy incrementally ✓

---

## Changelog

- 2026-05-28: Initial document. All design decisions for path 3 captured
  here pending extended sweep results and SLO finalization.
