# Analytical Risk Model: Math Specification

This document specifies the closed-form (analytical) risk model for path 2.
It replaces Monte Carlo as the inner loop of the planner: microsecond-scale
evaluation instead of seconds-scale.

Companion documents:
- `ARCHITECTURE_DECISIONS.md`: high-level design and decisions
- `RELATED_WORK_AND_INNOVATION.md`: paper positioning and prior art

---

## 1. Goal

Given:
- a workflow plan (memory tier per stage, entry prewarm count per window)
- the workload (entry arrival forecast)
- per-stage latency parameters fitted from real trace
- a target SLO (e.g., 20000 ms)

Compute `P(workflow E2E > SLO)` in closed form, in microseconds.

This drives the planner's inner loop and supports both static (one-shot)
and dynamic (online re-evaluation) planning.

---

## 2. Stage-level latency distribution

### Distribution choice: Lognormal

Each stage's execution time follows a lognormal distribution:

```
L_s ~ LogNormal(μ_s, σ_s)

PDF:  f(x; μ, σ) = (1 / (x σ √(2π))) · exp( -(ln x - μ)² / (2σ²) )
CDF:  Φ((ln x - μ) / σ)   where Φ is the standard normal CDF
```

Properties used:
- Always positive
- Right-skewed (matches observed long tails in serverless latency)
- Mean = exp(μ + σ²/2)
- Variance = (exp(σ²) - 1) · exp(2μ + σ²)
- Coefficient of variation: CV = sqrt(exp(σ²) - 1)

### Parameter fitting from real trace

For each (stage_name, latency_class) pair where latency_class is `warm`
or `cold`:

Step 1: extract sample set `S = {L_1, L_2, ..., L_n}` from
`latency_samples_for_monte_carlo.csv`, using the `dispatch_latency_ms`
column (total perceived latency at workflow runtime).

Step 2: fit lognormal via maximum likelihood (closed-form for lognormal):
```
μ = mean(ln L_i)
σ² = variance(ln L_i)
```

Step 3: store `(μ_warm, σ_warm)` and `(μ_cold, σ_cold)` per stage in
`reports/path2_lognormal_fit/per_stage_lognormal_params.csv`.

### Caveat: cold sample contamination (carried from P4)

The `cold_like` sample pool contains both clean cold starts and cold
starts under platform contention. For path 2 first version, accept this
contamination: fit lognormal to all cold samples per stage. This will
produce a wider σ_cold than the "clean cold" reality.

Future refinement: fit cold lognormal separately for "full cascade"
(workflow_cold_count == 5) workflows vs others, since the asymmetric
JIT model only cares about entry cold (path 2 first version uses entry
stage's cold distribution from the full cascade subset).

---

## 3. Resource scaling: lognormal under memory tier change

We measured lognormal parameters at one memory tier (1280 MB / 1.0 vCPU).
For a planner to choose between memory tiers, we need the distribution
at any tier.

### Mean scaling: Amdahl model

From `runner/stage6_resource/fit_amdahl_model.py`:
```
T_action(cpu) = S / min(cpu, 1.0) + P / min(cpu, W_eff) + C
```

where `S`, `P`, `C`, `W_eff` are fitted per stage. Gives mean execution
time at any CPU.

To get cpu from memory tier (OpenWhisk default):
```
cpu_millicores = 200 * ceil(memory_mb / 256)
cpu_cores = min(cpu_millicores, 3200) / 1000
```

### Variance scaling: CV constant assumption

Assume the coefficient of variation CV = σ / mean does not change with
memory tier:

```
CV_target = CV_1280   (constant across tiers)
σ_target = CV_1280 * mean_target
```

This is the standard assumption in serverless literature (Jolteon,
AQUATOPE).

### Lognormal parameter transformation under scaling

Given target_mean and CV_constant:
```
σ_target² = ln(1 + CV²) = ln(1 + (σ_1280 / mean_1280)²)
μ_target = ln(mean_target) - σ_target² / 2
```

This produces a lognormal at the new memory tier with the same shape
(CV) but scaled location (mean).

---

## 4. DAG aggregation: sums and maxes of lognormals

### civic_alert DAG topology

```
detect_object → estimate_pose → match_face → classify_scene → translate_alert
detect_object ─────────────────────────────→ classify_scene
```

Two paths converge at classify_scene:
- path1 (long): detect_object + estimate_pose + match_face
- path2 (short): detect_object alone (then classify_scene)

After classify_scene completes, translate_alert is the tail.

### Sum of lognormals: Fenton-Wilkinson moment matching

For independent lognormals `L_i ~ LogNormal(μ_i, σ_i)`, the sum is NOT
lognormal. Approximate by matching first two moments:

```
mean_sum = Σ mean_i = Σ exp(μ_i + σ_i²/2)
var_sum  = Σ var_i  = Σ (exp(σ_i²) - 1) · exp(2μ_i + σ_i²)
```

Approximate sum as `LogNormal(μ_sum, σ_sum)` with:
```
σ_sum² = ln(1 + var_sum / mean_sum²)
μ_sum  = ln(mean_sum) - σ_sum² / 2
```

This is the **Fenton-Wilkinson** approximation. Validated widely in
wireless and queueing literature.

### Max of two lognormals: Clark approximation

For `Y = max(A, B)` where A, B are lognormals (possibly correlated):

Step 1: convert to log-space (ln A, ln B are normal).

Step 2: apply Clark's formula for max of two normals:
```
a = sqrt(σ_lnA² + σ_lnB² - 2 ρ σ_lnA σ_lnB)
α = (μ_lnA - μ_lnB) / a
E[ln Y] = μ_lnA · Φ(α) + μ_lnB · Φ(-α) + a · φ(α)
E[(ln Y)²] = (μ_lnA² + σ_lnA²) · Φ(α) + (μ_lnB² + σ_lnB²) · Φ(-α)
           + (μ_lnA + μ_lnB) · a · φ(α)
Var(ln Y) = E[(ln Y)²] - (E[ln Y])²
```

where `φ`, `Φ` are standard normal PDF and CDF, ρ is correlation.

Step 3: treat Y as lognormal with `(E[ln Y], Var(ln Y))` and continue.

For path 2 first version, assume **ρ = 0** (independent paths). This is
slightly optimistic (in reality both paths share detect_object completion
time, so positive correlation), but simplifies math. Future refinement:
condition on detect_object completion and integrate.

### Full E2E aggregation for civic_alert

```
1. path1_sum = FW_sum([detect_object, estimate_pose, match_face])
2. classify_scene_start = max(path1_sum, detect_object_only)
   classify_scene_start ≈ path1_sum  (since path1 ≫ path2 in expected value)
   But use Clark approximation for tail accuracy.
3. E2E_pre_classify = classify_scene_start
4. E2E_through_classify = E2E_pre_classify + classify_scene_duration
5. E2E = E2E_through_classify + translate_alert_duration
```

All steps yield a lognormal (via moment matching).

### Simplification when path1 dominates

If `E[path1_sum] - E[path2] > 3 sqrt(var[path1_sum])`, then
`max ≈ path1_sum` to within negligible error. This is the case for
civic_alert (path1 ≈ 11s vs path2 ≈ 3.5s, gap > 5σ). The Clark step
collapses to a simple substitution.

For other workflows where paths are closer in expected value, the full
Clark math is needed.

---

## 5. Entry cold probability model

### Definition

`p_entry_cold(t)`: probability that an arriving workflow at time `t`
encounters a cold entry container.

### Naive formula

Given predicted entry demand `λ_t` over the prediction horizon and
planner's prewarm decision `W_t` (warm container count target):

```
p_entry_cold(t) = max(0, 1 - W_t / max(λ_t, ε))
```

where ε is a small constant to avoid division by zero.

### Refinement: account for container reuse

Each warm container can serve more than one request via the keepalive
mechanism. The effective warm capacity over a horizon H is:

```
effective_capacity(t, H) = W_t · (1 + K / D_entry_warm) · (H / K)
```

But this overestimates when arrivals are clustered. For first version,
use the naive formula.

### Calibration from real trace

From the 45-min real trace (W_t ≈ 0 because no manual prewarm, K = 20s):
- detect_object cold count = 101, total = 4011
- empirical p_entry_cold = 101 / 4011 = 2.52%

Use this as a baseline calibration: when planner sets `W_t = 0`, model
should predict `p_entry_cold ≈ 2.5%`, matching the OpenWhisk natural
container reuse rate at keepalive = 20s.

Adjust the naive formula to match this baseline:
```
p_entry_cold(t) = max(0, p_baseline - W_t · reuse_factor / max(λ_t, ε))
```

where `p_baseline = 2.5%` is the empirical zero-prewarm rate, and
`reuse_factor` is the average number of requests one prewarm serves.

For path 2 first version, use the naive formula with `p_baseline` floor:
```
p_entry_cold(t) = max(p_baseline_floor, 1 - W_t / max(λ_t, ε))
```
where `p_baseline_floor = 0.01` ensures the model never predicts zero
even when prewarm is aggressive.

Future refinement: fit a queueing model directly.

---

## 6. Workflow risk computation (top level)

### Two-scenario mixture

```
P(E2E > SLO) = (1 - p_entry_cold) · P(E2E_warm > SLO)
             +      p_entry_cold  · P(E2E_with_cold_entry > SLO)
```

### Computing each scenario's risk

For E2E_warm:
1. For each stage, use `(μ_warm_s, σ_warm_s)` scaled to its memory tier
2. Aggregate via Fenton-Wilkinson + Clark
3. Get final lognormal `(μ_E2E_warm, σ_E2E_warm)`
4. P(E2E_warm > SLO) = 1 - Φ((ln SLO - μ_E2E_warm) / σ_E2E_warm)

For E2E_with_cold_entry:
1. Replace entry's lognormal with `(μ_cold_entry, σ_cold_entry)`
2. Keep downstream stages' warm lognormals (JIT assumption: p_downstream_cold = 0)
3. Aggregate same way
4. P(E2E_with_cold_entry > SLO) computed identically

### Total runtime

Each evaluation:
- 5 stage lookups + scaling: ~50 floating point ops
- Fenton-Wilkinson for path1 sum: ~20 ops
- Clark for join: ~50 ops (or 5 ops if simplified)
- Final lognormal CDF: ~10 ops

**Total: ~150 ops per evaluation ≈ microseconds in Python with numpy.**

This is the key win over MC (which needs thousands of samples per evaluation).

---

## 7. Planner integration

### Planner objective (first version)

```
minimize  Cost(plan) = Σ_t Σ_s memory_tier_s · E[L_s | tier_s] · n_active_s_t
                     + Σ_t entry_prewarm_count_t · memory_tier_entry · prewarm_duration
                     
subject to  P(E2E > SLO) ≤ ε
            for each (workflow class, SLO target ε)
```

### Search algorithm (first version)

Brute force over:
- memory_tier_s ∈ {768, 1280, 2048, 2560 MB} per stage (5^4 = 625 combos)
- entry_prewarm_count_t ∈ {0, 1, ..., max_demand_t} per window

For each candidate plan:
1. Compute P(E2E > SLO) via analytical_risk
2. Compute Cost via cost model
3. Track Pareto frontier

For multi-SLO, repeat for each SLO class with its own ε threshold.

### Speedup

With ~150 ops per risk evaluation and ~10000 candidate plans:
- ~1.5M ops total
- ~milliseconds in vectorized numpy
- Acceptable even at per-window planning frequency

If brute force is too slow at scale, switch to Jolteon-style convex
relaxation: chance constraint becomes convex in (μ_s, σ_s) under
appropriate transformations.

---

## 8. Validation plan

### Step 1: per-stage lognormal fit quality

For each stage, compare:
- Empirical CDF from real trace samples
- Fitted lognormal CDF

Compute KS statistic. Should be < 0.1 for the fit to be reliable.

### Step 2: DAG aggregation accuracy

Generate ground truth via Monte Carlo:
- Draw stages independently from their lognormals
- Sum/max according to DAG
- Compute empirical CDF of E2E (100k samples)

Compare with analytical model's lognormal CDF:
- Match at p50, p90, p95, p99
- Allowable error: < 5% relative at p95

### Step 3: end-to-end calibration vs real trace

Use the actual plan from the 45-min replay (memory=1280, prewarm=0):
1. Compute predicted P(E2E > 20000ms) via analytical model
2. Compare to actual measured violation rate (11.2%)
3. Acceptable error: < 3pp absolute (so model predicts in [8%, 14%])

If error exceeds threshold:
- Check lognormal fit (Step 1)
- Check DAG aggregation (Step 2)
- Check p_entry_cold calibration

### Step 4: bound matching

Check that:
- P(E2E_warm > 20000) is small (matches Always-Warm violation rate ~1%)
- P(E2E_with_cold_entry > 20000) is small but larger
- The mixture matches the real measurement

---

## 9. Implementation plan

### Module structure

```
runner/stage4_risk/analytical_risk.py    # main risk computation
runner/stage4_risk/lognormal_fit.py      # parameter fitting
runner/stage4_risk/dag_aggregation.py    # FW + Clark
runner/stage4_risk/scaling.py            # CV-constant scaling
runner/stage4_risk/__init__.py
```

Keep existing `estimate_slo_risk.py` (MC) for ground truth comparison.

### Codex task split

Task R1: Fit lognormal parameters per stage from real trace.
Task R2: Implement DAG aggregation (FW + Clark) with civic_alert DAG.
Task R3: Wire up resource scaling + entry cold probability.
Task R4: End-to-end validation against MC and real trace.

Each task is 1-2 days of codex work, with verification.

---

## 9.5 Two Risk Computation Modes (IMPORTANT: do not conflate)

There are two distinct risk computation modes in the codebase. They
model DIFFERENT systems and produce DIFFERENT numbers. Conflating them
causes confusion (as happened during P3.1-apply review).

### Mode 1: no-JIT validation mode

Used by R4/R5 validation scripts (run_r4_path2_validation.py,
run_r5_multinode_validation.py).

- Models the CURRENT real system, which has NO JIT prewarming
- Each workflow's actual cold/warm pattern is taken from the trace
- Downstream stages CAN be cold (full partial_cold cascades happen)
- Aggregates per-workflow E2E from the observed cold/warm bits
- This is what we VALIDATE against real measured violation rates

R5 result at SLO=20s, 1280 MB multi-node:
- observed violation: 1.84%
- predicted violation: 2.74%
- error 0.89pp (PASS, validates the model math)

### Mode 2: JIT plan-risk mode

Used by compute_plan_risk() in plan_risk.py, consumed by the path 3
planner.

- Models the SYSTEM WE WILL DEPLOY, which HAS JIT prewarming
- Assumes JIT hides downstream cold starts (p_downstream_cold ≈ 0)
- Only the entry stage can be cold (p_entry_cold)
- Two-scenario mixture: entry-warm vs entry-cold (downstream always warm)

compute_plan_risk result at SLO=20s, 1280 MB:
- R5 multi-node params: 0.41% violation
- This is LOWER than the no-JIT 1.84% BECAUSE JIT eliminates downstream
  cold starts. This is the expected benefit of JIT.

### Why they differ and why that is correct

```
no-JIT (Mode 1):  P(violation) higher because downstream cold cascades
                  occur and inflate E2E latency
with-JIT (Mode 2): P(violation) lower because JIT hides downstream cold,
                   leaving only entry cold as a risk source
```

The difference 1.84% → 0.41% quantifies the JIT benefit at the modeling
level. Mode 1 validates that our distributions and aggregation math are
correct (vs real measurements). Mode 2 uses those validated distributions
to predict the deployed system's performance.

### Rule for using these modes

- For VALIDATION against real (no-JIT) traces: use Mode 1
- For PLANNING the deployed (with-JIT) system: use Mode 2
- NEVER compare a Mode 2 number directly to a Mode 1 measurement
- The path 3 planner uses Mode 2 exclusively
- The R4/R5 validation reports use Mode 1 exclusively

### Always use R5 multi-node params, not R1 single-node params

R1 fitted lognormal sigma ≈ 0.033 (single-node, low variance).
R5 refitted sigma ≈ 0.108 (multi-node, node-assignment variance).

The deployed system is multi-node, so the planner MUST use the R5
multi-node lognormal parameters
(reports/path2_lognormal_fit_multinode/). Using R1 single-node params
would drastically underestimate variance (and thus risk).

---

## 10. Out of scope (deferred to v2)

- Bayesian state inference (decided: closed-form is sufficient)
- FFT-based exact distribution convolution (use only if Clark insufficient)
- Path correlation modeling (assumed independent, may need refinement)
- Dynamic re-fitting of lognormal parameters online (use offline fit)
- Multi-stage cold cascade modeling (not needed under JIT assumption)
- Cold start tail modeling separately (treated uniformly via lognormal)
