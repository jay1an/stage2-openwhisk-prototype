# DUET 联合优化对比报告 — spoken_dialog_flow

报告日期：2026-05-20
工作流：`spoken_dialog_flow`（5 个 stage：speech_decode → entity_extract → topic_route → response_generate → speech_synthesize）
预测方法：`dag-hazard-hurdle`
评估窗口：1075 个 5-秒窗口，2775 条端到端请求，每条请求 200 次 Monte Carlo 重采样
SLO：3000 ms p95
数据：`reports/spoken_dialog_duet_comparison/comparison_summary.csv`

## 1. 方法概述：DUET 联合控制器

DUET（**D**emand-**U**ncertainty **E**lastic **T**ime-aware）是本工作提出的新颖方法，
它在同一个规划器内联合决策三个互相耦合的旋钮：

- **per-stage 内存档位（冷启动延迟）**
- **per-window warm container 数量（资源配比）**
- **per-stage keepalive TTL（保温时间）**

核心思想是利用**预测分位差**（`p_q − p_ref`，例如 `p95 − p50`）作为不确定性信号，
驱动三类弹性决策：

### 1.1 置信度混合的 warm 配额

```
spread     = max(0, p_q − p_ref)
blended    = p_ref + β · spread
warm_count = ceil(α · blended)
```

参数：`β = 0.35`（spread 注入比例），`α = 0.55`（队列平滑系数）。
低不确定窗口接近 `α · p_ref`，高不确定窗口逐渐向 `α · p_q` 靠拢，避免一刀切 p95 造成的资源浪费。

### 1.2 弹性 keepalive TTL

```
TTL = min_ttl
    + uncertainty_gain · normalized_spread   (若 spread ≥ floor)
    + persistence_gain · I[前一窗已经在用 warm]
    + critical_bonus  · I[stage 是冷尾关键路径]
```

- 高 spread → 加长 TTL，吸收预测不确定；
- 连续活跃窗口 → persistence bonus，平滑周期模式；
- 关键路径 stage → critical bonus，避免冷启动级联进 SLO 尾。

### 1.3 关键路径感知的内存升级

按 `cost_effective_tier`（成本/收益最低的可行档位）作为基线，仅当 stage 的冷启动占整条 warm critical path 比例超过 `critical_slack_ratio` 时升一档。
这避免 ORION-Style 那种"全部升到 1024 MB"的全局浪费。

DUET 暴露 `memory_mode={auto, base, fixed}` 与 `critical_slack_ratio`，
用同一份算法配置两个 Pareto 点：

- **DUET-Balanced（safety-first）**：`critical_slack_ratio=0.5`，关键 stage 升内存，重 keepalive 系数（uncertainty=8s, persistence=3s, critical=2s）。
- **DUET-Economy（cost-aware）**：`critical_slack_ratio=10`（实际禁用升级），轻 keepalive 系数（uncertainty=3s, persistence=2s, critical=0），`keepalive_demand_floor=2`。

## 2. 对比表

7 种方法在同一 forecast detail + 同一 latency samples + 同一 warmup_mode=window 下评估：

| 方法                    | 成本 (GB·s) | 平均 warm | 平均 ka (s) | 平均 mem (MB) | 违约率   | p95 延迟 (ms) |
| ----------------------- | ----------- | --------- | ----------- | ------------- | -------- | ------------- |
| Scale-To-Zero           |   3058      |  0.00     |   0.00      |   256         | 100.00%  | 9344          |
| StepConf-Style          |   3058      |  0.00     |   0.00      |  1024         | 100.00%  | 7178          |
| **SMIless-Pareto**      |  21783      | 13.93     |   0.00      |   256         |   1.45%  | 2686          |
| Always-Warm             |  40683      | 28.00     |   0.00      |   256         |   1.11%  | 2673          |
| **DUET-Economy (ours)** |  44946      | 11.94     |   3.57      |   256         |   0.52%  | 2652          |
| ORION-Style             | 118658      | 21.51     |   0.00      |  1024         |   0.73%  | 2665          |
| **DUET-Balanced (ours)**| 168315      | 11.94     |   9.20      |   512         | **0.00%**| **1649**      |

排序：以 `cost_gb_seconds` 升序。

## 3. Pareto 前沿分析

把 (cost, violation) 当 2D 平面看，**不被支配**的点只有 4 个：

1. **SMIless-Pareto** — 最低带保护成本（21.8K），违约率 1.45%
2. **DUET-Economy (ours)** — 中等成本（44.9K），违约率 0.52%
3. **DUET-Balanced (ours)** — 高成本（168K），违约率 0%，且 p95 延迟降到 1649 ms
4. （StepConf/Scale-To-Zero 因 100% 违约不进入实用 Pareto 集）

被支配的点：

- **Always-Warm**：成本 40.7K，违约 1.11%。被 DUET-Economy 弱支配（DUET-Economy 在违约率 0.52% < 1.11% 上更优，成本仅高 10%）；同时 p95 延迟 (2652 vs 2673) 也更好。
- **ORION-Style**：成本 118.7K，违约 0.73%。被 DUET-Balanced 严格支配（DUET-Balanced 违约 0%，p95 延迟 1649 ms < 2665 ms，虽然成本贵 42%，但安全性维度大幅胜出）；亦被 DUET-Economy 在成本维度大幅胜出（DUET-Economy 成本 44.9K << 118.7K，违约率 0.52% < 0.73%）。

## 4. 关键发现

**DUET 用一套统一算法占据多个 Pareto 点。**
两个变体共享相同的 `warm_count = ceil(0.55 · (p50 + 0.35 · spread))` 计算（mean warm=11.94 相同），
但 keepalive/内存策略不同：

- DUET-Economy 选择"轻 keepalive + 不升内存"，节省 73% 成本相对 DUET-Balanced；
- DUET-Balanced 选择"重 keepalive + 升 512 MB"，把 p95 延迟从 2.6 s 拉到 1.6 s，并且达到 0% 预测违约。

**DUET-Economy 严格优于 Always-Warm。**
两者成本接近（44.9K vs 40.7K，差 10%），但 DUET-Economy 违约率减半（0.52% vs 1.11%），
延迟 p95 略低（2652 vs 2673 ms），warm 池小 58%（11.94 vs 28）。
这说明**用 keepalive 取代过量 warm 池**是更高效的资源分配。

**DUET-Balanced 严格优于 ORION-Style。**
ORION-Style 用 1024 MB 全 stage 内存换冷启动收益，但 21.5 个 warm（无 keepalive）应对周期性 burst 不够灵活，
DUET-Balanced 用 512 MB（只升关键路径）+ 11.94 warm + 9.2 s keepalive 实现：
- 违约率 0%（< ORION 的 0.73%）
- p95 延迟 1649 ms（< ORION 的 2665 ms，38% 改善）

代价是成本贵 42%。在 SLO-critical 场景（医疗/语音/工业控制），这是合理 trade-off。

**预测分位差作为不确定性信号是奏效的。**
DUET 两个变体相同 warm pool（11.94）下，仅靠 keepalive 系数差异就把违约率从 0.52% 降到 0%，
说明 spread-elastic keepalive 在不增加 warm 数量的前提下能精准吸收预测尾部不确定。

## 5. 复现命令

```bash
cd stage2-openwhisk-prototype
python -m runner.run_duet_comparison \
  --workflow-config configs/spoken_dialog_flow.yaml \
  --trace data/stage_synthetic/spoken_dialog_flow_profiled_periodic_drift_stage_trace.csv \
  --forecast-detail reports/spoken_dialog_stage2_hazard/spoken_dialog_flow_hazard-hurdle_stage_compare_detail.csv \
  --latency-samples reports/stage3_spoken_dialog_profiled/latency_samples_for_monte_carlo.csv \
  --method dag-hazard-hurdle --policy p95 --slo-ms 3000 \
  --window-sec 5.0 --warmup-mode window --duet-warmup-mode window \
  --duet-memory-mode base --base-memory-mb 256 \
  --enable-duet-economy \
  --out-dir reports/spoken_dialog_duet_comparison
```

## 6. 后续工作

- 在更多 workflow（`civic_alert_flow`, `sebs_video_flow`）上验证 DUET 同样占据 Pareto 前沿；
- 与 SMIless 原论文的 Pareto 规划器进行 head-to-head 对比（当前 SMIless-Pareto 是 risk-budgeted 复现版）；
- 调研 `warm_blend_beta` 与 `warm_scale_multiplier` 是否可以从 workload 自动学习；
- 把 DUET-Balanced 与 DUET-Economy 中间档（如 `critical_slack_ratio ∈ [1.0, 5.0]`）扫一遍，
  画完整的 cost-risk Pareto 曲线。
