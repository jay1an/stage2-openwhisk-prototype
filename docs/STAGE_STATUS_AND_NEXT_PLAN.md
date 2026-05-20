# 各阶段当前情况与后续安排

更新时间：2026-05-13

本文档按 proposal 的系统主线整理当前项目进度。根目录仍保留历史名称
`stage2-openwhisk-prototype`，但代码已经按阶段拆到 `runner/stage*_*/`。

## 总体判断

当前项目已经从单纯 Stage 2 原型扩展成一个从 forecasting 到 latency profiling、
risk estimation、offline control planning 的研究原型。真正成熟的是 Stage 2 的离线
Workflow Forecastor 切片；Stage 3 和 Stage 4 已经有可运行的最小原型；Stage 5/6
目前仍是离线规划和资源画像探索，尚未进入真实在线控制。

必须保持的论文表述边界：

- 不要把当前系统写成完整闭环 controller。
- 不要声称 p95 calibration 在所有 workload 上已经解决。
- 不要把 synthetic-stage trace 写成真实 OpenWhisk replay evidence。
- 不要把 `allocated_count` 写成 OpenWhisk 已经创建的真实 container 数；它是窗口级
  capacity target / warm-capacity budget。

## 当前新增核心目标：Risk-Budgeted Pareto Planner

截至 2026-05-13，后续 Stage 4/5/6 的主线目标确定为：

```text
Risk-Budgeted Pareto Planner
```

它是本文后续最重要的系统优化贡献，不应退化成简单贪心或单独 memory tuning。

### 优化对象

一个候选 plan 包含：

```text
warm_count(stage, window)
keepalive_ttl(stage) / warmup refresh interval
memory_mb(stage)
```

第一版把 `memory_mb` 作为 CPU-correlated resource knob。不要一开始修改
OpenWhisk internals 做独立 CPU 控制。

这里的 `window` 是未来控制时间桶，不是 pod/container。若控制窗口为 5 秒，则：

```text
window = t
=> plan applies to [t * 5s, (t + 1) * 5s)
```

例如：

```text
decode, window=100, warm_count=1, keepalive=30, memory=512
```

表示在第 100 个 5 秒控制窗口内，目标是让 `decode` 至少有 1 个可用 warm
capacity，并用 keepalive/warmup refresh 近似维持 30 秒热状态。

预热 pod/action 的依据不是“所有 stage 都预热”，而是：

1. Stage 2 给出未来各 stage/window 的 demand distribution。
2. Stage 3 给出 warm/cold latency 和 memory-scaling model。
3. Stage 4 对候选 plan 估计 `P(workflow_latency > SLO | plan)`。
4. Risk-Budgeted Pareto Planner 选择满足 SLO risk 的最低 cost plan。
5. Stage 5 只对被选中 plan 中 `warm_count > 0` 的 stage 执行 warmup/refresh。

换言之，预热由最终 chosen plan 驱动，不由单独的 forecast count 或固定规则直接决定。

### 目标函数

```text
min cost(plan)

s.t.
  P(workflow_latency > SLO | plan) <= epsilon
```

其中：

```text
cost(plan) = C_exec + C_warm + C_reconfig
```

第一版 cost proxy：

```text
C_exec   = sum(memory_gb(stage) * action_duration_sec(stage))
C_warm   = sum(memory_gb(stage) * warm_idle_sec(stage))
C_reconf = lambda_reconf * number_of_memory_changes
```

### keepalive 的作用

keepalive 主要影响 cold probability 和 warm waste cost，不直接降低 action execution time。

近似关系：

```text
P(cold | tau) ~= P(next_interarrival_gap > tau)
```

因此：

- 增大 keepalive：cold-like risk 下降，warm idle cost 上升。
- 缩短 keepalive：warm waste cost 下降，cold-like risk 上升。
- sparse workload 下 keepalive 可能不划算。
- periodic / continuous moderate workload 下 keepalive 往往更有价值。

### 算法方向

不要把主算法设计成只保留一个当前解的普通 greedy。推荐：

```text
Risk-aware Pareto Beam Search + Slack-guided Action Expansion
```

算法框架：

1. 从最低成本 baseline plan 开始。
2. 生成 stage-prioritized one-step actions：
   - `warm_count +1/-1`
   - `keepalive` 上调/下调一个档位
   - `memory` 上调/下调一个档位
3. 用 Stage 4 Risk Estimator 评价每个 candidate plan：
   - `risk(plan) = P(workflow_latency > SLO | plan)`
   - `cost(plan)`
4. 用 `(cost, risk)` Pareto pruning 删除被支配方案。
5. 保留 beam 中若干非支配候选，避免陷入简单贪心局部最优。
6. 一旦存在 `risk <= epsilon` 的可行方案，选择最低 cost 的方案。
7. 做 compression pass：
   - 尝试降低 memory、减少 warm_count、缩短 keepalive。
   - 如果 risk 仍满足约束，则保留降成本动作。

### Stage 分工

- Stage 2：提供 demand distribution / desired capacity hints。
- Stage 3：提供 latency model 和 memory-scaling model。
- Stage 4：给定一个 plan，估计 workflow SLO risk。
- Stage 5：执行 warm_count / keepalive 相关动作。
- Stage 6：维护 memory/resource plan，并调用 Stage 4 评估候选方案。

### 在线决策原则

在线系统不应该实时试跑所有 memory tier。正确路径是：

1. Stage 3 做少量离线 profiling，拟合 per-stage memory-scaling model，例如：

```text
T_exec_i(m) = alpha_i + beta_i / (m / m0)^gamma_i
```

2. Stage 6 在线使用该模型估计 candidate memory tier 的 action duration。
3. Stage 4 用估计 latency distribution 计算 workflow risk。
4. Planner 选择满足 SLO risk 约束的最低成本 plan。

### 后续实现文件规范

后续实现按 stage 放入对应目录，避免再次出现 runner 根目录文件混乱：

```text
runner/stage5_control/control_plan.py
runner/stage5_control/cost_model.py
runner/stage5_control/risk_budgeted_pareto_planner.py
runner/stage3_latency/memory_scaling_model.py
runner/stage5_control/warmup_daemon.py
```

替换掉的临时脚本和缓存文件可以删除；原始 traces、最终报告、论文证据包不要未经确认删除。

## Stage 1：Problem Scope and Literature

### 当前情况

proposal 已经明确了研究范围：

- fixed-DAG serverless workflow。
- 第一版每次 workflow invocation 执行所有节点，暂不处理条件分支。
- CPU-only OpenWhisk + Kubernetes。
- 优化目标是 workflow-level SLO risk under cost。
- 对比对象包括 ORION、StepConf、AQUATOPE、SeBS-Flow、Azure trace、cold-start
  studies 和 serverless workload characterization。

当前不足：

- 相关工作仍然是初稿级别。
- 还没有形成论文中“本文和已有系统的差异表”。
- 对 StepConf/AQUATOPE/ORION 的可复现实验边界还没有最终确定。

### 下一步安排

1. 整理 related work 表格：
   - 单函数 sizing / cold start。
   - workflow DAG optimizer。
   - uncertainty-aware resource management。
   - serverless workload characterization。
2. 写清本文定位：
   - 不是只做 prediction。
   - 不是只做 prewarming。
   - 不是只做 resource sizing。
   - 核心是 uncertainty-calibrated forecasting + warm/cold latency + workflow risk +
     joint control。
3. 确定评测问题：
   - entry+DAG 是否比 per-stage independent 更稳？
   - conformal/risk-budget 是否改善 quantile calibration？
   - warm/cold latency 是否主导 workflow tail？
   - risk-aware warm plan 是否比 fixed warm plan 更省成本？

## Stage 2：Workflow Forecastor

### 当前情况

Stage 2 是目前最完整的部分。代码在：

```text
runner/stage2_forecastor/
```

已经完成：

- Azure-derived entry trace 构建。
- `sebs_video` 作为当前主 workflow。
- workflow entry forecasting。
- empirical DAG delay-kernel propagation。
- per-stage independent baseline。
- heuristic baselines：EWMA、hazard/hurdle、burst-localized。
- proper probabilistic baseline：Quantile LightGBM。
- LSTM entry/stage rolling forecast。
- rolling conformal calibration。
- online adaptive expert selector。
- p95 risk-budget fallback。
- Stage 2 final report。

当前主报告：

```text
reports/stage2_workflow_forecastor_final/
```

关键结论：

- 在 `sebs_video` Azure periodic/drift trace 上，online risk-budget selector 的 p95
  empirical quantile coverage 约为 0.957，demand coverage 约为 0.990。
- rolling conformal 更稳但成本更高。
- risk-budget selector 比 always-conformal 更省 replica-seconds。
- raw LSTM 和 raw LightGBM 不能直接声称校准完成。
- sparse/bursty/mixed_drift 目前不能作为主效果实验。

重要限制：

- 当前 Stage 2 主证据仍主要是 offline / synthetic-stage trace。
- `allocated_count` 是窗口级 capacity decision，不是实际 OpenWhisk container count。
- p95 calibration 只在选定 trace 和 warm-up-excluded 设置下表现较好。

### 下一步安排

优先级 1：补真实 replay 数据。

1. 在 OpenWhisk 上确认四个 Level-2 workflow 均可单次运行：
   - `sebs_trip_booking`
   - `sebs_video`
   - `sebs_map_reduce`
   - `sebs_ml`
2. 用 Azure 2021 选定 `(app, func)` schedule replay 到这些 workflow。
3. 采集真实 stage-level traces。
4. 复跑 Stage 2 entry+DAG vs per-stage independent 对比。

优先级 2：扩展 workload regime。

1. continuous/periodic-drift 保持主线。
2. sparse/bursty/mixed 先作为 robustness / limitation。
3. 如果要把 sparse/bursty 写成正面结果，需要新增更强模型：
   - active probability model。
   - hurdle/zero-inflated quantile model。
   - regime-switching selector。

优先级 3：让 Stage 2 输出更适合 Stage 4/5。

1. 明确区分：
   - demand forecast。
   - allocation decision。
   - desired warm capacity。
2. 输出 candidate warm targets：
   - `desired_warm_count(stage, window)`。
   - `confidence/risk_budget`。
   - `forecast_source/expert`。

## Stage 3：Latency Profiler

### 当前情况

代码在：

```text
runner/stage3_latency/
```

已经完成：

- coarse latency profiler。
- warm/cold-like 分类。
- 粗粒度分解：

```text
dispatch_latency_ms = platform_overhead_ms + action_duration_ms
```

- stage-level latency profile。
- workflow-level latency profile。
- Monte Carlo latency sample pool。
- augmented cold-like sample pool。

当前报告：

```text
reports/stage3_latency_profile/
reports/stage3_latency_augmented_cold_sebs_video_periodic_drift/
```

关键观察：

- 当前 pilot traces 显示 action duration 相对稳定。
- long-tail latency 主要来自 platform overhead / cold-like path。
- 真实 OpenWhisk pilot traces 主要覆盖 `sebs_trip_booking`，规模较小。
- `sebs_video` 主 trace 仍有 synthetic-stage 成分。

重要限制：

- 还没有完成 proposal 中完整分解：

```text
Lqueue + Lschedule + Lpod + Ldep + Lruntime + Lload + Lexec + Lnet
```

- `cold_like` 仍是 coarse label，不是平台内部精确 cold-start ground truth。
- augmented cold-like samples 只能用于 sensitivity / plumbing，不能替代真实测量。

### 下一步安排

优先级 1：补真实 latency trace。

1. 对四个 workflow 做真实 OpenWhisk replay。
2. 每个 stage 至少采集足够 warm 和 cold-like 样本。
3. 记录每次实验的 memory tier、OpenWhisk auth/API host、schedule、trace。

优先级 2：扩展 latency profile。

1. 按以下维度分组：
   - workflow。
   - stage。
   - warm/cold_like。
   - memory tier。
2. 输出：
   - p50/p90/p95/p99。
   - platform overhead quantiles。
   - action duration quantiles。
3. 形成 Stage 6 需要的：

```text
P(latency | stage, class, memory_mb)
```

优先级 3：改进 cold-like 判定。

1. 当前可继续用 threshold。
2. 后续应结合：
   - container id reuse。
   - inter-arrival idle time。
   - OpenWhisk activation metadata。
   - warmup invocation marker。

## Stage 4：Workflow SLO Risk Estimator

### 当前情况

代码在：

```text
runner/stage4_risk/
```

已经完成：

- offline Monte Carlo SLO-risk estimator。
- 输入 Stage 2 allocation detail 和 Stage 3 latency samples。
- 沿 workflow DAG 模拟 stage completion。
- 输出 workflow-level SLO violation probability。
- 支持 p90/p95 策略比较。

当前报告：

```text
reports/stage4_slo_risk_comparison/
reports/stage4_slo_risk_p90/
reports/stage4_slo_risk_p95/
```

当前示例：

- workflow：`sebs_video`
- SLO：2500 ms
- held-out fold：3
- p90 policy predicted violation probability：约 0.089
- p95 policy predicted violation probability：约 0.069
- observed violation rate：约 0.039

重要解释：

- 当前 Stage 4 不知道每个 stage 的真实 warm/cold 状态。
- 当前用 allocation deficit 作为 cold-like probability 的代理：

```text
p_cold ~= max(0, actual_count - allocated_count) / actual_count
```

- `allocated_count` 是窗口级 capacity target，不是真实 container count。

重要限制：

- 没有建模 queueing。
- 没有建模 keep-alive state transition。
- 没有建模 current warm replicas / in-flight requests。
- 没有建模 Stage 5 controller feedback。
- 当前风险估计偏保守，需要 calibration。

### 下一步安排

优先级 1：把 Stage 4 从 allocation-proxy 改成 warm-plan evaluator。

未来 Stage 4 应该显式输入 candidate warm plan：

```text
warm_count(stage, window)
keepalive_ttl(stage)
memory_mb(stage)
```

然后输出：

```text
P(workflow_latency > SLO | plan)
cost(plan)
marginal risk reduction by stage
```

优先级 2：引入状态变量。

需要建模：

- current warm replicas。
- in-flight requests。
- queue length / occupied containers。
- keep-alive remaining time。
- previous warmup success。

优先级 3：风险校准。

1. 重新跑最新 estimator，生成：
   - `risk_bin_table.csv`
   - `risk_calibration_summary.csv`
   - `stage_risk_contribution.csv`
2. 评估 Brier score / ECE。
3. 对不同 SLO 和 residual cold probability 做 sensitivity。

## Stage 5：Fast Warm Manager

### 当前情况

代码在：

```text
runner/stage5_control/
```

当前只有 offline joint-control planner：

```text
plan_joint_control.py
```

它可以根据 forecast detail、latency samples 和 workflow SLO 生成离线控制建议，但还没有
真正调用 OpenWhisk 执行 warmup。

当前定位：

- Stage 5 还不是 live controller。
- 它目前是 warm/prewarm/keep-alive planning prototype。

### 下一步安排

优先级 1：明确第一版 actuator。

第一版不要改 OpenWhisk internals，采用：

```text
controller-issued warmup invocations
```

即外部 daemon 周期性调用 action 来维持 warm。

优先级 2：实现最小闭环。

1. 输入 Stage 2/4 生成的 target：

```text
desired_warm_count(stage, next_window)
```

2. 对每个 stage 维护 warmup schedule。
3. 记录：
   - warmup invocation time。
   - target action。
   - success/failure。
   - observed latency。
   - possible container reuse。

优先级 3：实验比较。

至少比较：

- no warmup。
- fixed keep-alive warmup。
- Stage 2 allocation direct warmup。
- Stage 4 risk-aware warm plan。

## Stage 6：Slow Resource Configurator

### 当前情况

代码在：

```text
runner/stage6_resource/
```

已有：

- `benchmark_cpu_scaling.py`
- `summarize_memory_sweep.py`

当前定位：

- 资源配置仍在探索阶段。
- 第一版应使用 OpenWhisk action memory tier。
- CPU 暂时作为 memory-correlated proxy。

### 下一步安排

优先级 1：memory sweep。

对主要 workflow 和 stage 跑：

```text
128MB / 256MB / 512MB / 1024MB / 2048MB
```

输出：

- action duration p50/p95。
- platform overhead p50/p95。
- workflow latency p95。
- cost proxy。

优先级 2：接入 Stage 4。

让 Stage 4 可以评估：

```text
如果 stage i 从 256MB 调到 512MB，SLO risk 降多少，成本升多少？
```

优先级 3：联合优化。

先做离散候选搜索：

```text
warm_count in {0,1,2,4}
keepalive in {5s,15s,30s,60s}
memory in {128,256,512,1024}
```

暂不引入独立 CPU control 和 OpenWhisk internal 修改。

## 近期 4 周安排

### 第 1 周：整理和复现实验基线

- 固定代码结构和命令路径。
- 跑通 Stage 2 final report regeneration。
- 跑通 Stage 3 latency profile regeneration。
- 跑通 Stage 4 risk estimator regeneration。
- 建立统一 experiment manifest。

### 第 2 周：真实 OpenWhisk replay

- 四个 workflow 单次 smoke test。
- Azure schedule replay 到 `sebs_video`。
- 采集真实 stage trace。
- 对比 synthetic-stage trace 与 real replay trace。

### 第 3 周：Stage 4 warm-plan evaluator

- 把 `allocated_count` proxy 改为显式 warm plan。
- 实现 warm_count candidate enumeration。
- 输出 stage-level marginal risk reduction。
- 生成 risk calibration table。

### 第 4 周：Stage 5 最小闭环

- 实现外部 warmup daemon prototype。
- 先只支持 fixed warm target。
- 再接 Stage 4 risk-aware target。
- 比较 no warmup / fixed warmup / risk-aware warmup。

## 当前最重要的论文口径

当前可以写：

- 我们提出一个系统框架，统一 forecasting、latency profiling、risk estimation 和 joint
  control。
- Stage 2 已有较完整离线原型。
- Stage 3/4 已经打通从 latency sample 到 workflow SLO risk 的诊断链路。
- 初步结果表明，校准层和 risk-budget selector 能改善 p95 coverage/cost trade-off。

当前不能写：

- 系统已经完整在线运行。
- Stage 4 已经知道真实 warm/cold 状态。
- `allocated_count` 就是实际 container 数。
- 所有 workload 上 p95 calibration 都已解决。
- 当前 synthetic-stage `sebs_video` 是最终真实平台证据。
