# Stage-2 / Stage-3 / Stage-5 架构重构说明

## 重构动机

旧路径里，Stage-2 同时承担了两类职责：一类是预测 workflow entry 的到达量，另一类是把 entry 预测沿 DAG 传播成每个 stage 的窗口级预测。传播使用的是从历史 trace 估计出的无条件 delay kernel，也就是把 warm dispatch 和 cold dispatch 的延迟混在一起求经验分布。这样会把控制状态的偏差带进预测：训练 trace 偏冷时，下游 stage 的预测会被推迟；训练 trace 偏热时，下游 stage 的预测会被提前。Stage-5 再基于这个已经混合的 stage forecast 做预热和资源规划，容易出现少备 warm 或多备 warm 的级联误差。

新架构把三件事拆开：Stage-2 只回答“entry 会来多少”；Stage-3 从实测 trace 建立“workflow 时序如何展开”；Stage-5 根据自己的上一窗 warm 决策选择 warm 或 cold kernel，再把 entry forecast 展开到 stage 级别。这让新的 entry 预测方法可以直接接入控制器，不需要重复实现 DAG 传播。

## 新数据流

```text
+------------------+        +----------------------+
| Stage-2 Forecast |        | Stage-3 Latency Data |
| entry_forecast   |        | delay_kernel         |
+---------+--------+        +----------+-----------+
          |                            |
          +------------+---------------+
                       |
                       v
          +----------------------------+
          | Stage-5 Propagator         |
          | warm/cold state selection  |
          +------------+---------------+
                       |
                       v
          +----------------------------+
          | Stage-5 Planner            |
          | warm, keepalive, memory    |
          +----------------------------+
```

## 新产物 schema

| 产物 | 字段 |
| --- | --- |
| `entry_forecast.csv` | `workflow_name, method, target_window, policy, forecast_count, allocated_count` |
| `delay_kernel.csv` | `workflow_name, stage_name, prev_state, offset_windows, probability` |
| propagated stage forecast | `workflow_name, method, stage_name, target_window, policy, actual_count, forecast_count, allocated_count` |

`prev_state` 有三类取值：`warm` 表示上一控制窗该 stage 有 warm 状态，`cold` 表示没有 warm 状态，`any` 保留旧的无条件边际分布。每个 `(workflow_name, stage_name, prev_state)` 下，`probability` 会归一化到 1。

## state-conditional kernel 的作用

假设 entry 在窗口 100 的 p95 预测是 20。某个下游 stage 的 warm kernel 为 `offset=0:0.9, offset=1:0.1`，cold kernel 为 `offset=0:0.2, offset=1:0.8`。如果上一窗该 stage 已经 warm，Stage-5 在窗口 100 会主要把 20 个请求落到当前窗口；如果上一窗是 cold，则大部分请求会被传播到窗口 101。旧的 `any` kernel 可能平均成 `offset=0:0.55, offset=1:0.45`，在两种状态下都不准确。新路径让 Stage-5 的控制状态直接影响传播，减少“预测时序”和“执行状态”不一致造成的偏差。

## 兼容方式

旧的 `stage_compare_detail.csv` 仍然保留，`duet_planner.py`、`paper_baselines.py`、`risk_budgeted_pareto_planner.py` 和 `run_duet_comparison.py` 继续支持 `--forecast-detail`。新路径使用 `--entry-forecast` 与 `--delay-kernel`，并要求二者同时出现；如果同时传入旧路径和新路径参数，程序会直接报错，避免混用输入语义。

## 后续扩展

在线选择器可以只输出 entry-level 的多策略 forecast，然后由统一 propagator 传播到所有 stage。SRB-Ensemble、LSTM、LightGBM 等方法也只需要遵守 `entry_forecast.csv` schema。未来若 Stage-3 提供更细粒度的 kernel，例如按 memory tier、batch size 或 DAG 边区分，Stage-5 可以在同一 propagator 接口下扩展状态选择，而不必改动 Stage-2 预测器。
