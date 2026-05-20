# Stage 2 OpenWhisk Prototype

这个目录是 proposal 第二阶段的最小原型，目标不是马上做资源优化，而是先打通：

1. OpenWhisk action 调用链路
2. 固定 DAG workflow runner
3. trace collection
4. workflow entry forecasting
5. DAG arrival propagation

## 当前阶段目标

第二阶段只回答一个问题：

> 给定 workflow entry 的未来请求预测，如何沿固定 DAG 推导每个 stage 的未来到达分布？

我们先假设：

1. DAG 是固定的。
2. 每次 workflow invocation 都会执行所有节点。
3. edge selectivity = 1。
4. 暂不做 prewarm、resource sizing、batching。

## 目录结构

```text
stage2-openwhisk-prototype/
  actions/
    noop.py                  # 最小 OpenWhisk Python action
  configs/
    linear3.yaml             # 线性 3-stage workflow
    parallel_join4.yaml      # parallel + join workflow
  runner/
    openwhisk_client.py      # OpenWhisk REST client
    workflow.py              # DAG 配置解析和拓扑逻辑
    trace_store.py           # CSV trace 写入
    run_workflow.py          # 执行 workflow invocation
    forecast_entry.py        # entry arrival EWMA forecast
    propagate.py             # DAG delay-kernel propagation
  scripts/
    deploy_actions.sh        # 在 OpenWhisk 中部署测试 actions
  data/
    traces.csv               # 默认 trace 输出位置
```

## 0. 先验证 OpenWhisk CLI

在你的 master VM 上执行：

注意：OpenWhisk auth 的 secret 长度要足够。你当前环境里，24 字节 secret 会被识别为 `invalid auth`，实际可用的是更长的 36+ 字节 secret。后续实验统一使用你已经验证能 `wsk namespace list -i` 成功的 guest auth。

```bash
# 查看当前 wsk 配置
# 作用：确认 apihost 和 auth 已经设置到 guest namespace
wsk property get
```

```bash
# 查看 namespace
# 作用：确认 OpenWhisk API 和 auth 都已经可用
wsk namespace list -i
```

注意：`wsk --version` 在你当前 CLI 版本里不是有效参数，可以用 `wsk` 或 `wsk --help` 看 CLI 是否可用。

## 1. 拷贝或同步本目录到 master VM

如果你在 Windows 宿主机编辑代码，需要把 `stage2-openwhisk-prototype` 同步到 master VM，例如放到：

```text
~/stage2-openwhisk-prototype
```

后续命令默认你在 master VM 的该目录中执行。

## 2. 安装 Python 依赖

```bash
# 进入原型目录
# 作用：后续所有脚本都从这个目录运行
cd ~/stage2-openwhisk-prototype
```

```bash
# 安装 Python 依赖
# 作用：安装 runner 所需的 HTTP、YAML、数据处理库
python3 -m pip install -r requirements.txt
```

## 3. 部署 OpenWhisk actions

```bash
# 给脚本增加可执行权限
# 作用：允许直接运行部署脚本
chmod +x scripts/deploy_actions.sh
```

```bash
# 部署测试 actions
# 作用：创建/更新 wf_decode、wf_resize、wf_classify、wf_detect、wf_caption、wf_merge
./scripts/deploy_actions.sh
```

## 4. 跑通一条 linear workflow

```bash
# 执行 20 次 linear3 workflow
# 作用：生成第一批 trace 数据
python3 -m runner.run_workflow \
  --workflow configs/linear3.yaml \
  --count 20 \
  --apihost https://192.168.137.128:31001 \
  --auth "$(wsk property get --auth | awk '{print $3}')"
```

如果 `wsk property get --auth` 输出格式与你环境不同，可以手动传：

```bash
python3 -m runner.run_workflow \
  --workflow configs/linear3.yaml \
  --count 20 \
  --apihost https://192.168.137.128:31001 \
  --auth "你的-uuid:你的-secret"
```

## 5. 跑通 parallel + join workflow

```bash
# 执行 20 次 parallel_join4 workflow
# 作用：验证并行分支和 join 逻辑
python3 -m runner.run_workflow \
  --workflow configs/parallel_join4.yaml \
  --count 20 \
  --apihost https://192.168.137.128:31001 \
  --auth "你的-uuid:你的-secret"
```

## 6. 生成 entry forecast

```bash
# 基于 trace 聚合 entry arrival，并生成未来 12 个窗口预测
# 作用：得到 workflow entry arrival forecast
python3 -m runner.forecast_entry \
  --trace data/traces.csv \
  --workflow linear3 \
  --window-sec 5 \
  --horizon 12 \
  --method ewma \
  --out data/entry_forecast_linear3.csv
```

forecast 输出包含 `p50_count`、`p90_count`、`p95_count`、`p99_count`，以及对应的 `ceil_*` 和 `alloc_*` 整数列。

1. `ceil_*` 是纯数学向上取整。
2. `alloc_*` 会先应用 `--activation-threshold`，低于阈值则输出 0，更适合后续 prewarm replicas / resource allocation 决策。

## 7. 做 DAG propagation

```bash
# 使用 entry forecast 和历史 delay kernel 推导 stage arrival
# 作用：得到每个 stage 的未来到达预测
python3 -m runner.propagate \
  --trace data/traces.csv \
  --workflow configs/linear3.yaml \
  --entry-forecast data/entry_forecast_linear3.csv \
  --window-sec 5 \
  --out data/stage_arrival_forecast_linear3.csv
```

## 下一步

当这一版跑通后，我们再进入：

1. 更真实的 workload generator
2. activation record collector
3. cold/warm 标记
4. quantile forecasting
5. DAG propagation 误差评估

## 8. 生成更真实的 workload

除了 `runner.run_workflow` 的固定间隔调用，也可以用 `runner.run_workload` 生成不同入口到达模式。

```bash
# 运行 burst workload
# 作用：模拟周期性突发请求，用于观察 entry forecast 与 DAG propagation 在突发负载下的表现
python3 -m runner.run_workload \
  --workflow configs/linear3.yaml \
  --count 60 \
  --pattern burst \
  --base-interval-ms 500 \
  --burst-every 15 \
  --burst-size 5 \
  --burst-interval-ms 50 \
  --idle-interval-ms 2500 \
  --apihost https://192.168.137.128:31001 \
  --auth "UUID:SECRET"
```

支持的 workload pattern：

1. `constant`：固定间隔。
2. `burst`：短时间突发 + 较长空闲。
3. `periodic`：周期性快慢变化。
4. `sparse`：稀疏调用，随机插入长空闲。
5. `poisson`：指数间隔，近似泊松到达。

`runner.run_workload` 会同时写：

1. `data/traces.csv`：stage-level trace。
2. `data/workload_schedule.csv`：entry workload schedule 与每次 workflow 状态。

## 9. 使用 burst-aware entry predictor

`runner.forecast_entry` 支持两种 baseline：

1. `ewma`：指数加权移动平均 baseline。
2. `burst-aware`：基于最近窗口 `recent_max / zero_ratio / nonzero_mean` 的状态感知 baseline。

```bash
# 使用 burst-aware 方法生成 entry forecast
# 作用：在突发 workload 下给出比 EWMA 更保守的高分位预测
python3 -m runner.forecast_entry \
  --trace data/traces_burst_linear3.csv \
  --workflow linear3 \
  --window-sec 1 \
  --horizon 12 \
  --method burst-aware \
  --history-window 30 \
  --burst-threshold 2 \
  --idle-zero-ratio 0.8 \
  --activation-threshold 0.1 \
  --out data/entry_forecast_burst_linear3_1s_burstaware.csv
```

`burst-aware` 仍然是 baseline，不是最终论文方法。后续会继续实现 quantile regression 与 conformal calibration。
