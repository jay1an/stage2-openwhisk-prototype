from pathlib import Path

import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


nb = nbf.v4.new_notebook()

cells = [
    md(
        """
# Azure Functions Invocation Trace 2021 分析与 Replay Schedule 生成

这个 notebook 用来分析 `AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt`。

目标：

1. 不一次性读入大文件，使用 `chunksize` 做 streaming analysis，避免 OOM。
2. 按 `(app, func)` 统计真实 Azure Functions invocation arrival pattern。
3. 为论文 background 提供 workload characterization：sparse、bursty、periodic、drift。
4. 从候选 `(app, func)` 中导出真实 arrival 的 replay schedule，后续 replay 到 SeBS-Flow 风格的 OpenWhisk workflow。

重要约定：

- Azure trace 不是 DAG trace，不要从 Azure 的 app/function 推断 workflow DAG。
- Azure trace 只作为 workflow entry arrival source。
- DAG 结构来自 `sebs_trip_booking`、`sebs_video`、`sebs_map_reduce`、`sebs_ml`。
- 当前文件中的 `end_timestamp` 和 `duration` 是 seconds，相对 trace 起点计时。
- 生成 OpenWhisk replay schedule 时会转换成 milliseconds。
"""
    ),
    code(
        r"""
from pathlib import Path
import os
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 180)

# 如果你希望手动指定 trace 文件，可以设置环境变量：
# export AZURE_TRACE_PATH=/path/to/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt
TRACE_CANDIDATES = [
    Path(os.environ.get("AZURE_TRACE_PATH", "")) if os.environ.get("AZURE_TRACE_PATH") else None,
    Path("AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"),
    Path("../AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"),
    Path("../../AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"),
]

AZURE_TRACE_PATH = None
for candidate in TRACE_CANDIDATES:
    if candidate is not None and candidate.exists():
        AZURE_TRACE_PATH = candidate.resolve()
        break

if AZURE_TRACE_PATH is None:
    raise FileNotFoundError("找不到 AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt，请设置 AZURE_TRACE_PATH。")

# 输出目录：放在当前工作目录的 data/azure_analysis 下。
OUTPUT_DIR = Path("data/azure_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 官方 schema。
SCHEMA = ["app", "func", "end_timestamp", "duration"]

print("AZURE_TRACE_PATH =", AZURE_TRACE_PATH)
print("file_size_MB     =", AZURE_TRACE_PATH.stat().st_size / 1024 / 1024)
print("OUTPUT_DIR       =", OUTPUT_DIR.resolve())
"""
    ),
    md(
        """
## 1. 快速检查文件格式

这里只读取前几行，确认 header、字段类型和时间单位。
"""
    ),
    code(
        r"""
# 只读前 5 行，避免加载整个大文件。
preview = pd.read_csv(AZURE_TRACE_PATH, nrows=5)
preview
"""
    ),
    code(
        r"""
# 检查 timestamp 范围。这里 nrows 可以调大一点，但仍然不是全量读入。
small_sample = pd.read_csv(AZURE_TRACE_PATH, nrows=100_000)
print(small_sample.dtypes)
print("end_timestamp min/max:", small_sample["end_timestamp"].min(), small_sample["end_timestamp"].max())
print("duration min/max:", small_sample["duration"].min(), small_sample["duration"].max())
print("unique apps:", small_sample["app"].nunique(), "unique funcs:", small_sample["func"].nunique())
"""
    ),
    md(
        """
## 2. Streaming Reader

下面的函数会按 chunk 读取 CSV。每个 chunk 会：

- 保留 `app`, `func`, `end_timestamp`, `duration`
- 转成 numeric
- 过滤非法 duration
- 计算 `start_timestamp = end_timestamp - duration`

注意：`start_timestamp` 单位仍然是 seconds。
"""
    ),
    code(
        r"""
def read_chunks(path, chunksize=1_000_000):
    # 按 chunk 读取 Azure trace，避免一次性读入内存。
    return pd.read_csv(path, chunksize=chunksize)


def normalize_chunk(df):
    # 清洗一个 chunk，并计算 start_timestamp，单位是 seconds。
    df = df[SCHEMA].copy()
    df["app"] = df["app"].astype(str)
    df["func"] = df["func"].astype(str)
    df["end_timestamp"] = pd.to_numeric(df["end_timestamp"], errors="coerce")
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    df = df.dropna(subset=["end_timestamp", "duration"])
    df = df[df["duration"] >= 0].copy()
    df["start_timestamp"] = df["end_timestamp"] - df["duration"]
    df = df[df["start_timestamp"] >= 0].copy()
    return df
"""
    ),
    md(
        """
## 3. 第一遍扫描：每个 `(app, func)` 的基础统计

这一步只聚合，不保存每条 invocation，所以内存开销可控。

输出：

- `azure2021_basic_function_summary.csv`

用途：

- 筛掉调用太少的函数。
- 找到 invocation count、duration、时间跨度合适的候选函数。
"""
    ),
    code(
        r"""
def scan_basic_stats(path, chunksize=1_000_000, max_chunks=None):
    parts = []
    total_rows = 0
    for chunk_idx, raw in enumerate(read_chunks(path, chunksize=chunksize), start=1):
        df = normalize_chunk(raw)
        total_rows += len(df)

        grouped = df.groupby(["app", "func"]).agg(
            invocations=("func", "size"),
            first_start_s=("start_timestamp", "min"),
            last_start_s=("start_timestamp", "max"),
            duration_sum_s=("duration", "sum"),
            duration_max_s=("duration", "max"),
        ).reset_index()
        parts.append(grouped)

        if chunk_idx % 5 == 0:
            print(f"chunk={chunk_idx}, processed_rows={total_rows:,}, partial_groups={len(grouped):,}")
        if max_chunks is not None and chunk_idx >= max_chunks:
            break

    combined = pd.concat(parts, ignore_index=True)
    summary = combined.groupby(["app", "func"]).agg(
        invocations=("invocations", "sum"),
        first_start_s=("first_start_s", "min"),
        last_start_s=("last_start_s", "max"),
        duration_sum_s=("duration_sum_s", "sum"),
        duration_max_s=("duration_max_s", "max"),
    ).reset_index()

    summary["span_minutes"] = np.maximum(1.0, (summary["last_start_s"] - summary["first_start_s"]) / 60.0)
    summary["mean_rate_per_min"] = summary["invocations"] / summary["span_minutes"]
    summary["mean_duration_ms"] = (summary["duration_sum_s"] / summary["invocations"]) * 1000.0
    summary["max_duration_ms"] = summary["duration_max_s"] * 1000.0
    summary = summary.sort_values("invocations", ascending=False).reset_index(drop=True)
    return summary


# 如果只是测试 notebook，可以先 max_chunks=2；正式分析用 None。
basic_summary = scan_basic_stats(AZURE_TRACE_PATH, chunksize=1_000_000, max_chunks=None)
basic_summary.to_csv(OUTPUT_DIR / "azure2021_basic_function_summary.csv", index=False)
print("function_pairs =", len(basic_summary))
basic_summary.head(20)
"""
    ),
    md(
        """
## 4. 选择候选 `(app, func)` 做细粒度 time-series 分析

我们不对所有函数都建 time series，先筛调用量足够的候选函数，避免内存和计算开销过大。
"""
    ),
    code(
        r"""
# 可以根据实际数据规模调整。
MIN_INVOCATIONS = 200
MAX_CANDIDATES = 3000

candidate_basic = (
    basic_summary[basic_summary["invocations"] >= MIN_INVOCATIONS]
    .sort_values("invocations", ascending=False)
    .head(MAX_CANDIDATES)
    .copy()
)

candidate_basic["key"] = candidate_basic["app"] + "::" + candidate_basic["func"]
candidate_keys = set(candidate_basic["key"])

print("candidate functions =", len(candidate_keys))
candidate_basic.head(20)
"""
    ),
    md(
        """
## 5. 第二遍扫描：候选函数的 minute-level arrival counts

这一步把 invocation-level trace 聚合到 1-minute bins。

为什么先用 1-minute bins？

- workload characterization 更稳定。
- 适合判断 sparse / bursty / periodic / drift。
- 后续 replay schedule 仍然会使用 invocation-level timestamps，不会丢掉真实到达时间。
"""
    ),
    code(
        r"""
def build_candidate_counts(path, candidate_keys, bin_seconds=60, chunksize=1_000_000, max_chunks=None):
    parts = []
    total_rows = 0
    for chunk_idx, raw in enumerate(read_chunks(path, chunksize=chunksize), start=1):
        df = normalize_chunk(raw)
        total_rows += len(df)
        df["key"] = df["app"] + "::" + df["func"]
        df = df[df["key"].isin(candidate_keys)].copy()
        if not df.empty:
            df["bin"] = np.floor(df["start_timestamp"] / bin_seconds).astype("int64")
            grouped = df.groupby(["key", "bin"]).size().reset_index(name="count")
            parts.append(grouped)

        if chunk_idx % 5 == 0:
            print(f"chunk={chunk_idx}, processed_rows={total_rows:,}, matched_rows={len(df):,}")
        if max_chunks is not None and chunk_idx >= max_chunks:
            break

    if not parts:
        return pd.DataFrame(columns=["key", "bin", "count"])
    out = pd.concat(parts, ignore_index=True)
    out = out.groupby(["key", "bin"], as_index=False)["count"].sum()
    return out


minute_counts = build_candidate_counts(AZURE_TRACE_PATH, candidate_keys, bin_seconds=60, chunksize=1_000_000, max_chunks=None)
minute_counts.to_csv(OUTPUT_DIR / "azure2021_candidate_minute_counts.csv", index=False)
print("minute_count rows =", len(minute_counts))
minute_counts.head()
"""
    ),
    md(
        """
## 6. Workload Characterization 指标

这里计算几类可以写进论文 background 的指标：

- `zero_ratio`: 空窗口比例，越高越 sparse。
- `burst_score`: `max_per_min / active_mean_per_min`，越高越 bursty。
- `periodic_score`: 1h / 6h / 12h / 24h autocorrelation 的最大值。
- `drift_score`: 前半段与后半段调用量差异。
- `cv_per_min`: coefficient of variation。
- `fano_per_min`: variance-to-mean ratio。
"""
    ),
    code(
        r"""
def safe_autocorr(values, lag):
    if len(values) <= lag + 2:
        return np.nan
    x = values[:-lag]
    y = values[lag:]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def characterize_key(group):
    key = group["key"].iloc[0]
    group = group.sort_values("bin")
    first_bin = int(group["bin"].min())
    last_bin = int(group["bin"].max())
    span_bins = max(1, last_bin - first_bin + 1)
    active_bins = int(group["bin"].nunique())
    total = int(group["count"].sum())

    series = pd.Series(0.0, index=np.arange(first_bin, last_bin + 1))
    series.loc[group["bin"].to_numpy()] = group["count"].to_numpy(dtype=float)
    values = series.to_numpy(dtype=float)
    active_values = values[values > 0]

    mean = float(values.mean()) if len(values) else 0.0
    active_mean = float(active_values.mean()) if len(active_values) else 0.0
    max_count = int(values.max()) if len(values) else 0

    first_half = float(values[: len(values) // 2].sum())
    second_half = float(values[len(values) // 2 :].sum())
    drift_score = abs(second_half - first_half) / max(1.0, first_half + second_half)

    lag_scores = {
        "autocorr_1h": safe_autocorr(values, 60),
        "autocorr_6h": safe_autocorr(values, 360),
        "autocorr_12h": safe_autocorr(values, 720),
        "autocorr_24h": safe_autocorr(values, 1440),
    }
    valid_autocorr = [v for v in lag_scores.values() if not np.isnan(v)]
    periodic_score = max(valid_autocorr) if valid_autocorr else np.nan

    return {
        "key": key,
        "total_invocations": total,
        "first_bin": first_bin,
        "last_bin": last_bin,
        "span_minutes": span_bins,
        "active_minutes": active_bins,
        "zero_ratio": 1.0 - active_bins / span_bins,
        "mean_per_min": mean,
        "active_mean_per_min": active_mean,
        "max_per_min": max_count,
        "p95_per_min_all": float(np.quantile(values, 0.95)) if len(values) else 0.0,
        "p99_per_min_all": float(np.quantile(values, 0.99)) if len(values) else 0.0,
        "p95_per_min_active": float(np.quantile(active_values, 0.95)) if len(active_values) else 0.0,
        "p99_per_min_active": float(np.quantile(active_values, 0.99)) if len(active_values) else 0.0,
        "cv_per_min": float(np.std(values) / mean) if mean > 0 else np.nan,
        "fano_per_min": float(np.var(values) / mean) if mean > 0 else np.nan,
        "burst_score": max_count / max(1e-9, active_mean),
        "periodic_score": periodic_score,
        "drift_score": drift_score,
        **lag_scores,
    }


metrics = pd.DataFrame([characterize_key(g) for _, g in minute_counts.groupby("key")])
metrics[["app", "func"]] = metrics["key"].str.split("::", n=1, expand=True)
metrics = metrics.merge(candidate_basic[["key", "mean_duration_ms", "max_duration_ms"]], on="key", how="left")
metrics.to_csv(OUTPUT_DIR / "azure2021_candidate_characterization.csv", index=False)
metrics.sort_values("total_invocations", ascending=False).head(20)
"""
    ),
    md(
        """
## 7. 按 workload 类型展示候选 trace

建议最终每类选 2 条：

- sparse
- bursty
- periodic
- mixed / drift

这样得到 8 条真实 Azure entry traces。
"""
    ),
    code(
        r"""
def show_candidates(df, sort_by, ascending=False, n=20, query=None):
    out = df.copy()
    if query:
        out = out.query(query).copy()
    cols = [
        "app", "func", "total_invocations", "span_minutes", "active_minutes",
        "zero_ratio", "mean_per_min", "active_mean_per_min", "max_per_min",
        "burst_score", "periodic_score", "drift_score", "mean_duration_ms"
    ]
    return out.sort_values(sort_by, ascending=ascending)[cols].head(n)


print("Sparse candidates")
display(show_candidates(metrics, "zero_ratio", ascending=False, query="total_invocations >= 200"))

print("Bursty candidates")
display(show_candidates(metrics, "burst_score", ascending=False, query="total_invocations >= 200 and active_minutes >= 10"))

print("Periodic candidates")
display(show_candidates(metrics, "periodic_score", ascending=False, query="total_invocations >= 500 and active_minutes >= 60"))

print("Drift / mixed candidates")
display(show_candidates(metrics, "drift_score", ascending=False, query="total_invocations >= 500"))
"""
    ),
    md(
        """
## 8. 手动选择用于 replay 的 `(app, func)`

从上面的候选表复制 `app` 和 `func`。

`workflow` 字段只是默认把这条 arrival trace 导出给哪个 workflow 使用；后续同一条 arrival trace 也可以 replay 到其他 workflow。
"""
    ),
    code(
        r"""
# TODO: 根据候选表手动填写。
# 每类建议先选 2 条，总共 8 条。
SELECTED_TRACES = [
    # {"label": "sparse_0", "app": "APP_ID", "func": "FUNC_ID", "workload_type": "sparse", "workflow": "sebs_trip_booking"},
    # {"label": "bursty_0", "app": "APP_ID", "func": "FUNC_ID", "workload_type": "bursty", "workflow": "sebs_video"},
    # {"label": "periodic_0", "app": "APP_ID", "func": "FUNC_ID", "workload_type": "periodic", "workflow": "sebs_map_reduce"},
    # {"label": "drift_0", "app": "APP_ID", "func": "FUNC_ID", "workload_type": "drift", "workflow": "sebs_ml"},
]

selected_df = pd.DataFrame(SELECTED_TRACES)
selected_df
"""
    ),
    md(
        """
## 9. 加载某个 `(app, func)` 的 invocation-level timestamps

这里会重新 streaming 扫描大文件，但只保留选中的 `(app, func)`。

输出 schedule 时会把 seconds 转成 milliseconds。
"""
    ),
    code(
        r"""
def load_invocations_for_key(path, app, func, chunksize=1_000_000, max_rows=None):
    parts = []
    total = 0
    app = str(app)
    func = str(func)
    for raw in read_chunks(path, chunksize=chunksize):
        df = normalize_chunk(raw)
        df = df[(df["app"] == app) & (df["func"] == func)].copy()
        if df.empty:
            continue
        parts.append(df)
        total += len(df)
        if max_rows is not None and total >= max_rows:
            break
    if not parts:
        return pd.DataFrame(columns=SCHEMA + ["start_timestamp"])
    out = pd.concat(parts, ignore_index=True).sort_values("start_timestamp").reset_index(drop=True)
    if max_rows is not None:
        out = out.head(max_rows).copy()
    return out


def make_replay_schedule(invocations, workflow_name, source_label, speedup=1.0, max_invocations=None, min_interarrival_ms=0):
    # 把真实 Azure invocation timestamps 转成 OpenWhisk replay schedule。
    df = invocations.sort_values("start_timestamp").copy()
    if max_invocations is not None:
        df = df.head(max_invocations).copy()
    if df.empty:
        return pd.DataFrame()

    source_start_s = float(df["start_timestamp"].min())
    offsets_ms = ((df["start_timestamp"] - source_start_s) * 1000.0 / max(1e-9, speedup)).round().astype("int64").to_numpy()

    # 可选保护：限制最小 inter-arrival，避免本地 OpenWhisk 被真实高峰直接打爆。
    if min_interarrival_ms and min_interarrival_ms > 0:
        adjusted = []
        prev = None
        for off in offsets_ms:
            if prev is None:
                current = int(off)
            else:
                current = max(int(off), prev + int(min_interarrival_ms))
            adjusted.append(current)
            prev = current
        offsets_ms = np.array(adjusted, dtype="int64")

    schedule = pd.DataFrame({
        "workflow_name": workflow_name,
        "index": np.arange(len(df), dtype=int),
        "target_offset_ms": offsets_ms,
        "source_label": source_label,
        "source_app": df["app"].astype(str).to_numpy(),
        "source_func": df["func"].astype(str).to_numpy(),
        "source_start_s": df["start_timestamp"].to_numpy(),
        "source_end_s": df["end_timestamp"].to_numpy(),
        "source_duration_ms": (df["duration"] * 1000.0).round().astype("int64").to_numpy(),
    })
    return schedule
"""
    ),
    md(
        """
## 10. 导出 Replay Schedule

参数说明：

- `SPEEDUP`: 时间压缩倍率。`1.0` 表示保持原始时间间隔；`10.0` 表示压缩 10 倍。
- `MAX_INVOCATIONS_PER_TRACE`: 每条 trace 最多导出多少 invocation。
- `MIN_INTERARRIVAL_MS`: 最小 arrival 间隔保护，避免本地 OpenWhisk 触发 rate limit。

正式实验中必须记录这些参数。
"""
    ),
    code(
        r"""
SPEEDUP = 1.0
MAX_INVOCATIONS_PER_TRACE = 300
MIN_INTERARRIVAL_MS = 0

schedule_paths = []
for item in SELECTED_TRACES:
    inv = load_invocations_for_key(
        AZURE_TRACE_PATH,
        item["app"],
        item["func"],
        max_rows=MAX_INVOCATIONS_PER_TRACE,
    )
    print(item["label"], "invocations loaded =", len(inv))

    schedule = make_replay_schedule(
        inv,
        workflow_name=item.get("workflow", "sebs_video"),
        source_label=item["label"],
        speedup=SPEEDUP,
        max_invocations=MAX_INVOCATIONS_PER_TRACE,
        min_interarrival_ms=MIN_INTERARRIVAL_MS,
    )

    out_path = OUTPUT_DIR / f"schedule_{item['label']}_{item.get('workflow', 'workflow')}.csv"
    schedule.to_csv(out_path, index=False)
    schedule_paths.append(out_path)
    display(schedule.head())
    print("wrote", out_path)

schedule_paths
"""
    ),
    md(
        """
## 11. 可视化：用于 background 的 workload characterization

这些图可以帮助写 background：生产 serverless workload 具有 sparse、bursty、periodic 和 drift 特征。
"""
    ),
    code(
        r"""
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    metrics["zero_ratio"].hist(ax=axes[0], bins=50)
    axes[0].set_title("Zero ratio")
    axes[0].set_xlabel("empty minute ratio")

    metrics["burst_score"].replace([np.inf, -np.inf], np.nan).dropna().clip(upper=50).hist(ax=axes[1], bins=50)
    axes[1].set_title("Burst score")
    axes[1].set_xlabel("max / active mean, clipped at 50")

    metrics["periodic_score"].dropna().hist(ax=axes[2], bins=50)
    axes[2].set_title("Periodic score")
    axes[2].set_xlabel("max autocorrelation")

    plt.tight_layout()
    plt.show()
except Exception as exc:
    print("plot skipped:", type(exc).__name__, exc)
"""
    ),
]

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "pygments_lexer": "ipython3",
    },
}

out = Path(__file__).with_name("azure_trace_exploration_cn.ipynb")
nbf.write(nb, out)
print(out)
