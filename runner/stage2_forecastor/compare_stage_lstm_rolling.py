import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .compare_entry_lstm_forecasts import (
    bucket_upper_bound,
    import_torch,
    make_samples,
    predict_lstm,
    train_lstm,
)
from .compare_stage_forecasts import build_delay_kernel, load_split, resolve_window_ms
from .compare_stage_lightgbm_quantile_rolling import (
    POLICIES,
    actual_entry_counts,
    actual_stage_counts,
    alloc_count,
    ceil_count,
    complete_stage_forecast,
    count_series_from_windows,
    detail_from_forecast,
    summarize_detail,
)
from .compare_stage_rolling_forecasts import window_series
from ..workflow import load_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rolling-origin tuned LSTM workflow-entry forecast, DAG propagation, "
            "and per-stage independent LSTM baseline."
        )
    )
    parser.add_argument("--trace", required=True)
    parser.add_argument("--workflow-config", required=True)
    parser.add_argument("--split-index", required=True)
    parser.add_argument("--window-sec", type=int, default=5)
    parser.add_argument("--window-ms", type=int, default=None)
    parser.add_argument("--activation-threshold", type=float, default=0.1)
    parser.add_argument("--context-windows", type=int, default=30)
    parser.add_argument("--bucket-size", type=float, default=1.0)
    parser.add_argument("--hidden-size", type=int, default=30)
    parser.add_argument("--rnn-type", choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--loss", choices=["l1", "mse", "huber"], default="l1")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--calibration-ratio", type=float, default=0.20)
    parser.add_argument(
        "--tail-residual-scale",
        type=float,
        default=1.0,
        help="multiply positive residual shifts for p90/p95 to control tail conservativeness",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--write-forecast-csvs", action="store_true")
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def resolve_path(root: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    return path if path.is_absolute() else root / path


def method_label(prefix: str, args: argparse.Namespace) -> str:
    return (
        f"{prefix}-{args.rnn_type}-calibrated"
        f"-ctx{args.context_windows}-h{args.hidden_size}"
    )


def fallback_forecast_from_train(
    counts: pd.Series,
    eval_windows: list[int],
    method: str,
    window_ms: int,
    activation_threshold: float,
) -> pd.DataFrame:
    train_values = counts.to_numpy(dtype=float)
    if len(train_values) == 0:
        quantiles = {policy: 0.0 for policy in POLICIES}
    else:
        quantiles = {
            policy: float(np.quantile(train_values, quantile))
            for policy, quantile in POLICIES.items()
        }
    rows = []
    for window in eval_windows:
        row = {
            "method": method,
            "window": int(window),
            "window_start_ms": int(window) * window_ms,
        }
        previous = 0.0
        for policy in POLICIES:
            value = max(previous, max(0.0, quantiles[policy]))
            row[f"{policy}_count"] = value
            row[f"ceil_{policy}_count"] = ceil_count(value)
            row[f"alloc_{policy}_count"] = alloc_count(value, activation_threshold)
            previous = value
        rows.append(row)
    return pd.DataFrame(rows)


def forecast_lstm_from_counts(
    counts: pd.Series,
    train_end_window: int,
    eval_windows: list[int],
    method: str,
    window_ms: int,
    args: argparse.Namespace,
    torch,
    nn,
    optim,
    datautil,
) -> pd.DataFrame:
    counts = counts.sort_index().astype(float)
    train_counts = counts[counts.index <= train_end_window]
    if len(train_counts) < max(64, args.context_windows + 20):
        return fallback_forecast_from_train(
            train_counts,
            eval_windows,
            method,
            window_ms,
            args.activation_threshold,
        )

    windows = counts.index.to_numpy(dtype=int)
    raw_values = counts.to_numpy(dtype=float)
    binned_values = bucket_upper_bound(raw_values, args.bucket_size)
    x, y_binned, target_windows = make_samples(
        binned_values,
        windows,
        args.context_windows,
    )
    _, y_actual, _ = make_samples(raw_values, windows, args.context_windows)
    if len(x) < 64:
        return fallback_forecast_from_train(
            train_counts,
            eval_windows,
            method,
            window_ms,
            args.activation_threshold,
        )

    eval_set = set(int(window) for window in eval_windows)
    train_mask = target_windows <= train_end_window
    eval_mask = np.array([int(window) in eval_set for window in target_windows], dtype=bool)
    x_train_all = x[train_mask]
    y_train_all = y_binned[train_mask]
    y_train_actual_all = y_actual[train_mask]
    x_eval = x[eval_mask]
    y_eval_actual = y_actual[eval_mask]
    eval_target_windows = target_windows[eval_mask]
    if len(x_train_all) < 64 or len(x_eval) == 0:
        return fallback_forecast_from_train(
            train_counts,
            eval_windows,
            method,
            window_ms,
            args.activation_threshold,
        )

    cal_size = max(16, int(math.ceil(len(x_train_all) * args.calibration_ratio)))
    if cal_size >= len(x_train_all):
        cal_size = max(1, len(x_train_all) // 5)
    train_size = len(x_train_all) - cal_size
    x_fit = x_train_all[:train_size]
    y_fit = y_train_all[:train_size]
    x_cal = x_train_all[train_size:]
    y_cal_actual = y_train_actual_all[train_size:]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_fit_scaled = x_scaler.fit_transform(x_fit.reshape(-1, 1)).reshape(x_fit.shape)
    x_cal_scaled = x_scaler.transform(x_cal.reshape(-1, 1)).reshape(x_cal.shape)
    x_eval_scaled = x_scaler.transform(x_eval.reshape(-1, 1)).reshape(x_eval.shape)
    y_fit_scaled = y_scaler.fit_transform(y_fit.reshape(-1, 1)).reshape(-1)
    y_cal_binned_scaled = y_scaler.transform(y_train_all[train_size:].reshape(-1, 1)).reshape(-1)

    model, _history = train_lstm(
        torch=torch,
        nn=nn,
        optim=optim,
        datautil=datautil,
        x_train=x_fit_scaled,
        y_train=y_fit_scaled,
        x_val=x_cal_scaled,
        y_val=y_cal_binned_scaled,
        hidden_size=args.hidden_size,
        rnn_type=args.rnn_type,
        num_layers=args.num_layers,
        dropout=args.dropout,
        loss_name=args.loss,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        random_state=args.random_state,
    )

    cal_pred_scaled = predict_lstm(torch, model, x_cal_scaled)
    eval_pred_scaled = predict_lstm(torch, model, x_eval_scaled)
    cal_pred = y_scaler.inverse_transform(cal_pred_scaled.reshape(-1, 1)).reshape(-1)
    eval_pred = y_scaler.inverse_transform(eval_pred_scaled.reshape(-1, 1)).reshape(-1)
    cal_pred = np.maximum(0.0, cal_pred)
    eval_pred = np.maximum(0.0, eval_pred)

    residuals = y_cal_actual - cal_pred
    residual_quantiles = {}
    for policy, quantile in POLICIES.items():
        shift = float(np.quantile(residuals, quantile))
        if policy != "p50" and shift > 0.0:
            shift *= max(0.0, float(args.tail_residual_scale))
        residual_quantiles[policy] = shift
    forecast = pd.DataFrame(
        {
            "method": method,
            "window": eval_target_windows.astype(int),
            "actual_count": y_eval_actual.astype(float),
            "point_count": eval_pred.astype(float),
            "window_start_ms": eval_target_windows.astype(int) * window_ms,
        }
    )
    previous = None
    for policy in POLICIES:
        values = np.maximum(
            0.0,
            forecast["point_count"].to_numpy(dtype=float) + residual_quantiles[policy],
        )
        if previous is not None:
            values = np.maximum(previous, values)
        forecast[f"{policy}_count"] = values
        forecast[f"ceil_{policy}_count"] = [ceil_count(value) for value in values]
        forecast[f"alloc_{policy}_count"] = [
            alloc_count(value, args.activation_threshold) for value in values
        ]
        previous = values

    existing = set(int(window) for window in forecast["window"].tolist())
    missing = [int(window) for window in eval_windows if int(window) not in existing]
    if missing:
        fallback = fallback_forecast_from_train(
            train_counts,
            missing,
            method,
            window_ms,
            args.activation_threshold,
        )
        forecast = pd.concat([forecast, fallback], ignore_index=True, sort=False)
    return forecast.sort_values("window").reset_index(drop=True)


def build_entry_forecasts_for_fold(
    entries: pd.DataFrame,
    workflow_name: str,
    first_window: int,
    train_end_window: int,
    eval_windows: list[int],
    window_ms: int,
    args: argparse.Namespace,
    torch,
    nn,
    optim,
    datautil,
) -> pd.DataFrame:
    last_needed = max(max(eval_windows), train_end_window)
    counts = count_series_from_windows(entries, "window", first_window, last_needed)
    method = method_label("entry", args)
    forecast = forecast_lstm_from_counts(
        counts,
        train_end_window,
        eval_windows,
        method,
        window_ms,
        args,
        torch,
        nn,
        optim,
        datautil,
    )
    forecast["workflow_name"] = workflow_name
    return forecast


def propagate_entry_forecast(
    workflow_name: str,
    workflow,
    entry_forecast: pd.DataFrame,
    train_stage_rows: pd.DataFrame,
    window_ms: int,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    method = method_label("dag", args)
    for stage_name in workflow.nodes:
        stage_rows = train_stage_rows[train_stage_rows["stage_name"] == stage_name].copy()
        kernel = build_delay_kernel(stage_rows, window_ms) if not stage_rows.empty else {0: 1.0}
        for _, forecast_row in entry_forecast.iterrows():
            for offset, probability in kernel.items():
                target_window = int(forecast_row["window"]) + int(offset)
                row = {
                    "workflow_name": workflow_name,
                    "method": method,
                    "stage_name": stage_name,
                    "window": target_window,
                    "window_start_ms": target_window * window_ms,
                }
                for policy in POLICIES:
                    row[f"{policy}_count"] = float(forecast_row[f"{policy}_count"]) * float(
                        probability
                    )
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = (
        pd.DataFrame(rows)
        .groupby(["workflow_name", "method", "stage_name", "window", "window_start_ms"], as_index=False)[
            [f"{policy}_count" for policy in POLICIES]
        ]
        .sum()
    )
    for policy in POLICIES:
        out[f"ceil_{policy}_count"] = out[f"{policy}_count"].map(ceil_count)
        out[f"alloc_{policy}_count"] = out[f"{policy}_count"].map(
            lambda value: alloc_count(value, args.activation_threshold)
        )
    return out


def build_independent_stage_forecasts_for_fold(
    stage_rows: pd.DataFrame,
    workflow,
    workflow_name: str,
    first_window: int,
    train_end_window: int,
    eval_windows: list[int],
    window_ms: int,
    args: argparse.Namespace,
    torch,
    nn,
    optim,
    datautil,
) -> pd.DataFrame:
    outputs = []
    last_needed = max(max(eval_windows), train_end_window)
    method = method_label("independent", args)
    for stage_name in workflow.nodes:
        stage = stage_rows[stage_rows["stage_name"] == stage_name].copy()
        counts = count_series_from_windows(stage, "dispatch_window", first_window, last_needed)
        forecast = forecast_lstm_from_counts(
            counts,
            train_end_window,
            eval_windows,
            method,
            window_ms,
            args,
            torch,
            nn,
            optim,
            datautil,
        )
        forecast["workflow_name"] = workflow_name
        forecast["stage_name"] = stage_name
        outputs.append(forecast)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def write_report(out_dir: Path, entry_summary: pd.DataFrame, stage_summary: pd.DataFrame) -> None:
    def compact(frame: pd.DataFrame) -> pd.DataFrame:
        cols = [
            "method_family",
            "method",
            "policy",
            "demand_coverage_rate",
            "allocated_replica_seconds",
            "over_allocation_ratio",
            "allocation_utilization",
            "pinball_loss_mean",
            "empirical_quantile_coverage",
            "quantile_calibration_error",
        ]
        return frame[[col for col in cols if col in frame.columns]].copy()

    lines = [
        "# Tuned LSTM DAG Propagation Report",
        "",
        "## Scope",
        "",
        "- Entry method: tuned calibrated LSTM/GRU sequence forecaster.",
        "- Stage method 1: workflow-entry forecast with empirical DAG delay-kernel propagation.",
        "- Stage method 2: per-stage independent calibrated LSTM baseline.",
        "- Evaluation: rolling-origin folds from the provided split index.",
        "",
        "## Entry Summary",
        "",
        compact(entry_summary).to_markdown(index=False),
        "",
        "## Stage Summary",
        "",
        compact(stage_summary).to_markdown(index=False),
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch, nn, optim, datautil = import_torch()

    root = project_root()
    trace_path = resolve_path(root, args.trace)
    workflow_path = resolve_path(root, args.workflow_config)
    split_index_path = resolve_path(root, args.split_index)
    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    window_ms = resolve_window_ms(args)
    workflow = load_workflow(str(workflow_path))
    workflow_name = workflow.workflow_name
    trace = pd.read_csv(trace_path)
    split_index = pd.read_csv(split_index_path)

    workflow_rows = trace[
        (trace["workflow_name"] == workflow_name)
        & (trace["status"] == "ok")
    ].copy()
    entries = workflow_rows[workflow_rows["stage_name"] == "__entry__"].copy()
    entries["window"] = window_series(entries, "entry_ts_ms", window_ms)
    first_entry_window = int(entries["window"].min())

    stage_rows = workflow_rows[workflow_rows["stage_name"] != "__entry__"].copy()
    stage_rows["dispatch_window"] = window_series(stage_rows, "dispatch_start_ms", window_ms)
    first_stage_window = int(stage_rows["dispatch_window"].min())

    entry_detail_frames = []
    stage_detail_frames = []
    forecast_frames = []

    for _, fold in split_index.iterrows():
        fold_id = int(fold["fold_id"])
        split_path = Path(str(fold["split_path"]))
        if not split_path.is_absolute():
            split_path = resolve_path(root, str(split_path))
        train_ids, test_ids, split_map = load_split(
            trace,
            workflow_name,
            str(split_path),
            train_ratio=0.7,
            split_strategy="time",
        )
        split_map["entry_window"] = (split_map["entry_ts_ms"] // window_ms).astype(int)
        train_end_window = int(pd.to_numeric(split_map["split_cutoff_ms"]).dropna().iloc[0] // window_ms)

        test_entries = entries[entries["request_id"].isin(test_ids)].copy()
        if test_entries.empty:
            continue
        entry_eval_windows = list(
            range(int(test_entries["window"].min()), int(test_entries["window"].max()) + 1)
        )
        entry_actual = actual_entry_counts(test_entries, entry_eval_windows)
        entry_forecast = build_entry_forecasts_for_fold(
            entries=entries,
            workflow_name=workflow_name,
            first_window=first_entry_window,
            train_end_window=train_end_window,
            eval_windows=entry_eval_windows,
            window_ms=window_ms,
            args=args,
            torch=torch,
            nn=nn,
            optim=optim,
            datautil=datautil,
        )
        entry_detail_frames.append(
            detail_from_forecast(
                entry_forecast,
                entry_actual,
                fold_id,
                workflow_name,
                "entry-lstm",
                "entry",
                window_ms,
            )
        )

        test_stage_rows = stage_rows[stage_rows["request_id"].isin(test_ids)].copy()
        if test_stage_rows.empty:
            continue
        stage_eval_windows = list(
            range(
                int(test_stage_rows["dispatch_window"].min()),
                int(test_stage_rows["dispatch_window"].max()) + 1,
            )
        )
        stage_actual = actual_stage_counts(test_stage_rows, list(workflow.nodes), stage_eval_windows)

        train_stage_rows = stage_rows[stage_rows["dispatch_window"] <= train_end_window].copy()
        dag_forecast = propagate_entry_forecast(
            workflow_name=workflow_name,
            workflow=workflow,
            entry_forecast=entry_forecast,
            train_stage_rows=train_stage_rows,
            window_ms=window_ms,
            args=args,
        )
        forecast_frames.append(dag_forecast.assign(fold_id=fold_id))
        completed_dag = complete_stage_forecast(
            dag_forecast,
            workflow,
            workflow_name,
            str(dag_forecast["method"].iloc[0]),
            stage_eval_windows,
            window_ms,
        )
        stage_detail_frames.append(
            detail_from_forecast(
                completed_dag,
                stage_actual,
                fold_id,
                workflow_name,
                "entry-lstm-dag",
                "stage",
                window_ms,
            )
        )

        independent_forecast = build_independent_stage_forecasts_for_fold(
            stage_rows=stage_rows,
            workflow=workflow,
            workflow_name=workflow_name,
            first_window=first_stage_window,
            train_end_window=train_end_window,
            eval_windows=stage_eval_windows,
            window_ms=window_ms,
            args=args,
            torch=torch,
            nn=nn,
            optim=optim,
            datautil=datautil,
        )
        forecast_frames.append(independent_forecast.assign(fold_id=fold_id))
        completed_independent = complete_stage_forecast(
            independent_forecast,
            workflow,
            workflow_name,
            str(independent_forecast["method"].iloc[0]),
            stage_eval_windows,
            window_ms,
        )
        stage_detail_frames.append(
            detail_from_forecast(
                completed_independent,
                stage_actual,
                fold_id,
                workflow_name,
                "per-stage-independent-lstm",
                "stage",
                window_ms,
            )
        )

    entry_detail = pd.concat(entry_detail_frames, ignore_index=True)
    stage_detail = pd.concat(stage_detail_frames, ignore_index=True)
    entry_summary = summarize_detail(entry_detail, by_stage=False)
    stage_summary = summarize_detail(stage_detail, by_stage=False)
    stage_by_stage = summarize_detail(stage_detail, by_stage=True)

    entry_detail.to_csv(out_dir / "entry_detail.csv", index=False)
    entry_summary.to_csv(out_dir / "entry_summary.csv", index=False)
    stage_detail.to_csv(out_dir / "stage_detail.csv", index=False)
    stage_summary.to_csv(out_dir / "stage_summary.csv", index=False)
    stage_by_stage.to_csv(out_dir / "stage_by_stage.csv", index=False)
    if args.write_forecast_csvs and forecast_frames:
        pd.concat(forecast_frames, ignore_index=True).to_csv(out_dir / "forecasts.csv", index=False)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "trace": str(trace_path),
        "workflow_config": str(workflow_path),
        "split_index": str(split_index_path),
        "workflow_name": workflow_name,
        "window_ms": window_ms,
        "context_windows": args.context_windows,
        "bucket_size": args.bucket_size,
        "hidden_size": args.hidden_size,
        "rnn_type": args.rnn_type,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "loss": args.loss,
        "epochs": args.epochs,
        "patience": args.patience,
        "calibration_ratio": args.calibration_ratio,
        "tail_residual_scale": args.tail_residual_scale,
        "folds": int(len(split_index)),
        "methods": [
            method_label("entry", args),
            method_label("dag", args),
            method_label("independent", args),
        ],
        "online_assumption": (
            "Each target window forecast may use observed counts from earlier windows, "
            "matching a short control-loop one-step-ahead forecaster."
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(out_dir, entry_summary, stage_summary)

    print(f"wrote {out_dir}")
    print("entry summary:")
    print(entry_summary.to_string(index=False))
    print("stage summary:")
    print(stage_summary.to_string(index=False))


if __name__ == "__main__":
    main()

