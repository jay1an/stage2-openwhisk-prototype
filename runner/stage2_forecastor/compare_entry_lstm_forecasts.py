import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .compare_entry_ml_forecasts import alloc_count, build_entry_counts, summarize
from .compare_stage_forecasts import load_split, resolve_window_ms
from ..workflow import load_workflow


POLICIES = {"p50": 0.50, "p90": 0.90, "p95": 0.95}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SMIless-inspired LSTM baseline for workflow entry arrival forecasting. "
            "It predicts the next-window invocation upper bound, then derives "
            "p50/p90/p95 with residual calibration."
        )
    )
    parser.add_argument("--trace", required=True)
    parser.add_argument("--workflow-config", required=True)
    parser.add_argument("--split-map", required=True)
    parser.add_argument("--window-sec", type=int, default=5)
    parser.add_argument("--window-ms", type=int, default=None)
    parser.add_argument("--context-windows", type=int, default=60)
    parser.add_argument("--bucket-size", type=float, default=1.0)
    parser.add_argument("--hidden-size", type=int, default=30)
    parser.add_argument("--rnn-type", choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--loss", choices=["l1", "mse", "huber"], default="l1")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--activation-threshold", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--write-detail", action="store_true")
    parser.add_argument("--write-forecast-csv", action="store_true")
    return parser.parse_args()


def import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import torch.utils.data as datautil
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required for the SMIless-inspired LSTM baseline. "
            "Install a CPU PyTorch build first, then rerun this script."
        ) from exc
    return torch, nn, optim, datautil


def bucket_upper_bound(values: np.ndarray, bucket_size: float) -> np.ndarray:
    if bucket_size <= 0:
        return values.astype(float)
    return np.ceil(values.astype(float) / bucket_size) * bucket_size


def make_samples(values: np.ndarray, windows: np.ndarray, context: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if context <= 0:
        raise ValueError("--context-windows must be positive")
    x_rows = []
    y_rows = []
    target_windows = []
    for idx in range(context, len(values)):
        x_rows.append(values[idx - context : idx])
        y_rows.append(values[idx])
        target_windows.append(windows[idx])
    if not x_rows:
        return (
            np.empty((0, context), dtype=float),
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=int),
        )
    return (
        np.asarray(x_rows, dtype=float),
        np.asarray(y_rows, dtype=float),
        np.asarray(target_windows, dtype=int),
    )


class InvocationLSTM:
    def __init__(
        self,
        torch,
        nn,
        hidden_size: int,
        rnn_type: str = "lstm",
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        class Model(nn.Module):
            def __init__(
                self,
                hidden_size: int,
                rnn_type: str,
                num_layers: int,
                dropout: float,
            ):
                super().__init__()
                effective_dropout = dropout if num_layers > 1 else 0.0
                rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
                self.rnn = rnn_cls(
                    input_size=1,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=effective_dropout,
                    batch_first=True,
                )
                self.linear = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.rnn(x)
                return self.linear(out[:, -1, :]).squeeze(-1)

        if num_layers <= 0:
            raise ValueError("--num-layers must be positive")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("--dropout must be in [0, 1)")
        self.model = Model(hidden_size, rnn_type, num_layers, dropout)
        self.torch = torch


def train_lstm(
    torch,
    nn,
    optim,
    datautil,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    hidden_size: int,
    rnn_type: str,
    num_layers: int,
    dropout: float,
    loss_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    random_state: int,
):
    torch.manual_seed(random_state)
    wrapper = InvocationLSTM(
        torch,
        nn,
        hidden_size,
        rnn_type=rnn_type,
        num_layers=num_layers,
        dropout=dropout,
    )
    model = wrapper.model
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if loss_name == "mse":
        loss_fn = nn.MSELoss()
    elif loss_name == "huber":
        loss_fn = nn.SmoothL1Loss()
    else:
        loss_fn = nn.L1Loss()
    train_x = torch.tensor(x_train[:, :, None], dtype=torch.float32)
    train_y = torch.tensor(y_train, dtype=torch.float32)
    val_x = torch.tensor(x_val[:, :, None], dtype=torch.float32)
    val_y = torch.tensor(y_val, dtype=torch.float32)
    loader = datautil.DataLoader(datautil.TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)

    best_state = None
    best_val = float("inf")
    stale = 0
    history = []
    for epoch in range(max(1, epochs)):
        model.train()
        train_losses = []
        for bx, by in loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = loss_fn(pred, by)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_loss = float(loss_fn(val_pred, val_y).item())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss})
        if val_loss + 1e-9 < best_val:
            best_val = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict_lstm(torch, model, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(x[:, :, None], dtype=torch.float32)
        return model(tensor).detach().cpu().numpy().astype(float)


def build_detail(
    forecast: pd.DataFrame,
    workflow_name: str,
    method: str,
    activation_threshold: float,
) -> pd.DataFrame:
    rows = []
    for _, row in forecast.iterrows():
        actual = int(row["actual_count"])
        for policy in POLICIES:
            forecast_count = float(row[f"{policy}_count"])
            allocated = alloc_count(forecast_count, activation_threshold)
            rows.append(
                {
                    "workflow_name": workflow_name,
                    "method": method,
                    "policy": policy,
                    "window": int(row["window"]),
                    "actual_count": actual,
                    "forecast_count": forecast_count,
                    "allocated_count": allocated,
                    "under_count": max(0, actual - allocated),
                    "over_count": max(0, allocated - actual),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    torch, nn, optim, datautil = import_torch()
    window_ms = resolve_window_ms(args)
    workflow = load_workflow(args.workflow_config)
    workflow_name = workflow.workflow_name
    trace = pd.read_csv(args.trace)
    _, _, split_map = load_split(
        trace,
        workflow_name,
        args.split_map,
        train_ratio=0.7,
        split_strategy="time",
    )
    split_map["entry_window"] = (split_map["entry_ts_ms"] // window_ms).astype(int)
    train_end_window = int(split_map["split_cutoff_ms"].dropna().iloc[0] // window_ms)
    eval_start_window = train_end_window + 1
    eval_end_window = int(split_map[split_map["split"] == "test"]["entry_window"].max())

    counts = build_entry_counts(trace, workflow_name, window_ms)
    windows = counts.index.to_numpy(dtype=int)
    raw_values = counts.to_numpy(dtype=float)
    binned_values = bucket_upper_bound(raw_values, args.bucket_size)
    x, y_binned, target_windows = make_samples(binned_values, windows, args.context_windows)
    _, y_actual, _ = make_samples(raw_values, windows, args.context_windows)
    if len(x) < 64:
        raise ValueError("not enough supervised windows for LSTM training/evaluation")

    train_mask = target_windows <= train_end_window
    test_mask = (target_windows >= eval_start_window) & (target_windows <= eval_end_window)
    x_train_all = x[train_mask]
    y_train_all = y_binned[train_mask]
    y_train_actual_all = y_actual[train_mask]
    x_test = x[test_mask]
    y_test_actual = y_actual[test_mask]
    test_windows = target_windows[test_mask]
    if len(x_train_all) < 64 or len(x_test) == 0:
        raise ValueError("train/test samples are empty after applying the time split")

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
    x_test_scaled = x_scaler.transform(x_test.reshape(-1, 1)).reshape(x_test.shape)
    y_fit_scaled = y_scaler.fit_transform(y_fit.reshape(-1, 1)).reshape(-1)

    # Validation uses the calibration inputs but the binned target scale, matching SMIless upper-bound training.
    y_cal_binned_scaled = y_scaler.transform(y_train_all[train_size:].reshape(-1, 1)).reshape(-1)
    model, history = train_lstm(
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
    test_pred_scaled = predict_lstm(torch, model, x_test_scaled)
    cal_pred = y_scaler.inverse_transform(cal_pred_scaled.reshape(-1, 1)).reshape(-1)
    test_pred = y_scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).reshape(-1)
    cal_pred = np.maximum(0.0, cal_pred)
    test_pred = np.maximum(0.0, test_pred)

    residuals = y_cal_actual - cal_pred
    residual_quantiles = {
        policy: float(np.quantile(residuals, quantile))
        for policy, quantile in POLICIES.items()
    }
    forecast = pd.DataFrame(
        {
            "window": test_windows.astype(int),
            "actual_count": y_test_actual.astype(float),
            "upper_bound_point": test_pred.astype(float),
        }
    )
    for policy in POLICIES:
        forecast[f"{policy}_count"] = np.maximum(0.0, forecast["upper_bound_point"] + residual_quantiles[policy])
    forecast["p90_count"] = np.maximum(forecast["p90_count"], forecast["p50_count"])
    forecast["p95_count"] = np.maximum(forecast["p95_count"], forecast["p90_count"])
    method_name = f"smiless-{args.rnn_type}-calibrated"
    forecast["method"] = method_name

    detail = build_detail(
        forecast=forecast,
        workflow_name=workflow_name,
        method=method_name,
        activation_threshold=args.activation_threshold,
    )
    summary = summarize(detail, window_ms)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{workflow_name}_entry_lstm_compare_summary.csv"
    detail_path = out_dir / f"{workflow_name}_entry_lstm_compare_detail.csv"
    forecast_path = out_dir / f"{workflow_name}_entry_lstm_forecast.csv"
    metadata_path = out_dir / f"{workflow_name}_entry_lstm_compare_metadata.json"
    summary.to_csv(summary_path, index=False)
    if args.write_detail:
        detail.to_csv(detail_path, index=False)
    if args.write_forecast_csv:
        forecast.to_csv(forecast_path, index=False)
    metadata = {
        "workflow_name": workflow_name,
        "trace": args.trace,
        "workflow_config": args.workflow_config,
        "split_map": args.split_map,
        "window_ms": window_ms,
        "context_windows": args.context_windows,
        "bucket_size": args.bucket_size,
        "hidden_size": args.hidden_size,
        "rnn_type": args.rnn_type,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "loss": args.loss,
        "train_samples": int(train_size),
        "calibration_samples": int(cal_size),
        "test_samples": int(len(x_test)),
        "residual_quantiles": residual_quantiles,
        "epochs_run": int(len(history)),
        "model": (
            "SMIless-inspired LSTM invocation-number upper-bound with "
            "residual quantile calibration"
        ),
        "notes": [
            "SMIless uses LSTM for invocation-number upper-bound prediction.",
            "This adaptation adds residual calibration to expose p50/p90/p95 forecasts.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")
    if args.write_detail:
        print(f"wrote {detail_path}")
    if args.write_forecast_csv:
        print(f"wrote {forecast_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

