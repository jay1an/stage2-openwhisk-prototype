import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--window-sec", type=int, default=5)
    parser.add_argument(
        "--window-ms",
        type=int,
        default=None,
        help="override --window-sec with a millisecond-level window",
    )
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument(
        "--method",
        choices=[
            "ewma",
            "burst-aware",
            "burst-localized",
            "hurdle-ewma",
            "tsb",
            "hazard-hurdle",
            "fip-fourier",
        ],
        default="ewma",
    )
    parser.add_argument("--residual-window", type=int, default=60)
    parser.add_argument("--history-window", type=int, default=30)
    parser.add_argument("--burst-threshold", type=float, default=2.0)
    parser.add_argument(
        "--burst-period-windows",
        type=int,
        default=None,
        help="expected period between burst peaks, in forecast windows",
    )
    parser.add_argument(
        "--burst-width-windows",
        type=int,
        default=0,
        help="number of windows on each side of a predicted burst peak to mark as burst",
    )
    parser.add_argument(
        "--background-count",
        type=float,
        default=None,
        help="override non-burst forecast count for burst-localized",
    )
    parser.add_argument("--idle-zero-ratio", type=float, default=0.8)
    parser.add_argument("--activation-threshold", type=float, default=0.1)
    parser.add_argument(
        "--train-until-ms",
        type=int,
        default=None,
        help="use only entry rows with entry_ts_ms <= this timestamp",
    )
    parser.add_argument(
        "--train-until-window",
        type=int,
        default=None,
        help="use only entry rows whose window index is <= this value",
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def resolve_window_ms(args: argparse.Namespace) -> int:
    if args.window_ms is not None:
        if args.window_ms <= 0:
            raise ValueError("--window-ms must be positive")
        return args.window_ms
    if args.window_sec <= 0:
        raise ValueError("--window-sec must be positive")
    return args.window_sec * 1000


def ewma(values: np.ndarray, alpha: float) -> float:
    if len(values) == 0:
        return 0.0
    current = float(values[0])
    for value in values[1:]:
        current = alpha * float(value) + (1.0 - alpha) * current
    return current


def recent_residual_quantile(
    counts: np.ndarray,
    base: float,
    quantile: float,
    residual_window: int,
) -> float:
    if len(counts) == 0:
        return 0.0
    recent = counts[-max(1, residual_window):]
    residuals = recent - base
    return max(0.0, float(np.quantile(residuals, quantile)))


def ceil_count(value: float) -> int:
    return int(math.ceil(max(0.0, value)))


def alloc_count(value: float, activation_threshold: float) -> int:
    if value < activation_threshold:
        return 0
    return ceil_count(value)


def burst_aware_forecast(
    counts: np.ndarray,
    alpha: float,
    history_window: int,
    burst_threshold: float,
    idle_zero_ratio: float,
) -> tuple[float, float, float, float]:
    if len(counts) == 0:
        return 0.0, 0.0, 0.0, 0.0

    base = ewma(counts, alpha)
    recent = counts[-max(1, history_window):]
    recent_max = float(np.max(recent)) if len(recent) else 0.0
    zero_ratio = float(np.mean(recent == 0)) if len(recent) else 1.0
    nonzero = recent[recent > 0]
    nonzero_mean = float(np.mean(nonzero)) if len(nonzero) else 0.0

    if recent_max >= burst_threshold:
        p50 = max(base, nonzero_mean)
        p90 = max(p50, recent_max)
        p95 = max(p90, float(np.quantile(recent, 0.95)))
        p99 = max(p95, float(np.quantile(recent, 0.99)))
    elif zero_ratio >= idle_zero_ratio:
        p50 = min(base, nonzero_mean) if nonzero_mean > 0 else base
        p90 = max(p50, min(1.0, max(nonzero_mean, base)))
        p95 = max(p90, min(1.0, recent_max))
        p99 = max(p95, min(1.0, recent_max))
    else:
        p50 = base
        p90 = max(p50, float(np.quantile(recent, 0.90)))
        p95 = max(p90, float(np.quantile(recent, 0.95)))
        p99 = max(p95, float(np.quantile(recent, 0.99)))

    return max(0.0, p50), max(0.0, p90), max(0.0, p95), max(0.0, p99)


def hurdle_quantile(
    active_probability: float,
    positive_counts: np.ndarray,
    quantile: float,
) -> float:
    p_active = min(1.0, max(0.0, float(active_probability)))
    if p_active <= 0.0 or len(positive_counts) == 0:
        return 0.0
    zero_mass = 1.0 - p_active
    if quantile <= zero_mass:
        return 0.0
    adjusted = (quantile - zero_mass) / p_active
    adjusted = min(1.0, max(0.0, adjusted))
    return max(0.0, float(np.quantile(positive_counts, adjusted)))


def hurdle_ewma_forecast(
    counts: np.ndarray,
    alpha: float,
    residual_window: int,
    history_window: int,
) -> tuple[float, float, float, float]:
    if len(counts) == 0:
        return 0.0, 0.0, 0.0, 0.0

    active_indicator = (counts > 0).astype(float)
    active_probability = ewma(active_indicator, alpha)

    recent = counts[-max(1, history_window):]
    positive_recent = recent[recent > 0]
    if len(positive_recent) == 0:
        positive_recent = counts[counts > 0]
    if len(positive_recent) == 0:
        return 0.0, 0.0, 0.0, 0.0

    p50 = hurdle_quantile(active_probability, positive_recent, 0.50)
    p90 = hurdle_quantile(active_probability, positive_recent, 0.90)
    p95 = hurdle_quantile(active_probability, positive_recent, 0.95)
    p99 = hurdle_quantile(active_probability, positive_recent, 0.99)

    if len(positive_recent) > 0:
        positive_base = ewma(positive_recent.astype(float), alpha)
        p90 = max(p90, min(float(np.max(positive_recent)), positive_base))
        p95 = max(p95, p90)
        p99 = max(p99, p95)

    if active_probability < 0.5:
        p50 = 0.0

    return max(0.0, p50), max(0.0, p90), max(0.0, p95), max(0.0, p99)


def tsb_forecast(
    counts: np.ndarray,
    alpha: float,
    history_window: int,
) -> tuple[float, float, float, float]:
    """Teunter-Syntetos-Babai style intermittent-demand baseline.

    TSB separately smooths demand occurrence probability and positive demand
    size. Its quantiles are then read from a zero-inflated empirical mixture.
    """
    if len(counts) == 0:
        return 0.0, 0.0, 0.0, 0.0

    p_active = 1.0 if counts[0] > 0 else 0.0
    positive = counts[counts > 0]
    smoothed_size = float(positive[0]) if len(positive) else 0.0
    for value in counts[1:]:
        occurrence = 1.0 if value > 0 else 0.0
        p_active = alpha * occurrence + (1.0 - alpha) * p_active
        if value > 0:
            smoothed_size = alpha * float(value) + (1.0 - alpha) * smoothed_size

    recent = counts[-max(1, history_window):]
    positive_recent = recent[recent > 0]
    if len(positive_recent) == 0:
        positive_recent = positive
    if len(positive_recent) == 0:
        return 0.0, 0.0, 0.0, 0.0

    p50 = hurdle_quantile(p_active, positive_recent, 0.50)
    p90 = hurdle_quantile(p_active, positive_recent, 0.90)
    p95 = hurdle_quantile(p_active, positive_recent, 0.95)
    p99 = hurdle_quantile(p_active, positive_recent, 0.99)
    if p_active >= 0.5:
        p50 = max(p50, min(float(np.max(positive_recent)), smoothed_size))
    return max(0.0, p50), max(0.0, p90), max(0.0, p95), max(0.0, p99)


def empirical_next_active_probability(
    counts: np.ndarray,
    horizon_step: int,
    fallback_probability: float,
    smoothing: float = 5.0,
) -> float:
    active_positions = np.flatnonzero(counts > 0)
    if len(active_positions) < 2:
        return min(1.0, max(0.0, fallback_probability))

    idle_age = int(len(counts) - 1 - active_positions[-1])
    target_gap = idle_age + max(1, int(horizon_step))
    gaps = np.diff(active_positions).astype(int)
    risk = int(np.sum(gaps >= target_gap))
    events = int(np.sum(gaps == target_gap))
    if risk <= 0:
        return min(1.0, max(0.0, fallback_probability))

    # Smoothed discrete hazard: P(next active exactly at target_gap | gap >= target_gap).
    probability = (events + smoothing * fallback_probability) / (risk + smoothing)
    return min(1.0, max(0.0, float(probability)))


def hazard_hurdle_forecast(
    counts: np.ndarray,
    alpha: float,
    history_window: int,
    horizon_step: int = 1,
) -> tuple[float, float, float, float, float]:
    if len(counts) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    active_indicator = (counts > 0).astype(float)
    fallback_probability = ewma(active_indicator, alpha)
    p_active = empirical_next_active_probability(
        counts,
        horizon_step=horizon_step,
        fallback_probability=fallback_probability,
    )

    recent = counts[-max(1, history_window):]
    positive_recent = recent[recent > 0]
    if len(positive_recent) == 0:
        positive_recent = counts[counts > 0]
    if len(positive_recent) == 0:
        return 0.0, 0.0, 0.0, 0.0, p_active

    p50 = hurdle_quantile(p_active, positive_recent, 0.50)
    p90 = hurdle_quantile(p_active, positive_recent, 0.90)
    p95 = hurdle_quantile(p_active, positive_recent, 0.95)
    p99 = hurdle_quantile(p_active, positive_recent, 0.99)
    return max(0.0, p50), max(0.0, p90), max(0.0, p95), max(0.0, p99), p_active


def fourier_extrapolate(
    values: np.ndarray,
    n_predict: int,
    harmonics: int = 10,
) -> np.ndarray:
    """IceBreaker-style Fourier extrapolation after removing a linear trend."""
    x = np.asarray(values, dtype=float)
    n = int(x.size)
    if n == 0 or n_predict <= 0:
        return np.zeros(max(0, n_predict), dtype=float)
    if n < 4 or float(np.max(x) - np.min(x)) == 0.0:
        return np.full(n_predict, float(x[-1]), dtype=float)

    t = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(t, x, 1)
    x_notrend = x - (slope * t + intercept)
    freq_domain = np.fft.fft(x_notrend)
    freqs = np.fft.fftfreq(n)
    indexes = sorted(range(n), key=lambda idx: abs(freqs[idx]))

    future_t = np.arange(n, n + n_predict, dtype=float)
    restored = np.zeros(n_predict, dtype=float)
    for idx in indexes[: 1 + harmonics * 2]:
        amplitude = abs(freq_domain[idx]) / n
        phase = np.angle(freq_domain[idx])
        restored += amplitude * np.cos(2.0 * np.pi * freqs[idx] * future_t + phase)
    restored += slope * future_t + intercept
    return np.maximum(0.0, restored)


def fip_fourier_forecast(
    counts: np.ndarray,
    horizon_step: int = 1,
    local_window: int = 60,
    harmonics: int = 10,
    residual_window: int = 60,
) -> tuple[float, float, float, float]:
    """Fourier Invocation Prediction baseline inspired by IceBreaker.

    IceBreaker's FIP produces a point forecast. For our probabilistic entry
    forecast comparison, we calibrate that point forecast with recent one-step
    residual quantiles to obtain p50/p90/p95/p99.
    """
    values = np.asarray(counts, dtype=float)
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    local_window = max(4, min(int(local_window), int(values.size)))
    harmonics = max(1, min(int(harmonics), max(1, local_window // 2)))
    horizon_step = max(1, int(horizon_step))

    base = float(fourier_extrapolate(values[-local_window:], horizon_step, harmonics)[-1])

    residuals = []
    start = max(local_window, values.size - max(local_window + residual_window, local_window + 1))
    for idx in range(start, values.size):
        history = values[max(0, idx - local_window):idx]
        if history.size < 4:
            continue
        pred = float(fourier_extrapolate(history, 1, harmonics)[-1])
        residuals.append(float(values[idx] - pred))

    if residuals:
        residual_array = np.asarray(residuals, dtype=float)
        shifts = {
            0.50: float(np.quantile(residual_array, 0.50)),
            0.90: float(np.quantile(residual_array, 0.90)),
            0.95: float(np.quantile(residual_array, 0.95)),
            0.99: float(np.quantile(residual_array, 0.99)),
        }
    else:
        shifts = {0.50: 0.0, 0.90: 0.0, 0.95: 0.0, 0.99: 0.0}

    p50 = max(0.0, base + shifts[0.50])
    p90 = max(p50, max(0.0, base + shifts[0.90]))
    p95 = max(p90, max(0.0, base + shifts[0.95]))
    p99 = max(p95, max(0.0, base + shifts[0.99]))
    return p50, p90, p95, p99


def burst_groups(counts: pd.Series, burst_threshold: float) -> list[dict]:
    burst_windows = [
        (int(window), float(count))
        for window, count in counts.items()
        if float(count) >= burst_threshold
    ]
    if not burst_windows:
        return []

    groups = []
    current = [burst_windows[0]]
    for item in burst_windows[1:]:
        if item[0] == current[-1][0] + 1:
            current.append(item)
        else:
            groups.append(current)
            current = [item]
    groups.append(current)

    result = []
    for group in groups:
        peak_window, peak_count = max(group, key=lambda item: item[1])
        result.append(
            {
                "start": group[0][0],
                "end": group[-1][0],
                "peak_window": peak_window,
                "peak_count": peak_count,
            }
        )
    return result


def estimate_burst_period(groups: list[dict]) -> int | None:
    if len(groups) < 2:
        return None
    gaps = [
        groups[idx]["peak_window"] - groups[idx - 1]["peak_window"]
        for idx in range(1, len(groups))
    ]
    return int(round(float(np.median(gaps)))) if gaps else None


def is_predicted_burst_window(
    target_window: int,
    last_peak_window: int,
    period_windows: int,
    width_windows: int,
) -> bool:
    center = last_peak_window + period_windows
    while center + width_windows < target_window:
        center += period_windows
    return abs(target_window - center) <= width_windows


def burst_localized_forecast(
    counts: pd.Series,
    target_window: int,
    alpha: float,
    burst_threshold: float,
    burst_period_windows: int | None,
    burst_width_windows: int,
    background_count: float | None,
) -> tuple[float, float, float, float]:
    count_values = counts.to_numpy()
    base = ewma(count_values, alpha)
    if background_count is None:
        recent = count_values[-max(1, min(len(count_values), 30)) :]
        zero_ratio = float(np.mean(recent <= 0)) if len(recent) else 1.0
        background = 0.0 if zero_ratio >= 0.8 else base
    else:
        background = background_count
    background = max(0.0, float(background))

    groups = burst_groups(counts, burst_threshold)
    if not groups:
        return background, background, background, background

    period = burst_period_windows or estimate_burst_period(groups)
    if period is None or period <= 0:
        return background, background, background, background

    last_peak_window = int(groups[-1]["peak_window"])
    peak_count = max(float(group["peak_count"]) for group in groups)
    if not is_predicted_burst_window(
        target_window,
        last_peak_window,
        period,
        max(0, burst_width_windows),
    ):
        return background, background, background, background

    nonzero = count_values[count_values > 0]
    nonzero_mean = float(np.mean(nonzero)) if len(nonzero) else background
    p50 = max(background, min(peak_count, nonzero_mean))
    p90 = max(p50, peak_count)
    return p50, p90, p90, p90


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.trace)
    entries = df[
        (df["workflow_name"] == args.workflow)
        & (df["stage_name"] == "__entry__")
        & (df["status"] == "ok")
    ].copy()

    if entries.empty:
        raise SystemExit(f"no entry rows found for workflow={args.workflow}")

    window_ms = resolve_window_ms(args)
    if args.train_until_ms is not None:
        entries = entries[entries["entry_ts_ms"] <= args.train_until_ms].copy()
    if args.train_until_window is not None:
        entries = entries[
            (entries["entry_ts_ms"] // window_ms).astype(int)
            <= args.train_until_window
        ].copy()
    if entries.empty:
        raise SystemExit("no entry rows left after applying the training cutoff")

    first_window = int(entries["entry_ts_ms"].min() // window_ms)
    last_window = int(entries["entry_ts_ms"].max() // window_ms)

    counts = (
        entries.assign(window=(entries["entry_ts_ms"] // window_ms).astype(int))
        .groupby("window")
        .size()
        .reindex(range(first_window, last_window + 1), fill_value=0)
        .astype(float)
    )

    count_values = counts.to_numpy()
    if args.method == "ewma":
        base = ewma(count_values, args.alpha)
        p90_pad = recent_residual_quantile(count_values, base, 0.90, args.residual_window)
        p95_pad = recent_residual_quantile(count_values, base, 0.95, args.residual_window)
        p99_pad = recent_residual_quantile(count_values, base, 0.99, args.residual_window)
        p50_base = max(0.0, base)
        p90_base = max(0.0, base + p90_pad)
        p95_base = max(0.0, base + p95_pad)
        p99_base = max(0.0, base + p99_pad)
    elif args.method == "burst-aware":
        p50_base, p90_base, p95_base, p99_base = burst_aware_forecast(
            count_values,
            args.alpha,
            args.history_window,
            args.burst_threshold,
            args.idle_zero_ratio,
        )
    elif args.method == "hurdle-ewma":
        p50_base, p90_base, p95_base, p99_base = hurdle_ewma_forecast(
            count_values,
            args.alpha,
            args.residual_window,
            args.history_window,
        )
    elif args.method == "tsb":
        p50_base, p90_base, p95_base, p99_base = tsb_forecast(
            count_values,
            args.alpha,
            args.history_window,
        )
    elif args.method == "hazard-hurdle":
        p50_base, p90_base, p95_base, p99_base = 0.0, 0.0, 0.0, 0.0
    elif args.method == "fip-fourier":
        p50_base, p90_base, p95_base, p99_base = 0.0, 0.0, 0.0, 0.0
    elif args.method == "burst-localized":
        p50_base, p90_base, p95_base, p99_base = 0.0, 0.0, 0.0, 0.0
    else:
        raise ValueError(f"unsupported method: {args.method}")

    rows = []
    for step in range(1, args.horizon + 1):
        window = last_window + step
        p_active = 1.0
        if args.method == "burst-localized":
            p50_count, p90_count, p95_count, p99_count = burst_localized_forecast(
                counts,
                window,
                args.alpha,
                args.burst_threshold,
                args.burst_period_windows,
                args.burst_width_windows,
                args.background_count,
            )
        elif args.method == "hazard-hurdle":
            p50_count, p90_count, p95_count, p99_count, p_active = hazard_hurdle_forecast(
                count_values,
                args.alpha,
                args.history_window,
                horizon_step=step,
            )
        elif args.method == "fip-fourier":
            p50_count, p90_count, p95_count, p99_count = fip_fourier_forecast(
                count_values,
                horizon_step=step,
                local_window=max(60, args.history_window),
                harmonics=10,
                residual_window=args.residual_window,
            )
        else:
            p50_count = p50_base
            p90_count = p90_base
            p95_count = p95_base
            p99_count = p99_base
        rows.append(
            {
                "workflow_name": args.workflow,
                "method": args.method,
                "window": window,
                "window_start_ms": window * window_ms,
                "p_active": p_active,
                "p50_count": p50_count,
                "p90_count": p90_count,
                "p95_count": p95_count,
                "p99_count": p99_count,
                "ceil_p50_count": ceil_count(p50_count),
                "ceil_p90_count": ceil_count(p90_count),
                "ceil_p95_count": ceil_count(p95_count),
                "ceil_p99_count": ceil_count(p99_count),
                "alloc_p50_count": alloc_count(p50_count, args.activation_threshold),
                "alloc_p90_count": alloc_count(p90_count, args.activation_threshold),
                "alloc_p95_count": alloc_count(p95_count, args.activation_threshold),
                "alloc_p99_count": alloc_count(p99_count, args.activation_threshold),
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

