#!/usr/bin/env python3
"""Entry cold probability estimator.

This module connects an arrival-count forecast distribution D_t with the
currently available entry warm slots A_t.  The planner-facing probability is

    E[(D_t - A_t)+] / E[D_t]

which is the expected fraction of arrivals that cannot find a warm entry
container.  The same function can be fed with live /poolState rows where
``available = free + warming``.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from runner.stage2_forecastor.canon_count_forecaster import (
    LEAD_SEC,
    load_inputs,
    make_count_data,
    resolve,
    simulate_forecaster,
    validate_split,
    fit_count_model,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "reports" / "step5_entry_cold_estimator"
DEFAULT_LAG_K = 10
DEFAULT_KEEPALIVE_S = 10.0
REAL_CANON_ENTRY_COLD = {
    "premium": 0.072,
    "free": 0.128,
}


def normalize_action_for_poolstate(action_name: object) -> str:
    raw = str(action_name or "")
    if "/" in raw:
        return raw.rstrip("/").split("/")[-1]
    return raw


def available_from_poolstate(
    poolstate_rows: Iterable[Mapping[str, Any]] | Mapping[Any, Mapping[str, Any]],
    *,
    include_normalized_action: bool = True,
) -> dict[tuple[str, int], int]:
    """Return available warm slots keyed by ``(action, memoryMB)``.

    The invoker route returns rows with ``free`` and ``warming`` counts.  Busy
    containers are deliberately excluded because an entry arrival cannot use
    them immediately.
    """

    rows: Iterable[Mapping[str, Any]]
    if isinstance(poolstate_rows, Mapping):
        rows = poolstate_rows.values()
    else:
        rows = poolstate_rows

    available: dict[tuple[str, int], int] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        action = str(row.get("action", ""))
        if not action:
            continue
        try:
            memory = int(float(row.get("memoryMB")))
        except (TypeError, ValueError):
            continue
        free = int(float(row.get("free", 0) or 0))
        warming = int(float(row.get("warming", 0) or 0))
        value = max(0, free) + max(0, warming)
        keys = [(action, memory)]
        normalized = normalize_action_for_poolstate(action)
        if include_normalized_action and normalized != action:
            keys.append((normalized, memory))
        for key in keys:
            available[key] = max(available.get(key, 0), value)
    return available


def _broadcast_inputs(
    mean: float | np.ndarray,
    available: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    mean_arr = np.asarray(mean, dtype=float)
    avail_arr = np.asarray(available, dtype=float)
    scalar = mean_arr.ndim == 0 and avail_arr.ndim == 0
    mean_b, avail_b = np.broadcast_arrays(mean_arr, avail_arr)
    if np.any(~np.isfinite(mean_b)) or np.any(mean_b < 0):
        raise ValueError("mean/lambda values must be finite and >= 0")
    if np.any(~np.isfinite(avail_b)) or np.any(avail_b < 0):
        raise ValueError("available values must be finite and >= 0")
    return mean_b.astype(float), avail_b.astype(float), scalar


def _expected_shortage_scalar(
    mean: float,
    available: float,
    *,
    dist: str,
    alpha: float,
    tail_prob: float,
) -> float:
    if mean <= 0.0:
        return 0.0
    if available <= 0.0:
        return float(mean)

    if dist == "poisson" or alpha <= 1e-12:
        rv = poisson(mean)
        variance = mean
    elif dist == "nb":
        size = 1.0 / alpha
        prob = size / (size + mean)
        rv = nbinom(size, prob)
        variance = mean + alpha * mean * mean
    else:
        raise ValueError("dist must be 'nb' or 'poisson'")

    q = rv.ppf(max(0.0, min(1.0 - tail_prob, 0.999999999999)))
    if not np.isfinite(q):
        q = mean + 12.0 * math.sqrt(max(variance, 1e-9))
    upper = int(max(math.ceil(q), math.ceil(available) + 1, 1))
    xs = np.arange(0, upper + 1, dtype=float)
    shortage = np.maximum(xs - available, 0.0)
    return float(np.sum(shortage * rv.pmf(xs)))


def expected_entry_cold_count(
    mean: float | np.ndarray,
    available: float | np.ndarray,
    *,
    dist: str = "nb",
    alpha: float = 0.0,
    tail_prob: float = 1e-12,
) -> float | np.ndarray:
    """Expected unserved arrivals ``E[(D - available)+]``."""

    mean_b, avail_b, scalar = _broadcast_inputs(mean, available)
    if dist not in {"nb", "poisson"}:
        raise ValueError("dist must be 'nb' or 'poisson'")
    if alpha < 0 or not math.isfinite(float(alpha)):
        raise ValueError("alpha must be finite and >= 0")
    out = np.empty_like(mean_b, dtype=float)
    for index in np.ndindex(mean_b.shape):
        out[index] = _expected_shortage_scalar(
            float(mean_b[index]),
            float(avail_b[index]),
            dist=dist,
            alpha=float(alpha),
            tail_prob=float(tail_prob),
        )
    return float(out) if scalar else out


def p_entry_cold(
    mean: float | np.ndarray,
    available: float | np.ndarray,
    *,
    dist: str = "nb",
    alpha: float = 0.0,
    tail_prob: float = 1e-12,
) -> float | np.ndarray:
    """Expected entry-cold fraction for each forecast point."""

    mean_b, _, scalar = _broadcast_inputs(mean, available)
    shortage = np.asarray(
        expected_entry_cold_count(
            mean_b,
            available,
            dist=dist,
            alpha=alpha,
            tail_prob=tail_prob,
        ),
        dtype=float,
    )
    out = np.divide(
        shortage,
        mean_b,
        out=np.zeros_like(shortage, dtype=float),
        where=mean_b > 0,
    )
    out = np.clip(out, 0.0, 1.0)
    return float(out) if scalar else out


def estimate_entry_cold_rate(
    mean: np.ndarray,
    available: np.ndarray,
    *,
    dist: str = "nb",
    alpha: float = 0.0,
) -> float:
    """Aggregate expected entry-cold fraction over a horizon."""

    mean_arr, avail_arr, _ = _broadcast_inputs(mean, available)
    shortage = np.asarray(
        expected_entry_cold_count(mean_arr, avail_arr, dist=dist, alpha=alpha),
        dtype=float,
    )
    denominator = float(np.sum(mean_arr))
    if denominator <= 0.0:
        return 0.0
    return float(np.clip(float(np.sum(shortage)) / denominator, 0.0, 1.0))


def solve_effective_available(
    mean: np.ndarray,
    target_rate: float,
    *,
    dist: str = "nb",
    alpha: float = 0.0,
    tolerance: float = 1e-4,
) -> float:
    """Find a scalar available value that matches a target cold fraction."""

    if target_rate < 0.0 or target_rate > 1.0 or not math.isfinite(target_rate):
        raise ValueError("target_rate must be finite and in [0, 1]")
    mean_arr = np.asarray(mean, dtype=float)
    if mean_arr.sum() <= 0.0:
        return 0.0
    lo = 0.0
    hi = max(1.0, float(np.nanmax(mean_arr)) + 1.0)
    while estimate_entry_cold_rate(
        mean_arr,
        np.full_like(mean_arr, hi),
        dist=dist,
        alpha=alpha,
    ) > target_rate and hi < 1_000_000.0:
        hi *= 2.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        rate = estimate_entry_cold_rate(
            mean_arr,
            np.full_like(mean_arr, mid),
            dist=dist,
            alpha=alpha,
        )
        if abs(rate - target_rate) <= tolerance:
            return mid
        if rate > target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _empirical_bin_overflow(actual: np.ndarray, available: np.ndarray) -> float:
    denominator = float(np.sum(actual))
    if denominator <= 0.0:
        return 0.0
    return float(np.sum(np.maximum(np.asarray(actual) - np.asarray(available), 0.0)) / denominator)


def _class_counts_on_eval_bins(
    canon: pd.DataFrame,
    *,
    edges: np.ndarray,
    eval_indices: np.ndarray,
    eval_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if "slo_class" not in canon.columns:
        return out
    selected_indices = eval_indices[eval_mask]
    for slo_class, group in canon.groupby("slo_class"):
        counts, _ = np.histogram(group["source_start_s"].to_numpy(float), bins=edges)
        out[str(slo_class)] = counts[selected_indices].astype(float)
    return out


def run_offline_validation(
    *,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    lag_k: int = DEFAULT_LAG_K,
) -> dict[str, pd.DataFrame]:
    out = resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    arrivals, canon = load_inputs()
    split = validate_split(arrivals, canon)
    arrival_s = arrivals["arrival_s"].to_numpy(float)
    canon_times = canon["source_start_s"].to_numpy(float)
    canon_start = split["canon_start_day"] * 86400.0
    canon_end = split["canon_end_day"] * 86400.0
    data60 = make_count_data(arrival_s, canon_start, canon_end, bin_s=60, lag_k=lag_k)
    model60 = fit_count_model(data60)

    selfcheck_rows: list[dict[str, Any]] = []
    arms: list[tuple[str, np.ndarray]] = [
        ("constant_1", np.ones_like(model60.actual)),
        ("constant_2", np.full_like(model60.actual, 2.0)),
        ("constant_3", np.full_like(model60.actual, 3.0)),
        ("hgb_q90_60s", np.ceil(model60.q90_pred)),
        ("poisson_k90_60s", np.ceil(model60.poisson_k90)),
        ("nb_k90_60s", np.ceil(model60.nb_k90)),
    ]
    for name, available in arms:
        est_nb = estimate_entry_cold_rate(
            model60.mean_pred,
            available,
            dist="nb",
            alpha=model60.nb_alpha,
        )
        est_poisson = estimate_entry_cold_rate(
            model60.mean_pred,
            available,
            dist="poisson",
            alpha=0.0,
        )
        empirical = _empirical_bin_overflow(model60.actual, available)
        sim_cold, sim_seconds, sim_warmups = simulate_forecaster(
            canon_times,
            model60.starts,
            available,
            bin_s=60,
            keepalive_s=DEFAULT_KEEPALIVE_S,
        )
        sim_rate = sim_cold / max(1, len(canon_times))
        selfcheck_rows.append(
            {
                "arm": name,
                "mean_available": float(np.mean(available)),
                "est_nb_rate": est_nb,
                "est_poisson_rate": est_poisson,
                "empirical_bin_overflow_rate": empirical,
                "simulate_forecaster_cold_rate": sim_rate,
                "simulate_forecaster_cold": int(sim_cold),
                "simulate_container_seconds": float(sim_seconds),
                "simulate_warmups": int(sim_warmups),
                "abs_nb_minus_sim": abs(est_nb - sim_rate),
                "abs_poisson_minus_sim": abs(est_poisson - sim_rate),
            }
        )

    class_count_by_name = _class_counts_on_eval_bins(
        canon,
        edges=data60.edges,
        eval_indices=data60.indices,
        eval_mask=data60.eval_mask,
    )
    anchor_rows: list[dict[str, Any]] = []
    for slo_class, target_rate in REAL_CANON_ENTRY_COLD.items():
        class_mean = class_count_by_name.get(slo_class)
        if class_mean is None or float(np.sum(class_mean)) <= 0.0:
            continue
        for dist, alpha in [("poisson", 0.0), ("nb", model60.nb_alpha)]:
            effective_available = solve_effective_available(
                class_mean,
                target_rate,
                dist=dist,
                alpha=alpha,
            )
            reproduced = estimate_entry_cold_rate(
                class_mean,
                np.full_like(class_mean, effective_available),
                dist=dist,
                alpha=alpha,
            )
            anchor_rows.append(
                {
                    "slo_class": slo_class,
                    "target_real_entry_cold_rate": target_rate,
                    "dist": dist,
                    "alpha": float(alpha),
                    "effective_available": effective_available,
                    "reproduced_rate": reproduced,
                    "canon_arrivals": int(np.sum(class_mean)),
                    "mean_bin_count": float(np.mean(class_mean)),
                    "p95_bin_count": float(np.percentile(class_mean, 95)),
                    "max_bin_count": float(np.max(class_mean)),
                }
            )

    poolstate_example = [
        {
            "action": "/guest/wf_civic_detect_object_3072",
            "memoryMB": 3072,
            "free": 2,
            "busy": 3,
            "warming": 1,
            "oldestFreeIdleMs": 1200,
        },
        {
            "action": "wf_civic_detect_object_1280",
            "memoryMB": 1280,
            "free": 4,
            "busy": 0,
            "warming": 2,
        },
    ]
    available_example = available_from_poolstate(poolstate_example)

    selfcheck = pd.DataFrame(selfcheck_rows)
    anchor = pd.DataFrame(anchor_rows)
    selfcheck.to_csv(out / "selfcheck.csv", index=False)
    anchor.to_csv(out / "anchor_effective_available.csv", index=False)
    report = _render_report(
        split=split,
        model60=model60,
        selfcheck=selfcheck,
        anchor=anchor,
        available_example=available_example,
    )
    (out / "report.md").write_text(report, encoding="utf-8")
    return {
        "selfcheck": selfcheck,
        "anchor": anchor,
        "report": pd.DataFrame(),
    }


def _render_report(
    *,
    split: Mapping[str, Any],
    model60: Any,
    selfcheck: pd.DataFrame,
    anchor: pd.DataFrame,
    available_example: dict[tuple[str, int], int],
) -> str:
    lines = [
        "# Step 5 Entry Cold Estimator",
        "",
        "## Reuse Confirmation",
        "- `canon_count_forecaster`: confirmed usable for CANON 12/2 split and 60s HGBR/NB forecasts.",
        "- `simulate_forecaster`: reused for event-level self-check; it includes warm-container carry-over, so it is not identical to the point-bin estimator.",
        "- `PrewarmPool._query_pool_state`: confirmed fields `free`, `busy`, `warming`, `memoryMB`; estimator uses `available = free + warming`.",
        "",
        "## Data",
        f"- Train n={split['train_n']}; CANON n={split['canon_n']}; span={split['canon_span_h']:.2f}h.",
        f"- Train ends before CANON: `{split['train_ends_before_canon']}`.",
        f"- 60s forecaster: eval bins={len(model60.actual)}, actual_sum={float(np.sum(model60.actual)):.0f}, NB alpha={model60.nb_alpha:.6f}.",
        "",
        "## /poolState Availability Example",
        "",
        "| key | available |\n|---|---:|",
    ]
    for key, value in sorted(available_example.items(), key=lambda item: str(item[0])):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(
        [
            "",
            "## Self-Check",
            "",
            selfcheck.to_markdown(index=False),
            "",
            "## Real-Rate Anchor",
            "",
            "Targets use the measured CANON oracle-static entry cold rates supplied by the experiment notes: premium 7.2%, free 12.8%.",
            "",
            anchor.to_markdown(index=False) if not anchor.empty else "_No slo_class anchor rows available._",
            "",
            "## Interpretation",
            "",
            "- The estimator is a point-in-time shortage model: `E[(D-A)+]/E[D]`.",
            "- Event-level simulation can be lower because warm containers carry over between bins.",
            "- The anchor table turns observed real entry-cold rates into an effective available-slot explanation for planner `p_entry_cold`.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--lag-k", type=int, default=DEFAULT_LAG_K)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_offline_validation(out_dir=args.out_dir, lag_k=args.lag_k)
    selfcheck = result["selfcheck"]
    anchor = result["anchor"]
    print("SELFCHECK")
    print(selfcheck.to_string(index=False))
    print("\nANCHOR")
    print(anchor.to_string(index=False))
    print(f"\nWrote {resolve(args.out_dir) / 'report.md'}")


if __name__ == "__main__":
    main()
