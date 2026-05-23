"""Robustness check: re-run principled methods on bursty traces resampled
at time-compression=10 (real-deployment intensity), instead of 30.

Generates 50:50 time splits, then runs classical / pointprocess / cp /
hedge on bursty_dense and bursty_sparse at 5s and 2s windows.
"""

from __future__ import annotations

import math
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


APPS = {
    "bursty_dense": "configs/visual_qa_flow.yaml",
    "bursty_sparse": "configs/civic_alert_flow.yaml",
}
WINDOWS_SEC = [5, 2]
DATA_ROOT = Path("data/azure_multiapp_smooth10x")
OUT_ROOT = Path("data/entry_forecasts")
TRAIN_RATIO = 0.5


def make_split(label: str) -> Path:
    trace_path = DATA_ROOT / label / f"entry_trace_{label}.csv"
    trace = pd.read_csv(trace_path)
    entry = trace[(trace["stage_name"] == "__entry__") & (trace["status"] == "ok")]
    entry = entry[["request_id", "entry_ts_ms"]].drop_duplicates().sort_values(["entry_ts_ms", "request_id"]).reset_index(drop=True)
    start, end = int(entry["entry_ts_ms"].min()), int(entry["entry_ts_ms"].max())
    cutoff = start + int(math.floor((end - start) * TRAIN_RATIO))
    split_map = entry.copy()
    split_map["split"] = "test"
    split_map.loc[split_map["entry_ts_ms"] <= cutoff, "split"] = "train"
    split_map["split_strategy"] = "time"
    split_map["split_cutoff_ms"] = cutoff
    splits_dir = DATA_ROOT / label / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_path = splits_dir / f"entry_{label}_split.csv"
    split_map.to_csv(split_path, index=False)
    print(f"  split {label}: train={int((split_map['split']=='train').sum())} test={int((split_map['split']=='test').sum())}")
    return split_path


def run_module(module: str, app: str, win: int, suffix: str, cfg: str, extra: list[str] | None = None) -> tuple[bool, float, str]:
    out_dir = OUT_ROOT / f"smooth10x_{app}_{win}s_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    trace = str(DATA_ROOT / app / f"entry_trace_{app}.csv")
    split = str(DATA_ROOT / app / "splits" / f"entry_{app}_split.csv")
    cmd = [
        sys.executable, "-m", f"runner.stage2_forecastor.{module}",
        "--trace", trace, "--workflow-config", cfg, "--split-map", split,
        "--window-sec", str(win), "--out-dir", str(out_dir), "--write-detail",
    ]
    if extra:
        cmd.extend(extra)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    ok = r.returncode == 0
    err = "" if ok else (r.stderr or r.stdout or "").splitlines()[-1][:200]
    return ok, dt, err


def workflow_name(cfg: str) -> str:
    return Path(cfg).stem


def run_cp_for(app: str, win: int, src_suffix: str, dst_suffix: str, base_pattern: str) -> tuple[bool, float, str]:
    wf = workflow_name(APPS[app])
    src_detail = OUT_ROOT / f"smooth10x_{app}_{win}s_{src_suffix}" / base_pattern.format(wf=wf)
    out_dir = OUT_ROOT / f"smooth10x_{app}_{win}s_{dst_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{wf}_entry_cp_aci_{dst_suffix.replace('cp_','')}_detail.csv"
    cmd = [sys.executable, "-m", "runner.stage2_forecastor.apply_conformal_calibration",
           "--detail", str(src_detail), "--out", str(out), "--method", "aci"]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    ok = r.returncode == 0
    err = "" if ok else (r.stderr or r.stdout or "").splitlines()[-1][:200]
    if ok:
        # produce a summary CSV by reusing the rolling summarize logic
        det = pd.read_csv(out)
        if "window" not in det.columns and "target_window" in det.columns:
            det = det.rename(columns={"target_window": "window"})
        from runner.stage2_forecastor.compare_entry_rolling_forecasts import summarize as roll_summarize
        det2 = det.copy()
        if "origin_window" not in det2.columns:
            det2["origin_window"] = det2["window"]
        s = roll_summarize(det2, win * 1000)
        s.to_csv(out_dir / f"{wf}_entry_cp_aci_{dst_suffix.replace('cp_','')}_summary.csv", index=False)
    return ok, dt, err


def run_hedge_for(app: str, win: int) -> tuple[bool, float, str]:
    wf = workflow_name(APPS[app])
    details = []
    for src_suffix, pat in [
        ("classical",    f"{wf}_entry_classical_compare_detail.csv"),
        ("pointprocess", f"{wf}_entry_pointprocess_compare_detail.csv"),
        ("cp_classical", f"{wf}_entry_cp_aci_classical_detail.csv"),
        ("cp_pp",        f"{wf}_entry_cp_aci_pointprocess_detail.csv"),
    ]:
        p = OUT_ROOT / f"smooth10x_{app}_{win}s_{src_suffix}" / pat
        if p.exists():
            details.append(str(p))
    if not details:
        return False, 0.0, "no detail CSVs available"
    out_dir = OUT_ROOT / f"smooth10x_{app}_{win}s_hedge"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{wf}_entry_hedge_compare_detail.csv"
    cmd = [sys.executable, "-m", "runner.stage2_forecastor.apply_hedge_selector",
           "--detail", *details, "--out", str(out)]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    ok = r.returncode == 0
    err = "" if ok else (r.stderr or r.stdout or "").splitlines()[-1][:200]
    if ok:
        det = pd.read_csv(out)
        from runner.stage2_forecastor.compare_entry_rolling_forecasts import summarize as roll_summarize
        det2 = det.copy()
        if "origin_window" not in det2.columns:
            det2["origin_window"] = det2["window"]
        s = roll_summarize(det2, win * 1000)
        s.to_csv(out_dir / f"{wf}_entry_hedge_compare_summary.csv", index=False)
    return ok, dt, err


def main() -> None:
    print("== generating 50:50 time splits ==")
    for label in APPS:
        make_split(label)

    print("\n== running base methods ==")
    for app, cfg in APPS.items():
        for win in WINDOWS_SEC:
            for suffix, module in [
                ("classical", "compare_entry_classical_forecasts"),
                ("pointprocess", "compare_entry_pointprocess_forecasts"),
            ]:
                ok, dt, err = run_module(module, app, win, suffix, cfg)
                print(f"  [{app:14s} {win}s {suffix:14s}] {'OK' if ok else 'FAIL'} {dt:5.1f}s  {err}")

    print("\n== running CP-ACI on classical and pointprocess ==")
    for app, _ in APPS.items():
        wf = workflow_name(APPS[app])
        for win in WINDOWS_SEC:
            ok, dt, err = run_cp_for(app, win, "classical", "cp_classical",
                                     f"{wf}_entry_classical_compare_detail.csv")
            print(f"  [{app:14s} {win}s cp_classical ] {'OK' if ok else 'FAIL'} {dt:5.1f}s  {err}")
            ok, dt, err = run_cp_for(app, win, "pointprocess", "cp_pp",
                                     f"{wf}_entry_pointprocess_compare_detail.csv")
            print(f"  [{app:14s} {win}s cp_pp        ] {'OK' if ok else 'FAIL'} {dt:5.1f}s  {err}")

    print("\n== running Hedge over (classical + pointprocess + cp) ==")
    for app, _ in APPS.items():
        for win in WINDOWS_SEC:
            ok, dt, err = run_hedge_for(app, win)
            print(f"  [{app:14s} {win}s hedge        ] {'OK' if ok else 'FAIL'} {dt:5.1f}s  {err}")

    print("\nrobustness sweep done")


if __name__ == "__main__":
    main()
