"""Principled sweep: apply CP-ACI calibration and Hedge selector over all
5 apps x 2 window sizes, using base detail CSVs from the prior sweeps.

For each (app, win):
  1. Apply CP-ACI to each base family's detail CSV ->
     data/entry_forecasts/multiapp_<app>_<win>s_cp_<family>/.
  2. Run Hedge over the combined bank (base + CP) ->
     data/entry_forecasts/multiapp_<app>_<win>s_hedge/.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


APPS = ["periodic_dense", "periodic_sparse", "bursty_dense", "bursty_sparse", "drift"]
WORKFLOWS = {
    "periodic_dense": "sebs_video",
    "periodic_sparse": "civic_alert_flow",
    "bursty_dense": "sebs_video",
    "bursty_sparse": "civic_alert_flow",
    "drift": "spoken_dialog_flow",
}
WINDOWS_SEC = [5, 2]
FAMILY_DETAIL = {
    "classical": "{wf}_entry_classical_compare_detail.csv",
    "ml": "{wf}_entry_ml_compare_detail.csv",
    "twostage": "{wf}_entry_twostage_compare_detail.csv",
    "pointprocess": "{wf}_entry_pointprocess_compare_detail.csv",
}


def run_cmd(cmd: list[str], label: str) -> tuple[bool, float, str]:
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    ok = res.returncode == 0
    msg = "" if ok else (res.stderr or res.stdout or "").splitlines()[-1][:200]
    return ok, dt, msg


def run_cp(app: str, win: int) -> list[Path]:
    wf = WORKFLOWS[app]
    out_paths: list[Path] = []
    for family, fname_template in FAMILY_DETAIL.items():
        src = Path(f"data/entry_forecasts/multiapp_{app}_{win}s_{family}/{fname_template.format(wf=wf)}")
        if not src.exists():
            print(f"  skip cp:{family} (missing {src.name})", flush=True)
            continue
        out_dir = Path(f"data/entry_forecasts/multiapp_{app}_{win}s_cp_{family}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{wf}_entry_cp_aci_{family}_detail.csv"
        cmd = [
            sys.executable, "-m", "runner.stage2_forecastor.apply_conformal_calibration",
            "--detail", str(src),
            "--out", str(out_file),
            "--method", "aci",
            "--warmup", "30",
            "--gamma", "0.05",
        ]
        ok, dt, msg = run_cmd(cmd, f"cp-{family}")
        tag = "OK" if ok else "FAIL"
        print(f"    [cp-{family:12s}] {tag} {dt:5.1f}s  {msg}", flush=True)
        if ok:
            out_paths.append(out_file)
    return out_paths


def run_hedge(app: str, win: int, cp_paths: list[Path]) -> bool:
    wf = WORKFLOWS[app]
    base_paths: list[Path] = []
    for family, fname_template in FAMILY_DETAIL.items():
        src = Path(f"data/entry_forecasts/multiapp_{app}_{win}s_{family}/{fname_template.format(wf=wf)}")
        if src.exists():
            base_paths.append(src)
    bank = base_paths + cp_paths
    if not bank:
        print(f"  skip hedge (no banks)", flush=True)
        return False
    out_dir = Path(f"data/entry_forecasts/multiapp_{app}_{win}s_hedge")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{wf}_entry_hedge_compare_detail.csv"
    cmd = [
        sys.executable, "-m", "runner.stage2_forecastor.apply_hedge_selector",
        "--detail", *[str(p) for p in bank],
        "--out", str(out_file),
        "--policies", "p50,p90,p95",
        "--under-cost", "10",
        "--over-cost", "1",
    ]
    ok, dt, msg = run_cmd(cmd, "hedge")
    tag = "OK" if ok else "FAIL"
    print(f"    [hedge {len(bank):2d} experts] {tag} {dt:5.1f}s  {msg}", flush=True)
    return ok


def summarize_cp_and_hedge(app: str, win: int) -> None:
    """For each newly created detail, build a summary matching the standard format."""
    import pandas as pd
    from runner.stage2_forecastor.compare_entry_ml_forecasts import summarize

    wf = WORKFLOWS[app]
    candidate_paths = [
        Path(f"data/entry_forecasts/multiapp_{app}_{win}s_cp_{fam}/{wf}_entry_cp_aci_{fam}_detail.csv")
        for fam in FAMILY_DETAIL
    ] + [
        Path(f"data/entry_forecasts/multiapp_{app}_{win}s_hedge/{wf}_entry_hedge_compare_detail.csv")
    ]
    window_ms = win * 1000
    for path in candidate_paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        summary = summarize(df, window_ms)
        summary_path = path.with_name(path.stem.replace("_detail", "_summary") + ".csv")
        summary.to_csv(summary_path, index=False)


def main() -> None:
    jobs = [(app, w) for app in APPS for w in WINDOWS_SEC]
    total = len(jobs)
    for i, (app, w) in enumerate(jobs, 1):
        print(f"[{i}/{total}] {app:18s} {w}s", flush=True)
        cp_paths = run_cp(app, w)
        run_hedge(app, w, cp_paths)
        summarize_cp_and_hedge(app, w)
    print("principled sweep done")


if __name__ == "__main__":
    main()
