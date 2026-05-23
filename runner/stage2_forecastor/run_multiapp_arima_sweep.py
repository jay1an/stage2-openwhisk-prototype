"""ARIMA (p,d,q) order ablation on the 5 multiapp traces."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


APPS: dict[str, str] = {
    "periodic_dense": "configs/visual_qa_flow.yaml",
    "periodic_sparse": "configs/civic_alert_flow.yaml",
    "bursty_dense": "configs/visual_qa_flow.yaml",
    "bursty_sparse": "configs/civic_alert_flow.yaml",
    "drift": "configs/spoken_dialog_flow.yaml",
}
WINDOWS_SEC = [5, 2]
ORDERS = ["1,0,0", "0,0,1", "1,0,1", "2,0,1", "1,0,2", "2,0,2", "0,1,1", "1,1,1", "2,1,1", "1,1,2"]


def run_one(app: str, cfg: str, win: int, order: str) -> tuple[bool, float, str]:
    order_tag = order.replace(",", "")
    out_dir = Path(f"data/entry_forecasts/multiapp_{app}_{win}s_arima_{order_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "runner.stage2_forecastor.compare_entry_classical_forecasts",
        "--trace", f"data/azure_multiapp/{app}/entry_trace_{app}.csv",
        "--workflow-config", cfg,
        "--split-map", f"data/azure_multiapp/{app}/splits/entry_{app}_split.csv",
        "--window-sec", str(win),
        "--methods", "arima",
        "--arima-order", order,
        "--out-dir", str(out_dir),
        "--write-detail",
    ]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    ok = res.returncode == 0
    msg = "" if ok else (res.stderr or res.stdout or "").splitlines()[-1][:200]
    return ok, dt, msg


def main() -> None:
    jobs = [(app, cfg, w, o) for app, cfg in APPS.items() for w in WINDOWS_SEC for o in ORDERS]
    total = len(jobs)
    for i, (app, cfg, w, o) in enumerate(jobs, 1):
        ok, dt, msg = run_one(app, cfg, w, o)
        tag = "OK" if ok else "FAIL"
        print(f"[{i}/{total}] {tag} {app:18s} {w}s arima-{o:8s} {dt:5.1f}s  {msg}", flush=True)
    print("ARIMA sweep done")


if __name__ == "__main__":
    main()
