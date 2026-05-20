"""Point-process entry forecasters (homogeneous Poisson + Hawkes-exp) sweep."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


APPS: dict[str, str] = {
    "periodic_dense": "configs/sebs_video.yaml",
    "periodic_sparse": "configs/civic_alert_flow.yaml",
    "bursty_dense": "configs/sebs_video.yaml",
    "bursty_sparse": "configs/civic_alert_flow.yaml",
    "drift": "configs/spoken_dialog_flow.yaml",
}
WINDOWS_SEC = [5, 2]


def run_one(app: str, cfg: str, win: int) -> tuple[bool, float, str]:
    out_dir = Path(f"data/entry_forecasts/multiapp_{app}_{win}s_pointprocess")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "runner.stage2_forecastor.compare_entry_pointprocess_forecasts",
        "--trace", f"data/azure_multiapp/{app}/entry_trace_{app}.csv",
        "--workflow-config", cfg,
        "--split-map", f"data/azure_multiapp/{app}/splits/entry_{app}_split.csv",
        "--window-sec", str(win),
        "--methods", "homogeneous-poisson,hawkes-exp",
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
    jobs = [(app, cfg, w) for app, cfg in APPS.items() for w in WINDOWS_SEC]
    total = len(jobs)
    for i, (app, cfg, w) in enumerate(jobs, 1):
        ok, dt, msg = run_one(app, cfg, w)
        tag = "OK" if ok else "FAIL"
        print(f"[{i}/{total}] {tag} {app:18s} {w}s pointprocess {dt:6.1f}s  {msg}", flush=True)
    print("point-process sweep done")


if __name__ == "__main__":
    main()
