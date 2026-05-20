"""Sweep all non-deep entry forecasters over the 5 multiapp traces.

Runs the four method-family scripts (classical, ml, twostage, histogram) for
each of the 5 apps at both 5 s and 2 s windows. Logs per-run timing and
return code; final summary printed at the end.
"""

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

WINDOWS_SEC: list[int] = [5, 2]

FAMILIES: list[tuple[str, str]] = [
    ("classical", "compare_entry_classical_forecasts"),
    ("ml", "compare_entry_ml_forecasts"),
    ("twostage", "compare_entry_twostage_forecasts"),
    ("histogram", "compare_entry_histogram_forecasts"),
]


def run_one(app: str, cfg: str, win: int, family: str, module: str) -> tuple[bool, float, str]:
    out_dir = Path(f"data/entry_forecasts/multiapp_{app}_{win}s_{family}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        f"runner.stage2_forecastor.{module}",
        "--trace",
        f"data/azure_multiapp/{app}/entry_trace_{app}.csv",
        "--workflow-config",
        cfg,
        "--split-map",
        f"data/azure_multiapp/{app}/splits/entry_{app}_split.csv",
        "--window-sec",
        str(win),
        "--out-dir",
        str(out_dir),
        "--write-detail",
    ]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    ok = res.returncode == 0
    msg = "" if ok else (res.stderr or res.stdout or "").splitlines()[-1][:200]
    return ok, dt, msg


def main() -> None:
    results: list[tuple[str, int, str, bool, float, str]] = []
    total = len(APPS) * len(WINDOWS_SEC) * len(FAMILIES)
    i = 0
    for app, cfg in APPS.items():
        for win in WINDOWS_SEC:
            for family, module in FAMILIES:
                i += 1
                ok, dt, msg = run_one(app, cfg, win, family, module)
                tag = "OK" if ok else "FAIL"
                print(f"[{i}/{total}] {tag} {app:18s} {win}s {family:9s} {dt:6.1f}s  {msg}", flush=True)
                results.append((app, win, family, ok, dt, msg))

    print("\n=== summary ===")
    ok_n = sum(1 for r in results if r[3])
    print(f"{ok_n}/{len(results)} succeeded")
    for app, win, family, ok, dt, msg in results:
        if not ok:
            print(f"FAIL  {app} {win}s {family}  {msg}")


if __name__ == "__main__":
    main()
