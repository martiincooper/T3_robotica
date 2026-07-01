"""Batch runner: load HDF5 dataset → run perception → write CSV results."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from .g2t_perception.pipeline import GeometricEstimator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="path to dataset.h5")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    rows = []
    est = GeometricEstimator(cfg)

    with h5py.File(args.dataset, "r") as f:
        angles = f["scan/angles"][:]
        ranges = f["scan/ranges"][:]
        times = f["scan/time"][:]
        for i, t in enumerate(times):
            res = est.step(ranges[i], angles, float(t))
            rows.append({
                "time": t,
                "psi1_raw": res.psi1_raw,
                "psi2_raw": res.psi2_raw,
                "psi1_smoothed": res.psi1_smoothed,
                "psi2_smoothed": res.psi2_smoothed,
                "num_clusters": res.num_clusters,
                "num_lines": res.num_lines,
            })
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[run_perception] wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
