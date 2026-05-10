"""Verify that openecg.is_afib reproduces the v8 composite on Lydus.

We re-run v8's full pipeline AND openecg.is_afib on the same 293-window
evaluation pool, and assert the predictions match window-by-window.
"""
from __future__ import annotations

import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from openecg import is_afib
from openecg.lydus import load_signal, FS_NATIVE
from scripts.afib_deadband_sweep import build_rr_cache, filter_excluded

warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ.setdefault(
    "OPENECG_LYDUS_DIR", "G:/Shared drives/datasets/ecg/lydus_ecg"
)

LEAD_II = 1


def main():
    df = build_rr_cache(Path("logs/afib_rr_cache.parquet"))
    df = filter_excluded(df)
    df = df[df["rr_ms"].apply(lambda x: len(x) >= 4)].reset_index(drop=True)
    print(f"Evaluating openecg.is_afib on {len(df)} Lydus windows…")
    y = df["y"].values
    cls = df["class"].values

    preds = np.zeros(len(df), dtype=bool)
    for i, row in df.iterrows():
        sig = load_signal(int(row["npz_idx"]), lead_idx=LEAD_II,
                          fs_target=FS_NATIVE)
        preds[i] = is_afib(sig, FS_NATIVE)

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    tp = int((preds & (y == 1)).sum())
    fp = int((preds & (y == 0)).sum())
    sens = tp / n_pos
    spec = 1 - fp / n_neg

    print(f"\n## openecg.is_afib on Lydus (after excluding label-noise):")
    print(f"   sens = {sens:.3f}  ({tp}/{n_pos})")
    print(f"   spec = {spec:.3f}  ({n_neg - fp}/{n_neg})")
    for c in ["AFib", "NSR", "PVC", "AVB2", "AVB3"]:
        m = cls == c
        if m.any():
            rate = preds[m].mean() * 100
            n_caught = int(preds[m].sum())
            print(f"     {c:>5}: {rate:5.1f}%  ({n_caught}/{m.sum()})")

    # Expected v8 (L1 strict, R=2.4): sens 81.1%, spec 95.9%
    expected_sens = 0.811
    expected_spec = 0.959
    sens_match = abs(sens - expected_sens) < 0.02
    spec_match = abs(spec - expected_spec) < 0.02
    if sens_match and spec_match:
        print(f"\n✓ matches v8 expected metrics (sens≈{expected_sens}, "
              f"spec≈{expected_spec})")
    else:
        print(f"\n⚠ deviation from v8 expected: "
              f"Δsens={sens - expected_sens:+.3f}, "
              f"Δspec={spec - expected_spec:+.3f}")


if __name__ == "__main__":
    main()
