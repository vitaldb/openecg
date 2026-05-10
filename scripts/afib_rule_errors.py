"""Error analysis for the AFib rule tree.

Reload the features parquet, retrain the depth=3 tree, dump every
misclassified case with its full RR sequence + features, and bucket
errors by failure mode.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from openecg.qrs import detect_qrs
from openecg.lydus import load_signal, FS_NATIVE


os.environ.setdefault(
    "OPENECG_LYDUS_DIR", "G:/Shared drives/datasets/ecg/lydus_ecg"
)


FEAT_COLS = [
    "cv_rr", "rmssd_ms", "sarkar_fill", "origin_count",
    "abs_rho1_drr", "predictability", "poincare_clusters",
    "levene_p", "sign_flip", "pnn50", "rr_bimodality",
]


def rr_sequence(npz_idx: int) -> np.ndarray:
    sig = load_signal(npz_idx, lead_idx=1, fs_target=FS_NATIVE)
    peaks = detect_qrs(sig, FS_NATIVE)
    return np.diff(peaks) * (1000.0 / FS_NATIVE)


def main():
    df = pd.read_parquet("logs/afib_rule_features.parquet")
    print(f"Loaded {len(df)} windows")

    from sklearn.tree import DecisionTreeClassifier
    X = df[FEAT_COLS].fillna(df[FEAT_COLS].median()).values
    y = df["y"].values
    clf = DecisionTreeClassifier(
        max_depth=3, criterion="gini", min_samples_leaf=15,
        class_weight="balanced", random_state=42,
    )
    clf.fit(X, y)
    df = df.copy()
    df["pred"] = clf.predict(X)
    df["proba"] = clf.predict_proba(X)[:, 1]

    error_groups: dict[str, pd.DataFrame] = {}
    for cls in ["AFib", "NSR", "PVC", "AVB2", "AVB3"]:
        mask_cls = df["class"] == cls
        if cls == "AFib":
            errs = df[mask_cls & (df["pred"] == 0)]
        else:
            errs = df[mask_cls & (df["pred"] == 1)]
        error_groups[cls] = errs
        print(f"  {cls}: {len(errs)} errors / {mask_cls.sum()}")

    def fmt_rr(rr: np.ndarray) -> str:
        return "[" + " ".join(f"{int(round(v))}" for v in rr) + "]"

    out_lines: list[str] = []
    for cls, errs in error_groups.items():
        if len(errs) == 0:
            continue
        kind = "FN  (AFib → predicted non-AFib)" if cls == "AFib" \
            else f"FP  ({cls} → predicted AFib)"
        out_lines.append(f"\n========= {kind}   n={len(errs)} =========")
        for _, row in errs.iterrows():
            rr = rr_sequence(int(row["npz_idx"]))
            out_lines.append(
                f"\n  npz={int(row['npz_idx']):>6}  n_beats={int(row['n_beats']):>2}  "
                f"meanRR={row['mean_rr_ms']:.0f}ms  CV={row['cv_rr']:.2f}  "
                f"pnn50={row['pnn50']:.2f}  rmssd={row['rmssd_ms']:.0f}  "
                f"sarkar={row['sarkar_fill']:.3f}  origin={row['origin_count']:.2f}  "
                f"|ρ1(dRR)|={row['abs_rho1_drr']:.2f}  pred_proba={row['proba']:.2f}"
            )
            out_lines.append(f"     RR(ms) = {fmt_rr(rr)}")
            dr = np.diff(rr)
            out_lines.append(f"     dRR    = {fmt_rr(dr)}")

    report = "\n".join(out_lines)
    print(report)
    Path("logs/afib_rule_errors.txt").write_text(report, encoding="utf-8")
    print("\nSaved → logs/afib_rule_errors.txt")


if __name__ == "__main__":
    main()
