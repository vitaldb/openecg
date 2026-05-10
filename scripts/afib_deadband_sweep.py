"""Deadband-sweep over single-metric AFib discrimination.

Idea (user request): RR differences below T ms are physiological sinus arrhythmia,
not arrhythmic chaos. All magnitude-based metrics should treat |ΔRR| < T as 0.
T itself is a hyperparameter — sweep over it.
"""
from __future__ import annotations

import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import duckdb

from openecg.lydus import load_signal, FS_NATIVE
from openecg.qrs import detect_qrs

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="An input array is constant")


os.environ.setdefault(
    "OPENECG_LYDUS_DIR", "G:/Shared drives/datasets/ecg/lydus_ecg"
)
LYDUS_DIR = os.environ["OPENECG_LYDUS_DIR"]

RNG = np.random.default_rng(20260511)
LEAD_II_IDX = 1
N_PER_CLASS = 80

T_GRID = [0, 5, 10, 15, 20, 30, 40, 50, 70]


# Lydus npz indices flagged as label-noise during v6 FN review (2026-05-12):
# all four were labelled "A.fib" but visual inspection of lead I/II/V1
# shows clear P-waves and a regular sinus skeleton with occasional APCs
# (npz=101420, 36019, 156875, 126526) or RR clusters atypical for AFib
# (npz=58269, 123285). Excluded from the evaluation pool until clinical
# chart review confirms or corrects the Lydus rhythm label.
EXCLUDED_NPZ: set[int] = {101420, 36019, 156875, 126526, 58269, 123285}


def filter_excluded(df):
    """Drop label-noise-suspect rows. Returns a copy of df."""
    keep = ~df["npz_idx"].astype(int).isin(EXCLUDED_NPZ)
    return df.loc[keep].reset_index(drop=True)


def sample_npz_indices() -> dict[str, list[int]]:
    con = duckdb.connect(str(Path(LYDUS_DIR) / "lydus_ecg.duckdb"), read_only=True)
    queries = {
        "AFib": ("SELECT npz_idx FROM records WHERE rhythm='A.fib' AND "
                 "premature_beat='Nonspecific' AND avb='NSP' AND npz_idx IS NOT NULL"),
        "NSR":  ("SELECT npz_idx FROM records WHERE rhythm='NSR' AND "
                 "premature_beat='Nonspecific' AND avb='NSP' AND npz_idx IS NOT NULL"),
        "PVC":  ("SELECT npz_idx FROM records WHERE premature_beat='VPC' AND "
                 "rhythm IN ('NSR','S.brady','S.tachy') AND avb='NSP' AND npz_idx IS NOT NULL"),
        "AVB2": ("SELECT npz_idx FROM records WHERE avb='2''AVB' AND "
                 "rhythm IN ('NSR','S.brady') AND npz_idx IS NOT NULL"),
        "AVB3": ("SELECT npz_idx FROM records WHERE avb='3''AVB' AND npz_idx IS NOT NULL"),
    }
    out = {}
    for cls, q in queries.items():
        rows = [int(r[0]) for r in con.execute(q).fetchall()]
        RNG.shuffle(rows)
        out[cls] = rows[:N_PER_CLASS]
    return out


def build_rr_cache(path: Path) -> pd.DataFrame:
    if path.exists():
        df_existing = pd.read_parquet(path)
        # ensure widths_ms column exists; otherwise rebuild
        if "widths_ms" in df_existing.columns:
            return df_existing
    sampled = sample_npz_indices()
    rows = []
    for cls, ids in sampled.items():
        for npz_idx in ids:
            sig = load_signal(npz_idx, lead_idx=LEAD_II_IDX, fs_target=FS_NATIVE)
            peaks, widths_ms = detect_qrs(sig, FS_NATIVE, return_widths=True)
            rr_ms = np.diff(peaks) * (1000.0 / FS_NATIVE)
            rows.append({
                "npz_idx": int(npz_idx),
                "class": cls,
                "y": 1 if cls == "AFib" else 0,
                "rr_ms": rr_ms.astype(np.float32).tolist(),
                "widths_ms": widths_ms.astype(np.float32).tolist(),
            })
    df = pd.DataFrame(rows)
    df.to_parquet(path)
    return df


# --------------------------------------------------- deadband-aware features --


def cv_rr_db(rr, T):
    if len(rr) < 2:
        return 0.0
    s = float(np.std(rr))
    if s < T:
        return 0.0
    return s / float(np.mean(rr))


def rmssd_db(rr, T):
    drr = np.diff(rr)
    if len(drr) == 0:
        return 0.0
    masked = np.where(np.abs(drr) < T, 0.0, drr)
    return float(np.sqrt(np.mean(masked ** 2)))


def pRR_rel8_db(rr, T):
    if len(rr) < 2:
        return 0.0
    drr = np.abs(np.diff(rr))
    thresh = max(T, 0.08 * float(np.mean(rr)))
    return float(np.mean(drr >= thresh))


def sarkar_fill_db(rr, T, bin_ms=80, grid=13):
    drr = np.diff(rr)
    if len(drr) < 3:
        return 0.0
    drr_db = np.where(np.abs(drr) < T, 0.0, drr)
    pairs = np.column_stack([drr_db[:-1], drr_db[1:]])
    lo = -(grid // 2) * bin_ms
    idx = np.clip(np.floor((pairs - lo) / bin_ms).astype(int), 0, grid - 1)
    return len(np.unique(idx[:, 0] * grid + idx[:, 1])) / (grid * grid)


def origin_count_db(rr, T, center_ms=80):
    drr = np.diff(rr)
    if len(drr) < 3:
        return 1.0
    c = max(T, center_ms)
    pairs = np.abs(np.column_stack([drr[:-1], drr[1:]]))
    return float(np.mean(np.all(pairs < c, axis=1)))


def dom_cluster_db(rr, T, tol_frac=0.06):
    n = len(rr)
    if n == 0:
        return 0.0
    tol = max(T, tol_frac * float(np.mean(rr)))
    diffs = np.abs(rr[:, None] - rr[None, :])
    return float((diffs <= tol).sum(axis=1).max() / n)


def trimmed_cv_db(rr, T, k_iqr=1.5, max_iter=6, min_keep_frac=0.4):
    x = rr.astype(np.float64).copy()
    n0 = len(x)
    for _ in range(max_iter):
        if len(x) <= max(4, int(min_keep_frac * n0)):
            break
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        if iqr < 1e-6:
            break
        m = np.median(x)
        keep = np.abs(x - m) <= k_iqr * iqr
        if keep.all() or keep.sum() < max(4, int(min_keep_frac * n0)):
            break
        x = x[keep]
    if len(x) < 3:
        return 0.0
    s = float(np.std(x))
    if s < T:
        return 0.0
    return s / float(np.mean(x))


def bigeminy_acf_db(rr, T):
    if len(rr) < 7:
        return 0.0
    rr_q = np.round(rr / max(T, 1.0)) * max(T, 1.0) if T > 0 else rr
    from scipy.stats import spearmanr
    out = 0.0
    for lag in (2, 3, 4):
        if len(rr_q) > lag + 3:
            a, b = rr_q[:-lag], rr_q[lag:]
            if np.std(a) > 0 and np.std(b) > 0:
                r, _ = spearmanr(a, b)
                if np.isfinite(r) and abs(r) > out:
                    out = abs(r)
    return float(out)


def cosen_db(rr, T, m=1, r_grid_frac=(0.03, 0.05, 0.08, 0.12, 0.18, 0.25)):
    n = len(rr)
    if n < m + 2:
        return 0.0
    mean_rr = float(np.mean(rr))
    if float(np.std(rr)) < T:
        return 0.0
    best_se = None
    for rf in r_grid_frac:
        r = max(T, rf * mean_rr)
        tpl_m = np.array([rr[i:i + m] for i in range(n - m + 1)])
        tpl_m1 = np.array([rr[i:i + m + 1] for i in range(n - m)])
        if len(tpl_m) < 2 or len(tpl_m1) < 2:
            continue
        d_m = np.max(np.abs(tpl_m[:, None, :] - tpl_m[None, :, :]), axis=-1)
        d_m1 = np.max(np.abs(tpl_m1[:, None, :] - tpl_m1[None, :, :]), axis=-1)
        np.fill_diagonal(d_m, np.inf)
        np.fill_diagonal(d_m1, np.inf)
        B = (d_m <= r).sum()
        A = (d_m1 <= r).sum()
        if B == 0 or A == 0:
            continue
        se = -np.log(A / B) - np.log(mean_rr / 1000.0)
        if best_se is None or se > best_se:
            best_se = se
    return float(best_se) if best_se is not None else 0.0


def pvc_pair_db(rr, T, big_frac=0.20):
    drr = np.diff(rr)
    if len(drr) < 3:
        return 0.0
    mean_rr = float(np.mean(rr))
    big = max(T * 3, big_frac * mean_rr)
    a, b = drr[:-1], drr[1:]
    symmetric = np.abs(a + b) < 0.3 * big
    both_big = (np.abs(a) > big) & (np.abs(b) > big)
    return float((symmetric & both_big).sum() / max(1, len(drr) - 1))


FEATURES = {
    "cv_rr":          cv_rr_db,
    "rmssd_ms":       rmssd_db,
    "pRR_rel8":       pRR_rel8_db,
    "sarkar_fill":    sarkar_fill_db,
    "origin_count":   origin_count_db,
    "dom_cluster":    dom_cluster_db,
    "trimmed_cv":     trimmed_cv_db,
    "bigeminy_acf":   bigeminy_acf_db,
    "cosen":          cosen_db,
    "pvc_pair_score": pvc_pair_db,
}


def best_direction_auroc(scores, y):
    from sklearn.metrics import roc_auc_score, roc_curve
    s = np.nan_to_num(scores.astype(float), nan=float(np.nanmedian(scores)))
    a_hi = roc_auc_score(y, s)
    if a_hi >= 0.5:
        auc, sign, s_use = a_hi, "≥", s
    else:
        auc, sign, s_use = 1 - a_hi, "≤", -s
    fpr, tpr, thr = roc_curve(y, s_use)
    k = int(np.argmax(tpr - fpr))
    raw_thr = -thr[k] if sign == "≤" else thr[k]
    return auc, sign, float(raw_thr), float(tpr[k]), float(1 - fpr[k])


def main():
    cache_path = Path("logs/afib_rr_cache.parquet")
    cache_path.parent.mkdir(exist_ok=True)
    print(f"## RR cache → {cache_path}")
    df_cache = build_rr_cache(cache_path)
    print(f"   total rows: {len(df_cache)}")

    df_cache = df_cache[df_cache["rr_ms"].apply(lambda x: len(x) >= 4)].reset_index(drop=True)
    rr_arrays = [np.asarray(r, dtype=np.float64) for r in df_cache["rr_ms"]]
    y = df_cache["y"].values
    cls = df_cache["class"].values
    print(f"   usable (≥4 beats): {len(rr_arrays)}")
    print(f"   beats per window — min={min(len(r) for r in rr_arrays)}, "
          f"max={max(len(r) for r in rr_arrays)}, "
          f"median={int(np.median([len(r) for r in rr_arrays]))}")

    print("\n## Deadband sweep — AUROC per metric × T (ms)")
    header = f"{'metric':<16}" + "".join(f"  T={T:>2}" for T in T_GRID) + "  best"
    print(header)
    print("-" * len(header))
    table: dict[str, dict[int, float]] = {}
    all_scores: dict[tuple[str, int], np.ndarray] = {}
    for name, fn in FEATURES.items():
        row = {}
        for T in T_GRID:
            scores = np.array([fn(rr, T) for rr in rr_arrays])
            all_scores[(name, T)] = scores
            auc, _, _, _, _ = best_direction_auroc(scores, y)
            row[T] = auc
        table[name] = row
        best_T = max(row, key=lambda t: row[t])
        line = f"{name:<16}" + "".join(f"  {row[T]:.3f}" for T in T_GRID)
        line += f"   T={best_T:>2} ({row[best_T]:.3f})"
        print(line)

    print("\n## Optimal single-metric rule (at its best T)")
    best_Ts = {}
    for name in FEATURES:
        best_T = max(T_GRID, key=lambda t: table[name][t])
        best_Ts[name] = best_T
        sc = all_scores[(name, best_T)]
        auc, sign, thr, sens, spec = best_direction_auroc(sc, y)
        print(f"  {name:<16}  T={best_T:>2}  AUROC={auc:.3f}  "
              f"AFib if {name} {sign} {thr:.3g}  sens={sens:.2f}  spec={spec:.2f}")

    # global best T → composite tree
    print(f"\n## Best T per feature: {best_Ts}")
    feat_cols = list(FEATURES.keys())
    X = np.zeros((len(rr_arrays), len(feat_cols)), dtype=np.float64)
    for j, name in enumerate(feat_cols):
        X[:, j] = all_scores[(name, best_Ts[name])]

    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n## Decision tree using each metric at its best-T")
    for depth, weight in [(2, {0: 2.0, 1: 1.0}), (3, {0: 2.0, 1: 1.0}),
                          (3, "balanced")]:
        clf = DecisionTreeClassifier(
            max_depth=depth, criterion="gini",
            min_samples_leaf=12, class_weight=weight,
            random_state=42,
        )
        scores = cross_val_score(clf, X, y, scoring="roc_auc", cv=cv)
        clf.fit(X, y)
        pred = clf.predict(X)
        print(f"\n   depth={depth}  weight={weight}  "
              f"5-fold AUROC={scores.mean():.3f} ± {scores.std():.3f}")
        for line in export_text(clf, feature_names=feat_cols, max_depth=depth).splitlines():
            print("     " + line)
        print("     per-class AFib-call rate:")
        for c in ["AFib", "NSR", "PVC", "AVB2", "AVB3"]:
            mask = cls == c
            if mask.any():
                rate = pred[mask].mean() * 100
                print(f"        {c:>5}: {rate:5.1f}%  (n={mask.sum()})")


if __name__ == "__main__":
    main()
