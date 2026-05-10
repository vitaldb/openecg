"""AFib rule-based discriminator — 단일 지표 성능 + 2-3층 decision tree.

Lydus 10-sec 윈도우에서 클래스별 동수 추출 → openecg.detect_qrs (lead II) →
9개 RR 기반 지표 계산 → AUROC + Youden's J 최적 threshold → DecisionTree(d=2,3).

클래스 라벨:
  pos  = A.fib                                (rhythm == 'A.fib')
  neg0 = NSR                                  (rhythm == 'NSR', premature_beat == 'Nonspecific', avb == 'NSP')
  neg1 = isolated PVC on sinus                (premature_beat == 'VPC', rhythm sinus, avb == 'NSP')
  neg2 = 2°AVB                                (avb in {2'AVB})
  neg3 = 3°AVB                                (avb in {3'AVB})
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import duckdb

from openecg.qrs import detect_qrs
from openecg.lydus import load_signal, FS_NATIVE


RNG = np.random.default_rng(20260511)

LYDUS_DIR = os.environ.get(
    "OPENECG_LYDUS_DIR", "G:/Shared drives/datasets/ecg/lydus_ecg"
)
os.environ["OPENECG_LYDUS_DIR"] = LYDUS_DIR

LEAD_II_IDX = 1
N_PER_CLASS = 80          # 3'AVB max ~93 → 80 동수
MIN_BEATS = 4             # 10 s 윈도우에 4비트 미만이면 통계 무의미


# ---------------------------------------------------------------- sampling ---


def sample_windows() -> dict[str, list[int]]:
    con = duckdb.connect(str(Path(LYDUS_DIR) / "lydus_ecg.duckdb"), read_only=True)

    queries = {
        "AFib": (
            "SELECT npz_idx FROM records "
            "WHERE rhythm = 'A.fib' AND npz_idx IS NOT NULL "
            "AND premature_beat = 'Nonspecific' AND avb = 'NSP'"
        ),
        "NSR": (
            "SELECT npz_idx FROM records "
            "WHERE rhythm = 'NSR' AND premature_beat = 'Nonspecific' "
            "AND avb = 'NSP' AND npz_idx IS NOT NULL"
        ),
        "PVC": (
            "SELECT npz_idx FROM records "
            "WHERE premature_beat = 'VPC' AND rhythm IN ('NSR','S.brady','S.tachy') "
            "AND avb = 'NSP' AND npz_idx IS NOT NULL"
        ),
        "AVB2": (
            "SELECT npz_idx FROM records "
            "WHERE avb = '2''AVB' AND rhythm IN ('NSR','S.brady') "
            "AND npz_idx IS NOT NULL"
        ),
        "AVB3": (
            "SELECT npz_idx FROM records "
            "WHERE avb = '3''AVB' AND npz_idx IS NOT NULL"
        ),
    }
    out: dict[str, list[int]] = {}
    for cls, q in queries.items():
        rows = [int(r[0]) for r in con.execute(q).fetchall()]
        RNG.shuffle(rows)
        out[cls] = rows[:N_PER_CLASS]
        print(f"  {cls:>6}: pool={len(rows):>6}  sampled={len(out[cls])}")
    return out


# --------------------------------------------------------------- features ---


def _symbolize_terciles(x: np.ndarray) -> np.ndarray:
    """3-symbol encoding by sample tercile (S=0, N=1, L=2)."""
    q1, q2 = np.quantile(x, [1 / 3, 2 / 3])
    s = np.full(len(x), 1, dtype=np.int8)
    s[x <= q1] = 0
    s[x >= q2] = 2
    return s


def _predictability_index(symbols: np.ndarray, n_states: int = 3) -> float:
    """1 - H(X_{n+1} | X_n) / H(X). Higher = more deterministic."""
    if len(symbols) < 6:
        return 0.0
    # marginal entropy
    _, cnt = np.unique(symbols, return_counts=True)
    p = cnt / cnt.sum()
    H_marg = -np.sum(p * np.log(p + 1e-12))
    if H_marg < 1e-6:
        return 1.0
    # joint
    trans = np.zeros((n_states, n_states), dtype=np.float64)
    for a, b in zip(symbols[:-1], symbols[1:]):
        trans[a, b] += 1
    row_sum = trans.sum(axis=1, keepdims=True)
    P = np.divide(trans, row_sum, out=np.zeros_like(trans), where=row_sum > 0)
    pa = trans.sum(axis=1) / trans.sum()
    H_cond = 0.0
    for i in range(n_states):
        for j in range(n_states):
            if P[i, j] > 0:
                H_cond -= pa[i] * P[i, j] * np.log(P[i, j])
    return float(1.0 - H_cond / H_marg)


def _sarkar_bin_fill(drr_ms: np.ndarray, bin_ms: int = 80, grid: int = 13) -> float:
    """Sarkar dRR-map의 점유 bin 비율. AFib에서 높음."""
    if len(drr_ms) < 3:
        return 0.0
    pairs = np.column_stack([drr_ms[:-1], drr_ms[1:]])
    lo = -(grid // 2) * bin_ms
    hi = (grid // 2 + 1) * bin_ms
    idx = np.floor((pairs - lo) / bin_ms).astype(int)
    idx = np.clip(idx, 0, grid - 1)
    flat = idx[:, 0] * grid + idx[:, 1]
    occ = len(np.unique(flat))
    return occ / (grid * grid)


def _origin_count_ratio(drr_ms: np.ndarray, center_ms: int = 80) -> float:
    """|dRR_n| < center & |dRR_{n+1}| < center 비율. PVC에서 큼 (정상박 대다수)."""
    if len(drr_ms) < 3:
        return 1.0
    pairs = np.abs(np.column_stack([drr_ms[:-1], drr_ms[1:]]))
    return float(np.mean(np.all(pairs < center_ms, axis=1)))


def _poincare_clusters(rr_ms: np.ndarray, min_samples: int = 2) -> int:
    """DBSCAN on (RR_n, RR_{n+1}) with eps scaled to within-record variability."""
    if len(rr_ms) < 5:
        return 1
    from sklearn.cluster import DBSCAN
    eps = max(40.0, 0.6 * np.std(rr_ms))
    pairs = np.column_stack([rr_ms[:-1], rr_ms[1:]])
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pairs)
    return int(len(set(labels)) - (1 if -1 in labels else 0))


def _sign_flip_rate(drr_ms: np.ndarray) -> float:
    """Fraction of consecutive dRR pairs with opposite sign. Bigeminy/isolated PVC → ~1.0,
    AFib → ~0.5 (chance), Wenckebach → low (monotone runs)."""
    if len(drr_ms) < 3:
        return 0.5
    s = np.sign(drr_ms)
    flips = np.sum(s[:-1] * s[1:] < 0)
    return float(flips / (len(drr_ms) - 1))


def _pnn50(drr_ms: np.ndarray) -> float:
    if len(drr_ms) == 0:
        return 0.0
    return float(np.mean(np.abs(drr_ms) > 50))


def _dom_cluster_frac(rr_ms: np.ndarray, tol_frac: float = 0.06) -> float:
    """Max # beats within ±(tol_frac × mean RR) of any beat, divided by N.
    AFib has no regular skeleton → low; PVC w/ sinus + AVB3 escape → high.
    Scale-invariant via relative tolerance."""
    n = len(rr_ms)
    if n == 0:
        return 0.0
    tol = tol_frac * float(np.mean(rr_ms))
    diffs = np.abs(rr_ms[:, None] - rr_ms[None, :])
    counts = (diffs <= tol).sum(axis=1)
    return float(counts.max() / n)


def _trimmed_cv(rr_ms: np.ndarray, k_iqr: float = 1.5, max_iter: int = 6,
                min_keep_frac: float = 0.4) -> float:
    """Iteratively remove |x - median| > k×IQR. AFib's IQR stays large → little
    trim, CV preserved. PVC/escape collapses to a thin skeleton, CV → 0.
    Stops trimming once we'd lose more than 1-min_keep_frac of beats."""
    x = rr_ms.astype(float).copy()
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
        if keep.all():
            break
        if keep.sum() < max(4, int(min_keep_frac * n0)):
            break
        x = x[keep]
    if len(x) < 3 or np.mean(x) < 1e-6:
        return 0.0
    return float(np.std(x) / np.mean(x))


def _bigeminy_acf(rr_ms: np.ndarray) -> float:
    """Max |Spearman autocorr| of RR at lags 2,3,4. Rank-based for robustness
    to a single outlier with N=8-15. AFib≈0; bigeminy→0.7+; trigeminy→0.5+."""
    if len(rr_ms) < 7:
        return 0.0
    from scipy.stats import spearmanr
    out = 0.0
    for lag in (2, 3, 4):
        if len(rr_ms) > lag + 3:
            a, b = rr_ms[:-lag], rr_ms[lag:]
            try:
                r, _ = spearmanr(a, b)
                if np.isfinite(r) and abs(r) > out:
                    out = abs(r)
            except Exception:
                pass
    return float(out)


def _cosen(rr_ms: np.ndarray, m: int = 1, r_grid_frac=(0.03, 0.05, 0.08, 0.12, 0.18, 0.25)) -> float:
    """COSEn (Lake & Moorman 2011): SampEn with r auto-selected from a grid,
    plus mean-RR correction. Designed for as few as 12 beats."""
    n = len(rr_ms)
    if n < m + 2:
        return 0.0
    mean_rr = float(np.mean(rr_ms))
    std_rr = float(np.std(rr_ms))
    if std_rr < 1e-6:
        return 0.0
    # find r that maximises number of matches at template length m
    best_se = None
    for rf in r_grid_frac:
        r = rf * mean_rr
        # Build template count
        tpl_m = np.array([rr_ms[i:i + m] for i in range(n - m + 1)])
        tpl_m1 = np.array([rr_ms[i:i + m + 1] for i in range(n - m)])
        # cheby distances
        if len(tpl_m) < 2 or len(tpl_m1) < 2:
            continue
        d_m = np.max(np.abs(tpl_m[:, None, :] - tpl_m[None, :, :]), axis=-1)
        d_m1 = np.max(np.abs(tpl_m1[:, None, :] - tpl_m1[None, :, :]), axis=-1)
        # exclude self
        np.fill_diagonal(d_m, np.inf)
        np.fill_diagonal(d_m1, np.inf)
        B = (d_m <= r).sum()
        A = (d_m1 <= r).sum()
        if B == 0 or A == 0:
            continue
        # COSEn = -log(A/B) - log(2r) - log(mean_rr)
        # Equivalent expression with heart-rate correction (Lake 2010)
        se = -np.log(A / B) - np.log(mean_rr / 1000.0)
        if best_se is None or se > best_se:
            best_se = se
    return float(best_se) if best_se is not None else 0.0


def _pRR_relative(rr_ms: np.ndarray, x_pct: float = 8.0) -> float:
    """Fraction of consecutive RR pairs with |ΔRR| ≥ x% of mean RR.
    Scale-invariant pNN — robust to brady/tachy."""
    if len(rr_ms) < 2:
        return 0.0
    mean_rr = float(np.mean(rr_ms))
    drr = np.abs(np.diff(rr_ms))
    return float(np.mean(drr >= 0.01 * x_pct * mean_rr))


def _sarkar_pvc_pair_score(drr_ms: np.ndarray, big_frac_of_mean: float = 0.20,
                           mean_rr: float = 800.0) -> float:
    """Sarkar PVC-pair signature: count of (dRR[i], dRR[i+1]) pairs where
    dRR[i] ≈ -dRR[i+1] (symmetric off-diagonal in the RdR map) AND both
    have large magnitude. Subtract this evidence from AFib score."""
    if len(drr_ms) < 3:
        return 0.0
    big = big_frac_of_mean * mean_rr
    a, b = drr_ms[:-1], drr_ms[1:]
    symmetric = np.abs(a + b) < 0.3 * big
    both_big = (np.abs(a) > big) & (np.abs(b) > big)
    pvc_pairs = symmetric & both_big
    return float(pvc_pairs.sum() / max(1, len(drr_ms) - 1))


def _rr_bimodality(rr_ms: np.ndarray) -> float:
    """Gap between two largest peaks in a coarse RR histogram, normalised by std.
    Bigeminy/Mobitz → ~2 distinct peaks → large; AFib → unimodal → small."""
    if len(rr_ms) < 6:
        return 0.0
    nbins = max(5, int(np.sqrt(len(rr_ms))))
    h, e = np.histogram(rr_ms, bins=nbins)
    if h.max() == 0 or np.std(rr_ms) < 1e-6:
        return 0.0
    centers = 0.5 * (e[1:] + e[:-1])
    # rank bins by count
    order = np.argsort(h)[::-1]
    if h[order[1]] < 0.3 * h[order[0]]:
        return 0.0
    gap = abs(centers[order[0]] - centers[order[1]])
    return float(gap / np.std(rr_ms))


def _levene_p(rr_ms: np.ndarray) -> float:
    """Levene homoscedasticity: Var(RR_{n+1} | RR_n bucket) 균질 검정. AFib → high p."""
    if len(rr_ms) < 12:
        return 1.0
    from scipy.stats import levene
    nxt = rr_ms[1:]
    prev = rr_ms[:-1]
    # 3 buckets by tercile of prev
    q1, q2 = np.quantile(prev, [1 / 3, 2 / 3])
    g0 = nxt[prev <= q1]
    g1 = nxt[(prev > q1) & (prev < q2)]
    g2 = nxt[prev >= q2]
    if min(len(g0), len(g1), len(g2)) < 2:
        return 1.0
    try:
        return float(levene(g0, g1, g2).pvalue)
    except Exception:
        return 1.0


def _sample_entropy(x: np.ndarray, m: int = 2, r_frac: float = 0.2) -> float:
    """SampEn — short signal version (Richman 2000)."""
    n = len(x)
    if n < m + 2:
        return 0.0
    r = r_frac * np.std(x)
    if r < 1e-6:
        return 0.0
    def _phi(mm):
        if n - mm <= 0:
            return 1e-12
        tpl = np.array([x[i:i + mm] for i in range(n - mm + 1)])
        cnt = 0
        for i in range(len(tpl)):
            d = np.max(np.abs(tpl - tpl[i]), axis=1)
            cnt += np.sum(d <= r) - 1  # exclude self
        return cnt
    A = _phi(m + 1)
    B = _phi(m)
    if A == 0 or B == 0:
        return 0.0
    return float(-np.log(A / B))


def features_from_rr(rr_ms: np.ndarray) -> dict[str, float]:
    if len(rr_ms) < MIN_BEATS:
        return None
    drr = np.diff(rr_ms)
    mean_rr = float(np.mean(rr_ms))
    return {
        "n_beats": int(len(rr_ms)),
        "mean_rr_ms": mean_rr,
        "cv_rr": float(np.std(rr_ms) / mean_rr),
        "rmssd_ms": float(np.sqrt(np.mean(drr ** 2))) if len(drr) else 0.0,
        "rho1_drr": float(np.corrcoef(drr[:-1], drr[1:])[0, 1]) if len(drr) >= 3 else 0.0,
        "predictability": _predictability_index(_symbolize_terciles(rr_ms)),
        "sarkar_fill": _sarkar_bin_fill(drr),
        "origin_count": _origin_count_ratio(drr),
        "poincare_clusters": _poincare_clusters(rr_ms),
        "levene_p": _levene_p(rr_ms),
        "sign_flip": _sign_flip_rate(drr),
        "pnn50": _pnn50(drr),
        "rr_bimodality": _rr_bimodality(rr_ms),
        # v2 — literature-backed + skeleton tests
        "cosen": _cosen(rr_ms),
        "pRR_rel8": _pRR_relative(rr_ms, 8.0),
        "dom_cluster": _dom_cluster_frac(rr_ms),
        "trimmed_cv": _trimmed_cv(rr_ms),
        "bigeminy_acf": _bigeminy_acf(rr_ms),
        "pvc_pair_score": _sarkar_pvc_pair_score(drr, mean_rr=mean_rr),
    }


# --------------------------------------------------------------- evaluation ---


def auroc_best_direction(scores: np.ndarray, y: np.ndarray):
    """Try both directions, keep the one with AUROC > 0.5."""
    from sklearn.metrics import roc_auc_score, roc_curve
    s = np.nan_to_num(scores.astype(float), nan=float(np.nanmedian(scores)))
    auc_high = roc_auc_score(y, s)
    if auc_high >= 0.5:
        auc, direction = auc_high, "high"
        s_use = s
    else:
        auc, direction = 1 - auc_high, "low"
        s_use = -s
    fpr, tpr, thr = roc_curve(y, s_use)
    j = tpr - fpr
    k = int(np.argmax(j))
    raw_thr = -thr[k] if direction == "low" else thr[k]
    return float(auc), direction, float(raw_thr), float(tpr[k]), float(1 - fpr[k])


FEAT_COLS = [
    # baseline
    "cv_rr", "rmssd_ms", "sarkar_fill", "origin_count",
    "abs_rho1_drr", "predictability", "poincare_clusters",
    "levene_p", "sign_flip", "pnn50", "rr_bimodality",
    # v2 — literature + skeleton tests
    "cosen", "pRR_rel8", "dom_cluster", "trimmed_cv",
    "bigeminy_acf", "pvc_pair_score",
]


def main():
    print("# Lydus AFib rule-based discriminator\n")
    print("## Step 1 — sampling")
    sampled = sample_windows()

    print("\n## Step 2 — QRS detection + feature extraction (lead II @ 500 Hz)")
    rows: list[dict] = []
    for cls, idxs in sampled.items():
        n_ok = 0
        for npz_idx in idxs:
            sig = load_signal(npz_idx, lead_idx=LEAD_II_IDX, fs_target=FS_NATIVE)
            peaks = detect_qrs(sig, FS_NATIVE)
            rr_ms = np.diff(peaks) * (1000.0 / FS_NATIVE)
            feats = features_from_rr(rr_ms)
            if feats is None:
                continue
            feats["abs_rho1_drr"] = abs(feats["rho1_drr"])
            feats["class"] = cls
            feats["y"] = 1 if cls == "AFib" else 0
            feats["npz_idx"] = int(npz_idx)
            rows.append(feats)
            n_ok += 1
        print(f"  {cls:>6}: usable windows = {n_ok}/{len(idxs)}")

    import pandas as pd
    df = pd.DataFrame(rows)

    out_path = Path("logs/afib_rule_features.parquet")
    out_path.parent.mkdir(exist_ok=True)
    df.to_parquet(out_path)
    print(f"\nSaved features → {out_path}  (n={len(df)})")

    print("\n## Step 3 — class statistics (median per class)")
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print(df.groupby("class")[FEAT_COLS].median().T.to_string())

    print("\n## Step 4 — single-metric AUROC + best threshold (AFib=1)")
    y = df["y"].values
    summary = []
    for col in FEAT_COLS:
        auc, direction, thr, sens, spec = auroc_best_direction(df[col].values, y)
        op = "≥" if direction == "high" else "≤"
        summary.append({
            "metric": col, "AUROC": auc, "thr": thr, "direction": direction,
            "rule": f"AFib if {col} {op} {thr:.3g}",
            "sens": sens, "spec": spec,
        })
    summary.sort(key=lambda r: -r["AUROC"])
    for r in summary:
        print(f"  {r['metric']:<18}  AUROC={r['AUROC']:.3f}  "
              f"{r['rule']:<37}  sens={r['sens']:.2f}  spec={r['spec']:.2f}")

    print("\n## Step 5 — decision tree (depth=2, depth=3)")
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    X = df[FEAT_COLS].fillna(df[FEAT_COLS].median()).values
    feat_cols = FEAT_COLS
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Spec-favoring tree: penalty on FP > FN so PVC/AVB3 don't get called AFib.
    # 219 negatives vs 80 positives → 'balanced' = {0:0.68, 1:1.87}.
    # We invert: {0:2, 1:1} ≈ 3× penalty on each negative.
    for depth in (2, 3):
        clf = DecisionTreeClassifier(
            max_depth=depth, criterion="gini",
            min_samples_leaf=12, class_weight={0: 2.0, 1: 1.0},
            random_state=42,
        )
        scores = cross_val_score(clf, X, y, scoring="roc_auc", cv=cv)
        clf.fit(X, y)
        # per-class confusion on train data (illustrative)
        pred = clf.predict(X)
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y, pred)
        print(f"\n  depth={depth}  5-fold AUROC = {scores.mean():.3f} ± {scores.std():.3f}")
        print(f"    train confusion (rows=true, cols=pred): {cm.tolist()}")
        rep = classification_report(y, pred, target_names=["other", "AFib"], digits=3)
        print("   ", rep.replace("\n", "\n    "))
        rules = export_text(clf, feature_names=feat_cols, max_depth=depth)
        print("    rules:")
        for line in rules.splitlines():
            print("      " + line)

        # per-true-class breakdown
        print("    per-class predicted-as-AFib rate:")
        for cls in sampled:
            mask = df["class"].values == cls
            if not mask.any():
                continue
            rate = pred[mask].mean()
            print(f"      {cls:>6}: {rate * 100:.1f}%  (n={mask.sum()})")

    json_path = Path("logs/afib_rule_summary.json")
    json_path.write_text(json.dumps({
        "n_samples": int(len(df)),
        "per_metric": summary,
    }, indent=2))
    print(f"\nSaved single-metric summary → {json_path}")


if __name__ == "__main__":
    main()
