"""QTDB multi-annotator comparison: v4 C / NeuroKit / WTdelineator vs
q1c (expert 1, sparse) / q2c (expert 2, sparse, 11 records only) / pu (auto, dense, full record).

Key insight: q1c labels only ~2.6% of each 15-min record (the q1c span).
This makes PPV unfair to over-detection. pu covers the ENTIRE 15-min record
densely (≈1100 beats per record) so it gives a fair PPV measurement.

Three reference panels:
  Panel A: vs q1c (within q1c span) — strict expert reference
  Panel B: vs pu (full 15-min) — dense automatic reference
  Panel C: q1c vs q2c (where both exist) — human inter-rater agreement
           (defines the "human ceiling" — even two experts disagree at this level)
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import wfdb

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "WTdelineator"))

from openecg import qtdb
from openecg.stage2.evaluate import MARTINEZ_TOLERANCE_MS
from openecg.stage2.infer import (
    extract_boundaries, load_model_bundle,
    post_process_frames, predict_frames,
)
from openecg.stage2.multi_dataset import _normalize

CKPT = REPO / "data" / "checkpoints" / "stage2_v4_C.pt"
OUT_DIR = REPO / "out"
WINDOW_SAMPLES = 2500
FS = 250
FRAME_MS = 20
BOUNDARY_KEYS = ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off")


def signed_boundary_metrics(pred_indices, true_indices, tolerance_ms, fs=FS):
    """Fast O(N+M log N) greedy nearest-match with signed timing error.

    For each true index, find unmatched pred within tolerance using
    binary search + scanning a small candidate window. Each pred matches
    at most one true (greedy: closest first wins per the iteration order).
    """
    tol = tolerance_ms * fs / 1000.0
    pred = np.sort(np.asarray(pred_indices, dtype=np.int64))
    true = np.sort(np.asarray(true_indices, dtype=np.int64))
    n_pred, n_true = len(pred), len(true)
    matched = np.zeros(n_pred, dtype=bool)
    errors = []
    if n_pred and n_true:
        # For each true, search candidate range with searchsorted then linear
        lo_idx = np.searchsorted(pred, true - tol, side="left")
        hi_idx = np.searchsorted(pred, true + tol, side="right")
        for i in range(n_true):
            best_j = -1
            best_abs = float("inf")
            for j in range(lo_idx[i], hi_idx[i]):
                if matched[j]:
                    continue
                d = abs(int(pred[j]) - int(true[i]))
                if d < best_abs:
                    best_abs = d
                    best_j = j
            if best_j >= 0:
                matched[best_j] = True
                errors.append(int(pred[best_j]) - int(true[i]))
    n_hits = len(errors)
    sens = n_hits / n_true if n_true else 0.0
    ppv = n_hits / n_pred if n_pred else 0.0
    f1 = 2 * sens * ppv / (sens + ppv) if (sens + ppv) > 0 else 0.0
    if errors:
        e = np.asarray(errors, dtype=float) * 1000.0 / fs
        return {"f1": f1, "sens": sens, "ppv": ppv,
                "mean_signed_ms": float(np.mean(e)), "sd_ms": float(np.std(e)),
                "median_abs_ms": float(np.median(np.abs(e))),
                "n_true": int(n_true), "n_pred": int(n_pred), "n_hits": int(n_hits)}
    return {"f1": f1, "sens": sens, "ppv": ppv,
            "mean_signed_ms": 0.0, "sd_ms": 0.0, "median_abs_ms": 0.0,
            "n_true": int(n_true), "n_pred": int(n_pred), "n_hits": int(n_hits)}


def load_ann_as_super(record_path, ext):
    """Load WFDB annotation file (.q1c/.q2c/.pu) as our 6-key dict."""
    try:
        ann = wfdb.rdann(str(record_path), ext)
    except Exception:
        return None
    out = {k: [] for k in BOUNDARY_KEYS}
    for i, sym in enumerate(ann.symbol):
        s = int(ann.sample[i])
        if sym == "p":
            if i > 0 and ann.symbol[i - 1] == "(":
                out["p_on"].append(int(ann.sample[i - 1]))
            if i + 1 < len(ann.symbol) and ann.symbol[i + 1] == ")":
                out["p_off"].append(int(ann.sample[i + 1]))
        elif sym == "N":
            if i > 0 and ann.symbol[i - 1] == "(":
                out["qrs_on"].append(int(ann.sample[i - 1]))
            if i + 1 < len(ann.symbol) and ann.symbol[i + 1] == ")":
                out["qrs_off"].append(int(ann.sample[i + 1]))
        elif sym == "t":
            # t_on rarely marked; t_off is the closing ')'
            if i + 1 < len(ann.symbol) and ann.symbol[i + 1] == ")":
                out["t_off"].append(int(ann.sample[i + 1]))
            # Try preceding '(' for t_on (sometimes present)
            if i > 0 and ann.symbol[i - 1] == "(":
                out["t_on"].append(int(ann.sample[i - 1]))
    return out


def nk_delineate(sig, fs=FS):
    import neurokit2 as nk
    try:
        _, info = nk.ecg_delineate(sig, sampling_rate=fs, method="dwt")
    except Exception:
        return {k: [] for k in BOUNDARY_KEYS}
    return {
        "p_on":   [int(x) for x in info.get("ECG_P_Onsets",  []) if x is not None and not np.isnan(x)],
        "p_off":  [int(x) for x in info.get("ECG_P_Offsets", []) if x is not None and not np.isnan(x)],
        "qrs_on": [int(x) for x in info.get("ECG_R_Onsets",  []) if x is not None and not np.isnan(x)],
        "qrs_off":[int(x) for x in info.get("ECG_R_Offsets", []) if x is not None and not np.isnan(x)],
        "t_on":   [int(x) for x in info.get("ECG_T_Onsets",  []) if x is not None and not np.isnan(x)],
        "t_off":  [int(x) for x in info.get("ECG_T_Offsets", []) if x is not None and not np.isnan(x)],
    }


def wt_delineate(sig, fs=FS):
    import WTdelineator as wav
    n = len(sig)
    chunk = 65000
    out = {k: [] for k in BOUNDARY_KEYS}
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        if end - start < 1000:
            continue
        try:
            P, QRS, T = wav.signalDelineation(sig[start:end].astype(np.float64), fs)
        except Exception:
            continue
        for arr, on_key, off_key in [(P, "p_on", "p_off"),
                                       (QRS, "qrs_on", "qrs_off"),
                                       (T, "t_on", "t_off")]:
            for v in arr[:, 0]:
                v = int(v)
                if v != 0:
                    out[on_key].append(v + start)
            for v in arr[:, -1]:
                v = int(v)
                if v != 0:
                    out[off_key].append(v + start)
    return out


def v4_predict_sliding(model, sig, lead_idx, device, stride=WINDOW_SAMPLES):
    out = {k: [] for k in BOUNDARY_KEYS}
    n = len(sig)
    for start in range(0, n - WINDOW_SAMPLES + 1, stride):
        sig_win = sig[start:start + WINDOW_SAMPLES].astype(np.float32)
        sig_n = ((sig_win - sig_win.mean()) / (sig_win.std() + 1e-6)).astype(np.float32)
        raw = predict_frames(model, sig_n, lead_idx, device=device)
        pp = post_process_frames(raw, frame_ms=FRAME_MS)
        bds = extract_boundaries(pp, fs=FS, frame_ms=FRAME_MS)
        for k, v in bds.items():
            for x in v:
                if 0 <= x < WINDOW_SAMPLES:
                    out[k].append(int(x) + start)
    return out


def filter_in_range(samples, lo, hi):
    return [int(s) for s in samples if lo <= int(s) < hi]


def per_boundary_table(label, pred_by_b, true_by_b):
    print(f"\n=== {label} ===", flush=True)
    print(f"{'boundary':10s} {'tol':>5s} | {'F1':>6s} {'Se%':>6s} {'P+%':>6s} | "
          f"{'mean':>7s} {'SD':>6s} | {'n_true':>6s} {'n_pred':>6s}", flush=True)
    summary = {}
    for k in BOUNDARY_KEYS:
        m = signed_boundary_metrics(pred_by_b.get(k, []), true_by_b.get(k, []),
                                      tolerance_ms=MARTINEZ_TOLERANCE_MS[k])
        summary[k] = m
        print(f"{k:10s} {MARTINEZ_TOLERANCE_MS[k]:>4d}ms | "
              f"{m['f1']:.3f}  {m['sens']*100:5.1f}  {m['ppv']*100:5.1f} | "
              f"{m['mean_signed_ms']:+6.1f}  {m['sd_ms']:5.1f}  | "
              f"{m['n_true']:>6d} {m['n_pred']:>6d}", flush=True)
    return summary


def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {CKPT}", flush=True)
    bundle = load_model_bundle(CKPT, device=device)
    model = bundle["model"].train(False)

    inner = qtdb.ensure_extracted()
    rids = qtdb.records_with_q1c()
    print(f"Total q1c records: {len(rids)}", flush=True)

    # Per-record processing: predict once, compare against q1c/q2c/pu
    agg_pred = {n: {k: [] for k in BOUNDARY_KEYS}
                 for n in ("v4_C", "NeuroKit2_DWT", "WTdelineator")}
    agg_q1c = {k: [] for k in BOUNDARY_KEYS}    # within q1c span
    agg_pu  = {k: [] for k in BOUNDARY_KEYS}    # full 15-min
    agg_pred_full = {n: {k: [] for k in BOUNDARY_KEYS}
                      for n in ("v4_C", "NeuroKit2_DWT", "WTdelineator")}

    # For inter-rater (q1c vs q2c): only collect from records with q2c
    q1q2_q1c = {k: [] for k in BOUNDARY_KEYS}
    q1q2_q2c = {k: [] for k in BOUNDARY_KEYS}

    cum_q1c = 0
    cum_full = 0
    cum_q1q2 = 0
    n_q2c_records = 0
    t0 = time.time()

    for rid in rids:
        record = qtdb.load_record(rid)
        ann_q1c = load_ann_as_super(inner / rid, "q1c")
        ann_pu  = load_ann_as_super(inner / rid, "pu0")
        if ann_q1c is None or ann_pu is None:
            continue
        ann_q2c = load_ann_as_super(inner / rid, "q2c")  # may be None

        all_pos = []
        for k in BOUNDARY_KEYS:
            all_pos.extend(ann_q1c[k])
        if not all_pos:
            continue
        ann_min = min(all_pos)
        ann_max = max(all_pos)

        first_lead = list(record.keys())[0]
        sig_full = record[first_lead].astype(np.float32)
        n = len(sig_full)
        sig_norm = _normalize(sig_full)

        pred_v4 = v4_predict_sliding(model, sig_norm, lead_idx=1, device=device)
        pred_nk = nk_delineate(sig_norm)
        pred_wt = wt_delineate(sig_norm)

        # Panel A: vs q1c (within q1c span)
        for name, preds in [("v4_C", pred_v4), ("NeuroKit2_DWT", pred_nk),
                             ("WTdelineator", pred_wt)]:
            for k in BOUNDARY_KEYS:
                for s in filter_in_range(preds[k], ann_min, ann_max + 1):
                    agg_pred[name][k].append(int(s) + cum_q1c)
        for k in BOUNDARY_KEYS:
            for s in ann_q1c[k]:
                if ann_min <= s <= ann_max:
                    agg_q1c[k].append(int(s) + cum_q1c)

        # Panel B: vs pu (entire record)
        for name, preds in [("v4_C", pred_v4), ("NeuroKit2_DWT", pred_nk),
                             ("WTdelineator", pred_wt)]:
            for k in BOUNDARY_KEYS:
                for s in preds[k]:
                    if 0 <= s < n:
                        agg_pred_full[name][k].append(int(s) + cum_full)
        for k in BOUNDARY_KEYS:
            for s in ann_pu[k]:
                agg_pu[k].append(int(s) + cum_full)

        # Panel C: q1c vs q2c (only if q2c exists)
        if ann_q2c is not None:
            n_q2c_records += 1
            for k in BOUNDARY_KEYS:
                for s in ann_q1c[k]:
                    if ann_min <= s <= ann_max:
                        q1q2_q1c[k].append(int(s) + cum_q1q2)
                for s in ann_q2c[k]:
                    if ann_min <= s <= ann_max:
                        q1q2_q2c[k].append(int(s) + cum_q1q2)
            cum_q1q2 += (ann_max - ann_min) + 100

        cum_q1c += (ann_max - ann_min) + 100
        cum_full += n

    print(f"\nProcessed {len(rids)} records in {time.time()-t0:.0f}s", flush=True)
    print(f"q2c-available records used for Panel C: {n_q2c_records}", flush=True)

    print("\n" + "="*78, flush=True)
    print(f"{'PANEL A: vs q1c (sparse expert) within q1c span':^78}", flush=True)
    print("="*78, flush=True)
    A = {}
    for name in ("v4_C", "NeuroKit2_DWT", "WTdelineator"):
        A[name] = per_boundary_table(f"{name} vs q1c", agg_pred[name], agg_q1c)

    print("\n" + "="*78, flush=True)
    print(f"{'PANEL B: vs pu (dense automatic) over full 15-min record':^78}", flush=True)
    print("="*78, flush=True)
    B = {}
    for name in ("v4_C", "NeuroKit2_DWT", "WTdelineator"):
        B[name] = per_boundary_table(f"{name} vs pu (full)", agg_pred_full[name], agg_pu)

    print("\n" + "="*78, flush=True)
    print(f"{'PANEL C: human inter-rater q1c vs q2c (n={n_q2c_records} records)':^78}", flush=True)
    print("="*78, flush=True)
    C = per_boundary_table("q2c (treated as pred) vs q1c (treated as truth)",
                            q1q2_q2c, q1q2_q1c)

    # Final summary
    print("\n\n" + "="*88, flush=True)
    print(f"{'SUMMARY: avg Martinez F1 across 6 boundaries':^88}", flush=True)
    print("="*88, flush=True)
    print(f"  {'method':18s} | {'vs q1c span':>14s} | {'vs pu full':>14s}", flush=True)
    for name in ("v4_C", "NeuroKit2_DWT", "WTdelineator"):
        a = float(np.mean([A[name][k]["f1"] for k in BOUNDARY_KEYS]))
        b = float(np.mean([B[name][k]["f1"] for k in BOUNDARY_KEYS]))
        print(f"  {name:18s} | {a:14.3f} | {b:14.3f}", flush=True)
    c = float(np.mean([C[k]["f1"] for k in BOUNDARY_KEYS]))
    print(f"  {'q2c (human #2)':18s} | {c:14.3f} | {'-':>14s}  (human ceiling)", flush=True)

    def _safe(v):
        if isinstance(v, dict): return {k: _safe(x) for k, x in v.items()}
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        return v
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"qtdb_multi_annotator_{ts}.json"
    out_path.write_text(json.dumps(_safe({"A_vs_q1c": A, "B_vs_pu": B,
                                            "C_q1c_vs_q2c": C,
                                            "n_q2c_records": n_q2c_records}), indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
