"""QTDB full 15-min record comparison: v4 C (sliding) vs NeuroKit2 vs WTdelineator.

Why: previous baseline eval restricted to 10s windows around annotations,
which may have penalized NK/WT (their detectors typically warm up on context).
This script gives them the full 15-min recording. To avoid false-positive
inflation outside the q1c-annotated regions, predictions are matched against
GT only within [ann_min - margin, ann_max + margin] of the cardiologist
annotation span. v4 C runs in sliding 10s windows over the entire record.

Compares against the same Martinez per-boundary tolerances we already use.
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "WTdelineator"))

from ecgcode import qtdb
from ecgcode.stage2.evaluate import MARTINEZ_TOLERANCE_MS, signed_boundary_metrics
from ecgcode.stage2.infer import (
    extract_boundaries, load_model_bundle,
    post_process_frames, predict_frames,
)
from ecgcode.stage2.multi_dataset import _normalize

CKPT = REPO / "data" / "checkpoints" / "stage2_v4_C.pt"
OUT_DIR = REPO / "out"
WINDOW_SAMPLES = 2500   # 10s @ 250Hz for v4
FS = 250
FRAME_MS = 20
EVAL_MARGIN_SAMPLES = 0  # strict q1c span only (no margin) — q1c labels every beat in this region
BOUNDARY_KEYS = ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off")


def nk_delineate(sig, fs=FS):
    import neurokit2 as nk
    try:
        _, info = nk.ecg_delineate(sig, sampling_rate=fs, method="dwt")
    except Exception as e:
        print(f"  NK error: {e}", flush=True)
        return {k: [] for k in BOUNDARY_KEYS}
    out = {
        "p_on":   [int(x) for x in info.get("ECG_P_Onsets",  []) if x is not None and not np.isnan(x)],
        "p_off":  [int(x) for x in info.get("ECG_P_Offsets", []) if x is not None and not np.isnan(x)],
        "qrs_on": [int(x) for x in info.get("ECG_R_Onsets",  []) if x is not None and not np.isnan(x)],
        "qrs_off":[int(x) for x in info.get("ECG_R_Offsets", []) if x is not None and not np.isnan(x)],
        "t_on":   [int(x) for x in info.get("ECG_T_Onsets",  []) if x is not None and not np.isnan(x)],
        "t_off":  [int(x) for x in info.get("ECG_T_Offsets", []) if x is not None and not np.isnan(x)],
    }
    return out


def wt_delineate(sig, fs=FS):
    """WTdelineator on potentially long signal. Paper recommends max 2^16 samples;
    15-min @ 250Hz = 225000 (~3.4×). Process in non-overlapping chunks of
    65000 samples and stitch."""
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
        except Exception as e:
            print(f"  WT chunk [{start}:{end}] error: {e}", flush=True)
            continue
        for arr, on_idx, off_idx, on_key, off_key in [
            (P, 0, -1, "p_on", "p_off"),
            (QRS, 0, -1, "qrs_on", "qrs_off"),
            (T, 0, -1, "t_on", "t_off"),
        ]:
            for v in arr[:, on_idx]:
                v = int(v)
                if v != 0:
                    out[on_key].append(v + start)
            for v in arr[:, off_idx]:
                v = int(v)
                if v != 0:
                    out[off_key].append(v + start)
    return out


def v4_predict_sliding(model, sig, lead_idx, device, stride_samples=WINDOW_SAMPLES):
    """Run v4 in non-overlapping 10s windows over full record. Boundaries are
    offset by window start. No dedup needed because windows don't overlap."""
    out = {k: [] for k in BOUNDARY_KEYS}
    n = len(sig)
    for start in range(0, n - WINDOW_SAMPLES + 1, stride_samples):
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


def filter_by_range(samples, lo, hi):
    return [int(s) for s in samples if lo <= int(s) < hi]


def evaluate_record(rid, model, device):
    """Returns {model_name: {boundary: signed_metrics}} for one record."""
    record = qtdb.load_record(rid)
    ann = qtdb.load_q1c(rid)
    all_pos = []
    for k in BOUNDARY_KEYS:
        all_pos.extend(ann.get(k, []))
    if not all_pos:
        return None
    ann_min = max(0, min(all_pos) - EVAL_MARGIN_SAMPLES)
    ann_max = max(all_pos) + EVAL_MARGIN_SAMPLES

    first_lead = list(record.keys())[0]
    sig_full = record[first_lead].astype(np.float32)
    n = len(sig_full)
    ann_max = min(ann_max, n)

    # Normalize once for NK/WT (they accept arbitrary length)
    sig_norm = _normalize(sig_full)

    print(f"  {rid}: lead={first_lead}, n={n} samples ({n/FS:.0f}s), "
          f"ann range=[{ann_min}-{ann_max}] ({(ann_max-ann_min)/FS:.0f}s)", flush=True)

    pred_v4 = v4_predict_sliding(model, sig_norm, lead_idx=1, device=device)
    pred_nk = nk_delineate(sig_norm)
    pred_wt = wt_delineate(sig_norm)

    gt = {k: filter_by_range(ann.get(k, []), ann_min, ann_max) for k in BOUNDARY_KEYS}
    out = {}
    for name, preds in [("v4_C", pred_v4), ("NeuroKit2_DWT", pred_nk),
                         ("WTdelineator", pred_wt)]:
        per_b = {}
        for k in BOUNDARY_KEYS:
            pred_in = filter_by_range(preds.get(k, []), ann_min, ann_max)
            gt_in = gt[k]
            per_b[k] = signed_boundary_metrics(
                pred_in, gt_in, tolerance_ms=MARTINEZ_TOLERANCE_MS[k])
            per_b[k]["n_pred_total"] = len(preds.get(k, []))
        out[name] = per_b
    return out


def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {CKPT}", flush=True)
    bundle = load_model_bundle(CKPT, device=device)
    model = bundle["model"].train(False)

    rids = qtdb.records_with_q1c()
    # Restrict to T-annotated subset for fair t_on/t_off comparison
    t_subset = []
    for rid in rids:
        ann = qtdb.load_q1c(rid)
        all_pos = []
        for k in ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off"):
            all_pos.extend(ann.get(k, []))
        if not all_pos:
            continue
        n_q = len(ann.get("qrs_on", []))
        n_t = len(ann.get("t_on", []))
        if n_q > 0 and n_t / n_q >= 0.8:
            t_subset.append(rid)
    print(f"QTDB T-subset: {len(t_subset)} records (T:QRS >= 0.8)", flush=True)

    # Aggregate boundary lists across all records (cumulative offsets)
    agg_pred = {name: {k: [] for k in BOUNDARY_KEYS}
                 for name in ("v4_C", "NeuroKit2_DWT", "WTdelineator")}
    agg_gt = {k: [] for k in BOUNDARY_KEYS}
    cum = 0
    t0 = time.time()
    for rid in t_subset:
        record = qtdb.load_record(rid)
        ann = qtdb.load_q1c(rid)
        all_pos = []
        for k in BOUNDARY_KEYS:
            all_pos.extend(ann.get(k, []))
        if not all_pos:
            continue
        ann_min = max(0, min(all_pos) - EVAL_MARGIN_SAMPLES)
        first_lead = list(record.keys())[0]
        sig_full = record[first_lead].astype(np.float32)
        n = len(sig_full)
        ann_max = min(max(all_pos) + EVAL_MARGIN_SAMPLES, n)
        sig_norm = _normalize(sig_full)

        pred_v4 = v4_predict_sliding(model, sig_norm, lead_idx=1, device=device)
        pred_nk = nk_delineate(sig_norm)
        pred_wt = wt_delineate(sig_norm)

        for name, preds in [("v4_C", pred_v4), ("NeuroKit2_DWT", pred_nk),
                             ("WTdelineator", pred_wt)]:
            for k in BOUNDARY_KEYS:
                for s in preds.get(k, []):
                    if ann_min <= s < ann_max:
                        agg_pred[name][k].append(int(s) + cum)
        for k in BOUNDARY_KEYS:
            for s in ann.get(k, []):
                if ann_min <= s < ann_max:
                    agg_gt[k].append(int(s) + cum)
        cum += n
        print(f"  {rid} done (running cum={cum})", flush=True)
    print(f"\nAll records processed in {time.time()-t0:.1f}s", flush=True)

    # Per-model summary
    print("\n" + "="*78, flush=True)
    print(f"{'QTDB FULL-RECORD eval (within q1c annotated span ±2s)':^78}", flush=True)
    print("="*78, flush=True)
    full = {}
    for name in ("v4_C", "NeuroKit2_DWT", "WTdelineator"):
        print(f"\n=== {name} ===", flush=True)
        print(f"{'boundary':10s} {'tol':>5s} | {'F1':>6s} {'Se%':>6s} {'P+%':>6s} | "
              f"{'mean':>7s} {'SD':>6s} {'medAbs':>7s} | {'n_true':>6s} {'n_pred':>6s}", flush=True)
        per_b = {}
        for k in BOUNDARY_KEYS:
            m = signed_boundary_metrics(agg_pred[name][k], agg_gt[k],
                                          tolerance_ms=MARTINEZ_TOLERANCE_MS[k])
            per_b[k] = m
            print(f"{k:10s} {MARTINEZ_TOLERANCE_MS[k]:>4d}ms | "
                  f"{m['f1']:.3f}  {m['sens']*100:5.1f}  {m['ppv']*100:5.1f} | "
                  f"{m['mean_signed_ms']:+6.1f}  {m['sd_ms']:5.1f}  {m['median_abs_ms']:6.1f} | "
                  f"{m['n_true']:>6d} {m['n_pred']:>6d}", flush=True)
        full[name] = per_b

    # Final summary
    print("\n\n" + "="*78, flush=True)
    print(f"{'SUMMARY: avg Martinez F1 (QTDB FULL 15-min, q1c-span eval)':^78}", flush=True)
    print("="*78, flush=True)
    for name in ("v4_C", "NeuroKit2_DWT", "WTdelineator"):
        avg = float(np.mean([full[name][k]["f1"] for k in BOUNDARY_KEYS]))
        print(f"  {name:18s}  avg F1 = {avg:.3f}", flush=True)

    def _safe(v):
        if isinstance(v, dict): return {k: _safe(x) for k, x in v.items()}
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        return v
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"qtdb_fullrecord_comparison_{ts}.json"
    out_path.write_text(json.dumps(_safe(full), indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
