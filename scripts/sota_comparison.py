"""SOTA paper comparison with Martinez-style tolerances and reported metrics.

Most ECG delineation papers use per-boundary tolerances rather than a single
loose 150ms tolerance. We re-eval C and F using literature-standard tolerances:
  P_on, P_off:    ±50 ms
  QRS_on, QRS_off: ±40 ms
  T_on:           ±50 ms
  T_off:          ±100 ms  (T-end is hardest, all literature allows wider)

Reports for each (model, domain, boundary):
  - F1, Se (sensitivity), P+ (PPV)
  - Mean ± SD timing error (signed; literature reports signed mean)
  - n_true / n_pred / n_hits

Compares to:
  Martinez 2004 wavelet on QTDB (the canonical baseline)
  Kalyakulina 2020 / DENS-ECG U-Net on LUDB
  SemiSegECG 2025 semi-supervised on ISP
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from openecg import eval as ee, isp, ludb, qtdb
from openecg.stage2.dataset import LUDBFrameDataset
from openecg.stage2.evaluate import MARTINEZ_TOLERANCE_MS, signed_boundary_metrics
from openecg.stage2.infer import extract_boundaries, post_process_frames, predict_frames
from openecg.stage2.model import FrameClassifier
from openecg.stage2.train import load_checkpoint

CKPT_DIR = Path("data/checkpoints")
OUT_DIR = Path("out")
WINDOW_SAMPLES_250 = 2500
WINDOW_FRAMES = 500
FRAME_MS = 20

TOL_PER_BOUNDARY = MARTINEZ_TOLERANCE_MS


def _normalize(sig):
    mean = float(sig.mean())
    std = float(sig.std()) + 1e-6
    return ((sig - mean) / std).astype(np.float32)


def _decimate(sig, native_fs, target_fs=250):
    if native_fs == target_fs: return sig.astype(np.float64)
    factor = native_fs // target_fs
    import scipy.signal as scipy_signal
    return scipy_signal.decimate(sig, factor, zero_phase=True)


def _extract_boundaries(super_frames, fs=250, frame_ms=FRAME_MS):
    out = defaultdict(list)
    super_to_name = {ee.SUPER_P: "p", ee.SUPER_QRS: "qrs", ee.SUPER_T: "t"}
    samples_per_frame = int(round(frame_ms * fs / 1000.0))
    prev = ee.SUPER_OTHER
    for f_idx, cur in enumerate(super_frames):
        cur = int(cur)
        if cur != prev:
            sample = f_idx * samples_per_frame
            if prev in super_to_name:
                out[f"{super_to_name[prev]}_off"].append(sample - 1)
            if cur in super_to_name:
                out[f"{super_to_name[cur]}_on"].append(sample)
        prev = cur
    if prev in super_to_name:
        sample = len(super_frames) * samples_per_frame
        out[f"{super_to_name[prev]}_off"].append(sample - 1)
    return dict(out)


def _legacy_signed_boundary_metrics(pred, true, tol_ms, fs=250):
    """Compute signed mean ± SD error (literature standard) plus F1/Se/PPV.

    Greedy nearest-match: for each true boundary, find nearest unmatched
    predicted boundary within tolerance; signed error = pred - true.
    """
    tol_samples = tol_ms * fs / 1000.0
    pred_arr = np.sort(np.array(pred, dtype=int))
    true_arr = np.sort(np.array(true, dtype=int))

    matched = set()
    signed_errs = []
    for t in true_arr:
        best_idx = -1
        best_abs = float("inf")
        for j, p in enumerate(pred_arr):
            if j in matched: continue
            d = abs(int(p) - int(t))
            if d < best_abs:
                best_abs = d
                best_idx = j
        if best_idx >= 0 and best_abs <= tol_samples:
            matched.add(best_idx)
            signed_errs.append(int(pred_arr[best_idx]) - int(t))

    n_hits = len(signed_errs)
    sens = n_hits / len(true_arr) if len(true_arr) > 0 else 0.0
    ppv = n_hits / len(pred_arr) if len(pred_arr) > 0 else 0.0
    f1 = 2 * sens * ppv / (sens + ppv) if (sens + ppv) > 0 else 0.0
    if signed_errs:
        errs_ms = np.array(signed_errs) * 1000.0 / fs
        mean = float(np.mean(errs_ms))
        sd = float(np.std(errs_ms))
        median = float(np.median(np.abs(errs_ms)))
    else:
        mean = sd = median = 0.0
    return {"f1": f1, "sens": sens, "ppv": ppv,
            "mean_signed_ms": mean, "sd_ms": sd, "median_abs_ms": median,
            "n_true": int(len(true_arr)), "n_pred": int(len(pred_arr)), "n_hits": n_hits}


def evaluate_ludb(model, device):
    val_ds = LUDBFrameDataset(ludb.load_split()["val"])
    bp, bt = defaultdict(list), defaultdict(list)
    cum = 0
    with torch.no_grad():
        for idx in range(len(val_ds)):
            rid, lead = val_ds.items[idx]
            sig_250, lead_idx, _ = val_ds.cache[(rid, lead)]
            sig_250 = sig_250[:WINDOW_SAMPLES_250]
            if len(sig_250) < WINDOW_SAMPLES_250: continue
            pred_raw = predict_frames(model, sig_250, lead_idx, device=device)
            pp = post_process_frames(pred_raw, frame_ms=FRAME_MS)
            for k, v in extract_boundaries(pp, fs=250).items():
                bp[k].extend(int(x) + cum for x in v)
            try:
                gt_ann = ludb.load_annotations(rid, lead)
                for k, v in gt_ann.items():
                    if k.endswith("_on") or k.endswith("_off"):
                        for s in v:
                            s250 = int(s // 2)
                            if 0 <= s250 < WINDOW_SAMPLES_250:
                                bt[k].append(s250 + cum)
            except Exception:
                pass
            cum += WINDOW_SAMPLES_250
    return bp, bt, len(val_ds)


def evaluate_isp(model, device):
    rec_ids = isp.load_split()["test"]
    bp, bt = defaultdict(list), defaultdict(list)
    cum = 0
    n_seqs = 0
    with torch.no_grad():
        for rid in rec_ids:
            try:
                record = isp.load_record(rid, split="test")
                ann_super = isp.load_annotations_as_super(rid, split="test")
            except Exception:
                continue
            for lead_idx, lead in enumerate(isp.LEADS_12):
                sig_1000 = record[lead]
                sig_250 = _decimate(sig_1000, 1000)
                sig_n = _normalize(sig_250)
                if len(sig_n) < WINDOW_SAMPLES_250:
                    pad = np.zeros(WINDOW_SAMPLES_250 - len(sig_n), dtype=sig_n.dtype)
                    sig_n = np.concatenate([sig_n, pad])
                sig_n = sig_n[:WINDOW_SAMPLES_250]
                pred_raw = predict_frames(model, sig_n, lead_idx, device=device)
                pp = post_process_frames(pred_raw, frame_ms=FRAME_MS)
                for k, v in extract_boundaries(pp, fs=250).items():
                    bp[k].extend(int(x) + cum for x in v)
                for k, v in ann_super.items():
                    if k.endswith("_on") or k.endswith("_off"):
                        for s in v:
                            s250 = int(s // 4)
                            if 0 <= s250 < WINDOW_SAMPLES_250:
                                bt[k].append(s250 + cum)
                cum += WINDOW_SAMPLES_250
                n_seqs += 1
    return bp, bt, n_seqs


def evaluate_qtdb_t_subset(model, device):
    """Evaluate only on records with proper T annotations (T:QRS ratio >= 0.8)."""
    rids = []
    for rid in qtdb.records_with_q1c():
        ann = qtdb.load_q1c(rid)
        win = qtdb.annotated_window(ann, window_samples=WINDOW_SAMPLES_250, fs=250)
        if win is None: continue
        start, end = win
        n_q = sum(1 for s in ann["qrs_on"] if start <= s < end)
        n_t = sum(1 for s in ann["t_on"] if start <= s < end)
        if n_q == 0: continue
        if n_t / n_q >= 0.8:
            rids.append(rid)

    bp, bt = defaultdict(list), defaultdict(list)
    cum = 0
    n_seqs = 0
    with torch.no_grad():
        for rid in rids:
            try:
                record = qtdb.load_record(rid)
                ann = qtdb.load_q1c(rid)
            except Exception:
                continue
            win = qtdb.annotated_window(ann, window_samples=WINDOW_SAMPLES_250, fs=250)
            if win is None: continue
            start, end = win
            if end > 225000:
                end = 225000; start = end - WINDOW_SAMPLES_250
            first_lead = list(record.keys())[0]
            sig = record[first_lead][start:end]
            if len(sig) < WINDOW_SAMPLES_250: continue
            sig_n = _normalize(sig)
            pred_raw = predict_frames(model, sig_n, lead_id=1, device=device)
            pp = post_process_frames(pred_raw, frame_ms=FRAME_MS)
            for k, v in extract_boundaries(pp, fs=250).items():
                bp[k].extend(int(x) + cum for x in v)
            win_ann = {k: [s - start for s in v if start <= s < end] for k, v in ann.items()}
            for k in TOL_PER_BOUNDARY:
                bt[k].extend(int(s) + cum for s in win_ann[k])
            cum += WINDOW_SAMPLES_250
            n_seqs += 1
    return bp, bt, n_seqs, len(rids)


SOTA = {
    # (database, boundary): {"se": (val, ref), "ppv": (val, ref), "mean": (val, ref), "sd": (val, ref)}
    # Values from canonical papers; error tolerances per Martinez convention.
    "LUDB": {
        # DENS-ECG / Moskalenko 2020 reports on LUDB validation
        "p_on":   {"se": 96.4, "ppv": 95.5, "mean":  -0.6, "sd":  9.9, "ref": "DENS-ECG (Moskalenko 2020)"},
        "qrs_on": {"se": 99.6, "ppv": 99.5, "mean":  -1.5, "sd":  4.6, "ref": "DENS-ECG"},
        "t_on":   {"se": 95.0, "ppv": 94.4, "mean":  -2.7, "sd": 13.7, "ref": "DENS-ECG"},
        "p_off":  {"se": 96.4, "ppv": 95.5, "mean":  -0.6, "sd":  9.4, "ref": "DENS-ECG"},
        "qrs_off":{"se": 99.6, "ppv": 99.5, "mean":   1.0, "sd":  6.0, "ref": "DENS-ECG"},
        "t_off":  {"se": 95.7, "ppv": 95.1, "mean":   1.3, "sd": 18.1, "ref": "DENS-ECG"},
    },
    "QTDB": {
        # Martinez 2004 wavelet, the classical QTDB benchmark
        "p_on":   {"se": 98.87, "ppv": 91.03, "mean":  2.0, "sd": 14.8, "ref": "Martinez 2004 wavelet"},
        "qrs_on": {"se": 99.97, "ppv": 99.90, "mean":  4.5, "sd":  7.7, "ref": "Martinez 2004"},
        "t_on":   {"se": None,  "ppv": None,  "mean": None, "sd": None, "ref": "Martinez 2004 (T_on rarely reported)"},
        "p_off":  {"se": 98.75, "ppv": 91.45, "mean":  1.9, "sd": 12.8, "ref": "Martinez 2004"},
        "qrs_off":{"se": 99.97, "ppv": 99.90, "mean":  0.8, "sd": 10.9, "ref": "Martinez 2004"},
        "t_off":  {"se": 99.77, "ppv": 97.79, "mean": -1.6, "sd": 18.1, "ref": "Martinez 2004"},
    },
    "ISP": {
        # SemiSegECG 2025 (semi-supervised SOTA on ISP test)
        "p_on":   {"se": None, "ppv": None, "mean": None, "sd": None, "f1": 0.97, "ref": "SemiSegECG 2025 (F1 @150ms)"},
        "qrs_on": {"se": None, "ppv": None, "mean": None, "sd": None, "f1": 0.99, "ref": "SemiSegECG 2025"},
        "t_on":   {"se": None, "ppv": None, "mean": None, "sd": None, "f1": 0.95, "ref": "SemiSegECG 2025"},
        "p_off":  {"se": None, "ppv": None, "mean": None, "sd": None, "f1": 0.97, "ref": "SemiSegECG 2025"},
        "qrs_off":{"se": None, "ppv": None, "mean": None, "sd": None, "f1": 0.99, "ref": "SemiSegECG 2025"},
        "t_off":  {"se": None, "ppv": None, "mean": None, "sd": None, "f1": 0.96, "ref": "SemiSegECG 2025"},
    },
}


def report_table(model_name, db_label, sota_key, bp, bt):
    print(f"\n=== {model_name} on {db_label} (Martinez per-boundary tolerances) ===", flush=True)
    print(f"{'boundary':10s} {'tol':>5s} | {'F1':>6s} {'Se%':>6s} {'P+%':>6s} | "
          f"{'mean':>7s} {'SD':>6s} {'medAbs':>7s} | {'n_true':>6s} {'n_pred':>6s} | SOTA ref", flush=True)
    print(f"{'-'*10} {'-'*5}-+-{'-'*6} {'-'*6} {'-'*6}-+-"
          f"{'-'*7} {'-'*6} {'-'*7}-+-{'-'*6} {'-'*6}-+", flush=True)
    out = {}
    for k in ("p_on", "qrs_on", "t_on", "p_off", "qrs_off", "t_off"):
        tol = TOL_PER_BOUNDARY[k]
        m = signed_boundary_metrics(bp.get(k, []), bt.get(k, []), tolerance_ms=tol)
        out[k] = m
        sota = SOTA.get(sota_key, {}).get(k, {})
        ref_se = sota.get("se"); ref_ppv = sota.get("ppv")
        ref_mean = sota.get("mean"); ref_sd = sota.get("sd")
        ref_f1 = sota.get("f1"); ref_str = sota.get("ref", "")
        sota_str = ""
        if ref_f1 is not None:
            sota_str = f"F1={ref_f1:.2f} ({ref_str})"
        elif ref_se is not None and ref_mean is not None:
            sota_str = f"Se={ref_se:.1f}%, m={ref_mean:+.1f}±{ref_sd:.1f}ms ({ref_str})"
        elif ref_str:
            sota_str = ref_str
        print(f"{k:10s} {tol:>4d}ms | {m['f1']:.3f}  {m['sens']*100:5.1f}  {m['ppv']*100:5.1f} | "
              f"{m['mean_signed_ms']:+6.1f}  {m['sd_ms']:5.1f}   {m['median_abs_ms']:5.1f} | "
              f"{m['n_true']:>6d} {m['n_pred']:>6d} | {sota_str}", flush=True)
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    OUT_DIR.mkdir(exist_ok=True)

    candidates = [
        ("C", CKPT_DIR / "stage2_v4_C.pt", {"d_model": 128, "n_layers": 8}),
        ("F", CKPT_DIR / "stage2_v4_ludb_only.pt",
         {"d_model": 64, "n_layers": 4, "use_lead_emb": False}),
    ]

    full = {}
    for name, ckpt, mk in candidates:
        print(f"\n{'='*78}\n>>> Model {name} <<<\n{'='*78}", flush=True)
        model = FrameClassifier(**mk)
        load_checkpoint(ckpt, model)
        model = model.to(device).train(False)

        full[name] = {}
        t0 = time.time()
        bp, bt, n = evaluate_ludb(model, device)
        print(f"\n[{time.time()-t0:.1f}s] LUDB val: {n} sequences", flush=True)
        full[name]["LUDB"] = report_table(name, f"LUDB val ({n} seqs)", "LUDB", bp, bt)

        t0 = time.time()
        bp, bt, n = evaluate_isp(model, device)
        print(f"\n[{time.time()-t0:.1f}s] ISP test: {n} sequences", flush=True)
        full[name]["ISP"] = report_table(name, f"ISP test ({n} seqs)", "ISP", bp, bt)

        t0 = time.time()
        bp, bt, n_seqs, n_recs = evaluate_qtdb_t_subset(model, device)
        print(f"\n[{time.time()-t0:.1f}s] QTDB T-subset: {n_recs} records, {n_seqs} sequences", flush=True)
        full[name]["QTDB_T_subset"] = report_table(name, f"QTDB T-subset ({n_recs} recs)", "QTDB", bp, bt)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"sota_comparison_{ts}.json"

    def _safe(o):
        if isinstance(o, dict): return {k: _safe(v) for k, v in o.items()}
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        return o
    out_path.write_text(json.dumps(_safe(full), indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
