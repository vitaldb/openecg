"""Direct apples-to-apples comparison: v4 C vs NeuroKit2 (Martinez DWT) vs
WTdelineator (Ledezma reimplementation of Martinez 2004) on the same splits
and same Martinez tolerances we already use.

Why:
- DENS-ECG paper has no released code, so paper numbers cannot be verified.
- Original Martinez 2004 wavelet code never released; we use two open-source
  reimplementations on identical data instead.
- Reports per-boundary F1 / sens / PPV / mean signed error / SD ms.

Datasets:
- LUDB val (492 sequences, all 12 leads × 41 records OR lead II only)
- ISP test (864 sequences, all 12 leads × 72 records)
- QTDB T-subset (39 records, first lead per record)

Lead II is the conventional baseline lead for delineation; we report both
(a) lead II only and (b) all 12 leads averaged.
"""

import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "WTdelineator"))

from ecgcode import isp, ludb, qtdb
from ecgcode.stage2.dataset import LUDBFrameDataset
from ecgcode.stage2.evaluate import MARTINEZ_TOLERANCE_MS, signed_boundary_metrics
from ecgcode.stage2.infer import (
    extract_boundaries, load_model_bundle,
    post_process_frames, predict_frames,
)
from ecgcode.stage2.multi_dataset import _decimate_to_250, _normalize

CKPT = REPO / "data" / "checkpoints" / "stage2_v4_C.pt"
OUT_DIR = REPO / "out"
WINDOW_SAMPLES = 2500
FS = 250
FRAME_MS = 20
BOUNDARY_KEYS = ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off")


# ---------- Baseline: NeuroKit2 ----------

def nk_delineate(sig, fs=FS):
    import neurokit2 as nk
    try:
        _, info = nk.ecg_delineate(sig, sampling_rate=fs, method="dwt")
    except Exception:
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


# ---------- Baseline: WTdelineator (Ledezma reimpl Martinez 2004) ----------

def wt_delineate(sig, fs=FS):
    import WTdelineator as wav
    try:
        Pwav, QRS, Twav = wav.signalDelineation(sig.astype(np.float64), fs)
    except Exception:
        return {k: [] for k in BOUNDARY_KEYS}
    # WTdelineator returns 0 for non-detected entries
    p_on  = [int(p) for p in Pwav[:, 0] if int(p) != 0]
    p_off = [int(p) for p in Pwav[:, -1] if int(p) != 0]
    qrs_on  = [int(p) for p in QRS[:, 0] if int(p) != 0]
    qrs_off = [int(p) for p in QRS[:, -1] if int(p) != 0]
    t_on  = [int(p) for p in Twav[:, 0] if int(p) != 0]
    t_off = [int(p) for p in Twav[:, -1] if int(p) != 0]
    return {"p_on": p_on, "p_off": p_off, "qrs_on": qrs_on, "qrs_off": qrs_off,
            "t_on": t_on, "t_off": t_off}


# ---------- Our v4 C ----------

def v4_predict(model, sig, lead_idx, device):
    raw = predict_frames(model, sig, lead_idx, device=device)
    pp = post_process_frames(raw, frame_ms=FRAME_MS)
    bds = extract_boundaries(pp, fs=FS, frame_ms=FRAME_MS)
    return bds


# ---------- Aggregation ----------

def add_b(acc, local, cum):
    for k, v in local.items():
        acc[k].extend(int(x) + cum for x in v)


def metrics_table(label, bp, bt):
    print(f"\n=== {label} ===", flush=True)
    print(f"{'boundary':10s} {'tol':>5s} | {'F1':>6s} {'Se%':>6s} {'P+%':>6s} | "
          f"{'mean':>7s} {'SD':>6s} {'medAbs':>7s} | {'n_true':>6s} {'n_pred':>6s}", flush=True)
    summary = {}
    for k in BOUNDARY_KEYS:
        m = signed_boundary_metrics(bp.get(k, []), bt.get(k, []),
                                     tolerance_ms=MARTINEZ_TOLERANCE_MS[k])
        summary[k] = m
        print(f"{k:10s} {MARTINEZ_TOLERANCE_MS[k]:>4d}ms | "
              f"{m['f1']:.3f}  {m['sens']*100:5.1f}  {m['ppv']*100:5.1f} | "
              f"{m['mean_signed_ms']:+6.1f}  {m['sd_ms']:5.1f}  {m['median_abs_ms']:6.1f} | "
              f"{m['n_true']:>6d} {m['n_pred']:>6d}", flush=True)
    return summary


# ---------- Domain runners ----------

def run_ludb(model, device, leads_subset, edge_margin_ms=100):
    """Returns dict {model_name: (bp, bt, n)}.

    Restricts both predictions and GT to the LUDB labeled time range (per
    record-lead, ±edge_margin_ms) so correctly detected edge beats that the
    cardiologist did not annotate (~1.4s start, ~1.3s end on average) do not
    inflate FP and unfairly penalize PPV.
    """
    rec_ids = ludb.load_split()["val"]
    ds = LUDBFrameDataset(rec_ids)
    bp_v4, bp_nk, bp_wt, bt = (defaultdict(list) for _ in range(4))
    cum = 0
    n = 0
    t0 = time.time()
    margin_250 = int(round(edge_margin_ms * FS / 1000.0))
    with torch.no_grad():
        for idx in range(len(ds)):
            rid, lead = ds.items[idx]
            if leads_subset and lead not in leads_subset:
                continue
            sig_250, lead_idx, _ = ds.cache[(rid, lead)]
            sig_250 = sig_250[:WINDOW_SAMPLES]
            if len(sig_250) < WINDOW_SAMPLES:
                continue
            sig_raw = sig_250.astype(np.float32)
            rng = ludb.labeled_range(rid, lead)
            if rng is None:
                continue
            lo_250 = max(0, rng[0] // 2 - margin_250)
            hi_250 = min(WINDOW_SAMPLES, rng[1] // 2 + margin_250 + 1)

            def _filter_and_add(acc, preds):
                for k, vs in preds.items():
                    for s in vs:
                        if lo_250 <= s < hi_250:
                            acc[k].append(int(s) + cum)

            _filter_and_add(bp_v4, v4_predict(model, sig_raw, lead_idx, device))
            _filter_and_add(bp_nk, nk_delineate(sig_raw))
            _filter_and_add(bp_wt, wt_delineate(sig_raw))
            try:
                gt = ludb.load_annotations(rid, lead)
                for k, v in gt.items():
                    if k.endswith("_on") or k.endswith("_off"):
                        for s in v:
                            s250 = int(s // 2)
                            if lo_250 <= s250 < hi_250:
                                bt[k].append(s250 + cum)
            except Exception:
                pass
            cum += WINDOW_SAMPLES
            n += 1
    print(f"LUDB: {n} sequences in {time.time()-t0:.1f}s "
          f"(labeled-range filtered ±{edge_margin_ms}ms)", flush=True)
    return {"v4_C": (bp_v4, bt, n), "NeuroKit2_DWT": (bp_nk, bt, n),
            "WTdelineator": (bp_wt, bt, n)}


def run_isp(model, device, leads_subset):
    rec_ids = isp.load_split()["test"]
    bp_v4, bp_nk, bp_wt, bt = (defaultdict(list) for _ in range(4))
    cum = 0
    n = 0
    t0 = time.time()
    with torch.no_grad():
        for rid in rec_ids:
            try:
                record = isp.load_record(rid, split="test")
                ann = isp.load_annotations_as_super(rid, split="test")
            except Exception:
                continue
            for lead_idx, lead in enumerate(isp.LEADS_12):
                if leads_subset and lead not in leads_subset:
                    continue
                sig_1000 = record[lead]
                sig_250 = _decimate_to_250(sig_1000, 1000)
                sig_n = _normalize(sig_250)
                if len(sig_n) < WINDOW_SAMPLES:
                    pad = np.zeros(WINDOW_SAMPLES - len(sig_n), dtype=sig_n.dtype)
                    sig_n = np.concatenate([sig_n, pad])
                sig_n = sig_n[:WINDOW_SAMPLES]
                add_b(bp_v4, v4_predict(model, sig_n, lead_idx, device), cum)
                add_b(bp_nk, nk_delineate(sig_n), cum)
                add_b(bp_wt, wt_delineate(sig_n), cum)
                for k, v in ann.items():
                    if k.endswith("_on") or k.endswith("_off"):
                        for s in v:
                            s250 = int(s // 4)
                            if 0 <= s250 < WINDOW_SAMPLES:
                                bt[k].append(s250 + cum)
                cum += WINDOW_SAMPLES
                n += 1
    print(f"ISP: {n} sequences in {time.time()-t0:.1f}s", flush=True)
    return {"v4_C": (bp_v4, bt, n), "NeuroKit2_DWT": (bp_nk, bt, n),
            "WTdelineator": (bp_wt, bt, n)}


def run_qtdb(model, device):
    """QTDB T-subset, first lead per record (Martinez 2004 convention)."""
    rids = []
    for rid in qtdb.records_with_q1c():
        ann = qtdb.load_q1c(rid)
        win = qtdb.annotated_window(ann, window_samples=WINDOW_SAMPLES, fs=FS)
        if win is None:
            continue
        start, end = win
        n_q = sum(1 for s in ann["qrs_on"] if start <= s < end)
        n_t = sum(1 for s in ann["t_on"] if start <= s < end)
        if n_q > 0 and n_t / n_q >= 0.8:
            rids.append(rid)
    bp_v4, bp_nk, bp_wt, bt = (defaultdict(list) for _ in range(4))
    cum = 0
    n = 0
    t0 = time.time()
    with torch.no_grad():
        for rid in rids:
            try:
                record = qtdb.load_record(rid)
                ann = qtdb.load_q1c(rid)
            except Exception:
                continue
            win = qtdb.annotated_window(ann, window_samples=WINDOW_SAMPLES, fs=FS)
            if win is None:
                continue
            start, end = win
            if end > 225000:
                end = 225000
                start = end - WINDOW_SAMPLES
            first_lead = list(record.keys())[0]
            sig = record[first_lead][start:end]
            if len(sig) < WINDOW_SAMPLES:
                continue
            sig_n = _normalize(sig)
            add_b(bp_v4, v4_predict(model, sig_n, 1, device), cum)
            add_b(bp_nk, nk_delineate(sig_n), cum)
            add_b(bp_wt, wt_delineate(sig_n), cum)
            for k in BOUNDARY_KEYS:
                bt[k].extend(int(s - start) + cum
                              for s in ann[k] if start <= s < end)
            cum += WINDOW_SAMPLES
            n += 1
    print(f"QTDB: {n} records in {time.time()-t0:.1f}s", flush=True)
    return {"v4_C": (bp_v4, bt, n), "NeuroKit2_DWT": (bp_nk, bt, n),
            "WTdelineator": (bp_wt, bt, n)}


# ---------- Main ----------

def summarize_avg(by_model):
    """Returns dict {model: avg_F1_across_6_boundaries}."""
    out = {}
    for name, (bp, bt, _n) in by_model.items():
        f1s = []
        for k in BOUNDARY_KEYS:
            m = signed_boundary_metrics(bp.get(k, []), bt.get(k, []),
                                         tolerance_ms=MARTINEZ_TOLERANCE_MS[k])
            f1s.append(m["f1"])
        out[name] = float(np.mean(f1s))
    return out


def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {CKPT}", flush=True)
    bundle = load_model_bundle(CKPT, device=device)
    model = bundle["model"].train(False)

    # Lead II only first (conventional baseline). Then all 12 leads if time allows.
    print("\n" + "="*78, flush=True)
    print(f"{'Phase 1: lead II only (conventional baseline)':^78}", flush=True)
    print("="*78, flush=True)

    print("\n--- LUDB val (lead ii only) ---", flush=True)
    ludb_lead2 = run_ludb(model, device, leads_subset={"ii"})
    for name, (bp, bt, _) in ludb_lead2.items():
        metrics_table(f"{name} on LUDB val (lead ii)", bp, bt)

    print("\n--- ISP test (lead ii only) ---", flush=True)
    isp_lead2 = run_isp(model, device, leads_subset={"ii"})
    for name, (bp, bt, _) in isp_lead2.items():
        metrics_table(f"{name} on ISP test (lead ii)", bp, bt)

    print("\n--- QTDB T-subset (first lead, typically MLII) ---", flush=True)
    qtdb_res = run_qtdb(model, device)
    for name, (bp, bt, _) in qtdb_res.items():
        metrics_table(f"{name} on QTDB T-subset", bp, bt)

    # Summary table
    avg_lu = summarize_avg(ludb_lead2)
    avg_is = summarize_avg(isp_lead2)
    avg_qt = summarize_avg(qtdb_res)

    print("\n\n" + "="*78, flush=True)
    print(f"{'SUMMARY: avg Martinez boundary F1 across 6 boundaries':^78}", flush=True)
    print("="*78, flush=True)
    print(f"  {'model':18s}  {'LUDB val':>10s}  {'ISP test':>10s}  {'QTDB T-sub':>12s}", flush=True)
    for name in ("v4_C", "NeuroKit2_DWT", "WTdelineator"):
        print(f"  {name:18s}  {avg_lu[name]:10.3f}  {avg_is[name]:10.3f}  {avg_qt[name]:12.3f}", flush=True)

    # Save full
    import json
    def _safe(v):
        if isinstance(v, dict): return {k: _safe(x) for k, x in v.items()}
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        if isinstance(v, tuple) and len(v) == 3:
            bp, bt, n = v
            return {"n_seq": int(n)}
        return v
    raw = {
        "ludb_lead2": _safe({k: v for k, v in ludb_lead2.items()}),
        "isp_lead2":  _safe({k: v for k, v in isp_lead2.items()}),
        "qtdb": _safe({k: v for k, v in qtdb_res.items()}),
        "avg_lu": avg_lu, "avg_is": avg_is, "avg_qt": avg_qt,
    }
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"baseline_comparison_{ts}.json"
    out_path.write_text(json.dumps(raw, indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
