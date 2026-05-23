"""Apples-to-apples benchmark: openecg v56c (1-ch, soft-T α=0.9) vs
NeuroKit2 DWT vs WTdelineator (Ledezma reimpl of Martinez 2004).

Same datasets, same Martinez 2004 tolerances:
  * LUDB val   — 41 records × 12 leads (or lead II only)
  * ISP test   — 72 records × 12 leads (or lead II only)
  * QTDB T-sub — 39 records, first lead per record

Reports per-boundary F1 / Se / P+ / mean signed error / SD (ms) and the
6-boundary macro-F1 used as the canonical eval score.

Result of this script is reproducible: drop the existing
``out/baseline_comparison_*.json`` and re-run to refresh.

Run::

    python -m scripts.benchmark_v56c \\
        --ckpt data/checkpoints/stage2_v45k_noaux_L8_d128_1ch_v56c.pt \\
        --out out/benchmark_v56c.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "WTdelineator"))

from openecg import isp, ludb, qtdb
from openecg.dsp import rank_normalize
from openecg.stage2.dataset import LUDBFrameDataset
from openecg.stage2.evaluate import (
    MARTINEZ_TOLERANCE_MS, signed_boundary_metrics,
)
from openecg.stage2.infer import (
    apply_reg_to_boundaries, extract_boundaries,
    post_process_frames, predict_frames_with_reg,
)
from openecg.stage2.model import load_model_from_ckpt
from openecg.stage2.multi_dataset import _decimate_to_250

WINDOW_SAMPLES = 2500
FS = 250
FRAME_MS = 20
BOUNDARY_KEYS = ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off")


# ---------- Baseline: NeuroKit2 DWT ----------

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


# ---------- Baseline: WTdelineator (Ledezma re-impl of Martinez 2004) ----------

def wt_delineate(sig, fs=FS):
    import WTdelineator as wav
    try:
        Pwav, QRS, Twav = wav.signalDelineation(sig.astype(np.float64), fs)
    except Exception:
        return {k: [] for k in BOUNDARY_KEYS}
    p_on  = [int(p) for p in Pwav[:, 0] if int(p) != 0]
    p_off = [int(p) for p in Pwav[:, -1] if int(p) != 0]
    qrs_on  = [int(p) for p in QRS[:, 0] if int(p) != 0]
    qrs_off = [int(p) for p in QRS[:, -1] if int(p) != 0]
    t_on  = [int(p) for p in Twav[:, 0] if int(p) != 0]
    t_off = [int(p) for p in Twav[:, -1] if int(p) != 0]
    return {"p_on": p_on, "p_off": p_off, "qrs_on": qrs_on,
            "qrs_off": qrs_off, "t_on": t_on, "t_off": t_off}


# ---------- openecg v56c ----------

def v56c_predict(model, sig, lead_idx, device):
    """v56c forward + reg-head boundary refinement.

    Uses rank-normalized input to match training (input_norm=rank).
    Returns sample-indexed boundary dict in the 2500-sample (10s @ 250Hz)
    window's coordinate space.
    """
    sig_in = rank_normalize(sig.astype(np.float32))
    frames, reg = predict_frames_with_reg(model, sig_in, lead_idx, device=device)
    pp = post_process_frames(frames, frame_ms=FRAME_MS)
    bds = extract_boundaries(pp, fs=FS, frame_ms=FRAME_MS)
    if reg is not None:
        bds = apply_reg_to_boundaries(bds, reg,
                                      samples_per_frame=5,
                                      max_window=WINDOW_SAMPLES)
    return bds


# ---------- Aggregation helpers ----------

def _add_b(acc, local, cum):
    for k, v in local.items():
        acc[k].extend(int(x) + cum for x in v)


def metrics_table(label, bp, bt):
    print(f"\n=== {label} ===", flush=True)
    print(f"{'boundary':10s} {'tol':>5s} | {'F1':>6s} {'Se%':>6s} {'P+%':>6s} | "
          f"{'mean':>7s} {'SD':>6s} {'medAbs':>7s} | "
          f"{'n_true':>6s} {'n_pred':>6s}", flush=True)
    summary = {}
    for k in BOUNDARY_KEYS:
        m = signed_boundary_metrics(bp.get(k, []), bt.get(k, []),
                                     tolerance_ms=MARTINEZ_TOLERANCE_MS[k])
        summary[k] = {kk: (float(vv) if isinstance(vv, (np.floating, float))
                            else int(vv) if isinstance(vv, (np.integer, int))
                            else vv) for kk, vv in m.items()}
        print(f"{k:10s} {MARTINEZ_TOLERANCE_MS[k]:>4d}ms | "
              f"{m['f1']:.3f}  {m['sens']*100:5.1f}  {m['ppv']*100:5.1f} | "
              f"{m['mean_signed_ms']:+6.1f}  {m['sd_ms']:5.1f}  "
              f"{m['median_abs_ms']:6.1f} | "
              f"{m['n_true']:>6d} {m['n_pred']:>6d}", flush=True)
    return summary


# ---------- Dataset runners ----------

def run_ludb(model, device, leads_subset, edge_margin_ms=100):
    """LUDB val with labeled-range filtering so unannotated edge beats
    don't inflate FP."""
    rec_ids = ludb.load_split()["val"]
    ds = LUDBFrameDataset(rec_ids)
    bp_v56c, bp_nk, bp_wt, bt = (defaultdict(list) for _ in range(4))
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

            _filter_and_add(bp_v56c, v56c_predict(model, sig_raw, lead_idx, device))
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
    return {"openecg_v56c": (bp_v56c, bt, n),
            "NeuroKit2_DWT": (bp_nk, bt, n),
            "WTdelineator": (bp_wt, bt, n)}


def run_isp(model, device, leads_subset):
    """ISP test (1000 Hz native, decimated to 250 Hz)."""
    rec_ids = isp.load_split()["test"]
    bp_v56c, bp_nk, bp_wt, bt = (defaultdict(list) for _ in range(4))
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
                if len(sig_250) < WINDOW_SAMPLES:
                    pad = np.zeros(WINDOW_SAMPLES - len(sig_250),
                                    dtype=sig_250.dtype)
                    sig_250 = np.concatenate([sig_250, pad])
                sig_250 = sig_250[:WINDOW_SAMPLES].astype(np.float32)
                _add_b(bp_v56c, v56c_predict(model, sig_250, lead_idx, device), cum)
                _add_b(bp_nk, nk_delineate(sig_250), cum)
                _add_b(bp_wt, wt_delineate(sig_250), cum)
                for k, v in ann.items():
                    if k.endswith("_on") or k.endswith("_off"):
                        for s in v:
                            s250 = int(s // 4)
                            if 0 <= s250 < WINDOW_SAMPLES:
                                bt[k].append(s250 + cum)
                cum += WINDOW_SAMPLES
                n += 1
    print(f"ISP: {n} sequences in {time.time()-t0:.1f}s", flush=True)
    return {"openecg_v56c": (bp_v56c, bt, n),
            "NeuroKit2_DWT": (bp_nk, bt, n),
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
    bp_v56c, bp_nk, bp_wt, bt = (defaultdict(list) for _ in range(4))
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
            sig_n = sig.astype(np.float32)
            _add_b(bp_v56c, v56c_predict(model, sig_n, 1, device), cum)
            _add_b(bp_nk, nk_delineate(sig_n), cum)
            _add_b(bp_wt, wt_delineate(sig_n), cum)
            for k in BOUNDARY_KEYS:
                bt[k].extend(int(s - start) + cum
                              for s in ann[k] if start <= s < end)
            cum += WINDOW_SAMPLES
            n += 1
    print(f"QTDB: {n} records in {time.time()-t0:.1f}s", flush=True)
    return {"openecg_v56c": (bp_v56c, bt, n),
            "NeuroKit2_DWT": (bp_nk, bt, n),
            "WTdelineator": (bp_wt, bt, n)}


# ---------- Summary ----------

def summarize_avg(by_model):
    out = {}
    for name, (bp, bt, _n) in by_model.items():
        f1s = []
        for k in BOUNDARY_KEYS:
            m = signed_boundary_metrics(
                bp.get(k, []), bt.get(k, []),
                tolerance_ms=MARTINEZ_TOLERANCE_MS[k])
            f1s.append(m["f1"])
        out[name] = float(np.mean(f1s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        default="data/checkpoints/stage2_v45k_noaux_L8_d128_1ch_v56c.pt",
    )
    ap.add_argument("--out", default="out/benchmark_v56c.json")
    ap.add_argument("--leads", default="ii",
                    help="comma-separated lead subset (e.g. 'ii') or 'all'")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.ckpt} on {device}", flush=True)
    model, blob = load_model_from_ckpt(args.ckpt, device=device)
    model.train(False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"v56c — {n_params:,} params  "
          f"arch={blob['model_config'].get('arch')}", flush=True)

    leads_subset = None if args.leads == "all" else set(
        l.strip() for l in args.leads.split(",") if l.strip())
    leads_label = "all 12 leads" if leads_subset is None else (
        ", ".join(sorted(leads_subset)).upper())

    print("\n" + "=" * 78, flush=True)
    print(f"  openecg v56c vs NeuroKit2 DWT vs WTdelineator on {leads_label}",
          flush=True)
    print("=" * 78, flush=True)

    print("\n--- LUDB val ---", flush=True)
    ludb_res = run_ludb(model, device, leads_subset=leads_subset)
    ludb_per_boundary = {}
    for name, (bp, bt, _) in ludb_res.items():
        ludb_per_boundary[name] = metrics_table(
            f"{name} on LUDB val", bp, bt)

    print("\n--- ISP test ---", flush=True)
    isp_res = run_isp(model, device, leads_subset=leads_subset)
    isp_per_boundary = {}
    for name, (bp, bt, _) in isp_res.items():
        isp_per_boundary[name] = metrics_table(
            f"{name} on ISP test", bp, bt)

    print("\n--- QTDB T-subset ---", flush=True)
    qtdb_res = run_qtdb(model, device)
    qtdb_per_boundary = {}
    for name, (bp, bt, _) in qtdb_res.items():
        qtdb_per_boundary[name] = metrics_table(
            f"{name} on QTDB T-subset", bp, bt)

    avg_lu = summarize_avg(ludb_res)
    avg_is = summarize_avg(isp_res)
    avg_qt = summarize_avg(qtdb_res)

    print("\n\n" + "=" * 78, flush=True)
    print(f"  SUMMARY: 6-boundary macro-F1 (Martinez tolerances)",
          flush=True)
    print("=" * 78, flush=True)
    print(f"  {'model':18s}  {'LUDB val':>10s}  {'ISP test':>10s}  "
          f"{'QTDB T-sub':>12s}", flush=True)
    for name in ("openecg_v56c", "NeuroKit2_DWT", "WTdelineator"):
        print(f"  {name:18s}  {avg_lu[name]:10.3f}  {avg_is[name]:10.3f}  "
              f"{avg_qt[name]:12.3f}", flush=True)

    raw = {
        "ckpt": args.ckpt,
        "n_params": n_params,
        "leads": leads_label,
        "n_sequences": {
            "ludb": ludb_res["openecg_v56c"][2],
            "isp":  isp_res["openecg_v56c"][2],
            "qtdb": qtdb_res["openecg_v56c"][2],
        },
        "per_boundary": {
            "ludb": ludb_per_boundary,
            "isp":  isp_per_boundary,
            "qtdb": qtdb_per_boundary,
        },
        "macro_f1": {
            "ludb": avg_lu, "isp": avg_is, "qtdb": avg_qt,
        },
    }
    out_path.write_text(json.dumps(raw, indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
