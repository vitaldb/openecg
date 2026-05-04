"""Re-evaluate QTDB on the subset of records where q1c actually annotated T waves.

QTDB q1c annotates QRS+P on ~all examined beats but T on only ~half. Median
per-record T:QRS ratio in the 10s annotated window is 0. Full QTDB-ext eval
therefore penalises any T prediction on unannotated beats as FP, dragging
T_off F1 to ~0.48.

This script:
  1. Selects records where in-window T:QRS ratio >= 0.8 (T was the analysis target)
  2. Runs C and F on this T-annotated subset
  3. Reports boundary F1 — both raw QTDB-ext for context, and T-annotated subset
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ecgcode import eval as ee, qtdb
from ecgcode.stage2.infer import post_process_frames, predict_frames
from ecgcode.stage2.model import FrameClassifier
from ecgcode.stage2.train import load_checkpoint

CKPT_DIR = Path("data/checkpoints")
OUT_DIR = Path("out")
WINDOW_SAMPLES_250 = 2500
WINDOW_FRAMES = 500
FRAME_MS = 20
TOL_MS = 150


def _normalize(sig):
    mean = float(sig.mean())
    std = float(sig.std()) + 1e-6
    return ((sig - mean) / std).astype(np.float32)


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


def t_annotated_records(min_t_qrs_ratio=0.8):
    """Return list of QTDB record IDs where windowed T:QRS ratio >= threshold."""
    selected = []
    for rid in qtdb.records_with_q1c():
        ann = qtdb.load_q1c(rid)
        win = qtdb.annotated_window(ann, window_samples=WINDOW_SAMPLES_250, fs=250)
        if win is None:
            continue
        start, end = win
        n_q = sum(1 for s in ann["qrs_on"] if start <= s < end)
        n_t = sum(1 for s in ann["t_on"] if start <= s < end)
        if n_q == 0:
            continue
        if n_t / n_q >= min_t_qrs_ratio:
            selected.append(rid)
    return selected


def evaluate(model, rec_ids, device, label):
    """Returns dict with raw + post-proc boundary F1 metrics."""
    bp_raw, bp_pp, bt = defaultdict(list), defaultdict(list), defaultdict(list)
    cum = 0
    n_seqs = 0
    for rid in rec_ids:
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
        win_ann = {k: [s - start for s in v if start <= s < end] for k, v in ann.items()}
        with torch.no_grad():
            pred_raw = predict_frames(model, sig_n, lead_id=1, device=device)
            pred_pp = post_process_frames(pred_raw, frame_ms=FRAME_MS)
        for k, v in _extract_boundaries(pred_raw, fs=250).items():
            bp_raw[k].extend(int(x) + cum for x in v)
        for k, v in _extract_boundaries(pred_pp, fs=250).items():
            bp_pp[k].extend(int(x) + cum for x in v)
        for k in ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off"):
            bt[k].extend(int(s) + cum for s in win_ann[k])
        cum += WINDOW_SAMPLES_250
        n_seqs += 1

    out = {}
    for k in ("p_on", "qrs_on", "t_on", "p_off", "qrs_off", "t_off"):
        m = ee.boundary_f1(bp_pp.get(k, []), bt.get(k, []), tolerance_ms=TOL_MS, fs=250)
        be = ee.boundary_error(bp_pp.get(k, []), bt.get(k, []), tolerance_ms=TOL_MS, fs=250)
        out[k] = {"f1": m["f1"], "sens": m["sensitivity"], "ppv": m["ppv"],
                  "med_err_ms": be["median_error_ms"], "n_true": be["n_true"], "n_pred": be["n_pred"]}
    print(f"  [{label}] n_seqs={n_seqs}", flush=True)
    return out, n_seqs


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    OUT_DIR.mkdir(exist_ok=True)

    print("Selecting T-annotated subset (T:QRS ratio >= 0.8)...", flush=True)
    t_subset = t_annotated_records(min_t_qrs_ratio=0.8)
    all_records = qtdb.records_with_q1c()
    print(f"  T-annotated: {len(t_subset)} of {len(all_records)} records", flush=True)

    candidates = [
        ("C_combined_big_le", CKPT_DIR / "stage2_v4_C.pt", {"d_model": 128, "n_layers": 8}),
        ("F_ludb_only_no_le", CKPT_DIR / "stage2_v4_ludb_only.pt", {"d_model": 64, "n_layers": 4, "use_lead_emb": False}),
    ]

    full = {}
    for name, ckpt, mk in candidates:
        print(f"\n{'='*70}\n{name}\n{'='*70}", flush=True)
        model = FrameClassifier(**mk)
        load_checkpoint(ckpt, model)
        model = model.to(device).train(False)

        full[name] = {}
        print(f"\n  Eval on full QTDB-ext ({len(all_records)} records):", flush=True)
        full_metrics, full_n = evaluate(model, all_records, device, "full")
        print(f"  {'boundary':10s}  F1     Se    PPV   med_err  n_true  n_pred", flush=True)
        for k, m in full_metrics.items():
            print(f"  {k:10s}  {m['f1']:.3f} {m['sens']:.3f} {m['ppv']:.3f} {m['med_err_ms']:5.1f}ms  "
                  f"{m['n_true']:5d}  {m['n_pred']:5d}", flush=True)
        full[name]["full"] = {"metrics": full_metrics, "n_seqs": full_n}

        print(f"\n  Eval on T-annotated subset ({len(t_subset)} records):", flush=True)
        sub_metrics, sub_n = evaluate(model, t_subset, device, "T-subset")
        print(f"  {'boundary':10s}  F1     Se    PPV   med_err  n_true  n_pred", flush=True)
        for k, m in sub_metrics.items():
            print(f"  {k:10s}  {m['f1']:.3f} {m['sens']:.3f} {m['ppv']:.3f} {m['med_err_ms']:5.1f}ms  "
                  f"{m['n_true']:5d}  {m['n_pred']:5d}", flush=True)
        full[name]["t_annotated"] = {"metrics": sub_metrics, "n_seqs": sub_n}

    print("\n" + "="*78, flush=True)
    print(f"{'COMPARISON: Full QTDB-ext vs T-annotated subset':^78}", flush=True)
    print("="*78, flush=True)
    for name in [c[0] for c in candidates]:
        print(f"\n{name}:")
        print(f"  {'boundary':10s}  {'Full F1':>8s}  {'T-sub F1':>9s}  Δ", flush=True)
        for k in ("p_on", "qrs_on", "t_on", "p_off", "qrs_off", "t_off"):
            full_f1 = full[name]["full"]["metrics"][k]["f1"]
            sub_f1 = full[name]["t_annotated"]["metrics"][k]["f1"]
            print(f"  {k:10s}  {full_f1:.3f}     {sub_f1:.3f}      {sub_f1-full_f1:+.3f}", flush=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"eval_qtdb_t_annotated_{ts}.json"

    def _safe(o):
        if isinstance(o, dict): return {k: _safe(v) for k, v in o.items()}
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        return o
    out_path.write_text(json.dumps(_safe(full), indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
