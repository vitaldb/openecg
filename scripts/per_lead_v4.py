"""Per-lead boundary F1 breakdown for C and F on LUDB val.

The user's stated goal is single-lead model robust across any of the 12 leads.
This script verifies that by computing boundary F1 (150ms tol) and median timing
error per lead. Per-class consistency across leads = robust deployment.

Output: a 12x6 table (leads x boundary types) of F1, plus min/max/std per
boundary type to surface weak leads.
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from openecg import eval as ee, ludb
from openecg.stage2.dataset import LUDBFrameDataset
from openecg.stage2.infer import post_process_frames, predict_frames
from openecg.stage2.model import FrameClassifier
from openecg.stage2.train import load_checkpoint

CKPT_DIR = Path("data/checkpoints")
OUT_DIR = Path("out")
WINDOW_SAMPLES_250 = 2500
WINDOW_FRAMES = 500
FRAME_MS = 20
TOL_MS = 150


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


def evaluate_per_lead(model, val_ds, device):
    """Group sequences by lead, compute per-lead boundary F1."""
    model.train(False)
    by_lead = {lead: {"bp": defaultdict(list), "bt": defaultdict(list), "cum": 0}
               for lead in ludb.LEADS_12}

    with torch.no_grad():
        for idx in range(len(val_ds)):
            rid, lead = val_ds.items[idx]
            sig_250, lead_idx, true_frames = val_ds.cache[(rid, lead)]
            sig_250 = sig_250[:WINDOW_SAMPLES_250]
            if len(sig_250) < WINDOW_SAMPLES_250:
                continue
            pred_raw = predict_frames(model, sig_250, lead_idx, device=device)
            pred_pp = post_process_frames(pred_raw, frame_ms=FRAME_MS)
            entry = by_lead[lead]
            cum = entry["cum"]
            for k, v in _extract_boundaries(pred_pp, fs=250).items():
                entry["bp"][k].extend(int(x) + cum for x in v)
            try:
                gt_ann = ludb.load_annotations(rid, lead)
                for k, v in gt_ann.items():
                    if k.endswith("_on") or k.endswith("_off"):
                        for s in v:
                            s250 = int(s // 2)
                            if 0 <= s250 < WINDOW_SAMPLES_250:
                                entry["bt"][k].append(s250 + cum)
            except Exception:
                pass
            entry["cum"] += WINDOW_SAMPLES_250

    out = {}
    for lead in ludb.LEADS_12:
        entry = by_lead[lead]
        m = {}
        for k in ("p_on", "qrs_on", "t_on", "p_off", "qrs_off", "t_off"):
            res = ee.boundary_f1(entry["bp"].get(k, []), entry["bt"].get(k, []),
                                 tolerance_ms=TOL_MS, fs=250)
            be = ee.boundary_error(entry["bp"].get(k, []), entry["bt"].get(k, []),
                                   tolerance_ms=TOL_MS, fs=250)
            m[k] = {"f1": res["f1"], "sens": res["sensitivity"], "ppv": res["ppv"],
                    "med_err_ms": be["median_error_ms"], "n_true": be["n_true"]}
        out[lead] = m
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    OUT_DIR.mkdir(exist_ok=True)

    print("Loading LUDB val...", flush=True)
    val_ds = LUDBFrameDataset(ludb.load_split()["val"])
    print(f"  {len(val_ds)} sequences ({len(val_ds)//12} records x 12 leads)", flush=True)

    candidates = [
        ("C_combined_big_le", CKPT_DIR / "stage2_v4_C.pt", {"d_model": 128, "n_layers": 8}),
        ("F_ludb_only_no_le", CKPT_DIR / "stage2_v4_ludb_only.pt", {"d_model": 64, "n_layers": 4, "use_lead_emb": False}),
    ]

    full = {}
    for name, ckpt, mk in candidates:
        print(f"\n{'='*70}\n{name}\n{'='*70}", flush=True)
        model = FrameClassifier(**mk)
        load_checkpoint(ckpt, model)
        model = model.to(device)
        t0 = time.time()
        per_lead = evaluate_per_lead(model, val_ds, device)
        print(f"  Eval done in {time.time()-t0:.1f}s", flush=True)

        # Print per-lead F1 table
        boundaries = ("p_on", "qrs_on", "t_on", "p_off", "qrs_off", "t_off")
        print(f"\n  Per-lead boundary F1 @{TOL_MS}ms tol:", flush=True)
        print(f"  {'lead':5s} | " + " | ".join(f"{b:>7s}" for b in boundaries), flush=True)
        print(f"  {'-'*5}-+-" + "-+-".join("-"*7 for _ in boundaries), flush=True)
        for lead in ludb.LEADS_12:
            row = "  " + f"{lead:5s} | " + " | ".join(
                f"{per_lead[lead][b]['f1']:7.3f}" for b in boundaries)
            print(row, flush=True)

        # Min/max/std per boundary
        print(f"\n  Robustness (across 12 leads):", flush=True)
        print(f"  {'boundary':10s} | {'min':>6s} | {'max':>6s} | {'mean':>6s} | {'std':>6s} | {'min lead':>10s}", flush=True)
        robust = {}
        for b in boundaries:
            vals = np.array([per_lead[lead][b]["f1"] for lead in ludb.LEADS_12])
            min_idx = int(np.argmin(vals))
            min_lead = ludb.LEADS_12[min_idx]
            robust[b] = {"min": float(vals.min()), "max": float(vals.max()),
                         "mean": float(vals.mean()), "std": float(vals.std()),
                         "min_lead": min_lead}
            print(f"  {b:10s} | {vals.min():.3f} | {vals.max():.3f} | "
                  f"{vals.mean():.3f} | {vals.std():.3f} | {min_lead:>10s}", flush=True)

        # Median timing per lead
        print(f"\n  Median timing error (ms) per lead per boundary:", flush=True)
        print(f"  {'lead':5s} | " + " | ".join(f"{b:>7s}" for b in boundaries), flush=True)
        for lead in ludb.LEADS_12:
            row = "  " + f"{lead:5s} | " + " | ".join(
                f"{per_lead[lead][b]['med_err_ms']:7.1f}" for b in boundaries)
            print(row, flush=True)

        full[name] = {"per_lead": per_lead, "robustness": robust}

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"per_lead_v4_{ts}.json"

    def _safe(o):
        if isinstance(o, dict): return {k: _safe(v) for k, v in o.items()}
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        return o
    out_path.write_text(json.dumps(_safe(full), indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
