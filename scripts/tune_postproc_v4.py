"""Tune post-processing (min_duration_ms, merge_gap_ms) on C and F to close
the LUDB val PPV gap. No retraining; just re-runs inference with different
post-proc params.

Sweep:
  min_duration_ms ∈ {20, 40, 60, 80, 100, 120}
  merge_gap_ms    ∈ {100, 200, 300, 400}

Eval: boundary F1 @150ms tol on LUDB val (where the gap is largest).
Reports best per-boundary F1 and the params that achieve it.
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


def cache_raw_predictions(model, val_ds, device):
    """Cache (raw_pred_frames, true_boundaries dict) per item to avoid re-inference."""
    model.train(False)
    raw_preds = []
    true_boundaries = defaultdict(list)
    cum = 0
    with torch.no_grad():
        for idx in range(len(val_ds)):
            rid, lead = val_ds.items[idx]
            sig_250, lead_idx, true_frames = val_ds.cache[(rid, lead)]
            sig_250 = sig_250[:WINDOW_SAMPLES_250]
            if len(sig_250) < WINDOW_SAMPLES_250:
                continue
            pred = predict_frames(model, sig_250, lead_idx, device=device)
            raw_preds.append((pred, cum))
            try:
                gt_ann = ludb.load_annotations(rid, lead)
                for k, v in gt_ann.items():
                    if k.endswith("_on") or k.endswith("_off"):
                        for s in v:
                            s250 = int(s // 2)
                            if 0 <= s250 < WINDOW_SAMPLES_250:
                                true_boundaries[k].append(s250 + cum)
            except Exception:
                pass
            cum += WINDOW_SAMPLES_250
    return raw_preds, dict(true_boundaries)


def eval_with_params(raw_preds, true_boundaries, min_dur, merge_gap):
    pred_boundaries = defaultdict(list)
    for raw, cum in raw_preds:
        pp = post_process_frames(raw, frame_ms=FRAME_MS,
                                 min_duration_ms=min_dur, merge_gap_ms=merge_gap)
        for k, v in _extract_boundaries(pp, fs=250).items():
            pred_boundaries[k].extend(int(x) + cum for x in v)
    out = {}
    for k in ("p_on", "qrs_on", "t_on", "p_off", "qrs_off", "t_off"):
        m = ee.boundary_f1(pred_boundaries.get(k, []), true_boundaries.get(k, []),
                           tolerance_ms=TOL_MS, fs=250)
        out[k] = {"f1": m["f1"], "sens": m["sensitivity"], "ppv": m["ppv"],
                  "n_true": m["n_true"], "n_pred": m["n_pred"]}
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    OUT_DIR.mkdir(exist_ok=True)

    print("Loading LUDB val...", flush=True)
    val_ds = LUDBFrameDataset(ludb.load_split()["val"])
    print(f"  {len(val_ds)} sequences", flush=True)

    candidates = [
        ("C_combined_big_le", CKPT_DIR / "stage2_v4_C.pt", {"d_model": 128, "n_layers": 8}),
        ("F_ludb_only_no_le", CKPT_DIR / "stage2_v4_ludb_only.pt", {"d_model": 64, "n_layers": 4, "use_lead_emb": False}),
    ]

    min_dur_grid = [20, 40, 60, 80, 100, 120]
    merge_gap_grid = [100, 200, 300, 400]

    full_results = {}
    for name, ckpt, mk in candidates:
        print(f"\n{'='*70}\n{name} from {ckpt}\n{'='*70}", flush=True)
        model = FrameClassifier(**mk)
        load_checkpoint(ckpt, model)
        model = model.to(device)

        print("Caching raw predictions...", flush=True)
        t0 = time.time()
        raw_preds, true_b = cache_raw_predictions(model, val_ds, device)
        print(f"  done in {time.time()-t0:.1f}s ({len(raw_preds)} sequences)", flush=True)

        # Sweep
        sweep = {}
        print(f"\nSweeping {len(min_dur_grid)*len(merge_gap_grid)} param combos...", flush=True)
        for min_dur in min_dur_grid:
            for merge_gap in merge_gap_grid:
                key = f"min{min_dur}_merge{merge_gap}"
                sweep[key] = eval_with_params(raw_preds, true_b, min_dur, merge_gap)

        # Find best F1 per boundary type
        print(f"\nBest per-boundary F1 (sweep over {len(sweep)} combos):", flush=True)
        print(f"  {'boundary':10s}  {'best F1':>8s}  {'best params':>20s}  {'cur F1 (40,300)':>15s}", flush=True)
        cur_key = "min40_merge300"
        best_per_b = {}
        for b in ("p_on", "qrs_on", "t_on", "p_off", "qrs_off", "t_off"):
            best_combo = max(sweep.items(), key=lambda kv: kv[1][b]["f1"])
            best_key, best_metrics = best_combo
            cur_metrics = sweep[cur_key][b]
            best_per_b[b] = {"params": best_key, "f1": best_metrics[b]["f1"],
                             "sens": best_metrics[b]["sens"], "ppv": best_metrics[b]["ppv"],
                             "current_f1": cur_metrics["f1"]}
            print(f"  {b:10s}  {best_metrics[b]['f1']:.3f}    {best_key:>20s}    {cur_metrics['f1']:.3f}",
                  flush=True)

        # Find one combo that's best on average
        avg_scores = {k: np.mean([v[b]["f1"] for b in ("p_on", "qrs_on", "t_on", "p_off", "qrs_off", "t_off")])
                      for k, v in sweep.items()}
        best_avg = max(avg_scores.items(), key=lambda kv: kv[1])
        cur_avg = avg_scores[cur_key]
        print(f"\nBest avg F1: {best_avg[1]:.4f} @ {best_avg[0]}  (vs current {cur_avg:.4f} @ {cur_key})", flush=True)

        # Show full F1 table for current vs best-avg
        print(f"\nFull comparison: {cur_key} (current) vs {best_avg[0]} (best avg):", flush=True)
        print(f"  {'boundary':10s}  {'cur F1':>7s}  {'best F1':>8s}  {'cur PPV':>8s}  {'best PPV':>9s}", flush=True)
        for b in ("p_on", "qrs_on", "t_on", "p_off", "qrs_off", "t_off"):
            cur = sweep[cur_key][b]
            bst = sweep[best_avg[0]][b]
            print(f"  {b:10s}  {cur['f1']:.3f}    {bst['f1']:.3f}    {cur['ppv']:.3f}    {bst['ppv']:.3f}",
                  flush=True)

        full_results[name] = {
            "current_params": cur_key,
            "best_avg_params": best_avg[0],
            "best_avg_f1": best_avg[1],
            "current_avg_f1": cur_avg,
            "sweep": sweep,
            "best_per_boundary": best_per_b,
        }

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"tune_postproc_v4_{ts}.json"

    def _safe(o):
        if isinstance(o, dict): return {k: _safe(v) for k, v in o.items()}
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        return o
    out_path.write_text(json.dumps(_safe(full_results), indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
