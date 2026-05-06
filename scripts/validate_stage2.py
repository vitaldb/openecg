# scripts/validate_stage2.py
"""Validate Stage 2 checkpoint on LUDB val split.

Reports model frame F1, boundary error, side-by-side vs NK direct, per-lead breakdown.
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from openecg import codec, delineate, eval as ee, labeler, ludb, pacer
from openecg.stage2.dataset import LUDBFrameDataset
from openecg.stage2.infer import load_model, predict_frames

CKPT_PATH = Path("data/checkpoints/stage2_v1.pt")
OUT_DIR = Path("out")
FS = 500
FRAME_MS = 20
BOUNDARY_TOLERANCES = {
    "p_on": 50, "p_off": 50,
    "qrs_on": 40, "qrs_off": 40,
    "t_on": 50, "t_off": 100,
}


def _extract_pred_boundaries(events, fs=FS, already_super=False):
    """Extract on/off boundaries from RLE events.

    If `already_super`, event symbols are supercategory IDs (0/1/2/3 = other/P/QRS/T).
    Otherwise they are vocab IDs and get mapped via ee.to_supercategory.
    """
    out = defaultdict(list)
    cum_samples = 0
    prev_super = ee.SUPER_OTHER
    super_to_name = {ee.SUPER_P: "p", ee.SUPER_QRS: "qrs", ee.SUPER_T: "t"}
    for sym, ms in events:
        n = round(ms * fs / 1000.0)
        if already_super:
            cur_super = int(sym)
        else:
            cur_super = int(ee.to_supercategory(np.array([sym], dtype=np.uint8))[0])
        if cur_super != prev_super:
            if prev_super in super_to_name:
                out[f"{super_to_name[prev_super]}_off"].append(cum_samples - 1)
            if cur_super in super_to_name:
                out[f"{super_to_name[cur_super]}_on"].append(cum_samples)
        cum_samples += n
        prev_super = cur_super
    if prev_super in super_to_name:
        out[f"{super_to_name[prev_super]}_off"].append(cum_samples - 1)
    return dict(out)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint {CKPT_PATH}...")
    model = load_model(CKPT_PATH, device=device)

    val_ids = ludb.load_split()["val"]
    print(f"Validating on {len(val_ids)} val records x 12 leads")

    print("Loading val dataset...")
    val_ds = LUDBFrameDataset(val_ids)
    print(f"  {len(val_ds)} sequences cached")

    model_pred_frames = []
    model_true_frames = []
    nk_pred_frames = []
    nk_true_frames = []
    boundary_pred_model = defaultdict(list)
    boundary_pred_nk = defaultdict(list)
    boundary_true = defaultdict(list)
    per_lead_pred = defaultdict(list)
    per_lead_true = defaultdict(list)

    t0 = time.time()
    cum_offset = 0
    for idx in range(len(val_ds)):
        rid, lead = val_ds.items[idx]
        sig_250, lead_idx, true_frames = val_ds.cache[(rid, lead)]

        pred_frames_model = predict_frames(model, sig_250, lead_idx, device=device)
        n_common = min(len(pred_frames_model), len(true_frames))
        model_pred_frames.append(pred_frames_model[:n_common])
        model_true_frames.append(true_frames[:n_common].astype(np.uint8))
        per_lead_pred[lead_idx].append(pred_frames_model[:n_common])
        per_lead_true[lead_idx].append(true_frames[:n_common].astype(np.uint8))

        events_model = codec.from_frames(pred_frames_model, frame_ms=FRAME_MS)
        b_model = _extract_pred_boundaries(events_model, already_super=True)
        for k, v in b_model.items():
            boundary_pred_model[k].extend(int(x) + cum_offset for x in v)

        record = ludb.load_record(rid)
        sig_500 = record[lead]
        dr = delineate.run(sig_500, fs=FS)
        spikes = pacer.detect_spikes(sig_500, fs=FS)
        events_nk = labeler.label(dr, spikes.tolist(), n_samples=len(sig_500), fs=FS)
        nk_pred_super = ee.events_to_super_frames(events_nk, len(sig_500), fs=FS, frame_ms=FRAME_MS)
        n_common_nk = min(len(nk_pred_super), len(true_frames))
        nk_pred_frames.append(nk_pred_super[:n_common_nk])
        nk_true_frames.append(true_frames[:n_common_nk].astype(np.uint8))
        b_nk = _extract_pred_boundaries(events_nk)
        for k, v in b_nk.items():
            boundary_pred_nk[k].extend(int(x) + cum_offset for x in v)

        try:
            gt_ann = ludb.load_annotations(rid, lead)
            for k, v in gt_ann.items():
                if k.endswith("_on") or k.endswith("_off"):
                    boundary_true[k].extend(int(x) + cum_offset for x in v)
        except Exception:
            pass

        cum_offset += len(sig_500)

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(val_ds)}] {time.time()-t0:.1f}s")

    model_pred_concat = np.concatenate(model_pred_frames)
    model_true_concat = np.concatenate(model_true_frames)
    nk_pred_concat = np.concatenate(nk_pred_frames)
    nk_true_concat = np.concatenate(nk_true_frames)
    f1_model = ee.frame_f1(model_pred_concat, model_true_concat)
    f1_nk = ee.frame_f1(nk_pred_concat, nk_true_concat)

    boundary_metrics_model = {}
    boundary_metrics_nk = {}
    for key, tol in BOUNDARY_TOLERANCES.items():
        boundary_metrics_model[key] = ee.boundary_error(
            boundary_pred_model.get(key, []), boundary_true.get(key, []),
            tolerance_ms=tol, fs=FS,
        )
        boundary_metrics_nk[key] = ee.boundary_error(
            boundary_pred_nk.get(key, []), boundary_true.get(key, []),
            tolerance_ms=tol, fs=FS,
        )

    per_lead_f1 = {}
    for lid, preds in per_lead_pred.items():
        trues = per_lead_true[lid]
        p_concat = np.concatenate(preds)
        t_concat = np.concatenate(trues)
        per_lead_f1[lid] = ee.frame_f1(p_concat, t_concat)

    print("\n== Stage 2 v1.0 vs Stage 1 NK direct (LUDB val, supercategory F1) ==\n")
    print(f"{'Class':6s} | {'Model F1':>10s} | {'NK F1':>10s} | {'Delta':>7s}")
    for sc in (ee.SUPER_P, ee.SUPER_QRS, ee.SUPER_T):
        name = ee.SUPER_NAMES[sc]
        m = f1_model[sc]['f1']
        n = f1_nk[sc]['f1']
        print(f"{name:6s} | {m:10.3f} | {n:10.3f} | {m-n:+7.3f}")

    print("\nBoundary error: model / NK (median ms / sens / PPV)")
    for key in BOUNDARY_TOLERANCES:
        mm = boundary_metrics_model[key]
        nn = boundary_metrics_nk[key]
        print(f"  {key:7s} | model: {mm['median_error_ms']:5.1f}/{mm['sensitivity']:.2f}/{mm['ppv']:.2f}  "
              f"| NK: {nn['median_error_ms']:5.1f}/{nn['sensitivity']:.2f}/{nn['ppv']:.2f}")

    print("\nPer-lead F1 (Model):")
    print(f"{'Lead':5s} | {'P':>6s} | {'QRS':>6s} | {'T':>6s}")
    for lid in sorted(per_lead_f1.keys()):
        lead_name = ludb.LEADS_12[lid]
        f = per_lead_f1[lid]
        print(f"{lead_name:5s} | {f[ee.SUPER_P]['f1']:6.3f} | {f[ee.SUPER_QRS]['f1']:6.3f} | {f[ee.SUPER_T]['f1']:6.3f}")

    OUT_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = OUT_DIR / f"validation_stage2_{ts}.json"
    out_file.write_text(json.dumps({
        "model_f1": {ee.SUPER_NAMES[sc]: m for sc, m in f1_model.items()},
        "nk_f1": {ee.SUPER_NAMES[sc]: m for sc, m in f1_nk.items()},
        "boundary_model": boundary_metrics_model,
        "boundary_nk": boundary_metrics_nk,
        "per_lead_f1": {ludb.LEADS_12[lid]: {ee.SUPER_NAMES[sc]: m for sc, m in f.items()}
                        for lid, f in per_lead_f1.items()},
        "n_records": len(val_ids),
    }, indent=2))
    print(f"\nReport: {out_file}")


if __name__ == "__main__":
    main()
