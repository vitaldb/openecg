"""Visualize v12_reg predictions on the 5 LUDB 3rd-degree AV-block records
(rids 34, 74, 90, 104, 111) where the cardiologist GT does NOT contain any
P-wave annotations.

Goal: see whether the current model still predicts P-waves at sensible
positions even though training labels for these records lacked P. If yes
-> pseudo-label / verify path is viable; if no -> we need synthetic /
compositional augmentation as discussed.

Output: out/viz_ludb_avb3/<rid>_<lead>.png with signal + GT (no P) +
v12_reg + NK + WTdel rows for visual comparison.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "WTdelineator"))

from openecg import ludb
from openecg.stage2.dataset import LUDBFrameDataset
from openecg.stage2.infer import (
    apply_reg_to_boundaries, extract_boundaries,
    post_process_frames, predict_frames_with_reg,
)
from openecg.stage2.model import FrameClassifierViTReg
from openecg.stage2.train import load_checkpoint
from scripts.train_v9_q1c_pu_merge import KWARGS
from scripts.viz_test_all_datasets import (
    ann_to_bands, ann_to_lines, baselines_for_signal, frames_to_bands,
    plot_panel, FRAME_MS, FS, WINDOW_SAMPLES,
)

CKPT = REPO / "data" / "checkpoints" / "stage2_v12_reg.pt"
OUT_DIR = REPO / "out" / "viz_ludb_avb3"
AVB3_RIDS = (34, 74, 90, 104, 111)
LEADS = ("ii", "v1", "i", "v5")  # II/V1 best for P visibility


def predict_one(model, sig, lead_idx, device):
    sig = sig.astype(np.float32)
    sig_n = ((sig - sig.mean()) / (sig.std() + 1e-6)).astype(np.float32)
    frames, reg = predict_frames_with_reg(model, sig_n, lead_idx, device=device)
    pp = post_process_frames(frames, frame_ms=FRAME_MS)
    bds = extract_boundaries(pp, fs=FS, frame_ms=FRAME_MS)
    bds_refined = apply_reg_to_boundaries(bds, reg, max_window=len(sig))
    bands = frames_to_bands(pp, len(sig))
    return bds_refined, bands


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    print(f"Loading {CKPT.name}", flush=True)
    model = FrameClassifierViTReg(**KWARGS, n_reg=6)
    load_checkpoint(str(CKPT), model)
    model = model.to(device).train(False)

    ds = LUDBFrameDataset(list(AVB3_RIDS))

    for rid in AVB3_RIDS:
        for lead in LEADS:
            try:
                sig_250, lead_idx, _ = ds.cache[(rid, lead)]
            except KeyError:
                print(f"  miss cache rid={rid} lead={lead}")
                continue
            sig = sig_250[:WINDOW_SAMPLES]
            if len(sig) < WINDOW_SAMPLES:
                continue
            bds, bands = predict_one(model, sig, lead_idx, device)
            try:
                gt_500 = ludb.load_annotations(rid, lead)
            except Exception as e:
                print(f"  ann fail rid={rid} lead={lead}: {e}")
                continue
            gt_250 = {k: [int(s // 2) for s in v] for k, v in gt_500.items()}
            gt_bands = ann_to_bands(gt_250, 0, len(sig))
            gt_lines = ann_to_lines(gt_250, 0, len(sig))
            n_p_pred = len(bds.get("p_on", []))
            n_p_gt = len(gt_500.get("p_on", []))
            nk, nk_bands, wt, wt_bands = baselines_for_signal(sig)
            tracks = [
                (f"GT (cardio, P_gt={n_p_gt})", gt_bands, gt_lines),
                (f"v12_reg (P_pred={n_p_pred})", bands, bds),
                ("NeuroKit", nk_bands, nk),
                ("WTdel", wt_bands, wt),
            ]
            title = (f"LUDB 3°AVB rid={rid} lead={lead}  "
                     f"-- GT P={n_p_gt}, model P={n_p_pred}")
            plot_panel(sig, tracks, title, OUT_DIR / f"{rid:03d}_{lead}.png")
            print(f"  {rid:03d}_{lead}.png  GT_P={n_p_gt}  pred_P={n_p_pred}",
                  flush=True)

    print(f"\nWrote panels to {OUT_DIR}")


if __name__ == "__main__":
    main()
