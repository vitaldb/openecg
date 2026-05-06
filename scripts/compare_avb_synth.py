"""A/B compare stage2_v12_reg.pt vs stage2_v12_reg_with_synth.pt on AV-block cohorts.

Two test sets:

(A) LUDB 3rd-degree AV block records (rids 34, 74, 90, 104, 111).
    GT itself has zero P-wave annotations on these records (cardiologist
    skipped them). We can only count how many P-waves the model predicts on
    each lead and compare old vs new — more is better, since 10 s windows at
    ~60 bpm should contain ~10 atrial activations.

(B) BUT PDB AV-block records (rids 1, 3, 13, 22; BI/BII/BII/BIII).
    GT has expert-annotated P-peak positions. We compute peak F1 within a
    50 ms tolerance (= 13 samples @ 250 Hz). Predicted P 'peaks' come from
    the midpoint of each P band emitted by the post-processed frames.

Output:
    out/compare_avb_synth_<ts>.md / .json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.signal as scipy_signal
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from openecg import butpdb, ludb
from openecg.stage2.dataset import LUDBFrameDataset
from openecg.stage2.infer import (
    extract_boundaries, post_process_frames, predict_frames_with_reg,
)
from openecg.stage2.model import FrameClassifierViTReg
from openecg.stage2.multi_dataset import QTDB_LEAD_TO_LUDB_ID
from openecg.stage2.train import load_checkpoint
from scripts.train_v9_q1c_pu_merge import KWARGS

CKPT_DIR = REPO / "data" / "checkpoints"
OUT_DIR = REPO / "out"
WINDOW_SAMPLES = 2500
FS = 250
FRAME_MS = 20
PEAK_TOL_MS = 50
PEAK_TOL_SAMPLES = int(round(PEAK_TOL_MS * FS / 1000))

LUDB_AVB3 = (34, 74, 90, 104, 111)
LUDB_LEADS = ("ii", "v1", "i", "v5")
BUTPDB_AVB = butpdb.AVB_RECORDS
BUTPDB_PATHO = {1: "BII", 3: "BIII", 13: "BII", 22: "BI"}


def _load_model(name: str, device: str):
    p = CKPT_DIR / name
    m = FrameClassifierViTReg(**KWARGS, n_reg=6)
    load_checkpoint(str(p), m)
    m = m.to(device).train(False)
    return m


def _predict_p_peaks(model, sig_2500: np.ndarray, lead_id: int, device: str) -> list[int]:
    sig_n = ((sig_2500 - sig_2500.mean()) / (sig_2500.std() + 1e-6)).astype(np.float32)
    frames, _ = predict_frames_with_reg(model, sig_n, lead_id, device=device)
    pp = post_process_frames(frames, frame_ms=FRAME_MS)
    bds = extract_boundaries(pp, fs=FS, frame_ms=FRAME_MS)
    return [(int(on) + int(off)) // 2
            for on, off in zip(bds.get("p_on", []), bds.get("p_off", []))]


def _peak_f1(pred: list[int], gt: list[int], tol: int) -> tuple[float, int, int, int]:
    """1-to-1 match by greedy nearest within tolerance."""
    if not pred and not gt:
        return 1.0, 0, 0, 0
    if not pred or not gt:
        return 0.0, len(pred), len(gt), 0
    pred_s = sorted(pred); gt_s = sorted(gt)
    matched_gt = [False] * len(gt_s)
    tp = 0
    for p in pred_s:
        # nearest unmatched gt within tolerance
        best_idx, best_d = -1, tol + 1
        for j, g in enumerate(gt_s):
            if matched_gt[j]:
                continue
            d = abs(p - g)
            if d <= tol and d < best_d:
                best_d = d; best_idx = j
        if best_idx >= 0:
            matched_gt[best_idx] = True
            tp += 1
    fp = len(pred_s) - tp
    fn = sum(1 for m in matched_gt if not m)
    if tp == 0:
        return 0.0, fp, fn, tp
    prec = tp / (tp + fp); rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec), fp, fn, tp


def scan_ludb_avb3(model, device, label):
    rows = []
    ds = LUDBFrameDataset(list(LUDB_AVB3))
    for rid in LUDB_AVB3:
        for lead in LUDB_LEADS:
            try:
                sig_250, lead_idx, _ = ds.cache[(rid, lead)]
            except KeyError:
                continue
            sig = sig_250[:WINDOW_SAMPLES]
            if len(sig) < WINDOW_SAMPLES:
                continue
            peaks = _predict_p_peaks(model, sig.astype(np.float32), lead_idx, device)
            rows.append({"rid": rid, "lead": lead, "n_p_pred": len(peaks)})
    return rows


def scan_butpdb(model, device):
    rows = []
    for rid in BUTPDB_AVB:
        rec = butpdb.load_record(rid)
        fs_native = rec["fs"]
        sig_full = rec["signal"]
        p_peaks_native = butpdb.load_pwave_peaks(rid)
        if fs_native % FS == 0:
            factor = fs_native // FS
            sig_250 = np.stack(
                [scipy_signal.decimate(sig_full[:, c], factor, zero_phase=True)
                 for c in range(sig_full.shape[1])], axis=-1,
            )
        else:
            n_new = int(round(sig_full.shape[0] * FS / fs_native))
            sig_250 = np.stack(
                [scipy_signal.resample(sig_full[:, c], n_new)
                 for c in range(sig_full.shape[1])], axis=-1,
            )
        scale = sig_250.shape[0] / sig_full.shape[0]
        p_peaks_250 = (p_peaks_native * scale).astype(np.int64)

        n_total = sig_250.shape[0]
        n_windows = n_total // WINDOW_SAMPLES
        for lead_idx in (0, 1):
            model_lead = 1 if lead_idx == 0 else 10
            tp_sum = fp_sum = fn_sum = 0
            n_pred_total = n_gt_total = 0
            for w in range(n_windows):
                lo = w * WINDOW_SAMPLES; hi = lo + WINDOW_SAMPLES
                sig = sig_250[lo:hi, lead_idx].astype(np.float32)
                pred_local = _predict_p_peaks(model, sig, model_lead, "cuda" if torch.cuda.is_available() else "cpu")
                gt_local = [int(s - lo) for s in p_peaks_250 if lo <= s < hi]
                _f1, fp, fn, tp = _peak_f1(pred_local, gt_local, PEAK_TOL_SAMPLES)
                tp_sum += tp; fp_sum += fp; fn_sum += fn
                n_pred_total += len(pred_local); n_gt_total += len(gt_local)
            prec = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) else 0.0
            rec = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            rows.append({
                "rid": rid, "patho": BUTPDB_PATHO[rid], "lead_idx": lead_idx,
                "n_p_pred": n_pred_total, "n_p_gt": n_gt_total,
                "tp": tp_sum, "fp": fp_sum, "fn": fn_sum,
                "precision": prec, "recall": rec, "f1": f1,
            })
    return rows


def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    old = _load_model("stage2_v12_reg.pt", device)
    new = _load_model("stage2_v12_reg_with_synth.pt", device)

    print("Scanning LUDB 3°AVB cohort (no GT P)...", flush=True)
    ludb_old = scan_ludb_avb3(old, device, "old")
    ludb_new = scan_ludb_avb3(new, device, "new")

    print("Scanning BUT PDB AVB cohort (peak F1, tol=50ms)...", flush=True)
    butpdb_old = scan_butpdb(old, device)
    butpdb_new = scan_butpdb(new, device)

    payload = {
        "ludb_avb3": {"old": ludb_old, "new": ludb_new},
        "butpdb_avb": {"old": butpdb_old, "new": butpdb_new},
    }
    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = OUT_DIR / f"compare_avb_synth_{ts}.json"
    json_path.write_text(json.dumps(payload, indent=2))

    md = []
    md.append("# AV-block cohort A/B (v12_reg vs v12_reg+synth)\n")
    md.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}_\n")
    md.append("\n## LUDB 3°AVB (no GT P-wave; report predicted P count per 10 s)\n")
    md.append("| rid | lead | old | new | Δ |")
    md.append("|---|---|---|---|---|")
    by_key_old = {(r["rid"], r["lead"]): r["n_p_pred"] for r in ludb_old}
    by_key_new = {(r["rid"], r["lead"]): r["n_p_pred"] for r in ludb_new}
    for k in sorted(set(by_key_old) | set(by_key_new)):
        o = by_key_old.get(k, 0); n = by_key_new.get(k, 0)
        md.append(f"| {k[0]} | {k[1]} | {o} | {n} | {n - o:+d} |")
    sum_old = sum(by_key_old.values()); sum_new = sum(by_key_new.values())
    md.append(f"| **total** | — | **{sum_old}** | **{sum_new}** | "
              f"**{sum_new - sum_old:+d}** |")

    md.append("\n## BUT PDB AVB (50 ms peak F1)\n")
    md.append("| rid | patho | lead | n_pred old | n_pred new | F1 old | F1 new | Δ |")
    md.append("|---|---|---|---|---|---|---|---|")
    by_old = {(r["rid"], r["lead_idx"]): r for r in butpdb_old}
    by_new = {(r["rid"], r["lead_idx"]): r for r in butpdb_new}
    for k in sorted(by_old.keys()):
        o = by_old[k]; n = by_new[k]
        md.append(f"| {k[0]} | {o['patho']} | {k[1]} | "
                  f"{o['n_p_pred']} | {n['n_p_pred']} | "
                  f"{o['f1']:.3f} | {n['f1']:.3f} | "
                  f"{n['f1'] - o['f1']:+.3f} |")
    f1_old = [r["f1"] for r in butpdb_old]
    f1_new = [r["f1"] for r in butpdb_new]
    md.append(f"| **mean F1** | — | — | — | — | **{np.mean(f1_old):.3f}** | "
              f"**{np.mean(f1_new):.3f}** | "
              f"**{np.mean(f1_new) - np.mean(f1_old):+.3f}** |")

    md_path = OUT_DIR / f"compare_avb_synth_{ts}.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"\nSaved {json_path} and {md_path}")
    print(f"LUDB 3°AVB total predicted P: {sum_old} -> {sum_new}")
    print(f"BUT PDB AVB mean peak F1: {np.mean(f1_old):.3f} -> {np.mean(f1_new):.3f}")


if __name__ == "__main__":
    main()
