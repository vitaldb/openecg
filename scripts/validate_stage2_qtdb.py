# scripts/validate_stage2_qtdb.py
"""External validation of Stage 2 v1.0 model on QTDB cardiologist annotations.

Cross-DB generalization test: same model trained on LUDB, evaluated on QTDB.
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from openecg import codec, delineate, eval as ee, labeler, pacer, qtdb
from openecg.stage2.infer import load_model, predict_frames

CKPT_PATH = Path("data/checkpoints/stage2_v1.pt")
OUT_DIR = Path("out")
FS = 250  # QTDB native rate (matches our model input!)
FRAME_MS = 20
WINDOW_SAMPLES = 2500   # 10 s @ 250 Hz
BOUNDARY_TOLERANCES = {
    "p_on": 50, "p_off": 50,
    "qrs_on": 40, "qrs_off": 40,
    "t_on": 50, "t_off": 100,
}


def _extract_pred_boundaries_from_super_frames(super_frames, fs=FS, frame_ms=FRAME_MS):
    """Per-frame supercategory array → boundary sample indices dict."""
    out = defaultdict(list)
    super_to_name = {ee.SUPER_P: "p", ee.SUPER_QRS: "qrs", ee.SUPER_T: "t"}
    prev = ee.SUPER_OTHER
    for f_idx, cur in enumerate(super_frames):
        cur = int(cur)
        if cur != prev:
            sample = f_idx * (frame_ms * fs // 1000)
            if prev in super_to_name:
                out[f"{super_to_name[prev]}_off"].append(sample - 1)
            if cur in super_to_name:
                out[f"{super_to_name[cur]}_on"].append(sample)
        prev = cur
    if prev in super_to_name:
        sample = len(super_frames) * (frame_ms * fs // 1000)
        out[f"{super_to_name[prev]}_off"].append(sample - 1)
    return dict(out)


def _normalize_signal(sig):
    mean = float(sig.mean())
    std = float(sig.std()) + 1e-6
    return ((sig - mean) / std).astype(np.float32)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint {CKPT_PATH}...")
    model = load_model(CKPT_PATH, device=device)

    rec_ids = qtdb.records_with_q1c()
    print(f"QTDB records with q1c annotation: {len(rec_ids)}")

    model_pred_frames = []
    model_true_frames = []
    nk_pred_frames = []
    nk_true_frames = []
    boundary_pred_model = defaultdict(list)
    boundary_pred_nk = defaultdict(list)
    boundary_true = defaultdict(list)

    n_processed = 0
    n_skipped = 0
    cum_offset = 0

    t0 = time.time()
    for ridx, rid in enumerate(rec_ids):
        try:
            record = qtdb.load_record(rid)
            ann = qtdb.load_q1c(rid)
        except Exception as ex:
            print(f"  skip {rid}: {ex}")
            n_skipped += 1
            continue

        win = qtdb.annotated_window(ann, window_samples=WINDOW_SAMPLES, fs=FS)
        if win is None:
            n_skipped += 1
            continue
        start, end = win
        if end > 225000:
            end = 225000
            start = end - WINDOW_SAMPLES

        # First lead (typically MLII) → use lead_id=1 (≈ ii in LUDB)
        first_lead_name = list(record.keys())[0]
        sig = record[first_lead_name][start:end]
        if len(sig) < WINDOW_SAMPLES:
            n_skipped += 1
            continue
        sig_norm = _normalize_signal(sig)

        # Build GT super_frames for the window
        win_ann = {k: [s - start for s in v if start <= s < end] for k, v in ann.items()}
        sample_labels = np.full(WINDOW_SAMPLES, ee.SUPER_OTHER, dtype=np.uint8)
        for on, off in zip(win_ann["p_on"], win_ann["p_off"]):
            sample_labels[max(0, on):min(WINDOW_SAMPLES, off + 1)] = ee.SUPER_P
        for on, off in zip(win_ann["qrs_on"], win_ann["qrs_off"]):
            sample_labels[max(0, on):min(WINDOW_SAMPLES, off + 1)] = ee.SUPER_QRS
        for on, off in zip(win_ann["t_on"], win_ann["t_off"]):
            sample_labels[max(0, on):min(WINDOW_SAMPLES, off + 1)] = ee.SUPER_T
        n_frames = WINDOW_SAMPLES * 1000 // FS // FRAME_MS  # = 500
        samples_per_frame = WINDOW_SAMPLES // n_frames  # = 5
        true_frames = np.zeros(n_frames, dtype=np.uint8)
        for f in range(n_frames):
            seg = sample_labels[f * samples_per_frame: (f + 1) * samples_per_frame]
            if len(seg) == 0:
                continue
            vals, counts = np.unique(seg, return_counts=True)
            true_frames[f] = vals[np.argmax(counts)]

        # Model inference (input is already 250 Hz, normalized)
        pred_frames_model = predict_frames(model, sig_norm, lead_id=1, device=device)
        n_common = min(len(pred_frames_model), len(true_frames))
        model_pred_frames.append(pred_frames_model[:n_common])
        model_true_frames.append(true_frames[:n_common])

        # NK direct on the same window
        dr = delineate.run(sig.astype(np.float64), fs=FS)
        spikes = pacer.detect_spikes(sig.astype(np.float64), fs=FS)
        events_nk = labeler.label(dr, spikes.tolist(), n_samples=WINDOW_SAMPLES, fs=FS)
        nk_pred_super = ee.events_to_super_frames(events_nk, WINDOW_SAMPLES, fs=FS, frame_ms=FRAME_MS)
        n_common_nk = min(len(nk_pred_super), len(true_frames))
        nk_pred_frames.append(nk_pred_super[:n_common_nk])
        nk_true_frames.append(true_frames[:n_common_nk])

        # Boundaries (using consistent supercategory frame extraction for both)
        b_model = _extract_pred_boundaries_from_super_frames(pred_frames_model, fs=FS, frame_ms=FRAME_MS)
        b_nk = _extract_pred_boundaries_from_super_frames(nk_pred_super, fs=FS, frame_ms=FRAME_MS)
        for k, v in b_model.items():
            boundary_pred_model[k].extend(int(x) + cum_offset for x in v)
        for k, v in b_nk.items():
            boundary_pred_nk[k].extend(int(x) + cum_offset for x in v)
        for k in ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off"):
            boundary_true[k].extend(s + cum_offset for s in win_ann[k])

        cum_offset += WINDOW_SAMPLES
        n_processed += 1

        if (ridx + 1) % 20 == 0:
            print(f"  [{ridx+1}/{len(rec_ids)}] {time.time()-t0:.1f}s")

    print(f"\nProcessed {n_processed}, skipped {n_skipped}")

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

    print("\n== Stage 2 v1.0 vs NK direct on QTDB external (q1c, 10s windows) ==\n")
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

    # LUDB val baseline for reference
    print("\nFor reference — LUDB val (in-DB):")
    print("  P=0.604 QRS=0.806 T=0.695 (model)")
    print("  P=0.492 QRS=0.666 T=0.512 (NK direct)")

    OUT_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = OUT_DIR / f"validation_stage2_qtdb_{ts}.json"
    out_file.write_text(json.dumps({
        "n_processed": n_processed,
        "n_skipped": n_skipped,
        "model_f1": {ee.SUPER_NAMES[sc]: m for sc, m in f1_model.items()},
        "nk_f1": {ee.SUPER_NAMES[sc]: m for sc, m in f1_nk.items()},
        "boundary_model": boundary_metrics_model,
        "boundary_nk": boundary_metrics_nk,
    }, indent=2))
    print(f"\nReport: {out_file}")


if __name__ == "__main__":
    main()
