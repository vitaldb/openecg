# scripts/validate_v1.py
"""Validate Stage 1 v1.0 on LUDB val split.

Compares NK pseudo-labels (via our pipeline) against LUDB cardiologist
annotations. Reports per-class frame F1 (4-class supercategory) and
per-boundary-type error metrics (Martinez-style).

Usage:
    $env:ECGCODE_LUDB_ZIP = "..."
    uv run python scripts/validate_v1.py
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ecgcode import codec, delineate, eval as ee, labeler, ludb, pacer, vocab

FS = 500
FRAME_MS = 20
BOUNDARY_TOLERANCES = {
    "p_on": 50, "p_off": 50,
    "qrs_on": 40, "qrs_off": 40,
    "t_on": 50, "t_off": 100,
}
OUT_DIR = Path("out")


def _extract_pred_boundaries(events, fs=FS):
    """Extract predicted boundary sample indices from RLE events.
    Returns dict mirroring LUDB annotation keys (p_on, p_off, qrs_on, qrs_off, t_on, t_off)."""
    out = defaultdict(list)
    cum_samples = 0
    prev_super = ee.SUPER_OTHER
    super_to_name = {ee.SUPER_P: "p", ee.SUPER_QRS: "qrs", ee.SUPER_T: "t"}

    for sym, ms in events:
        n = round(ms * fs / 1000.0)
        cur_super = ee.to_supercategory(np.array([sym], dtype=np.uint8))[0]
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
    split = ludb.load_split()
    val_ids = split["val"]
    print(f"Validating on {len(val_ids)} val records x 12 leads = "
          f"{len(val_ids) * 12} sequences")

    all_pred_frames = []
    all_true_frames = []

    boundary_pred = defaultdict(list)
    boundary_true = defaultdict(list)

    nk_q_count = 0
    rle_q_count = 0

    t0 = time.time()
    for n, rid in enumerate(val_ids, 1):
        record = ludb.load_record(rid)
        cum_offset = 0
        for lead in ludb.LEADS_12:
            sig = record[lead]
            n_samples = len(sig)

            dr = delineate.run(sig, fs=FS)
            spikes = pacer.detect_spikes(sig, fs=FS)
            events = labeler.label(dr, spikes.tolist(), n_samples=n_samples, fs=FS)

            pred_frames = ee.events_to_super_frames(events, n_samples, fs=FS, frame_ms=FRAME_MS)
            try:
                gt_ann = ludb.load_annotations(rid, lead)
            except Exception:
                continue
            true_frames = ee.gt_to_super_frames(gt_ann, n_samples, fs=FS, frame_ms=FRAME_MS)

            n_common = min(len(pred_frames), len(true_frames))
            all_pred_frames.append(pred_frames[:n_common])
            all_true_frames.append(true_frames[:n_common])

            pred_b = _extract_pred_boundaries(events)
            for k, v in gt_ann.items():
                if not (k.endswith("_on") or k.endswith("_off")):
                    continue
                boundary_true[k].extend(int(x) + cum_offset for x in v)
            for k, v in pred_b.items():
                boundary_pred[k].extend(int(x) + cum_offset for x in v)

            cum_offset += n_samples

            if dr.n_beats > 0:
                nk_q_count += int(np.sum(~np.isnan(dr.q_peaks)))
            rle_q_count += sum(1 for s, _ in events if s == vocab.ID_Q)

        if n % 5 == 0:
            print(f"  [{n}/{len(val_ids)}] {time.time() - t0:.1f}s")

    pred_concat = np.concatenate(all_pred_frames)
    true_concat = np.concatenate(all_true_frames)
    f1_metrics = ee.frame_f1(pred_concat, true_concat)

    boundary_metrics = {}
    for key, tol_ms in BOUNDARY_TOLERANCES.items():
        boundary_metrics[key] = ee.boundary_error(
            boundary_pred.get(key, []), boundary_true.get(key, []),
            tolerance_ms=tol_ms, fs=FS,
        )

    q_loss = 1 - (rle_q_count / nk_q_count) if nk_q_count > 0 else 0.0

    print("\n== ECGCode v1.0 Validation on LUDB val ==\n")
    print("Frame-level F1 (4-class supercategory):")
    for sc, m in f1_metrics.items():
        name = ee.SUPER_NAMES[sc]
        print(f"  {name:6s} : F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}")

    print("\nBoundary error (median ms / p95 / sens / PPV):")
    for key, m in boundary_metrics.items():
        print(f"  {key:7s}: {m['median_error_ms']:5.1f} / {m['p95_error_ms']:5.1f} "
              f"/ {m['sensitivity']:.2f} / {m['ppv']:.2f}  "
              f"(hits={m['n_hits']}, true={m['n_true']}, pred={m['n_pred']})")

    print(f"\nQ-loss rate: {q_loss:.1%}  (NK={nk_q_count}, RLE={rle_q_count})")

    OUT_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = OUT_DIR / f"validation_v1_{ts}.json"
    out_file.write_text(json.dumps({
        "frame_f1": {ee.SUPER_NAMES[sc]: m for sc, m in f1_metrics.items()},
        "boundary": boundary_metrics,
        "q_loss_rate": q_loss,
        "n_records": len(val_ids),
    }, indent=2))
    print(f"\nReport saved: {out_file}")


if __name__ == "__main__":
    main()
