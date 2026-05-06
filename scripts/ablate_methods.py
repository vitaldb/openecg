# scripts/ablate_methods.py
"""Run validation pipeline with each NK delineate method, report comparison.

Usage:
    $env:OPENECG_LUDB_ZIP = "..."
    uv run python scripts/ablate_methods.py
"""

import json
import time
from pathlib import Path

import numpy as np

from openecg import delineate, eval as ee, labeler, ludb, pacer, vocab

METHODS = ["dwt", "cwt", "peak", "prominence"]
FS = 500
FRAME_MS = 20
OUT_DIR = Path("out")


def evaluate_method(method: str, val_ids: list[int]) -> dict:
    """Run pipeline + frame F1 evaluation with given NK method on val split."""
    all_pred = []
    all_true = []
    nk_q_count = 0
    rle_q_count = 0
    n_failed = 0

    for rid in val_ids:
        record = ludb.load_record(rid)
        for lead in ludb.LEADS_12:
            sig = record[lead]
            n_samples = len(sig)
            try:
                dr = delineate.run(sig, fs=FS, method=method)
            except Exception:
                n_failed += 1
                continue
            spikes = pacer.detect_spikes(sig, fs=FS)
            events = labeler.label(dr, spikes.tolist(), n_samples=n_samples, fs=FS)

            pred_frames = ee.events_to_super_frames(events, n_samples, fs=FS, frame_ms=FRAME_MS)
            try:
                gt_ann = ludb.load_annotations(rid, lead)
            except Exception:
                continue
            true_frames = ee.gt_to_super_frames(gt_ann, n_samples, fs=FS, frame_ms=FRAME_MS)

            n_common = min(len(pred_frames), len(true_frames))
            all_pred.append(pred_frames[:n_common])
            all_true.append(true_frames[:n_common])

            if dr.n_beats > 0:
                nk_q_count += int(np.sum(~np.isnan(dr.q_peaks)))
            rle_q_count += sum(1 for s, _ in events if s == vocab.ID_Q)

    if not all_pred:
        return {"f1": None, "q_loss": None, "failed": n_failed}

    pred_concat = np.concatenate(all_pred)
    true_concat = np.concatenate(all_true)
    f1 = ee.frame_f1(pred_concat, true_concat)
    q_loss = 1 - rle_q_count / nk_q_count if nk_q_count > 0 else 0.0
    return {"f1": f1, "q_loss": q_loss, "failed": n_failed,
            "nk_q": nk_q_count, "rle_q": rle_q_count}


def main():
    val_ids = ludb.load_split()["val"]
    print(f"Ablating {len(METHODS)} NK methods on {len(val_ids)} val records x 12 leads\n")

    results = {}
    for method in METHODS:
        print(f"--- {method} ---")
        t0 = time.time()
        try:
            results[method] = evaluate_method(method, val_ids)
        except Exception as exc:
            print(f"  FAILED entire method: {exc}\n")
            results[method] = None
            continue
        elapsed = time.time() - t0
        r = results[method]
        if r["f1"] is None:
            print(f"  no valid sequences ({elapsed:.1f}s)\n")
            continue
        print(f"  P F1   = {r['f1'][ee.SUPER_P]['f1']:.3f}  P={r['f1'][ee.SUPER_P]['precision']:.3f}  R={r['f1'][ee.SUPER_P]['recall']:.3f}")
        print(f"  QRS F1 = {r['f1'][ee.SUPER_QRS]['f1']:.3f}  P={r['f1'][ee.SUPER_QRS]['precision']:.3f}  R={r['f1'][ee.SUPER_QRS]['recall']:.3f}")
        print(f"  T F1   = {r['f1'][ee.SUPER_T]['f1']:.3f}  P={r['f1'][ee.SUPER_T]['precision']:.3f}  R={r['f1'][ee.SUPER_T]['recall']:.3f}")
        print(f"  Q-loss = {r['q_loss']:.1%}")
        print(f"  failed sequences: {r['failed']}")
        print(f"  ({elapsed:.1f}s)\n")

    valid = {m: r for m, r in results.items() if r and r["f1"] is not None}
    if valid:
        winner = max(valid.keys(),
                     key=lambda m: valid[m]["f1"][ee.SUPER_QRS]["f1"])
        print(f"== Winner: '{winner}' (highest QRS F1 = "
              f"{valid[winner]['f1'][ee.SUPER_QRS]['f1']:.3f}) ==")

    OUT_DIR.mkdir(exist_ok=True)
    out_file = OUT_DIR / f"ablation_{time.strftime('%Y%m%d_%H%M%S')}.json"
    serializable = {}
    for m, r in results.items():
        if r is None or r["f1"] is None:
            serializable[m] = None
            continue
        serializable[m] = {
            "f1": {ee.SUPER_NAMES[k]: v for k, v in r["f1"].items()},
            "q_loss": r["q_loss"],
            "failed": r["failed"],
            "nk_q": r["nk_q"],
            "rle_q": r["rle_q"],
        }
    out_file.write_text(json.dumps(serializable, indent=2))
    print(f"\nReport: {out_file}")


if __name__ == "__main__":
    main()
