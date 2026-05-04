# scripts/validate_stage2_v31.py
"""Validate Stage 2 v3.1 multi-DB checkpoint.

Reports:
- Frame-level F1 (raw model output, no post-proc)
- Frame-level F1 (with post-processing)
- Boundary-level F1 (literature-comparable, 150ms tolerance)

Sources: LUDB val + ISP test (and optional QTDB external).
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ecgcode import eval as ee, isp, ludb, qtdb
from ecgcode.stage2.dataset import LUDBFrameDataset
from ecgcode.stage2.infer import load_model, post_process_frames, predict_frames

CKPT_PATH = Path("data/checkpoints/stage2_v31.pt")
OUT_DIR = Path("out")
FRAME_MS = 20
WINDOW_SAMPLES_250 = 2500
WINDOW_FRAMES = 500
LITERATURE_TOL_MS = 150  # SemiSegECG / SOTA boundary tolerance


# -----------------------------------------------------------------------------
# Boundary extraction
# -----------------------------------------------------------------------------

def _extract_pred_boundaries_from_super_frames(super_frames, fs, frame_ms=FRAME_MS):
    """Per-frame supercategory array -> boundary sample indices dict."""
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


def _gt_to_super_frames_from_dict(gt_ann, n_samples, fs):
    """Convert LUDB-style annotation dict to per-frame super array (250Hz output)."""
    return ee.gt_to_super_frames(gt_ann, n_samples=n_samples, fs=fs, frame_ms=FRAME_MS)


# -----------------------------------------------------------------------------
# Per-DB evaluation
# -----------------------------------------------------------------------------

def _normalize(sig):
    mean = float(sig.mean())
    std = float(sig.std()) + 1e-6
    return ((sig - mean) / std).astype(np.float32)


def _decimate(sig, native_fs, target_fs=250):
    if native_fs == target_fs:
        return sig.astype(np.float64)
    factor = native_fs // target_fs
    import scipy.signal as scipy_signal
    return scipy_signal.decimate(sig, factor, zero_phase=True)


def evaluate_ludb_val(model, device):
    """Run on LUDB val (1908/12 = 159 records, 12 leads each)."""
    val_ids = ludb.load_split()["val"]
    print(f"  LUDB val: {len(val_ids)} records x 12 leads")
    val_ds = LUDBFrameDataset(val_ids)

    raw_pred, raw_true = [], []
    pp_pred, pp_true = [], []
    boundary_pred_raw = defaultdict(list)
    boundary_pred_pp = defaultdict(list)
    boundary_true = defaultdict(list)

    cum_offset = 0  # in samples at 250Hz (for boundary extraction)
    samples_per_window = WINDOW_SAMPLES_250

    for idx in range(len(val_ds)):
        rid, lead = val_ds.items[idx]
        sig_250, lead_idx, true_frames = val_ds.cache[(rid, lead)]
        # Truncate to window (some records may have slight differences)
        true_frames = true_frames[:WINDOW_FRAMES].astype(np.uint8)
        sig_250 = sig_250[:WINDOW_SAMPLES_250]
        if len(sig_250) < WINDOW_SAMPLES_250:
            continue

        pred_raw = predict_frames(model, sig_250, lead_idx, device=device)
        pred_pp = post_process_frames(pred_raw, frame_ms=FRAME_MS)

        n = min(len(pred_raw), len(true_frames))
        raw_pred.append(pred_raw[:n])
        raw_true.append(true_frames[:n])
        pp_pred.append(pred_pp[:n])
        pp_true.append(true_frames[:n])

        # Boundaries (in 250 Hz sample indices)
        b_raw = _extract_pred_boundaries_from_super_frames(pred_raw, fs=250)
        b_pp = _extract_pred_boundaries_from_super_frames(pred_pp, fs=250)
        for k, v in b_raw.items():
            boundary_pred_raw[k].extend(int(x) + cum_offset for x in v)
        for k, v in b_pp.items():
            boundary_pred_pp[k].extend(int(x) + cum_offset for x in v)

        # GT boundaries: load original ann (500Hz native), convert to 250Hz indices
        try:
            gt_ann = ludb.load_annotations(rid, lead)
            for k, v in gt_ann.items():
                if k.endswith("_on") or k.endswith("_off"):
                    for s in v:
                        sample_250 = int(s // 2)  # 500->250 Hz
                        if 0 <= sample_250 < samples_per_window:
                            boundary_true[k].append(sample_250 + cum_offset)
        except Exception:
            pass

        cum_offset += samples_per_window

    return {
        "raw_pred": np.concatenate(raw_pred),
        "raw_true": np.concatenate(raw_true),
        "pp_pred": np.concatenate(pp_pred),
        "pp_true": np.concatenate(pp_true),
        "boundary_pred_raw": dict(boundary_pred_raw),
        "boundary_pred_pp": dict(boundary_pred_pp),
        "boundary_true": dict(boundary_true),
        "fs": 250,
        "n_seqs": len(val_ds),
    }


def evaluate_isp_test(model, device):
    """Run on ISP test (held-out set)."""
    rec_ids = isp.load_split()["test"]
    print(f"  ISP test: {len(rec_ids)} records x 12 leads")

    raw_pred, raw_true = [], []
    pp_pred, pp_true = [], []
    boundary_pred_raw = defaultdict(list)
    boundary_pred_pp = defaultdict(list)
    boundary_true = defaultdict(list)

    cum_offset = 0
    samples_per_window = WINDOW_SAMPLES_250

    for ridx, rid in enumerate(rec_ids):
        try:
            record = isp.load_record(rid, split="test")
            ann_super = isp.load_annotations_as_super(rid, split="test")
        except Exception:
            continue

        for lead_idx, lead in enumerate(isp.LEADS_12):
            sig_1000 = record[lead]
            sig_250 = _decimate(sig_1000, native_fs=1000)
            sig_n = _normalize(sig_250)
            if len(sig_n) < WINDOW_SAMPLES_250:
                continue
            sig_n = sig_n[:WINDOW_SAMPLES_250]

            true_frames = ee.gt_to_super_frames(
                ann_super, n_samples=len(sig_1000), fs=1000, frame_ms=FRAME_MS,
            ).astype(np.uint8)
            true_frames = true_frames[:WINDOW_FRAMES]
            if len(true_frames) < WINDOW_FRAMES:
                continue

            pred_raw = predict_frames(model, sig_n, lead_idx, device=device)
            pred_pp = post_process_frames(pred_raw, frame_ms=FRAME_MS)

            n = min(len(pred_raw), len(true_frames))
            raw_pred.append(pred_raw[:n])
            raw_true.append(true_frames[:n])
            pp_pred.append(pred_pp[:n])
            pp_true.append(true_frames[:n])

            b_raw = _extract_pred_boundaries_from_super_frames(pred_raw, fs=250)
            b_pp = _extract_pred_boundaries_from_super_frames(pred_pp, fs=250)
            for k, v in b_raw.items():
                boundary_pred_raw[k].extend(int(x) + cum_offset for x in v)
            for k, v in b_pp.items():
                boundary_pred_pp[k].extend(int(x) + cum_offset for x in v)

            for k, v in ann_super.items():
                if k.endswith("_on") or k.endswith("_off"):
                    for s in v:
                        sample_250 = int(s // 4)  # 1000->250 Hz
                        if 0 <= sample_250 < samples_per_window:
                            boundary_true[k].append(sample_250 + cum_offset)

            cum_offset += samples_per_window

        if (ridx + 1) % 25 == 0:
            print(f"    [{ridx+1}/{len(rec_ids)}]")

    return {
        "raw_pred": np.concatenate(raw_pred),
        "raw_true": np.concatenate(raw_true),
        "pp_pred": np.concatenate(pp_pred),
        "pp_true": np.concatenate(pp_true),
        "boundary_pred_raw": dict(boundary_pred_raw),
        "boundary_pred_pp": dict(boundary_pred_pp),
        "boundary_true": dict(boundary_true),
        "fs": 250,
        "n_seqs": len(raw_pred),
    }


def evaluate_qtdb(model, device):
    """Run on QTDB external (q1c-annotated only)."""
    rec_ids = qtdb.records_with_q1c()
    print(f"  QTDB q1c: {len(rec_ids)} records")

    raw_pred, raw_true = [], []
    pp_pred, pp_true = [], []
    boundary_pred_raw = defaultdict(list)
    boundary_pred_pp = defaultdict(list)
    boundary_true = defaultdict(list)

    cum_offset = 0

    for rid in rec_ids:
        try:
            record = qtdb.load_record(rid)
            ann = qtdb.load_q1c(rid)
        except Exception:
            continue
        win = qtdb.annotated_window(ann, window_samples=WINDOW_SAMPLES_250, fs=250)
        if win is None:
            continue
        start, end = win
        if end > 225000:
            end = 225000
            start = end - WINDOW_SAMPLES_250

        first_lead = list(record.keys())[0]
        sig = record[first_lead][start:end]
        if len(sig) < WINDOW_SAMPLES_250:
            continue
        sig_n = _normalize(sig)

        # GT super_frames for the window
        win_ann = {k: [s - start for s in v if start <= s < end] for k, v in ann.items()}
        sample_labels = np.full(WINDOW_SAMPLES_250, ee.SUPER_OTHER, dtype=np.uint8)
        for on, off in zip(win_ann["p_on"], win_ann["p_off"]):
            sample_labels[max(0, on):min(WINDOW_SAMPLES_250, off + 1)] = ee.SUPER_P
        for on, off in zip(win_ann["qrs_on"], win_ann["qrs_off"]):
            sample_labels[max(0, on):min(WINDOW_SAMPLES_250, off + 1)] = ee.SUPER_QRS
        for on, off in zip(win_ann["t_on"], win_ann["t_off"]):
            sample_labels[max(0, on):min(WINDOW_SAMPLES_250, off + 1)] = ee.SUPER_T
        n_frames = WINDOW_FRAMES
        samples_per_frame = WINDOW_SAMPLES_250 // n_frames
        true_frames = np.zeros(n_frames, dtype=np.uint8)
        for f in range(n_frames):
            seg = sample_labels[f * samples_per_frame:(f + 1) * samples_per_frame]
            vals, counts = np.unique(seg, return_counts=True)
            true_frames[f] = vals[np.argmax(counts)]

        pred_raw = predict_frames(model, sig_n, lead_id=1, device=device)
        pred_pp = post_process_frames(pred_raw, frame_ms=FRAME_MS)

        n = min(len(pred_raw), len(true_frames))
        raw_pred.append(pred_raw[:n])
        raw_true.append(true_frames[:n])
        pp_pred.append(pred_pp[:n])
        pp_true.append(true_frames[:n])

        b_raw = _extract_pred_boundaries_from_super_frames(pred_raw, fs=250)
        b_pp = _extract_pred_boundaries_from_super_frames(pred_pp, fs=250)
        for k, v in b_raw.items():
            boundary_pred_raw[k].extend(int(x) + cum_offset for x in v)
        for k, v in b_pp.items():
            boundary_pred_pp[k].extend(int(x) + cum_offset for x in v)
        for k in ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off"):
            boundary_true[k].extend(int(s) + cum_offset for s in win_ann[k])

        cum_offset += WINDOW_SAMPLES_250

    if not raw_pred:
        return None
    return {
        "raw_pred": np.concatenate(raw_pred),
        "raw_true": np.concatenate(raw_true),
        "pp_pred": np.concatenate(pp_pred),
        "pp_true": np.concatenate(pp_true),
        "boundary_pred_raw": dict(boundary_pred_raw),
        "boundary_pred_pp": dict(boundary_pred_pp),
        "boundary_true": dict(boundary_true),
        "fs": 250,
        "n_seqs": len(raw_pred),
    }


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def _frame_f1_row(pred, true, label):
    f = ee.frame_f1(pred, true)
    return f"{label:10s}| {f[ee.SUPER_P]['f1']:5.3f} | {f[ee.SUPER_QRS]['f1']:5.3f} | {f[ee.SUPER_T]['f1']:5.3f}"


def _boundary_f1_row(boundary_pred, boundary_true, fs, label, tol_ms=LITERATURE_TOL_MS):
    parts = [f"{label:10s}"]
    keys = ("p_on", "qrs_on", "t_on", "p_off", "qrs_off", "t_off")
    metrics = {}
    for k in keys:
        m = ee.boundary_f1(
            boundary_pred.get(k, []),
            boundary_true.get(k, []),
            tolerance_ms=tol_ms, fs=fs,
        )
        metrics[k] = m
        parts.append(f"{m['f1']:5.3f}")
    return "| ".join(parts), metrics


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint {CKPT_PATH}...")
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    model = load_model(CKPT_PATH, device=device, d_model=128, n_layers=8)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    print("\nEvaluating LUDB val...")
    t0 = time.time()
    ludb_res = evaluate_ludb_val(model, device)
    print(f"  done in {time.time()-t0:.1f}s ({ludb_res['n_seqs']} sequences)")

    print("\nEvaluating ISP test...")
    t0 = time.time()
    isp_res = evaluate_isp_test(model, device)
    print(f"  done in {time.time()-t0:.1f}s ({isp_res['n_seqs']} sequences)")

    print("\nEvaluating QTDB external (cross-DB)...")
    t0 = time.time()
    qtdb_res = evaluate_qtdb(model, device)
    if qtdb_res is None:
        print("  no QTDB sequences processed")
    else:
        print(f"  done in {time.time()-t0:.1f}s ({qtdb_res['n_seqs']} sequences)")

    # Frame F1 raw
    print("\n== Stage 2 v3.1 multi-DB validation ==\n")
    print("Frame-level F1 (raw model output, no post-proc):")
    print(f"{'DB':10s}|   P   |  QRS  |   T")
    print(_frame_f1_row(ludb_res["raw_pred"], ludb_res["raw_true"], "LUDB val"))
    print(_frame_f1_row(isp_res["raw_pred"], isp_res["raw_true"], "ISP test"))
    if qtdb_res:
        print(_frame_f1_row(qtdb_res["raw_pred"], qtdb_res["raw_true"], "QTDB ext"))

    # Frame F1 post-proc
    print("\nFrame-level F1 (with post-processing):")
    print(f"{'DB':10s}|   P   |  QRS  |   T")
    print(_frame_f1_row(ludb_res["pp_pred"], ludb_res["pp_true"], "LUDB val"))
    print(_frame_f1_row(isp_res["pp_pred"], isp_res["pp_true"], "ISP test"))
    if qtdb_res:
        print(_frame_f1_row(qtdb_res["pp_pred"], qtdb_res["pp_true"], "QTDB ext"))

    # Boundary F1 (raw)
    print(f"\nBoundary-level F1 (raw, {LITERATURE_TOL_MS}ms tolerance):")
    print(f"{'DB':10s}| P_on  | QRS_on| T_on  | P_off | QRSoff| T_off")
    boundary_metrics_raw = {}
    for label, res in [("LUDB val", ludb_res), ("ISP test", isp_res),
                       ("QTDB ext", qtdb_res)]:
        if res is None:
            continue
        row, metrics = _boundary_f1_row(
            res["boundary_pred_raw"], res["boundary_true"], res["fs"], label,
        )
        print(row)
        boundary_metrics_raw[label] = metrics

    # Boundary F1 (post-proc, literature comparable)
    print(f"\nBoundary-level F1 (post-proc, {LITERATURE_TOL_MS}ms tolerance, literature-comparable):")
    print(f"{'DB':10s}| P_on  | QRS_on| T_on  | P_off | QRSoff| T_off")
    boundary_metrics_pp = {}
    for label, res in [("LUDB val", ludb_res), ("ISP test", isp_res),
                       ("QTDB ext", qtdb_res)]:
        if res is None:
            continue
        row, metrics = _boundary_f1_row(
            res["boundary_pred_pp"], res["boundary_true"], res["fs"], label,
        )
        print(row)
        boundary_metrics_pp[label] = metrics

    # Comparison
    f1_ludb_pp = ee.frame_f1(ludb_res["pp_pred"], ludb_res["pp_true"])
    f1_isp_pp = ee.frame_f1(isp_res["pp_pred"], isp_res["pp_true"])
    print("\n== Comparison ==")
    print("                | v1.0 frame F1     | v3.1 frame F1 (pp)")
    print(f"LUDB val P/QRS/T| 0.604/0.806/0.695 | "
          f"{f1_ludb_pp[ee.SUPER_P]['f1']:.3f}/{f1_ludb_pp[ee.SUPER_QRS]['f1']:.3f}/"
          f"{f1_ludb_pp[ee.SUPER_T]['f1']:.3f}")
    print(f"ISP test P/QRS/T| (n/a)             | "
          f"{f1_isp_pp[ee.SUPER_P]['f1']:.3f}/{f1_isp_pp[ee.SUPER_QRS]['f1']:.3f}/"
          f"{f1_isp_pp[ee.SUPER_T]['f1']:.3f}")
    print("\nLiterature SOTA (ISP): P/QRS/T boundary F1 ~ 0.97/0.99/0.96 (150ms tol)")

    # Save JSON report
    OUT_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = OUT_DIR / f"validation_stage2_v31_{ts}.json"

    def _json_safe(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out[k] = _json_safe(v)
            elif isinstance(v, (np.integer,)):
                out[k] = int(v)
            elif isinstance(v, (np.floating,)):
                out[k] = float(v)
            else:
                out[k] = v
        return out

    payload = {
        "checkpoint": str(CKPT_PATH),
        "tolerance_ms": LITERATURE_TOL_MS,
        "ludb_val": {
            "frame_f1_raw": _json_safe({ee.SUPER_NAMES[sc]: m
                                        for sc, m in ee.frame_f1(ludb_res["raw_pred"], ludb_res["raw_true"]).items()}),
            "frame_f1_pp": _json_safe({ee.SUPER_NAMES[sc]: m
                                       for sc, m in f1_ludb_pp.items()}),
            "boundary_f1_raw": _json_safe(boundary_metrics_raw.get("LUDB val", {})),
            "boundary_f1_pp": _json_safe(boundary_metrics_pp.get("LUDB val", {})),
            "n_seqs": ludb_res["n_seqs"],
        },
        "isp_test": {
            "frame_f1_raw": _json_safe({ee.SUPER_NAMES[sc]: m
                                        for sc, m in ee.frame_f1(isp_res["raw_pred"], isp_res["raw_true"]).items()}),
            "frame_f1_pp": _json_safe({ee.SUPER_NAMES[sc]: m
                                       for sc, m in f1_isp_pp.items()}),
            "boundary_f1_raw": _json_safe(boundary_metrics_raw.get("ISP test", {})),
            "boundary_f1_pp": _json_safe(boundary_metrics_pp.get("ISP test", {})),
            "n_seqs": isp_res["n_seqs"],
        },
    }
    if qtdb_res is not None:
        payload["qtdb_ext"] = {
            "frame_f1_raw": _json_safe({ee.SUPER_NAMES[sc]: m
                                        for sc, m in ee.frame_f1(qtdb_res["raw_pred"], qtdb_res["raw_true"]).items()}),
            "frame_f1_pp": _json_safe({ee.SUPER_NAMES[sc]: m
                                       for sc, m in ee.frame_f1(qtdb_res["pp_pred"], qtdb_res["pp_true"]).items()}),
            "boundary_f1_raw": _json_safe(boundary_metrics_raw.get("QTDB ext", {})),
            "boundary_f1_pp": _json_safe(boundary_metrics_pp.get("QTDB ext", {})),
            "n_seqs": qtdb_res["n_seqs"],
        }

    out_file.write_text(json.dumps(payload, indent=2))
    print(f"\nReport: {out_file}")


if __name__ == "__main__":
    main()
