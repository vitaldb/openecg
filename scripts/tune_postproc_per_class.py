"""Per-class post-proc tuning. Previous sweep found per-boundary optima differ
significantly: QRS prefers (min~20-40, merge~100), P/T prefer (min~60, merge~100-300).
This script tests several per-class param combos vs the single-default (60, 200).
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ecgcode import eval as ee, ludb
from ecgcode.stage2.dataset import LUDBFrameDataset
from ecgcode.stage2.infer import post_process_frames, predict_frames
from ecgcode.stage2.model import FrameClassifier
from ecgcode.stage2.train import load_checkpoint

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


def cache_raw(model, val_ds, device):
    model.train(False)
    raw = []
    true_b = defaultdict(list)
    cum = 0
    with torch.no_grad():
        for idx in range(len(val_ds)):
            rid, lead = val_ds.items[idx]
            sig_250, lead_idx, _ = val_ds.cache[(rid, lead)]
            sig_250 = sig_250[:WINDOW_SAMPLES_250]
            if len(sig_250) < WINDOW_SAMPLES_250: continue
            pred = predict_frames(model, sig_250, lead_idx, device=device)
            raw.append((pred, cum))
            try:
                gt_ann = ludb.load_annotations(rid, lead)
                for k, v in gt_ann.items():
                    if k.endswith("_on") or k.endswith("_off"):
                        for s in v:
                            s250 = int(s // 2)
                            if 0 <= s250 < WINDOW_SAMPLES_250:
                                true_b[k].append(s250 + cum)
            except Exception:
                pass
            cum += WINDOW_SAMPLES_250
    return raw, dict(true_b)


def eval_combo(raw, true_b, **kw):
    pred_b = defaultdict(list)
    for r, cum in raw:
        pp = post_process_frames(r, frame_ms=FRAME_MS, **kw)
        for k, v in _extract_boundaries(pp, fs=250).items():
            pred_b[k].extend(int(x) + cum for x in v)
    out = {}
    for k in ("p_on", "qrs_on", "t_on", "p_off", "qrs_off", "t_off"):
        m = ee.boundary_f1(pred_b.get(k, []), true_b.get(k, []),
                           tolerance_ms=TOL_MS, fs=250)
        out[k] = m["f1"]
    out["avg"] = float(np.mean([out[k] for k in ("p_on", "qrs_on", "t_on", "p_off", "qrs_off", "t_off")]))
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    OUT_DIR.mkdir(exist_ok=True)
    val_ds = LUDBFrameDataset(ludb.load_split()["val"])
    print(f"LUDB val: {len(val_ds)} sequences", flush=True)

    candidates = [
        ("C", CKPT_DIR / "stage2_v4_C.pt", {"d_model": 128, "n_layers": 8}),
        ("F", CKPT_DIR / "stage2_v4_ludb_only.pt", {"d_model": 64, "n_layers": 4, "use_lead_emb": False}),
    ]

    # Combos to test. Format: (label, dict for post_process_frames kwargs)
    combos = [
        ("default (60,200)",
         {"min_duration_ms": 60, "merge_gap_ms": 200}),
        ("PC: P(60,200)/QRS(20,100)/T(60,100)",
         {"per_class_min_ms": {1: 60, 2: 20, 3: 60},
          "per_class_merge_ms": {1: 200, 2: 100, 3: 100}}),
        ("PC: P(60,300)/QRS(20,100)/T(60,200)",
         {"per_class_min_ms": {1: 60, 2: 20, 3: 60},
          "per_class_merge_ms": {1: 300, 2: 100, 3: 200}}),
        ("PC: P(80,300)/QRS(40,100)/T(80,200)",
         {"per_class_min_ms": {1: 80, 2: 40, 3: 80},
          "per_class_merge_ms": {1: 300, 2: 100, 3: 200}}),
        ("PC: P(60,300)/QRS(40,100)/T(60,100)",
         {"per_class_min_ms": {1: 60, 2: 40, 3: 60},
          "per_class_merge_ms": {1: 300, 2: 100, 3: 100}}),
        ("PC: P(80,200)/QRS(40,100)/T(80,100)",
         {"per_class_min_ms": {1: 80, 2: 40, 3: 80},
          "per_class_merge_ms": {1: 200, 2: 100, 3: 100}}),
    ]

    full = {}
    for name, ckpt, mk in candidates:
        print(f"\n{'='*78}\n{name} from {ckpt}\n{'='*78}", flush=True)
        model = FrameClassifier(**mk)
        load_checkpoint(ckpt, model)
        model = model.to(device)
        t0 = time.time()
        raw, true_b = cache_raw(model, val_ds, device)
        print(f"Cached raw preds in {time.time()-t0:.1f}s", flush=True)

        results = {}
        for label, kw in combos:
            r = eval_combo(raw, true_b, **kw)
            results[label] = r
        # Print
        print(f"\n  {'combo':45s} {'p_on':>6s} {'qrs_on':>7s} {'t_on':>6s} {'p_off':>6s} {'qrs_off':>8s} {'t_off':>6s} {'AVG':>6s}", flush=True)
        baseline = results["default (60,200)"]["avg"]
        for label, r in results.items():
            avg_delta = r["avg"] - baseline
            mark = " *" if r["avg"] >= baseline else "  "
            print(f"  {label:45s} {r['p_on']:.3f} {r['qrs_on']:.3f}  {r['t_on']:.3f} {r['p_off']:.3f}   {r['qrs_off']:.3f} {r['t_off']:.3f} {r['avg']:.3f}{mark} (Δ{avg_delta:+.3f})", flush=True)
        full[name] = results

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"tune_postproc_per_class_{ts}.json"
    out_path.write_text(json.dumps(full, indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
