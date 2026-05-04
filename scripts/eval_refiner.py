"""Measure Stage 3 boundary refiner impact.

Compares post-processed frame-transition boundaries against the optional
signal-aware refiner on LUDB / ISP / QTDB T-annotated subset.
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ecgcode import isp, ludb, qtdb
from ecgcode.stage2.dataset import LUDBFrameDataset
from ecgcode.stage2.evaluate import (
    MARTINEZ_TOLERANCE_MS,
    average_boundary_f1,
    boundary_metrics_by_key,
)
from ecgcode.stage2.infer import (
    BOUNDARY_SHIFT_C,
    BOUNDARY_SHIFT_F,
    extract_boundaries,
    load_model_bundle,
    post_process_frames,
    predict_frames,
)
from ecgcode.stage2.refiner import refine_boundaries


CKPT_DIR = Path("data/checkpoints")
OUT_DIR = Path("out")
WINDOW_SAMPLES_250 = 2500
FRAME_MS = 20


MODEL_SPECS = {
    "C": {"ckpt": CKPT_DIR / "stage2_v4_C.pt", "shift": BOUNDARY_SHIFT_C},
    "F": {"ckpt": CKPT_DIR / "stage2_v4_ludb_only.pt", "shift": BOUNDARY_SHIFT_F},
}


def _normalize(sig):
    return ((sig - float(sig.mean())) / (float(sig.std()) + 1e-6)).astype(np.float32)


def _decimate(sig, native_fs, target_fs=250):
    if native_fs == target_fs:
        return sig.astype(np.float64)
    import scipy.signal as scipy_signal
    return scipy_signal.decimate(sig, native_fs // target_fs, zero_phase=True)


def _add_boundaries(acc, local_boundaries, cum):
    for key, values in local_boundaries.items():
        acc[key].extend(int(v) + cum for v in values)


def _predict_boundary_pair(model, sig_250, lead_idx, device, shift, refine_kwargs):
    raw = predict_frames(model, sig_250, lead_idx, device=device)
    frames = post_process_frames(raw, frame_ms=FRAME_MS)
    base = extract_boundaries(
        frames, fs=250, frame_ms=FRAME_MS, boundary_shift_ms=shift,
    )
    refined = refine_boundaries(sig_250, base, fs=250, **refine_kwargs)
    return base, refined


def evaluate_ludb(model, device, shift, refine_kwargs):
    ds = LUDBFrameDataset(ludb.load_split()["val"])
    base_pred, refined_pred, true = defaultdict(list), defaultdict(list), defaultdict(list)
    cum = 0
    with torch.no_grad():
        for idx in range(len(ds)):
            rid, lead = ds.items[idx]
            sig_250, lead_idx, _ = ds.cache[(rid, lead)]
            sig_250 = sig_250[:WINDOW_SAMPLES_250]
            if len(sig_250) < WINDOW_SAMPLES_250:
                continue
            base, refined = _predict_boundary_pair(
                model, sig_250, lead_idx, device, shift, refine_kwargs
            )
            _add_boundaries(base_pred, base, cum)
            _add_boundaries(refined_pred, refined, cum)
            gt_ann = ludb.load_annotations(rid, lead)
            for key, values in gt_ann.items():
                if key.endswith("_on") or key.endswith("_off"):
                    for sample in values:
                        s250 = int(sample // 2)
                        if 0 <= s250 < WINDOW_SAMPLES_250:
                            true[key].append(s250 + cum)
            cum += WINDOW_SAMPLES_250
    return base_pred, refined_pred, true, len(ds)


def evaluate_isp(model, device, shift, refine_kwargs):
    base_pred, refined_pred, true = defaultdict(list), defaultdict(list), defaultdict(list)
    cum = 0
    n_seq = 0
    with torch.no_grad():
        for rid in isp.load_split()["test"]:
            record = isp.load_record(rid, split="test")
            ann = isp.load_annotations_as_super(rid, split="test")
            for lead_idx, lead in enumerate(isp.LEADS_12):
                sig_1000 = record[lead]
                sig_250 = _decimate(sig_1000, 1000)
                sig_n = _normalize(sig_250)
                if len(sig_n) < WINDOW_SAMPLES_250:
                    sig_n = np.concatenate([
                        sig_n,
                        np.zeros(WINDOW_SAMPLES_250 - len(sig_n), dtype=sig_n.dtype),
                    ])
                sig_n = sig_n[:WINDOW_SAMPLES_250]
                base, refined = _predict_boundary_pair(
                    model, sig_n, lead_idx, device, shift, refine_kwargs
                )
                _add_boundaries(base_pred, base, cum)
                _add_boundaries(refined_pred, refined, cum)
                for key, values in ann.items():
                    if key.endswith("_on") or key.endswith("_off"):
                        for sample in values:
                            s250 = int(sample // 4)
                            if 0 <= s250 < WINDOW_SAMPLES_250:
                                true[key].append(s250 + cum)
                cum += WINDOW_SAMPLES_250
                n_seq += 1
    return base_pred, refined_pred, true, n_seq


def _qtdb_t_subset_ids():
    out = []
    for rid in qtdb.records_with_q1c():
        ann = qtdb.load_q1c(rid)
        win = qtdb.annotated_window(ann, window_samples=WINDOW_SAMPLES_250, fs=250)
        if win is None:
            continue
        start, end = win
        n_q = sum(1 for s in ann["qrs_on"] if start <= s < end)
        n_t = sum(1 for s in ann["t_on"] if start <= s < end)
        if n_q > 0 and n_t / n_q >= 0.8:
            out.append(rid)
    return out


def evaluate_qtdb(model, device, shift, refine_kwargs):
    base_pred, refined_pred, true = defaultdict(list), defaultdict(list), defaultdict(list)
    cum = 0
    n_seq = 0
    for rid in _qtdb_t_subset_ids():
        record = qtdb.load_record(rid)
        ann = qtdb.load_q1c(rid)
        win = qtdb.annotated_window(ann, window_samples=WINDOW_SAMPLES_250, fs=250)
        if win is None:
            continue
        start, end = win
        if end > 225000:
            end = 225000
            start = end - WINDOW_SAMPLES_250
        first_lead = next(iter(record))
        sig = record[first_lead][start:end]
        if len(sig) < WINDOW_SAMPLES_250:
            continue
        sig_n = _normalize(sig)
        base, refined = _predict_boundary_pair(
            model, sig_n, lead_idx=1, device=device, shift=shift, refine_kwargs=refine_kwargs
        )
        _add_boundaries(base_pred, base, cum)
        _add_boundaries(refined_pred, refined, cum)
        for key in MARTINEZ_TOLERANCE_MS:
            true[key].extend(int(s - start) + cum for s in ann[key] if start <= s < end)
        cum += WINDOW_SAMPLES_250
        n_seq += 1
    return base_pred, refined_pred, true, n_seq


def summarize(base_pred, refined_pred, true):
    baseline = boundary_metrics_by_key(base_pred, true, fs=250)
    refined = boundary_metrics_by_key(refined_pred, true, fs=250)
    return {
        "baseline": baseline,
        "refined": refined,
        "baseline_avg_f1": average_boundary_f1(baseline),
        "refined_avg_f1": average_boundary_f1(refined),
        "delta_avg_f1": average_boundary_f1(refined) - average_boundary_f1(baseline),
    }


def print_summary(model_name, domain, result):
    print(f"\n=== {model_name} {domain} ===", flush=True)
    print(
        f"avg F1 baseline={result['baseline_avg_f1']:.4f} "
        f"refined={result['refined_avg_f1']:.4f} "
        f"delta={result['delta_avg_f1']:+.4f}",
        flush=True,
    )
    print(f"{'boundary':8s} {'base':>8s} {'refined':>8s} {'delta':>8s} {'base_ms':>8s} {'ref_ms':>8s}", flush=True)
    for key in MARTINEZ_TOLERANCE_MS:
        base = result["baseline"][key]
        ref = result["refined"][key]
        print(
            f"{key:8s} {base['f1']:8.4f} {ref['f1']:8.4f} "
            f"{ref['f1'] - base['f1']:+8.4f} "
            f"{base['mean_signed_ms']:+8.2f} {ref['mean_signed_ms']:+8.2f}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["C", "F"], choices=sorted(MODEL_SPECS))
    parser.add_argument("--domains", nargs="+", default=["ludb", "isp", "qtdb"],
                        choices=["ludb", "isp", "qtdb"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--search-ms", type=int, default=80)
    parser.add_argument("--refine-p-t", action="store_true")
    args = parser.parse_args()

    domain_fns = {
        "ludb": ("ECGCODE_LUDB_ZIP", evaluate_ludb),
        "isp": ("ECGCODE_ISP_ZIP", evaluate_isp),
        "qtdb": ("ECGCODE_QTDB_ZIP", evaluate_qtdb),
    }
    refine_kwargs = {
        "search_ms": args.search_ms,
        "refine_qrs": True,
        "refine_p_t": bool(args.refine_p_t),
    }

    OUT_DIR.mkdir(exist_ok=True)
    full = {"config": {"refine_kwargs": refine_kwargs, "device": args.device}}
    for model_name in args.models:
        spec = MODEL_SPECS[model_name]
        if not spec["ckpt"].exists():
            print(f"skip {model_name}: missing {spec['ckpt']}", flush=True)
            continue
        bundle = load_model_bundle(spec["ckpt"], device=args.device)
        model = bundle["model"]
        full[model_name] = {}
        for domain in args.domains:
            env_name, fn = domain_fns[domain]
            if not os.environ.get(env_name):
                print(f"skip {domain}: {env_name} is not set", flush=True)
                continue
            t0 = time.time()
            base_pred, refined_pred, true, n_seq = fn(
                model, args.device, spec["shift"], refine_kwargs
            )
            result = summarize(base_pred, refined_pred, true)
            result["n_seq"] = n_seq
            result["seconds"] = time.time() - t0
            full[model_name][domain] = result
            print_summary(model_name, domain, result)
            print(f"n_seq={n_seq}, seconds={result['seconds']:.1f}", flush=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"eval_refiner_{ts}.json"

    def _safe(value):
        if isinstance(value, dict):
            return {k: _safe(v) for k, v in value.items()}
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        return value

    out_path.write_text(json.dumps(_safe(full), indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
