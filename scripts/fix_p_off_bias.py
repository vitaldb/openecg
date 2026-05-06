"""Test fixes for the p_off systematic +22ms bias.

Strategies:
  A. Baseline = current post_process (no trim)
  B. Signal-aware trim: walk back from end of each P segment, trim frames
     whose abs(signal - local_baseline) is within k*stdev of pre-P baseline noise.
     (k=1.0 conservative, k=2.0 aggressive)
  C. Fixed -22ms shift on p_off boundary samples (uniform offset)

Eval with Martinez tolerance (P 50ms): F1, Se, P+, mean ± SD timing error,
on LUDB val + ISP test.
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from openecg import eval as ee, isp, ludb
from openecg.stage2.dataset import LUDBFrameDataset
from openecg.stage2.infer import post_process_frames, predict_frames
from openecg.stage2.model import FrameClassifier
from openecg.stage2.train import load_checkpoint

CKPT_DIR = Path("data/checkpoints")
OUT_DIR = Path("out")
WINDOW_SAMPLES_250 = 2500
WINDOW_FRAMES = 500
FRAME_MS = 20
P_OFF_TOL_MS = 50
P_ON_TOL_MS = 50
SAMPLES_PER_FRAME = 5  # 20ms @ 250Hz


def _extract_boundaries(frames, fs=250, frame_ms=FRAME_MS, p_off_shift_samples=0):
    out = defaultdict(list)
    super_to_name = {ee.SUPER_P: "p", ee.SUPER_QRS: "qrs", ee.SUPER_T: "t"}
    spf = int(round(frame_ms * fs / 1000.0))
    prev = ee.SUPER_OTHER
    for f_idx, cur in enumerate(frames):
        cur = int(cur)
        if cur != prev:
            sample = f_idx * spf
            if prev in super_to_name:
                name = super_to_name[prev]
                offset_sample = sample - 1
                if name == "p":
                    offset_sample += p_off_shift_samples
                out[f"{name}_off"].append(offset_sample)
            if cur in super_to_name:
                out[f"{super_to_name[cur]}_on"].append(sample)
        prev = cur
    if prev in super_to_name:
        sample = len(frames) * spf
        name = super_to_name[prev]
        offset_sample = sample - 1
        if name == "p":
            offset_sample += p_off_shift_samples
        out[f"{name}_off"].append(offset_sample)
    return dict(out)


def signal_aware_trim_p(frames, signal, k_sd=1.5, baseline_window_frames=10):
    """Trim trailing frames of each P segment if their signal amplitude is
    within k_sd standard deviations of the local pre-P baseline.

    baseline = OTHER samples in the `baseline_window_frames` frames preceding P.
    For each P segment [i, j), walk back: while signal in frame (new_j - 1) is
    "near baseline", set new_j -= 1. Replace [new_j, j) with OTHER.
    """
    arr = np.asarray(frames, dtype=np.uint8).copy()
    sig = np.asarray(signal, dtype=np.float64)
    P = ee.SUPER_P
    OTHER = ee.SUPER_OTHER
    n = len(arr)
    spf = SAMPLES_PER_FRAME
    i = 0
    while i < n:
        if arr[i] != P:
            i += 1
            continue
        j = i
        while j < n and arr[j] == P:
            j += 1
        # Local baseline: OTHER samples in [i - baseline_window_frames, i)
        b_start = max(0, i - baseline_window_frames)
        baseline_samples = []
        for f in range(b_start, i):
            if arr[f] == OTHER:
                s0 = f * spf
                s1 = (f + 1) * spf
                if s1 <= len(sig):
                    baseline_samples.extend(sig[s0:s1])
        if len(baseline_samples) < spf:
            i = j
            continue
        baseline_mean = float(np.mean(baseline_samples))
        baseline_sd = float(np.std(baseline_samples)) + 1e-6
        threshold = baseline_sd * k_sd
        # Walk back from j
        new_j = j
        while new_j > i + 1:
            f = new_j - 1
            s0 = f * spf
            s1 = min((f + 1) * spf, len(sig))
            if s1 <= s0:
                break
            frame_amp = float(np.max(np.abs(sig[s0:s1] - baseline_mean)))
            if frame_amp < threshold:
                new_j -= 1
            else:
                break
        if new_j < j:
            arr[new_j:j] = OTHER
        i = j
    return arr


def signed_metrics(pred, true, tol_ms, fs=250):
    tol_samples = tol_ms * fs / 1000.0
    pred_arr = np.sort(np.array(pred, dtype=int))
    true_arr = np.sort(np.array(true, dtype=int))
    matched = set()
    signed_errs = []
    for t in true_arr:
        best = -1; best_abs = float("inf")
        for jj, p in enumerate(pred_arr):
            if jj in matched: continue
            d = abs(int(p) - int(t))
            if d < best_abs:
                best_abs = d; best = jj
        if best >= 0 and best_abs <= tol_samples:
            matched.add(best)
            signed_errs.append(int(pred_arr[best]) - int(t))
    n_hits = len(signed_errs)
    sens = n_hits / len(true_arr) if len(true_arr) > 0 else 0.0
    ppv = n_hits / len(pred_arr) if len(pred_arr) > 0 else 0.0
    f1 = 2 * sens * ppv / (sens + ppv) if (sens + ppv) > 0 else 0.0
    if signed_errs:
        e = np.array(signed_errs) * 1000.0 / fs
        return {"f1": f1, "sens": sens, "ppv": ppv,
                "mean_signed_ms": float(np.mean(e)), "sd_ms": float(np.std(e)),
                "med_abs_ms": float(np.median(np.abs(e))),
                "n_true": int(len(true_arr)), "n_pred": int(len(pred_arr))}
    return {"f1": f1, "sens": sens, "ppv": ppv,
            "mean_signed_ms": 0.0, "sd_ms": 0.0, "med_abs_ms": 0.0,
            "n_true": int(len(true_arr)), "n_pred": int(len(pred_arr))}


def cache_predictions_ludb(model, device):
    val_ds = LUDBFrameDataset(ludb.load_split()["val"])
    items = []
    bt_p_on, bt_p_off = [], []
    cum = 0
    with torch.no_grad():
        for idx in range(len(val_ds)):
            rid, lead = val_ds.items[idx]
            sig_250, lead_idx, _ = val_ds.cache[(rid, lead)]
            sig_250 = sig_250[:WINDOW_SAMPLES_250]
            if len(sig_250) < WINDOW_SAMPLES_250: continue
            pred = predict_frames(model, sig_250, lead_idx, device=device)
            pp = post_process_frames(pred, frame_ms=FRAME_MS)
            items.append({"sig": sig_250, "pp": pp, "cum": cum})
            try:
                gt_ann = ludb.load_annotations(rid, lead)
                for s in gt_ann.get("p_on", []):
                    s250 = int(s // 2)
                    if 0 <= s250 < WINDOW_SAMPLES_250:
                        bt_p_on.append(s250 + cum)
                for s in gt_ann.get("p_off", []):
                    s250 = int(s // 2)
                    if 0 <= s250 < WINDOW_SAMPLES_250:
                        bt_p_off.append(s250 + cum)
            except Exception:
                pass
            cum += WINDOW_SAMPLES_250
    return items, {"p_on": bt_p_on, "p_off": bt_p_off}


def cache_predictions_isp(model, device):
    rec_ids = isp.load_split()["test"]
    items = []
    bt_p_on, bt_p_off = [], []
    cum = 0
    with torch.no_grad():
        for rid in rec_ids:
            try:
                record = isp.load_record(rid, split="test")
                ann = isp.load_annotations_as_super(rid, split="test")
            except Exception:
                continue
            for lead_idx, lead in enumerate(isp.LEADS_12):
                sig_1000 = record[lead]
                from openecg.stage2.multi_dataset import _decimate_to_250, _normalize
                sig_250 = _decimate_to_250(sig_1000, 1000)
                sig_n = _normalize(sig_250)
                if len(sig_n) < WINDOW_SAMPLES_250:
                    pad = np.zeros(WINDOW_SAMPLES_250 - len(sig_n), dtype=sig_n.dtype)
                    sig_n = np.concatenate([sig_n, pad])
                sig_n = sig_n[:WINDOW_SAMPLES_250]
                pred = predict_frames(model, sig_n, lead_idx, device=device)
                pp = post_process_frames(pred, frame_ms=FRAME_MS)
                items.append({"sig": sig_n, "pp": pp, "cum": cum})
                for s in ann.get("p_on", []):
                    s250 = int(s // 4)
                    if 0 <= s250 < WINDOW_SAMPLES_250:
                        bt_p_on.append(s250 + cum)
                for s in ann.get("p_off", []):
                    s250 = int(s // 4)
                    if 0 <= s250 < WINDOW_SAMPLES_250:
                        bt_p_off.append(s250 + cum)
                cum += WINDOW_SAMPLES_250
    return items, {"p_on": bt_p_on, "p_off": bt_p_off}


def eval_strategy(items, true_b, label, trim=None, shift_ms=0):
    """trim: callable(frames, signal) -> frames, or None
    shift_ms: applied to p_off only via _extract_boundaries
    """
    bp_p_on, bp_p_off = [], []
    shift_samples = int(round(shift_ms * 250.0 / 1000.0))
    for item in items:
        frames = item["pp"]
        if trim is not None:
            frames = trim(frames, item["sig"])
        b = _extract_boundaries(frames, fs=250, p_off_shift_samples=shift_samples)
        cum = item["cum"]
        for s in b.get("p_on", []):
            bp_p_on.append(int(s) + cum)
        for s in b.get("p_off", []):
            bp_p_off.append(int(s) + cum)
    m_on = signed_metrics(bp_p_on, true_b["p_on"], P_ON_TOL_MS)
    m_off = signed_metrics(bp_p_off, true_b["p_off"], P_OFF_TOL_MS)
    print(f"  [{label}]", flush=True)
    for k, m in [("p_on", m_on), ("p_off", m_off)]:
        print(f"    {k:6s}: F1={m['f1']:.3f} Se={m['sens']*100:5.1f}% P+={m['ppv']*100:5.1f}% "
              f"mean={m['mean_signed_ms']:+6.1f}ms SD={m['sd_ms']:5.1f}ms med_abs={m['med_abs_ms']:5.1f}ms "
              f"(n_true={m['n_true']}, n_pred={m['n_pred']})", flush=True)
    return {"p_on": m_on, "p_off": m_off}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    OUT_DIR.mkdir(exist_ok=True)

    candidates = [
        ("C", CKPT_DIR / "stage2_v4_C.pt", {"d_model": 128, "n_layers": 8}),
        ("F", CKPT_DIR / "stage2_v4_ludb_only.pt",
         {"d_model": 64, "n_layers": 4, "use_lead_emb": False}),
    ]

    full = {}
    for name, ckpt, mk in candidates:
        print(f"\n{'='*78}\n>>> {name} <<<\n{'='*78}", flush=True)
        model = FrameClassifier(**mk)
        load_checkpoint(ckpt, model)
        model = model.to(device).train(False)

        full[name] = {}
        for dom_name, cache_fn in [("LUDB val", cache_predictions_ludb),
                                    ("ISP test", cache_predictions_isp)]:
            print(f"\n--- {name} on {dom_name} ---", flush=True)
            t0 = time.time()
            items, true_b = cache_fn(model, device)
            print(f"  Cached {len(items)} items in {time.time()-t0:.1f}s", flush=True)

            results = {}
            results["A_baseline"] = eval_strategy(items, true_b, "A: baseline (no trim)")
            results["B_trim_k1.0"] = eval_strategy(items, true_b, "B: signal trim k=1.0",
                                                   trim=lambda f, s: signal_aware_trim_p(f, s, k_sd=1.0))
            results["B_trim_k1.5"] = eval_strategy(items, true_b, "B: signal trim k=1.5",
                                                   trim=lambda f, s: signal_aware_trim_p(f, s, k_sd=1.5))
            results["B_trim_k2.0"] = eval_strategy(items, true_b, "B: signal trim k=2.0",
                                                   trim=lambda f, s: signal_aware_trim_p(f, s, k_sd=2.0))
            results["C_shift_-22ms"] = eval_strategy(items, true_b, "C: fixed -22ms shift",
                                                      shift_ms=-22)
            results["C_shift_-15ms"] = eval_strategy(items, true_b, "C: fixed -15ms shift",
                                                      shift_ms=-15)
            full[name][dom_name] = results

    # Comparison summary
    print("\n" + "="*90, flush=True)
    print(f"{'p_off F1 / mean_signed_ms summary':^90}", flush=True)
    print("="*90, flush=True)
    print(f"{'model/domain':24s} {'baseline':>15s} {'trim k=1.0':>15s} {'trim k=1.5':>15s} "
          f"{'trim k=2.0':>15s} {'shift -22':>12s} {'shift -15':>12s}", flush=True)
    for name in [c[0] for c in candidates]:
        for dom in ("LUDB val", "ISP test"):
            r = full[name][dom]
            def fmt(key):
                m = r[key]["p_off"]
                return f"{m['f1']:.3f} ({m['mean_signed_ms']:+.0f})"
            print(f"  {name}/{dom:18s}  {fmt('A_baseline'):>14s}  {fmt('B_trim_k1.0'):>14s}  "
                  f"{fmt('B_trim_k1.5'):>14s}  {fmt('B_trim_k2.0'):>14s}  "
                  f"{fmt('C_shift_-22ms'):>11s}  {fmt('C_shift_-15ms'):>11s}", flush=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"fix_p_off_bias_{ts}.json"

    def _safe(o):
        if isinstance(o, dict): return {k: _safe(v) for k, v in o.items()}
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        return o
    out_path.write_text(json.dumps(_safe(full), indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
