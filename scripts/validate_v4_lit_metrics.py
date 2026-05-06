"""Literature-style metric evaluation for v4 candidates: boundary F1, Se/PPV, mean+std/median timing error.

Trains C (combined big +lead_emb +CE) if its ckpt is missing, then evaluates
both C and F (LUDB only no_lead_emb) on LUDB val + ISP test + QTDB ext using
the boundary metrics framework from validate_stage2_v3.py.

Reported metrics (literature standard):
  - Boundary F1 (sens/PPV) with 150ms tolerance per boundary type
    (p_on, qrs_on, t_on, p_off, qrs_off, t_off)
  - Mean ± std and median timing error in ms
  - Frame F1 (raw + post-proc) for context

Reference papers cited in scripts/validate_stage2_v3.py and PLAN.md:
  - Martinez 2004 (QTDB wavelet baseline)
  - LUDB original / Kalyakulina 2020
  - SemiSegECG 2025 (semi-supervised SOTA on ISP)
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from openecg import eval as ee, isp, ludb, qtdb
from openecg.stage2.dataset import LUDBFrameDataset, compute_class_weights
from openecg.stage2.evaluate import boundary_metrics_by_key
from openecg.stage2.infer import post_process_frames, predict_frames
from openecg.stage2.model import FrameClassifier
from openecg.stage2.multi_dataset import CombinedFrameDataset
from openecg.stage2.train import TrainConfig, fit, load_checkpoint

OUT_DIR = Path("out")
CKPT_DIR = Path("data/checkpoints")
C_CKPT = CKPT_DIR / "stage2_v4_C.pt"
F_CKPT = CKPT_DIR / "stage2_v4_ludb_only.pt"
WINDOW_SAMPLES_250 = 2500
WINDOW_FRAMES = 500
FRAME_MS = 20
TOL_MS = 150
SEED = 42


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


def _extract_boundaries(super_frames, fs, frame_ms=FRAME_MS):
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


def boundary_summary(boundary_pred, boundary_true, fs, tol_ms=TOL_MS):
    """For each boundary type, return F1, sens, PPV, mean/std/median error ms."""
    keys = ("p_on", "qrs_on", "t_on", "p_off", "qrs_off", "t_off")
    signed = boundary_metrics_by_key(
        boundary_pred,
        boundary_true,
        tolerances_ms={k: tol_ms for k in keys},
        fs=fs,
    )
    return {
        k: {
            "f1": m["f1"], "sensitivity": m["sens"], "ppv": m["ppv"],
            "n_true": m["n_true"], "n_pred": m["n_pred"], "n_hits": m["n_hits"],
            "mean_error_ms": m["mean_abs_ms"],
            "median_error_ms": m["median_abs_ms"],
            "p95_error_ms": m["p95_abs_ms"],
        }
        for k, m in signed.items()
    }


@torch.no_grad()
def evaluate_ludb(model, device):
    val_ids = ludb.load_split()["val"]
    val_ds = LUDBFrameDataset(val_ids)
    raw_pred, raw_true, pp_pred = [], [], []
    bp_raw, bp_pp, bt = defaultdict(list), defaultdict(list), defaultdict(list)
    cum = 0
    for idx in range(len(val_ds)):
        rid, lead = val_ds.items[idx]
        sig_250, lead_idx, true_frames = val_ds.cache[(rid, lead)]
        true_frames = true_frames[:WINDOW_FRAMES].astype(np.uint8)
        sig_250 = sig_250[:WINDOW_SAMPLES_250]
        if len(sig_250) < WINDOW_SAMPLES_250: continue
        pred_raw = predict_frames(model, sig_250, lead_idx, device=device)
        pred_pp = post_process_frames(pred_raw, frame_ms=FRAME_MS)
        n = min(len(pred_raw), len(true_frames))
        raw_pred.append(pred_raw[:n]); raw_true.append(true_frames[:n]); pp_pred.append(pred_pp[:n])
        for k, v in _extract_boundaries(pred_raw, fs=250).items():
            bp_raw[k].extend(int(x)+cum for x in v)
        for k, v in _extract_boundaries(pred_pp, fs=250).items():
            bp_pp[k].extend(int(x)+cum for x in v)
        try:
            gt_ann = ludb.load_annotations(rid, lead)
            for k, v in gt_ann.items():
                if k.endswith("_on") or k.endswith("_off"):
                    for s in v:
                        s250 = int(s // 2)
                        if 0 <= s250 < WINDOW_SAMPLES_250:
                            bt[k].append(s250 + cum)
        except Exception:
            pass
        cum += WINDOW_SAMPLES_250
    return {
        "raw_pred": np.concatenate(raw_pred), "raw_true": np.concatenate(raw_true),
        "pp_pred": np.concatenate(pp_pred), "pp_true": np.concatenate(raw_true),
        "bp_raw": dict(bp_raw), "bp_pp": dict(bp_pp), "bt": dict(bt),
        "n_seqs": len(val_ds),
    }


@torch.no_grad()
def evaluate_isp(model, device):
    rec_ids = isp.load_split()["test"]
    raw_pred, raw_true, pp_pred = [], [], []
    bp_raw, bp_pp, bt = defaultdict(list), defaultdict(list), defaultdict(list)
    cum = 0
    for rid in rec_ids:
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
                pad = np.zeros(WINDOW_SAMPLES_250 - len(sig_n), dtype=sig_n.dtype)
                sig_n = np.concatenate([sig_n, pad])
            sig_n = sig_n[:WINDOW_SAMPLES_250]
            true_frames = ee.gt_to_super_frames(ann_super, n_samples=len(sig_1000),
                                                fs=1000, frame_ms=FRAME_MS).astype(np.uint8)
            if len(true_frames) < WINDOW_FRAMES:
                pad = np.full(WINDOW_FRAMES - len(true_frames), ee.SUPER_OTHER, dtype=true_frames.dtype)
                true_frames = np.concatenate([true_frames, pad])
            true_frames = true_frames[:WINDOW_FRAMES]
            pred_raw = predict_frames(model, sig_n, lead_idx, device=device)
            pred_pp = post_process_frames(pred_raw, frame_ms=FRAME_MS)
            n = min(len(pred_raw), len(true_frames))
            raw_pred.append(pred_raw[:n]); raw_true.append(true_frames[:n]); pp_pred.append(pred_pp[:n])
            for k, v in _extract_boundaries(pred_raw, fs=250).items():
                bp_raw[k].extend(int(x)+cum for x in v)
            for k, v in _extract_boundaries(pred_pp, fs=250).items():
                bp_pp[k].extend(int(x)+cum for x in v)
            for k, v in ann_super.items():
                if k.endswith("_on") or k.endswith("_off"):
                    for s in v:
                        s250 = int(s // 4)
                        if 0 <= s250 < WINDOW_SAMPLES_250:
                            bt[k].append(s250 + cum)
            cum += WINDOW_SAMPLES_250
    return {
        "raw_pred": np.concatenate(raw_pred), "raw_true": np.concatenate(raw_true),
        "pp_pred": np.concatenate(pp_pred), "pp_true": np.concatenate(raw_true),
        "bp_raw": dict(bp_raw), "bp_pp": dict(bp_pp), "bt": dict(bt),
        "n_seqs": len(raw_pred),
    }


@torch.no_grad()
def evaluate_qtdb(model, device):
    rec_ids = qtdb.records_with_q1c()
    raw_pred, raw_true, pp_pred = [], [], []
    bp_raw, bp_pp, bt = defaultdict(list), defaultdict(list), defaultdict(list)
    cum = 0
    for rid in rec_ids:
        try:
            record = qtdb.load_record(rid)
            ann = qtdb.load_q1c(rid)
        except Exception:
            continue
        win = qtdb.annotated_window(ann, window_samples=WINDOW_SAMPLES_250, fs=250)
        if win is None: continue
        start, end = win
        if end > 225000:
            end = 225000; start = end - WINDOW_SAMPLES_250
        first_lead = list(record.keys())[0]
        sig = record[first_lead][start:end]
        if len(sig) < WINDOW_SAMPLES_250: continue
        sig_n = _normalize(sig)
        win_ann = {k: [s - start for s in v if start <= s < end] for k, v in ann.items()}
        sample_labels = np.full(WINDOW_SAMPLES_250, ee.SUPER_OTHER, dtype=np.uint8)
        for on, off in zip(win_ann["p_on"], win_ann["p_off"]):
            sample_labels[max(0, on):min(WINDOW_SAMPLES_250, off + 1)] = ee.SUPER_P
        for on, off in zip(win_ann["qrs_on"], win_ann["qrs_off"]):
            sample_labels[max(0, on):min(WINDOW_SAMPLES_250, off + 1)] = ee.SUPER_QRS
        for on, off in zip(win_ann["t_on"], win_ann["t_off"]):
            sample_labels[max(0, on):min(WINDOW_SAMPLES_250, off + 1)] = ee.SUPER_T
        spf = WINDOW_SAMPLES_250 // WINDOW_FRAMES
        true_frames = np.zeros(WINDOW_FRAMES, dtype=np.uint8)
        for f in range(WINDOW_FRAMES):
            seg = sample_labels[f*spf:(f+1)*spf]
            vals, counts = np.unique(seg, return_counts=True)
            true_frames[f] = vals[np.argmax(counts)]
        pred_raw = predict_frames(model, sig_n, lead_id=1, device=device)
        pred_pp = post_process_frames(pred_raw, frame_ms=FRAME_MS)
        n = min(len(pred_raw), len(true_frames))
        raw_pred.append(pred_raw[:n]); raw_true.append(true_frames[:n]); pp_pred.append(pred_pp[:n])
        for k, v in _extract_boundaries(pred_raw, fs=250).items():
            bp_raw[k].extend(int(x)+cum for x in v)
        for k, v in _extract_boundaries(pred_pp, fs=250).items():
            bp_pp[k].extend(int(x)+cum for x in v)
        for k in ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off"):
            bt[k].extend(int(s)+cum for s in win_ann[k])
        cum += WINDOW_SAMPLES_250
    if not raw_pred: return None
    return {
        "raw_pred": np.concatenate(raw_pred), "raw_true": np.concatenate(raw_true),
        "pp_pred": np.concatenate(pp_pred), "pp_true": np.concatenate(raw_true),
        "bp_raw": dict(bp_raw), "bp_pp": dict(bp_pp), "bt": dict(bt),
        "n_seqs": len(raw_pred),
    }


def train_C(device):
    print(f"\n=== Training C (combined +big +lead_emb +CE) ===", flush=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    train_ds = CombinedFrameDataset(["ludb_train", "qtdb", "isp_train"])
    val_ds = LUDBFrameDataset(ludb.load_split()["val"])
    weights = torch.tensor(compute_class_weights(train_ds.label_counts()), dtype=torch.float32)
    cfg = TrainConfig(epochs=50, batch_size=64, lr=1e-3, early_stop_patience=10)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    model = FrameClassifier(d_model=128, n_layers=8)
    print(f"  params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    t0 = time.time()
    best = fit(model, train_loader, val_loader, weights, cfg, device=device,
               ckpt_path=C_CKPT, use_focal=False)
    print(f"  trained in {time.time()-t0:.1f}s, best={best}", flush=True)
    return C_CKPT


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    OUT_DIR.mkdir(exist_ok=True); CKPT_DIR.mkdir(parents=True, exist_ok=True)

    if not C_CKPT.exists():
        train_C(device)
    else:
        print(f"C ckpt exists at {C_CKPT}", flush=True)

    candidates = [
        ("C_combined_big_le", C_CKPT, {"d_model": 128, "n_layers": 8}),
        ("F_ludb_only_no_le", F_CKPT, {"d_model": 64,  "n_layers": 4, "use_lead_emb": False}),
    ]

    full_results = {}
    for name, ckpt, mk in candidates:
        print(f"\n{'='*70}\n=== {name} from {ckpt} ===\n{'='*70}", flush=True)
        if not ckpt.exists():
            print(f"  MISSING CKPT", flush=True)
            continue
        model = FrameClassifier(**mk)
        load_checkpoint(ckpt, model)
        model = model.to(device)

        domain_results = {}
        for dname, fn in [("LUDB val", evaluate_ludb), ("ISP test", evaluate_isp), ("QTDB ext", evaluate_qtdb)]:
            t0 = time.time()
            res = fn(model, device)
            if res is None: continue
            print(f"\n  -- {dname} ({res['n_seqs']} seqs, {time.time()-t0:.1f}s) --", flush=True)
            f1_raw = ee.frame_f1(res["raw_pred"], res["raw_true"])
            f1_pp = ee.frame_f1(res["pp_pred"], res["pp_true"])
            print(f"  Frame F1 raw  P/QRS/T = {f1_raw[ee.SUPER_P]['f1']:.3f} / "
                  f"{f1_raw[ee.SUPER_QRS]['f1']:.3f} / {f1_raw[ee.SUPER_T]['f1']:.3f}", flush=True)
            print(f"  Frame F1 pp   P/QRS/T = {f1_pp[ee.SUPER_P]['f1']:.3f} / "
                  f"{f1_pp[ee.SUPER_QRS]['f1']:.3f} / {f1_pp[ee.SUPER_T]['f1']:.3f}", flush=True)
            bs_raw = boundary_summary(res["bp_raw"], res["bt"], fs=250)
            bs_pp = boundary_summary(res["bp_pp"], res["bt"], fs=250)
            print(f"  Boundary metrics (post-proc, {TOL_MS}ms tol) — F1 / Se / PPV / med_err_ms:", flush=True)
            for k in ("p_on", "qrs_on", "t_on", "p_off", "qrs_off", "t_off"):
                m = bs_pp[k]
                print(f"    {k:8s}: F1={m['f1']:.3f}  Se={m['sensitivity']:.3f}  "
                      f"PPV={m['ppv']:.3f}  med_err={m['median_error_ms']:5.1f}ms  "
                      f"mean={m['mean_error_ms']:5.1f}ms  p95={m['p95_error_ms']:5.1f}ms  "
                      f"(n_true={m['n_true']}, n_pred={m['n_pred']})", flush=True)
            domain_results[dname] = {
                "frame_f1_raw": {ee.SUPER_NAMES[s]: f1_raw[s] for s in (1, 2, 3)},
                "frame_f1_pp": {ee.SUPER_NAMES[s]: f1_pp[s] for s in (1, 2, 3)},
                "boundary_raw_pp_tol150": bs_pp,
                "boundary_raw_tol150": bs_raw,
                "n_seqs": res["n_seqs"],
            }
        full_results[name] = domain_results

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"validate_v4_lit_metrics_{ts}.json"

    def _safe(o):
        if isinstance(o, dict): return {k: _safe(v) for k, v in o.items()}
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        return o
    out_path.write_text(json.dumps(_safe(full_results), indent=2))
    print(f"\nSaved {out_path}", flush=True)

    print("\n" + "="*90, flush=True)
    print(f"{'LITERATURE COMPARISON — Boundary F1 @150ms tol (post-processed)':^90}", flush=True)
    print("="*90, flush=True)
    print(f"{'metric':12s}  {'C LUDB':>8s} {'F LUDB':>8s} {'C ISP':>8s} {'F ISP':>8s} {'C QTDB':>8s} {'F QTDB':>8s}  literature", flush=True)
    LITERATURE = {
        "p_on":   "LUDB ~0.93–0.96 / SemiSegECG ISP ~0.97",
        "qrs_on": "LUDB ~0.98–0.99 / Martinez QTDB ~0.99 / SemiSegECG ISP ~0.99",
        "t_on":   "LUDB ~0.92–0.95 / SemiSegECG ISP ~0.95",
        "p_off":  "LUDB ~0.93–0.96",
        "qrs_off":"LUDB ~0.98–0.99 / Martinez QTDB ~0.99",
        "t_off":  "LUDB ~0.92–0.95 / Martinez QTDB ~0.93",
    }
    for k in ("p_on", "qrs_on", "t_on", "p_off", "qrs_off", "t_off"):
        def get(model, dom):
            try:
                return full_results[model][dom]["boundary_raw_pp_tol150"][k]["f1"]
            except KeyError:
                return None
        c_l = get("C_combined_big_le", "LUDB val")
        f_l = get("F_ludb_only_no_le", "LUDB val")
        c_i = get("C_combined_big_le", "ISP test")
        f_i = get("F_ludb_only_no_le", "ISP test")
        c_q = get("C_combined_big_le", "QTDB ext")
        f_q = get("F_ludb_only_no_le", "QTDB ext")
        def fmt(x): return f"{x:.3f}" if x is not None else "  n/a"
        print(f"{k:12s}  {fmt(c_l):>8s} {fmt(f_l):>8s} {fmt(c_i):>8s} {fmt(f_i):>8s} {fmt(c_q):>8s} {fmt(f_q):>8s}  {LITERATURE[k]}", flush=True)

    print("\nMedian timing error in ms (post-processed) — should be < 20ms per spec:", flush=True)
    print(f"{'metric':12s}  {'C LUDB':>8s} {'F LUDB':>8s} {'C ISP':>8s} {'F ISP':>8s} {'C QTDB':>8s} {'F QTDB':>8s}", flush=True)
    for k in ("p_on", "qrs_on", "t_on", "p_off", "qrs_off", "t_off"):
        def get(model, dom):
            try:
                return full_results[model][dom]["boundary_raw_pp_tol150"][k]["median_error_ms"]
            except KeyError:
                return None
        vals = [get("C_combined_big_le", "LUDB val"), get("F_ludb_only_no_le", "LUDB val"),
                get("C_combined_big_le", "ISP test"), get("F_ludb_only_no_le", "ISP test"),
                get("C_combined_big_le", "QTDB ext"), get("F_ludb_only_no_le", "QTDB ext")]
        def fmt(x): return f"{x:5.1f}ms" if x is not None else "    n/a"
        print(f"{k:12s}  " + " ".join(f"{fmt(v):>8s}" for v in vals), flush=True)


if __name__ == "__main__":
    main()
