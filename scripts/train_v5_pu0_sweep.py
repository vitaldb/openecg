"""Train v5: pu0/pu1 dense QTDB labels + LUDB edge-masked + ISP.

Two architectures × three sizes = 6 variants.

Architectures:
  cnn : Conv1d (k=15 s=5 + k=5 s=1) + Transformer  (= v4 baseline)
  vit : non-overlapping 5-sample (20ms) patches + Linear projection +
        sinusoidal positional encoding + Transformer

Sizes (same hyperparameters across architectures):
  s : d_model=64,  n_layers=4,  ff=128
  m : d_model=96,  n_layers=6,  ff=192
  l : d_model=128, n_layers=8,  ff=256  (= v4 C size)

Reference: v4 C ckpt for direct comparison.
Eval domains: LUDB val (edge-filtered), ISP test, QTDB q1c span, QTDB pu0 full.
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "WTdelineator"))

from openecg import isp, ludb, qtdb
from openecg.stage2.dataset import LUDBFrameDataset, compute_class_weights
from openecg.stage2.evaluate import MARTINEZ_TOLERANCE_MS, signed_boundary_metrics
from openecg.stage2.infer import (
    extract_boundaries, load_model_bundle, post_process_frames, predict_frames,
)
from openecg.stage2.model import FrameClassifier, FrameClassifierViT
from openecg.stage2.multi_dataset import (
    CombinedFrameDataset, QTDBPuFullDataset, _decimate_to_250, _normalize,
)
from openecg.stage2.train import TrainConfig, fit, load_checkpoint

CKPT_DIR = Path("data/checkpoints")
OUT_DIR = Path("out")
WINDOW_SAMPLES = 2500
WINDOW_FRAMES = 500
FRAME_MS = 20
FS = 250
SEED = 42
EDGE_MARGIN_MS = 100  # for LUDB edge filter in eval

SIZES = {
    "s": {"d_model": 64,  "n_heads": 4, "n_layers": 4, "ff": 128},
    "m": {"d_model": 96,  "n_heads": 4, "n_layers": 6, "ff": 192},
    "l": {"d_model": 128, "n_heads": 4, "n_layers": 8, "ff": 256},
}

ARCHS = {
    "cnn": FrameClassifier,
    "vit": FrameClassifierViT,
}


# ---------- Datasets ----------

class _LUDBOnlyTrain:
    """Adapter so LUDB train (edge-masked) cooperates with ConcatDataset."""
    pass


class _ConcatWithCounts(ConcatDataset):
    def label_counts(self):
        from openecg.stage2.multi_dataset import N_CLASSES
        total = np.zeros(N_CLASSES, dtype=np.int64)
        for d in self.datasets:
            if hasattr(d, "label_counts"):
                lc = d.label_counts()
                if lc is not None:
                    total += lc[:N_CLASSES]
        return np.maximum(total, 1)


def build_train_dataset(qtdb_windows_per_record=20):
    print("Building v5 train dataset...", flush=True)
    ludb_train = LUDBFrameDataset(
        ludb.load_split()["train"], mask_unlabeled_edges=True,
        edge_margin_ms=EDGE_MARGIN_MS,
    )
    print(f"  LUDB train (edge-masked): {len(ludb_train)} sequences", flush=True)

    isp_train = CombinedFrameDataset(["isp_train"])
    print(f"  ISP train: {len(isp_train)} sequences", flush=True)

    qtdb_pu = QTDBPuFullDataset(windows_per_record=qtdb_windows_per_record, seed=SEED)
    print(f"  QTDB pu0/pu1 sliding: {len(qtdb_pu)} windows", flush=True)

    combined = _ConcatWithCounts([ludb_train, isp_train, qtdb_pu])
    print(f"  TOTAL: {len(combined)}", flush=True)
    return combined


# ---------- Eval ----------

def add_b(acc, local, cum):
    for k, v in local.items():
        acc[k].extend(int(x) + cum for x in v)


def eval_model_ludb(model, device):
    """LUDB val with edge filter."""
    val_ds = LUDBFrameDataset(ludb.load_split()["val"])
    bp, bt = defaultdict(list), defaultdict(list)
    cum = 0
    margin_250 = int(round(EDGE_MARGIN_MS * FS / 1000.0))
    with torch.no_grad():
        for idx in range(len(val_ds)):
            rid, lead = val_ds.items[idx]
            sig_250, lead_idx, _ = val_ds.cache[(rid, lead)]
            sig_250 = sig_250[:WINDOW_SAMPLES]
            if len(sig_250) < WINDOW_SAMPLES:
                continue
            rng = ludb.labeled_range(rid, lead)
            if rng is None:
                continue
            lo = max(0, rng[0] // 2 - margin_250)
            hi = min(WINDOW_SAMPLES, rng[1] // 2 + margin_250 + 1)
            raw = predict_frames(model, sig_250, lead_idx, device=device)
            pp = post_process_frames(raw, frame_ms=FRAME_MS)
            for k, vs in extract_boundaries(pp, fs=FS, frame_ms=FRAME_MS).items():
                for s in vs:
                    if lo <= s < hi:
                        bp[k].append(int(s) + cum)
            gt = ludb.load_annotations(rid, lead)
            for k, vs in gt.items():
                if k.endswith("_on") or k.endswith("_off"):
                    for s in vs:
                        s250 = int(s // 2)
                        if lo <= s250 < hi:
                            bt[k].append(s250 + cum)
            cum += WINDOW_SAMPLES
    return bp, bt


def eval_model_isp(model, device):
    bp, bt = defaultdict(list), defaultdict(list)
    cum = 0
    with torch.no_grad():
        for rid in isp.load_split()["test"]:
            try:
                record = isp.load_record(rid, split="test")
                ann = isp.load_annotations_as_super(rid, split="test")
            except Exception:
                continue
            for lead_idx, lead in enumerate(isp.LEADS_12):
                sig_1000 = record[lead]
                sig_250 = _decimate_to_250(sig_1000, 1000)
                sig_n = _normalize(sig_250)
                if len(sig_n) < WINDOW_SAMPLES:
                    pad = np.zeros(WINDOW_SAMPLES - len(sig_n), dtype=sig_n.dtype)
                    sig_n = np.concatenate([sig_n, pad])
                sig_n = sig_n[:WINDOW_SAMPLES]
                raw = predict_frames(model, sig_n, lead_idx, device=device)
                pp = post_process_frames(raw, frame_ms=FRAME_MS)
                for k, vs in extract_boundaries(pp, fs=FS, frame_ms=FRAME_MS).items():
                    for s in vs:
                        bp[k].append(int(s) + cum)
                for k, vs in ann.items():
                    if k.endswith("_on") or k.endswith("_off"):
                        for s in vs:
                            s250 = int(s // 4)
                            if 0 <= s250 < WINDOW_SAMPLES:
                                bt[k].append(s250 + cum)
                cum += WINDOW_SAMPLES
    return bp, bt


QTDB_EVAL_SEED = 42
QTDB_PU0_WINDOWS_PER_RECORD = 5  # random windows per record for pu0 eval


def eval_model_qtdb_q1c(model, device):
    """QTDB q1c span eval — one window per record covering the q1c span."""
    bp, bt = defaultdict(list), defaultdict(list)
    cum = 0
    rng = np.random.default_rng(QTDB_EVAL_SEED)
    with torch.no_grad():
        for rid in qtdb.records_with_q1c():
            try:
                record = qtdb.load_record(rid)
                ann = qtdb.load_q1c(rid)
            except Exception:
                continue
            all_pos = []
            for k in ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off"):
                all_pos.extend(ann.get(k, []))
            if not all_pos:
                continue
            ann_min = min(all_pos)
            ann_max = max(all_pos)
            first_lead = list(record.keys())[0]
            sig_full = record[first_lead].astype(np.float32)
            n = len(sig_full)
            sig_norm = _normalize(sig_full)
            # One window centered on the q1c annotated span midpoint
            mid = (ann_min + ann_max) // 2
            start = max(0, min(n - WINDOW_SAMPLES, mid - WINDOW_SAMPLES // 2))
            sig_win = sig_norm[start:start + WINDOW_SAMPLES].astype(np.float32)
            sig_w = ((sig_win - sig_win.mean()) / (sig_win.std() + 1e-6)).astype(np.float32)
            raw = predict_frames(model, sig_w, lead_id=1, device=device)
            pp = post_process_frames(raw, frame_ms=FRAME_MS)
            for k, vs in extract_boundaries(pp, fs=FS, frame_ms=FRAME_MS).items():
                for s in vs:
                    abs_s = start + s
                    if ann_min <= abs_s <= ann_max:
                        bp[k].append(int(abs_s) + cum)
            for k in ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off"):
                for s in ann.get(k, []):
                    if ann_min <= s <= ann_max:
                        bt[k].append(int(s) + cum)
            cum += n
    return bp, bt


def eval_model_qtdb_pu0(model, device, n_windows=QTDB_PU0_WINDOWS_PER_RECORD):
    """QTDB pu0 eval — N random non-overlapping windows per record."""
    bp, bt = defaultdict(list), defaultdict(list)
    cum = 0
    rng = np.random.default_rng(QTDB_EVAL_SEED)
    with torch.no_grad():
        for rid in qtdb.records_with_q1c():
            try:
                record = qtdb.load_record(rid)
                ann = qtdb.load_pu(rid, lead=0)
            except Exception:
                continue
            first_lead = list(record.keys())[0]
            sig_full = record[first_lead].astype(np.float32)
            n = len(sig_full)
            sig_norm = _normalize(sig_full)
            # Random non-overlapping windows
            n_max_windows = n // WINDOW_SAMPLES
            k = min(n_windows, n_max_windows)
            chosen = rng.choice(n_max_windows, size=k, replace=False)
            covered_ranges = []
            for w in sorted(chosen):
                start = int(w) * WINDOW_SAMPLES
                end = start + WINDOW_SAMPLES
                covered_ranges.append((start, end))
                sig_win = sig_norm[start:end].astype(np.float32)
                sig_w = ((sig_win - sig_win.mean()) / (sig_win.std() + 1e-6)).astype(np.float32)
                raw = predict_frames(model, sig_w, lead_id=1, device=device)
                pp = post_process_frames(raw, frame_ms=FRAME_MS)
                for ck, vs in extract_boundaries(pp, fs=FS, frame_ms=FRAME_MS).items():
                    for s in vs:
                        bp[ck].append(int(start + s) + cum)
            # GT: keep only annotations within sampled windows
            for ck in ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off"):
                for s in ann.get(ck, []):
                    for lo, hi in covered_ranges:
                        if lo <= s < hi:
                            bt[ck].append(int(s) + cum)
                            break
            cum += n
    return bp, bt


def avg_f1(bp, bt):
    f1s = []
    for k in ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off"):
        m = signed_boundary_metrics(bp.get(k, []), bt.get(k, []),
                                      tolerance_ms=MARTINEZ_TOLERANCE_MS[k], fs=FS)
        f1s.append(m["f1"])
    return float(np.mean(f1s))


def eval_all_domains(model, device):
    bp_lu, bt_lu = eval_model_ludb(model, device)
    bp_is, bt_is = eval_model_isp(model, device)
    bp_q1, bt_q1 = eval_model_qtdb_q1c(model, device)
    bp_pu, bt_pu = eval_model_qtdb_pu0(model, device)
    return {
        "ludb_edge_filtered": avg_f1(bp_lu, bt_lu),
        "isp_test": avg_f1(bp_is, bt_is),
        "qtdb_q1c_span": avg_f1(bp_q1, bt_q1),
        "qtdb_pu0_full": avg_f1(bp_pu, bt_pu),
    }


# ---------- Train ----------

def train_one(name, train_ds, val_ds, arch_cls, model_kwargs, device, ckpt_path):
    print(f"\n=== TRAIN {name} ===", flush=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    weights = torch.tensor(compute_class_weights(train_ds.label_counts()), dtype=torch.float32)
    cfg = TrainConfig(epochs=80, batch_size=64, lr=1e-3, early_stop_patience=15)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                               num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                             num_workers=0, pin_memory=True)
    model = arch_cls(**model_kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  arch={arch_cls.__name__} params={n_params:,}, train n={len(train_ds)}", flush=True)
    t0 = time.time()
    best = fit(model, train_loader, val_loader, weights, cfg, device=device,
               ckpt_path=ckpt_path, use_focal=False)
    elapsed = time.time() - t0
    print(f"  trained in {elapsed:.1f}s, best={best}", flush=True)
    if ckpt_path and Path(ckpt_path).exists():
        load_checkpoint(ckpt_path, model)
    return model.to(device).train(False), elapsed, n_params


def main():
    OUT_DIR.mkdir(exist_ok=True); CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    ludb_val = LUDBFrameDataset(ludb.load_split()["val"],
                                  mask_unlabeled_edges=True,
                                  edge_margin_ms=EDGE_MARGIN_MS)
    print(f"LUDB val (edge-masked for early-stop): {len(ludb_val)}", flush=True)

    train_ds = build_train_dataset(qtdb_windows_per_record=20)

    full = {}

    # --- Reference: v4 C ---
    print("\n" + "=" * 78, flush=True)
    print(f"{'>>> v4 C (reference, no retrain) <<<':^78}", flush=True)
    print("=" * 78, flush=True)
    ref_ckpt = CKPT_DIR / "stage2_v4_C.pt"
    bundle = load_model_bundle(ref_ckpt, device=device)
    ref_model = bundle["model"].train(False)
    ref_n_params = sum(p.numel() for p in ref_model.parameters())
    t0 = time.time()
    full["v4_C_ref"] = {"params": ref_n_params, **eval_all_domains(ref_model, device),
                         "eval_seconds": time.time() - t0}
    for k, v in full["v4_C_ref"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    # --- v5 sweep: 2 archs × 3 sizes = 6 variants ---
    for arch_label, arch_cls in ARCHS.items():
        for size_label, model_kwargs in SIZES.items():
            name = f"v5_{arch_label}_{size_label}"
            print("\n" + "=" * 78, flush=True)
            print(f"{f'>>> {name} (pu0+pu1 train) <<<':^78}", flush=True)
            print("=" * 78, flush=True)
            ckpt = CKPT_DIR / f"stage2_{name}.pt"
            model, elapsed, n_params = train_one(name, train_ds, ludb_val,
                                                  arch_cls, model_kwargs, device, ckpt)
            t0 = time.time()
            results = eval_all_domains(model, device)
            full[name] = {"arch": arch_label, "size": size_label,
                           "params": n_params, "train_seconds": elapsed,
                           "eval_seconds": time.time() - t0, **results}
            for k, v in full[name].items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.3f}")
                else:
                    print(f"  {k}: {v}")

    # --- Summary ---
    print("\n" + "=" * 100, flush=True)
    print(f"{'v5 SUMMARY: avg Martinez F1 by domain':^100}", flush=True)
    print("=" * 100, flush=True)
    print(f"  {'model':14s} {'params':>10s} | {'LUDB(edge)':>10s} {'ISP':>10s} "
          f"{'QTDB q1c':>10s} {'QTDB pu0':>10s}", flush=True)
    for name, r in full.items():
        print(f"  {name:14s} {r['params']:>10,d} | "
              f"{r['ludb_edge_filtered']:>10.3f} {r['isp_test']:>10.3f} "
              f"{r['qtdb_q1c_span']:>10.3f} {r['qtdb_pu0_full']:>10.3f}", flush=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"train_v5_pu0_sweep_{ts}.json"

    def _safe(v):
        if isinstance(v, dict): return {k: _safe(x) for k, x in v.items()}
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        return v

    out_path.write_text(json.dumps(_safe(full), indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
