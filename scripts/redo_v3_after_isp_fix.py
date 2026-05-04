"""Re-run v3 regression ablation after fixing the ISP label alignment bug.

Bug: gt_to_super_frames previously used samples_per_frame = n_samples // n_frames,
which gave 19 (instead of 20) for ISP records of 9999 samples at 1000Hz with
frame_ms=20. Result: cumulative time drift up to 500ms by frame 499. Combined
training settings (which include ISP) were learning misaligned labels.

Affected previous experiments:
  - v3 (combined, big, focal+aug): training and eval both buggy
  - B/C/D/E (ablate_v3_regression): training and eval both buggy
  - G (ablate_v4_candidate): training and eval both buggy

Unaffected:
  - v1.0, A, F (LUDB only): LUDB is exactly 5000 samples → samples_per_frame=10 → no drift

This script re-runs only the affected settings and re-evaluates F's existing
ckpt on the now-fixed ISP test set.

Settings:
  B: combined, d=64/L=4,  CE,        lead_emb=on
  C: combined, d=128/L=8, CE,        lead_emb=on
  D: combined, d=128/L=8, focal+aug, lead_emb=on  (= v3 reproduce)
  E: combined, d=64/L=4,  CE,        lead_emb=OFF
  G: combined, d=64/L=4,  CE,        lead_emb=OFF + best-ckpt save

Eval all on: LUDB val (in-domain), ISP test (now clean), QTDB ext (was clean).
F's existing ckpt evaluated on same domains for reference.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ecgcode import eval as ee
from ecgcode import ludb
from ecgcode.stage2.dataset import LUDBFrameDataset, compute_class_weights
from ecgcode.stage2.model import FrameClassifier
from ecgcode.stage2.multi_dataset import (
    CombinedFrameDataset,
    CombinedFrameDatasetAugmented,
)
from ecgcode.stage2.train import TrainConfig, fit, load_checkpoint

OUT_DIR = Path("out")
CKPT_DIR = Path("data/checkpoints")
F_CKPT = CKPT_DIR / "stage2_v4_ludb_only.pt"  # unchanged, no need to retrain
EPOCHS = 50
PATIENCE = 10
SEED = 42


def eval_on(model, ds, device):
    model.train(False)
    pred_list, true_list = [], []
    with torch.no_grad():
        for idx in range(len(ds)):
            sig, lead_id, labels = ds[idx]
            x = sig.unsqueeze(0).to(device)
            lid = lead_id.unsqueeze(0).to(device)
            logits = model(x, lid)
            pred = logits.argmax(dim=-1).cpu().numpy().reshape(-1).astype(np.uint8)
            true = labels.numpy().reshape(-1).astype(np.uint8)
            pred_list.append(pred); true_list.append(true)
    f1 = ee.frame_f1(np.concatenate(pred_list), np.concatenate(true_list))
    return {
        "P": f1[ee.SUPER_P]["f1"],
        "QRS": f1[ee.SUPER_QRS]["f1"],
        "T": f1[ee.SUPER_T]["f1"],
        "other": f1[ee.SUPER_OTHER]["f1"],
        "n_seqs": len(ds),
    }


def train_setting(name, train_ds, val_ds, model_kwargs, use_focal, device, ckpt=None):
    print(f"\n{'='*70}\nTRAIN {name}: {model_kwargs}, focal={use_focal}, ckpt={'yes' if ckpt else 'no'}\n{'='*70}", flush=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    counts = train_ds.label_counts()
    weights = torch.tensor(compute_class_weights(counts), dtype=torch.float32)
    cfg = TrainConfig(epochs=EPOCHS, batch_size=64, lr=1e-3, early_stop_patience=PATIENCE)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    model = FrameClassifier(**model_kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params={n_params:,}, train n={len(train_ds)}", flush=True)
    t0 = time.time()
    best = fit(model, train_loader, val_loader, weights, cfg, device=device,
               ckpt_path=ckpt, use_focal=use_focal)
    elapsed = time.time() - t0
    print(f"  trained in {elapsed:.1f}s, best={best}", flush=True)
    if ckpt is not None and Path(ckpt).exists():
        load_checkpoint(ckpt, model)
        print(f"  reloaded best-ckpt from {ckpt}", flush=True)
    return model.to(device), {
        "n_params": n_params, "n_train": len(train_ds), "train_seconds": elapsed,
        "best_during_training": best,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    OUT_DIR.mkdir(exist_ok=True); CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading datasets...", flush=True)
    t0 = time.time()
    ludb_train = LUDBFrameDataset(ludb.load_split()["train"])
    ludb_val = LUDBFrameDataset(ludb.load_split()["val"])
    print(f"  LUDB train/val: {len(ludb_train)}/{len(ludb_val)} ({time.time()-t0:.1f}s)", flush=True)
    t0 = time.time()
    combined_train = CombinedFrameDataset(["ludb_train", "qtdb", "isp_train"])
    print(f"  combined: {len(combined_train)} ({time.time()-t0:.1f}s)", flush=True)
    print(f"  source_counts: {dict(combined_train.source_counts())}", flush=True)
    t0 = time.time()
    combined_train_aug = CombinedFrameDatasetAugmented(
        ["ludb_train", "qtdb", "isp_train"], n_ops=2, seed=SEED,
    )
    print(f"  combined aug: {len(combined_train_aug)} ({time.time()-t0:.1f}s)", flush=True)
    qtdb_test = CombinedFrameDataset(["qtdb"])
    isp_test = CombinedFrameDataset(["isp_test"])
    print(f"  QTDB ext / ISP test: {len(qtdb_test)} / {len(isp_test)}", flush=True)

    settings = [
        ("B_combined_small_le",  combined_train,     {"d_model": 64,  "n_layers": 4}, False, None),
        ("C_combined_big_le",    combined_train,     {"d_model": 128, "n_layers": 8}, False, None),
        ("D_v3_repro",           combined_train_aug, {"d_model": 128, "n_layers": 8}, True,  None),
        ("E_no_lead_emb",        combined_train,     {"d_model": 64,  "n_layers": 4, "use_lead_emb": False}, False, None),
        ("G_v4_combined_ckpt",   combined_train,     {"d_model": 64,  "n_layers": 4, "use_lead_emb": False}, False,
         CKPT_DIR / "stage2_v4_combined_fixed.pt"),
    ]

    results = {}
    for name, train_ds, mk, focal, ckpt in settings:
        try:
            model, meta = train_setting(name, train_ds, ludb_val, mk, focal, device, ckpt=ckpt)
            domain_f1 = {}
            for dname, ds in [("ludb_val", ludb_val), ("qtdb_ext", qtdb_test), ("isp_test", isp_test)]:
                m = eval_on(model, ds, device)
                domain_f1[dname] = m
                print(f"  {dname:10s} (n={m['n_seqs']:4d}): "
                      f"P={m['P']:.3f} QRS={m['QRS']:.3f} T={m['T']:.3f}", flush=True)
            results[name] = {**meta, "model_kwargs": mk, "use_focal": focal,
                             "ckpt_loaded_for_eval": ckpt is not None,
                             "domains": domain_f1}
        except Exception as ex:
            results[name] = {"error": repr(ex)}
            print(f"  FAILED: {ex}", flush=True)

    # Re-evaluate F (existing ckpt, no retraining) on the 3 domains with fixed labels.
    print(f"\n{'='*70}\nRE-EVAL F (no retrain, ckpt={F_CKPT})\n{'='*70}", flush=True)
    if F_CKPT.exists():
        f_model = FrameClassifier(d_model=64, n_layers=4, use_lead_emb=False)
        load_checkpoint(F_CKPT, f_model)
        f_model = f_model.to(device)
        f_domains = {}
        for dname, ds in [("ludb_val", ludb_val), ("qtdb_ext", qtdb_test), ("isp_test", isp_test)]:
            m = eval_on(f_model, ds, device)
            f_domains[dname] = m
            print(f"  {dname:10s} (n={m['n_seqs']:4d}): "
                  f"P={m['P']:.3f} QRS={m['QRS']:.3f} T={m['T']:.3f}", flush=True)
        results["F_ludb_only_ref"] = {"domains": f_domains, "ckpt_loaded_for_eval": True}

    summary = {"epochs": EPOCHS, "patience": PATIENCE, "settings": results}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"redo_v3_after_isp_fix_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}", flush=True)

    print("\n" + "=" * 90, flush=True)
    print(f"{'POST-FIX RESULTS (frame F1 P/QRS/T per domain)':^90}", flush=True)
    print("=" * 90, flush=True)
    print(f"{'setting':24s} {'LUDB val':>22s}  {'QTDB ext':>22s}  {'ISP test':>22s}", flush=True)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:24s} ERROR  {r['error']}", flush=True)
            continue
        d = r.get("domains", {})
        def fmt(m):
            if not m: return "n/a"
            return f"{m['P']:.3f}/{m['QRS']:.3f}/{m['T']:.3f}"
        print(f"{name:24s} {fmt(d.get('ludb_val')):>22s}  "
              f"{fmt(d.get('qtdb_ext')):>22s}  {fmt(d.get('isp_test')):>22s}", flush=True)

    print(f"\nReference (PRE-fix, for comparison):")
    print(f"  v1.0 (LUDB only, lead_emb on): LUDB val P=0.604/QRS=0.806/T=0.695")
    print(f"  A (LUDB only big, lead_emb on): LUDB val P=0.652/QRS=0.812/T=0.711 (still valid)")


if __name__ == "__main__":
    main()
