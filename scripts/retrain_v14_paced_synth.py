"""v14: retrain with FrameClassifierViTRegAux (auxiliary 4-class
head tapped after lower-4 transformer layers, supervising the same labels).

Same data mix as retrain_v12_reg_with_synth.py:
  LUDB train (mask_unlabeled_edges) + ISP train + QTDB-sliding +
  SyntheticAVBDataset (2000 windows, augmented v3 generator).

Loss = main_cls + alpha_aux * aux_cls + lambda_reg * reg_loss.

Output:
  data/checkpoints/stage2_v14_paced_synth.pt
  out/retrain_v14_paced_synth_<ts>.json
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "WTdelineator"))

from openecg import ludb, synth
from openecg.stage2.dataset import LUDBFrameDataset, compute_class_weights
from openecg.stage2.model import FrameClassifierViTRegAux
from openecg.stage2.reg_targets import RegLabelDataset
from openecg.stage2.synth_dataset import SyntheticAVBDataset
from openecg.stage2.train import TrainConfig, fit_reg_aux, load_checkpoint
from scripts.train_v12_reg import EDGE_MARGIN_MS, SEED, _eval_all
from scripts.train_v12_reg import _build_train_loader as _build_real_loader
from scripts.train_v9_q1c_pu_merge import KWARGS, _ConcatWithCounts

CKPT_DIR = REPO / "data" / "checkpoints"
OUT_DIR = REPO / "out"
LAMBDA = 0.05
ALPHA_AUX = 0.3
AUX_LAYER_SPLIT = 4
N_SYNTH_WINDOWS = 2000
SYNTH_LEADS = ("ii", "v1", "i", "v5", "v2")
SYNTH_SCENARIOS = ("mobitz1", "mobitz2", "complete")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    print("Building real-data train loader (LUDB + ISP + QTDB-sliding)...",
          flush=True)
    real_train = _build_real_loader()

    print("Building TemplateBank from LUDB sinus records...", flush=True)
    meta = ludb.load_metadata()
    sinus_ids = [r["id_int"] for r in meta if r["rhythm"].lower() == "sinus rhythm"]
    bank = synth.TemplateBank.from_ludb(
        record_ids=sinus_ids, leads=SYNTH_LEADS, max_per_lead=400,
    )
    for lead in SYNTH_LEADS:
        print(f"  {lead}: P={len(bank.p[lead])} QRS-T={len(bank.qrst[lead])}",
              flush=True)

    synth_ds = SyntheticAVBDataset(
        bank, leads=SYNTH_LEADS, scenarios=SYNTH_SCENARIOS,
        n_windows=N_SYNTH_WINDOWS, base_seed=SEED,
    )
    synth_reg = RegLabelDataset(synth_ds)

    train_ds = _ConcatWithCounts([real_train, synth_reg])
    print(f"Total train windows: {len(train_ds)}", flush=True)

    ludb_val = LUDBFrameDataset(
        ludb.load_split()["val"],
        mask_unlabeled_edges=True,
        edge_margin_ms=EDGE_MARGIN_MS,
    )
    weights = torch.tensor(
        compute_class_weights(train_ds.label_counts()), dtype=torch.float32,
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    cfg = TrainConfig(epochs=80, batch_size=64, lr=1e-3, early_stop_patience=15)
    train_loader = DataLoader(
        train_ds, batch_size=64, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        ludb_val, batch_size=64, shuffle=False, num_workers=0, pin_memory=True,
    )
    model = FrameClassifierViTRegAux(
        n_reg=6, aux_layer_split=AUX_LAYER_SPLIT, **KWARGS,
    )
    n_params = sum(p.numel() for p in model.parameters())
    ckpt_path = CKPT_DIR / "stage2_v14_paced_synth.pt"
    print(f"\n=== TRAIN v13 aux_qrs (lambda={LAMBDA}, alpha_aux={ALPHA_AUX}, "
          f"aux_layer_split={AUX_LAYER_SPLIT}, {n_params:,} params) ===",
          flush=True)

    t0 = time.time()
    best = fit_reg_aux(
        model, train_loader, val_loader, weights, cfg,
        device=device, ckpt_path=ckpt_path,
        lambda_reg=LAMBDA, alpha_aux=ALPHA_AUX,
    )
    elapsed = time.time() - t0

    if ckpt_path.exists():
        load_checkpoint(ckpt_path, model)
    model = model.to(device).train(False)

    res = _eval_all(model, device)
    payload = {
        "params": n_params,
        "lambda": LAMBDA,
        "alpha_aux": ALPHA_AUX,
        "aux_layer_split": AUX_LAYER_SPLIT,
        "n_synth_windows": N_SYNTH_WINDOWS,
        "train_seconds": elapsed,
        "best_metrics": best,
        "eval": res,
        "v12_reg_baseline_ref": {
            "ludb_edge_filtered": 0.947,
            "isp_test": 0.966,
            "qtdb_pu0_random": 0.847,
        },
        "v12_reg_with_synth_v3_ref": {
            "ludb_edge_filtered": 0.953,
            "isp_test": 0.963,
            "qtdb_pu0_random": 0.851,
        },
    }

    def _safe(v):
        if isinstance(v, dict):
            return {k: _safe(x) for k, x in v.items()}
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        return v

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"retrain_v14_paced_synth_{ts}.json"
    import json
    out_path.write_text(json.dumps(_safe(payload), indent=2))
    print(f"\nSaved {out_path}", flush=True)
    print(f"Final eval: {res}", flush=True)


if __name__ == "__main__":
    main()
