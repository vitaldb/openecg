"""v15 Phase 2: aux QRS head WITH concat (logits projected into upper
transformer input) + paced synth scenario.

Combines all the AVB-cohort knobs we have:
  * FrameClassifierViTRegAuxConcat: lower 4 layers supervised by aux 4-class
    head; aux logits softmax + concat with lower features → linear projection
    back to d_model → upper 4 layers + main head.
  * SyntheticAVBDataset: 2000 windows mixing mobitz1 / mobitz2 / complete /
    paced (paced uses LUDB-derived wide QRS templates from non-AVB pacers).
  * Loss = main_cls + alpha_aux * aux_cls + lambda_reg * reg_loss.

Outputs:
  data/checkpoints/stage2_v15_concat_paced.pt
  out/retrain_v15_concat_paced_<ts>.json
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
from openecg.stage2.model import FrameClassifierViTRegAuxConcat
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
SYNTH_SCENARIOS = ("mobitz1", "mobitz2", "complete", "paced")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    print("Building real-data train loader...", flush=True)
    real_train = _build_real_loader()

    print("Building TemplateBank (sinus + paced)...", flush=True)
    meta = ludb.load_metadata()
    sinus_ids = [r["id_int"] for r in meta if r["rhythm"].lower() == "sinus rhythm"]
    bank = synth.TemplateBank.from_ludb(
        record_ids=sinus_ids, leads=SYNTH_LEADS, max_per_lead=400,
    )
    for lead in SYNTH_LEADS:
        print(f"  {lead}: P={len(bank.p[lead])} QRS-T sinus={len(bank.qrst[lead])} "
              f"QRS-T paced={len(bank.qrst_paced[lead])}", flush=True)

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
    model = FrameClassifierViTRegAuxConcat(
        n_reg=6, aux_layer_split=AUX_LAYER_SPLIT, **KWARGS,
    )
    n_params = sum(p.numel() for p in model.parameters())
    ckpt_path = CKPT_DIR / "stage2_v15_concat_paced.pt"
    print(f"\n=== TRAIN v15 concat+paced ({n_params:,} params) ===", flush=True)

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
        "synth_scenarios": list(SYNTH_SCENARIOS),
        "train_seconds": elapsed,
        "best_metrics": best,
        "eval": res,
        "refs": {
            "v12_reg":   {"ludb": 0.947, "isp": 0.966, "qtdb": 0.847},
            "v3_synth":  {"ludb": 0.953, "isp": 0.963, "qtdb": 0.851},
            "v13_aux":   {"ludb": 0.953, "isp": 0.964, "qtdb": 0.856},
            "v14_paced": {"ludb": 0.949, "isp": 0.959, "qtdb": 0.852},
        },
    }

    def _safe(v):
        if isinstance(v, dict): return {k: _safe(x) for k, x in v.items()}
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        return v

    ts = time.strftime("%Y%m%d_%H%M%S")
    import json
    out_path = OUT_DIR / f"retrain_v15_concat_paced_{ts}.json"
    out_path.write_text(json.dumps(_safe(payload), indent=2))
    print(f"\nSaved {out_path}\nFinal eval: {res}", flush=True)


if __name__ == "__main__":
    main()
