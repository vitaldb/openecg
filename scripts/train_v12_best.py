# scripts/train_v12_best.py
"""v12_best - apply the winning boundary tweak (soft / reg / soft+reg) on top
of the winning SSL backbone (HuBERT-FT or ST-MEM-FT).

Run only after runs 1-6 have completed. The script reads each
out/train_v12_*_{ts}.json file and picks:
    * boundary tweak winner = arg max LUDB val avg F1 over {v12_soft, v12_reg}
      vs v9 baseline (0.923). If both beat baseline by >= +0.005, use both.
    * backbone winner = arg max LUDB val avg F1 over {v12_hubert_ft,
      v12_stmem_ft}. Tie-break by qtdb_pu0_random.

Spec: docs/superpowers/specs/2026-05-06-v12-ssl-boundary-design.md sec 5.4.
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "WTdelineator"))

from ecgcode import isp, ludb, qtdb
from ecgcode import eval as ecg_eval
from ecgcode.stage2.dataset import LUDBFrameDataset, compute_class_weights
from ecgcode.stage2.multi_dataset import (
    CombinedFrameDataset, QTDBSlidingDataset,
)
from ecgcode.stage2.reg_targets import RegLabelDataset
from ecgcode.stage2.soft_labels import SoftLabelDataset
from ecgcode.stage2.ssl.head import BackboneWithHeads
from ecgcode.stage2.ssl.hubert import HUBERT_DEFAULT_MODEL_ID, HubertECGAdapter
from ecgcode.stage2.ssl.stmem import STMEMAdapter
from ecgcode.stage2.soft_labels import soft_boundary_labels
from ecgcode.stage2.train import (
    TrainConfig, boundary_l1_loss, kl_cross_entropy, load_checkpoint,
    run_eval, run_eval_reg, save_checkpoint, score_val_metrics,
)
from scripts.train_v9_q1c_pu_merge import _ConcatWithCounts, eval_all
from scripts.train_v12_reg import _eval_all as _eval_all_reg

CKPT_DIR = REPO / "data" / "checkpoints"
OUT_DIR = REPO / "out"
SEED = 42
EDGE_MARGIN_MS = 100
V9_LUDB_F1 = 0.923
WIN_THRESHOLD = 0.005


def _latest_json(prefix):
    files = sorted(glob.glob(str(OUT_DIR / f"train_{prefix}_*.json")))
    if not files:
        return None
    return json.loads(Path(files[-1]).read_text())


def _metric(d, experiment_key, metric_key, candidates=None):
    """Look up `metric_key` from d[experiment_key], with optional fallback.

    Each train_v12_*_<ts>.json has shape {"v9_q1c_pu_merge_ref": {...},
    "<experiment_key>": {...}}, OR for the lambda sweep
    {"v9_q1c_pu_merge_ref": {...}, "v12_reg_sweep": {...}, "v12_reg_best": {...}}.
    Picking the first dict-valued entry would silently return the v9 reference
    numbers (the bug in the original plan helper). We look up by exact key
    instead, with a small candidate list for the lambda-sweep case.
    """
    if d is None:
        return float("nan")
    keys = [experiment_key] + list(candidates or [])
    for k in keys:
        v = d.get(k)
        if isinstance(v, dict) and metric_key in v:
            return float(v[metric_key])
    return float("nan")


def _ludb_f1(d, experiment_key, candidates=None):
    return _metric(d, experiment_key, "ludb_edge_filtered", candidates)


def _qtdb_f1(d, experiment_key, candidates=None):
    return _metric(d, experiment_key, "qtdb_pu0_random", candidates)


def _select_winners():
    soft = _latest_json("v12_soft")
    reg = _latest_json("v12_reg")
    hub_ft = _latest_json("v12_hubert_ft")
    stm_ft = _latest_json("v12_stmem_ft")
    # v12_reg.json keys the actual eval block under "v12_reg_best", not "v12_reg".
    reg_keys = ["v12_reg_best"]
    soft_f1 = _ludb_f1(soft, "v12_soft") if soft else 0.0
    reg_f1 = _ludb_f1(reg, "v12_reg", candidates=reg_keys) if reg else 0.0
    use_soft = soft_f1 >= V9_LUDB_F1 + WIN_THRESHOLD
    use_reg = reg_f1 >= V9_LUDB_F1 + WIN_THRESHOLD
    if not (use_soft or use_reg):
        use_soft = soft_f1 >= reg_f1
        use_reg = not use_soft
    hub_f1 = _ludb_f1(hub_ft, "v12_hubert_ft")
    stm_f1 = _ludb_f1(stm_ft, "v12_stmem_ft")
    if hub_f1 == stm_f1:
        backbone = "hubert" if (
            _qtdb_f1(hub_ft, "v12_hubert_ft")
            >= _qtdb_f1(stm_ft, "v12_stmem_ft")
        ) else "stmem"
    else:
        backbone = "hubert" if hub_f1 > stm_f1 else "stmem"
    return {
        "use_soft": bool(use_soft), "use_reg": bool(use_reg),
        "backbone": backbone, "soft_f1": soft_f1, "reg_f1": reg_f1,
        "hub_f1": hub_f1, "stm_f1": stm_f1,
    }


def _build_dataset(use_soft, use_reg, seed):
    ludb_train = LUDBFrameDataset(ludb.load_split()["train"],
                                    mask_unlabeled_edges=True,
                                    edge_margin_ms=EDGE_MARGIN_MS)
    isp_train = CombinedFrameDataset(["isp_train"])
    qtdb_merged = QTDBSlidingDataset(scale_factors=(1.0,),
                                       windows_per_record=20, seed=seed,
                                       q1c_pu_merge=True)
    if use_reg:
        ludb_w = RegLabelDataset(ludb_train)
        isp_w = RegLabelDataset(isp_train)
        qtdb_w = RegLabelDataset(qtdb_merged)
    else:
        ludb_w, isp_w, qtdb_w = ludb_train, isp_train, qtdb_merged
    if use_soft and not use_reg:
        ludb_w = SoftLabelDataset(ludb_w)
        isp_w = SoftLabelDataset(isp_w)
        qtdb_w = SoftLabelDataset(qtdb_w)
    return _ConcatWithCounts([ludb_w, isp_w, qtdb_w])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--stmem-weights", default=None)
    ap.add_argument("--lambda-reg", type=float, default=0.1)
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True); CKPT_DIR.mkdir(parents=True, exist_ok=True)
    winners = _select_winners()
    print(f"Selection: {winners}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    train_ds = _build_dataset(winners["use_soft"], winners["use_reg"], SEED)
    ludb_val = LUDBFrameDataset(ludb.load_split()["val"],
                                  mask_unlabeled_edges=True,
                                  edge_margin_ms=EDGE_MARGIN_MS)
    weights = torch.tensor(compute_class_weights(train_ds.label_counts()),
                            dtype=torch.float32)

    if winners["backbone"] == "hubert":
        backbone = HubertECGAdapter(model_id=HUBERT_DEFAULT_MODEL_ID, device=device)
    else:
        backbone = STMEMAdapter(weights_path=args.stmem_weights, device=device)
    model = BackboneWithHeads(backbone, hidden_dim=backbone.hidden_dim,
                                use_reg=winners["use_reg"])
    param_groups = [
        {"params": backbone.parameters(), "lr": 1e-5},
        {"params": model.cls_head.parameters(), "lr": 1e-3},
    ]
    if winners["use_reg"]:
        param_groups.append({"params": model.reg_head.parameters(), "lr": 1e-3})

    cfg = TrainConfig(epochs=args.epochs, batch_size=32, lr=1e-3,
                       early_stop_patience=7, warmup_frac=0.05)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                                num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(ludb_val, batch_size=32, shuffle=False,
                              num_workers=0, pin_memory=True)
    ckpt_path = CKPT_DIR / "stage2_v12_best.pt"

    model = model.to(device)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * max(1, len(train_loader))
    warmup_steps = int(total_steps * cfg.warmup_frac)
    def lr_lambda(step):
        if step < warmup_steps: return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best = -1; best_metrics = None; bad = 0
    cw = weights.to(device)
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train(); total = 0; n = 0
        for batch in train_loader:
            optimizer.zero_grad()
            if winners["use_reg"]:
                sigs, leads, labels, reg_t, reg_m = batch
                sigs = sigs.to(device); leads = leads.to(device)
                labels = labels.to(device); reg_t = reg_t.to(device).float(); reg_m = reg_m.to(device).bool()
                cls, reg = model(sigs, leads)
                if winners["use_soft"]:
                    # Build TRUE soft labels per-sample using soft_boundary_labels
                    # (ignore_index → all-zero row, transitions → 70/30 mix).
                    # NOT one-hot: that drops Approach A's softening and
                    # mistrains ignore frames as class 3.
                    labels_np = labels.cpu().numpy()
                    soft = torch.from_numpy(np.stack([
                        soft_boundary_labels(labels_np[i])
                        for i in range(labels_np.shape[0])
                    ])).to(device)
                    cls_loss = kl_cross_entropy(cls, soft, weight=cw)
                else:
                    cls_loss = torch.nn.functional.cross_entropy(
                        cls.transpose(1, 2), labels, weight=cw, ignore_index=255,
                    )
                reg_loss = boundary_l1_loss(reg, reg_t, reg_m)
                loss = cls_loss + args.lambda_reg * reg_loss
            else:
                sigs, leads, target = batch
                sigs = sigs.to(device); leads = leads.to(device); target = target.to(device)
                cls = model(sigs, leads)
                if winners["use_soft"]:
                    loss = kl_cross_entropy(cls, target.float(), weight=cw)
                else:
                    loss = torch.nn.functional.cross_entropy(
                        cls.transpose(1, 2), target, weight=cw, ignore_index=255,
                    )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step(); scheduler.step()
            total += float(loss.item()); n += 1
        val_fn = run_eval_reg if winners["use_reg"] else run_eval
        val = val_fn(model, val_loader, device)
        score = score_val_metrics(val, cfg.early_stop_metric)
        print(f"epoch {epoch:3d} train={total/max(1,n):.4f} score={score:.3f}", flush=True)
        if score > best:
            best = score
            best_metrics = {"epoch": epoch, "val_score": score, **winners,
                              "lambda_reg": args.lambda_reg}
            bad = 0
            save_checkpoint(ckpt_path, model, best_metrics, cfg)
        else:
            bad += 1
            if bad >= cfg.early_stop_patience: print(f"early stop {epoch}"); break
    elapsed = time.time() - t0
    if ckpt_path.exists():
        load_checkpoint(ckpt_path, model)
    model = model.to(device).train(False)
    # When use_reg=True the model returns (cls, reg) tuples and v9's
    # eval_all (which calls predict_frames -> .argmax on the output)
    # would crash. Use the reg-aware variant from train_v12_reg.
    res = _eval_all_reg(model, device) if winners["use_reg"] else eval_all(model, device)
    print(f"\n=== v12_best eval ===", flush=True)
    for k, v in res.items(): print(f"  {k}: {v:.3f}")

    full = {
        "v9_q1c_pu_merge_ref": {"ludb_edge_filtered": V9_LUDB_F1,
                                  "isp_test": 0.943,
                                  "qtdb_pu0_random": 0.779},
        "v12_best": {"selection": winners,
                       "lambda_reg": args.lambda_reg,
                       "train_seconds": elapsed,
                       "best_metrics": best_metrics or {}, **res},
    }
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"train_v12_best_{ts}.json"
    def _safe(v):
        if isinstance(v, dict): return {k: _safe(x) for k, x in v.items()}
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        return v
    out_path.write_text(json.dumps(_safe(full), indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
