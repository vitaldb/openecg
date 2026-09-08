"""Local boundary-F1 + median-ms eval for sweep checkpoints vs codec_v3.

The headline codec_v3 frame metric is boundary macro-F1 (Martinez per-channel
tolerance) + median timing error, NOT per-sample F1. This loads each checkpoint
at its OWN model_config (depth/width-agnostic), predicts the frame channel
per-sample @500 Hz, and scores boundaries with openecg.eval.boundary_error —
the same machinery scripts/eval_resolution_compare.py uses.

    python scripts/eval_boundary_sweep.py --ckpts pod_stage/pod_pull/sw_ft_*.pt \
        openecg/models/codec_v3.pt --ludb data/cache/v57_500hz/ludb_val.npz
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np
import torch

from openecg import eval as ee
from openecg.dsp import rank_normalize
from openecg.stage2.model_variants import FrameClassifierTransformerSampleResConvTokMH1Ch

TOL_MS = {"p_on": 50, "p_off": 50, "qrs_on": 40, "qrs_off": 40, "t_on": 50, "t_off": 100}
CTOR_KEYS = ("patch_size", "n_leads", "d_model", "n_heads", "n_layers", "ff",
             "n_classes", "dropout", "pos_type", "conv_stem", "max_seq_len",
             "mid_split", "lower_kernel")


def boundaries_ms(arr, ms_per_sample=2.0):
    arr = ee.fold_paced_to_qrs(np.asarray(arr, dtype=np.uint8))
    out = {k: [] for k in TOL_MS}; name_for = {1: "p", 2: "qrs", 3: "t"}; prev = 0
    for i, c in enumerate(arr):
        c = int(c)
        if c != prev:
            if prev in name_for:
                out[f"{name_for[prev]}_off"].append(int(round((i - 1) * ms_per_sample)))
            if c in name_for:
                out[f"{name_for[c]}_on"].append(int(round(i * ms_per_sample)))
        prev = c
    if prev in name_for:
        out[f"{name_for[prev]}_off"].append(int(round((len(arr) - 1) * ms_per_sample)))
    return out


def load_ckpt(ck, device):
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = blob["model_config"]
    kw = {k: cfg[k] for k in CTOR_KEYS if k in cfg}
    model = FrameClassifierTransformerSampleResConvTokMH1Ch(
        use_lead_emb=False, beat_n_classes=6, rhythm_n_classes=6, **kw).to(device)
    st = {(k[2:] if k.startswith("m.") else k): v for k, v in blob["model_state"].items()}
    model.load_state_dict(st, strict=False); model.eval()
    return model, cfg


@torch.no_grad()
def eval_one(model, sigs, lbls, device):
    f1s = {k: [] for k in TOL_MS}; med = {k: [] for k in TOL_MS}
    persamp_tp = {1: 0, 2: 0, 3: 0}; persamp_fp = dict(persamp_tp); persamp_fn = dict(persamp_tp)
    for i in range(len(sigs)):
        x = torch.from_numpy(rank_normalize(sigs[i].astype(np.float32))).unsqueeze(0).to(device)
        out = model(x, torch.zeros(1, dtype=torch.long, device=device))
        pred = out[0].argmax(-1)[0].cpu().numpy()
        gt = ee.fold_paced_to_qrs(lbls[i].astype(np.uint8))
        # per-sample P/QRS/T (ignore IGN if present)
        m = gt != 255
        for c in (1, 2, 3):
            persamp_tp[c] += int(((pred == c) & (gt == c) & m).sum())
            persamp_fp[c] += int(((pred == c) & (gt != c) & m).sum())
            persamp_fn[c] += int(((pred != c) & (gt == c) & m).sum())
        gm = boundaries_ms(lbls[i]); pm = boundaries_ms(pred)
        for k in TOL_MS:
            r = ee.boundary_error(pm[k], gm[k], tolerance_ms=TOL_MS[k], fs=1000)
            s, p = r["sensitivity"], r["ppv"]
            f1s[k].append(2 * s * p / (s + p) if (s + p) > 0 else 0.0)
            if r["n_hits"] > 0:
                med[k].append(r["median_error_ms"])
    per_f1 = {k: float(np.mean(v)) if v else 0.0 for k, v in f1s.items()}
    per_med = {k: float(np.mean(v)) if v else 0.0 for k, v in med.items()}
    ps = {}
    for c, nm in ((1, "P"), (2, "QRS"), (3, "T")):
        tp, fp, fn = persamp_tp[c], persamp_fp[c], persamp_fn[c]
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        ps[nm] = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return {"boundary_f1": float(np.mean(list(per_f1.values()))),
            "median_ms": float(np.mean(list(per_med.values()))),
            "per_channel_f1": per_f1, "per_channel_med": per_med,
            "persample_f1": ps, "persample_macro": float(np.mean(list(ps.values())))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--ludb", default="data/cache/v57_500hz/ludb_val.npz")
    ap.add_argument("--out", default="out/eval_boundary_sweep.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b = np.load(args.ludb); sigs, lbls = b["signals"], b["labels"]
    print(f"device={device}  LUDB N={len(sigs)}", flush=True)
    ckpts = []
    for pat in args.ckpts:
        ckpts.extend(sorted(glob.glob(pat)))
    rows = {}
    for ck in ckpts:
        name = Path(ck).stem
        model, cfg = load_ckpt(ck, device)
        r = eval_one(model, sigs, lbls, device)
        r["d_model"] = cfg.get("d_model"); r["n_layers"] = cfg.get("n_layers")
        rows[name] = r
        print(f"  {name:18s} d{cfg.get('d_model')} L{cfg.get('n_layers')} | "
              f"boundary_f1 {r['boundary_f1']:.4f}  median {r['median_ms']:.2f}ms | "
              f"per-sample {r['persample_macro']:.3f} "
              f"(P {r['persample_f1']['P']:.3f} QRS {r['persample_f1']['QRS']:.3f} T {r['persample_f1']['T']:.3f})",
              flush=True)
    print("\n==== BOUNDARY COMPARISON (LUDB 500Hz) ====")
    print(f"{'ckpt':18s} {'bound_f1':>9s} {'med_ms':>7s} {'persamp':>8s}")
    for n, r in sorted(rows.items(), key=lambda kv: -kv[1]['boundary_f1']):
        print(f"{n:18s} {r['boundary_f1']:9.4f} {r['median_ms']:7.2f} {r['persample_macro']:8.3f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
