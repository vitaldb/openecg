"""Per-sample macro-F1 for all three heads on a multihead cache — regression check.

    python3 eval_mh.py codec_v4.pt iter3_ft.pt --cache real_ml_v2_val.npz
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from kgpu_train import _MH
from eval_lydus_rhythm import load_model

IGN = 255


def macro_f1(pred, true, ncls, drop=None):
    f1s = []
    for c in range(ncls):
        if drop is not None and c == drop:
            continue
        if int((true == c).sum()) == 0:
            continue
        tp = int(((pred == c) & (true == c)).sum())
        fp = int(((pred == c) & (true != c)).sum())
        fn = int(((pred != c) & (true == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def evaluate(ckpt, cache, device):
    model = load_model(ckpt, device)
    dl = DataLoader(_MH(cache), batch_size=64, shuffle=False, num_workers=2)
    P = {"frame": [], "rhythm": [], "beat": []}
    T = {"frame": [], "rhythm": [], "beat": []}
    with torch.no_grad():
        for sig, lead, fr, rh, bt in dl:
            o = model(sig.to(device), lead.to(device))
            frame, beat, rhythm = o[0], o[1], o[2]
            for name, pred, tgt in (("frame", frame, fr), ("rhythm", rhythm, rh), ("beat", beat, bt)):
                p = pred.argmax(-1).cpu(); m = (tgt != IGN)
                if name == "beat":
                    m = m & (tgt != 0)
                if m.any():
                    P[name].append(p[m].numpy()); T[name].append(tgt[m].numpy())
    out = {}
    for name, ncls, drop in (("frame", 4, None), ("rhythm", 6, None), ("beat", 6, 0)):
        if T[name]:
            out[name] = macro_f1(np.concatenate(P[name]), np.concatenate(T[name]), ncls, drop)
        else:
            out[name] = float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--cache", default="real_ml_v2_val.npz")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== per-sample macro-F1 on {args.cache} ===")
    print(f"{'ckpt':24s} {'frame':>7s} {'rhythm':>7s} {'beat':>7s}")
    for ck in args.ckpts:
        s = evaluate(ck, args.cache, device)
        print(f"{ck:24s} {s['frame']:7.4f} {s['rhythm']:7.4f} {s['beat']:7.4f}")


if __name__ == "__main__":
    main()
