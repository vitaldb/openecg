"""Build v55a (1-channel) checkpoint from v54i (2-channel, ch1 discarded).

v55a uses :class:`FrameClassifierTransformerNoAux1Ch`, whose weights are
**identical** to v54i's :class:`FrameClassifierTransformerNoAux2Ch` — the
forward path uses only ``x[:, 0]`` in both. This script copies the v54i
state_dict into a freshly-built v55a model, verifies bit-equivalent
forward outputs on a random input, and saves the result as a new ckpt
that the training script's ``--eval-only`` mode can pick up.

Usage::

    python -m scripts.build_v55a \\
        --src data/checkpoints/stage2_v45k_noaux_L8_d128_v54i_tsoft0p7.pt \\
        --dst data/checkpoints/stage2_v45k_noaux_L8_d128_1ch_v55a.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from openecg.stage2.model import load_model_from_ckpt
from openecg.stage2.model_variants import FrameClassifierTransformerNoAux1Ch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/checkpoints/stage2_v45k_noaux_L8_d128_v54i_tsoft0p7.pt")
    ap.add_argument("--dst", default="data/checkpoints/stage2_v45k_noaux_L8_d128_1ch_v55a.pt")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        sys.exit(f"src ckpt not found: {src}")

    # Load v54i (2-channel) — gives back the parent class + the raw blob.
    src_model, blob = load_model_from_ckpt(str(src), device="cpu")
    src_model.train(False)
    src_cfg = dict(blob["model_config"])
    print(f"loaded v54i — arch={src_cfg.get('arch')}  "
          f"params={sum(p.numel() for p in src_model.parameters()):,}")

    # Build v55a (1-channel) with the same hyperparameters.
    ctor_cfg = {k: v for k, v in src_cfg.items() if k not in (
        "arch", "aux_target", "use_aux", "n_input_channels")}
    dst_model = FrameClassifierTransformerNoAux1Ch(**ctor_cfg)
    dst_model.train(False)
    print(f"built v55a — arch={dst_model.model_config['arch']}  "
          f"params={sum(p.numel() for p in dst_model.parameters()):,}")

    # State dicts must be identical — same module hierarchy, same shapes.
    src_sd = src_model.state_dict()
    dst_sd_keys = set(dst_model.state_dict().keys())
    if set(src_sd.keys()) != dst_sd_keys:
        only_src = set(src_sd) - dst_sd_keys
        only_dst = dst_sd_keys - set(src_sd)
        sys.exit(f"state_dict mismatch:\n  only in src: {only_src}\n  only in dst: {only_dst}")
    missing, unexpected = dst_model.load_state_dict(src_sd, strict=True)
    print(f"weight copy ok (no missing/unexpected)")

    # Numerical equivalence: same logits on a deterministic input. v54i
    # consumes (B, 2, T) and discards ch1; v55a consumes (B, T) directly.
    rng = np.random.default_rng(0)
    sig_np = rng.standard_normal((2, 2500)).astype(np.float32)
    sig = torch.from_numpy(sig_np)
    x_2ch = torch.stack([sig, torch.zeros_like(sig)], dim=1)  # (B=2, 2, T)
    lead = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        out_v54i = src_model(x_2ch, lead)[0]
        out_v55a = dst_model(sig, lead)[0]
    diff = float(torch.max(torch.abs(out_v54i - out_v55a)))
    print(f"equivalence check: max |Δ| = {diff:.2e}  "
          f"(expected ~0, since ch1 is unused in v54i)")
    if diff > 1e-6:
        sys.exit(f"forward outputs diverged (diff={diff}) — abort save")

    # Save with the v55a model_config so the train script can re-instantiate
    # the right class on --eval-only re-load.
    out_blob = dict(blob)
    out_blob["model_state"] = dst_model.state_dict()
    out_blob["model_config"] = dict(dst_model.model_config)
    # Preserve original metrics for traceability; they apply unchanged.
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_blob, str(dst))
    print(f"saved -> {dst}  ({dst.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
