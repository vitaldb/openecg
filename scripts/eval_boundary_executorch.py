"""F1 evaluation of an exported ExecuTorch .pte boundary model.

Mirrors ``eval_boundary_tflite.py`` but loads the model via
``executorch.runtime`` instead of tflite-runtime.

Run::

    python -m scripts.eval_boundary_executorch \\
        --pte data/deploy/boundary_et_int8.pte \\
        --src-ckpt data/checkpoints/stage2_v45k_noaux_L8_d128_1ch_v56c.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.retrain_v40_common import score_all_1ch


class ExecuTorchTorchAdapter(torch.nn.Module):
    """Adapt an ExecuTorch ``.pte`` runtime to the torch ``(x, lead_id)``
    calling convention used by ``score_all_1ch``.

    ExecuTorch's Python runtime exposes ``program.load_method("forward")``
    and ``method.execute([inputs])`` returning a list of output tensors.
    """

    def __init__(self, pte_path: str | Path):
        super().__init__()
        from executorch.runtime import Runtime
        self._rt = Runtime.get()
        self._program = self._rt.load_program(str(pte_path))
        self._method = self._program.load_method("forward")

    def forward(self, x: torch.Tensor, lead_id):
        # Boundary 1-ch input contract: ``(B, T)``; convert from (B, 1, T)
        # or (B, 2, T) if the upstream compose returned those.
        if x.dim() == 3:
            x = x[:, 0]
        # ExecuTorch fixed-shape graph was exported with B=1; loop windows.
        x = x.detach().cpu().to(torch.float32).contiguous()
        cls_outs = []
        reg_outs = []
        for b in range(x.shape[0]):
            outs = self._method.execute([x[b:b+1]])
            outs = [o if isinstance(o, torch.Tensor) else torch.from_numpy(o)
                    for o in outs]
            # Outputs are (cls_logits, reg_offsets) in that order from the
            # wrapper; we still locate by last-axis size to be robust.
            cls_o = next(o for o in outs if o.shape[-1] == 4)
            reg_o = next((o for o in outs if o.shape[-1] == 6), None)
            cls_outs.append(cls_o)
            if reg_o is not None:
                reg_outs.append(reg_o)
        cls_t = torch.cat(cls_outs, dim=0)
        reg_t = torch.cat(reg_outs, dim=0) if reg_outs else None
        return cls_t, reg_t, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pte", default="data/deploy/boundary_et_int8.pte")
    ap.add_argument(
        "--src-ckpt",
        default="data/checkpoints/stage2_v45k_noaux_L8_d128_1ch_v56c.pt",
    )
    ap.add_argument("--input-norm", default="rank")
    ap.add_argument(
        "--out-json", default="data/deploy/boundary_et_f1.json",
    )
    args = ap.parse_args()

    pte_path = Path(args.pte)
    if not pte_path.exists():
        raise SystemExit(f".pte not found: {pte_path}")

    model = ExecuTorchTorchAdapter(pte_path)
    print(f"loaded .pte ({pte_path.stat().st_size / 1e6:.2f} MB) - scoring "
          f"on LUDB/ISP/QTDB with input_norm={args.input_norm}")
    res = score_all_1ch(
        model, device="cpu", mask_afib_p=False,
        compose_fn=None, input_norm=args.input_norm,
    )

    blob = torch.load(args.src_ckpt, map_location="cpu", weights_only=False)
    src_metrics = (blob.get("metrics") or {}).get("per_set_scores") or {}
    src_mean = (blob.get("metrics") or {}).get("score")

    pte_mean = (float(np.mean(list(res.values()))) if isinstance(res, dict)
                else None)

    print("== ExecuTorch F1 ==")
    print(json.dumps(res, indent=2, default=float))
    print(f"\n== Δ vs torch fp32 ({Path(args.src_ckpt).name}) ==")
    print(f"torch mean : {src_mean}")
    print(f"pte mean   : {pte_mean}")
    if src_mean is not None and pte_mean is not None:
        print(f"Δ mean     : {pte_mean - src_mean:+.5f}")

    out = {
        "pte": str(pte_path),
        "pte_size_mb": pte_path.stat().st_size / 1e6,
        "pte_f1": res,
        "torch_per_set": src_metrics,
        "torch_mean": src_mean,
        "delta_mean": (
            (pte_mean - src_mean)
            if (src_mean is not None and pte_mean is not None)
            else None
        ),
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nsaved -> {args.out_json}")


if __name__ == "__main__":
    main()
