"""F1 evaluation of an exported .tflite boundary model.

Wraps a `tflite-runtime` interpreter in a torch-call signature so the
existing ``score_all_1ch`` evaluator (which expects ``model(x, lead_id)
-> (cls_logits, reg_offsets, ...)``) can be reused unchanged. Runs
boundary-F1 over LUDB / ISP / QTDB and compares against the torch fp32
baseline stored in the source checkpoint.

Run::

    python -m scripts.eval_boundary_tflite \\
        --tflite data/deploy/boundary_int8.tflite \\
        --src-ckpt data/checkpoints/stage2_v45k_noaux_L8_d128_1ch_v56c.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.retrain_v40_common import score_all_1ch


class TFLiteTorchAdapter(torch.nn.Module):
    """Adapt a `.tflite` interpreter to the torch ``(x, lead_id)`` calling
    convention used by ``score_all_1ch``.

    The TFLite graph exported from a 1-channel boundary model takes input
    ``(1, 2500)`` and returns two outputs identified by their last-axis
    size: ``(1, 500, 4)`` cls_logits and ``(1, 500, 6)`` reg_offsets.
    """

    def __init__(self, tflite_path: str | Path, num_threads: int = 1):
        super().__init__()
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter
        self._interp = Interpreter(model_path=str(tflite_path),
                                    num_threads=num_threads)
        self._interp.allocate_tensors()
        self._inp = self._interp.get_input_details()[0]
        outs = self._interp.get_output_details()
        self._cls = next(o for o in outs if o["shape"][-1] == 4)
        self._reg = next((o for o in outs if o["shape"][-1] == 6), None)

    def forward(self, x: torch.Tensor, lead_id):
        # x: (B, T) or (B, 1, T) or (B, 2, T) — the score path passes (B, T)
        # after compose_fn for 1-ch contracts.
        arr = x.detach().cpu().numpy().astype(np.float32)
        if arr.ndim == 3:
            arr = arr[:, 0]                                # (B, T)
        # tflite int8 model is fixed batch=1; loop windows if B > 1.
        cls_outs = []
        reg_outs = []
        for b in range(arr.shape[0]):
            self._interp.set_tensor(self._inp["index"], arr[b:b+1])
            self._interp.invoke()
            cls_outs.append(self._interp.get_tensor(self._cls["index"]))
            if self._reg is not None:
                reg_outs.append(self._interp.get_tensor(self._reg["index"]))
        cls_t = torch.from_numpy(np.concatenate(cls_outs, axis=0))
        reg_t = (torch.from_numpy(np.concatenate(reg_outs, axis=0))
                 if reg_outs else None)
        return cls_t, reg_t, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", default="data/deploy/boundary_int8.tflite")
    ap.add_argument(
        "--src-ckpt",
        default="data/checkpoints/stage2_v45k_noaux_L8_d128_1ch_v56c.pt",
    )
    ap.add_argument("--input-norm", default="rank",
                    help="must match training-time normalization (v56c=rank)")
    ap.add_argument("--out-json",
                    default="data/deploy/boundary_tflite_f1.json")
    args = ap.parse_args()

    tflite_path = Path(args.tflite)
    if not tflite_path.exists():
        raise SystemExit(f"tflite model not found: {tflite_path}")

    # 1-channel deploy contract: ``compose_fn=None`` triggers the default
    # ``_compose_input_native(...)[0]`` path which yields raw (T,) at FS=250 —
    # exactly what the TFLite adapter expects.
    model = TFLiteTorchAdapter(tflite_path)
    print(f"loaded tflite ({tflite_path.stat().st_size / 1e6:.2f} MB) - scoring "
          f"on LUDB/ISP/QTDB with input_norm={args.input_norm}")
    res = score_all_1ch(
        model, device="cpu", mask_afib_p=False,
        compose_fn=None, input_norm=args.input_norm,
    )

    # Compare to the source checkpoint's recorded torch fp32 score.
    blob = torch.load(args.src_ckpt, map_location="cpu", weights_only=False)
    src_metrics = (blob.get("metrics") or {}).get("per_set_scores") or {}
    src_mean = (blob.get("metrics") or {}).get("score")
    tflite_mean = res.get("score") if isinstance(res, dict) else None
    if tflite_mean is None:
        # Some score_all_1ch variants return per-set dict; reconstruct mean.
        per = res if isinstance(res, dict) else {}
        if per:
            tflite_mean = float(np.mean(list(per.values())))

    print("== TFLite F1 ==")
    print(json.dumps(res, indent=2, default=float))
    print(f"\n== Δ vs torch fp32 ({Path(args.src_ckpt).name}) ==")
    print(f"torch mean : {src_mean}")
    print(f"tflite mean: {tflite_mean}")
    if src_mean is not None and tflite_mean is not None:
        print(f"Δ mean     : {tflite_mean - src_mean:+.5f}")

    out = {
        "tflite": str(tflite_path),
        "tflite_size_mb": tflite_path.stat().st_size / 1e6,
        "tflite_f1": res,
        "torch_per_set": src_metrics,
        "torch_mean": src_mean,
        "delta_mean": (
            (tflite_mean - src_mean)
            if (src_mean is not None and tflite_mean is not None)
            else None
        ),
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nsaved -> {args.out_json}")


if __name__ == "__main__":
    main()
