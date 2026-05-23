"""Export 1-channel boundary checkpoint → TFLite (fp32 / fp16 / int8).

CLI wrapper around :func:`openecg.deploy.export_boundary_tflite`. The
library function does the actual conversion; this script orchestrates
"export all three variants + report size + sanity-check latency".

Quantization recap (post-training, no fine-tune):
  * **fp32**: full-precision baseline (~4 MB).
  * **fp16**: half-precision weights, fp32 activations (~2 MB).
  * **int8**: PT2E dynamic-range quantization on Conv1d + MatMul
    (~1.1 MB). Lossless on the v55a / v56c boundary task; matches
    torch fp32 eval within sub-0.001 F1 noise.

Default checkpoint: v56c (current SOTA — α=0.9 ep10, bf1=0.9299).

Run::

    python -m scripts.export_boundary_tflite \\
        --ckpt data/checkpoints/stage2_v45k_noaux_L8_d128_1ch_v56c.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from openecg.deploy import DEFAULT_CKPT


def _bench_latency(tflite_path: Path, n_iter: int = 50) -> float:
    """Return ms/window over ``n_iter`` synthetic inferences."""
    from openecg.deploy import Inference, WINDOW_SAMPLES
    det = Inference(str(tflite_path))
    rng = np.random.default_rng(1)
    warmup = rng.standard_normal(WINDOW_SAMPLES).astype(np.float32)
    bench = rng.standard_normal(WINDOW_SAMPLES).astype(np.float32)
    for _ in range(3):
        det.forward_window(warmup)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        det.forward_window(bench)
    return (time.perf_counter() - t0) * 1000.0 / n_iter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--out-dir", default="data/deploy")
    ap.add_argument("--name-stem", default="boundary",
                    help="prefix for output files: <stem>_<variant>.tflite")
    ap.add_argument(
        "--variants", nargs="+", default=["fp32", "int8"],
        choices=["fp32", "fp16", "int8"],
        help="fp16 currently unsupported (ai_edge_quantizer API churn)",
    )
    args = ap.parse_args()

    from openecg.deploy import export_boundary_tflite

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []
    print(f"== boundary TFLite export from {args.ckpt} ==")
    for q in args.variants:
        out_path = out_dir / f"{args.name_stem}_{q}.tflite"
        info = export_boundary_tflite(args.ckpt, out_path, quantize=q)
        ms = _bench_latency(out_path)
        info["ms_per_window"] = ms
        print(f"[{q:5s}] {out_path}  {info['tflite_size_mb']:.2f} MB  {ms:.2f} ms/win")
        summary.append(info)

    summary_path = out_dir / f"{args.name_stem}_tflite_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved -> {summary_path}")


if __name__ == "__main__":
    main()
