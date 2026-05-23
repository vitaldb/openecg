"""Export 1-channel boundary checkpoint → ExecuTorch (.pte) format.

ExecuTorch is Meta's PyTorch-native edge runtime. Compared to TFLite, it
keeps the model in a PyTorch graph all the way to the edge — no
MLIR/stablehlo conversion pass — which simplifies debugging and avoids
the kind of op-level incompatibility we hit when trying to quantize the
boundary transformer via litert-torch.

Pipeline:
  1. ``torch.export.export`` to capture the FX graph.
  2. (Optional) ``prepare_pt2e`` + calibration + ``convert_pt2e`` with
     ``XNNPACKQuantizer`` for int8 quantization on CPU.
  3. ``to_edge`` + ``to_executorch`` to serialize.

The exported ``.pte`` is loaded via ``executorch.runtime`` (Python) or
the C++ runtime on mobile / embedded targets. Both use the same file
format.

Run::

    python -m scripts.export_boundary_executorch \\
        --ckpt data/checkpoints/stage2_v45k_noaux_L8_d128_1ch_v56c.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from openecg.deploy import DEFAULT_CKPT, WINDOW_SAMPLES, _wrap_cls_reg_only
from openecg.stage2.model import load_model_from_ckpt


def export_pte(
    ckpt_path: str | Path,
    out_path: str | Path,
    quantize: str = "int8",
) -> dict:
    """Convert a 1-channel boundary torch checkpoint to ExecuTorch .pte.

    Parameters
    ----------
    ckpt_path : path to a 1-channel boundary checkpoint.
    out_path : destination ``.pte`` file path.
    quantize : ``"fp32"`` or ``"int8"`` (XNNPACK dynamic). fp16 is not
        a standard ExecuTorch backend recipe; skip.
    """
    from executorch.exir import to_edge_transform_and_lower
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackDynamicallyQuantizedPartitioner,
        XnnpackPartitioner,
        XnnpackQuantizedPartitioner,
    )

    model, blob = load_model_from_ckpt(str(ckpt_path), device="cpu")
    model.train(False)
    wrapped = _wrap_cls_reg_only(model)
    wrapped.train(False)

    sample = torch.zeros(1, WINDOW_SAMPLES, dtype=torch.float32)

    if quantize == "int8":
        # ExecuTorch's XNNPACK quantizer uses the same PT2E flow as
        # everywhere else. Apply BEFORE ``torch.export`` capture so the
        # exported graph has int8 ops baked in.
        try:
            from torch.ao.quantization.quantize_pt2e import (
                convert_pt2e, prepare_pt2e,
            )
        except ModuleNotFoundError:
            from torchao.quantization.pt2e.quantize_pt2e import (
                convert_pt2e, prepare_pt2e,
            )
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer, get_symmetric_quantization_config,
        )
        try:
            from torch.export import export_for_training
        except ImportError:
            from torch.export import export as export_for_training

        # Use STATIC int8 (is_dynamic=False) — the XNNPACK ExecuTorch
        # pipeline currently only fully lowers static-quantized graphs;
        # the dynamic recipe leaves choose_qparams/quantize/dequantize
        # ops in portable kernels which lack out_variants and crash the
        # ``to_executorch()`` lowering pass.
        with torch.no_grad():
            exported = export_for_training(wrapped, (sample,)).module()
            quantizer = XNNPACKQuantizer().set_global(
                get_symmetric_quantization_config(is_dynamic=False)
            )
            prepared = prepare_pt2e(exported, quantizer)
            # Calibration pass: a few synthetic windows are enough to
            # populate per-tensor activation ranges.
            rng = np.random.default_rng(0)
            for _ in range(8):
                prepared(torch.from_numpy(
                    rng.standard_normal((1, WINDOW_SAMPLES)).astype(np.float32)))
            quantized = convert_pt2e(prepared)
        captured = torch.export.export(quantized, (sample,))
    elif quantize == "fp32":
        captured = torch.export.export(wrapped, (sample,))
    else:
        raise ValueError(f"unknown quantize={quantize!r}; pick fp32/int8")

    # Static-quant graphs use XnnpackQuantizedPartitioner (folds
    # quantize/dequantize stubs into the delegated XNNPACK region).
    partitioners = (
        [XnnpackQuantizedPartitioner()] if quantize == "int8"
        else [XnnpackPartitioner()]
    )
    edge = to_edge_transform_and_lower(captured, partitioner=partitioners)
    et_program = edge.to_executorch()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(et_program.buffer)

    return {
        "params": sum(p.numel() for p in model.parameters()),
        "pte_size_mb": out_path.stat().st_size / 1e6,
        "model_config": dict(blob.get("model_config") or {}),
        "best_metrics": dict(blob.get("metrics") or {}),
        "quantize": quantize,
    }


def _bench_latency(pte_path: Path, n_iter: int = 50) -> float:
    """Return ms/window over ``n_iter`` synthetic inferences."""
    from executorch.runtime import Runtime

    rt = Runtime.get()
    program = rt.load_program(pte_path)
    method = program.load_method("forward")
    rng = np.random.default_rng(1)
    warmup = torch.from_numpy(
        rng.standard_normal((1, WINDOW_SAMPLES)).astype(np.float32))
    bench = torch.from_numpy(
        rng.standard_normal((1, WINDOW_SAMPLES)).astype(np.float32))
    for _ in range(3):
        method.execute([warmup])
    t0 = time.perf_counter()
    for _ in range(n_iter):
        method.execute([bench])
    return (time.perf_counter() - t0) * 1000.0 / n_iter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--out-dir", default="data/deploy")
    ap.add_argument("--name-stem", default="boundary_et")
    ap.add_argument(
        "--variants", nargs="+", default=["fp32", "int8"],
        choices=["fp32", "int8"],
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []
    print(f"== boundary ExecuTorch export from {args.ckpt} ==")
    for q in args.variants:
        out_path = out_dir / f"{args.name_stem}_{q}.pte"
        info = export_pte(args.ckpt, out_path, quantize=q)
        ms = _bench_latency(out_path)
        info["ms_per_window"] = ms
        print(f"[{q:5s}] {out_path}  {info['pte_size_mb']:.2f} MB  "
              f"{ms:.2f} ms/win")
        summary.append(info)

    summary_path = out_dir / f"{args.name_stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved -> {summary_path}")


if __name__ == "__main__":
    main()
