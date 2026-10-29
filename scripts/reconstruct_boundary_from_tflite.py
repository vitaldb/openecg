#!/usr/bin/env python3
"""Rebuild the v56c boundary detector in fp32 from ``boundary_int8.tflite``,
and cross-check the rebuild against the released fp32 checkpoint.

``boundary_int8.tflite`` stores int8 weights with per-channel scales, so its
dequantized weights are the trained weights up to int8 rounding. This script:

1. Instantiates ``vit_transformer_noaux_1ch`` (992,266 params).
2. Dequantizes every weight tensor of the TFLite file.
3. Maps the TFLite tensors onto the torch module. Nothing is fitted; three
   converter folds are undone:

   * ``patch_embed.bias`` was folded into the positional-encoding constant,
     so ``patch_embed.bias = 0`` and ``pos_enc[:500]`` takes the folded constant;
   * the pre-LN affine of every upper transformer layer was folded into the
     following ``in_proj`` / ``linear1`` weights and biases, so the upper
     ``norm1`` / ``norm2`` affine params become identity;
   * the lower LayerNorm constants are stored in reverse block order
     (block 0 uses MUL 77 / ADD 76 ... block 3 uses MUL 71 / ADD 70).

4. Checks that the rebuilt torch model reproduces the TFLite ``cls_logits``
   (``max |Δ| < 1e-3`` on any ``data/samples/*_10s_250hz.npy`` windows, and
   reported on the LUDB windows below).
5. Writes the rebuilt checkpoint and an fp32 ONNX export (opset 17, static
   ``1 × 2500``) under ``data/`` (not committed).

``--compare-ckpt`` then scores the rebuild against the released checkpoint
(``v56c.pt``, release ``boundary-v56c``):

* **weights** — the released weights are folded the same way the converter
  folds them, then compared tensor by tensor (max |Δ|, and max |Δ| relative to
  half an int8 step of that output channel, ``max|w_c| / 254``);
* **outputs** — torch fp32 (released) vs torch fp32 (rebuilt) vs the int8 TFLite
  on real LUDB windows (``--verify LUDB_NPZ`` or ``--ludb-val``).

Run (needs ``torch``, ``onnx`` and a TFLite interpreter such as ``ai-edge-litert``)::

    python -m scripts.reconstruct_boundary_from_tflite \\
        --compare-ckpt data/checkpoints/v56c.pt --ludb-val --json out/reconstruction.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# TFLite tensor index -> torch parameter, for boundary_int8.tflite (v56c).
_LOWER_CONV = [(51, 30), (49, 29), (48, 28), (47, 27)]          # (weight, bias), block 0..3
_ATTN_IN_W = {0: 25, 1: 19, 2: 13, 3: 7}
_ATTN_IN_B = {0: 26, 1: 20, 2: 14, 3: 8}
_ATTN_OUT_W = {0: 40, 1: 38, 2: 36, 3: 34}
_ATTN_OUT_B = {0: 24, 1: 18, 2: 12, 3: 6}
_LIN1_W = {0: 22, 1: 16, 2: 10, 3: 4}
_LIN1_B = {0: 23, 1: 17, 2: 11, 3: 5}
_LIN2_W = {0: 39, 1: 37, 2: 35, 3: 33}
_LIN2_B = {0: 21, 1: 15, 2: 9, 3: 3}


def _interpreter(path: Path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tflite_runtime.interpreter import Interpreter
    interp = Interpreter(model_path=str(path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    interp.set_tensor(inp["index"], np.zeros(inp["shape"], np.float32))
    interp.invoke()
    return interp


def _dequant(interp):
    details = {d["index"]: d for d in interp.get_tensor_details()}

    def get(idx: int) -> np.ndarray:
        d = details[idx]
        arr = interp.get_tensor(idx)
        q = d.get("quantization_parameters") or {}
        scales = np.asarray(q.get("scales", []), np.float32).reshape(-1)
        zps = np.asarray(q.get("zero_points", []), np.int32).reshape(-1)
        if arr.dtype == np.int8 and scales.size > 0:
            qdim = int(q.get("quantized_dimension", 0))
            if scales.size == 1:
                return (arr.astype(np.float32) - float(zps[0])) * float(scales[0])
            deq = np.moveaxis(arr.astype(np.float32), qdim, 0)
            for i in range(deq.shape[0]):
                deq[i] = (deq[i] - float(zps[i])) * float(scales[i])
            return np.moveaxis(deq, 0, qdim)
        return arr.astype(np.float32)

    return get


def build_from_tflite(tflite: Path):
    """Return the rebuilt torch model (eval mode)."""
    import torch

    from openecg.stage2.model_variants import FrameClassifierTransformerNoAux1Ch

    model = FrameClassifierTransformerNoAux1Ch(
        patch_size=5, n_leads=12, d_model=128, n_heads=4, n_layers=8, ff=256,
        n_classes=4, dropout=0.0, use_lead_emb=False, pos_type="sinusoidal",
        conv_stem=False, n_reg=6, mid_split=4, lower_kernel=7,
    ).eval()
    get = _dequant(_interpreter(tflite))
    sd = model.state_dict()

    def set_(key, arr):
        t = torch.from_numpy(np.ascontiguousarray(arr))
        assert sd[key].shape == t.shape, (key, sd[key].shape, t.shape)
        sd[key].copy_(t)

    set_("head.bias", get(2))
    set_("head.weight", get(32))
    set_("reg_head.bias", get(1))
    set_("reg_head.weight", get(31))
    set_("patch_embed.weight", get(41))
    sd["patch_embed.bias"].zero_()                       # folded into tensor 42
    sd["pos_enc"][:, :500, :].copy_(torch.from_numpy(get(42)))
    for i, (w_idx, b_idx) in enumerate(_LOWER_CONV):
        w = np.transpose(np.squeeze(get(w_idx), axis=2), (0, 2, 1))
        set_(f"lower_convs.{i}.weight", w)
        set_(f"lower_convs.{i}.bias", get(b_idx).reshape(128))
    for i in range(4):                                   # MUL (77-2i) = gamma, ADD (76-2i) = beta
        set_(f"lower_norms.{i}.weight", get(77 - 2 * i))
        set_(f"lower_norms.{i}.bias", get(76 - 2 * i))
    for L in range(4):
        p = f"upper_transformer.layers.{L}"
        set_(f"{p}.self_attn.in_proj_weight", get(_ATTN_IN_W[L]))
        set_(f"{p}.self_attn.in_proj_bias", get(_ATTN_IN_B[L]))
        set_(f"{p}.self_attn.out_proj.weight", get(_ATTN_OUT_W[L]))
        set_(f"{p}.self_attn.out_proj.bias", get(_ATTN_OUT_B[L]))
        set_(f"{p}.linear1.weight", get(_LIN1_W[L]))
        set_(f"{p}.linear1.bias", get(_LIN1_B[L]))
        set_(f"{p}.linear2.weight", get(_LIN2_W[L]))
        set_(f"{p}.linear2.bias", get(_LIN2_B[L]))
        for n in ("norm1", "norm2"):                     # folded into in_proj / linear1
            sd[f"{p}.{n}.weight"].fill_(1.0)
            sd[f"{p}.{n}.bias"].zero_()
    model.load_state_dict(sd)
    return model


def fold_like_converter(st: dict) -> dict:
    """Apply the TFLite converter's folds to a released state dict, so it can be
    compared tensor by tensor with the rebuild."""
    import torch

    st = {k: v.detach().clone().float() for k, v in st.items()}
    st["pos_enc"][:, :500, :] += st["patch_embed.bias"][None, None, :]
    st["patch_embed.bias"] = torch.zeros_like(st["patch_embed.bias"])
    for L in range(4):
        p = f"upper_transformer.layers.{L}"
        for norm, w, b in (("norm1", "self_attn.in_proj_weight", "self_attn.in_proj_bias"),
                           ("norm2", "linear1.weight", "linear1.bias")):
            g, beta = st[f"{p}.{norm}.weight"], st[f"{p}.{norm}.bias"]
            W = st[f"{p}.{w}"]
            st[f"{p}.{b}"] = st[f"{p}.{b}"] + W @ beta
            st[f"{p}.{w}"] = W * g[None, :]
            st[f"{p}.{norm}.weight"] = torch.ones_like(g)
            st[f"{p}.{norm}.bias"] = torch.zeros_like(beta)
    return st


def compare_weights(rebuilt: dict, released: dict) -> dict:
    rows = {}
    for k, r in rebuilt.items():
        if k not in released:
            continue
        a = r.detach().float().numpy()
        b = released[k].detach().float().numpy()
        if k == "pos_enc":
            a, b = a[:, :500], b[:, :500]
        d = np.abs(a - b)
        row = {"shape": list(a.shape), "max_abs": float(d.max()),
               "max_abs_ref": float(np.abs(b).max())}
        if a.ndim >= 2 and k != "pos_enc":
            # half an int8 step per output channel (symmetric per-channel int8)
            bc = np.abs(b).reshape(b.shape[0], -1).max(1) / 254.0
            dc = d.reshape(d.shape[0], -1).max(1)
            ok = bc > 0
            row["max_ratio_to_half_int8_step"] = float((dc[ok] / bc[ok]).max()) if ok.any() else 0.0
        rows[k] = row
    return rows


def _windows(args, n: int = 64) -> np.ndarray:
    from scripts.export_boundary_onnx import _windows_250hz

    if args.verify:
        return _windows_250hz(args.verify, n)
    from openecg import ludb
    from openecg.deploy import WINDOW_SAMPLES
    from openecg.dsp import rank_normalize

    out = []
    for rid in ludb.load_split()["val"]:
        rec = ludb.load_record(rid)
        for lead in ludb.LEADS_12:
            s = np.asarray(rec[lead], dtype=np.float32)[::2][:WINDOW_SAMPLES]
            out.append(rank_normalize(s))
            if len(out) >= n:
                return np.stack(out).astype(np.float32)
    return np.stack(out).astype(np.float32)


def _outputs_vs(a_cls, a_reg, b_cls, b_reg) -> dict:
    return {
        "max_abs_cls": float(np.abs(a_cls - b_cls).max()),
        "max_abs_reg": float(np.abs(a_reg - b_reg).max()),
        "argmax_agreement": float((a_cls.argmax(-1) == b_cls.argmax(-1)).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", default=str(ROOT / "openecg/models/boundary_int8.tflite"))
    ap.add_argument("--out-ckpt", default=str(ROOT / "data/checkpoints/boundary_v56c_from_tflite.pt"))
    ap.add_argument("--out-onnx", default=str(ROOT / "data/deploy/boundary_v56c_from_tflite.onnx"))
    ap.add_argument("--no-export", action="store_true", help="skip writing the .pt / .onnx")
    ap.add_argument("--compare-ckpt", default=None, help="released v56c.pt to compare against")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--verify", metavar="LUDB_NPZ", help="LUDB 500 Hz cache .npz")
    src.add_argument("--ludb-val", action="store_true", help="LUDB val split (OPENECG_LUDB_ZIP)")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    import torch

    from openecg.deploy import Inference, WINDOW_SAMPLES, _wrap_cls_reg_only, preprocess_window

    model = build_from_tflite(Path(args.tflite))
    det = Inference(args.tflite) if args.tflite else Inference()
    report: dict = {"tflite": args.tflite, "torch": torch.__version__}

    # 4. the rebuild must reproduce the TFLite logits
    for path in sorted((ROOT / "data" / "samples").glob("*_10s_250hz.npy")):
        sig = np.load(path)
        ref = det.forward_window(sig)
        with torch.no_grad():
            cls = _wrap_cls_reg_only(model)(torch.from_numpy(preprocess_window(sig)[None]))[0]
        d = float(np.abs(cls[0].numpy() - ref).max())
        print(f"{path.name}: max|rebuilt torch - tflite| cls = {d:.2e}")
        report.setdefault("samples_vs_tflite", {})[path.name] = d
        assert d < 1e-3, "TFLite mapping mismatch"

    # 5. write the rebuild
    if not args.no_export:
        ck = Path(args.out_ckpt)
        ck.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(),
                    "model_config": dict(model.model_config),
                    "note": "fp32 rebuild of boundary_int8.tflite (no fitting)"}, ck)
        onnx_path = Path(args.out_onnx)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(_wrap_cls_reg_only(model).eval(), torch.zeros(1, WINDOW_SAMPLES),
                          str(onnx_path), input_names=["signal"],
                          output_names=["cls_logits", "reg_offsets"],
                          opset_version=17, dynamo=False)
        print("wrote", ck, "and", onnx_path)

    if args.compare_ckpt:
        from scripts.export_boundary_onnx import _load

        tmp = Path(args.compare_ckpt).resolve().parent
        released, _ = _load(args.compare_ckpt, tmp)
        (tmp / "_stripped.pt").unlink(missing_ok=True)
        released.eval()
        rows = compare_weights(model.state_dict(), fold_like_converter(released.state_dict()))
        report["weights"] = rows
        worst = max(rows.items(), key=lambda kv: kv[1].get("max_ratio_to_half_int8_step", 0))
        print(f"weights: {len(rows)} tensors, worst max|diff|/(half int8 step) = "
              f"{worst[1]['max_ratio_to_half_int8_step']:.3f} ({worst[0]})")
        if args.verify or args.ludb_val:
            X = _windows(args)
            with torch.no_grad():
                r_cls, r_reg = (t.numpy() for t in _wrap_cls_reg_only(released)(torch.from_numpy(X)))
                b_cls, b_reg = (t.numpy() for t in _wrap_cls_reg_only(model)(torch.from_numpy(X)))
            t_cls = np.empty_like(r_cls)
            t_reg = np.empty_like(r_reg)
            for i in range(X.shape[0]):
                det._interp.set_tensor(det._input_detail["index"], X[i][None])
                det._interp.invoke()
                t_cls[i] = det._interp.get_tensor(det._cls_detail["index"])[0]
                t_reg[i] = det._interp.get_tensor(det._reg_detail["index"])[0]
            report["outputs"] = {
                "n_windows": int(X.shape[0]),
                "rebuilt_vs_released": _outputs_vs(b_cls, b_reg, r_cls, r_reg),
                "tflite_vs_released": _outputs_vs(t_cls, t_reg, r_cls, r_reg),
                "rebuilt_vs_tflite": _outputs_vs(b_cls, b_reg, t_cls, t_reg),
            }
            for k, v in report["outputs"].items():
                if isinstance(v, dict):
                    print(f"{k:22s} max|diff| cls {v['max_abs_cls']:.3e}  reg {v['max_abs_reg']:.3e}  "
                          f"argmax agree {v['argmax_agreement'] * 100:.4f}%")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
