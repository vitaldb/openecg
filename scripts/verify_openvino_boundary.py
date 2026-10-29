"""Parity check: OpenVINO vs ONNX Runtime on the v56c fp32 ONNX.

Companion to ``scripts/export_boundary_onnx.py --verify``. Same windows (real
LUDB windows, 500 Hz cache decimated to 250 Hz, rank-normalized), same report:
max |Δ| on ``cls_logits`` / ``reg_offsets``, frame argmax agreement, and a
``PARITY: OK / FAIL`` line. The reference is ONNX Runtime ``CPUExecutionProvider``
on the same ``v56c_fp32.onnx`` file; OpenVINO compiles that file unchanged on
``--device``.

It also reports boundary agreement: both outputs go through the same decoder
(``post_process_frames`` → ``extract_boundaries`` → ``apply_reg_to_boundaries``)
and the boundary lists are compared per window (exact match, and max |shift|
in samples for windows with the same boundary count).

Run::

    python -m scripts.verify_openvino_boundary \\
        --onnx data/deploy/v56c_fp32.onnx \\
        --verify data/cache/v57_500hz/ludb_val.npz --device CPU

Without the 500 Hz cache, ``--ludb-val`` builds the same kind of windows from
the LUDB val split (``OPENECG_LUDB_ZIP``; all 12 leads per record, split order)::

    python -m scripts.verify_openvino_boundary --ludb-val --device GPU

Default tolerances (1e-4 on both heads, 100% argmax agreement) are the ones
``export_boundary_onnx.py --verify`` uses. They apply as-is to devices that run
f32 (``CPU``, ``GPU`` with the backend's default ``INFERENCE_PRECISION_HINT=f32``).
``NPU`` computes in f16; pass ``--tol`` / ``--min-agree`` explicitly for it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from openecg.deploy import WINDOW_SAMPLES

FS = 250
FRAME_MS = 20


def _windows_250hz(npz_path: str, n: int = 64) -> np.ndarray:
    """Identical to ``scripts.export_boundary_onnx._windows_250hz`` (kept local so
    this check does not import torch)."""
    from openecg.dsp import rank_normalize

    sigs = np.load(npz_path)["signals"]
    out = []
    for i in range(min(n, len(sigs))):
        s = np.asarray(sigs[i], dtype=np.float32)[::2][:WINDOW_SAMPLES]
        if len(s) < WINDOW_SAMPLES:
            s = np.pad(s, (0, WINDOW_SAMPLES - len(s)))
        out.append(rank_normalize(s))
    return np.stack(out).astype(np.float32)


def _ludb_val_windows(n: int = 64) -> np.ndarray:
    """LUDB val split records (500 Hz, 10 s), every lead, decimated ``[::2]`` to
    250 Hz and rank-normalized — the same treatment as :func:`_windows_250hz`."""
    from openecg import ludb
    from openecg.dsp import rank_normalize

    out = []
    for rid in ludb.load_split()["val"]:
        rec = ludb.load_record(rid)
        for lead in ludb.LEADS_12:
            s = np.asarray(rec[lead], dtype=np.float32)[::2][:WINDOW_SAMPLES]
            if len(s) < WINDOW_SAMPLES:
                s = np.pad(s, (0, WINDOW_SAMPLES - len(s)))
            out.append(rank_normalize(s))
            if len(out) >= n:
                return np.stack(out).astype(np.float32)
    return np.stack(out).astype(np.float32)


def _boundaries(cls: np.ndarray, reg: np.ndarray) -> dict[str, list[int]]:
    from openecg.stage2.infer import (
        apply_reg_to_boundaries, extract_boundaries, post_process_frames,
    )

    pp = post_process_frames(cls.argmax(-1).astype(np.uint8), frame_ms=FRAME_MS)
    bds = extract_boundaries(pp, fs=FS, frame_ms=FRAME_MS)
    return apply_reg_to_boundaries(bds, reg, samples_per_frame=5,
                                   max_window=WINDOW_SAMPLES)


def _boundary_agreement(ref_cls, ref_reg, ov_cls, ov_reg) -> dict:
    n_exact = 0
    n_same_count = 0
    max_shift = 0
    n_ref = n_ov = 0
    for i in range(ref_cls.shape[0]):
        a = _boundaries(ref_cls[i], ref_reg[i])
        b = _boundaries(ov_cls[i], ov_reg[i])
        same = True
        same_count = True
        for k in sorted(set(a) | set(b)):
            va, vb = sorted(a.get(k, [])), sorted(b.get(k, []))
            n_ref += len(va)
            n_ov += len(vb)
            if va != vb:
                same = False
            if len(va) != len(vb):
                same_count = False
            elif va:
                max_shift = max(max_shift, int(np.max(np.abs(np.subtract(va, vb)))))
        n_exact += same
        n_same_count += same_count
    n = ref_cls.shape[0]
    return {
        "windows": n,
        "windows_identical": n_exact,
        "windows_same_count": n_same_count,
        "max_shift_samples": max_shift,
        "boundaries_ref": n_ref,
        "boundaries_openvino": n_ov,
    }


def verify(onnx_path: str, X: np.ndarray, device: str, config: dict | None,
           tol: float, min_agree: float) -> dict:
    import onnxruntime as ort

    from openecg.openvino_backend import OpenVINOBoundary, device_full_names

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    r_cls, r_reg = sess.run(["cls_logits", "reg_offsets"], {"signal": X})

    ovb = OpenVINOBoundary(onnx_path, device=device, config=config)
    o_cls = np.empty_like(r_cls)
    o_reg = np.empty_like(r_reg)
    for i in range(X.shape[0]):
        o_cls[i], o_reg[i] = ovb.forward_preprocessed(X[i])

    d_cls = float(np.abs(r_cls - o_cls).max())
    d_reg = float(np.abs(r_reg - o_reg).max())
    agree = float((r_cls.argmax(-1) == o_cls.argmax(-1)).mean())
    n_frames = r_cls.shape[0] * r_cls.shape[1]
    bnd = _boundary_agreement(r_cls, r_reg, o_cls, o_reg)
    full = device_full_names().get(ovb.device.split(".")[0], "")

    print(f"OpenVINO device    : {ovb.device} ({full})  config={ovb.config}")
    print(f"LUDB windows       : {X.shape}")
    print(f"max |diff| cls_logits : {d_cls:.3e}")
    print(f"max |diff| reg_offsets: {d_reg:.3e}")
    print(f"argmax agreement   : {agree * 100:.4f}%  ({int(round(agree * n_frames))}/{n_frames} frames)")
    print(f"boundaries         : {bnd['windows_identical']}/{bnd['windows']} windows identical, "
          f"max shift {bnd['max_shift_samples']} samples "
          f"({bnd['boundaries_openvino']} vs {bnd['boundaries_ref']} boundaries)")

    ok = d_cls < tol and d_reg < tol and agree >= min_agree
    print(f"PARITY (tol {tol:g}, min agree {min_agree:g}):", "OK" if ok else "FAIL")
    return {
        "device": ovb.device, "device_name": full, "config": ovb.config,
        "n_windows": int(X.shape[0]), "max_abs_cls": d_cls, "max_abs_reg": d_reg,
        "argmax_agreement": agree, "boundaries": bnd,
        "tol": tol, "min_agree": min_agree, "ok": bool(ok),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=None,
                    help="v56c_fp32.onnx (default: openecg.openvino_backend.boundary_onnx_path())")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--verify", metavar="LUDB_NPZ",
                     help="LUDB 500 Hz cache .npz (as for export_boundary_onnx.py --verify)")
    src.add_argument("--ludb-val", action="store_true",
                     help="build the windows from the LUDB val split instead (OPENECG_LUDB_ZIP)")
    ap.add_argument("--n", type=int, default=64, help="number of windows (default 64)")
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--config", default=None,
                    help='OpenVINO config as JSON, e.g. \'{"INFERENCE_PRECISION_HINT": "f16"}\'')
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--min-agree", type=float, default=1.0)
    ap.add_argument("--json", type=Path, default=None, help="write the result as JSON")
    args = ap.parse_args()

    if args.onnx is None:
        from openecg.openvino_backend import boundary_onnx_path
        args.onnx = str(boundary_onnx_path())
    X = _windows_250hz(args.verify, args.n) if args.verify else _ludb_val_windows(args.n)
    cfg = json.loads(args.config) if args.config else None
    res = verify(args.onnx, X, args.device, cfg, args.tol, args.min_agree)
    res["onnx"] = str(args.onnx)
    res["windows_source"] = args.verify or "ludb val split"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=2))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
