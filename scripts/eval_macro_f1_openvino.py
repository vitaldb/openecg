#!/usr/bin/env python3
"""Martinez boundary macro-F1 on LUDB + ISP + QTDB for several v56c runtimes.

Same protocol as ``scripts/benchmark_v56c.py`` and the v56c table in
``openecg/models/v56c_MODEL_CARD.md`` (lead II for LUDB val / ISP test; first
lead for the QTDB T-subset; repo splits; Martinez 2004 tolerances; reg-head
refinement). Every backend sees the same windows and the same decoder, so the
per-corpus numbers are directly comparable.

Backends (``--backends``, comma-separated):

* ``tflite``          bundled ``boundary_int8.tflite`` (TFLite / LiteRT) — reference
* ``ort``             ``v56c_fp32.onnx`` on ONNX Runtime CPU
* ``openvino:DEV``    ``v56c_fp32.onnx`` on OpenVINO device ``DEV`` (CPU, GPU, NPU, ...)
* ``torch:DEV``       ``v56c.pt`` on PyTorch device ``DEV`` (cpu, xpu, ...)

Besides macro-F1 it reports, per backend and corpus, frame-argmax agreement with
``tflite`` (when it is in the list) and mean inference time per window
(``ms_per_window``; model call only, batch 1, no pre/post-processing).

Requires::

    export OPENECG_LUDB_ZIP=... OPENECG_LUDB_CACHE=...
    export OPENECG_ISP_ZIP=...  OPENECG_ISP_CACHE=...
    export OPENECG_QTDB_CACHE=...                       # or OPENECG_QTDB_ZIP

Run::

    python -m scripts.eval_macro_f1_openvino --leads ii \
        --backends tflite,ort,openvino:CPU,openvino:GPU,openvino:NPU \
        --onnx data/deploy/v56c_fp32.onnx
    python -m scripts.eval_macro_f1_openvino --backends tflite,torch:xpu \
        --ckpt data/checkpoints/v56c.pt
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from collections import defaultdict
from pathlib import Path

import numpy as np

from openecg import ludb
from openecg.deploy import Inference, WINDOW_SAMPLES, preprocess_window
from openecg.stage2.evaluate import (
    BOUNDARY_KEYS,
    MARTINEZ_TOLERANCE_MS,
    signed_boundary_metrics,
)
from openecg.stage2.infer import (
    apply_reg_to_boundaries,
    extract_boundaries,
    post_process_frames,
)
from openecg.stage2.multi_dataset import _decimate_to_250

ROOT = Path(__file__).resolve().parents[1]
FS = 250
FRAME_MS = 20


def _boundaries_from_logits(cls_logits: np.ndarray, reg_offsets: np.ndarray | None):
    frames = cls_logits.argmax(-1).astype(np.uint8)
    pp = post_process_frames(frames, frame_ms=FRAME_MS)
    bds = extract_boundaries(pp, fs=FS, frame_ms=FRAME_MS)
    if reg_offsets is not None:
        bds = apply_reg_to_boundaries(
            bds, reg_offsets, samples_per_frame=5, max_window=WINDOW_SAMPLES,
        )
    return bds


class Backend:
    """``sig_250 -> boundaries`` for one runtime, recording frame argmax and the
    model-call time of every window it scores."""

    def __init__(self, name: str, forward: Callable[[np.ndarray], tuple], sync=None):
        self.name = name
        self._forward = forward          # rank-normalized (2500,) -> (cls, reg)
        self._sync = sync
        self.frames: list[np.ndarray] = []
        self.times: list[float] = []

    def reset(self) -> None:
        self.frames, self.times = [], []

    def __call__(self, sig_250: np.ndarray):
        x = preprocess_window(sig_250).astype(np.float32)
        t0 = time.perf_counter()
        cls, reg = self._forward(x)
        if self._sync is not None:
            self._sync()
        self.times.append(time.perf_counter() - t0)
        cls = np.asarray(cls)
        self.frames.append(cls.argmax(-1).astype(np.uint8))
        return _boundaries_from_logits(cls, None if reg is None else np.asarray(reg))


def _make_backend(spec: str, onnx_path: str | None, ckpt: str | None) -> tuple[Backend, dict]:
    kind, _, dev = spec.partition(":")
    kind = kind.lower()
    if kind == "tflite":
        det = Inference()

        def fwd(x):
            det._interp.set_tensor(det._input_detail["index"], x[None, ...])
            det._interp.invoke()
            cls = det._interp.get_tensor(det._cls_detail["index"])[0]
            reg = (det._interp.get_tensor(det._reg_detail["index"])[0]
                   if det._reg_detail else None)
            return cls, reg
        return Backend("tflite", fwd), {"model": "openecg/models/boundary_int8.tflite"}
    if kind == "ort":
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        def fwd(x):
            c, r = sess.run(["cls_logits", "reg_offsets"], {"signal": x[None, :]})
            return c[0], r[0]
        return Backend("ort", fwd), {"model": str(onnx_path), "onnxruntime": ort.__version__}
    if kind == "openvino":
        import openvino as ov

        from openecg.openvino_backend import OpenVINOBoundary, device_full_names

        ovb = OpenVINOBoundary(onnx_path, device=dev or "CPU")
        name = f"openvino_{ovb.device}"
        return Backend(name, ovb.forward_preprocessed), {
            "model": ovb.path, "config": ovb.config, "openvino": ov.get_version(),
            "device_name": device_full_names().get(ovb.device.split(".")[0], ""),
        }
    if kind == "torch":
        import torch

        from openecg.deploy import _wrap_cls_reg_only
        from scripts.export_boundary_onnx import _load

        dev = dev or "cpu"
        tmp = Path(ckpt).resolve().parent
        model, _ = _load(ckpt, tmp)
        (tmp / "_stripped.pt").unlink(missing_ok=True)
        wrap = _wrap_cls_reg_only(model).eval().to(dev)
        sync = getattr(torch, dev.split(":")[0], None)
        sync = getattr(sync, "synchronize", None) if dev != "cpu" else None

        def fwd(x):
            with torch.no_grad():
                c, r = wrap(torch.from_numpy(x[None, :]).to(dev))
            return c[0].float().cpu().numpy(), r[0].float().cpu().numpy()
        meta = {"model": str(ckpt), "torch": torch.__version__}
        if dev.startswith("xpu"):
            meta["device_name"] = torch.xpu.get_device_name(0)
        return Backend(f"torch_{dev}", fwd, sync), meta
    raise ValueError(f"unknown backend {spec!r}")


def _macro_f1(bp: dict, bt: dict) -> dict:
    per = {}
    f1s = []
    for k in BOUNDARY_KEYS:
        m = signed_boundary_metrics(bp.get(k, []), bt.get(k, []), MARTINEZ_TOLERANCE_MS[k], fs=FS)
        per[k] = m
        if not np.isnan(m["f1"]):
            f1s.append(m["f1"])
    return {
        "macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
        "per_boundary": {
            k: {
                kk: float(vv) if isinstance(vv, (float, np.floating, int, np.integer)) else vv
                for kk, vv in v.items()
                if kk in ("f1", "sens", "ppv", "n_true", "n_pred", "median_abs_ms")
            }
            for k, v in per.items()
        },
    }


def eval_ludb(predict_fn, leads_subset, edge_margin_ms=100):
    from openecg.stage2.dataset import LUDBFrameDataset

    rec_ids = ludb.load_split()["val"]
    ds = LUDBFrameDataset(rec_ids)
    bp, bt = defaultdict(list), defaultdict(list)
    cum = 0
    n = 0
    margin_250 = int(round(edge_margin_ms * FS / 1000.0))
    t0 = time.time()
    for idx in range(len(ds)):
        rid, lead = ds.items[idx]
        if leads_subset and lead not in leads_subset:
            continue
        sig_250, _lead_idx, _ = ds.cache[(rid, lead)]
        sig_250 = sig_250[:WINDOW_SAMPLES]
        if len(sig_250) < WINDOW_SAMPLES:
            continue
        sig_raw = sig_250.astype(np.float32)
        rng = ludb.labeled_range(rid, lead)
        if rng is None:
            continue
        lo_250 = max(0, rng[0] // 2 - margin_250)
        hi_250 = min(WINDOW_SAMPLES, rng[1] // 2 + margin_250 + 1)
        preds = predict_fn(sig_raw)
        for k, vs in preds.items():
            for s in vs:
                if lo_250 <= s < hi_250:
                    bp[k].append(int(s) + cum)
        try:
            gt = ludb.load_annotations(rid, lead)
            for k, v in gt.items():
                if k.endswith("_on") or k.endswith("_off"):
                    for s in v:
                        s250 = int(s // 2)
                        if lo_250 <= s250 < hi_250:
                            bt[k].append(s250 + cum)
        except Exception:
            pass
        cum += WINDOW_SAMPLES
        n += 1
    return _macro_f1(bp, bt) | {"n_windows": n, "elapsed_s": time.time() - t0}


def eval_isp(predict_fn, leads_subset):
    """ISP test, native 1000 Hz → 250 Hz (same as scripts/benchmark_v56c.py)."""
    try:
        from openecg import isp
        isp.ensure_extracted()
    except Exception as e:
        return {"skipped": True, "reason": str(e)}

    from openecg import isp

    # README / benchmark_v56c use load_split()['test'] (72), not eval_test_records()
    rec_ids = isp.load_split()["test"]
    bp, bt = defaultdict(list), defaultdict(list)
    cum = 0
    n = 0
    t0 = time.time()
    for rid in rec_ids:
        try:
            record = isp.load_record(rid, split="test")
            ann = isp.load_annotations_as_super(rid, split="test")
        except Exception:
            continue
        for lead_idx, lead in enumerate(isp.LEADS_12):
            if leads_subset and lead not in leads_subset:
                continue
            sig_1000 = record[lead]
            sig_250 = _decimate_to_250(sig_1000, 1000)
            if len(sig_250) < WINDOW_SAMPLES:
                pad = np.zeros(WINDOW_SAMPLES - len(sig_250), dtype=sig_250.dtype)
                sig_250 = np.concatenate([sig_250, pad])
            sig_250 = sig_250[:WINDOW_SAMPLES].astype(np.float32)
            preds = predict_fn(sig_250)
            for k, vs in preds.items():
                for s in vs:
                    bp[k].append(int(s) + cum)
            for k, v in ann.items():
                if k.endswith("_on") or k.endswith("_off"):
                    for s in v:
                        s250 = int(s // 4)
                        if 0 <= s250 < WINDOW_SAMPLES:
                            bt[k].append(s250 + cum)
            cum += WINDOW_SAMPLES
            n += 1
    if n == 0:
        return {"skipped": True, "reason": "no ISP windows scored"}
    return _macro_f1(bp, bt) | {"n_windows": n, "elapsed_s": time.time() - t0}


def eval_qtdb(predict_fn):
    try:
        from openecg import qtdb
        qtdb.ensure_extracted()
    except Exception as e:
        return {"skipped": True, "reason": str(e)}

    from openecg import qtdb

    rids = []
    for rid in qtdb.records_with_q1c():
        ann = qtdb.load_q1c(rid)
        win = qtdb.annotated_window(ann, window_samples=WINDOW_SAMPLES, fs=FS)
        if win is None:
            continue
        start, end = win
        n_q = sum(1 for s in ann["qrs_on"] if start <= s < end)
        n_t = sum(1 for s in ann["t_on"] if start <= s < end)
        if n_q > 0 and n_t / n_q >= 0.8:
            rids.append(rid)

    bp, bt = defaultdict(list), defaultdict(list)
    cum = 0
    n = 0
    t0 = time.time()
    for rid in rids:
        try:
            rec = qtdb.load_record(rid)
            lead = next(iter(rec))
            sig = rec[lead]
            if len(sig) < WINDOW_SAMPLES:
                continue
            ann = qtdb.load_q1c(rid)
            win = qtdb.annotated_window(ann, window_samples=WINDOW_SAMPLES, fs=FS)
            if win is None:
                continue
            start, end = win
            if end > len(sig):
                end = len(sig)
                start = max(0, end - WINDOW_SAMPLES)
            sig_250 = sig[start:start + WINDOW_SAMPLES].astype(np.float32)
            if len(sig_250) < WINDOW_SAMPLES:
                continue
            preds = predict_fn(sig_250)
            for k, vs in preds.items():
                for s in vs:
                    bp[k].append(int(s) + cum)
            for k, v in ann.items():
                if k.endswith("_on") or k.endswith("_off"):
                    for s in v:
                        if start <= s < start + WINDOW_SAMPLES:
                            bt[k].append(int(s - start) + cum)
            cum += WINDOW_SAMPLES
            n += 1
        except Exception:
            continue
    if n == 0:
        return {"skipped": True, "reason": "no QTDB windows scored"}
    return _macro_f1(bp, bt) | {"n_windows": n, "elapsed_s": time.time() - t0, "n_t_subset": len(rids)}


def _mean_over(report: dict, backend: str) -> dict:
    """Unweighted mean of per-corpus macro-F1 for corpora that actually ran."""
    vals, present = [], []
    for corpus in ("ludb", "isp", "qtdb"):
        block = report.get("corpora", {}).get(corpus, {}).get(backend)
        if not block or block.get("skipped"):
            continue
        f1 = block.get("macro_f1")
        if f1 is None or (isinstance(f1, float) and np.isnan(f1)):
            continue
        vals.append(float(f1))
        present.append(corpus)
    return {"macro_f1_mean": float(np.mean(vals)) if vals else float("nan"),
            "n_corpora": len(vals), "corpora": present}


def _ensure_env() -> None:
    if "OPENECG_LUDB_ZIP" not in os.environ:
        cand = ROOT / "data/physionet/ludb-1.0.1.zip"
        if cand.exists():
            os.environ["OPENECG_LUDB_ZIP"] = str(cand)
            os.environ.setdefault("OPENECG_LUDB_CACHE", str(ROOT / "data/physionet/ludb_cache"))
    if "OPENECG_ISP_ZIP" not in os.environ:
        cand = ROOT / "data/physionet/isp_delineation_dataset.zip"
        if cand.exists():
            os.environ["OPENECG_ISP_ZIP"] = str(cand)
            os.environ.setdefault("OPENECG_ISP_CACHE", str(ROOT / "data/physionet/isp_cache"))
    if "OPENECG_QTDB_CACHE" not in os.environ:
        cand = ROOT / "data/physionet/qtdb_cache"
        if cand.exists():
            os.environ["OPENECG_QTDB_CACHE"] = str(cand)
    if "OPENECG_QTDB_ZIP" not in os.environ:
        cand = ROOT / "data/physionet/qt-database-1.0.0.zip"
        if cand.exists() and cand.stat().st_size > 80_000_000:
            os.environ["OPENECG_QTDB_ZIP"] = str(cand)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="tflite,openvino:CPU",
                    help="comma-separated: tflite, ort, openvino:DEV, torch:DEV")
    ap.add_argument("--onnx", default=None,
                    help="v56c_fp32.onnx (default: openecg.openvino_backend.boundary_onnx_path())")
    ap.add_argument("--ckpt", default="data/checkpoints/v56c.pt", help="v56c.pt for torch:DEV")
    ap.add_argument("--leads", default="ii", help="'ii' or 'all' (LUDB/ISP)")
    ap.add_argument("--out", type=Path, default=ROOT / "out/macro_f1_v56c_backends.json")
    ap.add_argument("--skip-isp", action="store_true")
    ap.add_argument("--skip-qtdb", action="store_true")
    args = ap.parse_args()

    leads = None if args.leads == "all" else {args.leads.lower()}
    _ensure_env()
    specs = [b.strip() for b in args.backends.split(",") if b.strip()]
    if args.onnx is None and any(s.split(":")[0] in ("ort", "openvino") for s in specs):
        from openecg.openvino_backend import boundary_onnx_path
        args.onnx = str(boundary_onnx_path())

    backends, meta = [], {}
    for spec in specs:
        be, m = _make_backend(spec, args.onnx, args.ckpt)
        backends.append(be)
        meta[be.name] = m

    report = {"leads": args.leads, "backends": meta, "corpora": {}}
    runs = [("ludb", lambda f: eval_ludb(f, leads))]
    if not args.skip_isp:
        runs.append(("isp", lambda f: eval_isp(f, leads)))
    if not args.skip_qtdb:
        runs.append(("qtdb", eval_qtdb))

    for corpus, fn in runs:
        block = report["corpora"].setdefault(corpus, {})
        for be in backends:
            be.reset()
            res = fn(be)
            if not res.get("skipped"):
                t = np.asarray(be.times[1:] or be.times) * 1e3   # drop the first call
                res["ms_per_window"] = {"mean": float(t.mean()), "p50": float(np.median(t)),
                                        "p90": float(np.percentile(t, 90))}
                res["_frames"] = be.frames
            block[be.name] = res
            print(f"{corpus:5s} {be.name:22s} macro_f1={res.get('macro_f1', res)}  "
                  f"n={res.get('n_windows')}", flush=True)
        ref = block.get("tflite", {}).get("_frames")
        for name, res in block.items():
            fr = res.pop("_frames", None)
            if ref is not None and fr is not None and len(fr) == len(ref):
                a = np.concatenate(ref)
                b = np.concatenate(fr)
                res["argmax_agreement_vs_tflite"] = float((a == b).mean())

    for be in backends:
        report[f"mean_{be.name}"] = _mean_over(report, be.name)
        print(f"MEAN {be.name:22s} {report[f'mean_{be.name}']['macro_f1_mean']:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
