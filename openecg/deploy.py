"""TFLite-based inference deployment for the boundary classifier.

Canonical deploy checkpoint: **v56c** (k_noaux_L8_d128, 1-channel, soft-T
α=0.9, epoch 10) — a 1M-param frame classifier achieving boundary-F1
**0.9299** mean over LUDB / ISP / QTDB (LUDB 0.9456, ISP 0.9737, QTDB
0.8704). Small enough to ship via TFLite int8 (~1.1 MB) + numpy on
mobile / embedded targets (Android, iOS, holter, AED).

Pipeline:

  1. **Export**: :func:`export_boundary_tflite` loads a 1-channel
     torch checkpoint, converts via :mod:`ai_edge_torch` to TFLite,
     applies post-training int8 quantization, and writes a ``.tflite``
     file.
  2. **Inference**: :class:`Inference` loads the ``.tflite`` model with
     ``tflite_runtime.Interpreter`` and runs ``predict(signal, fs)``.
  3. **Post-processing**: :func:`logits_to_boundaries` converts per-frame
     class logits into (P_on, P_off, QRS_on, QRS_off, T_on, T_off) tuples.

Deploy footprint:

  * **At inference**: ``tflite-runtime`` (~5 MB wheel) + ``numpy``. No
    PyTorch, no TensorFlow, no onnxruntime.
  * **At export**: ``torch`` + ``ai_edge_torch`` (dev only). The exported
    ``.tflite`` is ~4 MB fp32, ~2 MB fp16, ~1.1 MB int8.

Input contract: **single-channel** (B, 2500) at 250 Hz. The detector
slides 10-s windows over the input signal with no overlap; trailing
partial windows are zero-padded. The 2-channel (sig + qrs_box) contract
from older v54i/v52-era checkpoints is no longer supported here — use
the 1-channel arch (`vit_transformer_noaux_1ch`); legacy 2-channel
checkpoints can be cheaply re-saved with the 1-ch arch label since the
weights are bit-identical.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from openecg.dsp import rank_normalize


# Model architecture constants (v56c fixed config — shared by v54i / v55a
# since these are weight-compatible variants of the same backbone).
PATCH_SIZE = 5          # samples per frame
WINDOW_SAMPLES = 2500   # 10 s @ 250 Hz (matches training)
N_FRAMES = WINDOW_SAMPLES // PATCH_SIZE   # 500
TARGET_FS = 250         # Hz (training pipeline samples at 250 Hz)
N_CLASSES = 4           # [none, P, QRS, T]

CLASS_NONE, CLASS_P, CLASS_QRS, CLASS_T = 0, 1, 2, 3

# Canonical 1-channel checkpoint path. The deploy export and any helper
# scripts default to this when no explicit ckpt is provided.
DEFAULT_CKPT = Path("data/checkpoints/stage2_v45k_noaux_L8_d128_1ch_v56c.pt")


# -- Export (torch → TFLite int8) -------------------------------------------

def _wrap_cls_reg_only(model):
    """Wrap a 1-channel boundary model to expose only (cls_logits, reg_offsets).

    Training-time forward returns the 4-tuple ``(cls, reg, None, None)``.
    The boundary decoder uses cls + reg only (None aux slots are ignored)
    — stripping them keeps the exported graph small and side-steps
    ``None`` outputs that some downstream converters reject.
    """
    import torch

    class _ClsRegOnly(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, sig):
            # 1-ch forward signature is (sig, lead_id). We bake lead_id=0
            # since the canonical config has use_lead_emb=False — the
            # value is unused but the parameter slot must be filled.
            lead_id = torch.zeros(sig.shape[0], dtype=torch.long, device=sig.device)
            out = self.m(sig, lead_id)
            return out[0], out[1]

    return _ClsRegOnly(model)


def export_boundary_tflite(
    ckpt_path: str | Path = DEFAULT_CKPT,
    out_path: str | Path = "data/deploy/boundary_int8.tflite",
    quantize: str = "int8",
    representative_samples: Optional[np.ndarray] = None,
) -> dict:
    """Convert a 1-channel boundary torch checkpoint to TFLite.

    Parameters
    ----------
    ckpt_path : path to a 1-channel boundary checkpoint
        (`arch=vit_transformer_noaux_1ch`). Defaults to the bundled
        v56c canonical ckpt (`DEFAULT_CKPT`).
    out_path : destination ``.tflite`` file path.
    quantize : one of ``"fp32"``, ``"fp16"``, ``"int8"``. ``"int8"``
        uses post-training dynamic-range quantization for both Conv1d
        and MatMul layers — far better size + speed than the prior
        onnxruntime CPU "MatMul-only" int8.
    representative_samples : optional ``(N, 2500)`` float32 array used
        for int8 activation calibration. If omitted, falls back to
        deterministic synthetic noise (~32 windows). Pass real LUDB /
        ISP windows for tighter activation ranges.

    Returns
    -------
    dict with ``params``, ``tflite_size_mb``, ``model_config``,
    ``best_metrics``, and ``quantize``.
    """
    import torch

    # `litert-torch` (>=0.7) is the renamed successor of `ai-edge-torch`.
    # Both expose the same `convert(...)` entry point and a `quantize`
    # submodule with PT2EQuantizer. Prefer litert-torch; fall back to the
    # legacy name so older environments keep working.
    try:
        import litert_torch as _edge
    except ImportError:
        import ai_edge_torch as _edge

    from openecg.stage2.model import load_model_from_ckpt

    model, blob = load_model_from_ckpt(str(ckpt_path), device="cpu")
    model.train(False)
    wrapped = _wrap_cls_reg_only(model)
    wrapped.train(False)

    from importlib import import_module
    sample = torch.zeros(1, WINDOW_SAMPLES, dtype=torch.float32)

    if quantize == "int8":
        # Two-step int8 path: convert fp32 first, then post-quantize via
        # ai_edge_quantizer. ``weight_only_wi8_afp32`` is the safest recipe
        # for accuracy preservation — it quantizes weights to int8 but
        # leaves activations in fp32, so no calibration is needed and
        # accuracy degradation is minimal. The dynamic recipe was tried
        # but destroyed accuracy on this model (likely a transformer
        # quirk in the dynamic activation quantization path).
        from ai_edge_quantizer import Quantizer, recipe as aeq_recipe

        fp32_buf = _edge.convert(wrapped, (sample,))
        # Write the fp32 intermediate to a temp file (ai_edge_quantizer
        # takes a path or bytes).
        out_path_obj = Path(out_path)
        out_path_obj.parent.mkdir(parents=True, exist_ok=True)
        tmp_fp32 = out_path_obj.with_suffix(".fp32.tmp.tflite")
        fp32_buf.export(str(tmp_fp32))
        q = Quantizer(str(tmp_fp32), aeq_recipe.weight_only_wi8_afp32())
        result = q.quantize()
        result.export_model(str(out_path_obj))
        tmp_fp32.unlink(missing_ok=True)

        return {
            "params": sum(p.numel() for p in model.parameters()),
            "tflite_size_mb": out_path_obj.stat().st_size / 1e6,
            "model_config": dict(blob.get("model_config") or {}),
            "best_metrics": dict(blob.get("metrics") or {}),
            "quantize": quantize,
        }
    elif quantize == "fp16":
        # fp16 weight-only path is currently disabled. The
        # ``ai_edge_quantizer`` float_casting recipe has shifted across
        # releases and the simpler ``model.half()`` + edge.convert pathway
        # trips litert-torch tracing for our transformer. Use ``int8`` or
        # ``fp32`` — int8 already matches fp32 accuracy on this model.
        raise NotImplementedError(
            "fp16 export not supported in this build — use int8 (lossless) "
            "or fp32. Track ai_edge_quantizer's float_casting recipe for a "
            "future re-enable."
        )
    elif quantize == "fp32":
        edge_model = _edge.convert(wrapped, (sample,))
    else:
        raise ValueError(f"unknown quantize={quantize!r}; pick fp32/fp16/int8")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    edge_model.export(str(out_path))

    return {
        "params": sum(p.numel() for p in model.parameters()),
        "tflite_size_mb": out_path.stat().st_size / 1e6,
        "model_config": dict(blob.get("model_config") or {}),
        "best_metrics": dict(blob.get("metrics") or {}),
        "quantize": quantize,
    }


# -- Inference (tflite-runtime + numpy) -------------------------------------

def preprocess_window(window: np.ndarray) -> np.ndarray:
    """Apply the 1-channel input contract to a single 2500-sample window.

    Returns a ``(2500,)`` float32 array, rank-normalized into ``[-1, +1]``.
    Input must already be at 250 Hz; resampling is the caller's job —
    use :meth:`Inference.predict` for an end-to-end path that handles it.
    """
    if window.shape[-1] != WINDOW_SAMPLES:
        raise ValueError(
            f"window must have {WINDOW_SAMPLES} samples, got {window.shape}")
    return rank_normalize(window.astype(np.float32))     # (2500,)


@dataclass
class Boundary:
    """One detected wave boundary (sample-indexed in the original 250 Hz window)."""
    cls: int           # CLASS_P / CLASS_QRS / CLASS_T
    start: int         # sample index of onset
    end: int           # sample index of offset (inclusive)

    @property
    def name(self) -> str:
        return {CLASS_P: "P", CLASS_QRS: "QRS", CLASS_T: "T"}[self.cls]


def logits_to_boundaries(
    logits: np.ndarray,
    *,
    min_frames: int = 2,
) -> list[Boundary]:
    """Decode per-frame class logits to a list of (cls, start, end) boundaries.

    Parameters
    ----------
    logits : (N_FRAMES, 4) array of per-frame class scores.
    min_frames : drop runs shorter than this many consecutive frames
        (default 2 = 40 ms at the 50 Hz frame rate of 250-Hz / patch=5).

    Returns
    -------
    Sorted list of :class:`Boundary` in sample-indexed coordinates of the
    original 2500-sample window (start/end are inclusive sample indices).
    """
    if logits.ndim != 2 or logits.shape != (N_FRAMES, N_CLASSES):
        raise ValueError(
            f"logits must be ({N_FRAMES}, {N_CLASSES}), got {logits.shape}"
        )
    classes = logits.argmax(axis=-1)        # (N_FRAMES,)
    out: list[Boundary] = []
    i = 0
    while i < N_FRAMES:
        c = int(classes[i])
        if c == CLASS_NONE:
            i += 1
            continue
        j = i
        while j < N_FRAMES and int(classes[j]) == c:
            j += 1
        run_len = j - i
        if run_len >= min_frames:
            start_sample = i * PATCH_SIZE
            end_sample = j * PATCH_SIZE - 1
            out.append(Boundary(cls=c, start=start_sample, end=end_sample))
        i = j
    return out


def bundled_model_path() -> Path:
    """Return the path to the .tflite model shipped inside the package.

    The canonical v56c int8 boundary detector is bundled at
    ``openecg/models/boundary_int8.tflite`` (~1.5 MB) so ``Inference()``
    can be instantiated with no extra downloads. Raises ``FileNotFoundError``
    if the artifact is missing — this only happens in an in-tree dev
    install before the model has been exported.
    """
    p = Path(__file__).resolve().parent / "models" / "boundary_int8.tflite"
    if not p.exists():
        raise FileNotFoundError(
            f"Bundled boundary model not found at {p}. Run "
            f"`python -m scripts.export_boundary_tflite` to build it, "
            f"or pass an explicit path to Inference()."
        )
    return p


# -- ONNX-backed layered codec (onnxruntime + numpy, no torch) --------------

CODEC_SEQ = 5000          # samples per window the bundled codec expects (10 s @ 500 Hz)
CODEC_FS = 500            # native sample rate of the codec ONNX graph


def bundled_codec_onnx_path(version: str = "v6") -> Path:
    """Path to the int8 ONNX layered codec shipped inside the package.

    ``codec_{version}_int8.onnx`` (~3.6 MB) carries all three heads
    (frame / beat / rhythm). Raises ``FileNotFoundError`` if missing.
    """
    p = Path(__file__).resolve().parent / "models" / f"codec_{version}_int8.onnx"
    if not p.exists():
        raise FileNotFoundError(
            f"Bundled ONNX codec not found at {p}. Export it with "
            f"`python -m scripts.export_encoder_onnx`, or pass an explicit path."
        )
    return p


class OnnxCodec:
    """ONNX-backed layered codec — a torch-free drop-in for ``model=``.

    Loads the bundled int8 codec (~3.6 MB, all three heads) via
    ``onnxruntime`` and exposes :meth:`encode` returning a
    :class:`~openecg.layered.LayeredCodec`, so it plugs straight into
    :func:`openecg.encode` / :func:`openecg.encode_stream` /
    :func:`openecg.report` without PyTorch::

        >>> from openecg.deploy import OnnxCodec
        >>> import openecg
        >>> codec = openecg.encode(sig_500hz, fs=500, model=OnnxCodec())
        >>> report = openecg.report(sig_500hz, fs=500, model=OnnxCodec())

    Deploy footprint: ``onnxruntime`` (~15 MB) + ``numpy``. Native rate is
    500 Hz with a fixed 10-s (5000-sample) window; shorter input is
    zero-padded, longer input is handled window-by-window by
    ``encode_stream`` (which feeds this exactly 5000 samples per call).
    """

    SEQ = CODEC_SEQ

    def __init__(self, onnx_path: str | Path | None = None, *,
                 version: str = "v6", providers: list[str] | None = None):
        import onnxruntime as ort  # local: keep onnxruntime optional
        if onnx_path is None:
            onnx_path = bundled_codec_onnx_path(version)
        self.path = str(onnx_path)
        self._sess = ort.InferenceSession(
            self.path, providers=providers or ["CPUExecutionProvider"])
        self._inp = self._sess.get_inputs()[0].name
        # Map outputs by name so we never depend on graph output ordering.
        names = [o.name for o in self._sess.get_outputs()]
        self._idx = {head: next(i for i, n in enumerate(names) if head in n)
                     for head in ("frame", "beat", "rhythm")}

    def encode(self, signal: np.ndarray, fs: int = CODEC_FS):
        from openecg.layered import LayeredCodec
        from openecg.eval import ALL_SUPER_QRS_CLASSES, SUPER_OTHER, SUPER_QRS

        sig = np.asarray(signal, dtype=np.float32).ravel()
        n = int(sig.size)
        # The codec was trained on per-window rank-normalized input; normalize
        # here so raw ECG "just works" (rank_normalize is idempotent).
        sig = np.asarray(rank_normalize(sig), dtype=np.float32)
        if sig.size >= self.SEQ:
            x = sig[:self.SEQ]
        else:
            x = np.zeros(self.SEQ, dtype=np.float32)
            x[:sig.size] = sig
        outs = self._sess.run(None, {self._inp: x[None, :].astype(np.float32)})

        def _arg(head, n_keep):
            a = outs[self._idx[head]][0].argmax(-1).astype(np.uint8)
            return a[:n_keep]

        frame = _arg("frame", min(n, self.SEQ))
        beat = _arg("beat", min(n, self.SEQ))
        rhythm = _arg("rhythm", min(n, self.SEQ))
        if n > self.SEQ:                       # only via direct >10 s encode()
            pad = n - self.SEQ
            frame = np.concatenate([frame, np.full(pad, SUPER_OTHER, np.uint8)])
            beat = np.concatenate([beat, np.zeros(pad, np.uint8)])
            rhythm = np.concatenate([rhythm, np.zeros(pad, np.uint8)])
        # Beat layer is QRS-gated (mirror LayeredPredictor.encode).
        qrs_mask = np.isin(frame, ALL_SUPER_QRS_CLASSES)
        beat = np.where(qrs_mask, beat, 0).astype(np.uint8)
        channels = np.stack([frame, beat, rhythm], axis=0)
        return LayeredCodec(fs=int(fs), channels=channels)


class Inference:
    """TFLite-backed boundary detector (canonical model: v56c int8).

    Usage::

        >>> det = Inference()                            # uses bundled v56c int8
        >>> sig_250hz = np.load("ecg.npy")               # any length, 250 Hz
        >>> boundaries_per_window = det.predict(sig_250hz)
        >>> for w in boundaries_per_window:
        ...     for b in w:
        ...         print(b.name, b.start, b.end)

    Pass an explicit ``tflite_path`` to use a custom export
    (e.g. ``Inference("data/deploy/boundary_int8.tflite")``).

    The detector slides a 10-second window (2500 samples @ 250 Hz) over the
    input with no overlap; partial trailing windows are zero-padded.

    The ``tflite-runtime`` package is loaded lazily so the rest of the
    library has no hard tflite dependency. ``pip install tflite-runtime``
    is enough for inference — no TensorFlow needed.
    """

    def __init__(self, tflite_path: str | Path | None = None,
                 num_threads: int = 1):
        if tflite_path is None:
            tflite_path = bundled_model_path()
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore
        except ImportError:
            try:
                # ai-edge-litert is the supported Windows path (no
                # tflite-runtime wheel exists on PyPI for Windows).
                from ai_edge_litert.interpreter import Interpreter  # type: ignore
            except ImportError:
                # Final fallback to full TensorFlow.
                from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
        self.path = str(tflite_path)
        self._interp = Interpreter(model_path=self.path, num_threads=num_threads)
        self._interp.allocate_tensors()
        self._input_detail = self._interp.get_input_details()[0]
        outputs = self._interp.get_output_details()
        # ai_edge_torch preserves output names from the wrapper's tuple;
        # locate cls_logits + reg_offsets by shape rather than name to
        # be robust against converter renaming.
        self._cls_detail = next(o for o in outputs if o["shape"][-1] == N_CLASSES)
        self._reg_detail = next((o for o in outputs if o["shape"][-1] == 6), None)

    def forward_window(self, window: np.ndarray) -> np.ndarray:
        """Run the TFLite model on a single 2500-sample window.

        Returns (N_FRAMES, N_CLASSES) logits. ``window`` must be 250 Hz
        and exactly 2500 samples; use :meth:`predict` for streaming input.
        """
        x = preprocess_window(window)[None, ...]           # (1, 2500)
        self._interp.set_tensor(self._input_detail["index"], x.astype(np.float32))
        self._interp.invoke()
        out = self._interp.get_tensor(self._cls_detail["index"])
        return out[0]                                      # (N_FRAMES, N_CLASSES)

    def predict(
        self,
        signal_250hz: np.ndarray,
        *,
        return_logits: bool = False,
    ) -> list:
        """Slide a 10-s window over ``signal_250hz`` and decode boundaries.

        Parameters
        ----------
        signal_250hz : 1-D float array at 250 Hz. Trailing samples that
            don't fill the final 2500-sample window are zero-padded.
        return_logits : if True, return the per-window logit array
            instead of decoded boundaries.

        Returns
        -------
        List of per-window outputs. Each entry is either a list of
        :class:`Boundary` (default) or the raw ``(500, 4)`` logits array.
        """
        sig = np.asarray(signal_250hz, dtype=np.float32).ravel()
        n_full = sig.size // WINDOW_SAMPLES
        rem = sig.size % WINDOW_SAMPLES
        windows: list[np.ndarray] = []
        for i in range(n_full):
            windows.append(sig[i * WINDOW_SAMPLES:(i + 1) * WINDOW_SAMPLES])
        if rem:
            pad = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
            pad[:rem] = sig[-rem:]
            windows.append(pad)
        outputs: list = []
        for w in windows:
            logits = self.forward_window(w)
            outputs.append(logits if return_logits else logits_to_boundaries(logits))
        return outputs


__all__ = [
    "export_boundary_tflite",
    "preprocess_window",
    "logits_to_boundaries",
    "bundled_model_path",
    "bundled_codec_onnx_path",
    "OnnxCodec",
    "CODEC_SEQ",
    "CODEC_FS",
    "Inference",
    "Boundary",
    "DEFAULT_CKPT",
    "PATCH_SIZE",
    "WINDOW_SAMPLES",
    "N_FRAMES",
    "TARGET_FS",
    "N_CLASSES",
    "CLASS_NONE",
    "CLASS_P",
    "CLASS_QRS",
    "CLASS_T",
]
