"""Layered ECG codec — sample-resolution multi-layer label channels.

Stacks N parallel label tracks at the input signal's sample rate. Each
layer encodes wave/beat/rhythm information at a different abstraction;
segment start/end is naturally represented by class transitions, and an
event-list view is provided for textual rendering and LLM ingestion.

Layers (low -> high abstraction):

  0 frame   wave class per sample: other / P / QRS / T / paced_QRS
            (output of the boundary delineator, nearest-neighbor
            upsampled from 50 Hz frames to the input fs).
  1 beat    beat type per sample, active inside QRS spans only:
            none / sinus / vpc / paced / fusion / unknown.  Outside
            QRS the value is ``BEAT_NONE``.
  2 rhythm  rhythm class per sample: sinus / avb / paced / afib / bbb
            / ventricular.  Currently window-constant — one label fills
            all samples.  Sub-window rhythm segmentation is a follow-up.

Sample resolution preserves on/off precision and makes channels
trivially zip-able with the raw signal for plotting and downstream
sequence models.  v0 covers a single 10-s window @ 250 Hz; multi-window
sliding is a thin loop on top.
"""
from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Callable, Optional

import numpy as np

from openecg.eval import (
    SUPER_OTHER, SUPER_P, SUPER_QRS, SUPER_T, SUPER_PACED_QRS,
    SUPER_NAMES, ALL_SUPER_QRS_CLASSES,
)

LAYER_NAMES: tuple[str, ...] = ("frame", "beat", "rhythm")
N_LAYERS = len(LAYER_NAMES)

# --- Continuous-use codec: 2-s edge guard --------------------------------
# Predictions in the outer ``EVAL_MARGIN_S`` seconds of any window have
# limited past/future context. Held-out evaluation, codec-string export,
# and stream stitching exclude these samples so adjacent windows can be
# concatenated seamlessly.  See the README "Continuous-use codec" section.
EVAL_MARGIN_S: float = 2.0
DEFAULT_WINDOW_S: float = 10.0

# ---- Layer 1: beat type --------------------------------------------------
BEAT_NONE    = 0
BEAT_SINUS   = 1
BEAT_VPC     = 2
BEAT_PACED   = 3
BEAT_FUSION  = 4
BEAT_UNKNOWN = 5
BEAT_NAMES = {
    BEAT_NONE: "none", BEAT_SINUS: "sinus", BEAT_VPC: "vpc",
    BEAT_PACED: "paced", BEAT_FUSION: "fusion", BEAT_UNKNOWN: "unknown",
}

# ---- Layer 2: rhythm (indices match openecg.rhythm.CLASS_NAMES) ----------
RHYTHM_SINUS, RHYTHM_AVB, RHYTHM_PACED, RHYTHM_AFIB, RHYTHM_BBB, RHYTHM_VENT = range(6)
RHYTHM_NAMES = {
    RHYTHM_SINUS: "sinus", RHYTHM_AVB: "avb", RHYTHM_PACED: "paced",
    RHYTHM_AFIB: "afib", RHYTHM_BBB: "bbb", RHYTHM_VENT: "ventricular",
}

CLASS_NAMES_BY_LAYER = {
    "frame":  SUPER_NAMES,
    "beat":   BEAT_NAMES,
    "rhythm": RHYTHM_NAMES,
}

# ---- Unified frame+beat view --------------------------------------------
# A single per-sample track that merges the frame (wave) and beat channels:
# the frame channel with each QRS sample REPLACED by its beat type (the beat
# channel is only meaningful inside QRS, so the two are complementary). This
# is a consumption/rendering convenience derived from the two trained heads —
# the model still has separate frame + beat heads (heterogeneous label sources
# make a single trained head impractical; see LayeredCodec.unified).
UNIFIED_OTHER, UNIFIED_P, UNIFIED_T = 0, 1, 2
UNIFIED_SINUS, UNIFIED_VPC, UNIFIED_PACED, UNIFIED_FUSION, UNIFIED_UNKNOWN = 3, 4, 5, 6, 7
UNIFIED_NAMES = {
    0: "other", 1: "P", 2: "T", 3: "sinus", 4: "vpc",
    5: "paced", 6: "fusion", 7: "unknown",
}
# beat class id -> unified QRS-type id
_BEAT_TO_UNIFIED = {
    BEAT_SINUS: UNIFIED_SINUS, BEAT_VPC: UNIFIED_VPC, BEAT_PACED: UNIFIED_PACED,
    BEAT_FUSION: UNIFIED_FUSION, BEAT_UNKNOWN: UNIFIED_UNKNOWN,
}

_DEFAULT_RENDER_CHARS = {
    "frame":  {SUPER_OTHER: "-", SUPER_P: "p", SUPER_QRS: "Q",
               SUPER_T: "t", SUPER_PACED_QRS: "P"},
    "beat":   {BEAT_NONE: "-", BEAT_SINUS: "s", BEAT_VPC: "v",
               BEAT_PACED: "p", BEAT_FUSION: "f", BEAT_UNKNOWN: "?"},
    "rhythm": {RHYTHM_SINUS: "S", RHYTHM_AVB: "A", RHYTHM_PACED: "P",
               RHYTHM_AFIB: "F", RHYTHM_BBB: "B", RHYTHM_VENT: "V"},
}


@dataclass
class LayeredCodec:
    """Output container — ``channels`` is uint8 (N_LAYERS, n_samples).

    The optional ``eval_margin_s`` (default 2.0) marks the front and back
    seconds of the window as outside the evaluation / emission band.
    Samples in ``[0, margin]`` and ``[n - margin, n]`` carry less past /
    future context and are therefore unsafe to emit when concatenating
    adjacent windows.  Use :attr:`eval_slice` or :attr:`eval_mask` to
    restrict downstream consumption to the inner band.
    """
    fs: int
    channels: np.ndarray
    layer_names: tuple[str, ...] = LAYER_NAMES
    eval_margin_s: float = EVAL_MARGIN_S

    def __post_init__(self) -> None:
        self.fs = int(self.fs)
        if self.fs <= 0:
            raise ValueError(f"fs must be positive, got {self.fs}")
        self.channels = np.asarray(self.channels, dtype=np.uint8)
        if self.channels.ndim != 2:
            raise ValueError(
                f"channels must be 2-D (n_layers, n_samples), got "
                f"shape {self.channels.shape}"
            )
        if self.channels.shape[0] != len(self.layer_names):
            raise ValueError(
                f"channels has {self.channels.shape[0]} layers but "
                f"layer_names has {len(self.layer_names)} entries"
            )
        if tuple(self.layer_names) != LAYER_NAMES:
            unknown = set(self.layer_names) - set(LAYER_NAMES)
            if unknown:
                raise ValueError(f"unknown layer names: {sorted(unknown)}")
        self.layer_names = tuple(self.layer_names)
        self.eval_margin_s = float(self.eval_margin_s)
        if self.eval_margin_s < 0:
            raise ValueError(
                f"eval_margin_s must be non-negative, got {self.eval_margin_s}"
            )

    @property
    def frame(self) -> np.ndarray:  return self.channels[0]
    @property
    def beat(self) -> np.ndarray:   return self.channels[1]
    @property
    def rhythm(self) -> np.ndarray: return self.channels[2]
    @property
    def n_samples(self) -> int:     return int(self.channels.shape[1])

    @property
    def unified(self) -> np.ndarray:
        """Single 8-class per-sample track merging frame + beat: the frame
        channel with each QRS sample replaced by its beat type. Classes:
        ``0 other, 1 P, 2 T, 3 sinus, 4 vpc, 5 paced, 6 fusion, 7 unknown``
        (see :data:`UNIFIED_NAMES`). The beat channel is meaningful only inside
        QRS, so this collapses the two complementary tracks into one readable
        stream — P/T waves and beat-typed QRS complexes on a single axis::

            codec.unified                          # uint8[n_samples]
            [UNIFIED_NAMES[c] for c in codec.unified]

        A QRS sample the beat head left untyped falls back to ``unknown``.
        """
        f, b = self.frame, self.beat
        u = np.zeros(self.n_samples, dtype=np.uint8)
        u[f == SUPER_P] = UNIFIED_P
        u[f == SUPER_T] = UNIFIED_T
        u[np.isin(f, ALL_SUPER_QRS_CLASSES)] = UNIFIED_UNKNOWN   # QRS, type t.b.d.
        for beat_id, uni_id in _BEAT_TO_UNIFIED.items():         # beat is QRS-gated
            u[b == beat_id] = uni_id
        return u

    @property
    def margin_samples(self) -> int:
        return int(round(self.eval_margin_s * self.fs))

    @property
    def eval_slice(self) -> slice:
        """``slice`` covering the inner [margin, n - margin] band."""
        m = self.margin_samples
        return slice(m, max(m, self.n_samples - m))

    @property
    def eval_mask(self) -> np.ndarray:
        """``bool[n_samples]`` — True inside the eval band, False in the
        2-s guards.  Use for masked-loss / masked-metric computation.
        """
        m = self.margin_samples
        mask = np.zeros(self.n_samples, dtype=bool)
        if self.n_samples > 2 * m:
            mask[m:self.n_samples - m] = True
        return mask

    def inner(self) -> "LayeredCodec":
        """Return a sliced copy clipped to ``eval_slice`` (margin = 0).

        Useful for ``encode_stream`` to concatenate seamless output:
        ``np.concatenate([w.inner().channels for w in windows], axis=1)``.
        """
        sl = self.eval_slice
        return LayeredCodec(fs=self.fs,
                             channels=self.channels[:, sl].copy(),
                             layer_names=self.layer_names,
                             eval_margin_s=0.0)

    def events(self, layer: str, *, drop_class: int | None = None,
               eval_only: bool = False
               ) -> list[tuple[int, int, int]]:
        """Run-length-encode one layer to (start, end_exclusive, class_id).

        Pass ``drop_class`` to skip segments of that class (e.g.
        ``BEAT_NONE`` to get only QRS-active beat spans).
        Pass ``eval_only=True`` to restrict to ``eval_slice`` — samples
        in the 2-s guards are excluded and segment indices are clipped.
        """
        if layer not in LAYER_NAMES:
            raise ValueError(f"unknown layer {layer!r}")
        arr = self.channels[LAYER_NAMES.index(layer)]
        if eval_only:
            sl = self.eval_slice
            arr = arr[sl]
            base = sl.start
        else:
            base = 0
        out: list[tuple[int, int, int]] = []
        i, n = 0, int(arr.size)
        while i < n:
            j = i + 1
            while j < n and arr[j] == arr[i]:
                j += 1
            cls = int(arr[i])
            if drop_class is None or cls != drop_class:
                out.append((base + i, base + j, cls))
            i = j
        return out

    def to_codec_string(self, *, layer: str = "frame",
                        chars: dict[int, str] | None = None,
                        downsample: int = 5,
                        eval_only: bool = False) -> str:
        """ASCII rendering of one layer.  Default ``downsample=5`` collapses
        250 Hz back to the 50 Hz frame-grid for readability.

        Pass ``eval_only=True`` to render only the inner [margin, -margin]
        band — the form intended for concatenation into a continuous codec
        string across many windows.
        """
        cmap = chars or _DEFAULT_RENDER_CHARS[layer]
        arr = self.channels[LAYER_NAMES.index(layer)]
        if eval_only:
            arr = arr[self.eval_slice]
        arr = arr[::max(1, downsample)]
        return "".join(cmap.get(int(c), "?") for c in arr)


# ---- Default predictor adapters -----------------------------------------

def _default_frame_predictor(signal: np.ndarray, fs: int, lead_id: int
                             ) -> np.ndarray:
    """Bundled TFLite delineator -> (n_frames,) uint8."""
    from openecg.deploy import Inference  # local import: tflite optional
    if not hasattr(_default_frame_predictor, "_inf"):
        _default_frame_predictor._inf = Inference()  # type: ignore[attr-defined]
    logits = _default_frame_predictor._inf.forward_window(signal)  # type: ignore[attr-defined]
    return np.argmax(logits, axis=-1).astype(np.uint8)


def _default_rhythm_predictor(signal: np.ndarray, fs: int) -> int:
    """Bundled 6-class rhythm router -> rhythm_id (int)."""
    from openecg.rhythm import classify, CLASS_NAMES, WINDOW_SAMPLES
    if fs != 250 or signal.size != WINDOW_SAMPLES:
        raise ValueError(
            f"default rhythm predictor needs 10 s @ 250 Hz "
            f"({WINDOW_SAMPLES} samples); got fs={fs}, len={signal.size}"
        )
    label, _ = classify(signal)
    return CLASS_NAMES.index(label)


def _default_beat_classifier(signal: np.ndarray,
                             qrs_spans: list[tuple[int, int]],
                             frames: np.ndarray, frame_ms: int, fs: int,
                             rhythm_id: int) -> list[int]:
    """Rule-stub: derive beat type from frame class + rhythm context.

    A dedicated per-beat morphology classifier will replace this; until
    then:
      * paced_QRS frame in span      -> BEAT_PACED
      * rhythm == ventricular        -> BEAT_VPC
      * rhythm in {sinus, avb}       -> BEAT_SINUS  (P-axis rhythms)
      * otherwise                    -> BEAT_UNKNOWN
    """
    spf = max(1, int(round(frame_ms * fs / 1000.0)))
    out: list[int] = []
    for s, e in qrs_spans:
        f_lo = s // spf
        f_hi = max(f_lo + 1, e // spf)
        seg = frames[f_lo:f_hi]
        if seg.size and (seg == SUPER_PACED_QRS).any():
            out.append(BEAT_PACED)
        elif rhythm_id == RHYTHM_VENT:
            out.append(BEAT_VPC)
        elif rhythm_id in (RHYTHM_SINUS, RHYTHM_AVB):
            out.append(BEAT_SINUS)
        else:
            out.append(BEAT_UNKNOWN)
    return out


# ---- Public encoder -----------------------------------------------------

_CODEC_CACHE: dict = {}


def load_codec(ckpt: str | None = None, device: str = "cpu"):
    """Load the bundled layered-codec model (frame/beat/rhythm) for use as the
    ``model=`` argument to :func:`encode`.

    Defaults to the packaged ``codec_v6.pt`` (pure-real + lydus hospital rhythm,
    sample-resolution multi-head, 500 Hz). codec_v6 adds a **structural-prior frame
    retrain** (penalize physiologically-forbidden wave→wave transitions + TV
    contiguity — from LUDB labels, P/QRS/T never directly touch) then re-derives the
    beat & rhythm heads (frozen-head, vitaldb VPC + natural-prior). Frame boundary-F1
    **0.829 → 0.855** (+0.026), median timing 11.6 → 11.1 ms; beat VPC 0.935 → 0.929,
    rhythm 0.805 → 0.797 (tiny). Requires torch. Cached per (ckpt, device).

        >>> import openecg
        >>> m = openecg.load_codec()
        >>> codec = openecg.encode(signal_500hz, fs=500, model=m)
    """
    from pathlib import Path as _Path
    if ckpt is None:
        ckpt = str(_Path(__file__).with_name("models") / "codec_v6.pt")
    key = (ckpt, device)
    if key in _CODEC_CACHE:
        return _CODEC_CACHE[key]
    from openecg.stage2.model import load_model_from_ckpt
    model, _ = load_model_from_ckpt(ckpt, device=device)
    model.eval()
    _CODEC_CACHE[key] = model
    return model


def load_codec_onnx(onnx_path: str | None = None, *, version: str = "v6"):
    """Load the bundled **int8 ONNX** layered codec as a ``model=`` argument
    for :func:`encode` — a torch-free path that needs only ``onnxruntime``.

        >>> import openecg
        >>> m = openecg.load_codec_onnx()            # ~3.6 MB, no torch
        >>> codec = openecg.encode(signal_500hz, fs=500, model=m)

    Same heads as :func:`load_codec` (frame / beat / rhythm) at 500 Hz; the
    int8 graph is byte-light enough for on-device / agent deployment. See
    :class:`openecg.deploy.OnnxCodec`.
    """
    from openecg.deploy import OnnxCodec
    return OnnxCodec(onnx_path, version=version)


def _infer_model_device(model) -> str:
    """Best-effort device string for torch modules."""
    try:
        return str(next(model.parameters()).device)
    except (AttributeError, StopIteration):
        return "cpu"


def _is_layered_encoder(model) -> bool:
    return isinstance(model, LayeredPredictor) or (
        hasattr(model, "encode") and not hasattr(model, "parameters")
    )


def _is_codec_ref(model) -> bool:
    return isinstance(model, (str, PathLike))


def _resolve_codec_ref(model) -> str | None:
    if isinstance(model, str) and model.lower() in {"default", "bundled", "v1"}:
        return None
    return str(model)


FramePredictor   = Callable[[np.ndarray, int, int], np.ndarray]
RhythmPredictor  = Callable[[np.ndarray, int], int]
BeatClassifier   = Callable[[np.ndarray, list, np.ndarray, int, int, int], list]


class LayeredPredictor:
    """Adapter for a multi-head ``FrameClassifierTransformerLayered1Ch``.

    Wraps a model whose forward returns ``(cls, reg, beat, rhythm)`` —
    all per-patch over ``(B, n_patches, n_classes)`` — and exposes
    either:

      * :meth:`encode` — return a full :class:`LayeredCodec` from one
        forward pass (preferred);
      * three individual callbacks (``frame_predictor`` etc.) compatible
        with :func:`encode`'s injection points (kept for symmetry).

    Caches the per-patch logits keyed by ``id(signal)`` so the three
    callback path also runs forward only once per call.
    """

    def __init__(self, model, *, lead_id: int = 0, device: str | None = None,
                 frame_ms: int = 20):
        self.model = model.eval()
        self.device = device or _infer_model_device(model)
        self.model.to(self.device)
        self.lead_id = int(lead_id)
        self.frame_ms = int(frame_ms)
        self._cache_key: int | None = None
        self._cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def _forward(self, signal: np.ndarray):
        import torch  # local: keep numpy-only callers torch-free
        if self._cache_key == id(signal) and self._cache is not None:
            return self._cache
        sig = np.asarray(signal, dtype=np.float32).ravel()
        # The codec was trained on per-window rank-normalized signals; feeding a
        # raw ECG silently degrades every head (frame F1 0.85 -> 0.65). Normalize
        # here so the public API "just works" on raw input. rank_normalize is
        # idempotent, so callers that already normalized are unaffected.
        from openecg.dsp import rank_normalize as _rank_normalize
        sig = np.asarray(_rank_normalize(sig), dtype=np.float32)
        x = torch.from_numpy(sig).unsqueeze(0).to(self.device)
        lid = torch.tensor([self.lead_id], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model(x, lid)
        # 4-tuple (frame, reg, beat, rhythm) = old layered arch;
        # 3-tuple (frame, beat, rhythm) = sample-res conv-tok multi-head (v59b+).
        if len(out) == 4:
            cls, _reg, beat, rhythm = out
        else:
            cls, beat, rhythm = out
        f = cls[0].argmax(-1).cpu().numpy().astype(np.uint8)
        b = beat[0].argmax(-1).cpu().numpy().astype(np.uint8)
        r = rhythm[0].argmax(-1).cpu().numpy().astype(np.uint8)
        self._cache_key, self._cache = id(signal), (f, b, r)
        return self._cache

    def encode(self, signal: np.ndarray, fs: int = 250) -> LayeredCodec:
        sig = np.asarray(signal, dtype=np.float32).ravel()
        n = int(sig.size)
        f_patches, b_patches, r_patches = self._forward(sig)
        spf = max(1, int(round(self.frame_ms * fs / 1000.0)))

        def _upsample(arr_50hz: np.ndarray, fill: int) -> np.ndarray:
            # Sample-resolution heads (v59b+) already emit one label per input
            # sample — no patch upsampling needed.
            if arr_50hz.size == n:
                return arr_50hz.astype(np.uint8)
            up = np.repeat(arr_50hz, spf)[:n]
            if up.size < n:
                up = np.concatenate(
                    [up, np.full(n - up.size, fill, dtype=np.uint8)])
            return up

        layer_frame  = _upsample(f_patches, SUPER_OTHER)
        layer_rhythm = _upsample(r_patches, RHYTHM_SINUS)
        beat_full    = _upsample(b_patches, BEAT_NONE)
        # Beat layer is QRS-gated: zero outside QRS frames so downstream
        # callers see BEAT_NONE in non-QRS regions regardless of what the
        # head emitted there.
        qrs_mask = np.isin(layer_frame, ALL_SUPER_QRS_CLASSES)
        layer_beat = np.where(qrs_mask, beat_full, BEAT_NONE).astype(np.uint8)

        channels = np.stack([layer_frame, layer_beat, layer_rhythm], axis=0)
        return LayeredCodec(fs=fs, channels=channels)

    # ---- Injection-point callbacks (compose with `encode()` if mixing) --

    def frame_predictor(self, signal: np.ndarray, fs: int, lead_id: int
                        ) -> np.ndarray:
        return self._forward(signal)[0]

    def rhythm_predictor(self, signal: np.ndarray, fs: int) -> int:
        r = self._forward(signal)[2]
        return int(np.bincount(r, minlength=6).argmax()) if r.size else RHYTHM_SINUS

    def beat_classifier(self, signal: np.ndarray,
                        qrs_spans: list[tuple[int, int]],
                        frames: np.ndarray, frame_ms: int, fs: int,
                        rhythm_id: int) -> list[int]:
        b = self._forward(signal)[1]
        spf = max(1, int(round(frame_ms * fs / 1000.0)))
        out: list[int] = []
        for s, e in qrs_spans:
            f_lo, f_hi = s // spf, max(s // spf + 1, e // spf)
            seg = b[f_lo:f_hi]
            out.append(int(np.bincount(seg, minlength=6).argmax())
                       if seg.size else BEAT_UNKNOWN)
        return out


def encode(
    signal: np.ndarray,
    fs: int = 250,
    *,
    model=None,
    frame_predictor: Optional[FramePredictor] = None,
    rhythm_predictor: Optional[RhythmPredictor] = None,
    beat_classifier: Optional[BeatClassifier] = None,
    lead_id: int = 0,
    frame_ms: int = 20,
    eval_margin_s: float = EVAL_MARGIN_S,
    device: str | None = None,
) -> LayeredCodec:
    """Encode one ECG window into the layered codec.

    Parameters
    ----------
    signal : 1-D float array at ``fs`` Hz.
    fs : sample rate of ``signal``.  Channels are returned at this rate.
    model : optional multi-head model, pre-built :class:`LayeredPredictor`,
        or checkpoint reference.  Strings / PathLike objects are loaded lazily
        through :func:`load_codec`; pass ``"default"`` for the bundled codec.
        When model is given, it fills *all three* layers and the
        ``frame_predictor`` / ``rhythm_predictor`` / ``beat_classifier`` kwargs
        are ignored.
    frame_predictor : ``(signal, fs, lead_id) -> uint8 frames`` at
        ``frame_ms`` resolution with values in ``SUPER_*``.  Defaults to
        the bundled TFLite delineator.
    rhythm_predictor : ``(signal, fs) -> rhythm_id`` (0..5).  Defaults to
        ``openecg.rhythm.classify``.
    beat_classifier : ``(signal, qrs_spans, frames, frame_ms, fs,
        rhythm_id) -> list[beat_id]`` of length ``len(qrs_spans)``.
        Defaults to the rule-stub above.

    Returns
    -------
    LayeredCodec with ``channels`` shape ``(N_LAYERS, len(signal))``.
    """
    if model is not None:
        if _is_codec_ref(model):
            model = load_codec(_resolve_codec_ref(model), device=device or "cpu")
        predictor = (
            model if _is_layered_encoder(model)
            else LayeredPredictor(model, lead_id=lead_id, device=device,
                                  frame_ms=frame_ms)
        )
        c = predictor.encode(signal, fs=fs)
        c.eval_margin_s = float(eval_margin_s)
        return c
    sig = np.asarray(signal, dtype=np.float32).ravel()
    n = int(sig.size)
    spf = max(1, int(round(frame_ms * fs / 1000.0)))

    # Layer 0: frame (predict at 50 Hz, upsample by nearest-neighbor).
    fp = frame_predictor or _default_frame_predictor
    frames = np.asarray(fp(sig, fs, lead_id), dtype=np.uint8)
    layer_frame = np.repeat(frames, spf)[:n]
    if layer_frame.size < n:
        layer_frame = np.concatenate([
            layer_frame,
            np.full(n - layer_frame.size, SUPER_OTHER, dtype=np.uint8),
        ])

    # Layer 2: rhythm (window-constant for v0).
    rp = rhythm_predictor or _default_rhythm_predictor
    rhythm_id = int(rp(sig, fs))
    layer_rhythm = np.full(n, rhythm_id, dtype=np.uint8)

    # Layer 1: beat (per-QRS, expanded across the span).
    qrs_mask = np.isin(layer_frame, ALL_SUPER_QRS_CLASSES).astype(np.int8)
    edges = np.diff(np.concatenate([[0], qrs_mask, [0]]))
    starts = np.where(edges == 1)[0].tolist()
    ends   = np.where(edges == -1)[0].tolist()
    qrs_spans = list(zip(starts, ends))
    bc = beat_classifier or _default_beat_classifier
    beat_ids = bc(sig, qrs_spans, frames, frame_ms, fs, rhythm_id)
    layer_beat = np.full(n, BEAT_NONE, dtype=np.uint8)
    for (s, e), bid in zip(qrs_spans, beat_ids):
        layer_beat[s:e] = bid

    channels = np.stack([layer_frame, layer_beat, layer_rhythm], axis=0)
    return LayeredCodec(fs=fs, channels=channels,
                         eval_margin_s=float(eval_margin_s))


def encode_stream(
    signal: np.ndarray,
    fs: int = 250,
    *,
    window_s: float = DEFAULT_WINDOW_S,
    eval_margin_s: float = EVAL_MARGIN_S,
    **encode_kwargs,
) -> LayeredCodec:
    """Slide :func:`encode` over a long signal, emit a seamless codec.

    Adjacent windows overlap by ``2 * eval_margin_s``; each window
    contributes its *inner* ``[margin, window - margin]`` portion to the
    output.  Head and tail of the full signal are filled from the first
    and last windows respectively (no past / future to overlap with).

    For ``window_s=10`` and ``eval_margin_s=2`` the stride is 6 s — every
    sample is covered by a prediction made with at least 2 s of past
    *and* 2 s of future context (except the very first and last 2 s of
    the whole recording, which are unavoidable).

    Returns a :class:`LayeredCodec` covering the full ``len(signal)``
    samples, with ``eval_margin_s=0`` (the stitched output has no
    internal guard region; only the absolute head / tail are still
    context-light).
    """
    sig = np.asarray(signal, dtype=np.float32).ravel()
    fs = int(fs)
    win_n = int(round(window_s * fs))
    mar_n = int(round(eval_margin_s * fs))
    if win_n <= 2 * mar_n:
        raise ValueError(
            f"window_s ({window_s}s) must exceed 2 * eval_margin_s "
            f"({2 * eval_margin_s}s)")
    inner_n = win_n - 2 * mar_n
    n = sig.size
    if _is_codec_ref(encode_kwargs.get("model")):
        encode_kwargs = dict(encode_kwargs)
        encode_kwargs["model"] = load_codec(
            _resolve_codec_ref(encode_kwargs["model"]),
            device=encode_kwargs.get("device") or "cpu",
        )
    if encode_kwargs.get("model") is not None and not _is_layered_encoder(
        encode_kwargs["model"]
    ):
        encode_kwargs = dict(encode_kwargs)
        encode_kwargs["model"] = LayeredPredictor(
            encode_kwargs["model"],
            lead_id=int(encode_kwargs.get("lead_id", 0)),
            device=encode_kwargs.get("device"),
            frame_ms=int(encode_kwargs.get("frame_ms", 20)),
        )

    if n <= win_n:                                              # single window
        return encode(sig, fs=fs, eval_margin_s=eval_margin_s,
                       **encode_kwargs)

    # Sliding positions so each window's inner band covers a disjoint
    # contiguous span of the output. Window k spans [k*inner_n, k*inner_n + win_n).
    pieces: list[np.ndarray] = []
    k = 0
    while True:
        start = k * inner_n
        end = start + win_n
        if end >= n:                                            # final window
            end = n
            start = max(0, end - win_n)
            w = np.zeros(win_n, dtype=np.float32)
            tail = sig[start:end]
            w[:tail.size] = tail
            c = encode(w, fs=fs, eval_margin_s=eval_margin_s, **encode_kwargs)
            # First window: keep [0, win_n - mar_n]; mid: [mar_n, win_n - mar_n];
            # last: keep [mar_n, win_n] but trim trailing pad.
            inner_start = 0 if k == 0 else mar_n
            valid_end = tail.size                               # ignore pad
            pieces.append(c.channels[:, inner_start:valid_end])
            break
        c = encode(sig[start:end], fs=fs, eval_margin_s=eval_margin_s,
                    **encode_kwargs)
        inner_start = 0 if k == 0 else mar_n
        inner_end = win_n - mar_n
        pieces.append(c.channels[:, inner_start:inner_end])
        k += 1

    channels = np.concatenate(pieces, axis=1)
    # In rare cases (window_s coverage rounding) channels may overshoot
    # the signal length by a sample. Trim to exact length.
    channels = channels[:, :n]
    return LayeredCodec(fs=fs, channels=channels, eval_margin_s=0.0)


def frames_to_events(frames: np.ndarray, frame_ms: int = 20
                     ) -> list[tuple[int, int]]:
    """Run-length encode per-frame class IDs to ``(class_id, length_ms)``.

    This is a lightweight layered-codec utility for validation and boundary
    extraction. It intentionally returns plain Python events instead of a
    separate packed token format.
    """
    arr = np.asarray(frames, dtype=np.uint8).ravel()
    if arr.size == 0:
        return []
    change_idx = np.flatnonzero(np.diff(arr)) + 1
    boundaries = np.concatenate(([0], change_idx, [arr.size]))
    out: list[tuple[int, int]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        out.append((int(arr[start]), int(end - start) * int(frame_ms)))
    return out


__all__ = [
    "encode", "encode_stream", "load_codec", "load_codec_onnx",
    "LayeredCodec", "LayeredPredictor", "frames_to_events",
    "LAYER_NAMES", "N_LAYERS", "EVAL_MARGIN_S", "DEFAULT_WINDOW_S",
    "BEAT_NONE", "BEAT_SINUS", "BEAT_VPC", "BEAT_PACED", "BEAT_FUSION",
    "BEAT_UNKNOWN", "BEAT_NAMES",
    "RHYTHM_SINUS", "RHYTHM_AVB", "RHYTHM_PACED", "RHYTHM_AFIB",
    "RHYTHM_BBB", "RHYTHM_VENT", "RHYTHM_NAMES",
    "CLASS_NAMES_BY_LAYER", "UNIFIED_NAMES",
]
