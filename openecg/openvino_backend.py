"""OpenVINO backend for the v56c boundary detector.

Reads ``v56c_fp32.onnx`` — the fp32 v56c boundary detector published as a
release asset (``boundary-v56c``; see ``openecg/models/v56c_MODEL_CARD.md``)
— directly. It is not bundled: pass its path, set ``OPENECG_V56C_ONNX``, or
call :func:`fetch_v56c_fp32_onnx` once.

Device selection: any OpenVINO device name, e.g. ``CPU`` (default), ``GPU``,
``NPU``, or ``AUTO``.

Precision: on ``CPU`` and ``GPU`` the model compiles with
``INFERENCE_PRECISION_HINT=f32`` by default (OpenVINO would otherwise pick bf16
on CPUs with native bf16 support and f16 on GPUs), so results match the
reference runtimes. ``NPU`` uses its own default. Pass ``config=`` to override, e.g.
``config={"INFERENCE_PRECISION_HINT": "bf16"}`` or ``config={}`` for the device
default.

Requires the ``openvino`` extra::

    pip install "openecg[openvino]"
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

from openecg.deploy import preprocess_window

F32_HINT = {"INFERENCE_PRECISION_HINT": "f32"}


def _device_base(device: str) -> str:
    """``"GPU.0"`` -> ``"GPU"``; virtual devices (``AUTO:...``) keep their prefix."""
    return device.upper().split(":", 1)[0].split(".", 1)[0]


BOUNDARY_TFLITE_ERROR = (
    "OpenVINO TFLite frontend cannot load boundary_int8.tflite: "
    "GeneralFailure on ADD near patch_embed. "
    "Use the fp32 ONNX release asset (v56c_fp32.onnx) instead."
)

#: fp32 ONNX release asset of the v56c boundary detector (Apache-2.0).
V56C_ONNX_URL = (
    "https://github.com/vitaldb/openecg/releases/download/boundary-v56c/v56c_fp32.onnx"
)
#: sha256 from ``openecg/models/v56c_MODEL_CARD.md``.
V56C_ONNX_SHA256 = "6ae046ee058cbe05b28176d77760e19c2587efcab027ef13ea1990b1283a6c8e"
#: Environment variable that points at a local copy of ``v56c_fp32.onnx``.
V56C_ONNX_ENV = "OPENECG_V56C_ONNX"


def _import_openvino():
    try:
        import openvino as ov
    except ImportError as e:
        raise ImportError(
            "openecg.openvino_backend requires OpenVINO: "
            'pip install "openecg[openvino]"'
        ) from e
    return ov


def available_devices() -> list[str]:
    ov = _import_openvino()
    return list(ov.Core().available_devices)


def device_full_names() -> dict[str, str]:
    """``{device: FULL_DEVICE_NAME}`` for every available OpenVINO device."""
    ov = _import_openvino()
    core = ov.Core()
    out = {}
    for d in core.available_devices:
        try:
            out[d] = str(core.get_property(d, "FULL_DEVICE_NAME"))
        except Exception:  # noqa: BLE001
            out[d] = "unknown"
    return out


def _default_cache_dir() -> Path:
    return Path(os.environ.get("OPENECG_CACHE", Path.home() / ".cache" / "openecg"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def boundary_onnx_path() -> Path:
    """Locate ``v56c_fp32.onnx``.

    Looks at ``$OPENECG_V56C_ONNX``, then ``data/deploy/v56c_fp32.onnx`` under the
    current directory (the default output of ``scripts/export_boundary_onnx.py``),
    then the cache used by :func:`fetch_v56c_fp32_onnx`.
    """
    cands = []
    env = os.environ.get(V56C_ONNX_ENV)
    if env:
        cands.append(Path(env).expanduser())
    cands.append(Path("data") / "deploy" / "v56c_fp32.onnx")
    cands.append(_default_cache_dir() / "v56c_fp32.onnx")
    for c in cands:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "v56c_fp32.onnx not found (looked at: "
        + ", ".join(str(c) for c in cands)
        + "). Download it once with "
        "`python -c \"from openecg.openvino_backend import fetch_v56c_fp32_onnx; "
        "fetch_v56c_fp32_onnx()\"`, or set "
        f"{V56C_ONNX_ENV} to a local copy of {V56C_ONNX_URL}."
    )


def fetch_v56c_fp32_onnx(dest: str | Path | None = None, *, url: str = V56C_ONNX_URL) -> Path:
    """Download the v56c fp32 ONNX release asset and verify its sha256.

    Writes to ``dest`` (default ``~/.cache/openecg/v56c_fp32.onnx``, or
    ``$OPENECG_CACHE``). An existing file with the right sha256 is reused.
    """
    dest = Path(dest) if dest is not None else _default_cache_dir() / "v56c_fp32.onnx"
    if dest.is_file() and _sha256(dest) == V56C_ONNX_SHA256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    os.close(fd)
    tmp = Path(tmp)
    try:
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        got = _sha256(tmp)
        if got != V56C_ONNX_SHA256:
            raise RuntimeError(
                f"sha256 mismatch for {url}: got {got}, expected {V56C_ONNX_SHA256}"
            )
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def _compile(model_path: str | Path, device: str, config: dict | None = None):
    ov = _import_openvino()
    core = ov.Core()
    device = device.upper()
    virtual = device.split(":", 1)[0] in ("AUTO", "HETERO", "MULTI", "BATCH")
    if not virtual and device not in core.available_devices:
        raise RuntimeError(
            f"OpenVINO device {device!r} not available; have {core.available_devices}. "
            f"Pass any OpenVINO device name, e.g. CPU, GPU, NPU, or AUTO."
        )
    model = core.read_model(str(model_path))
    _make_static(model)
    return core.compile_model(model, device, config or {})


def _make_static(model) -> None:
    """Pin dynamic input dims (the ONNX ``batch``) to 1.

    Every caller here runs single-window batch-1 inference, so fix the
    shapes before compile.
    """
    ov = _import_openvino()
    new_shapes = {}
    for inp in model.inputs:
        ps = inp.get_partial_shape()
        if ps.is_dynamic:
            dims = [1 if d.is_dynamic else d.get_length() for d in ps]
            new_shapes[inp.get_any_name()] = ov.PartialShape(dims)
    if new_shapes:
        model.reshape(new_shapes)


class OpenVINOBoundary:
    """OpenVINO-backed v56c boundary detector, read from the fp32 ONNX release asset.

    ``model_path`` defaults to :func:`boundary_onnx_path`. Direct TFLite load is
    unsupported (:data:`BOUNDARY_TFLITE_ERROR`). Inference is batch 1, one
    2500-sample window at a time.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        device: str = "CPU",
        config: dict | None = None,
    ):
        if model_path is None:
            model_path = boundary_onnx_path()
        self.path = str(model_path)
        self.device = device.upper()
        if config is None and _device_base(self.device) in ("CPU", "GPU"):
            # f32 instead of the bf16 (CPU) / f16 (GPU) device defaults, so the
            # outputs track the fp32 ONNX reference.
            config = F32_HINT
        self.config = dict(config or {})
        if self.path.endswith(".tflite"):
            # Explicit legacy path — document the failure clearly.
            try:
                self._compiled = _compile(self.path, self.device, self.config)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    f"{BOUNDARY_TFLITE_ERROR}\nUnderlying: {e}"
                ) from e
        else:
            self._compiled = _compile(self.path, self.device, self.config)
        self._input = self._compiled.inputs[0]
        self._cls = None
        self._reg = None
        for o in self._compiled.outputs:
            names = o.get_names()
            if "cls_logits" in names:
                self._cls = o
            elif "reg_offsets" in names:
                self._reg = o
        if self._cls is None:
            raise RuntimeError(f"no 'cls_logits' output in {self.path}")

    def forward_preprocessed(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """``(cls_logits[500, 4], reg_offsets[500, 6])`` for one window that is
        already rank-normalized (the ONNX ``signal`` input, shape ``(2500,)``)."""
        x = np.asarray(x, dtype=np.float32).reshape(1, -1)
        rq = self._compiled.create_infer_request()
        rq.infer({self._input: x})
        cls = rq.get_tensor(self._cls).data.copy()[0]
        reg = rq.get_tensor(self._reg).data.copy()[0] if self._reg is not None else None
        return cls, reg

    def forward_window(self, window: np.ndarray) -> np.ndarray:
        """Return (N_FRAMES, 4) cls logits for a single 2500-sample window."""
        return self.forward_preprocessed(preprocess_window(window))[0]

    def forward_window_cls_reg(self, window: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """``(cls_logits, reg_offsets)`` for a single raw 2500-sample 250 Hz window."""
        return self.forward_preprocessed(preprocess_window(window))


__all__ = [
    "OpenVINOBoundary",
    "BOUNDARY_TFLITE_ERROR",
    "F32_HINT",
    "available_devices",
    "device_full_names",
    "boundary_onnx_path",
    "fetch_v56c_fp32_onnx",
    "V56C_ONNX_URL",
    "V56C_ONNX_SHA256",
    "V56C_ONNX_ENV",
]
