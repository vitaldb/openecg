"""Optional OpenVINO backend tests.

Skipped unless ``openvino`` is installed (``pip install "openecg[openvino]"``).
The TFLite comparison additionally needs a TFLite interpreter
(``tflite-runtime`` or ``ai-edge-litert``); the ONNX Runtime comparison needs
``onnxruntime``. The boundary tests need the ``v56c_fp32.onnx`` release asset
(``OPENECG_V56C_ONNX``, ``data/deploy/`` or the fetch cache) and are skipped
without it. Everything runs on the ``CPU`` device.
"""
import sys

import numpy as np
import pytest

ov = pytest.importorskip("openvino")

from openecg import openvino_backend as ovb  # noqa: E402

F32 = {"INFERENCE_PRECISION_HINT": "f32"}


def _synth_sinus(fs: int, seconds: float = 10.0, rr_s: float = 0.75, seed: int = 0) -> np.ndarray:
    """Deterministic synthetic sinus ECG (P / QRS / T bumps + wander + noise)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    t = np.arange(n) / fs
    sig = 0.03 * np.sin(2 * np.pi * 0.25 * t) + rng.normal(0, 0.008, n)
    qrs = np.hanning(max(3, int(0.08 * fs)))
    pw = 0.25 * np.hanning(max(3, int(0.10 * fs)))
    tw = 0.35 * np.hanning(max(3, int(0.18 * fs)))
    c = int(rr_s * fs)
    while c + int(0.16 * fs) + len(tw) < n:
        lo = c - len(qrs) // 2
        sig[lo:lo + len(qrs)] += qrs
        p0 = c - int(0.18 * fs)
        if p0 >= 0:
            sig[p0:p0 + len(pw)] += pw
        t0 = c + int(0.16 * fs)
        sig[t0:t0 + len(tw)] += tw
        c += int(rr_s * fs)
    return sig.astype(np.float32)


def test_import_error_names_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "openvino", None)
    with pytest.raises(ImportError, match=r"openecg\[openvino\]"):
        ovb._import_openvino()


def _boundary_onnx():
    try:
        return ovb.boundary_onnx_path()
    except FileNotFoundError:
        pytest.skip("v56c_fp32.onnx release asset not available")


def test_boundary_onnx_path_env(tmp_path, monkeypatch):
    f = tmp_path / "v56c_fp32.onnx"
    f.write_bytes(b"x")
    monkeypatch.setenv(ovb.V56C_ONNX_ENV, str(f))
    assert ovb.boundary_onnx_path() == f


def test_boundary_onnx_path_missing_names_fetch(tmp_path, monkeypatch):
    monkeypatch.delenv(ovb.V56C_ONNX_ENV, raising=False)
    monkeypatch.setenv("OPENECG_CACHE", str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="fetch_v56c_fp32_onnx"):
        ovb.boundary_onnx_path()


def test_fetch_rejects_sha256_mismatch(tmp_path):
    src = tmp_path / "bad.onnx"
    src.write_bytes(b"not the release asset")
    dest = tmp_path / "out" / "v56c_fp32.onnx"
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        ovb.fetch_v56c_fp32_onnx(dest, url=src.as_uri())
    assert not dest.exists()
    assert not list(dest.parent.glob("*.part"))


def test_cpu_defaults_to_f32_and_config_overrides():
    # The OpenVINO CPU plugin would pick bf16 on CPUs with native bf16 support.
    path = _boundary_onnx()
    default = ovb.OpenVINOBoundary(path, device="CPU")
    assert default.config == F32
    assert default._compiled.get_property("INFERENCE_PRECISION_HINT") == ov.Type.f32
    assert ovb.OpenVINOBoundary(path, device="CPU", config={}).config == {}


def test_make_static_pins_boundary_batch():
    model = ov.Core().read_model(str(_boundary_onnx()))
    assert any(i.get_partial_shape().is_dynamic for i in model.inputs)
    ovb._make_static(model)
    assert all(i.get_partial_shape().is_static for i in model.inputs)
    assert list(model.inputs[0].get_shape()) == [1, 2500]
    assert all(o.get_partial_shape().is_static for o in model.outputs)


def test_boundary_cpu_matches_onnxruntime():
    ort = pytest.importorskip("onnxruntime")
    from openecg.deploy import preprocess_window

    path = _boundary_onnx()
    x = preprocess_window(_synth_sinus(250))
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    r_cls, r_reg = sess.run(["cls_logits", "reg_offsets"], {"signal": x[None, :]})
    cls, reg = ovb.OpenVINOBoundary(path, device="CPU").forward_preprocessed(x)
    assert cls.shape == (500, 4) and reg.shape == (500, 6)
    assert float(np.abs(cls - r_cls[0]).max()) < 1e-4
    assert float(np.abs(reg - r_reg[0]).max()) < 1e-4
    assert np.array_equal(cls.argmax(-1), r_cls[0].argmax(-1))


def test_boundary_frame_argmax_vs_tflite():
    from openecg.deploy import N_FRAMES, Inference

    try:
        det = Inference()
    except ImportError:
        pytest.skip("no TFLite interpreter (tflite-runtime / ai-edge-litert)")
    ob = ovb.OpenVINOBoundary(_boundary_onnx(), device="CPU")

    sig = _synth_sinus(250)
    ref = det.forward_window(sig).argmax(-1)
    got = ob.forward_window(sig).argmax(-1)
    assert got.shape == ref.shape == (N_FRAMES,)
    assert float(np.mean(got == ref)) >= 0.99
