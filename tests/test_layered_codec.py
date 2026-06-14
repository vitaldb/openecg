import numpy as np
import pytest

from openecg import eval as ee
from openecg.layered import (
    BEAT_NONE,
    BEAT_SINUS,
    LayeredCodec,
    LayeredPredictor,
    RHYTHM_SINUS,
    encode,
    encode_stream,
    frames_to_events,
)


def test_layered_codec_validates_shape():
    with pytest.raises(ValueError, match="2-D"):
        LayeredCodec(fs=250, channels=np.zeros(10, dtype=np.uint8))


def test_layered_codec_validates_fs():
    with pytest.raises(ValueError, match="fs must be positive"):
        LayeredCodec(fs=0, channels=np.zeros((3, 10), dtype=np.uint8))


def test_layered_codec_coerces_channels_to_uint8():
    c = LayeredCodec(fs=250, channels=np.zeros((3, 10), dtype=np.int64))
    assert c.channels.dtype == np.uint8


def test_frames_to_events():
    frames = np.array([0, 0, 1, 1, 1, 0], dtype=np.uint8)
    assert frames_to_events(frames, frame_ms=20) == [(0, 40), (1, 60), (0, 20)]


class CountingPredictor:
    def __init__(self):
        self.calls = 0

    def encode(self, signal, fs=250):
        self.calls += 1
        n = len(signal)
        channels = np.zeros((3, n), dtype=np.uint8)
        channels[0, n // 4:n // 2] = ee.SUPER_QRS
        channels[1, n // 4:n // 2] = BEAT_SINUS
        channels[2, :] = RHYTHM_SINUS
        return LayeredCodec(fs=fs, channels=channels)


def test_encode_stream_accepts_prebuilt_predictor_without_wrapping():
    predictor = CountingPredictor()
    sig = np.zeros(20 * 250, dtype=np.float32)
    c = encode_stream(sig, fs=250, model=predictor)
    assert c.channels.shape == (3, sig.size)
    assert c.eval_margin_s == 0
    assert predictor.calls > 1


def test_layered_predictor_is_reused_across_stream_windows(monkeypatch):
    class DummyModel:
        def eval(self):
            return self

        def to(self, device):
            self.device = device
            return self

        def parameters(self):
            return iter(())

    created = []

    class DummyPredictor(LayeredPredictor):
        def __init__(self, *args, **kwargs):
            created.append((args, kwargs))

        def encode(self, signal, fs=250):
            n = len(signal)
            return LayeredCodec(
                fs=fs,
                channels=np.stack(
                    [
                        np.zeros(n, dtype=np.uint8),
                        np.full(n, BEAT_NONE, dtype=np.uint8),
                        np.full(n, RHYTHM_SINUS, dtype=np.uint8),
                    ],
                    axis=0,
                ),
            )

    monkeypatch.setattr("openecg.layered.LayeredPredictor", DummyPredictor)
    sig = np.zeros(20 * 250, dtype=np.float32)
    encode_stream(sig, fs=250, model=DummyModel(), device="cpu")
    assert len(created) == 1


def test_encode_loads_string_model_reference_lazily(monkeypatch):
    calls = []
    predictor = CountingPredictor()

    def fake_load_codec(ckpt=None, device="cpu"):
        calls.append((ckpt, device))
        return predictor

    monkeypatch.setattr("openecg.layered.load_codec", fake_load_codec)
    sig = np.zeros(2500, dtype=np.float32)
    c = encode(sig, fs=250, model="default", device="cpu")
    assert c.channels.shape == (3, sig.size)
    assert calls == [(None, "cpu")]


def test_encode_stream_loads_string_model_reference_once(monkeypatch):
    calls = []
    predictor = CountingPredictor()

    def fake_load_codec(ckpt=None, device="cpu"):
        calls.append((ckpt, device))
        return predictor

    monkeypatch.setattr("openecg.layered.load_codec", fake_load_codec)
    sig = np.zeros(5000, dtype=np.float32)
    c = encode_stream(sig, fs=250, model="custom.pt", device="cuda")
    assert c.channels.shape == (3, sig.size)
    assert calls == [("custom.pt", "cuda")]
    assert predictor.calls > 1


def test_unified_merges_frame_and_beat():
    """`unified` = frame with QRS samples replaced by beat type (8-class)."""
    from openecg.eval import SUPER_OTHER, SUPER_P, SUPER_QRS, SUPER_T
    from openecg.layered import (
        BEAT_VPC, UNIFIED_NAMES, UNIFIED_OTHER, UNIFIED_P, UNIFIED_T,
        UNIFIED_SINUS, UNIFIED_VPC, UNIFIED_UNKNOWN,
    )
    # frame: other P QRS T ; beat: sinus on first QRS, vpc on second
    frame = np.array([SUPER_OTHER, SUPER_P, SUPER_QRS, SUPER_T,
                      SUPER_OTHER, SUPER_QRS], dtype=np.uint8)
    beat = np.array([BEAT_NONE, BEAT_NONE, BEAT_SINUS, BEAT_NONE,
                     BEAT_NONE, BEAT_VPC], dtype=np.uint8)
    rhythm = np.full(6, RHYTHM_SINUS, dtype=np.uint8)
    c = LayeredCodec(fs=250, channels=np.stack([frame, beat, rhythm]))
    u = c.unified
    assert u.dtype == np.uint8 and u.shape == (6,)
    assert list(u) == [UNIFIED_OTHER, UNIFIED_P, UNIFIED_SINUS, UNIFIED_T,
                       UNIFIED_OTHER, UNIFIED_VPC]
    assert UNIFIED_NAMES[UNIFIED_VPC] == "vpc"


def test_unified_untyped_qrs_falls_back_to_unknown():
    """A QRS sample with no beat type reads as 'unknown', not a wave class."""
    from openecg.eval import SUPER_QRS
    from openecg.layered import UNIFIED_UNKNOWN
    frame = np.array([SUPER_QRS, SUPER_QRS], dtype=np.uint8)
    beat = np.array([BEAT_NONE, BEAT_NONE], dtype=np.uint8)   # QRS but untyped
    rhythm = np.zeros(2, dtype=np.uint8)
    c = LayeredCodec(fs=250, channels=np.stack([frame, beat, rhythm]))
    assert list(c.unified) == [UNIFIED_UNKNOWN, UNIFIED_UNKNOWN]
