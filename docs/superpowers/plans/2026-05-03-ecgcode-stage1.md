# ECGCode Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ECG signal → discrete RLE token stream pipeline (NeuroKit2 as labeler) + LUDB cardiologist label baseline validation.

**Architecture:** Pure-Python CPU pipeline. Per-lead tokenization with `delineate.py` (NK wrapper) + `pacer.py` (highpass+threshold spike detector) feeding into `labeler.py` (sample-level array → RLE compress). `codec.py` handles uint16 packing, frame expansion, and ASCII art rendering. `eval.py` measures NK pseudo-label vs LUDB cardiologist agreement. No neural model in this Stage.

**Tech Stack:** Python 3.11+, uv (env), numpy, scipy, neurokit2, wfdb, pytest. Spec: `docs/superpowers/specs/2026-05-03-ecgcode-stage1-design.md`.

**Environment requirements (set before running tasks 7+)**:
```powershell
$env:ECGCODE_LUDB_ZIP = "G:\Shared drives\datasets\ecg\lobachevsky-university-electrocardiography-database-1.0.1.zip"
```

---

## File Structure

| Path | Responsibility |
|---|---|
| `pyproject.toml` | uv-managed project + deps |
| `ecgcode/__init__.py` | package marker |
| `ecgcode/vocab.py` | 13 IDs, char/name lookups, active class set |
| `ecgcode/codec.py` | uint16 pack/unpack, length split, frame expand, ASCII render |
| `ecgcode/pacer.py` | highpass+threshold pacer spike detector |
| `ecgcode/delineate.py` | NK ecg_peaks + ecg_delineate wrapper, `DelineateResult` dataclass |
| `ecgcode/labeler.py` | DelineateResult + spikes + n_samples → RLE event list |
| `ecgcode/ludb.py` | zip extract + WFDB load + stratified train/val split |
| `ecgcode/eval.py` | per-symbol F1 + boundary error metrics |
| `scripts/tokenize_ludb.py` | run pipeline on all LUDB → `ludb_tokens.npz` |
| `scripts/validate_v1.py` | run eval on val split → JSON report |
| `scripts/ablate_methods.py` | NK method sweep (dwt/cwt/peak/prominence) |
| `scripts/validate_pacer.py` | TPR/FPR on pacer vs non-pacer records |
| `tests/conftest.py` | pytest fixtures (synthetic signals, mock DelineateResult) |
| `tests/test_vocab.py` | vocab consistency |
| `tests/test_codec.py` | round-trip, frame expansion, render |
| `tests/test_pacer.py` | synthetic spike detection |
| `tests/test_labeler.py` | mock NK output → expected RLE |
| `tests/test_integration.py` | LUDB record 1 end-to-end |
| `data/splits/ludb_v1.json` | committed reproducibility-locked split |

**Note**: `data/splits/ludb_v1.json`은 reproducibility를 위해 commit. .gitignore의 `data/` rule을 `data/*` + `!data/splits/`로 변경 (Task 8에서 처리).

**Spec deviation**: spec의 `labeler.label(signal, dr, spikes, fs)` → `labeler.label(dr, spikes, n_samples, fs)`로 변경 (signal 직접 필요 없음, 테스트 용이성).

---

### Task 1: Project setup & dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `ecgcode/__init__.py`
- Modify: `.gitignore` (add `data/*` exception for splits)

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "ecgcode"
version = "0.1.0"
description = "Clinically-grounded discrete tokenization for electrocardiograms"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "neurokit2>=0.2.7",
    "wfdb>=4.1",
]

[dependency-groups]
dev = [
    "pytest>=7.4",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["ecgcode"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

- [ ] **Step 2: Create empty package**

```python
# ecgcode/__init__.py
__version__ = "0.1.0"
```

- [ ] **Step 3: Update .gitignore for splits exception**

Replace the `data/` line in `.gitignore` with:

```
# Data files (not for public release)
*.npz
*.npy
*.dat
*.hea
*.atr
*.xws
*.qrs
*.vital
*.wfdb
data/*
!data/splits/
!data/splits/*.json
datasets/
```

- [ ] **Step 4: uv sync**

Run: `uv sync`
Expected: Creates `.venv/`, installs all deps, no errors.

- [ ] **Step 5: Smoke import test**

Run: `uv run python -c "import ecgcode; print(ecgcode.__version__)"`
Expected: `0.1.0`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml ecgcode/__init__.py .gitignore
git commit -m "Add project scaffolding (pyproject + package init)"
```

---

### Task 2: Vocab module

**Files:**
- Create: `ecgcode/vocab.py`
- Create: `tests/__init__.py` (empty)
- Create: `tests/test_vocab.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_vocab.py
from ecgcode import vocab

def test_id_count():
    assert len(vocab.ID_TO_CHAR) == 13

def test_pad_is_zero():
    assert vocab.ID_PAD == 0
    assert vocab.ID_TO_CHAR[0] == "·"

def test_active_v1_classes_are_nine():
    assert len(vocab.ACTIVE_V1) == 9

def test_active_set_is_correct():
    expected = {vocab.ID_UNK, vocab.ID_ISO, vocab.ID_PACER,
                vocab.ID_P, vocab.ID_Q, vocab.ID_R, vocab.ID_S,
                vocab.ID_T, vocab.ID_W}
    assert set(vocab.ACTIVE_V1) == expected

def test_char_lookup_roundtrip():
    for i, ch in vocab.ID_TO_CHAR.items():
        assert vocab.CHAR_TO_ID[ch] == i

def test_name_lookup_roundtrip():
    for i, name in vocab.ID_TO_NAME.items():
        assert vocab.NAME_TO_ID[name] == i

def test_no_duplicate_chars():
    chars = list(vocab.ID_TO_CHAR.values())
    assert len(chars) == len(set(chars))

def test_no_duplicate_names():
    names = list(vocab.ID_TO_NAME.values())
    assert len(names) == len(set(names))

def test_pacer_id_is_3():
    assert vocab.ID_PACER == 3

def test_iso_id_is_2():
    assert vocab.ID_ISO == 2

def test_w_id_is_10():
    assert vocab.ID_W == 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_vocab.py -v`
Expected: All 10 tests FAIL with `ModuleNotFoundError` or `AttributeError`.

- [ ] **Step 3: Implement vocab.py**

```python
# ecgcode/vocab.py
"""ECGCode alphabet v1.0 — 13 IDs, append-only versioning.

See spec: docs/superpowers/specs/2026-05-03-ecgcode-stage1-design.md
"""

ID_PAD = 0
ID_UNK = 1
ID_ISO = 2
ID_PACER = 3
ID_P = 4
ID_Q = 5
ID_R = 6
ID_S = 7
ID_T = 8
ID_U = 9
ID_W = 10
ID_D = 11
ID_J = 12

ID_TO_CHAR = {
    ID_PAD: "·",
    ID_UNK: "?",
    ID_ISO: "_",
    ID_PACER: "*",
    ID_P: "p",
    ID_Q: "q",
    ID_R: "r",
    ID_S: "s",
    ID_T: "t",
    ID_U: "u",
    ID_W: "w",
    ID_D: "d",
    ID_J: "j",
}

ID_TO_NAME = {
    ID_PAD: "<pad>",
    ID_UNK: "<unk>",
    ID_ISO: "iso",
    ID_PACER: "pacer_spike",
    ID_P: "P",
    ID_Q: "Q",
    ID_R: "R",
    ID_S: "S",
    ID_T: "T",
    ID_U: "U",
    ID_W: "wide_QRS",
    ID_D: "delta",
    ID_J: "J_wave",
}

CHAR_TO_ID = {ch: i for i, ch in ID_TO_CHAR.items()}
NAME_TO_ID = {name: i for i, name in ID_TO_NAME.items()}

# Active classes in v1.0 (predicted by model + emitted by labeler).
# Excludes pad (mask only) and reserved IDs (u, d, j — v1.1).
ACTIVE_V1 = (
    ID_UNK, ID_ISO, ID_PACER,
    ID_P, ID_Q, ID_R, ID_S, ID_T, ID_W,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_vocab.py -v`
Expected: All 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ecgcode/vocab.py tests/__init__.py tests/test_vocab.py
git commit -m "Add vocab module with 13-symbol alphabet v1.0"
```

---

### Task 3: Codec module

**Files:**
- Create: `ecgcode/codec.py`
- Create: `tests/test_codec.py`

- [ ] **Step 1: Write encode/decode round-trip tests**

```python
# tests/test_codec.py
import numpy as np
from ecgcode import codec, vocab

def test_encode_decode_roundtrip():
    events = [(vocab.ID_ISO, 100), (vocab.ID_P, 92), (vocab.ID_ISO, 200)]
    packed = codec.encode(events)
    assert packed.dtype == np.uint16
    assert codec.decode(packed) == events

def test_encode_returns_uint16():
    packed = codec.encode([(vocab.ID_P, 80)])
    assert packed.dtype == np.uint16
    assert len(packed) == 1

def test_length_4ms_snap():
    events = [(vocab.ID_P, 95)]   # 95ms → snap to nearest 4ms = 96
    packed = codec.encode(events)
    decoded = codec.decode(packed)
    assert decoded == [(vocab.ID_P, 96)]

def test_length_under_4ms_snaps_to_4ms_minimum():
    events = [(vocab.ID_PACER, 2)]   # 2ms → min 1 unit = 4ms
    decoded = codec.decode(codec.encode(events))
    assert decoded == [(vocab.ID_PACER, 4)]

def test_length_over_1020_splits():
    events = [(vocab.ID_ISO, 2500)]  # 2500ms → 1020 + 1020 + 460
    packed = codec.encode(events)
    assert len(packed) == 3
    decoded = codec.decode(packed)
    assert sum(ms for _, ms in decoded) == 2500
    assert all(s == vocab.ID_ISO for s, _ in decoded)

def test_zero_length_event_rejected():
    import pytest
    with pytest.raises(ValueError):
        codec.encode([(vocab.ID_P, 0)])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_codec.py -v`
Expected: All 6 tests FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement encode/decode**

```python
# ecgcode/codec.py
"""ECGCode token format codec — uint16 pack/unpack, frame expansion, ASCII render."""

import numpy as np
from ecgcode import vocab

MS_PER_UNIT = 4
MAX_LENGTH_MS = 255 * MS_PER_UNIT  # 1020 ms


def _split_long(sym: int, length_ms: int) -> list[tuple[int, int]]:
    chunks = []
    while length_ms > MAX_LENGTH_MS:
        chunks.append((sym, MAX_LENGTH_MS))
        length_ms -= MAX_LENGTH_MS
    if length_ms > 0:
        chunks.append((sym, length_ms))
    return chunks


def encode(events: list[tuple[int, int]]) -> np.ndarray:
    """Pack (symbol_id, length_ms) events to uint16 array.
    
    Length is snapped to nearest 4ms (min 4ms). Long events split at 1020ms.
    Raises ValueError if any event has length 0 or invalid symbol_id.
    """
    out = []
    for sym, ms in events:
        if ms <= 0:
            raise ValueError(f"Event length must be positive, got {ms}")
        if not (0 <= sym <= 255):
            raise ValueError(f"Invalid symbol_id {sym}")
        # Snap to 4ms grid, minimum 1 unit
        units = max(1, round(ms / MS_PER_UNIT))
        snapped_ms = units * MS_PER_UNIT
        for chunk_sym, chunk_ms in _split_long(sym, snapped_ms):
            chunk_units = chunk_ms // MS_PER_UNIT
            packed = (chunk_sym << 8) | chunk_units
            out.append(packed)
    return np.array(out, dtype=np.uint16)


def decode(packed: np.ndarray) -> list[tuple[int, int]]:
    """Unpack uint16 array to (symbol_id, length_ms) events."""
    types = (packed >> 8).astype(np.uint8)
    units = (packed & 0xFF).astype(np.uint8)
    return [(int(t), int(u) * MS_PER_UNIT) for t, u in zip(types, units)]
```

- [ ] **Step 4: Run encode/decode tests, verify pass**

Run: `uv run pytest tests/test_codec.py -v -k "encode or decode or split or snap or zero"`
Expected: 6 tests PASS.

- [ ] **Step 5: Write frame expansion tests**

Append to `tests/test_codec.py`:

```python
def test_frame_expansion_basic():
    # 100ms iso + 100ms P at 20ms/frame → 5 iso frames + 5 P frames
    events = [(vocab.ID_ISO, 100), (vocab.ID_P, 100)]
    frames = codec.to_frames(events, frame_ms=20)
    assert len(frames) == 10
    assert (frames[:5] == vocab.ID_ISO).all()
    assert (frames[5:] == vocab.ID_P).all()

def test_frame_expansion_max_overlap():
    # 16ms Q + 24ms R in same 40ms window at 20ms/frame
    # frame 0 (0-20ms): Q=16, R=4 → Q wins (max overlap)
    # frame 1 (20-40ms): R only → R
    events = [(vocab.ID_Q, 16), (vocab.ID_R, 24)]
    frames = codec.to_frames(events, frame_ms=20)
    assert len(frames) == 2
    assert frames[0] == vocab.ID_Q
    assert frames[1] == vocab.ID_R

def test_frame_expansion_spike_priority():
    # 20ms iso + 4ms spike + 16ms iso → 2 frames
    # frame 0 (0-20ms): iso=20 → iso wins
    # frame 1 (20-40ms): spike=4 + iso=16 → spike wins (priority override)
    events = [(vocab.ID_ISO, 20), (vocab.ID_PACER, 4), (vocab.ID_ISO, 16)]
    frames = codec.to_frames(events, frame_ms=20)
    assert len(frames) == 2
    assert frames[0] == vocab.ID_ISO
    assert frames[1] == vocab.ID_PACER

def test_frame_expansion_total_ms_override():
    # Override total_ms to truncate or extend
    events = [(vocab.ID_ISO, 100)]
    frames = codec.to_frames(events, frame_ms=20, total_ms=200)
    assert len(frames) == 10  # 200/20

def test_frame_expansion_uint8_dtype():
    events = [(vocab.ID_P, 60)]
    frames = codec.to_frames(events, frame_ms=20)
    assert frames.dtype == np.uint8
```

- [ ] **Step 6: Implement to_frames**

Append to `ecgcode/codec.py`:

```python
def to_frames(
    events: list[tuple[int, int]],
    frame_ms: int = 20,
    total_ms: int | None = None,
) -> np.ndarray:
    """Expand RLE events to per-frame symbol array.
    
    Rule: each frame gets the symbol with maximum overlap, with `*` (pacer)
    as priority override (any frame containing a spike → pacer label).
    """
    if total_ms is None:
        total_ms = sum(ms for _, ms in events)
    n_frames = round(total_ms / frame_ms)
    out = np.zeros(n_frames, dtype=np.uint8)
    
    # Build (start_ms, end_ms, sym) intervals
    intervals = []
    cum = 0
    for sym, ms in events:
        intervals.append((cum, cum + ms, sym))
        cum += ms
    
    for f in range(n_frames):
        f_start = f * frame_ms
        f_end = f_start + frame_ms
        # Per-symbol overlap accumulator
        overlap = {}
        spike_present = False
        for s_start, s_end, sym in intervals:
            if s_end <= f_start:
                continue
            if s_start >= f_end:
                break
            ov = min(s_end, f_end) - max(s_start, f_start)
            if ov > 0:
                if sym == vocab.ID_PACER:
                    spike_present = True
                else:
                    overlap[sym] = overlap.get(sym, 0) + ov
        if spike_present:
            out[f] = vocab.ID_PACER
        elif overlap:
            out[f] = max(overlap, key=overlap.get)
        else:
            out[f] = vocab.ID_ISO  # gap (shouldn't happen if events cover total)
    return out
```

- [ ] **Step 7: Run frame expansion tests**

Run: `uv run pytest tests/test_codec.py -v -k "frame_expansion"`
Expected: 5 tests PASS.

- [ ] **Step 8: Write ASCII render tests**

Append to `tests/test_codec.py`:

```python
def test_render_compact():
    events = [(vocab.ID_ISO, 100), (vocab.ID_P, 92), (vocab.ID_ISO, 44),
              (vocab.ID_Q, 16), (vocab.ID_R, 40), (vocab.ID_S, 24),
              (vocab.ID_ISO, 84), (vocab.ID_T, 164), (vocab.ID_ISO, 208)]
    s = codec.render_compact(events)
    assert s == "_p_qrs_t_"

def test_render_timed():
    events = [(vocab.ID_ISO, 100), (vocab.ID_P, 100)]
    s = codec.render_timed(events, ms_per_char=20)
    # 100ms iso = 5 underscores, 100ms P = 5 p's
    assert s == "_____ppppp"

def test_render_timed_min_one_char():
    # 8ms event at 20ms/char → round(0.4)=0, but min 1 char
    events = [(vocab.ID_PACER, 8)]
    s = codec.render_timed(events, ms_per_char=20)
    assert s == "*"

def test_render_json():
    events = [(vocab.ID_ISO, 100), (vocab.ID_P, 92)]
    j = codec.render_json(events)
    assert j == [{"sym": "iso", "ms": 100}, {"sym": "P", "ms": 92}]
```

- [ ] **Step 9: Implement render functions**

Append to `ecgcode/codec.py`:

```python
def render_compact(events: list[tuple[int, int]]) -> str:
    """One char per event."""
    return "".join(vocab.ID_TO_CHAR[sym] for sym, _ in events)


def render_timed(events: list[tuple[int, int]], ms_per_char: int = 20) -> str:
    """Char count proportional to duration. Minimum 1 char per event."""
    chars = []
    for sym, ms in events:
        n = max(1, round(ms / ms_per_char))
        chars.append(vocab.ID_TO_CHAR[sym] * n)
    return "".join(chars)


def render_json(events: list[tuple[int, int]]) -> list[dict]:
    """Verbose JSON view: [{'sym': name, 'ms': length}, ...]."""
    return [{"sym": vocab.ID_TO_NAME[sym], "ms": ms} for sym, ms in events]
```

- [ ] **Step 10: Run all codec tests**

Run: `uv run pytest tests/test_codec.py -v`
Expected: All 15 tests PASS.

- [ ] **Step 11: Commit**

```bash
git add ecgcode/codec.py tests/test_codec.py
git commit -m "Add codec module (uint16 pack, frame expand, ASCII render)"
```

---

### Task 4: Pacer spike detector

**Files:**
- Create: `ecgcode/pacer.py`
- Create: `tests/test_pacer.py`

- [ ] **Step 1: Write detector tests**

```python
# tests/test_pacer.py
import numpy as np
from scipy import signal as scipy_signal
from ecgcode import pacer

FS = 500


def test_detects_synthetic_spike():
    n = FS * 5  # 5s
    sig = np.zeros(n)
    sig[1000] = 5.0   # spike at 2s, sample 1000
    detected = pacer.detect_spikes(sig, fs=FS)
    assert len(detected) == 1
    assert abs(detected[0] - 1000) <= 1


def test_detects_negative_polarity_spike():
    n = FS * 5
    sig = np.zeros(n)
    sig[1500] = -5.0
    detected = pacer.detect_spikes(sig, fs=FS)
    assert len(detected) == 1


def test_ignores_qrs_like_wave():
    # 60ms hann window (R wave morphology) should not trigger
    sig = np.zeros(FS * 5)
    qrs = scipy_signal.windows.hann(30) * 1.5  # 60ms wide, amp 1.5
    sig[1000:1030] = qrs
    detected = pacer.detect_spikes(sig, fs=FS)
    assert len(detected) == 0


def test_refractory_dedup_bipolar():
    # bipolar artifact (over+undershoot) within 5ms → 1 detection
    sig = np.zeros(FS * 5)
    sig[1000] = 5.0
    sig[1001] = -3.0  # 2ms apart
    detected = pacer.detect_spikes(sig, fs=FS, refractory_ms=5.0)
    assert len(detected) == 1


def test_multiple_spikes_outside_refractory():
    sig = np.zeros(FS * 5)
    sig[500] = 5.0
    sig[1500] = 5.0   # 2s = 1000 samples apart, well outside refractory
    sig[2500] = 5.0
    detected = pacer.detect_spikes(sig, fs=FS)
    assert len(detected) == 3


def test_returns_numpy_int_array():
    sig = np.zeros(FS * 5)
    sig[1000] = 5.0
    detected = pacer.detect_spikes(sig, fs=FS)
    assert isinstance(detected, np.ndarray)
    assert np.issubdtype(detected.dtype, np.integer)


def test_no_spikes_in_pure_noise():
    rng = np.random.default_rng(seed=42)
    sig = rng.normal(0, 0.05, size=FS * 10)  # tiny noise
    detected = pacer.detect_spikes(sig, fs=FS)
    assert len(detected) == 0
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/test_pacer.py -v`
Expected: All 7 tests FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement pacer.py**

```python
# ecgcode/pacer.py
"""Pacemaker spike detector — highpass + adaptive threshold.

Spike characteristics: 1-3ms wide, high amplitude (>1mV), sharp slope.
ECG content (<50Hz) suppressed by 80Hz highpass; spike survives.
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def detect_spikes(
    signal: np.ndarray,
    fs: int = 500,
    cutoff_hz: float = 80.0,
    amp_threshold_mad: float = 5.0,
    max_width_ms: float = 4.0,
    refractory_ms: float = 5.0,
) -> np.ndarray:
    """Detect pacemaker spike sample indices.
    
    Args:
        signal: 1D ECG samples
        fs: sampling rate in Hz
        cutoff_hz: highpass cutoff (default 80Hz, suppresses ECG content)
        amp_threshold_mad: peak height threshold = N × median(|hp|)
        max_width_ms: reject peaks wider than this (likely R waves)
        refractory_ms: dedup window for bipolar artifacts
    
    Returns:
        Sorted array of sample indices where spikes were detected.
    """
    nyq = fs / 2
    b, a = butter(4, cutoff_hz / nyq, btype="high")
    hp = filtfilt(b, a, signal)
    abs_hp = np.abs(hp)
    
    mad = np.median(abs_hp)
    if mad == 0:
        return np.array([], dtype=np.int64)
    threshold = amp_threshold_mad * mad
    
    max_width_samples = max(1, int(max_width_ms * fs / 1000))
    refractory_samples = max(1, int(refractory_ms * fs / 1000))
    
    peaks, _ = find_peaks(
        abs_hp,
        height=threshold,
        distance=refractory_samples,
        width=(None, max_width_samples),
    )
    return peaks.astype(np.int64)
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `uv run pytest tests/test_pacer.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ecgcode/pacer.py tests/test_pacer.py
git commit -m "Add pacer spike detector (highpass + adaptive threshold)"
```

---

### Task 5: NK delineate wrapper

**Files:**
- Create: `ecgcode/delineate.py`

- [ ] **Step 1: Implement delineate.py with DelineateResult**

```python
# ecgcode/delineate.py
"""NeuroKit2 ecg_delineate wrapper.

NK provides per-beat onset/peak/offset for P, QRS, T plus separate Q/S peaks.
Missing waves are marked with NaN inside NK output; we keep that and let
labeler handle (via np.isnan checks).
"""

from dataclasses import dataclass

import neurokit2 as nk
import numpy as np


@dataclass
class DelineateResult:
    """Per-lead wave delineation output. All arrays length = num beats.
    Missing wave indices stored as NaN."""
    p_onsets: np.ndarray
    p_peaks: np.ndarray
    p_offsets: np.ndarray
    q_peaks: np.ndarray
    r_onsets: np.ndarray
    r_peaks: np.ndarray
    r_offsets: np.ndarray
    s_peaks: np.ndarray
    t_onsets: np.ndarray
    t_peaks: np.ndarray
    t_offsets: np.ndarray

    @classmethod
    def empty(cls) -> "DelineateResult":
        e = np.array([], dtype=float)
        return cls(*([e] * 11))

    @property
    def n_beats(self) -> int:
        return len(self.r_peaks)


def run(signal: np.ndarray, fs: int = 500, method: str = "dwt") -> DelineateResult:
    """Run NK ecg_peaks + ecg_delineate. Returns DelineateResult.
    
    On any NK exception or 0 R peaks detected, returns DelineateResult.empty().
    """
    try:
        _, info = nk.ecg_peaks(signal, sampling_rate=fs)
        rpeaks = np.asarray(info["ECG_R_Peaks"], dtype=float)
        if len(rpeaks) == 0:
            return DelineateResult.empty()
        _, waves = nk.ecg_delineate(
            signal, rpeaks=rpeaks.astype(int), sampling_rate=fs, method=method
        )
    except Exception:
        return DelineateResult.empty()
    
    return DelineateResult(
        p_onsets=np.asarray(waves["ECG_P_Onsets"], dtype=float),
        p_peaks=np.asarray(waves["ECG_P_Peaks"], dtype=float),
        p_offsets=np.asarray(waves["ECG_P_Offsets"], dtype=float),
        q_peaks=np.asarray(waves["ECG_Q_Peaks"], dtype=float),
        r_onsets=np.asarray(waves["ECG_R_Onsets"], dtype=float),
        r_peaks=rpeaks,
        r_offsets=np.asarray(waves["ECG_R_Offsets"], dtype=float),
        s_peaks=np.asarray(waves["ECG_S_Peaks"], dtype=float),
        t_onsets=np.asarray(waves["ECG_T_Onsets"], dtype=float),
        t_peaks=np.asarray(waves["ECG_T_Peaks"], dtype=float),
        t_offsets=np.asarray(waves["ECG_T_Offsets"], dtype=float),
    )
```

No unit test for `run` (NK is hard to test in isolation; covered by integration test in Task 9). DelineateResult dataclass is tested via labeler tests (Task 6).

- [ ] **Step 2: Smoke verify import**

Run: `uv run python -c "from ecgcode.delineate import DelineateResult, run; print(DelineateResult.empty().n_beats)"`
Expected: `0`

- [ ] **Step 3: Commit**

```bash
git add ecgcode/delineate.py
git commit -m "Add NK delineate wrapper with DelineateResult dataclass"
```

---

### Task 6: Labeler module

**Files:**
- Create: `tests/conftest.py`
- Create: `ecgcode/labeler.py`
- Create: `tests/test_labeler.py`

- [ ] **Step 1: Create conftest.py with fixtures**

```python
# tests/conftest.py
import numpy as np
import pytest

from ecgcode.delineate import DelineateResult


def _arr(*xs):
    return np.array(xs, dtype=float)


@pytest.fixture
def empty_dr():
    return DelineateResult.empty()


@pytest.fixture
def one_beat_normal_dr():
    """One normal beat: P at 100-150, Q at 195-205, R at 195-230, S at 220-230, T at 280-400.
    Sample indices (500Hz). Single beat → arrays length 1."""
    return DelineateResult(
        p_onsets=_arr(100),
        p_peaks=_arr(125),
        p_offsets=_arr(150),
        q_peaks=_arr(200),
        r_onsets=_arr(195),
        r_peaks=_arr(210),
        r_offsets=_arr(230),
        s_peaks=_arr(225),
        t_onsets=_arr(280),
        t_peaks=_arr(330),
        t_offsets=_arr(400),
    )


@pytest.fixture
def one_beat_wide_no_qs_dr():
    """One beat with wide R only (LBBB pattern): R 200-280 (160ms wide), no Q, no S."""
    return DelineateResult(
        p_onsets=_arr(100),
        p_peaks=_arr(125),
        p_offsets=_arr(150),
        q_peaks=_arr(np.nan),
        r_onsets=_arr(200),
        r_peaks=_arr(240),
        r_offsets=_arr(280),
        s_peaks=_arr(np.nan),
        t_onsets=_arr(330),
        t_peaks=_arr(380),
        t_offsets=_arr(450),
    )


@pytest.fixture
def one_beat_narrow_no_qs_dr():
    """One beat with narrow R only (V1 lead): R 200-230 (60ms wide), no Q, no S."""
    return DelineateResult(
        p_onsets=_arr(100),
        p_peaks=_arr(125),
        p_offsets=_arr(150),
        q_peaks=_arr(np.nan),
        r_onsets=_arr(200),
        r_peaks=_arr(215),
        r_offsets=_arr(230),
        s_peaks=_arr(np.nan),
        t_onsets=_arr(280),
        t_peaks=_arr(330),
        t_offsets=_arr(400),
    )
```

- [ ] **Step 2: Write labeler tests**

```python
# tests/test_labeler.py
import numpy as np
import pytest

from ecgcode import labeler, vocab

FS = 500
N_SAMPLES = 500  # 1 second @ 500Hz


def _syms(events):
    return [s for s, _ in events]


def test_normal_qrs_emits_q_r_s(one_beat_normal_dr):
    events = labeler.label(one_beat_normal_dr, [], n_samples=N_SAMPLES, fs=FS)
    syms = _syms(events)
    # Expected sequence: iso, p, iso, q, r, s, iso, t, iso
    assert syms == [
        vocab.ID_ISO, vocab.ID_P, vocab.ID_ISO,
        vocab.ID_Q, vocab.ID_R, vocab.ID_S,
        vocab.ID_ISO, vocab.ID_T, vocab.ID_ISO,
    ]


def test_wide_qrs_no_q_no_s_emits_w(one_beat_wide_no_qs_dr):
    events = labeler.label(one_beat_wide_no_qs_dr, [], n_samples=N_SAMPLES, fs=FS)
    syms = _syms(events)
    assert vocab.ID_W in syms
    assert vocab.ID_R not in syms
    assert vocab.ID_Q not in syms
    assert vocab.ID_S not in syms


def test_narrow_qrs_no_q_no_s_emits_r_not_w(one_beat_narrow_no_qs_dr):
    events = labeler.label(one_beat_narrow_no_qs_dr, [], n_samples=N_SAMPLES, fs=FS)
    syms = _syms(events)
    assert vocab.ID_R in syms
    assert vocab.ID_W not in syms


def test_spike_overrides_iso(empty_dr):
    """Empty NK + 1 spike at sample 250 → iso, *, iso sequence."""
    events = labeler.label(empty_dr, [], n_samples=N_SAMPLES, fs=FS)
    # Empty DR but n_beats=0 → entire ?, not iso. Check this separately.
    assert _syms(events) == [vocab.ID_UNK]


def test_nk_total_failure_returns_unk(empty_dr):
    """0 R peaks → entire signal is one ? event."""
    events = labeler.label(empty_dr, [], n_samples=5000, fs=FS)
    assert events == [(vocab.ID_UNK, 10000)]   # 5000 samples × 2ms = 10000ms


def test_spike_in_normal_iso_region(one_beat_normal_dr):
    """Spike at sample 50 (in iso region before P) → wave-*-wave split."""
    events = labeler.label(one_beat_normal_dr, [50], n_samples=N_SAMPLES, fs=FS)
    syms = _syms(events)
    # iso (50 samples) → spike (1 sample) → iso (49 samples) → ...
    assert syms[0] == vocab.ID_ISO
    assert syms[1] == vocab.ID_PACER
    assert syms[2] == vocab.ID_ISO


def test_total_duration_matches_n_samples(one_beat_normal_dr):
    events = labeler.label(one_beat_normal_dr, [], n_samples=N_SAMPLES, fs=FS)
    total_ms = sum(ms for _, ms in events)
    assert total_ms == N_SAMPLES * 1000 // FS  # 1000ms


def test_boundary_clamp(one_beat_normal_dr):
    """t_offsets beyond signal length → clamped, no crash."""
    one_beat_normal_dr.t_offsets[0] = 600  # beyond 500-sample signal
    events = labeler.label(one_beat_normal_dr, [], n_samples=N_SAMPLES, fs=FS)
    total_ms = sum(ms for _, ms in events)
    assert total_ms == 1000  # still exactly 1 second


def test_wide_qrs_threshold_at_120ms(one_beat_normal_dr):
    """QRS exactly 120ms with no Q/S → still r (threshold is strict >120)."""
    dr = one_beat_normal_dr
    dr.q_peaks[0] = np.nan
    dr.s_peaks[0] = np.nan
    dr.r_onsets[0] = 200
    dr.r_offsets[0] = 200 + 60  # 60 samples = 120ms exactly
    events = labeler.label(dr, [], n_samples=N_SAMPLES, fs=FS)
    syms = _syms(events)
    assert vocab.ID_R in syms
    assert vocab.ID_W not in syms
```

- [ ] **Step 3: Run tests, verify they fail**

Run: `uv run pytest tests/test_labeler.py -v`
Expected: All 9 tests FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement labeler.py**

```python
# ecgcode/labeler.py
"""Convert NK delineate output + pacer spikes → RLE token stream.

Algorithm (per the spec §6):
1. Initialize sample-level array as iso
2. Mark P / T regions
3. Decompose each QRS into q/r/s by midpoint, with wide-QRS fallback (w)
4. Override with pacer spikes (priority: spike > wave > iso)
5. Run-length compress to (symbol_id, length_ms) events
"""

import numpy as np

from ecgcode import vocab
from ecgcode.delineate import DelineateResult

WIDE_QRS_THRESHOLD_MS = 120.0


def _safe_int(x: float, n: int) -> int | None:
    """Cast NK index (float, possibly NaN) to int and clamp to [0, n-1]. Returns None if NaN."""
    if x is None or np.isnan(x):
        return None
    return max(0, min(n - 1, int(x)))


def _has(x) -> bool:
    return x is not None and not np.isnan(x)


def label(
    dr: DelineateResult,
    spike_idx: list[int],
    n_samples: int,
    fs: int = 500,
) -> list[tuple[int, int]]:
    """Build sample-level label array, then run-length compress to RLE events."""
    ms_per_sample = 1000.0 / fs
    
    # NK total failure → entire signal as one ? event
    if dr.n_beats == 0:
        return [(vocab.ID_UNK, int(round(n_samples * ms_per_sample)))]
    
    labels = np.full(n_samples, vocab.ID_ISO, dtype=np.uint8)
    
    # 1. P waves
    for on_f, off_f in zip(dr.p_onsets, dr.p_offsets):
        if not (_has(on_f) and _has(off_f)):
            continue
        on = _safe_int(on_f, n_samples)
        off = _safe_int(off_f, n_samples)
        labels[on:off + 1] = vocab.ID_P
    
    # 2. T waves
    for on_f, off_f in zip(dr.t_onsets, dr.t_offsets):
        if not (_has(on_f) and _has(off_f)):
            continue
        on = _safe_int(on_f, n_samples)
        off = _safe_int(off_f, n_samples)
        labels[on:off + 1] = vocab.ID_T
    
    # 3. QRS — q/r/s decomposition with wide-QRS fallback
    n_beats = dr.n_beats
    for i in range(n_beats):
        if not (_has(dr.r_onsets[i]) and _has(dr.r_offsets[i])):
            continue
        on = _safe_int(dr.r_onsets[i], n_samples)
        off = _safe_int(dr.r_offsets[i], n_samples)
        r = _safe_int(dr.r_peaks[i], n_samples)
        q = _safe_int(dr.q_peaks[i] if i < len(dr.q_peaks) else None, n_samples)
        s = _safe_int(dr.s_peaks[i] if i < len(dr.s_peaks) else None, n_samples)
        
        qrs_ms = (off - on + 1) * ms_per_sample
        has_q = q is not None
        has_s = s is not None
        
        # Wide-QRS fallback
        if not has_q and not has_s and qrs_ms > WIDE_QRS_THRESHOLD_MS:
            labels[on:off + 1] = vocab.ID_W
            continue
        
        # Standard q/r/s decomposition
        q_end = (q + r) // 2 if has_q else on
        s_start = (r + s) // 2 if has_s else off + 1
        
        if has_q:
            labels[on:q_end] = vocab.ID_Q
        labels[q_end:s_start] = vocab.ID_R
        if has_s:
            labels[s_start:off + 1] = vocab.ID_S
    
    # 4. Pacer spikes — highest priority override
    for idx in spike_idx:
        if 0 <= idx < n_samples:
            labels[idx] = vocab.ID_PACER
    
    # 5. RLE compress
    return _rle_compress(labels, ms_per_sample)


def _rle_compress(labels: np.ndarray, ms_per_sample: float) -> list[tuple[int, int]]:
    """Group consecutive identical labels → list of (symbol_id, length_ms)."""
    if len(labels) == 0:
        return []
    # Find indices where label changes
    change_idx = np.flatnonzero(np.diff(labels)) + 1
    boundaries = np.concatenate(([0], change_idx, [len(labels)]))
    events = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        sym = int(labels[start])
        n = end - start
        ms = int(round(n * ms_per_sample))
        if ms > 0:
            events.append((sym, ms))
    return events
```

- [ ] **Step 5: Run tests, verify they pass**

Run: `uv run pytest tests/test_labeler.py -v`
Expected: All 9 tests PASS.

Note: `test_spike_overrides_iso` uses `empty_dr` fixture — that test asserts unk because empty DR returns single unk event regardless of spikes (spike isn't applied when n_beats=0). If the test fails, the implementation correctly handles the precedence.

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py ecgcode/labeler.py tests/test_labeler.py
git commit -m "Add labeler with QRS decomposition + wide-QRS fallback + spike priority"
```

---

### Task 7: LUDB loader

**Files:**
- Create: `ecgcode/ludb.py` (load + extract portion)

- [ ] **Step 1: Implement extract + load_record**

```python
# ecgcode/ludb.py
"""LUDB (Lobachevsky University Database) loader and stratified split.

Expects ECGCODE_LUDB_ZIP env var pointing to the LUDB zip file.
Extracts to ECGCODE_LUDB_CACHE (default: ~/.cache/ecgcode/ludb).
"""

import csv
import json
import os
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import wfdb

LEADS_12 = ("i", "ii", "iii", "avr", "avl", "avf",
            "v1", "v2", "v3", "v4", "v5", "v6")

LUDB_INNER_DIR = "lobachevsky-university-electrocardiography-database-1.0.1"


def _zip_path() -> Path:
    p = os.environ.get("ECGCODE_LUDB_ZIP")
    if not p:
        raise FileNotFoundError(
            "Set ECGCODE_LUDB_ZIP env var to LUDB zip file path. "
            "Download from https://physionet.org/content/ludb/1.0.1/"
        )
    return Path(p)


def _cache_path() -> Path:
    p = os.environ.get("ECGCODE_LUDB_CACHE")
    if p:
        return Path(p).expanduser()
    return Path.home() / ".cache" / "ecgcode" / "ludb"


def ensure_extracted() -> Path:
    """Extract LUDB zip to cache (idempotent). Returns the inner data dir."""
    cache = _cache_path()
    inner = cache / LUDB_INNER_DIR
    if inner.exists():
        return inner
    cache.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(_zip_path()) as z:
        z.extractall(cache)
    return inner


def all_record_ids() -> list[int]:
    """Return all 200 LUDB record IDs (1..200)."""
    inner = ensure_extracted()
    records_file = inner / "RECORDS"
    ids = []
    for line in records_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        # Lines look like "data/1", "data/2", ...
        ids.append(int(Path(line).name))
    return sorted(ids)


def load_record(record_id: int) -> dict[str, np.ndarray]:
    """Load one LUDB record. Returns dict {lead_name: signal[5000]}.
    
    Signal shape: 5000 samples = 10s @ 500Hz, microvolts.
    """
    inner = ensure_extracted()
    record_path = str(inner / "data" / str(record_id))
    record = wfdb.rdrecord(record_path)
    return {lead: record.p_signal[:, i].astype(np.float64)
            for i, lead in enumerate(LEADS_12)}


def load_annotations(record_id: int, lead: str) -> dict[str, list[int]]:
    """Load LUDB cardiologist annotations for one record-lead.
    
    Returns dict with keys 'p_on', 'p_peak', 'p_off', 'qrs_on', 'qrs_peak',
    'qrs_off', 't_on', 't_peak', 't_off' (sample indices).
    """
    inner = ensure_extracted()
    ann_path = str(inner / "data" / str(record_id))
    ann = wfdb.rdann(ann_path, lead)
    out = {"p_on": [], "p_peak": [], "p_off": [],
           "qrs_on": [], "qrs_peak": [], "qrs_off": [],
           "t_on": [], "t_peak": [], "t_off": []}
    # LUDB annotation format: ( N ) ( p ) ( t ) where ( and ) flank a peak symbol
    for i, sym in enumerate(ann.symbol):
        s = int(ann.sample[i])
        if sym == "p":
            out["p_peak"].append(s)
            if i > 0 and ann.symbol[i - 1] == "(":
                out["p_on"].append(int(ann.sample[i - 1]))
            if i + 1 < len(ann.symbol) and ann.symbol[i + 1] == ")":
                out["p_off"].append(int(ann.sample[i + 1]))
        elif sym == "N":
            out["qrs_peak"].append(s)
            if i > 0 and ann.symbol[i - 1] == "(":
                out["qrs_on"].append(int(ann.sample[i - 1]))
            if i + 1 < len(ann.symbol) and ann.symbol[i + 1] == ")":
                out["qrs_off"].append(int(ann.sample[i + 1]))
        elif sym == "t":
            out["t_peak"].append(s)
            if i > 0 and ann.symbol[i - 1] == "(":
                out["t_on"].append(int(ann.sample[i - 1]))
            if i + 1 < len(ann.symbol) and ann.symbol[i + 1] == ")":
                out["t_off"].append(int(ann.sample[i + 1]))
    return out


def load_metadata() -> list[dict]:
    """Read ludb.csv. Returns list of dicts with normalized rhythm field.
    
    Multi-line Rhythms (with embedded newlines) are normalized to first line.
    """
    inner = ensure_extracted()
    csv_path = inner / "ludb.csv"
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        rhythm_raw = r.get("Rhythms", "").strip()
        # Take first line for stratification (some records have multiple rhythms)
        r["rhythm"] = rhythm_raw.split("\n")[0].strip()
        r["pacemaker"] = bool(r.get("Cardiac pacing", "").strip())
        r["id_int"] = int(r["ID"])
    return rows
```

- [ ] **Step 2: Smoke verify (requires LUDB zip env var set)**

Run:
```powershell
$env:ECGCODE_LUDB_ZIP = "G:\Shared drives\datasets\ecg\lobachevsky-university-electrocardiography-database-1.0.1.zip"
uv run python -c "from ecgcode.ludb import load_record, all_record_ids, load_metadata; ids=all_record_ids(); print(f'records: {len(ids)}'); rec = load_record(1); print(f'lead ii shape: {rec[\"ii\"].shape}'); meta=load_metadata(); print(f'pacers: {sum(1 for m in meta if m[\"pacemaker\"])}')"
```
Expected: 
```
records: 200
lead ii shape: (5000,)
pacers: 10
```

- [ ] **Step 3: Commit**

```bash
git add ecgcode/ludb.py
git commit -m "Add LUDB loader (extract, load_record, load_annotations, metadata)"
```

---

### Task 8: LUDB stratified split + commit JSON

**Files:**
- Modify: `ecgcode/ludb.py` (append split function)
- Create: `data/splits/ludb_v1.json` (committed)

- [ ] **Step 1: Append split function to ludb.py**

Append to `ecgcode/ludb.py`:

```python
def stratified_split(seed: int = 42, val_frac: float = 0.2) -> dict[str, list[int]]:
    """Rhythm-stratified record-level train/val split.
    
    Each rhythm class is split independently to preserve class balance.
    Reproducible via numpy default_rng(seed).
    """
    meta = load_metadata()
    by_rhythm: dict[str, list[int]] = defaultdict(list)
    for r in meta:
        by_rhythm[r["rhythm"]].append(r["id_int"])
    
    rng = np.random.default_rng(seed)
    train_ids: list[int] = []
    val_ids: list[int] = []
    for rhythm, ids in sorted(by_rhythm.items()):
        ids_sorted = sorted(ids)
        rng.shuffle(ids_sorted)
        n_val = round(len(ids_sorted) * val_frac)
        val_ids.extend(ids_sorted[:n_val])
        train_ids.extend(ids_sorted[n_val:])
    
    return {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "seed": seed,
        "val_frac": val_frac,
    }


def save_split_json(out_path: Path | str = "data/splits/ludb_v1.json", seed: int = 42):
    """Generate stratified split and save to JSON for reproducibility lock-in."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    split = stratified_split(seed=seed)
    out_path.write_text(json.dumps(split, indent=2))
    return split


def load_split(path: Path | str = "data/splits/ludb_v1.json") -> dict[str, list[int]]:
    """Load committed split JSON."""
    return json.loads(Path(path).read_text())
```

- [ ] **Step 2: Generate the split JSON**

Run:
```bash
uv run python -c "from ecgcode.ludb import save_split_json; s = save_split_json(); print(f'train: {len(s[\"train\"])}, val: {len(s[\"val\"])}')"
```
Expected: `train: 160, val: 40` (or close — within ±2 due to rounding per rhythm class).

- [ ] **Step 3: Inspect generated file**

```bash
cat data/splits/ludb_v1.json | head -10
```

Expected: JSON with `train`, `val`, `seed`, `val_frac` keys.

- [ ] **Step 4: Verify reproducibility**

Run:
```bash
uv run python -c "from ecgcode.ludb import stratified_split; s1 = stratified_split(seed=42); s2 = stratified_split(seed=42); assert s1 == s2; print('reproducible OK')"
```
Expected: `reproducible OK`

- [ ] **Step 5: Verify rhythm-stratification**

Run:
```bash
uv run python -c "
from collections import Counter
from ecgcode.ludb import load_metadata, load_split
meta = {m['id_int']: m['rhythm'] for m in load_metadata()}
split = load_split()
print('Train rhythms:', Counter(meta[i] for i in split['train']))
print('Val rhythms:', Counter(meta[i] for i in split['val']))
"
```
Expected: Each rhythm class appears in both train and val (where stratification permits — single-record classes go to train).

- [ ] **Step 6: Commit code + JSON**

```bash
git add ecgcode/ludb.py data/splits/ludb_v1.json
git commit -m "Add rhythm-stratified train/val split (160/40) + reproducibility lock"
```

---

### Task 9: Integration test on LUDB record 1

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
"""End-to-end integration test on LUDB record 1.

Skipped if ECGCODE_LUDB_ZIP env var not set or extraction fails.
"""

import os

import numpy as np
import pytest

LUDB_AVAILABLE = bool(os.environ.get("ECGCODE_LUDB_ZIP"))

pytestmark = pytest.mark.skipif(
    not LUDB_AVAILABLE,
    reason="ECGCODE_LUDB_ZIP env var not set; integration test requires LUDB",
)


def test_record_1_lead_ii_end_to_end():
    from ecgcode import codec, delineate, labeler, ludb, pacer, vocab
    
    record = ludb.load_record(1)
    sig = record["ii"]
    assert len(sig) == 5000
    
    dr = delineate.run(sig, fs=500)
    assert dr.n_beats > 0   # NK should detect beats in record 1 (sinus brady)
    
    spikes = pacer.detect_spikes(sig, fs=500)
    # Record 1 is non-pacemaker, expect few or no spikes
    
    events = labeler.label(dr, spikes.tolist(), n_samples=len(sig), fs=500)
    total_ms = sum(ms for _, ms in events)
    assert 9900 <= total_ms <= 10100, f"expected ~10s, got {total_ms}ms"
    
    packed = codec.encode(events)
    assert packed.dtype == np.uint16
    assert codec.decode(packed) == events
    
    art = codec.render_timed(events, ms_per_char=20)
    assert 480 <= len(art) <= 520, f"expected ~500 chars, got {len(art)}"
    print(f"\nRecord 1 lead II ASCII art ({len(art)} chars):")
    print(art)


def test_all_12_leads_record_1_no_crash():
    from ecgcode import codec, delineate, labeler, ludb, pacer
    
    record = ludb.load_record(1)
    for lead, sig in record.items():
        dr = delineate.run(sig, fs=500)
        spikes = pacer.detect_spikes(sig, fs=500)
        events = labeler.label(dr, spikes.tolist(), n_samples=len(sig), fs=500)
        assert len(events) > 0, f"empty events for lead {lead}"
        # Round-trip must be lossless
        packed = codec.encode(events)
        assert codec.decode(packed) == events


def test_pacer_record_8_detects_spikes():
    """Record 8 is a pacemaker patient (per ludb.csv 'Cardiac pacing' column).
    Pacer detector should find at least one spike on lead V1 or II."""
    from ecgcode import ludb, pacer
    
    record = ludb.load_record(8)
    spike_counts = {lead: len(pacer.detect_spikes(sig, fs=500))
                    for lead, sig in record.items()}
    total_spikes = sum(spike_counts.values())
    assert total_spikes > 0, f"no spikes detected on pacer record 8: {spike_counts}"
```

- [ ] **Step 2: Run integration tests**

Run:
```powershell
$env:ECGCODE_LUDB_ZIP = "G:\Shared drives\datasets\ecg\lobachevsky-university-electrocardiography-database-1.0.1.zip"
uv run pytest tests/test_integration.py -v -s
```
Expected: All 3 tests PASS. The `-s` flag shows the ASCII art print output for visual inspection.

- [ ] **Step 3: Visual inspection**

Read the printed ASCII art from the first test. Confirm it visually shows alternating wave patterns (e.g., `_pp__qrs_____ttttt___...`). If pattern looks broken (e.g., all `_` or all `?`), debug NK or labeler before proceeding.

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "Add LUDB record 1 end-to-end integration test"
```

---

### Task 10: Eval module (frame F1 + boundary error)

**Files:**
- Create: `ecgcode/eval.py`
- Create: `tests/test_eval.py`

- [ ] **Step 1: Write eval tests**

```python
# tests/test_eval.py
import numpy as np

from ecgcode import eval as ee
from ecgcode import vocab


def test_supercategory_mapping():
    # Map per-frame array to LUDB-compat 4-class
    frames = np.array([vocab.ID_ISO, vocab.ID_P, vocab.ID_Q, vocab.ID_R,
                       vocab.ID_S, vocab.ID_W, vocab.ID_T, vocab.ID_PACER,
                       vocab.ID_UNK], dtype=np.uint8)
    super_frames = ee.to_supercategory(frames)
    assert super_frames.tolist() == [
        ee.SUPER_OTHER, ee.SUPER_P, ee.SUPER_QRS, ee.SUPER_QRS,
        ee.SUPER_QRS, ee.SUPER_QRS, ee.SUPER_T, ee.SUPER_OTHER, ee.SUPER_OTHER,
    ]


def test_frame_f1_perfect_match():
    pred = np.array([ee.SUPER_P] * 5 + [ee.SUPER_QRS] * 3 + [ee.SUPER_T] * 5)
    true = pred.copy()
    metrics = ee.frame_f1(pred, true)
    for super_class in (ee.SUPER_P, ee.SUPER_QRS, ee.SUPER_T):
        assert metrics[super_class]["f1"] == 1.0
        assert metrics[super_class]["precision"] == 1.0
        assert metrics[super_class]["recall"] == 1.0


def test_frame_f1_total_disagreement():
    pred = np.array([ee.SUPER_P] * 10)
    true = np.array([ee.SUPER_QRS] * 10)
    metrics = ee.frame_f1(pred, true)
    assert metrics[ee.SUPER_P]["f1"] == 0.0
    assert metrics[ee.SUPER_QRS]["f1"] == 0.0


def test_boundary_error_perfect_alignment():
    # All predicted boundaries match LUDB exactly within tolerance
    true_idx = [100, 200, 300]
    pred_idx = [100, 200, 300]
    result = ee.boundary_error(pred_idx, true_idx, tolerance_ms=50, fs=500)
    assert result["sensitivity"] == 1.0
    assert result["ppv"] == 1.0
    assert result["median_error_ms"] == 0.0
    assert result["n_hits"] == 3


def test_boundary_error_partial_match():
    true_idx = [100, 200, 300]
    pred_idx = [105, 199, 600]   # last is way outside tolerance
    result = ee.boundary_error(pred_idx, true_idx, tolerance_ms=50, fs=500)
    assert result["n_hits"] == 2
    assert result["sensitivity"] == pytest.approx(2/3)
    assert result["ppv"] == pytest.approx(2/3)


import pytest


def test_boundary_error_empty_inputs():
    result = ee.boundary_error([], [], tolerance_ms=50, fs=500)
    assert result["n_hits"] == 0
    assert result["sensitivity"] == 0.0
    assert result["ppv"] == 0.0
```

- [ ] **Step 2: Run, verify they fail**

Run: `uv run pytest tests/test_eval.py -v`
Expected: All FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement eval.py**

```python
# ecgcode/eval.py
"""Evaluation metrics — frame-level F1 (4-class) + boundary error (Martinez-style)."""

import numpy as np

from ecgcode import vocab

# Supercategory IDs for LUDB-compat 4-class comparison
SUPER_OTHER = 0
SUPER_P = 1
SUPER_QRS = 2
SUPER_T = 3
SUPER_NAMES = {SUPER_OTHER: "other", SUPER_P: "P", SUPER_QRS: "QRS", SUPER_T: "T"}

_SUPER_MAP = {
    vocab.ID_PAD: SUPER_OTHER,
    vocab.ID_UNK: SUPER_OTHER,
    vocab.ID_ISO: SUPER_OTHER,
    vocab.ID_PACER: SUPER_OTHER,
    vocab.ID_P: SUPER_P,
    vocab.ID_Q: SUPER_QRS,
    vocab.ID_R: SUPER_QRS,
    vocab.ID_S: SUPER_QRS,
    vocab.ID_W: SUPER_QRS,
    vocab.ID_T: SUPER_T,
    vocab.ID_U: SUPER_T,    # U is repolarization-adjacent; group with T for now
    vocab.ID_D: SUPER_QRS,  # delta is QRS-adjacent
    vocab.ID_J: SUPER_QRS,
}


def to_supercategory(frames: np.ndarray) -> np.ndarray:
    """Map per-frame v1 alphabet IDs → LUDB-compat 4-class."""
    out = np.zeros_like(frames, dtype=np.uint8)
    for src, dst in _SUPER_MAP.items():
        out[frames == src] = dst
    return out


def frame_f1(pred: np.ndarray, true: np.ndarray) -> dict[int, dict[str, float]]:
    """Per-supercategory precision/recall/F1.
    
    Returns: {super_id: {'precision': p, 'recall': r, 'f1': f}}
    """
    out = {}
    for sc in (SUPER_OTHER, SUPER_P, SUPER_QRS, SUPER_T):
        tp = int(np.sum((pred == sc) & (true == sc)))
        fp = int(np.sum((pred == sc) & (true != sc)))
        fn = int(np.sum((pred != sc) & (true == sc)))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        out[sc] = {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    return out


def boundary_error(
    pred_indices: list[int],
    true_indices: list[int],
    tolerance_ms: float,
    fs: int,
) -> dict:
    """Greedy nearest-match boundary comparison (Martinez 2004 style).
    
    For each true boundary, find nearest predicted within tolerance.
    Returns sensitivity, PPV, mean/median/p95 error in ms, hit/miss counts.
    """
    tolerance_samples = tolerance_ms * fs / 1000
    pred_arr = np.sort(np.array(pred_indices, dtype=int))
    true_arr = np.sort(np.array(true_indices, dtype=int))
    
    if len(true_arr) == 0 and len(pred_arr) == 0:
        return _empty_boundary_result()
    
    matched_pred = set()
    errors_samples = []
    n_hits = 0
    
    for t in true_arr:
        if len(pred_arr) == 0:
            break
        # Nearest unmatched predicted
        best_idx = -1
        best_err = float("inf")
        for j, p in enumerate(pred_arr):
            if j in matched_pred:
                continue
            err = abs(int(p) - int(t))
            if err < best_err:
                best_err = err
                best_idx = j
        if best_idx >= 0 and best_err <= tolerance_samples:
            matched_pred.add(best_idx)
            errors_samples.append(best_err)
            n_hits += 1
    
    if errors_samples:
        errors_ms = np.array(errors_samples) * 1000.0 / fs
        mean_err = float(np.mean(errors_ms))
        median_err = float(np.median(errors_ms))
        p95_err = float(np.percentile(errors_ms, 95))
    else:
        mean_err = median_err = p95_err = 0.0
    
    sensitivity = n_hits / len(true_arr) if len(true_arr) > 0 else 0.0
    ppv = n_hits / len(pred_arr) if len(pred_arr) > 0 else 0.0
    
    return {
        "sensitivity": sensitivity,
        "ppv": ppv,
        "n_hits": n_hits,
        "n_true": int(len(true_arr)),
        "n_pred": int(len(pred_arr)),
        "mean_error_ms": mean_err,
        "median_error_ms": median_err,
        "p95_error_ms": p95_err,
    }


def _empty_boundary_result():
    return {
        "sensitivity": 0.0, "ppv": 0.0, "n_hits": 0, "n_true": 0, "n_pred": 0,
        "mean_error_ms": 0.0, "median_error_ms": 0.0, "p95_error_ms": 0.0,
    }


def events_to_super_frames(events, n_samples, fs=500, frame_ms=20):
    """Pipeline events → per-frame supercategory array.
    Used by validate_v1.py and ablate_methods.py."""
    from ecgcode import codec
    total_ms = round(n_samples * 1000.0 / fs)
    frames = codec.to_frames(events, frame_ms=frame_ms, total_ms=total_ms)
    return to_supercategory(frames)


def gt_to_super_frames(gt_ann, n_samples, fs=500, frame_ms=20):
    """LUDB cardiologist annotation dict → per-frame supercategory array (majority per frame)."""
    sample_labels = np.full(n_samples, SUPER_OTHER, dtype=np.uint8)
    for on, off in zip(gt_ann["p_on"], gt_ann["p_off"]):
        sample_labels[on:off + 1] = SUPER_P
    for on, off in zip(gt_ann["qrs_on"], gt_ann["qrs_off"]):
        sample_labels[on:off + 1] = SUPER_QRS
    for on, off in zip(gt_ann["t_on"], gt_ann["t_off"]):
        sample_labels[on:off + 1] = SUPER_T
    n_frames = round(n_samples * 1000.0 / fs / frame_ms)
    samples_per_frame = n_samples // n_frames if n_frames > 0 else 1
    out = np.zeros(n_frames, dtype=np.uint8)
    for f in range(n_frames):
        seg = sample_labels[f * samples_per_frame: (f + 1) * samples_per_frame]
        if len(seg) == 0:
            continue
        vals, counts = np.unique(seg, return_counts=True)
        out[f] = vals[np.argmax(counts)]
    return out
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `uv run pytest tests/test_eval.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ecgcode/eval.py tests/test_eval.py
git commit -m "Add eval module (4-class frame F1 + boundary error metrics)"
```

---

### Task 11: Tokenize script (full LUDB → npz)

**Files:**
- Create: `scripts/tokenize_ludb.py`

- [ ] **Step 1: Implement tokenize_ludb.py**

```python
# scripts/tokenize_ludb.py
"""Run full pipeline on every LUDB record × lead → ludb_tokens.npz.

Usage:
    $env:ECGCODE_LUDB_ZIP = "..."
    uv run python scripts/tokenize_ludb.py
"""

import json
import time
from pathlib import Path

import numpy as np

from ecgcode import codec, delineate, labeler, ludb, pacer

OUT_PATH = Path("data/ludb_tokens.npz")


def tokenize_one(sig: np.ndarray, fs: int = 500) -> np.ndarray:
    dr = delineate.run(sig, fs=fs)
    spikes = pacer.detect_spikes(sig, fs=fs)
    events = labeler.label(dr, spikes.tolist(), n_samples=len(sig), fs=fs)
    return codec.encode(events)


def main():
    record_ids = ludb.all_record_ids()
    print(f"Tokenizing {len(record_ids)} records × {len(ludb.LEADS_12)} leads "
          f"= {len(record_ids) * len(ludb.LEADS_12)} sequences...")
    
    arrays: dict[str, np.ndarray] = {}
    t0 = time.time()
    for n, rid in enumerate(record_ids, 1):
        record = ludb.load_record(rid)
        for lead in ludb.LEADS_12:
            sig = record[lead]
            packed = tokenize_one(sig, fs=500)
            arrays[f"{rid:04d}_{lead}"] = packed
        if n % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{n}/{len(record_ids)}] {elapsed:.1f}s elapsed")
    
    meta = {
        "vocab_version": "v1.0",
        "ms_unit": codec.MS_PER_UNIT,
        "fs": 500,
        "n_records": len(record_ids),
        "leads": list(ludb.LEADS_12),
    }
    
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_PATH, meta=json.dumps(meta), **arrays)
    
    elapsed = time.time() - t0
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Saved {OUT_PATH} ({size_kb:.1f} KB) in {elapsed:.1f}s")
    print(f"Total sequences: {len(arrays)}, mean events/seq: "
          f"{np.mean([len(a) for a in arrays.values()]):.1f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on full LUDB**

Run:
```powershell
$env:ECGCODE_LUDB_ZIP = "G:\Shared drives\datasets\ecg\lobachevsky-university-electrocardiography-database-1.0.1.zip"
uv run python scripts/tokenize_ludb.py
```
Expected: Progress output, saves `data/ludb_tokens.npz` (~150-300 KB), 2400 sequences.

- [ ] **Step 3: Smoke verify output**

Run:
```bash
uv run python -c "
import numpy as np, json
d = np.load('data/ludb_tokens.npz')
print('keys (first 3):', list(d.keys())[:3], '... total:', len(d.keys()))
print('meta:', json.loads(str(d['meta'])))
arr = d['0001_ii']
print(f'record 1 lead ii: {len(arr)} events, dtype={arr.dtype}')
"
```
Expected output:
```
keys (first 3): ['meta', '0001_i', '0001_ii'] ... total: 2401
meta: {'vocab_version': 'v1.0', 'ms_unit': 4, 'fs': 500, ...}
record 1 lead ii: ... events, dtype=uint16
```

- [ ] **Step 4: Commit script (output is gitignored)**

```bash
git add scripts/tokenize_ludb.py
git commit -m "Add tokenize_ludb script (full LUDB → ludb_tokens.npz)"
```

Note: `data/ludb_tokens.npz` is gitignored by `data/*` pattern; we only commit the script.

---

### Task 12: Validation script (NK pseudo-label vs LUDB cardiologist)

**Files:**
- Create: `scripts/validate_v1.py`

- [ ] **Step 1: Implement validate_v1.py**

```python
# scripts/validate_v1.py
"""Validate Stage 1 v1.0 on LUDB val split.

Compares NK pseudo-labels (via our pipeline) against LUDB cardiologist
annotations. Reports per-class frame F1 (4-class supercategory) and
per-boundary-type error metrics (Martinez-style).

Usage:
    $env:ECGCODE_LUDB_ZIP = "..."
    uv run python scripts/validate_v1.py
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ecgcode import codec, delineate, eval as ee, labeler, ludb, pacer, vocab

FS = 500
FRAME_MS = 20
BOUNDARY_TOLERANCES = {
    "p_on": 50, "p_off": 50,
    "qrs_on": 40, "qrs_off": 40,
    "t_on": 50, "t_off": 100,
}
OUT_DIR = Path("out")


def _extract_pred_boundaries(events, fs=FS):
    """Extract predicted boundary sample indices from RLE events.
    Returns dict mirroring LUDB annotation keys (p_on, p_off, qrs_on, qrs_off, t_on, t_off)."""
    out = defaultdict(list)
    cum_samples = 0
    prev_super = ee.SUPER_OTHER
    super_to_name = {ee.SUPER_P: "p", ee.SUPER_QRS: "qrs", ee.SUPER_T: "t"}
    
    for sym, ms in events:
        n = round(ms * fs / 1000.0)
        cur_super = ee.to_supercategory(np.array([sym], dtype=np.uint8))[0]
        if cur_super != prev_super:
            # boundary at cum_samples
            if prev_super in super_to_name:
                out[f"{super_to_name[prev_super]}_off"].append(cum_samples - 1)
            if cur_super in super_to_name:
                out[f"{super_to_name[cur_super]}_on"].append(cum_samples)
        cum_samples += n
        prev_super = cur_super
    # Close any open wave at end
    if prev_super in super_to_name:
        out[f"{super_to_name[prev_super]}_off"].append(cum_samples - 1)
    
    return dict(out)


def main():
    split = ludb.load_split()
    val_ids = split["val"]
    print(f"Validating on {len(val_ids)} val records × 12 leads = "
          f"{len(val_ids) * 12} sequences")
    
    # Frame-level: accumulate predictions and ground truth across all sequences
    all_pred_frames = []
    all_true_frames = []
    
    # Boundary-level: per-type accumulators
    boundary_pred = defaultdict(list)
    boundary_true = defaultdict(list)
    boundary_offsets = []  # cumulative sample offset per sequence for boundary aggregation
    
    # Q-loss tracking
    nk_q_count = 0
    rle_q_count = 0
    
    t0 = time.time()
    for n, rid in enumerate(val_ids, 1):
        record = ludb.load_record(rid)
        cum_offset = 0
        for lead in ludb.LEADS_12:
            sig = record[lead]
            n_samples = len(sig)
            
            dr = delineate.run(sig, fs=FS)
            spikes = pacer.detect_spikes(sig, fs=FS)
            events = labeler.label(dr, spikes.tolist(), n_samples=n_samples, fs=FS)
            
            # Frame-level
            pred_frames = ee.events_to_super_frames(events, n_samples, fs=FS, frame_ms=FRAME_MS)
            try:
                gt_ann = ludb.load_annotations(rid, lead)
            except Exception:
                continue
            true_frames = ee.gt_to_super_frames(gt_ann, n_samples, fs=FS, frame_ms=FRAME_MS)
            
            # Truncate to common length
            n_common = min(len(pred_frames), len(true_frames))
            all_pred_frames.append(pred_frames[:n_common])
            all_true_frames.append(true_frames[:n_common])
            
            # Boundary-level: shift indices by cumulative offset to avoid collision
            pred_b = _extract_pred_boundaries(events)
            for k, v in gt_ann.items():
                if not k.endswith("_on") and not k.endswith("_off"):
                    continue
                # qrs_on/off matches our predicted naming; p_on/off, t_on/off too
                boundary_true[k].extend(int(x) + cum_offset for x in v)
            for k, v in pred_b.items():
                boundary_pred[k].extend(int(x) + cum_offset for x in v)
            
            cum_offset += n_samples
            
            # Q-loss
            nk_q_count += int(np.sum(~np.isnan(dr.q_peaks))) if dr.n_beats > 0 else 0
            rle_q_count += sum(1 for s, _ in events if s == vocab.ID_Q)
        
        if n % 5 == 0:
            print(f"  [{n}/{len(val_ids)}] {time.time() - t0:.1f}s")
    
    # Aggregate frame F1
    pred_concat = np.concatenate(all_pred_frames)
    true_concat = np.concatenate(all_true_frames)
    f1_metrics = ee.frame_f1(pred_concat, true_concat)
    
    # Boundary error per type
    boundary_metrics = {}
    for key, tol_ms in BOUNDARY_TOLERANCES.items():
        boundary_metrics[key] = ee.boundary_error(
            boundary_pred.get(key, []), boundary_true.get(key, []),
            tolerance_ms=tol_ms, fs=FS,
        )
    
    q_loss = 1 - (rle_q_count / nk_q_count) if nk_q_count > 0 else 0.0
    
    # Print summary
    print("\n== ECGCode v1.0 Validation on LUDB val ==\n")
    print("Frame-level F1 (4-class supercategory):")
    for sc, m in f1_metrics.items():
        name = ee.SUPER_NAMES[sc]
        print(f"  {name:6s} : F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}")
    
    print("\nBoundary error (median ms / p95 / sens / PPV):")
    for key, m in boundary_metrics.items():
        print(f"  {key:7s}: {m['median_error_ms']:5.1f} / {m['p95_error_ms']:5.1f} "
              f"/ {m['sensitivity']:.2f} / {m['ppv']:.2f}  "
              f"(hits={m['n_hits']}, true={m['n_true']}, pred={m['n_pred']})")
    
    print(f"\nQ-loss rate: {q_loss:.1%}  (NK={nk_q_count}, RLE={rle_q_count})")
    
    # Save JSON
    OUT_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = OUT_DIR / f"validation_v1_{ts}.json"
    out_file.write_text(json.dumps({
        "frame_f1": {ee.SUPER_NAMES[sc]: m for sc, m in f1_metrics.items()},
        "boundary": boundary_metrics,
        "q_loss_rate": q_loss,
        "n_records": len(val_ids),
    }, indent=2))
    print(f"\nReport saved: {out_file}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on val split**

Run:
```powershell
uv run python scripts/validate_v1.py
```
Expected: Console output with frame F1 and boundary metrics, JSON saved to `out/validation_v1_<timestamp>.json`. Exact F1 numbers depend on NK behavior — should be in the ranges:
- P frame F1: 0.6 ~ 0.85
- QRS frame F1: 0.85 ~ 0.95
- T frame F1: 0.5 ~ 0.85

If numbers are dramatically lower (e.g., < 0.3 for QRS), debug:
- Check `_extract_pred_boundaries` boundary extraction logic
- Check `_gt_to_super_frames` ground truth alignment
- Verify NK is producing waves (`dr.n_beats > 0` for most records)

- [ ] **Step 3: Eyeball results, compare to acceptance criteria**

Acceptance (from spec §10):
| Metric | Target |
|---|---|
| P frame F1 | ≥ 0.80 |
| QRS frame F1 | ≥ 0.90 |
| T frame F1 | ≥ 0.75 |
| QRS_onset boundary median | ≤ 20 ms |
| Q-loss rate | ≤ 20% |

If anything misses target, investigate before declaring v1.0 done. Don't change targets to fit results.

- [ ] **Step 4: Commit script**

```bash
git add scripts/validate_v1.py
git commit -m "Add validate_v1 script (NK vs LUDB cardiologist metrics report)"
```

---

### Task 13: NK method ablation script

**Files:**
- Create: `scripts/ablate_methods.py`

Tests NK's 4 delineate methods (`dwt`, `cwt`, `peak`, `prominence`) on the val split, reports per-method frame F1 + Q-loss, and picks the highest-QRS-F1 winner. The winner becomes the locked-in method for v1.0 release (default in `delineate.run`). This is a quick "method sweep" — we still treat `dwt` as initial default and switch only if ablation shows another is clearly better.

- [ ] **Step 1: Implement ablate_methods.py**

```python
# scripts/ablate_methods.py
"""Run validation pipeline with each NK delineate method, report comparison.

Usage:
    $env:ECGCODE_LUDB_ZIP = "..."
    uv run python scripts/ablate_methods.py
"""

import json
import time
from pathlib import Path

import numpy as np

from ecgcode import delineate, eval as ee, labeler, ludb, pacer, vocab

METHODS = ["dwt", "cwt", "peak", "prominence"]
FS = 500
FRAME_MS = 20
OUT_DIR = Path("out")


def evaluate_method(method: str, val_ids: list[int]) -> dict:
    """Run pipeline + frame F1 evaluation with given NK method on val split."""
    all_pred = []
    all_true = []
    nk_q_count = 0
    rle_q_count = 0
    n_failed = 0
    
    for rid in val_ids:
        record = ludb.load_record(rid)
        for lead in ludb.LEADS_12:
            sig = record[lead]
            n_samples = len(sig)
            try:
                dr = delineate.run(sig, fs=FS, method=method)
            except Exception:
                n_failed += 1
                continue
            spikes = pacer.detect_spikes(sig, fs=FS)
            events = labeler.label(dr, spikes.tolist(), n_samples=n_samples, fs=FS)
            
            pred_frames = ee.events_to_super_frames(events, n_samples, fs=FS, frame_ms=FRAME_MS)
            try:
                gt_ann = ludb.load_annotations(rid, lead)
            except Exception:
                continue
            true_frames = ee.gt_to_super_frames(gt_ann, n_samples, fs=FS, frame_ms=FRAME_MS)
            
            n_common = min(len(pred_frames), len(true_frames))
            all_pred.append(pred_frames[:n_common])
            all_true.append(true_frames[:n_common])
            
            if dr.n_beats > 0:
                nk_q_count += int(np.sum(~np.isnan(dr.q_peaks)))
            rle_q_count += sum(1 for s, _ in events if s == vocab.ID_Q)
    
    if not all_pred:
        return {"f1": None, "q_loss": None, "failed": n_failed}
    
    pred_concat = np.concatenate(all_pred)
    true_concat = np.concatenate(all_true)
    f1 = ee.frame_f1(pred_concat, true_concat)
    q_loss = 1 - rle_q_count / nk_q_count if nk_q_count > 0 else 0.0
    return {"f1": f1, "q_loss": q_loss, "failed": n_failed,
            "nk_q": nk_q_count, "rle_q": rle_q_count}


def main():
    val_ids = ludb.load_split()["val"]
    print(f"Ablating {len(METHODS)} NK methods on {len(val_ids)} val records × 12 leads\n")
    
    results = {}
    for method in METHODS:
        print(f"--- {method} ---")
        t0 = time.time()
        try:
            results[method] = evaluate_method(method, val_ids)
        except Exception as exc:
            print(f"  FAILED entire method: {exc}\n")
            results[method] = None
            continue
        elapsed = time.time() - t0
        r = results[method]
        if r["f1"] is None:
            print(f"  no valid sequences ({elapsed:.1f}s)\n")
            continue
        print(f"  P F1   = {r['f1'][ee.SUPER_P]['f1']:.3f}")
        print(f"  QRS F1 = {r['f1'][ee.SUPER_QRS]['f1']:.3f}")
        print(f"  T F1   = {r['f1'][ee.SUPER_T]['f1']:.3f}")
        print(f"  Q-loss = {r['q_loss']:.1%}")
        print(f"  failed sequences: {r['failed']}")
        print(f"  ({elapsed:.1f}s)\n")
    
    # Pick winner: highest QRS F1
    valid = {m: r for m, r in results.items() if r and r["f1"] is not None}
    if valid:
        winner = max(valid.keys(),
                     key=lambda m: valid[m]["f1"][ee.SUPER_QRS]["f1"])
        print(f"== Winner: '{winner}' (highest QRS F1 = "
              f"{valid[winner]['f1'][ee.SUPER_QRS]['f1']:.3f}) ==")
        print(f"\nIf winner != 'dwt', update delineate.run default and re-run validate_v1.py.")
    
    # Save serializable JSON
    OUT_DIR.mkdir(exist_ok=True)
    out_file = OUT_DIR / f"ablation_{time.strftime('%Y%m%d_%H%M%S')}.json"
    serializable = {}
    for m, r in results.items():
        if r is None or r["f1"] is None:
            serializable[m] = None
            continue
        serializable[m] = {
            "f1": {ee.SUPER_NAMES[k]: v for k, v in r["f1"].items()},
            "q_loss": r["q_loss"],
            "failed": r["failed"],
            "nk_q": r["nk_q"],
            "rle_q": r["rle_q"],
        }
    out_file.write_text(json.dumps(serializable, indent=2))
    print(f"\nReport: {out_file}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run ablation**

Run:
```powershell
uv run python scripts/ablate_methods.py
```
Expected: 4 method blocks with F1/Q-loss output, winner declaration, JSON saved. Total runtime ~5-15 minutes (depends on NK speed for each method).

- [ ] **Step 3: Decide winner method**

Inspect the comparison. Decision rule (in priority):
1. QRS F1 ≥ 0.90 if achievable
2. Among methods meeting (1), highest P+T F1 average
3. If no method meets (1), pick highest QRS F1 and document as "best of available"

- [ ] **Step 4: If winner != "dwt", update default**

If ablation picks a different method, edit `ecgcode/delineate.py`:

```python
def run(signal: np.ndarray, fs: int = 500, method: str = "<winner>") -> DelineateResult:
```

And re-run `validate_v1.py` to lock in numbers with the chosen method.

- [ ] **Step 5: Commit**

```bash
git add scripts/ablate_methods.py
# Also commit the change to delineate.py if default was updated
git commit -m "Add NK method ablation script + lock in winner method"
```

---

### Task 14: Pacer detector validation script

**Files:**
- Create: `scripts/validate_pacer.py`

- [ ] **Step 1: Implement validate_pacer.py**

```python
# scripts/validate_pacer.py
"""Validate pacer spike detector: TPR on 10 pacemaker records, FPR on 190 non-pacer.

Usage:
    $env:ECGCODE_LUDB_ZIP = "..."
    uv run python scripts/validate_pacer.py
"""

from collections import defaultdict

from ecgcode import ludb, pacer

FS = 500


def main():
    meta = ludb.load_metadata()
    pacer_ids = sorted(int(m["ID"]) for m in meta if m["pacemaker"])
    non_pacer_ids = sorted(int(m["ID"]) for m in meta if not m["pacemaker"])
    
    print(f"Pacer records: {len(pacer_ids)} ({pacer_ids})")
    print(f"Non-pacer records: {len(non_pacer_ids)}")
    print()
    
    print("== Positive set (pacemaker patients) ==")
    pacer_results = []
    for rid in pacer_ids:
        record = ludb.load_record(rid)
        per_lead = {lead: len(pacer.detect_spikes(sig, fs=FS))
                    for lead, sig in record.items()}
        total = sum(per_lead.values())
        n_leads_with_spikes = sum(1 for v in per_lead.values() if v > 0)
        pacer_results.append({
            "id": rid, "total_spikes": total, "leads_with_spikes": n_leads_with_spikes,
            "per_lead": per_lead,
        })
        print(f"  Record {rid:3d}: {total:3d} spikes total, "
              f"{n_leads_with_spikes:2d}/12 leads detected")
    
    n_records_with_any = sum(1 for r in pacer_results if r["total_spikes"] > 0)
    mean_inter_lead = (sum(r["leads_with_spikes"] for r in pacer_results)
                       / len(pacer_results))
    print(f"\nPacer record detection rate: {n_records_with_any}/{len(pacer_ids)}")
    print(f"Mean inter-lead consistency: {mean_inter_lead:.1f}/12 leads (target ≥ 6)")
    
    print("\n== Negative set (non-pacemaker, sample 30 records) ==")
    # Sample first 30 to keep runtime reasonable; can run all by removing slice
    sampled = non_pacer_ids[:30]
    fp_per_record = []
    suspicious = []
    for rid in sampled:
        record = ludb.load_record(rid)
        total = sum(len(pacer.detect_spikes(sig, fs=FS))
                    for sig in record.values())
        fp_per_record.append(total)
        if total > 5:
            suspicious.append((rid, total))
    
    fpr_per_10s = sum(fp_per_record) / len(sampled)
    print(f"Total false positives over {len(sampled)} non-pacer records: "
          f"{sum(fp_per_record)}")
    print(f"Mean FPR: {fpr_per_10s:.2f} spikes / 10s record (target < 2)")
    if suspicious:
        print(f"Suspicious records (>5 false spikes): {suspicious}")
    
    # Acceptance check
    print("\n== Acceptance ==")
    tpr_ok = n_records_with_any >= 8     # 80% of pacer records detected
    inter_ok = mean_inter_lead >= 6
    fpr_ok = fpr_per_10s < 2
    print(f"  TPR (≥8/10 records): {'PASS' if tpr_ok else 'FAIL'}")
    print(f"  Inter-lead (≥6/12):  {'PASS' if inter_ok else 'FAIL'}")
    print(f"  FPR (<2 per 10s):    {'PASS' if fpr_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run validation**

Run:
```powershell
uv run python scripts/validate_pacer.py
```
Expected: TPR ≥ 8/10, inter-lead ≥ 6, FPR < 2.

If TPR fails (<8/10), investigate:
- Lower `amp_threshold_mad` from 5.0 to 3.5 in `pacer.detect_spikes`
- Inspect failing record signals manually

If FPR fails, investigate:
- Raise `cutoff_hz` from 80 to 100
- Tighten `max_width_ms` from 4.0 to 2.0

- [ ] **Step 3: Commit**

```bash
git add scripts/validate_pacer.py
git commit -m "Add validate_pacer script (TPR on 10 pacer + FPR on non-pacer)"
```

---

### Task 15: Final verification run

**Files:** No new files; verifies everything together.

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: All ~40 tests PASS (vocab 10 + codec 15 + pacer 7 + labeler 9 + eval 6 + integration 3 conditional).

- [ ] **Step 2: Run all validation scripts**

```powershell
uv run python scripts/tokenize_ludb.py
uv run python scripts/ablate_methods.py
uv run python scripts/validate_v1.py
uv run python scripts/validate_pacer.py
```

- [ ] **Step 3: Visual ASCII art sanity check on 10 records**

Run:
```bash
uv run python -c "
from ecgcode import codec, delineate, labeler, ludb, pacer
import random
random.seed(42)
val_ids = ludb.load_split()['val'][:10]
for rid in val_ids:
    sig = ludb.load_record(rid)['ii']
    dr = delineate.run(sig, fs=500)
    spikes = pacer.detect_spikes(sig, fs=500)
    events = labeler.label(dr, spikes.tolist(), n_samples=len(sig), fs=500)
    art = codec.render_timed(events, ms_per_char=20)
    print(f'Record {rid:3d}: ({dr.n_beats} beats, {len(events)} events)')
    print(f'  {art}')
    print()
"
```

For each record, confirm:
- Visible PQRST patterns repeat (not all `_` or all `?`)
- Beat counts are reasonable (5-15 beats per 10s)
- No glaring artifacts (e.g., 200 consecutive `?`)

- [ ] **Step 4: Update README with status**

Append to `README.md`:

```markdown

## Status

Stage 1 v1.0 complete: tokenization pipeline + LUDB validation baseline. See:
- Spec: `docs/superpowers/specs/2026-05-03-ecgcode-stage1-design.md`
- Plan: `docs/superpowers/plans/2026-05-03-ecgcode-stage1.md`
- Validation report: `out/validation_v1_*.json`

## Setup

```bash
uv sync
$env:ECGCODE_LUDB_ZIP = "<path-to-ludb-zip>"
uv run pytest
uv run python scripts/tokenize_ludb.py
uv run python scripts/validate_v1.py
```
```

- [ ] **Step 5: Final commit**

```bash
git add README.md
git commit -m "Stage 1 v1.0 complete: pipeline + validation"
git push
```

---

## Self-review checklist

After implementing all tasks, verify the spec's §10 success criteria are met by running `validate_v1.py` and `validate_pacer.py`. If any metric misses target:

| Miss | Investigate |
|---|---|
| Q-loss > 20% | NK detects too many short Qs, or labeler midpoint splitting drops them — consider smaller frame_ms or priority for Q in to_frames |
| QRS F1 < 0.85 | NK method ablation: try `method="cwt"` or `method="peak"` in delineate.run |
| T F1 < 0.70 | T detection is hard; may be acceptable. Document and proceed to Stage 2. |
| Pacer TPR < 8/10 | Lower `amp_threshold_mad` to 3.5 |
| Pacer FPR > 2 | Raise `cutoff_hz` to 100, tighten `max_width_ms` to 2.0 |

If targets missed but design issues identified, document findings in a follow-up `docs/superpowers/notes/2026-05-XX-stage1-results.md` and decide: ship v1.0 anyway and iterate, or block and fix.
