# OpenECG

*Clinically-grounded ECG wave segmentation that ships an int8 TFLite model in 1.5 MB.*

[![PyPI](https://img.shields.io/pypi/v/openecg.svg)](https://pypi.org/project/openecg/)
[![Python](https://img.shields.io/pypi/pyversions/openecg.svg)](https://pypi.org/project/openecg/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

OpenECG ships:

- A **pretrained per-frame P / QRS / T classifier** with a parallel
  boundary-regression head — 0.99 M params, trained on
  LUDB + QTDB + ISP + a synthetic AV-block mix. Beats NeuroKit2 DWT and
  WTdelineator (Martínez 2004) on every public benchmark we tested
  (see [Performance](#performance)).
- A **TFLite int8 deploy artifact** (~1.5 MB) bundled inside the wheel
  so `Inference()` works with no extra downloads — usable on Android,
  iOS, Raspberry Pi, AED-class embedded targets. Inference uses only
  `tflite-runtime` (~5 MB) + numpy; no PyTorch, no TensorFlow.
- A **layered ECG codec** (`openecg.encode`, `openecg.encode_stream`)
  that emits sample-resolution frame / beat / rhythm label channels.
- **Loaders** for LUDB, QTDB, ISP, BUT PDB, PTB-XL and the
  synthetic AV-block dataset so every number in this README is
  reproducible from a clean clone.

## Install

```bash
# Core (numpy-only): tokenizer + signal processing primitives
pip install openecg

# Structured report on the int8 ONNX codec — torch-free (onnxruntime + numpy)
pip install "openecg[report]"

# Inference (ships int8 TFLite boundary model + ONNX codec — no torch needed)
pip install "openecg[deploy]"

# Training / evaluation (loaders, stage2 transformer, optional NeuroKit2)
pip install "openecg[loaders,stage2]"
```

PyTorch is **only** required for training and for `[deploy-export]`. The
inference paths are torch-free: TFLite (`tflite-runtime`) for the boundary
detector, ONNX (`onnxruntime`) for the layered codec and `openecg.report()`.

## Quickstart

### Boundary detection from a numpy signal

```python
import numpy as np
from openecg.deploy import Inference

# Loads the bundled v56c int8 TFLite model — no download needed.
det = Inference()

# Any 1-D float array at 250 Hz; e.g. one lead of a 10-second clip.
ecg_250hz = np.load("my_ecg.npy")                 # shape (N,) at 250 Hz

# Slides a 10-s window with no overlap; trailing samples are zero-padded.
windows = det.predict(ecg_250hz)
for w in windows:
    for b in w:
        print(b.name, b.start, b.end)             # "P 145 215", "QRS 320 365", ...
```

`b.start` / `b.end` are sample-indexed (0-based) inside the 10-s
window. Each window yields up to ~50 boundaries (P + QRS + T per beat).
The model expects **single-channel input at 250 Hz** — resample
upstream if your source is 500 / 1000 Hz.

### Layered codec — three label streams, one call

```python
import openecg
codec = openecg.encode(ecg_500hz, fs=500, model="default")   # 10-s window @ 500 Hz
codec.channels                              # uint8 (3, 5000) at sample resolution
codec.frame, codec.beat, codec.rhythm       # per-layer views
codec.unified                               # merged frame+beat: 1 track, 8 classes
codec.events("beat", drop_class=0)          # [(start, end, class_id), ...]
codec.to_codec_string(layer="frame")        # ASCII rendering
```

`codec.unified` collapses the frame and (QRS-gated) beat tracks into one readable
8-class per-sample stream — `other / P / T / sinus / vpc / paced / fusion / unknown`
(each QRS replaced by its beat type). A consumption convenience derived from the two
trained heads (the model keeps separate frame+beat heads — heterogeneous label sources
make a single trained head impractical, and a merged-head ablation showed a small beat
cost for no frame gain).

The bundled codec is **codec_v6** (`openecg.load_codec()`, 1.16 M params), trained
on **pure real, human-expert annotations only** — including **lydus cardiologist
hospital rhythm** — no synthetic, no pseudo-labels. Held-out:

- **frame** boundary macro-F1 **0.855** (**0.867** excl. 10 broken-label LUDB windows),
  median timing **11.1 ms** (LUDB, 500 Hz)
- **beat** sinus F1 **0.99** / VPC **0.929** (MIT-BIH DS2)
- **rhythm** on real hospital ECG (lydus-test, n=22 229) macro **0.797**; cross-cohort
  audit on **CODE-test** (827, Brazil cardiologist gold) macro **0.767** — avb/afib
  generalise (AUROC 0.99), bbb is at the single-lead-II ceiling (AUROC 0.946)

**v6 over v5: a structural-prior FRAME upgrade.** The frame head is retrained with a
**data-derived physiological structure loss** — from LUDB labels the waves P/QRS/T
*never directly touch* (a baseline sample always separates them), so the loss penalizes
forbidden wave→wave transitions + adds a total-variation contiguity term, directly
fixing the over-segmentation that capped frame quality. This lifts boundary-F1
**0.829 → 0.855 (+0.026)** and timing 11.6 → 11.1 ms — breaking the prior
"annotation-noise ceiling". The beat & rhythm heads are then re-derived on the new
backbone (frozen-head, +vitaldb VPC and natural-prior), recovering them to VPC 0.929 /
rhythm 0.797 (small ~0.01 cost). Earlier codec_v5 was a pure rhythm-calibration upgrade
(bbb 0.618 → 0.651). See the [model card](openecg/models/codec_v6_MODEL_CARD.md).
`encode()` auto rank-normalizes its input. Deploy artifacts (int8 ONNX) ship in
`openecg/models/`.

The three channels run in parallel at the input signal's sample rate.
Each layer is a separate label stream at a different abstraction — wave
boundaries on the bottom, beat type per QRS in the middle, rhythm class
on top — and segment starts / ends are recoverable from class
transitions on any single channel.

| Layer | Granularity | Classes | Convention aligned with |
|---|---|---|---|
| 0 `frame` | per sample | other / P / QRS / T / paced_QRS | LUDB, QTDB, ISP wave annotation |
| 1 `beat` | per QRS span | none / sinus / VPC / paced / fusion / unknown | WFDB AAMI EC57 beat codes (N/V/F/`/`/Q) |
| 2 `rhythm` | per sample (sub-window) | sinus / AVB / paced / AFib / BBB / ventricular | WFDB rhythm aux notes `(N` `(AFIB` `(VT` ... |

VTach, AFib, and similar rhythm-level events sit on layer 2 — they're
*not* a new frame class, matching MIT-BIH / WFDB convention. See
`openecg/layered.py` for the predictor-injection points.

### Structured report — one call, agent-ready JSON

For monitoring agents (or anyone who wants the conclusion, not the per-sample
channels), `openecg.report()` fuses the codec with the pure-numpy QRS detector
and the rule-based AFib check into one JSON-serialisable reading:

```python
import openecg
rep = openecg.report(ecg_500hz, fs=500)      # torch-free: runs the int8 ONNX codec
rep.summary
# 'Sinus rhythm, HR 84 bpm, regular. No ectopy. (AFib rule: negative.)'
rep.to_json(indent=2)                        # drop straight into an LLM / API payload
```

```jsonc
{
  "rhythm":      {"label": "sinus", "confidence": 1.0, "distribution": {"sinus": 1.0}},
  "heart_rate":  {"bpm": 84, "rr_mean_ms": 714, "rr_cv": 0.03, "regularity": "regular"},
  "beats":       {"count": 14, "by_type": {"sinus": 14}, "vpc_count": 0, "events": [...]},
  "intervals_ms":{"p_duration": 112, "qrs_duration": 103, "t_duration": 178,
                  "pr": 148, "qt": 370},          // wave intervals from the delineator
  "afib_check":  {"is_afib": false, "reason": "no rule fired", "agrees_with_codec": true},
  "flags":       []                                // bradycardia / tachycardia / ectopy / ...
}
```

Heart rate and R-peaks come from the validated `detect_qrs`; rhythm, beat type
and wave intervals from the codec; AFib is an **independent rule-based second
opinion** cross-checked against the codec's rhythm head (a disagreement raises a
flag). By default it runs the **int8 ONNX codec** (`pip install openecg[report]`
— onnxruntime + numpy, **no PyTorch**); `model="torch"` uses the checkpoint,
`model="rules"` skips the codec entirely (heart rate + AFib on pure numpy).

### Continuous-use codec — 2-s edge guard

Predictions in the outer **2 seconds** of any 10-s window have limited
past / future context. Held-out evaluation and stream stitching exclude
these samples so adjacent windows can be concatenated **seamlessly**:

```python
codec = openecg.encode(ecg_250hz)            # 10-s window
codec.eval_mask                               # bool[2500], True only in [2s, 8s]
codec.events("beat", eval_only=True)         # restricts to the inner band
inner = codec.inner()                         # sliced copy, margin=0

# Long signal -> sliding inference with stride = window - 2*margin = 6 s.
# Every emitted sample had ≥2 s of past AND ≥2 s of future context.
holter_codec = openecg.encode_stream(long_signal_250hz)   # arbitrary length
```

The same philosophy applies to training and evaluation: loss is computed
on the full 10-s window (the model needs context to learn), but
held-out metrics are masked to the inner 6-s band. This is the only way
to get a codec whose output is **continuously stitchable** without
boundary artefacts — a prerequisite for any honest foundation-model
claim.

## Performance

The shipped model is **v56c** — `vit_transformer_noaux_1ch`, L8/d=128
(0.99 M params), trained with soft-T α=0.9 on LUDB + QTDB + ISP +
synthetic AV-block data and rank-normalised input. The exported TFLite
int8 is bit-equivalent (Δ macro-F1 = -0.0025 vs torch fp32).

Macro-F1 across the six P / QRS / T on/off boundaries, with
Martínez 2004 tolerances (P 50 ms, QRS 40 ms, T_on 50 ms, T_off 100 ms),
**lead II only**:

| Dataset (n records) | **openecg v56c** | NeuroKit2 DWT | WTdelineator |
|---|---:|---:|---:|
| LUDB val (41)          | **0.963** | 0.788 | 0.596 |
| ISP test (72)          | **0.971** | 0.703 | 0.604 |
| QTDB T-subset (44)     | **0.908** | 0.605 | 0.535 |

openecg also hits **≤16 ms median timing error on every boundary**,
meeting the clinical 20 ms spec target — the wavelet baselines miss it
on T_off (~44 ms) and on every P boundary. Full per-boundary
F1 / Se / P+ / SD / median error tables are in
[`docs/benchmarks/v56c_vs_baselines.md`](docs/benchmarks/v56c_vs_baselines.md).

### Representative cases

Each figure overlays the four detectors on the same ECG strip from each
benchmark dataset, lead II. P = red, QRS = blue, T = green; shaded
regions are the predicted wave durations, vertical ticks at the top
mark predicted onsets and offsets.

**LUDB val record 16** — clean sinus rhythm with prominent P / QRS / T;
openecg matches the cardiologist annotation, NeuroKit2 places P
boundaries off the true wave and the WTdelineator drops most P / T
detections after the first beat.

![LUDB comparison](docs/figures/v56c_vs_baselines_ludb.png)

**ISP test record 2** — dense rhythm with subtle P and biphasic T;
openecg locks onto every beat, NeuroKit2 misses the first beat entirely
and shifts QRS/T positions, WTdelineator's T spans run far beyond
the true T-wave.

![ISP comparison](docs/figures/v56c_vs_baselines_isp.png)

**QTDB record sel100 (MLII)** — low-amplitude T waves, the regime
where wavelet methods struggle. openecg keeps tight P and QRS spans
on every beat; NeuroKit2 produces sporadic T detections far from the
T wave; WTdelineator drops two of the four beats.

![QTDB comparison](docs/figures/v56c_vs_baselines_qtdb.png)

Reproduce these figures:

```bash
python -m scripts.viz_benchmark_v56c
# writes docs/figures/v56c_vs_baselines_{ludb,isp,qtdb}.png
```

```bash
python -m scripts.benchmark_v56c --leads ii --out out/benchmark_v56c.json
```

### Deploy footprint

| Path | Size | Macro-F1 | Latency / 10-s window |
|---|---:|---:|---:|
| Torch fp32 (training) | 4.0 MB | 0.9299 | 44 ms |
| TFLite fp32           | 4.4 MB | 0.9299 | 44 ms |
| **TFLite int8 (bundled)** | **1.5 MB** | **0.9274** | 44 ms |

We benchmarked ExecuTorch on the same checkpoint and TFLite int8 won
by 3.5× on latency and -0.004 less F1 loss — TFLite stays canonical
until ExecuTorch ships a weight-only int8 recipe. See
[`docs/benchmarks/v56c_vs_baselines.md`](docs/benchmarks/v56c_vs_baselines.md)
for the full backend comparison.

## Optional extras

`pyproject.toml` declares optional dependency groups so each install is
minimal:

- `[deploy]` — `tflite-runtime` + `numpy`; what end users install. Pulls
  the bundled `.tflite` model from the wheel; no PyTorch needed.
- `[loaders]` — `wfdb` + `scipy` for LUDB / ISP / QTDB / BUT PDB / PTB-XL.
- `[stage2]` — torch + transformers for the training-time backbones.
- `[delineate]` — NeuroKit2 + scipy for the baseline comparison.
- `[deploy-export]` — torch + ai-edge-torch for re-exporting the
  `.tflite` from a torch checkpoint (Linux / WSL only).

```bash
pip install "openecg[deploy]"            # end-user inference
pip install "openecg[loaders,delineate]" # reproduce the benchmark table
```

## Toward an ECG foundation model

OpenECG's design treats the three-channel **layered codec** as the
output interface of a single foundation model:

> Wave → beat → rhythm, all at sample resolution, stitchable across
> windows, learned jointly from every public corpus that carries the
> matching label level.

The three pieces required to call this a foundation model are:

1. **A common output schema across datasets.** No dataset is large
   enough on its own — LUDB has wave labels for 200 records, MIT-BIH
   Arrhythmia has beat labels for 48 — but each is a partial label of
   the same codec. Training is multi-task with a per-sample loss mask
   over the layers each dataset annotates.
2. **A continuously-stitchable output.** Honest inference on Holter or
   24-h streams requires that adjacent windows produce a seamless
   codec. The 2-s edge guard (see above) is the mechanism: training and
   evaluation only count the inner band, so the model is *rewarded* for
   producing predictions that are stable when re-evaluated 2 s later
   with new future context.
3. **Convention alignment.** Frame layer matches LUDB / QTDB / ISP;
   beat layer matches WFDB AAMI EC57 codes; rhythm layer matches WFDB
   aux-note rhythms. A foundation model that invents its own taxonomy
   is unusable downstream.

### Public datasets, mapped to codec layers

The annotation level dictates which layer's loss is unmasked for each
record. Datasets carrying both beat *and* rhythm annotations (in **bold**
below) supervise two layers simultaneously and are the spine of the
multi-task pool.

| Dataset | Records | Hours | Layer 0 (frame) | Layer 1 (beat) | Layer 2 (rhythm) |
|---|---:|---:|:---:|:---:|:---:|
| LUDB                          |    200 |  0.6 | ✓ |   |   |
| QTDB                          |    105 |  0.9 | ✓ | ✓ |   |
| ISP                           |    160 |   —  | ✓ |   |   |
| BUT PDB                       |     50 |  1.7 | ✓ (P-peak) |   | ✓ (AVB) |
| **MIT-BIH Arrhythmia**        |     48 |   24 |   | ✓ | ✓ |
| **MIT-BIH SVDB**              |     78 |   39 |   | ✓ | ✓ |
| **MIT-BIH LTDB**              |      7 |  ~120 |   | ✓ | ✓ |
| MIT-BIH NSR                   |     18 |  432 |   | ✓ (sinus) |   |
| MIT-BIH AFDB                  |     25 |  250 |   |   | ✓ (AFib) |
| **MIT-BIH MVE (VFDB)**        |     22 |   11 |   | ✓ | ✓ (VT/VF) |
| MIT-BIH Polysomnographic      |     18 |  ~110 |   | ✓ |   |
| Fantasia                      |     40 |   80 |   | ✓ (sinus) |   |
| **Sudden Cardiac Death Holter** | 23 |   552 |   | ✓ | ✓ |
| INCART (12-lead)              |     75 |   37 |   | ✓ |   |
| European ST-T                 |     90 |  180 |   | ✓ |   |
| BIDMC CHF                     |     15 |  300 |   | ✓ |   |
| PTB-XL                        | 21,837 |   61 |   |   | ✓ (SCP, window) |
| Chapman-Shaoxing 12-lead      | 10,646 |   30 |   |   | ✓ (window) |

**Excluded by design.** CinC 2021 (SNOMED codes aggregated from six
heterogeneous sources), Icentia 11k (model-predicted pseudo-labels),
and CODE-15% (AI-derived binary diag flags) are deliberately *not* in
the foundation pool. The codec layers are defined relative to
human-expert annotation conventions (cardiologist wave boundaries, AAMI
beat codes, WFDB rhythm aux-notes); mixing in machine-derived labels
would erode the very ground truth the codec is supposed to represent.
These corpora remain useful for downstream external validation but
not for training the codec heads.

All listed corpora are PhysioNet- or Zenodo-distributed (CC-BY /
ODC-BY). Loaders for LUDB / QTDB / ISP / BUT PDB / PTB-XL ship in
`openecg.*`; the remaining ones are pulled in over the standard WFDB
interface during multi-task training.

The dataset survey above is mirrored in
[`G:\Shared drives\Datasets\ECG\DATASETS.md`](.) for the SNUH research
group; the public OpenECG repo will publish a smaller `DATASETS.md`
restricted to the open corpora once the multi-task model ships.

### Status

| Component | Today | Roadmap (v0.5+) |
|---|---|---|
| Layer 0 — frame delineator        | v56c TFLite int8 (bundled, 1.5 MB) | Re-export v56d (AVB-augmented) once `torch.int1` mismatch resolved |
| Layer 1 — per-beat classifier     | rule-stub (paced / VT-rhythm fallback) | Multi-head v57 trained on MIT-BIH Arrhythmia + SVDB + LTDB + INCART + Fantasia |
| Layer 2 — rhythm classifier       | 6-class CNN (`openecg.rhythm`), window-constant | Per-patch head from multi-head v57 → sub-window rhythm segmentation |
| 2-s edge-guarded codec            | `eval_mask` / `eval_only` / `encode_stream` | Used as the gating metric for all v57 training checkpoints |
| Continuous-stitch inference       | `openecg.encode_stream(signal)` works today | Native multi-window deploy path in TFLite |

The single-pass multi-head architecture lives at
`openecg.stage2.model_variants.FrameClassifierTransformerLayered1Ch`
(arch id `vit_transformer_layered_1ch`). It loads from the v56d weight
file via `strict=False`; the new beat / rhythm heads start zero-init so
the untrained model emits safe defaults (`BEAT_NONE` / `RHYTHM_SINUS`)
until per-layer supervision lands.
