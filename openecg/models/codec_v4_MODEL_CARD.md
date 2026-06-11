# openecg-codec v4 — layered ECG codec (frame / beat / rhythm)

A small, single-lead ECG **layered codec**: maps a 500 Hz ECG signal to **three
per-sample label channels** at the input's own resolution —

```
openecg.encode(signal_500hz)  ->  frame[T], beat[T], rhythm[T]
```

| layer | classes (canonical index order) |
|-------|----------|
| **frame** | other, P, QRS, T |
| **beat**  | none, sinus, vpc, paced, fusion, unknown |
| **rhythm**| **sinus, avb, paced, afib, bbb, ventricular** |

> **v4 over v3: a pure VPC-beat upgrade.** Only the beat head is retrained — on
> +vitaldb anesthesiologist-validated intra-operative VPC beats. **Frame and
> rhythm are byte-identical to codec_v3** (their weights are frozen), so there is
> **zero regression** on delineation or hospital rhythm: VPC F1 **0.858 → 0.935**
> is the only change. Still pure-real.

## Architecture
`vit_transformer_sample_res_convtok_mh_1ch` — conv-tokenizer ViT, sample-
resolution head, three parallel per-sample heads, no regression head.
**1,159,572 parameters.** Single-lead, 10 s window = 5000 samples @ 500 Hz,
rank-normalized; output length == input length.

codec_v4 = codec_v3 with the **backbone + frame head + rhythm head frozen** and
**only the 774-parameter beat head retrained**. Every non-beat tensor is
bit-for-bit codec_v3 (verified). So frame/rhythm outputs are *identical* to
codec_v3; the beat head simply learned a better VPC decision boundary from a
second expert domain.

## Training data — PURE REAL, human-expert annotation only
No pseudo-labels, no synthetic data (CinC2021 / Icentia11k / CODE-15 and all
model-generated labels excluded). PTB-XL, Chapman, lydus **and vitaldb** are
clinician-validated → human-expert. **lydus rhythm is from the cardiologist
over-read (`conclusion`); vitaldb beats are anesthesiologist-validated (κ = 0.93,
a DL classifier only screened candidates — final labels are expert).**

The frame and rhythm layers are inherited unchanged from codec_v3 (see
`codec_v3_MODEL_CARD.md` for their full source list). The **beat** head retrain
adds one source:

| layer | expert sources |
|-------|----------------|
| beat (v3) | MIT-BIH Arrhythmia DS1 + INCART + LTDB + SVDB + MIT-BIH paced (102/104/217) |
| **beat (v4, added)** | **+ vitaldb-arrhythmia (482 surgical patients, 612 k expert VPC beat-samples; rhythm labels masked out so hospital rhythm is unaffected)** |

> vitaldb's *rhythm* labels (afib-heavy, no bbb/paced — an intra-operative
> distribution) were deliberately **excluded** from supervision: a full
> fine-tune that used them lifted VPC but cost −0.024 hospital-rhythm macro. The
> frozen beat-only recipe keeps the VPC win with no rhythm cost.

## Evaluation

**(A) Frame delineation — IDENTICAL to codec_v3** (weights frozen):
LUDB held-out per-sample P/QRS/T **0.827** (P 0.801 / QRS 0.850 / T 0.831);
boundary macro-F1 **0.840**, median timing **8.7 ms** (Martinez tolerances, 500 Hz).

**(B) Rhythm — IDENTICAL to codec_v3** (weights frozen):
hospital held-out (lydus-test, patient-disjoint) macro **0.79** — sinus 0.94 /
avb 0.76 / paced 0.76 / afib 0.88 / bbb 0.62. (PTB-XL-distribution macro 0.74.)

**(C) Beat — MIT-BIH DS2 held-out (record-disjoint, 49,555 expert beats) — the v4 change**

| class | **codec_v4** | codec_v3 |
|---|---|---|
| sinus | **0.992** | 0.985 |
| **vpc** | **0.935** (P 0.94 / R 0.94) | 0.858 (P 0.78 / R 0.95) |

The VPC gain is a **precision** fix (0.78 → 0.94 at unchanged recall): codec_v3
over-called VPCs; vitaldb's real VPC beats taught the head to stop. The gain is
robust (five independent vitaldb runs all 0.915–0.943) and **cross-domain**
(intra-operative training → ambulatory MIT-BIH evaluation).

## Limitations (read before use)
- **`ventricular` rhythm is untested on hospital data**; `fusion`/`paced` beats
  remain weak/untested in DS2. Use beat for **sinus / VPC** discrimination.
- Rhythm performance is **eval-distribution-dependent** (hospital ≫ v2; PTB-XL < v2).
- Single-lead, 500 Hz, 10 s windows. **Research/educational only. Not a medical
  device; not for diagnosis.**

## Artifacts
| file | size | notes |
|------|------|-------|
| `codec_v4.pt` | 4.9 MB | torch checkpoint (`openecg.load_codec()` default) |
| `codec_v4_int8.onnx` | ~3.8 MB | dynamic int8 (MatMul); argmax ≈ fp32 (0.993 agree) |

ONNX I/O: input `signal (batch, 5000)` float32 (rank-normalized);
outputs `frame_logits (B,5000,4)`, `beat_logits (B,5000,6)`, `rhythm_logits (B,5000,6)`.
