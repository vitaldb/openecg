# openecg-codec v3 — layered ECG codec (frame / beat / rhythm)

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

> v3 over v2 (= the shipped `codec_v1.pt`): **adds cardiologist hospital rhythm
> (lydus) → fixes the avb / paced / bbb rhythm classes that v1/v2 flagged as
> "experimental, do not rely"**, at a small delineation cost. Still pure-real.

## Architecture
`vit_transformer_sample_res_convtok_mh_1ch` — conv-tokenizer ViT, sample-
resolution head, three parallel per-sample heads, no regression head.
**1,159,572 parameters.** Single-lead, 10 s window = 5000 samples @ 500 Hz,
rank-normalized; output length == input length. Initialised from codec_v2 and
fine-tuned (frame-boosted sampling to protect delineation).

## Training data — PURE REAL, human-expert annotation only
No pseudo-labels, no synthetic data (CinC2021 / Icentia11k / CODE-15 and all
model-generated labels excluded). PTB-XL, Chapman **and lydus** are
cardiologist-validated → human-expert. **lydus rhythm is derived from the
cardiologist over-read (`conclusion`), never the machine statement.**

| layer | expert sources |
|-------|----------------|
| frame | LUDB + QTDB + ISP (P/QRS/T) + BUT-PDB (P/QRS, incl. AV-block) |
| beat  | MIT-BIH Arrhythmia DS1 + INCART + LTDB + SVDB + MIT-BIH paced (102/104/217) |
| rhythm| AFDB (afib) + VFDB/CUDB/MVEDB/SDDB/SCDDB (ventricular) + NSRDB (sinus) + BUT-PDB (avb) + PTB-XL (bbb/paced/afib/sinus) **+ lydus (49,492 hospital windows: avb/paced/afib/bbb/sinus, cardiologist `conclusion`)** |

Real expert pool (≈25.6k windows) + lydus (49.5k rhythm windows), source-balanced,
record/patient-disjoint splits. Each window supervises only the layer(s) it has
expert GT for (masking — a source never labels a layer it doesn't annotate).

> A controlled synthetic-augmentation ablation was run and **rejected**: synth
> (synarrdb) was net-neutral-to-negative on every held-out class real data covers.
> A 2³ factorial (mask × lydus × synth, 3 seeds) isolated the effect of each —
> **lydus is the rare-rhythm engine, masking is essential for delineation, synth
> earns no place.** This codec is pure-real *by evidence*.

## Evaluation — the rhythm gain is **distribution-specific** (read carefully)

The rhythm classes' value depends on the evaluation distribution. v2 was strong on
PTB-XL-distribution rhythm but near-blind on real hospital ECG; v3 (lydus) inverts
the rare-class story for the hospital distribution it targets.

**(A) Frame delineation — LUDB held-out (per-sample P/QRS/T F1)**

| | P | QRS | T | **mean** |
|---|---|---|---|---|
| **codec_v3** | 0.801 | 0.850 | 0.831 | **0.827** |
| codec_v2 (shipped) | 0.819 | 0.877 | 0.854 | 0.850 |
| v56c TFLite (framing-only) | 0.714 | 0.795 | 0.748 | 0.752 |

v3 frame is **+0.075 over the previous framing-only delineator**, **−0.023 vs
codec_v2** — the (small) cost of adding rhythm-distribution coverage.

Clinical boundary metric (P/QRS/T on/off macro-F1 at Martinez tolerances, 500 Hz):
**0.840, median timing error 8.7 ms** — sample-resolution localization (vs the
bundled 250 Hz/20 ms-grid TFLite delineator).

**(B) Rhythm — hospital held-out (lydus-test, patient-disjoint, per-class F1)**

| class | **codec_v3** | codec_v2 |
|---|---|---|
| sinus | 0.94 | 0.87 |
| **avb** | **0.76** | 0.00 |
| **paced** | **0.76** | 0.22 |
| **afib** | **0.88** | 0.74 |
| **bbb** | **0.62** | 0.19 |
| ventricular | — (no lydus source) | — |
| **macro** | **0.79** | 0.40 |

PTB-XL-distribution rhythm (in-distribution for v2): v3 macro **0.74** vs v2 0.81 —
the flip side of the trade. **Use the codec's rhythm output for hospital-style
ECG; for PTB-XL-style data v2's rhythm is stronger.**

## Limitations (read before use)
- **`ventricular` rhythm is untested on hospital data** (lydus has no ventricular
  source; it is trained from VFDB/CUDB in the real pool but not audited here).
- Rhythm performance is **eval-distribution-dependent** (hospital ≫ v2; PTB-XL < v2).
- **Beat — MIT-BIH DS2 held-out** (record-disjoint, 49,555 expert beats):
  sinus F1 **0.985**, vpc F1 **0.858** (recall 0.95); `fusion` F1 0.04 and
  `paced`/`unknown` are weak/untested (rare or absent in DS2). Use beat for
  sinus/VPC discrimination; do not rely on fusion/paced.
- Single-lead, 500 Hz, 10 s windows. **Research/educational only. Not a medical
  device; not for diagnosis.**

## Artifacts
| file | size | notes |
|------|------|-------|
| `codec_v3.pt` | 4.9 MB | torch checkpoint (`openecg.load_codec()` default) |
| `codec_v3_int8.onnx` | ~3.8 MB | dynamic int8 (MatMul); argmax ≈ fp32 |

ONNX I/O: input `signal (batch, 5000)` float32 (rank-normalized);
outputs `frame_logits (B,5000,4)`, `beat_logits (B,5000,6)`, `rhythm_logits (B,5000,6)`.

— Staged for review; not yet released.
