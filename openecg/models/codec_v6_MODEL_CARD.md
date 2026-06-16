# openecg-codec v6 — layered ECG codec (frame / beat / rhythm)

A small, single-lead ECG **layered codec**: a 500 Hz signal → **three per-sample
label channels** at the input's own resolution.

```
openecg.encode(signal_500hz)  ->  frame[T], beat[T], rhythm[T]
```

| layer | classes (canonical index order) |
|-------|----------|
| **frame** | other, P, QRS, T |
| **beat**  | none, sinus, vpc, paced, fusion, unknown |
| **rhythm**| sinus, avb, paced, afib, bbb, ventricular |

> **v6 over v5: a structural-prior FRAME upgrade.** The frame head is retrained with a
> **data-derived physiological structure loss**, then the beat & rhythm heads are
> re-derived on the new backbone (frozen-head, the codec_v4/v5 trick). Frame boundary-F1
> **0.829 → 0.855 (+0.026)** and median timing **11.6 → 11.1 ms**; beat VPC 0.935 → 0.929
> and hospital rhythm 0.805 → 0.797 (small, near seed-noise). Net: a clear delineation
> gain at a tiny beat/rhythm cost. Still pure-real.

## What changed — a physiological structure loss on the frame head
LUDB labels encode a hard constraint: **the waves P/QRS/T never directly touch** — a
baseline ("other") sample always separates them (empirical per-sample transition
frequency P↔QRS↔T == 0), and each wave is one contiguous run. Over-segmentation (the
known frame weakness — spurious short waves, illegal adjacencies) violates exactly this.
The loss adds two differentiable, data-derived terms on the frame softmax:

- **forbidden-transition penalty** — minimize expected mass on `wave_t → DIFFERENT-wave_{t+1}`
  (pushes a baseline sample between waves; encodes ordering + refractoriness);
- **total-variation contiguity** — suppress short spurious waves.

Sweep optimum: forbid≈8, tv≈0.6 (robust over 3 seeds; a no-struct full-FT control only
reaches 0.832, so the +0.023 gain is the prior, not the retrain). This breaks the prior
"annotation-noise ceiling" of 0.829 that earlier data/head/soft-boundary attempts could not.

## Architecture & build
`vit_transformer_sample_res_convtok_mh_1ch`, 1,159,572 params, single-lead, 10 s @ 500 Hz,
rank-normalized. codec_v6 = **(1)** struct-prior frame retrain (full-FT) → **(2)** frozen
backbone + beat-only retrain on +vitaldb VPC → **(3)** frozen backbone + rhythm-only
natural-prior re-fit + logit-bias. Stages 2-3 recover beat/rhythm that the frame full-FT
slightly perturbed.

## Training data — PURE REAL, human-expert annotation only
No pseudo-labels, no synthetic. LUDB/QTDB/ISP frame, MIT-BIH DS1 + vitaldb-arrhythmia beats
(anesthesiologist-validated VPC), lydus cardiologist hospital rhythm (`conclusion`).

## Evaluation (held-out)
| metric | **codec_v6** | codec_v5 |
|---|---|---|
| frame boundary macro-F1 (LUDB val) | **0.855** | 0.829 |
| frame boundary macro-F1 (LUDB val, clean) | **0.867** | — |
| frame median timing | **11.1 ms** | 11.6 ms |
| beat VPC F1 (MIT-BIH DS2) | 0.929 | 0.935 |
| rhythm macro (lydus hospital test) | 0.797 | 0.805 |

> **Frame "clean" val (0.867).** 10 of the 492 LUDB-val windows (2.0%) carry a P/T
> label but **no QRS** — physiologically impossible; an artifact of LUDB's *per-lead*
> partial annotation (the QRS was simply not marked on that lead). The model is penalised
> for correctly predicting the un-annotated QRS, so 0.855 *understates* delineation.
> Excluding those 10 broken-label windows gives **0.867**. Training was likewise cleaned
> (11 such windows, 0.16%, IGN-masked; builder now guards against this).

### Cross-cohort rhythm audit — CODE-test (independent, cardiologist gold standard)
[CODE-test](https://zenodo.org/records/3765780) = 827 12-lead ECGs, Brazil/TNMG, multi-
cardiologist consensus — **patient-disjoint from training** (lydus=Korea, PTB-XL=Germany),
so an external rare-class generalisation check. Exam-level labels mapped to lead II:
1dAVb→avb, RBBB/LBBB→bbb, AF→afib, SB/ST→sinus (818 single-label exams).

| class | lydus F1 | CODE-test F1 | CODE-test AUROC |
|---|---|---|---|
| sinus | 0.943 | 0.931 | 0.944 |
| avb | 0.760 | 0.778 | **0.994** |
| afib | 0.898 | 0.846 | **0.999** |
| bbb | 0.625 | 0.513 | 0.946 |
| macro | **0.797** | **0.767** | — |

**avb and afib generalise cross-cohort** (near-identical F1 on an unseen country; afib
recall 1.000 on CODE-test). **bbb F1 understates detection:** its ranking is good
(AUROC 0.946, recall 0.87) — the low F1/precision is the rare-class base rate plus the
genuine single-lead limit below, not a detection failure.

## Limitations (read before use)
- **Small beat/rhythm cost** vs v5 (VPC −0.006, rhythm −0.008): the struct full-FT perturbs
  the shared backbone and the frozen-head re-derive recovers ~99%, not 100%. If your use is
  beat/rhythm-critical and not frame-critical, codec_v5 may suit better.
- `ventricular` rhythm untested on hospital data; `fusion`/`paced` beats weak.
- **`paced` rhythm: prefer the rule-based spike detector, not the neural class.** The
  neural `paced` scores recall 0.65 on lydus (`Possible pacing`, uncertain labels) but only
  ~0.02 on *confirmed* pacing (MIT-BIH 102/104/217) — it never learned the pacemaker spike.
  `openecg.detect_pacings` / `is_paced_record` (rule-based) recovers confirmed pacing
  (recall ~0.99 at `threshold_z≈12`; the default `threshold_z=32` is a deliberate
  high-specificity point that under-detects on lower-bandwidth/older sources like mitdb —
  **tune `threshold_z` per source**, it does not transfer). Caveat: this works only when
  the spike is in the signal — modern
  **bipolar** pacing often has no visible spike (much of lydus), an inherent limit no
  method overcomes. Use the spike detector for spike-present pacing; treat the neural
  `paced` class as unreliable.
- **bbb is at the single-lead-II ceiling** (AUROC 0.946, F1 ~0.51–0.57). It detects bbb
  well (recall 0.87 cross-cohort) but over-calls normal sinus (precision ~0.36). This is a
  genuine *information* limit, not a fixable model deficiency: BBB's discriminating
  morphology (rSR′/notched R) lives in V1/V6, while lead II shows only QRS widening — so
  avb (PR/dropped beats) and afib (irregular/no-P) reach AUROC 0.99 but bbb cannot.
  Verified 2026-06-16: neither hemiblock-label cleaning nor threshold recalibration gives a
  transferable gain; closing it needs multi-lead input (out of scope for a single-lead codec).
- Single-lead, 500 Hz, 10 s windows.
- **Research/educational only. Not a medical device; not for diagnosis.**

## Artifacts
| file | size | notes |
|------|------|-------|
| `codec_v6.pt` | 4.9 MB | torch checkpoint (`openecg.load_codec()` default) |
| `codec_v6_int8.onnx` | ~3.8 MB | dynamic int8; `openecg.report()` / `OnnxCodec` default |

ONNX I/O: input `signal (batch, 5000)` float32 (rank-normalized); outputs
`frame_logits (B,5000,4)`, `beat_logits (B,5000,6)`, `rhythm_logits (B,5000,6)`.
