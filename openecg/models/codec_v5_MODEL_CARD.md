# openecg-codec v5 — layered ECG codec (frame / beat / rhythm)

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

> **v5 over v4: a pure rhythm-head upgrade.** Only the rhythm head changes — it is
> re-fit at the **natural hospital class prior** and **logit-bias calibrated**.
> **Frame and beat are byte-identical to codec_v4** (backbone + frame head + beat
> head weights are bit-for-bit v4, verified by state-dict diff), so there is **zero
> regression** on delineation (frame boundary-F1 **0.840**) or VPC (DS2 **0.935**).
> Hospital rhythm macro-F1 **0.792 → 0.805** is the only change. Still pure-real.

## Architecture
`vit_transformer_sample_res_convtok_mh_1ch` — conv-tokenizer ViT, sample-
resolution head, three parallel per-sample heads. **1,159,572 parameters.**
Single-lead, 10 s window = 5000 samples @ 500 Hz, rank-normalized; output length
== input length.

codec_v5 = codec_v4 with the **backbone + frame head + beat head frozen** and
**only the rhythm head retrained** (then a per-class logit bias folded into
`rhythm_sample.bias`). Every non-rhythm tensor is bit-for-bit codec_v4 (verified),
so frame/beat outputs are *identical* to codec_v4.

## What changed and why — class-prior calibration
The rhythm head's weakness was **not feature quality but calibration**. The lydus
training set is sinus-*capped* (≈27 % sinus, balanced for learning), but the
deployment/hospital test prior is sinus-*heavy* (≈70 %). codec_v4's head, plus any
retrain on the balanced data, **over-calls rare classes** on the natural-prior test
(bbb precision 0.57). codec_v5 fixes this two ways, both zero-regression:

1. **Natural-prior re-fit** — the rhythm linear head is retrained with the sampler
   reweighted to the natural class prior (frame/beat frozen).
2. **Logit-bias calibration** — a per-class additive bias, tuned on a prior-matched
   held-out dev split, folded into the head bias.

A full fine-tune and a balanced-prior retrain were both tried and **rejected** (they
moved calibration the wrong way; rhythm *features* are at ceiling for single lead II).

## Training data — PURE REAL, human-expert annotation only
No pseudo-labels, no synthetic data. Frame and beat layers are inherited unchanged
from codec_v4 (see `codec_v4_MODEL_CARD.md`). The rhythm head re-fit uses the same
**lydus cardiologist over-read (`conclusion`)** rhythm labels as v3/v4 — only the
*sampling prior and calibration* changed, no new data source.

## Evaluation

**(A) Frame delineation — IDENTICAL to codec_v4** (weights frozen):
LUDB held-out per-sample P/QRS/T **0.827**; boundary macro-F1 **0.840**, median
timing **8.7 ms** (Martinez tolerances, 500 Hz).

**(B) Beat — IDENTICAL to codec_v4** (weights frozen):
MIT-BIH DS2 sinus **0.992** / **VPC 0.935** (P 0.94 / R 0.94).

**(C) Rhythm — hospital held-out (lydus-test, patient-disjoint, n=22,229) — the v5 change**

| class | **codec_v5** | codec_v4 |
|---|---|---|
| sinus | **0.946** | 0.936 |
| avb | **0.767** | 0.758 |
| paced | **0.768** | 0.761 |
| afib | **0.890** | 0.886 |
| **bbb** | **0.651** | 0.618 |
| **macro** | **0.805** | 0.792 |

Every class improves; **bbb +0.033** (the rare-class equity target — over-calling
fixed, precision 0.57 → 0.62). Window accuracy 0.882 → 0.899. Selection used a
prior-matched dev split (never the test set).

**Robustness (4-seed replication).** The natural-prior re-fit was repeated at 4
independent seeds (rtx4090/5090/3090/v100): held-out test macro **0.800–0.806, every
seed > codec_v4's 0.792** (mean ≈ 0.803), bbb 0.63–0.65 throughout. The improvement is
not seed-luck. The **natural-prior re-fit is the robust driver**; the per-class
logit-bias is a *marginal* per-model calibration (tuned on 3272 dev windows — it helped
the shipped seed but was within noise on others). codec_v5 is the seed at the top of
this cluster.

## Limitations (read before use)
- **`ventricular` rhythm is untested on hospital data**; `fusion`/`paced` beats
  remain weak/untested in DS2. Use beat for **sinus / VPC** discrimination.
- **bbb is lead-II-limited** — bundle-branch block is best seen in V1/V6; single-lead
  rhythm is near its information ceiling for bbb. Rhythm performance is
  **eval-distribution-dependent** (hospital ≫ PTB-XL).
- Single-lead, 500 Hz, 10 s windows. **Research/educational only. Not a medical
  device; not for diagnosis.**

## Artifacts
| file | size | notes |
|------|------|-------|
| `codec_v5.pt` | 4.9 MB | torch checkpoint (`openecg.load_codec()` default) |
| `codec_v5_int8.onnx` | ~3.8 MB | dynamic int8 (MatMul); argmax ≈ fp32. `openecg.report()` / `OnnxCodec` default |

ONNX I/O: input `signal (batch, 5000)` float32 (rank-normalized);
outputs `frame_logits (B,5000,4)`, `beat_logits (B,5000,6)`, `rhythm_logits (B,5000,6)`.
