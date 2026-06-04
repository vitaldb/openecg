# openecg-codec v1 — layered ECG codec (frame / beat / rhythm)

A small, single-lead ECG **layered codec**: maps a 500 Hz ECG signal to **three
per-sample label channels** at the input's own resolution —

```
openecg.encode(signal_500hz)  ->  frame[T], beat[T], rhythm[T]
```

| layer | classes |
|-------|---------|
| **frame** | other, P, QRS, T |
| **beat**  | none, sinus, vpc, paced, fusion, unknown |
| **rhythm**| sinus, avb, paced, afib, bbb, ventricular |

## Architecture
`vit_transformer_sample_res_convtok_mh_1ch` — conv-tokenizer ViT, sample-
resolution head (nearest-upsample + zero-init refine conv + per-sample linear),
three parallel per-sample heads, no regression head. **1,159,572 parameters.**
Single-lead, 10 s window = 5000 samples @ 500 Hz, rank-normalized; output length
== input length.

## Training data — PURE REAL, human-expert annotation only
No pseudo-labels, no synthetic data (CinC2021 / Icentia11k / CODE-15 and all
model-generated labels are excluded by design). PTB-XL and Chapman are
cardiologist-validated and therefore count as human-expert.

| layer | expert sources |
|-------|----------------|
| frame | LUDB + QTDB + ISP (P/QRS/T delineation) + BUT-PDB (P/QRS, incl. AV-block) |
| beat  | MIT-BIH Arrhythmia DS1 + INCART + LTDB + SVDB + MIT-BIH paced (102/104/217) |
| rhythm| AFDB (afib) + VFDB/CUDB/MVEDB/SDDB/SCDDB (ventricular) + NSRDB (sinus) + BUT-PDB (AV block) + PTB-XL (bbb/paced/afib/sinus, window-level) |

25,593 training windows, record/fold-disjoint train/val/test. Each window
supervises only the layer(s) with expert GT (rest ignored in loss). Early-stop
on a real-only, record-disjoint validation set.

> A controlled synthetic-augmentation sweep was run and **rejected** — adding
> synthetic data did not improve, and degraded, the rare classes real data
> already covers (e.g. real afib F1 0.73→~0.48). This codec is pure-real by
> evidence, not only by principle.

## Evaluation — held-out, record/fold-disjoint TEST
MIT-BIH DS2 beat (49,555 expert beats) + held-out AFDB/VFDB/NSRDB rhythm:

| layer.class | precision | recall | F1 | support |
|-------------|-----------|--------|----|---------|
| beat sinus  | 0.98 | 0.98 | **0.985** | 45,947 |
| beat vpc    | 0.87 | 0.90 | **0.884** | 3,213 |
| beat fusion | 0.30 | 0.31 | 0.307 | 388 |
| rhythm sinus| 0.92 | 0.88 | **0.899** | 1,271 |
| rhythm afib | 0.77 | 0.81 | **0.791** | 352 |
| rhythm vent | 0.51 | 0.38 | 0.436 | 63 |

## Limitations (read before use)
- **`bbb`, `paced`, `avb` rhythm are EXPERIMENTAL / weak.** Trained (PTB-XL
  bbb/paced/afib; BUT-PDB avb) but held-out recall is low — PTB-XL labels are
  window-level while the head is per-sample and sinus dominates, so these rare
  rhythms are under-emitted. Held-out PTB-XL: afib F1 0.63, bbb F1 0.14
  (recall ~0.10), paced F1 0.06; avb has no held-out labels to score.
  **Do not rely on bbb/paced/avb rhythm output.** (Loss reweighting and
  oversampling were tried and did not fix it; a window-level pooled rhythm head
  is the planned next step.)
- `fusion` / `unknown` beat are weak (rare in the pool).
- Single-lead, 500 Hz, 10 s windows. Research and educational use only.
  **Not a medical device; not for diagnosis.**

## Artifacts
| file | size | notes |
|------|------|-------|
| `codec_v1.pt` | 4.9 MB | torch checkpoint (`openecg.load_codec()`) |
| `codec_v1_int8.onnx` | 3.78 MB | dynamic int8 (MatMul); argmax agrees 99.4% with fp32 |
| `openecg_codec_v1.onnx` (deploy dir) | 5.38 MB | fp32, single file, opset 17 |

ONNX I/O: input `signal (batch, 5000)` float32 (rank-normalized);
outputs `frame_logits (B,5000,4)`, `beat_logits (B,5000,6)`, `rhythm_logits (B,5000,6)`.
