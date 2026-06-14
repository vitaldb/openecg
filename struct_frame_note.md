# codec_v6 — structural-prior frame breakthrough (2026-06-14)

**The long-standing frame "annotation-noise ceiling" (boundary-F1 0.829, which v5/v6/v7
all failed to beat) was broken by a NEW lever: a data-derived physiological structure
loss on the frame head.** boundary-F1 **0.829 → 0.855 (+0.026)**, median timing 11.6 → 11.1 ms.

## The idea (physiology → data → loss)
Cardiac waves come out in order and never overlap: from LUDB labels the per-sample
transition frequency between different waves (P↔QRS, QRS↔T, P↔T) is **exactly 0** — a
baseline ("other") sample ALWAYS separates two waves, and each wave is one contiguous
run. Over-segmentation (the known frame weakness: spurious short waves, illegal
adjacencies) violates exactly this. Two differentiable terms on the frame softmax:
- `--struct-forbid` — minimize expected mass on `wave_t → DIFFERENT-wave_{t+1}`
  (push a baseline between waves; encodes ordering + refractoriness)
- `--struct-tv` — total-variation contiguity (suppress short spurious waves)

Sweep: forbid≈8, tv≈0.6 optimal, robust over 3 seeds (0.850–0.855). A no-struct full-FT
control reaches only 0.832, so the +0.023 is the PRIOR, not the retrain. In
`pod_stage/kgpu_train.py`. So 0.829 was not annotation noise — over-seg was reducible.

## codec_v6 = 3-stage assembly (struct frame + frozen-head beat/rhythm re-derive)
The struct retrain is full-FT, which perturbs the shared backbone → beat VPC 0.935→0.901
and rhythm 0.805→0.761 regress (DATA artifact: trained on real_ml_v2 only). Fixed with
the codec_v4/v5 frozen-head trick on the new struct backbone:
1. struct backbone+frame → **frame 0.855**
2. freeze backbone, beat-only retrain (+vitaldb VPC) → **VPC 0.901→0.929**
3. freeze backbone, rhythm-only natural-prior re-fit + logit-bias → **rhythm 0.761→0.797**

## codec_v6 vs codec_v5 (held-out)
| metric | codec_v6 | codec_v5 |
|---|---|---|
| frame boundary-F1 (LUDB) | **0.855** | 0.829 |
| frame median timing | **11.1 ms** | 11.6 ms |
| beat VPC (DS2) | 0.929 | 0.935 |
| rhythm (lydus test) | 0.797 | 0.805 |

**Net win: frame +0.026 at a tiny beat/rhythm cost (−0.006 / −0.008, near seed-noise).**
Not fully zero-regression — the frozen-head re-derive recovers ~99%, not 100%. Shipped as
openecg 0.10.0 (codec_v6, default). Residual gap could close with a frame-preserving
struct application. Tooling added: `--struct-forbid/--struct-tv`, `--unified/--unified-loss`,
unified-head ablation (separate finding: 2-head wins, see memory), `scripts/eval_unified_headline.py`.
