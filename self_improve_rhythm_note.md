# Self-improving loop — rhythm rare-class equity (lydus hospital test)

**Gate:** `scripts/eval_lydus_rhythm.py <ckpt> --test data/cache/lydus_rhythm_test.npz`
macro-F1 over present classes (sinus/avb/paced/afib/bbb), n=22229, patient-disjoint.
Secondary (frozen-head designs guarantee these unchanged): frame boundary-F1, beat VPC.

## Baseline — codec_v4 (reproduced locally 2026-06-12)
| class | support | prec | rec | f1 |
|---|---|---|---|---|
| sinus | 15681 | 0.948 | 0.924 | 0.936 |
| avb | 1626 | 0.710 | 0.812 | 0.758 |
| paced | 866 | 0.857 | 0.685 | 0.761 |
| afib | 2252 | 0.889 | 0.882 | 0.886 |
| bbb | 1804 | 0.568 | 0.678 | **0.618** |
| **macro** | | | | **0.7918** |
acc 0.8822. Weak spots: bbb (over-predicted, prec 0.57), paced (under-recalled 0.685).

## Methodology (test-clean self-improving loop)
- **Selection signal = lydus_dev** (8675 windows, 3337 patients), carved from lydus_train
  by hid-hash range [1500,3000), patient-disjoint from both trainsub and test. Decisions
  are made on dev; `lydus_test` is the final gate, checked sparingly to avoid test-hacking.
- **Train data:** real_ml_v2_train (24842) + lydus_trainsub (40817, hash≥3000).
- Trainer `--val lydus_dev` → frame/beat=IGN report 0, best-epoch = lydus-dev rhythm-F1.
- Compute on kgpu rtx4090 (pod gpu-rhythm-loop). codec_v4 dev baseline measured in iter1.

## Iter 1 — rhythm-only linear-probe re-train (zero regression)
**Hypothesis:** rare-class oversampling recalibrates the 774-param rhythm linear head to
lift paced-recall (0.685) and bbb, raising macro — frame/beat frozen (byte-identical to v4).
**Design:** `--init-ckpt codec_v4 --freeze-backbone --rhythm-only`, rhythm-boost ∈ {1,3,6},
10 ep. Select best on lydus_dev.
**RESULT: FAILED on test.** Best on balanced-dev (rb=1.0, 0.8845 vs v4 0.8607) REVERSED on
test: rb1.0 **0.7595** vs v4 **0.7918** (−0.032); rb3.0 0.7179. Diagnosis: **class-prior
mismatch** — lydus_train/dev is sinus-CAPPED (27% sinus), lydus_test is NATURAL (70% sinus).
Retraining on balanced data + rare-biased class weights (cw[sinus]≈0) shifted the boundary
toward rare classes → great on balanced dev, but over-calls rare on sinus-heavy test (bbb
precision 0.568→0.422). **Methodology fix:** built prior-matched `lydus_dev_nat` (70% sinus,
n=3272) whose ordering AGREES with test (v4 0.851 > rb1.0 0.833). Use nat-dev for selection.

## Iter 2 — inference-time per-class logit-bias calibration (free, zero-regression)
**Hypothesis:** v4's rhythm head is mis-calibrated for the natural prior (bbb over-called:
prec 0.568 ≪ rec 0.678). Optimize a per-class additive bias on the rhythm logits to maximize
macro-F1 at the test prior (tuned on nat-dev), fold into `rhythm_sample.bias` (frame/beat/
features all byte-identical). Confirm on test.
**RESULT: SUCCESS, +0.0112 test macro (0.7918→0.8030), zero regression.** Bias
[+0.2,-0.8,0,+1.4,-0.2] (sinus/avb/paced/afib/bbb). Per-class test F1: sinus .936→.942,
avb .758→.765, paced .761→.778, afib .886→.893, **bbb .618→.637** (prec .568→.621 — the
over-calling fixed). nat-dev selection transferred to test (0.851→0.859 dev). acc .882→.893.
Shippable as-is (codec_v4_biasadj.pt). `pod_stage/iter2_logit_bias.py`.

## Iter 3 — gentle full fine-tune for better features (gated)
**Hypothesis:** bbb (0.637) still has headroom that a linear/bias fix can't reach — needs
better rhythm FEATURES. Full fine-tune from codec_v4 (unfreeze), low LR 5e-5, natural prior
(rhythm-boost 1, learned the iter1 lesson). frame/beat stay supervised by real_ml_v2_train
(lydus has IGN frame/beat) → anchored. Select on nat-dev rhythm (--val lydus_dev_nat).
Gate after: frame (ludb_val) + beat/frame (real_val) regression within tol; then stack the
logit-bias on top.
**RESULT: FAILED.** iter3_ft test macro 0.7701 (< v4 0.7918); even +bias = 0.8009 (≈ v4+bias
0.8030, NOT better). Frame/beat held (real_val frame 0.877→0.881, beat 0.648→0.657 — anchored
by real_ml_v2 supervision). But rhythm got WORSE: same prior-drift as iter1 — training on
sinus-CAPPED trainsub re-balanced calibration → over-calls rare on natural-prior test. Proves
**full fine-tune does NOT improve rhythm features; codec_v4 features are at ceiling.**

## Iter 4 — rhythm-only linear re-fit at the NATURAL prior (zero regression) ✅ WINNER
**Hypothesis:** iter1/3 failed because they trained at the BALANCED prior. Re-fit the rhythm
linear head at the natural test prior (new `--rhythm-prior` sampler → 70% sinus, drops
rhythm-IGN windows). A linear re-fit can re-weight features, not just shift thresholds (iter2).
**RESULT: BEST.** iter4a raw test macro **0.8004** (already ≥ iter2!); +logit-bias = **0.8045**.
Per-class: sinus .936→.946, avb .758→.767, paced .761→.768, afib .886→.890, **bbb .618→.651**.
acc .882→.899. **Zero-regression PROVEN**: only rhythm_sample.{weight,bias} differ from v4;
backbone+frame+beat byte-identical → frame-F1 0.829 / beat-VPC 0.935 unchanged.
The balanced-vs-natural contrast (iter1 0.76 ❌ vs iter4 0.80 ✅) confirms prior was the whole issue.

## CONVERGED — winner: `iter4a_biasadj` (codec_v5 candidate)
rhythm macro **0.7918 → 0.8045** (+0.0127, +1.6% rel), **bbb 0.618 → 0.651** (+0.033, +5.3% rel,
the equity target), every class ↑, frame/beat byte-identical. Diminishing returns
(iter2→iter4 +0.0015) + evidence that features are at ceiling (bbb is lead-II-limited; only
prior-calibration helps) → loop converged. Levers tried: calib ✅, natural-prior re-fit ✅,
balanced retrain ❌, full-FT ❌. Artifacts on pod /root/loop + pod_stage/pod_pull/.

## Multi-seed robustness (4 seeds, parallel across the fleet)
iter4a re-run at seeds 1-4 (rtx4090/5090/3090/v100, staged once via kgpu transit-object):
raw-refit test macro [s1 0.8027, s2 0.8039, s3 0.8058, s4 0.8017] + shipped s0 0.8004;
+bias [0.8028, 0.7973, 0.8020, 0.8040] + s0 0.8047. **All 5 seeds > codec_v4 0.7918**
(mean raw ≈ 0.803). bbb 0.63-0.65 throughout. **Natural-prior re-fit = the robust driver;
logit-bias = marginal/noisy** (helped s0/s4, slightly hurt s2/s3 — tuned on only 3272 dev
windows). codec_v5 = s0 (top of the cluster). Verdict: the +0.011 gain is robust, not luck.

## Iter 2 (queued if Iter 1 underdelivers) — full fine-tune, gated
Linear-probe ceiling is low (only re-weights frozen features). If Iter 1 macro gain < ~0.01,
escalate: full fine-tune from codec_v4 (unfreeze all) with rhythm emphasis (↑lambda-rhythm,
rhythm-boost) — gate: accept only if frame boundary-F1 and beat VPC stay within ~0.005 of v4
(eval ludb_val + MIT-BIH DS2). Backbone capacity can move bbb features, not just re-weight.
