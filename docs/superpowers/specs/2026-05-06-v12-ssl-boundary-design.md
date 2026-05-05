# Stage 2 v12 — SSL transfer + boundary engineering

**Date**: 2026-05-06
**Baseline**: `v9_q1c_pu_merge` (ViT d=128 L=8, q1c+pu0 merged labels, no lead_emb).
LUDB edge-filtered avg Martinez F1 **0.923**, ISP test **0.943**, QTDB pu0_random **0.779**.
**Scope**: 7 new training runs that probe two orthogonal axes — boundary-aware loss/regression on the existing ViT, and transfer from open-weight ECG SSL backbones (HuBERT-ECG, ST-MEM).
**Out of scope**: 12-channel joint input (forbidden — Stage 2 stays single-lead), Stage 1 / Stage 3 changes, alphabet expansion, downstream beat/rhythm classification, paper writing.

---

## 1. Goal

Lift v9_q1c_pu_merge avg Martinez F1 by **≥ +0.02** on LUDB and **≥ +0.05** on QTDB pu0, without regressing ISP. Produce a single comparison table that distinguishes (a) what the existing architecture is missing in boundary precision and (b) whether large-corpus SSL pretraining breaks the LUDB 200-record data ceiling identified by the v4 capacity scan.

## 2. Design principles

1. **Single-lead at inference, always.** Stage 2 must remain robust across any of the 12 standard leads with a 1-channel input. SSL backbones that are natively 12-channel are adapted (replicate-then-fallback-reinit), never used in joint 12-channel mode.
2. **All 7 runs evaluated identically to v9.** LUDB val (edge-filtered) + ISP test + QTDB pu0_random with `seed=42`, `windows_per_record=5`. Per-class boundary F1 + avg Martinez F1.
3. **Diff-from-v9 minimal where possible.** A and B reuse v9 dataset, model, training config, and eval; only the labelled-target / loss / head differ.
4. **YAGNI on SSL.** No HuBERT-style iterative re-clustering. No multi-stage pretrain. We adopt published weights, adapt input/head, and fine-tune.
5. **Comparison table is the deliverable.** Each run writes `out/v12_<name>_<ts>.json` in the v9 schema; `scripts/compare_v12.py` aggregates.

## 3. Experiment matrix

| # | name | backbone | boundary tweak | training mode |
|---|---|---|---|---|
| 0 | v9_q1c_pu_merge (ref) | ViT d=128 L=8 | — | already trained |
| 1 | v12_soft | ViT d=128 L=8 | A: soft labels at transitions | scratch (= v9 config + KL) |
| 2 | v12_reg | ViT d=128 L=8 | B: 6-channel regression head | scratch (= v9 config + L1 head) |
| 3 | v12_hubert_lp | HuBERT-ECG | — | linear probe (encoder frozen) |
| 4 | v12_hubert_ft | HuBERT-ECG | — | full finetune (encoder LR 1e-5) |
| 5 | v12_stmem_lp | ST-MEM | — | linear probe |
| 6 | v12_stmem_ft | ST-MEM | — | full finetune |
| 7 | v12_best | best of {3..6} | best of {A, B, A+B} | full finetune |

Run 7 is decided after runs 1–6 land; the spec commits to running it but not to the combination.

## 4. Boundary engineering on v9 backbone

### 4.1 v12_soft (Approach A: soft labels)

**Motivation.** The current per-frame CE punishes any disagreement at the boundary frame, but Martinez tolerance is ±40–100 ms, i.e., 2–5 frames. The signal at frame `t` straddling a wave on/off is genuinely ambiguous.

**Label transformation.**
- For each transition in the GT label sequence (`labels[i] != labels[i+1]`), the two adjacent frames receive a 70/30 split of probability mass between the two classes:
  ```
  soft[i  ] = 0.7 · onehot(labels[i  ]) + 0.3 · onehot(labels[i+1])
  soft[i+1] = 0.3 · onehot(labels[i  ]) + 0.7 · onehot(labels[i+1])
  ```
- Non-boundary frames remain hard one-hot.
- `IGNORE_INDEX` frames produce an all-zero target row and are excluded from the loss.

**Loss.** KL divergence with reduction='batchmean' over non-ignore rows. Class weights from `compute_class_weights` are folded into the target row before normalization.

**No other change.** Same dataset (`LUDBFrameDataset` + ISP + QTDB merge), same model (`FrameClassifierViT` v9 KWARGS), same `TrainConfig`, same SEED, same eval.

### 4.2 v12_reg (Approach B: boundary regression head)

**Motivation.** Frame-level argmax is bounded by the 20 ms frame grid; LUDB median |err| is already at this floor (8 ms QRS, 12–16 ms P/T). Direct sub-frame regression at the boundary frame can take this down without changing inference latency.

**Architecture.** New class `FrameClassifierViTReg` — same backbone as v9 plus a parallel regression head:
```
backbone(x) → h ∈ [B, 500, 128]
class_head : Linear(128, 4)   → CE / KL on supercategories
reg_head   : Linear(128, 6)   → signed offset in samples for
                                {p_on, p_off, qrs_on, qrs_off, t_on, t_off}
```

**Regression target.** For each frame `f` and each boundary class `k`, the target is the signed sample-offset to the nearest GT boundary of class `k` *only if* such a boundary lies within ±5 frames of `f`; otherwise the target is masked.

**Loss.** `total = CE_class + λ · L1(reg, target, mask)` with λ swept ∈ {0.05, 0.1, 0.5}; final spec value chosen by LUDB val avg F1. `λ=0.1` is the prior.

**Inference.** Apply `extract_boundaries` on the post-processed frame argmax as today. For each predicted boundary at sample `s = f * spf`, look up `reg[f, k]` and emit `s_refined = s + reg[f, k]` (clamped to window). This does NOT use the existing rule-based `ecgcode/stage2/refiner.py`, which the eval_refiner experiment showed degrades QRS by –0.15 F1.

### 4.3 Combination (deferred to run 7)

A and B are loss-only / head-only deltas and combine cleanly (KL on `class_head` + L1 on `reg_head`). The combination is **not** evaluated standalone on v9 — it is the candidate boundary tweak for run 7 if both A and B individually beat v9 on LUDB val avg F1 by ≥ +0.005. This keeps the new-run count at 7.

## 5. SSL backbone transfer

### 5.1 Common adaptation framework

Each SSL backbone exposes (after our adapter) a function `encode(sig: [B, 2500] float32, lead_id: [B] long) -> h: [B, 500, d]` so the existing 4-class classification head and (for run 7) regression head plug in unchanged.

The adapter handles the per-model differences in (a) sampling rate, (b) window length, (c) channel count, (d) native temporal resolution.

### 5.2 HuBERT-ECG (`Edoardo-BS/hubert-ecg` on HuggingFace)

| native | ours | adapter |
|---|---|---|
| 100 Hz input | 250 Hz | `scipy.signal.decimate(sig, 2, zero_phase=True)` then resample 125→100 via `scipy.signal.resample`, single channel |
| 5 s window (500 samples) | 10 s (2500) | split 10 s into 2 × 5 s; run encoder twice; concat per-frame outputs along time |
| 50 Hz post-conv frames | 50 Hz | direct 1:1 mapping (250 + 250 → 500 frames) |
| 12-lead or 1-lead variants | 1-lead | use the published 1-lead variant if available, else pass single channel into the 1-channel input slot |

**Linear probe (v12_hubert_lp).** Freeze all encoder weights. Train only the 4-class linear head on top of the encoder's final-layer hidden state. AdamW LR=1e-3, weight decay 0.01, 20 epochs, early stop patience 5.

**Full finetune (v12_hubert_ft).** Two parameter groups: encoder (LR=1e-5), head (LR=1e-3). Linear warmup over 1 epoch then cosine decay. 30 epochs, early stop patience 7.

**Boundary at the 5 s split.** The two 5 s chunks are encoded independently; receptive field at the seam is one-sided. Inference accepts this; training masks the 5 frames either side of the seam in the loss when this measurably hurts validation (decided after first hubert_ft run).

### 5.3 ST-MEM (`bakqui/ST-MEM` on GitHub)

| native | ours | adapter |
|---|---|---|
| 250 Hz input | 250 Hz | direct |
| 5 s × 12 channels | 10 s × 1 channel | split into 2 × 5 s; replicate the single lead across all 12 input channels (default); concat outputs |
| spatiotemporal patches (lead × time) | 50 Hz / 4-class | linearly project the time-pooled patch outputs onto a 250-frame-per-window grid then upsample to 500 frames; final `Linear(d, 4)` head |

**Single-lead default**: replicate. The model's lead-axis attention becomes degenerate (every channel identical) but the time-axis attention and patch CNN still operate on the real signal — this is the cheapest adapter.

**Single-lead fallback**: if v12_stmem_ft underperforms v12_hubert_ft by ≥ 0.02 F1, reinitialize the input projection / patch-embed weights to take 1 channel, reload all transformer weights, and retrain. Documented as v12_stmem_ft_1ch if executed.

**LP / FT protocol**: identical to HuBERT (same LRs, schedules, early-stop).

### 5.4 v12_best (run 7)

After runs 1–6 are evaluated:
1. **Boundary tweak** = winner on LUDB val avg F1 among {v12_soft (A), v12_reg (B)} vs v9. If both beat v9 by ≥ +0.005, use A+B (combined cls + reg head). If neither beats v9, run 7 still executes with the tied/closer of A/B for completeness.
2. **Backbone** = best LUDB val avg F1 among the **full-finetune** SSL runs only ({v12_hubert_ft, v12_stmem_ft}); LP runs are diagnostic and not eligible. Tie-break by QTDB pu0 F1.
3. Train one full-finetune that applies the chosen tweak on top of the chosen backbone, using the same LP/FT protocol (encoder LR 1e-5, head LR 1e-3, warmup 1ep, cosine decay).

## 6. Training data and eval (unchanged from v9)

**Train**: `LUDBFrameDataset(train, mask_unlabeled_edges=True, edge_margin_ms=100)`
∪ `CombinedFrameDataset(["isp_train"])`
∪ `QTDBSlidingDataset(scale_factors=(1.0,), windows_per_record=20, q1c_pu_merge=True)`.

`SEED=42`. `EDGE_MARGIN_MS=100`. Batch size 64. AdamW. Class weights from
`compute_class_weights` on the merged label histogram.

**Eval**: `eval_all` from v9_q1c_pu_merge — LUDB val (edge_filtered), ISP test (all 12 leads concatenated cumulative), QTDB pu0_random (5 random windows / record, seed 42). Average Martinez F1 over the 6 boundary classes. Same `signed_boundary_metrics` (`tolerance_ms` from `MARTINEZ_TOLERANCE_MS`).

## 7. Code structure

```
ecgcode/stage2/
  ssl/
    __init__.py
    hubert.py        # Edoardo-BS/hubert-ecg loader + 250→100Hz adapter + 5s×2 forward
    stmem.py         # bakqui/ST-MEM loader + 12-ch replicate adapter + 5s×2 forward
    head.py          # FrameHead(d_model, n_classes=4); FrameRegHead(d_model, n_reg=6)
  model.py           # + FrameClassifierViTReg (parallel cls + reg head)
  dataset.py         # + soft_boundary_labels(labels, alpha=0.7) helper used by training scripts
  train.py           # + soft-target KL path; + regression L1 path; + LP/FT param groups
scripts/
  train_v12_soft.py   # 1 run, calls fit() with KL loss
  train_v12_reg.py    # sweeps λ ∈ {0.05, 0.1, 0.5} → reports best
  train_v12_hubert.py # CLI flag --mode {lp, ft}
  train_v12_stmem.py  # CLI flag --mode {lp, ft}
  train_v12_best.py   # depends on outputs of 1..6 to pick combination
  compare_v12.py      # aggregate out/v12_*.json into one Markdown + JSON comparison table
```

`docs/superpowers/plans/` will hold the implementation plan written by `writing-plans` after this spec is approved.

## 8. Risks and mitigations

1. **SSL weight licenses.** HuBERT-ECG and ST-MEM are research releases on GitHub / HuggingFace. Confirm the LICENSE on each at download time; if either prohibits the use, drop that backbone and document.
2. **HuBERT 5 s seam bleeding.** Two independent 5 s forwards lose attention context across the seam. Quantify the seam-frame error in v12_hubert_ft eval; mitigate via overlap-add (run twice with 2.5 s offset, average) only if measurable.
3. **ST-MEM single-lead degeneracy.** Replicate-12 may waste pretrained signal because lead attention is uninformative. Fallback: reinit 1-channel input projection (`v12_stmem_ft_1ch`).
4. **Regression head scaling.** λ at 0.1 may be off by 5×. Sweep at one model only (v9 backbone) on LUDB val; carry the winning λ to v12_best.
5. **HuggingFace / pip dependencies.** Adding `transformers`, `huggingface-hub` and ST-MEM's pyproject. Pinned versions go into `pyproject.toml`. CPU-only loading must work for inference scripts.
6. **GPU budget.** 7 runs × 30 epochs × current per-epoch cost. v9 took ≈14 min; SSL FT runs are 2–3× larger model so ≈30–45 min each. Total ≈4 h GPU. Acceptable.

## 9. Success criteria

The spec is judged a success if **at least one of v12_hubert_ft / v12_stmem_ft / v12_best** clears:

| Dataset | v9 baseline | target |
|---|---|---|
| LUDB val edge_filtered | 0.923 | ≥ 0.943 |
| ISP test | 0.943 | ≥ 0.943 (no regression) |
| QTDB pu0_random | 0.779 | ≥ 0.829 |

If none of the runs hits the LUDB target, the failure modes (which boundary classes regressed; whether SSL features are simply orthogonal to delineation needs) are documented in `out/v12_postmortem.md` so the next iteration is informed rather than blind.

## 10. Non-goals (write down so they stay non-goals)

- Pretraining on SNUH 78만 from scratch. (Defer to a v13+ if v12 SSL fails.)
- 12-channel joint input. (Forbidden by Stage 2 charter.)
- Replacing the rule-based `ecgcode/stage2/refiner.py`. (Boundary regression head supersedes it for predicted boundaries; `refiner.py` stays as opt-in for ablation.)
- Alphabet expansion (P_tall, T_inv, etc.) — Stage 2 v12 stays at 4-class supercategory.
- Re-running v9. The published v9 numbers (LUDB 0.923 / ISP 0.943 / QTDB 0.779) are the comparison anchor.

---

*Owner*: vital@snu.ac.kr
*Last updated*: 2026-05-06
