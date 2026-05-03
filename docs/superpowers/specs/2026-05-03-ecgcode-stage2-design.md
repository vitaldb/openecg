# ECGCode Stage 2 — Design Spec

**Date**: 2026-05-03
**Scope**: Per-frame supercategory classifier (P/QRS/T/other @ 50Hz) supervised on LUDB cardiologist labels
**Out of scope**: Fine-grained 9-class output, multi-lead joint model, NK pseudo-label training, other datasets, boundary refinement

---

## 1. Goal

LUDB cardiologist label로 supervised 학습한 frame classifier가 **Stage 1 v1.0 baseline (NK direct)을 의미 있게 넘는다**:

| Metric | Stage 1 baseline (NK dwt) | Stage 2 v1.0 target |
|---|---|---|
| P frame F1 | 0.49 | ≥ 0.80 |
| QRS frame F1 | 0.67 | ≥ 0.90 |
| T frame F1 | 0.51 | ≥ 0.75 |
| QRS_on boundary median | 8 ms | ≤ 20 ms |
| **Model > NK direct** | — | 모든 wave class에서 ≥ NK |

마지막 row가 핵심 — 같은 LUDB val에서 model이 NK 자체보다 더 cardiologist에 가까워야 의미.

## 2. Design principles

1. **Supervised on cardiologist labels** (Stage 1 design은 NK pseudo-label 가설 — measurement 결과 NK가 너무 noisy. Stage 2는 직접 GT 학습으로 전환.)
2. **Single-lead model with lead embedding** — 1908 sequences (159 records × 12 leads), lead-id를 embedding으로 주입.
3. **Match Stage 1 alphabet at supercategory granularity** — 4-class (other/P/QRS/T)이 LUDB GT의 자연 해상도.
4. **Stage 1 eval 인프라 재사용** — `ecgcode/eval.py`의 frame F1, boundary error 그대로.
5. **YAGNI** — multi-lead, fine-grained, 다른 데이터셋, fine-tuning은 v2+.

## 3. Module structure

기존 `ecgcode/` 패키지에 신규 모듈 추가 (Stage 1 모듈 재사용):

```
ecgcode/
├── ... (기존 Stage 1 modules)
├── stage2/
│   ├── __init__.py
│   ├── dataset.py             # PyTorch Dataset: LUDB → (signal, lead_id, frame_labels)
│   ├── model.py               # FrameClassifier: Conv + Transformer + Linear
│   ├── train.py               # 학습 loop + checkpoint + early stopping
│   └── infer.py               # checkpoint → per-frame predictions on val/test
scripts/
├── ... (기존)
├── train_stage2.py            # CLI for stage2/train.py
└── validate_stage2.py         # checkpoint → frame F1 + boundary error vs LUDB cardiologist
                                # + model vs NK direct comparison table
tests/
├── ... (기존)
├── test_stage2_dataset.py     # synthetic LUDB-like → expected (signal, label) shapes
├── test_stage2_model.py       # forward pass shape + parameter count
└── test_stage2_train.py       # 1 epoch on tiny synthetic batch → loss decreases
```

`ecgcode/stage2/`로 격리하여 Stage 1과 명확히 분리. 의존성: `stage2 → ecgcode.{vocab, eval, ludb, codec}` (단방향).

## 4. Training data

**Train split**: Stage 1 locked split의 train (`data/splits/ludb_v1.json`) — 159 records.
**Val split**: 동일 locked split의 val — 41 records.
**Sequence count**: 159 × 12 = 1908 train, 41 × 12 = 492 val.

### Per-sequence transformation

```
LUDB record (rid, lead) → (sig_500hz[5000], lead_id[1], frame_labels[500])
                                  ↓
                    sig_250hz[2500] = decimate(sig_500hz, 2)
                                  ↓
                    z-score normalize per sequence: (x - mean) / std
```

**Frame labels** (50Hz, 500 frames per 10s record):
- LUDB cardiologist annotation `(P_on, P_off)`, `(QRS_on, QRS_off)`, `(T_on, T_off)` triplets per beat
- Sample-level array: iso default, mark P/QRS/T regions
- Per-frame: majority vote within 10-sample frame (50Hz at 250Hz input → 5 samples/frame; at 500Hz reference → 10 samples/frame; use cardiologist boundaries at 500Hz before decimation)

Use `ecgcode.eval.gt_to_super_frames` (already exists).

### Lead encoding

`lead_id` ∈ {0..11} mapped to 12 standard leads (`i, ii, iii, avr, avl, avf, v1, v2, v3, v4, v5, v6`). Index from `ecgcode.ludb.LEADS_12` tuple.

### Augmentation (v1.0)

**없음**. v1.1 후순위 (gaussian noise, time shift, amplitude scaling).

## 5. Model architecture

```python
# ecgcode/stage2/model.py
class FrameClassifier(nn.Module):
    def __init__(self, n_leads=12, d_model=64, n_heads=4, n_layers=4, ff=256, n_classes=4):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=15, stride=5, padding=7)   # 250Hz → 50Hz
        self.conv2 = nn.Conv1d(32, d_model, kernel_size=5, stride=1, padding=2)
        self.lead_emb = nn.Embedding(n_leads, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, ff, batch_first=True),
            num_layers=n_layers,
        )
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x, lead_id):
        # x: [B, 2500] @ 250Hz, lead_id: [B]
        h = self.conv1(x.unsqueeze(1)).gelu()                       # [B, 32, 500]
        h = self.conv2(h).gelu()                                     # [B, d_model, 500]
        h = h.transpose(1, 2)                                        # [B, 500, d_model]
        h = h + self.lead_emb(lead_id).unsqueeze(1)                  # broadcast lead embed
        h = self.transformer(h)                                      # [B, 500, d_model]
        return self.head(h)                                          # [B, 500, 4]
```

**Parameter budget**:
- Conv1: 1×32×15 + 32 = 512
- Conv2: 32×64×5 + 64 = 10,304
- Lead emb: 12×64 = 768
- TransformerEncoder × 4 layers (d=64, ff=256, heads=4): ~80K each = 320K
- Head: 64×4 + 4 = 260

**Total ≈ 332K params**.

**No positional encoding for v1.0** — sequence length fixed (500 frames), and Conv stride 5 inherently encodes position. v1.1 candidate: sinusoidal PE if global context attention shows position confusion.

## 6. Loss + optimization

### Loss: weighted cross-entropy

Iso class dominates (~60% of frames). Per-class weights:

```python
# Computed once from train split label distribution
class_weights = 1.0 / np.sqrt(class_freq)         # softer than 1/freq
class_weights = class_weights / class_weights.sum() * n_classes
loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
```

Soft inverse-sqrt prevents over-correction (full inverse-freq punishes model for predicting iso even when correct).

### Optimizer

- AdamW, lr=1e-3, weight_decay=1e-4
- Cosine schedule with warmup (5% of total steps)
- Batch 64
- Mixed precision (`torch.cuda.amp.autocast`)
- Epochs: 50 max, early stopping (patience=10) on val QRS F1

### Checkpoint policy

- Save best model on val QRS F1 (primary metric)
- Save final epoch (for full-training comparison)
- Format: `{model_state, optimizer_state, val_metrics, epoch, config}` in single .pt file at `data/checkpoints/stage2_v1_<timestamp>.pt`

## 7. Evaluation methodology

### A. Frame-level F1 (Stage 1 metric, supercategory)

Same as Stage 1 §7 — per-class precision/recall/F1 on val frames. Use `ecgcode.eval.frame_f1`.

### B. Boundary error (Stage 1 metric)

Decode model softmax → argmax frames → run-length encode → extract boundaries → compare to LUDB cardiologist via `ecgcode.eval.boundary_error` with same Martinez tolerances:

| Boundary | Tolerance |
|---|---|
| P_onset, P_offset | ±50 ms |
| QRS_onset, QRS_offset | ±40 ms |
| T_onset | ±50 ms |
| T_offset | ±100 ms |

### C. Model vs NK direct comparison (CRITICAL)

Same val sequences, same metrics, two pipelines:
- **NK pipeline**: Stage 1 baseline (delineate.run + labeler.label) — already measured (P=0.49, QRS=0.67, T=0.51)
- **Model pipeline**: signal → model → argmax frames → metrics

Report side-by-side table. **Pass criterion**: model F1 ≥ NK F1 for all 3 wave classes.

### D. Per-lead breakdown

Per-class F1 grouped by lead_id (12 buckets) — identifies if model struggles on specific leads (e.g., V1 P-wave often biphasic).

### Acceptance criteria

| Metric | Target | Stretch |
|---|---|---|
| P frame F1 | ≥ 0.80 | ≥ 0.90 |
| QRS frame F1 | ≥ 0.90 | ≥ 0.95 |
| T frame F1 | ≥ 0.75 | ≥ 0.85 |
| QRS_on boundary median | ≤ 20 ms | ≤ 10 ms |
| Model F1 ≥ NK F1 | all 3 wave classes | margin ≥ 0.10 each |
| Train time on RTX 4090 | ≤ 30 min | ≤ 15 min |

미달 시 진단:
- All metrics flat / below NK → loss formulation 또는 lr 문제
- High train F1, low val F1 → overfitting → augmentation / dropout 추가 (v1.1)
- Specific class미달 → 해당 class weight 상향 또는 architecture 조정

## 8. Testing strategy

### Unit tests (no GPU required, synthetic data)

| Test file | Coverage |
|---|---|
| `test_stage2_dataset.py` | Synthetic LUDB-like input → correct (signal, lead_id, label) shapes; label distribution roughly matches expected |
| `test_stage2_model.py` | Forward pass output shape `[B, 500, 4]`; param count ~330K; CPU + GPU 둘 다 작동 |
| `test_stage2_train.py` | 1 epoch on tiny synthetic batch (e.g., 4 sequences × 2500 samples) → loss strictly decreases; checkpoint save/load round-trip |

### Integration test

`test_integration.py`에 추가: 학습된 small model (1 epoch on 5 records) → val 1 record inference → output shape + class probability sanity (sum to 1).

CI에서는 unit + small integration만. Full training은 manual.

## 9. Risks + mitigations

| Risk | Mitigation |
|---|---|
| 1908 sequences가 transformer overfit | val F1 monitoring, early stopping, dropout=0.1 (in TransformerEncoderLayer default) |
| Lead embedding이 학습 안 됨 (lead 0/1만 학습됨) | per-lead F1 breakdown으로 검증, mismatch 발견 시 lead-stratified batch sampler |
| Class imbalance (iso ~60%) | weighted CE, val에서는 unweighted F1 (skewed metric 방어) |
| Boundary error가 frame quantization (20ms)으로 한계 | model output frame rate 50Hz → boundary 정확도 ±10ms 이론적 한계, 실제로는 더 큼; v2.x boundary refinement head 고려 |
| LUDB cardiologist label도 perfect 아님 | 실제 ceiling은 ~0.95 (inter-rater agreement 추정); 0.90 target 달성 시 사실상 saturate |

## 10. Open questions / 후속

- v2.x: 9-class fine-grained output (NK pseudo-label로 q/r/s/w/* 학습)
- v2.x: Multi-lead joint model (12-channel input, 159 examples)
- v2.x: Augmentation suite (gaussian noise, time shift)
- v2.x: Boundary refinement head (regression for sub-frame timing)
- v3: Other datasets (QTDB validation, MIT-BIH/SNUH 학습 추가)
- v3: 모델을 다른 데이터셋에 적용해 pseudo-label 만들고 self-training
