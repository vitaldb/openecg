# ECGCode Stage 1 — Design Spec

**Date**: 2026-05-03
**Scope**: Tokenization pipeline (signal → RLE token stream) + LUDB validation
**Out of scope**: Frame classifier neural model (Stage 2), v1.1 detectors (`u`/`d`/`j`), other datasets

---

## 1. Goal

ECG signal을 임상학적으로 해석 가능한 discrete token stream으로 변환하는 **non-neural pipeline 완성**, 그리고 **NeuroKit2(NK) pseudo-label vs LUDB cardiologist label baseline 측정**.

**Stage 1 v1.0 완료 기준**:
1. `python scripts/tokenize_ludb.py` → 200 records × 12 leads = 2,400 sequences 토큰화 완료, `ludb_tokens.npz` 저장
2. `python scripts/validate_v1.py` → per-symbol F1 + boundary error report 출력
3. ASCII art로 LUDB 10개 record 시각 검토 → 명백한 오류 없음

이 baseline이 Stage 2 frame classifier가 넘어야 할 기준선.

## 2. Design principles

1. **Morphology-pure alphabet**: token은 wave가 무엇·언제만 인코딩. 진폭/극성 같은 derivable 정보는 alphabet에서 제외 (downstream에서 raw signal 또는 별도 amplitude 채널로 처리).
2. **Append-only vocab**: ID는 한 번 할당되면 재사용 안 됨. 신규 symbol은 항상 끝에 추가.
3. **NK as labeler, LUDB as validator**: 학습 라벨은 NK pseudo-label, gold standard는 LUDB cardiologist label. train/val 분리.
4. **Per-lead independence**: 각 lead가 독립적으로 토큰화 (12 leads → 12 sequences). Lead context는 Stage 2 모델이 학습.
5. **YAGNI**: Stage 1은 pipeline + validation까지. 모델, 다른 데이터셋, boundary refinement는 후속 단계.

## 3. Module structure

```
ecgcode/
├── pyproject.toml                # uv-managed
├── README.md
├── .gitignore
├── docs/
│   └── superpowers/specs/
│       └── 2026-05-03-ecgcode-stage1-design.md   # 이 spec
├── ecgcode/
│   ├── __init__.py
│   ├── vocab.py                  # 13 IDs + char↔name↔ID 매핑 + 활성 클래스
│   ├── codec.py                  # uint16 pack/unpack + ASCII art + frame expansion
│   ├── ludb.py                   # zip extract + WFDB read + 160/40 stratified split
│   ├── delineate.py              # NK ecg_delineate(method="dwt") wrapper
│   ├── pacer.py                  # high-pass + threshold spike detector
│   ├── labeler.py                # delineation + spikes → RLE token stream
│   └── eval.py                   # NK vs cardiologist: per-symbol F1, boundary error
├── scripts/
│   ├── tokenize_ludb.py          # 전체 LUDB 토큰화 → .tokens.npz
│   └── validate_v1.py            # eval 실행 + JSON report
├── tests/
│   ├── conftest.py
│   ├── test_vocab.py
│   ├── test_codec.py
│   ├── test_pacer.py
│   ├── test_labeler.py
│   └── test_integration.py
└── data/                         # gitignored, LUDB extract / token outputs
```

각 모듈 단일 책임, 200줄 미만 목표. 의존성 단방향: `delineate`/`pacer`/`vocab` → `labeler` → `codec`.

## 4. Alphabet v1 (locked)

| ID | char | name | v1.0 active? | trigger / 출처 |
|---|---|---|---|---|
| 0 | `·` | `<pad>` | mask only | sequence padding (loss 제외) |
| 1 | `?` | `<unk>` | ✅ | NK 완전 실패 / artifact / VFib chaos |
| 2 | `_` | iso | ✅ | wave 사이 gap |
| 3 | `*` | pacer_spike | ✅ | high-pass + threshold detector |
| 4 | `p` | P | ✅ | NK ECG_P_Onsets/Offsets |
| 5 | `q` | Q | ✅ | NK ECG_Q_Peaks |
| 6 | `r` | R | ✅ | NK ECG_R_Peaks |
| 7 | `s` | S | ✅ | NK ECG_S_Peaks |
| 8 | `t` | T | ✅ | NK ECG_T_Onsets/Offsets |
| 9 | `u` | U | 🚧 v1.1 | rule (T_offset+200ms region) |
| 10 | `w` | wide_QRS | ✅ | labeler fallback (no Q/S + duration>120ms) |
| 11 | `d` | delta | 🚧 v1.1 | rule (PR<120ms + slope) |
| 12 | `j` | J wave | 🚧 v1.1 | rule (J_point notch) |

**v1.0 학습 대상 9 classes**: `?, _, *, p, q, r, s, t, w`. 미래 CTC blank 또는 v2 symbol 필요 시 ID 13부터 append.

**구조적 정합성**:
- 0–3: meta + non-wave events
- 4–10: wave letters lowercase, atrial → ventricular 순 (p → q,r,s → t → u → w)
- 11–12: extensions (delta, J — v1.1 detector 구현 시 활성화)
- 13 IDs ≤ 16 → 4 bits에 fit (저장 효율)

## 5. Token format & codec

### Per-event packing

```
Each event = uint16 little-endian
  bits [15..8] = symbol_id  (0..255, v1 사용은 0..12)
  bits [7..0]  = length     (1..255, 단위 4ms → 4..1020 ms)
```

**불변량**:
- `length == 0` 금지
- ms → 4ms grid snap: `round(ms/4)`, 최소 1
- `symbol_id == 0` (pad)는 codec에 등장 안 함

### Long-event splitting (length > 1020 ms)

```python
def split_long(sym: int, length_ms: int) -> list[tuple[int, int]]:
    chunks = []
    while length_ms > 1020:
        chunks.append((sym, 1020))
        length_ms -= 1020
    if length_ms > 0:
        chunks.append((sym, length_ms))
    return chunks
```

LUDB 10초 record는 split 거의 발생 안 함. Holter 적용 시 중요.

### Frame rate (학습 라벨용)

**50 Hz / 20 ms per frame** (Option B, Q/S coverage 보장).

| Wave | ms | @ 20ms/frame |
|---|---|---|
| P (80–120) | 4–6 frames |
| Q (10–30) | 1–2 frames |
| R (30–60) | 2–3 frames |
| S (20–40) | 1–2 frames |
| T (160–200) | 8–10 frames |
| Pacer spike (1–3) | priority override (1 frame fat label) |

10s LUDB record → 500 frames per-frame label array.

### Per-frame expansion 규칙

1. Spike (`*`) 우선: frame과 1ms라도 겹치면 그 frame label = `*`
2. 그 외 wave: max-overlap symbol (frame과 가장 길게 겹치는 symbol)
3. 모두 없으면 iso

### Storage layout

```
ludb_tokens.npz
├── meta              # JSON: vocab_version, ms_unit, fs, ...
├── record_ids        # uint16[N]
├── lead_names        # str[12]
├── 0001_i            # uint16[K]
├── 0001_ii
├── ...
└── 0200_v6
```

### Three views

```python
events = [(2,128), (4,92), (2,44), (5,16), (6,40), (7,24), (2,84), (8,164), (2,208)]
# = [(iso,128ms), (P,92ms), (iso,44ms), (Q,16ms), (R,40ms), (S,24ms), (iso,84ms), (T,164ms), (iso,208ms)]

ID array      : array([2, 4, 2, 5, 6, 7, 2, 8, 2], dtype=uint8)
Compact ASCII : "_p_qrs_t_"
Timed ASCII   : "______ppppp__qrrs____tttttttt__________"  # 1 char per 20ms
JSON verbose  : [{"sym":"iso","ms":128}, {"sym":"P","ms":92}, ...]
```

### Codec 인터페이스

```python
# ecgcode/codec.py
def encode(events: list[tuple[int, int]]) -> np.ndarray  # uint16
def decode(packed: np.ndarray) -> list[tuple[int, int]]
def to_frames(events, frame_ms=20, total_ms=None) -> np.ndarray  # uint8 per-frame
def from_frames(frames: np.ndarray, frame_ms=20) -> list[tuple[int, int]]
def render_compact(events) -> str
def render_timed(events, ms_per_char=20) -> str
def render_json(events, vocab) -> list[dict]
```

## 6. Pipeline architecture

### Data flow

```
ludb.zip → ludb.load_record(id) → {lead_name: signal[5000]}
                                          ↓ per lead
    ┌─────────────────────────────────────┴─────────────────────────────────┐
    ↓                                                                       ↓
delineate.run(signal, fs=500)                              pacer.detect_spikes(signal, fs=500)
  → DelineateResult                                        → list[spike_sample_idx]
    p_onsets, p_peaks, p_offsets
    q_peaks
    r_onsets, r_peaks, r_offsets
    s_peaks
    t_onsets, t_peaks, t_offsets
    ↓                                                                       ↓
    └─────────────────────────────────────┬─────────────────────────────────┘
                                          ↓
                          labeler.label(signal, dr, spikes, fs=500)
                              → list[(symbol_id, length_ms)]
                                          ↓
                          codec.encode(events) → np.ndarray[uint16]
                                          ↓
                              save → ludb_tokens.npz
```

### Labeler core 알고리즘

Sample-level array를 채우고 마지막에 RLE로 압축:

```python
def label(signal, dr, spike_idx, fs=500) -> list[tuple[int, int]]:
    n = len(signal)
    
    # NK total failure 처리
    if len(dr.r_peaks) == 0:
        return [(VOCAB.ID_UNK, int(n * 1000 / fs))]
    
    labels = np.full(n, VOCAB.ID_ISO, dtype=np.uint8)
    
    # 1. P waves
    for onset, offset in zip(dr.p_onsets, dr.p_offsets):
        labels[onset:offset+1] = VOCAB.ID_P
    
    # 2. T waves (polarity 무관)
    for onset, offset in zip(dr.t_onsets, dr.t_offsets):
        labels[onset:offset+1] = VOCAB.ID_T
    
    # 3. QRS — q/r/s/w 분해
    for i in range(len(dr.r_onsets)):
        on, off = dr.r_onsets[i], dr.r_offsets[i]
        q = dr.q_peaks[i] if i < len(dr.q_peaks) else None
        r = dr.r_peaks[i]
        s = dr.s_peaks[i] if i < len(dr.s_peaks) else None
        
        has_q = q is not None and not np.isnan(q)
        has_s = s is not None and not np.isnan(s)
        qrs_ms = (off - on) * 1000 / fs
        
        # Wide QRS fallback
        if not has_q and not has_s and qrs_ms > 120:
            labels[on:off+1] = VOCAB.ID_W
            continue
        
        q_end = (int(q) + r) // 2 if has_q else on
        s_start = (r + int(s)) // 2 if has_s else off + 1
        
        if has_q:
            labels[on:q_end] = VOCAB.ID_Q
        labels[q_end:s_start] = VOCAB.ID_R
        if has_s:
            labels[s_start:off+1] = VOCAB.ID_S
    
    # 4. Pacer spike (highest priority)
    for idx in spike_idx:
        labels[idx] = VOCAB.ID_PACER
    
    # 5. RLE compress
    return rle_compress(labels, ms_per_sample=1000/fs)
```

**Priority**: spike > wave > iso. P/T 먼저 채우고 QRS 후속.

### Edge cases

| 상황 | 처리 |
|---|---|
| NK가 R peak 0개 검출 | 전체 1 event `?` |
| NK가 일부 wave만 검출 | 검출된 것만 라벨, 나머지 iso |
| Wave 경계가 signal 넘어감 | clamp to [0, n) |
| Spike가 wave 내부 | spike sample이 wave를 잠시 split (RLE에서 wave-`*`-wave) |
| QRS_offset > 다음 P_onset (NK 오류) | 후순위 wave가 이김 |
| 첫/마지막 wave 외부 | iso 자동 |

## 7. Data split & validation methodology

### LUDB train/val split

**원칙**: record-level (lead-level 금지 — leakage), rhythm 기준 stratified.

| Rhythm | Total | Train (80%) | Val (20%) |
|---|---|---|---|
| Sinus rhythm | 143 | 114 | 29 |
| Sinus tachycardia | 4 | 3 | 1 |
| Sinus bradycardia | 25 | 20 | 5 |
| Sinus arrhythmia | 8 | 6 | 2 |
| Irregular sinus | 2 | 2 | 0 |
| Atrial fibrillation | 15 | 12 | 3 |
| Atrial flutter | 3 | 3 | 0 |
| **Total** | **200** | **160** | **40** |

`seed=42` 고정. `data/splits/ludb_v1.json`에 commit하여 lock-in.

### Validation metrics

#### A. 4-class supercategory frame-level F1

Mapping (v1 → LUDB-compat):

| Our v1 | LUDB super |
|---|---|
| `p` | P |
| `q`, `r`, `s`, `w` | QRS |
| `t` | T |
| `_`, `*`, `?` | iso/other |

50Hz per-frame array로 expand → per-class precision/recall/F1.

#### B. Boundary error (Martinez 2004 / LUDB paper 표준)

| Boundary | Tolerance |
|---|---|
| P_onset, P_offset | ±50 ms |
| QRS_onset, QRS_offset | ±40 ms |
| T_onset | ±50 ms |
| T_offset | ±100 ms |

각 LUDB-annotated boundary에 대해 ±tolerance 내 nearest predicted boundary 검색 → hit이면 `error_ms = |LUDB_t - pred_t|` 기록.

**Per boundary type**:
- Sensitivity (recall): hit / total LUDB
- PPV (precision): hit / total predicted
- Mean error ± SD, Median, p95 (ms)

#### C. Q-loss rate

```
Q-loss rate = 1 - (RLE에 등장한 q events / NK delineate가 검출한 Q peaks)
```

50Hz frame 양자화로 짧은 Q wave 소실률.

#### D. NK 자체 baseline

Frame F1 / boundary error를 두 가지로 계산:
- **NK direct**: NK output 직접 평가
- **Pipeline**: NK → labeler → RLE → expand-to-frames

차이가 거의 0이어야 함 (labeler는 lossless re-encoding이 목표).

### Acceptance criteria for Stage 1 v1.0

| Metric | Target | Stretch |
|---|---|---|
| P frame F1 | ≥ 0.80 | ≥ 0.90 |
| QRS frame F1 | ≥ 0.90 | ≥ 0.95 |
| T frame F1 | ≥ 0.75 | ≥ 0.85 |
| QRS_onset boundary median | ≤ 20 ms | ≤ 10 ms |
| QRS_onset sensitivity | ≥ 0.90 | ≥ 0.95 |
| Q-loss rate | ≤ 20% | ≤ 10% |
| ASCII art 가독성 | 10/10 records | — |

미달 시:
- Q-loss > 20% → multi-resolution head 재검토
- QRS F1 < 0.85 → NK method 변경 시도 (cwt, peak)
- T F1 < 0.70 → NK method ablation을 v1.1으로

## 8. Pacer spike detector

### Spike 신호 특성

| 속성 | 값 |
|---|---|
| Width | 1–3 ms (0.5–1.5 samples @ 500Hz) |
| Amplitude | 1–10 mV (ADC saturate 가능) |
| Slope | >50 mV/ms |
| Polarity | 양극·음극 모두 가능 |
| Bipolar artifact | overshoot/undershoot 가능 |

### 알고리즘 (highpass + adaptive threshold)

```python
# ecgcode/pacer.py
from scipy.signal import butter, filtfilt, find_peaks

def detect_spikes(
    signal: np.ndarray,
    fs: int = 500,
    cutoff_hz: float = 80.0,
    amp_threshold_mad: float = 5.0,
    max_width_ms: float = 4.0,
    refractory_ms: float = 5.0,
) -> np.ndarray:
    nyq = fs / 2
    b, a = butter(4, cutoff_hz / nyq, btype='high')
    hp = filtfilt(b, a, signal)
    abs_hp = np.abs(hp)
    threshold = amp_threshold_mad * np.median(abs_hp)
    peaks, _ = find_peaks(
        abs_hp,
        height=threshold,
        distance=int(refractory_ms * fs / 1000),
        width=(None, int(max_width_ms * fs / 1000)),
    )
    return peaks
```

### Per-lead 적용

각 lead 독립 호출 (lead마다 spike visibility 다름). Labeler가 해당 lead의 spike list 사용.

### Validation

**Positive set**: LUDB의 pacemaker 환자 10 records (`ludb.csv`의 axis = "None (pacemaker)" 필터).
**Negative set**: 비-pacemaker 190 records.

**Metric**:
- TPR per beat: 0.85 이상 목표
- FPR: < 2 spikes / 10s record (non-pacer)
- Inter-lead consistency: 같은 시각 spike가 ≥ 6/12 lead 검출

`scripts/validate_pacer.py` 별도 스크립트로 측정.

### Failure mode 대응

| 실패 | 대응 |
|---|---|
| 일부 pacer record에서 spike 0 | `amp_threshold_mad` 3.5로 낮춤 |
| Non-pacer FPR > 5 | `cutoff_hz` 100Hz로 올림 |
| 전체적 too aggressive | wavelet detector 시도 (v1.1) |

## 9. Testing strategy

### 계층

| 계층 | 목적 | 외부 의존성 |
|---|---|---|
| Unit | 순수 함수 round-trip, 경계값 | 없음 (synthetic) |
| Integration | LUDB record 1로 end-to-end smoke | LUDB record 1 |
| Validation | 전체 split 메트릭 | LUDB val 40 records |

CI: unit + integration. Validation은 manual.

### 핵심 테스트 케이스

**`test_vocab.py`**: ID 유일성, char/name 매핑 round-trip, active class count, pad ID = 0.

**`test_codec.py`**:
- `encode(decode(x)) == x`
- Length > 1020 split 검증
- 4ms snap (95ms → 96ms)
- Frame expansion at 20ms
- Spike priority override
- Q max-overlap rule

**`test_pacer.py`**:
- Synthetic spike inject → detect
- QRS-shape wave 무시 (false positive 방어)
- Refractory dedup

**`test_labeler.py`** (핵심):
- Normal QRS → q-r-s emission
- Wide QRS no Q/S → w emission
- Narrow QRS no Q/S → r (not w)
- Spike during iso → * in stream
- NK total failure → entire ?
- NK boundary overshoots → clamp

**`test_integration.py`** (LUDB 필요):
- Record 1 lead II end-to-end
- Total ms ≈ 10000 ± rounding
- 12 lead 전체 crash 없음

### CI

```bash
uv run pytest                 # unit + integration (LUDB 있으면)
uv run pytest -m "not slow"   # unit only
```

### 비-범위

- NK internal, scipy filter coefficient 테스트 안 함
- Property-based testing — v1.1
- Mutation, performance benchmark — 후순위

## 10. Open questions / 후속

- v1.1: `u`/`d`/`j` detector 구현
- v1.1: NK method ablation (`dwt` vs `cwt` vs `peak`)
- Stage 2: 50Hz frame classifier 학습 (NK pseudo-label로 학습, LUDB cardiologist label로 평가)
- Stage 3: boundary refinement (PLAN §5)
- 다른 데이터셋 확장: QTDB validation, 추가 학습 데이터
