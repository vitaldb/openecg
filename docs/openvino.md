# OpenVINO path

`openecg.openvino_backend` runs the v56c boundary detector on OpenVINO, alongside the
reference runtimes. It is an optional extra: nothing else in
openecg imports it, and the TFLite and ONNX Runtime paths are unchanged.

```bash
pip install "openecg[openvino]"
```

```python
from openecg.openvino_backend import OpenVINOBoundary, available_devices

print(available_devices())                     # e.g. ['CPU', 'GPU', 'NPU']
det = OpenVINOBoundary(device="CPU")           # v56c boundary, released fp32 ONNX
cls_logits = det.forward_window(sig_250hz)     # (500, 4) for one 2500-sample window
```

`device` takes any OpenVINO device name available on the host, e.g. `CPU` (default),
`GPU`, `NPU`, or `AUTO`; an absent device raises a clear error.

## Artifacts

| Model | Reference runtime | OpenVINO input |
|---|---|---|
| Boundary detector (v56c) | `boundary_int8.tflite` via TFLite / LiteRT | `v56c_fp32.onnx` (release `boundary-v56c`), read directly |

The OpenVINO TFLite frontend cannot load `boundary_int8.tflite`, so `OpenVINOBoundary`
reads the fp32 ONNX that `scripts/export_boundary_onnx.py` exports from `v56c.pt`
(see [`v56c_MODEL_CARD.md`](../openecg/models/v56c_MODEL_CARD.md)). It is not bundled in
the wheel. `OpenVINOBoundary()` looks for it at `$OPENECG_V56C_ONNX`, then
`data/deploy/v56c_fp32.onnx`, then `~/.cache/openecg/v56c_fp32.onnx` (`$OPENECG_CACHE`).
To download the release asset into the cache (sha256-checked against
`SHA256SUMS`):

```python
from openecg.openvino_backend import fetch_v56c_fp32_onnx
fetch_v56c_fp32_onnx()
```

## Parity with the fp32 ONNX

`scripts/verify_openvino_boundary.py` is the OpenVINO counterpart of
`scripts/export_boundary_onnx.py --verify`: same windows (first 64 LUDB windows, 500 →
250 Hz by `[::2]`, rank-normalized), same tolerance (`max |Δ| < 1e-4` on both heads, 100 %
frame-argmax agreement), with ONNX Runtime CPU on the same ONNX as the reference. It also
decodes boundaries from both outputs (`post_process_frames` → `extract_boundaries` →
`apply_reg_to_boundaries`) and reports how many windows give identical boundaries.

```bash
python -m scripts.verify_openvino_boundary --ludb-val --device CPU
python -m scripts.verify_openvino_boundary --verify data/physionet/ludb_cache/<record>.npz --device GPU
```

`--tol` / `--min-agree` relax the gate for devices that do not compute in f32 (`NPU`; see
the results below for the size of the f16 differences).

## Compile step (applied automatically)

The dynamic batch dim of the ONNX input is fixed to 1 before `compile_model`
(input `[1, 2500]`), since inference is one window at a time.

## Precision

`OpenVINOBoundary` takes an OpenVINO `config=` dict. When it is not given, it sets
`INFERENCE_PRECISION_HINT=f32` on:

- **`CPU`**, so results are reproducible across hosts. Without the hint, OpenVINO picks
  bf16 on CPUs with native bf16 support.
- **`GPU`**. With the device default (f16) the outputs no longer meet the `1e-4` parity
  tolerance (Arc 140T: max |Δ| `cls_logits` 2.8e-2, 99.99 % frame-argmax agreement,
  boundaries within one sample).

Other devices use their own defaults. To opt into bf16 on a CPU that supports it, or
into the device default:

```python
OpenVINOBoundary(device="CPU", config={"INFERENCE_PRECISION_HINT": "bf16"})
OpenVINOBoundary(device="GPU", config={})  # device default (f16 on GPU)
```

The results below are for these defaults.

## Reproduce

```bash
pip install "openecg[openvino,loaders]"
python -m scripts.verify_openvino_boundary --ludb-val --device CPU   # boundary vs fp32 ONNX
```

`verify_openvino_boundary.py` and `eval_macro_f1_openvino.py` decode boundaries with
`openecg.stage2`, which imports `torch` (CPU build is enough); `pip install torch`
first. `OpenVINOBoundary` itself does not need it. The `tflite` reference in
`eval_macro_f1_openvino.py` needs a TFLite interpreter (`tflite-runtime` from
`[deploy]`, or `ai-edge-litert` on Windows).

Boundary macro-F1 on LUDB / ISP / QTDB (PhysioNet / Zenodo data, not redistributed),
same protocol as the v56c model-card table (lead II; first lead for QTDB):

```bash
export OPENECG_LUDB_ZIP=data/physionet/ludb-1.0.1.zip
export OPENECG_LUDB_CACHE=data/physionet/ludb_cache
export OPENECG_ISP_ZIP=data/physionet/isp_delineation_dataset.zip   # Zenodo 14679837
export OPENECG_ISP_CACHE=data/physionet/isp_cache
export OPENECG_QTDB_CACHE=data/physionet/qtdb_cache
python -m scripts.eval_macro_f1_openvino --backends tflite,ort,openvino:CPU,openvino:GPU,openvino:NPU
python -m scripts.eval_macro_f1_openvino --backends tflite,torch:xpu   # PyTorch XPU from v56c.pt
```

Tests: `pytest tests/test_openvino_backend.py` (skipped when `openvino` is not installed;
boundary tests skipped when `v56c_fp32.onnx` is not found).

## Results

Tested host: Intel Core Ultra 9 285H (`CPU`), Intel Arc 140T integrated GPU (`GPU`),
Intel AI Boost (`NPU`); Windows 11 Pro
(10.0.26200), Python 3.12.10, OpenVINO 2026.4.1, ONNX Runtime 1.31.1, ai-edge-litert
2.3.0, torch 2.15.0 (`+xpu` build for the XPU rows). Defaults as above (`CPU` / `GPU`
f32, `NPU` device default).

### Parity with the fp32 ONNX (boundary)

`scripts/verify_openvino_boundary.py --ludb-val`: 64 LUDB val windows, reference ONNX
Runtime CPU on the same `v56c_fp32.onnx`.

| Device | max \|Δ\| `cls_logits` | max \|Δ\| `reg_offsets` | frame-argmax agreement | windows with identical boundaries | result (tol 1e-4, 100 %) |
|---|---:|---:|---:|---:|---|
| `CPU` | 1.6e-5 | 6.7e-5 | 100 % (32000/32000) | 64/64 | OK |
| `GPU` | 1.5e-5 | 6.3e-5 | 100 % (32000/32000) | 64/64 | OK |
| `NPU` | 2.8e-2 | 0.21 | 99.98 % (31994/32000) | 35/64 (max shift 1 sample, same 4306 boundaries) | outside the f32 tolerance |

The `NPU` computes in f16, so it does not meet the f32 tolerance; its boundaries move
by at most one sample (4 ms at 250 Hz).

### Boundary macro-F1 vs the int8 TFLite

Martínez tolerances, same protocol and splits as the v56c table in
[`v56c_MODEL_CARD.md`](../openecg/models/v56c_MODEL_CARD.md) (lead II; first lead for the
QTDB T-subset). OpenVINO and ONNX Runtime read `v56c_fp32.onnx`; PyTorch reads `v56c.pt`.

| Corpus (n) | TFLite int8 (ref) | ORT `CPU` | OpenVINO `CPU` | OpenVINO `GPU` | OpenVINO `NPU` | PyTorch `xpu` (Arc 140T) | PyTorch `cpu` |
|---|---:|---:|---:|---:|---:|---:|---:|
| LUDB val (41) | 0.9633 | 0.9633 | 0.9633 | 0.9633 | 0.9633 | 0.9633 | 0.9633 |
| ISP test (72) | 0.9711 | 0.9711 | 0.9711 | 0.9711 | 0.9711 | 0.9711 | 0.9711 |
| QTDB T-subset (44) | 0.9076 | 0.9083 | 0.9083 | 0.9083 | 0.9083 | 0.9083 | 0.9083 |
| **mean** | **0.9473** | **0.9476** | **0.9476** | **0.9476** | **0.9476** | **0.9476** | **0.9476** |

Frame-argmax agreement with the TFLite model over all scored windows is 0.9987 /
0.9985 / 0.9978 (LUDB / ISP / QTDB) for every fp32 path, and 0.9986 / 0.9985 / 0.9978 on
`NPU`. The difference from TFLite is the int8 weight rounding in `boundary_int8.tflite`
(see the model card), not the runtime.
