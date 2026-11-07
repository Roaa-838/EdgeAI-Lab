# EdgeAI-Lab

**Reproducible Accuracy–Latency–Size Trade-offs for Vision Model Deployment**

A research-engineering benchmark that compares **one trained image classifier** across:

1. **PyTorch FP32 CPU** (baseline)
2. **ONNX Runtime FP32 CPU** (exported graph)
3. **ONNX Runtime static QDQ INT8 CPU** (post-training calibrated on training-only data)
4. **PyTorch CUDA**, optional (reported separately; not equated with CPU benchmark)

The same held-out validation images, image preprocessing, class-to-index mapping, and fixed batch sizes are used for all backends. Individual backends execute in fresh Python worker processes to limit measurement contamination. It reports measured **top-1 and top-5 accuracy, latency p50/p95, throughput, whole-process peak RSS, and model-file size**, with provenance and reproducibility metadata.

> **Research status (Nov 2026):** Implemented research *framework*. Offline unit tests and a synthetic smoke test are engineering validation only. **No ONNX or INT8 inference performance, model accuracy, or speed-up is claimed until you run the real benchmark.** Full ONNX Runtime execution and end-to-end INT8 compatibility require testing on your hardware.

## Why this project?

A compressed model is not automatically faster or more accurate. Conversion can introduce numerical differences; INT8 speedups depend on operator coverage, CPU instructions, and calibration. The central question is whether the lower model size and potentially lower latency justify any accuracy loss on the **same** held-out evaluation images.

## Choose your research path

**A. Use an existing trained PyTorch vision model (preferred).** If you have a compatible `state_dict` saved from your own trained `mobilenet_v3_small`, `resnet18`, `efficientnet_b0`, or `vgg16` model, run `edgeai adopt`. The original architecture, class order and preprocessing must match. **Do not use `adopt` to pretend that Keras `.h5`, arbitrary TorchScript, or a different classifier is directly supported.** If you used another architecture, extend `src/edgeai/models.py` first.

**B. Create a defensible baseline from scratch.** Download fastai's Imagenette-160, train MobileNetV3-small **from random initialization**, choose the checkpoint using a small training-only development split, then evaluate *once* on its separate `val/` split. This avoids falsely presenting an ImageNet-pretrained model tested on Imagenette as independent validation (the datasets can overlap).

## 1. Setup on Windows 10/11

Requirements: Python 3.12, git, internet for first installation / dataset. CUDA is optional. Install **CPU-only** ONNX Runtime for cross-framework comparisons; GPU benchmarking is optional and separate.

```powershell
cd "D:\EdgeAI-Lab"
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[full]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m edgeai status
.\.venv\Scripts\python.exe -m edgeai smoke-test
```

If you already have a working CUDA PyTorch installation, use its matching torchvision build. Consult the [official PyTorch installer](https://pytorch.org/get-started/locally/); do not install incompatible torchvision wheels over an existing GPU setup.

**The smoke test creates temporary random images and checks preprocessing / model forward pass. It is not accuracy research.**

## 2. Option B — Train a new benchmark model from scratch

Download the official fastai dataset into ignored `data/` (no copyrighted images are committed):

```powershell
.\.venv\Scripts\python.exe -m edgeai download-imagenette
```

Train MobileNetV3-small from scratch on its `train/` images (CUDA if available), while holding out a 10% *development subset within train*. The provided `val/` is not touched by training:

```powershell
.\.venv\Scripts\python.exe -m edgeai train --data-root data/imagenette2-160 --arch mobilenet_v3_small --epochs 8 --batch-size 32 --img-size 160 --seed 42 --checkpoint models/classifier.pth
```

Check `models/training_history.json` and the training-only development accuracy. More training may be needed for satisfactory performance; **do not tune directly on external `val/` after benchmarking**. Report architecture, number of epochs and checkpoint provenance honestly.

## 3. Option A — Adopt your existing trained model

For an **existing compatible torchvision state_dict** (example: `resnet18` with `fresh` and `rotten` output labels in that exact order):

```powershell
.\.venv\Scripts\python.exe -m edgeai adopt --source "D:\Models\my_resnet18.pth" --arch resnet18 --classes fresh rotten --input-size 224 --resize-size 256 --checkpoint models/classifier.pth
```

`models/metadata.json` then records class mappings, model architecture and preprocessing settings. Ensure that your **original training normalization and resize/crop match this repository's ImageNet mean/std inference recipe**. If not, edit the preprocessing module to reproduce your training pipeline; otherwise accuracy comparisons are not trustworthy.

Provide a labeled `data/my_dataset/val/<class_name>/*.jpg` folder and a **separate** `data/my_dataset/train/<class_name>/*.jpg` folder for INT8 calibration. The class names must be identical to those passed in `--classes`. For trustworthy accuracy, use an external held-out set not used to train or select the checkpoint.

In subsequent commands replace `data/imagenette2-160` with `data/my_dataset`.

## 4. Export FP32 to ONNX

```powershell
.\.venv\Scripts\python.exe -m edgeai export --checkpoint models/classifier.pth --output models/classifier.fp32.onnx --check-parity
```

The file `models/classifier.fp32.meta.json` contains checkpoint hashes, opset, export settings, and file size. Export uses a **legacy dynamic-axes exporter deliberately** for compatibility with the target PyTorch 2.x range; PyTorch's newer `dynamo=True` exporter is the recommended modern approach and can be substituted when validated. `--check-parity` tests FP32 ONNX against PyTorch and fails if their outputs diverge beyond a small tolerance.

## 5. Calibrate static INT8 on **train only**

```powershell
.\.venv\Scripts\python.exe -m edgeai quantize --onnx models/classifier.fp32.onnx --checkpoint models/classifier.pth --data-root data/imagenette2-160 --output models/classifier.int8.onnx --calibration-count 64 --validation-check-count 0 --seed 42
```

This produces ONNX Runtime **QDQ static quantization** using MinMax calibration and Conv/MatMul/Gemm operators. The calibration sample is drawn from `train/`, not from external `val/`. A content-hash overlap guard checks **every held-out validation image by default** and rejects duplicate content against the calibration sample. The quantized model is test-loaded on CPU immediately; if the runtime does not support a quantized operator, it reports the error instead of pretending success.

ONNX Runtime recommends static PTQ for CNNs; INT8 inference is **not guaranteed faster** for every CPU/architecture. If static QDQ cannot be loaded on your specific runtime, retain the FP32 comparison and document the incompatible operator—don't manufacture an INT8 result.

## 6. Benchmark the same held-out data

```powershell
.\.venv\Scripts\python.exe -m edgeai benchmark --checkpoint models/classifier.pth --onnx models/classifier.fp32.onnx --int8 models/classifier.int8.onnx --metadata models/metadata.json --data-root data/imagenette2-160 --backends torch-cpu onnx-cpu onnx-int8-cpu --batch-sizes 1 8 --max-eval 200 --warmup 10 --repeats 40 --threads 4 --seed 42
```

Every backend and batch size runs as a distinct subprocess. The code checks that ONNX and INT8 files match the same FP32 checkpoint (SHA256 metadata). `results/benchmark.csv` and `results/benchmark.json` hold real measured values only. Per-backend predictions allow measuring conversion agreement. Input preprocessing is excluded from forward latency for all runtimes.

Optional GPU **separate** benchmark:

```powershell
.\.venv\Scripts\python.exe -m edgeai benchmark --backends torch-cuda --batch-sizes 1 8 --max-eval 200
```

Compare **CPU to CPU** for backend speedups; do not claim that CUDA throughput and CPU INT8 are equivalent device tests. GPU measurements synchronize the device. The sampled RSS value is **whole-process memory, not VRAM or model-only memory**.

## 7. Generate measured research report and dashboard

```powershell
.\.venv\Scripts\python.exe -m edgeai report
.\.venv\Scripts\python.exe -m streamlit run src/edgeai/dashboard.py
```

Expected actual outputs **after running the benchmark**:

- `results/benchmark.csv` — measured per-backend and per-batch accuracy, speed, size, RSS
- `results/benchmark.json` — complete benchmark metadata and model provenance
- `results/figures/cpu_latency.png` — actual CPU latency comparison
- `reports/measured_results.md` — auto-generated research report with limitations

The included `reports/research_report.pdf` is a **methodology-only draft**, not a claim of experimental performance.

## Experiment checklist

- [ ] Checkpoint trained on legitimate data, with matching class order and normalization
- [ ] External evaluation images have never been used for training, checkpoint selection or INT8 calibration
- [ ] Same FP32 PyTorch checkpoint used for both ONNX exports
- [ ] Same preprocessing, held-out file order, and output-label mapping for every backend
- [ ] Test `--check-parity` and record any quantized accuracy change
- [ ] Compare latency at batch 1 and 8 with warmup/repeats and fixed CPU thread count
- [ ] Record processor, OS, runtime versions, sample sizes and artifacts' hashes
- [ ] Report actual numbers without assuming compression always means speedup

## Technology choices

Python · PyTorch · torchvision · ONNX · ONNX Runtime · QDQ INT8 PTQ · Pillow · NumPy · psutil · pytest · Streamlit.

## Structure

```text
EdgeAI-Lab/
├── src/edgeai/
│   ├── cli.py               # Command-line interface
│   ├── models.py            # Supported architectures and checkpoint provenance
│   ├── dataset.py           # Split checks + identical input processing
│   ├── training.py          # From-scratch train and training-only dev selection
│   ├── exporting.py         # ONNX dynamic-batch export + optional parity check
│   ├── quantization.py      # Static QDQ INT8 calibration on train-only images
│   ├── worker.py            # Isolated CPU/GPU backend inference + timing
│   ├── benchmark.py         # Cross-backend runner and experiment output
│   ├── report.py            # Metrics tables and evidence report
│   └── dashboard.py         # Streamlit comparisons (real results only)
├── configs/experiment.example.json
├── docs/EXPERIMENT_PROTOCOL.md
├── docs/RESEARCH_INTEGRITY.md
├── notebooks/
├── reports/research_report.pdf
├── tests/
├── .github/workflows/ci.yml
└── pyproject.toml
```

**Repository name:** `EdgeAI-Lab` | Suggested GitHub topic tags: `edge-ai`, `onnxruntime`, `quantization`, `computer-vision`, `benchmarking`, `pytorch`, `research-reproducibility`.

## Research ethics and limitations

No patient data, restricted training datasets, pretrained model weights or API secrets are included. Imagenette is a third-party dataset and must be downloaded separately under its applicable terms. The PyTorch model can be trained with random initialization or imported as an existing state_dict; no accuracy claim is made until the held-out evaluation runs. An isolated dataset benchmark is not evidence of safety or production robustness.

## Sources

- [PyTorch ONNX export](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html)
- [ONNX Runtime static/dynamic quantization guidance](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [Torchvision MobileNetV3 small](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html)
- [Imagenette](https://github.com/fastai/imagenette)
