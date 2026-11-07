# EdgeAI-Lab — Real Model Deployment Benchmark

Generated: 2026-11-07T10:48:37+00:00

## Research question

How do conversion and post-training INT8 quantization affect accuracy, inference latency,
model file size, and observed process memory for **the same trained classifier**?

## Measured results

| backend | device | batch_size | n_eval | top1_accuracy | median_latency_ms | p95_latency_ms | images_per_second | model_file_mb | peak_process_rss_mb | prediction_agreement_vs_torch_cpu |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| torch-cpu | cpu | 1 | 200 | 0.71 | 11.5613 | 15.7113 | 80.762 | 5.9464 | 362.57 | 1.0 |
| onnx-cpu | cpu | 1 | 200 | 0.71 | 1.21995 | 1.48222 | 799.611 | 5.8384 | 135.539 | 1.0 |
| onnx-int8-cpu | cpu | 1 | 200 | 0.7 | 2.755 | 3.47433 | 354.826 | 1.6428 | 133.723 | 0.965 |
| torch-cpu | cpu | 8 | 200 | 0.71 | 25.8877 | 29.81466 | 308.306 | 5.9464 | 372.039 | 1.0 |
| onnx-cpu | cpu | 8 | 200 | 0.71 | 4.70735 | 6.26404 | 1577.65 | 5.8384 | 147.148 | 1.0 |
| onnx-int8-cpu | cpu | 8 | 200 | 0.7 | 14.01635 | 17.35885 | 578.835 | 1.6428 | 155.352 | 0.965 |

## Experimental design

- Training: ImageFolder `train/` with development split drawn from training data only, or a separately provided trained checkpoint.
- Calibration: a documented subset of **train/**; never the held-out validation data.
- Evaluation: same held-out **val/** images, labels and preprocessing for each backend.
- Latency: model forward only, preprocessed tensors, warm-up iterations, repeated measurements.
- CPU: PyTorch CPU, ONNX Runtime CPU, and ORT static QDQ INT8 CPU where supported.
- Batch sizes: separate experiments; compare equal batch-size and equal device rows only.
- Memory: peak **whole-process RSS** sampled during the worker; this is not isolated model memory or CUDA VRAM.
- Accuracy: top-1 and top-5 on this **specific evaluation sample**, not official ImageNet test accuracy.

## Interpretive cautions

1. These are measured software results on the local test cohort, not general model guarantees.
2. Faster/slower INT8 execution depends on operator coverage, CPU instruction support and runtime.
3. QDQ static INT8 involves quantization/calibration and can reduce accuracy; inspect both accuracy and runtime.
4. Training and evaluation provenance should be reported in the accompanying model metadata.
5. Quantized ONNX weights and Torch state dictionaries have different serialization and graph overheads.
6. Choosing the best checkpoint using the official external validation set would invalidate its held-out interpretation.

## Suggested follow-up research

Repeat runs on another CPU, add calibration-size ablation, compare per-class errors,
and investigate operators causing accuracy differences. Do not declare superiority from one machine.
