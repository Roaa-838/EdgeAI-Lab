# Experiment protocol

## Hypothesis
Static INT8 post-training quantization and ONNX conversion trade file size,
FP32 prediction agreement, model forward latency, and classification accuracy.
Performance changes depend on operator support and device; faster runtime is
not a foregone conclusion.

## Fair-comparison controls

1. Single trained checkpoint. Hashes must link Torch -> ONNX -> QDQ INT8.
2. Official Imagenette `train` split used for training and internal dev set;
   untouched `val` split is final evaluation only. User checkpoints require
   similarly held-out external images.
3. Calibration images are sampled from train only. Record sample size & seed.
4. Same class-index mapping, image transforms and validation samples for every
   backend. Same CPU thread count and batch sizes.
5. Warm up each backend, then record repeated model-forward latency. Use
   process isolation to avoid major cross-run state contamination.
6. For CPU comparisons, use the same machine; report CPU model, OS and
   runtime versions. Report CUDA separately.
7. Repeat a complete benchmark on a different day or 3 independent runs to
   assess machine-level variability. Isolated timing repeats measure within-run
   variability, not between-day reproducibility.

## Dataset requirements

`data/DATASET/train/CLASS/*.jpg` for training and PTQ calibration, and
`data/DATASET/val/CLASS/*.jpg` as external holdout. Classes and index ordering
must match metadata exactly. Tests on a dataset previously used to train the
model cannot be presented as independent model accuracy.

## Primary outcomes

- Top-1 accuracy (%) on validation; top-5 accuracy secondary
- Median and p95 inference latency (ms) at batch 1 and 8
- Images per second for each fixed batch size
- Model file bytes / MiB, on-disk serialized artifact comparison
- Whole-process peak resident set size (RSS): a diagnostic only, not isolated
  model RAM and not comparable to GPU memory
- Agreement rate against PyTorch FP32 predictions

## Statistical limitations

The same samples are reused across backends and timing repeats. For accuracy,
report n and optionally paired confidence intervals or McNemar statistics.
Use actual held-out data, not synthetic fixtures, in results tables. A training
checkpoint selected using test/val invalidates the holdout claim. A single
machine's timing result should not be universalized.

## Extensions

- Quantization calibration size ablation (e.g. 32/64/128)
- Per-class accuracy and confusion matrix
- Thread sensitivity, batch sensitivity
- Real device memory instrumentation with appropriate vendor tools
