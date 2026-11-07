# Positioning EdgeAI-Lab in 2027 master's applications

## Application evidence hierarchy

1. Finished real experiments with a reproducible benchmark and actual validation data.
2. Independently runnable source, unit tests, clear hardware/version provenance.
3. A clear failure analysis (such as unexpected INT8 accuracy loss or unsupported ONNX operators).
4. A methodology-only prototype **without unmeasured performance claims**.

## Which master’s programs benefit most?

| Program | What to emphasize |
|---|---|
| CNAM AI for Connected Industries | Practical model integration, CPU latency, deployability, ONNX Runtime |
| Grenoble MoSIG | Quantization effects, controlled computer-vision experiments, reproducibility |
| VIBOT | Vision inference constraints, predictive fidelity, device-specific limitations |
| IPCVAI | Representation preservation and experimental evaluation after conversion |
| Côte d'Azur Computer Science | Inference efficiency, systems metrics, research methodology |
| Lille Data Science | Paired evaluations, baselines and experimental reporting |

## CV bullet before real results

> Implemented a reproducible computer-vision deployment benchmark comparing
> PyTorch FP32, ONNX Runtime FP32, and calibrated static INT8 inference, with
> automated holdout checks, inference measurement, and provenance tracking.

Use that bullet only after all named components are working as stated, particularly the ONNX/INT8 run. Until then, substitute:

> Developed a tested Python benchmark framework for comparing computer-vision
> deployment runtimes, with held-out evaluation design, checkpoint provenance,
> and model-size/latency measurement infrastructure.

## CV bullet after real results

> Evaluated [trained model architecture] on [n] held-out [dataset] images,
> comparing PyTorch, ONNX Runtime and INT8 at batch sizes 1 and 8;
> measured [verified change] in model size and [verified change] in latency,
> with [verified change] in accuracy.

Replace every placeholder from `results/benchmark.csv` after real experiments.
Never use a university scholarship CV to claim model speedups measured on the
synthetic smoke test or inferred from compressed file size alone.
