"""Produce a defensible research summary from real measured results only."""
from __future__ import annotations

import csv
from pathlib import Path
from .utils import now_utc


def generate_report(benchmark_csv: str, report_path: str, plot_path: str | None = None) -> str:
    source=Path(benchmark_csv)
    if not source.is_file():raise FileNotFoundError('No actual benchmark CSV found. Run benchmark first.')
    with source.open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    if not rows or any(z.get('run_type')!='real_labeled_evaluation' for z in rows):
        raise ValueError('Only actual labeled evaluation rows can produce a research report')
    fields=('backend','device','batch_size','n_eval','top1_accuracy','median_latency_ms','p95_latency_ms',
            'images_per_second','model_file_mb','peak_process_rss_mb','prediction_agreement_vs_torch_cpu')
    head='| '+' | '.join(fields)+' |'
    sep='| '+' | '.join(['---']*len(fields))+' |'
    table='\n'.join(['| '+' | '.join(str(z.get(k,'')) for k in fields)+' |' for z in rows])
    result=f'''# EdgeAI-Lab — Real Model Deployment Benchmark

Generated: {now_utc()}

## Research question

How do conversion and post-training INT8 quantization affect accuracy, inference latency,
model file size, and observed process memory for **the same trained classifier**?

## Measured results

{head}
{sep}
{table}

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
'''
    path=Path(report_path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(result,encoding='utf-8')
    if plot_path:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:raise RuntimeError('Install plotting support: pip install -e ".[visualize]"') from exc
        cpu=[z for z in rows if z['device']=='cpu' and z['batch_size']==rows[0]['batch_size']]
        fig,ax=plt.subplots(figsize=(7.4,3.7))
        ax.bar([z['backend'] for z in cpu],[float(z['median_latency_ms']) for z in cpu]);
        ax.set_ylabel('Median latency, ms (lower is better)')
        ax.set_title(f"CPU inference comparison — batch size {rows[0]['batch_size']}")
        ax.tick_params(axis='x',labelrotation=20)
        fig.tight_layout()
        p=Path(plot_path);p.parent.mkdir(parents=True,exist_ok=True);fig.savefig(p,dpi=150)
        plt.close(fig)
    return result
