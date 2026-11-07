"""Run backend comparisons in isolated subprocesses.

The exact same evaluation files and image transform are used for all backends.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from .utils import file_sha256, read_json, write_json, now_utc

FIELDS=('backend','device','batch_size','n_eval','top1_accuracy','top5_accuracy',
        'median_latency_ms','p95_latency_ms','mean_latency_ms','std_latency_ms',
        'images_per_second','peak_process_rss_mb','process_rss_before_backend_mb',
        'peak_minus_before_backend_mb','model_file_mb','model_sha256','n_warmup',
        'n_timing_repeats','torch_threads','ort_threads','prediction_agreement_vs_torch_cpu',
        'run_type','timestamp_utc','dataset')


def verify_provenance(checkpoint, onnx, int8, backends):
    if any(z.startswith('onnx-') for z in backends):
        meta_path=Path(onnx).with_suffix('.meta.json')
        if not meta_path.exists():raise FileNotFoundError(f'Export metadata missing: {meta_path}')
        m=read_json(meta_path)
        if m['checkpoint_sha256']!=file_sha256(checkpoint) or m['onnx_sha256']!=file_sha256(onnx):
            raise ValueError('ONNX does not match the supplied PyTorch checkpoint. Re-export.')
    if 'onnx-int8-cpu' in backends:
        qmeta=Path(int8).with_suffix('.meta.json')
        if not qmeta.exists():raise FileNotFoundError(f'Quantization metadata missing: {qmeta}')
        m=read_json(qmeta)
        if m['float_onnx_sha256']!=file_sha256(onnx) or m['sha256']!=file_sha256(int8):
            raise ValueError('INT8 model does not match the FP32 ONNX model. Re-quantize.')


def benchmark_models(checkpoint: str, onnx: str, int8: str, metadata: str, data_root: str,
                     output_dir: str='results',backends=None,batch_sizes=None,max_eval=200,
                     warmup=10,repeats=40,threads=4,seed=42) -> list[dict]:
    backends=backends or ['torch-cpu','onnx-cpu','onnx-int8-cpu']
    batch_sizes=batch_sizes or [1,8]
    allowed={'torch-cpu','torch-cuda','onnx-cpu','onnx-int8-cpu'}
    if not set(backends).issubset(allowed):raise ValueError('Invalid backend selection')
    if not Path(metadata).exists():raise FileNotFoundError(f'Missing metadata: {metadata}')
    verify_provenance(checkpoint,onnx,int8,backends)
    p=Path(output_dir);p.mkdir(parents=True,exist_ok=True)
    rows=[]
    for batch in batch_sizes:
        for backend in backends:
            path=p/f'run_{backend}_b{batch}.json'
            cmd=[sys.executable,'-m','edgeai.worker','--checkpoint',str(checkpoint),
                 '--onnx',str(onnx),'--int8',str(int8),'--metadata',str(metadata),
                 '--data-root',str(data_root),'--output',str(path),'--backend',backend,
                 '--batch-size',str(batch),'--max-eval',str(max_eval),'--warmup',str(warmup),
                 '--repeats',str(repeats),'--threads',str(threads),'--seed',str(seed)]
            print('Running:',backend,'batch=',batch,flush=True)
            subprocess.run(cmd,check=True)
            rows.append(read_json(path))
    by_batch={}
    for r in rows:
        by_batch.setdefault(r['batch_size'],{})[r['backend']]=r
    for batch,group in by_batch.items():
        baseline=group.get('torch-cpu')
        if not baseline:continue
        ref=baseline['predictions']
        for r in group.values():
            if len(ref)!=len(r['predictions']):raise ValueError('Different validation sets used by backends')
            r['prediction_agreement_vs_torch_cpu']=round(sum(a==b for a,b in zip(ref,r['predictions']))/len(ref),6)
    with (p/'benchmark.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    write_json(p/'benchmark.json',{'run_type':'real_labeled_evaluation','created_at':now_utc(),
                                  'model_checkpoint':str(checkpoint),'all_runs':rows,
                                  'memory_caveat':'Sampled process RSS; compare backend memory with caution.',
                                  'dataset_caveat':'ImageFolder val data, distinct from train/dev/calibration.'})
    return rows
