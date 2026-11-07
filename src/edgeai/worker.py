"""Isolated inference process: one backend, one batch size, one device.

Inputs are preprocessed before timing. RSS includes this worker's dependencies;
RSS is not GPU VRAM and is *not* an isolated per-model allocation metric.
"""
from __future__ import annotations
import argparse
import os
import platform
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import psutil

from .dataset import list_images, stratified_limit, load_eval_arrays, validate_dataset_class_folders
from .utils import write_json, file_sha256, mb, now_utc


class RSSSampler:
    def __init__(self,interval=0.01):
        self.interval=interval;self.stop=threading.Event()
        self.process=psutil.Process(os.getpid())
        self.start_mb=self.process.memory_info().rss/(1024*1024)
        self.peak_mb=self.start_mb
        self.thread=None
    def __enter__(self):
        def sample():
            while not self.stop.wait(self.interval):
                self.peak_mb=max(self.peak_mb,self.process.memory_info().rss/(1024*1024))
        self.thread=threading.Thread(target=sample,daemon=True)
        self.thread.start();return self
    def __exit__(self,*args):
        self.peak_mb=max(self.peak_mb,self.process.memory_info().rss/(1024*1024))
        self.stop.set();self.thread.join()


def topk_counts(logits: np.ndarray, labels: np.ndarray, k=5):
    if logits.ndim!=2 or labels.ndim!=1 or logits.shape[0]!=labels.size:
        raise ValueError('Expected logits [N,C] and labels [N]')
    if labels.size==0 or np.any(labels < 0) or logits.shape[1]<=max(labels):
        raise ValueError('Invalid labels or output dimension')
    order=np.argsort(-logits,axis=1,kind='stable')
    top1=int(np.sum(order[:,0]==labels))
    topk=int(sum(int(labels[i] in order[i,:min(k,logits.shape[1])]) for i in range(labels.size)))
    return top1,topk


def run_once(args):
    from .models import load_checkpoint
    import json
    meta=json.loads(Path(args.metadata).read_text(encoding='utf-8'))
    validate_dataset_class_folders(Path(args.data_root)/'val',meta['class_to_idx'])
    items=stratified_limit(list_images(Path(args.data_root)/'val',meta['class_to_idx']),args.max_eval,args.seed)
    if len(items)<args.batch_size:
        raise ValueError('At least batch_size validation images are required')
    images,labels=load_eval_arrays(items,int(meta['input_size']),int(meta['resize_size']))
    n=len(items)
    device='cpu' if args.backend!='torch-cuda' else 'cuda'
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    batch=args.batch_size
    inputs=[]
    # Use full-size batches for timing; avoid tail-batch performance distortion.
    for start in range(0,n-batch+1,batch):inputs.append(images[start:start+batch])
    if not inputs:inputs=[images[:batch]]
    measured=[];predictions=[];total_correct=0;total_top5=0
    model_path=args.checkpoint if args.backend.startswith('torch') else (args.onnx if args.backend=='onnx-cpu' else args.int8)
    if not Path(model_path).exists():
        raise FileNotFoundError(f'File required for {args.backend}: {model_path}')
    with RSSSampler() as memory:
        if args.backend.startswith('torch'):
            import torch
            torch.set_num_threads(args.threads)
            if device=='cuda' and not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable; run torch-cpu or install CUDA torch')
            model,_=load_checkpoint(args.checkpoint)
            model=model.to(device).eval()
            def forward(batch_array):
                tensor=torch.from_numpy(batch_array)
                if device=='cuda':tensor=tensor.to('cuda',non_blocking=False)
                with torch.inference_mode():
                    output=model(tensor)
                if device=='cuda':torch.cuda.synchronize()
                return output.cpu().numpy()
            hardware_name=torch.cuda.get_device_name(0) if device=='cuda' else platform.processor()
        else:
            import onnxruntime as ort
            opts=ort.SessionOptions()
            opts.intra_op_num_threads=args.threads
            opts.inter_op_num_threads=1
            opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session=ort.InferenceSession(str(model_path),sess_options=opts,providers=['CPUExecutionProvider'])
            inp_name=session.get_inputs()[0].name
            def forward(batch_array):
                return session.run(None,{inp_name:batch_array})[0]
            hardware_name=platform.processor()
        # Accuracy over every selected image; no artificial padding, not timed.
        for start in range(0,n,batch):
            end=min(start+batch,n)
            logits=forward(images[start:end])
            if logits.ndim!=2 or logits.shape[1]!=int(meta['num_classes']):
                raise ValueError(f'Output logits shape {logits.shape} incompatible with {meta["num_classes"]} classes')
            a,b=topk_counts(logits,labels[start:end]);total_correct+=a;total_top5+=b
            predictions.extend(logits.argmax(axis=1).astype(int).tolist())
        for i in range(args.warmup):forward(inputs[i%len(inputs)])
        for i in range(args.repeats):
            batch_input=inputs[i%len(inputs)]
            t0=time.perf_counter_ns()
            forward(batch_input)
            measured.append((time.perf_counter_ns()-t0)/1e6)
    a=np.asarray(measured)
    result={
      'run_type':'real_labeled_evaluation', 'timestamp_utc':now_utc(),
      'backend':args.backend,'device':device,'device_name':hardware_name,
      'batch_size':batch,'n_eval':n,'n_timing_repeats':args.repeats,'n_warmup':args.warmup,
      'torch_threads':args.threads if args.backend.startswith('torch') else None,
      'ort_threads':args.threads if args.backend.startswith('onnx') else None,
      'top1_accuracy':round(total_correct/n,6),'top5_accuracy':round(total_top5/n,6),
      'median_latency_ms':round(float(np.median(a)),5),'p95_latency_ms':round(float(np.percentile(a,95)),5),
      'mean_latency_ms':round(float(np.mean(a)),5),'std_latency_ms':round(float(np.std(a,ddof=1)),5) if len(a)>1 else 0.,
      'images_per_second':round(float(batch*1000/np.mean(a)),3),
      'peak_process_rss_mb':round(memory.peak_mb,3),'process_rss_before_backend_mb':round(memory.start_mb,3),
      'peak_minus_before_backend_mb':round(max(0.,memory.peak_mb-memory.start_mb),3),
      'model_file_mb':mb(Path(model_path).stat().st_size),'model_sha256':file_sha256(model_path),
      'dataset':'held_out_val','preprocessing':'resize_shorter_side_center_crop_imagenet_normalization',
      'input_size':meta['input_size'],'resize_size':meta['resize_size'],'seed':args.seed,
      'accuracy_note':'Top-1/top-5 on held-out labels. Not official ImageNet accuracy.',
      'latency_note':'Model forward only, preprocessed input; GPU synchronization included; transfer included for CUDA.',
      'memory_note':'Sampled peak RSS of whole worker; not model-only RAM or GPU VRAM.',
      'predictions':predictions,
    }
    write_json(args.output,result)
    print(f"{args.backend} batch={batch}: top1={result['top1_accuracy']:.3f}, median={result['median_latency_ms']:.3f} ms")
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ('checkpoint','onnx','int8','metadata','data-root','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--backend',choices=['torch-cpu','torch-cuda','onnx-cpu','onnx-int8-cpu'],required=True)
    p.add_argument('--batch-size',type=int,required=True)
    p.add_argument('--max-eval',type=int,default=200)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--repeats',type=int,default=40)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if args.batch_size<1 or args.warmup<0 or args.repeats<2 or args.threads<1:raise ValueError('Invalid benchmark configuration')
    run_once(args)

if __name__=='__main__':main()
