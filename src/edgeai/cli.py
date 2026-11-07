"""CLI entrypoint for EdgeAI-Lab."""
from __future__ import annotations
import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path

from .utils import write_json


def build_parser():
    p=argparse.ArgumentParser(prog='edgeai',description='EdgeAI-Lab: honest CV deployment research')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('status',help='Check installed components and detected hardware')
    demo=sub.add_parser('smoke-test',help='Synthetic engineering smoke test (no research accuracy metrics)')
    demo.add_argument('--output',default='results/smoke_test.json')
    down=sub.add_parser('download-imagenette',help='Download official Imagenette-160 archive to data/')
    down.add_argument('--output-root',default='data')
    tr=sub.add_parser('train',help='Train a torchvision model from scratch on train/; val/ stays untouched')
    tr.add_argument('--data-root',default='data/imagenette2-160')
    tr.add_argument('--checkpoint',default='models/classifier.pth')
    tr.add_argument('--arch',choices=['mobilenet_v3_small','resnet18','efficientnet_b0','vgg16'],default='mobilenet_v3_small')
    tr.add_argument('--epochs',type=int,default=8)
    tr.add_argument('--batch-size',type=int,default=32)
    tr.add_argument('--max-train',type=int,default=0)
    tr.add_argument('--img-size',type=int,default=160)
    tr.add_argument('--seed',type=int,default=42)
    tr.add_argument('--lr',type=float,default=.001)
    tr.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    ad=sub.add_parser('adopt',help='Wrap your existing supported torchvision state_dict (.pth)')
    ad.add_argument('--source',required=True);ad.add_argument('--checkpoint',default='models/classifier.pth')
    ad.add_argument('--arch',choices=['mobilenet_v3_small','resnet18','efficientnet_b0','vgg16'],required=True)
    ad.add_argument('--classes',nargs='+',required=True,help='Class folder names in exact classifier output order')
    ad.add_argument('--input-size',type=int,default=160)
    ad.add_argument('--resize-size',type=int,default=192)
    ex=sub.add_parser('export',help='Export to dynamic-batch FP32 ONNX')
    ex.add_argument('--checkpoint',default='models/classifier.pth')
    ex.add_argument('--output',default='models/classifier.fp32.onnx')
    ex.add_argument('--opset',type=int,default=17)
    ex.add_argument('--check-parity',action='store_true')
    qu=sub.add_parser('quantize',help='Use train-only images to calibrate QDQ static INT8')
    qu.add_argument('--checkpoint',default='models/classifier.pth')
    qu.add_argument('--onnx',default='models/classifier.fp32.onnx')
    qu.add_argument('--output',default='models/classifier.int8.onnx')
    qu.add_argument('--data-root',default='data/imagenette2-160')
    qu.add_argument('--calibration-count',type=int,default=64)
    qu.add_argument('--validation-check-count',type=int,default=0,help='0 checks all val images for duplicate content')
    qu.add_argument('--seed',type=int,default=42)
    qu.add_argument('--per-channel',action='store_true')
    bn=sub.add_parser('benchmark',help='Run each backend+batch in a fresh subprocess')
    bn.add_argument('--checkpoint',default='models/classifier.pth')
    bn.add_argument('--onnx',default='models/classifier.fp32.onnx')
    bn.add_argument('--int8',default='models/classifier.int8.onnx')
    bn.add_argument('--metadata',default='models/metadata.json')
    bn.add_argument('--data-root',default='data/imagenette2-160')
    bn.add_argument('--output-dir',default='results')
    bn.add_argument('--backends',nargs='+',default=['torch-cpu','onnx-cpu','onnx-int8-cpu'],choices=['torch-cpu','torch-cuda','onnx-cpu','onnx-int8-cpu'])
    bn.add_argument('--batch-sizes',nargs='+',type=int,default=[1,8])
    bn.add_argument('--max-eval',type=int,default=200)
    bn.add_argument('--warmup',type=int,default=10)
    bn.add_argument('--repeats',type=int,default=40)
    bn.add_argument('--threads',type=int,default=4)
    bn.add_argument('--seed',type=int,default=42)
    rep=sub.add_parser('report',help='Generate measured research report and comparison figure')
    rep.add_argument('--csv',default='results/benchmark.csv')
    rep.add_argument('--output',default='reports/measured_results.md')
    rep.add_argument('--plot',default='results/figures/cpu_latency.png')
    return p


def main():
    a=build_parser().parse_args()
    if a.command=='status':
        print('Platform:',platform.platform(),'Python:',sys.version.split()[0])
        for pkg in ('torch','torchvision','onnx','onnxruntime','psutil','numpy','PIL','pytest','streamlit'):
            print(f'  {pkg:14}', 'installed' if importlib.util.find_spec(pkg) else 'missing')
        try:
            import torch
            print('CUDA:',torch.cuda.is_available())
            if torch.cuda.is_available():print('GPU:',torch.cuda.get_device_name(0))
        except ImportError:pass
        return
    if a.command=='smoke-test':
        from .smoke import smoke_test
        print(json.dumps(smoke_test(a.output),indent=2));return
    if a.command=='download-imagenette':
        from .download import download_imagenette
        print(download_imagenette(a.output_root));return
    if a.command=='train':
        from .training import train
        from .models import load_checkpoint
        train(a.data_root,a.checkpoint,arch=a.arch,epochs=a.epochs,batch_size=a.batch_size,
              max_train=a.max_train,img_size=a.img_size,seed=a.seed,lr=a.lr,device=a.device)
        _,meta=load_checkpoint(a.checkpoint)
        write_json(Path(a.checkpoint).parent/'metadata.json',meta);print('Saved:',a.checkpoint);return
    if a.command=='adopt':
        from .models import adopt_checkpoint
        meta=adopt_checkpoint(a.source,a.checkpoint,a.arch,a.classes,a.input_size,a.resize_size)
        write_json(Path(a.checkpoint).parent/'metadata.json',meta)
        print('Adopted checkpoint:',a.checkpoint);return
    if a.command=='export':
        from .exporting import export_model,check_fp32_parity
        out=export_model(a.checkpoint,a.output,a.opset)
        if a.check_parity:out['max_abs_error']=check_fp32_parity(a.checkpoint,a.output)
        print(json.dumps(out,indent=2));return
    if a.command=='quantize':
        from .quantization import quantize_cnn
        stats=quantize_cnn(a.onnx,a.output,a.checkpoint,a.data_root,
                           a.calibration_count,a.validation_check_count,a.seed,a.per_channel)
        print(json.dumps(stats,indent=2));return
    if a.command=='benchmark':
        from .benchmark import benchmark_models
        rows=benchmark_models(a.checkpoint,a.onnx,a.int8,a.metadata,a.data_root,a.output_dir,
                              a.backends,a.batch_sizes,a.max_eval,a.warmup,a.repeats,a.threads,a.seed)
        for r in rows:
            print(r['backend'],r['batch_size'],r['top1_accuracy'],r['median_latency_ms'])
        return
    if a.command=='report':
        from .report import generate_report
        print(generate_report(a.csv,a.output,a.plot));return

if __name__=='__main__':main()
