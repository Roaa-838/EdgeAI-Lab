"""Export a real trained checkpoint; verify ONNX graph (optional ORT parity)."""
from __future__ import annotations

from pathlib import Path

from .models import load_checkpoint
from .utils import file_sha256, write_json, now_utc, mb


def export_model(checkpoint: str, output: str, opset: int = 17) -> dict:
    try:
        import torch
        import onnx
    except ImportError as exc:
        raise RuntimeError('Install export dependencies: python -m pip install -e ".[full]"') from exc
    model,meta = load_checkpoint(checkpoint)
    size=int(meta["input_size"])
    example=torch.zeros((1,3,size,size),dtype=torch.float32)
    out=Path(output)
    out.parent.mkdir(parents=True,exist_ok=True)
    # Legacy exporter is used intentionally for dynamic batch portability across
    # torch 2.3–2.6; the modern dynamo exporter is recommended upstream.
    torch.onnx.export(model,example,str(out), export_params=True,do_constant_folding=True,
                      input_names=["images"],output_names=["logits"],opset_version=opset,
                      dynamic_axes={"images":{0:"batch"},"logits":{0:"batch"}},dynamo=False)
    onnx.checker.check_model(str(out))
    payload={"checkpoint":str(Path(checkpoint).resolve()),"checkpoint_sha256":file_sha256(checkpoint),
             "onnx":str(out.resolve()),"onnx_sha256":file_sha256(out),"onnx_mb":mb(out.stat().st_size),
             "input_name":"images","output_name":"logits","opset":opset,
             "arch":meta["arch"],"num_classes":meta["num_classes"],
             "input_size":meta["input_size"],"resize_size":meta["resize_size"],
             "exporter":"torch.onnx.export legacy with dynamic_axes", "timestamp":now_utc()}
    write_json(out.with_suffix('.meta.json'),payload)
    return payload


def check_fp32_parity(checkpoint: str, onnx_path: str, atol: float = 1e-3):
    import numpy as np
    import torch
    import onnxruntime as ort
    model,meta=load_checkpoint(checkpoint)
    session=ort.InferenceSession(onnx_path,providers=["CPUExecutionProvider"])
    rng=np.random.default_rng(11)
    x=rng.normal(size=(2,3,int(meta['input_size']),int(meta['input_size']))).astype(np.float32)
    with torch.inference_mode():
        p=model(torch.from_numpy(x)).numpy()
    o=session.run(None,{session.get_inputs()[0].name:x})[0]
    err=float(np.max(np.abs(p-o)))
    if err>atol:
        raise ValueError(f"FP32 export disagrees with PyTorch: max_abs_error={err:.5g} > {atol}")
    return err
