"""CPU-oriented post-training static QDQ INT8 quantization for CNNs."""
from __future__ import annotations

from pathlib import Path

from .dataset import assert_disjoint_by_content, list_images, stratified_limit, preprocess, validate_dataset_class_folders
from .models import load_checkpoint
from .utils import write_json, read_json, file_sha256, now_utc, mb


def quantize_cnn(onnx_path: str, output: str, checkpoint: str, data_root: str,
                 calibration_count: int = 64, validation_check_count: int = 0,
                 seed: int = 42, per_channel: bool = False) -> dict:
    try:
        import onnxruntime as ort
        from onnxruntime.quantization import (
            CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
        )
    except ImportError as exc:
        raise RuntimeError('Missing ONNX Runtime: python -m pip install -e ".[onnx]"') from exc
    _, meta=load_checkpoint(checkpoint)
    exported_meta_path=Path(onnx_path).with_suffix('.meta.json')
    if not exported_meta_path.is_file():
        raise FileNotFoundError(f'Missing ONNX export provenance: {exported_meta_path}')
    exported=read_json(exported_meta_path)
    if exported.get('checkpoint_sha256')!=file_sha256(checkpoint) or exported.get('onnx_sha256')!=file_sha256(onnx_path):
        raise ValueError('FP32 ONNX provenance differs from checkpoint; re-export the model')
    classes=meta['class_to_idx']
    root=Path(data_root)
    validate_dataset_class_folders(root/'train',classes)
    validate_dataset_class_folders(root/'val',classes)
    calib=stratified_limit(list_images(root/'train',classes),calibration_count,seed)
    validation=list_images(root/'val',classes)
    if validation_check_count > 0:
        validation=stratified_limit(validation,validation_check_count,seed)
    assert_disjoint_by_content(calib,validation)
    input_name=ort.InferenceSession(str(onnx_path),providers=['CPUExecutionProvider']).get_inputs()[0].name
    size=int(meta['input_size']);resize=int(meta['resize_size'])

    class Reader(CalibrationDataReader):
        def __init__(self):self.i=0
        def get_next(self):
            if self.i>=len(calib):return None
            path=calib[self.i]['path'];self.i+=1
            return {input_name:preprocess(path,size,resize)[None,...]}
        def rewind(self):self.i=0

    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    quantize_static(model_input=str(onnx_path),model_output=str(out),calibration_data_reader=Reader(),
                    quant_format=QuantFormat.QDQ,activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8,per_channel=per_channel,
                    calibrate_method=CalibrationMethod.MinMax,
                    op_types_to_quantize=['Conv','MatMul','Gemm'])
    # Fail here if the target CPU runtime cannot load the quantized graph.
    sess=ort.InferenceSession(str(out),providers=['CPUExecutionProvider'])
    first=preprocess(calib[0]['path'],size,resize)[None,...]
    sess.run(None,{sess.get_inputs()[0].name:first})
    stats={"method":"static post-training QDQ (Conv/MatMul/Gemm)","quantized_file":str(out.resolve()),
           "file_mb":mb(out.stat().st_size),"sha256":file_sha256(out),
           "float_onnx_sha256":file_sha256(onnx_path),"calibration_n":len(calib),
           "calibration_source":"train_only", "calibration_seed":seed,"evaluated_disjoint_validation_count":len(validation),
           "activation_type":"QUInt8","weight_type":"QInt8","per_channel":per_channel,
           "input_name":input_name,"timestamp":now_utc()}
    write_json(out.with_suffix('.meta.json'),stats)
    return stats
