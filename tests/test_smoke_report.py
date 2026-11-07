import csv
import json
import pytest
from pathlib import Path
from edgeai.smoke import smoke_test
from edgeai.report import generate_report
from edgeai.utils import write_json,read_json,file_sha256
from edgeai.benchmark import verify_provenance


def test_smoke(tmp_path):
    output=tmp_path/'smoke.json'
    result=smoke_test(output)
    assert output.exists() and result['passed']
    assert result['run_type']=='synthetic_engineering_smoke_test'
    assert result['accuracy_metrics'] is None

def test_json_atomic(tmp_path):
    p=tmp_path/'test.json';write_json(p,{'a':[1,2]})
    assert read_json(p)=={'a':[1,2]}
    assert len(file_sha256(p))==64

def test_no_benchmark_csv(tmp_path):
    with pytest.raises(FileNotFoundError):generate_report(str(tmp_path/'none.csv'),str(tmp_path/'report.md'))

def test_report_rejects_synthetic(tmp_path):
    src=tmp_path/'benchmark.csv';src.write_text('run_type,backend,device,batch_size\nsynthetic_engineering_smoke_test,mock,cpu,1\n')
    with pytest.raises(ValueError,match='Only actual'):generate_report(str(src),str(tmp_path/'report.md'))

def test_report_real_measured(tmp_path):
    src=tmp_path/'benchmark.csv'
    src.write_text('run_type,backend,device,batch_size,n_eval,top1_accuracy,median_latency_ms\nreal_labeled_evaluation,torch-cpu,cpu,1,10,0.5,12.0\n')
    report=generate_report(str(src),str(tmp_path/'report.md'))
    assert '0.5' in report and (tmp_path/'report.md').exists()

def test_provenance_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):verify_provenance('nothing.pth',str(tmp_path/'float.onnx'),str(tmp_path/'int8.onnx'),['onnx-cpu'])
