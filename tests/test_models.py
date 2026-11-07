import pytest
from edgeai.models import create_model, load_checkpoint, adopt_checkpoint


def test_unknown_arch():
    with pytest.raises(ValueError,match='Unsupported'):create_model('fake',2)

def test_mobilenet_forward():
    torch=pytest.importorskip('torch')
    model=create_model('mobilenet_v3_small',3).eval()
    with torch.inference_mode():
        output=model(torch.zeros((1,3,64,64)))
    assert tuple(output.shape)==(1,3)

def test_model_checkpoint_roundtrip(tmp_path):
    torch=pytest.importorskip('torch')
    source=tmp_path/'model.pth';output=tmp_path/'adopted.pth'
    model=create_model('resnet18',2)
    torch.save(model.state_dict(),source)
    adopt_checkpoint(str(source),str(output),'resnet18',['a','b'],64,80)
    restored,meta=load_checkpoint(output)
    assert meta['class_to_idx']=={'a':0,'b':1}
    assert meta['input_size']==64
    assert restored.fc.out_features==2

def test_missing_metadata(tmp_path):
    torch=pytest.importorskip('torch')
    path=tmp_path/'broken.pth';torch.save({'state_dict':{}},path)
    with pytest.raises(ValueError,match='metadata'):load_checkpoint(path)
