from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from edgeai.dataset import (classes_in_directory,list_images,stratified_limit,train_dev_split,
                            preprocess,assert_disjoint_by_content,resized_center_crop)


def test_class_names(tiny_data):
    assert classes_in_directory(tiny_data/'train')==['apple','orange']

def test_list_images(tiny_data):
    items=list_images(tiny_data/'train',{'apple':0,'orange':1})
    assert len(items)==8 and {z['label'] for z in items}=={0,1}

def test_stratified_limit_balanced(tiny_data):
    items=list_images(tiny_data/'train',{'apple':0,'orange':1})
    x=stratified_limit(items,4,42)
    assert len(x)==4 and sum(z['label']==0 for z in x)==2

def test_stratified_limit_deterministic(tiny_data):
    items=list_images(tiny_data/'train',{'apple':0,'orange':1})
    assert stratified_limit(items,4,42)==stratified_limit(items,4,42)

def test_train_dev_split(tiny_data):
    items=list_images(tiny_data/'train',{'apple':0,'orange':1})
    train,dev=train_dev_split(items,0.25,42)
    assert len(train)==6 and len(dev)==2
    assert not ({z['path'] for z in train}&{z['path'] for z in dev})

def test_invalid_train_dev_fraction():
    with pytest.raises(ValueError):train_dev_split([],0,42)

def test_preprocess_shape_dtype(tiny_data):
    x=preprocess(tiny_data/'val'/'apple'/'00.png',32,40)
    assert x.shape==(3,32,32) and x.dtype==np.float32 and np.isfinite(x).all()

def test_preprocess_deterministic(tiny_data):
    p=tiny_data/'val'/'apple'/'00.png'
    assert np.array_equal(preprocess(p,32,40),preprocess(p,32,40))

def test_resize_guard():
    with pytest.raises(ValueError):resized_center_crop(Image.new('RGB',(20,20)),32,24)

def test_content_leakage_detected(tiny_data):
    items=list_images(tiny_data/'train',{'apple':0,'orange':1})
    with pytest.raises(ValueError,match='overlap'):
        assert_disjoint_by_content(items,items[:1])

def test_content_leakage_copy_detected(tiny_data,tmp_path):
    a=list_images(tiny_data/'train',{'apple':0,'orange':1})
    copied=tmp_path/'copied.png'
    copied.write_bytes(Path(a[0]['path']).read_bytes())
    with pytest.raises(ValueError,match='overlap'):
        assert_disjoint_by_content(a,[{'path':str(copied),'label':0}])

def test_distinct_contents(tiny_data):
    a=list_images(tiny_data/'train',{'apple':0,'orange':1})
    b=list_images(tiny_data/'val',{'apple':0,'orange':1})
    # The generated test fixture has the same bytes in train and val, so a
    # robust content guard must reject even though filenames are different.
    with pytest.raises(ValueError):assert_disjoint_by_content(a,b)


def test_preprocessing_matches_torchvision(tiny_data):
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    with Image.open(tiny_data/'val'/'apple'/'00.png') as im:
        tv=transforms.Compose([
            transforms.Resize(40,interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(32),transforms.ToTensor(),
            transforms.Normalize((.485,.456,.406),(.229,.224,.225))
        ])(im.convert('RGB')).numpy()
    ours=preprocess(tiny_data/'val'/'apple'/'00.png',32,40)
    assert np.max(np.abs(tv-ours))<1e-5
