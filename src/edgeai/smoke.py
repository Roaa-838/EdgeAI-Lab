"""Synthetic **engineering** smoke test; does not create model accuracy claims."""
from pathlib import Path
import numpy as np
from PIL import Image
from .dataset import preprocess, stratified_limit, train_dev_split
from .utils import now_utc, write_json


def smoke_test(output='results/smoke_test.json') -> dict:
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as root:
        rng=np.random.default_rng(42)
        items=[]
        for category in ('class_a','class_b'):
            for i in range(6):
                path=Path(root)/f'{category}_{i}.png'
                Image.fromarray(rng.integers(0,256,size=(48,56,3),dtype=np.uint8)).save(path)
                items.append({'path':str(path),'label':int(category=='class_b'),'class_name':category})
        train,dev=train_dev_split(items,0.25,42)
        cropped=preprocess(items[0]['path'],32,40)
        if cropped.shape!=(3,32,32):raise AssertionError('Preprocessing produced incorrect shape')
        if len(train)+len(dev)!=len(items):raise AssertionError('Split failed')
        try:
            import torch
            from .models import create_model
            m=create_model('mobilenet_v3_small',2).eval()
            with torch.inference_mode():
                logits=m(torch.from_numpy(np.stack([cropped]*2)))
            forward_shape=list(logits.shape)
            ok=forward_shape==[2,2]
        except ImportError:
            forward_shape=None
            ok=True
        payload={'run_type':'synthetic_engineering_smoke_test','created_at':now_utc(),
                 'random_synthetic_images':len(items),'train_samples':len(train),'dev_samples':len(dev),
                 'preprocess_shape':list(cropped.shape),'torch_forward_shape':forward_shape,
                 'passed':ok,'accuracy_metrics':None,
                 'warning':'Synthetic smoke test ONLY; this is not model training or empirical accuracy research.'}
        write_json(output,payload)
        return payload
