from pathlib import Path
import numpy as np
import pytest
from PIL import Image

@pytest.fixture
def tiny_data(tmp_path: Path):
    for split in ('train','val'):
        for name in ('apple','orange'):
            folder=tmp_path/split/name
            folder.mkdir(parents=True)
            for i in range(4):
                color=(i*30, 40 if name=='apple' else 140, 120)
                Image.fromarray(np.full((40,52,3),color,dtype=np.uint8)).save(folder/f'{i:02}.png')
    return tmp_path
