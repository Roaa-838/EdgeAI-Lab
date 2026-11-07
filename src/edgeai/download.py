"""User-invoked Imagenette download with archive-path safety checks.

No dataset or pretrained weights are silently downloaded.
"""
from pathlib import Path
import tarfile
import urllib.request

IMAGENETTE_URL='https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz'


def download_imagenette(output_root: str = 'data', url: str = IMAGENETTE_URL) -> Path:
    root=Path(output_root)
    root.mkdir(parents=True,exist_ok=True)
    archive=root/'imagenette2-160.tgz'
    if not archive.exists():
        print(f'Downloading {url} -> {archive}',flush=True)
        urllib.request.urlretrieve(url, str(archive))
    expected=root/'imagenette2-160'
    if expected.is_dir() and (expected/'train').exists() and (expected/'val').exists():
        return expected
    with tarfile.open(archive,'r:gz') as tf:
        for member in tf.getmembers():
            candidate=(root/member.name).resolve()
            if not candidate.is_relative_to(root.resolve()):
                raise ValueError(f'Unsafe archive path: {member.name}')
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f'Unsafe archive entry: {member.name}')
        tf.extractall(root,filter='data')
    if not (expected/'train').is_dir() or not (expected/'val').is_dir():
        raise RuntimeError('Archive did not create expected train and val folders')
    return expected
