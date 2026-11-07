"""ImageFolder helpers and **identical** inference preprocessing for all runtimes.

All labels refer to classifier output indices. The training and held-out testing
folders must be separate, even if copying filenames would make paths distinct.
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

EXTS = {".jpeg", ".jpg", ".png", ".bmp", ".webp"}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def classes_in_directory(folder: str | Path) -> list[str]:
    p = Path(folder)
    if not p.is_dir():
        raise FileNotFoundError(f"Image folder not found: {p}")
    classes = sorted(d.name for d in p.iterdir() if d.is_dir())
    if len(classes) < 2:
        raise ValueError(f"Expected >=2 class directories in {p}; got {classes}")
    return classes


def list_images(root: str | Path, class_to_idx: dict[str, int]) -> list[dict]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    items = []
    for label, idx in sorted(class_to_idx.items()):
        folder = root / label
        if not folder.is_dir():
            raise FileNotFoundError(f"Expected class directory {folder}")
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.suffix.lower() in EXTS:
                items.append({"path": str(p.resolve()), "label": int(idx), "class_name": label})
    if not items:
        raise ValueError(f"No images found under {root}")
    return items


def stratified_limit(items: list[dict], max_count: int | None, seed: int) -> list[dict]:
    if max_count is None or max_count <= 0 or len(items) <= max_count:
        return list(items)
    rng = random.Random(seed)
    groups: dict[int, list[dict]] = {}
    for item in items:
        groups.setdefault(item["label"], []).append(item)
    # Round-robin across classes to avoid a sample accidentally containing one class only.
    for v in groups.values():
        rng.shuffle(v)
    selected = []
    while len(selected) < max_count and any(groups.values()):
        for klass in sorted(groups):
            if len(selected) >= max_count:
                break
            if groups[klass]:
                selected.append(groups[klass].pop())
    return sorted(selected, key=lambda x: x["path"])


def train_dev_split(items: list[dict], dev_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    if not 0 < dev_fraction < 0.5:
        raise ValueError("dev_fraction must be between 0 and 0.5")
    rng = random.Random(seed)
    groups: dict[int, list[dict]] = {}
    for item in items:
        groups.setdefault(item["label"], []).append(item)
    train, dev = [], []
    for _, group in sorted(groups.items()):
        if len(group) < 2:
            raise ValueError("Each class needs >=2 training images for a separate development set")
        copy = list(group)
        rng.shuffle(copy)
        n_dev = max(1, round(len(copy) * dev_fraction))
        dev.extend(copy[:n_dev])
        train.extend(copy[n_dev:])
    return train, dev


def resized_center_crop(image: Image.Image, size: int = 160, resize: int = 192) -> Image.Image:
    if resize < size:
        raise ValueError("resize must be >= crop size")
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    # Match torchvision.transforms.Resize(int): truncate the long edge.
    if w < h:
        w2, h2 = resize, int(resize * h / w)
    elif h < w:
        w2, h2 = int(resize * w / h), resize
    else:
        w2 = h2 = resize
    image = image.resize((w2, h2), resample=Image.Resampling.BILINEAR)
    # Match torchvision.transforms.CenterCrop rounding (including odd differences).
    left = int(round((w2 - size) / 2.0))
    top = int(round((h2 - size) / 2.0))
    return image.crop((left, top, left + size, top + size))


def preprocess(path: str | Path, size: int = 160, resize: int = 192) -> np.ndarray:
    with Image.open(path) as img:
        img = resized_center_crop(img, size, resize)
        array = np.asarray(img, dtype=np.float32) / 255.0
    return np.ascontiguousarray(((array - MEAN) / STD).transpose(2, 0, 1), dtype=np.float32)


def load_eval_arrays(items: list[dict], size: int, resize: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([preprocess(z["path"], size, resize) for z in items])
    y = np.array([z["label"] for z in items], dtype=np.int64)
    return x, y


def _digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_disjoint_by_content(first: list[dict], second: list[dict]) -> None:
    first_hashes = {_digest(z["path"]) for z in first}
    overlap = [z["path"] for z in second if _digest(z["path"]) in first_hashes]
    if overlap:
        raise ValueError(f"Calibration/evaluation image overlap detected: {overlap[:3]}")


def validate_dataset_class_folders(split_root: Path, class_to_idx: dict[str, int]) -> None:
    actual = set(classes_in_directory(split_root))
    expected = set(class_to_idx)
    if actual != expected:
        raise ValueError(f"Dataset label mismatch in {split_root}: expected {sorted(expected)}, found {sorted(actual)}")
