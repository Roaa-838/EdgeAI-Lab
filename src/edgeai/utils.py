from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: str | Path, payload: dict | list) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replace avoids partially written results.
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=p.parent, suffix=".tmp", delete=False) as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        temp = f.name
    os.replace(temp, p)


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def mb(bytes_count: int) -> float:
    return round(bytes_count / (1024 * 1024), 4)
