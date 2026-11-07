"""Supported torchvision architectures and safe state-dict checkpoint loading."""
from __future__ import annotations

from pathlib import Path

ARCHITECTURES = ("mobilenet_v3_small", "resnet18", "efficientnet_b0", "vgg16")


def create_model(arch: str, num_classes: int):
    import torch.nn as nn
    import torchvision.models as models
    if arch == "mobilenet_v3_small":
        return models.mobilenet_v3_small(weights=None, num_classes=num_classes)
    if arch == "resnet18":
        return models.resnet18(weights=None, num_classes=num_classes)
    if arch == "efficientnet_b0":
        return models.efficientnet_b0(weights=None, num_classes=num_classes)
    if arch == "vgg16":
        return models.vgg16(weights=None, num_classes=num_classes)
    raise ValueError(f"Unsupported architecture {arch!r}. Choose from {ARCHITECTURES}")


def load_checkpoint(path: str | Path):
    import torch
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a dictionary checkpoint with state_dict and metadata")
    if "state_dict" not in checkpoint or "metadata" not in checkpoint:
        raise ValueError("Checkpoint missing 'state_dict' or 'metadata'. Use 'edgeai adopt' for a raw state_dict")
    metadata = checkpoint["metadata"]
    for field in ("arch", "num_classes", "class_to_idx", "input_size", "resize_size"):
        if field not in metadata:
            raise ValueError(f"Checkpoint metadata missing {field}")
    model = create_model(metadata["arch"], int(metadata["num_classes"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, metadata


def adopt_checkpoint(source: str, destination: str, arch: str, class_names: list[str], input_size: int = 160, resize_size: int = 192):
    """Wrap a trusted local *torchvision state_dict* with full evaluation provenance.

    Never load pickled modules or run arbitrary checkpoint code.
    """
    import torch
    original = torch.load(source, map_location="cpu", weights_only=True)
    weights = original.get("state_dict", original.get("model_state_dict", original)) if isinstance(original, dict) else None
    if not isinstance(weights, dict):
        raise ValueError("A state_dict (mapping of tensor names to tensors) is required")
    weights = {k.removeprefix("module."): v for k, v in weights.items()}
    model = create_model(arch, len(class_names))
    model.load_state_dict(weights, strict=True)
    meta = {
        "arch": arch, "num_classes": len(class_names),
        "class_to_idx": {n:i for i,n in enumerate(class_names)},
        "input_size": input_size, "resize_size": resize_size,
        "checkpoint_origin": "imported_user_model", "training_dataset": "provided_by_user",
        "validation_dataset": "external_held_out_required",
        "notes": "Confirm preprocessing matches original model training before evaluation.",
    }
    p = Path(destination); p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict":model.state_dict(), "metadata":meta}, str(p))
    return meta
