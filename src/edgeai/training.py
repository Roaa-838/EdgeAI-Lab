"""Train from scratch using train/, select best checkpoint using train-only dev.

The external val/ split is NEVER read during training or checkpoint selection.
"""
from __future__ import annotations

import time
from pathlib import Path

from .dataset import classes_in_directory, list_images, stratified_limit, train_dev_split
from .models import create_model
from .utils import now_utc, seed_everything, write_json


def train(data_root: str, checkpoint: str, arch: str = "mobilenet_v3_small", epochs: int = 8,
          batch_size: int = 32, max_train: int = 0, img_size: int = 160,
          seed: int = 42, lr: float = 0.001, device: str = "auto", num_workers: int = 0) -> list[dict]:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from PIL import Image, ImageOps
    seed_everything(seed)
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    data_root = Path(data_root)
    classes = classes_in_directory(data_root / "train")
    class_to_idx = {name: i for i, name in enumerate(classes)}
    samples = stratified_limit(list_images(data_root / "train", class_to_idx), max_train, seed)
    train_items, dev_items = train_dev_split(samples, 0.1, seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected but torch.cuda.is_available() is false")
    norm = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    training_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size), transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm
    ])
    resize_size = round(img_size * 1.2)
    eval_transform = transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(img_size), transforms.ToTensor(), norm
    ])
    class ImageList(Dataset):
        def __init__(self, records, transform):
            self.records, self.transform = records, transform
        def __len__(self):return len(self.records)
        def __getitem__(self, i):
            with Image.open(self.records[i]["path"]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            return self.transform(image), self.records[i]["label"]
    train_loader = DataLoader(ImageList(train_items, training_transform), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dev_loader = DataLoader(ImageList(dev_items, eval_transform), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model = create_model(arch, len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    history=[]; best=-1.0
    dest = Path(checkpoint); dest.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,epochs+1):
        t0=time.perf_counter(); model.train(); losses=[]
        for x,y in train_loader:
            x,y=x.to(device),y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred=model(x); loss=criterion(pred,y); loss.backward(); optimizer.step()
            losses.append(float(loss.item()))
        model.eval(); correct=0; n=0
        with torch.inference_mode():
            for x,y in dev_loader:
                pred=model(x.to(device)).argmax(dim=1).cpu()
                correct+=int((pred==y).sum().item());n+=len(y)
        dev_acc=correct/n
        item={"epoch":epoch,"train_loss":round(sum(losses)/len(losses),5),"development_top1":round(dev_acc,5),"epoch_seconds":round(time.perf_counter()-t0,1)}
        history.append(item)
        print(f"epoch={epoch:02d} train_loss={item['train_loss']:.4f} dev_top1={dev_acc:.4f}",flush=True)
        if dev_acc>best:
            best=dev_acc
            meta={"arch":arch,"num_classes":len(classes),"class_to_idx":class_to_idx,
                  "input_size":img_size,"resize_size":resize_size,"training_dataset":"imagenette_or_user_imagefolder_train_only",
                  "validation_dataset":"external_val_held_out","seed":seed,"created_at":now_utc(),
                  "train_size":len(train_items),"dev_size":len(dev_items),
                  "training_origin":"random_initialization","selected_dev_accuracy":dev_acc,
                  "note":"Official external val split not used for training or checkpoint selection."}
            torch.save({"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"metadata":meta},dest)
    write_json(dest.parent / "training_history.json",history)
    return history
