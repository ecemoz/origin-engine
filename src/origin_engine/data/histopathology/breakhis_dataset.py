import os
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class BreakHisDataset(Dataset):

    VALID_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict[str, Any]]:
        samples = []

        breast_root = self.root / "histology_slides" / "breast"

        for label_name in ["benign", "malignant"]:
            label_dir = breast_root / label_name
            if not label_dir.exists():
                continue

            sob_dir = label_dir / "SOB"
            if not sob_dir.exists():
                continue


            for subtype in os.listdir(sob_dir):
                subtype_dir = sob_dir / subtype
                if not subtype_dir.is_dir():
                    continue


                for slide_id in os.listdir(subtype_dir):
                    slide_dir = subtype_dir / slide_id
                    if not slide_dir.is_dir():
                        continue


                    for magnification in os.listdir(slide_dir):
                        mag_dir = slide_dir / magnification
                        if not mag_dir.is_dir():
                            continue


                        for fname in os.listdir(mag_dir):
                            if fname.lower().endswith(self.VALID_EXT):
                                samples.append({
                                    "path": str(mag_dir / fname),
                                    "label": 0 if label_name == "benign" else 1,
                                    "subtype": subtype,
                                    "slide_id": slide_id,
                                    "magnification": magnification
                                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        img = Image.open(item["path"]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return {
            "image": img,
            "label": item["label"],
            "subtype": item["subtype"],
            "slide_id": item["slide_id"],
            "magnification": item["magnification"],
            "path": item["path"]
        }

def default_breakhis_transform(img_size=224):

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
