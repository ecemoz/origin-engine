import os
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class LC25000Dataset(Dataset):

    VALID_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    LABEL_MAP = {
        "colon_aca": 1,
        "colon_n": 0,
        "lung_aca": 1,
        "lung_scc": 1,
        "lung_n": 0,
    }

    ORGAN_MAP = {
        "colon_aca": "colon",
        "colon_n": "colon",
        "lung_aca": "lung",
        "lung_scc": "lung",
        "lung_n": "lung",
    }

    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict[str, Any]]:
        samples = []

        organ_dirs = ["colon_image_sets", "lung_image_sets"]

        for organ_dir in organ_dirs:
            organ_path = self.root / organ_dir
            if not organ_path.exists():
                continue

            for class_name in os.listdir(organ_path):
                class_dir = organ_path / class_name
                if not class_dir.is_dir():
                    continue

                label = self.LABEL_MAP[class_name]
                organ = self.ORGAN_MAP[class_name]

                for fname in os.listdir(class_dir):
                    if fname.lower().endswith(self.VALID_EXT):
                        samples.append({
                            "path": str(class_dir / fname),
                            "label": label,
                            "organ": organ,
                            "class": class_name,
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
            "organ": item["organ"],
            "class": item["class"],
            "path": item["path"],
        }


def default_lc25000_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
