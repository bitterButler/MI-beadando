"""
COCO allat-osztalyozasi adathalmaz PyTorch Dataset kente.

Az allat kategoriak (COCO kategoriaazonosito -> nev):
    16: bird, 17: cat, 18: dog, 19: horse, 20: sheep,
    21: cow, 22: elephant, 23: bear, 24: zebra, 25: giraffe

Mivel a COCO detekcios adathalmaz (egy kepen tobb objektum is lehet),
az osztaly-cimket az adott kephez tartozo legnagyobb befoglalo terletu
allat-annotacio alapjan rendeljuk hozza.
"""

import os
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# ---------------------------------------------------------------------------
# Osztalydefiniciok
# ---------------------------------------------------------------------------

ANIMAL_CATEGORIES: dict[int, str] = {
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
}

# Rendezett kategoriaazonositok -> konzisztens index-eles
_SORTED_CAT_IDS: list[int] = sorted(ANIMAL_CATEGORIES.keys())
CAT_ID_TO_IDX: dict[int, int] = {
    cat_id: idx for idx, cat_id in enumerate(_SORTED_CAT_IDS)
}
IDX_TO_LABEL: dict[int, str] = {
    idx: ANIMAL_CATEGORIES[cat_id] for cat_id, idx in CAT_ID_TO_IDX.items()
}
CLASS_NAMES: list[str] = [IDX_TO_LABEL[i] for i in range(len(IDX_TO_LABEL))]
NUM_CLASSES: int = len(ANIMAL_CATEGORIES)

# ImageNet normalizacios parameterei (ResNet es sajat CNN is ezeket hasznalja)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Annotacio betoltese
# ---------------------------------------------------------------------------

def load_coco_samples(
    images_dir: str | Path,
    ann_file: str | Path,
) -> list[tuple[str, int]]:
    """
    Betolti a COCO annotaciokat es visszaad egy lista(kep_utvonal, osztaly_index)
    parokbol. Minden kephez a legnagyobb befoglalo terulet allat-annotacioja
    alapjan rendeljuk hozza az osztalyt.

    Args:
        images_dir: A val2017/ kepek mappaja.
        ann_file:   Az instances_val2017.json utvonala.

    Returns:
        Lista (abs_image_path, class_idx) tuple-okbol.
    """
    images_dir = Path(images_dir)
    ann_file = Path(ann_file)

    print(f"Annotaciok betoltese: {ann_file}")
    with open(ann_file, "r") as f:
        coco_data = json.load(f)

    # kep-id -> annotaciok lista
    img_to_anns: dict[int, list[dict]] = defaultdict(list)
    for ann in coco_data["annotations"]:
        if ann["category_id"] in ANIMAL_CATEGORIES:
            img_to_anns[ann["image_id"]].append(ann)

    # kep-id -> fajlnev
    img_id_to_fname: dict[int, str] = {
        img["id"]: img["file_name"] for img in coco_data["images"]
    }

    samples: list[tuple[str, int]] = []
    missing = 0

    for img_id, anns in img_to_anns.items():
        # Legjelentosebb annotacio: legnagyobb befoglalo terulet (szelesseg * magassag)
        best = max(anns, key=lambda a: a["bbox"][2] * a["bbox"][3])
        cat_id = best["category_id"]

        if cat_id not in CAT_ID_TO_IDX:
            continue

        img_path = images_dir / img_id_to_fname[img_id]
        if img_path.exists():
            samples.append((str(img_path), CAT_ID_TO_IDX[cat_id]))
        else:
            missing += 1

    if missing:
        print(f"Figyelem: {missing} kep nem talalhato a lemezen.")

    print(
        f"Betoltott mintak: {len(samples)} db, "
        f"{NUM_CLASSES} osztaly"
    )
    return samples


# ---------------------------------------------------------------------------
# Dataset osztaly
# ---------------------------------------------------------------------------

class COCOAnimalsDataset(Dataset):
    """
    PyTorch Dataset COCO allat-kepek osztalyozasahoz.

    Args:
        samples:   Lista (image_path, label_idx) tuple-okbol.
        transform: torchvision transzformacio pipeline.
    """

    def __init__(
        self,
        samples: list[tuple[str, int]],
        transform: T.Compose | None = None,
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# ---------------------------------------------------------------------------
# Transzformaciok
# ---------------------------------------------------------------------------

def get_train_transform() -> T.Compose:
    """Tanitas: data augmentaciokal (flip, crop, szinvaltozas, forgatas)."""
    return T.Compose([
        T.Resize((256, 256)),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomRotation(degrees=15),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def get_eval_transform() -> T.Compose:
    """Validacio / teszt: csak atmeretezas es normalizacio."""
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloaders(
    images_dir: str | Path,
    ann_file: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Letrehozza a tanito, validacios es teszt DataLoader-eket.

    A teljes adathalmazt veletlenszeruen osztja fel 70 / 15 / 15 aranyban.
    A tanito halmazon augmentaciok vannak bekapcsolva, a validacion
    es teszten nincsenek.

    Args:
        images_dir:  A kepeket tartalmazo mappa (val2017/).
        ann_file:    Az instances_val2017.json utvonala.
        batch_size:  Mini-batch meret.
        num_workers: DataLoader worker szamok (Windows-on 0 ajanlott).
        train_ratio: Tanito halmaz arannya (0.70 = 70%).
        val_ratio:   Validacios halmaz arannya (0.15 = 15%).
        seed:        Veletlenszam mag a reprodukalhato felosztashoz.

    Returns:
        (train_loader, val_loader, test_loader) tuple.
    """
    all_samples = load_coco_samples(images_dir, ann_file)

    rng = random.Random(seed)
    rng.shuffle(all_samples)

    n = len(all_samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_s = all_samples[:n_train]
    val_s = all_samples[n_train : n_train + n_val]
    test_s = all_samples[n_train + n_val :]

    print(
        f"Adathalmaz felosztasa: "
        f"{len(train_s)} tanito | {len(val_s)} validacios | {len(test_s)} teszt"
    )

    train_ds = COCOAnimalsDataset(train_s, transform=get_train_transform())
    val_ds = COCOAnimalsDataset(val_s, transform=get_eval_transform())
    test_ds = COCOAnimalsDataset(test_s, transform=get_eval_transform())

    loader_kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)

    return train_loader, val_loader, test_loader
