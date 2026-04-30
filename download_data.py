"""
COCO 2017 val adathalmaz letöltese – allatos kepek kiszurese.

Hasznalat:
    python download_data.py --data-dir ./data
    python download_data.py --data-dir ./data --max-samples 500
"""

import os
import json
import zipfile
import argparse
import time
from pathlib import Path

import requests
from tqdm import tqdm

ANNOTATIONS_ZIP_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
COCO_VAL_IMAGE_BASE = "http://images.cocodataset.org/val2017"

# COCO kategoriazonositok az allat osztalyokhoz
ANIMAL_CATEGORY_IDS = {16, 17, 18, 19, 20, 21, 22, 23, 24, 25}


def download_file(url: str, dest_path: str, description: str = "Letoltes") -> None:
    """Fajl letoltese haladassavval."""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(
        desc=description, total=total, unit="B", unit_scale=True, unit_divisor=1024
    ) as pbar:
        for chunk in response.iter_content(chunk_size=65536):
            f.write(chunk)
            pbar.update(len(chunk))


def ensure_annotations(data_dir: Path) -> Path:
    """Letolti es kicsomagolja az annotaciokat, ha meg nem leteznek."""
    ann_dir = data_dir / "annotations"
    ann_file = ann_dir / "instances_val2017.json"

    if ann_file.exists():
        print(f"Annotaciok mar elerhetek: {ann_file}")
        return ann_file

    ann_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "annotations_trainval2017.zip"

    if not zip_path.exists():
        print("COCO 2017 annotaciok letoltese (~250 MB)...")
        download_file(str(ANNOTATIONS_ZIP_URL), str(zip_path), "Annotaciok")

    print("Kicsomagolas...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Csak az instances_val2017.json kell
        target = "annotations/instances_val2017.json"
        if target in zf.namelist():
            zf.extract(target, str(data_dir))
        else:
            zf.extractall(str(data_dir))

    zip_path.unlink()
    print(f"Annotaciok mentve: {ann_file}")
    return ann_file


def find_animal_images(ann_file: Path) -> list[dict]:
    """
    Visszaadja azoknak a kepeknek az adatait, amelyek legalabb egy
    allat kategoriat tartalmaznak.
    """
    print(f"\nAnnotaciok elemzese: {ann_file}")
    with open(ann_file, "r") as f:
        coco = json.load(f)

    # Kepek, amelyek allat annotaciot tartalmaznak
    animal_img_ids: set[int] = set()
    for ann in coco["annotations"]:
        if ann["category_id"] in ANIMAL_CATEGORY_IDS:
            animal_img_ids.add(ann["image_id"])

    # id -> kepinformacio
    id_to_info = {img["id"]: img for img in coco["images"]}

    images = [
        id_to_info[img_id]
        for img_id in animal_img_ids
        if img_id in id_to_info
    ]
    print(f"Talalt allatos kepek: {len(images)} db")
    return images


def download_images(
    images: list[dict],
    img_dir: Path,
    max_samples: int | None = None,
) -> None:
    """Letolti a hianyzó COCO val2017 kepeket."""
    img_dir.mkdir(parents=True, exist_ok=True)

    if max_samples is not None:
        images = images[:max_samples]
        print(f"Korlat alkalmazva: legfeljebb {max_samples} kep")

    missing = [img for img in images if not (img_dir / img["file_name"]).exists()]

    if not missing:
        print(f"Minden kep mar elerheto ({len(images)} db).")
        return

    already = len(images) - len(missing)
    print(f"Letoltendo: {len(missing)} kep  (mar megvan: {already} db)")

    failed = []
    for img_info in tqdm(missing, desc="Kepek letoltese", unit="kep"):
        filename = img_info["file_name"]
        url = f"{COCO_VAL_IMAGE_BASE}/{filename}"
        dest = img_dir / filename
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(dest, "wb") as f:
                f.write(r.content)
        except Exception as exc:
            failed.append((filename, str(exc)))
        # Kismerteku keses a CDN terhelts mersekelese erdekeben
        time.sleep(0.02)

    if failed:
        print(f"\nFigyelem: {len(failed)} kep letoltese sikertelen volt.")

    total_ok = len(images) - len(failed)
    print(f"\nLetoltes befejezve. Osszes elerheto kep: {total_ok} db")


def download_coco_animals(data_dir: str = "./data", max_samples: int | None = None) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    ann_file = ensure_annotations(data_dir)
    images = find_animal_images(ann_file)
    img_dir = data_dir / "val2017"
    download_images(images, img_dir, max_samples=max_samples)

    print(f"\nKesz! Adatok elerheto helye:")
    print(f"  Kepek:       {img_dir}")
    print(f"  Annotaciok: {ann_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COCO allatos kepek letoltese")
    parser.add_argument("--data-dir", default="./data", help="Adatok mentesi mappaja")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Legfeljebb ennyi kepet toltsunk le (tesztelashez hasznos)",
    )
    args = parser.parse_args()
    download_coco_animals(args.data_dir, args.max_samples)
