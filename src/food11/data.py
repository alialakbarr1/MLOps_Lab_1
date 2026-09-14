"""Prepare the Food-11 dataset for ResNet training.

Reads the raw dataset from ``data/food11_raw`` where every split folder holds a
flat list of images named ``<class-index>_<counter>.jpg`` and writes two
ImageFolder-style datasets (the layout ``torchvision.datasets.ImageFolder`` and
therefore ResNet expects):

    data/food11_processed/<split>/<Category>/<image>.jpg
    data/food11_processed_mini/<split>/<Category>/<image>.jpg

Both copies are resized to 128x128; the ``_mini`` one keeps at most
``MINI_PER_CATEGORY`` images per category per split so experiments run fast.

Usage:
    uv run python ./src/food11/data.py
"""

from __future__ import annotations

import shutil
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

CATEGORIES = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
]

SPLITS = ["training", "evaluation", "validation"]

IMAGE_SIZE = (128, 128)
MINI_PER_CATEGORY = 100
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "food11_raw"
PROCESSED_DIR = DATA_DIR / "food11_processed"
MINI_DIR = DATA_DIR / "food11_processed_mini"


def category_of(image_path: Path) -> str | None:
    """Food-11 encodes the label as the part of the file name before the '_'."""
    label = image_path.stem.split("_")[0]
    if not label.isdigit():
        return None
    index = int(label)
    if index >= len(CATEGORIES):
        return None
    return CATEGORIES[index]


def iter_images(split_dir: Path):
    """Yield the images of a split, sorted so the run is reproducible."""
    for path in sorted(split_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def reset(directory: Path) -> None:
    """Rebuild from scratch so a rerun never leaves stale files behind."""
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)


def prepare_split(split: str) -> tuple[int, int, int]:
    raw_split = RAW_DIR / split
    if not raw_split.is_dir():
        print(f"  ! {raw_split} is missing, skipping")
        return 0, 0, 0

    for category in CATEGORIES:
        (PROCESSED_DIR / split / category).mkdir(parents=True, exist_ok=True)
        (MINI_DIR / split / category).mkdir(parents=True, exist_ok=True)

    kept = defaultdict(int)
    processed = mini = skipped = 0

    for image_path in iter_images(raw_split):
        category = category_of(image_path)
        if category is None:
            skipped += 1
            continue

        with Image.open(image_path) as image:
            resized = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
            target = PROCESSED_DIR / split / category / f"{image_path.stem}.jpg"
            resized.save(target, format="JPEG", quality=90)
            processed += 1

            if kept[category] < MINI_PER_CATEGORY:
                resized.save(
                    MINI_DIR / split / category / f"{image_path.stem}.jpg",
                    format="JPEG",
                    quality=90,
                )
                kept[category] += 1
                mini += 1

    return processed, mini, skipped


def main() -> int:
    if not RAW_DIR.is_dir():
        print(f"Raw dataset not found at {RAW_DIR}", file=sys.stderr)
        print(
            "Download Food-11 and unpack it as "
            "data/food11_raw/{training,evaluation,validation}",
            file=sys.stderr,
        )
        return 1

    reset(PROCESSED_DIR)
    reset(MINI_DIR)

    totals = [0, 0, 0]
    for split in SPLITS:
        print(f"Processing {split} ...")
        processed, mini, skipped = prepare_split(split)
        print(f"  {processed} images -> food11_processed, {mini} -> food11_processed_mini")
        if skipped:
            print(f"  {skipped} files skipped (unrecognised label)")
        totals = [a + b for a, b in zip(totals, (processed, mini, skipped))]

    print(f"\nDone. {totals[0]} processed, {totals[1]} in mini, {totals[2]} skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
