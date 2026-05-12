"""
Download COCO 2017 val split into data/raw/.
Run once: python scripts/download_coco.py
"""

import zipfile
import shutil
from pathlib import Path
import urllib.request

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def download(url: str, dest: Path) -> None:
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def _progress(count, block_size, total_size):
    pct = min(count * block_size * 100 / total_size, 100)
    print(f"\r  {pct:.1f}%", end="", flush=True)


def main():
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    images_zip = DATA_RAW / "val2017.zip"
    annotations_zip = DATA_RAW / "annotations_trainval2017.zip"

    if not images_zip.exists():
        download(IMAGES_URL, images_zip)
    else:
        print("val2017.zip already present, skipping download.")

    if not annotations_zip.exists():
        download(ANNOTATIONS_URL, annotations_zip)
    else:
        print("annotations_trainval2017.zip already present, skipping download.")

    print("Extracting images ...")
    with zipfile.ZipFile(images_zip) as z:
        z.extractall(DATA_RAW)

    print("Extracting annotations ...")
    with zipfile.ZipFile(annotations_zip) as z:
        z.extractall(DATA_RAW)

    image_count = len(list((DATA_RAW / "val2017").glob("*.jpg")))
    print(f"Done. {image_count} images in {DATA_RAW / 'val2017'}")


if __name__ == "__main__":
    main()
