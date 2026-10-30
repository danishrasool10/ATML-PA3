"""COCO 2014 / VQAv2 assets: use copies attached under /kaggle/input when present, else download
from the official hosts (with retries).  Only the images that are actually selected are fetched."""
from __future__ import annotations

import io
import os
import re
import shutil
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

COCO_ANN_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
VQA_BASE = "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa"
FILES = {
    # name of the json -> zip url that contains it
    "captions_train2014.json": COCO_ANN_ZIP,
    "captions_val2014.json": COCO_ANN_ZIP,
    "instances_train2014.json": COCO_ANN_ZIP,
    "instances_val2014.json": COCO_ANN_ZIP,
    "v2_OpenEnded_mscoco_train2014_questions.json": f"{VQA_BASE}/v2_Questions_Train_mscoco.zip",
    "v2_OpenEnded_mscoco_val2014_questions.json": f"{VQA_BASE}/v2_Questions_Val_mscoco.zip",
    "v2_mscoco_train2014_annotations.json": f"{VQA_BASE}/v2_Annotations_Train_mscoco.zip",
    "v2_mscoco_val2014_annotations.json": f"{VQA_BASE}/v2_Annotations_Val_mscoco.zip",
    "v2_mscoco_train2014_complementary_pairs.json": f"{VQA_BASE}/v2_Complementary_Pairs_Train_mscoco.zip",
    "v2_mscoco_val2014_complementary_pairs.json": f"{VQA_BASE}/v2_Complementary_Pairs_Val_mscoco.zip",
}
IMG_URL = "http://images.cocodataset.org/{split}/COCO_{split}_{id:012d}.jpg"
INPUT_ROOTS = [p for p in os.environ.get("PA3_INPUT_ROOTS", "/kaggle/input").split(os.pathsep) if p]
UA = {"User-Agent": "Mozilla/5.0 (pa3-task2 data prep)"}


def _http_get(url: str, tries: int = 5, timeout: float = 120) -> bytes:
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception as ex:  # noqa: BLE001
            last = ex
            time.sleep(min(60, 5 * 2 ** i))
    raise RuntimeError(f"download failed after {tries} tries: {url} ({last})")


def _download_to(url: str, dst: Path, tries: int = 5) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, open(tmp, "wb") as fh:
                shutil.copyfileobj(r, fh, 1 << 22)
            os.replace(tmp, dst)
            return dst
        except Exception as ex:  # noqa: BLE001
            last = ex
            print(f"[download] {url}: {ex}; retry {i + 1}/{tries}", flush=True)
            time.sleep(min(60, 5 * 2 ** i))
    raise RuntimeError(f"download failed: {url} ({last})")


def find_attached(name: str) -> Path | None:
    for root in INPUT_ROOTS:
        r = Path(root)
        if r.exists():
            for p in r.rglob(name):
                if p.is_file():
                    return p
    return None


def ensure_annotation(name: str, raw_dir: Path) -> Path:
    """Path to a json (attached copy, cached copy, or extracted from the official zip)."""
    local = raw_dir / name
    if local.exists():
        return local
    att = find_attached(name)
    if att is not None:
        print(f"[data] using attached {att}", flush=True)
        return att
    url = FILES[name]
    zpath = raw_dir / "zips" / url.rsplit("/", 1)[1]
    if not zpath.exists():
        print(f"[data] downloading {url}", flush=True)
        _download_to(url, zpath)
    with zipfile.ZipFile(zpath) as z:
        member = next(m for m in z.namelist() if m.endswith(name))
        local.parent.mkdir(parents=True, exist_ok=True)
        with z.open(member) as src, open(local, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 22)
    return local


# ------------------------------------------------------------------------------------------- images
_ID = re.compile(r"(\d{6,12})\.jpe?g$", re.IGNORECASE)


def attached_image_index(split: str) -> dict[int, Path]:
    """image_id -> path for any attached directory named like the split (train2014 / val2014)."""
    idx: dict[int, Path] = {}
    for root in INPUT_ROOTS:
        r = Path(root)
        if not r.exists():
            continue
        for d in r.rglob(split):
            if d.is_dir():
                for f in d.iterdir():
                    m = _ID.search(f.name)
                    if m:
                        idx.setdefault(int(m.group(1)), f)
    return idx


def image_path(img_dir: Path, key: str) -> Path:
    split, i = key.split(":")
    return img_dir / split / f"COCO_{split}_{int(i):012d}.jpg"


def ensure_images(keys: list[str], img_dir: Path, workers: int = 32) -> dict:
    """Make every image in `keys` ("train2014:123") available under img_dir (copy/link attached or download)."""
    from PIL import Image

    t0 = time.time()
    att = {s: attached_image_index(s) for s in ("train2014", "val2014")}
    todo, n_att = [], 0
    for k in keys:
        dst = image_path(img_dir, k)
        if dst.exists():
            continue
        split, i = k.split(":")
        src = att[split].get(int(i))
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src is not None:
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            n_att += 1
        else:
            todo.append(k)

    def fetch(k: str) -> str | None:
        split, i = k.split(":")
        dst = image_path(img_dir, k)
        try:
            data = _http_get(IMG_URL.format(split=split, id=int(i)))
            Image.open(io.BytesIO(data)).verify()
            tmp = dst.with_suffix(".part")
            tmp.write_bytes(data)
            os.replace(tmp, dst)
            return None
        except Exception as ex:  # noqa: BLE001
            return f"{k}: {ex}"

    errors = []
    if todo:
        print(f"[data] downloading {len(todo)} images ({n_att} attached)", flush=True)
        with ThreadPoolExecutor(workers) as ex:
            for j, err in enumerate(ex.map(fetch, todo)):
                if err:
                    errors.append(err)
                if (j + 1) % 1000 == 0:
                    print(f"[data]   {j + 1}/{len(todo)} images", flush=True)
    if errors:
        raise RuntimeError(f"{len(errors)} images failed, e.g. {errors[:3]}")
    return {"n_images": len(keys), "attached": n_att, "downloaded": len(todo), "seconds": time.time() - t0}
