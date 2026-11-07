"""Stanford Cars annotations and the leak-free train / validation / test manifests."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import scipy.io as sio
from sklearn.model_selection import StratifiedShuffleSplit


def _find_file(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name), key=lambda p: len(p.parts))
    if not matches:
        raise FileNotFoundError(f"{name} not found under {root}. See the README dataset section.")
    return matches[0]


def _find_images(root: Path, folder: str, sample: str) -> Path:
    matches = [p.parent for p in root.rglob(sample) if folder in p.parent.as_posix().split("/")]
    if not matches:
        raise FileNotFoundError(f"Could not find {folder}/{sample} under {root}.")
    return sorted(matches, key=lambda p: len(p.parts))[0]


def _read(mat_path: Path, image_dir: Path, split: str) -> list[dict]:
    annotations = sio.loadmat(mat_path, squeeze_me=True)["annotations"]
    rows = []
    for a in np.atleast_1d(annotations):
        rows.append({"split": split, "path": str(image_dir / str(a["fname"])), "label": int(a["class"]) - 1,
                     "x1": int(a["bbox_x1"]) - 1, "y1": int(a["bbox_y1"]) - 1,
                     "x2": int(a["bbox_x2"]), "y2": int(a["bbox_y2"])})
    return rows


def load_dataset(root: str | Path, val_fraction: float = 0.1, seed: int = 42):
    """Split validation off the training images. Test images are never used for any choice."""
    root = Path(root)
    names = [str(n) for n in sio.loadmat(_find_file(root, "cars_meta.mat"), squeeze_me=True)["class_names"]]
    train = _read(_find_file(root, "cars_train_annos.mat"), _find_images(root, "cars_train", "00001.jpg"), "train")
    test = _read(_find_file(root, "cars_test_annos_withlabels.mat"), _find_images(root, "cars_test", "00001.jpg"), "test")
    labels = np.array([r["label"] for r in train])
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    _, val_index = next(splitter.split(np.zeros(len(labels)), labels))
    for index in val_index:
        train[index]["split"] = "val"
    return train + test, names


def write_manifest(rows: list[dict], path: str | Path) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: str | Path, split: str | None = None) -> list[dict]:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("label", "x1", "y1", "x2", "y2"):
            row[key] = int(row[key])
    return [r for r in rows if split is None or r["split"] == split]
