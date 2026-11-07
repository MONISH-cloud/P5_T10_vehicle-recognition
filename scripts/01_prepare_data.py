"""Create the train/val/test manifest, class list and YOLO detector dataset. No model involved."""
import argparse, json, os, sys
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vehicle_id.data import load_dataset, write_manifest
from vehicle_id.decision import parse_class_name

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", required=True, help="Folder holding the Stanford Cars files")
parser.add_argument("--out", default="data")
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
rows, names = load_dataset(args.data_root)
if len(names) != 196:
    print(f"WARNING: expected 196 classes, found {len(names)}. Check the dataset copy.")
for name in names:
    parse_class_name(name)
write_manifest(rows, out / "manifest.csv")
(out / "classes.json").write_text(json.dumps(names, indent=1))

# The detector dataset uses train and val only, so test images remain unseen.
for row in rows:
    if row["split"] == "test":
        continue
    image_dir = out / "yolo/images" / row["split"]
    label_dir = out / "yolo/labels" / row["split"]
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    source = Path(row["path"]).resolve()
    link = image_dir / source.name
    if not link.exists():
        os.symlink(source, link)
    with Image.open(source) as image:
        width, height = image.size
    x, y = (row["x1"] + row["x2"]) / 2 / width, (row["y1"] + row["y2"]) / 2 / height
    w, h = (row["x2"] - row["x1"]) / width, (row["y2"] - row["y1"]) / height
    (label_dir / f"{source.stem}.txt").write_text(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
(out / "yolo/cars.yaml").write_text(
    f"path: {(out / 'yolo').resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: car\n")
counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
print("Images per split:", counts)
