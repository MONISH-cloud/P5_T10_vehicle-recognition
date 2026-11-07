"""Fine-tune YOLO11 on Stanford Cars boxes, one 'car' class. Calibration later compares it with stock COCO weights."""
import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/yolo/cars.yaml")
parser.add_argument("--weights", default="yolo11n.pt")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--project", default="checkpoints/detector")
args = parser.parse_args()
project = Path(args.project).resolve()
model = YOLO(args.weights)
model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, patience=6,
            project=str(project), name="finetune", exist_ok=True, single_cls=True, seed=42)
# Ultralytics may choose its own output folder, so always copy the weights to the path later steps expect.
target = project / "finetune" / "weights" / "best.pt"
best = Path(model.trainer.best).resolve()
if not best.exists():
    raise SystemExit(f"Detector weights not found at {best}")
target.parent.mkdir(parents=True, exist_ok=True)
if best != target:
    shutil.copy(best, target)
print("Detector weights saved to", target)
