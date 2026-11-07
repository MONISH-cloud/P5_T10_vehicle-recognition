"""Choose stricter confidence thresholds on VALIDATION images, without retraining.

Use this when calibration produced thresholds of 0.0, which means the app never gives a less specific answer.
--update-test then re-measures the confidence section of results/metrics.json on the test images.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vehicle_id import classical
from vehicle_id.data import read_manifest
from vehicle_id.decision import Hierarchy, fuse, softmax
from vehicle_id.evaluation import Evaluator, read_rgb
from vehicle_id.metrics import threshold_for_accuracy
from vehicle_id.pipeline import ClassicalModel, OnnxClassifier, OnnxDetector

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", default="data/manifest.csv")
parser.add_argument("--models", default="models")
parser.add_argument("--results", default="results")
parser.add_argument("--target", type=float, default=0.97)
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--update-test", action="store_true")
args = parser.parse_args()
models = Path(args.models)
meta = json.loads((models / "meta.json").read_text())
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.gpu else ["CPUExecutionProvider"]
file = "classifier_fp32.onnx" if (models / "classifier_fp32.onnx").exists() else meta["classifier_file"]
classifier = OnnxClassifier(models / file, meta["size"], meta["mean"], meta["std"], providers)
detector = OnnxDetector(models / meta["detector_file"], class_ids=meta["detector_classes"], providers=providers)
classical_model = ClassicalModel(models / "classical.npz") if (models / "classical.npz").exists() else None
beta = meta.get("beta", 0.0)
print(f"Classifier {file}, temperature {meta['temperature']:.3f}, fusion weight {beta}")

# Validation probabilities, computed exactly as in 06_calibrate.py.
rows = read_manifest(args.manifest, "val")
labels = np.array([r["label"] for r in rows])
locator = Evaluator(detector, classifier, None, meta["class_names"])
probs = []
for row in rows:
    image = read_rgb(row["path"])
    box, _ = locator.locate(row, image, "detector")
    crop = classical.crop(image, box)
    p = softmax(classifier(crop, flip=True), meta["temperature"])
    if classical_model is not None and beta > 0:
        p = fuse(p, classical_model.probabilities(classical.extract_features(crop, box, image.shape)), beta)
    probs.append(p)
probs = np.stack(probs)
hierarchy = Hierarchy(meta["class_names"])
model_probs = np.stack([hierarchy.rollup(p, "model") for p in probs])
make_probs = np.stack([hierarchy.rollup(p, "make") for p in probs])
old = meta.get("thresholds")
meta["thresholds"] = {"full": threshold_for_accuracy(probs, labels, args.target),
                      "model": threshold_for_accuracy(model_probs, hierarchy.model_of[labels], args.target),
                      "make": threshold_for_accuracy(make_probs, hierarchy.make_of[labels], args.target)}
meta["threshold_target"] = args.target
(models / "meta.json").write_text(json.dumps(meta, indent=1))
print(f"Thresholds {old} -> {meta['thresholds']} (validation target {args.target:.0%})")

if args.update_test:
    metrics_path = Path(args.results) / "metrics.json"
    results = json.loads(metrics_path.read_text())
    evaluator = Evaluator(detector, classifier, classical_model, meta["class_names"], meta["temperature"], beta, meta["thresholds"])
    saved = Path(args.results) / "hybrid_test_probs.npz"
    if saved.exists():
        data = np.load(saved)
        test_probs, test_labels = data["probs"], data["labels"]
    else:
        test_rows = read_manifest(args.manifest, "test")
        test_labels = np.array([r["label"] for r in test_rows])
        print(f"Re-running the hybrid on {len(test_rows)} test images...", flush=True)
        test_probs = evaluator.run_variant(test_rows, box_mode="detector", enhance=True, refine=True, tta=True, fusion=True)[0]
    results["confidence"] = evaluator.evaluate_confidence(test_probs, test_labels)
    metrics_path.write_text(json.dumps(results, indent=2))
    print("Updated the confidence section of", metrics_path)
    for level, row in results["confidence"]["decision_levels"].items():
        print(f"  {level}: {row['share']:.1%} of test images")
