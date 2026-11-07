"""Make every remaining choice on VALIDATION images: detector, quantized model, temperature, beta, thresholds."""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vehicle_id import classical
from vehicle_id.data import read_manifest
from vehicle_id.decision import Hierarchy, fit_temperature, fuse, softmax
from vehicle_id.evaluation import Evaluator, read_rgb
from vehicle_id.metrics import threshold_for_accuracy, topk_accuracy
from vehicle_id.pipeline import ClassicalModel, OnnxClassifier, OnnxDetector

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", default="data/manifest.csv")
parser.add_argument("--models", default="models")
parser.add_argument("--target", type=float, default=0.97, help="Accuracy the reported answers must reach on validation")
parser.add_argument("--max-drop", type=float, default=0.005, help="Top-1 loss allowed for a quantized model")
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--limit", type=int, default=0)
args = parser.parse_args()
models = Path(args.models)
meta = json.loads((models / "meta.json").read_text())
providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if args.gpu else ["CPUExecutionProvider"])
rows = read_manifest(args.manifest, "val")
if args.limit:
    rows = rows[:args.limit]
labels = np.array([r["label"] for r in rows])
classifier = OnnxClassifier(models / "classifier_fp32.onnx", meta["size"], meta["mean"], meta["std"], providers)

# 1. Detector: stock COCO weights against the fine-tuned model.
best_detector, best_map = None, -1.0
for file, classes in (("detector_coco.onnx", [2, 5, 7]), ("detector_finetuned.onnx", [0])):
    if not (models / file).exists():
        continue
    report = Evaluator(OnnxDetector(models / file, class_ids=classes, providers=providers), classifier, None, meta["class_names"]).evaluate_detection(rows)
    print(file, report)
    if report["map50_95"] > best_map:
        best_detector, best_map = (file, classes), report["map50_95"]
meta["detector_file"], meta["detector_classes"] = best_detector
detector = OnnxDetector(models / best_detector[0], class_ids=best_detector[1], providers=providers)

# 2. Temperature, using the same crops the live app sees.
evaluator = Evaluator(detector, classifier, None, meta["class_names"])
crops, placements, logits = [], [], []
for row in rows:
    image = read_rgb(row["path"])
    box, _ = evaluator.locate(row, image, "detector")
    crops.append(classical.crop(image, box))
    placements.append((box, image.shape))
    logits.append(classifier(crops[-1], flip=True))
logits = np.stack(logits)
temperature = fit_temperature(logits, labels)
probs = softmax(logits, temperature)
meta["temperature"] = temperature
print(f"Temperature {temperature:.3f}; validation top-1 {topk_accuracy(probs, labels):.4f}")

# 3. Classical fusion weight. beta=0 is kept if classical features do not help.
beta = 0.0
if (models / "classical.npz").exists():
    model = ClassicalModel(models / "classical.npz")
    classical_probs = np.stack([model.probabilities(classical.extract_features(crop, box, shape)) for crop, (box, shape) in zip(crops, placements)])
    scores = {b: topk_accuracy(np.stack([fuse(p, c, b) for p, c in zip(probs, classical_probs)]), labels) for b in (0.0, 0.02, 0.05, 0.1, 0.2, 0.3)}
    print("Fusion weight -> validation top-1:", scores)
    beta = max(scores, key=lambda b: (scores[b], -b))
    probs = np.stack([fuse(p, c, beta) for p, c in zip(probs, classical_probs)])
meta["beta"] = beta

# 4. Decision thresholds for each answer level.
hierarchy = Hierarchy(meta["class_names"])
model_probs = np.stack([hierarchy.rollup(p, "model") for p in probs])
make_probs = np.stack([hierarchy.rollup(p, "make") for p in probs])
meta["thresholds"] = {"full": threshold_for_accuracy(probs, labels, args.target),
                      "model": threshold_for_accuracy(model_probs, hierarchy.model_of[labels], args.target),
                      "make": threshold_for_accuracy(make_probs, hierarchy.make_of[labels], args.target)}
print("Thresholds:", meta["thresholds"])

# 5. Deployed classifier: the smallest whose validation top-1 stays within the allowed drop.
base = topk_accuracy(softmax(np.stack([classifier(c) for c in crops[:500]])), labels[:500])
chosen = "classifier_fp32.onnx"
for file in ("classifier_int8_dynamic.onnx", "classifier_int8_static.onnx"):
    if not (models / file).exists():
        continue
    candidate = OnnxClassifier(models / file, meta["size"], meta["mean"], meta["std"])
    accuracy = topk_accuracy(softmax(np.stack([candidate(c) for c in crops[:500]])), labels[:500])
    print(f"{file}: validation top-1 {accuracy:.4f} (fp32 {base:.4f})")
    if base - accuracy <= args.max_drop and (models / file).stat().st_size < (models / chosen).stat().st_size:
        chosen = file
meta["classifier_file"] = chosen
(models / "meta.json").write_text(json.dumps(meta, indent=1))
print("Deploying", chosen, "with detector", meta["detector_file"])
