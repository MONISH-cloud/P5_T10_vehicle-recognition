"""Final, one-time evaluation on unseen TEST images. Do not tune anything after reading these numbers."""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vehicle_id import classical
from vehicle_id.data import read_manifest
from vehicle_id.evaluation import Evaluator, measure_latency, read_rgb, save_json
from vehicle_id.pipeline import ClassicalModel, OnnxClassifier, OnnxDetector, Options, load_pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", default="data/manifest.csv")
parser.add_argument("--models", default="models")
parser.add_argument("--out", default="results")
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--limit", type=int, default=0, help="Smoke test only: reports from a limited run are marked partial")
FAST = os.environ.get("P5_FAST_EVAL") == "1"  # Backup mode: smaller subsets for the slowest sections
parser.add_argument("--ablation-images", type=int, default=2000 if FAST else 0, help="0 = all test images")
parser.add_argument("--robustness-images", type=int, default=300 if FAST else 500)
parser.add_argument("--deployed-images", type=int, default=500 if FAST else 1000)
parser.add_argument("--unseen", default="data/unseen", help="Team-collected modern_cars/ and non_cars/ folders")
args = parser.parse_args()
models, out = Path(args.models), Path(args.out)
out.mkdir(parents=True, exist_ok=True)
meta = json.loads((models / "meta.json").read_text())
providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if args.gpu else ["CPUExecutionProvider"])
rows = read_manifest(args.manifest, "test")
if args.limit:
    rows = rows[:args.limit]
labels = np.array([r["label"] for r in rows])
detector = OnnxDetector(models / meta["detector_file"], class_ids=meta["detector_classes"], providers=providers)
classifier = OnnxClassifier(models / "classifier_fp32.onnx", meta["size"], meta["mean"], meta["std"], providers)
classical_model = ClassicalModel(models / "classical.npz")
evaluator = Evaluator(detector, classifier, classical_model, meta["class_names"], meta["temperature"], meta["beta"], meta["thresholds"])
results = {"partial_run": bool(args.limit), "test_images": len(rows), "classifier": meta["classifier_name"],
           "detector_file": meta["detector_file"], "temperature": meta["temperature"], "fusion_beta": meta["beta"]}


def checkpoint():
    save_json(results, out / "metrics.json")
    print("Saved progress to", out / "metrics.json", flush=True)


sample = np.random.default_rng(42).permutation(len(rows))
hybrid = dict(box_mode="detector", enhance=True, refine=True, tta=True, fusion=True)
results["detection"] = evaluator.evaluate_detection(rows)
checkpoint()
subset_size = args.ablation_images if 0 < args.ablation_images < len(rows) else len(rows)
ablation_rows = rows if subset_size == len(rows) else [rows[i] for i in sample[:subset_size]]
table, kept = evaluator.evaluate_ablation(ablation_rows)
results["ablation"], results["ablation_images"] = table, len(ablation_rows)
if subset_size == len(rows):
    deep_scores, hybrid_scores = table["Detector crop (deep-only baseline)"], table["Full hybrid"]
    hybrid_probs, hybrid_boxes = kept["Full hybrid"]
else:  # The three-way comparison always uses every test image.
    deep_scores = evaluator.scores(evaluator.run_variant(rows, box_mode="detector")[0], labels)
    hybrid_probs, hybrid_boxes = evaluator.run_variant(rows, **hybrid)
    hybrid_scores = evaluator.scores(hybrid_probs, labels)
np.savez_compressed(out / "hybrid_test_probs.npz", probs=hybrid_probs, labels=labels)
results["comparison"] = {"classical_only": evaluator.scores(evaluator.classical_only(rows), labels),
                         "deep_only": deep_scores, "hybrid": hybrid_scores}
checkpoint()
results["confidence"] = evaluator.evaluate_confidence(hybrid_probs, labels)
results["subgroups"] = evaluator.evaluate_subgroups(rows, hybrid_probs)
results["failures"] = evaluator.mine_failures(rows, hybrid_probs, hybrid_boxes, out / "failures")
checkpoint()
results["robustness"] = evaluator.evaluate_robustness([rows[i] for i in sample[:args.robustness_images]])
checkpoint()

# Efficiency on CPU, the deployment hardware.
latency_rows = [rows[i] for i in sample[:100]]
crops = []
for row in latency_rows:
    image = read_rgb(row["path"])
    crops.append(classical.crop(image, classical.expand_box((row["x1"], row["y1"], row["x2"], row["y2"]), image.shape[1], image.shape[0])))
efficiency = {}
for path in sorted(models.glob("classifier_*.onnx")):
    candidate = OnnxClassifier(path, meta["size"], meta["mean"], meta["std"])
    efficiency[path.name] = {"size_mb": path.stat().st_size / 1e6, "cpu_latency": measure_latency(candidate, crops)}
detector_cpu = OnnxDetector(models / meta["detector_file"], class_ids=meta["detector_classes"])
images = [read_rgb(r["path"]) for r in latency_rows]
efficiency[meta["detector_file"]] = {"size_mb": (models / meta["detector_file"]).stat().st_size / 1e6, "cpu_latency": measure_latency(detector_cpu, images)}
pipeline = load_pipeline(models)
end_to_end = measure_latency(lambda r: pipeline.run(r["path"], Options()), latency_rows)
results["efficiency"] = {"models": efficiency, "deployed_classifier": meta["classifier_file"], "end_to_end_cpu": end_to_end}

# The deployed model, on the same subset as the fp32 model, for an honest comparison.
subset = [rows[i] for i in sample[:args.deployed_images]]
subset_labels = np.array([r["label"] for r in subset])
fp32_scores = evaluator.scores(evaluator.run_variant(subset, **hybrid)[0], subset_labels)
deployed = OnnxClassifier(models / meta["classifier_file"], meta["size"], meta["mean"], meta["std"])
deployed_eval = Evaluator(detector, deployed, classical_model, meta["class_names"], meta["temperature"], meta["beta"], meta["thresholds"])
deployed_eval.detection_cache = evaluator.detection_cache
results["deployed_vs_fp32"] = {"images": len(subset), "fp32": fp32_scores, "deployed": deployed_eval.scores(deployed_eval.run_variant(subset, **hybrid)[0], subset_labels)}
checkpoint()

# Open-set behaviour on images the team collected: modern cars and non-vehicles.
unseen = {}
for folder, description in (("modern_cars", "cars outside the catalogue"), ("non_cars", "non-vehicle images")):
    files = sorted(p for p in (Path(args.unseen) / folder).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if files:
        outcomes = [pipeline.run(p) for p in files]
        levels = [r.vehicles[0]["level"] if r.vehicles else r.status for r in outcomes]
        unseen[folder] = {"description": description, "images": len(files), "outcomes": {l: levels.count(l) for l in set(levels)}}
results["unseen_open_set"] = unseen
checkpoint()
