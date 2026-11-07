"""The whole non-training flow on synthetic images: data, classical training, pipeline, evaluation, reports."""
import json
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np
import pytest
from conftest import CLASS_NAMES
from vehicle_id.data import load_dataset, read_manifest, write_manifest
from vehicle_id.evaluation import Evaluator
from vehicle_id.pipeline import ClassicalModel, Options, VehiclePipeline
from vehicle_id.reporting import ablation_markdown, eval_markdown

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def prepared(dataset, tmp_path_factory):
    out = tmp_path_factory.mktemp("work")
    subprocess.run([sys.executable, str(ROOT / "scripts/01_prepare_data.py"), "--data-root", str(dataset.parent), "--out", str(out / "data")], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/02_train_classical.py"), "--manifest", str(out / "data/manifest.csv"), "--out", str(out / "classical.npz"), "--pca", "32", "--jobs", "1"], check=True)
    return out


def test_splits_do_not_leak(dataset):
    rows, names = load_dataset(dataset.parent)
    assert names == CLASS_NAMES
    by_split = {s: {r["path"] for r in rows if r["split"] == s} for s in ("train", "val", "test")}
    assert all(by_split.values())
    assert not by_split["train"] & by_split["val"]
    assert not by_split["test"] & (by_split["train"] | by_split["val"])
    assert all("cars_test" in p for p in by_split["test"])
    assert {r["label"] for r in rows if r["split"] == "val"} == set(range(len(CLASS_NAMES)))
    first = rows[0]
    assert 0 <= first["x1"] < first["x2"] and 0 <= first["y1"] < first["y2"]


def test_yolo_labels_are_normalised(prepared):
    labels = list((prepared / "data/yolo/labels/train").glob("*.txt"))
    assert labels
    values = [float(v) for v in labels[0].read_text().split()]
    assert values[0] == 0 and all(0 < v <= 1 for v in values[1:])
    assert not list((prepared / "data/yolo/images").glob("test/*"))


def test_classical_model_beats_chance(prepared, fakes):
    rows = read_manifest(prepared / "data/manifest.csv", "test")
    detector, classifier = fakes
    evaluator = Evaluator(detector, classifier, ClassicalModel(prepared / "classical.npz"), CLASS_NAMES)
    scores = evaluator.scores(evaluator.classical_only(rows), np.array([r["label"] for r in rows]))
    assert scores["top1"] > 2 / len(CLASS_NAMES)


def test_pipeline_end_to_end(prepared, fakes):
    detector, classifier = fakes
    pipeline = VehiclePipeline(detector, classifier, ClassicalModel(prepared / "classical.npz"), CLASS_NAMES, beta=0.05)
    row = read_manifest(prepared / "data/manifest.csv", "test")[0]
    result = pipeline.run(row["path"])
    assert result.status == "ok" and result.vehicles[0]["make"] and "Classified crop" in result.intermediates
    for view in ("Edges (Canny)", "Contour analysis", "Shape (HOG)", "Texture (LBP)", "Colour (hue histogram)"):
        assert view in result.intermediates and result.intermediates[view].dtype == np.uint8
    assert result.geometry["aspect_ratio"] > 0
    assert set(result.timings_ms) == {"preprocess", "detect", "classify"}


def test_pipeline_bad_and_empty_inputs(tmp_path, fakes):
    detector, classifier = fakes
    pipeline = VehiclePipeline(detector, classifier, None, CLASS_NAMES)
    bad = tmp_path / "bad.png"
    bad.write_text("nope")
    assert pipeline.run(bad).status == "bad_input"
    blank = np.full((200, 200, 3), 128, np.uint8)
    assert pipeline.run(blank).status == "no_vehicle"


def test_multiple_vehicles_are_returned(fakes):
    _, classifier = fakes
    image = np.full((240, 480, 3), 200, np.uint8)
    image[40:200, 20:200] = (200, 30, 30)
    image[60:200, 280:440] = (30, 30, 200)
    detector = lambda _: [{"box": (20, 40, 200, 200), "score": .9}, {"box": (280, 60, 440, 200), "score": .8}]
    result = VehiclePipeline(detector, classifier, None, CLASS_NAMES).run(image, Options(use_refinement=False))
    assert len(result.vehicles) == 2


def test_full_evaluation_and_reports(prepared, fakes, tmp_path):
    detector, classifier = fakes
    rows = read_manifest(prepared / "data/manifest.csv", "test")
    labels = np.array([r["label"] for r in rows])
    evaluator = Evaluator(detector, classifier, ClassicalModel(prepared / "classical.npz"), CLASS_NAMES, beta=0.05)
    table, kept = evaluator.evaluate_ablation(rows)
    probs, boxes = kept["Full hybrid"]
    latency = {"mean_ms": 10.0, "p95_ms": 12.0, "samples": 1}
    results = {"test_images": len(rows), "classifier": "fake", "detector_file": "fake", "partial_run": True,
               "detection": evaluator.evaluate_detection(rows), "ablation": table,
               "comparison": {"classical_only": evaluator.scores(evaluator.classical_only(rows), labels),
                              "deep_only": table["Detector crop (deep-only baseline)"], "hybrid": table["Full hybrid"]},
               "confidence": evaluator.evaluate_confidence(probs, labels),
               "failures": evaluator.mine_failures(rows, probs, boxes, tmp_path / "failures"),
               "robustness": evaluator.evaluate_robustness(rows[:6]), "subgroups": evaluator.evaluate_subgroups(rows, probs),
               "efficiency": {"models": {"m.onnx": {"size_mb": 1.0, "cpu_latency": latency}}, "deployed_classifier": "m.onnx", "end_to_end_cpu": latency},
               "deployed_vs_fp32": {"images": 5, "fp32": table["Full hybrid"], "deployed": table["Full hybrid"]}}
    json.dumps(results)
    assert results["robustness"]["conditions"]["viewpoint_proxy/1"]["deep_iou"] is None
    assert results["robustness"]["conditions"]["low_light/1"]["deep_iou"] is not None
    assert sum(row["images"] for row in results["subgroups"]["car_size"].values()) == len(rows)
    assert results["detection"]["mean_iou"] > 0.5
    assert len(table) == 12 and len(results["robustness"]["conditions"]) == 21
    text = eval_markdown(results, "1ABC, 2DEF")
    lines = text.splitlines()
    assert lines[0] == "1ABC,2DEF" and "PARTIAL RUN" in text
    order = [text.index(f"## {n}.") for n in range(1, 7)]
    assert order == sorted(order)
    assert "TEAM TO EXPLAIN" in ablation_markdown(results) and "Measured change" in text
