import json
import numpy as np
import pytest
from conftest import CLASS_NAMES, synthetic_car
from vehicle_id import classical, corruptions, metrics
from vehicle_id.decision import Hierarchy, decide, fit_temperature, fuse, parse_class_name, softmax
from vehicle_id.pipeline import BadInput, annotate, letterbox, load_image, parse_yolo_output
from vehicle_id.reporting import eval_markdown


@pytest.mark.parametrize("name,make,model,year", [
    ("Audi S4 Sedan 2012", "Audi", "S4 Sedan", 2012), ("AM General Hummer SUV 2000", "AM General", "Hummer SUV", 2000),
    ("Aston Martin V8 Vantage Coupe 2012", "Aston Martin", "V8 Vantage Coupe", 2012), ("Land Rover LR2 SUV 2012", "Land Rover", "LR2 SUV", 2012),
    ("Ram C/V Cargo Van Minivan 2012", "Ram", "C/V Cargo Van Minivan", 2012)])
def test_parse_class_name(name, make, model, year):
    parsed = parse_class_name(name)
    assert (parsed["make"], parsed["model"], parsed["year"]) == (make, model, year)


def test_parse_rejects_bad_name():
    with pytest.raises(ValueError):
        parse_class_name("not a class")


def _probs(values):
    probs = np.full(len(CLASS_NAMES), 1e-6)
    for index, value in values.items():
        probs[index] = value
    return probs / probs.sum()


def test_decision_levels_back_off():
    hierarchy = Hierarchy(CLASS_NAMES)
    limits = {"full": 0.8, "model": 0.8, "make": 0.6}
    assert decide(_probs({0: 0.95, 2: 0.05}), hierarchy, limits)["level"] == "full"
    years = decide(_probs({0: 0.5, 1: 0.45, 2: 0.05}), hierarchy, limits)
    assert years["level"] == "model" and years["model"] == "S4 Sedan" and years["year"] is None
    assert "2007" in years["label"] and "2012" in years["label"]
    flat = np.ones(len(CLASS_NAMES)) / len(CLASS_NAMES)
    assert decide(flat, hierarchy, limits)["level"] == "abstain"


def test_rollup_preserves_probability():
    hierarchy = Hierarchy(CLASS_NAMES)
    probs = softmax(np.random.default_rng(0).normal(size=len(CLASS_NAMES)))
    assert hierarchy.rollup(probs, "make").sum() == pytest.approx(1)
    assert hierarchy.rollup(probs, "model").sum() == pytest.approx(1)


def test_temperature_fixes_overconfidence():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 10, 4000)
    logits = rng.normal(0, 1, (4000, 10))
    logits[np.arange(4000), labels] += 2.0
    assert fit_temperature(logits * 5, labels) > 2.0


def test_fusion_zero_beta_is_noop():
    deep = softmax(np.array([2.0, 1.0, 0.0]))
    assert np.array_equal(fuse(deep, softmax(np.array([0, 5.0, 0])), 0.0), deep)


def test_topk_and_map():
    probs = np.array([[.7, .2, .1], [.2, .5, .3], [.5, .4, .1]])
    labels = np.array([0, 2, 1])
    assert metrics.topk_accuracy(probs, labels, 1) == pytest.approx(1 / 3)
    assert metrics.topk_accuracy(probs, labels, 2) == pytest.approx(1)
    truth = {"a": (0, 0, 10, 10), "b": (0, 0, 10, 10)}
    assert metrics.average_precision([("a", .9, (0, 0, 10, 10)), ("b", .8, (0, 0, 10, 10))], truth) == pytest.approx(1)
    assert metrics.average_precision([("a", .9, (50, 50, 60, 60))], truth) == 0
    half = metrics.average_precision([("a", .9, (0, 0, 10, 10))], truth)
    assert half == pytest.approx(51 / 101)


def test_box_utilities():
    assert classical.box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert classical.box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0
    assert classical.clip_box((-5, -5, 500, 500), 100, 100) == (0, 0, 100, 100)
    assert classical.clip_box((5, 5, 5, 5), 100, 100) == (5, 5, 6, 6)


def test_feature_length_is_constant_and_finite():
    rng = np.random.default_rng(1)
    image, box = synthetic_car(rng, 2)
    first = classical.extract_features(classical.crop(image, box), box, image.shape)
    second = classical.extract_features(np.zeros((20, 90, 3), np.uint8))
    assert first.shape == second.shape and np.isfinite(first).all() and np.isfinite(second).all()


def test_enhancement_only_acts_when_needed():
    rng = np.random.default_rng(2)
    image, _ = synthetic_car(rng, 1)
    dark = (image * 0.1).astype(np.uint8)
    enhanced, steps = classical.adaptive_enhance(dark)
    assert any(s.startswith("gamma") for s in steps) and enhanced.mean() > dark.mean() * 1.5
    assert classical.adaptive_enhance(image)[0] is image or classical.adaptive_enhance(image)[1] != []


def test_contour_roi_and_refine_find_the_car():
    rng = np.random.default_rng(3)
    image, truth = synthetic_car(rng, 4)
    assert classical.box_iou(classical.contour_roi(image), truth) > 0.5
    loose = classical.clip_box((truth[0] - 10, truth[1] - 10, truth[2] + 10, truth[3] + 10), image.shape[1], image.shape[0])
    assert classical.box_iou(classical.refine_box(image, loose), truth) >= classical.box_iou(loose, truth) - 0.05


@pytest.mark.parametrize("name", corruptions.CORRUPTIONS)
@pytest.mark.parametrize("severity", corruptions.SEVERITIES)
def test_corruptions(name, severity):
    rng = np.random.default_rng(4)
    image, _ = synthetic_car(rng, 3)
    output = corruptions.apply(name, image, severity, seed=1)
    assert output.shape == image.shape and output.dtype == np.uint8
    assert np.array_equal(output, corruptions.apply(name, image, severity, seed=1))


def test_yolo_output_parsing_maps_back_to_original():
    image = np.zeros((480, 640, 3), np.uint8)
    canvas, scale, left, top = letterbox(image, 640)
    assert canvas.shape == (640, 640, 3) and scale == 1 and top == 80
    raw = np.zeros((1, 4 + 8, 2), np.float32)
    raw[0, :4, 0] = [320, 320, 200, 100]
    raw[0, 4 + 2, 0] = 0.9
    raw[0, :4, 1] = [100, 100, 20, 20]
    raw[0, 4 + 1, 1] = 0.99  # A non-vehicle class must be ignored.
    results = parse_yolo_output(raw, 640, scale, left, top, (2, 5, 7), shape=image.shape)
    assert len(results) == 1 and results[0]["box"] == (220, 190, 420, 290) and results[0]["score"] == pytest.approx(0.9)


def test_load_image_rejects_bad_files(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"this is not an image")
    with pytest.raises(BadInput):
        load_image(bad)
    with pytest.raises(BadInput):
        load_image(np.zeros((5, 5, 3), np.uint8))
    assert max(load_image(np.zeros((3000, 4000, 3), np.uint8)).shape) == 1600


def test_annotate_handles_all_levels():
    image = np.zeros((200, 300, 3), np.uint8)
    for level in ("full", "model", "make", "abstain"):
        vehicle = {"box": (10, 10, 100, 100), "level": level, "label": "X" * 80, "confidence": .5}
        assert annotate(image, [vehicle]).any()


def test_report_refuses_incomplete_metrics():
    with pytest.raises(ValueError):
        eval_markdown({"comparison": {}}, "USN1")


def test_batch_size_falls_back_until_it_fits(capsys):
    from vehicle_id.training_utils import accumulation_for, pick_batch_size
    memory = {20: None, 16: None, 12: 0.95, 10: 0.79}
    assert pick_batch_size(20, memory.get) == 10
    assert accumulation_for(10, 40) == 4 and accumulation_for(32, 32) == 1 and accumulation_for(8, 40) == 5
    with pytest.raises(SystemExit):
        pick_batch_size(20, lambda size: None)
    assert pick_batch_size(32, {32: 0.5}.get) == 32
