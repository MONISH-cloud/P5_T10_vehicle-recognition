"""Ablation, comparison, robustness, confidence and failure-case evaluation.

Detector and classifier are passed in, so this code runs the same with real models or test fakes.
"""
from __future__ import annotations
import json
import shutil
import time
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from . import classical, corruptions, metrics
from .decision import Hierarchy, decide, fuse, softmax


def read_rgb(path) -> np.ndarray:
    """No resizing or EXIF rotation, because the annotated boxes use the stored pixels."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def gt_box(row) -> tuple:
    return row["x1"], row["y1"], row["x2"], row["y2"]


def primary_detection(detections, shape, min_area_fraction=0.02):
    """The same rule as the live app: the largest sufficiently large detection."""
    area = lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1])
    usable = [d for d in detections if area(d) >= min_area_fraction * shape[0] * shape[1]]
    return max(usable, key=area) if usable else None


class Evaluator:
    def __init__(self, detector, classifier, classical_model, class_names, temperature=1.0, beta=0.0, thresholds=None):
        self.detector, self.classifier, self.classical_model = detector, classifier, classical_model
        self.hierarchy = Hierarchy(class_names)
        self.temperature, self.beta = temperature, beta
        self.thresholds = thresholds or {"full": 0.6, "model": 0.6, "make": 0.6}
        self.detection_cache: dict[str, list] = {}

    def detect_cached(self, path, image):
        if path not in self.detection_cache:
            self.detection_cache[path] = self.detector(image)
        return self.detection_cache[path]

    def locate(self, row, image, mode, enhanced=False, refine=False, transform_key=None):
        height, width = image.shape[:2]
        if mode == "full":
            return (0, 0, width, height), None
        if mode == "gt":
            return classical.expand_box(gt_box(row), width, height), gt_box(row)
        if mode == "contour":
            box = classical.contour_roi(image)
            return classical.expand_box(box, width, height), box
        detections = self.detector(image) if (enhanced or transform_key) else self.detect_cached(row["path"], image)
        primary = primary_detection(detections, image.shape)
        raw = primary["box"] if primary else classical.contour_roi(image)
        if refine and primary:
            raw = classical.refine_box(image, raw)
        return classical.expand_box(raw, width, height), raw

    def probabilities(self, row, box_mode="detector", enhance=False, refine=False, tta=False, fusion=False,
                      corruption=None, severity=1):
        image = read_rgb(row["path"])
        if corruption:
            image = corruptions.apply(corruption, image, severity, seed=_stable_seed(row["path"]))
        enhanced = False
        if enhance:
            result, steps = classical.adaptive_enhance(image)
            enhanced, image = bool(steps), result
        box, raw = self.locate(row, image, box_mode, enhanced, refine, corruption)
        region = classical.crop(image, box)
        probs = softmax(self.classifier(region, flip=tta), self.temperature)
        if fusion and self.classical_model is not None and self.beta > 0:
            features = classical.extract_features(region, box, image.shape)
            probs = fuse(probs, self.classical_model.probabilities(features), self.beta)
        return probs, raw

    def run_variant(self, rows, **flags):
        results = [self.probabilities(row, **flags) for row in rows]
        return np.stack([r[0] for r in results]), [r[1] for r in results]

    def classical_only(self, rows):
        output = []
        for row in rows:
            image = read_rgb(row["path"])
            box = classical.contour_roi(image)
            region = classical.crop(image, box)
            output.append(self.classical_model.probabilities(classical.extract_features(region, box, image.shape)))
        return np.stack(output)

    def scores(self, probs, labels):
        make_true, model_true = self.hierarchy.make_of[labels], self.hierarchy.model_of[labels]
        make_pred = np.array([self.hierarchy.rollup(p, "make").argmax() for p in probs])
        model_pred = np.array([self.hierarchy.rollup(p, "model").argmax() for p in probs])
        return {"images": int(len(labels)), "top1": metrics.topk_accuracy(probs, labels, 1),
                "top5": metrics.topk_accuracy(probs, labels, 5),
                "mean_per_class": metrics.mean_per_class_accuracy(probs, labels),
                "make_accuracy": float((make_pred == make_true).mean()),
                "make_model_accuracy": float((model_pred == model_true).mean())}

    def evaluate_detection(self, rows):
        predictions, ground_truth, ious, misses = [], {}, [], 0
        for row in rows:
            image = read_rgb(row["path"])
            detections = self.detect_cached(row["path"], image)
            ground_truth[row["path"]] = gt_box(row)
            predictions += [(row["path"], d["score"], d["box"]) for d in detections]
            primary = primary_detection(detections, image.shape)
            if primary is None:
                misses += 1
                ious.append(0.0)
            else:
                ious.append(classical.box_iou(primary["box"], gt_box(row)))
        return {"images": len(rows), "mean_iou": float(np.mean(ious)), "median_iou": float(np.median(ious)),
                "iou_at_least_0_5": float((np.array(ious) >= 0.5).mean()), "no_detection_rate": misses / len(rows),
                "ap50": metrics.average_precision(predictions, ground_truth, 0.5),
                "map50_95": metrics.map_50_95(predictions, ground_truth)}

    def evaluate_ablation(self, rows):
        labels = np.array([r["label"] for r in rows])
        detector = dict(box_mode="detector")
        hybrid = dict(box_mode="detector", enhance=True, refine=True, tta=True, fusion=True)
        variants = {
            "Full image, no detector": dict(box_mode="full"),
            "Ground-truth box (upper bound, not deployable)": dict(box_mode="gt"),
            "Detector crop (deep-only baseline)": detector,
            "Detector + adaptive enhancement": {**detector, "enhance": True},
            "Detector + GrabCut refinement": {**detector, "refine": True},
            "Detector + flip TTA": {**detector, "tta": True},
            "Detector + classical fusion": {**detector, "fusion": True},
            "Full hybrid": hybrid,
            "Hybrid without enhancement": {**hybrid, "enhance": False},
            "Hybrid without refinement": {**hybrid, "refine": False},
            "Hybrid without TTA": {**hybrid, "tta": False},
            "Hybrid without classical fusion": {**hybrid, "fusion": False},
        }
        table, kept = {}, {}
        for name, flags in variants.items():
            probs, boxes = self.run_variant(rows, **flags)
            table[name] = self.scores(probs, labels)
            if name in ("Detector crop (deep-only baseline)", "Full hybrid"):
                kept[name] = (probs, boxes)
        return table, kept

    def _condition_scores(self, rows, labels, flags, measure_iou=True):
        probs, boxes = self.run_variant(rows, **flags)
        iou = float(np.mean([classical.box_iou(b, gt_box(r)) for b, r in zip(boxes, rows)])) if measure_iou else None
        return self.scores(probs, labels)["top1"], iou

    def evaluate_robustness(self, rows):
        """Classification and localization. The warp moves the car, so its IoU is not measured."""
        labels = np.array([r["label"] for r in rows])
        deep = dict(box_mode="detector")
        hybrid = dict(box_mode="detector", enhance=True, refine=True, tta=True, fusion=True)
        deep_top1, deep_iou = self._condition_scores(rows, labels, deep)
        hybrid_top1, hybrid_iou = self._condition_scores(rows, labels, hybrid)
        output = {"images": len(rows),
                  "clean": {"deep_top1": deep_top1, "hybrid_top1": hybrid_top1, "deep_iou": deep_iou, "hybrid_iou": hybrid_iou},
                  "conditions": {}}
        for name in corruptions.CORRUPTIONS:
            for severity in corruptions.SEVERITIES:
                condition = {"corruption": name, "severity": severity}
                measure = name != "viewpoint_proxy"
                deep_top1, deep_iou = self._condition_scores(rows, labels, {**deep, **condition}, measure)
                hybrid_top1, hybrid_iou = self._condition_scores(rows, labels, {**hybrid, **condition}, measure)
                output["conditions"][f"{name}/{severity}"] = {"deep_top1": deep_top1, "hybrid_top1": hybrid_top1,
                                                              "deep_iou": deep_iou, "hybrid_iou": hybrid_iou}
        return output

    def evaluate_subgroups(self, rows, probs):
        """Accuracy on real, uncorrupted images, grouped by lighting, car size and box shape."""
        groups: dict[str, dict[str, list]] = {"brightness": {}, "car_size": {}, "box_shape": {}}
        correct = probs.argmax(axis=1) == np.array([r["label"] for r in rows])
        for index, row in enumerate(rows):
            brightness = classical.quality_metrics(read_rgb(row["path"]))["brightness"]
            width, height = row["x2"] - row["x1"], row["y2"] - row["y1"]
            short_side, aspect = min(width, height), width / max(height, 1)
            lighting = "dark (below 0.25)" if brightness < 0.25 else "bright (above 0.75)" if brightness > 0.75 else "normal"
            size = "small (under 150 px)" if short_side < 150 else "large (over 300 px)" if short_side > 300 else "medium (150-300 px)"
            shape = "narrow box (under 1.4)" if aspect < 1.4 else "wide box (over 1.9)" if aspect > 1.9 else "medium box (1.4-1.9)"
            for group, key in (("brightness", lighting), ("car_size", size), ("box_shape", shape)):
                groups[group].setdefault(key, []).append(bool(correct[index]))
        return {group: {key: {"images": len(values), "top1": float(np.mean(values))} for key, values in sorted(members.items())}
                for group, members in groups.items()}

    def evaluate_confidence(self, probs, labels):
        records = [decide(p, self.hierarchy, self.thresholds) for p in probs]
        levels = {}
        for level in ("full", "model", "make", "abstain"):
            members = [i for i, r in enumerate(records) if r["level"] == level]
            correct = 0
            for i in members:
                truth = self.hierarchy.classes[labels[i]]
                record = records[i]
                correct += (record["make"] == truth["make"] and (record["model"] in (None, truth["model"]))
                            and (record["year"] in (None, truth["year"]))) if level != "abstain" else 0
            levels[level] = {"share": len(members) / len(records),
                             "reported_answer_correct": correct / len(members) if members and level != "abstain" else None}
        return {"ece": metrics.expected_calibration_error(probs, labels), "thresholds_used": self.thresholds,
                "decision_levels": levels, "threshold_curve": metrics.threshold_curve(probs, labels)}

    def mine_failures(self, rows, probs, boxes, out_dir, per_tag=8):
        out_dir = Path(out_dir)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        labels = np.array([r["label"] for r in rows])
        predictions = probs.argmax(axis=1)
        confusion, tagged = Counter(), {}
        for index in np.where(predictions != labels)[0]:
            row, truth, guess = rows[index], self.hierarchy.classes[labels[index]], self.hierarchy.classes[predictions[index]]
            confusion[f"{truth['name']} -> {guess['name']}"] += 1
            image = read_rgb(row["path"])
            quality = classical.quality_metrics(image)
            iou = classical.box_iou(boxes[index], gt_box(row))
            area = (row["x2"] - row["x1"]) * (row["y2"] - row["y1"]) / (image.shape[0] * image.shape[1])
            tags = []
            if truth["make"] == guess["make"] and truth["model"] == guess["model"]:
                tags.append("year_only_confusion")
            elif truth["make"] == guess["make"]:
                tags.append("same_make_wrong_model")
            else:
                tags.append("wrong_make")
            if iou < 0.5:
                tags.append("wrong_or_missed_box")
            if quality["brightness"] < 0.25 or quality["brightness"] > 0.8:
                tags.append("difficult_exposure")
            if quality["sharpness"] < 40:
                tags.append("blurred")
            if area < 0.2:
                tags.append("small_vehicle")
            record = {"path": row["path"], "truth": truth["name"], "predicted": guess["name"],
                      "confidence": float(probs[index].max()), "iou": iou}
            for tag in tags:
                tagged.setdefault(tag, []).append(record)
        summary = {"errors": int((predictions != labels).sum()), "images": len(rows),
                   "top_confusions": confusion.most_common(15),
                   "tag_counts": {t: len(v) for t, v in sorted(tagged.items(), key=lambda kv: -len(kv[1]))}, "examples": {}}
        for tag, records in tagged.items():
            target = out_dir / tag
            target.mkdir(parents=True, exist_ok=True)
            chosen = sorted(records, key=lambda r: -r["confidence"])[:per_tag]
            summary["examples"][tag] = chosen
            for number, record in enumerate(chosen):
                shutil.copy(record["path"], target / f"{number:02d}_{Path(record['path']).name}")
        return summary


def _stable_seed(text: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(text)) % (2 ** 32)


def measure_latency(function, inputs, warmup: int = 2) -> dict:
    for item in inputs[:warmup]:
        function(item)
    times = []
    for item in inputs:
        start = time.perf_counter()
        function(item)
        times.append((time.perf_counter() - start) * 1000)
    return {"mean_ms": float(np.mean(times)), "p95_ms": float(np.percentile(times, 95)), "samples": len(times)}


def save_json(data: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2))
