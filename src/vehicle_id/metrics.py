"""Evaluation metrics, implemented here so they can be explained line by line in the viva."""
from __future__ import annotations
import numpy as np
from .classical import box_iou


def topk_accuracy(probs: np.ndarray, labels: np.ndarray, k: int = 1) -> float:
    top = np.argpartition(-probs, kth=min(k, probs.shape[1] - 1), axis=1)[:, :k]
    return float((top == labels[:, None]).any(axis=1).mean())


def mean_per_class_accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    predictions = probs.argmax(axis=1)
    return float(np.mean([(predictions[labels == c] == c).mean() for c in np.unique(labels)]))


def average_precision(predictions, ground_truth: dict, iou_threshold: float = 0.5) -> float:
    """COCO-style 101-point AP. predictions: (image_id, score, box); ground_truth: image_id -> box."""
    ranked = sorted(predictions, key=lambda p: -p[1])
    matched, flags = set(), []
    for image_id, _, box in ranked:
        hit = (image_id not in matched and image_id in ground_truth
               and box_iou(box, ground_truth[image_id]) >= iou_threshold)
        if hit:
            matched.add(image_id)
        flags.append(hit)
    if not ground_truth or not flags:
        return 0.0
    hits = np.cumsum(flags)
    recall = hits / len(ground_truth)
    precision = hits / np.arange(1, len(flags) + 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    samples = [envelope[recall >= r].max() if (recall >= r).any() else 0.0 for r in np.linspace(0, 1, 101)]
    return float(np.mean(samples))


def map_50_95(predictions, ground_truth: dict) -> float:
    return float(np.mean([average_precision(predictions, ground_truth, t) for t in np.arange(0.5, 0.96, 0.05)]))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    confidence, predictions = probs.max(axis=1), probs.argmax(axis=1)
    error = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        members = (confidence > low) & (confidence <= high)
        if members.any():
            error += members.mean() * abs((predictions[members] == labels[members]).mean() - confidence[members].mean())
    return float(error)


def threshold_curve(probs: np.ndarray, labels: np.ndarray, thresholds=None) -> list[dict]:
    """Share of images answered, and accuracy on those, at each confidence threshold."""
    thresholds = np.arange(0, 1.0, 0.05) if thresholds is None else thresholds
    confidence, correct = probs.max(axis=1), probs.argmax(axis=1) == labels
    return [{"threshold": round(float(t), 3), "coverage": float((confidence >= t).mean()),
             "accuracy": float(correct[confidence >= t].mean()) if (confidence >= t).any() else None}
            for t in thresholds]


def threshold_for_accuracy(probs: np.ndarray, labels: np.ndarray, target: float) -> float:
    """Lowest threshold whose answered images reach the target accuracy. Use validation data."""
    for point in threshold_curve(probs, labels, np.arange(0, 1.0, 0.01)):
        if point["accuracy"] is not None and point["accuracy"] >= target:
            return point["threshold"]
    return 0.99
