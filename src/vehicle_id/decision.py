"""Class hierarchy, calibration, classical fusion and low-confidence decisions."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar

MULTI_WORD_MAKES = ("AM General", "Aston Martin", "Land Rover")


def parse_class_name(name: str) -> dict:
    """'Audi S4 Sedan 2012' -> make 'Audi', model 'S4 Sedan', year 2012."""
    tokens = name.strip().split()
    if len(tokens) < 3 or not tokens[-1].isdigit():
        raise ValueError(f"Unexpected Stanford Cars class name: {name!r}")
    make = next((m for m in MULTI_WORD_MAKES if name.startswith(m + " ")), tokens[0])
    return {"name": name, "make": make, "model": name[len(make):].rsplit(" ", 1)[0].strip(),
            "year": int(tokens[-1])}


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled -= scaled.max(axis=-1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=-1, keepdims=True)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit on validation data only."""
    rows = np.arange(len(labels))

    def nll(temperature: float) -> float:
        return float(-np.log(softmax(logits, temperature)[rows, labels] + 1e-12).mean())

    return float(minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x)


def fuse(deep_probs: np.ndarray, classical_probs: np.ndarray | None, beta: float) -> np.ndarray:
    """Log-linear fusion. beta is tuned on validation; beta=0 means classical adds nothing."""
    if classical_probs is None or beta == 0:
        return deep_probs
    scores = np.log(deep_probs + 1e-12) + beta * np.log(classical_probs + 1e-12)
    return softmax(scores)


class Hierarchy:
    """Rolls 196 class probabilities up to make+model and make levels."""

    def __init__(self, class_names: list[str]):
        self.classes = [parse_class_name(n) for n in class_names]
        self.makes = sorted({c["make"] for c in self.classes})
        self.models = sorted({(c["make"], c["model"]) for c in self.classes})
        self.make_of = np.array([self.makes.index(c["make"]) for c in self.classes])
        self.model_of = np.array([self.models.index((c["make"], c["model"])) for c in self.classes])

    def rollup(self, probs: np.ndarray, level: str) -> np.ndarray:
        index = self.make_of if level == "make" else self.model_of
        size = len(self.makes) if level == "make" else len(self.models)
        return np.bincount(index, weights=probs, minlength=size)


def decide(probs: np.ndarray, hierarchy: Hierarchy, thresholds: dict, top_k: int = 5) -> dict:
    """Report the most specific answer the probabilities support, otherwise abstain."""
    order = np.argsort(probs)[::-1][:top_k]
    top = [{**hierarchy.classes[i], "probability": float(probs[i])} for i in order]
    best = top[0]
    model_probs = hierarchy.rollup(probs, "model")
    make_probs = hierarchy.rollup(probs, "make")
    model_index, make_index = int(model_probs.argmax()), int(make_probs.argmax())
    make, model = hierarchy.models[model_index]
    result = {"top": top, "class_confidence": best["probability"],
              "model_confidence": float(model_probs[model_index]),
              "make_confidence": float(make_probs[make_index])}
    if best["probability"] >= thresholds["full"]:
        return {**result, "level": "full", "confidence": best["probability"],
                "label": best["name"], "make": best["make"], "model": best["model"], "year": best["year"]}
    if result["model_confidence"] >= thresholds["model"]:
        years = sorted(c["year"] for i, c in enumerate(hierarchy.classes)
                       if hierarchy.model_of[i] == model_index and probs[i] >= 0.05)
        year_text = " / ".join(map(str, years)) if years else "uncertain"
        return {**result, "level": "model", "confidence": result["model_confidence"],
                "label": f"{make} {model} (year: {year_text})", "make": make, "model": model, "year": None}
    if result["make_confidence"] >= thresholds["make"]:
        name = hierarchy.makes[make_index]
        return {**result, "level": "make", "confidence": result["make_confidence"],
                "label": f"{name} (model uncertain)", "make": name, "model": None, "year": None}
    return {**result, "level": "abstain", "confidence": best["probability"],
            "label": "Unrecognised: possibly outside the 196 known classes",
            "make": None, "model": None, "year": None}
