"""Reproducible stress tests for the robustness metric."""
from __future__ import annotations
import cv2
import numpy as np

SEVERITIES = (1, 2, 3)


def low_light(image, severity, rng):
    scale = {1: 0.5, 2: 0.3, 3: 0.15}[severity]
    noisy = image.astype(np.float32) * scale + rng.normal(0, 4 * severity, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def overexposure(image, severity, rng):
    return np.clip(image.astype(np.float32) * {1: 1.5, 2: 2.0, 3: 3.0}[severity], 0, 255).astype(np.uint8)


def motion_blur(image, severity, rng):
    size = {1: 7, 2: 13, 3: 21}[severity]
    kernel = np.zeros((size, size), np.float32)
    kernel[size // 2] = 1 / size
    return cv2.filter2D(image, -1, kernel)


def low_resolution(image, severity, rng):
    scale = {1: 0.5, 2: 0.25, 3: 0.125}[severity]
    height, width = image.shape[:2]
    small = cv2.resize(image, (max(8, int(width * scale)), max(8, int(height * scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def occlusion(image, severity, rng):
    """Blocks part of the middle of the frame, where the car usually is."""
    fraction = {1: 0.15, 2: 0.3, 3: 0.5}[severity]
    height, width = image.shape[:2]
    h, w = max(1, int(height * fraction ** 0.5)), max(1, int(width * fraction ** 0.5))
    y = int(rng.integers(height // 4, max(height // 4 + 1, height - h - height // 8)))
    x = int(rng.integers(width // 8, max(width // 8 + 1, width - w - width // 8)))
    out = image.copy()
    out[y:y + h, x:x + w] = 90
    return out


def jpeg(image, severity, rng):
    quality = {1: 30, 2: 15, 3: 7}[severity]
    _, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def viewpoint(image, severity, rng):
    """Perspective warp. A proxy for angle change: the dataset has no viewpoint labels."""
    shift = {1: 0.08, 2: 0.15, 3: 0.25}[severity]
    height, width = image.shape[:2]
    source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    target = np.float32([[width * shift, height * shift], [width, 0], [width, height], [width * shift, height * (1 - shift)]])
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(source, target), (width, height), borderMode=cv2.BORDER_REPLICATE)


CORRUPTIONS = {"low_light": low_light, "overexposure": overexposure, "motion_blur": motion_blur,
               "low_resolution": low_resolution, "occlusion": occlusion, "jpeg": jpeg, "viewpoint_proxy": viewpoint}


def apply(name: str, image: np.ndarray, severity: int, seed: int = 0) -> np.ndarray:
    return CORRUPTIONS[name](image, severity, np.random.default_rng(seed))
