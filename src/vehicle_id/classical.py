"""Classical computer vision. All functions take RGB uint8 arrays."""
from __future__ import annotations
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

Box = tuple[int, int, int, int]
FEATURE_SIZE = 128


def clip_box(box, width: int, height: int) -> Box:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, x2 = sorted((min(max(x1, 0), width - 1), min(max(x2, 0), width)))
    y1, y2 = sorted((min(max(y1, 0), height - 1), min(max(y2, 0), height)))
    return x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)


def expand_box(box, width: int, height: int, margin: float = 0.08) -> Box:
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    return clip_box((x1 - dx, y1 - dy, x2 + dx, y2 + dy), width, height)


def crop(image: np.ndarray, box) -> np.ndarray:
    x1, y1, x2, y2 = clip_box(box, image.shape[1], image.shape[0])
    return image[y1:y2, x1:x2]


def box_iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def quality_metrics(image: np.ndarray) -> dict[str, float]:
    """Cheap measurements that drive the adaptive pre-processing decisions."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return {
        "brightness": float(np.median(gray) / 255.0),
        "contrast": float(gray.std() / 255.0),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "dark_fraction": float((gray < 30).mean()),
        "bright_fraction": float((gray > 240).mean()),
        "min_side": float(min(image.shape[:2])),
    }


def adaptive_enhance(image: np.ndarray, quality: dict | None = None):
    """Correct only what is measurably wrong; a good image is returned unchanged."""
    quality = quality or quality_metrics(image)
    out, steps = image, []
    if quality["brightness"] < 0.25:
        gamma = float(np.clip(np.log(0.45) / np.log(max(quality["brightness"], 0.02)), 0.3, 1.0))
        table = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
        out = cv2.LUT(out, table)
        steps.append(f"gamma={gamma:.2f}")
        out = cv2.bilateralFilter(out, 5, 30, 5)
        steps.append("bilateral denoise")
    elif quality["brightness"] > 0.8 or quality["bright_fraction"] > 0.4:
        table = (np.linspace(0, 1, 256) ** 1.6 * 255).astype(np.uint8)
        out = cv2.LUT(out, table)
        steps.append("gamma=1.60")
    if quality["contrast"] < 0.14 or steps:
        lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[..., 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        steps.append("CLAHE")
    if quality["sharpness"] < 40:
        blurred = cv2.GaussianBlur(out, (0, 0), 1.5)
        out = cv2.addWeighted(out, 1.5, blurred, -0.5, 0)
        steps.append("unsharp")
    return out, steps


def auto_canny(image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    median = float(np.median(gray))
    return cv2.Canny(gray, int(max(0, (1 - sigma) * median)), int(min(255, (1 + sigma) * median)))


def contour_roi(image: np.ndarray) -> Box:
    """Detector-free region of interest, also the ROI for the classical-only baseline."""
    height, width = image.shape[:2]
    scale = 320 / max(height, width)
    small = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
    scale = min(scale, 1.0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(auto_canny(small), cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = small.shape[0] * small.shape[1]
    best, best_score = None, 0.0
    centre = np.array([small.shape[1] / 2, small.shape[0] / 2])
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < 0.08 * frame_area or w * h > 0.98 * frame_area or not 0.5 < w / h < 4.0:
            continue
        distance = np.linalg.norm(np.array([x + w / 2, y + h / 2]) - centre) / np.linalg.norm(centre)
        score = (w * h / frame_area) * (1 - 0.5 * distance)
        if score > best_score:
            best, best_score = (x, y, x + w, y + h), score
    if best is None:
        return clip_box((width * .1, height * .1, width * .9, height * .9), width, height)
    return clip_box([v / scale for v in best], width, height)


def refine_box(image: np.ndarray, box, min_iou: float = 0.6) -> Box:
    """Tighten a detector box with GrabCut; keep the original if the result disagrees."""
    height, width = image.shape[:2]
    original = clip_box(box, width, height)
    scale = min(1.0, 256 / max(height, width))
    small = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
    x1, y1, x2, y2 = [int(v * scale) for v in original]
    if x2 - x1 < 16 or y2 - y1 < 16:
        return original
    mask = np.zeros(small.shape[:2], np.uint8)
    rect = (x1, y1, x2 - x1, min(y2 - y1, small.shape[0] - y1))
    try:
        cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), mask, rect, np.zeros((1, 65)),
                    np.zeros((1, 65)), 2, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return original
    ys, xs = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD))
    if len(xs) < 64:
        return original
    refined = clip_box([v / scale for v in (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)], width, height)
    return refined if box_iou(refined, original) >= min_iou else original


def extract_features(region: np.ndarray, box=None, frame_shape=None) -> np.ndarray:
    """Shape, texture, colour and geometry descriptor (fixed length)."""
    resized = cv2.resize(region, (FEATURE_SIZE, FEATURE_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    shape = hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                block_norm="L2-Hys", feature_vector=True)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    texture = []
    for row in np.array_split(lbp, 3, axis=0):
        for cell in np.array_split(row, 3, axis=1):
            hist = np.histogram(cell, bins=10, range=(0, 10))[0].astype(np.float32)
            texture.append(hist / max(hist.sum(), 1))
    hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
    colour = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    colour = colour / max(colour.sum(), 1)
    return np.concatenate([shape, np.concatenate(texture), colour,
                           geometric_features(resized, region.shape, box, frame_shape)]).astype(np.float32)


def geometry_summary(region: np.ndarray, box=None, frame_shape=None) -> dict[str, float]:
    resized = cv2.resize(region, (FEATURE_SIZE, FEATURE_SIZE), interpolation=cv2.INTER_AREA)
    values = geometric_features(resized, region.shape, box, frame_shape)
    return {"aspect_ratio": float(values[0] * 4), "frame_fraction": float(values[1]), "edge_density": float(values[2]),
            "solidity": float(values[3]), "extent": float(values[4])}


def feature_views(region: np.ndarray) -> dict[str, np.ndarray]:
    """Pictures of the classical features, for the app's processing-steps tab."""
    size = 256
    resized = cv2.resize(region, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(auto_canny(resized), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outlined = resized.copy()
    cv2.drawContours(outlined, contours, -1, (0, 220, 120), 1)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cv2.rectangle(outlined, (x, y), (x + w, y + h), (240, 80, 60), 2)
    _, hog_image = hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
    hog_image = np.clip(hog_image / max(float(hog_image.max()), 1e-6) * 255, 0, 255).astype(np.uint8)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_image = np.clip(lbp / 9 * 255, 0, 255).astype(np.uint8)
    chart = np.full((size, size, 3), 255, np.uint8)
    hue = cv2.calcHist([cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)], [0], None, [32], [0, 180]).ravel()
    hue = hue / max(float(hue.max()), 1.0)
    for index, value in enumerate(hue):
        colour = cv2.cvtColor(np.uint8([[[index * 180 // 32, 200, 220]]]), cv2.COLOR_HSV2RGB)[0, 0].tolist()
        cv2.rectangle(chart, (index * 8, size - 1), (index * 8 + 6, size - 1 - int(value * (size - 8))), colour, -1)
    return {"Contour analysis": outlined, "Shape (HOG)": hog_image, "Texture (LBP)": lbp_image, "Colour (hue histogram)": chart}


def geometric_features(resized: np.ndarray, region_shape, box=None, frame_shape=None) -> np.ndarray:
    edges = auto_canny(resized)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solidity = extent = 0.0
    hu = np.zeros(7)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        hull_area = cv2.contourArea(cv2.convexHull(largest))
        x, y, w, h = cv2.boundingRect(largest)
        solidity = area / hull_area if hull_area else 0.0
        extent = area / (w * h) if w * h else 0.0
        raw = cv2.HuMoments(cv2.moments(largest)).ravel()
        hu = -np.sign(raw) * np.log10(np.abs(raw) + 1e-12) / 12.0
    frame_fraction = 0.0
    if box is not None and frame_shape is not None:
        frame_fraction = ((box[2] - box[0]) * (box[3] - box[1])) / (frame_shape[0] * frame_shape[1])
    aspect = region_shape[1] / max(region_shape[0], 1)
    return np.array([aspect / 4.0, frame_fraction, float((edges > 0).mean()), solidity, extent, *hu],
                    dtype=np.float32)
