import sys
from pathlib import Path
import cv2
import numpy as np
import pytest
import scipy.io as sio
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CLASS_NAMES = ["Audi S4 Sedan 2007", "Audi S4 Sedan 2012", "AM General Hummer SUV 2000", "Aston Martin V8 Vantage Coupe 2012",
               "Ram C/V Cargo Van Minivan 2012", "Mercedes-Benz SL-Class Coupe 2009", "Land Rover LR2 SUV 2012", "smart fortwo Convertible 2012"]


def synthetic_car(rng, label, width=320, height=240):
    image = np.full((height, width, 3), rng.integers(170, 215), np.uint8)
    image = np.clip(image.astype(int) + rng.normal(0, 3, image.shape), 0, 255).astype(np.uint8)
    x1, y1 = int(rng.integers(20, 60)), int(rng.integers(40, 70))
    x2, y2 = int(rng.integers(230, 300)), int(rng.integers(170, 220))
    hue = (label * 22, 200, 170)
    colour = cv2.cvtColor(np.uint8([[hue]]), cv2.COLOR_HSV2RGB)[0, 0].tolist()
    cv2.rectangle(image, (x1, y1 + 40), (x2, y2), colour, -1)
    cv2.rectangle(image, (x1 + 30, y1), (x2 - 30 - label * 3, y1 + 45), colour, -1)
    cv2.circle(image, (x1 + 35, y2), 16, (20, 20, 20), -1)
    cv2.circle(image, (x2 - 35, y2), 16, (20, 20, 20), -1)
    return image, (x1, y1, x2, min(height - 1, y2 + 16))


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    """A miniature copy of the Stanford Cars file layout, with synthetic pictures."""
    root = tmp_path_factory.mktemp("cars") / "stanford_cars"
    rng = np.random.default_rng(0)
    (root / "devkit").mkdir(parents=True)
    dtype = [("bbox_x1", "O"), ("bbox_y1", "O"), ("bbox_x2", "O"), ("bbox_y2", "O"), ("class", "O"), ("fname", "O")]
    for folder, per_class, mat in (("cars_train", 12, root / "devkit/cars_train_annos.mat"), ("cars_test", 5, root / "cars_test_annos_withlabels.mat")):
        (root / folder).mkdir()
        records = []
        number = 0
        for label in range(len(CLASS_NAMES)):
            for _ in range(per_class):
                number += 1
                image, box = synthetic_car(rng, label)
                name = f"{number:05d}.jpg"
                cv2.imwrite(str(root / folder / name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                records.append((box[0] + 1, box[1] + 1, box[2], box[3], label + 1, name))
        sio.savemat(mat, {"annotations": np.array(records, dtype=dtype)})
    sio.savemat(root / "devkit/cars_meta.mat", {"class_names": np.array(CLASS_NAMES, dtype=object)})
    return root


class FakeDetector:
    """Returns the region that differs from the plain background."""

    def __call__(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask = np.abs(gray.astype(int) - int(np.median(gray))) > 25
        if mask.mean() < 0.02:
            return []
        ys, xs = np.where(mask)
        return [{"box": (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), "score": 0.9}]


class FakeClassifier:
    """Reads the car's hue, so results are deterministic and exercise the whole flow."""

    def __call__(self, region, flip=False):
        hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        pixels = hsv[hsv[..., 1] > 100]
        logits = np.zeros(len(CLASS_NAMES))
        if len(pixels):
            logits[int(round(np.median(pixels[:, 0]) / 22)) % len(CLASS_NAMES)] = 8.0
        return logits


@pytest.fixture
def fakes():
    return FakeDetector(), FakeClassifier()
