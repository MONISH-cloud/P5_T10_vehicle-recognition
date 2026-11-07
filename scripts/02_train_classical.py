"""Classical baseline: HOG, LBP, colour and geometry features with softmax regression.

Trains on train images, tunes C on validation, and writes plain arrays to models/classical.npz.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vehicle_id import classical
from vehicle_id.data import read_manifest
from vehicle_id.evaluation import gt_box, read_rgb
from vehicle_id.metrics import topk_accuracy

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", default="data/manifest.csv")
parser.add_argument("--out", default="models/classical.npz")
parser.add_argument("--pca", type=int, default=256)
parser.add_argument("--jobs", type=int, default=-1)
args = parser.parse_args()


def features(row):
    image = read_rgb(row["path"])
    box = classical.expand_box(gt_box(row), image.shape[1], image.shape[0])
    return classical.extract_features(classical.crop(image, box), box, image.shape)


def matrix(rows):
    return np.stack(Parallel(n_jobs=args.jobs)(delayed(features)(r) for r in rows)), np.array([r["label"] for r in rows])


X_train, y_train = matrix(read_manifest(args.manifest, "train"))
X_val, y_val = matrix(read_manifest(args.manifest, "val"))
scaler = StandardScaler().fit(X_train)
n_components = min(args.pca, X_train.shape[0] - 1, X_train.shape[1])
pca = PCA(n_components, random_state=42).fit(scaler.transform(X_train))
Z_train, Z_val = pca.transform(scaler.transform(X_train)), pca.transform(scaler.transform(X_val))
best, best_score = None, -1.0
for c in (0.003, 0.01, 0.03, 0.1, 0.3):
    model = LogisticRegression(C=c, max_iter=300).fit(Z_train, y_train)
    score = topk_accuracy(model.predict_proba(Z_val), y_val, 1)
    print(f"C={c}: validation top-1 {score:.4f}")
    if score > best_score:
        best, best_score = model, score
if best.coef_.shape[0] != len(json.load(open(Path(args.manifest).parent / "classes.json"))):
    raise SystemExit("Some classes have no training images: the classical model is invalid.")
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(args.out, mean=scaler.mean_, scale=scaler.scale_, pca_mean=pca.mean_, components=pca.components_,
                    coef=best.coef_, intercept=best.intercept_)
print(f"Saved {args.out}; validation top-1 {best_score:.4f}")
