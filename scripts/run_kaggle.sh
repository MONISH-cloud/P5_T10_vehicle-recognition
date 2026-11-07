#!/usr/bin/env bash
# Runs training and evaluation in order. Enable GPU and Internet in the Kaggle notebook settings.
# Usage: bash scripts/run_kaggle.sh /kaggle/input/<stanford-cars-dataset-folder>
set -euo pipefail
DATA_ROOT="${1:?Pass the dataset folder}"
pip install -q timm ultralytics onnx onnxscript onnxruntime-gpu scikit-image
python scripts/01_prepare_data.py --data-root "$DATA_ROOT"
python scripts/02_train_classical.py
python scripts/04_train_detector.py
python scripts/03_train_classifier.py --max-minutes 420
python scripts/05_export.py
python scripts/06_calibrate.py --gpu
python scripts/07_evaluate.py --gpu
