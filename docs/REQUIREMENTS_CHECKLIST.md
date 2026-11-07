# P5 Requirements Checklist

"Code" means written and covered by tests or the plan's smoke test. "Result" items exist only after the real run.

| Brief requirement | Where it is |
|---|---|
| Preprocessing and normalization | `classical.quality_metrics`, `classical.adaptive_enhance`, `pipeline.classifier_tensor` |
| Edge detection and contour analysis | `classical.auto_canny`, `classical.contour_roi`, `classical.feature_views` |
| Colour and texture features | `classical.extract_features` (HSV histogram, LBP), `classical.feature_views` |
| Region-of-interest extraction | `classical.contour_roi`, `classical.refine_box`, `classical.crop` |
| Geometric feature analysis | `classical.geometric_features`, `classical.geometry_summary` |
| CNN or ViT classification | `scripts/03_train_classifier.py` (ConvNeXt V2) |
| YOLO or R-CNN detection | `scripts/04_train_detector.py`, `pipeline.OnnxDetector` (YOLO11) |
| Transfer learning and fine-tuning | `scripts/03_train_classifier.py`, `scripts/04_train_detector.py` |
| 196-class classification | `scripts/03_train_classifier.py` |
| Classical and deep integration | `pipeline.VehiclePipeline`, `decision.fuse` |
| Detected regions used for classification | `VehiclePipeline.run` |
| Boxes with labels and confidence | `pipeline.annotate`, `app.py` |
| Low-confidence handling | `decision.decide`, `decision.fit_temperature` |
| Similar makes, models and years | Fine-grained training; make and make + model accuracy; year-only and same-make failure tags; top confusions |
| Illumination and viewing angles | Low-light and overexposure tests with accuracy and IoU; brightness groups; perspective warp and box-shape groups (both proxies) |
| Occlusion and resolutions | Occlusion and low-resolution tests; car-size groups |
| Classical, deep and hybrid comparison | `evaluate_ablation`, `results["comparison"]`, app Compare tab |
| Ablation analysis | `evaluate_ablation` (12 variants), `docs/ABLATION_AND_FAILURES.md` |
| Model optimization | `scripts/05_export.py` (int8 quantization) |
| Unseen test images | `scripts/07_evaluate.py`; test split unused for choices |
| Three failure cases | `mine_failures` finds them. **The team must write the explanations.** |
| Top-1 and Top-5, IoU, mAP | `metrics.py`, `eval_P5_<TeamID>.md` sections 1 to 3 |
| Size and inference time | Section 4 |
| Robustness | Section 5 |
| Confidence and thresholds | Section 6 |
| Upload image | `app.py` |
| Bounding box | `pipeline.annotate` |
| Make, model and year | `app.vehicle_card` |
| Confidence | `app.vehicle_card` |
| Intermediate results | App Processing steps tab |
| Classical, deep and hybrid comparison in the app | App Compare tab |
| Measurable accuracy and efficiency change | "Measured change" section of the eval file. **The direction depends on real results.** |
