"""Markdown reports. Every number comes from metrics.json; missing data raises instead of being filled in."""
from __future__ import annotations


def _pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _iou(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _ms(value) -> str:
    return f"{value:,.0f} ms"


def _required(results: dict, *keys):
    missing = [k for k in keys if k not in results]
    if missing:
        raise ValueError(f"metrics.json is missing {missing}. Finish scripts/07_evaluate.py first.")


def eval_markdown(results: dict, usns: str) -> str:
    _required(results, "comparison", "detection", "efficiency", "robustness", "confidence", "deployed_vs_fp32", "subgroups")
    usn_line = ",".join(u.strip() for u in usns.split(",") if u.strip())
    if not usn_line:
        raise ValueError("At least one USN is required.")
    comparison, detection, efficiency = results["comparison"], results["detection"], results["efficiency"]
    hybrid, robustness, confidence = comparison["hybrid"], results["robustness"], results["confidence"]
    deployed = results["deployed_vs_fp32"]
    lines = [usn_line, "", "# Evaluation: P5 Fine-Grained Vehicle Recognition", ""]
    if results.get("partial_run"):
        lines += ["> **PARTIAL RUN: smoke-test numbers, not valid for submission.**", ""]
    lines += [f"Measured on {results['test_images']} unseen test images. Model: `{results['classifier']}`, detector: `{results['detector_file']}`.", "",
              "## 1. Classification performance", "",
              "| Approach | Top-1 | Top-5 | Mean per class | Make | Make + model |", "|---|---|---|---|---|---|"]
    for name, key in (("Classical only", "classical_only"), ("Deep learning only", "deep_only"), ("Hybrid", "hybrid")):
        row = comparison[key]
        lines.append(f"| {name} | {_pct(row['top1'])} | {_pct(row['top5'])} | {_pct(row['mean_per_class'])} | {_pct(row['make_accuracy'])} | {_pct(row['make_model_accuracy'])} |")
    lines += ["", "These are end-to-end results: boxes come from the model, not from the annotations.", "",
              "## 2. Object localization (IoU)", "",
              f"- Mean IoU: {detection['mean_iou']:.4f}", f"- Median IoU: {detection['median_iou']:.4f}",
              f"- Images with IoU of at least 0.5: {_pct(detection['iou_at_least_0_5'])}", f"- Images with no usable detection: {_pct(detection['no_detection_rate'])}", "",
              "## 3. Object detection (mAP)", "", f"- AP@0.5: {detection['ap50']:.4f}", f"- mAP@0.5:0.95: {detection['map50_95']:.4f}", "",
              "Stanford Cars annotates one car per image, so correct detections of background cars count as false positives.", "",
              "## 4. Model efficiency", "", "| Model file | Size | Mean CPU latency | P95 CPU latency |", "|---|---|---|---|"]
    for name, row in efficiency["models"].items():
        lines.append(f"| {name} | {row['size_mb']:.1f} MB | {_ms(row['cpu_latency']['mean_ms'])} | {_ms(row['cpu_latency']['p95_ms'])} |")
    lines += ["", f"Deployed classifier: `{efficiency['deployed_classifier']}`. End-to-end CPU time per image: mean {_ms(efficiency['end_to_end_cpu']['mean_ms'])}, P95 {_ms(efficiency['end_to_end_cpu']['p95_ms'])}.", "",
              f"On the same {deployed['images']} test images, top-1 was {_pct(deployed['fp32']['top1'])} before optimisation and {_pct(deployed['deployed']['top1'])} for the deployed model.", "",
              f"## 5. Robustness ({robustness['images']} test images)", "",
              f"Clean: deep top-1 {_pct(robustness['clean']['deep_top1'])}, IoU {_iou(robustness['clean']['deep_iou'])}; hybrid top-1 {_pct(robustness['clean']['hybrid_top1'])}, IoU {_iou(robustness['clean']['hybrid_iou'])}.", "",
              "| Condition / severity | Deep top-1 | Hybrid top-1 | Deep IoU | Hybrid IoU |", "|---|---|---|---|---|"]
    for name, row in robustness["conditions"].items():
        lines.append(f"| {name} | {_pct(row['deep_top1'])} | {_pct(row['hybrid_top1'])} | {_iou(row['deep_iou'])} | {_iou(row['hybrid_iou'])} |")
    lines += ["", "`viewpoint_proxy` is a synthetic perspective warp, because the dataset has no viewpoint labels. It moves the car, so IoU is not measured for it.", "",
              "Hybrid top-1 on real, uncorrupted test images, by group. Box shape is only a rough proxy for viewing angle.", "",
              "| Group | Value | Images | Top-1 |", "|---|---|---|---|"]
    for group, members in results["subgroups"].items():
        for key, row in members.items():
            lines.append(f"| {group} | {key} | {row['images']} | {_pct(row['top1'])} |")
    lines += ["",
              "## 6. Confidence and threshold analysis", "", f"Expected calibration error: {confidence['ece']:.4f}. Thresholds were chosen on validation data: {confidence['thresholds_used']}.", "",
              "| Answer level | Share of images | Reported answer correct |", "|---|---|---|"]
    for name, row in confidence["decision_levels"].items():
        lines.append(f"| {name} | {_pct(row['share'])} | {_pct(row['reported_answer_correct'])} |")
    lines += ["", "| Confidence threshold | Images answered | Accuracy on answered |", "|---|---|---|"]
    for row in confidence["threshold_curve"]:
        lines.append(f"| {row['threshold']:.2f} | {_pct(row['coverage'])} | {_pct(row['accuracy'])} |")
    deep_only, classical_only = comparison["deep_only"], comparison["classical_only"]
    original = efficiency["models"].get("classifier_fp32.onnx")
    final = efficiency["models"].get(efficiency["deployed_classifier"])
    lines += ["", "## Measured change", "", f"- Hybrid top-1 against deep only: {(hybrid['top1'] - deep_only['top1']) * 100:+.2f} points.",
              f"- Hybrid top-1 against classical only: {(hybrid['top1'] - classical_only['top1']) * 100:+.2f} points.",
              f"- Top-1 after optimisation, same images: {(deployed['deployed']['top1'] - deployed['fp32']['top1']) * 100:+.2f} points."]
    if original and final:
        lines += [f"- Classifier size after optimisation: {original['size_mb']:.1f} MB to {final['size_mb']:.1f} MB.",
                  f"- Mean classifier CPU latency after optimisation: {original['cpu_latency']['mean_ms']:,.0f} ms to {final['cpu_latency']['mean_ms']:,.0f} ms."]
    return "\n".join(lines) + "\n"


def ablation_markdown(results: dict) -> str:
    _required(results, "ablation", "failures")
    failures = results["failures"]
    baseline = results["ablation"]["Detector crop (deep-only baseline)"]["top1"]
    full = results["ablation"]["Full hybrid"]["top1"]
    images = results.get("ablation_images", results["test_images"])
    scope = "all" if images == results["test_images"] else f"a fixed random subset of {images} of the"
    lines = ["# Ablation Study and Failure-Case Analysis", "", f"Ablation measured on {scope} {results['test_images']} unseen test images. Failure analysis uses all of them.", "", "## Ablation study", "",
             "| Pipeline | Top-1 | Top-5 | Change from deep-only |", "|---|---|---|---|"]
    for name, row in results["ablation"].items():
        lines.append(f"| {name} | {_pct(row['top1'])} | {_pct(row['top5'])} | {(row['top1'] - baseline) * 100:+.2f} pts |")
    lines += ["", f"Full hybrid against deep-only: {(full - baseline) * 100:+.2f} points. A component with no positive effect is reported as such.", "",
              "## Error profile", "", f"{failures['errors']} of {failures['images']} images were misclassified.", "", "| Tag | Errors |", "|---|---|"]
    for tag, count in failures["tag_counts"].items():
        lines.append(f"| {tag} | {count} |")
    lines += ["", "Tags can overlap. Most frequent confusions:", ""]
    for pair, count in failures["top_confusions"]:
        lines.append(f"- {pair}: {count}")
    lines += ["", "## Failure cases", "", "Example images are in `results/failures/<tag>/`.", ""]
    for number in (1, 2, 3):
        lines += [f"### Failure case {number}", "", "TEAM TO EXPLAIN: image, true and predicted class, confidence, the cause you found, and a possible fix.", ""]
    return "\n".join(lines)
