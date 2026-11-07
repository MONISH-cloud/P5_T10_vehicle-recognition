"""Report and slide figures. Reads only results/metrics.json, checkpoints/classifier/log.csv and the
failure images named in docs/ABLATION_AND_FAILURES.md. Writes PNGs to docs/figures/ at 2x resolution.

Run: python scripts/10_figures.py   (needs matplotlib, which the app does not)
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
M = json.loads((ROOT / "results" / "metrics.json").read_text())

# One style for every figure: reference palette slots 1 and 2, recessive grid, text in ink colours.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6,
})
DPI = 200  # 2x a 100-dpi base size


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def pct(x):
    return 100 * x


# 1. Training curve ---------------------------------------------------------------------------------
rows = [r for r in csv.reader((ROOT / "checkpoints" / "classifier" / "log.csv").open()) if r]
epoch = [int(r[0]) for r in rows]
top1 = [pct(float(r[2])) for r in rows]
top5 = [pct(float(r[3])) for r in rows]
best = max(range(len(top1)), key=top1.__getitem__)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(epoch, top5, color=ORANGE, marker="o", label="Top-5")
ax.plot(epoch, top1, color=BLUE, marker="o", label="Top-1")
ax.axvline(epoch[best], color=INK2, linestyle=":", linewidth=1.2)
ax.annotate(f"Best epoch {epoch[best]}\ntop-1 {top1[best]:.2f}%, top-5 {top5[best]:.2f}%",
            xy=(epoch[best], top1[best]), xytext=(epoch[best] - 6.6, 45),
            arrowprops=dict(arrowstyle="-", color=INK2, linewidth=1), color=INK, fontsize=10)
ax.set(xlabel="Epoch", ylabel="Validation accuracy (%)", ylim=(0, 100), xticks=epoch,
       title="Classifier training: validation accuracy by epoch")
ax.legend(loc="lower right")
save(fig, "training_curve.png")

# 2. Approach comparison ----------------------------------------------------------------------------
names = [("Classical only", "classical_only"), ("Deep learning only", "deep_only"), ("Hybrid", "hybrid")]
c = M["comparison"]
x = range(len(names))
w = 0.36
fig, ax = plt.subplots(figsize=(7, 4))
b1 = ax.bar([i - w / 2 - 0.01 for i in x], [pct(c[k]["top1"]) for _, k in names], w, color=BLUE, label="Top-1")
b5 = ax.bar([i + w / 2 + 0.01 for i in x], [pct(c[k]["top5"]) for _, k in names], w, color=ORANGE, label="Top-5")
for bars in (b1, b5):
    ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=10, color=INK)
ax.set_xticks(list(x), [n for n, _ in names])
ax.set(ylabel="Test accuracy (%)", ylim=(0, 110), yticks=range(0, 101, 20),
       title=f"Classical vs deep vs hybrid, {M['test_images']:,} unseen test images")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left")
save(fig, "comparison.png")

# 3. Robustness: small multiples, one per condition -------------------------------------------------
r = M["robustness"]
conds = [("low_light", "Low light"), ("overexposure", "Overexposure"), ("motion_blur", "Motion blur"),
         ("low_resolution", "Low resolution"), ("occlusion", "Occlusion"), ("jpeg", "JPEG damage"),
         ("viewpoint_proxy", "Perspective warp\n(viewpoint proxy)")]
fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharex=True, sharey=True)
levels = [0, 1, 2, 3]
for ax, (key, label) in zip(axes.flat, conds):
    deep = [pct(r["clean"]["deep_top1"])] + [pct(r["conditions"][f"{key}/{s}"]["deep_top1"]) for s in (1, 2, 3)]
    hyb = [pct(r["clean"]["hybrid_top1"])] + [pct(r["conditions"][f"{key}/{s}"]["hybrid_top1"]) for s in (1, 2, 3)]
    ax.plot(levels, deep, color=BLUE, marker="o", label="Deep only")
    ax.plot(levels, hyb, color=ORANGE, marker="s", label="Hybrid")
    ax.set_title(label, fontsize=11)
    ax.set_xticks(levels, ["Clean", "1", "2", "3"])
    ax.set_ylim(0, 100)
for ax in axes[1]:
    ax.set_xlabel("Severity level")
for ax in axes[:, 0]:
    ax.set_ylabel("Top-1 accuracy (%)")
axes[0, 3].tick_params(labelbottom=True)
axes[0, 3].set_xlabel("Severity level")
legend_ax = axes.flat[-1]
legend_ax.axis("off")
handles, labels = axes.flat[0].get_legend_handles_labels()
legend_ax.legend(handles, labels, loc="center", fontsize=11)
legend_ax.text(0.5, 0.18, f"{r['images']} fixed test images\nper condition and level", ha="center",
               transform=legend_ax.transAxes, color=INK2, fontsize=10)
fig.suptitle("Robustness: deep only vs hybrid, top-1 under seven conditions", x=0.01, ha="left",
             fontsize=13, fontweight="bold")
fig.tight_layout()
save(fig, "robustness.png")

# 4. Confidence: images answered vs accuracy --------------------------------------------------------
conf = M["confidence"]
curve = conf["threshold_curve"]
cov = [pct(p["coverage"]) for p in curve]
acc = [pct(p["accuracy"]) for p in curve]
used = conf["thresholds_used"]["full"]
full = conf["decision_levels"]["full"]
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(cov, acc, color=BLUE, marker="o", markersize=4, label="Threshold 0.00 to 0.95, steps of 0.05")
for p, xv, yv in zip(curve, cov, acc):
    if round(p["threshold"], 2) in (0.0, 0.5, 0.9, 0.95):
        ax.annotate(f"{p['threshold']:.2f}", (xv, yv), textcoords="offset points", xytext=(6, -12),
                    fontsize=9, color=INK2)
ax.plot([pct(full["share"])], [pct(full["reported_answer_correct"])], marker="o", markersize=10,
        color=ORANGE, markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none",
        label=f"Threshold used: {used:.2f}", zorder=5)
ax.annotate(f"{used:.2f} (used)\n{pct(full['share']):.2f}% answered fully,\n{pct(full['reported_answer_correct']):.2f}% correct",
            (pct(full["share"]), pct(full["reported_answer_correct"])), textcoords="offset points",
            xytext=(-150, 8), fontsize=10, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, linewidth=1))
ax.invert_xaxis()
ax.set(xlabel="Images answered at full confidence (%)", ylabel="Accuracy on answered images (%)",
       title=f"Confidence threshold trade-off (ECE {conf['ece']:.4f})")
ax.legend(loc="lower right", fontsize=9)
save(fig, "confidence.png")

# 5. Failure cases: the three named in docs/ABLATION_AND_FAILURES.md -------------------------------
examples = {Path(e["path"]).name: e for group in M["failures"]["examples"].values() for e in group}
cases = [("Case 1: unusual viewpoint", "wrong_or_missed_box/00_00353.jpg"),
         ("Case 2: probable label error", "wrong_make/03_07470.jpg"),
         ("Case 3: wrong model year", "year_only_confusion/02_02012.jpg")]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
for ax, (title, rel) in zip(axes, cases):
    e = examples[rel.split("/")[1].split("_", 1)[1]]  # "00_00353.jpg" -> "00353.jpg"
    img = Image.open(ROOT / "results" / "failures" / rel).convert("RGB")
    ax.imshow(img)
    ax.set_anchor("N")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(f"True: {e['truth']}\nPredicted: {e['predicted']}\nConfidence: {e['confidence']:.2f}"
                  f"  ·  test image {Path(e['path']).name}", fontsize=9.5, color=INK, linespacing=1.5)
fig.suptitle("Three analysed failure cases", x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout()
save(fig, "failure_cases.png")
