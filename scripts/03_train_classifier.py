"""Fine-tune a timm classifier on vehicle crops. Resumable, for Kaggle or Colab sessions.

Validation drives all choices; this script never reads the test split.
"""
import argparse, csv, gc, math, os, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
import numpy as np
import timm
import torch
from PIL import Image
from timm.data import Mixup, create_transform, resolve_model_data_config
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV3
from torch.utils.data import DataLoader, Dataset
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vehicle_id.classical import expand_box
from vehicle_id.data import read_manifest
from vehicle_id.pipeline import classifier_tensor
from vehicle_id.training_utils import accumulation_for, pick_batch_size

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", default="data/manifest.csv")
parser.add_argument("--model", default="convnextv2_base.fcmae_ft_in22k_in1k_384")
parser.add_argument("--out", default="checkpoints/classifier")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch-size", type=int, default=20)
parser.add_argument("--accumulate", type=int, default=2)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--layer-decay", type=float, default=0.7)
parser.add_argument("--weight-decay", type=float, default=0.05)
parser.add_argument("--drop-path", type=float, default=0.2)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--max-minutes", type=float, default=0, help="Stop cleanly after this many minutes (0 = no limit)")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)


class Crops(Dataset):
    """Train crops get random box jitter, so the classifier tolerates imperfect detector boxes."""

    def __init__(self, rows, transform=None, config=None):
        self.rows, self.transform, self.config = rows, transform, config

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as handle:
            image = handle.convert("RGB")
        box = (row["x1"], row["y1"], row["x2"], row["y2"])
        if self.transform:
            w, h = box[2] - box[0], box[3] - box[1]
            shift = np.random.uniform(-0.05, 0.12, 4) * np.array([w, h, w, h])
            box = (box[0] - shift[0], box[1] - shift[1], box[2] + shift[2], box[3] + shift[3])
            x1, y1, x2, y2 = expand_box(box, image.width, image.height, 0.0)
            return self.transform(image.crop((x1, y1, x2, y2))), row["label"]
        x1, y1, x2, y2 = expand_box(box, image.width, image.height, 0.08)
        tensor = classifier_tensor(np.asarray(image.crop((x1, y1, x2, y2))), self.config["input_size"][1], self.config["mean"], self.config["std"])
        return torch.from_numpy(tensor), row["label"]


model = timm.create_model(args.model, pretrained=True, num_classes=196, drop_path_rate=args.drop_path).to(device)
config = resolve_model_data_config(model)


def try_batch(size):
    """Run one forward and backward pass on random images, and report peak GPU memory."""
    fraction = None
    try:
        model.train()
        torch.cuda.reset_peak_memory_stats()
        images = torch.randn(size, *config["input_size"], device=device)
        with torch.amp.autocast(device):
            loss = model(images).mean()
        loss.backward()
        fraction = torch.cuda.max_memory_reserved() / torch.cuda.get_device_properties(0).total_memory
    except torch.OutOfMemoryError:
        pass
    images = loss = None
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    return fraction


if device == "cuda":
    effective_batch = args.batch_size * args.accumulate
    args.batch_size = pick_batch_size(args.batch_size, try_batch)
    args.accumulate = accumulation_for(args.batch_size, effective_batch)
    print(f"Using batch size {args.batch_size}, accumulation {args.accumulate}", flush=True)
train_transform = create_transform(input_size=config["input_size"], is_training=True, scale=(0.75, 1.0), ratio=(0.8, 1.25),
                                   hflip=0.5, color_jitter=0.2, auto_augment="rand-m7-mstd0.5-inc1",
                                   interpolation="bicubic", mean=config["mean"], std=config["std"],
                                   re_prob=0.25, re_mode="pixel", re_count=1)
train_rows = read_manifest(args.manifest, "train")
val_rows = read_manifest(args.manifest, "val")
if len({r["label"] for r in train_rows}) != 196:
    raise SystemExit("Training rows do not cover all 196 classes.")
train_loader = DataLoader(Crops(train_rows, train_transform), args.batch_size, shuffle=True, num_workers=args.workers,
                          pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
val_loader = DataLoader(Crops(val_rows, config=config), args.batch_size * 2, num_workers=args.workers, pin_memory=True)


def stage_of(name: str) -> int:
    if name.startswith("stem"):
        return 0
    if name.startswith("stages."):
        return int(name.split(".")[1]) + 1
    return 5


groups = {}
for name, parameter in model.named_parameters():
    decay = 0.0 if parameter.ndim <= 1 else args.weight_decay
    key = (stage_of(name), decay)
    groups.setdefault(key, {"params": [], "weight_decay": decay, "lr": args.lr * args.layer_decay ** (5 - key[0])})["params"].append(parameter)
optimizer = torch.optim.AdamW(list(groups.values()))
steps_per_epoch = max(1, len(train_loader) // args.accumulate)
total_steps, warmup_steps = args.epochs * steps_per_epoch, 2 * steps_per_epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / warmup_steps) * (
    1.0 if step < warmup_steps else 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))))
scaler = torch.amp.GradScaler(device, enabled=device == "cuda")
mixup = Mixup(mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.8, label_smoothing=0.1, num_classes=196)
criterion = SoftTargetCrossEntropy()
ema = ModelEmaV3(model, decay=0.999, device=device)
start_epoch, best_top1 = 0, 0.0
last_path, best_path = out / "last.pt", out / "best.pt"
if last_path.exists():
    state = torch.load(last_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    ema.module.load_state_dict(state["ema"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    start_epoch, best_top1 = state["epoch"] + 1, state["best_top1"]
    print(f"Resumed at epoch {start_epoch}, best validation top-1 {best_top1:.4f}")


@torch.no_grad()
def validate(network):
    network.eval()
    top1 = top5 = seen = 0
    for images, labels in val_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.amp.autocast(device, enabled=device == "cuda"):
            logits = network(images)
        best = logits.topk(5, dim=1).indices
        top1 += (best[:, 0] == labels).sum().item()
        top5 += (best == labels[:, None]).any(dim=1).sum().item()
        seen += labels.size(0)
    return top1 / seen, top5 / seen


began = time.time()
log_path = out / "log.csv"
for epoch in range(start_epoch, args.epochs):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running, batches = 0.0, 0
    for step, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        images, targets = mixup(images, labels)
        with torch.amp.autocast(device, enabled=device == "cuda"):
            loss = criterion(model(images), targets) / args.accumulate
        scaler.scale(loss).backward()
        running += loss.item() * args.accumulate
        batches += 1
        if (step + 1) % args.accumulate == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            ema.update(model)
    top1, top5 = validate(ema.module)
    print(f"epoch {epoch + 1}/{args.epochs} loss {running / max(batches, 1):.3f} val_top1 {top1:.4f} val_top5 {top5:.4f} "
          f"elapsed {(time.time() - began) / 60:.1f} min", flush=True)
    with open(log_path, "a", newline="") as handle:
        csv.writer(handle).writerow([epoch + 1, running / max(batches, 1), top1, top5])
    if top1 > best_top1:
        best_top1 = top1
        torch.save({"model": ema.module.state_dict(), "name": args.model, "config": config, "val_top1": top1, "val_top5": top5}, best_path)
    torch.save({"model": model.state_dict(), "ema": ema.module.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch, "best_top1": best_top1}, last_path)
    if args.max_minutes and (time.time() - began) / 60 > args.max_minutes:
        print("Time budget reached. Run the same command again in a new session to resume.")
        break
print(f"Best validation top-1: {best_top1:.4f}")
