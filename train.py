"""
train.py
────────
Training script for PlantVillage disease classification with EfficientNet-B0.

Two-phase training:
  Phase 1 — Freeze backbone, train only the classification head (fast warm-up).
  Phase 2 — Unfreeze all layers, fine-tune with a lower learning rate.

Mixed-precision training (AMP) is used automatically when a CUDA GPU is available.

Usage:
    python train.py                             # default settings
    python train.py --epochs 20 --batch-size 64 --output-dir runs/exp1
    python train.py --config segmented --model efficientnet_b2

Arguments:
    --epochs        Total training epochs (default: 20)
    --phase1-epochs Epochs for head-only warm-up (default: 5)
    --batch-size    Mini-batch size (default: 32; reduce if OOM)
    --lr            Initial learning rate for Phase 1 (default: 1e-3)
    --model         timm model name (default: efficientnet_b0)
    --config        Dataset config: color | grayscale | segmented (default: color)
    --output-dir    Checkpoint and log directory (default: runs/exp1)
    --num-workers   DataLoader workers — use 0 on Windows if issues arise
    --patience      Early-stopping patience in epochs (default: 5)
"""

import json
import os
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from src.dataset import load_plantvillage
from src.model import build_model, freeze_backbone, unfreeze_all, count_parameters


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train PlantVillage classifier")
    p.add_argument("--epochs",        type=int,   default=20)
    p.add_argument("--phase1-epochs", type=int,   default=5,
                   help="Epochs for head-only warm-up (backbone frozen)")
    p.add_argument("--batch-size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=1e-3,
                   help="Learning rate for Phase 1 (Phase 2 uses lr/10)")
    p.add_argument("--model",         type=str,   default="efficientnet_b0")
    p.add_argument("--config",        type=str,   default="color",
                   choices=["color", "grayscale", "segmented"])
    p.add_argument("--output-dir",    type=str,   default="runs/exp1")
    p.add_argument("--num-workers",   type=int,   default=0,
                   help="Set to 0 on Windows to avoid multiprocessing issues")
    p.add_argument("--patience",      type=int,   default=5,
                   help="Early-stopping patience (epochs without val improvement)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Training / evaluation loops
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total_loss = correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(dim=1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        preds       = outputs.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device} | AMP: {use_amp}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds, test_ds, class_names = load_plantvillage(config=args.config)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    with open(os.path.join(args.output_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(num_classes=len(class_names), model_name=args.model).to(device)
    total_p, trainable_p = count_parameters(model)
    print(f"Parameters — total: {total_p:,} | trainable: {trainable_p:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

    history      = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_preds   = []
    best_labels  = []

    # ── Phase 1: Head-only warm-up ────────────────────────────────────────────
    p1 = args.phase1_epochs
    print(f"\n{'='*55}")
    print(f" Phase 1 — head warm-up ({p1} epochs, backbone frozen)")
    print(f"{'='*55}")

    freeze_backbone(model)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=p1, eta_min=1e-5)
    patience_cnt = 0

    for epoch in range(1, p1 + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp)
        va_loss, va_acc, preds, labels = evaluate(
            model, test_loader, criterion, device, use_amp)
        scheduler.step()

        _log_epoch(epoch, p1, tr_loss, tr_acc, va_loss, va_acc, t0)
        history = _update_history(history, tr_loss, tr_acc, va_loss, va_acc)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_preds, best_labels = preds, labels
            _save_checkpoint(model, args.output_dir)
            patience_cnt = 0
        else:
            patience_cnt += 1

    # ── Phase 2: Full fine-tuning ─────────────────────────────────────────────
    p2 = args.epochs - p1
    print(f"\n{'='*55}")
    print(f" Phase 2 — full fine-tuning ({p2} epochs, all layers)")
    print(f"{'='*55}")

    unfreeze_all(model)
    optimizer = AdamW(model.parameters(), lr=args.lr / 10)
    scheduler = CosineAnnealingLR(optimizer, T_max=p2, eta_min=1e-6)
    patience_cnt = 0

    for epoch in range(1, p2 + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp)
        va_loss, va_acc, preds, labels = evaluate(
            model, test_loader, criterion, device, use_amp)
        scheduler.step()

        _log_epoch(epoch, p2, tr_loss, tr_acc, va_loss, va_acc, t0)
        history = _update_history(history, tr_loss, tr_acc, va_loss, va_acc)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_preds, best_labels = preds, labels
            _save_checkpoint(model, args.output_dir)
            patience_cnt = 0
            print(f"    ✓ Best so far: {best_val_acc:.4f}")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"  Early stopping (patience={args.patience})")
                break

    # ── Save outputs ──────────────────────────────────────────────────────────
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    report = classification_report(best_labels, best_preds, target_names=class_names)
    print("\nClassification Report:\n", report)

    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(f"Best val accuracy: {best_val_acc:.4f}\n\n{report}")

    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"Outputs saved to: {args.output_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers (keep main() readable)
# ──────────────────────────────────────────────────────────────────────────────

def _log_epoch(epoch, total, tr_loss, tr_acc, va_loss, va_acc, t0):
    print(
        f"  [{epoch:02d}/{total}] "
        f"loss {tr_loss:.4f}/{va_loss:.4f}  "
        f"acc {tr_acc:.4f}/{va_acc:.4f}  "
        f"({time.time()-t0:.0f}s)"
    )


def _update_history(history, tr_loss, tr_acc, va_loss, va_acc):
    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(va_loss)
    history["val_acc"].append(va_acc)
    return history


def _save_checkpoint(model, output_dir):
    torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))


if __name__ == "__main__":
    main()
