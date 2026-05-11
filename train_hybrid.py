"""Train the Hybrid CNN + Geometric-Ratio model for finger geometry.

Usage:
  python train_hybrid.py                          # full training
  python train_hybrid.py --epochs 30 --batch 16   # custom settings
  python train_hybrid.py --precompute-only         # just build caches

Outputs:
  models/hybrid_model.pth           - trained model weights
  models/hybrid_report.json         - training metrics
  models/hybrid_training_curve.png  - loss/accuracy plot
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, f1_score

from feature_extraction import FEATURE_COLUMNS
from hybrid_model import build_model
from hybrid_dataset import (
    FingerGeometryDataset,
    get_train_transforms,
    get_val_transforms,
    precompute_crops,
    precompute_ratios,
    load_ratio_cache,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train hybrid CNN model")
    p.add_argument("--data", default="super_database", help="Image directory")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=12,
                   help="Early stopping patience")
    p.add_argument("--out", default="models/hybrid_model.pth")
    p.add_argument("--precompute-only", action="store_true",
                   help="Only build caches then exit")
    return p.parse_args()


def make_splits(dataset, test_ratio=0.2, seed=42):
    """Stratified train/val split."""
    labels = [dataset.samples[i][2] for i in range(len(dataset))]
    label_indices = [dataset.label_to_idx[l] for l in labels]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio,
                                 random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), label_indices))
    return train_idx.tolist(), val_idx.tolist()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, ratios, labels in loader:
        images = images.to(device)
        ratios = ratios.to(device)
        labels = labels.to(device)

        logits = model(images, ratios)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for images, ratios, labels in loader:
        images = images.to(device)
        ratios = ratios.to(device)
        labels = labels.to(device)

        logits = model(images, ratios)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    acc = correct / total
    macro_f1 = f1_score(all_labels, all_preds, average="macro",
                        zero_division=0)
    return total_loss / total, acc, macro_f1, all_preds, all_labels


def save_training_curve(train_losses, val_losses, train_accs, val_accs,
                        output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label="Train Loss", color="#e74c3c")
    ax1.plot(val_losses, label="Val Loss", color="#3498db")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_accs, label="Train Acc", color="#e74c3c")
    ax2.plot(val_accs, label="Val Acc", color="#3498db")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    image_root = project_root / args.data
    crop_dir = project_root / "hand_crops"
    cache_path = project_root / "ratio_cache.csv"

    # Step 1: Pre-compute crops and ratios if needed
    needs_crops = not crop_dir.exists() or len(list(crop_dir.glob("*.png"))) < 100
    needs_ratios = not cache_path.exists()

    if needs_crops or args.precompute_only:
        print("=" * 60)
        print("Step 1a: Pre-cropping hand images (one-time)...")
        print("=" * 60)
        precompute_crops(image_root, crop_dir)

    if needs_ratios or args.precompute_only:
        print("=" * 60)
        print("Step 1b: Pre-computing geometric ratios (one-time)...")
        print("=" * 60)
        precompute_ratios(image_root, cache_path)

    if args.precompute_only:
        print("Pre-computation complete. Run again without --precompute-only to train.")
        return

    ratio_cache, label_cache = load_ratio_cache(cache_path)
    print(f"Loaded {len(ratio_cache)} cached ratios")

    # Step 2: Build datasets (FAST - no MediaPipe at all!)
    print("\n" + "=" * 60)
    print("Step 2: Building datasets...")
    print("=" * 60)

    full_ds = FingerGeometryDataset(
        crop_dir, ratio_cache, label_cache, transform=get_val_transforms()
    )
    print(f"Found {len(full_ds)} samples, {full_ds.num_classes} classes")

    train_idx, val_idx = make_splits(full_ds)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    # Create train/val datasets with proper transforms
    train_ds = FingerGeometryDataset(
        crop_dir, ratio_cache, label_cache,
        transform=get_train_transforms(),
        label_to_idx=full_ds.label_to_idx
    )
    val_ds = FingerGeometryDataset(
        crop_dir, ratio_cache, label_cache,
        transform=get_val_transforms(),
        label_to_idx=full_ds.label_to_idx
    )

    train_loader = DataLoader(
        Subset(train_ds, train_idx), batch_size=args.batch,
        shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx), batch_size=args.batch,
        shuffle=False, num_workers=0, pin_memory=True
    )

    # Step 3: Build model
    print("\n" + "=" * 60)
    print("Step 3: Building hybrid model...")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = build_model(num_classes=full_ds.num_classes, device=device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Step 4: Training loop
    print("\n" + "=" * 60)
    print("Step 4: Training...")
    print("=" * 60)

    best_f1 = 0.0
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        t_loss, t_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        v_loss, v_acc, v_f1, v_preds, v_labels = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        train_accs.append(t_acc)
        val_accs.append(v_acc)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss={t_loss:.4f} Acc={t_acc:.3f} | "
              f"Val Loss={v_loss:.4f} Acc={v_acc:.3f} F1={v_f1:.3f} | "
              f"LR={lr:.6f}")

        if v_f1 > best_f1:
            best_f1 = v_f1
            patience_counter = 0
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": full_ds.num_classes,
                "label_to_idx": full_ds.label_to_idx,
                "idx_to_label": full_ds.idx_to_label,
                "feature_columns": FEATURE_COLUMNS,
                "best_val_f1": best_f1,
                "epoch": epoch,
            }, out_path)
            print(f"  -> Saved best model (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed:.1f}s")

    # Step 5: Final evaluation with best model
    print("\n" + "=" * 60)
    print("Step 5: Final evaluation with best model...")
    print("=" * 60)

    checkpoint = torch.load(args.out, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, final_acc, final_f1, preds, labels = evaluate(
        model, val_loader, criterion, device
    )

    idx_to_label = checkpoint["idx_to_label"]
    label_names = [idx_to_label[i] for i in range(full_ds.num_classes)]
    report = classification_report(
        labels, preds, target_names=label_names, zero_division=0
    )

    print(f"\nFinal Val Accuracy: {final_acc:.4f}")
    print(f"Final Val Macro-F1: {final_f1:.4f}")

    # Save reports
    out_dir = Path(args.out).parent
    report_data = {
        "dataset_images": len(full_ds),
        "dataset_classes": full_ds.num_classes,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_f1": float(best_f1),
        "final_val_accuracy": float(final_acc),
        "final_val_macro_f1": float(final_f1),
        "training_time_seconds": float(elapsed),
        "device": str(device),
        "model_params_total": model.get_total_params(),
        "model_params_trainable": model.get_trainable_params(),
    }

    (out_dir / "hybrid_report.json").write_text(
        json.dumps(report_data, indent=2), encoding="utf-8"
    )
    (out_dir / "hybrid_classification_report.txt").write_text(
        report, encoding="utf-8"
    )
    save_training_curve(
        train_losses, val_losses, train_accs, val_accs,
        out_dir / "hybrid_training_curve.png"
    )

    print(f"\nSaved: {args.out}")
    print(f"Saved: {out_dir / 'hybrid_report.json'}")
    print(f"Saved: {out_dir / 'hybrid_classification_report.txt'}")
    print(f"Saved: {out_dir / 'hybrid_training_curve.png'}")


if __name__ == "__main__":
    main()
