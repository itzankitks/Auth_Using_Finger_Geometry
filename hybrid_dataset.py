"""PyTorch Dataset for the Hybrid Finger Geometry model.

Key optimization: Pre-crops all hand images ONCE into a cache folder
so MediaPipe doesn't run during training (which was extremely slow).

Workflow:
  1. precompute_crops()  - runs MediaPipe once, saves cropped PNGs
  2. precompute_ratios() - runs feature extraction once, saves CSV
  3. FingerGeometryDataset - loads from cache (fast!)
"""

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from feature_extraction import FEATURE_COLUMNS, extract_feature_vector

mp_hands = mp.solutions.hands
IMAGE_PATTERN = re.compile(r"IMG_(\d+)\s*\((\d+)\)", re.IGNORECASE)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(size=224):
    return transforms.Compose([
        transforms.Resize((size + 32, size + 32)),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.05),
        transforms.RandomAffine(0, translate=(0.08, 0.08),
                                scale=(0.9, 1.1)),
        transforms.GaussianBlur(3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])


def get_val_transforms(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ──────────────────────────────────────────────
# Pre-computation (run once)
# ──────────────────────────────────────────────

def crop_hand(image_bgr, padding=40):
    """Detect hand and crop region. Returns RGB numpy array or None."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = image_bgr.shape
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                        min_detection_confidence=0.5) as hands:
        results = hands.process(rgb)
    if not results.multi_hand_landmarks:
        return None
    lm = results.multi_hand_landmarks[0]
    xs = [l.x * w for l in lm.landmark]
    ys = [l.y * h for l in lm.landmark]
    x0 = max(0, int(min(xs)) - padding)
    y0 = max(0, int(min(ys)) - padding)
    x1 = min(w, int(max(xs)) + padding)
    y1 = min(h, int(max(ys)) + padding)
    crop = rgb[y0:y1, x0:x1]
    return crop if crop.size > 0 else None


def precompute_crops(image_root, crop_dir):
    """Crop all hand images once and save as PNG files.
    
    This is the key optimization: MediaPipe runs once here instead of
    on every training batch.
    """
    image_root = Path(image_root)
    crop_dir = Path(crop_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    skipped = 0
    files = sorted(image_root.iterdir())
    total = len([f for f in files if f.is_file() and
                 f.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    for p in files:
        if not p.is_file() or p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        m = IMAGE_PATTERN.search(p.stem)
        if not m:
            continue

        out_path = crop_dir / f"{p.stem}.png"
        if out_path.exists():
            count += 1
            continue

        bgr = cv2.imread(str(p))
        if bgr is None:
            skipped += 1
            continue

        hand = crop_hand(bgr)
        if hand is None:
            # Fallback: use full image resized
            hand = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            skipped += 1

        # Save as PNG
        hand_bgr = cv2.cvtColor(hand, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), hand_bgr)
        count += 1

        if count % 100 == 0:
            print(f"  Cropped {count}/{total} images...")

    print(f"Done. {count} crops saved to {crop_dir}")
    if skipped:
        print(f"  ({skipped} used full image fallback)")


def precompute_ratios(image_root, output_csv):
    """Pre-compute all geometric ratios to CSV."""
    root = Path(image_root)
    rows = []
    for p in sorted(root.iterdir()):
        if not p.is_file() or p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        m = IMAGE_PATTERN.search(p.stem)
        if not m:
            continue
        try:
            fv = extract_feature_vector(str(p))
            row = {"_path": str(p), "_stem": p.stem,
                   "Label": f"Person {int(m.group(1))}"}
            row.update(fv)
            rows.append(row)
            if len(rows) % 100 == 0:
                print(f"  Cached {len(rows)} ratios...")
        except Exception as e:
            print(f"  Skipped {p.name}: {e}")

    fields = ["_path", "_stem"] + FEATURE_COLUMNS + ["Label"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {len(rows)} entries -> {output_csv}")


def load_ratio_cache(csv_path) -> Tuple[dict, dict]:
    """Load pre-computed ratios. Returns (stem->ratios, stem->label) dicts."""
    ratios = {}
    labels = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stem = row["_stem"]
            ratios[stem] = [float(row[c]) for c in FEATURE_COLUMNS]
            labels[stem] = row["Label"]
    return ratios, labels


# ──────────────────────────────────────────────
# Fast Dataset (loads from pre-computed cache)
# ──────────────────────────────────────────────

class FingerGeometryDataset(Dataset):
    """Fast dataset that loads pre-cropped images + cached ratios.
    
    No MediaPipe at training time - everything is pre-computed.
    """

    def __init__(self, crop_dir, ratio_cache, label_cache,
                 transform=None, label_to_idx=None):
        self.crop_dir = Path(crop_dir)
        self.transform = transform or get_val_transforms()
        self.ratio_cache = ratio_cache    # stem -> [10 floats]
        self.label_cache = label_cache    # stem -> "Person N"

        # Discover crop files that have matching ratios
        self.samples = []
        for p in sorted(self.crop_dir.glob("*.png")):
            stem = p.stem
            if stem in self.ratio_cache and stem in self.label_cache:
                self.samples.append((p, stem, self.label_cache[stem]))

        # Label mapping
        if label_to_idx:
            self.label_to_idx = label_to_idx
        else:
            unique = sorted(set(lbl for _, _, lbl in self.samples))
            self.label_to_idx = {l: i for i, l in enumerate(unique)}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, stem, label_str = self.samples[idx]
        label_idx = self.label_to_idx[label_str]

        # Load pre-cropped image (fast - no MediaPipe!)
        img = Image.open(path).convert("RGB")
        img_tensor = self.transform(img)

        # Load pre-computed ratios (fast - no feature extraction!)
        ratios = self.ratio_cache[stem]
        ratio_tensor = torch.tensor(ratios, dtype=torch.float32)

        return img_tensor, ratio_tensor, label_idx

    @property
    def num_classes(self):
        return len(self.label_to_idx)

    @property
    def class_names(self):
        return [self.idx_to_label[i] for i in range(self.num_classes)]


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    db = root / "super_database"
    
    print("Step 1: Pre-computing hand crops...")
    precompute_crops(db, root / "hand_crops")
    
    print("\nStep 2: Pre-computing geometric ratios...")
    precompute_ratios(db, root / "ratio_cache.csv")
