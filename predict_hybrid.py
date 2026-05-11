"""Predict person identity using the Hybrid CNN + Ratio model.

Usage:
  python predict_hybrid.py --image "super_database/IMG_0067 (1).JPG"
  python predict_hybrid.py --image path/to/hand.jpg --model models/hybrid_model.pth
"""

import argparse
from pathlib import Path

import cv2
import torch
from torchvision import transforms

from feature_extraction import FEATURE_COLUMNS, extract_feature_vector
from hybrid_model import HybridFingerGeometryNet
from hybrid_dataset import crop_hand, IMAGENET_MEAN, IMAGENET_STD


def load_hybrid_model(model_path, device):
    """Load saved hybrid model checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = HybridFingerGeometryNet(
        num_classes=checkpoint["num_classes"],
        num_ratio_features=len(checkpoint["feature_columns"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def prepare_image(image_path, device):
    """Load image, crop hand, and return tensor."""
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read: {image_path}")

    hand = crop_hand(bgr)
    if hand is None:
        print("  Warning: No hand detected, using full image")
        hand = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    img_tensor = transform(hand).unsqueeze(0).to(device)
    return img_tensor


def prepare_ratios(image_path, feature_columns, device):
    """Extract geometric ratios and return tensor."""
    fv = extract_feature_vector(str(image_path))
    ratios = [fv[c] for c in feature_columns]
    ratio_tensor = torch.tensor([ratios], dtype=torch.float32).to(device)
    return ratio_tensor, fv


@torch.no_grad()
def predict(model, img_tensor, ratio_tensor, idx_to_label, top_k=5):
    """Run prediction and return top-K results."""
    logits = model(img_tensor, ratio_tensor)
    probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_indices = probs.topk(top_k)
    results = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        label = idx_to_label.get(idx, idx_to_label.get(str(idx), f"Class {idx}"))
        results.append((label, prob))
    return results


def main():
    p = argparse.ArgumentParser(description="Predict with hybrid model")
    p.add_argument("--image", required=True, help="Path to hand image")
    p.add_argument("--model", default="models/hybrid_model.pth",
                   help="Path to hybrid model checkpoint")
    p.add_argument("--top-k", type=int, default=5, help="Show top-K predictions")
    args = p.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        # Try in super_database
        alt = Path(__file__).resolve().parent / "super_database" / args.image
        if alt.exists():
            image_path = alt
        else:
            print(f"Image not found: {args.image}")
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model, checkpoint = load_hybrid_model(args.model, device)
    idx_to_label = checkpoint["idx_to_label"]
    feature_columns = checkpoint["feature_columns"]
    print(f"Model loaded: {checkpoint['num_classes']} classes, "
          f"best val F1={checkpoint['best_val_f1']:.4f}")

    # Prepare inputs
    img_tensor = prepare_image(image_path, device)
    ratio_tensor, raw_features = prepare_ratios(image_path, feature_columns,
                                                 device)

    # Predict
    results = predict(model, img_tensor, ratio_tensor, idx_to_label,
                      top_k=args.top_k)

    # Display
    print(f"\nImage: {image_path}")
    print(f"\nPredicted: {results[0][0]} (confidence: {results[0][1]:.4f})")
    print(f"\nTop-{args.top_k} predictions:")
    for rank, (label, prob) in enumerate(results, 1):
        bar = "=" * int(prob * 30)
        print(f"  {rank}. {label:<15s} {prob:.4f} {bar}")

    print(f"\nGeometric features used:")
    for col in feature_columns:
        print(f"  {col}: {raw_features[col]:.6f}")


if __name__ == "__main__":
    main()
