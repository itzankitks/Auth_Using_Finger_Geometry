import argparse
import torch
import torch.nn as nn
from pathlib import Path
import cv2
from torchvision import transforms
from PIL import Image

from siamese_model import SiameseHybridNet
from feature_extraction import extract_feature_vector, FEATURE_COLUMNS
from hybrid_dataset import crop_hand, IMAGENET_MEAN, IMAGENET_STD

def load_siamese_model(model_path, device):
    model = SiameseHybridNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_embedding(model, image_path, device):
    # Prepare Image
    bgr = cv2.imread(str(image_path))
    hand = crop_hand(bgr)
    if hand is None:
        hand = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    img_tensor = transform(hand).unsqueeze(0).to(device)
    
    # Prepare Ratios
    fv = extract_feature_vector(str(image_path))
    ratios = [fv[c] for c in FEATURE_COLUMNS]
    ratio_tensor = torch.tensor([ratios], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        embedding = model.forward_one(img_tensor, ratio_tensor)
    return embedding

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img1", required=True)
    parser.add_argument("--img2", required=True)
    parser.add_argument("--model", default="models/siamese_hybrid.pth")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_siamese_model(args.model, device)
    
    emb1 = get_embedding(model, args.img1, device)
    emb2 = get_embedding(model, args.img2, device)
    
    # Calculate Euclidean distance
    dist = torch.pow(emb1 - emb2, 2).sum().sqrt().item()
    
    # Calculate Cosine Similarity
    sim = torch.nn.functional.cosine_similarity(emb1, emb2).item()
    
    print(f"\nResults for Comparison:")
    print(f"  Image 1: {args.img1}")
    print(f"  Image 2: {args.img2}")
    print(f"  -------------------------")
    print(f"  Euclidean Distance: {dist:.4f} (Lower is more similar)")
    print(f"  Cosine Similarity:  {sim:.4f} (Higher is more similar)")
    
    threshold = 0.5 # Example threshold for distance
    if dist < threshold:
        print("\n  VERDICT: MATCH (Same Person)")
    else:
        print("\n  VERDICT: NO MATCH (Different Persons)")

if __name__ == "__main__":
    main()
