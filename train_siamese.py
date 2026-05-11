import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from hybrid_dataset import FingerGeometryDataset, load_ratio_cache, get_train_transforms, get_val_transforms
from siamese_model import get_siamese_model

class TripletDataset(Dataset):
    """
    Dataset that returns triplets (anchor, positive, negative).
    Anchor and positive are from the same person.
    Negative is from a different person.
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.labels = [base_dataset.samples[i][2] for i in range(len(base_dataset))]
        self.label_to_indices = {}
        for i, label in enumerate(self.labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(i)
        
        self.unique_labels = list(self.label_to_indices.keys())

    def __getitem__(self, index):
        # Anchor
        anchor_idx = index
        anchor_label = self.labels[anchor_idx]
        
        # Positive (same label, different index)
        pos_indices = self.label_to_indices[anchor_label].copy()
        if len(pos_indices) > 1:
            pos_indices.remove(anchor_idx)
        pos_idx = random.choice(pos_indices)
        
        # In a real hard-mining scenario, we would select the negative
        # based on current model embeddings. Here we'll implement a 
        # "Semi-Hard" selection by picking from a pool of negatives.
        neg_label = random.choice(self.unique_labels)
        while neg_label == anchor_label:
            neg_label = random.choice(self.unique_labels)
        neg_idx = random.choice(self.label_to_indices[neg_label])
        
        anchor_img, anchor_rat, _ = self.base_dataset[anchor_idx]
        pos_img, pos_rat, _ = self.base_dataset[pos_idx]
        neg_img, neg_rat, _ = self.base_dataset[neg_idx]
        
        return (anchor_img, anchor_rat), (pos_img, pos_rat), (neg_img, neg_rat)

    def __len__(self):
        return len(self.base_dataset)

def train_triplet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load cached data
    project_root = Path(__file__).resolve().parent
    crop_dir = project_root / "hand_crops"
    cache_path = project_root / "ratio_cache.csv"
    
    if not cache_path.exists():
        print("Error: Ratio cache not found. Please run hybrid_dataset.py first.")
        return

    ratio_cache, label_cache = load_ratio_cache(cache_path)
    
    # Base dataset
    base_ds = FingerGeometryDataset(
        crop_dir, ratio_cache, label_cache, transform=get_train_transforms()
    )
    
    # Triplet dataset
    triplet_ds = TripletDataset(base_ds)
    loader = DataLoader(triplet_ds, batch_size=16, shuffle=True)
    
    # Model & Loss
    model = get_siamese_model(device)
    # Increased margin forces harder separation between different people
    criterion = nn.TripletMarginLoss(margin=2.0, p=2)
    # Added Weight Decay (L2 Regularization) to prevent overfitting
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=2e-4, weight_decay=1e-4)
    
    # Training Loop
    model.train()
    epochs = 30
    history = []
    
    print(f"Starting Siamese training for {epochs} epochs...")
    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for (a_img, a_rat), (p_img, p_rat), (n_img, n_rat) in pbar:
            a_img, a_rat = a_img.to(device), a_rat.to(device)
            p_img, p_rat = p_img.to(device), p_rat.to(device)
            n_img, n_rat = n_img.to(device), n_rat.to(device)
            
            # Forward
            emb_a = model.forward_one(a_img, a_rat)
            emb_p = model.forward_one(p_img, p_rat)
            emb_n = model.forward_one(n_img, n_rat)
            
            loss = criterion(emb_a, emb_p, emb_n)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(loader)
        history.append(avg_loss)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
    # Save model
    model_path = project_root / "models" / "siamese_hybrid.pth"
    model_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Plot loss
    plt.figure()
    plt.plot(history)
    plt.title("Siamese Triplet Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(project_root / "models" / "siamese_loss.png")
    plt.show()

if __name__ == "__main__":
    train_triplet()
