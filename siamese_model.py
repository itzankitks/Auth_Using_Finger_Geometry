import torch
import torch.nn as nn
from torchvision import models

class SiameseHybridNet(nn.Module):
    """
    Siamese Hybrid Model for Metric Learning.
    
    This model takes a hand image and its geometric ratios and maps them 
    into a 128-dimensional embedding space where hands from the same 
    person are close together and hands from different people are far apart.
    """
    def __init__(self, num_ratio_features=10, embedding_dim=128):
        super().__init__()
        
        # 1. CNN Branch (Visual Features)
        # Using MobileNetV2 for efficiency on local machines
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # Freeze early layers to preserve ImageNet features
        # (important for small datasets)
        all_features = list(backbone.features.parameters())
        num_freeze = int(len(all_features) * 0.7)
        for param in all_features[:num_freeze]:
            param.requires_grad = False
            
        self.cnn_backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # CNN Projection
        self.cnn_proj = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # 2. Ratio Branch (Geometric Features) - STRONGER FOCUS
        self.ratio_proj = nn.Sequential(
            nn.Linear(num_ratio_features, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # 3. Fusion & Embedding Head
        # Concatenate 256 (CNN) + 128 (Ratio) = 384
        self.embedding_head = nn.Sequential(
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim)
        )

    def forward_one(self, x_img, x_ratio):
        """Compute embedding for a single sample."""
        # Visual features
        cnn_feat = self.cnn_backbone(x_img)
        cnn_feat = self.pool(cnn_feat).flatten(1)
        cnn_proj = self.cnn_proj(cnn_feat)
        
        # Geometric features
        ratio_proj = self.ratio_proj(x_ratio)
        
        # Fusion
        combined = torch.cat([cnn_proj, ratio_proj], dim=1)
        embedding = self.embedding_head(combined)
        
        # Normalize embedding to unit hypersphere
        return nn.functional.normalize(embedding, p=2, dim=1)

    def forward(self, img1, rat1, img2, rat2):
        """Forward pass for Siamese pair (used in contrastive loss)."""
        emb1 = self.forward_one(img1, rat1)
        emb2 = self.forward_one(img2, rat2)
        return emb1, emb2

def get_siamese_model(device):
    model = SiameseHybridNet().to(device)
    print(f"Siamese Hybrid Model initialized on {device}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")
    return model

if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_siamese_model(dev)
    
    # Test pass
    dummy_img = torch.randn(2, 3, 224, 224).to(dev)
    dummy_rat = torch.randn(2, 10).to(dev)
    emb = model.forward_one(dummy_img, dummy_rat)
    print(f"Embedding shape: {emb.shape}") # (2, 128)
