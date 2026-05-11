"""Hybrid Finger Geometry CNN Model.

Combines two branches:
  - CNN branch (MobileNetV2 transfer learning) for visual features
    (palm lines, knuckle texture, skin patterns)
  - Ratio branch for the 10 hand-crafted geometric features

Architecture:
  Image → MobileNetV2 → GlobalAvgPool → Dense(128)  ─┐
                                                       ├→ Dense(64) → Dense(N) → Softmax
  Ratios(10) → Dense(32)                             ─┘
"""

import torch
import torch.nn as nn
from torchvision import models


class PalmLineEnhancer(nn.Module):
    """Learnable palm-line enhancement using depthwise convolutions
    that mimic Gabor-filter-like edge detection at multiple orientations.

    This replaces the hand-crafted Gabor preprocessing with a lightweight
    trainable front-end that can adapt to the data.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.enhance = nn.Sequential(
            # Depthwise conv at larger kernel to capture line structures
            nn.Conv2d(in_channels, 16, kernel_size=5, padding=2, groups=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enhance(x)


class HybridFingerGeometryNet(nn.Module):
    """Hybrid model combining CNN visual features with geometric ratio features.

    Parameters
    ----------
    num_classes : int
        Number of person classes to predict.
    num_ratio_features : int
        Number of hand-crafted geometric features (default 10).
    cnn_embed_dim : int
        Dimensionality of the CNN embedding (default 128).
    freeze_ratio : float
        Fraction of MobileNetV2 layers to freeze (0.0 = none, 1.0 = all).
        Default 0.7 freezes the first 70% of layers.
    use_palm_enhancer : bool
        Whether to prepend the learnable palm-line enhancer.
    """

    def __init__(
        self,
        num_classes: int,
        num_ratio_features: int = 10,
        cnn_embed_dim: int = 128,
        freeze_ratio: float = 0.7,
        use_palm_enhancer: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_palm_enhancer = use_palm_enhancer

        # Optional learnable palm-line enhancer
        if use_palm_enhancer:
            self.palm_enhancer = PalmLineEnhancer(in_channels=3, out_channels=3)

        # ============================
        # CNN Branch (MobileNetV2)
        # ============================
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # Freeze early layers
        all_params = list(backbone.features.parameters())
        num_freeze = int(len(all_params) * freeze_ratio)
        for param in all_params[:num_freeze]:
            param.requires_grad = False

        self.cnn_features = backbone.features  # All conv layers
        self.pool = nn.AdaptiveAvgPool2d(1)    # Global Average Pooling

        # MobileNetV2 outputs 1280 features after pool
        self.cnn_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, cnn_embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # ============================
        # Ratio Branch
        # ============================
        self.ratio_head = nn.Sequential(
            nn.Linear(num_ratio_features, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        # ============================
        # Fusion & Classification
        # ============================
        fusion_dim = cnn_embed_dim + 32  # 128 + 32 = 160
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        image: torch.Tensor,
        ratios: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        image : Tensor of shape (B, 3, 224, 224)
        ratios : Tensor of shape (B, 10)

        Returns
        -------
        logits : Tensor of shape (B, num_classes)
        """
        # Optional palm-line enhancement
        if self.use_palm_enhancer:
            x = self.palm_enhancer(image)
        else:
            x = image

        # CNN branch
        x = self.cnn_features(x)          # (B, 1280, 7, 7)
        x = self.pool(x)                  # (B, 1280, 1, 1)
        x = x.view(x.size(0), -1)         # (B, 1280)
        cnn_out = self.cnn_head(x)         # (B, 128)

        # Ratio branch
        ratio_out = self.ratio_head(ratios)  # (B, 32)

        # Fusion
        combined = torch.cat([cnn_out, ratio_out], dim=1)  # (B, 160)
        logits = self.classifier(combined)                  # (B, num_classes)
        return logits

    def get_trainable_params(self) -> int:
        """Return count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())


def build_model(num_classes: int, device: torch.device, **kwargs) -> HybridFingerGeometryNet:
    """Convenience function to build and move model to device."""
    model = HybridFingerGeometryNet(num_classes=num_classes, **kwargs)
    model = model.to(device)
    print(f"Model built: {model.get_total_params():,} total params, "
          f"{model.get_trainable_params():,} trainable")
    return model


if __name__ == "__main__":
    # Quick sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=164, device=device)

    # Dummy forward pass
    dummy_img = torch.randn(2, 3, 224, 224, device=device)
    dummy_ratios = torch.randn(2, 10, device=device)
    out = model(dummy_img, dummy_ratios)
    print(f"Output shape: {out.shape}")  # (2, 164)
    print("Sanity check passed!")
