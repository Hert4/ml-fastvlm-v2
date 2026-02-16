# FastViT HD Vision Tower for Belle-VLM
# Upload file nay len HuggingFace repo: beyoru/Belle-VLM

import torch
import torch.nn as nn


class FastViTHD(nn.Module):
    """
    FastViT High-Definition Vision Tower.
    Output dimension: 3072 (de match voi mm_projector da train)
    """

    def __init__(self, image_size=384, output_dim=3072):
        super().__init__()
        self.image_size = image_size
        self.output_dim = output_dim
        self.backbone = None
        self.projection = None
        self._initialized = False

    def _init_backbone(self, device, dtype):
        """Lazy init backbone on correct device"""
        if self._initialized:
            return

        import timm

        # Load base FastViT model directly on target device
        self.backbone = timm.create_model(
            "fastvit_mci2.apple_mclip",
            pretrained=True,
            num_classes=0
        ).to(device=device, dtype=dtype)
        self.backbone.eval()

        # FastViT MCI2 output: 1280 features
        backbone_dim = 1280

        # Projection layer: 1280 -> 3072
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.GELU(),
        ).to(device=device, dtype=dtype)

        self._initialized = True
        print(f"FastViTHD initialized on {device}")

    def forward_features(self, x):
        """
        Forward pass returning spatial features.

        Args:
            x: [B, C, H, W] input images

        Returns:
            features: [B, num_patches, output_dim]
        """
        # Lazy init on first forward
        self._init_backbone(x.device, x.dtype)

        # Ensure backbone is on same device as input
        if self.backbone is not None:
            self.backbone = self.backbone.to(device=x.device, dtype=x.dtype)
            self.projection = self.projection.to(device=x.device, dtype=x.dtype)

        with torch.no_grad():
            features = self.backbone.forward_features(x)

        # features shape: [B, C, H, W] -> [B, H*W, C]
        if features.dim() == 4:
            b, c, h, w = features.shape
            features = features.flatten(2).transpose(1, 2)  # [B, H*W, C]
        elif features.dim() == 2:
            features = features.unsqueeze(1)  # [B, 1, C]

        # Project to output_dim
        features = self.projection(features)

        return features

    def forward(self, x):
        return self.forward_features(x)

    def to(self, *args, **kwargs):
        # Don't move backbone here - will be lazy initialized
        return super().to(*args, **kwargs)
