from typing import List

import torch
import torch.nn as nn

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock


class ViTEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        img_size: List[int] | int,
        patch_size: List[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        proj_type: str = "conv",
        pos_embed_type: str = "learnable",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            proj_type=proj_type,
            pos_embed_type=pos_embed_type,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class ViTDecoder(nn.Module):
    def __init__(
        self, hidden_size: int, num_classes: int, post_activation: str = "Tanh"
    ) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        if post_activation == "Tanh":
            self.classification_head = nn.Sequential(
                nn.Linear(hidden_size, num_classes), nn.Tanh()
            )
        else:
            self.classification_head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.classification_head(x[:, 0])
        return x


if __name__ == "__main__":
    encoder = ViTEncoder(
        in_channels=3, img_size=(3, 224, 224), patch_size=(3, 16, 16)
    )  # Adjusted patch_size
    decoder = ViTDecoder(hidden_size=768, num_classes=2)

    x = torch.randn(1, 3, 224, 224)
    encoder_output = encoder(x)

    # Pass through the decoder
    decoder_output = decoder(encoder_output)
