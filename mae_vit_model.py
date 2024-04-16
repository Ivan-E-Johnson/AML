from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.layers import Conv
from monai.networks.nets import ViTAutoEnc
from monai.utils import ensure_tuple_rep


class MAEViTAutoEnc(ViTAutoEnc):
    """
    Vision Transformer (ViT),  That will be used for A Masked Autoencoder

    """

    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        out_channels: int = 1,
        deconv_chns: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        proj_type: str = "conv",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            img_size,
            patch_size,
            out_channels,
            deconv_chns,
            hidden_size,
            mlp_dim,
            num_layers,
            num_heads,
            pos_embed,
            proj_type,
            dropout_rate,
            spatial_dims,
            qkv_bias,
            save_attn,
        )

    def forward(self, x):
        """
        Args:
            x: input tensor must have isotropic spatial dimensions,
                such as ``[batch_size, channels, sp_size, sp_size[, sp_size]]``.
        """
        spatial_size = x.shape[2:]
        batch_size = x.shape[0]
        raw_patched_image_with_embeddings = self.patch_embedding(x)
        raw_positional_embeddings = self.patch_embedding.position_embeddings.data
        print(f"Patch Embedding: {x.shape}")
        # Mask some inputs

        # Select random 25% indexes
        x = raw_patched_image_with_embeddings[:, :56, :]

        # Apply mask to 75% indexes

        masked = raw_positional_embeddings[:, 56:, :]

        batch_mask = masked
        for i in range(batch_size - 1):
            batch_mask = torch.cat((batch_mask, masked), 0)

        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)

        # This needs
        concated_masked = torch.cat((x, batch_mask), 1)
        concated_masked = concated_masked.transpose(1, 2)
        d = [s // p for s, p in zip(spatial_size, self.patch_size)]
        ## possible latent space
        x = torch.reshape(
            concated_masked, [concated_masked.shape[0], concated_masked.shape[1], *d]
        )

        x = self.conv3d_transpose(x)
        x = self.conv3d_transpose_1(x)
        return x, hidden_states_out
