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
        mask_rate: float = 0.75,
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
        self.mask_rate = mask_rate
        self.mask_dict = {}  # Dict of index of patches to mask and the masked patches
        self.unmasked_dict = (
            {}
        )  # Dict of index of patches to mask and the unmasked patches
        self.masked_tensor = None
        self.unmasked_tensor = None

    def forward(self, x):
        """
        Args:
            x: input tensor must have isotropic spatial dimensions,
                such as ``[batch_size, channels, sp_size, sp_size[, sp_size]]``.
        """
        spatial_size = x.shape[2:]

        batch_size = x.shape[0]
        x, random_masked_patches = self.split_patches(x)

        batch_mask = random_masked_patches
        for i in range(batch_size - 1):
            batch_mask = torch.cat((batch_mask, random_masked_patches), 0)

        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)

        # TODO reconstruct the image from the masked patches and the normalized output of the transformer blocks
        # TODO make sure the order of the patches is preserved
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

    def split_patches(self, x):
        # THIS IS A HACK
        raw_patched_image_with_embeddings = self.patch_embedding(x)
        raw_positional_embeddings = self.patch_embedding.position_embeddings

        number_of_all_patches = raw_patched_image_with_embeddings.shape[1]
        number_of_patches_to_mask = int(number_of_all_patches * self.mask_rate)

        # Select random 75% indexes to mask
        mask_indexes = torch.randperm(number_of_all_patches)[:number_of_patches_to_mask]

        unmasked_indexes = torch.tensor(
            [i for i in range(number_of_all_patches) if i not in mask_indexes]
        )
        # The idea is to keep the original index order of the patches in the index by using 2 dicts.

        # Possible other implemation ideas:
        # 1)  Use a class that extends PatchEmbeddingBlock like we did with MAEViTAutoEnc
        # 2)  Create a Completely Seperate Class that will handle the masking and unmasking of the patches
        # 3)  Figure out a way to change to keep the add it to metadata of the tensors?? Is that a thing?

        for mask_index in sorted(mask_indexes):
            if self.masked_tensor is None:
                self.masked_tensor = raw_positional_embeddings[:, mask_index, :]
            else:
                self.masked_tensor = torch.cat(
                    (self.masked_tensor, raw_positional_embeddings[:, mask_index, :]), 0
                )
            self.mask_dict[mask_index] = raw_positional_embeddings[:, mask_index, :]

        for unmasked_index in sorted(unmasked_indexes):
            if self.unmasked_tensor is None:
                self.unmasked_tensor = raw_patched_image_with_embeddings[
                    :, unmasked_index, :
                ]
            else:
                self.unmasked_tensor = torch.cat(
                    (
                        self.unmasked_tensor,
                        raw_patched_image_with_embeddings[:, unmasked_index, :],
                    ),
                    0,
                )
            self.unmasked_dict[unmasked_index] = raw_patched_image_with_embeddings[
                :, unmasked_index, :
            ]

        return self.masked_tensor, self.unmasked_tensor

    def reconstruct_image(self, normalized_x):
        batch_size = normalized_x.shape[0]

        batched_masked_patches = self.masked_tensor
        for i in range(batch_size - 1):
            batched_masked_patches = torch.cat(
                (batched_masked_patches, self.masked_tensor), 0
            )

        # Create a tensor we can use to reconstruct the image
        # TODO Figure out how to reconstruct the image from the masked patches and the
        #  normalized output of the transformer blocks
        # THEY MUST GO IN THE SAME ORDER BUT MUST BE RANDOMLY SELECTED
