from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.layers import Conv
from monai.networks.nets import ViTAutoEnc
from monai.utils import ensure_tuple_rep
from sklearn.model_selection import train_test_split


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
        self.mask_mapper = MaskMapper(mask_rate, self.patch_embedding.n_patches)

    def forward(self, x):
        """
        Args:
            x: input tensor must have isotropic spatial dimensions,
                such as ``[batch_size, channels, sp_size, sp_size[, sp_size]]``.
        """
        spatial_size = x.shape[2:]

        batch_size = x.shape[0]
        raw_patched_image_with_embeddings = self.patch_embedding(x)
        raw_positional_embeddings = self.patch_embedding.position_embeddings
        masked_patches = raw_positional_embeddings.repeat(batch_size, 1, 1)
        unmasked_tensor, masked_patches = (
            self.mask_mapper.split_tensor_and_record_new_indexes(
                raw_patched_image_with_embeddings, masked_patches
            )
        )

        # Masked patches

        hidden_states_out = []
        for blk in self.blocks:
            unmasked_tensor = blk(unmasked_tensor)
            hidden_states_out.append(unmasked_tensor)
        unmasked_tensor = self.norm(unmasked_tensor)

        # TODO reconstruct the image from the masked patches and the normalized output of the transformer blocks
        # TODO make sure the order of the patches is preserved
        concated_masked = self.mask_mapper.reconstruct_image(
            unmasked_tensor, raw_patched_image_with_embeddings
        )
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

        return self.masked_tensor

    def reconstruct_image(self, normalized_x):
        batch_size = normalized_x.shape[0]
        # Repeat the masked patches for each sample in the batch
        masked_patches = self.masked_tensor.repeat(batch_size, 1, 1)
        # Use an index mapping to reconstruct the image


class MaskMapper:
    # TODO's are organized Highest Priority to Lowest Priority @ Joslin
    # TODO Make Sure that reconstructed image is valid and has the same shape as the original image
    # TODO Figure out L1 loss and Contrastive Loss and make a tldr for the team
    # TODO make the env file example and update the Encoder to use the .env file
    # TODO Test to see if we can use the stacked images as the input to the model -- I will probably end up doing this
    # TODO Figure out where the "latent space" is and how to get it / visualize it
    # TODO This class is a bit of a hack. It should be refactored to be more general and possibly moved into the Autoencoder class

    def __init__(self, mask_rate: float, number_of_patch_tensors: int):
        self.mask_rate = mask_rate
        self.index_order = [
            i for i in range(number_of_patch_tensors)
        ]  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.orig_unmasked_indexes, self.orig_masked_indexes = train_test_split(
            self.index_order, test_size=self.mask_rate
        )

    def split_tensor_and_record_new_indexes(
        self, tensor: torch.Tensor, raw_positional_embeddings: torch.Tensor
    ):
        self.mask_index_mapping = {}
        self.masked_tensor = None
        for index in self.orig_masked_indexes:
            extracol = raw_positional_embeddings[:, index, :].unsqueeze(1)
            if self.masked_tensor is None:
                self.masked_tensor = extracol
                new_index = 0
            else:
                new_index = self.masked_tensor.shape[1]
                self.masked_tensor = torch.cat((self.masked_tensor, extracol), 1)

            self.mask_index_mapping[index] = new_index

        unmasked_tensor = None
        self.unmasked_index_mapping = {}
        for index in self.orig_unmasked_indexes:
            extracol = tensor[:, index, :].unsqueeze(1)
            if unmasked_tensor is None:
                unmasked_tensor = extracol
                new_index = 0
            else:
                new_index = unmasked_tensor.shape[1]
                unmasked_tensor = torch.cat((unmasked_tensor, extracol), 1)
            self.unmasked_index_mapping[index] = new_index

        return unmasked_tensor, self.masked_tensor

    def reconstruct_image(self, normalized_x, orig_tensor):
        batch_dim = normalized_x.shape[0]
        if self.masked_tensor.shape[0] != batch_dim:  # CHECK THAT THE
            raise ValueError(
                "The batch dimension of the masked tensor and the normalized tensor must be the same"
            )

        # Use an index mapping to reconstruct the image
        reconstructed_image = torch.zeros_like(orig_tensor)
        for orig_index, new_index in self.mask_index_mapping.items():
            reconstructed_image[:, orig_index, :] = self.masked_tensor[:, new_index, :]

        for orig_index, new_index in self.unmasked_index_mapping.items():
            reconstructed_image[:, orig_index, :] = normalized_x[:, new_index, :]

        return reconstructed_image
