from __future__ import annotations

from collections.abc import Sequence

import torch

from monai.networks.nets import ViTAutoEnc
from sklearn.model_selection import train_test_split
import unittest


# TODO's are organized Highest Priority to Lowest Priority @ Joslin
# Done Make Sure that reconstructed image is valid and has the same shape as the original image
# Done This class is a bit of a hack. It should be refactored to be more general and possibly moved into the Autoencoder class
# TODO Figure out L1 loss and Contrastive Loss and make a tldr for the team
# TODO make the env file example and update the Encoder to use the .env file
# TODO Test to see if we can use the stacked images as the input to the model -- I will probably end up doing this
# TODO Figure out where the "latent space" is and how to get it / visualize it


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
        test: bool = False,
        training_unsupervised: bool = False,
        number_of_patch_tensors: int = 3,
    ) -> None:

        self.mask_rate = mask_rate
        self.mask_dict = {}  # Dict of index of patches to mask and the masked patches
        self.unmasked_dict = (
            {}
        )  # Dict of index of patches to mask and the unmasked patches
        self.training_unsupervised = training_unsupervised
        if not test:
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
            self.index_order = [i for i in range(self.patch_embedding.n_patches)]
            self.orig_unmasked_indexes, self.orig_masked_indexes = train_test_split(
                self.index_order, test_size=self.mask_rate
            )
        else:
            self.index_order = [i for i in range(number_of_patch_tensors)]
            self.orig_unmasked_indexes, self.orig_masked_indexes = train_test_split(
                self.index_order, test_size=self.mask_rate, random_state=42
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
        raw_positional_embeddings = self.patch_embedding.position_embeddings
        masked_patches = raw_positional_embeddings.repeat(batch_size, 1, 1)
        if self.training_unsupervised:

            unmasked_tensor, masked_patches = self.split_tensor_and_record_new_indexes(
                raw_patched_image_with_embeddings, masked_patches
            )
        else:
            unmasked_tensor = raw_patched_image_with_embeddings
        # Masked patches

        hidden_states_out = []
        for blk in self.blocks:
            unmasked_tensor = blk(unmasked_tensor)
            hidden_states_out.append(unmasked_tensor)
        unmasked_tensor = self.norm(unmasked_tensor)
        concated_masked = self.reconstruct_image(
            unmasked_tensor, raw_patched_image_with_embeddings
        )
        concated_masked = concated_masked.transpose(1, 2)
        d = [s // p for s, p in zip(spatial_size, self.patch_size)]
        ## possible latent space
        x = torch.reshape(
            concated_masked, [concated_masked.shape[0], concated_masked.shape[1], *d]
        )
        #decoder
        x = self.decoder(x)
        return x, hidden_states_out

    def decoder(self, x):
        x = self.conv3d_transpose(x)
        x = self.conv3d_transpose_1(x)
        return x
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
        if self.masked_tensor.shape[0] != batch_dim:
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


class TestMaskMapper(unittest.TestCase):

    def test_mapping(self):
        # Example values
        original_tensor = torch.tensor(
            [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]
        )
        positional_embeddings = torch.tensor(
            [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]]
        )
        expected_reconstructed_image = torch.tensor(
            [[[1, 2], [3, 4], [5, 6]], [[0.1, 0.2], [0.9, 1.0], [0.5, 0.6]]]
        )

        mask_mapper = MAEViTAutoEnc(
            mask_rate=0.5, img_size=[1, 2, 3], in_channels=1, test=True, patch_size=1
        )

        # Split tensor and record new indexes
        unmasked_tensor, masked_tensor = (
            mask_mapper.split_tensor_and_record_new_indexes(
                original_tensor, positional_embeddings
            )
        )

        # Check if the mappings are correct
        for orig_index, new_index in mask_mapper.mask_index_mapping.items():
            orig_patch = positional_embeddings[:, orig_index, :]
            masked_patch = masked_tensor[:, new_index, :]
            self.assertTrue(
                torch.all(masked_patch == orig_patch),
                f"Mismatch at masked index {new_index}",
            )

        for orig_index, new_index in mask_mapper.unmasked_index_mapping.items():
            orig_patch = original_tensor[:, orig_index, :]
            unmasked_patch = unmasked_tensor[:, new_index, :]
            self.assertTrue(
                torch.all(unmasked_patch == orig_patch),
                f"Mismatch at unmasked index {new_index}",
            )
