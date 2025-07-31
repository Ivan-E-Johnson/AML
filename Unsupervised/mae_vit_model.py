from __future__ import annotations

from collections.abc import Sequence

import torch

from monai.networks.nets import ViTAutoEnc
from sklearn.model_selection import train_test_split
import unittest

from torch import nn


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
        # Save the original indexes of the masked and unmasked patches
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
            # Save the original indexes of the masked and unmasked patches
            self.index_order = [i for i in range(self.patch_embedding.n_patches)]
            self.orig_unmasked_indexes, self.orig_masked_indexes = train_test_split(
                self.index_order, test_size=self.mask_rate
            )
            print(f"Orig unmasked indexes: {len(self.orig_masked_indexes)}")
            self.orig_image_size = img_size

        else:
            # For testing purposes, we want to be able to set the indexes of the masked and unmasked patches
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
        # Get the patch embeddings
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
        # reshape the tensor
        d = [s // p for s, p in zip(spatial_size, self.patch_size)]
        ## possible latent space
        x = torch.reshape(
            concated_masked, [concated_masked.shape[0], concated_masked.shape[1], *d]
        )
        print(f"Shape of x: {x.shape}")
        # decoder
        x = self.decoder(x)
        print(f"Shape of x after decoder: {x.shape}")
        return x, hidden_states_out

    def decoder(self, x):
        """
        Forward pass of the decoder.
        Parameters
        ----------
        x

        Returns
        -------
        """
        x = self.conv3d_transpose(x)
        x = self.conv3d_transpose_1(x)
        return x

    def split_tensor_and_record_new_indexes(
        self, tensor: torch.Tensor, raw_positional_embeddings: torch.Tensor
    ):
        """
        Splits the tensor into masked and unmasked patches and records the new indexes of the patches.
        Parameters
        ----------
        tensor
        raw_positional_embeddings

        Returns
        -------
        unmasked_tensor
        """
        self.mask_index_mapping = {}
        self.masked_tensor = None
        # add the masked patches to the mask_dict
        for index in self.orig_masked_indexes:
            extracol = raw_positional_embeddings[:, index, :].unsqueeze(1)
            # loop through the masked patches and add them to the mask_dict
            if self.masked_tensor is None:
                self.masked_tensor = extracol
                new_index = 0
            else:
                new_index = self.masked_tensor.shape[1]
                self.masked_tensor = torch.cat((self.masked_tensor, extracol), 1)

            self.mask_index_mapping[index] = new_index

        unmasked_tensor = None
        self.unmasked_index_mapping = {}
        # add the unmasked patches to the unmasked_dict
        for index in self.orig_unmasked_indexes:
            extracol = tensor[:, index, :].unsqueeze(1)
            # loop through the unmasked patches and add them to the unmasked_dict
            if unmasked_tensor is None:
                unmasked_tensor = extracol
                new_index = 0
            else:
                new_index = unmasked_tensor.shape[1]
                unmasked_tensor = torch.cat((unmasked_tensor, extracol), 1)
            self.unmasked_index_mapping[index] = new_index

        return unmasked_tensor, self.masked_tensor

    def reconstruct_image(self, normalized_x, orig_tensor):
        """
        Reconstructs the image from the normalized tensor and the original tensor.
        Parameters
        ----------
        normalized_x
        orig_tensor

        Returns
        -------
        reconstructed_image
        """
        # Reconstruct the image from the normalized tensor and the original tensor.
        batch_dim = normalized_x.shape[0]
        if self.masked_tensor.shape[0] != batch_dim:
            # Check if the batch dimension of the masked tensor and the normalized tensor are the same
            raise ValueError(
                "The batch dimension of the masked tensor and the normalized tensor must be the same"
            )
        print(
            "----------------------STARTING RECONSTRUCTION----------------------------------"
        )
        print(f"Shape of normalized tensor: {normalized_x.shape}")
        print(f"Shape of original tensor: {orig_tensor.shape}")
        print(
            "-----------------------END RECONSTRUCTION---------------------------------"
        )

        # Use an index mapping to reconstruct the image
        reconstructed_image = torch.zeros_like(orig_tensor)
        print("maskIndexMapping: ", len(self.mask_index_mapping.keys()))
        print("unmaskedIndexMapping: ", len(self.unmasked_index_mapping.keys()))
        allIndexes = self.mask_index_mapping.keys() & self.unmasked_index_mapping.keys()
        print("All Indexes: ", len(allIndexes))
        # Loop through the masked indexes and add them to the reconstructed image
        for orig_index, new_index in self.mask_index_mapping.items():
            reconstructed_image[:, orig_index, :] = self.masked_tensor[:, new_index, :]
        # Loop through the unmasked indexes and add them to the reconstructed image
        for orig_index, new_index in self.unmasked_index_mapping.items():
            reconstructed_image[:, orig_index, :] = normalized_x[:, new_index, :]
        return reconstructed_image


# Unit tests for the MAEViTAutoEnc class
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


class TestMAEViTAutoEncReconstruction(unittest.TestCase):
    def setUp(self):
        # Initialize the model with a controlled configuration
        self.model = MAEViTAutoEnc(
            in_channels=1,
            img_size=[28, 28],  # Smaller size for test simplicity
            patch_size=7,
            hidden_size=32,
            mlp_dim=64,
            num_layers=1,
            num_heads=1,
            deconv_chns=16,
            training_unsupervised=True,
            test=True,  # Ensure we're in testing mode if needed
        )
        # Manually set the mask and unmasked indices for predictable behavior
        self.model.orig_masked_indexes = [1, 3]  # Indices of patches that are masked
        self.model.orig_unmasked_indexes = [
            0,
            2,
            4,
        ]  # Indices of patches that are unmasked

        # Mock data for masked and unmasked patches
        self.masked_tensor = torch.tensor([[1000.0, 1000.0], [3000.0, 3000.0]])
        self.normalized_x = torch.tensor(
            [[0.0, 0.0], [2000.0, 2000.0], [4000.0, 4000.0]]
        )
        self.orig_tensor = torch.zeros((1, 5, 2))  # Simulate an original tensor

        # Set these tensors in the model for direct access
        self.model.masked_tensor = self.masked_tensor
        self.model.normalized_x = self.normalized_x

    def test_reconstruct_image(self):
        # Perform the reconstruction
        reconstructed = self.model.reconstruct_image(
            self.normalized_x, self.orig_tensor
        )

        # Expected reconstructed image based on the set indices
        expected_tensor = torch.tensor(
            [
                [0.0, 0.0],
                [1000.0, 1000.0],
                [2000.0, 2000.0],
                [3000.0, 3000.0],
                [4000.0, 4000.0],
            ]
        )
        expected_tensor = expected_tensor.unsqueeze(
            0
        )  # Adding batch dimension for comparison

        # Verify the reconstructed tensor matches the expected tensor
        self.assertTrue(
            torch.equal(reconstructed, expected_tensor),
            f"Reconstructed tensor does not match expected output. Got {reconstructed}",
        )


if __name__ == "__main__":
    unittest.main()
