import pytorch_lightning as pl
import torch
from monai.networks.blocks import PatchEmbeddingBlock
from torch import nn
from torch.nn.functional import unfold


class GenericMAE(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        input_channels,
        image_size,
        patch_size,
        hidden_size,
        num_heads,
        proj_type,
        dropout_rate,
        mask_rate=0.75,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_rate = mask_rate

        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=input_channels,
            img_size=image_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            proj_type=proj_type,
            dropout_rate=dropout_rate,
            spatial_dims=self.spatial_dims,
        )
        self.mask_rate = mask_rate
        self.num_patches = self.patch_embedding.n_patches
        self.mask_indices = torch.randperm(self.num_patches)[
            : int(self.num_patches * mask_rate)
        ]
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, training=True):
        patch_embeddings = self.patch_embedding(x)
        if training:
            batch_size = x.shape[0]
            raw_positional_embeddings = self.patch_embedding.position_embeddings
            masked_patches = raw_positional_embeddings.repeat(batch_size, 1, 1)
            unmasked_tensor, masked_patches = self.split_tensor_and_record_new_indexes(
                patch_embeddings, masked_patches
            )
            encoded_patches = self.encoder(masked_patches)
            encoded_patches = self.norm(encoded_patches)
            recustructed_patches = self.reconstruct_image(
                encoded_patches, patch_embeddings
            )
            spatial_size = x.shape[2:]
            d = [s // p for s, p in zip(spatial_size, self.patch_size)]
            ## possible latent space
            x = torch.reshape(
                recustructed_patches,
                [recustructed_patches.shape[0], recustructed_patches.shape[1], *d],
            )
            return self.decoder(x)
        else:
            return self.decoder(self.encoder(patch_embeddings))

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
