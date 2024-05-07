import torch
from torch import nn as nn


class SplitAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, hidden_size, patch_size):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.norm = nn.LayerNorm(hidden_size)
        self.patch_size = patch_size

    def forward(self, x):
        print(f"Orig X Shape: {x.shape}")
        spatial_size = x.shape[2:]
        x, down_x = self.encoder(x)

        print(f"X Shape: {x.shape}")
        # print(f"Down X Shape: {down_x}")
        x = self.norm(x)
        x = x.transpose(1, 2)
        d = [s // p for s, p in zip(spatial_size, self.patch_size)]
        x = torch.reshape(x, [x.shape[0], x.shape[1], *d])
        x = self.decoder(x)
        return x, down_x


class CustomDecoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        decov_chns,
        up_kernel_size,
        out_channels,
        classsification=False,
    ):
        super().__init__()
        conv_trans = (
            nn.ConvTranspose3d
        )  # Assuming you are using 3D convolution transpose
        self.conv3d_transpose = conv_trans(
            in_channels=hidden_size,
            out_channels=decov_chns,
            kernel_size=up_kernel_size,
            stride=up_kernel_size,
        )
        self.conv3d_transpose_1 = conv_trans(
            in_channels=decov_chns,
            out_channels=out_channels,
            kernel_size=up_kernel_size,
            stride=up_kernel_size,
        )
        # Optionally add normalization and activation layers if needed
        self.activation = nn.ReLU()
        self.classification = classsification

    def forward(self, x):
        x = self.conv3d_transpose(x)
        if self.classification:
            x = self.activation(x)
        x = self.conv3d_transpose_1(x)
        if (
            self.classification
        ):  # Assuming you want to add an activation function at the end
            x = self.activation(x)
        return x
