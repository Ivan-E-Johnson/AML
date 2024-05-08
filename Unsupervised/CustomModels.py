import torch
from monai.networks.layers import Norm, Act
from monai.networks.nets.flexible_unet import UNetDecoder
from torch import nn as nn


class SplitAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, hidden_size, patch_size):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.norm = nn.LayerNorm(hidden_size)
        self.patch_size = patch_size

    def forward(self, x):
        spatial_size = x.shape[2:]
        x, down_x = self.encoder(x)

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
        self.activation = nn.LeakyReLU()
        self.normalize1 = nn.BatchNorm3d(decov_chns)
        self.normalize2 = nn.BatchNorm3d(out_channels)
        self.classification = classsification

    def forward(self, x):
        x = self.conv3d_transpose(x)
        if self.classification:
            x = self.activation(x)
            x = self.normalize1(x)
        x = self.conv3d_transpose_1(x)
        if (
            self.classification
        ):  # Assuming you want to add an activation function at the end
            x = self.activation(x)
            x = self.normalize2(x)
        return x


# class ClassificationDecoder(nn.Module):
#     def __init__(self, hidden_size, decov_chns, up_kernel_size, out_channels, dropout_rate=0.1,
#                  classification=False):
#         super(CustomDecoder, self).__init__()
#         conv_trans = nn.ConvTranspose3d  # Using 3D convolution transpose as assumed
#         self.up1 = conv_trans(in_channels=hidden_size, out_channels=decov_chns, kernel_size=up_kernel_size,
#                               stride=2)
#         self.norm1 = Norm.BATCH(decov_chns)  # Using Monai's batch normalization
#         self.act1 = Act.RELU()  # Using Monai's ReLU activation
#         self.dropout1 = nn.Dropout3d(dropout_rate)  # Adding dropout layer for regularization
#
#         # Adjusting in_channels to account for concatenated skip connection
#         self.up2 = conv_trans(in_channels=decov_chns + hidden_size, out_channels=out_channels,
#                               kernel_size=up_kernel_size, stride=2)
#         self.norm2 = Norm.BATCH(out_channels)  # Adding normalization to the second layer
#         self.act2 = Act.RELU()  # Activation for the second layer
#         self.dropout2 = nn.Dropout3d(dropout_rate)  # Dropout for the second layer
#
#         self.classification = classification
#
#     def forward(self, x, encoder_features):
#
#         for i in range(len(encoder_features)):
#             print(f"Encoder Features: {encoder_features[i].shape}")
#             x
#
#         # Up-sampling and integrating first set of features
#         x = self.up1(x)
#         x = self.norm1(x)
#         x = self.act1(x)
#         x = self.dropout1(x)
#
#         # Concatenate the corresponding encoder features with the current state
#         if len(encoder_features) > 0:
#             x = torch.cat([x, encoder_features[0]], dim=1)  # Adjust dimension as needed
#
#         # Second up-sampling and integration
#         x = self.up2(x)
#         x = self.norm2(x)
#         x = self.act2(x)
#         x = self.dropout2(x)
#
#         return x
#
