from typing import Sequence

import torch
from monai.networks.blocks import UnetrPrUpBlock
from monai.networks.layers import Norm, Act
from monai.networks.nets import UNETR
from monai.networks.nets.flexible_unet import UNetDecoder
from torch import nn as nn

from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.networks.blocks.segresnet_block import (
    ResBlock,
    get_conv_layer,
    get_upsample_layer,
)


class SplitAutoEncoder(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        hidden_size,
        patch_size,
        freeze_encoder,
        pass_hidden_to_decoder=False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.norm = nn.LayerNorm(hidden_size)
        self.patch_size = patch_size
        self.pass_hidden_to_decoder = pass_hidden_to_decoder

    def forward(self, x):
        spatial_size = x.shape[2:]
        x, down_x = self.encoder(x)

        # print(f"Down X Shape: {down_x}")
        x = self.norm(x)
        x = x.transpose(1, 2)
        d = [s // p for s, p in zip(spatial_size, self.patch_size)]
        x = torch.reshape(x, [x.shape[0], x.shape[1], *d])
        if self.pass_hidden_to_decoder:
            x = self.decoder(x, down_x)
        else:
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
        self.activation = nn.LeakyReLU()
        # Optionally add normalization and activation layers if needed
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


class CustomDecoder3D(nn.Module):
    def __init__(
        self,
        hidden_size,
        initial_channels,
        final_channels,
        patch_size,
        img_dim,
        num_up_blocks=2,
        kernel_size=3,
        upsample_kernel_size=2,
        norm_name="batch",
        conv_block=True,
        res_block=True,
    ):
        super().__init__()

        # Calculate the number of patches and the dimensionality of each patch feature
        # Assuming img_dim is a tuple (D, H, W)
        self.num_patches = (
            (img_dim[0] // patch_size[0])
            * (img_dim[1] // patch_size[1])
            * (img_dim[2] // patch_size[2])
        )
        self.hidden_size = hidden_size
        self.final_channels = final_channels
        self.num_up_blocks = num_up_blocks
        self.kernel_size = kernel_size
        self.upsample_kernel_size = upsample_kernel_size
        self.norm_name = norm_name
        self.patch_size = patch_size
        self.img_dim = img_dim

        # Initial layer to reshape and upsample the sequence to a feature map
        self.initial_upsample = nn.ConvTranspose3d(
            in_channels=hidden_size,
            out_channels=initial_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Upsampling blocks
        self.blocks = nn.ModuleList()
        current_channels = initial_channels
        for _ in range(num_up_blocks):
            out_channels = current_channels // 2  # Example for channel reduction
            block = UnetrPrUpBlock(
                spatial_dims=3,  # Since it's 3D
                in_channels=current_channels,
                out_channels=out_channels,
                num_layer=1,
                kernel_size=kernel_size,
                stride=1,
                upsample_kernel_size=upsample_kernel_size,
                norm_name=norm_name,
                conv_block=conv_block,
                res_block=res_block,
            )
            self.blocks.append(block)
            current_channels = out_channels

        # Final layer to match the desired output channels
        self.final_conv = nn.Conv3d(current_channels, final_channels, kernel_size=1)

    def forward(self, x, down_x):
        # Process input through the initial upsample
        print(f"Original X Shape: {x.shape}")
        print(f"Original Down X Shape: {down_x[0].shape}")
        batch_size = x.shape[0]
        x = x.permute(0, 2, 1).reshape(
            batch_size, self.initial_channels, *self.patch_size
        )
        print(f"X After Reshaping Shape: {x.shape}")
        x = self.initial_upsample(x)
        print(f"X After Initial Upsample Shape: {x.shape}")

        # Iterate over blocks, adding skip connections
        for block, dx in zip(self.blocks, down_x):
            print(f"X Shape: {x.shape} | DX Shape: {dx.shape}")
            x = torch.cat([x, dx], dim=1)  # Concatenate feature maps from encoder
            print(f"X After Concatenation Shape: {x.shape}")
            x = block(x)

        x = self.final_conv(x)
        print(f"X After Final Conv Shape: {x.shape}")
        return x

    import torch
    import torch.nn as nn
    from torch.nn.functional import interpolate


class ModifiedUnetR(UNETR):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        checkpoint_path: str,
        img_size: Sequence[int] | int,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "conv",
        proj_type: str = "conv",
        norm_name: tuple | str = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            pos_embed=pos_embed,
            proj_type=proj_type,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            qkv_bias=qkv_bias,
            save_attn=save_attn,
        )

        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage
        )
        self.vit.load_state_dict(checkpoint["encoder"], strict=False, assign=True)
