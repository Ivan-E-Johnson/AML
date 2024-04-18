import os
from pathlib import Path

import pytorch_lightning as pl
from monai.data import DataLoader, CacheDataset
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from mae_vit_model import MAEViTAutoEnc
from monai.networks.nets import ViTAutoEnc
import torch.nn as nn
from monai.losses import ContrastiveLoss
import torch
from monai.transforms import (
    Compose,
    DataStatsD,
    EnsureChannelFirstD,
    ToTensord,
    EnsureChannelFirstd,
    CropForegroundd,
    RandSpatialCropSamplesd,
    RandCoarseDropoutd,
    OneOf,
    RandCoarseShuffled,
    CopyItemsd,
    SaveImageD,
    LoadImageD,
)


def _create_image_dict(base_data_path: Path, is_testing: bool = False) -> list:
    data_dicts = []
    for dir in base_data_path.iterdir():
        if dir.is_dir():
            data_dicts.append({"image": dir / f"{dir.name}_pp_t2w.nii.gz"})
    if is_testing:
        data_dicts = data_dicts[:10]
    return data_dicts


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, img_size, patch_size, in_channels, lr=1e-4):
        super().__init__()
        # self.model = ViTAutoEnc(
        #     in_channels=in_channels,
        #     img_size=img_size,
        #     patch_size=patch_size,
        #     pos_embed="conv",
        #     hidden_size=768,
        #     mlp_dim=3072,
        # )
        self.model = MAEViTAutoEnc(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            proj_type="conv",
            hidden_size=768,
            mlp_dim=3072,
        )
        self.batch_size = 4
        self.number_workers = 4
        self.cache_rate = 0.8

        self.recon_loss = nn.L1Loss()
        self.contrastive_loss = ContrastiveLoss(temperature=0.05)
        self.lr = lr

        # base_data_path = Path(os.enviorn["DATA_PATH"])
        base_data_path = Path(
            "/Users/iejohnson/School/spring_2024/AML/Supervised_learning/DATA/ALL_PROSTATEx/WITHOUT_SEGMENTATION/PreProcessed"
        )
        data_dicts = _create_image_dict(base_data_path, is_testing=True)
        train_image_paths, test_image_paths = train_test_split(
            data_dicts, test_size=0.2
        )

        self.train_dataset = CacheDataset(
            data=train_image_paths,
            transform=self._get_train_transforms(),
            cache_rate=self.cache_rate,
            num_workers=self.number_workers,
            runtime_cache=True,
        )
        self.val_dataset = CacheDataset(
            data=test_image_paths,
            transform=self._get_train_transforms(),
            cache_rate=self.cache_rate,
            num_workers=self.number_workers,
            runtime_cache=True,
        )

    def _get_train_transforms(self):
        return Compose(
            [
                # ModalityStackTransformd(keys=["image"]),
                LoadImageD(keys=["image"], image_only=False),
                DataStatsD(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                # DataStatsD(keys=["image"]),
                # CropForegroundd(keys=["image"], source_key="image", k_divisible=4),
                # RandSpatialCropSamplesd(keys=["image"], roi_size=patch_size, random_size=False),
                # DataStatsD(keys=["image"]),
                CopyItemsd(
                    keys=["image"],
                    times=2,
                    names=["reference_patched", "contrastive_patched"],
                    allow_missing_keys=False,
                ),
                OneOf(
                    transforms=[
                        RandCoarseDropoutd(
                            keys=["contrastive_patched"],
                            prob=1.0,
                            holes=6,
                            spatial_size=5,
                            dropout_holes=True,
                            max_spatial_size=32,
                        ),
                        RandCoarseDropoutd(
                            keys=["contrastive_patched"],
                            prob=1.0,
                            holes=6,
                            spatial_size=20,
                            dropout_holes=False,
                            max_spatial_size=64,
                        ),
                    ]
                ),
                RandCoarseShuffled(
                    keys=["contrastive_patched"], prob=0.8, holes=10, spatial_size=8
                ),
                ToTensord(keys=["image", "reference_patched", "contrastive_patched"]),
                # SaveImageD(keys=["contrastive_patched"], folder_layout=layout),
            ]
        )

    def train_dataloader(self):
        print(f"Train Dataset: {len(self.train_dataset)}")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        print(f"Train Dataset: {len(self.train_dataset)}")
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, inputs_2, gt_input = (
            batch["image"],
            batch["contrastive_patched"],
            batch["reference_patched"],
        )
        # inputs, gt_input = batch['image'], batch['reference_patched']
        #

        outputs_v1, latent_v1 = self.forward(inputs)
        outputs_v2, latent_v2 = self.forward(inputs_2)
        r_loss = self.recon_loss(outputs_v1, gt_input)
        cl_loss = self.contrastive_loss(
            outputs_v1.flatten(start_dim=1), outputs_v2.flatten(start_dim=1)
        )
        loss = r_loss + cl_loss * r_loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, gt_input = batch["image"], batch["reference_patched"]
        outputs, _ = self.forward(inputs)
        val_loss = self.recon_loss(outputs, gt_input)
        self.log("val_loss", val_loss)
        return val_loss

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def save_latent_space(self, latent, filename):
        torch.save(latent, filename)


# Usage with your datasets


def do_main():

    model = LitAutoEncoder(
        img_size=(240, 240, 16), patch_size=(16, 16, 16), in_channels=1
    )
    trainer = pl.Trainer(max_epochs=10, accelerator="cpu")
    trainer.fit(model)


if __name__ == "__main__":
    do_main()
