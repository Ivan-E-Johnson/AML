import time

import monai
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
import einops
from matplotlib import pyplot as plt
from monai.losses import ContrastiveLoss
from monai.networks.nets import ViTAutoEnc
from monai.utils import set_determinism, first
from monai.data import FolderLayout

from monai.data import PatchIter, GridPatchDataset, DataLoader, CacheDataset
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
from sklearn.model_selection import train_test_split
from torch.nn import L1Loss

from support_functions import *


def _create_dict_of_preprocessed_unlabeled_data(
    base_data_path: Path, is_testing: bool = False
) -> list:
    data_dicts = []

    for dir in base_data_path.iterdir():
        if dir.is_dir():
            images_list = []
            images_list.append(dir / f"{dir.name}_pp_t2w.nii.gz")
            images_list.append(dir / f"{dir.name}_pp_adc.nii.gz")
            images_list.append(dir / f"{dir.name}_pp_blow.nii.gz")
            images_list.append(dir / f"{dir.name}_pp_tracew.nii.gz")
            data_dicts.append({"image": images_list})
    if is_testing:
        data_dicts = data_dicts[:10]
    return data_dicts


def _create_image_dict(base_data_path: Path, is_testing: bool = False) -> list:
    data_dicts = []
    for dir in base_data_path.iterdir():
        if dir.is_dir():
            data_dicts.append({"image": dir / f"{dir.name}_pp_t2w.nii.gz"})
    if is_testing:
        data_dicts = data_dicts[:10]
    return data_dicts


layout = FolderLayout(
    output_dir="/Users/iejohnson/School/spring_2024/AML/Supervised_learning/DATA/ALL_PROSTATEx/WITHOUT_SEGMENTATION/Patched/",
    postfix="train_transformed",
    parent=True,
    extension="nii.gz",
    makedirs=True,
)


image_size = (240, 240, 16)
patch_size = (16, 16, 16)
# num_samples = 240 // 60 * 240 // 60 * 16 // 4
num_samples = 4**3
train_transforms = Compose(
    [
        # ModalityStackTransformd(keys=["image"]),
        LoadImageD(keys=["image"], image_only=False),
        DataStatsD(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        DataStatsD(keys=["image"]),
        # CropForegroundd(keys=["image"], source_key="image", k_divisible=4),
        # RandSpatialCropSamplesd(keys=["image"], roi_size=patch_size, random_size=False),
        DataStatsD(keys=["image"]),
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
        # SaveImageD(keys=["contrastive_patched"], folder_layout=layout),
    ]
)


if __name__ == "__main__":
    base_data_path = Path("/ALL_PROSTATEx/WITHOUT_SEGMENTATION/PreProcessed")
    print(f"Base data path: {base_data_path}")
    data_dicts = _create_image_dict(base_data_path, is_testing=True)
    train_image_paths, test_image_paths = train_test_split(data_dicts, test_size=0.2)

    print(f"Data dicts: {data_dicts}")

    cache_dataset = CacheDataset(
        data=train_image_paths,
        transform=train_transforms,
        cache_rate=1.0,
        num_workers=2,
    )
    val_cache_dataset = CacheDataset(data=test_image_paths, transform=train_transforms)
    images = first(cache_dataset)
    val_loader = DataLoader(
        val_cache_dataset, batch_size=4, shuffle=True, num_workers=2
    )
    train_loader = DataLoader(cache_dataset, batch_size=4, shuffle=True, num_workers=2)

    logdir = "/Users/iejohnson/School/spring_2024/AML/Supervised_learning/DATA/ALL_PROSTATEx/WITHOUT_SEGMENTATION/logs"
    Path(logdir).mkdir(parents=True, exist_ok=True)
    print("item size:", images)
    print("CacheDataset: ", CacheDataset)

    model = ViTAutoEnc(
        in_channels=1,
        img_size=image_size,
        patch_size=patch_size,
        pos_embed="conv",
        hidden_size=768,
        mlp_dim=3072,
    )
    device = torch.device("cpu")

    model = model.to(device)

    # Define Hyper-paramters for training loop
    max_epochs = 500
    val_interval = 2
    batch_size = 4
    lr = 1e-4
    epoch_loss_values = []
    step_loss_values = []
    epoch_cl_loss_values = []
    epoch_recon_loss_values = []
    val_loss_values = []
    best_val_loss = 1000.0

    recon_loss = L1Loss()
    contrastive_loss = ContrastiveLoss(temperature=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        epoch_cl_loss = 0
        epoch_recon_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            start_time = time.time()

            inputs, inputs_2, gt_input = (
                batch_data["image"].to(device),
                batch_data["contrastive_patched"].to(device),
                batch_data["reference_patched"].to(device),
            )
            print(f"I1: {inputs.shape}, I2: {inputs_2.shape}, GT: {gt_input.shape}")
            optimizer.zero_grad()
            outputs_v1, hidden_v1 = model(inputs)
            outputs_v2, hidden_v2 = model(inputs_2)
            print(f"outputs_v1: {outputs_v1.shape}, outputs_v2: {outputs_v2.shape}")
            flat_out_v1 = outputs_v1.flatten(start_dim=1, end_dim=4)
            flat_out_v2 = outputs_v2.flatten(start_dim=1, end_dim=4)
            print(f"flat_out_v1: {flat_out_v1.shape}, flat_out_v2: {flat_out_v2.shape}")
            print(f"GT: {gt_input.shape}")
            r_loss = recon_loss(outputs_v1, gt_input)
            cl_loss = contrastive_loss(flat_out_v1, flat_out_v2)

            # Adjust the CL loss by Recon Loss
            total_loss = r_loss + cl_loss * r_loss

            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            step_loss_values.append(total_loss.item())

            # CL & Recon Loss Storage of Value
            epoch_cl_loss += cl_loss.item()
            epoch_recon_loss += r_loss.item()

            end_time = time.time()
            print(
                f"{step}/{len(cache_dataset) // train_loader.batch_size}, "
                f"train_loss: {total_loss.item():.4f}, "
                f"time taken: {end_time - start_time}s"
            )

        epoch_loss /= step
        epoch_cl_loss /= step
        epoch_recon_loss /= step

        epoch_loss_values.append(epoch_loss)
        epoch_cl_loss_values.append(epoch_cl_loss)
        epoch_recon_loss_values.append(epoch_recon_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if epoch % val_interval == 0:
            print("Entering Validation for epoch: {}".format(epoch + 1))
            total_val_loss = 0
            val_step = 0
            model.eval()
            for val_batch in val_loader:
                val_step += 1
                start_time = time.time()
                inputs, gt_input = (
                    val_batch["image"].to(device),
                    val_batch["reference_patched"].to(device),
                )
                print("Input shape: {}".format(inputs.shape))
                outputs, outputs_v2 = model(inputs)

                # TODO Add visualization of the latent space

                val_loss = recon_loss(outputs, gt_input)
                total_val_loss += val_loss.item()
                end_time = time.time()

            total_val_loss /= val_step
            val_loss_values.append(total_val_loss)
            print(
                f"epoch {epoch + 1} Validation avg loss: {total_val_loss:.4f}, "
                f"time taken: {end_time - start_time}s"
            )

            if total_val_loss < best_val_loss:
                print(f"Saving new model based on validation loss {total_val_loss:.4f}")
                best_val_loss = total_val_loss
                checkpoint = {
                    "epoch": max_epochs,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(checkpoint, os.path.join(logdir, "best_model.pt"))

            plt.figure(1, figsize=(8, 8))
            plt.subplot(2, 2, 1)
            plt.plot(epoch_loss_values)
            plt.grid()
            plt.title("Training Loss")

            plt.subplot(2, 2, 2)
            plt.plot(val_loss_values)
            plt.grid()
            plt.title("Validation Loss")

            plt.subplot(2, 2, 3)
            plt.plot(epoch_cl_loss_values)
            plt.grid()
            plt.title("Training Contrastive Loss")

            plt.subplot(2, 2, 4)
            plt.plot(epoch_recon_loss_values)
            plt.grid()
            plt.title("Training Recon Loss")

            plt.savefig(os.path.join(logdir, "loss_plots.png"))
            plt.close(1)

    print("Done")
