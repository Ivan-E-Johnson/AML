import argparse
import os

import itk
import matplotlib.pyplot as plt
import monai.visualize
import numpy as np
import pytorch_lightning
import torch
from monai.config import print_config
from monai.data import CacheDataset, list_data_collate
from monai.data import DataLoader
from monai.losses import DiceCELoss, TverskyLoss
from monai.metrics import (
    DiceMetric,
    HausdorffDistanceMetric,
    SurfaceDistanceMetric,
)
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    EnsureChannelFirstD,
    RandFlipd,
    AsDiscreted,
    DataStatsD,
    AsDiscrete,
)
from monai.transforms import (
    LoadImaged,
    ToTensord,
)
from monai.transforms import (
    RandGaussianNoiseD,
    RandHistogramShiftD,
    RandRotated,
)
from monai.visualize import img2tensorboard
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import ReduceLROnPlateau
from pytorch_lightning.profilers import SimpleProfiler
from sklearn.model_selection import train_test_split

from support_functions import (
    init_stacked_data_lists,
    ModalityStackTransformd,
    BestModelCheckpoint,
    LoadAndSplitLabelsToChannelsd,
    convert_logits_to_one_hot,
)

from pathlib import Path

# weights obtained from running weight_labels() in itk_preprocessing.py
label_weights = {
    "Background": 94.66189125389738,
    "Peripheral Zone": 1.6434384300595233,
    "Transition Zone": 3.4090448842297345,
    "Distal Prostatic Urethra": 0.26201520647321425,
    "Fibromuscular Stroma": 0.023610225340136057,
}
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class StackedNet(pytorch_lightning.LightningModule):
    """
    This class defines the network architecture.

    Attributes:
    _model (UNet): The UNet model.
    dice_loss_function (DiceLoss): The loss function.
    post_pred (Compose): The post prediction transformations.
    post_label (Compose): The post label transformations.
    dice_metric (DiceMetric): The dice metric.
    best_val_dice (float): The best validation dice score.
    best_val_epoch (int): The epoch with the best validation dice score.
    validation_step_outputs (list): The outputs of the validation step.
    """

    # More aggressive early downsampling: (4, 2, 2, 2)
    # Default downsampling: (2, 2, 2, 2)
    # More gradual downsampling: (2, 2, 1, 1)
    def __init__(
        self,
        batch_size: int,
        learning_rate: float,
        dropout: float,
        norm: Norm,
        number_res_units: int,
        kernel_size: int,
        kernel_upsample: int,
        channels: tuple,
        strides: tuple,
        activation: str,
        is_testing=False,
        multi_gpu=False,
    ):

        super().__init__()
        self.number_of_classes = 5  # INCLUDES BACKGROUND
        self.is_testing = is_testing
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.multi_gpu = multi_gpu

        self._model = UNet(
            spatial_dims=3,
            in_channels=3,
            out_channels=self.number_of_classes,
            channels=channels,  # Number of features in each layer
            strides=strides,
            num_res_units=number_res_units,
            norm=norm,
            kernel_size=kernel_size,
            up_kernel_size=kernel_upsample,
            dropout=dropout,
            act=activation,
        )
        # TODO Add the weights to the loss function and allow multichannel output

        class_weights = torch.tensor([0.0002, 0.6085, 0.2933, 3.8166, 42.3545]).to(
            self.device
        )

        self.dice_loss_function = DiceCELoss(
            softmax=True,
            ce_weight=class_weights,
            include_background=False,
        )
        self.tv_loss = TverskyLoss(
            softmax=True,
            alpha=0.3,
            beta=0.7,
        )
        self.dice_metric = DiceMetric(
            include_background=False,
            num_classes=self.number_of_classes,
        )
        self.hausdorff_metric = HausdorffDistanceMetric(
            include_background=False,
        )
        self.surface_distance_metric = SurfaceDistanceMetric(
            include_background=False,
        )

        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.prepare_data()
        self.save_hyperparameters()

    def forward(self, x):
        """
        This function defines the forward pass of the network.

        Args:
        x (Tensor): The input tensor.

        Returns:
        Tensor: The output tensor.
        """
        return self._model(x)

    def prepare_data(self):
        """
        This function prepares the data for training and validation.
        """
        # set up the correct data path

        image_paths, mask_paths = init_stacked_data_lists()
        train_image_paths, test_image_paths, train_mask_paths, test_mask_paths = (
            train_test_split(image_paths, mask_paths, test_size=0.2)
        )
        # Prepare data
        train_files = [
            {"image": img_path, "label": mask_path}
            for img_path, mask_path in zip(train_image_paths, train_mask_paths)
        ]
        val_files = [
            {"image": img_path, "label": mask_path}
            for img_path, mask_path in zip(test_image_paths, test_mask_paths)
        ]
        if self.is_testing:
            train_files = train_files[:5]
            val_files = val_files[:1]

        # set the data transforms
        RandFlipd_prob = 0.5

        # set deterministic training for reproducibility

        # define the data transforms
        self.train_transforms = Compose(
            [
                LoadAndSplitLabelsToChannelsd(keys=["label"]),
                ModalityStackTransformd(keys=["image"]),
                EnsureChannelFirstD(
                    keys=["image", "label"], channel_dim=0
                ),  # Add channel to image and mask so
                # Coarse Segmentation combine all mask
                RandFlipd(keys=["image", "label"], prob=RandFlipd_prob, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=RandFlipd_prob, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=RandFlipd_prob, spatial_axis=2),
                RandRotated(
                    keys=["image", "label"],
                    range_x=15,
                    range_y=15,
                    range_z=15,
                    prob=RandFlipd_prob,
                ),
                RandGaussianNoiseD(keys=["image"], prob=RandFlipd_prob / 2),
                RandHistogramShiftD(keys=["image"], prob=RandFlipd_prob / 2),
                ToTensord(keys=["image", "label"]),
                # DataStatsD(keys=["image", "label"]),
            ]
        )
        self.validation_transforms = Compose(
            [
                LoadAndSplitLabelsToChannelsd(keys=["label"]),
                ModalityStackTransformd(keys=["image"]),
                EnsureChannelFirstD(
                    keys=["image", "label"], channel_dim=0
                ),  # Add channel to image and mask so
                DataStatsD(keys=["image", "label"]),
                EnsureChannelFirstD(
                    keys=["image"], channel_dim=0
                ),  # Add channel to image and mask so
                EnsureChannelFirstD(keys=["label"]),
                ToTensord(keys=["image", "label"]),
                # DataStatsD(keys=["image", "label"]),
            ]
        )

        # we use cached datasets - these are 10x faster than regular datasets
        self.train_ds = CacheDataset(
            data=train_files,
            transform=self.train_transforms,
            cache_rate=0.8,
            num_workers=4,
            runtime_cache=True,
        )
        self.val_ds = CacheDataset(
            data=val_files,
            transform=self.validation_transforms,
            cache_rate=0.8,
            num_workers=4,
            runtime_cache=True,
        )

    def train_dataloader(self):
        """
        This function returns the training data loader.

        Returns:
        DataLoader: The training data loader.
        """
        train_loader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=12,
            collate_fn=list_data_collate,
        )
        return train_loader

    def val_dataloader(self):
        """
        This function returns the validation data loader.

        Returns:
        DataLoader: The validation data loader.
        """

        val_loader = DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=8)
        return val_loader

    def configure_optimizers(self):
        torch.enable_grad()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # return optimizer
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=10,
            verbose=True,
            monitor="val_loss",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def training_step(self, batch, batch_idx):
        """
        This function defines the training step.

        Args:
        batch (dict): The batch of data.
        batch_idx (int): The index of the batch.

        Returns:
        dict: The loss and the logs.
        """
        torch.enable_grad()
        images, labels = batch["image"], batch["label"]
        outputs = self.forward(images)

        if self.is_testing:
            print("Training Step")
            print(f"Images.Shape = {images.shape}")
            print(f"Labels.Shape = {labels.shape}")
            print(f"outputs shape {outputs.shape}")
        loss = self.dice_loss_function(outputs, labels)
        tv_loss = self.tv_loss(outputs, labels)
        onehot_predictions = convert_logits_to_one_hot(outputs)
        dice = self.dice_metric(y_pred=onehot_predictions, y=labels).mean()
        # Log metrics
        self.log(
            "train_dice",
            dice,
            on_step=False,
            on_epoch=True,
            reduce_fx=torch.mean,
            sync_dist=self.multi_gpu,
            batch_size=self.batch_size,
        )
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            reduce_fx=torch.mean,
            sync_dist=self.multi_gpu,
            batch_size=self.batch_size,
        )
        self.log(
            "train_tv_loss",
            tv_loss,
            on_step=False,
            on_epoch=True,
            reduce_fx=torch.mean,
            sync_dist=self.multi_gpu,
            batch_size=self.batch_size,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        torch.no_grad()
        images, labels = batch["image"], batch["label"]
        outputs = self.forward(images)

        if self.is_testing:
            print("Validation Step")
            print(f"Images.Shape = {images.shape}")
            print(f"Labels.Shape = {labels.shape}")
            print(f"Shape after post_pred: {outputs.shape}")

        loss = self.dice_loss_function(outputs, labels)
        tv_loss = self.tv_loss(outputs, labels)
        onehot_predictions = convert_logits_to_one_hot(outputs)
        # Calculate accuracy, precision, recall, and F1 score
        dice = self.dice_metric(y_pred=onehot_predictions, y=labels).mean()
        haussdorf = self.hausdorff_metric(y_pred=onehot_predictions, y=labels).mean()
        surface_distance = self.surface_distance_metric(
            y_pred=onehot_predictions, y=labels
        ).mean()

        if self.is_testing:
            print(f"Predictions shape: {outputs.shape}")
            print(f"Dice: {dice}")
            print(f"Haussdorf: {haussdorf}")
            print(f"Surface Distance: {surface_distance}")

        self.log(
            name="haussdorf_distance",
            value=haussdorf,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
            sync_dist=self.multi_gpu,
        )
        self.log(
            name="tv_loss",
            value=tv_loss,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
            sync_dist=self.multi_gpu,
        )
        self.log(
            name="surface_distance",
            value=surface_distance,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
            sync_dist=self.multi_gpu,
        )
        # Log metrics
        self.log(
            "val_dice",
            dice,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
            sync_dist=self.multi_gpu,
        )
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
            sync_dist=self.multi_gpu,
        )
        torch.enable_grad()
        return loss

    def on_validation_epoch_end(self):
        torch.no_grad()
        # Get the first batch of the validation data
        val_loader = self.val_dataloader()
        images, labels = (
            next(iter(val_loader))["image"],
            next(iter(val_loader))["label"],
        )

        # Move images and labels to device
        images = images.to(self.device)
        labels = labels.to(self.device)

        # Onlys select the first image and label

        # Run the inference
        outputs = self.forward(images)
        outputs = convert_logits_to_one_hot(outputs)

        print(f"Outputs values: {np.unique(outputs.cpu().numpy())}")
        print(f"Output shape: {outputs.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Images shape: {images.shape}")

        monai.visualize.plot_2d_or_3d_image(
            data=labels,
            tag=f"Labels",
            step=self.current_epoch,
            writer=self.logger.experiment,
            max_channels=self.number_of_classes,
            frame_dim=-1,
        )
        monai.visualize.plot_2d_or_3d_image(
            data=outputs,
            tag=f"Predictions",
            step=self.current_epoch,
            writer=self.logger.experiment,
            max_channels=self.number_of_classes,
            frame_dim=-1,
        )

        torch.enable_grad()


def train_model(
    learning_rate: float,
    batch_size: int,
    dropout_prob: float,
    kernel_size: int,
    kernel_upsample: int,
    number_res_units: int,
    strides: tuple,
    channels: tuple,
    activation: str,
    experiment_name: str,
    epochs: int,
    reload_dataloaders_every_n_epochs: int = 10,
    using_multi_gpu: bool = False,
    is_testing: bool = False,
):
    os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    accelerator = os.environ.get("ACCELERATOR", "gpu")
    gpu_id = os.environ.get("GPU_ID", 0)

    devices: list[int] | str
    if accelerator == "cpu":
        using_multi_gpu = False
        devices = "auto"
    else:
        devices = [int(gpu_id)]
    print("*" * 80)
    print(f"Cuda available: {torch.cuda.is_available()}")
    print(f"Using Multi GPU: {using_multi_gpu}")
    print(f"Devices: {devices}")
    print(f"Accelerator: {accelerator}")
    print("*" * 80)

    net = StackedNet(
        batch_size=batch_size,
        learning_rate=learning_rate,
        dropout=dropout_prob,
        norm=Norm.BATCH,
        kernel_size=kernel_size,
        kernel_upsample=kernel_upsample,
        strides=strides,
        channels=channels,
        activation=activation,
        number_res_units=number_res_units,
        is_testing=True,
        multi_gpu=using_multi_gpu,
    )
    current_file_loc = Path(__file__).parent
    log_dir = current_file_loc / "logs"
    tb_logger = pytorch_lightning.loggers.TensorBoardLogger(
        save_dir=log_dir.as_posix(), name=experiment_name
    )

    checkpoint_fn = experiment_name + "checkpoint-{epoch:02d}-{val_dice:.2f}"
    checkpoint_callback = ModelCheckpoint(
        monitor="val_dice",
        mode="max",
        save_last=True,
        dirpath=log_dir.as_posix(),
        filename=checkpoint_fn,
    )
    log_every_n_steps = int((98 * 0.8) // batch_size)
    trainer = pytorch_lightning.Trainer(
        max_epochs=epochs,
        logger=tb_logger,
        accelerator=accelerator,
        devices=devices,
        enable_checkpointing=True,
        num_sanity_val_steps=1,
        log_every_n_steps=log_every_n_steps,
        reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
        # profiler=profiler,
        callbacks=[
            BestModelCheckpoint(
                monitor="val_dice", mode="max", experiment_name=experiment_name
            ),
            checkpoint_callback,
        ],  # Add the custom callback
    )

    trainer.fit(net)

    print("Finished training")
    return net.best_val_dice.cpu()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--learning_rate", type=float, default=1e-3)
    args.add_argument("--batch_size", type=int, default=10)
    args.add_argument("--dropout_prob", type=float, default=0.2)
    args.add_argument("--kernel_size", type=int, default=5)
    args.add_argument("--kernel_upsample", type=int, default=5)
    args.add_argument("--number_res_units", type=int, default=4)
    args.add_argument("--strides", type=tuple, default=(2, 2, 2, 2))
    args.add_argument("--channels", type=tuple, default=(16, 32, 128, 256))
    args.add_argument("--activation", type=str, default="PReLU")
    args.add_argument(
        "--experiment_name", type=str, default="basic_stacked_unet_stacked_labels"
    )
    args.add_argument("--epochs", type=int, default=1000)
    # args.add_argument("--using_multi_gpu", type=bool, default=False)
    # args.add_argument("--is_testing", type=bool, default=False)
    args = args.parse_args()

    model_params = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "dropout_prob": args.dropout_prob,
        "kernel_size": args.kernel_size,
        "kernel_upsample": args.kernel_upsample,
        "number_res_units": args.number_res_units,
        "strides": args.strides,
        "channels": args.channels,
        "activation": args.activation,
        "epochs": args.epochs,
        "experiment_name": args.experiment_name,
        # "using_multi_gpu": args.using_multi_gpu,
        # "is_testing": args.is_testing,
    }

    train_model(**model_params)
