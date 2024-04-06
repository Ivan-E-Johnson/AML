import argparse
import os

import itk
import matplotlib.pyplot as plt
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
from monai.networks.nets import UNet, SegResNet
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
from monai.utils import UpsampleMode
from monai.visualize import img2tensorboard
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profilers import SimpleProfiler
from sklearn.model_selection import train_test_split

from support_functions import (
    init_stacked_data_lists,
    ModalityStackTransformd,
    BestModelCheckpoint,
    LoadAndSplitLabelsToChannelsd,
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

RUN_NAME = "z16_basic_seg_3_29_24"


class SegNet(pytorch_lightning.LightningModule):
    """
    This class defines the network architecture.

    Attributes:
    _model (SegNet): The network architecture.
    loss_function (DiceLoss): The loss function.
    post_pred (Compose): The post prediction transformations.
    post_label (Compose): The post label transformations.
    dice_metric (DiceMetric): The dice metric.
    best_val_dice (float): The best validation dice score.
    best_val_epoch (int): The epoch with the best validation dice score.
    validation_step_outputs (list): The outputs of the validation step.
    """

    def __init__(
        self,
        batch_size: int,
        learning_rate: float,
        init_filters: int,
        dropout_prob: float,
        blocks_down: tuple,
        blocks_up: tuple,
        using_multi_gpu: bool = False,
        is_testing=False,
    ):

        super().__init__()
        self.number_of_classes = 5  # INCLUDES BACKGROUND
        self.is_testing = is_testing
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.multi_gpu = using_multi_gpu

        self._model = SegResNet(
            spatial_dims=3,
            init_filters=init_filters,
            in_channels=3,
            out_channels=self.number_of_classes,
            dropout_prob=dropout_prob,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            upsample_mode=UpsampleMode.DECONV,
        )
        # TODO Add the weights to the loss function and allow multichannel output
        inverted_weights = [1 / v for v in label_weights.values()]
        class_weights = torch.tensor(
            np.array(inverted_weights).astype(np.float16), requires_grad=True
        ).to(self.device)

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
            reduction="mean",
        )
        self.hausdorff_metric = HausdorffDistanceMetric(
            include_background=False,
            reduction="mean",
        )
        self.surface_distance_metric = SurfaceDistanceMetric(
            include_background=False,
            reduction="mean",
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

        return optimizer
        # scheduler = ReduceLROnPlateau(
        #     optimizer,
        #     mode="min",
        #     factor=0.1,
        #     patience=10,
        #     verbose=True,
        #     monitor="val_loss",
        # )
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": scheduler,
        #     "monitor": "val_loss",
        # }

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
        output = self.forward(images)

        if self.is_testing:
            print("Training Step")
            print(f"Images.Shape = {images.shape}")
            print(f"Labels.Shape = {labels.shape}")
            print(f"Output shape {output.shape}")

        loss = self.dice_loss_function(output, labels)
        tv_loss = self.tv_loss(output, labels)

        prob_output = torch.softmax(output, dim=1)
        dice = self.dice_metric(y_pred=prob_output, y=labels).mean()
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
        prob_output = torch.softmax(outputs, dim=1)

        # Calculate accuracy, precision, recall, and F1 score
        dice = self.dice_metric(y_pred=prob_output, y=labels).mean()
        haussdorf = self.hausdorff_metric(y_pred=prob_output, y=labels).mean()
        surface_distance = self.surface_distance_metric(
            y_pred=prob_output, y=labels
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

        # Run the inference
        outputs = self.forward(images)
        print(f"Output shape: {outputs.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Images shape: {images.shape}")
        print(f"Images min: {images.shape[:-3]}")

        # center_slice = images.shape[0] //2
        # images = images[0,:, center_slice, :, :]
        # outputs = outputs[0,:, center_slice, :, :]
        # TODO @joslin can you figure out why these are plotting incorrectly?
        img2tensorboard.plot_2d_or_3d_image(
            writer=self.logger.experiment,
            data=images,
            step=self.current_epoch,
            max_channels=3,
            tag="Validation/Image",
        )

        img2tensorboard.plot_2d_or_3d_image(
            writer=self.logger.experiment,
            data=outputs,
            step=self.current_epoch,
            max_channels=self.number_of_classes,
            tag="Validation/Label",
        )

        # oneHotLabel = AsDiscrete(to_onehot=self.number_of_classes)(labels)
        # img2tensorboard.plot_2d_or_3d_image(
        #     writer=self.logger.experiment,
        #     data=oneHotLabel,
        #     step=self.current_epoch,
        #     max_channels=self.number_of_classes,
        #     tag="Validation/Prediction",
        # )
        torch.enable_grad()


def train_model(
    learning_rate: float,
    batch_size: int,
    dropout_prob: float,
    init_filters: int,
    blocks_down: tuple,
    blocks_up: tuple,
    epochs: int,
    experiment_name: str,
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

    log_every_n_steps: int = int(
        (98 * 0.8) // batch_size
    )  # Log every 80% of the training data

    init_filters = int(init_filters)
    batch_size = int(batch_size)
    epochs = int(epochs)

    net = SegNet(
        learning_rate=learning_rate,
        batch_size=batch_size,
        dropout_prob=dropout_prob,
        using_multi_gpu=False,
        is_testing=False,
        init_filters=init_filters,
        blocks_down=blocks_down,
        blocks_up=blocks_up,
    )
    current_file_loc = Path(__file__).parent
    log_dir = current_file_loc / "logs"
    tb_logger = pytorch_lightning.loggers.TensorBoardLogger(
        save_dir=log_dir.as_posix(), name=experiment_name
    )

    checkpoint_fn = experiment_name + "checkpoint-{epoch:02d}-{val_dice:.2f}"
    # initialise Lightning's trainer.
    checkpoint_callback = ModelCheckpoint(
        monitor="val_dice",
        mode="max",
        save_last=True,
        dirpath=log_dir.as_posix(),
        filename=checkpoint_fn,
    )

    trainer = pytorch_lightning.Trainer(
        max_epochs=epochs,
        logger=tb_logger,
        accelerator=accelerator,
        devices=devices,
        enable_checkpointing=True,
        num_sanity_val_steps=1,
        log_every_n_steps=log_every_n_steps,
        callbacks=[
            BestModelCheckpoint(),
            checkpoint_callback,
        ],  # Add the custom callback
    )

    trainer.fit(net)

    print("Finished training")
    return net.best_val_dice.cpu()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--learning_rate", type=float, default=1e-3)
    args.add_argument("--batch_size", type=int, default=5)
    args.add_argument("--dropout_prob", type=float, default=0.2)
    args.add_argument("--init_filters", type=int, default=8)
    args.add_argument("--blocks_down", type=tuple, default=(1, 2, 2, 4))
    args.add_argument("--blocks_up", type=tuple, default=(1, 1, 1))
    args.add_argument("--epochs", type=int, default=100)
    args.add_argument(
        "--experiment_name", type=str, default="default_seg_net_experiment"
    )
    # args.add_argument("--using_multi_gpu", type=bool, default=False)
    # args.add_argument("--is_testing", type=bool, default=False)

    args = args.parse_args()
    model_params = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "dropout_prob": args.dropout_prob,
        "init_filters": args.init_filters,
        "blocks_down": args.blocks_down,
        "blocks_up": args.blocks_up,
        "epochs": args.epochs,
        "experiment_name": args.experiment_name,
    }

    train_model(**model_params)
