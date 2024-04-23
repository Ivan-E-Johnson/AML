import argparse
import os

import itk
import matplotlib.pyplot as plt
import monai
import numpy as np
import pytorch_lightning
import torch
import torchmetrics.clustering
from skimage import measure

from monai.config import print_config
from monai.data import CacheDataset, list_data_collate
from monai.data import DataLoader
from monai.losses import DiceCELoss, TverskyLoss, FocalLoss, DiceLoss, DiceFocalLoss
from monai.metrics import (
    DiceMetric,
    HausdorffDistanceMetric,

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
    KeepLargestConnectedComponent,
    FillHoles,
    Spacingd,
    ResizeWithPadOrCropd,
)
from monai.transforms import (
    LoadImaged,
    ToTensord,
    ToTensor,
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
    convert_logits_to_one_hot, init_t2w_only_data_lists,
)

from pathlib import Path

# weights obtained from running weight_labels() in itk_preprocessing.py

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

RUN_NAME = "CoarseProstate"

# TODO REVERT BACK TO USING THE RAW DATA WITH BASIC NORMALIZATION
def init_single_channel_raw_data():
    raw_dir = Path(os.getenv("RAW_WITH_SEGMENTATION_PATH"))
    subject_dirs = [
        subject_dir
        for subject_dir in raw_dir.iterdir()
        if subject_dir.is_dir() and "ProstateX" in subject_dir.name
    ]
    image_paths = []
    mask_paths = []
    for subject_dir in subject_dirs:
        mask_paths.append(subject_dir / f"{subject_dir.name}_Segmentation.nii.gz")
        image_paths.append(subject_dir / f"{subject_dir.name}_T2w.nii.gz")
        assert mask_paths[-1].exists(), f"Mask path {mask_paths[-1]} does not exist"
        assert image_paths[-1].exists(), f"Image path {image_paths[-1]} does not exist"
    return image_paths, mask_paths


class CoarseSegNet(pytorch_lightning.LightningModule):
    """
    This class defines the network architecture.

    Attributes:
    _model (CoarseSegNet): The network architecture.
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
        self.number_of_classes = 1  # INCLUDES BACKGROUND
        self.is_testing = is_testing
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.multi_gpu = using_multi_gpu

        self._model = SegResNet(
            spatial_dims=3,
            init_filters=init_filters,
            in_channels=1,
            out_channels=self.number_of_classes,
            dropout_prob=dropout_prob,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            # upsample_mode=UpsampleMode.DECONV,
        )
        # TODO Add the weights to the loss function and allow multichannel output

        self.dice_loss_function = DiceFocalLoss(
            lambda_focal=0.5,
            sigmoid=True,
            smooth_nr=1e-5,
            smooth_dr=1e-5,
        )
        self.dice_metric = DiceMetric(
            reduction="mean",
        )
        self.hausdorff_metric = HausdorffDistanceMetric(
            reduction="mean",
        )
        # self.rand_score = torchmetrics.clustering.RandScore()

        self.threshold_prob = 0.5 # Threshold for converting voxel_probability to binary
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.prepare_data()
        self.save_hyperparameters()

        # self.PostPred = Compose(
        #     [
        #         AsDiscrete(),
        #         KeepLargestConnectedComponent(),
        #         FillHoles(),
        #         ToTensor(),
        #     ]
        # )

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

        image_paths, mask_paths = init_t2w_only_data_lists()
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
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstD(
                    keys=["image", "label"], channel_dim="no_channel"
                ),  # Add channel to image and mask so
                # Spacingd(
                #     keys=["image", "label"],
                #     pixdim=(0.5, 0.5, 3.0),
                #     mode=("bilinear", "nearest"),
                # ),
                # ResizeWithPadOrCropd(
                #     keys=["image", "label"], spatial_size=(384, 384, 24)
                # ),
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
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstD(
                    keys=["image", "label"], channel_dim="no_channel"
                ),  # Add channel to image and mask so
                # Spacingd(
                #     keys=["image", "label"],
                #     pixdim=(0.5, 0.5, 3.0),
                #     mode=("bilinear", "nearest"),
                # ),
                # ResizeWithPadOrCropd(
                #     keys=["image", "label"], spatial_size=(384, 364, 24)
                # ),
                ToTensord(keys=["image", "label"]),
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
        # Following the hyperparamters as described in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9511435/
        # Initialize the optimizer with the initial learning rate
        torch.set_grad_enabled(True)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Step decay scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)

        # Return both optimizer and scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",  # If your validation strategy involves monitoring loss
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
        # with torch.set_grad_enabled(True):
        #     outputs = self.PostPred(outputs)

        if self.is_testing:
            print("Training Step")
            print(f"Images.Shape = {images.shape}")
            print(f"Labels.Shape = {labels.shape}")
            print(f"outputs shape {outputs.shape}")
        loss = self.dice_loss_function(outputs, labels)
        # onehot_predictions = convert_logits_to_one_hot(outputs)
        # dice = self.dice_metric(y_pred=onehot_predictions, y=labels).mean()
        outputs = self.convert_logits_to_binary(outputs)
        dice = self.dice_metric(y_pred=outputs, y=labels).mean()
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
        # self.log(
        #     "train_rand_score",
        #     self.rand_score(preds=outputs, target=labels).mean(),
        #     on_step=False,
        #     on_epoch=True,
        #     reduce_fx=torch.mean,
        #     sync_dist=self.multi_gpu,
        #     batch_size=self.batch_size,
        # )

        return loss

    def validation_step(self, batch, batch_idx):
        torch.no_grad()
        images, labels = batch["image"], batch["label"]
        outputs = self.forward(images)
        # with torch.set_grad_enabled(True):
        #     outputs = self.PostPred(outputs)

        if self.is_testing:
            print("Validation Step")
            print(f"Images.Shape = {images.shape}")
            print(f"Labels.Shape = {labels.shape}")
            print(f"Shape after post_pred: {outputs.shape}")

        loss = self.dice_loss_function(outputs, labels)

        outputs = self.convert_logits_to_binary(outputs)
        dice = self.dice_metric(y_pred=outputs, y=labels).mean()
        haussdorf = self.hausdorff_metric(y_pred=outputs, y=labels).mean()
        # rand_score = self.rand_score(preds=outputs, target=labels).mean()

        if self.is_testing:
            print(f"Predictions shape: {outputs.shape}")
            print(f"Dice: {dice}")
            print(f"Haussdorf: {haussdorf}")
            print(f"Rand Score: {rand_score}")

        self.log(
            name="haussdorf_distance",
            value=haussdorf,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
            sync_dist=self.multi_gpu,
        )

        # self.log(
        #     name="rand_score",
        #     value=rand_score,
        #     on_step=False,
        #     on_epoch=True,
        #     batch_size=self.batch_size,
        #     sync_dist=self.multi_gpu,
        # )
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

    def convert_logits_to_binary(self, logits):
        probs = torch.sigmoid(logits)
        return probs > self.threshold_prob

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
        probs = torch.sigmoid(outputs) # Convert to probabilities

        outputs = probs > self.threshold_prob

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
        monai.visualize.plot_2d_or_3d_image(
            data=probs,
            tag=f"Probabilities",
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
    using_multi_gpu = False
    devices: list[int] | str
    if accelerator == "cpu":
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

    net = CoarseSegNet(
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
    log_dir.mkdir(exist_ok=True, parents=True)
    tb_logger = pytorch_lightning.loggers.TensorBoardLogger(
        save_dir=log_dir.as_posix(), name=experiment_name
    )

    checkpoint_fn = experiment_name + "checkpoint-{epoch:02d}-{val_dice:.2f}"
    # initialise Lightning's trainer.
    checkpoint_callback = ModelCheckpoint(
        monitor="val_dice",
        mode="max",
        save_last=True,
        save_top_k=5,
        enable_version_counter=True,
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
        reload_dataloaders_every_n_epochs=100,
        callbacks=[
            BestModelCheckpoint(monitor="val_dice", mode="max", experiment_name=f"best_val_model_{experiment_name}"),
            BestModelCheckpoint(monitor="haussdorf_distance", mode="min", experiment_name=f"best_haussdorf_model_{experiment_name}"),
            checkpoint_callback,
        ],  # Add the custom callback
    )

    trainer.fit(net)

    print("Finished training")
    return net.best_val_dice.cpu()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--learning_rate", type=float, default=0.0008)
    args.add_argument("--batch_size", type=int, default=10)
    args.add_argument("--dropout_prob", type=float, default=0.2)
    args.add_argument("--init_filters", type=int, default=8)
    args.add_argument("--blocks_down", type=tuple, default=(1, 2, 2, 4))
    args.add_argument("--blocks_up", type=tuple, default=(1, 1, 1))
    args.add_argument("--epochs", type=int, default=100)
    args.add_argument(
        "--experiment_name", type=str, default="default_coarse_prostate_experiment"
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
