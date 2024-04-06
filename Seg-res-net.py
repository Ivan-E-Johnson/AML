import itk
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning
import torch
from monai.config import print_config
from monai.data import (
    CacheDataset,
    list_data_collate,
)
from monai.data import DataLoader
from monai.losses import DiceCELoss
from monai.metrics import (
    DiceMetric,
    HausdorffDistanceMetric,
    SurfaceDistanceMetric,
)
from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose,
    EnsureChannelFirstD,
    RandFlipd,
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
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from support_functions import init_t2w_only_data_lists, BestModelCheckpoint

print_config()
from pathlib import Path
import argparse

# weights obtained from running weight_labels() in itk_preprocessing.py
label_weights = {
    "Background": 94.66189125389738,
    "Peripheral Zone": 1.6434384300595233,
    "Transition Zone": 3.4090448842297345,
    "Distal Prostatic Urethra": 0.26201520647321425,
    "Fibromuscular Stroma": 0.023610225340136057,
}
from dotenv import load_dotenv
import os

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
        self.using_multi_gpu = using_multi_gpu

        self._model = SegResNet(
            spatial_dims=3,
            init_filters=init_filters,
            in_channels=1,
            out_channels=self.number_of_classes,
            dropout_prob=dropout_prob,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            upsample_mode=UpsampleMode.DECONV,
        )

        self.loss_function = DiceCELoss(
            softmax=True,
            to_onehot_y=True,
            squared_pred=True,
        )
        self.dice_metric = DiceMetric(
            include_background=False,
            reduction="mean",
            get_not_nans=False,
            num_classes=None,  # Infered from the data
        )
        self.hausdorff_metric = HausdorffDistanceMetric(
            include_background=False,
            reduction="mean",
            get_not_nans=False,
        )
        self.surface_distance_metric = SurfaceDistanceMetric(
            include_background=False,
            reduction="mean",
            get_not_nans=False,
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
                LoadImaged(keys=["image", "label"], reader="ITKReader"),
                EnsureChannelFirstD(
                    keys=["image", "label"]
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
            ]
        )
        self.validation_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstD(keys=["image", "label"]),
                ToTensord(keys=["image", "label"]),
            ]
        )

        number_dataset_workers = 8
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
        # self.train_ds = SmartCacheDataset(
        #     data=train_files,
        #     transform=self.train_transforms,
        #     cache_num=10,
        #     replace_rate=0.2,
        #     num_init_workers=number_dataset_workers,
        #     num_replace_workers=number_dataset_workers // 2,
        # )
        #
        # self.val_ds = SmartCacheDataset(
        #     data=val_files,
        #     transform=self.validation_transforms,
        #     cache_num=10,
        #     replace_rate=0.2,
        #     num_init_workers=number_dataset_workers,
        #     num_replace_workers=number_dataset_workers // 2,
        # )

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
        images, labels = batch["image"], batch["label"]
        output = self.forward(images)
        if self.is_testing:
            print("Training Step")
            print(f"Images.Shape = {images.shape}")
            print(f"Labels.Shape = {labels.shape}")
            print(f"Output shape {output.shape}")
        loss = self.loss_function(output, labels)
        # Collapsing the output to a single channel
        predictions = torch.argmax(output, dim=1, keepdim=True)
        dice = self.dice_metric(y_pred=predictions, y=labels).mean()
        # Log metrics
        self.log(
            "train_dice",
            dice,
            on_step=False,
            on_epoch=True,
            reduce_fx=torch.mean,
            sync_dist=self.using_multi_gpu,
            batch_size=self.batch_size,
        )
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            reduce_fx=torch.mean,
            sync_dist=self.using_multi_gpu,
            batch_size=self.batch_size,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]

        torch.no_grad()
        outputs = self.forward(images)
        if self.is_testing:
            print("Validation Step")
            print(f"Images.Shape = {images.shape}")
            print(f"Labels.Shape = {labels.shape}")
            print(f"Shape after post_pred: {outputs.shape}")

        loss = self.loss_function(outputs, labels)

        predictions = torch.argmax(outputs, dim=1, keepdim=True)

        dice = self.dice_metric(y_pred=predictions, y=labels).mean()
        haussdorf = self.hausdorff_metric(y_pred=predictions, y=labels).mean()
        surface_distance = self.surface_distance_metric(
            y_pred=predictions, y=labels
        ).mean()

        if self.is_testing:
            print(f"Predictions shape: {predictions.shape}")
            print(f"Dice: {dice}")
            print(f"Haussdorf: {haussdorf}")
            print(f"Surface Distance: {surface_distance}")

        self.log(
            name="haussdorf_distance",
            value=haussdorf,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
            sync_dist=self.using_multi_gpu,
        )
        self.log(
            name="surface_distance",
            value=surface_distance,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
            sync_dist=self.using_multi_gpu,
        )
        # Log metrics
        self.log(
            "val_dice",
            dice,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
            sync_dist=self.using_multi_gpu,
        )
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
            sync_dist=self.using_multi_gpu,
        )
        return loss

    def on_validation_epoch_end(self):
        # Get the first batch of the validation data
        val_loader = self.val_dataloader()
        images, labels = (
            next(iter(val_loader))["image"],
            next(iter(val_loader))["label"],
        )

        # Move images and labels to device
        images = images.to(self.device)
        labels = labels.to(self.device)

        torch.no_grad()

        # Run the inference
        outputs = self.forward(images)
        predictions = torch.argmax(outputs, dim=1, keepdim=True)

        # Convert the tensors to numpy arrays for saving with ITK
        images_np = images.cpu().numpy()

        labels_np = labels.cpu().numpy()

        predictions_np = predictions.cpu().numpy()

        images_np = np.squeeze(images_np, axis=1)
        predictions_np = np.squeeze(predictions_np, axis=1)
        labels_np = np.squeeze(labels_np, axis=1)
        for i in range(min(3, images_np.shape[0])):
            single_image = images_np[i, :, :, :]
            single_label = labels_np[i, :, :, :]
            single_prediction = predictions_np[i, :, :, :]
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 3, 1)
            middle_slice = single_image.shape[-1] // 2
            plt.imshow(single_image[:, :, middle_slice], cmap="gray")
            plt.title("Image")
            plt.subplot(1, 3, 2)
            plt.imshow(single_prediction[:, :, middle_slice])
            plt.title("Prediction")
            plt.subplot(1, 3, 3)
            plt.imshow(single_label[:, :, middle_slice])
            plt.title("Label")
            # TODO Add legend so differentiate the classes

            self.logger.experiment.add_figure(
                f"image_prediction_{i}", plt.gcf(), global_step=self.current_epoch
            )
            # TODO See if this works
            # self.logger.experiment.add_figure(f"image_{i}", plt.imshow(single_image[:, :, middle_slice], cmap="gray"), global_step=self.current_epoch)
            fig = plt.figure(figsize=(10, 10))
            plt.imshow(single_prediction[:, :, middle_slice])
            self.logger.experiment.add_figure(
                f"prediction_{i}", fig, global_step=self.current_epoch
            )

    def run_example_inference(self):
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
        predictions = torch.argmax(outputs, dim=1, keepdim=True)

        # Calculate the dice score
        dice = self.dice_metric(y_pred=predictions, y=labels)
        print(f"Dice: {dice}")

        # Convert the tensors to numpy arrays for saving with ITK
        images_np = images.cpu().numpy()
        labels_np = labels.cpu().numpy()
        predictions_np = predictions.cpu().numpy()
        print(f"Images_np shape: {images_np.shape}")
        print(f"Labels_np shape: {labels_np.shape}")
        print(f"Predictions_np shape: {predictions_np.shape}")

        images_np = np.squeeze(images_np, axis=1)
        labels_np = np.squeeze(labels_np, axis=1)
        predictions_np = np.squeeze(predictions_np, axis=1)
        print(f"Images_np shape: {images_np.shape}")
        print(f"Labels_np shape: {labels_np.shape}")
        print(f"Predictions_np shape: {predictions_np.shape}")
        for i in range(np.shape(images_np)[0]):
            single_image = images_np[i, :, :, :]
            single_label = labels_np[i, :, :, :]
            single_prediction = predictions_np[i, :, :, :]
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 3, 1)
            middle_slice = single_image.shape[-1] // 2
            plt.imshow(single_image[:, :, middle_slice], cmap="gray")
            plt.title("Image")
            plt.subplot(1, 3, 2)
            plt.imshow(single_label[:, :, middle_slice], cmap="gray")
            plt.title("Label")
            plt.subplot(1, 3, 3)
            plt.imshow(single_prediction[:, :, middle_slice])

            plt.savefig(f"image_prediction_{i}.png")

            itk.imwrite(itk.GetImageFromArray(single_image), f"image_{i}.nii.gz")
            itk.imwrite(itk.GetImageFromArray(single_label), f"label_{i}.nii.gz")

            itk.imwrite(
                itk.GetImageFromArray(single_prediction.astype(np.uint8)),
                f"prediction_{i}.nii.gz",
            )

        return images, labels, predictions


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
    print("*" * 80)
    print(f"Cuda available: {torch.cuda.is_available()}")
    print("*" * 80)
    log_every_n_steps: int = int(
        (98 * 0.8) // batch_size
    )  # Want to ensure that we log every epoch
    accelerator = os.environ.get("ACCELERATOR")
    gpu_id = os.environ.get("GPU_ID")
    gpus = [int(gpu_id)]
    # if len(gpus) > 1
    #     # Not implemented yet
    #     using_multi_gpu = True
    # set up loggers and checkpoints
    # initialise the LightningModule
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
    os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
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
        devices=gpus,
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
