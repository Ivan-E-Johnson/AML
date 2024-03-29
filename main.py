import itk
import numpy as np
import pytorch_lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureType,
    DataStats,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, list_data_collate, decollate_batch, DataLoader
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
from dotenv import load_dotenv
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImaged,
    EnsureChannelFirstD,
    SpatialResampleD,
    SpacingD,
    ResizeD,
    LoadImage,
    RandFlipD,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    ToTensor,
    RandFlipd,
    DataStatsD,
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import UNet
from monai.transforms import (
    LoadImaged,
    ScaleIntensityd,
    RandCropByPosNegLabeld,
    ToTensord,
)
from pytorch_lightning.cli import ReduceLROnPlateau
from pytorch_lightning.profilers import SimpleProfiler
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard._utils import make_grid

print_config()
from pathlib import Path


# Function to initialize data lists
def init_data_lists():
    """
    This function initializes the image and mask paths.

    Args:
    base_data_path (Path): The base path of the data.

    Returns:
    list: A list of image paths.
    list: A list of mask paths.
    """
    base_data_path = Path(
        "/Users/iejohnson/School/spring_2024/AML/Supervised_learning/DATA/preprocessed_data"
    )
    mask_paths = []
    image_paths = []
    for dir in base_data_path.iterdir():
        if dir.is_dir():

            image_paths.append(dir / f"{dir.name}_resampled_normalized_t2w.nii.gz")
            mask_paths.append(dir / f"{dir.name}_resampled_segmentations.nii.gz")

    return image_paths, mask_paths


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AddChannelD(object):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            im = d[key]
            im = np.expand_dims(im, axis=0)
            d[key] = im
        return d


class Net(pytorch_lightning.LightningModule):
    """
    This class defines the network architecture.

    Attributes:
    _model (UNet): The UNet model.
    loss_function (DiceLoss): The loss function.
    post_pred (Compose): The post prediction transformations.
    post_label (Compose): The post label transformations.
    dice_metric (DiceMetric): The dice metric.
    best_val_dice (float): The best validation dice score.
    best_val_epoch (int): The epoch with the best validation dice score.
    validation_step_outputs (list): The outputs of the validation step.
    """

    def __init__(self, is_testing=False):
        super().__init__()
        self.number_of_classes = 5  # INCLUDES BACKGROUND
        self._model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=self.number_of_classes,
            channels=(16, 32, 64, 128, 256),  # Number of features in each layer
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )
        self.is_testing = is_testing

        self.loss_function = DiceCELoss(
            softmax=True, to_onehot_y=True, squared_pred=True
        )  # TODO Implement secondary loss functions

        # TODO implement this sensitivity metric
        # self.sensitivity_metric = ConfusionMatrixMetric(metric_name="sensitivity", compute_sample=True,)
        self.post_pred = Compose(
            [
                EnsureType(),  # Ensure tensor type
            ]
        )

        self.post_label = Compose(
            [
                EnsureType(),  # Ensure tensor type
            ]
        )
        # TODO ADD Other metrics
        self.dice_metric = DiceMetric(
            include_background=False,
            reduction="mean",
            get_not_nans=False,
            num_classes=None,  # Infered from the data
        )
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.prepare_data()

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

        image_paths, mask_paths = init_data_lists()
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
                    keys=["image", "label"]
                ),  # Add channel to image and mask so
                # Coarse Segmentation combine all mask
                RandFlipd(keys=["image", "label"], prob=RandFlipd_prob, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=RandFlipd_prob, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=RandFlipd_prob, spatial_axis=2),
                ToTensord(keys=["image", "label"]),
                # DataStatsD(keys=["image", "label"]),
            ]
        )
        self.validation_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstD(
                    keys=["image", "label"]
                ),  # Add channel to image and mask so
                ToTensord(keys=["image", "label"]),
                # DataStatsD(keys=["image", "label"]),
            ]
        )

        # we use cached datasets - these are 10x faster than regular datasets
        self.train_ds = CacheDataset(
            data=train_files,
            transform=self.train_transforms,
            cache_rate=0.5,
            num_workers=4,
        )
        self.val_ds = CacheDataset(
            data=val_files,
            transform=self.validation_transforms,
            cache_rate=0.5,
            num_workers=4,
        )

    def train_dataloader(self):
        """
        This function returns the training data loader.

        Returns:
        DataLoader: The training data loader.
        """
        train_loader = DataLoader(
            self.train_ds,
            batch_size=5,
            shuffle=True,
            num_workers=4,
            collate_fn=list_data_collate,
        )
        return train_loader

    def val_dataloader(self):
        """
        This function returns the validation data loader.

        Returns:
        DataLoader: The validation data loader.
        """
        val_loader = DataLoader(self.val_ds, batch_size=1, num_workers=4)
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
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
        # print("Training Step")
        # print(f"Images.Shape = {images.shape}")
        print(f"Labels.Shape = {labels.shape}")
        output = self.forward(images)
        print(f"Output shape {output.shape}")
        loss = self.loss_function(output, labels)
        # Colapsing the output to a single channel
        predictions = torch.argmax(
            output, dim=1, keepdim=True
        )  # Assuming output is logits
        dice = self.dice_metric(y_pred=predictions, y=labels).mean()
        # Log metrics
        self.log(
            "train_dice",
            dice,
            on_step=True,
            on_epoch=True,
            reduce_fx=torch.mean,
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True, reduce_fx=torch.mean)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]

        outputs = self.forward(images)
        if self.is_testing:
            print("Validation Step")
            print(f"Images.Shape = {images.shape}")
            print(f"Labels.Shape = {labels.shape}")
            print(f"Shape after post_pred: {outputs.shape}")

        loss = self.loss_function(outputs, labels)

        # Calculate accuracy, precision, recall, and F1 score
        predictions = torch.argmax(
            outputs, dim=1, keepdim=True
        )  # Assuming output is logits

        dice = self.dice_metric(y_pred=predictions, y=labels).mean()

        if self.is_testing:
            print(f"Predictions shape: {predictions.shape}")
            print(f"Dice: {dice}")
        targets = labels  # Assuming labels are already one-hot encoded
        precision = precision_score(
            targets.flatten(), predictions.flatten(), average="weighted"
        )
        recall = recall_score(
            targets.flatten(), predictions.flatten(), average="weighted"
        )

        # Log metrics
        self.log(
            "val_dice",
            dice,
            on_step=False,
            on_epoch=True,
            batch_size=images.shape[0],
        )
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, batch_size=images.shape[0]
        )
        self.log(
            "val_precision",
            precision,
            on_step=False,
            on_epoch=True,
            batch_size=images.shape[0],
        )
        self.log(
            "val_recall",
            recall,
            on_step=False,
            on_epoch=True,
            batch_size=images.shape[0],
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
        images = images.to(device)
        labels = labels.to(device)

        # Run the inference
        outputs = self.forward(images)
        predictions = torch.argmax(outputs, dim=1, keepdim=True)

        # Convert the tensors to numpy arrays for saving with ITK
        images_np = images.cpu().numpy()
        predictions_np = predictions.cpu().numpy()

        images_np = np.squeeze(images_np, axis=1)
        predictions_np = np.squeeze(predictions_np, axis=1)

        for i in range(min(5, images_np.shape[0])):
            single_image = images_np[i, :, :, :]
            single_prediction = predictions_np[i, :, :, :]
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 2, 1)
            middle_slice = single_image.shape[-1] // 2
            plt.imshow(single_image[:, :, middle_slice], cmap="gray")
            plt.title("Image")
            plt.subplot(1, 2, 2)
            plt.imshow(single_prediction[:, :, middle_slice], cmap="gray")
            plt.title("Prediction")
            self.logger.experiment.add_figure(
                f"image_prediction_{i}", plt.gcf(), global_step=self.current_epoch
            )
            # TODO See if this works
            # self.logger.experiment.add_figure(f"image_{i}", plt.imshow(single_image[:, :, middle_slice], cmap="gray"), global_step=self.current_epoch)
            # self.logger.experiment.add_figure(f"prediction_{i}", plt.imshow(single_prediction[:, :, middle_slice], cmap="gray"), global_step=self.current_epoch)

    def run_example_inference(self):
        # Get the first batch of the validation data
        val_loader = self.val_dataloader()
        images, labels = (
            next(iter(val_loader))["image"],
            next(iter(val_loader))["label"],
        )

        # Move images and labels to device
        images = images.to(device)
        labels = labels.to(device)

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
            plt.imshow(single_prediction[:, :, middle_slice], cmap="gray")
            plt.savefig(f"image_prediction_{i}.png")

            itk.imwrite(itk.GetImageFromArray(single_image), f"image_{i}.nii.gz")
            itk.imwrite(itk.GetImageFromArray(single_label), f"label_{i}.nii.gz")

            itk.imwrite(
                itk.GetImageFromArray(single_prediction.astype(np.uint8)),
                f"prediction_{i}.nii.gz",
            )

        return images, labels, predictions


class BestModelCheckpoint(pytorch_lightning.callbacks.Callback):
    def __init__(self, monitor="val_dice", mode="max"):
        super().__init__()
        self.monitor = monitor
        self.mode = mode

    def on_validation_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        if logs is not None:
            val_dice = logs.get(self.monitor)
            if val_dice is not None:
                if self.mode == "max" and val_dice >= pl_module.best_val_dice:
                    pl_module.best_val_dice = val_dice
                    pl_module.best_val_epoch = trainer.current_epoch
                    # Save the best model
                    checkpoint_callback = (
                        trainer.checkpoint_callback
                    )  # Access checkpoint callback from trainer
                    checkpoint_callback.best_model_path = os.path.join(
                        checkpoint_callback.dirpath, "best_model.pth"
                    )
                    trainer.save_checkpoint(checkpoint_callback.best_model_path)


if __name__ == "__main__":
    print("*" * 80)
    print(f"Device: {device}")
    print(f"Cuda available: {torch.cuda.is_available()}")
    print("*" * 80)

    # set up loggers and checkpoints
    # initialise the LightningModule
    net = Net(is_testing=True)
    current_file_loc = Path(__file__).parent
    log_dir = current_file_loc / "logs"
    tb_logger = pytorch_lightning.loggers.TensorBoardLogger(
        save_dir=log_dir.as_posix(), name="3_26_24_lightning_logs"
    )
    os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # initialise Lightning's trainer.
    profiler = SimpleProfiler()
    checkpoint_callback = ModelCheckpoint(
        monitor="val_dice",
        mode="max",
        save_last=True,
        dirpath=log_dir.as_posix(),
        filename="checkpoint-{epoch:02d}-{val_dice:.2f}",
    )

    #
    # print(f"Images shape: {images.shape}")
    # print(f"Labels shape: {labels.shape}")
    # print(f"Predictions shape: {predictions.shape}")

    trainer = pytorch_lightning.Trainer(
        max_epochs=1,
        logger=tb_logger,
        # accelerator="gpu",
        enable_checkpointing=True,
        num_sanity_val_steps=1,
        log_every_n_steps=1,
        profiler=profiler,
        callbacks=[
            BestModelCheckpoint(),
            checkpoint_callback,
        ],  # Add the custom callback
    )

    trainer.fit(net)
    net = Net.load_from_checkpoint(checkpoint_callback.best_model_path)
    net.run_example_inference()

    print("Finished training")
