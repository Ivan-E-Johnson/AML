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
    RandGaussianNoiseD,
    RandHistogramShiftD,
    RandRotated,
)
from monai.networks.nets import UNet, SwinUNETR
from monai.networks.layers import Norm
from monai.metrics import (
    DiceMetric,
    ConfusionMatrixMetric,
    HausdorffDistanceMetric,
    SurfaceDistanceMetric,
)
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
import torchmetrics

from pytorch_lightning.cli import ReduceLROnPlateau
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.tuner import Tuner
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torch import device
from torch.utils.tensorboard._utils import make_grid

print_config()
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
import os

# Load environment variables from .env file
load_dotenv()

RUN_NAME = "z32_basic_unet_3_29_24"


def convert_to_bi_name(name: str):
    study_num = name.split("-")[1]
    return f"PRX{study_num}"


BASE_DATA_PATH = Path(
    "/home/iejohnson/programing/Supervised_learning/DATA/preprocessed_data"
)

RUN_NAME = "Baysian_Opt_z16_basic_unet_3_30_24"


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
    base_data_path = Path(os.environ.get("DATA_PATH"))
    print(f"Base data path: {base_data_path}")
    # base_data_path = Path(
    #     "/Users/iejohnson/School/spring_2024/AML/Supervised_learning/DATA/preprocessed_data"
    # )
    mask_paths = []
    image_paths = []
    for dir in base_data_path.iterdir():
        if dir.is_dir():
            image_list = []

            image_list.append(dir / f"{dir.name}_resampled_normalized_t2w.nii.gz")
            mask_paths.append(dir / f"{dir.name}_resampled_segmentations.nii.gz")

            adc_path = list(dir.glob("*ADC.nii_registered.nii.gz"))[0]
            bvalue_path = list(dir.glob("*bVal.nii_registered.nii.gz"))[0]
            image_list.append(adc_path)

            image_list.append(bvalue_path)

            image_paths.append(image_list)
    return image_paths, mask_paths


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


class ModalityStackTransform:
    """
    A custom transformation to stack different modalities into a single multi-channel image.
    Assumes modalities are stored in separate files with a consistent naming scheme.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            # Assuming the key is a list of paths
            # Assuming 'data' is a dictionary with keys 'image' and 'label'
            # And 'image' is a list of paths: [path_to_t1, path_to_t2, path_to_adc]
            images = [itk.imread(path) for path in d[key]]
            arrays = [itk.array_from_image(image) for image in images]
            stacked_image = np.stack(
                arrays, axis=0
            )  # Stack along the channel dimension
            d["image"] = stacked_image
            print(f"Stacked image shape: {stacked_image.shape}")
            print(f"Stacked image dtype: {stacked_image.dtype}")

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


    # TEST OUT
    # Shallow Progression (16, 24, 48, 96, 192)
    # Default Progression (16, 32, 64, 128, 256)
    # Deep Progression (32, 64, 128, 256, 512)

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
        kernel_size: int = 3,
        kernel_upsample: int = 3,
        is_testing=False,

    ):

        super().__init__()
        self.number_of_classes = 5  # INCLUDES BACKGROUND

        self._model = UNet(
            spatial_dims=3,
            in_channels=3,
            out_channels=self.number_of_classes,
            channels=(16, 32, 64, 128, 256),  # Number of features in each layer
            strides=(2, 2, 2, 2),
            num_res_units=number_res_units,
            norm=norm,
            kernel_size=kernel_size,
            up_kernel_size=kernel_upsample,
            dropout=dropout,
            # act="relu",
        )
        self.is_testing = is_testing

        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.loss_function = DiceCELoss(
            softmax=True, to_onehot_y=True, squared_pred=True
        )

        # TODO implement this sensitivity metric
        # self.sensitivity_metric = ConfusionMatrixMetric(metric_name="sensitivity", compute_sample=True,)
        # TODO ADD Other metrics
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
        self.iou_metric = torchmetrics.JaccardIndex(
            task="multiclass",
            threshold=0.5,
            num_classes=self.number_of_classes,
            average="macro",
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
                ModalityStackTransform(keys=["image"]),
                LoadImaged(keys=["label"]),
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
                ModalityStackTransform(keys=["image"]),
                LoadImaged(keys=["label"]),
                EnsureChannelFirstD(
                    keys=["image", "label"], channel_dim=0
                ),  # Add channel to image and mask so
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
        torch.set_grad_enabled(True)
        images, labels = batch["image"], batch["label"]
        output = self.forward(images)
        if self.is_testing:
            print("Training Step")
            print(f"Images.Shape = {images.shape}")
            print(f"Labels.Shape = {labels.shape}")
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
            on_step=False,
            on_epoch=True,
            reduce_fx=torch.mean,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            reduce_fx=torch.mean,
            sync_dist=True,
            batch_size=self.batch_size,
        )

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
        # Run these metrics before the squashing of the output

        # Calculate accuracy, precision, recall, and F1 score
        predictions = torch.argmax(
            outputs, dim=1, keepdim=True
        )  # Assuming output is logits

        dice = self.dice_metric(y_pred=predictions, y=labels).mean()
        haussdorf = self.hausdorff_metric(y_pred=predictions, y=labels).mean()
        surface_distance = self.surface_distance_metric(
            y_pred=predictions, y=labels
        ).mean()

        iou = self.iou_metric(predictions, labels)


        if self.is_testing:
            print(f"Predictions shape: {predictions.shape}")
            print(f"Dice: {dice}")
            print(f"Haussdorf: {haussdorf}")
            print(f"Surface Distance: {surface_distance}")


        if self.best_val_dice < dice:
            self.best_val_dice = dice
            self.best_val_epoch = self.current_epoch
            
        self.log(
            name="iou",
            value=iou,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
            sync_dist=True,
        )
        self.log(
            name="haussdorf_distance",
            value=haussdorf,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
            sync_dist=True,
        )
        self.log(
            name="surface_distance",
            value=surface_distance,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
            sync_dist=True,
        )
        # Log metrics
        self.log(
            "val_dice",
            dice,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
            sync_dist=True,
        )
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
            sync_dist=True,
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
        single_image = images_np[0, :, :, :]
        single_label = labels_np[0, :, :, :]
        single_prediction = predictions_np[0, :, :, :]
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
            f"image_prediction", plt.gcf(), global_step=self.current_epoch
        )
        # TODO See if this works
        # self.logger.experiment.add_figure(f"image_{i}", plt.imshow(single_image[:, :, middle_slice], cmap="gray"), global_step=self.current_epoch)
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(single_prediction[:, :, middle_slice])
        self.logger.experiment.add_figure(
            f"prediction", fig, global_step=self.current_epoch
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
                        checkpoint_callback.dirpath, f"{RUN_NAME}_best_model.pth"
                    )
                    trainer.save_checkpoint(checkpoint_callback.best_model_path)



def do_main(
    learning_rate,
    batch_size,
    dropout,
    number_res_units,
    kernel_size,
    kernel_upsample,
    epocs=200,
    log_name=RUN_NAME,
):
    checkpoint_fn = RUN_NAME + "checkpoint-{epoch:02d}-{val_dice:.2f}"
    current_file_loc = Path(__file__).parent
    log_dir = current_file_loc / "logs"
    tb_logger = pytorch_lightning.loggers.TensorBoardLogger(
        save_dir=log_dir.as_posix(), name=log_name
    )
    kernel_size = int(kernel_size)
    kernel_upsample = int(kernel_upsample)
    number_res_units = int(number_res_units)
    batch_size = int(batch_size)
    # Ensure that the values are valid for the model
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_upsample % 2 == 0:
        kernel_upsample += 1

    net = Net(
        learning_rate=learning_rate,
        batch_size=batch_size,
        dropout=dropout,
        norm=Norm.BATCH,
        number_res_units=number_res_units,
        is_testing=False,
        kernel_size=kernel_size,
        kernel_upsample=kernel_upsample,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_dice",
        mode="max",
        save_last=True,
        dirpath=log_dir.as_posix(),
        filename=checkpoint_fn,
    )
    trainer = pytorch_lightning.Trainer(
        max_epochs=epocs,

        logger=tb_logger,
        accelerator="cpu",
        # devices=[0],
        enable_checkpointing=True,
        num_sanity_val_steps=1,
        log_every_n_steps=5,
        # profiler=profiler,
        callbacks=[
            BestModelCheckpoint(),
            checkpoint_callback,
        ],  # Add the custom callback
    )

    trainer.fit(net)
    return net.best_val_dice


def do_baysian_optimization(
    number_of_iterations: int = 10, number_of_points_to_probe: int = 10
):
    from bayes_opt import BayesianOptimization
    from bayes_opt.event import Events
    from bayes_opt.logger import JSONLogger
    from bayes_opt.util import load_logs

    min_learning_rate = 1e-6
    max_learning_rate = 1e-2
    min_batch_size = 1
    max_batch_size = 30
    min_dropout = 0.1
    max_dropout = 0.5
    min_number_res_units = 0
    max_number_res_units = 5
    min_kernel_size = 1
    max_kernel_size = 5
    min_kernel_upsample = 1
    max_kernel_upsample = 5

    optimizer_bounds = {
        "learning_rate": (min_learning_rate, max_learning_rate),
        "batch_size": (min_batch_size, max_batch_size),
        "dropout": (min_dropout, max_dropout),
        "number_res_units": (min_number_res_units, max_number_res_units),
        "kernel_size": (min_kernel_size, max_kernel_size),
        "kernel_upsample": (min_kernel_upsample, max_kernel_upsample),
    }
    optimizer_best_position = {
        "learning_rate": 0.001,
        "batch_size": 5,
        "dropout": 0.1,
        "number_res_units": 3,
        "kernel_size": 3,
        "kernel_upsample": 3,
    }

    current_iteration = 0

    optimizer = BayesianOptimization(
        f=do_main,
        pbounds=optimizer_bounds,
        random_state=42,
        verbose=2,
        allow_duplicate_points=True,
    )
    all_optimizer_logs: list[str] = list()
    if len(all_optimizer_logs) > 0:
        print(
            f"Initializing optimizer with previous computations: {all_optimizer_logs}"
        )
        load_logs(optimizer, logs=all_optimizer_logs)
    else:
        optimizer.probe(
            params=optimizer_best_position,
            lazy=True,
        )
        # probe min and max
        optimizer.probe(
            params={
                "learning_rate": min_learning_rate,
                "batch_size": min_batch_size,
                "dropout": min_dropout,
                "number_res_units": min_number_res_units,
                "kernel_size": min_kernel_size,
                "kernel_upsample": min_kernel_upsample,
            },
            lazy=True,
        )
        optimizer.probe(
            params={
                "learning_rate": max_learning_rate,
                "batch_size": max_batch_size,
                "dropout": max_dropout,
                "number_res_units": max_number_res_units,
                "kernel_size": max_kernel_size,
                "kernel_upsample": max_kernel_upsample,
            },
            lazy=True,
        )
        optimization_logs_folder_path: Path = (
            Path(__file__).parent / "optimization_logs"
        )
        optimization_logs_folder_path.mkdir(parents=True, exist_ok=True)
        optimizer_logs_fn: Path = (
            optimization_logs_folder_path
            / f"logs{str(RUN_NAME)}_iter_{current_iteration}.json"
        )
        logger = JSONLogger(path=optimizer_logs_fn.as_posix())
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
        all_optimizer_logs.append(optimizer_logs_fn.as_posix())
        optimizer.maximize(init_points=10, n_iter=10)


if __name__ == "__main__":
    print("*" * 80)
    print(f"Device: {device}")
    print(f"Cuda available: {torch.cuda.is_available()}")
    print("*" * 80)
    do_baysian_optimization(number_of_iterations=100, number_of_points_to_probe=100)

    print("Finished training")
