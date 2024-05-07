import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
from monai.data import DataLoader, CacheDataset
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from mae_vit_model import MAEViTAutoEnc
from monai.networks.nets import ViTAutoEnc
import torch.nn as nn
from monai.losses import ContrastiveLoss, DiceCELoss, TverskyLoss
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
    RandFlipd,
    RandRotated,
    RandGaussianNoiseD,
    RandHistogramShiftD,
)
from dotenv import load_dotenv
import os

import os
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

load_dotenv()


def _create_image_dict(
    base_data_path: Path, is_testing: bool = False
) -> tuple[list, list]:
    image_data_dicts = []
    label_data_dicts = []

    for dir in base_data_path.iterdir():
        if dir.is_dir():
            image_data_dicts.append({"image": dir / f"{dir.name}_pp_t2w.nii.gz"})
            label_data_dicts.append(
                {"label": dir / f"{dir.name}_pp_segmentation.nii.gz"}
            )
    if is_testing:
        image_data_dicts = image_data_dicts[:10]
        label_data_dicts = label_data_dicts[:10]
    return image_data_dicts, label_data_dicts


class VitSupervisedAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        img_size,
        patch_size,
        in_channels,
        hidden_size,
        mlp_dim,
        proj_type,
        num_layers,
        decov_chns,
        num_heads,
        testing,
        lr=1e-4,
    ):
        super().__init__()
        self.testing = testing
        self.batch_size = 4
        self.number_workers = 4
        self.cache_rate = 0.8
        self.hidden_size = hidden_size
        self.mlp_dim = mlp_dim
        self.proj_type = proj_type
        self.in_channels = in_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.decov_chns = decov_chns
        self.num_heads = num_heads
        self.save_hyperparameters(
            "img_size",
            "patch_size",
            "in_channels",
            "hidden_size",
            "mlp_dim",
            "proj_type",
            "num_layers",
            "decov_chns",
            "num_heads",
            "lr",
        )

        self.model = MAEViTAutoEnc(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            proj_type=proj_type,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            deconv_chns=decov_chns,
            num_layers=num_layers,
            num_heads=num_heads,
            training_unsupervised=True,
            out_channels=1,  ## Adjust as needed
        )

        self.recon_loss = nn.L1Loss()
        self.contrastive_loss = ContrastiveLoss(temperature=0.05)
        self.lr = lr

        base_data_path = Path(os.getenv("PP_WITH_SEGMENTATION_PATH"))
        image_data_dict, label_data_dict = _create_image_dict(
            base_data_path, is_testing=self.testing
        )
        data_dicts = [
            dict(**image, **label)
            for image, label in zip(image_data_dict, label_data_dict)
        ]

        train_image_paths, test_image_paths = train_test_split(
            data_dicts, test_size=0.2, random_state=42
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
            transform=self._get_val_transforms(),
            cache_rate=self.cache_rate,
            num_workers=self.number_workers,
            runtime_cache=True,
        )
        # TODO Check this loss function 1 channel output for now
        self.dice_loss_function = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
            jaccard=False,
            reduction="mean",
        )
        self.tv_loss = TverskyLoss(
            softmax=True,
            alpha=0.3,  # weight of false positive
            beta=0.7,  # weight of false negative
        )
        self.dice_metric = DiceMetric()
        self.hausdorff_metric = HausdorffDistanceMetric()

    def _get_train_transforms(self):
        RandFlipd_prob = 0.5
        return Compose(
            [
                # ModalityStackTransformd(keys=["image"]),
                LoadImageD(keys=["image", "label"], image_only=False),
                # DataStatsD(keys=["image"]),
                EnsureChannelFirstd(keys=["image", "label"]),
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
                RandGaussianNoiseD(keys=["image", "label"], prob=RandFlipd_prob / 2),
                RandHistogramShiftD(keys=["image", "label"], prob=RandFlipd_prob / 2),
                ToTensord(keys=["image", "label"]),
                # SaveImageD(keys=["contrastive_patched"], folder_layout=layout),
            ]
        )

    def _get_val_transforms(self):
        return Compose(
            [
                LoadImageD(keys=["image", "label"], image_only=False),
                EnsureChannelFirstd(keys=["image", "label"]),
                ToTensord(keys=["image", "label"]),
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
        inputs, label = (
            batch["image"],
            batch["label"],
        )
        outputs_v1, _hidden_states = self.forward(inputs)
        # TODO RUN PCA ON HIDDEN STATES ect
        diceCE_loss = self.dice_loss_function(outputs_v1, label)
        tv_loss = self.tv_loss(outputs_v1, label)
        hausdorff_metric = self.hausdorff_metric(outputs_v1, label)
        dice_metric = self.dice_metric(outputs_v1, label)

        loss = diceCE_loss + 0.5 * tv_loss
        self.log("train_combined_loss", loss)
        self.log("train_dice_loss", diceCE_loss)
        self.log("train_tv_loss", tv_loss)
        self.log("train_hausdorff_distance", hausdorff_metric.mean())  # This
        self.log("train_dice_metric", dice_metric.mean())
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, label = (
            batch["image"],
            batch["label"],
        )
        outputs_v1, _hidden_states = self.forward(inputs)
        # TODO RUN PCA ON HIDDEN STATES ect
        diceCE_loss = self.dice_loss_function(outputs_v1, label)
        tv_loss = self.tv_loss(outputs_v1, label)
        hausdorff_metric = self.hausdorff_metric(outputs_v1, label)
        dice_metric = self.dice_metric(outputs_v1, label)

        loss = diceCE_loss + 0.5 * tv_loss
        self.log("val_combined_loss", loss)
        self.log("val_dice_loss", diceCE_loss)
        self.log("val_tv_loss", tv_loss)
        self.log("val_hausdorff_distance", hausdorff_metric.mean())
        self.log("val_dice_metric", dice_metric.mean())

        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        return optimizer


def train_model(
    learning_rate: float,
    batch_size: int,
    epochs: int,
    experiment_name: str,
    in_channels: int,
    hidden_size: int,
    mlp_dim: int,
    proj_type: str,
    num_layers: int,
    decov_chns: int,
    num_heads: int,
    patch_size: list,
    testing: bool = False,
):
    # Environment configuration for CUDA
    os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    accelerator = os.environ.get("ACCELERATOR", "gpu")
    gpu_id = os.environ.get("GPU_ID", "0")

    # Configure devices based on the accelerator type
    devices = [int(gpu_id)] if accelerator != "cpu" else "auto"
    using_multi_gpu = len(devices) > 1

    print("*" * 80)
    print(f"Cuda available: {torch.cuda.is_available()}")
    print(f"Using Multi GPU: {using_multi_gpu}")
    print(f"Devices: {devices}")
    print(f"Accelerator: {accelerator}")
    print("*" * 80)

    log_every_n_steps = int((98 * 0.8) // batch_size)

    # Ensure the model architecture matches the one used when the checkpoint was saved
    net = VitSupervisedAutoEncoder(
        img_size=(320, 320, 32),
        patch_size=patch_size,
        in_channels=in_channels,
        hidden_size=hidden_size,  # Ensure this matches the hidden_size used when the checkpoint was saved
        mlp_dim=mlp_dim,
        proj_type=proj_type,
        num_layers=num_layers,  # Ensure this matches the num_layers used when the checkpoint was saved
        decov_chns=decov_chns,  # Ensure this matches the decov_chns used when the checkpoint was saved
        num_heads=num_heads,  # Ensure this matches the num_heads used when the checkpoint was saved
        lr=learning_rate,
        testing=testing,
    )

    # Then load the state_dict from the checkpoint
    checkpoint = torch.load(
        "/Users/iejohnson/School/spring_2024/AML/Supervised_learning/AML_Project_Supervised/Unsupervised/Results/UnsupervisedEncoderLogs/Unsupervised_16layer_perceptron_double_hiddensize_mlp_5000e-checkpoint-epoch=4156-val_loss=0.39.ckpt",
        map_location="cpu",
    )
    net.load_state_dict(checkpoint["state_dict"], strict=False)
    # Logging and checkpointing setup
    current_file_loc = Path(__file__).parent
    log_dir = current_file_loc / "SupervisedEncoderLogs"
    log_dir.mkdir(exist_ok=True, parents=True)
    tb_logger = TensorBoardLogger(save_dir=log_dir.as_posix(), name=experiment_name)

    loss_checkpoint_fn = experiment_name + "-checkpoint-{epoch:02d}-{val_loss:.2f}"
    dice_checkpoint_fn = (
        experiment_name + "-checkpoint-{epoch:02d}-{val_dice_metric:.2f}"
    )
    top_k = 3
    checkpoint_callback = ModelCheckpoint(
        monitor="val_combined_loss",
        mode="min",
        dirpath=log_dir.as_posix(),
        filename=loss_checkpoint_fn,
        save_top_k=top_k,
    )
    best_model_checkpoint = ModelCheckpoint(
        monitor="val_dice_metric",
        mode="max",
        dirpath=log_dir.as_posix(),
        filename=dice_checkpoint_fn,
        save_top_k=top_k,
    )

    # Configure the trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=tb_logger,
        accelerator=accelerator,
        devices=devices,
        enable_checkpointing=True,
        num_sanity_val_steps=1,
        log_every_n_steps=log_every_n_steps,
        callbacks=[checkpoint_callback],
    )

    # Start training
    trainer.fit(net)


def do_main():
    parser = argparse.ArgumentParser(
        description="Train the AutoEncoder with Vision Transformer."
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs="+",
        default=[16, 16, 16],
        help="Dimensions of each image patch.",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=1536,
        help="Hidden size for the transformer model.",
    )
    parser.add_argument(
        "--mlp_dim",
        type=int,
        default=6144,
        help="MLP dimension for the transformer model.",
    )
    parser.add_argument(
        "--proj_type",
        type=str,
        default="conv",
        help="Projection type for the transformer model.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=24,
        help="Number of layers in the transformer model.",
    )
    #### NOTE DECONV(DECODER) ARE WHAT MAKES THIS SUPERVISED ####

    parser.add_argument(
        "--decov_chns",
        type=int,
        default=16,
        help="Number of channels for the deconvolution layer.",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=12,
        help="Number of heads for the transformer model.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="UnsupervisedEncoder",
        help="Name of the experiment to be logged.",
    )

    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training."
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=1,
        help="Number of input channels in the image.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1000,
        help="Maximum number of epochs to train.",
    )
    parser.add_argument(
        "--testing", type=bool, default=False, help="Testing the model."
    )

    args = parser.parse_args()
    train_model(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.max_epochs,
        experiment_name=args.experiment_name,
        in_channels=args.in_channels,
        hidden_size=args.hidden_size,
        mlp_dim=args.mlp_dim,
        proj_type=args.proj_type,
        num_layers=args.num_layers,
        decov_chns=args.decov_chns,
        num_heads=args.num_heads,
        patch_size=args.patch_size,
        testing=args.testing,
    )


if __name__ == "__main__":
    do_main()
