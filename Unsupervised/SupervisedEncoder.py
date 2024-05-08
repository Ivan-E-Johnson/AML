import argparse
import math
import os
from PIL import Image
import io
from pathlib import Path
from CustomModels import CustomDecoder, SplitAutoEncoder, CustomDecoder3D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import monai
import numpy as np
import pytorch_lightning as pl
from monai.data import DataLoader, CacheDataset
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from sklearn.manifold import TSNE
from mae_vit_model import MAEViTAutoEnc
from monai.networks.nets import ViTAutoEnc, ViT
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
from torchviz import make_dot
import os
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


from support_functions import (
    convert_logits_to_one_hot,
    convert_AutoEncoder_output_to_labelpred,
)

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
        encoder_path: str,
        decov_chns: int,
        testing=True,
        lr=1e-4,
        out_channels=5,
    ):
        super().__init__()
        self.testing = testing
        self.batch_size = 4
        self.number_workers = 11
        self.cache_rate = 0.8
        checkpoint = torch.load(encoder_path, map_location=lambda storage, loc: storage)
        self.hidden_size = checkpoint["hyper_parameters"]["hidden_size"]
        self.patch_size = checkpoint["hyper_parameters"]["patch_size"]
        self.in_channels = checkpoint["hyper_parameters"]["in_channels"]
        self.image_size = checkpoint["hyper_parameters"]["img_size"]
        self.mlp_dim = checkpoint["hyper_parameters"]["mlp_dim"]
        self.num_heads = checkpoint["hyper_parameters"]["num_heads"]
        self.num_layers = checkpoint["hyper_parameters"]["num_layers"]
        self.proj_type = checkpoint["hyper_parameters"]["proj_type"]

        print(f"hidden_size: {self.hidden_size}")
        self.encoder = ViT(
            in_channels=self.in_channels,
            img_size=self.image_size,
            patch_size=self.patch_size,
            hidden_size=self.hidden_size,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            proj_type=self.proj_type,
            # classification=True,
        )
        self.encoder.load_state_dict(checkpoint["encoder"], strict=False, assign=True)

        self.deconv_chns = decov_chns
        self.save_hyperparameters(
            dict(
                patch_size=self.patch_size,
                hidden_size=self.hidden_size,
                decov_chns=self.deconv_chns,
                lr=lr,
                out_channels=out_channels,
            )
        )
        self.up_kernel_size = [int(math.sqrt(i)) for i in self.patch_size]
        # self.Decoder = CustomDecoder3D(
        #     hidden_size=self.hidden_size,
        #     initial_channels=self.deconv_chns,
        #     final_channels=out_channels,
        #     patch_size=self.patch_size,
        #     img_dim=self.image_size,
        #     num_up_blocks=2,  # Example
        #     kernel_size=3,
        #     upsample_kernel_size=self.up_kernel_size,
        #     norm_name="batch",
        #     conv_block=True,
        #     res_block=True,
        # )
        self.Decoder = CustomDecoder(
            hidden_size=self.hidden_size,
            decov_chns=self.deconv_chns,
            up_kernel_size=self.up_kernel_size,
            out_channels=out_channels,
            classsification=True,
        )

        self.model = SplitAutoEncoder(
            encoder=self.encoder,
            decoder=self.Decoder,
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
        )

        # make_dot(self.model, params=dict(self.model.named_parameters())).render( "SupervisedBackbonedAutoEncoder", format="png")

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
            softmax=True,
            to_onehot_y=True,
            squared_pred=True,
            jaccard=False,
            reduction="mean",
        )
        self.tv_loss = TverskyLoss(
            include_background=False,
            softmax=True,
            to_onehot_y=True,
            alpha=0.5,  # weight of false positive
            beta=0.5,  # weight of false negative
        )
        self.dice_metric = DiceMetric(
            include_background=False,
            reduction="mean_channel",
        )
        self.hausdorff_metric = HausdorffDistanceMetric(
            include_background=False, reduction="mean_channel", percentile=95
        )

    def _get_train_transforms(self):
        RandFlipd_prob = 0.5
        return Compose(
            [
                # ModalityStackTransformd(keys=["image"]),
                LoadImageD(keys=["image", "label"], image_only=False, dtype=np.float32),
                # DataStatsD(keys=["image"]),
                ToTensord(keys=["image", "label"], dtype=torch.float32),
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
                ToTensord(keys=["image", "label"], dtype=torch.float32),
                # SaveImageD(keys=["contrastive_patched"], folder_layout=layout),
            ]
        )

    def _get_val_transforms(self):
        return Compose(
            [
                LoadImageD(keys=["image", "label"], image_only=False, dtype=np.float32),
                EnsureChannelFirstd(keys=["image", "label"]),
                ToTensord(keys=["image", "label"], dtype=torch.float32),
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
        # print(f"label: {label.shape}")
        diceCE_loss = self.dice_loss_function(outputs_v1, label)
        tv_loss = self.tv_loss(outputs_v1, label)

        # TODO Implement Post processing of removing small objects here
        single_channel_preds = convert_AutoEncoder_output_to_labelpred(outputs_v1)
        self.hausdorff_metric(single_channel_preds, label)
        self.dice_metric(single_channel_preds, label)

        loss = diceCE_loss + 0.25 * tv_loss
        self.log("train_combined_loss", loss, batch_size=self.batch_size)
        self.log("train_dice_loss", diceCE_loss, batch_size=self.batch_size)
        self.log("train_tv_loss", tv_loss, batch_size=self.batch_size)
        self.log(
            "train_hausdorff_distance",
            self.hausdorff_metric.aggregate().mean(),
            batch_size=self.batch_size,
            on_epoch=True,
        )
        self.log(
            "train_dice_metric",
            self.dice_metric.aggregate().mean(),
            batch_size=self.batch_size,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, label = (
            batch["image"],
            batch["label"],
        )
        outputs_v1, _hidden_states = self.forward(inputs)
        # TODO RUN PCA ON HIDDEN STATES ect
        # print(f"Outputs values: {np.unique(outputs_v1.cpu().numpy())}")

        # print(f"Output shape: {outputs_v1.shape}")
        # print(f"label: {label.shape}")
        #
        diceCE_loss = self.dice_loss_function(outputs_v1, label)
        tv_loss = self.tv_loss(outputs_v1, label)

        # TODO Implement Post processing of removing small objects here
        single_channel_preds = convert_AutoEncoder_output_to_labelpred(outputs_v1)

        self.hausdorff_metric(single_channel_preds, label)
        self.dice_metric(single_channel_preds, label)

        loss = diceCE_loss + 0.25 * tv_loss
        self.log("val_combined_loss", loss)
        self.log("val_dice_loss", diceCE_loss)
        self.log("val_tv_loss", tv_loss)
        # TODO Track each metric for each class
        self.log(
            "val_hausdorff_distance",
            self.hausdorff_metric.aggregate().mean(),
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.log(
            "val_dice_metric",
            self.dice_metric.aggregate().mean(),
            on_epoch=True,
            batch_size=self.batch_size,
        )

        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        return optimizer

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
        # TODO RUN PCA ON HIDDEN STATES ect
        outputs, hidden_states = self.forward(images)

        single_channel_preds = convert_AutoEncoder_output_to_labelpred(outputs)
        print(f"Outputs values: {np.unique(single_channel_preds.cpu().numpy())}")
        print(f"Output shape: {outputs.shape}")
        print(f"one_hot_label shape: {single_channel_preds.shape}")
        print(f"Images shape: {images.shape}")

        monai.visualize.plot_2d_or_3d_image(
            data=labels,
            tag=f"label",
            step=self.current_epoch,
            writer=self.logger.experiment,
            frame_dim=-1,
        )
        monai.visualize.plot_2d_or_3d_image(
            data=single_channel_preds,
            tag=f"Predictions",
            step=self.current_epoch,
            writer=self.logger.experiment,
            frame_dim=-1,
        )
        # Select a layer's hidden states; assuming using layer index 0 for visualization
        selected_hidden_states = hidden_states[0].cpu().numpy()

        # Flatten the hidden states if necessary
        reshaped_states = selected_hidden_states.reshape(
            -1, selected_hidden_states.shape[-1]
        )

        # Perform PCA
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(reshaped_states)

        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(reshaped_states)

        # Plot PCA and t-SNE results and log to TensorBoard
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        scatter = axs[0].scatter(
            pca_results[:, 0], pca_results[:, 1], alpha=0.6, cmap="viridis"
        )
        axs[0].set_title("PCA Projection of Hidden States")
        axs[0].set_xlabel("Principal Component 1")
        axs[0].set_ylabel("Principal Component 2")
        plt.colorbar(scatter, ax=axs[0], label="Class Label")

        scatter = axs[1].scatter(
            tsne_results[:, 0],
            tsne_results[:, 1],
            alpha=0.6,
            cmap="viridis",
        )
        axs[1].set_title("t-SNE Projection of Hidden States")
        axs[1].set_xlabel("t-SNE Dimension 1")
        axs[1].set_ylabel("t-SNE Dimension 2")
        plt.colorbar(scatter, ax=axs[1], label="Class Label")

        plt.tight_layout()
        self.logger.experiment.add_figure(
            "PCA and t-SNE Projections", fig, global_step=self.current_epoch
        )

        torch.enable_grad()


def train_model(
    learning_rate: float,
    batch_size: int,
    epochs: int,
    experiment_name: str,
    decov_chns: int,
    encoder_path: str,
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
    net = VitSupervisedAutoEncoder(
        encoder_path=encoder_path,
        decov_chns=decov_chns,  # Ensure this matches the decov_chns used when the checkpoint was saved
        lr=learning_rate,
        testing=testing,
    )

    current_file_loc = Path(__file__).parent
    log_dir = current_file_loc / "SupervisedEncoderLogs"
    log_dir.mkdir(exist_ok=True, parents=True)
    tb_logger = TensorBoardLogger(save_dir=log_dir.as_posix(), name=experiment_name)

    loss_checkpoint_fn = (
        experiment_name + "-checkpoint-{epoch:02d}--loss-{val_combined_loss:.2f}"
    )
    dice_checkpoint_fn = (
        experiment_name + "-checkpoint-{epoch:02d}-dice--{val_dice_metric:.2f}"
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
        callbacks=[checkpoint_callback, best_model_checkpoint],
    )

    # Start training
    trainer.fit(net)


def do_main():
    parser = argparse.ArgumentParser(
        description="Train a supervised encoder model using a pretrained unsupervised encoder."
    )
    #### NOTE DECONV(DECODER) ARE WHAT MAKES THIS SUPERVISED ####
    parser.add_argument(
        "--decov_chns",
        type=int,
        default=24,
        help="Number of channels for the deconvolution layer.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="UnsupervisedEncoder",
        help="Name of the experiment to be logged.",
    )
    parser.add_argument(
        "--encoder_path",
        type=str,
        required=True,
        help="Path to the pretrained unsupervised encoder model.",
    )

    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training."
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
        decov_chns=args.decov_chns,
        testing=args.testing,
        encoder_path=args.encoder_path,
    )


if __name__ == "__main__":
    do_main()
