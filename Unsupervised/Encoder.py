import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
from matplotlib import pyplot as plt
from monai.data import DataLoader, CacheDataset
from pytorch_msssim import ssim
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


def _create_image_dict(base_data_path: Path, is_testing: bool = False) -> list:
    data_dicts = []
    for dir in base_data_path.iterdir():
        if dir.is_dir():
            data_dicts.append({"image": dir / f"{dir.name}_pp_t2w.nii.gz"})
    if is_testing:
        data_dicts = data_dicts[:10]
    return data_dicts


class LitAutoEncoder(pl.LightningModule):
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
        )

        self.recon_loss = nn.L1Loss()
        self.contrastive_loss = ContrastiveLoss(temperature=0.05)
        self.lr = lr

        base_data_path = Path(os.getenv("DATA_PATH"))
        data_dicts = _create_image_dict(base_data_path, is_testing=self.testing)
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
            transform=self._get_train_transforms(),
            cache_rate=self.cache_rate,
            num_workers=self.number_workers,
            runtime_cache=True,
        )

    def _get_train_transforms(self):
        RandFlipd_prob = 0.5
        return Compose(
            [
                # ModalityStackTransformd(keys=["image"]),
                LoadImageD(keys=["image"], image_only=False),
                # DataStatsD(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                RandFlipd(keys=["image"], prob=RandFlipd_prob, spatial_axis=0),
                RandFlipd(keys=["image"], prob=RandFlipd_prob, spatial_axis=1),
                RandFlipd(keys=["image"], prob=RandFlipd_prob, spatial_axis=2),
                RandRotated(
                    keys=["image"],
                    range_x=15,
                    range_y=15,
                    range_z=15,
                    prob=RandFlipd_prob,
                ),
                RandGaussianNoiseD(keys=["image"], prob=RandFlipd_prob / 2),
                RandHistogramShiftD(keys=["image"], prob=RandFlipd_prob / 2),
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
        inputs, gt_input = batch["image"], batch["reference_patched"]
        outputs, _ = self.forward(inputs)
        ssim_loss = 1 - ssim(outputs, gt_input, data_range=1,
                             size_average=True)  # Assuming your data is normalized to [0,1]
        loss = ssim_loss
        self.log("train_loss", loss)
        return loss


    def validation_step(self, batch, batch_idx):
        inputs, gt_input = batch["image"], batch["reference_patched"]
        outputs, _ = self.forward(inputs)
        ssim_loss = 1 - ssim(outputs, gt_input, data_range=1, size_average=True)
        self.log("val_loss", ssim_loss)
        return ssim_loss
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

    # Instantiate the model
    net = LitAutoEncoder(
        img_size=(320, 320, 32),
        patch_size=patch_size,
        in_channels=in_channels,
        hidden_size=hidden_size,
        mlp_dim=mlp_dim,
        proj_type=proj_type,
        num_layers=num_layers,
        decov_chns=decov_chns,
        num_heads=num_heads,
        lr=learning_rate,
        testing=testing,
    )

    # Logging and checkpointing setup
    current_file_loc = Path(__file__).parent
    log_dir = current_file_loc / "UnsupervisedEncoderLogs"
    log_dir.mkdir(exist_ok=True, parents=True)
    tb_logger = TensorBoardLogger(save_dir=log_dir.as_posix(), name=experiment_name)

    checkpoint_fn = experiment_name + "-checkpoint-{epoch:02d}-{val_loss:.2f}"
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=log_dir.as_posix(),
        filename=checkpoint_fn,
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
        default=1024,
        help="Hidden size for the transformer model.",
    )
    parser.add_argument(
        "--mlp_dim",
        type=int,
        default=3072,
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
    parser.add_argument(
        "--decov_chns",
        type=int,
        default=16,
        help="Number of channels for the deconvolution layer.",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=16,
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
        "--max_epochs", type=int, default=10, help="Maximum number of epochs to train."
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

import torchvision.utils as vutils
def evaluate_model(checkpoint_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # use the first GPU if available, otherwise use the CPU
    model = LitAutoEncoder.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        testing=True  # Adjust this based on your needs
    )
    model.to(device)  # ensure the model is on the correct device
    val_dataloader = model.val_dataloader()
    model.eval()
    model.freeze()

    for batch in val_dataloader:
        inputs, targets = batch["image"].to(device), batch["reference_patched"].to(device)
        print(f"Input Shape: {inputs.shape}")
        outputs, latent = model(targets)

        print(f"Output Shape: {outputs.shape}")
        visualize_output(targets, outputs)
        break  # Just show one batch for example purposes


def visualize_output(inputs, outputs, num_images=5):

    inputs = inputs.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()

    fig, axs = plt.subplots(nrows=num_images, ncols=2, figsize=(10, num_images * 5))

    for i in range(num_images):
        # Check if we have fewer images than num_images
        if i >= inputs.shape[0]:
            break

        # Assuming images are in [C, H, W] format and grayscale
        ax = axs[i, 0]
        ax.imshow(inputs[i].squeeze(), cmap='gray', aspect='auto')
        ax.set_title('Original Image')
        ax.axis('off')

        ax = axs[i, 1]
        ax.imshow(outputs[i].squeeze(), cmap='gray', aspect='auto')
        ax.set_title('Reconstructed Image')
        ax.axis('off')

    plt.tight_layout()
    plt.show()
def visualize_output(inputs, outputs, file_path=None):
    inputs = inputs.detach().cpu()
    outputs = outputs.detach().cpu()
    print(f"Inputs Shape: {inputs.shape}")
    print(f"Outputs Shape: {outputs.shape}")
     # SELECT THE MIDDLE SLICE
    midSlice = inputs.shape[-1] // 2
    inputs = inputs[:, :, :, :,midSlice]
    outputs = outputs[:, :, :, :,midSlice]
    print(f"Inputs Shape: {inputs.shape}")
    print(f"Outputs Shape: {outputs.shape}")

    # Create a grid of images: original, reconstructed, difference
    for i in range(inputs.shape[0]):
        # Remove unneeded channel dim
        images = torch.stack([inputs[i], outputs[i]], dim=0)

        print(f"Images Shape: {images.shape}")
        grid = vutils.make_grid(images, nrow=2, normalize=True, scale_each=True)
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.show()


if __name__ == "__main__":
    do_main()
    # evaluate_model('/home/ssome/PycharmProjects/pythonProject/AML/Unsupervised/UnsupervisedEncoderLogs/UnsupervisedEncoder-checkpoint-epoch=09-val_loss=0.77.ckpt')