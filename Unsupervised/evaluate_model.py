import os
from pathlib import Path

import scipy
import numpy as np

from monai.data.image_reader import nib
from monai.transforms import ToTensord, EnsureChannelFirstD, LoadImaged, Compose, LoadImageD, EnsureChannelFirstd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import scipy.ndimage
import torch
from Encoder import LitAutoEncoder, _create_image_dict
from monai.data import DataLoader, CacheDataset


def load_model(checkpoint_path):
    model = LitAutoEncoder.load_from_checkpoint(
        checkpoint_path,
        img_size=(320, 320, 32),
        patch_size=(16, 16, 16),
        in_channels=1
    )
    model.eval()
    return model


def evaluate_model(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create lists to store results
    all_images = []
    all_labels = []
    all_predictions = []
    for batch in dataloader:
        images_tensor = batch["image"].to(device)
        print(batch.keys())
        labels_tensor = batch["label"].to(device)
        outputs, _ = model(images_tensor)
        predictions = torch.argmax(outputs, dim=1)
        all_images.append(images_tensor.cpu().numpy())
        all_labels.append(labels_tensor.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())

    images_np = np.concatenate(all_images, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    predictions_np = np.concatenate(all_predictions, axis=0)
    for i in range(len(images_np)):
        single_image = images_np[i, 0]  # single-channel images
        single_label = labels_np[i]
        single_prediction = predictions_np[i]
        single_label = single_label.squeeze(axis=0)
        print(
            "single image:",
            single_image.shape,
            "single label:",
            single_label.shape,
            "single prediction:",
            single_prediction.shape,
        )
        visualize_labels(single_image, single_label, single_prediction)

def prepare_data():
    transforms = Compose([
        LoadImageD(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
    ])
    base_data_path = Path(os.getenv("DATA_PATH"))
    data_dicts = _create_image_dict(base_data_path, is_testing=True)
    dataset = CacheDataset(data=data_dicts, transform=transforms, cache_rate=1.0, num_workers=4)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader

def simple_test_visualization():
    # Load NIfTI image
    base_pp_with_seg_path = Path(os.environ["PP_WITH_SEGMENTATION_PATH"])
    # base_pp_with_seg_path = Path("/localscratch/Users/iejohnson/DATA/ALL_PROSTATEx/WITH_SEGMENTATION/PreProcessed/ProstateX-0004")
    subject_id = "ProstateX-0004"
    image_path = base_pp_with_seg_path /f"{subject_id}/{subject_id}_pp_t2w.nii.gz"

    # Load NIfTI segmentation label
    label_path = base_pp_with_seg_path / f"{subject_id}/{subject_id}_pp_segmentation.nii.gz"

    # Load image and label data
    image_data = nib.load(image_path).get_fdata()
    label_data = nib.load(label_path).get_fdata()

    # Define custom colormap for segmentation labels
    # Map the label values to colors
    label_colors = ListedColormap([
        'black',  # Background (0)
        'red',  # Peripheral Zone (1)
        'green',  # Transition Zone (2)
        'blue',  # Fibromuscular Stroma (3)
        'yellow',  # Distal Prostatic Urethra (4)
    ])

    # The names of the labels
    label_names = [
        'Background',
        'Peripheral Zone',
        'Transition Zone',
        'Fibromuscular Stroma',
        'Distal Prostatic Urethra'
    ]

    # Display NIfTI image
    slice_index = image_data.shape[2] // 2  # Index for the central slice
    plt.figure(figsize=(8, 8))
    plt.imshow(image_data[:, :, slice_index], cmap='gray')
    plt.imshow(label_data[:, :, slice_index], cmap=label_colors, alpha=0.5)  # Overlay labels on central slice
    plt.title('Segmentation Labels Overlay')
    plt.axis('off')

    # Create a colorbar with label names
    colorbar = plt.colorbar(ticks=range(len(label_names)))
    colorbar.set_ticklabels(label_names)

    # Adjust colorbar height to match the figure
    ax = plt.gca()
    image_aspect = image_data.shape[0] / image_data.shape[1]
    colorbar.ax.set_aspect(20.0 * image_aspect)

    plt.show()

def visualize_labels(image_data, label_data, prediction_data):

    # Map the label values to colors
    label_colors = ListedColormap(
        [
            "black",  # Background
            "red",  # Peripheral Zone
            "green",  # Transition Zone
            "blue",  # Fibromuscular Stroma
            "yellow",  # Urethra
        ]
    )

    # The names of the labels
    label_names = [
        "Background",
        "Peripheral Zone",
        "Transition Zone",
        "Fibromuscular Stroma",
        "Urethra",
    ]
    print(image_data.ndim, image_data.shape)
    if image_data.ndim == 3 and image_data.shape[0] == 1:
        image_data = image_data.squeeze(0)
        print("HERE")

    slice_index = image_data.shape[2] // 2  # Index for the central slice
    plt.figure(figsize=(8, 8))
    plt.imshow(image_data[:, :, slice_index], cmap="gray")
    plt.imshow(
        label_data[:, :, slice_index], cmap=label_colors, alpha=0.5
    )  # Overlay labels on central slice
    plt.title("True Labels Overlay")
    plt.axis("off")
    # mqke colorbar with label names
    # colorbar = plt.colorbar(ticks=range(len(label_names)))
    # colorbar.set_ticklabels(label_names)

    # match colorbar height to figure
    ax = plt.gca()
    image_aspect = image_data.shape[0] / image_data.shape[1]
    # colorbar.ax.set_aspect(20.0 * image_aspect)

    plt.show()
    plt.figure(figsize=(8, 8))
    prediction_data = scipy.ndimage.rotate(prediction_data, 90, reshape=False)
    plt.imshow(image_data[:, :, slice_index], cmap="gray")
    plt.imshow(
        prediction_data[:, :, slice_index], cmap=label_colors, alpha=0.5
    )  # Overlay predictions on central slice
    plt.title("Predictions Overlay")
    plt.axis("off")

    # Create a colorbar with label names
    colorbar = plt.colorbar(ticks=range(len(label_names)))
    colorbar.set_ticklabels(label_names)

    # Match colorbar height to figure
    ax = plt.gca()
    image_aspect = image_data.shape[0] / image_data.shape[1]
    colorbar.ax.set_aspect(20.0 * image_aspect)

    plt.show()


if __name__ == "__main__":
    # saved_model_path = (
    #     "lightning_logs/best-model-699-0.77|700..epoch_run.ckpt"  # Path to the saved model
    # )
    # model = load_model(saved_model_path)
    # dataloader = prepare_data()
    # evaluate_model(model, dataloader)
    simple_test_visualization()