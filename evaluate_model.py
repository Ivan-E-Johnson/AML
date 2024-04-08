import scipy
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap
from monai.data import DataLoader, CacheDataset
from monai.transforms import ToTensord, EnsureChannelFirstD, LoadImaged, Compose
from sklearn.model_selection import train_test_split

from AML_Project_Supervised.basic_unet import Net
from AML_Project_Supervised.support_functions import init_t2w_only_data_lists


def run_inference(saved_model_path, val_loader):
    # load model
    device = 'cpu'
    model = Net.load_from_checkpoint(saved_model_path)
    model = model.to(device)
    model.eval()

    # Create lists to store results
    all_images = []
    all_labels = []
    all_predictions = []

    # Iterate through the validation loader
    for batch in val_loader:
        # Move batch to device
        images_tensor = batch["image"].to(device)
        labels_tensor = batch["label"].to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(images_tensor)
            predictions = torch.argmax(outputs, dim=1)

        # Append results to lists
        all_images.append(images_tensor.cpu().numpy())
        all_labels.append(labels_tensor.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())

    # Concatenate results
    images_np = np.concatenate(all_images, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    predictions_np = np.concatenate(all_predictions, axis=0)

    # Calculate Dice score
    dice = model.dice_metric(torch.tensor(predictions_np), torch.tensor(labels_np))

    # Visualize the results
    for i in range(len(images_np)):
        single_image = images_np[i, 0]  #  single-channel images
        single_label = labels_np[i]
        single_prediction = predictions_np[i]
        single_label = single_label.squeeze(axis=0)
        print("single image:",single_image.shape, "single label:", single_label.shape, "single prediction:", single_prediction.shape)
        # Call visualize_labels to display the results
        visualize_labels(single_image, single_label, single_prediction)

    return dice


def visualize_labels(image_data, label_data, prediction_data):

    # Map the label values to colors
    label_colors = ListedColormap([
        'black',  # Background
        'red',  # Peripheral Zone
        'green',  # Transition Zone
        'blue',  # Fibromuscular Stroma
        'yellow',  # Urethra
    ])

    # The names of the labels
    label_names = [
        'Background',
        'Peripheral Zone',
        'Transition Zone',
        'Fibromuscular Stroma',
        'Urethra'
    ]

    slice_index = image_data.shape[2] // 2  # Index for the central slice
    plt.figure(figsize=(8, 8))
    plt.imshow(image_data[:, :, slice_index], cmap='gray')
    plt.imshow(label_data[:, :, slice_index], cmap=label_colors, alpha=0.5)  # Overlay labels on central slice
    plt.title('True Labels Overlay')
    plt.axis('off')
    # mqke colorbar with label names
    colorbar = plt.colorbar(ticks=range(len(label_names)))
    colorbar.set_ticklabels(label_names)

    # match colorbar height to figure
    ax = plt.gca()
    image_aspect = image_data.shape[0] / image_data.shape[1]
    colorbar.ax.set_aspect(20.0 * image_aspect)

    plt.show()
    plt.figure(figsize=(8, 8))
    prediction_data = scipy.ndimage.rotate(prediction_data, 90, reshape=False)
    plt.imshow(image_data[:, :, slice_index], cmap='gray')
    plt.imshow(prediction_data[:, :, slice_index], cmap=label_colors, alpha=0.5)  # Overlay predictions on central slice
    plt.title('Predictions Overlay')
    plt.axis('off')

    # Create a colorbar with label names
    colorbar = plt.colorbar(ticks=range(len(label_names)))
    colorbar.set_ticklabels(label_names)

    # Match colorbar height to figure
    ax = plt.gca()
    image_aspect = image_data.shape[0] / image_data.shape[1]
    colorbar.ax.set_aspect(20.0 * image_aspect)

    plt.show()

if __name__ == '__main__':
    saved_model_path = 'logs/basic_unet_final_project_best_model.pth' # Path to the saved model
    image_paths, mask_paths = init_t2w_only_data_lists() # Get the paths to the images and masks
    train_image_paths, test_image_paths, train_mask_paths, test_mask_paths = (
        train_test_split(image_paths, mask_paths, test_size=0.2)
    ) # Split the data into training and testing sets
    val_files = [
        {"image": img_path, "label": mask_path}
        for img_path, mask_path in zip(test_image_paths, test_mask_paths)
    ] # Create a list of dictionaries containing the image and mask paths
    val_files = val_files[:1] # Use only the first image for testing
    validation_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"], reader="ITKReader", image_only=False),
                EnsureChannelFirstD(
                    keys=["image", "label"]
                ),
                ToTensord(keys=["image", "label"]),
                # DataStatsD(keys=["image", "label"]),
            ]
        ) # Define the validation transforms
    val_ds = CacheDataset(
        data=val_files,
        transform=validation_transforms,
        cache_rate=0.8,
        num_workers=4,
        runtime_cache=True,
    )   # Create the validation dataset
    val_loader = DataLoader(val_ds, batch_size=10, num_workers=8) # Create the validation data loader

    run_inference(saved_model_path, val_loader) # Run inference on the validation data
