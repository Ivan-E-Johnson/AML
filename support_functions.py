import os
from pathlib import Path

import itk
import numpy as np
import pytorch_lightning
import torch


def convert_to_bi_name(name: str):
    study_num = name.split("-")[1]
    return f"PRX{study_num}"


def init_stacked_data_lists():
    """
    This function initializes the image and mask paths.

    Args:
    base_data_path (Path): The base path of the data.

    Returns:
    list: A list of image paths.
    list: A list of mask paths.
    """
    base_data_path = Path(os.environ.get("STACKED_DATA_PATH"))
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


def init_t2w_only_data_lists():
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
            image_paths.append(dir / f"{dir.name}_resampled_normalized_t2w.nii.gz")
            mask_paths.append(dir / f"{dir.name}_resampled_segmentations.nii.gz")

    return image_paths, mask_paths


class ModalityStackTransformd:
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
            #
            # transposed = np.transpose(
            #     stacked_image, axes=(0, 2, 3, 1)
            # )  # Move channel dimension to the last

            stacked_image = np.moveaxis(stacked_image, 1, -1)
            # assert transposed.shape == stacked_image.shape
            # assert np.allclose(transposed, stacked_image)
            d["image"] = stacked_image
            # print(f"Stacked image shape: {stacked_image.shape}")
            # print(f"Stacked image dtype: {stacked_image.dtype}")

        return d


class LoadAndSplitLabelsToChannelsd:
    """
    A custom transformation to convert a label image to a one-hot encoded tensor.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = itk.imread(d[key])
            label_array = itk.array_from_image(label)
            unique_values = np.unique(label_array)
            num_classes = len(unique_values)

            # Create a one-hot encoded tensor
            one_hot = np.zeros((num_classes, *label_array.shape), dtype=np.float32)
            for i, value in enumerate(unique_values):
                one_hot[i] = label_array == value
            one_hot = np.moveaxis(one_hot, 1, -1)

            d["label"] = one_hot
        return d


class BestModelCheckpoint(pytorch_lightning.callbacks.Callback):
    def __init__(
        self, monitor="val_dice", mode="max", experiment_name="no_experiment_name_given"
    ):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.experiment_name = experiment_name

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
                        checkpoint_callback.dirpath,
                        f"{self.experiment_name}_best_model.pth",
                    )
                    trainer.save_checkpoint(checkpoint_callback.best_model_path)


def convert_logits_to_one_hot(logits):
    """
    This function converts the logits to a one-hot encoded tensor.

    Args:
    logits (torch.Tensor): The logits tensor.

    Returns:
    torch.Tensor: The one-hot encoded tensor.
    """
    num_classes = logits.shape[1]
    prob_outputs = torch.softmax(logits, dim=1)
    prob_outputs = prob_outputs.argmax(dim=1)
    prob_outputs = torch.nn.functional.one_hot(prob_outputs, num_classes=num_classes)
    prob_outputs = prob_outputs.moveaxis(-1, 1)
    return prob_outputs


def convert_one_hot_to_label(one_hot):
    """
    This function converts a one-hot encoded tensor to a label tensor.

    Args:
    one_hot (torch.Tensor): The one-hot encoded tensor.

    Returns:
    torch.Tensor: The label tensor.
    """
    label = one_hot.argmax(dim=1)
    return label


def convert_AutoEncoder_output_to_labelpred(logits):
    """
    This function converts a one-hot encoded tensor to a label tensor.

    Args:
    one_hot (torch.Tensor): The one-hot encoded tensor.

    Returns:
    torch.Tensor: The label tensor.
    """
    prob_outputs = torch.softmax(logits, dim=1)
    return prob_outputs.argmax(dim=1).unsqueeze(1)
