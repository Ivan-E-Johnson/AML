from pathlib import Path
import os
import shutil
import tempfile
import matplotlib.pyplot as plt
import itk
import torch
import numpy as np
from sklearn.metrics import classification_report
import pydicom
from monai.apps import download_and_extract
from monai.config import print_config
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    ToTensor,
)
from monai.data import Dataset, DataLoader
from monai.utils import set_determinism
import numpy as np
print_config()

# Load Data from Data Folder
base_data_path = Path('/Users/iejohnson/School/spring_2024/AML/Supervised_learning/Data')
input_prostate_data = base_data_path / "OrigProstate/PROSTATEx"
segmentation_data = base_data_path / "Segmentations/PROSTATEx"

prostate_dirs = [x for x in input_prostate_data.iterdir() if x.is_dir()]

def create_data_dict(prostate_dirs, segmentation_base):
    data_dict = {}
    for prostate_dir in prostate_dirs:
        patient_id = prostate_dir.name
        data_dict[patient_id] = {}
        data_dict[patient_id]['prostate'] = prostate_dir
        data_dict[patient_id]['segmentation'] = segmentation_base / patient_id


print(prostate_dirs)
print(segmentation_dirs)


# Starter code for creating and augmenting datasets



# Create a list of dictionaries, each containing the filename for an image and its corresponding label
# data_dicts = [{'image': img, 'label': lbl} for img, lbl in zip(image_files, label_files)]
#
# # Define transformations for augmenting the dataset
# # MONAI's dictionary-based transforms expect dictionaries with keys matching those in the data_dicts
# transforms = Compose([
#     LoadImaged(keys=['image', 'label']),       # Load image and label
#     AddChanneld(keys=['image', 'label']),     # Add channel dimension
#     ScaleIntensityd(keys=['image']),          # Normalize image intensity
#     RandRotated(keys=['image', 'label'], range=np.pi/4, prob=0.5),  # Random rotation
#     RandFlipd(keys=['image', 'label'], spatial_axis=1, prob=0.5),   # Random flip
#     RandAffined(keys=['image', 'label'], prob=0.5, translate_range=5), # Random affine transformations
#     ToTensord(keys=['image', 'label'])        # Convert to PyTorch Tensor
# ])
#
# # Create the dataset
# # The Dataset class in MONAI is a simple wrapper around a list of data_dicts and the composed transforms
# dataset = Dataset(data=data_dicts, transform=transforms)

# Note: In this code, we apply a series of augmentations, including random rotations, flips,
# and affine transformations. You can customize these based on your specific requirements.

# The actual instantiation of the dataset object and the augmentation will occur when this
# code is run in an environment with the necessary libraries installed.
