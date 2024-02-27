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
base_data_path = Path(
    "/Users/iejohnson/School/spring_2024/AML/Supervised_learning/Data/SortedProstateData"
)


def init_data_lists(base_data_path):
    mask_paths = []
    image_paths = []
    for dir in base_data_path.iterdir():
        if dir.is_dir():
            mask_paths.append(dir / f"{dir.name}_prostate.nii.gz")
            image_paths.append(dir / f"{dir.name}_segmentation.nii.gz")
    return image_paths, mask_paths


print("finished")
