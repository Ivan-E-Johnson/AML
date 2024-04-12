import monai
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from support_functions import *



if __name__ == "__main__":

    image_paths, mask_paths = init_stacked_data_lists()
    # only use the first 10 images for testing
    image_paths = image_paths[:10]
    mask_paths = mask_paths[:10]

    train_files = [
        {"image": img_path, "label": mask_path}
        for img_path, mask_path in zip(image_paths, mask_paths)
    ]




