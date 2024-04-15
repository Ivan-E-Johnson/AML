import monai
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path

from monai.data import PatchIter, GridPatchDataset, DataLoader, CacheDataset
from monai.transforms import Compose, DataStatsD, EnsureChannelFirstD, ToTensord

from support_functions import *


def _create_dict_of_preprocessed_unlabeled_data(base_data_path: Path, is_testing:bool = False)-> list:
    data_dicts = []

    for dir in base_data_path.iterdir():
        if dir.is_dir():
            images_list = []
            images_list.append(dir / f"{dir.name}_pp_t2w.nii.gz")
            images_list.append(dir / f"{dir.name}_pp_adc.nii.gz")
            images_list.append(dir / f"{dir.name}_pp_blow.nii.gz")
            images_list.append(dir / f"{dir.name}_pp_tracew.nii.gz")
            data_dicts.append({"image": images_list})
    if is_testing:
        data_dicts = data_dicts[:10]
    return data_dicts


def get_validation_transforms():
    return Compose([
        EnsureChannelFirstD(keys=["image"]),
        DataStatsD(keys=["image"]),
        ToTensord(keys=["image"]),
    ])


if __name__ == "__main__":
    base_data_path = Path("/Users/iejohnson/School/spring_2024/AML/Supervised_learning/DATA/ALL_PROSTATEx/WITHOUT_SEGMENTATION/PreProcessed")
    print(f"Base data path: {base_data_path}")
    data_dicts = _create_dict_of_preprocessed_unlabeled_data(base_data_path, is_testing=True)
    print(f"Data dicts: {data_dicts}")

    cache_dataset = CacheDataset(data=data_dicts, transform=Compose(ModalityStackTransformd(keys=["image"])))
    images = next(iter(cache_dataset))["image"]

    print("item size:", images)
    print("CacheDataset: ", cache_dataset)



    patch_iter = PatchIter(patch_size=(60,60,4), start_pos=(0,0,0))
    PatchDataset = GridPatchDataset(images,patch_iter=patch_iter, transform=get_validation_transforms())
    test_loader = DataLoader(PatchDataset, batch_size=2, num_workers=2)
    images = next(iter(test_loader))["image"]

    print("item size:", images)
    print("CacheDataset: ", CacheDataset)

    # for item in DataLoader(PatchDataset, batch_size=2, num_workers=2):
    #     print("patch size:", item[0].shape)
    #     print("coordinates:", item[1])
    # print("PatchDataset: ", PatchDataset)
    # print("PatchDataset[0]: ", PatchDataset[0])

    print(data_dicts)
    print("Done")