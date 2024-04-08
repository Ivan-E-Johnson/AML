import os
import random
from collections import defaultdict

import itk
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from data_connector import get_exploratory_image_info, get_image_stats
import pandas as pd
import itertools
import nibabel as nib
from dotenv import load_dotenv
import os

load_dotenv()

def show_subject_images(image: itk.Image, mask: itk.Image):
    # Convert the images to numpy arrays
    image_np = itk.GetArrayViewFromImage(image)
    mask_np = itk.GetArrayViewFromImage(mask)

    # Plot the images
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image_np[5, :, :], cmap="gray")
    ax[0].set_title("Image")
    ax[1].imshow(mask_np[5, :, :], cmap="gray")
    ax[1].set_title("Mask")
    plt.show()


def init_data_lists(base_data_path):
    mask_paths = []
    image_paths = []
    for dir in base_data_path.iterdir():
        if dir.is_dir():
            image_paths.append(dir / f"{dir.name}_prostate.nii.gz")
            mask_paths.append(dir / f"{dir.name}_segmentation.nii.gz")
    return image_paths, mask_paths


def get_mask_bounding_box(
        mask: itk.Image, label: int
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    # Initialize the LabelStatisticsImageFilter
    stats_filter = itk.LabelStatisticsImageFilter.New(Input=mask, LabelInput=mask)
    stats_filter.Update()

    # Extract the bounding box for the specified label
    bounding_box = stats_filter.GetBoundingBox(label)
    # Unpack the bounding box values
    x_start, x_end, y_start, y_end, z_start, z_end = bounding_box
    print(
        f"X Start: {x_start}, X End: {x_end}, Y Start: {y_start}, Y End: {y_end}, Z Start: {z_start}, Z End: {z_end}"
    )
    return (x_start, y_start, z_start), (x_end, y_end, z_end)


def calculate_bounding_box_extent(mask_image: itk.Image) -> tuple[int, int, int]:
    # Get the bounding box
    bounding_box = get_mask_bounding_box(mask_image, 1)
    x_extent = bounding_box[1][0] - bounding_box[0][0]
    y_extent = bounding_box[1][1] - bounding_box[0][1]
    z_extent = bounding_box[1][2] - bounding_box[0][2]
    print(f"X Extent: {x_extent}, Y Extent: {y_extent}, Z Extent: {z_extent}")
    return x_extent, y_extent, z_extent


def calculate_bounding_box_extent_in_mm(mask_image):
    # Get the image spacing
    spacing = mask_image.GetSpacing()
    # Get the bounding box
    x_extent, y_extent, z_extent = calculate_bounding_box_extent(mask_image)
    x_extent_mm = x_extent * spacing[0]
    y_extent_mm = y_extent * spacing[1]
    z_extent_mm = z_extent * spacing[2]
    # Calculate the extent in mm
    extent_str = f"X Extent (mm): {x_extent_mm}, Y Extent (mm): {y_extent_mm}, Z Extent (mm): {z_extent_mm}"
    return extent_str


def convert_multiclass_mask_to_binary(mask: itk.Image) -> itk.Image:
    # Initialize the BinaryThresholdImageFilter
    binary_filter = itk.BinaryThresholdImageFilter.New(
        Input=mask, LowerThreshold=0, UpperThreshold=0, InsideValue=0, OutsideValue=1
    )
    binary_filter.Update()

    return binary_filter.GetOutput()


def get_initial_reportable_image_stats(itk_image: itk.Image) -> dict:
    mean, std, max, min = get_image_stats(itk_image)
    spacing = list(itk_image.GetSpacing())
    size = list(itk_image.GetLargestPossibleRegion().GetSize())
    return {
        "max": max,
        "mean": mean,
        "min": min,
        "std": std,
        "spacing": spacing,
        "size": size,
    }


def write_exploratry_image_info(image_paths, mask_paths, output_path):
    image_df = pd.DataFrame()
    mask_df = pd.DataFrame()
    for image_path, mask_path in zip(image_paths, mask_paths):
        print(image_path, mask_path)
        image = itk.imread(str(image_path), itk.F)
        mask = itk.imread(str(mask_path), itk.UC)
        image_info = get_initial_reportable_image_stats(image)
        mask_info = get_initial_reportable_image_stats(mask)
        image_info["image_name"] = image_path.name
        mask_info["mask_name"] = mask_path.name

        print(f"Image Info: {image_info}")
        print(f"Mask Info: {mask_info}")
        # show_subject_images(image, mask)
        # Convert the multiclass mask to a binary mask
        binary_mask = convert_multiclass_mask_to_binary(mask)

        # show_subject_images(image,binary_mask)

        # Calculate the extent of the bounding box
        prostate_extent = calculate_bounding_box_extent(binary_mask)
        x_extent, y_extent, z_extent = prostate_extent
        mask_info["prostate_x_extent"] = x_extent
        mask_info["prostate_y_extent"] = y_extent
        mask_info["prostate_z_extent"] = z_extent
        prostate_extent_mm = calculate_bounding_box_extent_in_mm(binary_mask)
        mask_info["prostate_extent_mm"] = prostate_extent_mm
        if image_df.empty:
            image_df = pd.Series(image_info).to_frame().T
            mask_df = pd.Series(mask_info).to_frame().T
        else:
            image_df = pd.concat(
                [image_df, pd.Series(image_info).to_frame().T], ignore_index=True
            )
            mask_df = pd.concat(
                [mask_df, pd.Series(mask_info).to_frame().T], ignore_index=True
            )
    print(image_df.head())
    image_df.to_csv("image_info.csv", index=False)
    mask_df.to_csv("mask_info.csv", index=False)


def find_center_of_gravity_in_index_space(image: itk.Image) -> tuple[int, int, int]:
    """
    This function calculates the center of gravity of an image in index space.
    It first casts the image to unsigned char type, then uses the ImageMomentsCalculator
    from the itk library to compute the center of gravity in index space.
    Finally, it transforms the center of gravity from physical space to continuous index space.

    Parameters:
    image (itk.Image): The input image.

    Returns:
    tuple[int, int, int]: The center of gravity in continuous index space.
    """

    UC_Image_Type = itk.Image[itk.UC, 3]
    UC_image = rescale_and_cast_to_unsigned_char(image)

    moments_filter = itk.ImageMomentsCalculator[UC_Image_Type].New()
    moments_filter.SetImage(UC_image)
    moments_filter.Compute()
    center_of_gravity_in_physical_space = moments_filter.GetCenterOfGravity()
    center_of_gravity_in_index_space = image.TransformPhysicalPointToContinuousIndex(
        center_of_gravity_in_physical_space
    )

    return center_of_gravity_in_index_space


def rescale_and_cast_to_unsigned_char(image: itk.Image) -> itk.Image:
    UC_Image_Type = itk.Image[itk.UC, 3]
    normalizer = itk.RescaleIntensityImageFilter[image, UC_Image_Type].New()
    normalizer.SetInput(image)
    normalizer.SetOutputMinimum(0)
    normalizer.SetOutputMaximum(255)
    normalizer.Update()
    return normalizer.GetOutput()


def simple_otsu_thresholding(image: itk.Image) -> itk.Image:
    # Initialize the OtsuThresholdImageFilter
    uc_image = rescale_and_cast_to_unsigned_char(image)
    otsu_filter = itk.OtsuThresholdImageFilter[uc_image, uc_image].New()
    otsu_filter.SetInput(uc_image)
    otsu_filter.Update()
    return otsu_filter.GetOutput()


def multi_otsu_thresholding(
        image: itk.Image, number_of_thresholds: int, number_of_histogram_bins: int
) -> itk.Image:
    # Initialize the OtsuThresholdImageFilter
    uc_image = rescale_and_cast_to_unsigned_char(image)
    otsu_filter = itk.OtsuMultipleThresholdsImageFilter[uc_image, uc_image].New()
    otsu_filter.SetInput(uc_image)
    otsu_filter.SetNumberOfHistogramBins(number_of_histogram_bins)
    otsu_filter.SetNumberOfThresholds(number_of_thresholds)
    otsu_filter.Update()
    return otsu_filter.GetOutput()


def calculate_distance_between_points(
        point1: tuple[int, int, int], point2: tuple[int, int, int]
) -> float:
    x_distance = point1[0] - point2[0]
    y_distance = point1[1] - point2[1]
    z_distance = point1[2] - point2[2]
    distance = (x_distance ** 2 + y_distance ** 2 + z_distance ** 2) ** 0.5
    return distance


def plot_dict_images(
        threshold_dict: dict, center_dict: dict, true_center: tuple[int, int, int]
):
    SAVE_IMAGES = False

    for key, value in threshold_dict.items():
        center = center_dict[key]
        plt.imshow(value[5, :, :], cmap="gray")
        plt.scatter(center[0], center[1], c="r", marker="*")
        plt.scatter(true_center[0], true_center[1], c="b", marker="x")
        plt.title(f"Threshold: {key}")
        if SAVE_IMAGES:
            plt.savefig(f"Threshold_{key}.png")
        plt.show()

    # for key, value in threshold_dict.items():
    #
    #     plt.imshow(value[5,:,:],cmap="gray")
    #     plt.title(f"Threshold: {key}")
    #     plt.show()


def find_closest_center(true_center: tuple[int, int, int], center_dict: dict):
    closest_center = None
    closest_distance = float("inf")
    for key, value in center_dict.items():
        distance = calculate_distance_between_points(true_center, value)
        print(
            f"For {key}:"
            f"True Center: {true_center}, Center: {value}, Distance: {distance}"
            f"Closest Center: {closest_center}, Closest Distance: {closest_distance} for {key}"
            f"************************************"
        )
        if distance < closest_distance:
            closest_distance = distance
            closest_center = key
    return closest_center


def compare_average_center_of_mass(base_data_path):
    output_data_path = Path("heck_Centers")
    output_data_path.mkdir(exist_ok=True, parents=True)

    thresholding_list = [1, 3, 5]
    histogram_list = [10, 30, 75]
    image_types = ["raw_image", "simple_otsu"] + [
        f"{t}_{h}"
        for t, h in itertools.product(thresholding_list, histogram_list)
        if t < h
    ]

    # Initialize dictionaries to store total distances and counts for each image type
    total_distances = {image_type: 0.0 for image_type in image_types}
    counts = {image_type: 0 for image_type in image_types}

    image_paths, mask_paths = init_data_lists(base_data_path)
    # selected_indices = random.sample(range(len(image_paths)), 10)
    # selected_image_paths = [image_paths[i] for i in selected_indices]
    # selected_mask_paths = [mask_paths[i] for i in selected_indices]
    selected_image_paths = image_paths
    selected_mask_paths = mask_paths
    for image_path, mask_path in zip(selected_image_paths, selected_mask_paths):
        print(f"Processing {image_path} and {mask_path}")
        image = itk.imread(str(image_path), itk.F)
        mask = itk.imread(str(mask_path), itk.UC)

        # Raw image
        raw_image_center = find_center_of_gravity_in_index_space(image)
        mask_center = find_center_of_gravity_in_index_space(mask)
        raw_distance = calculate_distance_between_points(raw_image_center, mask_center)
        total_distances["raw_image"] += raw_distance
        counts["raw_image"] += 1

        # Simple Otsu thresholding
        simple_otsu_image = simple_otsu_thresholding(image)
        simple_otsu_center = find_center_of_gravity_in_index_space(simple_otsu_image)
        simple_otsu_distance = calculate_distance_between_points(
            simple_otsu_center, mask_center
        )
        total_distances["simple_otsu"] += simple_otsu_distance
        counts["simple_otsu"] += 1

        # Multi Otsu thresholding
        for threshold, histogram in itertools.product(
                thresholding_list, histogram_list
        ):
            if threshold >= histogram:
                continue
            image_type = f"{threshold}_{histogram}"
            threshed_image = multi_otsu_thresholding(image, threshold, histogram)
            threshed_center = find_center_of_gravity_in_index_space(threshed_image)
            distance = calculate_distance_between_points(threshed_center, mask_center)
            total_distances[image_type] += distance
            counts[image_type] += 1

    # Calculate and print running average distances
    average_distances = {
        image_type: total_distances[image_type] / counts[image_type]
        for image_type in image_types
    }
    for image_type, avg_distance in average_distances.items():
        print(f"{image_type}: Average Distance = {avg_distance}")


def calculate_new_origin_from_center_of_mass(
        center_of_mass_in_phsyical_space: tuple[float, float, float], spacing, size
):
    return (
            np.array(center_of_mass_in_phsyical_space)
            - (
                    np.array([(size[0] / 2 - 1), (size[1] / 2 - 1), (size[2] / 2 - 1)])
                    * np.array(spacing)
            ).tolist()
    )


def get_recentered_image_from_center_of_mass(
        mask: itk.Image,
        center_of_mass: tuple[int, int, int],
        x_extent: int,
        y_extent: int,
        z_extent: int,
):
    PIXEL_TYPE = itk.UC
    IMAGE_TYPE = itk.Image[PIXEL_TYPE, 3]
    # Get the region from the bounding box
    region = itk.ImageRegion[3]()
    direction = mask.GetDirection()
    direction.SetIdentity()
    size = itk.Size[3]()
    size[0] = x_extent
    size[1] = y_extent
    size[2] = z_extent
    index = itk.Index[3]()
    index[0] = 0
    index[1] = 0
    index[2] = 0
    region.SetSize(size)
    region.SetIndex(index)

    # Leave direction as identity
    new_image_origin = calculate_new_origin_from_center_of_mass(
        center_of_mass, mask.GetSpacing(), size
    )

    new_centered_image = IMAGE_TYPE.New()
    new_centered_image.SetRegions(region)
    new_centered_image.SetSpacing(mask.GetSpacing())
    new_centered_image.SetOrigin(new_image_origin)
    new_centered_image.SetDirection(mask.GetDirection())
    new_centered_image.Allocate()

    return new_centered_image


def resample_image_to_reference(image: itk.Image, reference_image: itk.Image):
    IMAGETYPE = itk.Image[itk.UC, 3]

    nearest_neighbor_interpolator = itk.NearestNeighborInterpolateImageFunction[
        IMAGETYPE, itk.D
    ].New()

    resampler = itk.ResampleImageFilter[IMAGETYPE, IMAGETYPE].New()
    resampler.SetInput(image)
    resampler.SetReferenceImage(reference_image)
    resampler.UseReferenceImageOn()
    resampler.UpdateLargestPossibleRegion()

    resampler.SetInterpolator(
        nearest_neighbor_interpolator
    )  # Nearest Neighbor interpolation for mask
    resampler.SetTransform(itk.IdentityTransform[itk.D, 3].New())
    resampler.Update()
    return resampler.GetOutput()


def round_to_next_largest_16_multiple(number):
    return int(np.ceil(number / 16) * 16)


def find_largest_needed_region_for_prostate(base_data_path: Path):
    extenstion_factor = 1.3
    # Get the image and mask paths
    im, mask_paths = init_data_lists(base_data_path)

    # Write the exploratory image info
    # Calculate center of mass

    df = pd.DataFrame()

    for image_path, mask_path in zip(im, mask_paths):
        image = itk.imread(str(image_path), itk.F)
        mask = itk.imread(str(mask_path), itk.UC)
        binary_mask = convert_multiclass_mask_to_binary(mask)
        # number_non_zero = np.count_nonzero(itk.GetArrayViewFromImage(mask))
        # print(f"Number of non zero pixels in mask: {number_non_zero}")

        center_of_mass = mask.TransformContinuousIndexToPhysicalPoint(
            find_center_of_gravity_in_index_space(binary_mask)
        )

        # Todo figure out what region will fit the largest prostate available + 10%
        bounding_box = get_mask_bounding_box(mask, 1)
        x_extent, y_extent, z_extent = calculate_bounding_box_extent(binary_mask)
        x_extended = int(x_extent * extenstion_factor)
        y_extended = int(y_extent * extenstion_factor)
        z_extended = int(z_extent * extenstion_factor)
        x_rounded = round_to_next_largest_16_multiple(x_extended)
        y_rounded = round_to_next_largest_16_multiple(y_extended)
        z_rounded = round_to_next_largest_16_multiple(z_extended)
        physical_rounded_size_x = x_rounded * mask.GetSpacing()[0]
        physical_rounded_size_y = y_rounded * mask.GetSpacing()[1]
        physical_rounded_size_z = z_rounded * mask.GetSpacing()[2]
        uc_image = rescale_and_cast_to_unsigned_char(image)
        recentered_image = get_recentered_image_from_center_of_mass(
            mask, center_of_mass, x_extent, y_extent, z_extent
        )
        extended_recentered_image = get_recentered_image_from_center_of_mass(
            mask,
            center_of_mass,
            int(x_extent * extenstion_factor),
            int(y_extent * extenstion_factor),
            int(z_extent * extenstion_factor),
        )
        rounded_extended_recentered_image = get_recentered_image_from_center_of_mass(
            mask, center_of_mass, x_rounded, y_rounded, z_rounded
        )
        # # # Resample the mask to the recentered image
        recentered_image = resample_image_to_reference(mask, recentered_image)
        extended_recentered_image = resample_image_to_reference(
            mask, extended_recentered_image
        )
        rounded_extended_recentered_image = resample_image_to_reference(
            mask, rounded_extended_recentered_image
        )
        # itk.imwrite(recentered_image, f"recentered_{mask_path.name}")
        # itk.imwrite(extended_recentered_image, f"extended_recentered_{mask_path.name}")
        # itk.imwrite(rounded_extended_recentered_image, f"rounded_extended_recentered_{mask_path.name}")
        expected_non_zero = np.count_nonzero(itk.GetArrayViewFromImage(mask))
        number_non_zero = np.count_nonzero(itk.GetArrayViewFromImage(recentered_image))
        number_non_zero_extended = np.count_nonzero(
            itk.GetArrayViewFromImage(extended_recentered_image)
        )
        number_non_zero_rounded = np.count_nonzero(
            itk.GetArrayViewFromImage(rounded_extended_recentered_image)
        )
        # assert number_non_zero_extended == number_non_zero
        # assert number_non_zero_extended == np.count_nonzero(itk.GetArrayViewFromImage(rounded_extended_recentered_image))
        # assert number_non_zero == np.count_nonzero(itk.GetArrayViewFromImage(mask))
        # print(f"Number of non zero pixels in recentered image: {number_non_zero}")
        # print(f"Number of non zero pixels in extended recentered image: {number_non_zero_extended}")

        df = df._append(
            {
                "mask_name": mask_path.name,
                "x_extent": x_extent,
                "y_extent": y_extent,
                "z_extent": z_extent,
                "x_extended": x_extended,
                "y_extended": y_extended,
                "z_extended": z_extended,
                "x_rounded": x_rounded,
                "y_rounded": y_rounded,
                "z_rounded": z_rounded,
                "physical_rounded_size_x": physical_rounded_size_x,
                "physical_rounded_size_y": physical_rounded_size_y,
                "physical_rounded_size_z": physical_rounded_size_z,
                "expected_number_non_zero": expected_non_zero,
                "number_non_zero": number_non_zero,
                "number_non_zero_extended": number_non_zero_extended,
                "number_non_zero_rounded": number_non_zero_rounded,
            },
            ignore_index=True,
        )
        print(f"Bounding Box: {bounding_box}")

        print(f"Center of Mass for {mask_path.name}: {center_of_mass}")
        # TODO Fix this and verify
        # Unsure why this is not working but can skip for now

        # break
    # Largest possible size for prostate given a 10% buffer itkSize3 ([224, 192, 32])

    df.to_csv("largest_region_info.csv", index=False)


def create_blank_image(spacing, desired_size, origin, PIXEL_TYPE):
    IMAGE_TYPE = itk.Image[PIXEL_TYPE, 3]
    # Get the region from the bounding box
    region = itk.ImageRegion[3]()

    size = itk.Size[3]()
    size[0] = desired_size[0]
    size[1] = desired_size[1]
    size[2] = desired_size[2]
    index = itk.Index[3]()
    index[0] = 0
    index[1] = 0
    index[2] = 0
    region.SetSize(size)
    region.SetIndex(index)

    new_centered_image = IMAGE_TYPE.New()
    new_centered_image.SetRegions(region)
    new_centered_image.SetSpacing(spacing)
    new_centered_image.SetOrigin(origin)
    new_centered_image.Allocate()
    return new_centered_image


def pre_process_images(base_data_path: Path):
    # Get the image and mask paths
    image_paths, mask_paths = init_data_lists(base_data_path)
    output_data_path = base_data_path.parent / "preprocessed_data"
    output_data_path.mkdir(exist_ok=True, parents=True)

    # Write the exploratory image info
    # Calculate center of mass
    data_csv = output_data_path / "image_info.csv"
    df = pd.DataFrame()
    for image_path, mask_path in zip(image_paths, mask_paths):
        subject_name = image_path.name.split(".")[0]
        image = itk.imread(str(image_path), itk.F)
        mask = itk.imread(str(mask_path), itk.UC)
        resampled_image, resampled_mask = resample_images_for_training(image, mask)
        resampled_normalized_image = normalize_t2w_images(resampled_image)
        resampled_output_path = (
                output_data_path
                / subject_name
                / f"{subject_name}_resampled_normalized_t2w.nii.gz"
        )
        resampled_mask_path = (
                output_data_path
                / subject_name
                / f"{subject_name}_resampled_segmentations.nii.gz"
        )
        assert np.unique(itk.GetArrayViewFromImage(resampled_mask)).size == 5
        resampled_output_path.parent.mkdir(exist_ok=True, parents=True)
        # itk.imwrite(resampled_normalized_image, resampled_output_path)
        # itk.imwrite(resampled_mask, resampled_mask_path)
        # TODO: Normalize the images to have a mean of 0 and a standard deviation of 1 for t2W
        df = df._append(
            {
                "subject_name": subject_name,
                "original_image_path": image_path,
                "original_mask_path": mask_path,
                "preprocessed_image_path": resampled_output_path,
                "preprocessed_mask_path": resampled_mask_path,
            },
            ignore_index=True,
        )

    df.to_csv(data_csv, index=False)


def normalize_t2w_images(image: itk.Image):
    image_type = itk.Image[itk.F, 3]
    # Initialize the RescaleIntensityImageFilter
    normalizer = itk.RescaleIntensityImageFilter[image_type, image_type].New()
    normalizer.SetInput(image)
    normalizer.SetOutputMinimum(-1)
    normalizer.SetOutputMaximum(1)
    normalizer.Update()
    return normalizer.GetOutput()


def resample_images_for_training(image: itk.Image, mask: itk.Image):
    x_y_size = 288  # Index
    z_size = 16  # Index
    x_y_fov = 140  # mm
    z_fov = 70  # mm

    IMAGE_PIXEL_TYPE = itk.F
    MASK_PIXEL_TYPE = itk.UC
    IMAGE_TYPE = itk.Image[itk.F, 3]
    MASK_TYPE = itk.Image[itk.UC, 3]
    # Get the center of mass of the mask
    center_of_mass = find_center_of_gravity_in_index_space(mask)

    new_size = [x_y_size, x_y_size, z_size]
    # Get the new in plane spacing for the recentered image
    new_in_plane_spacing = x_y_fov / x_y_size
    new_out_of_plane_spacing = z_fov / z_size
    new_spacing = [new_in_plane_spacing, new_in_plane_spacing, new_out_of_plane_spacing]

    center_of_mass_in_physical_space = mask.TransformContinuousIndexToPhysicalPoint(
        center_of_mass
    )
    print(f"Center of Mass in Physical Space: {center_of_mass_in_physical_space}")
    new_origin = calculate_new_origin_from_center_of_mass(
        center_of_mass_in_physical_space,
        new_spacing,
        new_size,
    )
    new_mask_blank = create_blank_image(
        new_spacing, new_size, new_origin, MASK_PIXEL_TYPE
    )
    new_image_blank = create_blank_image(
        new_spacing, new_size, new_origin, IMAGE_PIXEL_TYPE
    )
    print(
        f"New Origin: {new_origin}, New Size: {new_size}, New In Plane Spacing: {new_in_plane_spacing}"
    )
    # Get the new spacing for the recentered image

    nearest_neighbor_interpolator = itk.NearestNeighborInterpolateImageFunction[
        MASK_TYPE, itk.D
    ].New()
    linear_interpolator = itk.LinearInterpolateImageFunction[IMAGE_TYPE, itk.D].New()
    identity_transform = itk.IdentityTransform[itk.D, 3].New()

    # Initialize the ResampleImageFilter
    itk_t2w_resampler = itk.ResampleImageFilter[IMAGE_TYPE, IMAGE_TYPE].New()
    itk_t2w_resampler.SetInput(image)
    itk_t2w_resampler.SetReferenceImage(new_image_blank)
    itk_t2w_resampler.SetTransform(identity_transform)
    itk_t2w_resampler.UseReferenceImageOn()
    itk_t2w_resampler.UpdateLargestPossibleRegion()
    itk_t2w_resampler.SetInterpolator(linear_interpolator)
    itk_t2w_resampler.Update()
    resampled_image = itk_t2w_resampler.GetOutput()

    itk_mask_resampler = itk.ResampleImageFilter[MASK_TYPE, MASK_TYPE].New()
    itk_mask_resampler.SetInput(mask)
    itk_mask_resampler.SetReferenceImage(new_mask_blank)
    itk_mask_resampler.SetTransform(identity_transform)
    itk_mask_resampler.UseReferenceImageOn()
    itk_mask_resampler.UpdateLargestPossibleRegion()
    itk_mask_resampler.SetInterpolator(nearest_neighbor_interpolator)
    itk_mask_resampler.Update()
    resampled_mask = itk_mask_resampler.GetOutput()

    return resampled_image, resampled_mask


def get_label_percentage(image_path, label_path):
    # Paths to your image and label files
    # image_path = "/home/jsome/PycharmProjects/AML/DATA/SortedProstateData/ProstateX-0004/ProstateX-0004_prostate.nii.gz"
    #
    # # Load NIfTI segmentation label
    # label_path = ("/home/jsome/PycharmProjects/AML/DATA/SortedProstateData/ProstateX-0004/ProstateX-0004_segmentation"
    #               ".nii.gz")


    # Load image and label data
    image_data = nib.load(image_path).get_fdata()
    label_data = nib.load(label_path).get_fdata()
    weight_dict = {}
    # Define custom colormap for segmentation labels
    # Map the label values to colors
    label_colors = ListedColormap([
        'black',  # Background (0)
        'red',  # Peripheral Zone (1)
        'green',  # Transition Zone (2)
        'blue',  # Fibromuscular Stroma (3)
        'yellow',  # Distal Prostatic Urethra (4)
    ])

    # Label names mapped to their respective numbers
    label_names = {
        0: 'Background',
        1: 'Peripheral Zone',
        2: 'Transition Zone',
        3: 'Fibromuscular Stroma',
        4: 'Distal Prostatic Urethra'
    }

    # Calculate the total number of pixels per slice (for 2D image slice)
    total_pixels_per_slice = label_data.shape[0] * label_data.shape[1]

    # Calculate the percentage of each label
    unique, counts = np.unique(label_data[:, :, image_data.shape[2] // 2], return_counts=True)
    label_percentages = {label_names[label]: (count / total_pixels_per_slice) * 100 for label, count in
                         zip(unique, counts)}

    # Print out the percentages with label names
    print("Percentage of each label in the central slice:")
    for label_name, percentage in label_percentages.items():
        print(f"{label_name}: {percentage:.2f}%")
        weight_dict[label_name] = percentage
    return weight_dict
def weight_labels(data_path):

    image_path: str =""
    label_path: str =""
    weights = defaultdict(lambda : 0.0)
    subject_count = 0
    for filename in os.listdir(data_path):
        subject = os.path.join(data_path, filename)
        #check if is directory
        if os.path.isdir(subject):
            for file in os.listdir(subject):
                if file.endswith("segmentation.nii.gz"):
                    #grab label info
                    label_path = os.path.join(subject, file)
                if file.endswith("prostate.nii.gz"):
                    #grab image info
                    image_path = os.path.join(subject, file)
            subject_weights = get_label_percentage(image_path,label_path)
            subject_count += 1
            for key in subject_weights.keys():
                weights[key] += subject_weights[key]
    for key in weights.keys():
        weights[key] = weights[key]/subject_count
    print(weights)





if __name__ == "__main__":
    # base_data_path = Path(
    #     "/Users/iejohnson/School/spring_2024/AML/Supervised_learning/DATA/SortedProstateData"
    # )
    base_data_path = Path(
        os.environ.get("SortedProstateData")
    )
    # find_largest_needed_region_for_prostate(base_data_path)
    # compare_average_center_of_mass(base_data_path)
    pre_process_images(base_data_path)
    print("Finshed")
    weight_labels(base_data_path)
