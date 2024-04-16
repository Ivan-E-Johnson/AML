from pathlib import Path

import itk
import numpy as np
from botimageai.preprocessing.random_selection_mask import (
    make_nonzero_mask_image,
    make_random_sampling_mask,
    make_dilated_mask_region,
    inject_mask_into_segmentation,
)
from dcm_classifier.dicom_volume import DicomSingleVolumeInfoBase

from itk_preprocessing import (
    get_initial_reportable_image_stats,
    convert_multiclass_mask_to_binary,
    resample_images_for_training,
    normalize_t2w_images,
)
from botimageai.preprocessing.bayes_optimized_euler3d import (
    run_optimizer_for_mmi_metric,
    Timer,
    _median_filter_test,
    euler3d_human_readable_str,
)


def normalize_image(image):
    median_image = _median_filter_test(image)
    rescaled_image = normalize_t2w_images(median_image)
    return rescaled_image


def find_renamed_images(sorted_prostate_path: Path, bi_prostatex_path: Path):
    desired_output_path = Path(
        "/Users/iejohnson/School/spring_2024/AML/Supervised_learning/DATA/bi_prostatex"
    )
    subject_names = [
        x.name for x in sorted_prostate_path.iterdir() if "ProstateX" in x.name
    ]
    converted_names = [convert_standard_name_to_bi_name(x) for x in subject_names]
    for converted_name, actual_name in zip(converted_names, subject_names):
        subject_output_path = desired_output_path / actual_name
        subject_output_path.mkdir(exist_ok=True, parents=True)
        given_data_path = sorted_prostate_path / actual_name
        given_mask_path = list(given_data_path.rglob("*segmentation.nii.gz"))[0]
        given_mask_image = itk.imread(str(given_mask_path), itk.UC)
        given_t2w_path = list(given_data_path.rglob("*prostate.nii.gz"))[0]
        given_t2w_image = itk.imread(str(given_t2w_path), itk.F)

        t2w_path = list((bi_prostatex_path / converted_name).rglob("*.nii.gz"))[0]
        derivatives_path = bi_prostatex_path / "derivatives"
        adc_path = list((derivatives_path / "ADC" / converted_name).rglob("*.nii.gz"))[
            0
        ]
        tracew_path = list(
            (derivatives_path / "bVal" / converted_name).rglob("*.nii.gz")
        )[0]

        t2w_im = itk.imread(str(t2w_path), itk.F)
        adc_im = itk.imread(str(adc_path), itk.F)
        tracew_im = itk.imread(str(tracew_path), itk.F)
        t2w_info = get_initial_reportable_image_stats(t2w_im)
        adc_info = get_initial_reportable_image_stats(adc_im)
        tracew_info = get_initial_reportable_image_stats(tracew_im)
        print(f"T2W Info: {t2w_info}")
        print(f"ADC Info: {adc_info}")
        print(f"TraceW Info: {tracew_info}")
        # run_study_registration(
        #     given_t2w_path, given_mask_path, adc_path, tracew_path, subject_output_path
        # )
        pre_processed_t2w_path = (
            subject_output_path / f"{actual_name}_resampled_normalized_t2w.nii.gz"
        )
        pre_processed_mask_path = (
            subject_output_path / f"{actual_name}_resampled_segmentations.nii.gz"
        )
        resampled_image, resampled_mask = resample_images_for_training(
            given_t2w_image, given_mask_image
        )
        resampled_normalized_image = normalize_image(resampled_image)
        assert np.unique(itk.GetArrayViewFromImage(resampled_mask)).size == 5
        itk.imwrite(resampled_normalized_image, pre_processed_t2w_path)
        itk.imwrite(resampled_mask, pre_processed_mask_path)


def convert_standard_name_to_bi_name(standard_name: str):
    new_name = "sub-PRX"
    study_number = standard_name.split("-")[1]
    return f"{new_name}{study_number}"


def make_fixed_image_mask_region(protate_mask_image, fraction: float):
    random_background_label_code = 25
    random_image_samples = make_random_sampling_mask(
        protate_mask_image, fraction, random_background_label_code
    )
    dilated_mask_region = make_dilated_mask_region(protate_mask_image)
    dilated_mask_region_label_code = 50
    final_mask = inject_mask_into_segmentation(
        random_image_samples, dilated_mask_region, dilated_mask_region_label_code
    )

    protstate_mask_label_code = 100
    final_mask = inject_mask_into_segmentation(
        final_mask, protate_mask_image, protstate_mask_label_code
    )
    return final_mask


def apply_transfrom_to_image(moving_image, transform):
    imageType = type(moving_image)
    OutputImageType = itk.Image[itk.F, 3]
    resampler = itk.ResampleImageFilter[imageType, OutputImageType].New()
    resampler.SetInput(moving_image)
    resampler.SetTransform(transform)
    resampler.SetSize(moving_image.GetLargestPossibleRegion().GetSize())
    resampler.SetOutputOrigin(moving_image.GetOrigin())
    resampler.SetOutputSpacing(moving_image.GetSpacing())
    resampler.SetOutputDirection(moving_image.GetDirection())
    resampler.SetDefaultPixelValue(-1)
    resampler.Update()
    return resampler.GetOutput()


def run_study_registration(
    t2w_orig_path, mask_orig_path, adc_bi_path, tracew_bi_path, output_folder
):
    temp_dir = output_folder / "registration_tempfiles"
    temp_dir.mkdir(parents=True, exist_ok=True)
    study_name = output_folder.name
    for moving_image_path in [adc_bi_path, tracew_bi_path]:
        registration_name = moving_image_path.stem
        moving_to_fixed_tfm_fn = temp_dir / f"{registration_name}_moving_to_fixed.h5"
        with Timer(f"======Run {study_name}  Registration Bayesian Optimizer ======"):

            fixed_image_raw = itk.imread(t2w_orig_path, itk.F)
            moving_image_raw = itk.imread(moving_image_path, itk.F)

            moving_image = _median_filter_test(moving_image_raw)
            fixed_image = _median_filter_test(fixed_image_raw)

            mask_pixel_type = itk.ctype("unsigned char")
            dimension: int = 3
            multiclass_prostate_mask = itk.imread(mask_orig_path, mask_pixel_type)
            protate_mask_image = convert_multiclass_mask_to_binary(
                multiclass_prostate_mask
            )

            prostate_mask_orig_mask = itk.ImageMaskSpatialObject[dimension].New()
            prostate_mask_orig_mask.SetImage(protate_mask_image)
            prostate_mask_orig_mask.Update()
            prostate_mask_centroid = (
                prostate_mask_orig_mask.GetMyBoundingBoxInWorldSpace().GetCenter()
            )
            del prostate_mask_orig_mask

            mi_fixed_mask_image = make_fixed_image_mask_region(protate_mask_image, 1.0)

            fixed_object_mask = itk.ImageMaskSpatialObject[dimension].New()
            fixed_object_mask.SetImage(mi_fixed_mask_image)
            fixed_object_mask.Update()

            non_zero_moving_mask = make_nonzero_mask_image(moving_image)

            moving_object_mask = itk.ImageMaskSpatialObject[dimension].New()
            moving_object_mask.SetImage(non_zero_moving_mask)
            moving_object_mask.Update()

            itk.imwrite(
                mi_fixed_mask_image,
                temp_dir / "mi_fixed_mask_image.nii.gz",
            )
            itk.imwrite(
                non_zero_moving_mask,
                temp_dir / "non_zero_moving_mask.nii.gz",
            )
            itk.imwrite(fixed_image, temp_dir / "fixed_image.nii.gz")
            itk.imwrite(moving_image, temp_dir / "moving_image.nii.gz")
            euler3d_transform = run_optimizer_for_mmi_metric(
                random_state_seed=20,
                initial_number_of_points_to_probe=200,
                number_of_iterations=100,
                registration_name=registration_name,
                fixed_image=fixed_image,
                fixed_object_mask=fixed_object_mask,
                moving_image=moving_image,
                moving_object_mask=moving_object_mask,
                prostate_mask_centroid=prostate_mask_centroid,
                optimization_logs_folder_path=temp_dir,
            )

            tfm_writer = itk.TransformFileWriterTemplate[itk.D].New()
            tfm_writer.SetInput(euler3d_transform)
            tfm_writer.SetFileName(moving_to_fixed_tfm_fn.as_posix())
            tfm_writer.Update()

            print(
                f"{registration_name}: {euler3d_human_readable_str(euler3d_transform)}\n\n\n\n"
            )
            resampled_moving = apply_transfrom_to_image(
                moving_image_raw, euler3d_transform
            )
            pre_processed_moving_image = normalize_image(resampled_moving)
            padded_moving_image, _ = resample_images_for_training(
                pre_processed_moving_image, protate_mask_image
            )
            output_im_path = output_folder / f"{registration_name}_registered.nii.gz"
            itk.imwrite(padded_moving_image, output_im_path.as_posix())


if __name__ == "__main__":
    # base_data_path = Path(
    #     "/Users/iejohnson/School/spring_2024/AML/Supervised_learning/DATA/SortedProstateData"
    # )
    sorted_prostate_path = Path(
        "/Users/iejohnson/School/spring_2024/AML/Supervised_learning/DATA/SortedProstateData"
    )
    # base_secondary_output_path = Path(
    #     "/Users/iejohnson/School/spring_2024/AML/Supervised_learning/DATA/SecondaryData"
    # )
    # suplimentary_data_path = Path(
    #     "/Users/iejohnson/School/spring_2024/AML/Supervised_learning/ProstateX/ProstateX_DICOM/manifest-A3Y4AE4o5818678569166032044/PROSTATEx"
    # )
    bi_prostatex_path = Path(
        "/Users/iejohnson/Botimageai/ImageData/PublicData/PROSTATEx"
    )
    find_renamed_images(sorted_prostate_path, bi_prostatex_path)
