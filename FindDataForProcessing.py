import tempfile

import botimageai
from pathlib import Path
import itk
import pydicom
import dicom2nifti

from botimageai.dicom_processing.process_one_dicom_study_to_volumes_mapping import (
    ProstatIDDicomStudyToVolumesMapping,
)
from dcm_classifier.utility_functions import itk_read_from_dicomfn_list
from itk_preprocessing import resample_image_to_reference


if __name__ == "__main__":
    prostatX_data = Path(
        "/Users/iejohnson/School/spring_2024/AML/Supervised_learning/ProstateX/ProstateX_DICOM/manifest-A3Y4AE4o5818678569166032044/PROSTATEx"
    )
    prostatX_segmentation_data = Path(
        "/Users/iejohnson/School/spring_2024/AML/Supervised_learning/ProstateX/ProstateX_DICOM/manifest-1605042674814/PROSTATEx"
    )
    default_output_dir = Path(
        "/Users/iejohnson/School/spring_2024/AML/Supervised_learning/DATA/ALL_PROSTATEx/WITHOUT_SEGMENTATION/RAW"
    )
    with_segmentation_output_dir = Path(
        "/Users/iejohnson/School/spring_2024/AML/Supervised_learning/DATA/ALL_PROSTATEx/WITH_SEGMENTATION/RAW"
    )
    default_output_dir.mkdir(parents=True, exist_ok=True)
    with_segmentation_output_dir.mkdir(parents=True, exist_ok=True)

    subject_dirs = [
        x for x in prostatX_data.iterdir() if x.is_dir() and "ProstateX" in x.name
    ]
    segmentation_dirs = [
        x
        for x in prostatX_segmentation_data.iterdir()
        if x.is_dir() and "ProstateX" in x.name
    ]
    segmentation_names = [x.name for x in segmentation_dirs]
    subject_names = [x.name for x in subject_dirs]

    for subject_dir in subject_dirs:
        has_corresponding_segmentation = subject_dir.name in segmentation_names
        # Check if the segmentation data exists

        try:
            volume_mapping = ProstatIDDicomStudyToVolumesMapping(subject_dir)
        except Exception as e:
            print(f"Error processing {subject_dir.name}: {e}")
            continue
        best_images: dict[str, itk.Image[itk.F, 3]] = (
            volume_mapping.get_best_inputs_images()
        )
        image_output_dir = default_output_dir / "RAW" / subject_dir.name

        if has_corresponding_segmentation:
            segmentation_dicom_dir = prostatX_segmentation_data / subject_dir.name
            image_output_dir = with_segmentation_output_dir / "RAW" / subject_dir.name
            dcm_file_list = list(segmentation_dicom_dir.rglob("*.dcm"))
            itk_segmentation = itk_read_from_dicomfn_list(dcm_file_list)
            segmentation_output_file = (
                image_output_dir / f"{subject_dir.name}_segmentation.nii.gz"
            )
            itk_segmentation = resample_image_to_reference(
                image=itk_segmentation, reference_image=best_images.get("t2w")
            )
            image_output_dir.mkdir(parents=True, exist_ok=True)
            itk.imwrite(itk_segmentation, segmentation_output_file)

            print(f"Segmentation output file: {segmentation_output_file}")
        image_output_dir.mkdir(parents=True, exist_ok=True)
        for image_name, image in best_images.items():
            # print(f"Image name: {image_name}")
            # print(f"Image: {image}")
            output_file = image_output_dir / f"{subject_dir.name}_{image_name}.nii.gz"
            itk.imwrite(image, output_file)
