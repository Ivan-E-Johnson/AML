import tempfile

import botimageai
from pathlib import Path
import itk
import numpy as np
import pydicom
import dicom2nifti

from botimageai.dicom_processing.process_one_dicom_study_to_volumes_mapping import (
    ProstatIDDicomStudyToVolumesMapping,
)
from dcm_classifier.dicom_volume import DicomSingleVolumeInfoBase
from dcm_classifier.utility_functions import itk_read_from_dicomfn_list

from data_connector import (
    check_images_in_same_space,
    check_and_adjust_image_to_same_space,
)
from itk_preprocessing import resample_image_to_reference


class SinglePatientWithSegmentation:
    """
    A class used to manage single patient data.

    ...

    Attributes
    ----------
    prostate_dcms : list
        a list of paths to the prostate dicom files
    segmentation_dcm : list
        a list of paths to the segmentation dicom files
    prostate_volume : itk.Image
        an itk.Image object that represents the prostate volume
    orig_segmentation_volume : itk.Image
        an itk.Image object that represents the segmentation volume

    Methods
    -------
    _ensure_all_data_in_same_space():
        Ensures that the prostate volume and segmentation volume are in the same space.
    get_prostate_volume():
        Returns the prostate volume.
    get_segmentation_volume():
        Returns the segmentation volume.
    write_prostate_volume(output_path):
        Writes the prostate volume to the specified path.
    write_segmentation_volume(output_path):
        Writes the segmentation volume to the specified path.
    """

    def __init__(self, prostate_volume, segmentation):
        """
        Constructs all the necessary attributes for the SinglePatientData object.

        Parameters
        ----------
            prostate_volume : list
                a list of paths to the prostate dicom files
            segmentation : list
                a list of paths to the segmentation dicom files
        """
        self.segmentation_dcm = list(segmentation)

        self.segmentation_volume = None
        self.prostate_volume = prostate_volume
        self.orig_segmentation_volume = itk_read_from_dicomfn_list(
            self.segmentation_dcm
        )
        self.segmentation_list = []
        self._split_segmentation()
        self._squash_segmentations_with_one_hot()

    def _squash_segmentations_with_one_hot(self):
        """
        Squashes the 4 segmentations into a single one-hot encoded segmentation.
        """
        # squash the segmentations into a single one-hot encoded segmentation
        # The four-class segmentation encompasses the PZ, TZ, AFMS (anterior fibromuscular stroma), and the urethra
        # 0: background, 1: peripheral zone, 2: transition zone, 3: urethra , 4 AFMS
        # The original segmentation has a Z dimension that is 4 times the prostate volume
        squashed_seg_arr = np.zeros_like(itk.GetArrayFromImage(self.prostate_volume))
        for i in range(4):
            seg_array = itk.GetArrayFromImage(self.segmentation_list[i])
            squashed_seg_arr += (i + 1) * seg_array / 255
        squashed_seg = itk.GetImageFromArray(squashed_seg_arr)
        squashed_seg.CopyInformation(self.prostate_volume)
        self.segmentation_volume = squashed_seg

    def _split_segmentation(self):
        """
        Splits the segmentation volume into its 4 components.
        From manual inspection the segmentation has a Z dimension that is 4 times the prostate volume.

        """
        desired_shape = self.prostate_volume.GetLargestPossibleRegion().GetSize()
        segmentation_array = itk.GetArrayFromImage(self.orig_segmentation_volume)
        for i in range(4):
            seg_array = segmentation_array[
                i * desired_shape[2] : (i + 1) * desired_shape[2], :, :
            ]
            seg_image = itk.GetImageFromArray(seg_array)
            seg_image.CopyInformation(self.prostate_volume)
            self.segmentation_list.append(seg_image)

    def _ensure_all_data_in_same_space(self):
        """
        Ensures that the prostate volume and segmentation volume are in the same space.
        """
        self.segmentation_volume = check_and_adjust_image_to_same_space(
            self.prostate_volume, self.segmentation_volume
        )

    def get_prostate_volume(self):
        """
        Returns the prostate volume.
        """
        return self.prostate_volume

    def get_segmentation_volume(self):
        """
        Returns the segmentation volume.
        """
        return self.segmentation_volume

    def write_prostate_volume(self, output_path):
        """
        Writes the prostate volume to the specified path.

        Parameters
        ----------
            output_path : Path
                a Path object that represents the path where the prostate volume will be written
        """
        itk.imwrite(self.get_prostate_volume(), output_path)

    def write_segmentation_volume(self, output_path: Path):
        """
        Writes the segmentation volume to the specified path.

        Parameters
        ----------
            output_path : Path
                a Path object that represents the path where the segmentation volume will be written
        """
        itk.imwrite(self.segmentation_volume, output_path.as_posix())


if __name__ == "__main__":
    prostatX_data = Path(
        "/Users/iejohnson/School/spring_2024/AML/Supervised_learning/ProstateX/ProstateX_DICOM/manifest-A3Y4AE4o5818678569166032044/PROSTATEx"
    )
    prostatX_segmentation_data = Path(
        "/Users/iejohnson/School/spring_2024/AML/Supervised_learning/DATA/Segmentations/PROSTATEx"
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
        image_output_dir = default_output_dir / subject_dir.name
        if not has_corresponding_segmentation:
            continue
        if has_corresponding_segmentation:
            segmentation_dicom_dir = prostatX_segmentation_data / subject_dir.name
            image_output_dir = with_segmentation_output_dir / subject_dir.name
            image_output_dir.mkdir(parents=True, exist_ok=True)
            dcm_file_list = list(segmentation_dicom_dir.rglob("*.dcm"))

            segmentation_output_file = (
                image_output_dir / f"{subject_dir.name}_segmentation.nii.gz"
            )
            if segmentation_output_file.exists():
                segmentation_output_file.unlink()
            segmentation_data = SinglePatientWithSegmentation(
                prostate_volume=best_images["t2w"],
                segmentation=dcm_file_list,
            )

            segmentation_data.write_segmentation_volume(segmentation_output_file)

            print(f"Segmentation output file: {segmentation_output_file}")
        image_output_dir.mkdir(parents=True, exist_ok=True)
        for image_name, image in best_images.items():
            # print(f"Image name: {image_name}")
            # print(f"Image: {image}")
            output_file = image_output_dir / f"{subject_dir.name}_{image_name}.nii.gz"
            itk.imwrite(image, output_file)
