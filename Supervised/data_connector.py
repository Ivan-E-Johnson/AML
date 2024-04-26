import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import itk
from dcm_classifier.dicom_volume import (
    DicomSingleVolumeInfoBase,
    itk_read_from_dicomfn_list,
)


class DataConverter:
    """
    A class used to convert and manage prostate data.

    ...

    Attributes
    ----------
    input_prostate_data_path : Path
        a Path object that represents the directory where the original prostate data is stored
    segmentation_data_path : Path
        a Path object that represents the directory where the segmentation data is stored
    data_dict : dict
        a dictionary that stores the patient data with patient_id as key

    Methods
    -------
    __getitem__(patient_id):
        Returns the patient data for the given patient_id.
    __iter__():
        Returns an iterator for the data dictionary.
    __len__():
        Returns the number of patients in the data dictionary.
    _init_data_dict():
        Initializes the data dictionary with patient data.
    write_all_prostate_volumes(output_base_dir):
        Writes all prostate volumes to the specified directory.
    """

    def __init__(self, base_data_path):
        """
        Constructs all the necessary attributes for the DataConverter object.

        Parameters
        ----------
            base_data_path : Path
                a Path object that represents the base directory where the data is stored
        """
        self.input_prostate_data_path = base_data_path / "OrigProstate/PROSTATEx"
        self.segmentation_data_path = base_data_path / "Segmentations/PROSTATEx"
        self.data_dict = {}  # todo: change to defaultdict
        self._init_data_dict()
        self.data_df = pd.DataFrame(
            columns=[
                "PatientID",
                "Size",
                "Spacing",
                "Origin",
                "Direction",
                "UniqueValues",
                "Stats",
            ]
        )

    def __getitem__(self, patient_id):
        """
        Returns the patient data for the given patient_id.

        Parameters
        ----------
            patient_id : str
                the id of the patient
        """
        return self.data_dict[patient_id]

    def __iter__(self):
        """
        Returns an iterator for the data dictionary.
        """
        return iter(self.data_dict)

    def __len__(self):
        """
        Returns the number of patients in the data dictionary.
        """
        return len(self.data_dict)

    class SinglePatientData:
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

        def __init__(self, prostate_dcms, segmentation):
            """
            Constructs all the necessary attributes for the SinglePatientData object.

            Parameters
            ----------
                prostate_dcms : list
                    a list of paths to the prostate dicom files
                segmentation : list
                    a list of paths to the segmentation dicom files
            """
            self.prostate_dcms = list(prostate_dcms)
            self.segmentation_dcm = list(segmentation)

            self.segmentation_volume = None
            self.prostate_volume = itk_read_from_dicomfn_list(self.prostate_dcms)
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
            squashed_seg_arr = np.zeros_like(
                itk.GetArrayFromImage(self.prostate_volume)
            )
            for i in range(4):
                seg_array = itk.GetArrayFromImage(self.segmentation_list[i])
                squashed_seg_arr += (i + 1) * seg_array / 255
            squashed_seg = itk.GetImageFromArray(squashed_seg_arr)
            squashed_seg.SetSpacing(self.prostate_volume.GetSpacing())
            squashed_seg.SetDirection(self.prostate_volume.GetDirection())
            squashed_seg.SetOrigin(self.prostate_volume.GetOrigin())
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
                seg_image.SetSpacing(self.prostate_volume.GetSpacing())
                seg_image.SetDirection(self.prostate_volume.GetDirection())
                seg_image.SetOrigin(self.prostate_volume.GetOrigin())
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

        def write_segmentation_volume(self, output_path):
            """
            Writes the segmentation volume to the specified path.

            Parameters
            ----------
                output_path : Path
                    a Path object that represents the path where the segmentation volume will be written
            """
            itk.imwrite(self.segmentation_volume, output_path)
            # for i in range(len(self.segmentation_list)):
            #     itk.imwrite(self.segmentation_list[i], output_path.parent / f"{output_path.stem}_{i}.nii.gz")

        # TODO Add functionality to write out a "rough" segmentation of the prostate which is just all the masks combined

    def _init_data_dict(self):
        """
        Initializes the data dictionary with patient data.
        """
        prostate_dirs = [
            x for x in self.input_prostate_data_path.iterdir() if x.is_dir()
        ]

        # dict of dicts to store the data with key as patient_id and value as a
        # dict containing the prostate and segmentation paths

        for prostate_dir in prostate_dirs:
            print(f"Processing {prostate_dir.name}...")
            patient_id = prostate_dir.name
            patient_segmentation_path = self.segmentation_data_path / patient_id
            patient_data = self.SinglePatientData(
                list(prostate_dir.rglob("*.dcm")),
                list(patient_segmentation_path.rglob("*.dcm")),
            )
            self.data_dict[patient_id] = patient_data

    def write_all_prostate_volumes(self, output_base_dir):
        """
        Writes all prostate volumes to the specified directory.

        Parameters
        ----------
            output_base_dir : Path
                a Path object that represents the directory where the prostate volumes will be written
        """
        for patient_id in self.data_dict.keys():
            patient_data = self.data_dict[patient_id]

            # self.data_df = pd.concat([self.data_df, pd.DataFrame(get_exploratory_image_info(patient_data.get_prostate_volume()))], ignore_index=True)
            output_dir = output_base_dir / patient_id
            print(f"Writing to {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
            try:
                patient_data.write_prostate_volume(
                    output_dir / f"{patient_id}_prostate.nii.gz"
                )
                patient_data.write_segmentation_volume(
                    output_dir / f"{patient_id}_segmentation.nii.gz"
                )
            except Exception as e:
                print(f"Error writing {patient_id}: {e}")
        # self.data_df.to_csv(output_base_dir / "data_collection.csv", index=False)


def get_image_stats(image: itk.Image):
    """
    Returns the statistics of the given image.

    Parameters
    ----------
        image : itk.Image
            an itk.Image object

    Returns
    -------
        mean : float
            the mean intensity of the image
        std : float
            the standard deviation of the image intensities
        max : float
            the maximum intensity of the image
        min : float
            the minimum intensity of the image
    """
    stats = itk.StatisticsImageFilter.New(image)
    stats.SetInput(image)
    stats.Update()
    mean = stats.GetMean()
    std = stats.GetSigma()
    max = stats.GetMaximum()
    min = stats.GetMinimum()
    return mean, std, max, min


def check_images_in_same_space(
    image1: itk.Image, image2: itk.Image, tolerance: float = 1e-5
) -> bool:
    """
    Checks if two images are in the same space.

    Parameters
    ----------
        image1 : itk.Image
            the first image
        image2 : itk.Image
            the second image
        tolerance : float, optional
            the tolerance for the comparison (default is 1e-5)

    Returns
    -------
        bool
            True if the images are in the same space, False otherwise
    """
    msg: str = ""
    is_in_same_space: bool = True

    if image1.GetImageDimension() != image2.GetImageDimension():
        msg += f"Dimension mismatch: {image1.GetImageDimension()} != {image2.GetImageDimension()}\n"
        is_in_same_space = False

    if not np.allclose(image1.GetSpacing(), image2.GetSpacing(), atol=tolerance):
        msg += f"Spacing mismatch: {image1.GetSpacing()} != {image2.GetSpacing()}\n"
        is_in_same_space = False

    if not np.allclose(image1.GetDirection(), image2.GetDirection(), atol=tolerance):
        msg += (
            f"Direction mismatch: {image1.GetDirection()} != {image2.GetDirection()}\n"
        )
        is_in_same_space = False

    if not np.allclose(image1.GetOrigin(), image2.GetOrigin(), atol=tolerance):
        msg += f"Origin mismatch: {image1.GetOrigin()} != {image2.GetOrigin()}\n"
        is_in_same_space = False

    print(msg)
    return is_in_same_space


def check_and_adjust_image_to_same_space(
    reference_image: itk.Image, target_image: itk.Image, tolerance: float = 1e-5
) -> itk.Image:
    """
    Adjusts the target image's metadata to match the reference image's.

    Parameters
    ----------
        reference_image : itk.Image
            the reference image
        target_image : itk.Image
            the target image that will be adjusted
        tolerance : float, optional
            the tolerance for the comparison (default is 1e-5)

    Returns
    -------
        itk.Image
            the adjusted target image
    """
    if not check_images_in_same_space(reference_image, target_image, tolerance):
        # Adjust dimension indirectly by ensuring we are using the same type
        # Note: ITK's Python wrapping does not allow changing the dimensionality of an existing image directly

        # Adjust spacing, direction, and origin
        target_image.SetSpacing(reference_image.GetSpacing())
        target_image.SetDirection(reference_image.GetDirection())
        target_image.SetOrigin(reference_image.GetOrigin())

    return target_image


def get_exploratory_image_info(image: itk.Image):
    """
    Returns the exploratory information of the given image.

    Parameters
    ----------
        image : itk.Image
            an itk.Image object

    Returns
    -------
        dict
            a dictionary containing the exploratory information of the image
    """
    exploratory_info = {}
    exploratory_info["Size"] = image.GetLargestPossibleRegion().GetSize()
    exploratory_info["Spacing"] = image.GetSpacing()
    exploratory_info["Origin"] = image.GetOrigin()
    exploratory_info["Direction"] = image.GetDirection()
    exploratory_info["UniqueValues"] = np.unique(itk.GetArrayFromImage(image))
    exploratory_info["Stats"] = get_image_stats(image)
    print(exploratory_info)
    return exploratory_info


if __name__ == "__main__":
    base_data_path = Path(
        "/Users/iejohnson/School/spring_2024/AML/Supervised_learning/Data"
    )
    output_base_dir = base_data_path / "SortedProstateData"

    # test_segmentation_dcms = list(
    #     (base_data_path / "Segmentations/PROSTATEx/ProstateX-0004/").rglob("*.dcm")
    # )
    # test_prostate_dcms = list(
    #     (base_data_path / "OrigProstate/PROSTATEx/ProstateX-0004/").rglob("*.dcm")
    # )
    # test_data = DataConverter.SinglePatientData(test_prostate_dcms, test_segmentation_dcms)
    # test_prostate = test_data.get_prostate_volume()
    # test_segmentation = test_data.get_segmentation_volume()
    # print(f"Prostate volume: {test_data.get_prostate_volume()}")
    # print(f"Segmentation volume: {test_data.get_segmentation_volume()}")
    Converter = DataConverter(base_data_path)
    Converter.write_all_prostate_volumes(base_data_path / "SortedProstateData")
