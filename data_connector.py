import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pydicom
import itk
from dcm_classifier.dicom_volume import DicomSingleVolumeInfoBase

class DataConverter:
    def __init__(self, base_data_path):
        # NOTE
        self.input_prostate_data_path = base_data_path / "OrigProstate/PROSTATEx"
        self.segmentation_data_path = base_data_path / "Segmentations/PROSTATEx"
        self.data_dict = {} # todo: change to defaultdict
        self._init_data_dict()


    def __getitem__(self, patient_id):
        return self.data_dict[patient_id]

    def __iter__(self):
        return iter(self.data_dict)

    def __len__(self):
        return len(self.data_dict)


    class SinglePatientData:
        def __init__(self, prostate_dcms, segmentation):
            self.prostate_dcms = prostate_dcms
            self.segmentation_dcm = segmentation

            self._prostate_volume_info_base = DicomSingleVolumeInfoBase(prostate_dcms)
            self._segmentation_volume_info_base = DicomSingleVolumeInfoBase(segmentation)

            self.prostate_volume = self._prostate_volume_info_base.get_itk_image()
            self.segmentation_volume = self._segmentation_volume_info_base.get_itk_image()

        def _ensure_all_data_in_same_space(self):
            self.segmentation_volume = check_and_adjust_image_to_same_space(self.prostate_volume, self.segmentation_volume)

        def get_prostate_volume(self):
            return self.prostate_volume

        def get_segmentation_volume(self):
            return self.segmentation_volume

        def write_prostate_volume(self, output_path):
            itk.imwrite(self.get_prostate_volume(), output_path)

        def write_segmentation_volume(self, output_path):
            itk.imwrite(self.get_segmentation_volume(), output_path)

    def _init_data_dict(self):
        prostate_dirs = [x for x in self.input_prostate_data_path.iterdir() if x.is_dir()]

        # dict of dicts to store the data with key as patient_id and value as a
        # dict containing the prostate and segmentation paths

        for prostate_dir in prostate_dirs:
            print(f"Processing {prostate_dir.name}...")
            patient_id = prostate_dir.name
            patient_segmentation_path = self.segmentation_data_path / patient_id
            patient_data = self.SinglePatientData(list(prostate_dir.rglob("*.dcm")), list(patient_segmentation_path.rglob("*.dcm")))
            self.data_dict[patient_id] = patient_data

    def write_all_prostate_volumes(self, output_base_dir):
        for patient_id in self.data_dict.keys():
            patient_data = self.data_dict[patient_id]
            assert isinstance(patient_data, self.SinglePatientData)
            print(f"Converting {patient_id}...")
            output_dir = output_base_dir / patient_id
            print(f"Writing to {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
            try:
                patient_data.write_prostate_volume(output_dir / f"{patient_id}_prostate.nii.gz")
                patient_data.write_segmentation_volume(output_dir / f"{patient_id}_segmentation.nii.gz")
            except Exception as e:
                print(f"Error writing {patient_id}: {e}")




def get_image_stats(image: itk.Image):
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
    msg: str = ""
    is_in_same_space: bool = True

    if image1.GetImageDimension() != image2.GetImageDimension():
        msg += f"Dimension mismatch: {image1.GetImageDimension()} != {image2.GetImageDimension()}\n"
        is_in_same_space = False

    if not np.allclose(image1.GetSpacing(), image2.GetSpacing(), atol=tolerance):
        msg += f"Spacing mismatch: {image1.GetSpacing()} != {image2.GetSpacing()}\n"
        is_in_same_space = False

    if not np.allclose(image1.GetDirection(), image2.GetDirection(), atol=tolerance):
        msg += f"Direction mismatch: {image1.GetDirection()} != {image2.GetDirection()}\n"
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
    Adjusts target_image's metadata to match reference_image's.
    """

    # Check if images are already in the same space, if not, adjust target_image
    if not check_images_in_same_space(reference_image, target_image, tolerance):
        # Adjust dimension indirectly by ensuring we are using the same type
        # Note: ITK's Python wrapping does not allow changing the dimensionality of an existing image directly

        # Adjust spacing, direction, and origin
        target_image.SetSpacing(reference_image.GetSpacing())
        target_image.SetDirection(reference_image.GetDirection())
        target_image.SetOrigin(reference_image.GetOrigin())

        print("Target image adjusted to match the reference image's space.")
    else:
        print("Images are already in the same space. No adjustments needed.")

    return target_image


# Example usage:
if __name__ == "__main__":
    base_data_path = Path('/Users/iejohnson/School/spring_2024/AML/Supervised_learning/Data')
    output_base_dir = base_data_path / "SortedProstateData"
    test_dicom_path = base_data_path / "OrigProstate/PROSTATEx/ProstateX-0004/10-18-2011-NA-MR prostaat kanker detectie WDSmc MCAPRODETW-45493/5.000000-t2tsetra-75680"

    Converter = DataConverter(base_data_path)
    Converter.write_all_prostate_volumes(base_data_path / "SortedProstateData")
    # test_seg = Converter['ProstateX-0004'].get_segmentation_volume()
    # print(test_seg)
    # test_vol = Converter['ProstateX-0004'].get_prostate_volume()
    # print(test_vol)
    # assert check_images_in_same_space(test_seg, test_vol)
    #
    #
    # test_vol = check_and_adjust_image_to_same_space(test_seg, test_vol)



    # prostate_data_instance = DataConverter(base_data_path)



# Now you can access the loaded DICOM files using prostate_data_instance.prostate_dcms
# and prostate_data_instance.segmentation_dcm
