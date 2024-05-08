from pathlib import Path
import itk
import numpy as np
from tqdm import tqdm

from Supervised.itk_preprocessing import (
    find_center_of_gravity_in_index_space,
    calculate_new_origin_from_center_of_mass,
    create_blank_image,
)


class SingleStudyStackedDataBase:
    """
    A class used to manage single study data.
    """
    x_y_size = 320  # Index
    z_size = 32  # Index
    x_y_fov = 160  # mm
    z_fov = 96  # mm
    IMAGE_PIXEL_TYPE = itk.F
    MASK_PIXEL_TYPE = itk.UC
    IMAGE_TYPE = itk.Image[IMAGE_PIXEL_TYPE, 3]
    MASK_TYPE = itk.Image[MASK_PIXEL_TYPE, 3]
    VECTOR_IMAGE_TYPE = itk.VectorImage[IMAGE_PIXEL_TYPE, 3]

    def __init__(self, study_path: Path):
        # Initialize the paths to the images
        self.study_path = study_path
        self.t2w_path = study_path / f"{study_path.name}_t2w.nii.gz"
        self.adc_path = study_path / f"{study_path.name}_adc.nii.gz"
        self.blow = study_path / f"{study_path.name}_b0low.nii.gz"
        self.tracew = study_path / f"{study_path.name}_tracew.nii.gz"

    def load_images(self):
        """
        Loads the images from the paths.
        Returns
        -------
        None
        """
        self.t2w = itk.imread(self.t2w_path.as_posix())
        self.adc = itk.imread(self.adc_path.as_posix())
        self.blow = itk.imread(self.blow.as_posix())
        self.tracew = itk.imread(self.tracew.as_posix())

    def get_t2w_image(self):
        """
        Returns the T2W image.
        -------
        """
        return self.t2w

    def get_adc_image(self):
        """
        Returns the ADC image.
        -------

        """
        return self.adc

    def get_blow_image(self):
        """
        Returns the BLOW image.
        -------

        """
        return self.blow

    def get_tracew_image(self):
        """
        Returns the TRACEW image.
        -------

        """
        return self.tracew

    def _get_all_resampled_images(self):
        """
        Resamples all the images for training.
]        -------

        """
        resampled_t2w = self._resample_images_for_training(self.get_t2w_image())
        resampled_adc = self._resample_images_for_training(self.get_adc_image())
        resampled_blow = self._resample_images_for_training(self.get_blow_image())
        resampled_tracew = self._resample_images_for_training(self.get_tracew_image())
        return resampled_t2w, resampled_adc, resampled_blow, resampled_tracew

    def resample_coresponding_label(self, label: itk.Image[itk.UC, 3]):
        """
        Resamples the label for training.
        Parameters
        ----------
        label

        Returns
        -------
        resampled_label
        """
        orig_values, orig_counts = np.unique(
            itk.GetArrayFromImage(label), return_counts=True
        )
        resampled_label = self._resample_images_for_training(label, is_mask=True)
        resampled_values, resampled_counts = np.unique(
            itk.GetArrayFromImage(resampled_label), return_counts=True
        )
        print(f"Original Values: {orig_values}, Original Counts: {orig_counts}")
        print(
            f"Resampled Values: {resampled_values}, Resampled Counts: {resampled_counts}"
        )
        return resampled_label

    def _resample_images_for_training(
        self, image_to_resample: itk.Image[itk.F, 3], is_mask=False
    ):
        """
        Resamples the image for training.
        Parameters
        ----------
        image_to_resample
        is_mask

        Returns
        -------
        resampled_image
        """
        # Get the center of mass of the mask
        center_of_mass = find_center_of_gravity_in_index_space(self.t2w)

        new_size = [self.x_y_size, self.x_y_size, self.z_size]
        # Get the new in plane spacing for the recentered image
        new_in_plane_spacing = self.x_y_fov / self.x_y_size
        new_out_of_plane_spacing = self.z_fov / self.z_size
        new_spacing = [
            new_in_plane_spacing,
            new_in_plane_spacing,
            new_out_of_plane_spacing,
        ]

        center_of_mass_in_physical_space = (
            self.get_t2w_image().TransformContinuousIndexToPhysicalPoint(center_of_mass)
        )
        print(f"Center of Mass in Physical Space: {center_of_mass_in_physical_space}")
        new_origin = calculate_new_origin_from_center_of_mass(
            center_of_mass_in_physical_space,
            new_spacing,
            new_size,
        )

        new_image_blank = create_blank_image(
            new_spacing, new_size, new_origin, self.IMAGE_PIXEL_TYPE
        )
        print(
            f"New Origin: {new_origin}, New Size: {new_size}, New In Plane Spacing: {new_in_plane_spacing}, New Out of Plane Spacing: {new_out_of_plane_spacing}"
        )
        # Get the new spacing for the recentered image

        interpolator = itk.LinearInterpolateImageFunction[self.IMAGE_TYPE, itk.D].New()
        if is_mask:
            interpolator = itk.NearestNeighborInterpolateImageFunction[
                self.IMAGE_TYPE, itk.D
            ].New()
        identity_transform = itk.IdentityTransform[itk.D, 3].New()
        # Initialize the ResampleImageFilter
        itk_t2w_resampler = itk.ResampleImageFilter[
            self.IMAGE_TYPE, self.IMAGE_TYPE
        ].New()
        itk_t2w_resampler.SetInput(image_to_resample)
        itk_t2w_resampler.SetReferenceImage(new_image_blank)
        itk_t2w_resampler.SetTransform(identity_transform)
        itk_t2w_resampler.UseReferenceImageOn()
        itk_t2w_resampler.UpdateLargestPossibleRegion()
        itk_t2w_resampler.SetInterpolator(interpolator)
        itk_t2w_resampler.Update()
        resampled_image = itk_t2w_resampler.GetOutput()
        return resampled_image

    def _fill_outliers_with_median(
        self, image: itk.Image[itk.F, 3], radius: list[int, int, int]
    ):
        """
        Fills the outliers with the median.
        Parameters
        ----------
        image
        radius

        Returns
        -------
        multiply_filter.GetOutput()
        """
        lower_percentile = 0.1
        upper_percentile = 0.99
        # Get the lower and upper thresholds
        lower_threshold = np.quantile(itk.GetArrayFromImage(image), lower_percentile)
        upper_threshold = np.quantile(itk.GetArrayFromImage(image), upper_percentile)
        print(
            f"Study {self.study_path.name}:\tLower Threshold: {lower_threshold}, Upper Threshold: {upper_threshold}"
        )
        # Calculate the median image
        median_image = self._calculate_median_image(image, radius)
        mask = self._find_mask_of_outliers(
            median_image, lower_threshold, upper_threshold
        )
        # Multiply the image with the mask
        multiply_filter = itk.MultiplyImageFilter[
            self.IMAGE_TYPE, self.IMAGE_TYPE, self.IMAGE_TYPE
        ].New()
        multiply_filter.SetInput1(image)
        multiply_filter.SetInput2(mask)
        multiply_filter.Update()
        return multiply_filter.GetOutput()

    def _convert_images_to_zscore_images(self, image_to_rescale: itk.Image[itk.F, 3]):
        # Calculate the mean and standard deviation of the image
        stats = itk.StatisticsImageFilter[self.IMAGE_TYPE].New()
        stats.SetInput(image_to_rescale)
        stats.Update()
        mean = stats.GetMean()
        std = stats.GetSigma()
        # print(f"Study {self.study_path.name}:\tMean: {mean}, Std: {std}")
        # Subtract the mean and divide by the standard deviation
        subtract_filter = itk.SubtractImageFilter[
            self.IMAGE_TYPE, self.IMAGE_TYPE, self.IMAGE_TYPE
        ].New()
        subtract_filter.SetInput1(image_to_rescale)
        subtract_filter.SetConstant2(mean)
        subtract_filter.Update()
        divide_filter = itk.DivideImageFilter[
            self.IMAGE_TYPE, self.IMAGE_TYPE, self.IMAGE_TYPE
        ].New()
        divide_filter.SetInput1(subtract_filter.GetOutput())
        divide_filter.SetConstant2(std)
        divide_filter.Update()
        return divide_filter.GetOutput()  # Z-score image

    def _find_mask_of_outliers(
        self, image: itk.Image[itk.F, 3], lower_threshold, upper_threshold
    ):
        """
        Finds the mask of the outliers.
        Parameters
        ----------
        image
        lower_threshold
        upper_threshold

        Returns
        -------
        mask.GetOutput()
        """
        mask = itk.BinaryThresholdImageFilter[self.IMAGE_TYPE, self.IMAGE_TYPE].New()
        mask.SetInput(image)
        mask.SetLowerThreshold(lower_threshold)
        mask.SetUpperThreshold(upper_threshold)
        mask.SetOutsideValue(1)
        mask.SetInsideValue(0)
        mask.Update()
        return mask.GetOutput()

    def _calculate_median_image(
        self, img: itk.Image[itk.F, 3], radius: list[int, int, int]
    ) -> itk.Image[itk.F, 3]:
        """
        Calculates the median image.
        Parameters
        ----------
        img
        radius

        Returns
        -------
        median_image_filter.GetOutput()
        """
        median_image_filter = itk.MedianImageFilter[
            itk.Image[itk.F, 3], itk.Image[itk.F, 3]
        ].New()
        median_image_filter.SetRadius(radius)
        median_image_filter.SetInput(img)
        median_image_filter.Update()
        return median_image_filter.GetOutput()

    def pre_process_images_for_training(self):
        """
        Preprocesses the images for training.
        Returns
        -------
        rescaled_t2w, rescaled_adc, rescaled_blow, rescaled_tracew
        """
        resampled_t2w, resampled_adc, resampled_blow, resampled_tracew = (
            self._get_all_resampled_images()
        )
        rescaled_t2w = self._convert_images_to_zscore_images(resampled_t2w)
        rescaled_adc = self._convert_images_to_zscore_images(resampled_adc)
        rescaled_blow = self._convert_images_to_zscore_images(resampled_blow)
        rescaled_tracew = self._convert_images_to_zscore_images(resampled_tracew)
        return rescaled_t2w, rescaled_adc, rescaled_blow, rescaled_tracew


def preprocess_images(
    base_data_path: Path, output_path: Path, has_segmentation: bool = False
):
    """
    This function initializes the image and mask paths.

    Args:
    base_data_path (Path): The base path of the data.
    output_path (Path): The path to output the preprocessed images.

    Returns:
        None

    """

    for dir in tqdm(base_data_path.iterdir()):
        if dir.is_dir():
            subject_output_dir = output_path / f"{dir.name}"
            subject_output_dir.mkdir(parents=True, exist_ok=True)

            study = SingleStudyStackedDataBase(dir)
            try:
                study.load_images()
            except:
                print(f"Study {dir.name} failed to load images")
                continue
            # Fill outliers with median
            pp_t2w, pp_adc, pp_blow, pp_tracew = study.pre_process_images_for_training()
            itk.imwrite(pp_t2w, subject_output_dir / f"{dir.name}_pp_t2w.nii.gz")
            itk.imwrite(pp_adc, subject_output_dir / f"{dir.name}_pp_adc.nii.gz")
            itk.imwrite(pp_blow, subject_output_dir / f"{dir.name}_pp_blow.nii.gz")
            itk.imwrite(pp_tracew, subject_output_dir / f"{dir.name}_pp_tracew.nii.gz")
            # Fill outliers with median
            if has_segmentation:
                segmentation_path = dir / f"{dir.name}_segmentation.nii.gz"
                segmentation = itk.imread(segmentation_path.as_posix())
                pp_segmentation = study.resample_coresponding_label(segmentation)

                pp_seg_output_dir: Path = (
                    subject_output_dir / f"{dir.name}_pp_segmentation.nii.gz"
                )
                if pp_seg_output_dir.exists():
                    pp_seg_output_dir.unlink()
                itk.imwrite(pp_segmentation, pp_seg_output_dir)


if __name__ == "__main__":
    with_out_seg_data_path = Path("/ALL_PROSTATEx/WITHOUT_SEGMENTATION/RAW")
    with_out_seg_output_path = Path("/ALL_PROSTATEx/WITHOUT_SEGMENTATION/PreProcessed")
    preprocess_images(
        with_out_seg_data_path, with_out_seg_output_path, has_segmentation=False
    )