from pathlib import Path
import itk
import numpy as np
from tqdm import tqdm

from itk_preprocessing import find_center_of_gravity_in_index_space, calculate_new_origin_from_center_of_mass, \
    create_blank_image


class SingleStudyStackedDataBase:
    x_y_size = 240  # Index
    z_size = 16  # Index
    x_y_fov = 120  # mm
    z_fov = 70  # mm
    IMAGE_PIXEL_TYPE = itk.F
    MASK_PIXEL_TYPE = itk.UC
    IMAGE_TYPE = itk.Image[IMAGE_PIXEL_TYPE, 3]
    MASK_TYPE = itk.Image[MASK_PIXEL_TYPE, 3]
    VECTOR_IMAGE_TYPE = itk.VectorImage[IMAGE_PIXEL_TYPE, 3]
    def __init__(self, study_path: Path):
        self.study_path = study_path
        self.t2w_path = study_path / f"{study_path.name}_t2w.nii.gz"
        self.adc_path = study_path / f"{study_path.name}_adc.nii.gz"
        self.blow = study_path / f"{study_path.name}_b0low.nii.gz"
        self.tracew = study_path / f"{study_path.name}_tracew.nii.gz"


        self.t2w = itk.imread(self.t2w_path.as_posix())
        self.adc = itk.imread(self.adc_path.as_posix())
        self.blow = itk.imread(self.blow.as_posix())
        self.tracew = itk.imread(self.tracew.as_posix())

    def get_t2w_image(self):
        return self.t2w
    def get_adc_image(self):
        return self.adc

    def get_blow_image(self):
        return self.blow

    def get_tracew_image(self):
        return self.tracew

    def _get_all_resampled_images(self):
        resampled_t2w = self._resample_images_for_training(self.get_t2w_image())
        resampled_adc = self._resample_images_for_training(self.get_adc_image())
        resampled_blow = self._resample_images_for_training(self.get_blow_image())
        resampled_tracew = self._resample_images_for_training(self.get_tracew_image())
        return resampled_t2w, resampled_adc, resampled_blow, resampled_tracew

    def resample_coresponding_label(self, label: itk.Image[itk.UC, 3]):
        return self._resample_images_for_training(label)
    def _resample_images_for_training(self, image_to_resample: itk.Image[itk.F, 3]):

        # Get the center of mass of the mask
        center_of_mass = find_center_of_gravity_in_index_space(self.t2w)

        new_size = [self.x_y_size, self.x_y_size, self.z_size]
        # Get the new in plane spacing for the recentered image
        new_in_plane_spacing = self.x_y_fov / self.x_y_size
        new_out_of_plane_spacing = self.z_fov / self.z_size
        new_spacing = [new_in_plane_spacing, new_in_plane_spacing, new_out_of_plane_spacing]

        center_of_mass_in_physical_space = self.get_t2w_image().TransformContinuousIndexToPhysicalPoint(
            center_of_mass
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

        linear_interpolator = itk.LinearInterpolateImageFunction[self.IMAGE_TYPE, itk.D].New()
        identity_transform = itk.IdentityTransform[itk.D, 3].New()
        # Initialize the ResampleImageFilter
        itk_t2w_resampler = itk.ResampleImageFilter[self.IMAGE_TYPE, self.IMAGE_TYPE].New()
        itk_t2w_resampler.SetInput(image_to_resample)
        itk_t2w_resampler.SetReferenceImage(new_image_blank)
        itk_t2w_resampler.SetTransform(identity_transform)
        itk_t2w_resampler.UseReferenceImageOn()
        itk_t2w_resampler.UpdateLargestPossibleRegion()
        itk_t2w_resampler.SetInterpolator(linear_interpolator)
        itk_t2w_resampler.Update()
        resampled_image = itk_t2w_resampler.GetOutput()
        return resampled_image
    def _fill_outliers_with_median(self, image: itk.Image[itk.F, 3], radius: list[int, int, int]):
        lower_percentile = .1
        upper_percentile = .99

        lower_threshold = np.quantile(itk.GetArrayFromImage(image), lower_percentile)
        upper_threshold = np.quantile(itk.GetArrayFromImage(image), upper_percentile)
        print(f"Study {self.study_path.name}:\tLower Threshold: {lower_threshold}, Upper Threshold: {upper_threshold}")
        median_image = self._calculate_median_image(image, radius)
        mask = self._find_mask_of_outliers(median_image, lower_threshold, upper_threshold)

        multiply_filter = itk.MultiplyImageFilter[self.IMAGE_TYPE, self.IMAGE_TYPE, self.IMAGE_TYPE].New()
        multiply_filter.SetInput1(image)
        multiply_filter.SetInput2(mask)
        multiply_filter.Update()
        return multiply_filter.GetOutput()
    def _rescale_images_for_training(self, image_to_rescale: itk.Image[itk.F, 3]):
        # Initialize the RescaleIntensityImageFilter
        normalizer = itk.RescaleIntensityImageFilter[self.IMAGE_TYPE, self.IMAGE_TYPE].New()
        normalizer.SetInput(image_to_rescale)
        normalizer.SetOutputMinimum(0)
        normalizer.SetOutputMaximum(1)
        normalizer.Update()
        return normalizer.GetOutput()
    def _find_mask_of_outliers(self, image: itk.Image[itk.F, 3], lower_threshold, upper_threshold):
        mask = itk.BinaryThresholdImageFilter[
            self.IMAGE_TYPE, self.IMAGE_TYPE
        ].New()
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
        median_image_filter = itk.MedianImageFilter[
            itk.Image[itk.F, 3], itk.Image[itk.F, 3]
        ].New()
        median_image_filter.SetRadius(radius)
        median_image_filter.SetInput(img)
        median_image_filter.Update()
        return median_image_filter.GetOutput()


    def pre_process_images_for_training(self):
        resampled_t2w, resampled_adc, resampled_blow, resampled_tracew = self._get_all_resampled_images()
        rescaled_t2w = self._rescale_images_for_training(resampled_t2w)
        rescaled_adc = self._rescale_images_for_training(resampled_adc)
        rescaled_blow = self._rescale_images_for_training(resampled_blow)
        rescaled_tracew = self._rescale_images_for_training(resampled_tracew)
        return rescaled_t2w, rescaled_adc, rescaled_blow, rescaled_tracew

    def create_stacked_image(self):
        pp_t2w, pp_adc, _, pp_tracew = self.pre_process_images_for_training()
        ImageToVectorImageFilterType = itk.ComposeImageFilter[
            self.IMAGE_TYPE, self.VECTOR_IMAGE_TYPE
        ].New()
        ImageToVectorImageFilterType.SetInput1(pp_t2w)
        ImageToVectorImageFilterType.SetInput2(pp_adc)
        # ImageToVectorImageFilterType.SetInput3(pp_blow)
        ImageToVectorImageFilterType.SetInput3(pp_tracew)
        ImageToVectorImageFilterType.Update()
        return ImageToVectorImageFilterType.GetOutput()


def preprocess_images(base_data_path: Path, output_path: Path, has_segmentation: bool = False):
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
            pp_t2w, pp_adc, pp_blow, pp_tracew = study.pre_process_images_for_training()
            itk.imwrite(pp_t2w, subject_output_dir / f"{dir.name}_pp_t2w.nii.gz")
            itk.imwrite(pp_adc, subject_output_dir / f"{dir.name}_pp_adc.nii.gz")
            itk.imwrite(pp_blow, subject_output_dir / f"{dir.name}_pp_blow.nii.gz")
            itk.imwrite(pp_tracew, subject_output_dir / f"{dir.name}_pp_tracew.nii.gz")

            if has_segmentation:
                segmentation_path = dir / f"{dir.name}_Segmentation.nii.gz"
                segmentation = itk.imread(segmentation_path.as_posix())
                pp_segmentation = study.resample_coresponding_label(segmentation)
                itk.imwrite(pp_segmentation, subject_output_dir / f"{dir.name}_pp_segmentation.nii.gz")

            stacked_image = study.create_stacked_image()
            itk.imwrite(stacked_image, subject_output_dir / f"{dir.name}_stacked_image.nii.gz")


if __name__ == "__main__":
    base_data_path = Path("/Users/iejohnson/School/spring_2024/AML/Supervised_learning/DATA/ALL_PROSTATEx/WITH_SEGMENTATION/RAW")
    output_path = Path("/Users/iejohnson/School/spring_2024/AML/Supervised_learning/DATA/ALL_PROSTATEx/WITH_SEGMENTATION/PreProcessed")
    preprocess_images(base_data_path, output_path, has_segmentation=True)