import collections
import logging
import os
from pathlib import Path

import highdicom as hd
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import tensorflow as tf
import hashlib
from pydicom.sr.codedict import codes
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import binarize

from dense_unet import denseunet
from new_r2udensenet import r2udensenet
from new_r2unet import r2unet
from res_unet import resunet
from unet import unet_model

logging.captureWarnings(True)


class SegmentationPipeline:

    T1_IMAGE = 'T1'
    T2_IMAGE = 'T2'
    UNK_IMAGE = 'UNK'

    def __init__(self, input_dir: str, output_dir: str, weight_dir: str, model: str = 'r2udensenet', **kwargs):
        logging.info('Initializing TNBC Segmentation Pipeline')
        logging.debug(f'Input directory: {input_dir}, Output directory: {output_dir}, '
                      f'Weight directory: {weight_dir}, Model: {model}, Additional arguments: {kwargs}')

        self.input_dir: str = input_dir
        self.output_dir: str = output_dir
        self.weight_dir: str = weight_dir
        self.model: str = model
        self.preprocessed_images: np.ndarray | None = None
        self.segmentation_model: tf.keras.Model | None = None
        self.segmentation_masks: np.ndarray | None = None
        self.dicom_series: dict = collections.defaultdict(list)
        self.dicom_series_types: dict | None = None
        self.dicom_files: dict = collections.defaultdict(list)
        self.dicom_segmentations: dict = collections.defaultdict(list)
        self.qc_fig = None
        self.image_shape = None
        self.series_length = None

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_dicom_series(self):
        """
        Load DICOM series from the input directory.
        """
        logging.info(f'Loading DICOM series from {self.input_dir}')

        # Get all DICOM files in the input directory
        dicom_files = [str(p) for p in Path(self.input_dir).rglob('*.dcm')]

        logging.info(f'Found {len(dicom_files)} DICOM files')

        # Sort DICOM files into a dictionary for each series
        dicom_series = collections.defaultdict(list)
        dicom_series_types = {}
        for fp in dicom_files:
            ds = pydicom.dcmread(fp)

            # Skip files without a SeriesInstanceUID or not of modality 'MR'
            if not hasattr(ds, 'SeriesInstanceUID') or ds.Modality != 'MR':
                logging.warning(f'Skipping file {fp} without SeriesInstanceUID or not of modality MR')
                continue

            series = ds.SeriesInstanceUID
            dicom_series[series].append((fp, ds))

            # Need to determine which series is T1 and which is T2 for the model
            if 't1' in ds.SeriesDescription.lower() or 't1' in ds.ProtocolName.lower() or 't1' in ds.SequenceName.lower():
                dicom_series_types[series] = self.T1_IMAGE
            elif 't2' in ds.SeriesDescription.lower() or 't2' in ds.ProtocolName.lower() or 't2' in ds.SequenceName.lower():
                dicom_series_types[series] = self.T2_IMAGE
            else:
                dicom_series_types[series] = self.UNK_IMAGE

        # Sort the files and series by InstanceNumber
        for series, datasets in dicom_series.items():
            dicom_series[series] = sorted(datasets, key=lambda x: x[1].InstanceNumber)

        # Log if T1 and T2 series are not found
        errors = []
        if self.T1_IMAGE not in dicom_series_types.values():
            logging.error('No T1 series found')
            errors.append('No T1 series found')

        if self.T2_IMAGE not in dicom_series_types.values():
            logging.error('No T2 series found')
            errors.append('No T2 series found')

        if errors:
            raise ValueError(errors)

        # Log if no series are found
        if len(dicom_series) == 0:
            logging.error('No series found')
            raise ValueError('No series found')

        self.dicom_series = dicom_series
        self.dicom_series_types = dicom_series_types

        # All series should have the same number of files
        series_lengths = set([len(datasets) for datasets in dicom_series.values()])
        if len(series_lengths) > 1:
            logging.error('Series have different number of files')
            raise ValueError('Series have different number of files')

        self.series_length = series_lengths.pop()

        # Set the image shape, presuming all images have the same shape
        ds = dicom_series[list(dicom_series.keys())[0]][0][1]
        self.image_shape = (ds.Rows, ds.Columns)

        logging.info(f'Found {len(dicom_series)} series, '
                     f'each with {self.series_length} slices, '
                     f'each with shape {self.image_shape}')

        return dicom_series

    def load_model(self):
        """
        Load the segmentation model.
        """
        logging.info(f'Loading segmentation model: {self.model}')

        switcher = {
            'denseunet': denseunet,
            'unet': unet_model,
            'resunet': resunet,
            'r2udensenet': r2udensenet,
            'r2unet': r2unet
        }

        model_func = switcher.get(self.model)
        self.segmentation_model = model_func()
        self.segmentation_model.load_weights(os.path.join(self.weight_dir, f'model_{self.model}.hdf5'))

    def preprocess_images(self):
        """
        Preprocess the images.
        """
        logging.info('Preprocessing images')

        images = np.empty((self.series_length, *self.image_shape, len(self.dicom_series)))

        logging.debug(f'Preprocessed images shape: {images.shape}')

        # Load and normalize by series
        for i, (series, datasets) in reversed(list(enumerate(self.dicom_series.items()))):
            normalized_images = self.normalize_images([ds for (fp, ds) in datasets])

            # T2 images are the first channel, T1 images are the second channel
            if self.dicom_series_types[series] == self.T1_IMAGE:
                images[:, :, :, 1] = normalized_images
            elif self.dicom_series_types[series] == self.T2_IMAGE:
                images[:, :, :, 0] = normalized_images
            else:
                logging.error(f'Unknown image type for series {series}, not T1 or T2, skipping series')

        images = images.astype('float32')
        hash_md5 = hashlib.md5(images.tobytes()).hexdigest()

        logging.info(f'Preprocessed images shape: {images.shape}')
        logging.info(f'Preprocessed images type: {images.dtype}')
        logging.info(f'Preprocessed images min: {np.min(images)}')
        logging.info(f'Preprocessed images max: {np.max(images)}')
        logging.info(f'Preprocessed images md5 hash: {hash_md5}')

        self.preprocessed_images = images

    def normalize_images(self, datasets: list) -> np.ndarray:
        """
        Normalize the pixel arrays of the DICOM datasets.
        """
        images = np.array([ds.pixel_array for ds in datasets])
        images = images.astype('float32')
        images = np.squeeze(images)
        mean = np.mean(images)
        std = np.std(images)
        images_normalized = (images - mean) / std

        return images_normalized

    def segment_images(self):
        """
        Segment the images using the segmentation model.
        """
        logging.info('Segmenting images')

        masks = self.segmentation_model.predict(self.preprocessed_images, batch_size=1, verbose=1)
        masks = np.squeeze(masks, axis=3)
        masks = np.around(masks, decimals=0)

        logging.debug(f'Segmentation masks shape: {masks.shape}')

        self.segmentation_masks = masks

    def evaluate_performance(self):
        """
        Evaluate the performance of the segmentation.
        """
        pass

    def save_masks(self):
        """
        Save the segmentation masks as PNG images.
        """
        logging.info('Saving segmentation masks')

        for i in range(self.segmentation_masks.shape[0]):
            plt.imsave(f'{self.output_dir}/mask_{i}.png', self.segmentation_masks[i], cmap=plt.cm.gray)

    def save_dicom_seg_objects(self):
        """
        Save the mask as a DICOM Segmentation object for each series.
        """
        logging.info('Saving segmentation masks')

        # Create a single segmentation mask containing all segmentations
        mask = self.segmentation_masks.astype(bool)

        # Describe the algorithm that created the segmentation
        algorithm_identification = hd.AlgorithmIdentificationSequence(
            name='DR2UNet',
            version='v0.0.1',
            family=codes.cid7162.ArtificialIntelligence
        )

        # Create a description for each segment
        description_segment = hd.seg.SegmentDescription(
                segment_number=1,
                segment_label=f'Segment',
                segmented_property_category=codes.cid7150.Tissue,
                segmented_property_type=codes.cid7166.Tissue,
                algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
                algorithm_identification=algorithm_identification,
                tracking_uid=hd.UID(),
                tracking_id=f'1',
        )

        # Save the segmentation mask for each series
        for series, datasets in self.dicom_series.items():
            for (fp, ds) in datasets:
                if not hasattr(ds, 'AccessionNumber'):
                    ds.AccessionNumber = ds.StudyInstanceUID

            seg_dataset = hd.seg.Segmentation(
                source_images=[ds for (fp, ds) in datasets],
                pixel_array=mask,
                segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
                segment_descriptions=[description_segment],
                series_instance_uid=hd.UID(),
                series_number=9999,
                sop_instance_uid=hd.UID(),
                instance_number=1,
                manufacturer='Manufacturer',
                manufacturer_model_name='Model',
                software_versions='v1',
                device_serial_number='Device XYZ',
            )

            seg_dataset.SeriesDescription = 'DR2UNet Segmentation'

            # Delete some tags to avoid downstream issues with XNAT OHIF viewer
            if hasattr(seg_dataset, 'PatientWeight'):
                delattr(seg_dataset, 'PatientWeight')
            if hasattr(seg_dataset, 'PatientAge'):
                delattr(seg_dataset, 'PatientAge')

            series_short = series.split('.')[-1]
            output_path = f'{self.output_dir}/mask_{series_short}.dcm'
            seg_dataset.save_as(output_path)
            self.dicom_segmentations[series].append((output_path, seg_dataset))

            logging.info(f'Saved segmentation mask for series {series} to {output_path}')

        logging.debug('Finished saving segmentation masks')

    def visualize_segmentation(self, save=False):
        """
        Visualize the segmentation by plotting the center slice of each series with the segmentation mask overlayed.
        Plot the first, middle, and last slice of each series with the segmentation mask overlayed.
        """

        # Five images per row for each series
        col = 4
        rows = int(np.ceil(self.series_length / col))

        if rows < 2:
            rows = 2
            col = 3

        rows = rows * len(self.dicom_series)

        logging.info(f'Visualizing segmentation masks with {rows} rows and {col} columns')

        fig, axs = plt.subplots(rows, col, figsize=(20, 4 * rows))
        fig.subplots_adjust(wspace=0.01, hspace=0.03)

        for i, (series, datasets) in enumerate(self.dicom_series.items()):
            for j, (fp, ds) in enumerate(datasets):
                r = (j // col) + (i * rows // len(self.dicom_series))
                c = j % col

                logging.debug(f'Plotting series {series} slice {j} at row {r} column {c}')

                image = self.preprocessed_images[j, :, :, i]
                mask = self.segmentation_masks[j, :, :]
                pixel_aspect_ratio = ds.PixelAspectRatio.as_integer_ratio()
                axs[r, c].imshow(image, cmap=plt.cm.gray)
                axs[r, c].imshow(mask, cmap=plt.cm.jet, alpha=0.2)
                axs[r, c].axis('off')
                axs[r, c].set_aspect(pixel_aspect_ratio[1] / pixel_aspect_ratio[0])

        if save:
            plt.savefig(f'{self.output_dir}/segmentation.png')

        return fig

    def run_pipeline(self):
        """
        Run the DICOM segmentation pipeline.
        """
        logging.info('Running DICOM segmentation pipeline')

        self.load_dicom_series()
        self.load_model()
        self.preprocess_images()
        self.segment_images()
        self.save_masks()
        self.save_dicom_seg_objects()

    def load_train_data(self):
        """
        Original function from data2D.py for loading training data.
        """
        logging.info('====== Loading of Training Images and Masks ===================')
        images_train = np.load('images_train.npy')
        images_train_T1 = np.load('images_train_T1.npy')
        mask_train = np.load('mask_train.npy')
        return images_train, images_train_T1, mask_train

    def load_test_data(self):
        """
        Original function from data2D.py for loading test data.
        """
        logging.info('======Loading of Test Data=======')
        images_test = np.load('images_test.npy')
        return images_test

    def gen_precision_recall(self):
        """
        Original function from main_testing.py for generating precision recall curve.
        :return:
        """
        logging.info('====== Generate Precision Recall Curve ========')
        images_train, images_train_T1, mask_train = self.load_train_data()
        images_train = images_train.astype('float32')
        images_train_T1 = images_train_T1.astype('float32')
        mask_train = mask_train.astype('float32')

        images_train_mean = np.mean(images_train)
        images_train_std = np.std(images_train)
        images_train = (images_train - images_train_mean) / images_train_std

        images_train_mean = np.mean(images_train_T1)
        images_train_std = np.std(images_train_T1)
        images_train_T1 = (images_train_T1 - images_train_mean) / images_train_std
        mask_train /= 255.
        mask_train = mask_train.ravel().reshape(-1, 1)
        mask_train = binarize(mask_train, threshold=0.5)

        images_train_final = np.append(images_train, images_train_T1, axis=3)

        switcher = {
            'denseunet': denseunet,
            'unet': unet_model,
            'resunet': resunet,
            'r2udensenet': r2udensenet,
            'r2unet': r2unet
        }
        model_func = switcher.get(self.model)
        model = model_func() if model_func else None
        model.load_weights(os.path.join(self.weight_dir, f'model_{self.model}.hdf5'))
        y_pred = model.predict(images_train_final, batch_size=1, verbose=1)
        y_pred = np.squeeze(y_pred, axis=3)

        y_test = mask_train
        nn_fpr_keras, nn_tpr_keras, nn_thresholds_keras = precision_recall_curve(y_test.ravel(), y_pred.ravel())
        diff = np.abs(nn_fpr_keras - nn_tpr_keras)
        place = np.argmin(diff)
        logging.info('Optimal Threshold = %s', str(nn_thresholds_keras[place]))
        logging.info('Recall = %s', nn_fpr_keras[place])
        logging.info('Precision = %s', nn_tpr_keras[place])
        # plt.plot(nn_tpr_keras, nn_fpr_keras)
        # plt.ylabel('Precision',fontweight='bold',fontsize = 20)
        # plt.xlabel('Recall',fontweight='bold',fontsize = 20)
        # plt.xticks(fontsize=18)
        # plt.yticks(fontsize=18)
        # plt.savefig('precision_recall.png')
        opt_thresh = nn_thresholds_keras[place]

        return opt_thresh
