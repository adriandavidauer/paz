'''
VVAD-LRS3 Dataset Generator
'''

# System imports
import os
import h5py
import random
import numpy as np
import math
from typing import Literal, get_args
# 3rd party imports
from paz.datasets.utils import get_class_names
import keras
# local imports
from generator import Generator
# end file header
__author__      = 'Adrian Auer'


REDUCTION_METHODS = ['cut', 'reduce']
DATA_VARIANTS = ['face_images', 'lip_images', 'face_features', 'lip_features']
URL= "https://www.kaggle.com/api/v1/datasets/download/adrianlubitz/vvadlrs3"

class VvadLrs3Dataset(Generator):
    """Class for generating the VVAD-LRS3 dataset.
    # Arguments
        path: String. Full path to vvadlrs3 folder containing 'faceImages_small.h5', 'lipImages.h5', 'faceFeatures.h5' and 'lipFeatures.h5' files.
        split: String. Valid option contain 'train', 'validation' or 'test'.
        validation_split: Float. Percentage of the dataset to be used for validation (valid options between 0.0 to 1.0). Set
            to 0.0 to disable.
        test_split: Float. Percentage of the dataset to be used for testing (valid options between 0.0 to 1.0). Set
            to 0.0 to disable.
        testing: Boolean. If True, a smaller dataset of 200 samples is used for testing. This is useful for testing
        evaluating: Boolean. If True, the dataset is used for evaluation. This means that the dataset is not shuffled
            and the indexes will be stored.
        reduction_method: String. Valid options are 'cut' or 'reduce'. If 'cut' is selected, the video is cut to the
            reduction_length. If 'reduce' is selected, reduction_length many single frames of the video is removed form
            the clip.
        reduced_length: Float. The length of the video after the reduction_method is applied. Choose None if you
            want to keep the original size. None is the default
        data_variant: String. The type of data to be used. Valid options are 'face_images', 'lip_images', 'face_features', 'lip_features'.

    # References
        -[VVAD-LRS3](https://www.kaggle.com/datasets/adrianlubitz/vvadlrs3)
    """
    def __init__(
            self, path='~/.keras/paz/datasets', split='train', validation_split=0.2, test_split=0.1, testing=False,
            evaluating=False, reduction_method: str = 'cut', reduced_length=None, data_variant:str='face_images'):
        if split != 'train' and split != 'validation' and split != 'test':
            raise ValueError('Invalid split name')
        if validation_split < 0.0 or validation_split > 1.0:
            raise ValueError('Invalid validation split')
        if test_split < 0.0 or test_split > 1.0:
            raise ValueError('Invalid test split')
        if validation_split + test_split > 1.0:
            raise ValueError('The sum of val_split and test_split must be less than 1.0')
        if data_variant not in DATA_VARIANTS:
            raise ValueError(f"Invalid data_variant '{data_variant}'. Valid options are {DATA_VARIANTS}")
        if reduction_method not in REDUCTION_METHODS:
            raise ValueError(f"Invalid reduction_method '{reduction_method}'. Valid options are {REDUCTION_METHODS}")

        # TODO: Check downloading with different paths + remove print statements
        print("Downloading VVAD-LRS3 dataset...")
        path = os.path.expanduser(path)
        print("Data will be downloaded to:", path)
        extract_dir = os.path.join(path, "vvadlrs3_extracted") 
        print("Data extracted path:", extract_dir)
        if not os.path.exists(extract_dir):
            data_path = keras.utils.get_file(origin=URL, fname="vvadlrs3.zip", cache_dir=path, extract=True)
        else:
            print("Already extracted, skipping extraction.")
        # print the content of the data_path
        print("Data path:", data_path)
        
        print("Files in data extracted path:", os.listdir(extract_dir))

        if data_variant == 'face_images':
            path = os.path.join(extract_dir, 'faceImages_small.h5')
        elif data_variant == 'lip_images':
            path = os.path.join(extract_dir, 'lipImages.h5')
        elif data_variant == 'face_features':
            path = os.path.join(extract_dir, 'faceFeatures.h5')
        elif data_variant == 'lip_features':
            path = os.path.join(extract_dir, 'lipFeatures.h5')
        print("Using dataset path:", path)
            # path = os.path.join(path, 'vvadlrs3_faceImages_small.h5')

        class_names = get_class_names('VVAD_LRS3')

        super(VvadLrs3Dataset, self).__init__(path, split, class_names, 'VVAD_LRS3')
        self.validation_split = validation_split
        self.test_split = test_split
        self.use_test_data = testing
        self.evaluating = evaluating
        self.index = []
        self.reduction_method = reduction_method
        self.reduced_length = reduced_length

        # Get total dataset size and length of the video clips
        self.total_dataset_size, self.video_length = self.get_data_and_video_size()

        # Split dataset indexes
        self.indexes_train, self.indexes_validation, self.indexes_test = self.split_data_indices()

        # Reduction init
        self.dropout_ids = []
        self.reduction_init()

        random.seed(445363)

    def __call__(self):
        indexes = []
        if self.split == 'train':
            indexes = self.indexes_train
            random.shuffle(indexes)  # Use always the same seed so every model gets the same shuffle
        elif self.split == 'validation':
            indexes = self.indexes_validation
        elif self.split == 'test':
            indexes = self.indexes_test

        data = h5py.File(self.path, mode='r')
        if not self.use_test_data:
            x_train = data.get('x_train')
            y_train = data.get('y_train')
        else:
            indexes = reversed(indexes)
            x_train = data.get('x_test')
            y_train = data.get('y_test')

        for index in indexes:
            yield self.generate_x_data(x_train, index), y_train[index]
        data.close()

    def get_data_and_video_size(self):
        """Gets the total size of the dataset and the length of the video clips."""
        data = h5py.File(self.path, mode='r')

        total_size = 0
        if not self.use_test_data:
            total_size = data.get('x_train').shape[0]
            length = data.get('x_test').shape[1]
        else:
            total_size = data.get('x_test').shape[0]
            length = data.get('x_test').shape[1]
        data.close()
        return total_size, length

    def split_data_indices(self):
        """Splits the data indices into train, validation and test."""
        indexes_positive = list(range(self.total_dataset_size // 2))
        indexes_negative = list(range(self.total_dataset_size // 2, self.total_dataset_size))

        random.Random(445363).shuffle(indexes_positive)
        random.Random(848641).shuffle(indexes_negative)

        indexes_validation = []
        indexes_test = []

        if self.validation_split > 0.0:
            indexes_validation, indexes_positive, indexes_negative = \
                self.split_by_ratio(self.validation_split, indexes_positive, indexes_negative)
        if self.test_split > 0.0:
            indexes_test, indexes_positive, indexes_negative = \
                self.split_by_ratio(self.test_split, indexes_positive, indexes_negative)

        indexes_train = indexes_positive + indexes_negative

        return indexes_train, indexes_validation, indexes_test

    def split_by_ratio(self, split_ratio, indexes_positive, indexes_negative):
        """Splits the indexes positive and negative by a ratio.

        # Arguments
            split_ratio: Float. Ratio of the split. Valid options are between 0.0 and 1.0.
            indexes_positive: List of integers. Indexes of the positive samples.
            indexes_negative: List of integers. Indexes of the negative samples.
        """
        split_size = int(split_ratio * 0.5 * self.total_dataset_size)
        indexes_split = indexes_positive[:split_size] + indexes_negative[:split_size]
        indexes_positive = indexes_positive[split_size:]
        indexes_negative = indexes_negative[split_size:]
        return indexes_split, indexes_positive, indexes_negative

    def reduction_init(self):
        """Initializes the reduction method.

        # Arguments
            reduction_method: String. Valid options are 'cut' or 'reduce'. If 'cut' is selected, the video is cut to the
                reduction_length. If 'reduce' is selected, reduction_length many single frames of the video is removed
                form the clip.
            reduced_length: Float. The length of the video after the reduction_method is applied. Choose None if you
                want to keep the original size. None is the default
        """
        if self.reduced_length is None:
            self.reduced_length = self.video_length
        elif self.reduced_length > self.video_length:
            raise ValueError('reduction_length must be smaller than the length of the video')
        else:
            self.reduced_length = math.ceil(self.reduced_length / 25 * self.video_length)

        if 'reduce' in self.reduction_method:
            if self.reduced_length == self.video_length:
                self.reduction_method = 'cut'
            else:
                count_dropouts = self.video_length - self.reduced_length

                cal_drop_every = self.video_length / count_dropouts

                self.dropout_ids = [int(i * cal_drop_every - (cal_drop_every / 2))
                                    for i in range(1, count_dropouts + 1)]

    def generate_x_data(self, x_train, i):
        """Generates the x data for the given index.

        # Arguments
            x_train: h5py dataset. Dataset containing the x data.
            i: Integer. Index of the data to be generated.
        """

        if self.evaluating:
            self.index.append(i)

        if 'reduce' in self.reduction_method:  # First tested using appending all wanted frames, but it is
            # more efficient to remove the unwanted frames
            x_out = x_train[i]
            x_out = np.delete(x_out, self.dropout_ids, 0)
        else:
            x_out = x_train[i][:self.reduced_length]
        return x_out

    def get_index(self):
        """Gets the index of the current sample. Only available if evaluating is set to True."""
        return self.index.pop(0)

    def __len__(self):
        if self.total_dataset_size == -1:
            raise ValueError('You need to call __call__ first to set the total_size')

        if self.split == 'train':
            return (self.total_dataset_size - int(self.validation_split * 0.5 * self.total_dataset_size) * 2 -
                    int(self.test_split * 0.5 * self.total_dataset_size) * 2)
        elif self.split == 'validation':
            return int(self.validation_split * 0.5 * self.total_dataset_size) * 2
        elif self.split == 'test':
            print('total_size_test', self.total_dataset_size)
            return int(self.test_split * 0.5 * self.total_dataset_size) * 2

if __name__ == '__main__':
    # Example usage
    dataset = VvadLrs3Dataset(path='~/.keras/paz/datasets', split='train', validation_split=0.2, test_split=0.1, testing=False,
                              evaluating=False, reduction_method='cut', reduced_length=None, data_variant='face_images')
    for sample, label in dataset():
        print(sample.shape, label)
        break  # Just to show the first sample
    print("Number of classes:", dataset.num_classes)