import utils
import config as config

#import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from typing import Any
import cv2
#from tqdm import notebook, tnrange
import glob

from datasets import load_dataset

from torch.utils.data import Dataset
import os
from transformers import SegformerImageProcessor, SegformerFeatureExtractor

from torch.utils.data import DataLoader
from torch import Tensor
from sklearn.model_selection import train_test_split

def import_data_image(sample_dir: str):

    # images
    img = cv2.imread(sample_dir)
    img = cv2.resize(img, (config.im_width, config.im_height), interpolation=cv2.INTER_AREA)

    return img

def read_mask(folder_path: str)->NDArray:
    iris = cv2.imread(os.path.join(folder_path, 'iris.png'), cv2.IMREAD_GRAYSCALE)
    pupil = cv2.imread(os.path.join(folder_path, 'pupil.png'), cv2.IMREAD_GRAYSCALE)

    return (iris-pupil)//255*150 + (pupil//255)*29


def import_data_mask(sample_dir: str):

    # masks
    # mask = cv2.imread(sample_dir)
    # mask_a = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_a = read_mask(sample_dir)
    lut = np.arange(256, dtype=np.uint8)*0
    lut[29] = 1
    lut[150] = 2
    mask_a = cv2.LUT(mask_a, lut)
    return mask_a

def import_data(sample_dir: str):

    # images
    img = import_data_image(sample_dir + '/image_resize.png')

    # masks
    
    mask = import_data_mask(sample_dir)

    return img, mask


def add_conversion_tp_bw_images(images: NDArray, labels: NDArray) -> tuple[NDArray, NDArray]:
    n = len(images)
    imagesBW = np.sum(images, axis=3)/3
    imagesBW = np.repeat(imagesBW[..., np.newaxis], 3, axis=-1)

    images = np.append(images, imagesBW, axis=0)
    labels = np.append(labels, labels, axis=0)
    return images, labels


def load_data_inference(path_data: str = config.path_data)->tuple[NDArray, NDArray]:
    print('Getting and resizing inference images and masks ... ')

    sample_dirs = glob.glob(os.path.join(path_data,  '*.png'))
    if config.reduced_full_dataset:
        config.N_samples = np.minimum(len(sample_dirs), config.N_samples)
    else:
        config.N_samples = len(sample_dirs)
    train_ids = range(0,config.N_samples)

    id_map = np.zeros(config.N_samples, dtype=np.uint32)
    X_train = np.zeros((config.N_samples, config.im_height, config.im_width, config.im_chan), dtype=np.float32)
    
    n = 0
    for id_ in train_ids:
        X_train[n] = import_data_image(sample_dirs[id_])
        id_map[n] = id_
        n+=1

    return X_train, id_map


def load_data(path_data: str = config.path_data, validation_split: float = config.validation_split)->tuple[NDArray, NDArray, NDArray, NDArray, list[int], list[int], NDArray, tuple[int, int]]:
    img = cv2.imread(path_data + '/sample_1/image_resize.png')[:,:,1]
    sizes_test = [img.shape[0], img.shape[1]]

    print('Getting and resizing train images and masks ... ')

    sample_dirs = glob.glob(os.path.join(path_data,  'sample_*'))
    if config.reduced_full_dataset:
        config.N_samples = np.minimum(len(sample_dirs), config.N_samples)
    else:
        config.N_samples = len(sample_dirs)
    train_ids = range(0,config.N_samples)

    id_map = np.zeros(config.N_samples, dtype=np.uint32)
    X_train = np.zeros((config.N_samples, config.im_height, config.im_width, config.im_chan), dtype=np.float32)
    Y_train = np.zeros((config.N_samples, config.out_height, config.out_width), dtype=np.float32) 
    
    n = 0
    for id_ in train_ids:
    
        X, mask = import_data(sample_dirs[id_])
        X_train[n] = X
        Y_train[n] = mask

        id_map[n] = int(sample_dirs[id_].split('_')[-1])

        n += 1
    
    config.N_samples = n

    X_train = X_train[:config.N_samples]
    Y_train = Y_train[:config.N_samples]

    X_train, X_test, Y_train, Y_test, idx1, idx2 = train_test_split(X_train,Y_train, range(config.N_samples),test_size=validation_split, random_state=1337)

    meanTrain = np.mean(X_train, axis=(0,1,2))
    np.savetxt('meanValueTrain.csv', meanTrain, delimiter=',')

    if config.load_additional_data:
        X_train, Y_train = load_additional_data(X_train, Y_train, path_data=config.path_additional_data)

    X_test = np.append(X_test, X_train[-10:-1], axis=0)
    Y_test = np.append(Y_test, Y_train[-10:-1], axis=0)

    return X_train, X_test, Y_train, Y_test, idx1, idx2, id_map, sizes_test


def load_additional_data(X_train: NDArray, Y_train: NDArray, path_data: str = config.path_data):
    img = cv2.imread(path_data + '/sample_1/image_resize.png')[:,:,1]
    sizes_test = [img.shape[0], img.shape[1]]

    print('Getting and resizing train images and masks ... ')

    sample_dirs = glob.glob(os.path.join(path_data,  'sample_*'))

    config.N_samples_add = np.minimum(len(sample_dirs), config.N_samples_add)
    train_ids = range(0,config.N_samples_add)
    n_base = X_train.shape[0]

    id_map = np.zeros(config.N_samples_add, dtype=np.uint32)
    X_train0 = np.zeros((config.N_samples_add + n_base, config.im_height, config.im_width, config.im_chan), dtype=np.float32)
    Y_train0 = np.zeros((config.N_samples_add + n_base, config.out_height, config.out_width), dtype=np.float32) 
    X_train0[:n_base, :, :, :] = X_train
    Y_train0[:n_base, :, :] = Y_train
    n = 0

    for id_ in train_ids:
    
        X, mask = import_data(sample_dirs[id_])

        X_train0[n + n_base] = X
        Y_train0[n + n_base] = mask

        id_map[n] = int(sample_dirs[id_].split('_')[-1])

        n += 1
        if n > config.N_samples_add:
            break
    
    print('Finished')

    config.N_samples_add = n

    X_train0 = X_train0[:(config.N_samples_add + n_base)]
    Y_train0 = Y_train0[:(config.N_samples_add + n_base)]

    return X_train0, Y_train0


class EyeSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, X, Y, idx, image_processor: SegformerImageProcessor, train: bool=True) -> None:
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            image_processor (SegFormerImageProcessor): image processor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.image_processor = image_processor
        self.train = train
        self.transform_image = None
        self.shared_transform = None

        self.images = X
        self.images /= 255.0
        self.annotations = Y
        
        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = self.images[idx]
        segmentation_map = self.annotations[idx]

        if self.transform_image:
            image = self.transform_image(image=image)['image']

        if self.shared_transform:
            augmented = self.shared_transform(image=image, label=segmentation_map)
            image = augmented['image']
            segmentation_map = augmented['label']

        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.image_processor(image, segmentation_map, return_tensors="pt")

        for k,v in encoded_inputs.items():
            if isinstance(encoded_inputs[k],list):
                encoded_inputs[k] = encoded_inputs[k][0].squeeze_()
            else:
                encoded_inputs[k].squeeze_()
            
        return encoded_inputs
    
    def set_transform(self, transform: Any) -> None:
        """Update the dataset's transformation."""
        self.shared_transform = transform

    def set_transform_image(self, transform: Any) -> None:
        """Update the dataset's transformation."""
        self.transform_image = transform


def ade_palette() -> list[list[int]]:
    """ADE20K palette that maps each class to RGB values."""
    return [[0, 0, 0], [0, 0, 255], [0, 255, 0]]