
import sys
import random
import warnings
from zipfile import ZipFile
import json 
import shutil
import re
import config as config

#import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import cv2
#from tqdm import notebook, tnrange
import glob

from datasets import load_dataset
import requests, zipfile, io
from torch.utils.data import Dataset
import torch
import os
from PIL import Image, ExifTags
from transformers import SegformerImageProcessor

from torch.utils.data import DataLoader
from torch import Tensor
from torchvision.transforms import Compose, ColorJitter, ToTensor, Lambda
from albumentations.pytorch import ToTensorV2
from transformers.image_utils import to_numpy_array
import albumentations

def reduce_label(label: Tensor) -> np.ndarray:
        label = to_numpy_array(label)
        # Avoid using underflow conversion
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255
        return label

def visualize_seg_mask(image: np.ndarray, mask: np.ndarray):
   color_seg = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
   palette = np.array(ade_palette())
   for label, color in enumerate(palette):
       color_seg[mask == label, :] = color

   color_seg = color_seg[..., ::-1]  # convert to BGR
   img = np.array(image) * 0.5 + color_seg * 0.5  # plot the image with the segmentation map
   img = img.astype(np.uint8)
   plt.figure(figsize=(15, 10))
   plt.imshow(img)
   plt.axis("off")
   plt.show()

class ADE20KDataset(Dataset):
    def __init__(self, root_dir, processor, size=(512, 512)):
        self.root_dir = root_dir
        self.processor = processor
        self.size = size

        # Find all image files recursively
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "**", "*.jpg"), recursive=True))
        self.mask_paths = [p.replace(".jpg", ".png") for p in self.image_paths]  # Corresponding masks

        # Ensure mask files exist
        self.image_paths, self.mask_paths = zip(*[
            (img, mask) for img, mask in zip(self.image_paths, self.mask_paths) if os.path.exists(mask)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        # Apply Segformer preprocessing
        processed = self.processor(image, segmentation_maps=mask, size=self.size, return_tensors="pt")

        return {
            "pixel_values": processed["pixel_values"].squeeze(0),  # (3, H, W)
            "labels": processed["labels"].squeeze(0)  # (H, W)
        }

def transformsA(examples):
    examples["pixel_values"] = [image.convert("RGB").resize((100,100)) for image in examples["image"]]
    examples["labels"] = [annotation.convert("RGB").resize((100, 100)) for annotation in examples["annotation"]
    ]
    return examples

transformBase = albumentations.Compose(
    [
        # ToTensor(),
        albumentations.Resize(config.im_width, config.im_height),
        albumentations.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        albumentations.ToRGB(),
        ToTensorV2()
    ]
)

def transforms(examples):
    transformed_images, transformed_masks = [], []

    for image, seg_mask in zip(examples["image"], examples["annotation"]):
        image, seg_mask = np.array(image, np.float32)/255.0, np.array(seg_mask, np.int64)
        # image, seg_mask = ToTensor()(image), ToTensor()(seg_mask)
        transformed = transformBase(image=image, mask=seg_mask)
        transformed_images.append(transformed["image"])
        transformed_masks.append(reduce_label(transformed["mask"].long()))
    examples["pixel_values"] = transformed_images
    examples["labels"] = transformed_masks

    del examples["image"]
    del examples["annotation"]
    return examples

jitter = Compose(
    [
         ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.2),
         ToTensor(),
    ]
)

def transformsB(examples):
    examples["pixel_values"] = [jitter(image.convert("RGB")) for image in examples["image"]]
    return examples


def importData(root_dir: str ='/app/ADE20k_toy_dataset'):

    if config.load_entire_dataset:#False:#
        image_processor = SegformerImageProcessor(reduce_labels=True)
        # dataset = load_dataset("scene_parse_150", split="train+test")
        # dataset = dataset.train_test_split(test_size=0.1)
        # train_dataset = dataset['train']
        # valid_dataset = dataset['test']
        train_dataset = load_dataset("scene_parse_150", split="train")
        valid_dataset = load_dataset("scene_parse_150", split="validation")
        # test_dataset = load_dataset("scene_parse_150", "instance_segmentation", split="test")
        train_dataset.set_transform(transforms)
        valid_dataset.set_transform(transforms)
    else:
        image_processor = SegformerImageProcessor(reduce_labels=True)

        # train_dataset = ADE20KDataset(os.path.join(root_dir, 'training'), image_processor)
        # valid_dataset = ADE20KDataset(os.path.join(root_dir, 'validation'), image_processor)

        train_dataset = SemanticSegmentationDataset(root_dir=root_dir, image_processor=image_processor)
        valid_dataset = SemanticSegmentationDataset(root_dir=root_dir, image_processor=image_processor, train=False)
    train_dataset[0]
    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(valid_dataset))

    # encoded_inputs = train_dataset[0]
    # print(encoded_inputs["pixel_values"].shape)
    # print(encoded_inputs["labels"].shape)
    # print(encoded_inputs["labels"])
    # print(encoded_inputs["labels"].squeeze().unique())

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size)

    batch = next(iter(train_dataloader))

    for k,v in batch.items():
        print(k, v.shape)

    print(batch["labels"].shape)

    mask = (batch["labels"] != 255)
    print(mask)
    print(batch["labels"][mask])

    return image_processor, train_dataloader, valid_dataloader


def download_data():
    url = "https://www.dropbox.com/s/l1e45oht447053f/ADE20k_toy_dataset.zip?dl=1"
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()


class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir: str, image_processor: SegformerImageProcessor, train: bool=True):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            image_processor (SegFormerImageProcessor): image processor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.root_dir = root_dir
        self.image_processor = image_processor
        self.train = train

        sub_path = "training" if self.train else "validation"
        self.img_dir = os.path.join(self.root_dir, "images", sub_path)
        self.ann_dir = os.path.join(self.root_dir, "annotations", sub_path)

        # read images
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
            image_file_names.extend(files)
        self.images = sorted(image_file_names)

        # read annotations
        annotation_file_names = []
        for root, dirs, files in os.walk(self.ann_dir):
            annotation_file_names.extend(files)
        self.annotations = sorted(annotation_file_names)

        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.img_dir, self.images[idx]))
        segmentation_map = Image.open(os.path.join(self.ann_dir, self.annotations[idx]))

        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.image_processor(image, segmentation_map, return_tensors="pt")

        for k,v in encoded_inputs.items():
            encoded_inputs[k].squeeze_() # remove batch dimension

        return encoded_inputs
    

def ade_palette():
    """ADE20K palette that maps each class to RGB values."""
    return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]
     

def export_single_result(predicted_segmentation_map, image: Image.Image, output_name: str, path: str):
    color_seg = np.zeros((predicted_segmentation_map.shape[0],
                      predicted_segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3

    palette = np.array(ade_palette())
    for label, color in enumerate(palette):
        color_seg[predicted_segmentation_map == label, :] = color
    # Convert to BGR
    color_seg = color_seg[..., ::-1]

    # Show image + mask
    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)

    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.savefig(os.path.join(path, output_name))
    plt.close()


def export_resuls(predicted_maps: list[Image.Image], images: list[Image.Image], path: str):

    for index, (pred, image) in enumerate(zip(predicted_maps, images)):
        export_single_result(pred, image, f'result_{index}.png', path)