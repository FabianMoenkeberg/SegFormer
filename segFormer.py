
import os
import sys
import random
import warnings
from zipfile import ZipFile

import debugpy
import config as config
import utils as utils
import model as modelDef

#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import cv2

from datetime import datetime
from accelerate.utils import write_basic_config



np.random.seed(config.seed)

# utils.download_data()

image_processor, train_dataloader, valid_dataloader = utils.import_data()

model = modelDef.ModelNN()

model.load_model()
model.setup_lora()

if config.train_model:
    model.train(image_processor, train_dataloader)


# Test on some images
image = Image.open('/app/ADE20k_toy_dataset/images/training/ADE_train_00000001.jpg')
prediction_org = np.array(Image.open('/app/ADE20k_toy_dataset/annotations/training/ADE_train_00000001.png'))
examples = utils.transforms({"image": [image], "annotation": [prediction_org]})
examples_red = image_processor(images=examples["pixel_values"], segmentation_maps=examples["labels"], return_tensors="pt")
image0 = Image.fromarray(np.transpose(np.array(examples["pixel_values"][0]),(1,2,0)).astype(np.uint8))
prediction_red = examples_red['labels'][0]
image_red = Image.fromarray(np.transpose(np.array(examples_red["pixel_values"][0]),(1,2,0)).astype(np.uint8))

# Note: model predicts reduced label
prediction = model.inference(image_processor, image0)
prediction = utils.invert_reduce_label(prediction)
prediction_red1 = utils.invert_reduce_label(prediction_red)


utils.export_single_result(prediction, image, "prediction_test.png", "")
utils.export_single_result(prediction_org, image, "prediction_org.png", "")
utils.export_single_result(prediction_red1, image.resize((512,512)), "prediction_test1.png", "")
