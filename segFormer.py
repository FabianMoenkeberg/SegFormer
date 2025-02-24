
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



np.random.seed(config.seed)

# utils.download_data()

image_processor, train_dataloader, valid_dataloader = utils.importData()

model = modelDef.ModelNN()

model.load_model()

model.train(image_processor, train_dataloader)

image = Image.open('/app/ADE20k_toy_dataset/images/training/ADE_train_00000001.jpg')
pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(model.device)
prediction = model.inference(image_processor, image)

utils.export_single_result(prediction, image, "result_test.png", "")
