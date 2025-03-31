import config as config
import utils as utils
import utils_eye
import model as modelDef

#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

from accelerate.utils import write_basic_config
from utils_eye import load_data_inference
from transformers import SegformerImageProcessor

path = '/data_eye/eyeSegmentationFemto/topview_hardcases'
out_path = os.path.join(path, "results")
if not os.path.exists(out_path):
    os.mkdir(out_path)

np.random.seed(config.seed)

X_train, id_map = load_data_inference(path_data=path)
X_train /= 255.0
model = modelDef.ModelNN()

image_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
image_processor.do_reduce_labels = False
image_processor.do_rescale = False
image_processor.size['height']=config.im_height
image_processor.size['width']=config.im_width

model.load_model()

# Test on some images
for id, image in enumerate(X_train):
    prediction = model.inference_image(image_processor=image_processor, image=image)
    color_seg = model.convert_colors(utils_eye.ade_palette(), prediction)
    cv2.imwrite(os.path.join(out_path, f"results_{id}.png"), image*255)
    cv2.imwrite(os.path.join(out_path, f"results_{id}_det.png"), color_seg)
