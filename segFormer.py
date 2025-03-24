import config as config
import utils as utils
import utils_eye
import model as modelDef

#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from accelerate.utils import write_basic_config
from utils_eye import load_data


np.random.seed(config.seed)

image_processor, train_dataloader, valid_dataloader = utils.import_data()

model = modelDef.ModelNN()

model.load_model()
model.setup_lora()
# model.set_weights_for_training()

if config.train_model:
    model.train(image_processor, train_dataloader, valid_dataloader=valid_dataloader)
    model.save()


# Test on some images
model.inference_dataset(image_processor, valid_dataloader, "valid_results", utils_eye.ade_palette())
