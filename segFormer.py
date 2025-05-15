import config as config
import utils as utils
import utils_eye
import model as modelDef

import numpy as np
import matplotlib.pyplot as plt



np.random.seed(config.seed)

image_processor, train_dataloader, valid_dataloader = utils.import_data()

model = modelDef.ModelNN()

model.load_model()
model.setup_lora()
# model.set_weights_for_training()

if config.train_model:
    model.train(image_processor, train_dataloader, valid_dataloader=valid_dataloader)
    model.save()
else:
    model.model.to(model.device)
    val_loss = model.validate(valid_dataloader)
    print(f'Validation Loss: {val_loss}')


# Test on some images
model.inference_dataset(image_processor, valid_dataloader, "valid_results", utils_eye.ade_palette())
