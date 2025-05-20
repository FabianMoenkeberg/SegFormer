import torchmetrics.classification
import transformers
import config
from transformers import SegformerForSemanticSegmentation, AutoImageProcessor, Mask2FormerConfig, Mask2FormerForUniversalSegmentation
import json
from huggingface_hub import hf_hub_download
import datasets
from torcheval.metrics import BinaryF1Score
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy
from torchmetrics import MetricCollection
import torch
from torch import nn, optim
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
from transformers import SegformerImageProcessor, SegformerFeatureExtractor, SegformerForImageClassification
from torch.utils.data import DataLoader
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
from numpy.typing import NDArray
import os
import cv2
import numpy as np

class ModelNN:
    def __init__(self) -> None:
        self.model = None
        self.id2label = None
        self.repo_id = "huggingface/label-files"
        self.filename = "ade20k-id2label.json"
        # self.modelname = "nvidia/mit-b0"
        self.modelname = "nvidia/segformer-b2-finetuned-ade-512-512"

        self.device = None
        self.useLora = True
        self.metric = MetricCollection({
            "mean_accuracy": MulticlassAccuracy(num_classes=config.N_segClasses+1, average='macro'),
            "f1": MulticlassF1Score(num_classes=config.N_segClasses+1, average='macro')
            })

        self.image_name = "pixel_values"
        self.label_name = "labels"
        self.label_class = None

        if config.GPU_USAGE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

    def get_femto_id_to_label(self) -> dict:
        return {0: 'background', 1: 'pupil', 2: 'iris'}

    def load_model(self) -> None:
        # load id2label mapping from a JSON on the hub
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
        self.id2label = self.get_femto_id_to_label()

        self.id2label = {int(k): v for k, v in self.id2label.items()}
        label2id = {v: k for k, v in self.id2label.items()}

        # define model
        if config.model_type == 'SegFormer':
            if config.load_local_model:
                model = SegformerForSemanticSegmentation.from_pretrained(config.name_loadModel)
            else:
                model = SegformerForSemanticSegmentation.from_pretrained(self.modelname,
                                                                        num_labels=config.N_segClasses+1,
                                                                        id2label=self.id2label,
                                                                        label2id=label2id,
                                                                        ignore_mismatched_sizes = True
        )
        elif config.model_type == 'Mask2Former':
            self.label_name = "mask_labels"
            self.label_class = 'class_labels'
            if config.load_local_model:
                model = Mask2FormerForUniversalSegmentation.from_pretrained(config.name_loadModel)
            else:
                model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-coco-instance",
                                                     num_labels=config.N_segClasses+1,
                                                                        id2label=self.id2label,
                                                                        label2id=label2id,
                                                                        ignore_mismatched_sizes = True
                                                                        )
            # model.config.num_labels = 3  # 3 classes: background, class 1, and class 2
            # model.config.id2label = self.id2label
            # model.config.label2id = label2id
            # model.classifier = torch.nn.Conv2d(model.config.hidden_size, 3, kernel_size=(1, 1))

        self.model = model

    def set_weights_for_training(self) -> None:
        # Freeze all model layers except the last one
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the final classification head
        # for param in self.model.decode_head.parameters():
        #     param.requires_grad = True

        # Freeze the decoder head except for the last layer (classifier)
        for name, param in self.model.decode_head.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False  # Freeze
            else:
                param.requires_grad = True   # Train only classifier

    def setup_lora(self) -> None:
        if self.useLora:
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Trainable parameters before Lora: {trainable_params}")

            # Define LoRA Configuration
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["query", "value"],
                lora_dropout=0.1,
                bias="lora_only",
                modules_to_save=["decode_head.classifier"],
            )

            self.model = get_peft_model(self.model, lora_config)
            print("Trainable parameters after Lora:")
            self.model.print_trainable_parameters()

    def train(self, image_processor: SegformerImageProcessor, train_dataloader: DataLoader, valid_dataloader: DataLoader = None) -> None:
        image_processor.do_reduce_labels

        # define optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=config.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
        loss_fn = nn.CrossEntropyLoss(ignore_index=255)

        # move model to GPU
        if config.GPU_USAGE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)

        self.model.train()
        for epoch in range(config.nEpochs):  # loop over the dataset multiple times
            print("Epoch:", epoch)
            running_tloss = 0.0
            n = 0
            for idx, batch in enumerate(tqdm(train_dataloader)):
                self.model.train()

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs, pixel_values, labels = self.evaluate_batch(batch)

                loss = outputs.loss
                loss.backward()
                optimizer.step()

                # evaluate
                with torch.no_grad():
                    predicted, labels = self.get_prediction(outputs, image_processor, labels)

                    # labels = labels.detach().cpu().numpy()
                    labels.astype(np.int32) 
                    predicted = predicted.astype(np.int32)
                    # note that the metric expects predictions + labels as numpy arrays
                    self.metric.update(torch.tensor(predicted), torch.tensor(labels))

                # let's print loss and metrics every 100 batches
                if idx % 50 == 0:
                    # currently using _compute instead of compute
                    # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
                    metrics = self.metric.compute()

                    print("Loss:", loss.detach().cpu().item())
                    print("f1:", metrics["f1"])
                    print("Mean accuracy:", metrics["mean_accuracy"])
                    self.metric.reset()

                running_tloss += loss.detach().cpu()*pixel_values.size(0)
                n += pixel_values.size(0)
            
            running_tloss /= len(train_dataloader)*config.batch_size
            
            if valid_dataloader:
                running_vloss = self.validate(valid_dataloader)
            
                scheduler.step(running_vloss)

            print(f'Epoch {epoch} \t\t Training Loss: {running_tloss} \t\t Validation Loss: {running_vloss}')

    def evaluate_batch(self, batch: transformers.image_processing_base.BatchFeature) -> tuple:
        # move model to GPU
        # get the inputs;
        pixel_values = batch[self.image_name].to(self.device)
        labels = batch[self.label_name].to(self.device)

        # forward + backward + optimize
        if self.label_class:
            classes = batch[self.label_class].to(self.device)
            outputs = self.model(pixel_values=pixel_values, mask_labels=labels, class_labels=classes)
        else:
            outputs = self.model(pixel_values=pixel_values, labels=labels)

        return outputs, pixel_values, labels

    def get_prediction(self, outputs, image_processor: SegformerImageProcessor, labels=None):
        if config.model_type == 'SegFormer':
            logits = outputs.logits
            upsampled_logits = nn.functional.interpolate(logits, size=[config.im_width, config.im_height], mode="bilinear", align_corners=False)

            predicted = upsampled_logits.argmax(dim=1).detach().cpu().numpy()
            if not labels is None:
                labels = labels.detach().cpu().numpy()
        elif config.model_type == 'Mask2Former':
            shapes = [[config.im_width, config.im_height] for _ in range(outputs.masks_queries_logits.shape[0])]
            predicted = image_processor.post_process_semantic_segmentation(outputs, shapes)
            predicted = np.array([el.detach().cpu().numpy() for el in predicted])
            if labels:
                labels = labels.argmax(dim=1).detach().cpu().numpy()
        return predicted, labels
    
    def get_prediction_val(self, outputs, image_processor: SegformerImageProcessor):
        if config.model_type == 'SegFormer':
            logits = outputs.logits.detach().cpu()
            
            predicted_segmentation_map = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[[config.im_width, config.im_height] for el in range(logits.shape[0])])
            predicted_segmentation_map = [el.detach().cpu().numpy() for el in predicted_segmentation_map]
        elif config.model_type == 'Mask2Former':
            shapes = [[config.im_width, config.im_height] for el in range(outputs.masks_queries_logits.shape[0])]
            predicted_segmentation_map = image_processor.post_process_semantic_segmentation(outputs, shapes)
            predicted_segmentation_map = np.array([el.detach().cpu().numpy() for el in predicted_segmentation_map])

        return predicted_segmentation_map

    def validate(self, valid_dataloader):
        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        self.model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(valid_dataloader):
                voutputs, pixel_values, labels = self.evaluate_batch(vdata)
                
                vloss = voutputs.loss.detach().cpu()
                running_vloss += vloss*pixel_values.size(0)
        
        running_vloss /= len(valid_dataloader)*config.batch_size

        return running_vloss

    def inference_dataset(self, image_processor: SegformerImageProcessor, dataset: DataLoader, output_dir: str, palette: list[list[int]])->None:
        
        self.model.eval()
        os.makedirs(output_dir, exist_ok=True)

        # Set device (make sure to use the same device as your model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        id_img = 0

        with torch.no_grad():
            for batch_idx, vdata in enumerate(dataset):
                inputs = vdata[self.image_name].to(self.device)
                labels = vdata[self.label_name]
                if len(labels.shape)>3:
                    labels = labels.argmax(dim=1)
                outputs = self.inference(image_processor, inputs)
                for i in range(inputs.size(0)):
                     # height, width, 3
                    color_seg = self.convert_colors(palette, outputs[i])
                    color_seg_ref = self.convert_colors(palette, labels[i])

                    image = self.reverse_transform_images(inputs.cpu().numpy()[i], image_processor=image_processor)
                    cv2.imwrite(os.path.join(output_dir, f"results_{id_img}_det.png"), color_seg)
                    cv2.imwrite(os.path.join(output_dir, f"results_{id_img}.png"), image*255)
                    cv2.imwrite(os.path.join(output_dir, f"results_{id_img}_ref.png"), color_seg_ref)
                    id_img+=1


    def reverse_transform_images(self, input: NDArray, image_processor: SegformerImageProcessor):
        image = np.transpose(input, (1,2,0))
        image  = image*image_processor.image_std +image_processor.image_mean
        return image

    def convert_colors(self, palette: list[list[int]], labels: NDArray) -> NDArray:
        color_seg = np.zeros((config.im_width, config.im_height, 3), dtype=np.uint8)

        for label, color in enumerate(palette):
            color_seg[labels == label, :] = color

        # Convert to BGR
        color_seg = color_seg[..., ::-1]

        return color_seg

    def inference(self, image_processor: SegformerImageProcessor, pixel_values: NDArray) -> NDArray:
        # forward pass
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
        
        predicted_segmentation_map = self.get_prediction_val(outputs, image_processor)
        
        return predicted_segmentation_map

    def inference_image(self, image_processor: SegformerImageProcessor, image: Image.Image) -> NDArray:
        pixel_values = image_processor(image, return_tensors="pt")[self.image_name].to(self.device)
        self.model.to(self.device)
        predicted_segmentation_map = self.inference(image_processor, pixel_values)[0]
        print(predicted_segmentation_map)
        return predicted_segmentation_map
    
    def save(self, name: str = config.model_type) -> None:
        self.model.save_pretrained(name + "_lora_adapter")
        if self.useLora:
            merged_model = self.model.merge_and_unload()
            merged_model.save_pretrained(name + "full_finetuned")

    def load_lora_separate(self, name: str = "segformer_model_lora_adapter") -> None:
        self.load_model()
        lora_model = PeftModel.from_pretrained(self.model, name)

        self.model = lora_model
