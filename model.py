import config
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import json
from huggingface_hub import hf_hub_download
import evaluate
import torch
from torch import nn, optim
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
from transformers import SegformerImageProcessor, SegformerFeatureExtractor, SegformerForImageClassification
from torch.utils.data import DataLoader
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
from numpy.typing import NDArray

class ModelNN:
    def __init__(self) -> None:
        self.model = None
        self.id2label = None
        self.repo_id = "huggingface/label-files"
        self.filename = "ade20k-id2label.json"
        # self.modelname = "nvidia/mit-b0"
        self.modelname = "nvidia/segformer-b2-finetuned-ade-512-512"
        # self.modelname = "nvidia/mit-b0"

        self.device = None

        self.metric = evaluate.load("mean_iou")

    def load_model(self) -> None:
        # load id2label mapping from a JSON on the hub
        self.id2label = json.load(open(hf_hub_download(repo_id=self.repo_id, filename=self.filename, repo_type="dataset"), "r"))
        self.id2label = {int(k): v for k, v in self.id2label.items()}
        label2id = {v: k for k, v in self.id2label.items()}

        # define model
        model = SegformerForSemanticSegmentation.from_pretrained(self.modelname,
                                                                num_labels=150,
                                                                id2label=self.id2label,
                                                                label2id=label2id,
        )

        self.model = model

    def set_weights_for_training(self) -> None:
        # Freeze all model layers except the last one
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the final classification head
        for param in self.model.decode_head.parameters():
            param.requires_grad = True

    def setup_lora(self) -> None:
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters before Lora: {trainable_params}")

        # Define LoRA Configuration
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="lora_only",
            modules_to_save=["decode_head"],
        )

        self.model = get_peft_model(self.model, lora_config)
        print("Trainable parameters after Lora:")
        self.model.print_trainable_parameters()

    def train(self, image_processor: SegformerImageProcessor, train_dataloader: DataLoader, valid_dataloader: DataLoader = None) -> None:
        image_processor.do_reduce_labels

        # define optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=config.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True)

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
                # get the inputs;
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(pixel_values=pixel_values, labels=labels)
                loss, logits = outputs.loss, outputs.logits

                loss.backward()
                optimizer.step()

                # evaluate
                with torch.no_grad():
                    upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                    predicted = upsampled_logits.argmax(dim=1)

                    # note that the metric expects predictions + labels as numpy arrays
                    self.metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

                # let's print loss and metrics every 100 batches
                if idx % 100 == 0:
                    # currently using _compute instead of compute
                    # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
                    metrics = self.metric._compute(
                            predictions=predicted.cpu(),
                            references=labels.cpu(),
                            num_labels=len(self.id2label),
                            ignore_index=255,
                            reduce_labels=False, # we've already reduced the labels ourselves
                        )

                    print("Loss:", loss.detach().cpu().item())
                    print("Mean_iou:", metrics["mean_iou"])
                    print("Mean accuracy:", metrics["mean_accuracy"])

                running_tloss += loss.detach().cpu()*pixel_values.size(0)
                n += pixel_values.size(0)
            
            running_tloss /= len(train_dataloader)*config.batch_size
            
            if valid_dataloader:
                running_vloss = 0.0
                # Set the model to evaluation mode, disabling dropout and using population
                # statistics for batch normalization.
                self.model.eval()

                # Disable gradient computation and reduce memory consumption.
                with torch.no_grad():
                    for i, vdata in enumerate(valid_dataloader):
                        pixel_values = vdata["pixel_values"].to(self.device)
                        labels = vdata["labels"].to(self.device)
                        voutputs = self.model(pixel_values=pixel_values, labels=labels)
                        # loss, logits = voutputs.loss, voutputs.logits
                        # upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                        # predicted = upsampled_logits.argmax(dim=1)

                        # # note that the metric expects predictions + labels as numpy arrays
                        # self.metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

                        vloss = voutputs.loss.detach().cpu()
                        running_vloss += vloss*pixel_values.size(0)

            running_vloss /= len(valid_dataloader)*config.batch_size
            scheduler.step(running_vloss)

            print(f'Epoch {epoch} \t\t Training Loss: {running_tloss} \t\t Validation Loss: {running_vloss}')

    def inference(self, image_processor: SegformerImageProcessor, image: Image.Image) -> NDArray:
        pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(self.device)

        # forward pass
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)

        logits = outputs.logits.cpu()
        print(logits.shape)
        
        predicted_segmentation_map = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()
        print(predicted_segmentation_map)
        return predicted_segmentation_map
    
    def save(self, name: str = "segformer_model") -> None:
        self.model.save_pretrained(name + "_lora_adapter")
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(name + "full_finetuned")

    def load_lora_separate(self, name: str = "segformer_model_lora_adapter") -> None:
        self.load_model()
        lora_model = PeftModel.from_pretrained(self.model, name)

        self.model = lora_model
