# SegFormer for Eye Segmentation
SegFormer is a test project to test the possibilities of vision transformer segmentation methods.
It is mainly based on the tutorial from `https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SegFormer/Fine_tune_SegFormer_on_custom_dataset.ipynb`.
The training is based on [lora transformer fine tuning](https://huggingface.co/docs/peft/main/conceptual_guides/lora).

## Questions
* How to use them?
* Are they better than CNN-UNets

## Implemented Tools
* [segFormer.py](segFormer.py) train neural network based on [config settings](config.py)
* To test the network on images in a test folder: [inference.py](inference.py)

# Installation
##  Without Docker
Please note that you should use the anaconda environment requirements.txt for training.

## With Docker and VS Code
We use docker in WSL to run tensorflow version 2.17, which is not supported for Windows anymore. 
Some details on how to setup Docker for WSL and run it with VSCode can be found [here](doc/docker.md)

