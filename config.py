import yaml

import torch
from torchvision import transforms

import coder

# General config
# Datasets

def get_model(device):

    predictor = coder.ResNet([4,4,4])
    model = coder.Network(predictor, device=device)
    return model


