import os
import yaml
import json 
import argparse
import torchvision.utils as vutils 
from torchvision import transforms
import cv2
cv2.setNumThreads(0)

def get_config(config_Path):
    #
    with open(config_Path) as f:
        config = yaml.safe_load(f)
    config = json.dumps(config)
    config = json.loads(config, object_hook=lambda d: argparse.Namespace(**d))
    return config


to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    
to_tensor =  transforms.Compose([
        transforms.ToTensor(),
    ])


def img_read(imag_Path):
    img = cv2.cvtColor(
        cv2.imread(imag_Path),
        cv2.COLOR_BGR2RGB
    )
    return to_tensor_norm(img)