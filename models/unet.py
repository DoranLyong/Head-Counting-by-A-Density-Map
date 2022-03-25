'''
inspired by https://github.com/roeiherz/MobileUNET/blob/master/nets/MobileNetV2_unet.py

'''


import logging 
import math 
import sys 
import os 
import os.path as osp

import torch 
import torch.nn as nn 

#from .backbones.mobilenetv2 import MobileNetV2, InvertedResidual
from .backbones.mobilenetv2 import MobileNetV2, InvertedResidual



# Design U-Net 
# ------------------------------
class UNET(nn.Module):
    def __init__(self, cfg):
        super(UNET, self).__init__()
        self.cfg = cfg 

        # Backbone
        # --------------------------
        if cfg.MODEL.BACKBONE == "mobilenetv2":
            self.backbone = MobileNetV2()
        
        else: 
            raise ValueError(f"Wrong backbone model is requested.")

        if cfg.WEIGHTS.BACKBONE:
            print("load pretrained weights")

        # UNET structure 
        # ----------------------------
        

        
        
        




if __name__ == "__main__":
    print("ready")

    model = UNET()