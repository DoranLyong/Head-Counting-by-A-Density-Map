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


from backbones.mobilenetv2 import MobileNetV2, InvertedResidual


# Design U-Net with MobileNet_V2
# ------------------------------
class MobileNetV2_UNET(nn.Module):
    def __init__(self, pre_trained:str=osp.join('weights', 'mobilenet_v2.pth.tar'), mode:str='train'):
        super(MobileNetV2_UNET, self).__init__()

        self.mode = mode 
        




if __name__ == "__main__":
    print("ready")

    model = MobileNetV2_UNET()