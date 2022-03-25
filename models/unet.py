'''
code inspired by https://github.com/roeiherz/MobileUNET/blob/master/nets/MobileNetV2_unet.py

more information:
- https://idiotdeveloper.com/unet-segmentation-with-pretrained-mobilenetv2-as-encoder/
- https://www.kaggle.com/code/balraj98/unet-for-building-segmentation-pytorch/notebook
- https://github.com/zym1119/DeepLabv3_MobileNetv2_PyTorch/blob/master/network.py
'''


import logging 
import math 
import sys 
import os 
import os.path as osp
from turtle import forward

import torch 
import torch.nn as nn 

from .backbones.mobilenetv2 import MobileNetV2, InvertedResidual


# Design U-Net 
# ------------------------------
class UNET(nn.Module):
    def __init__(self, cfg, mode='train'):
        super(UNET, self).__init__()
        self.cfg = cfg 
        self.mode = mode

        # Backbone as Encoder(Downsampling)
        # --------------------------
        if cfg.MODEL.BACKBONE == "mobilenetv2":
            self.backbone = MobileNetV2()
        
        else: 
            raise ValueError(f"Wrong backbone model is requested.")



        # Up-sampling as Decoder
        # ----------------------------
        upsampling_setting = [
            [1280, 96, 192, 96],
            [  96, 32, 64,  32],
            [  32, 24, 48,  24],
            [  24, 16, 32,  16]
        ]

        self.dconv = nn.ModuleList()
        self.invres = nn.ModuleList()

        for din, dout, ivin, ivout in upsampling_setting: 
            self.dconv.append(nn.ConvTranspose2d(din, dout, kernel_size=4, padding=1, stride=2))
            self.invres.append(InvertedResidual(inp=ivin, oup=ivout, stride=1, expand_ratio=6))

        self.interpolate = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Density prediction  
        # -----------------------------
        self.conv_last = nn.Conv2d(16, 3, 1) # 1x1 conv; 16-channel -> 3-channel
        self.density_pred = nn.Conv2d(3, 1, 1, bias=False) 


        # init. weigths 
        # ----------------------------
        self._init_weights()

        if cfg.WEIGHTS.BACKBONE:
            print("load pretrained weights")


    def forward(self, x:torch.Tensor):
        assert x.ndimension() == 4, "input tensor should be 4D, but given {n.ndimension()}D."

        # Downsampling 
        # ------------------- 
        for n in range(0, 2):
            x = self.backbone.features[n](x) 
        x1 = x # (3, 224, 224) -> (16, 112, 112)
        
        for n in range(2, 4): 
            x = self.backbone.features[n](x)
        x2 = x # (16, 112, 112) -> (24, 56, 56) 
        
        for n in range(4, 7):
            x = self.backbone.features[n](x)
        x3 = x # (24, 56, 56) -> (32, 28, 28)
        
        for n in range(7, 14): 
            x = self.backbone.features[n](x)
        x4 = x # (32, 28, 28) -> (96, 14, 14)
        
        for n in range(14, 19): 
            x = self.backbone.features[n](x)
        x5 = x # (96, 14, 14) -> (1280, 7, 7)

        
        # Upsampling 
        # -------------------
        up1 = torch.cat([x4, self.dconv[0](x)], dim=1) # (96*2, 14, 14)
        up1 = self.invres[0](up1) # (96, 14, 14)
        
        up2 = torch.cat([x3, self.dconv[1](up1)], dim=1) # (32*2, 28, 28)
        up2 = self.invres[1](up2) # (32, 28, 28)
        
        up3 = torch.cat([x2, self.dconv[2](up2)], dim=1) # (24*2, 56, 56)
        up3 = self.invres[2](up3) # (24, 56, 56)
        
        up4 = torch.cat([x1, self.dconv[3](up3)], dim=1) # (16*2, 112, 112)
        up4 = self.invres[3](up4) # (16, 112, 112)

        # Density prediction 
        # -----------------------
        x = self.conv_last(up4) # (3, 112, 112)
        x = self.density_pred(x) # (1, 112, 112)

        x = self.interpolate(x) # (1, 112, 112) -> (1, 224, 224)            
        return x 

    def _init_weights(self): 
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels # # kernel elements; h x w x c
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None: 
                    m.bias.data.zero_() 
            elif isinstance(m, nn.BatchNorm2d): 
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear): 
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == "__main__":
    # checking 
    # # (ref) https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#from-a-yaml-file
    # -------------- 
    from omegaconf import DictConfig, OmegaConf 
    from backbones.mobilenetv2 import MobileNetV2, InvertedResidual

    cfg = OmegaConf.load('../cfg/default.yaml')
    
    model = UNET(cfg, mode='eval')
#    model = UNET(cfg, mode='train')
    
    input = torch.randn(1, 3, 480, 640)
    output = model(input)
    print(output.size())

    assert input.shape[2:] == output.shape[2:], "input shape != ouput shape"
    
