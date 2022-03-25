import time 
import os 
import os.path as osp 

import cv2 
import numpy as np 
import hydra
from omegaconf import DictConfig, OmegaConf

import torch 


from models.unet import UNET



@hydra.main(config_path="./cfg", config_name="default")
def main(cfg: DictConfig): 
    print(OmegaConf.to_yaml(cfg))

    ### Create model 
    # ---------------------
    model = UNET(cfg)


    ### Create optimizer 
    # ---------------------


    ### Load resume path if necessary 
    # ---------------------


    ### DataLoader & Training scheme & Loss function
    # ---------------------



    ### Training and TEsting Schedule 
    # ---------------------







if __name__ == "__main__":
    main()