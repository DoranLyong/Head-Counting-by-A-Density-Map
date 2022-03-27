import time 
import os 

import numpy as np 
import hydra
from omegaconf import DictConfig, OmegaConf

import torch 
import torch. nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


from models.unet import UNET
from utils.data import H5Dataset
from looper import Looper


# Set global variables 
# -----------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
seed = int(time.time())
torch.manual_seed(seed)
use_cuda = True
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' # TODO: add to config e.g. 0,1,2,3
    torch.cuda.manual_seed(seed)


@hydra.main(config_path="./cfg", config_name="default")
def main(cfg: DictConfig): 
    print(OmegaConf.to_yaml(cfg))

    ### Create model 
    # ---------------------
    model = UNET(cfg, mode='train')
    model = model.cuda()
    model = nn.DataParallel(model)


    ### Create optimizer & Training scheme & Loss function
    # ---------------------
    loss = nn.MSELoss()

    parameters = model.parameters() 
#    optimizer = optim.Adam(parameters, lr=cfg.TRAIN.LEARNING_RATE, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer = optim.SGD(parameters, lr=cfg.TRAIN.LEARNING_RATE/cfg.TRAIN.BATCH_SIZE, momentum=cfg.SOLVER.MOMENTUM, dampening=0, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    ### Load resume path if necessary 
    # ---------------------



    ### DataLoader 
    # ---------------------

    T_train = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Rotate(limit=35, p=0.5),
            A.VerticalFlip(p=0.5),
#        A.Normalize(mean=[0.43216, 0.394666, 0.37645], 
#                    std=[0.22803, 0.22145, 0.216989], 
#                    max_pixel_value=255.),
            ToTensorV2()
            ])

    T_val = A.Compose([
#        A.Normalize(mean=[0.43216, 0.394666, 0.37645], 
#                    std=[0.22803, 0.22145, 0.216989], 
#                    max_pixel_value=255.),        
        ToTensorV2()])

    dataset = dict()
    dataloader = dict() 
    dataset['train'] = H5Dataset(cfg, split="train", transform=T_train)
    dataset['valid'] = H5Dataset(cfg, split="valid", transform=T_val)

    dataloader['train'] = DataLoader(dataset['train'], batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, 
                            num_workers=cfg.DATA_LOADER.NUM_WORKERS, drop_last=True, pin_memory=True)

    dataloader['valid'] = DataLoader(dataset['valid'], batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.DATA_LOADER.NUM_WORKERS, drop_last=False, pin_memory=True)


    ### Training and Testing Schedule 
    # ---------------------
    plots = [None] * 2

    train_looper = Looper(model, device, loss, optimizer,
                          dataloader['train'], len(dataset['train']), plots[0])
    valid_looper = Looper(model, device, loss, optimizer,
                          dataloader['valid'], len(dataset['valid']), plots[1],
                          validation=True)


    # current best results (lowest mean absolute error on validation set)
    dataset_name = "mall"
    network_architecture = "mobilenetV2_UNET"


    current_best = np.infty

    for epoch in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH + 1):
        print(f"Epoch {epoch + 1}\n")

        # run training epoch and update learning rate
        train_looper.run()
        lr_scheduler.step()

        # run validation epoch
        with torch.no_grad():
            result = valid_looper.run()

        # update checkpoint if new best is reached
        if result < current_best:
            current_best = result
            torch.save(model.state_dict(),
                       f'{dataset_name}_{network_architecture}.pth')

            print(f"\nNew best result: {result}")

        print("\n", "-"*80, "\n", sep='')

    print(f"[Training done] Best result: {current_best}")






if __name__ == "__main__":
    main()