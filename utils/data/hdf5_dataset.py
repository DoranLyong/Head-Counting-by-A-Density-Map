import os.path as osp

import h5py 
import numpy as np 

from torch.utils.data import Dataset, DataLoader



class H5Dataset(Dataset): 
    # PyTorch dataset for HDF5 files
    # ------------------------------
    def __init__(self, 
                cfg,
                split="train",
                transform=None):
        """
        Args: 
            path: a path to a HDF5 file
        """
        super(H5Dataset, self).__init__()

        basePath = cfg.LISTDATA.BASE_PTH
        if split == "train":
            fileName = cfg.LISTDATA.TRAIN_FILE
        elif split == "valid": 
            fileName = cfg.LISTDATA.TEST_FILE

        path = osp.join(basePath, fileName)

        self.h5 = h5py.File(path, 'r')  # path in read
        self.images = self.h5['images']
        self.labels = self.h5['labels']

        self.T = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx:int):
        img = self.images[idx].transpose(1,2,0) # (C,H,W) -> (H,W,C)
        label = self.labels[idx] # (1, H, W)

        if self.T:
            label = self.labels[idx].squeeze() # (1, H, W) -> (H, W)
            transformed = self.T(image=img, mask=label)
            return transformed["image"], transformed["mask"].unsqueeze(dim=0) 

        return img, label