""" A tool to download and preprocess data, and generate HDF5 file. 
this code is inspired by 'https://github.com/NeuroSYS-pl/objects_counting_dmap/blob/master/get_data.py'

If you want to know more about HDF5, 
check 'https://realpython.com/storing-images-in-python/'

Available datasets: 
    - mall: http://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html
    - ShanghaiTech: https://www.kaggle.com/datasets/tthien/shanghaitech-with-people-density-map?resource=download
"""
#%%
import os 
import os.path as osp 
from typing import List, Tuple

import click 
import h5py 
import numpy as np 
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
import albumentations as A


# %%
def process_data(dataset: str): 
    """
    Get chosen dataset and generate HDF5 files with training
    and validation samples.
    """
    # dictionary-based switch statement
    {
        'mall': generate_mall_data,
        'shanghai': generate_shanghai_data,
    }[dataset]()

#%%
def generate_mall_data():
    """Generate HDF5 files for mall dataset."""

    # create training and validation HDF5 files
    # -----------------------------------------
    train_h5, valid_h5 = create_hdf5('mall',
                                     train_size=1500,
                                     valid_size=500,
                                     img_size=(480, 640),
                                     in_channels=3) 

def generate_shanghai_data():
    """Generate HDF5 files for ShanghaiTech dataset."""

    # create training and validation HDF5 files
    # -----------------------------------------
    train_h5, valid_h5 = create_hdf5('shanghai',
                                     train_size=300,
                                     valid_size=100,
                                     img_size=(480, 640),
                                     in_channels=3)

    # load labels infomation from provided MATLAB file
    # it is a numpy array with (x, y) objects position for subsequent frames
    # ---------------------------------------------------------------------
    path = f"../Dataset/ShanghaiTech/part_B/train_data/ground-truth"

    # 파일들을 이름순으로 불러와서 
    # 리스트에 저장
    labels = [] 
    for file in Path 경로에 있는 것 이름 순서대로: 

        labels.append(loadmat(f'{path}/{file}.mat')['image_info'][0][0][0][0][0])


#%% 
def create_hdf5(dataset_name: str,
                train_size: int,
                valid_size: int,
                img_size: Tuple[int, int],
                in_channels: int=3):
    
    # create output folder if it does not exist
    os.makedirs(dataset_name, exist_ok=True)

    # create HDF5 files: [dataset_name]/(train | valid).h5
    train_h5 = h5py.File(osp.join(dataset_name, 'train.h5'), 'w')
    valid_h5 = h5py.File(osp.join(dataset_name, 'valid.h5'), 'w')

    # add two HDF5 datasets (images and labels) for each HDF5 file
    for h5, size in ((train_h5, train_size), (valid_h5, valid_size)):
        h5.create_dataset('images', (size, in_channels, *img_size))
        h5.create_dataset('labels', (size, 1, *img_size))

    return train_h5, valid_h5
