# Copyright (c) 2022, Benjamin Holzschuh

import astropy
import os

from skimage.transform import resize
from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np
from multiprocessing import *
from tqdm import tqdm
import h5py
import tensorflow as tf
import pickle
from astropy.convolution import convolve

H5PY_PATH_SOURCE = "/mnt/data/denoising/StyleGAN-TNG.h5"
H5PY_PATH_CONVOLVED = "/mnt/data/denoising/StyleGAN-TNG-convolved.h5"
BOUND = 64
        
size = 20  # on each side from the center
sigma_psf = 2.0
y, x = np.mgrid[-size:size+1, -size:size+1]
psf = np.exp(-(x**2 + y**2)/(2.0*sigma_psf**2))
psf /= np.sum(psf)

def transform(data):
    
    if data.shape[0] in [1,2,3,4,5]:
            data = np.transpose(data, (1,2,0))
    
    size_transformed = np.mean(data[...,-1])
    min_size = 2.0
    max_size = 2516.0

    size = np.exp(size_transformed * (np.log(max_size) - np.log(min_size)) + np.log(min_size)) / 2 # cropped original size

    size = min(int(size), 2516)
    size = max(size, 2)

    data = resize(data[...,:4], (size, size, 4))

    return data

def save_part(rank):
  
    with h5py.File(H5PY_PATH_SOURCE, 'r') as hf_source:
        with h5py.File(H5PY_PATH_CONVOLVED, 'a') as hf_convolved:
        
            for key in tqdm(hf_source.keys()):
                
                group_selector = hf_source[str(key)]
                data = group_selector['data'][:]
                z_space = group_selector['z_space'][:]
                w_space = group_selector['w_space'][:]
                data = transform(data)
                data = np.maximum(0, data)
                
                if data.shape[0] > 1024:
                    continue
                
                orig = np.array(data)
                data[...,0] = convolve(data[...,0], psf)
                data[...,1] = convolve(data[...,1], psf)
                data[...,2] = convolve(data[...,2], psf)
                data[...,3] = convolve(data[...,3], psf)
                
                grp = hf_convolved.create_group(str(key))
                
                _ = grp.create_dataset(
                    name='convolved', data=data.astype('float32'),
                    shape=data.shape, compression="gzip")
                
                _ = grp.create_dataset(
                    name='orig', data=orig.astype('float32'),
                    shape=data.shape, compression="gzip")
                
                _ = grp.create_dataset(
                    name='w_space', data=w_space.astype('float32'),
                    shape=w_space.shape, compression="gzip")
                
                _ = grp.create_dataset(
                    name='z_space', data=z_space.astype('float32'),
                    shape=z_space.shape, compression="gzip")
    
    return 0

if __name__ == '__main__':
    
    with Pool(processes=1) as pool:

        _ = pool.starmap(save_part, [(0,)])
        
        print('finished')
