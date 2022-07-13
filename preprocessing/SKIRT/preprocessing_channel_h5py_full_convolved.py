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

H5PY_PATH_SOURCE = "../../data/Denoising/TNG_channel_a64_sdss99_full_r%1d.h5"
H5PY_PATH_CONVOLVED = "../../data/Denoising/TNG_channel_a64_sdss99_full_convolved_r%1d.h5"
BOUND = 64
        
size = 20  # on each side from the center
sigma_psf = 2.0
y, x = np.mgrid[-size:size+1, -size:size+1]
psf = np.exp(-(x**2 + y**2)/(2.0*sigma_psf**2))
psf /= np.sum(psf)
    
def save_part(rank):
  
    with h5py.File(H5PY_PATH_SOURCE % rank, 'r') as hf_source:
        with h5py.File(H5PY_PATH_CONVOLVED % rank, 'a') as hf_convolved:
        
            for key in tqdm(hf_source.keys()):
                
                group_selector = hf_source[str(key)]
                data = group_selector['data'][:]
                
                if data.shape[0] > 1024:
                    continue
                
                orig = np.array(data)
                data[...,0] = convolve(data[...,0], psf)
                data[...,1] = convolve(data[...,1], psf)
                data[...,2] = convolve(data[...,2], psf)
                data[...,3] = convolve(data[...,3], psf)
                
                grp = hf_convolved.create_group(str(key))
                
                image_data = grp.create_dataset(
                    name='convolved', data=data.astype('float32'),
                    shape=data.shape, compression="gzip")
                
                image_data = grp.create_dataset(
                    name='orig', data=orig.astype('float32'),
                    shape=data.shape, compression="gzip")
    
    return 0

if __name__ == '__main__':
    
    with Pool(processes=1) as pool:

        _ = pool.starmap(save_part, [(0,)])
        
        print('finished')
