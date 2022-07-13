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
from PIL import Image

BOUND = 256
DATA_ROOT = '../../data/COSMOS/cosmos_galaxy_postage_stamps'
STATS_FILE = f'../../data/COSMOS/COSMOS_{BOUND}_stats.p'
H5PY_PATH = f'../../data/COSMOS/COSMOS_{BOUND}_p%1d.h5'
TFRECORDS_PATH = f'../../data/COSMOS/COSMOS_{BOUND}_r%1dp%1d.tfrecord'
SAMPLE_PATH = f'../../data/COSMOS/COSMOS_{BOUND}_samples'
NUM_SAMPLES = 50
PARTS_TFRECORDS = 4
PARTS_H5PY = 1

FILES = os.listdir(DATA_ROOT)
FILES = [file for file in FILES if file.endswith('_processed.fits.gz')]

def crop(data, resolution=BOUND):
    result = [] 
    center = int(data.shape[0] / 2)
    start = center-int(resolution/2)
    end = center+int(resolution/2)
    temp = data[start:end, start:end]
    
    return temp

def estimate_mean(file_list):
    
    mean_acc = 0
    count = 0
    
    for fit_file in tqdm(file_list):
        
        with fits.open(os.path.join(DATA_ROOT, fit_file)) as hdul:
            orig_size = hdul[0].data.shape[1]
            image = hdul[0].data
        
            
        if orig_size < BOUND:
            continue
            
        count += 1
        mean_acc += np.mean(image, axis=(0,1))
        
        
    return mean_acc / count

class FitReader:
    def __init__(self):
        pass
        
    
    def open_fit_file(self, fit_file):
        with fits.open(os.path.join(DATA_ROOT, fit_file)) as hdul:
            orig_size = hdul[0].data.shape[1]

            if orig_size < BOUND:
                return None

            image = crop(hdul[0].data)
            
        return image[:,:,None]
    
    def close(self):
        pass
    
def create_tfrecord(_image, _writer):
    # image = open(image_dir + image_file, mode='rb').read()
    # image_decoded = tf.image.decode_image(image)

    ex = tf.train.Example(features=tf.train.Features(feature={
        'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=_image.shape)),
        'label': tf.train.Feature(float_list=tf.train.FloatList(value=[0])),
        'data': tf.train.Feature(float_list=tf.train.FloatList(value=_image.reshape(-1)))
    }))
    _writer.write(ex.SerializeToString())

def save_part_tfrecords(file_list, rank, mean):
    
    n = rank
    images = []
    writer_dict = {}

    for resolution_level in range(2, 8 + 1):
        writer_dict[resolution_level] = tf.io.TFRecordWriter(TFRECORDS_PATH % (resolution_level,n))

    reader = FitReader()
    vars_ = []
    
    for fit_file in tqdm(file_list):
        
        image = reader.open_fit_file(fit_file)
        
        if image is None: 
            continue
        
        var = np.mean(np.square(image - mean), axis=(0,1)) 
        vars_.append(var)
     
        for resolution_level in range(2,8+1):

        
            resolution = image.shape[0] // (2 ** (8 - resolution_level))
            downsampled_image = resize(image, (resolution, resolution), order=1)            
            downsampled_image = np.transpose(downsampled_image, (2,0,1))
            create_tfrecord(downsampled_image, writer_dict[resolution_level])

    for resolution_level in range(2,8+1):
        writer_dict[resolution_level].close()
    
    vars_ = np.array(vars_)
    vars_ = np.mean(vars_, axis=0)
        
    reader.close()
    
    return vars_
    
def save_part_h5py(file_list, rank, mean):

    images = []
    
    reader = FitReader()
    vars_ = []

    with h5py.File(H5PY_PATH % rank, 'a') as hf:
        
        for fit_file, n in tqdm(zip(file_list, range(len(file_list)))):

            image = reader.open_fit_file(fit_file)
            if image is None:
                continue

            var = np.mean(np.square(image - mean), axis=(0,1)) 
            vars_.append(var)

            image_shape = image.shape

            grp = hf.create_group(str(n))

            image_data = grp.create_dataset(
                name='data', data=image.astype('float32'),
                shape=image.shape, maxshape=image_shape, compression="gzip")
                
    vars_ = np.array(vars_)
    vars_ = np.mean(vars_, axis=0)
    
    reader.close()
    
    return vars_

def save_samples(file_list):
    
    reader = FitReader()
    
    for fit_file, n in tqdm(zip(file_list, range(len(file_list)))):
        
        image = reader.open_fit_file(fit_file)
        if image is None:
            continue
        
        cm = plt.cm.ScalarMappable(None, cmap='magma')
        cmap = cm.get_cmap()
        image_ = image[:,:,0]
        image_ = (image_ - np.min(image_)) / (np.max(image_) - np.min(image_))
       
        rgb_img = cmap(image_)[..., :3]
        rgb_img = rgb_img * 255
        rgb_img = rgb_img.astype(np.uint8)

        id = n
        result = Image.fromarray(rgb_img)
        result.save(os.path.join(SAMPLE_PATH, f'{id}.png'))
        with open(os.path.join(SAMPLE_PATH, f'{id}.npy'), 'wb') as handle:
            
            np.save(handle, image)

def splitList(list_, parts:int):
    splitted_list = []
    for i in range(parts):
        splitted_list.append(list_[i::parts])
    return splitted_list

if __name__ == '__main__':
     
    save_samples(FILES[:NUM_SAMPLES])
    
    num_proccesses_mean = 4
    print('Computing mean...')
    with Pool(processes=num_proccesses_mean) as pool:
        files = splitList(FILES, num_proccesses_mean)
        means_per_file_list = pool.map(estimate_mean, files)
        mean = np.zeros_like(means_per_file_list[0])
        for e in means_per_file_list: 
            mean += e
        mean = mean / len(means_per_file_list)
        
    print('Computing variance and saving galaxies')
    
    print('preparing TFRECORDS...')
    with Pool(processes=PARTS_TFRECORDS) as pool:

        files = splitList(FILES, PARTS_TFRECORDS)
        
        print('Computing variance and saving galaxies')
        
        vars_per_file_list = pool.starmap(save_part_tfrecords, zip(files, range(PARTS_TFRECORDS), [mean] * PARTS_TFRECORDS))
        
        var = np.zeros_like(vars_per_file_list[0])
        for e in vars_per_file_list:
            var += e
        var = var / len(vars_per_file_list)
        std = np.sqrt(var)
        
        print('mean and variance tfrecords: ', [mean,std])
        
    print('preparing H5PY...')
    with Pool(processes=PARTS_H5PY) as pool:

        files = splitList(FILES, PARTS_H5PY)
        
        print('Computing variance and saving galaxies')
        
        vars_per_file_list = pool.starmap(save_part_h5py, zip(files, range(PARTS_H5PY), [mean] * PARTS_H5PY))
        
        var = np.zeros_like(vars_per_file_list[0])
        for e in vars_per_file_list:
            var += e
        var = var / len(vars_per_file_list)
        std = np.sqrt(var)
        
        print('mean and std h5py: ', [mean,std])
    
    
    with open(STATS_FILE, 'wb') as handle:
        pickle.dump([mean, std], handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print('finished')