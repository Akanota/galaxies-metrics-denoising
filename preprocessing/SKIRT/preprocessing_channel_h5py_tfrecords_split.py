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
from pathlib import Path

BOUND = 64
SPLIT_FILE = f'../data/TNG_{BOUND}_sdss99_split.p'

DATA_ROOT = os.path.join('..', '..', 'sdss', 'snapnum_099', 'data')
HDF5_ROOT = os.path.join('..', '..', 'sdss', 'snapnum_099')

STATS_FILE = f'../data/%s/%s_sdss99_{BOUND}_stats.p'
H5PY_PATH = f"../data/%s/%s_sdss99_{BOUND}_r%1d.h5"
TFRECORDS_PATH = f"../data/%s/%s_sdss99_{BOUND}_r%1dp%1d.tfrecords"

FITS_FILES = os.listdir(DATA_ROOT)

NUM_SAMPLES = 100

SAMPLE_PATH = f'../data/%s/%s_sdss99_{BOUND}_samples'

PARTS_H5PY = 1
PARTS_TFRECORDS = 4

def size_conversion(size):
    min_size = 2.0
    max_size = 2516.0
    return (np.log(size) - np.log(min_size)) / (np.log(max_size) - np.log(min_size))

def crop_and_resize(data, resolution=256, crop_ratio=0.25):
    result = np.zeros((4,resolution, resolution))
    for n in range(4):
        res_data = data.shape[1]
        start = int(crop_ratio * res_data)
        end = int(((1-crop_ratio) * res_data))
        temp = data[n][start:end, start:end]
        result[n] = resize(temp, (resolution, resolution))
        
    return result

def generate_subfind_to_id_dict():
    n = 0
    dict_map = {}
    with open(os.path.join(HDF5_ROOT,'subfind_ids.txt'), 'r') as file:
        for subfind in file:
            dict_map[int(subfind)] = n
            n += 1
    return dict_map
        
halo_to_id = generate_subfind_to_id_dict()

def create_tfrecord(_image, _labels, _writer):
    # image = open(image_dir + image_file, mode='rb').read()
    # image_decoded = tf.image.decode_image(image)

    ex = tf.train.Example(features=tf.train.Features(feature={
        'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=_image.shape)),
        'label': tf.train.Feature(float_list=tf.train.FloatList(value=_labels)),
        'data': tf.train.Feature(float_list=tf.train.FloatList(value=_image.reshape(-1)))
    }))
    _writer.write(ex.SerializeToString())

def estimate_mean(file_list):
    
    mean_acc = np.zeros(shape=(5,))
    count = 0
    
    for fit_file in tqdm(file_list):
        
        with fits.open(os.path.join(DATA_ROOT, fit_file)) as hdul:
            orig_size = hdul[0].data.shape[1]
            image = crop_and_resize(hdul[0].data)
            image = np.transpose(image, (1,2,0))
            # add 5th channel containing size information
            image = np.concatenate([image, np.ones(image.shape[:-1] + (1,)) * size_conversion(hdul[0].data.shape[1])], axis=2)

            
        if orig_size < BOUND:
            continue
            
        count += 1
        mean_acc += np.mean(image, axis=(0,1))
        
        
    return mean_acc / len(file_list)
        
class FitReader:
    def __init__(self):
        
        self.morphs_g = h5py.File(os.path.join(HDF5_ROOT, 'morphs_g.hdf5'), "r")
        self.morphs_i = h5py.File(os.path.join(HDF5_ROOT, 'morphs_i.hdf5'), "r")
        self.morphs_keys = list(self.morphs_g.keys())
        
    
    def open_fit_file(self, fit_file):
        with fits.open(os.path.join(DATA_ROOT, fit_file)) as hdul:
            orig_size = hdul[0].data.shape[1]

            if orig_size < BOUND:
                return None, None

            image = crop_and_resize(hdul[0].data)

            image = np.transpose(image, (1,2,0))

            # add 5th channel containing size information
            image = np.concatenate([image, np.ones(image.shape[:-1] + (1,)) * size_conversion(orig_size)], axis=2)

        halo_id = int(fit_file[fit_file.find('_')+1:fit_file.find('.')])
        id = halo_to_id[halo_id]

        labels = [self.morphs_i[key][id] for key in self.morphs_keys] + [self.morphs_g[key][id] for key in self.morphs_keys] 
        labels = np.array(labels)
            
        return image, labels
    
    def close(self):
        self.morphs_g.close()
        self.morphs_i.close()
        
def save_part_tfrecords(file_list, rank, mean, split_key):
    
    n = rank
    images = []
    writer_dict = {}

    for resolution_level in range(2, 8 + 1):
        writer_dict[resolution_level] = tf.io.TFRecordWriter(TFRECORDS_PATH % (split_key, split_key, resolution_level,n))

    reader = FitReader()
    vars_ = []
    
    for fit_file in tqdm(file_list):
        
        image, labels = reader.open_fit_file(fit_file)
        if image is None: 
            continue
        
        var = np.mean(np.square(image - mean), axis=(0,1)) 
        vars_.append(var)
     
        for resolution_level in range(2,8+1):

        
            resolution = image.shape[0] // (2** (8 - resolution_level))
            downsampled_image = resize(image, (resolution, resolution), order=1)

            downsampled_image = np.transpose(downsampled_image, (2,0,1))
            
            
            create_tfrecord(downsampled_image, labels, writer_dict[resolution_level])

    for resolution_level in range(2,8+1):
        writer_dict[resolution_level].close()
    
    vars_ = np.array(vars_)
    vars_ = np.mean(vars_, axis=0)
        
    reader.close()
    
    return vars_
    
def save_samples(file_list, split_key):
    
    reader = FitReader()
    
    Path(SAMPLE_PATH % (split_key, split_key)).mkdir(parents=True, exist_ok=True)
    
    for fit_file, n in tqdm(zip(file_list, range(len(file_list)))):
        
        image, labels = reader.open_fit_file(fit_file)
        if image is None:
            continue
        
        r_channel = image[...,1] # r-channel
        cm = plt.cm.ScalarMappable(None, cmap='magma')
        cmap = cm.get_cmap()
        r_channel = (r_channel - np.min(r_channel)) / (np.max(r_channel) - np.min(r_channel))
        rgb_img = cmap(r_channel)[..., :3]
        rgb_img = rgb_img * 255
        rgb_img = rgb_img.astype(np.uint8)
            
        halo_id = int(fit_file[fit_file.find('_')+1:fit_file.find('.')])
        id = halo_to_id[halo_id]

        result = Image.fromarray(rgb_img)
        result.save(os.path.join(SAMPLE_PATH % (split_key, split_key), f'{id}.png'))
        with open(os.path.join(SAMPLE_PATH % (split_key, split_key), f'{id}.npy'), 'wb') as handle:
            np.save(handle, image)
        
def save_part_h5py(file_list, rank, mean, split_key):

    images = []
    writer_dict = {}
    
    reader = FitReader()
    
    vars_ = []
    
    with h5py.File(H5PY_PATH % (split_key, split_key, rank), 'a') as hf:
        
        for fit_file, n in tqdm(zip(file_list, range(len(file_list)))):

            image, labels = reader.open_fit_file(fit_file)
            if image is None:
                continue

            var = np.mean(np.square(image - mean), axis=(0,1)) 
            vars_.append(var)

            image_shape = image.shape

            grp = hf.create_group(split_key + '_' + str(n))

            image_data = grp.create_dataset(
                name='data', data=image.astype('float32'),
                shape=image.shape, maxshape=image_shape, compression="gzip")

            label_data = grp.create_dataset(
                name='labels', data=labels.astype('float32'),
                shape=labels.shape)
                
    vars_ = np.array(vars_)
    vars_ = np.mean(vars_, axis=0)
    
    reader.close()
    
    return vars_

def splitList(list_, parts:int):
    splitted_list = []
    for i in range(parts):
        splitted_list.append(list_[i::parts])
    return splitted_list

if __name__ == '__main__':
        
    with open(SPLIT_FILE, 'rb') as f:
        split_dict = pickle.load(f)
        
    for split_key in split_dict:
        print(f'processing data split {split_key}')
    
        FITS_FILES_SPLIT = [FITS_FILES[int(i)] for i in split_dict[split_key]]
    
        save_samples(FITS_FILES_SPLIT[:NUM_SAMPLES], split_key)
    
        num_proccesses_mean = 4
        print('Computing mean...')
        with Pool(processes=num_proccesses_mean) as pool:
            files = splitList(FITS_FILES_SPLIT,num_proccesses_mean)
            means_per_file_list = pool.map(estimate_mean, files)
            mean = np.zeros_like(means_per_file_list[0])
            for e in means_per_file_list: 
                mean += e
            mean = mean / len(means_per_file_list)
        
        print('Computing variance and saving galaxies')
    
        print('preparing TFRECORDS...')
        with Pool(processes=PARTS_TFRECORDS) as pool:

            files = splitList(FITS_FILES_SPLIT, PARTS_TFRECORDS)
        
            print('Computing variance and saving galaxies')

            vars_per_file_list = pool.starmap(save_part_tfrecords, zip(files, range(PARTS_TFRECORDS), [mean] * PARTS_TFRECORDS, [split_key] * PARTS_TFRECORDS))

            var = np.zeros_like(vars_per_file_list[0])
            for e in vars_per_file_list:
                var += e
            var = var / len(vars_per_file_list)
            var = np.sqrt(var)

            print('mean and variance tfrecords: ', [mean,var])
        
        print('preparing H5PY...')
        with Pool(processes=PARTS_H5PY) as pool:

            files = splitList(FITS_FILES_SPLIT, PARTS_H5PY)

            print('Computing variance and saving galaxies')

            vars_per_file_list = pool.starmap(save_part_h5py, zip(files, range(PARTS_H5PY), [mean] * PARTS_H5PY, [split_key] * PARTS_H5PY))

            var = np.zeros_like(vars_per_file_list[0])
            for e in vars_per_file_list:
                var += e
            var = var / len(vars_per_file_list)
            var = np.sqrt(var)

            print('mean and variance h5py: ', [mean,var])


        with open(STATS_FILE % (split_key, split_key), 'wb') as handle:
            pickle.dump([mean,var], handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print('finished')
