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

from lensmcmc.tools import generic_yaml_loader
from lensmcmc.tools.lensing import pixels
from lensmcmc.tools.lensing import counts_AB
from lensmcmc.models.sourcemodels import Sersic

STATS_FILE = f'../../data/Sersic/Sersic_mag_22_stats.p'
H5PY_PATH = f"../../data/Sersic/Sersic_mag_22r%1d.h5"
TFRECORDS_PATH = f"../../data/Sersic_mag_22r%1dp%1d.tfrecords"

instrument = generic_yaml_loader('instruments/bells.yaml')

pix = pixels(instrument['pixel_size'], instrument['field_of_view'] + 5.9 * instrument['pixel_size'])

NUM_SAMPLES = 100
SAMPLE_PATH = f'../../data/Sersic/Sersic_mag_22_samples'


PARTS_H5PY = 1
PARTS_TFRECORDS = 4

def create_tfrecord(_image, _labels, _writer):
    # image = open(image_dir + image_file, mode='rb').read()
    # image_decoded = tf.image.decode_image(image)

    ex = tf.train.Example(features=tf.train.Features(feature={
        'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=_image.shape)),
        'label': tf.train.Feature(float_list=tf.train.FloatList(value=_labels)),
        'data': tf.train.Feature(float_list=tf.train.FloatList(value=_image.reshape(-1)))
    }))
    _writer.write(ex.SerializeToString())

def estimate_mean(h5py_file):
    
    mean_acc = np.zeros(shape=(5,))
    count = 0
    
    for fit_file in tqdm(file_list):
            
        count += 1
        mean_acc += np.mean(image, axis=(0,1))
        
        
    return mean_acc / len(file_list)
        
class SersicGenerator:
    def __init__(self):
        pass
    
    def generate(self):
        
        galaxy = {
            'x_position': np.random.uniform(-1.0, 1.0),
            'y_position': np.random.uniform(-1.0, 1.0),
            'radius': np.random.uniform(1.0, 4.0),
            'sersic_index': np.random.uniform(1.0, 4.0),
            'magnitude': 22, #np.random.uniform(16.0, 28.0),
            'axis_ratio': np.random.uniform(0.4, 1.0),
            'position_angle': np.random.uniform(0.0, np.pi)
        }

        # Convert magnitude to counts
        total_counts = counts_AB(galaxy['magnitude'], instrument)

        # Initialise galaxy model
        galaxy_light_model = Sersic({
           'x_position': galaxy['x_position'],
           'y_position': galaxy['y_position'],
           'radius': galaxy['radius'],
           'sersic_index': galaxy['sersic_index'],
           'total_counts': total_counts * (instrument['pixel_size'] ** 2),
           'axis_ratio': galaxy['axis_ratio'],
           'position_angle': galaxy['position_angle']
        })

        labels = [galaxy['x_position'], galaxy['y_position'], galaxy['radius'], galaxy['sersic_index'],
           galaxy['axis_ratio'], galaxy['position_angle'], total_counts * (instrument['pixel_size'] ** 2)]

        # Calculate brightness
        image = instrument['exposure_time'] * galaxy_light_model.ray_trace(pix, sub=10)[:,:,None]
       
        return image, np.array(labels)
    
    def close(self):
        pass
        
def save_part_tfrecords(num_data, rank):
    
    n = rank
    images = []
    writer_dict = {}

    for resolution_level in range(2, 8 + 1):
        writer_dict[resolution_level] = tf.io.TFRecordWriter(TFRECORDS_PATH % (resolution_level,n))

    generator = SersicGenerator()
    mean_ = []
    
    for i in tqdm(range(num_data)):
        
        image, labels = generator.generate()
        mean = np.mean(image, axis=(0,1))
        mean_.append(mean)
     
        for resolution_level in range(2,8+1):

        
            resolution = image.shape[0] // (2** (8 - resolution_level))
            downsampled_image = resize(image, (resolution, resolution), order=1)

            downsampled_image = np.transpose(downsampled_image, (2,0,1))
            
            
            create_tfrecord(downsampled_image, labels, writer_dict[resolution_level])

    for resolution_level in range(2,8+1):
        writer_dict[resolution_level].close()
    
    mean_ = np.array(mean_)
    mean_ = np.mean(mean_, axis=0)
        
    generator.close()
    
    return mean_
    
def save_samples(num_samples):
    
    generator = SersicGenerator()
    
    for i in tqdm(range(num_samples)):
        
        image, labels = generator.generate()
        channel = image[...,0] 
        
        cm = plt.cm.ScalarMappable(None, cmap='magma')
        cmap = cm.get_cmap()
        channel = (channel - np.min(channel)) / (np.max(channel) - np.min(channel))
        rgb_img = cmap(channel)[..., :3]
        rgb_img = rgb_img * 255
        rgb_img = rgb_img.astype(np.uint8)

        result = Image.fromarray(rgb_img)
        result.save(os.path.join(SAMPLE_PATH, '%05d.png' % i))
        with open(os.path.join(SAMPLE_PATH, '%05d.npy' % i), 'wb') as handle:
            np.save(handle, image)
        
def save_part_h5py(num_data, rank):

    images = []
    writer_dict = {}
    
    generator = SersicGenerator()   
    mean_ = []

    with h5py.File(H5PY_PATH % rank, 'a') as hf:
        
        for i in tqdm(range(num_data)):

            image, labels = generator.generate()
            mean = np.mean(image, axis=(0,1))
            mean_.append(mean)
            
            image_shape = image.shape

            grp = hf.create_group("%05d" % i)

            image_data = grp.create_dataset(
                name='data', data=image.astype('float32'),
                shape=image.shape, maxshape=image_shape, compression="gzip")

            label_data = grp.create_dataset(
                name='labels', data=labels.astype('float32'),
                shape=labels.shape)
                
    mean_ = np.array(mean_)
    mean_ = np.mean(mean_, axis=0)
        
    generator.close()
    
    return mean

if __name__ == '__main__':
    
    num_galaxies = 50000
    
    save_samples(50)

    print('preparing H5PY...')
    with Pool(processes=PARTS_H5PY) as pool:
        
        print('Saving galaxies')
        
        means_per_file_list = pool.starmap(save_part_h5py, zip([num_galaxies], [0]))
        mean = np.zeros_like(means_per_file_list[0])
        for e in means_per_file_list: 
            mean += e
        mean = mean / len(means_per_file_list)
        
        print('mean h5py: ', [mean])
        
    with h5py.File(H5PY_PATH % 0, 'r') as f:
        keys = list(f.keys())
        var_ = []
        for key in tqdm(keys):
            group_selector = f[key]
            data = group_selector['data'][:]
            var_.append(np.mean(np.square(data - mean), axis=(0,1)))
            
        var_ = np.array(var_)
        
        var = np.mean(var_, axis=0)
        
        std = np.sqrt(var)
            
    print('mean, std', [mean, std])
                
    
    print('preparing TFRECORDS...')
    with Pool(processes=PARTS_TFRECORDS) as pool:
        
        print('Saving galaxies')
        
        means_per_file_list = pool.starmap(save_part_tfrecords, zip([num_galaxies // PARTS_TFRECORDS] * PARTS_TFRECORDS, range(PARTS_TFRECORDS)))
        mean = np.zeros_like(means_per_file_list[0])
        for e in means_per_file_list: 
            mean += e
        mean = mean / len(means_per_file_list)
        
        print('mean tfrecords: ', [mean])
        
    with open(STATS_FILE, 'wb') as handle:
        pickle.dump([mean,std], handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print('finished')
