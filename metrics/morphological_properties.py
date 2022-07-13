# Copyright (c) 2022, Benjamin Holzschuh

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from astropy.visualization import simple_norm
from astropy.modeling import models
from astropy.convolution import convolve
import photutils
import multiprocessing
import time
import statmorph
import h5py
import uuid
import pickle
import os
from pathlib import Path
import time
from data_iterator import *
from tqdm import tqdm
from statmorph.utils.image_diagnostics import make_figure
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.convolution import Gaussian2DKernel

size = 20  # on each side from the center
sigma_psf = 2.0
y, x = np.mgrid[-size:size+1, -size:size+1]
psf = np.exp(-(x**2 + y**2)/(2.0*sigma_psf**2))
psf /= np.sum(psf)

snp = 100.0
npixels = 5  # minimum number of connected pixels

gain = 10000.0



morph_properties_names = ['xc_centroid', 'yc_centroid', 'ellipticity_centroid',
    'elongation_centroid', 'orientation_centroid', 'xc_asymmetry', 'yc_asymmetry', 'ellipticity_asymmetry', 
    'elongation_asymmetry', 'orientation_asymmetry', 'rpetro_circ', 'rpetro_ellip', 'rhalf_circ', 'rhalf_ellip',
    'r20', 'r80', 'gini', 'm20', 'gini_m20_bulge', 'gini_m20_merger', 'sn_per_pixel', 'concentration', 'asymmetry',
    'smoothness', 'sersic_amplitude', 'sersic_rhalf', 'sersic_n', 'sersic_xc', 'sersic_yc', 'sersic_ellip', 
    'sersic_theta', 'sky_mean', 'sky_median', 'sky_sigma', 'flag', 'flag_sersic']


def preprocessing_COSMOS(image):
    
    ny, nx = image.shape
    npixels = 5  # minimum number of connected pixels
    
    nsigma = 1.5
    psf = None
    
    threshold = photutils.detect_threshold(image, nsigma=nsigma) #, background=0.0)
    
    segm = photutils.detect_sources(image, threshold, npixels)
    label = np.argmax(segm.areas) + 1
    segmap = segm.data == label
    segmap_float = ndi.uniform_filter(np.float64(segmap), size=10)
    segmap = segmap_float > 0.5
    
    gain = 1 
    
    return image, segmap, gain, psf

def preprocessing_Sersic(image):
    
    ny, nx = image.shape
    npixels = 5  # minimum number of connected pixels
    
    nsigma = 1.5
    psf = None
    
    threshold = photutils.detect_threshold(image, nsigma=nsigma) #, background=0.0)
    
    segm = photutils.detect_sources(image, threshold, npixels)
    label = np.argmax(segm.areas) + 1
    segmap = segm.data == label
    segmap_float = ndi.uniform_filter(np.float64(segmap), size=10)
    segmap = segmap_float > 0.5
    
    gain = 1 
    
    return image, segmap, gain, psf

def preprocessing_SKIRT(image):
    
    # The PSF used in Rodriguez-Gomez et al. is chosen close to the PSF of the PS1 instrument, which is 1.18 arcseconds in the i band
    # pixel scale is 0.267 ckpch-1. 3 pixels roughly correspond to 1.16 kpc which translates to 1.18 arcsec for objects at z=0.0485 
    
    ny, nx = image.shape
    npixels = 5  # minimum number of connected pixels

    
    sigma = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3
    kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
    kernel.normalize()
    psf = kernel 
    sigma_background = 1 / 15 # for the i band in TNG
    image = convolve(image, psf) 
    image += sigma_background * np.random.standard_normal(size=(ny, nx))
    
    nsigma = 1.5
    
    threshold = photutils.detect_threshold(image, nsigma=nsigma) #, background=0.0)
    
    segm = photutils.detect_sources(image, threshold, npixels)
    
    label = np.argmax(segm.areas) + 1
    segmap = segm.data == label
    segmap_float = ndi.uniform_filter(np.float64(segmap), size=10)
    segmap = segmap_float > 0.5
    
    gain = 750 # according to Rodriguez-Gomez et al.
    
    return image, segmap, gain, np.array(psf)


def fit_galaxy(image, savefile=None, config='SKIRT'):
    
    if config=='SKIRT':
        image, segmap, gain, psf = preprocessing_SKIRT(image)
    elif config=='Sersic':
        image, segmap, gain, psf = preprocessing_Sersic(image)
    elif config=='COSMOS':
        image, segmap, gain, psf = preprocessing_COSMOS(image)
    else:
        return None
    
    source_morphs = statmorph.source_morphology(
        image, segmap, gain=gain, psf=psf)
    
    features = {}
    for key in morph_properties_names:
        features[key] = getattr(source_morphs[0],key)
        
    if savefile:
        fig = make_figure(source_morphs[0])
        print(savefile)
        fig.savefig(savefile)
    
    return features
    
def processing(input_, savefile=None):
    images, start, config = input_
    if images.shape[3] == 1:
        images = images.repeat([1, 1, 1, 3])
    r = {}
    for n in range(images.shape[0]):
        # try:
            i_channel = images[n][...,2].numpy()
            if savefile:
                r[start+n] = fit_galaxy(i_channel, savefile=savefile % (start+n), config=config)
            else:
                r[start+n] = fit_galaxy(i_channel, config=config)   
        # except Exception as e:
        #     print(e)
        #     print(f'Skipping image {start+n}')
    return r

def compute_morph_for_dataset(dataset, max_items=50000):

    
    tag = dataset['tag']
    config = dataset['config']
    cache_folder = dataset['cache']
    if tag:
        cache_file = os.path.join(cache_folder, tag + '_morph.pkl')
        

        # Check if the file exists (all processes must agree).
        flag = os.path.isfile(cache_file)
        
        # Load.
        if flag:
            
            print(f'Loaded {cache_file}')
            with open(cache_file, 'rb') as handle:
                return pickle.load(handle)
    
    num_items = len(dataset['file'])
    if max_items is not None:
        num_items = min(num_items, max_items)

    batch_size = 1
    iterator = H5PyIterator(dataset['file'], transform=dataset['transform'])
    dl = torch.utils.data.DataLoader(iterator, batch_size=batch_size)
    
    morph_properties = {}

    processed_items = 0
    
    p = multiprocessing.Pool(12)
    ids = [i * batch_size for i in range(len(dl))]
    configs = [config for _ in range(len(dl))]
    
    print('plotting images.......')
    
    iter_dl = iter(dl)
    for i in range(10):
        Path(os.path.join('cache', tag)).mkdir(parents=True, exist_ok=True)
        savefile = os.path.join('cache', tag, 'morph_%d.png')
        processing((next(iter_dl), i*batch_size, config), savefile=savefile)
    
    print('starting fits........')
    
    for elem in tqdm(p.imap_unordered(processing, zip(dl, ids, configs)), total=len(dl)):
        morph_properties.update(elem)
        
    
    if 'max_items' in dataset and len(dl) >= dataset['max_items']:
            for id in range(dataset['max_items'], len(dl)):
                del morph_properties[id]
    
    if tag:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + '.' + uuid.uuid4().hex
       
        with open(temp_file, 'wb') as handle:
            pickle.dump(morph_properties, handle)
 
        os.replace(temp_file, cache_file) # atomic

    return morph_properties

def compute_morph(SOURCE, TARGET):
    
    morph_source = compute_morph_for_dataset(SOURCE)
 
    morph_generated = compute_morph_for_dataset(TARGET)

    return morph_source, morph_generated