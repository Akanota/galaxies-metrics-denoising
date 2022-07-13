# Copyright (c) 2022, Benjamin Holzschuh

import numpy as np
import scipy.linalg
import torch
from scipy import fftpack
from scipy.stats import wasserstein_distance
import copy
import hashlib
import os
import uuid
import pickle
from tqdm import tqdm
from data_iterator import *
from matplotlib import pyplot as plt
from skimage.transform import resize

# https://stackoverflow.com/questions/21242011/most-efficient-way-to-calculate-radial-profile
def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(np.int32)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

def get_partition_list(num_partitions = 15):
    
    max_frequency = 0.5 * 1 / (0.276 / 0.7)
    min_frequency = 0.0
    image_dim = 128
    num_bins = int(np.floor(image_dim * np.sqrt(2)))
    dist = np.linspace(min_frequency, max_frequency, num_bins)
    loguniform_array = np.log(np.linspace(1/1024, max_frequency, num_partitions))
    partition_list = list(loguniform_array)
    spacing_list = []
    for x0, x1 in zip(partition_list, partition_list[1:]):
        spacing_list.append(1 / (x1 - x0))

    sum_spacing = np.sum(spacing_list)
    interval_length = np.array(spacing_list) * (num_bins / sum_spacing)
    count = 0
    partition_class = []
    for i in range(num_bins):
        if i < np.cumsum(interval_length)[count]:
            partition_class.append(count)
        else:
            count += 1
            partition_class.append(count)

    partition_list = []
    previous_class = 0
    previous_frequency = 0
    count = 0

    partition_list.append(0)

    for i in partition_class:

        if previous_class != i:

            partition_list.append((previous_frequency, max_frequency * ((count + 1) / num_bins)))
            previous_frequency = max_frequency * ((count + 1) / num_bins)
            previous_class = i

        count += 1

    partition_list.append((previous_frequency, max_frequency))
    return partition_list

partition_list = get_partition_list()

def bin_to_partition(bin_number, num_bins, partition_list):
    assert bin_number < num_bins
    
    if bin_number == 0:
        return 0, 0
    elif bin_number == num_bins - 1:
        return len(partition_list) - 1, partition_list[-1][-1]
    
    frequency = (bin_number / (num_bins-1)) * partition_list[-1][-1]
    
    for n, range_ in zip(range(len(partition_list)), partition_list):
        if not isinstance(range_, tuple):
            continue
        x0, x1 = range_
        
        if x0 < frequency and frequency <= x1:
         
            return n, frequency


def powerspectra_to_partition(powerspectra, partition=partition_list):

    bins = np.linspace(-2, 5, 100)

    modes_dict = {}
    N_modes = len(partition)
    
    for n in range(N_modes):
        modes_dict[n] = []
    
    for p in powerspectra:
        for n, value in zip(range(len(p)), p):
            bin_, _ = bin_to_partition(n, len(p), partition)
            modes_dict[bin_].append(value)
            
    
    for i in range(N_modes):
        modes_dict[i] = []
        for p in powerspectra:
            if len(p) < i + 1:
                continue
            modes_dict[i].append(p[i])

    # hist_dict = {}    
    
    # for mode in range(N_modes):
    #     log_modes = [np.log10(x) for x in modes_dict[mode]]
    #     hist = np.histogram(log_modes, bins=bins)
    #     hist_dict[mode] = hist

    # return hist_dict
    return modes_dict
    

def powerspectra_to_hist(powerspectra, modes=None):
    powerspectra_normalized = []

    for p in powerspectra:
        # s = sum(p)
        # p = [x / s for x in p]
        powerspectra_normalized.append(p)

    bins = np.linspace(-9, 1, 100)

    modes_dict = {}
    N_modes = 2500 # pick any large enough number 
    for i in range(N_modes):
        modes_dict[i] = []
        for p in powerspectra_normalized:
            if len(p) < i + 1:
                continue
            modes_dict[i].append(p[i])

    hist_dict = {}
    
    if modes is None:
        modes = list(modes_dict.keys())
        
    
    for mode in modes:
        log_modes = [np.log10(x) for x in modes_dict[mode]]
        hist = np.histogram(log_modes, bins=bins)
        hist_dict[mode] = hist

    return hist_dict


def channel_to_power_spectra(r_channel):
    dim = r_channel.shape[0]
    galaxy_fft = fftpack.fft2(r_channel)
    galaxy_shiftfft = np.fft.fftshift(galaxy_fft)

    galaxy_shiftfft_normalize = galaxy_shiftfft / (dim * dim)

    rasp = radial_profile(abs(galaxy_shiftfft_normalize), (dim / 2, dim / 2))

    return rasp


def compute_modes_hist_for_dataset(dataset, modes=None, max_items=50000, data_loader_kwargs=None, batch_size=64):

    tag = dataset['tag']
    cache_folder = dataset['cache']
    if tag:
        cache_file = os.path.join(cache_folder, tag + '_raps.pkl')
    
        # Check if the file exists (all processes must agree).
        flag = os.path.isfile(cache_file)
        
        # Load.
        if flag:
            
            print(f'Loaded {cache_file}')
            with open(cache_file, 'rb') as handle:
                return pickle.load(handle)
    
    # dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)


    # dataset.reset(opts.dataset_kwargs.lod, opts.dataset_kwargs.per_GPU_batch_size)

    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    num_items = len(dataset['file'])
    if max_items is not None:
        num_items = min(num_items, max_items)
        
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # pm = ProgressMonitor()
    # progress = pm.sub(tag='dataset features', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)

    iterator = H5PyIterator(dataset['file'], transform=dataset['transform'])
    dl = torch.utils.data.DataLoader(iterator, batch_size=1)
    
    powerspectra = []
    sizes = []

    processed_items = 0
    for images in tqdm(dl):

        if images.shape[3] == 1:
            images = images.repeat([1, 1, 1, 3])

        for n in range(images.shape[0]):

            r_channel = images[n, :, :, 1].cpu().numpy()

            size, _ = r_channel.shape
            sizes.append(size)
            
            powerspectra.append(channel_to_power_spectra(r_channel))
            
        processed_items += images.shape[0]
        # progress.update(processed_items)

    histogram_dict = powerspectra_to_hist(powerspectra, modes)
    
    #if opts.num_gpus > 1:
    #    for key in histogram_dict:
    #        histogram_key_list = []
    #        for src in range(opts.num_gpus):
    #            value = torch.as_tensor(histogram_dict[key][0], dtype=torch.float32, device=opts.device)
    #            torch.distributed.broadcast(value, src=src)
    #            histogram_key_list.append(value.cpu().numpy())
    #        histogram_dict[key] = (np.sum(np.stack(histogram_key_list), axis=0), histogram_dict[key][1])
    #    sizes_list = []
    #    for src in range(opts.num_gpus):
    #        # Size list can have different lengths; this is a problem for broadcast
    #        size_placeholder = len(sizes)
    #        value_size = torch.as_tensor(size_placeholder, dtype=torch.int32, device=opts.device)
    #        torch.distributed.broadcast(value_size, src=src)
    #        if opts.rank == src:
    #            value = torch.as_tensor(sizes, dtype=torch.int32, device=opts.device)
    #        else:
    #            value = torch.zeros((value_size.cpu().numpy()), dtype=torch.int32).to(opts.device)
    #
    #        torch.distributed.broadcast(value, src=src)
    #        sizes_list.append(value.cpu().numpy())
    #    sizes = sum([list(s) for s in sizes_list], [])
    
    if tag:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + '.' + uuid.uuid4().hex
       
        with open(temp_file, 'wb') as handle:
            pickle.dump((histogram_dict, sizes, powerspectra), handle)
 
        os.replace(temp_file, cache_file) # atomic

    return histogram_dict, sizes, powerspectra


def plt_mode_joint(ax, data_generated, data_training, mode):

    # log_modes_generated = [np.log10(x) for x in data_generated[mode]]
    # log_modes_training = [np.log10(x) for x in data_training[mode]]
    # vert_hist_generated = np.histogram(log_modes_generated, bins=bins)
    # vert_hist_training = np.histogram(log_modes_training, bins=bins)

    vert_hist_generated = data_generated[mode]
    vert_hist_training = data_training[mode]

    # Compute height of plot.
    height_generated = np.ceil(max(vert_hist_generated[1])) - np.floor(min(vert_hist_generated[1]))
    height_training = np.ceil(max(vert_hist_training[1])) - np.floor(min(vert_hist_training[1]))
    height = max(height_generated, height_training)

    # compute height of each horizontal bar.
    height = height / len(vert_hist_training[0])

    ax.barh(vert_hist_training[1][:-1], vert_hist_training[0] / sum(vert_hist_training[0]), height=height, color='blue', alpha=0.7)
    ax.barh(vert_hist_generated[1][:-1], vert_hist_generated[0] / sum(vert_hist_generated[0]), height=height, color='orange', alpha=0.7)

    # ax.set_yscale('log')
    ax.set_xlabel(mode)

def compute_raps(SOURCE, TARGET):
    
    modes = [0, 1, 2, 3, 4, 6, 8, 10, 15, 20, 25, 30, 40]

    hist_dataset, sizes_dataset, powerspectra_dataset = compute_modes_hist_for_dataset(SOURCE, modes=modes)
    hist_generated, sizes_generated, powerspectra_generated = compute_modes_hist_for_dataset(TARGET, modes=modes)

    stats = {}

    return stats, {'SOURCE': (hist_dataset, sizes_dataset, powerspectra_dataset), 'TARGET': (hist_generated, sizes_generated, powerspectra_generated)}
