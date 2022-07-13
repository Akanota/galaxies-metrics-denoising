# Copyright (c) 2022, Benjamin Holzschuh

import h5py
from matplotlib import pyplot as plt
import numpy as np
import os
import argparse
from radially_averaged_power_spectrum import *
from morphological_properties import compute_morph
from frechet_inception_distance import compute_fid
from kernel_inception_distance import compute_kid
import torch
from multiprocessing import *

def calculate_metrics(generated_file, training_file, config, cache='cache'):
    
    start_generated = generated_file.rfind('/')+1
    end_generated = generated_file.rfind('.')
    tag_generated = generated_file[start_generated:end_generated]
    
    start_training = training_file.rfind('/')+1
    end_training = training_file.rfind('.')
    tag_training = training_file[start_training:end_training]
    
    dataset_dict_source = {'file': training_file, 'transform': False, 'tag': tag_training, 'config': config, 'cache': cache}

    dataset_dict_target = {'file': generated_file, 'transform': True, 'tag': tag_generated, 'config': config, 'cache': cache}
    
    torch.cuda.set_per_process_memory_fraction(0.75)
    
    print(f'Calculating 2D powerspectra')
    
    result_raps = compute_raps(dataset_dict_source, dataset_dict_target)
    
    print(f'Calculating FID')
    
    result_fid = compute_fid(dataset_dict_source, dataset_dict_target)
    
    print(f'Calculating KID')
    
    result_kid = compute_kid(dataset_dict_source, dataset_dict_target)
    
    print(f'Calculating morphological properties')
    
    result_morph = compute_morph(dataset_dict_source, dataset_dict_target)
    
    return {'raps': result_raps, 'fid': result_fid, 'kid': result_kid, 'morph': result_morph}

# with Pool(processes=4) as pool:
#     print(files)
#     _ = pool.starmap(calculate_metrics, [ (file,) for file in files ] )
    
if __name__== "__main__":

    parser = argparse.ArgumentParser(description='Parameter Parser', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--generated-file', required=True, help='Generated data file')
    parser.add_argument('--training-file', required=True, help='Training data file')
    parser.add_argument('--config', required=True, help='Either "SKIRT", "COSMOS" or "Sersic"')
    
    args, unknown = parser.parse_known_args()
    
    _ = calculate_metrics(**vars(args))
