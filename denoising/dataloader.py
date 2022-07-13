# Copyright (c) 2022, Benjamin Holzschuh

from random import shuffle as shuffleList
from random import choice, choices
from tensorflow.keras.utils import Sequence

from scipy import ndimage
import scipy
import imageio
import numpy as np
from copy import deepcopy
import os
from skimage.transform import resize
import h5py
from astropy.convolution import convolve

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
    
def listdir_nohidden(path, suffix = None):
    list_ = []
    for f in os.listdir(path):
        if (not f.startswith('.')):
            if suffix is None:
                list_.append(f)
            else: 
                if f.endswith(suffix):
                    list_.append(f)
    return list_

size = 20  # on each side from the center
sigma_psf = 2.0
y, x = np.mgrid[-size:size+1, -size:size+1]
psf = np.exp(-(x**2 + y**2)/(2.0*sigma_psf**2))
psf /= np.sum(psf)
    
def make_generator(dl):
    def f():
        for x,y in dl:
            for n in range(x.shape[0]):
                yield x[n], y[n]
            
    return f

class MixingDataLoader(Sequence):
    
    def __init__(self, dataLoaderList, alphaList, size):
        self.dataLoaderList = dataLoaderList
        self.dataLoaderIdx = list(range(len(self.dataLoaderList)))
        self.dataLoaderItems = [0] * len(self.dataLoaderList)
        self.alphaList = alphaList
        self.size = size
        
        assert np.sum(self.alphaList) == 1.0
        
    def __len__(self):
        
        self.on_epoch_end()
        return 50
        # return self.size
    
    def on_epoch_end(self):
    
        for loader in self.dataLoaderList:
            loader.on_epoch_end()
        self.dataLoaderItems = [0] * len(self.dataLoaderList)
        
    def __getitem__(self, index):
        
        idx = choices(population=self.dataLoaderIdx, weights=self.alphaList, k=1)[0]
        item = self.dataLoaderList[idx].__getitem__(self.dataLoaderItems[idx])
        self.dataLoaderItems[idx] += 1
        
        return item
        
    
    
class DataLoader(Sequence):
    
    def __init__(self, files, keys, batchSize, shuffle = True, name='', augmentation=None):
    
        self.augmentation = augmentation
        self.name = name
        self.keys = deepcopy(keys)
        self.shuffle = shuffle
        self.batchSize = batchSize
        self.transform = []
        
        self._open_h5file(files)
        # self.keys = list(self.h5file.keys())
        
        self.numSamples = len(self.keys)
        
        self.normalize = True
        self.mean = np.array([0.,0.,0.,0.])
        self.std = np.array([1.,1.,1.,1.])
    
        if self.shuffle:
            shuffleList(self.keys)    
        
        self.batches = list(chunks(self.keys, self.batchSize))
        self.numBatches = len(self.batches)
                
        print("Length: %d" % self.numSamples)
        
        
    def _open_h5file(self, paths):
        
        self.h5files = {}
        for path in paths:
            self.h5files[path] = h5py.File(path, 'r')
        
        return self.h5files
        
    def __del__(self):
        try:
            if self.h5files is not None:
                self.h5files.close()
        finally:
            self.h5files = None
            # print('closed ', self._path)
        
            
    def load(self, key, transform = True, augmentation=None):
        
        file, sample_id = key
        group_selector = self.h5files[file][str(sample_id)]
        orig = group_selector['orig'][:][...,:4]
        convolved = group_selector['convolved'][:][...,:4]
         
        if transform:
            flipAxes: List[int] = []

            if choice([0, 1]) == 1:
                flipAxes.append(0)
            if choice([0, 1]) == 1:
                flipAxes.append(1)
            orig = np.flip(orig, axis=flipAxes)
            convolved = np.flip(convolved, axis=flipAxes)
            rotation = choice([0, 1, 2, 3])
            orig = np.rot90(orig, k=rotation, axes=[0, 1])
            convolved = np.rot90(convolved, k=rotation, axes=[0, 1])
         
        # orig = np.array(data)
        # data[...,0] = convolve(data[...,0], psf)
        # data[...,1] = convolve(data[...,1], psf)
        # data[...,2] = convolve(data[...,2], psf)
        # data[...,3] = convolve(data[...,3], psf)
        
        
        if augmentation is None:
        
            ny, nx, nc = convolved.shape
            snp = 0.25
            convolved += (1.0 / snp) * np.random.standard_normal(size=(ny, nx, nc))
        
        else:
           
            downsampling = augmentation['downsampling']
            orig = resize(orig, (int(orig.shape[0] // downsampling), int(orig.shape[1] // downsampling), orig.shape[2]))
            convolved = resize(convolved, orig.shape)
            
            ny, nx, nc = convolved.shape
            snp = augmentation['snp']
            convolved += (1.0 / snp) * np.random.standard_normal(size=(ny, nx, nc))
        
        return convolved, orig
            
    def __len__(self):
        
        return 50
        
        # return self.numBatches

    def setNormAndStd(self, mean, std):
        
        self.normalize = True
        self.mean = mean
        self.std = std
    
    def __getitem__(self, index):
        
        data_orig = []
        data_noised = []
        
        for sample in self.batches[index]:
        
            noised, orig = self.load(sample, augmentation=self.augmentation)
            data_orig.append(orig)
            data_noised.append(noised)
        
        item_orig = np.stack(data_orig)
        item_noised = np.stack(data_noised)
        
        if self.normalize:
            
            item_orig = (item_orig - self.mean) / self.std
            item_noised = (item_noised - self.mean) / self.std
        
        return item_noised, item_orig

    def on_epoch_end(self):
    
        if self.shuffle:
            shuffleList(self.keys)   
            self.batches = list(chunks(self.keys, self.batchSize))
            
    def setTransformations(self, transformations):
        
        self.transform = transformations
            
    def getNormAndStd(self, mean = None, std = None):
        
        mean = 0
        std = 0
        count_elems = 0
    
        normalize_ = self.normalize
        self.normalize = False
    
        for i in range(self.__len__()):
            batch, _ = self.__getitem__(i)
            count_elems += batch.shape[0] * batch.shape[1] * batch.shape[2]
            mean += np.sum(batch,axis = (0,1,2))
            
        mean = mean / count_elems
            
        for i in range(self.__len__()):
            batch, _ = self.__getitem__(i)
            
            std += np.sum(np.abs(batch-mean)**2, axis=(0,1,2))
        
        std = std / count_elems
        std = np.sqrt(std)
        std[2] = 1
    
        self.normalize = normalize_
    
        return mean, std
