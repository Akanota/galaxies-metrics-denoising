# Copyright (c) 2022, Benjamin Holzschuh

import numpy as np
import h5py
import torch
from skimage.transform import resize

def len_dataset(file):
    with h5py.File(file, 'r') as f:
        return len(f)
    

def counter_dataset(file):
    with h5py.File(file, 'r') as f:
        for key in f.keys:
            group_selector = f[str(idx)]
            _image = group_selector['data'][:]
            # _image = _image * 0.5 + 0.5
            # _image = np.clip(_image, 0, 1)
            # _image = _image * 255
            # _image = _image.astype(np.uint8)
            yield _image
            
class H5PyIterator(torch.utils.data.IterableDataset):
    def __init__(self, file, resize_transform=-1, transform = True):
        self.file = file
        with h5py.File(file, 'r') as f:
            self.keys = list(f.keys())
            
        self.transform = transform
        self.resize_transform = resize_transform
        self.counter = 0
        self.numItems = len(self.keys)
        super(H5PyIterator).__init__()
        
    def __len__(self):
        return self.numItems
        
    def __iter__(self):
        return self
    
    def get_item(self, id):
        with h5py.File(self.file, 'r') as f:
            group_selector = f[str(id)]
            data = group_selector['data'][:]

            return self.transform_data(data)
    
    def transform_data(self, data):
        if data.shape[0] in [1,2,3,4,5]:
            data = np.transpose(data, (1,2,0))

        if self.transform:

            size_transformed = np.mean(data[...,-1])
            min_size = 2.0
            max_size = 2516.0

            size = np.exp(size_transformed * (np.log(max_size) - np.log(min_size)) + np.log(min_size)) / 2 

            size = min(int(size), 2516)
            size = max(size, 2)

            data = resize(data[...,:4], (size, size, 4))
        if self.resize_transform > 0:
            data = resize(data, (self.resize_transform, self.resize_transform))

        return data
        

    def __next__(self):
        if self.counter >= self.numItems:
            raise StopIteration
        else:
            with h5py.File(self.file, 'r') as f:
                group_selector = f[str(self.keys[self.counter])]
                self.counter += 1
                data = group_selector['data'][:]
                
                return self.transform_data(data)
                
            
        self.current += 1
        if self.current < self.high:
            return self.current
        raise StopIteration
            
class SimpleH5PyLoader(torch.utils.data.IterableDataset):
        def __init__(self, file):
            self.file = file
            super(SimpleH5PyLoader).__init__()

        def __iter__(self):
            return H5PyIterator(self.file)
