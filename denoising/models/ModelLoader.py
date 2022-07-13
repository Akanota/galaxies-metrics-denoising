# Copyright (c) 2022, Benjamin Holzschuh

import os

file_directory = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.append(os.path.join(file_directory, '..'))

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import model_from_json


from models.Architectures import *

import random
import string

class ModelLoader(object):
    
    save_path = os.path.join(file_directory, '..', 'weights')
    
    file_list = set()
    
    def __init__(self, id = None, env='local', **kwargs):
        
        if (not isinstance(id, str)) and (not (id is None)):
            raise TypeError('Id must be a string identifier')
    
        if id is None:
            id = ''.join(random.choice(string.ascii_letters) for i in range(8))
            
        self.id = id  
        self.kwargs = kwargs
        
        #self.model_path_json = os.path.join(self.save_path, id + '.json')
        self.weight_path = os.path.join(self.save_path, id + '.h5')
        
        self.modelDict = getModelDict(**kwargs)
        self.input = self.modelDict['input']
        
        if hasattr(self.modelDict['full_model'], 'output'):
            self.model = keras.Model(inputs=self.input, outputs=self.modelDict['full_model'].output)
        else:
            self.model = keras.Model(inputs=self.input, outputs=self.modelDict['full_model'])
    
        # print(self.model_encoder.summary())
        
    
    def snapshot(self, file):
        self.model.save_weights(file, save_format='h5')
        
    def save_weights(self, id, budget=None):
        if budget:
            self.model.save_weights(os.path.join(self.save_path, f'{id}_{budget}.h5'), save_format='h5')
        else:
            self.model.save_weights(os.path.join(self.save_path, f'{id}.h5'), save_format='h5')
        
    def restore(self, file=None):
        
        if file is None:
            weight_path_ = self.weight_path
        else:
            weight_path_ = file
        
        if os.path.isfile(weight_path_):
                self.model.load_weights(weight_path_)
                print(f'loading weights from {weight_path_}')
        else:
            print(f'did not find file {weight_path_} for best model weights')
            # raise ValueError(f'did not find file {weight_path_} for best model weights')
    
            
    def buildModel(self):
        pass
    
    def get_model(self):
        return self.model
