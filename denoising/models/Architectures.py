# Copyright (c) 2022, Benjamin Holzschuh

import re
from enum import Enum

import os
file_directory = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.join(file_directory, '..'))

class ARCHITECTURES(Enum):
    ResNetV1 = 1 
    
def getArchitectureEnum(architecture):
        
        if architecture == 'ResNetV1':
            _architecture = ARCHITECTURES.ResNetV1
        else:
            raise ValueError(f'Architecture {architecture} not supported')
        
        return _architecture
        
def getModelDict(architecture, **params):
    
    architecture = getArchitectureEnum(architecture)
    
    if architecture == ARCHITECTURES.ResNetV1:
            from models.ResNetV1 import loadModel
            return loadModel(**params)

    else:
        raise ValueError(f'Architecture {architecture} not supported')
