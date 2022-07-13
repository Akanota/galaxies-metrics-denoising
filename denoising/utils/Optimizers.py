# Copyright (c) 2022, Benjamin Holzschuh

from tensorflow.keras.optimizers import RMSprop, SGD, Adam, Nadam

def get_optimizer(optimizer='adam', learning_rate=0.001, clipnorm=None, clipvalue = None, **kwargs):
    
    if optimizer=='rmsprop':
        optimizer = RMSprop
    elif optimizer=='adam':
        optimizer = Adam
    elif optimizer=='nadam':
        optimizer = Nadam
    elif optimizer=='sgd':
        optimizer = SGD
        
    print(learning_rate)
        
    if not clipvalue is None:
        return optimizer(lr=learning_rate, clipvalue=clipvalue)
    elif not clipnorm is None:
        return optimizer(lr=learning_rate, clipnorm=clipnorm)
    else:
        return optimizer(lr=learning_rate)
