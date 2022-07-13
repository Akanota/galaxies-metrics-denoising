# Copyright (c) 2022, Benjamin Holzschuh

import sys
import os
# sys.path.append(os.path.join("..", ".."))
file_directory = os.path.dirname(os.path.abspath(__file__))


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, Callback, EarlyStopping
from tensorflow.keras.optimizers import RMSprop, SGD, Adam, Nadam
from dataloader import *
from sklearn.model_selection import *
import numpy as np
from matplotlib import pyplot as plt
import argparse
import random
import string
import math
import time
import pickle
from copy import deepcopy
import zipfile
import h5py
from utils.Optimizers import *
from models.ModelLoader import *

import wandb
from wandb.keras import WandbCallback

hyperparameter_defaults = {
    'architecture' : 'ResNetV1',
    'api_key' : None,
    'name' : None,
    'optimizer' : 'adam',
    'learning_rate' : 0.0001,
    'batch_size': 1,
    'epochs' : 1,
    'normalization_data' : 'fixed',
    'rotate_data' : True,
    'gpu' : None,
    'budget_steps' : 10,
    'data_mixing_gen' : None,
    'data_mixing_orig' : 1.0,
    'alpha_mixing' : 0.0,
}

ENTITY = 'mpa'
    
def log_videos(keys, gen, loader, name=None):
        if name is None:
            name = gen.name
        NUM_SAMPLES = 50
        video_log = {}
        
        if len(keys) < NUM_SAMPLES:
            return video_log
        
        video_log[f'{name}_plots'] = []
        
        for id in range(NUM_SAMPLES):
            key = keys[id]
            noised, orig = gen.load(key, transform=False)
            
            channel_g_orig = orig[...,0]
            channel_r_orig = orig[...,1]
            channel_i_orig = orig[...,2]
            channel_z_orig = orig[...,3]
            
            channel_g_noised = noised[...,0]
            channel_r_noised = noised[...,1]
            channel_i_noised = noised[...,2]
            channel_z_noised = noised[...,3]
            
            denoised = loader.model.predict(noised[None,...])[0]
            
            channel_g_denoised = denoised[...,0]
            channel_r_denoised = denoised[...,1]
            channel_i_denoised = denoised[...,2]
            channel_z_denoised = denoised[...,3]
            
            orig_stacked = np.concatenate([channel_g_orig, channel_r_orig, channel_i_orig, channel_z_orig], axis=1)
            denoised_stacked = np.concatenate([channel_g_denoised, channel_r_denoised, channel_i_denoised, channel_z_denoised], axis=1)
            noised_stacked = np.concatenate([channel_g_noised, channel_r_noised, channel_i_noised, channel_z_noised], axis=1)
            
            cm = plt.cm.ScalarMappable(None, cmap='bwr')
            cmap = cm.get_cmap()
            residual_stacked = (orig_stacked - denoised_stacked) # ** 2
            residual_stacked = cmap(residual_stacked)
            
            
            cm = plt.cm.ScalarMappable(None, cmap='magma')
            cmap = cm.get_cmap()
            pic1 = np.concatenate([orig_stacked, denoised_stacked, noised_stacked], axis=0)
            pic1 = (pic1 - np.min(pic1)) / (np.max(pic1) - np.min(pic1))
            pic1 = cmap(pic1)
            
            pic2 = np.concatenate([pic1, residual_stacked], axis=0)
            
            video_log[f'{name}_plots'].append(wandb.Image(pic2))
        
        return video_log
             

def train(params):
    
    
    hyperparameter_defaults.update(params)
    params = hyperparameter_defaults
    
    store = {}

    architecture_name = params['architecture']

    if (not 'api_key' in params) or (params['api_key'] is None):
        os.environ["WANDB_MODE"] = 'dryrun'
    else:
        os.environ["WANDB_API_KEY"] = params['api_key']
    
    if (not 'name' in params) or (params['name'] is None):
        name = architecture_name
    else:
        name = params['name'] 
        
    if 'gpu' in params and not (params['gpu'] is None):
        os.environ["CUDA_VISIBLE_DEVICES"]=params['gpu']
    
    params['alpha_mixing'] = float(params['alpha_mixing'])
    
    # config_tf = tf.ConfigProto()
    # config_tf.gpu_options.allow_growth = True
    # sess = tf.Session(config=config_tf)
    # keras.backend.set_session(sess)
    
    if params['continue_id']:
        wandb.init(id=params['continue_id'], resume='allow', config=params, project='denoising', tags = [params['architecture']], name=name, entity=ENTITY)
    else:
        wandb.init(config=params, project='denoising', tags = [params['architecture']], name=name, entity=ENTITY)
    
    wandb.run.name = name
 
    config = wandb.config
    
    store['config'] = params
        
    dataKeys = {}
    with h5py.File(params['file'], 'r') as f:
        train_keys, val_keys = train_test_split(list(f.keys()), test_size=0.4, train_size=0.6, random_state=2021, shuffle=True, stratify=None)
        val_keys, test_keys = train_test_split(val_keys, test_size=0.5, train_size=0.5, random_state=2021, shuffle=True, stratify=None)
        dataKeys['train'] = list(zip([params['file']] * len(train_keys), train_keys))
        dataKeys['val'] = list(zip([params['file']] * len(val_keys), val_keys))
        dataKeys['test'] = list(zip([params['file']] * len(test_keys), test_keys))
        
    SYNTHETIC_DATA_FILE = params['gen_file'] # '../../data/Denoising/alae_lat512_full_convolved.h5'
  
    print(f"Training set length: {len(dataKeys['train'])}")
    print(f"Validation set length: {len(dataKeys['val'])}")
    print(f"Test set length: {len(dataKeys['test'])}")
    
    with h5py.File(SYNTHETIC_DATA_FILE, 'r') as f:
        gen_keys = list(f.keys())
        # include params['data_mixing'] * len(train_keys) more generated samples in the training 
        gen_keys = list(zip([SYNTHETIC_DATA_FILE] * len(gen_keys), gen_keys))
    
    orig_train_dataset_length = len(dataKeys['train'])
    
    # if params['data_mixing_orig']:
    #     dataKeys['train'] = dataKeys['train'][:int(float(params['data_mixing_orig']) * orig_train_dataset_length)]
     
    orig_keys = dataKeys['train']
    
    # if params['data_mixing_gen']:
    #     gen_keys = gen_keys[:int(float(params['data_mixing_gen']) * orig_train_dataset_length)]
    
    dataKeys['gen'] = gen_keys
    # dataKeys['train'].extend(gen_keys)
    
    fileList = [params['file'], SYNTHETIC_DATA_FILE]
    # fileList = [params['file']]
    
    print(f"Generated set length: {len(dataKeys['gen'])}")

    
    # trainGenerator = DataLoader(fileList, dataKeys['train'], config['batch_size'], name='train')
    trainGenerator = DataLoader([params['file']], dataKeys['train'], config['batch_size'], name='train')
    genGenerator = DataLoader([SYNTHETIC_DATA_FILE], dataKeys['gen'], config['batch_size'], name='gen')
    valGenerator = DataLoader([params['file']], dataKeys['val'], config['batch_size'], name='val')
    testGenerator = DataLoader([params['file']], dataKeys['test'], config['batch_size'], name='test', shuffle=False)
    augmentation_1 = {'snp' : 0.25, 'downsampling' : 2}
    testGeneratorAugmentation1 = DataLoader([params['file']], dataKeys['test'], config['batch_size'], name='test', shuffle=False, augmentation=augmentation_1)
    augmentation_2 = {'snp' : 0.05, 'downsampling' : 1}
    testGeneratorAugmentation2 = DataLoader([params['file']], dataKeys['test'], config['batch_size'], name='test', shuffle=False, augmentation=augmentation_2)
    augmentation_3 = {'snp' : 0.05, 'downsampling' : 2}
    testGeneratorAugmentation3 = DataLoader([params['file']], dataKeys['test'], config['batch_size'], name='test', shuffle=False, augmentation=augmentation_3)

    mixingGenerator = MixingDataLoader([trainGenerator, genGenerator], [1-params['alpha_mixing'], params['alpha_mixing']], 5000)
    
    id = wandb.run.id

    if config.normalization_data == 'dynamic':
        mean, std = trainGenerator.getNormAndStd()

        trainGenerator.setNormAndStd(mean, std)
        valGenerator.setNormAndStd(mean, std)
        testGenerator.setNormAndStd(mean, std)
       
        print('mean: ', mean)
        print('std: ', std)
        params['std'] = std
        
    params['input_dim'] = (None, None, 4)
    
    
    gpu_list = params['gpu'].split(',')
    # strategy = tf.distribute.MirroredStrategy(devices=[f'/gpu:{gpu}' for gpu in gpu_list])
    # print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    BUFFER_SIZE = 5000 # len(trainGenerator)

    BATCH_SIZE_PER_REPLICA = params['batch_size']
    
    # GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA 
    
    # trainGeneratorDataset = tf.data.Dataset.from_generator(make_generator(trainGenerator), (tf.float32, tf.float32)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
    # mixingGeneratorDataset = tf.data.Dataset.from_generator(make_generator(mixingGenerator), (tf.float32, tf.float32)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
    # valGeneratorDataset = tf.data.Dataset.from_generator(make_generator(valGenerator), (tf.float32, tf.float32)).batch(GLOBAL_BATCH_SIZE)
    # testGeneratorDataset = tf.data.Dataset.from_generator(make_generator(testGenerator), (tf.float32, tf.float32)).batch(GLOBAL_BATCH_SIZE)
    # train_dist_dataset = strategy.experimental_distribute_dataset(trainGeneratorDataset)
    # val_dist_dataset = strategy.experimental_distribute_dataset(valGeneratorDataset)
    
    test_list = [('test', testGenerator)]
    
    best_val_loss = np.inf
    if 'best_val_loss' in wandb.run.summary.keys():
        best_val_loss = wandb.run.summary["best_val_loss"]
    
    if 'last_lr' in wandb.run.summary.keys():
        params['learning_rate'] = wandb.run.summary["last_lr"]
    
    if "epochs_finished" in wandb.run.summary.keys():
        initial_epoch = wandb.run.summary["epochs_finished"]
    else:
        initial_epoch = 0
    
    def lr_scheduler(epoch, lr):
        if epoch > 0 and epoch % 50 == 0:
            return max(lr * 0.8, 0.000001)
        else:
            return lr
            
    # Open a strategy scope.
    # with strategy.scope():
    
    loader = ModelLoader(id = id, **params)

    model = loader.get_model()

    model.summary()
    optimizer = get_optimizer(**params)
    model.compile(loss='mse', optimizer=optimizer)

    # load existing weights if exists
    loader.restore()

    save_path = loader.weight_path

    reduce_lr_loss = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    # reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.8, patience=20, verbose=1, mode='min')
    early_stopping_val_loss = EarlyStopping(monitor='val_loss', patience=500, verbose=1)
    # checkpoint = ModelCheckpoint(save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    
    class NANCallback(keras.callbacks.Callback):
        def on_batch_end(self, batch, logs={}):
            loss = logs.get('loss')
            if math.isnan(loss):
                model.stop_training = True
    
    class CustomWandbCallback(WandbCallback):
        
        def __init__(self, best_val_loss):
            self.best_val_loss = best_val_loss
            super().__init__()
        
        def on_epoch_end(self, epoch, logs={}):

            logs_ = deepcopy(logs)

            wandb.run.summary["epochs_finished"] = epoch
            if logs['val_loss'] < self.best_val_loss:

                print(f'val_loss improved from {self.best_val_loss} to {logs["val_loss"]} -- saving model')
                
                self.best_val_loss = logs['val_loss']
                wandb.run.summary["best_val_loss"] = self.best_val_loss

                next_budget = epoch + (params['budget_steps'] - (epoch % params['budget_steps']))

                #https://github.com/keras-team/keras/issues/11101 solved :)
                #model.save(os.path.join('weights', f'model_{id}_{next_budget}.h5'))

                loader.save_weights(id=id, budget=next_budget)
                loader.save_weights(id=id)
            
            logs_.update(log_videos(gen_keys, genGenerator, loader, name='gen'))
            logs_.update(log_videos(orig_keys, trainGenerator, loader))
            logs_.update(log_videos(dataKeys['val'], valGenerator, loader))
            logs_.update({'epoch' : epoch})

            if 'lr' in logs:
                wandb.run.summary['last_lr'] = logs['lr']

            super().on_epoch_end(epoch, logs_)
    
    callbacks = [reduce_lr_loss, early_stopping_val_loss, CustomWandbCallback(best_val_loss), NANCallback()]

    start_time = time.time()

    # history = model.fit_generator(generator=mixingGeneratorDataset,
    #                     validation_data=valGeneratorDataset, epochs=config.epochs,
    #                     verbose=1, callbacks=callbacks, use_multiprocessing = False, workers = 1, initial_epoch = initial_epoch)
    
    history = model.fit_generator(generator=mixingGenerator,
                        validation_data=valGenerator, epochs=config.epochs,
                        verbose=1, callbacks=callbacks, use_multiprocessing = False, workers = 1, initial_epoch = initial_epoch)
        
    store['loss_history'] = history.history['loss']
    store['val_loss_history'] = history.history['val_loss']
    
    # load best weights
    loader.restore()
    
    NUM_EVALS = 4
    test_eval = []
    for _ in range(NUM_EVALS):
        test_eval.append(model.evaluate(x=testGenerator))
    test_eval_augmentation1 = []
    for _ in range(NUM_EVALS):
        test_eval_augmentation1.append(model.evaluate(x=testGeneratorAugmentation1))
    test_eval_augmentation2 = []
    for _ in range(NUM_EVALS):
        test_eval_augmentation2.append(model.evaluate(x=testGeneratorAugmentation2))
    test_eval_augmentation3 = []
    for _ in range(NUM_EVALS):
        test_eval_augmentation3.append(model.evaluate(x=testGeneratorAugmentation3))
    
    for i, v in zip(range(NUM_EVALS), test_eval):
        wandb.run.summary[f'test_loss_{i}'] = v 
        
    for i, v in zip(range(NUM_EVALS), test_eval_augmentation1):
        wandb.run.summary[f'test_loss_augmentation1_{i}'] = v 
        
    for i, v in zip(range(NUM_EVALS), test_eval_augmentation2):
        wandb.run.summary[f'test_loss_augmentation2_{i}'] = v 
        
    for i, v in zip(range(NUM_EVALS), test_eval_augmentation3):
        wandb.run.summary[f'test_loss_augmentation3_{i}'] = v 
    
    store['test_eval'] = test_eval
    store['test_eval_augmentation1'] = test_eval_augmentation1
    store['test_eval_augmentation2'] = test_eval_augmentation2
    store['test_eval_augmentation3'] = test_eval_augmentation3
                      
    wandb.run.summary['runtime'] = time.time() - start_time
    store['runtime'] = wandb.run.summary['runtime']
    
        
    store_path = os.path.join(file_directory, 'store', f'{id}.p')
    with open(store_path, 'wb') as handle:
        pickle.dump(store, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__== "__main__":

    parser = argparse.ArgumentParser(description='Parameter Parser', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--architecture', default=hyperparameter_defaults['architecture'], help='Architecture of transfer network')
    parser.add_argument('--name', default=None, help='Name of experiment')
    
    parser.add_argument('--continue-id', default=None, help='ID of run to continue')
    parser.add_argument('--gpu', default=None, help='Visible GPUs')
    parser.add_argument('--alpha-mixing', default=0.0, help='Dataset mixing coefficient')
    parser.add_argument('--api-key', default=hyperparameter_defaults['api_key'], help='Wandb API key')
    
    parser.add_argument('--file', help='Data file (training data)', required=True)
    parser.add_argument('--gen-file', default=None, help='Data file (generated data)', required=True)
    args, unknown = parser.parse_known_args()
   
    hyperparameter_defaults.update(vars(args))
    train(hyperparameter_defaults)
