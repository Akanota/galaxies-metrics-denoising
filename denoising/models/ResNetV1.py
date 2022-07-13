# Copyright (c) 2022, Benjamin Holzschuh

from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow import keras
import tensorflow as tf

import numpy as np

def loadModel(input_dim, **kwargs):
        
        # Encoder
        input = keras.layers.Input(shape=input_dim)
        enc_block_0 = keras.layers.Conv2D(filters=32, kernel_size=5, padding='same')(input)
        enc_block_0 = keras.layers.LeakyReLU()(enc_block_0)

        enc_l_conv1 = keras.layers.Conv2D(filters=32, kernel_size=5, padding='same')(enc_block_0)
        enc_l_conv1 = keras.layers.LeakyReLU()(enc_l_conv1)
        enc_l_conv2 = keras.layers.Conv2D(filters=32, kernel_size=5, padding='same')(enc_l_conv1)
        enc_l_skip1 = keras.layers.add([enc_block_0, enc_l_conv2])
        enc_block_1 = keras.layers.LeakyReLU()(enc_l_skip1)

        enc_l_conv3 = keras.layers.Conv2D(filters=32, kernel_size=5, padding='same')(enc_block_1)
        enc_l_conv3 = keras.layers.LeakyReLU()(enc_l_conv3)
        enc_l_conv4 = keras.layers.Conv2D(filters=32, kernel_size=5, padding='same')(enc_l_conv3)
        enc_l_skip2 = keras.layers.add([enc_block_1, enc_l_conv4])
        enc_block_2 = keras.layers.LeakyReLU()(enc_l_skip2)

        enc_l_conv5 = keras.layers.Conv2D(filters=32, kernel_size=5, padding='same')(enc_block_2)
        enc_l_conv5 = keras.layers.LeakyReLU()(enc_l_conv5)
        enc_l_conv6 = keras.layers.Conv2D(filters=32, kernel_size=5, padding='same')(enc_l_conv5)
        enc_l_skip3 = keras.layers.add([enc_block_2, enc_l_conv6])
        enc_block_3 = keras.layers.LeakyReLU()(enc_l_skip3)

        enc_l_conv7 = keras.layers.Conv2D(filters=32, kernel_size=5, padding='same')(enc_block_3)
        enc_l_conv7 = keras.layers.LeakyReLU()(enc_l_conv7)
        enc_l_conv8 = keras.layers.Conv2D(filters=32, kernel_size=5, padding='same')(enc_l_conv7)
        enc_l_skip4 = keras.layers.add([enc_block_3, enc_l_conv8])
        enc_block_4 = keras.layers.LeakyReLU()(enc_l_skip4)

        enc_l_conv9 = keras.layers.Conv2D(filters=32, kernel_size=5, padding='same')(enc_block_4)
        enc_l_conv9 = keras.layers.LeakyReLU()(enc_l_conv9)
        enc_l_convA = keras.layers.Conv2D(filters=32, kernel_size=5, padding='same')(enc_l_conv9)
        enc_l_skip5 = keras.layers.add([enc_block_4, enc_l_convA])
        enc_block_5 = keras.layers.LeakyReLU()(enc_l_skip5)

        encoder = keras.layers.Conv2D(filters=2,  kernel_size=5, padding='same')(enc_block_5)
        
        # Decoder
        dec_block_0 = keras.layers.Conv2DTranspose(filters=32, kernel_size=5, padding='same')(encoder)
        dec_block_0 = keras.layers.LeakyReLU()(dec_block_0)

        dec_l_conv1 = keras.layers.Conv2DTranspose(filters=32, kernel_size=5, padding='same')(dec_block_0)
        dec_l_conv1 = keras.layers.LeakyReLU()(dec_l_conv1)
        dec_l_conv2 = keras.layers.Conv2DTranspose(filters=32, kernel_size=5, padding='same')(dec_l_conv1)
        dec_l_skip1 = keras.layers.add([dec_block_0, dec_l_conv2])
        dec_block_1 = keras.layers.LeakyReLU()(dec_l_skip1)

        dec_l_conv3 = keras.layers.Conv2DTranspose(filters=32, kernel_size=5, padding='same')(dec_block_1)
        dec_l_conv3 = keras.layers.LeakyReLU()(dec_l_conv3)
        dec_l_conv4 = keras.layers.Conv2DTranspose(filters=32, kernel_size=5, padding='same')(dec_l_conv3)
        dec_l_skip2 = keras.layers.add([dec_block_1, dec_l_conv4])
        dec_block_2 = keras.layers.LeakyReLU()(dec_l_skip2)

        dec_l_conv5 = keras.layers.Conv2DTranspose(filters=32, kernel_size=5, padding='same')(dec_block_2)
        dec_l_conv5 = keras.layers.LeakyReLU()(dec_l_conv5)
        dec_l_conv6 = keras.layers.Conv2DTranspose(filters=32, kernel_size=5, padding='same')(dec_l_conv5)
        dec_l_skip3 = keras.layers.add([dec_block_2, dec_l_conv6])
        dec_block_3 = keras.layers.LeakyReLU()(dec_l_skip3)

        dec_l_conv7 = keras.layers.Conv2DTranspose(filters=32, kernel_size=5, padding='same')(dec_block_3)
        dec_l_conv7 = keras.layers.LeakyReLU()(dec_l_conv7)
        dec_l_conv8 = keras.layers.Conv2DTranspose(filters=32, kernel_size=5, padding='same')(dec_l_conv7)
        dec_l_skip4 = keras.layers.add([dec_block_3, dec_l_conv8])
        dec_block_4 = keras.layers.LeakyReLU()(dec_l_skip4)

        dec_l_conv9 = keras.layers.Conv2DTranspose(filters=32, kernel_size=5, padding='same')(dec_block_4)
        dec_l_conv9 = keras.layers.LeakyReLU()(dec_l_conv9)
        dec_l_convA = keras.layers.Conv2DTranspose(filters=32, kernel_size=5, padding='same')(dec_l_conv9)
        dec_l_skip5 = keras.layers.add([dec_block_4, dec_l_convA])
        dec_block_5 = keras.layers.LeakyReLU()(dec_l_skip5)

        # TODO: filters should depend on input_tensor
        decoder = keras.layers.Conv2DTranspose(filters=4, kernel_size=5, padding='same')(dec_block_5)
        
        return {'encoder' : encoder, 'decoder': decoder, 'input': input, 'full_model': decoder}
