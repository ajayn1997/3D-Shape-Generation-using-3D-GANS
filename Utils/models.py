# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 13:30:07 2020

@author: theaj
"""

from keras import Sequential
from keras.layers import Input
from keras.layers.convolutional import Conv3D, Deconv3D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU

def build_Generator():
    '''
    Build the Generator Model of the 3D-GAN.
    Returns: Generator Network
    '''
    
    # Hyperparameters
    z_size = 200
    gen_filters = [512, 256, 128, 64, 1]
    gen_kernel_sizes = [4, 4, 4, 4, 4]
    gen_strides = [1, 2, 2, 2, 2]
    gen_input_shape = (1, 1, 1, z_size)
    gen_activations = ['relu', 'relu', 'relu', 'relu', 'sigmoid']
    gen_convolution_blocks = 5
    
    # Input layer of generator. Input is a vector sampled out a prob distribution
    input_layer = Input(shape=gen_input_shape)
    
    # 1st 3D Deconvolution block
    a = Deconv3D(filters = gen_filters[0],
                 kernel_size = gen_kernel_sizes[0],
                 strides = gen_strides[0])(input_layer)
    a = BatchNormalization()(a, training=True)
    a = Activation(gen_activations[0])(a)
    
    # Next 4 3D Deconvolution blocks
    for i in range(gen_convolution_blocks - 1):
        a = Deconv3D(filters = gen_filters[i + 1],
                 kernel_size = gen_kernel_sizes[i + 1],
                 strides = gen_strides[i + 1])(a)
        a = BatchNormalization()(a, training=True)
        a = Activation(gen_activations[0])(a)
    
    # Putting all the blocks together in a Model object
    gen_model = Model(inputs=[input_layer], outputs=[a])
    return gen_model
    
def build_Discriminator():
    '''
    Build Discrimator Model of the GAN
    Returns: Discriminator Network
    '''
    # Hyperparameters
    dis_input_shape = (64, 64, 64, 1)
    dis_filters = [64, 128, 216, 512, 1]
    dis_kernel_sizes = [4, 4, 4, 4, 4]
    dis_strides = [2, 2, 2, 2, 1]
    dis_paddings = ['same', 'same', 'same', 'same', 'valid']
    dis_alphas = [0.2, 0.2, 0.2, 0.2, 0.2]
    dis_activations = ['leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu', 'sigmoid']
    dis_convolution_blocks = 5
    
    # Input layer of discriminator. Input of discriminator is a 3D image of size (64,64,64,1)
    dis_input_layer = Input(shape = dis_input_shape)
    
    # 1st 3D convolution block.
    a = Conv3D(filters = dis_filters[0],
               kernel_size = dis_kernel_sizes[0],
               strides = dis_strides[0],
               padding = dis_paddings[0])(dis_input_layer)
    a = BatchNormalization()(a, training=True)
    a = LeakyReLU(dis_alphas[0])(a)
    
    # The next 4 3D convolution blocks
    for i in range(dis_convolution_blocks - 1):
        a = Conv3D(filters = dis_filters[i + 1],
                   kernel_size = dis_kernel_sizes[i + 1],
                   strides = dis_strides[i + 1],
                   padding = dis_paddings[i + 1])(a)
        a = BatchNormalization()(a, training=True)
        if dis_activations[i + 1] == 'leaky_relu':
            a = LeakyReLU(dis_alphas[i + 1])(a)
        else:
            a = Activation(activation='sigmoid')(a)
    
    # Putting all the blocks together
    dis_model = Model(inputs=[dis_input_layer], outputs=[a])
    return dis_model
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    