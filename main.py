# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 11:27:18 2020

@author: theaj
"""
from keras.optimizers import Adam
from Utils.models import build_Generator, build_Discriminator
from Utils.image_preprocessing import get3DImages, saveFromVoxels
from keras.layers import Input
from keras.models import Model
from keras.callbacks import TensorBoard
import tensorflow as tf
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import time
import os

def write_log(callback, name, value, batch_no):
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = value
    summary_value.tag = name
    callback.writer.add_summary(summary, batch_no)
    callback.writer.flush()



def main():
    
    '''
    Hyperparameter specification
    '''
    object_name = 'airplane'
    data_dir = "data/3DShapeNets/volumetric_data/{}/30/train/*.mat".format(object_name)
    gen_learning_rate = 0.0025
    dis_learning_rate = 10e-5
    beta = 0.5
    batch_size = 1
    z_size = 200
    epochs = 10
    MODE = 'train'
    
    # Create the models
    
    gen_optimizer = Adam(lr=gen_learning_rate, beta_1=beta)
    dis_optimizer = Adam(lr=dis_learning_rate, beta_1=beta)
    
    discriminator = build_Discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer=dis_optimizer)
    
    generator = build_Generator()
    generator.compile(loss='binary_crossentropy', optimizer=gen_optimizer)
    
    discriminator.trainable = False
    
    # Putting both generator and discrimator together to form a GAN
    
    input_layer = Input(shape=(1, 1, 1, z_size))
    generated_volumes = generator(input_layer)
    validity = discriminator(generated_volumes)
    adversial_model = Model(inputs=[input_layer], outputs=[validity])
    adversial_model.compile(loss='binary_crossentropy', optimizer=gen_optimizer)
    
    # Load datasets
    print('Loading data.....')
    volumes = get3DImages(data_dir)
    # print(volumes)
    volumes = volumes[...,np.newaxis].astype(np.float)
    print('Data loaded')
    
    # TensorBoard init
    tensorboard = TensorBoard(log_dir='logs/{}'.format(time.time()))
    tensorboard.set_model(generator)
    tensorboard.set_model(discriminator)
    
    labels_real = np.reshape(np.ones((batch_size,)), (-1, 1, 1, 1, 1))
    labels_fake = np.reshape(np.zeros((batch_size,)), (-1, 1, 1, 1, 1))    
    
    if MODE=='train':
        for epoch in range(epochs):
            print('Epoch:' , epoch)
            
            gen_losses = []
            dis_losses = []
            
            number_of_batches = int(volumes.shape[0] / batch_size)
            print('Number of batches:' , number_of_batches)
            for index in range(number_of_batches):
                print('Batch:' , index + 1)

                z_sample = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
                volumes_batch = volumes[index * batch_size:(index + 1) * batch_size, :, :, :]

                # Next, generate volumes using generator network
                gen_volumes = generator.predict_on_batch(z_sample)

                '''
                Train the discriminator network
                '''
                discriminator.trainable = True
                if index % 2 == 0:
                    loss_real = discriminator.train_on_batch(volumes_batch, labels_real)
                    loss_fake = discriminator.train_on_batch(gen_volumes, labels_fake)

                    d_loss = 0.5 * np.add(loss_real, loss_fake)
                    print('d_loss: {}'.format(d_loss))
                else:
                    d_loss = 0
                discriminator.trainable = False
                '''
                Train the generator network
                '''
                z = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
                g_loss = adversial_model.train_on_batch(z, labels_real)
                print('g_loss:{}'.format(g_loss))

                gen_losses.append(g_loss)
                dis_losses.append(d_loss)

                # Every 10th mini-batch , generate volumes and save them
                if index % 10 == 0:
                    z_sample2 = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
                    generated_volumes = generator.predict(z_sample2, verbose=3)
                    for i, generated_volume in enumerate(generated_volumes[:5]):
                        voxels = np.squeeze(generated_volume)
                        voxels[voxels < 5] = 0.
                        voxels[voxels >= 5] = 1.
                        saveFromVoxels(voxels, "results/img_{}_{}_{}".format(epoch, index, i))

            # Write losses to Tensorboard
            write_log(tensorboard, 'g_loss', np.mean(g_loss), epoch)
            write_log(tensorboard, 'd_loss', np.mean(d_loss), epoch)

        generator.save_weights(os.path.join("models", "generator_weights.h5"))
        discriminator.save_weights(os.path.join("models", "discriminator_weights.h5"))    
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
               