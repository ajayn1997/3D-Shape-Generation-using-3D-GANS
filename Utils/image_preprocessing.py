# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 12:10:39 2020

@author: theaj
"""
import numpy as np
import glob
import scipy.io as io
import scipy.ndimage as nd
import matplotlib.pyplot as plt



def get3DImages(data_dir):
    all_files = np.random.choice(glob.glob(data_dir), size=10)
    # all_files = glob.glob(data_dir)
    all_volumes = np.asarray([getVoxelsFromMat(f) for f in all_files], dtype=np.bool)
    return all_volumes

def getVoxelsFromMat(path, cube_length=64):
    voxels = io.loadmat(path)['instance']
    '''
    Size of mat Voxel array is (30,30,30). The network 
    requires the size of the voxel to be (64,64,64). 
    We use np.pad() function to add 0s to the array and
    increase the size to (32,32,32).
    '''
    voxels = np.pad(voxels, (1,1), 'constant', constant_values=(0,0))
    '''
    The scipy.ndimage library has a function called zoom,
    which increases the size of ndimage without loss of data.
    We use it here to increase the size of voxel from
    (32,32,32) to (64,64,64).
    '''
    if cube_length!=32 and cube_length==64:
        voxels = nd.zoom(voxels, (2,2,2), mode='constant', order=0)
    return voxels

def saveFromVoxels(voxels, path):
    '''
    function to plot the voxel as a 3d
    projection and save the plot to the given path
    '''
    z, x, y = voxels.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c='red')
    plt.savefig(path)
        