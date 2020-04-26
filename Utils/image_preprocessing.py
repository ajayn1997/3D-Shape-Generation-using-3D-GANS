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


def getVoxelsFromMat(path, cube_len=64):
    voxels = io.loadmat(path)['instance']
    print(voxels)
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
    if cube_len != 32 and cube_len == 64:
        voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
    return voxels


def saveFromVoxels(voxels, path):
    z, x, y = voxels.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c='red')
    plt.savefig(path)


def plotAndSaveVoxel(file_path, voxel):
    """
    Plot a voxel
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    ax.voxels(voxel, edgecolor="red")
    # plt.show()
    plt.savefig(file_path)