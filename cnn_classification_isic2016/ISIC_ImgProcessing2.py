#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 15:36:41 2018

@author: sherryrodas
"""

import skimage
import os
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import h5py
from skimage import io
from os import listdir
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage import transform
from sklearn.model_selection import train_test_split



# Train

# Read in labels file
class_label_file = 'isic2016/Part3_TrainCropAug_CI_Labels.csv'
f = open(class_label_file)
lines = f.read().splitlines()
train_labels = {}
f.close()

# Create dictionary from labels file where key: value = Filename: Class
for i in range(0,len(lines)):
    tmp = lines[i].split(",")
    filename = tmp[0]
    Class = tmp[1]
    train_labels[filename] = Class
print('number of training images ', len(train_labels.keys()))

# Get list of images in augmented train directory
#imagedir = 'Downloads/isic2016/train_augmented/'
imagedir = 'isic2016/train_crop_aug_ci_rot/'
files = listdir(imagedir)
print("number of training images ", len(files))
print("First 5 files ", files[0:5])

# Get Dimensions for empty array
I = io.imread(imagedir+files[0])
io.imshow(I)
plt.show()
print("Dimensions of the image ", I.shape)
       
# Get Dimensions of Reduced Size for empty array
I = transform.resize(I,(120, 160)) #can change to 96x128 for smaller dimension
I.shape
io.imshow(I)

# Process Reduced Images into COLOR vectors with Labels
x_train = np.empty(shape=(len(files),I.shape[0],I.shape[1],I.shape[2]),dtype=np.int)
y_train = np.empty(shape=(len(files),), dtype=np.int)
print(x_train.shape)
print(y_train.shape)

# Create x_train (array with images) and y_train (array with labels) in the same order
for i in range(0,len(files)):
    if (i%100 == 0):
        print('done processing ' + str(i) + ' images')
    I = io.imread(imagedir+files[i])
    I = transform.resize(I,(120, 160)) #Reduce Size
    #I = rgb2gray(I)
    #I = I.reshape(I.shape[0],I.shape[1],1)
    x_train[i,:,:,:] = I #train aug files image vectors
    y_train[i] = train_labels[files[i]] #train aug file label to file image


# Test

# Read in labels file
class_label_file = 'isic2016/Part3_Test_Labels.csv'
f = open(class_label_file)
lines = f.read().splitlines()
test_labels = {}
f.close()

# Create dictionary from labels file where key: value = Filename: Class
for i in range(0,len(lines)):
    tmp = lines[i].split(",")
    filename = tmp[0]
    Class = tmp[1]
    test_labels[filename] = Class
print('number of testing images ', len(test_labels.keys()))

# Get list of images in augmented test directory
imagedir = 'isic2016/test_cropped/'
files = listdir(imagedir)
print('number of testing images ', len(files))
print('first 5 files ', files[0:5])

# Process Reduced Images into COLOR vectors with Labels
x_test = np.empty(shape=(len(files),I.shape[0],I.shape[1],I.shape[2]),dtype=np.int)
y_test = np.empty(shape=(len(files),), dtype=np.int)
print(x_test.shape)
print(y_test.shape)

# Create x_train (array with images) and y_train (array with labels) in the same order
for i in range(0,len(files)):
    if (i%100 == 0):
        print('done processing ' + str(i) + ' images')
    I = io.imread(imagedir+files[i])
    I = transform.resize(I,(120, 160)) #Reduce Size
    #I = rgb2gray(I)
    #I = I.reshape(I.shape[0],I.shape[1],1)
    x_test[i,:,:,:] = I #test files image vectors
    y_test[i] = test_labels[files[i]] #test file label to file image


# Store into h5 matrices
hf = h5py.File("isic2016/isic_crop_aug_ci.h5", 'w')
hf.create_dataset('x_train', data=x_train)
hf.create_dataset('y_train', data=y_train)
hf.create_dataset('x_test', data=x_test)
hf.create_dataset('y_test', data=y_test)
hf.close()
