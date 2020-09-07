#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 09:08:36 2018

@author: sherryrodas
"""

import h5py
import numpy as np

filename = 'isic2016/isic_bw_aug_ci.h5'
#filename = 'isic2016/isic_crop_aug_ci.h5'
f = h5py.File(filename, 'r')
x_train = np.array(f['x_train'])
x_test = np.array(f['x_test'])
y_train = np.array(f['y_train'])
y_test = np.array(f['y_test'])


#normalize between 0 and 1
x_train = x_train/255
x_test  = x_test/255

print('training size ', x_train.shape)
print('testing size ',x_test.shape)


from keras.utils import np_utils
NB_CLASSES = 2
print('shape of y_train and y_test before categorical conversion')
print(y_train.shape)
print(y_test.shape)
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)
print('shape of y_train and y_test after categorical conversion')
print(y_train.shape)
print(y_test.shape)


from keras.models import Sequential # what kind of model ? a sequenctial model
from keras.layers.core import Dense, Activation, Dropout # different layers, activation function, and dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam,RMSprop,Adagrad,Adadelta,Nadam # optimization algorithms
from Models import Model5, Model5d, GetMetrics, plotHistory

Model5(x_train,y_train,x_test,y_test,'isic2016/bw_cnn5.hdf5',1) 
Model5d(x_train,y_train,x_test,y_test,'isic2016/bw_cnn5_1d.hdf5',1)

#Model5(x_train,y_train,x_test,y_test,'isic2016/crop_cnn5.hdf5',3) 
#Model5d(x_train,y_train,x_test,y_test,'isic2016/crop_cnn5_1d.hdf5', 3)












