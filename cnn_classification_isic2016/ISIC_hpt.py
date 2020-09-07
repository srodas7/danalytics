#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 09:08:36 2018

@author: sherryrodas
"""

import h5py
import numpy as np

filename = 'isic2016/isic_data_augci_rot.h5'
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
from talos.model.normalizers import lr_normalizer
from keras.optimizers import SGD, Adam,RMSprop,Adagrad,Adadelta,Nadam # optimization algorithms
import talos
from Models import cnn_model, OutputResults

seed = 7
np.random.seed(seed)


parameters = {'learning_rate': [1,5,10,0.5,0.1],
     'num_kernels':[16,32,64],
     'batch_size': [50,100,150],
     'drop_out': [0.2,0.3],
     'optimizer': [Adam,Adadelta],
     'activation':['relu'],
     'kernel_initializer': ['glorot_uniform','glorot_normal','RandomNormal','RandomUniform'],
     'cnn_layers' :[0,1,2,3]
     }

parameters_small = {'learning_rate': [1],
     'num_kernels':[32,40,64,90],
     'batch_size': [50],
     'drop_out': [0.3],
     'optimizer': [Adam],
     'activation':['relu'],
     'kernel_initializer': ['glorot_uniform'],
     'h_nodes': [100, 128],
     'kernel_size': [3,5,9]
     }


##probabilistic reduction based on correlation
t_nonrandom_downsampling = talos.Scan(x=x_train,
            y=y_train,
            model=cnn_model,
            dataset_name='isic_hpt',
            experiment_no='2',
            params=parameters_small,
            reduction_method='correlation',
            reduction_interval = 10,
            reduction_window = 10,
            grid_downsample=1,
            talos_log_name='hpt_nonrandom') 

try:
    talos.Deploy(t_nonrandom_downsampling,'t_nonrandom_tuning')
except:
    print(" ") 


OutputResults(x_train,y_train,'t_nonrandom_tuning/t_nonrandom_tuning_model.json','t_nonrandom_tuning/t_nonrandom_tuning_model.h5')
OutputResults(x_test,y_test,'t_nonrandom_tuning/t_nonrandom_tuning_model.json','t_nonrandom_tuning/t_nonrandom_tuning_model.h5')
 












