#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 17:28:28 2018

@author: sherryrodas
"""

import h5py
import numpy as np

#filename = 'Downloads/isic2016/isic_data.h5'
filename = 'isic2016/isic_data.h5'
f = h5py.File(filename, 'r')
x_train = np.array(f['x_train'])
y_train = np.array(f['y_train'])
x_test = np.array(f['x_test'])
y_test = np.array(f['y_test'])

x_train = x_train/255
x_test = x_test/255

print('training size ', x_train.shape)
print('testing size ', x_train.shape)
print('training class labels ', y_train.shape)
print('testing class labels ', y_test.shape)

# =============================================================================
# from keras.utils import np_utils
# NB_CLASSES = 2
# print('shape of y_train and y_test before categorical conversion')
# print(y_train.shape)
# print(y_test.shape)
# y_train = np_utils.to_categorical(y_train, NB_CLASSES)
# y_test = np_utils.to_categorical(y_test, NB_CLASSES)
# print('shape of y_train and y_test after categorical conversion')
# print(y_train.shape)
# print(y_test.shape)
# =============================================================================

#Function to print metrics
    
def GetMetrics(model,predictors,response):
    #y_classes = [np.argmax(y, axis=None, out=None) for y in response]
    pred_class = model.predict_classes(predictors)
    from sklearn.metrics import accuracy_score
    print('accuracy: ', accuracy_score(response,pred_class))
    from sklearn.metrics import cohen_kappa_score
    print('kappa ', cohen_kappa_score(response, pred_class))
    from sklearn.metrics import confusion_matrix
    print('confusion_matrix\n', confusion_matrix(response, pred_class))
    from sklearn.metrics import roc_auc_score
    print('roc_auc_score ', roc_auc_score(response, pred_class))


from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten

import matplotlib
import matplotlib.pyplot as plt

# Function to get plots
def plotHistory(Tuning,pltname):
    fig,axs = plt.subplots(1,2,figsize=(15,5))
    axs[0].plot(Tuning.history['loss'])
    axs[0].plot(Tuning.history['val_loss'])
    axs[0].set_title('loss vs epoch')
    axs[0].set_ylabel('loss')
    axs[0].set_xlabel('epoch')
    axs[0].legend(['train','vali'], loc='upper left')
    
    axs[1].plot(Tuning.history['acc'])
    axs[1].plot(Tuning.history['val_acc'])
    axs[1].set_title('accuracy vs epoch')
    axs[1].set_ylabel('accuracy')
    axs[1].set_xlabel('epoch')
    axs[1].set_ylim([0.0,1.0])
    axs[1].legend(['train','vali'], loc='upper left') 
    plt.savefig('opt_history/'+str(pltname)+'.png')
    plt.show(block=False)
    plt.show()

NB_EPOCH = 50
BATCH_SIZE = 50
VERBOSE = 1
NB_CLASSES = 2
OPTIMIZER = Adam()
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2
METRICS = ['accuracy']
LOSS = 'categorical_crossentropy'

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

IMG_ROWS = 120
IMG_COLS = 160
IMG_CHANNELS = 3 #color rgb

# Model 1
#input shape
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)
# 2D CNN layer (use 'Conv2D' in Keras) with 30 kernels (5x5 size)
NUM_KERNELS = 64
cnn_model1 = Sequential()
cnn_model1.add(Conv2D(NUM_KERNELS,kernel_size=(5,5),padding="valid",kernel_initializer='glorot_uniform',input_shape = INPUT_SHAPE, data_format='channels_last'))
cnn_model1.add(Activation("relu"))
cnn_model1.add(MaxPooling2D(pool_size=(2,2)))
cnn_model1.add(BatchNormalization())
cnn_model1.add(Dropout(0.2))

cnn_model1.add(Flatten())
cnn_model1.add(Dense(50))
cnn_model1.add(Activation("relu"))
cnn_model1.add(BatchNormalization())
cnn_model1.add(Dropout(0.3))

cnn_model1.add(Dense(NB_CLASSES))
cnn_model1.add(Activation("softmax"))
cnn_model1.compile(loss=LOSS,optimizer=OPTIMIZER,metrics=METRICS)

print(cnn_model1.summary())

#filepath = 'Downloads/cifar/cifar_best_cnn_model1.hdf5'
filepath = 'isic2016/best_cnn_model1_augci.hdf5'
checkpoint = ModelCheckpoint(filepath,monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_monitor = EarlyStopping(monitor='val_loss', patience = 8)

Tuning_cnn_model1 = cnn_model1.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT, callbacks=[checkpoint,early_stopping_monitor])


# Model 2
cnn_model2=Sequential()
cnn_model2.add(Conv2D(64,kernel_size=(5,5),padding="valid",kernel_initializer="glorot_uniform",input_shape=INPUT_SHAPE))
cnn_model2.add(Activation("relu"))
cnn_model2.add(MaxPooling2D(pool_size=(2,2)))
cnn_model2.add(BatchNormalization())
cnn_model2.add(Dropout(0.2))

cnn_model2.add(Conv2D(64,kernel_size=(3,3),padding="valid",kernel_initializer="glorot_uniform"))
cnn_model2.add(Activation("relu"))
cnn_model2.add(MaxPooling2D(pool_size=(2,2)))
cnn_model2.add(BatchNormalization())
cnn_model2.add(Dropout(0.3))

cnn_model2.add(Flatten())
cnn_model2.add(Dense(100))
cnn_model2.add(Activation("relu"))
cnn_model2.add(BatchNormalization())
cnn_model2.add(Dropout(0.5))

cnn_model2.add(Dense(NB_CLASSES))
cnn_model2.add(Activation("softmax"))
cnn_model2.compile(loss=LOSS,optimizer=OPTIMIZER,metrics=METRICS)

print(cnn_model2.summary())

filepath = 'isic2016/best_cnn_model2_augci.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_monitor = EarlyStopping(monitor='val_loss',patience=8)

Tuning_cnn_model2 = cnn_model2.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT, callbacks=[checkpoint,early_stopping_monitor])


# Explore Layers for best 2 Models
from keras.models import load_model
filepath = 'isic2016/best_cnn_model1_augci.hdf5'
finalModel1 = load_model(filepath)
print("Layers in Model 1")
print(finalModel1.layers)
print(" ")

filepath = 'isic2016/best_cnn_model2_augci.hdf5'
finalModel2 = load_model(filepath)
print("Layers in Model 2")
print(finalModel2.layers)
print(" ")


# Get Accuracy for both models
print("Training Accuracy for 1 CNN Layer Model:")
print(GetMetrics(cnn_model1,x_train,y_train))
print("Testing Accuracy for 1 CNN Layer Model:")
print(GetMetrics(cnn_model1,x_test,y_test))
print(" ")
print("Training Accuracy for 2 CNN Layer Model:")
print(GetMetrics(cnn_model2,x_train,y_train))
print("Testing Accuracy for 2 CNN Layer Model:")
print(GetMetrics(cnn_model2,x_test,y_test))

# Get Optimization History Plot for both models
plotHistory(Tuning_cnn_model1, '1cnn_softmax')
plotHistory(Tuning_cnn_model2, '2cnn_softmax')







