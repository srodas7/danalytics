#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 12:46:24 2018

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


# SPLIT TRAIN DATA PROVIDED INTO TRAIN AND TEST DATA

# Read in labels file
class_label_file = 'isic2016/Part3_Train_GroundTruth.csv'
f = open(class_label_file)
lines = f.read().splitlines()

# Create dictionary from labels file where key: value = Filename: Class
train_labels = {}
#dict.keys(), dict.values(), dict.items()
f.close()
for i in range(0,len(lines)):
    tmp = lines[i].split(",")
    filename = tmp[0] + '_Segmentation.png'
    Class = 0 if tmp[1] == 'benign' else 1
    train_labels[filename] = Class
print('number of training images ', len(train_labels.keys()))

# Get list of all files in the Training Images directory provided to us
imagedir = 'Data_BW/'
files = listdir(imagedir)
print("number of training images ", len(files))
print("First 5 files ", files[0:5])

# Split into Train and Test datasets
ind = list(range(0,900))
train_ind, test_ind = train_test_split(ind, test_size=0.3, random_state=123)



# CREATE NEW DIRECTORIES FOR SPLIT DATASETS AND NEW LABELS FILE

# Create new directory for Train images and move Train images there
os.system('mkdir -p isic2016/train_bw/')
train_folder = 'isic2016/train_bw/'
for i in range(0,len(train_ind)):
    I = io.imread(imagedir+files[train_ind[i]])
    io.imsave(train_folder+files[train_ind[i]], I)

# Create new directory for Test images and move Test images there
os.system('mkdir -p isic2016/test_bw/')
test_folder = 'isic2016/test_bw/'
for i in range(0,len(test_ind)):
    I = io.imread(imagedir+files[test_ind[i]])
    io.imsave(test_folder+files[test_ind[i]], I)

# Create new dictionary with Train File Name as Keys and Class as Value
train_keys = listdir(train_folder)
new_train_labels = {x:train_labels[x] for x in train_keys}   

# Create new dictionary with Test File Name as Keys and Class as Value
test_keys = listdir(test_folder)
new_test_labels = {x:train_labels[x] for x in test_keys} 
    
# Create new label file for Train dataset by using the new train labels dictionary
new_train_label_file = 'isic2016/Part3_Train_Labels_BW.csv'
f = open(new_train_label_file, "w")
for keys in new_train_labels.keys():
    f.write(keys + "," + str(new_train_labels[keys]) + "\n")
f.close()
print("Finished creating BW Train Labels with: " + str(len(new_train_labels.keys())) + "files")

# Create new label file for Train dataset by using the new test labels dictionary
new_test_label_file = 'isic2016/Part3_Test_Labels_BW.csv'
f = open(new_test_label_file, "w")
for keys in new_test_labels.keys():
    f.write(keys + "," + str(new_test_labels[keys]) + "\n")
f.close()
print("Finished creating BW Test Labels with: " + str(len(new_test_labels.keys())) + "files")
