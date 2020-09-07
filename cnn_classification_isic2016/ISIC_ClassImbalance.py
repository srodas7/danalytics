#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 12:52:33 2018

@author: sherryrodas
"""
import skimage
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from skimage import io
from os import listdir
from skimage.color import rgb2gray
from skimage.transform import resize
from numpy import flip
from PIL import ImageEnhance
from PIL import Image

# Open train augmented labels file and put into a dictionary
class_label_file = 'isic2016/Part3_TrainAug_Labels.csv'
f = open(class_label_file)
lines = f.read().splitlines()
aug_train_labels = {}
f.close()
for i in range(0,len(lines)):
    tmp = lines[i].split(",")
    filename = tmp[0]
    Class = tmp[1]
    aug_train_labels[filename] = Class

# Get count for Malignant and Benign in Aug Train Labels
malignant_count = len([v for v in aug_train_labels.values() if v == '1'])
benign_count = len([v for v in aug_train_labels.values() if v == '0'])
print(malignant_count)
print(benign_count)

# Get filename and label of original train dataset
o_class_label_file = 'isic2016/Part3_Train_Labels.csv'
f = open(o_class_label_file)
lines = f.read().splitlines()
o_train_labels = {}
f.close()
for i in range(0, len(lines)):
    tmp = lines[i].split(",")
    filename = tmp[0]
    Class = tmp[1]
    o_train_labels[filename] = Class

# Get dictionary subset of malignant samples
malignant_samples = {k:v for (k,v) in o_train_labels.items() if v == '1'}


# Add new images to Augmented directory
imagedir = 'isic2016/train/'
augmentation_folder = 'isic2016/train_augmented/'
ci_train_labels = {}
for key in malignant_samples.keys():
    if (i%50 == 0):
        print('Done processing ' + str(i) + ' images')
    I = io.imread(imagedir+key)
    # Flip Vertical
    Iv = flip(I,0)
    v_file = "vertical_" + key
    ci_train_labels[v_file] = malignant_samples[key]
    io.imsave(augmentation_folder + v_file, Iv)
    
    # Flip Horizontal
    Ih = flip(I,1)
    h_file = "horizontal_" + key
    ci_train_labels[h_file] = malignant_samples[key]
    io.imsave(augmentation_folder + h_file, Ih)
    
    # Rotate
    Ir = skimage.transform.rotate(I, angle=np.random.uniform(-45,45))
    Ir_file = "ci_rotated_" + key
    ci_train_labels[Ir_file] = malignant_samples[key]
    io.imsave(augmentation_folder + Ir_file, Ir)
    
    ie = Image.open(imagedir+key)
    #Enhance Color
    color = ImageEnhance.Color(ie)
    Icolor_low = color.enhance(0.9)
    c1_file = "color_enh1_" + key
    ci_train_labels[c1_file] = malignant_samples[key]
    Icolor_low.save(augmentation_folder + c1_file)
    
    Icolor_high = color.enhance(1.1)
    c2_file = "color_enh2_" + key
    ci_train_labels[c2_file] = malignant_samples[key]
    Icolor_high.save(augmentation_folder + c2_file)
    
    #Enhance Sharpness
    sharpness = ImageEnhance.Sharpness(ie)
    Isharp = sharpness.enhance(1.2)
    sharp_file = "sharp_" + key
    ci_train_labels[sharp_file] = malignant_samples[key]
    Isharp.save(augmentation_folder + sharp_file)

ci_aug_new_train_labels = {**aug_train_labels, **ci_train_labels}
print("augmented labels length: ", len(aug_train_labels.keys()))
print("class labels processed length: ", len(ci_train_labels.keys()))
print("augmented and class labels processed length: ", len(ci_aug_new_train_labels.keys()))

m = len([v for v in ci_aug_new_train_labels.values() if v == '1'])
b = len([v for v in ci_aug_new_train_labels.values() if v == '0'])
print("malignant count in class imbalance and augmented: ", m)
print("benign count in class imbalance and augmented: ", b)

ci_aug_new_train_labels_file = 'isic2016/Part3_TrainAug_CI_Labels.csv'
f = open(ci_aug_new_train_labels_file, "w")
for keys in ci_aug_new_train_labels.keys():
    f.write(keys + "," + ci_aug_new_train_labels[keys] + "\n")
f.close()

