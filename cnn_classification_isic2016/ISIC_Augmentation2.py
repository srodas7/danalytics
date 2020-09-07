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

# AUGMENT AND REDUCE IMAGES IN NEW TRAIN DIRECTORY

# Read in train labels file
#class_label_file = 'Downloads/isic2016/Part3_Train_Labels.csv'
class_label_file = 'isic2016/Part3_Train_Labels_BW.csv'
f = open(class_label_file)
lines = f.read().splitlines()
train_labels = {}
f.close()
for i in range(0,len(lines)):
    tmp = lines[i].split(",")
    filename = tmp[0]
    Class = tmp[1]
    train_labels[filename] = Class
print('number of training images ', len(train_labels.keys()))

# Get images in Train directory into a list
#imagedir = 'Downloads/isic2016/train/'
imagedir = 'isic2016/train_bw/'
files = listdir(imagedir)

# Make a new directory for augmented images - augmented AND original images will reside there
os.system('mkdir -p isic2016/train_bw_aug/')
augmentation_folder = 'isic2016/train_bw_aug/'
new_train_labels = {}
NUM_ROTATE = 2
for i in range(0,len(files)):
    if (i%100 == 0):
        print('done processing ' + str(i) + ' images')
    I = io.imread(imagedir + files[i])
    new_train_labels[files[i]] = train_labels[files[i]]
    io.imsave(augmentation_folder + files[i], I)
    
    Iv = flip(I,0)
    v_file = "vertical_" + files[i]
    new_train_labels[v_file] = train_labels[files[i]]
    io.imsave(augmentation_folder + v_file, Iv)
    
    Ih = flip(I,1)
    h_file = "horizontal_" + files[i]
    new_train_labels[h_file] = train_labels[files[i]]
    io.imsave(augmentation_folder + h_file, Ih)


#new_train_label_file = 'Downloads/isic2016/Part3_TrainAug_Labels.csv'
new_train_label_file = 'isic2016/Part3_TrainBWAug_Labels.csv'
f = open(new_train_label_file, "w")
for keys in new_train_labels.keys():
    f.write(keys + "," + new_train_labels[keys] + "\n")
f.close()



