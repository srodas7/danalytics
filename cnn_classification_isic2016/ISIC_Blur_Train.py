#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 13:22:51 2018

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
from PIL import ImageFilter

#Blur Train Aug CI
imagedir = 'isic2016/train_augmented_ci/'
files = listdir(imagedir)
new_folder = 'isic2016/train_blurred/'

for i in range(0,len(files)):
    if (i%50 == 0):
        print("Processed " + str(i) + " images")
    Ie = Image.open(imagedir + files[i])
    Ib = Ie.filter(ImageFilter.BLUR)
    Ib.save(new_folder + files[i])

#Blur Test
imagedir = 'isic2016/test/'
files = listdir(imagedir)
new_folder = 'isic2016/test_blurred/'

for i in range(0,len(files)):
    if (i%50 == 0):
        print("Processed " + str(i) + " images")
    Ie = Image.open(imagedir + files[i])
    Ib = Ie.filter(ImageFilter.BLUR)
    Ib.save(new_folder + files[i])
    
