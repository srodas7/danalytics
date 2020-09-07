#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 18:23:51 2018

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
import math
import cv2
import numpy as np


imagedir = 'isic2016/train_augmented_ci_rot/'
files = listdir(imagedir)

newdir = 'isic2016/train_grey/'

for file in files:
    img = Image.open(imagedir+file).convert('LA')
    img.save(newdir+file)
    