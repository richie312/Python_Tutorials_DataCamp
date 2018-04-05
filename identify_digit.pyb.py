# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 16:03:22 2018

@author: Administrator
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import imshow
from skimage import data
import skimage
import numpy as np
import os


def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".png")]
        for f in file_names:
            images.append(skimage.data.imread(f))
    return images

ROOT_PATH = "E:\Data Science\Python\Identify_Digits\Train\Train"
train_data_directory = os.path.join(ROOT_PATH, "Images")
test_data_directory = os.path.join(ROOT_PATH, "Images") 

images = load_data(train_data_directory)
images_test=load_data(test_data_directory)

## Check the image and the array values


images[10]

imshow(images[10])

images[10]

