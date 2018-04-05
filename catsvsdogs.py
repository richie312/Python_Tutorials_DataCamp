# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 09:10:19 2018

@author: Richie
"""
from os import listdir
from os.path import isfile, join
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import os
import tensorflow as tf 

## Create the user defined function for labels

def create_label(image_name):
    """ Create an one-hot encoded vector from image name """
    word_label = image_name.split('.')[-3]
    if word_label == 'cat':
        return np.array([1,0])
    elif word_label == 'dog':
        return np.array([0,1])



## Create the user function for loading the data

def load_data(data_directory):
    images = []
    labels = []
    file_names = [os.path.join(data_directory, f) 
                     for f in os.listdir(data_directory) 
                     if f.endswith(".jpg")]
    for f in file_names:
        images.append(skimage.color.rgb2gray(skimage.data.imread(f)))
        labels.append(create_label(f))
    return images , labels
    
ROOT_PATH = "D:\cats_dogs"
train_data_directory = os.path.join(ROOT_PATH, "train_ver")
test_data_directory = os.path.join(ROOT_PATH, "tes")

images, labels = load_data(train_data_directory)
images_test =load_data(test_data_directory)

## Plot the image

create_label(images[133])


train_data[1]

word_label = train_data[1].split('.')[-3]
plt.show(imshow(images[1]))

training_data.append([np.array(images),create_label(images)])
    
    return images
    return(training_data)
labels
import cv2