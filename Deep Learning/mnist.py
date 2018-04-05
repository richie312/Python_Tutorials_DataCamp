# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 11:12:01 2018

@author: Richie
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD

## Preporcess the data and create the predictor and target variables

# Create the model: model
model = Sequential()

# Add the first hidden layer
model.add(Dense(50, activation = 'relu', input_shape=(784,)))

# Add the second hidden layer
model.add(Dense(50, activation = 'relu'))

# Add the output layer
model.add(Dense(10, activation = 'softmax'))

# Compile the model
model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(X,y, validation_split=0.3)
