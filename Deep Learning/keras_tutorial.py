# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential


## Load the data

predictors=np.loadtxt('predictors_data.csv',delimiter=',')
n_cols=predictors.shape[1]

# Build the model
model=Sequential()
# Add the first layer with 100 nodes
model.add(Dense(100,activation = 'relu', input_shape=(n_cols,)))
# Add the Second layer with 100  nodes
model.add(Dense(100,activation = 'relu'))
# Add the output layer
model.add(Dense(1))

# Compiling a model to get the optimum weights in order to get the mini
#mum loss for which prediction is better

model.compile(optimizer='adam',loss='mean_squared_error')

## Fitting the model

model.fit(predictors,target)


