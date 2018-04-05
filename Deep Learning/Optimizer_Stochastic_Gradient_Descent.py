# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 09:28:09 2018

@author: Richie
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD

## User Defined Function

def get_new_model(input_shape=input_shape):
    # Build the model
    model=Sequential()
# Add the first layer with 100 nodes
    model.add(Dense(100,activation = 'relu', input_shape=(n_cols,)))
# Add the Second layer with 100  nodes
    model.add(Dense(100,activation = 'relu'))
# Add the third layer
    model.add(Dense(2, activation = 'softmax'))
    return(model)
    
lr_to_test = [0.000001,0.01,1]

for lr_to_test:
    model=get_new_model()
    my_optimizer=SGD(lr=lr)
    model.compile(optimizer=my_optimizer, loss = 'categorical_crossentropy')
    model.fit(predictors,target)
    
## Early Stopping
    
from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience = 2)

## Fitting the model

model.fit(predictors, target, validation_split = 0.3, epochs = 20, callbacks = [early_stopping_monitor])

## Compare the model


# Create the plot
plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()
    
    