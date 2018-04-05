# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 08:55:08 2018

@author: Richie
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

data=pd.read_csv('basket_ball_shot_log.csv')
predictors= data.drop(['shot_result'],axis=1).as_matrix
target=to_categorical(data.shot_result)

# Build the model
model=Sequential()
# Add the first layer with 100 nodes
model.add(Dense(100,activation = 'relu', input_shape=(n_cols,)))
# Add the Second layer with 100  nodes
model.add(Dense(100,activation = 'relu'))
# Add the third layer
model.add(Dense(100, activation = 'relu'))

# Add the output layer

model.add(Dense(2, activation = 'softmax'))
## Compile the model
model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
## Fit the model
model(predictors, target)

## Saving, Reloading and making predcitions with the model

from keras.models import load_model
model.save('model_file.h5')
my_model=load_model('my_model.h5')
predictions=my_model.predict(data_to_predict_with)
probability_true=predictions[:,1]

## Verify the model structure

my_model.summary()


