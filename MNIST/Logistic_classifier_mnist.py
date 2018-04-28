# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 07:24:50 2018

@author: Richie
"""
## Change the working directory
import os
os.chdir('D:\Python_Tutorials\MNIST')

## Import the necessary modules

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from multilable_classifier import multilabel_train_test_split

## Load the datset and examine

mnist_data = fetch_mldata('MNIST original')
X, y = mnist_data["data"], mnist_data["target"]
some_digit = X_train[10]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
interpolation="nearest")
plt.axis("off")


## First split the data into sample and hold out set

X_sample,y_sample=X[:15000],y[:15000]


# Get labels and convert to dummy variables: label_dummies
label_dummies = pd.get_dummies(y_sample)



# Calculate number of unique values for each label: num_unique_labels
num_unique_labels = pd_Data.Frame(y_sample).apply(pd.Series.nunique,axis =0)

# Plot number of unique values for each label
num_unique_labels.plot(kind = 'bar')

# Label the axes
plt.xlabel('Labels')
plt.ylabel('Number of unique valu



# Create training and test sets
X_train, X_test, y_train, y_test = multilabel_train_test_split(X_sample,
                                                               label_dummies,
                                                               size=0.2, 
                                                               seed=123)
## Reshape the data
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')


## Convert 3D to 2D array
X_train = X_train.reshape((X_train.shape[0],-1), order ='F')

# Instantiate the classifier: clf
clf = OneVsRestClassifier(LogisticRegression())

# Fit the classifier to the training data
clf.fit(X_train,y_train)

# Print the accuracy
print("Accuracy: {}".format(clf.score(X_test, y_test)))

## Predict the above model on the holdout data set(which is usually the test set)

# Instantiate the classifier: clf
clf = LogisticRegression(multi_class='multinomial', solver='newton-cg')

# Fit it to the training data
clf.fit(X_train, y_train)

# Load the holdout data: holdout
holdout = pd.read_csv("HoldoutData.csv", index_col=0)

# Generate predictions: predictions
predictions = clf.predict_proba(holdout[NUMERIC_COLUMNS].fillna(-1000))

