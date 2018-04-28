# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 09:39:18 2018

@author: Richie
"""

import os
os.chdir('D:\Python_Tutorials\wine_quality_classifier')

## Import the necessary modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from multilable_classifier import multilabel_train_test_split

## Load the dataset
data = pd.DataFrame.from_csv("wine.csv")

## create a dictionary to map

mymap = {"red": 1, "white":2}

## user defined function with lambda

data = data.applymap(lambda x: mymap.get(x) if x in mymap else x)



cols = ('volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides',
       'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH',
       'sulphates', 'alcohol', 'style')

## Get the feature and target variables
X,y = data.loc[:,cols], data.loc[:,("quality")]

# Get labels and convert to dummy variables: label_dummies
label_dummies = pd.get_dummies(y)


# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


clf = LogisticRegression(class='multinomial')

# Fit the classifier to the training data
clf.fit(X_train,y_train)

# Print the accuracy
print("Accuracy: {}".format(clf.score(X_test, y_test)))
























