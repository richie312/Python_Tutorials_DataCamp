# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata




mnist = fetch_mldata('MNIST original')
mnist
X, y = mnist["data"], mnist["target"]
some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
interpolation="nearest")
plt.axis("off")

### Split the dataset into test and train for each class image and label respectively,

X_train,X_test,y_train,y_test=X[:60000],X[60000:],y[:60000],y[60000:]

## Lets shuffle the data for cross validation

shuffle_index=np.random.permutation(60000)
X_train,y_train=X_train[shuffle_index],y_train[shuffle_index]


## Training a Binary Classifier

y_train_5=(y_train == 5)

## Train the model using Stochastic Gradient Descent Classifier

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)

## Lets predict the true case for digit 5
sgd_clf.fit(X_train, y_train_5)


## Predict the case for any random digit
sgd_clf.predict([some_digit])
random = X[22000]
sgd_clf.predict([random])

## Implementing Cross Validation

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
skfolds = StratifiedKFold(n_splits=3, random_state=42)
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_5[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_5[test_index])
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred)) # prints 0.9502, 0.96565 and 0.96495
    
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
    
    