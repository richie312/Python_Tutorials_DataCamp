# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

####  Codes to update the weight and calulate the slopes

weights = np.array([1,2])
input_data = np.array([3,4])
target = 6
learning_rate = 0.01
preds = (weights*input_data).sum()
error = preds - target
print(error)

## Slope Calculation



