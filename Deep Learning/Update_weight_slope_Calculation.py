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

gradient = 2* input_data* error
weights_updated = weights - learning_rate*gradient
preds_updated = (weights_updated*input_data).sum()
error_updated = preds_updated - target
print(error_updated)



### Calculate the slope
weights = np.array([0,2,1])
target = 0
input_data= np.array([1,2,3])



# Calculate the predictions: preds

preds = (weights*input_data).sum()

# Calculate the error: error
error = preds - target

# Calculate the slope: slope
slope = 2 * input_data * error

# Print the slope
print(slope)

# Set the learning rate: learning_rate
learning_rate = 0.01

# Calculate the predictions: preds
preds = (weights * input_data).sum()

# Calculate the error: error
error = preds - target

# Calculate the slope: slope
slope = 2 * input_data * error

# Update the weights: weights_updated
weights_updated = weights - learning_rate*slope

# Get updated predictions: preds_updated
preds_updated = (weights_updated*input_data).sum()

# Calculate updated error: error_updated
error_updated = preds_updated - target


# Print the original error
print(error)

# Print the updated error
print(error_updated)

### Calcuate the optimal interation required to get the optimal weights

n_updates = 20
mse_hist = []


## User Defined function

def get_slope(input_data,target,weights):
    preds = (weights * input_data).sum()
    error = preds - target
    slope = 2 * input_data * error
    return(slope)
   
    
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def get_mse(input_data,target,weights):
    preds = (weights * input_data).sum()
    mse = mean_squared_error(target,preds)
    return(mse)
    
    


# Iterate over the number of updates
for i in range(n_updates):
    # Calculate the slope: slope
    slope = get_slope(input_data,target,weights)
    
    # Update the weights: weights
    weights = weights - learning_rate * slope
    
    # Calculate mse with new weights: mse
    mse = get_mse(input_data, target, weights)
    
    # Append the mse to mse_hist
    mse_hist.append(mse)

# Plot the mse history
plt.plot(mse_hist)
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.show()

