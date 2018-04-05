# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
df=np.array([1,2,3,4])

import tensorflow as tf
import numpy as np



## Forward Propagation
input_data = np.array([2,3]),
weights = {'node_01': np.array([1,1]),
           'node_02': np.array([-1,1]),
           'output_node': np.array([2,-1])}

nodes_01_value = (input_data*weights['node_01']).sum()
nodes_02_value = (input_data*weights['node_02']).sum()

hidden_layer_values=np.array([nodes_01_value,nodes_02_value])
print(hidden_layer_values)

## Output

Output = (hidden_layer_values*weights['output_node']).sum()
print(Output)

######   Activation Function ###############

input_data = np.array([2,3]),
weights = {'node_01': np.array([1,1]),
           'node_02': np.array([-1,1]),
           'output_node': np.array([2,-1])}

nodes_01_value = np.tanh((input_data*weights['node_01']).sum())
nodes_02_value = np.tanh((input_data*weights['node_02']).sum())

hidden_layer_values=np.array([nodes_01_value,nodes_02_value])
print(hidden_layer_values)

## Output

Output = np.tanh((hidden_layer_values*weights['output_node']).sum())
print(Output)

#### User defined Function


### UDF for relu

def relu(input):
    d= max(input,0)
    return d



# Define predict_with_network()
def predict_with_network(input_data_row, weights):

    # Calculate node 0 value
    node_0_input = (input_data_row*weights['node_0']).sum()
    node_0_output = relu(node_0_input)

    # Calculate node 1 value
    node_1_input = (input_data_row*weights['node_1']).sum()
    node_1_output = relu(node_1_input)
    

    # Put node values into array: hidden_layer_outputs
    hidden_layer_outputs = np.array([node_0_output, node_1_output])
    
    # Calculate model output
    input_to_final_layer = (hidden_layer_outputs*weights['output']).sum()
    model_output = relu(input_to_final_layer)
    
    # Return model output
    return(model_output)


# Create empty list to store prediction results
results = []
for input_data_row in input_data:
    # Append prediction to results
    results.append(predict_with_network(input_data_row,weights))
    
# Print results
print(results)


#######   Need for the optimisation    ###################


## Optimising the model with one weight

# The data point you will make a prediction for
input_data = np.array([0, 3])

# Sample weights
weights_0 = {'node_0': [2, 1],
             'node_1': [1, 2],
             'output': [1, 1]
            }

# The actual target value, used to calculate the error
target_actual = 3

# Make prediction using original weights
model_output_0 = predict_with_network(input_data, weights_0)

# Calculate error: error_0
error_0 = model_output_0 - target_actual

# Create weights that cause the network to make perfect prediction (3): weights_1
weights_1 = {'node_0': [2,1],
             'node_1': [1, 0],
             'output': [1, 1]
            }

# Make prediction using new weights: model_output_1
model_output_1 = predict_with_network(input_data, weights_1)

# Calculate error: error_1
error_1 = model_output_1 - target_actual

# Print error_0 and error_1
print(error_0)
print(error_1)

#### Using Sklearn to get the Mean Square Error #####################

target_actuals = np.array([1,3,5,7])

from sklearn.metrics import mean_squared_error

# Create model_output_0 
model_output_0 = []
# Create model_output_0
model_output_1 = []

# Loop over input_data
for row in input_data:
    # Append prediction to model_output_0
    model_output_0.append(predict_with_network(row,weights_0))
    
    # Append prediction to model_output_1
    model_output_1.append(predict_with_network(row,weights_1))
    

# Calculate the mean squared error for model_output_0: mse_0
mse_0 = mean_squared_error(target_actuals,model_output_0)


# Calculate the mean squared error for model_output_1: mse_1
mse_1 = mean_squared_error(target_actuals, model_output_1)

# Print mse_0 and mse_1
print("Mean squared error with weights_0: %f" %mse_0)
print("Mean squared error with weights_1: %f" %mse_1)        


####  Codes to update the weight and calulate the slopes

weights = np.array([1,2])
input_data = np.array([3,4])
target = 6
learning_rate = 0.01
preds = (weights*input_data).sum()
error = preds - target
print(error)

## Slope Calculation























