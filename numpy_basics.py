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
        




















