# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 08:24:41 2018

@author: Administrator
"""

import tensorflow as tf


a= tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)

adder_node=a+b
    
W=tf.Variable([.3],tf.float32)
b=tf.Variable([-.3],tf.float32)
x=tf.placeholder(tf.float32)

linear_model=W*x+b
init=tf.global_variables_initializer()

sess=tf.Session()
sess.run(init)
print(sess.run(linear_model,{x:[1,2,3,4]}))


## To Evaluate the model on training data, we nee a placeholder to provide desired values and 
##we need to write a loss functions


## Model Parameters

W=tf.Variable([.3],tf.float32)
b=tf.Variable([-.3],tf.float32)

x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)

linear_model=W*x+b

## Loss function
squared_deltas=tf.square(linear_model-y)
loss=tf.reduce_sum(squared_deltas)

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))


#########  REDUCING THE LOSS ###################

## Optimiser modifies the each variable according to magnitude of the derivative of loss wrt variable,
## Here we will use gradient descent optimiser.

## Optimize

optimizer=tf.train.GradientDescentOptimizer(0.01)
train=optimizer.minimize(loss)
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})
##For Visualising the tensorflow graphs, tensorflow boards are used.
File_writer=tf.summary.FileWriter("C:\TensorFlow_Projects\Computation_Graph_Sample\graph",sess.graph)
print(sess.run([W,b]))
sess.close()


################ Computational Graph  ####################################

##For Visualising the tensorflow graphs, tensorflow boards are used.

##The first argument when creating the filewriter is an outout directory name, which will be created if 
##it does not exist.

File_writer=tf.summary.FileWriter("C:\TensorFlow_Projects\Computation_Graph_Sample\graph",sess.graph)

