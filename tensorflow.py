# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 09:15:42 2018

@author: Richie
"""

import tensorflow as tf

## Build the computational graph

node1=tf.constant(3.0, tf.float32)
node2=tf.constant(4.0)
print(node1, node2)

## Run the compoutational graph

sess=tf.Session()
print(sess.run([node1,node2]))


## Example

a = tf.constant(5)
b= tf.constant(2)
c = tf.constant(3)

d= tf.multiply(a,b)
e=tf.add(c,b)
f=tf.subtract(d,e)

sess=tf.Session()
outs= sess.run(f)


## Example 2

a= tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)

adder_node=a+b

sess=tf.Session()

print(sess.run(adder_node,{a: [1,3],b: [2,4]}))


## Example 3

W=tf.Variable([0.3],tf.float32)
b=tf.Variable([-0.3],tf.float32)
x=tf.placeholder(tf.float32)

Linear_model=W*x+b
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
print(sess.run(Linear_model,{x:[1,2,3,4]}))

##Compute the loss

y=tf.placeholder(tf.float32)
squared_deltas=tf.square(Linear_model-y)
loss=tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y: [0,-1,-2,-3]}))

## Gradient Descent Loss function

optimizer=tf.train.GradientDescentOptimizer(0.01)
train=optimizer.minimize(loss)

sess.run(init)
for i in range(1000):
    sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})
print(sess.run([W,b]))

## AND Logic Gate for True and False


T, F= 1, -1

bias=1

train_in  = [
        [T,T, bias],
        [T, F, bias],
        [F, T, bias],
        [F,F, bias],
]

train_out = [
        [T],
        [F],
        [F],
        [F],
]

w= tf.Variable(tf.random_normal([3,1]))

def step(x):
    is_greater=tf.greater(x,0)
    as_float=tf.to_float(is_greater)
    doubled=tf.multiply(as_float,2)
    return tf.subtract(doubled,1)
output=step(tf.matmul(train_in,w))
error=tf.subrtract(train_out, output)
mse=tf.reduce_mean(tf.square(error))

delta=tf.matmul(train_in,error,transpose_a=True)
train=tf.assign(w,tf.add(w,delta))

sess=tf.Session()
sess.run(tf.initialize_all_variables())

err, target= 1, 0

epoch, max_epoch=0,10
while err>target and err<max_epoch:
    epoch+=1
    err,_=sess.run([mse,train])
    print('epoch',epoch,'mse:',err)

tf.Variable.

import numpy as np

import tensorflow as tf

df=np.array([1,2,3,4])
















































































