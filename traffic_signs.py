# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from os import listdir
from os.path import isfile, join
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score
import pandas as pd
import scipy
import skimage
from skimage import data
import tensorflow as tf 
    
 def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH = "D:\Belgium_Traffic"
train_data_directory = os.path.join(ROOT_PATH, "Training")
test_data_directory = os.path.join(ROOT_PATH, "Testing")

images, labels = load_data(train_data_directory)
images_test,labels_test=load_data(test_data_directory)

## Plot the images using skimage
images[10]

imshow(images[10])


## convert the images to numpy array in order to inspect the dimension of the images

images= np.array(images)
print(images.ndim)
print(labels.ndim)

## Print the image size
print(images.size)
## Count the number of elements in an array

len(images)
len(set(labels))

## Make Histogram using matplotlib

plt.hist(labels,62)

### Determine the (random) indexes of the images that you want to see 

traffic_signs = [300, 2250, 3650, 4000]

# Fill out the subplots with the random images that you defined 
for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images[traffic_signs[i]])
    plt.subplots_adjust(wspace=1)
plt.show()

## Resizing the images

for i in range(len(traffic_signs)):
    plt.subplot(1,4,i+1)
    plt.axis('off')
    plt.imshow(images[traffic_signs[i]])
    plt.subplots_adjust(wspace=1)
    plt.show()
    print("shape{0}, min{1}, max{2}".format(images[traffic_signs[i]].shape,
                                            images[traffic_signs[i]].min(),
                                            images[traffic_signs[i]].max()))
    
## Get the unique labels
    
    unique_labels=set(labels)
    
    ## Initialise the figure
    plt.figure(figsize=(15,15))
    
    ## Set a counter
    i=1
    
for label in unique_labels:
    image=images[labels.index(label)]
    plt.subplot(8,8,i)
    plt.axis("off")
    plt.title("Label {0} ({1})".format(label, labels.count(label)))
    i+=1
    plt.imshow(image)
plt.show()
    
# Resize the images in the `images` array
images28 = [skimage.transform.resize(image, (28, 28)) for image in images]

## Examine  the images

imshow(images28[1])

## Image conversion to grayscale
images28=np.array(images28)

images28=skimage.color.rgb2gray(images28)

imshow(images28[230])

## Check the result

traffic_signs = [300, 2250, 3650, 4000]

for i in range(len(traffic_signs)):
    plt.subplot(1,4,i+1)
    plt.axis('off')
    plt.imshow(images28[traffic_signs[i]],cmap='gray')
    plt.subplots_adjust(wspace=0.5)
plt.show()

##### Modelling the Neural Network

#############   TENSORFLOW      #############################

# Import `tensorflow` 
import tensorflow as tf 

# Initialize placeholders 
x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28])
y = tf.placeholder(dtype = tf.int32, shape = [None])

# Flatten the input data
images_flat = tf.contrib.layers.flatten(x)

# Fully connected layer 
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

# Define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, 
                                                                    logits = logits))
# Define an optimizer 
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#################  Run THe Model ####################

tf.set_random_seed(1234)
sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(201):
        print('EPOCH', i)
        _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images28, y: labels})
        if i % 10 == 0:
            print("Loss: ", loss)
        print('DONE WITH EPOCH')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(201):
        _, loss_value = sess.run([train_op, loss], feed_dict={x: images28, y: labels})
        if i % 10 == 0:
            print("Loss: ", loss)
            
 ######################   Evaluating Your Neural Network  #######################

# Import Random
import random
import matplotlib.pyplot as plt


# Pick 10 random images
sample_indexes = random.sample(range(len(images28)), 10)
sample_images = [images28[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

# Run the "correct_pred" operation
sess=tf.Session()
sess.run(tf.global_variables_initializer())
 predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]
 print(predicted)


    
                        
# Print the real and predicted labels
print(sample_labels)

# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction), 
             fontsize=12, color=color)
    plt.imshow(sample_images[i],  cmap="gray")

plt.show()       
        
##################  Prediction on test Data  #########################

# Load the test data
test_images, test_labels = load_data(test_data_directory)

# Transform the images to 28 by 28 pixels
test_images28 = [skimage.transform.resize(image, (28, 28)) for image in test_images]

# Convert to grayscale
test_images28 = skimage.color.rgb2gray(np.array(test_images28))

# Run predictions against the full test set.
predicted = sess.run([correct_pred], feed_dict={x: test_images28})[0]

# Calculate correct matches 
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])

# Calculate the accuracy
accuracy = match_count / len(test_labels)

# Print the accuracy
print("Accuracy: {:.3f}".format(accuracy))























