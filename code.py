# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 11:46:44 2021

@author: Dell
"""

### Importing Libraries
import tensorflow as tf
import os
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


#### Dataset Load
############  Training Dataset   ############
Training_Path = os.path.join('C:/Users/USER/Desktop/project_new_dataset/flowerss/training') 

############  Validation Dataset   ############
Testing_Path = os.path.join('C:/Users/USER/Desktop/project_new_dataset/flowerss/testing') 

#### Data Pre-Processing
### Resize the image
# Height and Width of Image
Image_Size = 256 

#Batch Size
BATCH_SIZE = 32  ##### To pass the images in 32 slots

# Data Augmentation
Training_data = ImageDataGenerator(
                rescale = 1./255, #normalizing the input image convert 0 and 1
                rotation_range = 30, # randomly rotate images in the range (degrees, 0 to 180)
                vertical_flip=True) # randomly flip images

#
Testing_data = ImageDataGenerator(
                rescale = 1./255) # Normalizing the image

Train_set = Training_data.flow_from_directory(
                Training_Path,
                target_size=(Image_Size,Image_Size),
                batch_size=BATCH_SIZE,
                class_mode = 'categorical')


Test_set = Testing_data.flow_from_directory(
                Testing_Path,
                target_size = (Image_Size,Image_Size),
                batch_size = BATCH_SIZE,
                class_mode = 'categorical')

labels_values,no_of_images = np.unique(Train_set.classes,return_counts = True)
dict(zip(Train_set.class_indices,no_of_images))
labels = Test_set.class_indices
labels = { v:k for k,v in labels.items() } # Flipping keys and values
values_lbl = list(labels.values()) # Taking out only values from dictionary


# Defining all layers.
dense_layer = tf.keras.layers.Dense ## Define Dense layer
convolution = tf.keras.layers.Conv2D  ## Define convolutinal layer
max_pooling= tf.keras.layers.MaxPooling2D ## Define max_pooling layer
flattening = tf.keras.layers.Flatten() ## Define flattening layer
dropout = tf.keras.layers.Dropout(0.2)  ## Define dropout layer

# Sequential Model
model = tf.keras.Sequential()
# 1st layer
model.add(convolution(16,(3,3),input_shape = (256,256,3),padding='same',activation='relu'))
model.add(max_pooling(2,2))

# 2nd layer
model.add(convolution(16,(3,3),padding='same',activation='relu'))
model.add(max_pooling(2,2))

# 3rd layer
model.add(convolution(32,(3,3),padding='same',activation='relu'))
model.add(max_pooling(2,2))

# 4th layer
model.add(convolution(32,(3,3),padding='same',activation='relu'))
model.add(max_pooling(2,2))

# 5th layer
model.add(convolution(32,(3,3),padding='same',activation='relu'))
model.add(max_pooling(2,2))

# Flatten Layer
model.add(flattening)

model.add(dense_layer(512,activation='relu',))
model.add(dropout)
model.add(dense_layer(256,activation='relu'))

# Output Layer
model.add(dense_layer(5,activation='softmax'))

# Summary
model.summary()

# Compiling model
model.compile(loss='categorical_crossentropy',optimizer = tf.keras.optimizers.RMSprop(lr = 0.001),metrics=['acc'] )

### Fitting the Model
cnn_network = model.fit_generator(Train_set,
                                    epochs=2,
                                    validation_data=Test_set,
                                    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience = 105)])

#Save our model
model.save("first_ones.h5")

## Accuracy comparison
plt.plot(cnn_network.history['loss'])
plt.plot(cnn_network.history['val_loss'])
plt.legend(['loss','val_loss'])
plt.show()

plt.plot(cnn_network.history['acc'])
plt.plot(cnn_network.history['val_acc'])
plt.legend(['acc','val_acc'])
plt.show()

