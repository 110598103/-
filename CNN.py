# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 20:53:43 2021

@author: 張景堯
"""
from PIL import Image
import numpy as np
import tensorflow as tf
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import random
import cv2
#圖像翻轉，壓縮
img_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,rotation_range=20,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True,
                              # fill_mode='nearest'
                              )
#建dataset
images_Train= img_train.flow_from_directory('theSimpsons-train/train/',target_size=(128,128),
                      batch_size=64,)
images_vaild= img_train.flow_from_directory('theSimpsons-train/valid/',target_size=(128,128),
                      batch_size=64,)
#建CNN
from keras.models import Sequential  #用來啟動 NN
from keras.layers import Conv2D  # Convolution Operation
from keras.layers import MaxPooling2D # Pooling
from keras.layers import Flatten
from keras.layers import Dense # Fully Connected Networks
from keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
import tensorflow.keras as keras  
model = Sequential()
model.add(Conv2D(filters=16,kernel_size=5,padding='same',input_shape=(128,128,3),activation = 'relu'))
model.add(Conv2D(filters=16,kernel_size=5,padding='same',activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(filters=24,kernel_size=5,padding='same',activation = 'relu'))
model.add(Conv2D(filters=24,kernel_size=5,padding='same',activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32,kernel_size=5,padding='same',activation = 'relu'))
model.add(Conv2D(filters=32,kernel_size=5,padding='same',activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=48,kernel_size=5,padding='same',activation = 'relu'))
model.add(Conv2D(filters=48,kernel_size=5,padding='same',activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64,kernel_size=5,padding='same',activation = 'relu'))
model.add(Conv2D(filters=64,kernel_size=5,padding='same',activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=96,kernel_size=5,padding='same',activation = 'relu'))
model.add(Conv2D(filters=96,kernel_size=5,padding='same',activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units =2048, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(units =1024, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(units =256, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units =128, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(50,activation='softmax'))
model.summary()
#處存權重
checkpoint_filepath = '.\\tmp5\\checkpoint'
model_checkpoint_callback = [tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True,save_best_only=True,mode = 'max',preiod = 1)]
from keras.models import load_model
model.load_weights('model_weights_128.h5')
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
histroy=model.fit(images_Train,
          epochs=150,
          validation_data=images_vaild,
        callbacks=model_checkpoint_callback
                 )
# 儲存整個模型
model.save('model_128.hdf5')

