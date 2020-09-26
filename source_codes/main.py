# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 14:21:25 2020

@author: Casper
"""
# %%
import pandas as pd
import numpy as np 
import cv2
import glob
import os
# %%
data_path=r"C:\Users\Casper\Desktop\sss\train_data\*.png"
label_path = r"C:\Users\Casper\Desktop\sss\label_data\*.png"

data_path = glob.glob(data_path)
label_path = glob.glob(label_path)
data_arr=[]
label_arr=[]
# %%
"""

for i in range(len(data_path)):
    img=cv2.imread(data_path[i],cv2.IMREAD_UNCHANGED)
    img=cv2.resize(img,(64,64),cv2.INTER_CUBIC)
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    img_YCrCb = img_YCrCb / 255.
    img_y=img_YCrCb[:,:,0]
    data_arr.append(img_y)
# %%

a=np.array([np.array(x) for x in data_arr])

# %%
np.save("x.npy",a)
"""
# %%
for i in range(len(label_path)):
    img=cv2.imread(label_path[i],cv2.IMREAD_UNCHANGED)
    img=cv2.resize(img,(64,64),cv2.INTER_CUBIC)
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    img_YCrCb = img_YCrCb / 255.
    img_y=img_YCrCb[:,:,0]
    label_arr.append(img_y)

# %%
b=np.array([np.array(x) for x in label_arr])

# %%
np.save("y.npy",b)

# %%
x=np.load("x.npy")
y=np.load("y.npy")

# %%

x=x.reshape(-1,64,64,1)
y=y.reshape(-1,64,64,1)



# %%
import cv2
import numpy as np
import glob
import os
import math
#from skimage.measure import compare_ssim as ssim
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam
 

def model():
    
    # define model type
    SRCNN = Sequential()
    
    # add model layers
    SRCNN.add(Conv2D(filters = 128, kernel_size = (9, 9), kernel_initializer = 'random_uniform',
                     activation = 'relu', padding = 'same', use_bias = True, input_shape = (64, 64, 1)))
    SRCNN.add(Conv2D(filters = 64, kernel_size = (3, 3), kernel_initializer = 'random_uniform',
                     activation = 'relu', padding = 'same', use_bias = True))
    SRCNN.add(Conv2D(filters = 1, kernel_size = (5, 5), kernel_initializer = 'random_uniform',
                     activation = 'linear', padding = 'same', use_bias = True))
    
    # define optimizer
    optimizer = Adam(lr = 0.0001)
    
    # compile model
    SRCNN.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])
    
    return SRCNN
# %%
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint
import h5py

print(tf.test.gpu_device_name())




# %%
filepath=r"C:/Users/Casper/Desktop/sss/modelsss/nts4/nts-nnweights{epoch:02d}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', 
verbose=0, save_weights_only=False, save_best_only=False, mode='max') 
EPOCHS=100
# %%
net=model()
net.fit(x,y,batch_size=64,epochs=EPOCHS, callbacks=[checkpoint])
net.save_weights(filepath)

print(net.summary())
#model.save(filepath)


