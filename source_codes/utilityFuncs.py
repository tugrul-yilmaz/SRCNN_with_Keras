# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 17:31:30 2020

@author: Casper
"""
import pandas as pd
import numpy as np
import math
import cv2
import glob
import os

#from skimage.measure import compare_ssim as ssim
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam
 

# %%
def model():
    
    # define model type
    SRCNN = Sequential()
    
    # add model layers
    SRCNN.add(Conv2D(filters = 64, kernel_size = (9, 9), kernel_initializer = 'random_uniform',
                     activation = 'relu', padding = 'same', use_bias = True, input_shape = (64, 64, 1)))
    SRCNN.add(Conv2D(filters = 32, kernel_size = (3, 3), kernel_initializer = 'random_uniform',
                     activation = 'relu', padding = 'same', use_bias = True))
    SRCNN.add(Conv2D(filters = 1, kernel_size = (5, 5), kernel_initializer = 'random_uniform',
                     activation = 'linear', padding = 'same', use_bias = True))
    
    # define optimizer
    optimizer = Adam(lr = 0.0003)
    
    # compile model
    SRCNN.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])
    print(model.summary())
    return SRCNN

# %%

def savingLR(data_path, save_path, factor):
    data_path = glob.glob(data_path)
    for i in range(len(data_path)):
        fn = data_path[i].split('\\')[-1]
        img = cv2.imread(data_path[i], cv2.IMREAD_UNCHANGED)
        width = img.shape[1]
        height = img.shape[0]
        newWidth = int(width / factor)
        newHeight = int(height / factor)
        img = cv2.resize(img, (newWidth, newHeight), cv2.INTER_CUBIC)
        LRImg = cv2.resize(img, (width, height), cv2.INTER_CUBIC)
        cv2.imwrite(save_path + "\{}".format(str(fn)), LRImg)
        print("{}. dosya kaydediliyor.".format(str(i)))
    print("İşlem bitti.")


# %% 
def save_images(data_path,save_path,size1,size2):
    data_path=glob.glob(data_path)
    
    for i in range(len(data_path)): 
        fn=data_path[i].split("\\")[-1]
        img=cv2.imread(data_path[i],cv2.IMREAD_UNCHANGED)
        n_img=cv2.resize(img,(size1,size2),cv2.INTER_AREA)
        cv2.imwrite(save_path+"\{}.png".format(str(fn)),n_img)
        print("{}. dosya kaydediliyor.".format(str(i)))
    print("İşlem bitti.")     
    

# %%
def convertTestLR(data_path, factor, size):
    img = cv2.imread(data_path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (size, size), cv2.INTER_AREA)
    orijinalResim = img
    width = img.shape[1]
    height = img.shape[0]
    newWidth = int(width / factor)
    newHeight = int(height / factor)
    img = cv2.resize(img, (newWidth, newHeight), cv2.INTER_CUBIC)
    LRImg = cv2.resize(img, (width, height), cv2.INTER_CUBIC)
    bozukResim = LRImg
    LRImg_YCrCb = cv2.cvtColor(LRImg, cv2.COLOR_RGB2YCrCb)
    y = LRImg_YCrCb[:, :, 0]
    cb = LRImg_YCrCb[:, :, 1]
    cr = LRImg_YCrCb[:, :, 2]
    y = y.astype(float) / 255.
    return y, cb, cr, orijinalResim, bozukResim

# %%
def PSNR(changedImg, originalImg):   
    changedImg = changedImg.astype(float)
    originalImg = originalImg.astype(float)

    diff = originalImg - changedImg
    diff = diff.flatten('C')

    RMSE = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(255. / RMSE)


# mean squared error (MSE)
def MSE(changedImg, originalImg):
    err = np.sum((changedImg.astype('float') - originalImg.astype('float')) ** 2)
    err /= float(changedImg.shape[0] * changedImg.shape[1])
    return err


def prepareData(data_path, label_path):
  data_path = glob.glob(data_path)
  label_path = glob.glob(label_path)
  for i in range(len(data_path)):
    img = cv2.imread(data_path[i], cv2.IMREAD_UNCHANGED)
    label = cv2.imread(label_path[i], cv2.IMREAD_UNCHANGED)
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    label_YCrCb = cv2.cvtColor(label, cv2.COLOR_RGB2YCrCb)
    img_YCrCb = (img_YCrCb).astype(float) / 255.
    label_YCrCb = (label_YCrCb).astype(float) / 255.
    img_y = img_YCrCb[:, :, 0]
    label_y = label_YCrCb[:, :, 0]
  return img_y, label_y

# %%
def crop2(data_path, save_path, crop_size, totalIterNumber):
    data_path = glob.glob(data_path)
    os.makedirs(save_path)
    for i in range(len(data_path)):
        fn = data_path[i].split('\\')[-1].split('.')[0]
        #print(fn)
        img = cv2.imread(data_path[i], cv2.IMREAD_UNCHANGED)
        print("{}. resim kırpılıyor.".format(str(i)))
        for j in range(totalIterNumber):
            for x in range(totalIterNumber):
                crop_img =  img[j*crop_size:(j+1)*crop_size, x*crop_size:(x+1)*crop_size].copy()
                print("{}_{}_{}.png kaydediliyor...".format(str(fn), str(j), str(x)))
                cv2.imwrite(save_path + "{}_{}_{}.png".format(str(fn), str(j), str(x)), crop_img)
        
    print("İşlem bitti!")
    
def crop_saving(data_path, save_path):
    data_path = glob.glob(data_path)
    print(data_path)
    #os.makedirs(save_path)
    for i in range(len(data_path)):
        fn=data_path[i].split("\\")[-1].split('.')[0]
        img = cv2.imread(data_path[i], cv2.IMREAD_UNCHANGED)
        
        print("{}. resim kırpılıyor.".format(str(i)))
        counter=1
        for j in range(8):
            for k in range(8):
                crop_img=img[j*64:(j+1)*64,k*64:(k+1)*64].copy()
                cv2.imwrite(save_path + "\{}_{}.png".format(str(fn), str(counter)), crop_img)
                counter=counter+1
    
def crop(data_path,save_path,crop_size):
    data_path=glob.glob(data_path)
    
        