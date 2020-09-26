# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 18:41:12 2020

@author: Casper
"""
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


import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint
import h5py
import utilityFuncs

# %%cropping


# %%
data_path=r"C:\Users\Casper\Desktop\sss\test\*.png"
save_path=r"C:\Users\Casper\Desktop\sss\test_512"
size1=512
size2=512
utilityFuncs.save_images(data_path,save_path,size1,size2)

# %%

data_path=r"C:\Users\Casper\Desktop\sss\test_org_512\*.png"
save_path=r"C:\Users\Casper\Desktop\sss\test_lr_512"
factor=2
utilityFuncs.savingLR(data_path,save_path,factor)


# %%
data_path=r"C:\Users\Casper\Desktop\sss\lr_512\*.png"
save_path=r"C:\Users\Casper\Desktop\sss\train_data"
utilityFuncs.crop_saving(data_path,save_path)

# %%
data_path=r"C:\Users\Casper\Desktop\sss\org_512\*.png"
save_path=r"C:\Users\Casper\Desktop\sss\label_data"
utilityFuncs.crop_saving(data_path,save_path)

