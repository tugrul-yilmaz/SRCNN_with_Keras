# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 19:23:23 2020

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

import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint
import h5py
import utilityFuncs


# %%
new_model = tf.keras.models.load_model('nts-nnweights62.hdf5')
new_model.summary()
# %%
size=512

trial=cv2.imread("0801lr.png.png")
#trial=cv2.resize(trial,(size,size),cv2.INTER_AREA)
trial=cv2.cvtColor(trial,cv2.COLOR_RGB2YCrCb)
cb=trial[:,:,1]
cr=trial[:,:,2]
trial=trial/255.
trialy=trial[:,:,0]

trialy=trialy.reshape(-1,size,size,1)

pred=new_model.predict(trialy)
pred=pred*255
pred[pred[:] > 255] = 255
pred[pred[:] < 0] = 0
pred= pred.astype(np.uint8)
pred =pred[0,:,:, 0]
print(pred.shape)
print(cb.shape)
print(cr.shape)

img = cv2.merge([pred, cb, cr])

img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
cv2.imwrite("SRtahmin.png", img)

# %%
orijinal = cv2.imread("0801.png.png".format(size, size), cv2.IMREAD_COLOR)
SRCNN = cv2.imread("SRtahmin.png", cv2.IMREAD_COLOR)
bicubic = cv2.imread("0801lr.png.png", cv2.IMREAD_COLOR)
psnrCubic = utilityFuncs.PSNR(bicubic, orijinal)
psnrSR = utilityFuncs.PSNR(SRCNN, orijinal)
#ssimCubic = ssim(bicubic, orijinal, multichannel = True)
#ssimSR = ssim(SRCNN, orijinal, multichannel = True)
mseCubic = utilityFuncs.MSE(bicubic, orijinal)
mseSR = utilityFuncs.MSE(SRCNN, orijinal)

#03 --25.936
#16   25.97
#42   26.0319
#59   26.0341
#60   26.0423

print("Bicubic_PSNR: {} \tSRCNN_PSNR: {}".format(psnrCubic, psnrSR))
#print("Bicubic_SSIM: {} \tSRCNN_SSIM: {}".format(ssimCubic, ssimSR))
print("Bicubic_MSE: {} \tSRCNN_MSE: {}".format(mseCubic, mseSR))


# %%
fig, axs = plt.subplots(1, 3, figsize=(20, 8))
axs[0].imshow(cv2.cvtColor(orijinal, cv2.COLOR_BGR2RGB))
axs[0].set_title('Orijinal')
axs[1].imshow(cv2.cvtColor(bicubic, cv2.COLOR_BGR2RGB))
axs[1].set_title('Düşük Çözünürlük')
axs[1].set_xlabel(xlabel = 'PSNR: {:.3f}\nMSE: {:.3f} '.format(psnrCubic, mseCubic))
axs[2].imshow(cv2.cvtColor(SRCNN, cv2.COLOR_BGR2RGB))
axs[2].set_title('SRCNN')
axs[2].set_xlabel(xlabel = 'PSNR: {:.3f} \nMSE: {:.3f} '.format(psnrSR, mseSR), color = "green")
fig.savefig('demo5.png')

"""   # remove the x and y ticks
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

fig.savefig('.\\demo\\demo{}.png'.format(i))
plt.close()
"""