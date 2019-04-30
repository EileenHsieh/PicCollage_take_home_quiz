#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:08:29 2019

@author: eileen
"""


from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import joblib
from skimage import io, transform, color


#%%
DATAROOT = '/home/eileen/piccollage/train_imgs'
LABPATH = '/home/eileen/piccollage/train_responses.csv'
TOTAL_NUM = 150000

TRAIN_NUM = int(TOTAL_NUM*0.6)
VAL_NUM = int(TOTAL_NUM*0.01)
TEST_NUM = TOTAL_NUM-(TRAIN_NUM+VAL_NUM)

print('Train:{} | Val:{} | Test:{}'.format(TRAIN_NUM, VAL_NUM, TEST_NUM))

RANDSEED = 0
np.random.seed(RANDSEED)

#%% load label
df = pd.read_csv(LABPATH)
df.index = df['id']
df = df['corr']


#%% generate
imgPaths = glob(DATAROOT+'/*.png')
sampleIdxs = np.random.choice(len(imgPaths), TRAIN_NUM+VAL_NUM, replace=False)
testIdxs = [idx for idx in range(TOTAL_NUM) if idx not in sampleIdxs]
trainPaths = [imgPaths[idx] for idx in sampleIdxs[:TRAIN_NUM]]
valPaths = [imgPaths[idx] for idx in sampleIdxs[TRAIN_NUM:]]
testPaths = [imgPaths[idx] for idx in testIdxs]


#%% test functions: crop, grayscale, resize
RESIZE_SHAPE = 64
img = io.imread(trainPaths[0])
img = img[2:129,21:148,:]       # crop
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
plt.title('original')
plt.imshow(img)

img_gray = color.rgb2gray(img)  # grayscale
fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('gray')
plt.imshow(img_gray)

img_resize = transform.resize(img, (RESIZE_SHAPE,RESIZE_SHAPE))
img_resize_gray = color.rgb2gray(img_resize)
img_resize_gray_exp = np.expand_dims(img_resize_gray, axis=0)
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
plt.title('resize')
plt.imshow(img_resize_gray_exp[0,:,:])


#%% check distribution
trainLabs = [df.loc[p.split('/')[-1].split('.')[0]] for p in trainPaths]
valLabs = [df.loc[p.split('/')[-1].split('.')[0]] for p in valPaths]
testLabs = [df.loc[p.split('/')[-1].split('.')[0]] for p in testPaths]

style.use('seaborn-darkgrid')
fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('data distribution')
plt.hist(testLabs, label='test')
plt.hist(trainLabs,label='train')
plt.legend()
plt.savefig('../sample_data/train{}_val{}_test{}.png'.format(TRAIN_NUM, VAL_NUM, TEST_NUM))


dataLists = {'train':list(zip(trainPaths, trainLabs)), 'val':list(zip(valPaths, valLabs)), 'test':list(zip(testPaths, testLabs))}
joblib.dump(dataLists, '../sample_data/train{}_val{}_test{}.pkl'.format(TRAIN_NUM, VAL_NUM, TEST_NUM))

