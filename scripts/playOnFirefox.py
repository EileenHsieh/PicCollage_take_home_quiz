#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:09:52 2019

@author: eileen
"""


from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from skimage import io
import matplotlib.pyplot as plt
from resnet import PreTrainedModel
from skimage.transform import resize
from DataTool import ImgDataset, Rgb2gray, Binarize, Binarize01, Crop, CountParam
import torch
import joblib
import numpy as np
import time
from torchvision import transforms



#%% parameters

game_home = 'http://guessthecorrelation.com/'

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#%% load model
#model = joblib.load('./models/resnet_1conv-tr30000-te118500-ep40-in64-bz128-lr0.0005-k164_k264_k332.pkl')
model = joblib.load('../models/resnet_1conv-tr60000-te88500-ep20-in64-bz256-lr0.05-k164_k264_k332.pkl')


#%% automatically play games
# connect
browser = webdriver.Firefox(executable_path="./geckodriver") # Use Firefox
browser.set_window_size(600, 600)
browser.get(game_home)
print(browser.current_url)

# click to play

#%%
for i in range(100):
    #==============================================================================
    # get the screenshot
    #==============================================================================
    browser.save_screenshot('./screenshot.png') 
    
    # cut out the interested area
    screenshot = io.imread('./screenshot.png') # (623, 1280, 4)
    screenshot = screenshot[:,:,:3]

    img = screenshot[100:450, 140:510,:]
#    img = screenshot[130:450,570:900,:]
    img[img>200] = 255
    img[img<200] = 0
    plt.imshow(img)

    
    #==============================================================================
    # fill in the form
    #==============================================================================
    img_resize = resize(img, (64,64), anti_aliasing=True)

#    p2 = Rgb2gray()
    p3 = transforms.ToTensor()
#    p4 = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_preprocessed = p3((img_resize))
    img_preprocessed[img_preprocessed<0.9]=0
    img_preprocessed[img_preprocessed>=0.9]=1    

    img_input = torch.from_numpy(np.expand_dims(img_preprocessed, axis=0)).float().to(DEVICE)
    output, pretrain_feat = model(img_input)
    predict = round(output.item(),3)
    
    #==============================================================================
    # fill in the blank
    #==============================================================================
    search_elem = browser.find_element_by_css_selector("#guess-input")
    search_elem.send_keys(str(predict).split('.')[1])
    time.sleep(3)
    
    browser.find_element_by_css_selector("#submit-btn").click()
    time.sleep(3)
    browser.find_element_by_css_selector("#next-btn").click()



#%% close
browser.close()

