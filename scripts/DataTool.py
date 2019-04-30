#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:03:05 2019

@author: eileen
"""
from torch.utils.data import Dataset
from skimage import io, transform, color
import numpy as np
from skimage.transform import resize



class ImgDataset(Dataset):
    def __init__(self, dataList, transform=None):
        self.dataList = dataList
        self.transform = transform

    def __len__(self):
        return len(self.dataList)

    def __getitem__(self, idx):
        img = io.imread(self.dataList[idx][0])
        lab = self.dataList[idx][1]
        name = self.dataList[idx][0].split('/')[-1].split('.')[0]
        if self.transform:
            img = self.transform(img)
        
        sample = {'name':name, 'img': img, 'lab': lab}
        return sample

def CountParam(model, requires_grad=True):
    if requires_grad:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def weighted_mse_loss(logit,lab,err_ws):
    pct_var = (logit-lab)**2
    out = pct_var * err_ws
    loss = out.mean()
    return loss


#%%
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, img):
        h, w = img.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(img, (new_h, new_w))
        return img
    
    
class Rgb2gray(object):
    def __init__(self):
        pass
    def __call__(self, img):
        img_gray = color.rgb2gray(img)
        img_gray = np.expand_dims(img_gray, axis=0)
        return img_gray
    
class Crop(object):
    def __init__(self, h_start,h_end,w_start,w_end):
        self.h_start = h_start
        self.h_end = h_end
        self.w_start = w_start
        self.w_end = w_end
        pass
    def __call__(self, img):
        img = img[self.h_start:self.h_end,self.w_start:self.w_end,:]
        return img
    
class Binarize(object):
    def __init__(self):
        pass
    def __call__(self, img):
        img[img>=200] = 255
        img[img<200] = 0
        return img
    
class Binarize01(object):
    def __init__(self):
        pass
    def __call__(self, img):
        img[img<0.9] = 0
        img[img>=0.9] = 1
        return img

class Resize(object):
    def __init__(self, output_size):
        self.output_size = output_size
    def __call__(self, img):
        img = resize(img, (self.output_size, self.output_size),
                       anti_aliasing=True)
        return img    

