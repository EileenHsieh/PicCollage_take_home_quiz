#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 01:32:46 2019

@author: eileen
"""


import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from math import pi
from copy import deepcopy
import torch.autograd as autograd



#%%
class CNN(nn.Module):
    def __init__(self,ks1, ks2, ks3):
        super(CNN, self).__init__()
        
        
        self.encoder = nn.Sequential(
            ## conv formula: (N+2*P-K)/S
            nn.Conv2d(1, ks1, kernel_size=3, stride=1,padding=1), #4,64,80,64
            nn.BatchNorm2d(ks1),
            nn.MaxPool2d(2),
            nn.Hardtanh(),           

            nn.Conv2d(ks1, ks2, kernel_size=3, stride=1,padding=1), # 8,32,40,32
            nn.BatchNorm2d(ks2),
            nn.MaxPool2d(2),
            nn.Hardtanh(),
                       
                   
            nn.Conv2d(ks2, ks3, kernel_size=3, stride=1,padding=1), # 16,16,20,16
            nn.BatchNorm2d(ks3),
            nn.Hardtanh(),
        
                   
            nn.Conv2d(ks3, 1, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(1),
            nn.Hardtanh(),
                      )
        self.upSample1 = nn.Upsample(scale_factor=4) 
        self.glob_avg_pool = nn.AvgPool2d(kernel_size = 16,stride=1)
#            nn.MaxPool2d(2),)
        
        

    def forward(self, x):
        x1 = self.encoder(x)
        x1 = self.upSample1(x1)
        output = self.glob_avg_pool(x1)

        return output, x



