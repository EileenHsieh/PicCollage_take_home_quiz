#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 09:59:34 2019

@author: eileen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PreTrainedModel(nn.Module):
    def __init__(self, num_res_layers, freeze,ks1, ks2, ks3):
        super(PreTrainedModel, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        for param in resnet18.parameters():
            param.requires_grad = not(freeze)
        self.res18_1 = nn.Sequential(*list(resnet18.children())[:2]) # output:64, 64, 16, 16
        self.res18_2 = nn.Sequential(*list(resnet18.children())[3:num_res_layers])

        self.encoder1 = nn.Sequential(
            ## conv formula: (N+2*P-K+1)/S
            nn.Conv2d(64, ks1, kernel_size=3, stride=1,padding=1), #4,64,80,64
            nn.BatchNorm2d(ks1),
            nn.MaxPool2d(2),
            nn.Hardtanh(),                       
            )
        
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(ks1, ks2, kernel_size=3, stride=1,padding=1), # 8,32,40,32
            nn.BatchNorm2d(ks2),
            nn.MaxPool2d(2),
            nn.Hardtanh(),
            )
        
        self.encoder3 = nn.Sequential(          
            nn.Conv2d(ks2, ks3, kernel_size=3, stride=1,padding=1), # 16,16,20,16
            nn.BatchNorm2d(ks3),
            nn.Hardtanh(),

                            
            nn.Conv2d(ks3, 1, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(1),
            nn.Hardtanh(),)
        
        self.upSample1 = nn.Upsample(scale_factor=4) 
        

        self.glob_avg_pool = nn.AvgPool2d(kernel_size = 16,stride=1)
        
     	
    def forward(self,x):
        out1 = self.res18_1(x)
        out1 = self.res18_2(out1)
        ###### follow with my finetune layers
        x1 = self.encoder1(out1)
        x1 = self.encoder2(x1)
        x1 = self.encoder3(x1)
#        ###### up sampling to create output with the same size
        x1 = self.upSample1(x1)       
        pred = self.glob_avg_pool(x1)
        
        return pred, out1
 
    

