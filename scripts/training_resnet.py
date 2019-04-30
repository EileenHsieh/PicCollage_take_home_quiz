#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 12:14:54 2019

@author: eileen
"""

from DataTool import ImgDataset, Rgb2gray, Resize, Binarize, Binarize01, Crop, CountParam
import joblib
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from vanilla_CNN import CNN
from resnet import PreTrainedModel
import matplotlib.pyplot as plt
import numpy as np


#%% parameters
MODINFO = 'resnet_1conv'
TOTAL_NUM = 150000
TRAIN_NUM = int(TOTAL_NUM*0.2)
VAL_NUM = int(TOTAL_NUM*0.01)
TEST_NUM = TOTAL_NUM-(TRAIN_NUM+VAL_NUM)
OPTIM = 'SGD'

MAXEPOCH = 20
inputShape = 64
batch_size = 256
num_workers = 2
lr = 5e-2

ks1, ks2, ks3 = 64,64,32


#%% datalists
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataLists = joblib.load('../sample_data/train{}_val{}_test{}.pkl'.format(TRAIN_NUM, VAL_NUM, TEST_NUM))

trainList = dataLists['train']
valList = dataLists['val']
testList = dataLists['test']

#%% data loader

#composed = transforms.Compose([Crop(2,130,20,148), Rescale(inputShape),AxisTranspose()])
composed = transforms.Compose([Binarize(), 
                               Crop(2,129,21,148),
                               Resize(inputShape), 
                               Binarize01(),
                               transforms.ToTensor(),])
trainLoader = DataLoader(ImgDataset(trainList, transform=composed), batch_size=batch_size,
                        shuffle=True, num_workers=num_workers)
valLoader = DataLoader(ImgDataset(valList, transform=composed), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)
testLoader = DataLoader(ImgDataset(testList, transform=composed), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)



#%%
#model = CNN(ks1,ks2,ks3).to(DEVICE)
model = PreTrainedModel(5,True,ks1,ks2,ks3).to(DEVICE)

loss_mse = torch.nn.MSELoss()  #L1Loss()
#optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

#%% training 
lossTrains, lossVals = [], []
for epoch in range(1,MAXEPOCH+1):
    print(epoch)

    # train
    tr_loss, tr_lab, tr_pred = [], [], []
    model = model.train()
    for i_batch, sample_batched in enumerate(trainLoader):
        optimizer.zero_grad()
        
        output, pretrain_feat = model(sample_batched['img'].float()[:,:3,:,:].to(DEVICE))
        tr_pred.append(output.data.cpu().numpy().reshape(-1))
        lab = sample_batched['lab']        
        tr_lab.append(lab.numpy())
        
        lTrain = loss_mse(output.view(-1), lab.float().to(DEVICE))

        lTrain.backward()
        optimizer.step()
        lossTrains.append(lTrain.item())
        
    tr_benchmark = np.std(np.hstack(tr_lab))**2 
    tr_MSE = np.mean((np.hstack(tr_pred) - np.hstack(tr_lab))**2)
    print("train benchmark: %f, train MSE: %f" %(tr_benchmark, tr_MSE))

    # validation
    model = model.eval()
    val_lab, val_pred = [], []

    for i_batch, sample_batched in enumerate(valLoader):
        output, pretrain_feat = model(sample_batched['img'].float()[:,:3,:,:].to(DEVICE))
        val_pred.append(output.data.cpu().numpy().reshape(-1))
        lab = sample_batched['lab']
        val_lab.append(lab.numpy())
        
        lVal = loss_mse(output.view(-1), lab.float().to(DEVICE))
  
        
    benchmark = np.std(np.hstack(val_lab))**2 #RMSE benchmark: std
    lossVals.append(np.mean((np.hstack(val_pred) - np.hstack(val_lab))**2))
     
print("test benchmark: %f, test MSE: %f" %(benchmark, lossVals[-1]))

#%% plot
modName = '{}-tr{}-te{}-ep{}-in{}-bz{}-opt{}-lr{}-k1{}_k2{}_k3{}'.format(MODINFO, TRAIN_NUM, TEST_NUM, MAXEPOCH,
            inputShape, batch_size, OPTIM, lr, ks1,ks2,ks3)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('loss train')
plt.plot(lossTrains)
plt.savefig('../models/lTrain_{}.png'.format(modName))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('loss test')
plt.plot(lossVals)
plt.savefig('../models/lTest_{}.png'.format(modName))

joblib.dump(model, '../models/{}-tr{}-te{}-ep{}-in{}-bz{}-lr{}-k1{}_k2{}_k3{}.pkl'.format(MODINFO, TRAIN_NUM, TEST_NUM, MAXEPOCH,
            inputShape, batch_size, lr, ks1,ks2,ks3))


#%% evaluate on test
model = model.eval()
ts_lab, ts_pred = [], []
for i_batch, sample_batched in enumerate(testLoader):
    output, pretrain_feat = model(sample_batched['img'].float()[:,:3,:,:].to(DEVICE))
    ts_pred.append(output.data.cpu().numpy().reshape(-1))
    lab = sample_batched['lab']
    ts_lab.append(lab.numpy())
    
ts_MSE = np.mean((np.hstack(ts_pred) - np.hstack(ts_lab))**2)
print('\n\nFinal losses: train:{:.5} | val:{:.5} | test:{:.5}'.format(tr_MSE, lossVals[-1], ts_MSE ))

#%% calculate parameters
cnt_paras_all = CountParam(model, requires_grad=False)
cnt_paras_grad = CountParam(model, requires_grad=True)
print('Total params:{}'.format(cnt_paras_all))
print('Trainable params:{}'.format(cnt_paras_grad))


    
    
   

        