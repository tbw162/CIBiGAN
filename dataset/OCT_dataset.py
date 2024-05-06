# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 14:00:09 2022

@author: lab503
"""

import argparse
import os
import numpy as np
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image
from itertools import product
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torch.autograd import Variable
import tqdm
import copy
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import torch
from PIL import Image
import inspect
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from itertools import cycle
import warnings

'''def create_dataloader():

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    train_dir = r'~/data/'
    train_dataset = datasets.ImageFolder(train_dir, transform)

    print(train_dataset.class_to_idx)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers = 1, pin_memory=True)
    return train_loader

train_loader = create_dataloader()
train_data = []
train_label = []
for idx, (image, target) in enumerate(train_loader):
    train_data.append(image.data.cpu().numpy())
    train_label.append(target.cpu().numpy())
train_data = np.concatenate(train_data)
train_label = np.concatenate(train_label)
np.save('test_data.npy',train_data)
np.save('test_label.npy',train_label)'''


def create_dataset(opt,kwargs):
    
    data_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        ])
    
    train_data = np.load(r'/data/train_data.npy')
    train_labels= np.load(r'data/train_label.npy')
    normal_train_data = train_data[train_labels==3]
    normal_train_labels = train_labels[train_labels==3]
    anomaly_train_data = train_data[train_labels!=3]
    anomaly_train_labels = train_labels[train_labels!=3]
    print('all anomalous', anomaly_train_data.shape[0])
    normal_amount = normal_train_data.shape[0]
    print('normal_amount',normal_amount)
    anomaly_amount = int((normal_amount//(1-opt.gamma_p))-normal_amount)
    print(anomaly_amount,'anomaly_amount')
   
    randIdx_anomaly = np.arange(anomaly_train_data.shape[0])
    np.random.shuffle(randIdx_anomaly)
    
    
    train_unlabeled_data = np.concatenate((normal_train_data,anomaly_train_data[randIdx_anomaly[:anomaly_amount]]),axis=0)
    train_unlabeled_labels = np.concatenate((normal_train_labels,anomaly_train_labels[randIdx_anomaly[:anomaly_amount]]),axis=0)
    
    anomaly_train_data = anomaly_train_data[randIdx_anomaly[anomaly_amount:]]
    anomaly_train_labels = anomaly_train_labels[randIdx_anomaly[anomaly_amount:]]
    
    auxiliary_amount = int(opt.gamma_l*normal_amount/(1-opt.gamma_l))
    print('auxiliary_amount',auxiliary_amount) 
    #要改不完备的话就是把这里的!=normal_digit 改成==auxiliary_digit
    auxiliary_data = anomaly_train_data[anomaly_train_labels!=opt.normal_digit]
    auxiliary_labels = anomaly_train_labels[anomaly_train_labels!=opt.normal_digit]
    
    randIdx_auxiliary = np.arange(auxiliary_data.shape[0])
    auxiliary_data = auxiliary_data[randIdx_auxiliary[:auxiliary_amount]]
    auxiliary_labels = auxiliary_labels[randIdx_auxiliary[:auxiliary_amount]]
    
    unlabeled_dataset = TrainDataset(train_unlabeled_data,train_unlabeled_labels)
    auxiliary_dataset = TrainDataset(auxiliary_data,auxiliary_labels)
    
    
    test_data = np.load(r'/home/tbw/AAAI1/OCT2017/test_data.npy')
    test_labels = np.load(r'/home/tbw/AAAI1/OCT2017/test_label.npy')
    normal_test_data = test_data[test_labels==3]
    normal_test_labels = test_labels[test_labels==3]
    
    anomaly_test_data = test_data[test_labels!=3]
    anomaly_test_labels = test_labels[test_labels!=3]
    
    test_random_normal = int(0.2*normal_test_data.shape[0]) #划分测试集和验证集
    test_random_anomaly = int(0.2*anomaly_test_data.shape[0])
    randIdx_test_normal = np.arange(normal_test_data.shape[0])
    randIdx_test_anomaly = np.arange(anomaly_test_data.shape[0])
    val_data = np.concatenate((normal_test_data[randIdx_test_normal[:test_random_normal]],anomaly_test_data[randIdx_test_anomaly[:test_random_anomaly]]),axis=0)
    val_labels = np.concatenate((normal_test_labels[randIdx_test_normal[:test_random_normal]],anomaly_test_labels[randIdx_test_anomaly[:test_random_anomaly]]),axis=0)
    
    test_data = np.concatenate((normal_test_data[randIdx_test_normal[test_random_normal:]],anomaly_test_data[randIdx_test_anomaly[test_random_anomaly:]]),axis=0)
    test_labels = np.concatenate((normal_test_labels[randIdx_test_normal[test_random_normal:]],anomaly_test_labels[randIdx_test_anomaly[test_random_anomaly:]]),axis=0)
    
    val_dataset = TrainDataset(val_data,val_labels)
    test_dataset = TrainDataset(test_data,test_labels)
    return unlabeled_dataset, auxiliary_dataset, val_dataset, test_dataset
class TrainDataset(Dataset):
    def __init__(self,data,targets):
        
        self.data = data
        self.targets = targets
     
       
        
    def __getitem__(self, index):
 
        
        return (self.data[index],self.targets[index])
        
    def __len__(self):
        return len(self.data)
    
    
