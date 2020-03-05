# -*- coding: utf-8 -*-

import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from utils import getDataset
from model import MainNet
from torchvision import transforms



device = torch.device('cuda')

#creating model dictionary(saving the trained model)
#if not os.path.exists(model_path):
#    os.makedir(model_path)
    
#Image Processing 
transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor])

ques_stoi_path = "../ques_stoi.tsv" 
ans_itos_path  = "../ans_itos.tsv"
samples_tsv_path = "../train.tsv" 
image_path = "../train2014" 
#Building the getDataset
d = getDataset(ques_stoi_path, ans_itos_path,samples_tsv_path,image_path,transform)
dataloader = torch.utils.data.DataLoader(dataset = d,
                                             batch_size = 4,
                                             shuffle = True,
                                             num_workers = 2)

#Building the model
net = MainNet(256,30,256,256,256)

#Loss and Optimizer
criterion = nn.MSELoss()
params = list(net.parameters())
optimizer = torch.optim.Adam(params, lr= 0.001)

#Train the model
steps = len(dataloader)
for epoch in range(10):
    for i, (ques,img,ans) in enumerate(dataloader):
       net = MainNet()
       loss = criterion(input,targets)
       loss.backward()
       optimizer.step()
       if i % 10 == 0:
          print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, 10, i, steps, loss.item(), np.exp(loss.item())))
  

if __name__ == "__main__":
    
#    parser = argparse.ArgumentParser()
#    args = parser.parse_args()
#    ques_stoi_path = "../ques_stoi.tsv" 
#    ans_itos_path  = "../ans_itos.tsv"
#    samples_tsv_path = "../train.tsv" 
#    image_path = "../train2014" 
#    num_epoch = 10
#    model_path = "models/"
#    d = getDataset(ques_stoi_path, ans_itos_path,samples_tsv_path,image_path,transform)
#    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models') 
#    parser.add_argument('--learning_rate', type=float, default=0.001) 
#    parser.add_argument('--num_epochs', type=int, default=5)    
#    args = parser.parse_args()
#    print(args)