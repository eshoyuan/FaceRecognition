import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from PIL import Image, ImageDraw, ExifTags
import datetime
from torch.autograd import Variable
import torch
import cv2
import numpy as np
from facenet_pytorch import MTCNN, extract_face
import matplotlib.pyplot as plt
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms, utils, datasets, models
import re
import torch.utils
import torch.optim.lr_scheduler as lr_scheduler
from  torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from mynet import *
from centerloss import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
"""
模型选择:
不同的batch_size(16,32,64)
不同的epoch(5,10,20,40,80)
不同的loss weight(0.001,0.01,0.1,1)
保存模型 注意命名
"""
class face_dataset(torch.utils.data.Dataset):
    def __init__(self):
        data=np.load("saved_file/train_mtcnn.npy")
        label=np.load("saved_file/train_label.npy")
        self.label=torch.LongTensor(label).to(device)
        self.img=torch.tensor(data).to(device)
    def __getitem__(self,index):
        img=self.img[index]
        label=self.label[index]
        return img,label
    def __len__(self):
        return self.label.shape[0]

def train(epoch,train_loader,net,loss_weight,optimizer4nn,optimzer4center,centerloss):
    for i,data in enumerate(train_loader):
        img, target = data[0].to(device), data[1].to(device)
        ip1, pred = net(img)
        loss = F.cross_entropy(pred, target)+loss_weight * centerloss(target, ip1)#+
        print(loss)
        optimizer4nn.zero_grad()
        optimzer4center.zero_grad()
        loss.backward()
        optimizer4nn.step()
        optimzer4center.step()

def eval(batch,epoch,loss_weight):
    set=face_dataset()
    train_loader = DataLoader(set, batch_size=batch, shuffle=True, num_workers=0)
    net=Net(device=device)
    #加载pretrained参数
    resnet = InceptionResnetV1(pretrained='vggface2').to(device)
    pretrained_dict = resnet.state_dict()
    model_dict = net.state_dict()
    pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    ####
    centerloss = CenterLoss(num_classes=41, feat_dim=512).to(device)
    # optimzer4nn
    optimizer4nn = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),lr=0.001,momentum=0.9, weight_decay=0.0005)
    sheduler = lr_scheduler.StepLR(optimizer4nn,20,gamma=0.8)
    # optimzer4center
    optimzer4center = optim.SGD(centerloss.parameters(), lr =0.5)
    net=net.eval()
    for epoch in range(epoch):
        sheduler.step()
        print(epoch)
        train(epoch+1,train_loader,net,loss_weight,optimizer4nn,optimzer4center,centerloss)
    dataX=set.img.detach().cpu()
    dataY=set.label.detach().cpu().numpy()
    #torch.save(net.state_dict(), f'saved_weights/net_batch{batch}_epoch_{epoch}_lossW{loss_weight}.pth')
    testX=np.load("saved_file/test_mtcnn.npy")
    testY=np.load("saved_file/test_label.npy") 
    with open("result.txt","a") as f:
        f.write(f"batch{batch}_epoch{epoch}_lossW{loss_weight}\n")
    net.to('cpu')
    X,_=net(dataX)
    X=X.numpy()
    for k in range(1,5):
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(X, dataY)
        #with open("result.txt","a") as f:
        temp,_=net(torch.tensor(testX))
        #f.write(f"\t k={k}: acc:  {sum((neigh.predict(temp.numpy())==testY))/41}\n")
        print(f"\t k={k}: acc:  {sum((neigh.predict(temp.numpy())==testY))/41}\n")
    
# eval(batch=16,epoch=0,loss_weight=1)
# for loss_weight in [0.001,0.01,0.1,1]:
#     for epoch in [5,10,20,40]:
#         for batch in [8,16,32,64]:
eval(32,10,0.01)

    

