import pandas as pd
import numpy as np
import os, time, random
import argparse
import json
from torch import optim
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import torchvision
import torch.nn as nn
from sklearn.model_selection import train_test_split
from net_modules import PhyFusNet, MultiscaleBlock,INNFusion,RFNet
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torch.autograd as autograd


os.environ['CUDA_VISIBLE_DEVICES']='0,1'

def dU_xx(x0, z0, x1, z1, x, z, rho):
    du_xx = (rho / (2 * torch.pi)) * (1 / ((x-x0)**2 + (z-z0)**2)**(3/2)
                                        - (3 * (x-x0)**2) / ((x-x0)**2 + (z-z0)**2)**(5/2)
                                        - 1 / ((x-x1)**2 + (z-z1)**2)**(3/2)
                                        + (3 * (x-x1)**2) / ((x-x1)**2 + (z-z1)**2)**(5/2))
    return du_xx

def dU_zz(x0, z0, x1, z1, x, z, rho):
    du_zz = (rho / (2 * torch.pi)) * (1 / ((x-x0)**2 + (z-z0)**2)**(3/2)
                                        - (3 * (z-z0)**2) / ((x-x0)**2 + (z-z0)**2)**(5/2)
                                        - 1 / ((x-x1)**2 + (z-z1)**2)**(3/2)
                                        + (3 * (z-z1)**2) / ((x-x1)**2 + (z-z1)**2)**(5/2))
    return du_zz

def U(x0, z0, x1, z1, x, z, rho):
    u = (rho / (2 * torch.pi)) * (1 / ((x-x0)**2 + (z-z0)**2)**(1/2) 
                                  - 1 / ((x-x1)**2 + (z-z1)**2)**(1/2))
    return u


# 自定义损失函数
class PhyLoss(nn.Module):
    def __init__(self, x0, z0, x1, z1, x, z,s_l):
        super(PhyLoss, self).__init__()
        self.x0 = x0
        self.z0 = z0
        self.x1 = x1
        self.z1 = z1
        self.x = x
        self.z = z
        self.lambda_factor = 2 * torch.pi
        self.s_l = s_l  # 标签电阻
    
    
    def forward(self, rho):
        du_xx = dU_xx(self.x0, self.z0, self.x1, self.z1, self.x, self.z, rho)
        du_zz = dU_zz(self.x0, self.z0, self.x1, self.z1, self.x, self.z, rho)
        
        t1 = (1/rho) * du_xx + (1/rho) * du_zz - self.lambda_factor**2 * (1/rho) * U(self.x0, self.z0, self.x1, self.z1, self.x, self.z, rho)
        
        du_xx_l = dU_xx(self.x0, self.z0, self.x1, self.z1, self.x, self.z, self.s_l)
        du_zz_l = dU_zz(self.x0, self.z0, self.x1, self.z1, self.x, self.z, self.s_l)
        
        t2 = (1/self.s_l) * du_xx_l + (1/self.s_l) * du_zz_l - self.lambda_factor**2 * (1/self.s_l) * U(self.x0, self.z0, self.x1, self.z1, self.x, self.z, self.s_l)
        
        loss = torch.mean((t1 - t2)**2)
        return loss
    
def train_net(net, device, dataset, epochs=1000, batch_size=3, lr=0.00001):
    Loss = []
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    
    # criterion = nn.SmoothL1Loss().cuda()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 训练epochs次
    for epoch in range(epochs):
        # 训练模式
        net.train()
        # 按照batch_size开始训练
        for xx_train, yy_train in train_loader:
            optimizer.zero_grad()
            # 将数据拷贝到device中
            i1 = xx_train[:,3:,:,:].to(device=device, dtype=torch.float32)
            i2 = xx_train[:,0:3,:,:].to(device=device, dtype=torch.float32)
    
            label = yy_train.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = net(i1,i2)
            # 计算loss
            phy = PhyLoss(l_ax, l_az, l_bx, l_bz, l_mx, l_mz, label)
            loss_phy = phy(pred)

            mse = nn.MSELoss()
            loss_mse = mse(pred,label)

            loss = loss_mse + 0*loss_phy

            print('Loss/train', loss.item())
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), '1l+0l_best_model.pth')
            # 更新参数
            loss.backward()
            optimizer.step()
        Loss.append(loss.item())
    Loss0 = np.array(Loss)
    np.save('1l+0l_epoch_{}'.format(epochs),Loss0)
if __name__ == "__main__":
    import time
    
    l_ax = torch.tensor(np.load('Loc_Ax.npy')).cuda()
    l_az = torch.tensor(np.load('Loc_Az.npy')).cuda()
    l_bx = torch.tensor(np.load('Loc_Bx.npy')).cuda()
    l_bz = torch.tensor(np.load('Loc_Bz.npy')).cuda()
    l_mx = torch.tensor(np.load('Loc_Mx.npy')).cuda()
    l_mz = torch.tensor(np.load('Loc_Mz.npy')).cuda()
    
    # start = time.perf_counter()
    # # 加载数据
    # filepath1 = "C:/Users/DELL/Desktop/孙晓晨/合并数据2/"
    # pathdir1 = os.listdir(filepath1)
    
    # filepath2 = "C:/Users/DELL/Desktop/孙晓晨/label2/"
    # pathdir2 = os.listdir(filepath2)
    
    # x_train, x_test, y_train, y_test = train_test_split(pathdir1, pathdir2, test_size=0.1,random_state=1994)
    # r1 = []
    # for file in x_train:
    #     d = np.load(filepath1 + file)
    #     r1.append(d)
    # xx_train = np.array(r1)
    # r2 = []
    # for file in y_train:
    #     d = np.array(pd.read_csv(filepath2 + file))
    #     d = d.reshape(1, d.shape[0], d.shape[1])
    #     r2.append(d)
    # yy_train = np.array(r2)
    # r3 = []
    # for file in x_test:
    #     d = np.load(filepath1 + file)
    #     r3.append(d)
    # xx_test = np.array(r3)
    # r4 = []
    # for file in y_test:
    #     d = np.array(pd.read_csv(filepath2 + file))
    #     d = d.reshape(1, d.shape[0], d.shape[1])
    #     r4.append(d)
    # yy_test = np.array(r4)
    # xx_train = torch.tensor(xx_train)
    # yy_train = torch.tensor(yy_train)
    # dataset = Data.TensorDataset(xx_train, yy_train)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net = PhyFusNet(channel_in_i1=1,channel_in_i2=3)
    
    # net.to(device=device)
    # # 开始训练
    # train_net(net, device, dataset)
    # end = time.perf_counter()
    # print("运行时间为", round(end - start), 'seconds')

    #ceshi
    # filepath3 = "C:/Users/DELL/Desktop/孙晓晨/测试集/"
    # pathdir3 = os.listdir(filepath3)
    # r5 = []
    # for file in pathdir3:
    #     d = np.load(filepath3 + file)
    #     r5.append(d)
    # test = np.array(r5)
    # xx_test = torch.tensor(xx_test)
    # xx_test = xx_test.to(device=device, dtype=torch.float32)
    # predict = net(xx_test[:,3:,:,:], xx_test[:,0:3:,:,:])
    # predict = predict.cpu().detach().numpy()
    # np.save('1l+0l_predict.npy',predict)
    
    # a = predict[7,0,:,:]
    # plt.imshow(a)
    

    #现场数据测试
    f_path = 'E:/物理信息网络/数据集/现场数据/数据组合/'
    pathdir = os.listdir(f_path)
    net = PhyFusNet(channel_in_i1=1,channel_in_i2=3).to(device)
    net.load_state_dict(torch.load('D:/博士期间/物理信息网络/epoch=1000/1l+2l_best_model.pth'))
    # d = np.load('E:/物理信息网络/数据集/现场数据/数据组合/5.22-11.06.npy')
    
    f_d = []
    for file in pathdir:
        d = np.load(f_path + file)
        f_d.append(d)
    f_d = np.array(f_d)
    f_d = torch.tensor(f_d)
    f_d = f_d.to(device=device, dtype=torch.float32)
    
    f_pred = net(f_d[:,3:,:,:], f_d[:,0:3:,:,:])
    f_pred = f_pred.cpu().detach().numpy()
    np.save('E:/物理信息网络/数据集/现场数据/结果/'+file[:-4],f_pred)
    
    

