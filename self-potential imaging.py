# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 18:03:49 2023

@author: 噜啦啦
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import Polygon

# 电荷概率成像反演程序
# 由电势数据（10个电极点）计算电场数据（9个点），只能算x方向的E
def Cal_E(num,V):
    # V = pd.read_excel('D:/博士期间/联合反演/data01.xlsx',header=None)
    # 假设中点的E可以这样计算
    # V = np.array(V)
    Ex = []
    for i in range(num-1):
        Ex.append((V[0][i]-V[0][i+1])/0.3)
    return Ex

# 线电荷模型的扫描函数
def SDS(dx,zq):
    Sx = dx / (dx**2 + zq**2)**1.5
    return Sx

# 概率成像函数
def COP(xq,zq,Ex,num):
    x = np.arange(1.65,3.4,0.3)  #比X少一个，X的中点坐标
    # 对离散数据积分,即：trapz(值y,坐标x)
    f1 = []
    f2 = []
    for i in range(num-1):
        f1.append(Ex[i] * SDS(x[i]-xq, zq))
        f2.append(Ex[i]**2)
    inte_1 = np.trapz(f1,x)
    inte_2 = np.trapz(f2,x)
    # 计算概率
    C = 2*2**0.5 / (np.pi * inte_2)**0.5
    eta = C * zq**1.5 * inte_1
    return eta

# In[1]
def ShowProbability_contourf(x, z, value):
    
    # 注意：需要设置一下可绘制区域的位置和大小
    # levels = np.arange(-1,1,0.05)
    # levels = [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
    #           0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    C = plt.contour(x,z,value,18,colors='black')
    plt.contourf(x,z,value,18)
    plt.clabel(C,inline=1,fontsize=10)
    ax = plt.gca()
    ax.set_title("Probability Diagram")
    # ax.set_title("Probability Diagram for field downward")
    plt.xlabel(u'X-direction(m)',fontsize=10)
    plt.ylabel(u'Z-direction(m)',fontsize=10)
    
    plt.show()
    
# In[2]
def main(V,X,Z,num):
    
    # 电极中点位置处的Ex
    Ex = Cal_E(num,V)
    # 计算结果
    r = []
    result = []
    for height in range(Z.shape[0]):
        for length in range(num):
            # if corr[height-1][length-1] != 0:
            #     eta = 1.5*COP(X[length], Z[height], Ex, num)
            # else:
            eta = COP(X[length], Z[height], Ex, num)
            r.append(eta)
        result.append(r)
        r = []
    result = np.array(result)
    
    return result

# In[3]
# =============================================================================
# 主程序运行
# =============================================================================
if __name__ == '__main__':
    import os
    from scipy import interpolate
    path = 'E:/物理信息网络/数据集/现场数据/SP/'
    path2 = 'E:/物理信息网络/数据集/现场数据/SP_inv/'
    # path3 = 'E:/物理信息网络/数据集/现场数据/SP_img/'
    pathdir = os.listdir(path)
    num = 7
    
    X = np.arange(1.5,3.4,0.3)
    Z = np.arange(0,2,0.1)
    real_Z = np.arange(0,-2,-0.1)
    xx, zz = np.meshgrid(X, real_Z)
    for file in pathdir:
        V = pd.read_excel(path+file,header=None)
        V = np.array(V)
        result = main(V,X,Z,num)
        # ShowProbability_contourf(xx, zz, result)
        
        interpolator = interpolate.interp2d(np.arange(7), np.arange(20), result, kind='linear')
        new_x = np.linspace(0, 7, 160)
        new_y = np.linspace(0, 19, 40)
        new_array = interpolator(new_x, new_y)
        np.save(path2+file[:-5]+'.npy',new_array)