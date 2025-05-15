# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import Polygon

# Charge probability imaging inversion program
def Cal_E(num,V):
    # V = pd.read_excel('D:/博士期间/联合反演/data01.xlsx',header=None)
    # Assuming that the midpoint E can be calculated as follows
    # V = np.array(V)
    Ex = []
    for i in range(num-1):
        Ex.append((V[0][i]-V[0][i+1])/0.3)
    return Ex

# Scanning function of linear charge model
def SDS(dx,zq):
    Sx = dx / (dx**2 + zq**2)**1.5
    return Sx

# Probability imaging function
def COP(xq,zq,Ex,num):
    x = np.arange(1.65,3.4,0.3) 
    # Integrating discrete data
    f1 = []
    f2 = []
    for i in range(num-1):
        f1.append(Ex[i] * SDS(x[i]-xq, zq))
        f2.append(Ex[i]**2)
    inte_1 = np.trapz(f1,x)
    inte_2 = np.trapz(f2,x)
    # Calculate probability
    C = 2*2**0.5 / (np.pi * inte_2)**0.5
    eta = C * zq**1.5 * inte_1
    return eta

# In[1]
def ShowProbability_contourf(x, z, value):
    
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
    
    # Ex at the midpoint of the electrode
    Ex = Cal_E(num,V)
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
# Main program running
# =============================================================================
if __name__ == '__main__':
    import os
    from scipy import interpolate
    path = 'E:/物理信息网络/数据集/现场数据/SP/'
    path2 = 'E:/物理信息网络/数据集/现场数据/SP_inv/'
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
