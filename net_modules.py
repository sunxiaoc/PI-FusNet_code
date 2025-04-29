
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def initialize_weights(net_l, scale=1):
    # 如果net_l不是一个列表，将其转换为列表
    if not isinstance(net_l, list):
        net_l = [net_l]
    # 遍历每个神经网络
    for net in net_l:
        # 遍历网络的每个模块
        for m in net.modules():
            # 如果当前模块是一个2D卷积层
            if isinstance(m, nn.Conv2d):
                # 使用Kaiming初始化权重，a=0表示使用线性激活函数，mode='fan_in'表示以输入通道数为分母
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                # 缩放权重（通常在残差块中使用）
                m.weight.data *= scale  # for residual block
                # 如果存在偏置项，则将其初始化为零
                if m.bias is not None:
                    m.bias.data.zero_()
            # 如果当前模块是线性层
            elif isinstance(m, nn.Linear):
                # 使用Kaiming初始化权重，a=0表示使用线性激活函数，mode='fan_in'表示以输入通道数为分母
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                # 缩放权重
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            # 如果当前模块是BatchNormalization层
            elif isinstance(m, nn.BatchNorm2d):
                # 将BatchNormalization层的权重初始化为1，偏置项初始化为0
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class Laplace(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Laplace,self).__init__()
        kernel = [[0,1,0],[1,-4,1],[0,1,0]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.conv1=nn.Conv2d(in_channels,out_channels,self.weight,stride=1,padding='same',bias=False)
        # nn.init.constant_(self.conv1.weight,1)
        # nn.init.constant_(self.conv1.weight[0,0,1,1],-8)
        # nn.init.constant_(self.conv1.weight[0,1,1,1],-8)
        # nn.init.constant_(self.conv1.weight[0,2,1,1],-8)
      
    def forward(self,x1):
        edge_map=self.conv1(x1)
        print(edge_map.shape)
        return edge_map

class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=96, bias=True):
        super(DenseBlock, self).__init__()
        # 定义密集连接线，包括4个卷积层
        # in_channels,out_channels,kernel_size,stride,padding,bias
        self.conv1 = nn.Conv2d(channel_in, gc, 3, stride=1, padding='same', bias=bias)  
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, stride=1, padding='same', bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, stride=1, padding='same', bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, channel_out, 1, stride=1, padding='same', bias=bias) # 1*1 conv
        
        # 定义Laplace梯度算子线
        self.conv_l1 = nn.Conv2d(channel_in, channel_in, 1, stride=1, padding='same', bias=bias)  # 1*1 conv
        self.conv_l2= nn.Conv2d(channel_in,channel_in,3,stride=1, padding='same',bias=False)
        nn.init.constant_(self.conv_l2.weight,1)
        nn.init.constant_(self.conv_l2.weight[0,0,1,1],-8)
        nn.init.constant_(self.conv_l2.weight[0,1,1,1],-8)
        nn.init.constant_(self.conv_l2.weight[0,2,1,1],-8)
        self.conv_l3 = nn.Conv2d(channel_in, channel_out, 1, stride=1, padding='same', bias=bias)  # 1*1 conv
    
        
        # 定义LeakyReLU激活函数
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # 根据指定的初始化方法初始化权重
        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4, self.conv_l1, self.conv_l3], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv_l1, self.conv_l3], 0.1)
       # 初始化第4个卷积层的权重
        # initialize_weights(self.conv4, 0)
    
    def forward(self, x):
        # 前向传播函数，输入x经过一系列卷积和激活层
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        # print(x4.shape)
        x_l1 = self.lrelu(self.conv_l1(x))
        x_l2 = self.conv_l2(x_l1)
        x_l3 = self.lrelu(self.conv_l3(x_l2))
        
        out = x4 + x_l3

        return out
    
# multiscale residual attention module
class MultiscaleBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', bias=True):
        super(MultiscaleBlock, self).__init__()
        # 定义密集连接线，包括4个卷积层
        # in_channels,out_channels,kernel_size,stride,padding,bias
        self.conv_f11 = nn.Conv2d(channel_in, channel_out, 7, stride=1, padding='same', bias=bias)  
        self.conv_f12 = nn.Conv2d(channel_in, channel_out, 5, stride=1, padding='same', bias=bias)
        self.conv_f13 = nn.Conv2d(channel_in, channel_out, 3, stride=1, padding='same', bias=bias)
        
        # 定义LeakyReLU激活函数
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # 根据指定的初始化方法初始化权重
        if init == 'xavier':
            initialize_weights_xavier([self.conv_f11, self.conv_f12, self.conv_f13], 0.1)
        else:
            initialize_weights([self.conv_f11, self.conv_f12, self.conv_f13], 0.1)
       
    def forward(self, x):
        # 前向传播函数，输入x经过一系列卷积和激活层
        f11 = self.lrelu(self.conv_f11(x))
        f12 = self.lrelu(self.conv_f12(x))
        f13 = self.lrelu(self.conv_f13(x))
        
        f1 = torch.mul(f12, f11)
        f2 = torch.mul(f12, f13)
        # out = torch.cat((f12, f1, f2), 1)
        out = f12 + f1 +f2

        return out

class MultiscaleBlock2(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', bias=True):
        super(MultiscaleBlock2, self).__init__()
        # 定义密集连接线，包括4个卷积层
        # in_channels,out_channels,kernel_size,stride,padding,bias
        # self.conv_f11 = nn.Conv2d(channel_in, channel_out, 7, stride=1, padding='same', bias=bias)  
        # self.conv_f12 = nn.Conv2d(channel_in, channel_out, 5, stride=1, padding='same', bias=bias)
        self.conv_f13 = nn.Conv2d(channel_in, channel_out, 3, stride=1, padding='same', bias=bias)
        
        # 定义LeakyReLU激活函数
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # 根据指定的初始化方法初始化权重
        if init == 'xavier':
            initialize_weights_xavier([self.conv_f13], 0.1)
        else:
            initialize_weights([self.conv_f13], 0.1)
       
    def forward(self, x):
        # 前向传播函数，输入x经过一系列卷积和激活层
        # f11 = self.lrelu(self.conv_f11(x))
        # f12 = self.lrelu(self.conv_f12(x))
        f13 = self.lrelu(self.conv_f13(x))
        
        
        out = f13

        return out


def subnet(net_structure, init='xavier'):
    # 定义构造函数constructor，该函数接受输入通道数和输出通道数，返回一个特定类型的网络模块
    def constructor(channel_in, channel_out):
        # 如果net_structure是'DR_LGO'，则创建DenseBlock模块
        if net_structure == 'DR_LGO':
            if init == 'xavier':
                return DenseBlock(channel_in, channel_out, init)
            else:
                return DenseBlock(channel_in, channel_out)
            # return UNetBlock(channel_in, channel_out)
        # 如果要使用不同的网络结构，可以在这里添加其他条件分支
        else:
            return None  # 如果net_structure不是已知的结构，返回None

    return constructor

# 创建一个可逆模块    
class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num1, channel_num2, clamp=0.8):
        super(InvBlock, self).__init__()
        
        # 两个multiscalebolck的输出通道数分别作为InvBlock的两个输入通道数
        self.split_len1 = channel_num1 
        self.split_len2 = channel_num2  
        
        self.clamp = clamp
        # 创建三个子网络，分别为F、G和H，使用传入的subnet_constructor构造这些子网络
        # F网络的输入通道数为self.split_len2，输出通道数为self.split_len1
        self.F = subnet_constructor(self.split_len2, self.split_len1)
        # G网络的输入通道数为self.split_len1，输出通道数为self.split_len2
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        # H网络的输入通道数为self.split_len1，输出通道数为self.split_len2
        self.H = subnet_constructor(self.split_len1, self.split_len2)          
        
    def forward(self, x):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
        # 计算y1，y1由x2和经过F子网络后的x1相加得到
        y1 = x2 + self.F(x1) # 1 channel 
        # 计算可学习的缩放因子s，并通过sigmoid函数和线性变换进行缩放
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        # 计算y2，y2由x2经过缩放因子和G子网络后的y1相加得到
        y2 = x1.mul(torch.exp(self.s)) + self.G(y1) # 2 channel
        # 将y1和y2拼接在一起，得到输出out
        out = torch.cat((y1, y2), 1)

        return out    
    
class INNFusion(nn.Module):
    def __init__(self, channel_in, channel_out, subnet_constructor=subnet('DR_LGO'), block_num=3):   
        super(INNFusion, self).__init__()
        operations = []

        channel_num1 = channel_in
        channel_num2 = channel_out 
        # 创建多个可逆块，并将它们添加到operations列表中
        for j in range(block_num): 
            b = InvBlock(subnet_constructor, channel_num1, channel_num2) # one block is one flow step. 
            operations.append(b)
        
        self.operations = nn.ModuleList(operations)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            # 如果是卷积层，则使用Xavier正态分布初始化权重
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= 1.  # for residual block# 用于残差块（通常是标准化权重的一个技巧）
                # 如果存在偏置项，则将其初始化为零
                if m.bias is not None:
                    m.bias.data.zero_() 
            # 如果是线性层，则使用Xavier正态分布初始化权重
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= 1.
                if m.bias is not None:
                    m.bias.data.zero_()
            # 如果是批归一化层，则将权重初始化为1，将偏置项初始化为零
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
    
    def forward(self, x):
        out = x # x: [N,3,H,W] 
        #assert 0
         
        for op in self.operations:
            out = op.forward(out)
        
        return out
    
    
# 特征重建网络   
class RFNet(nn.Module):
    def __init__(self, channel_in, channel_out=1, init='xavier', gc=16, bias=True):
        super(RFNet, self).__init__()
        # 定义6个卷积层
        # in_channels,out_channels,kernel_size,stride,padding,bias
        self.conv1 = nn.Conv2d(channel_in, gc, 3, stride=1, padding='same', bias=bias)  
        self.conv2 = nn.Conv2d(gc, gc, 3, stride=1, padding='same', bias=bias)
        self.conv3 = nn.Conv2d(gc, gc, 3, stride=1, padding='same', bias=bias)
        self.conv4 = nn.Conv2d(gc*2, gc, 3, stride=1, padding='same', bias=bias)
        self.conv5 = nn.Conv2d(gc*2, gc, 3, stride=1, padding='same', bias=bias)
        self.conv6 = nn.Conv2d(gc, channel_out, 3, stride=1, padding='same', bias=bias)
        
        # 定义LeakyReLU激活函数
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # 根据指定的初始化方法初始化权重
        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3, 
                                       self.conv4, self.conv5, self.conv6], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3, 
                                self.conv4, self.conv5, self.conv6], 0.1)
          
    def forward(self, x):
        # 前向传播函数，输入x经过一系列卷积和激活层
        x1 = self.lrelu(self.conv1(x))
        # x2 = self.lrelu(self.conv2(x1))
        # x3 = self.lrelu(self.conv3(x2))
        # x4 = self.lrelu(self.conv4(torch.cat((x2,x3),1)))
        # x5 = self.lrelu(self.conv5(torch.cat((x1,x4),1)))
        x6 = self.conv6(x1)                
        
        return x6    
    
# reversible fusion networks based on physical information
class PhyFusNet(nn.Module): 
    def __init__(self, channel_in_i1, channel_in_i2, out=1, init='xavier', gc=5, bias=True):
        super(PhyFusNet, self).__init__()
        # channel_in_i1 第一个输入的通道数=1，channel_in_i2 第二个输入的通道数=3
        self.f1 = MultiscaleBlock2(channel_in_i1, 16)
        self.f2 = MultiscaleBlock2(channel_in_i2, 16)
        self.innfus = INNFusion(16, 16)
        self.rfnet = RFNet(32)
        
    def forward(self, x1, x2):
        # 分别将两个输入传递给两个MultiscaleBlock
        
        output1 = self.f1(x1)
        output2 = self.f2(x2)
        
        # 将两个输出在维度1上拼接
        combined_output = torch.cat((output1, output2), dim=1)
        
        # 将拼接后的输出传递给INNFusion和RFNet
        innfusion_output = self.innfus(combined_output)
        # print(innfusion_output.shape)   # torch.Size([1, 64, 40, 160])
        out = self.rfnet(innfusion_output)

        return out
        

    
    
    
    
