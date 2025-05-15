
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def initialize_weights(net_l, scale=1):
    # If net_1 is not a list, convert it to a list
    if not isinstance(net_l, list):
        net_l = [net_l]
    # Traverse each neural network
    for net in net_l:
        # Traverse every module of the network
        for m in net.modules():
            # If the current module is a 2D convolutional layer
            if isinstance(m, nn.Conv2d):
                # Initialize weights using Kaiming, where a=0 indicates the use of a linear activation function, and mode='fan-in 'indicates the denominator is the number of input channels
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                # Scaling weights (usually used in residual blocks)
                m.weight.data *= scale  # for residual block
                # If there is a bias term, initialize it to zero
                if m.bias is not None:
                    m.bias.data.zero_()
            # If the current module is a linear layer
            elif isinstance(m, nn.Linear):
               
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                # Scale weight
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            # If the current module is a BatchNormalization layer
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize the weights of BatchNormalization layer to 1 and the bias term to 0
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
        # Define dense connection lines, including 4 convolutional layers
        # in_channels,out_channels,kernel_size,stride,padding,bias
        self.conv1 = nn.Conv2d(channel_in, gc, 3, stride=1, padding='same', bias=bias)  
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, stride=1, padding='same', bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, stride=1, padding='same', bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, channel_out, 1, stride=1, padding='same', bias=bias) # 1*1 conv
        
        # Define Laplace gradient operator line
        self.conv_l1 = nn.Conv2d(channel_in, channel_in, 1, stride=1, padding='same', bias=bias)  # 1*1 conv
        self.conv_l2= nn.Conv2d(channel_in,channel_in,3,stride=1, padding='same',bias=False)
        nn.init.constant_(self.conv_l2.weight,1)
        nn.init.constant_(self.conv_l2.weight[0,0,1,1],-8)
        nn.init.constant_(self.conv_l2.weight[0,1,1,1],-8)
        nn.init.constant_(self.conv_l2.weight[0,2,1,1],-8)
        self.conv_l3 = nn.Conv2d(channel_in, channel_out, 1, stride=1, padding='same', bias=bias)  # 1*1 conv
    
        
        # Define LeakyReLU activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # Initialize weights according to the specified initialization method
        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4, self.conv_l1, self.conv_l3], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv_l1, self.conv_l3], 0.1)
       # Initialize the weights of the fourth convolutional layer
        # initialize_weights(self.conv4, 0)
    
    def forward(self, x):
        # Forward propagation function, where the input x undergoes a series of convolutions and activation layers
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
        # Define dense connection lines, including 4 convolutions
        # in_channels,out_channels,kernel_size,stride,padding,bias
        self.conv_f11 = nn.Conv2d(channel_in, channel_out, 7, stride=1, padding='same', bias=bias)  
        self.conv_f12 = nn.Conv2d(channel_in, channel_out, 5, stride=1, padding='same', bias=bias)
        self.conv_f13 = nn.Conv2d(channel_in, channel_out, 3, stride=1, padding='same', bias=bias)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if init == 'xavier':
            initialize_weights_xavier([self.conv_f11, self.conv_f12, self.conv_f13], 0.1)
        else:
            initialize_weights([self.conv_f11, self.conv_f12, self.conv_f13], 0.1)
       
    def forward(self, x):
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
        # in_channels,out_channels,kernel_size,stride,padding,bias
        # self.conv_f11 = nn.Conv2d(channel_in, channel_out, 7, stride=1, padding='same', bias=bias)  
        # self.conv_f12 = nn.Conv2d(channel_in, channel_out, 5, stride=1, padding='same', bias=bias)
        self.conv_f13 = nn.Conv2d(channel_in, channel_out, 3, stride=1, padding='same', bias=bias)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        if init == 'xavier':
            initialize_weights_xavier([self.conv_f13], 0.1)
        else:
            initialize_weights([self.conv_f13], 0.1)
       
    def forward(self, x):
        # f11 = self.lrelu(self.conv_f11(x))
        # f12 = self.lrelu(self.conv_f12(x))
        f13 = self.lrelu(self.conv_f13(x))
        
        
        out = f13

        return out


def subnet(net_structure, init='xavier'):
    # Define a constructor function that accepts the number of input channels and output channels, and returns a specific type of network module
    def constructor(channel_in, channel_out):
        # If net_stucture is' DR_LGO ', create DenseBlock module
        if net_structure == 'DR_LGO':
            if init == 'xavier':
                return DenseBlock(channel_in, channel_out, init)
            else:
                return DenseBlock(channel_in, channel_out)
            # return UNetBlock(channel_in, channel_out)
        # If you want to use different network structures, you can add other conditional branches here
        else:
            return None  # If net_stucture is not a known structure, return None

    return constructor

# Create a reversible module 
class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num1, channel_num2, clamp=0.8):
        super(InvBlock, self).__init__()
        
        # The number of output channels for two multiscale blocks is used as the number of input channels for InvBlock
        self.split_len1 = channel_num1 
        self.split_len2 = channel_num2  
        
        self.clamp = clamp
        # Create three subnetworks, namely F, G, and H, and construct these subnetworks using the passed subnet-struct
        # The input channel number of the F network is self. split-len2, and the output channel number is self. split-len1
        self.F = subnet_constructor(self.split_len2, self.split_len1)
        # The input channel number of the G network is self. split-len1, and the output channel number is self. split-len2
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        # The input channel number of the H network is self. split-len1, and the output channel number is self. split-len2
        self.H = subnet_constructor(self.split_len1, self.split_len2)          
        
    def forward(self, x):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
        # Calculate y1, which is obtained by adding x2 and x1 after passing through the F sub network
        y1 = x2 + self.F(x1) # 1 channel 
        # Calculate the learnable scaling factor s and scale it using sigmoid function and linear transformation
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        # Calculate y2, which is obtained by adding y1 from x2 after scaling factor and G sub network
        y2 = x1.mul(torch.exp(self.s)) + self.G(y1) # 2 channel
        # Concatenate y1 and y2 together to obtain output out
        out = torch.cat((y1, y2), 1)

        return out    
    
class INNFusion(nn.Module):
    def __init__(self, channel_in, channel_out, subnet_constructor=subnet('DR_LGO'), block_num=3):   
        super(INNFusion, self).__init__()
        operations = []

        channel_num1 = channel_in
        channel_num2 = channel_out 
        # Create multiple reversible blocks and add them to the operations list
        for j in range(block_num): 
            b = InvBlock(subnet_constructor, channel_num1, channel_num2) # one block is one flow step. 
            operations.append(b)
        
        self.operations = nn.ModuleList(operations)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            # If it is a convolutional layer, use Xavier normal distribution to initialize weights
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= 1.  # for residual block
                # If there is a bias term, initialize it to zero
                if m.bias is not None:
                    m.bias.data.zero_() 
            # If it is a linear layer, use Xavier normal distribution to initialize weights
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= 1.
                if m.bias is not None:
                    m.bias.data.zero_()
            # If it is a batch normalization layer, initialize the weights to 1 and the bias term to zero
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
    
    def forward(self, x):
        out = x # x: [N,3,H,W] 
        #assert 0
         
        for op in self.operations:
            out = op.forward(out)
        
        return out
    
    
# Feature reconstruction network  
class RFNet(nn.Module):
    def __init__(self, channel_in, channel_out=1, init='xavier', gc=16, bias=True):
        super(RFNet, self).__init__()
        # Define 6 convolutional layers
        # in_channels,out_channels,kernel_size,stride,padding,bias
        self.conv1 = nn.Conv2d(channel_in, gc, 3, stride=1, padding='same', bias=bias)  
        self.conv2 = nn.Conv2d(gc, gc, 3, stride=1, padding='same', bias=bias)
        self.conv3 = nn.Conv2d(gc, gc, 3, stride=1, padding='same', bias=bias)
        self.conv4 = nn.Conv2d(gc*2, gc, 3, stride=1, padding='same', bias=bias)
        self.conv5 = nn.Conv2d(gc*2, gc, 3, stride=1, padding='same', bias=bias)
        self.conv6 = nn.Conv2d(gc, channel_out, 3, stride=1, padding='same', bias=bias)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3, 
                                       self.conv4, self.conv5, self.conv6], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3, 
                                self.conv4, self.conv5, self.conv6], 0.1)
          
    def forward(self, x):
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
        # The number of channels for the first input of channel_1_i1 is 1, and the number of channels for the second input of channel_1_i2 is 3
        self.f1 = MultiscaleBlock2(channel_in_i1, 16)
        self.f2 = MultiscaleBlock2(channel_in_i2, 16)
        self.innfus = INNFusion(16, 16)
        self.rfnet = RFNet(32)
        
    def forward(self, x1, x2):
        # Pass two inputs separately to two Multiscale Locks
        
        output1 = self.f1(x1)
        output2 = self.f2(x2)
        
        # Concatenate two outputs on dimension 1
        combined_output = torch.cat((output1, output2), dim=1)
        
        # Pass the concatenated output to INNFusion and RFNet
        innfusion_output = self.innfus(combined_output)
        # print(innfusion_output.shape)   # torch.Size([1, 64, 40, 160])
        out = self.rfnet(innfusion_output)

        return out
        

    
    
    
    
