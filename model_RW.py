import math
import torch
import numpy as np
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable, Function
from torch.nn.parameter import Parameter
from torch.nn import functional as Func
from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

from IPython.core import debugger
debug = debugger.Pdb().set_trace

affine_par = True

def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=False)

class RW_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, shrink_factor):
        super(RW_Module, self).__init__()
        self.chanel_in = in_dim
        self.shrink_factor = shrink_factor

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        
    def own_softmax1(self, x):
    
        maxes1 = torch.max(x, 1, keepdim=True)[0]
        maxes2 = torch.max(x, 2, keepdim=True)[0]
        x_exp = torch.exp(x-0.5*maxes1-0.5*maxes2)
        x_exp_sum_sqrt = torch.sqrt(torch.sum(x_exp, 2, keepdim=True))

        return (x_exp/x_exp_sum_sqrt)/torch.transpose(x_exp_sum_sqrt, 1, 2)
    
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        x_shrink = x
        m_batchsize, C, height, width = x.size()
        if self.shrink_factor != 1:
            height = (height - 1) // self.shrink_factor + 1
            width = (width - 1) // self.shrink_factor + 1
            x_shrink = Func.interpolate(x_shrink, size=(height, width), mode='bilinear', align_corners=True)
            
        
        proj_query = self.query_conv(x_shrink).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x_shrink).view(m_batchsize, -1, width*height)
        
        energy = torch.bmm(proj_query, proj_key)

        attention = self.softmax(energy)

        proj_value = self.value_conv(x_shrink).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        
        if self.shrink_factor != 1:
            height = (height - 1) * self.shrink_factor + 1
            width = (width - 1) * self.shrink_factor + 1
            out = Func.interpolate(out, size=(height, width), mode='bilinear', align_corners=True)

        out = self.gamma*out + x
        return out,energy


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = BatchNorm(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)

        padding = dilation
        self.conv2 = conv3x3(planes, planes, stride=1, padding=padding, dilation = dilation)
        self.bn2 = BatchNorm(planes, affine = affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,BatchNorm=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = BatchNorm(planes, affine = affine_par)

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=padding, bias=False, dilation = dilation)
        self.bn2 = BatchNorm(planes, affine = affine_par)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4, affine = affine_par)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, BatchNorm=nn.BatchNorm2d):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm(64, affine = affine_par)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0],BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2,BatchNorm=BatchNorm)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4,BatchNorm=BatchNorm)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,BatchNorm=nn.BatchNorm2d,dilations=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion,affine = affine_par))
        layers = []
        if dilations == None:
            layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample,BatchNorm=BatchNorm))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))
        else:
            layers.append(block(self.inplanes, planes, stride,dilation=dilations[0], downsample=downsample,BatchNorm=BatchNorm))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, dilation=dilations[i], BatchNorm=BatchNorm))
        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, inplane, dilation_series, BatchNorm, num_classes):
        return block(inplane, dilation_series, BatchNorm, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        out = self.layer4(x3)

        return out, x1, x2, x3



class DifNet(nn.Module):
    def __init__(self, num_classes, layers ,BatchNorm=nn.BatchNorm2d, shrink_factor=2):
        super(DifNet, self).__init__()
        inter_channels = 2048 // 4
        self.conv5a = nn.Sequential(nn.Conv2d(2048, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm(inter_channels),
                                   nn.ReLU())
        self.conv5b = nn.Sequential(nn.Conv2d(2048, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm(inter_channels),
                                   nn.ReLU())
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm(inter_channels),
                                   nn.ReLU())
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, num_classes, 1))
        
        self.PAM_Module = RW_Module(512,shrink_factor)
        
        #self.aux = nn.Sequential(
        #        nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
        #        nn.BatchNorm2d(256),
        #        nn.ReLU(inplace=True),
        #        nn.Dropout2d(p=0.1),
        #        nn.Conv2d(256, num_classes, kernel_size=1)
        #    )
        
        #self.edge_layer = Edge_Module()

        if layers == 18:
            self.model_sed = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, BatchNorm=BatchNorm)
        elif layers == 34:
            self.model_sed = ResNet(BasicBlock, [3, 4, 6, 3], num_classes, BatchNorm=BatchNorm)
        elif layers == 50: 
            self.model_sed = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, BatchNorm=BatchNorm)
        elif layers == 101:
            self.model_sed = ResNet(Bottleneck, [3, 4, 23, 3], num_classes, BatchNorm=BatchNorm)
        elif layers == 152:
            self.model_sed = ResNet(Bottleneck, [3, 8, 36, 3], num_classes, BatchNorm=BatchNorm)
        elif layers == 1850:
            self.model_sed = ResNet(BasicBlock, [3, 2, 2, 2], num_classes, BatchNorm=BatchNorm)
        else:
            print('unsupport layer number: {}'.format(layers))
            exit()
        

    def forward(self, x):
        out, x1, x2, x3 = self.model_sed(x)
        
        #edge_map = self.edge_layer(x1, x2, out)
        #edge_out = torch.sigmoid(edge_map)
        
        sed = self.conv5a(out)
  
        pred4, P = self.PAM_Module(sed)
      
        pred5 = self.conv51(pred4)
        pred = self.conv6(pred5)

        return pred, P, sed, pred4
    

def Res_Deeplab(num_classes=21, layers=18, shrink_factor=2):
    difnet = DifNet(num_classes, layers, SynchronizedBatchNorm2d, shrink_factor)
    return difnet


def Res_Deeplab2(num_classes=21, layers=18):
    difnet = DifNet(num_classes, int(layers.replace('+','')))
    return difnet