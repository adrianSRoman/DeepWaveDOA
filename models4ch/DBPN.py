import os
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import *

import torch
import math

class DenseBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation='relu', norm='batch'):
        super(DenseBlock, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm1d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm1d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.fc(x))
        else:
            out = self.fc(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias, dtype=torch.double)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU(dtype=torch.double)
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias, dtype=torch.double)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU(dtype=torch.double)
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm='batch'):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(num_filter)
        elif norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(num_filter)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU(dtype=torch.double)
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()


    def forward(self, x):
        residual = x
        if self.norm is not None:
            out = self.bn(self.conv1(x))
        else:
            out = self.conv1(x)

        if self.activation is not None:
            out = self.act(out)

        if self.norm is not None:
            out = self.bn(self.conv2(out))
        else:
            out = self.conv2(out)

        out = torch.add(out, residual)
        return out

class UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(UpBlock, self).__init__()
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)        

    def forward(self, x):
    	h0 = self.up_conv1(x)
    	l0 = self.up_conv2(h0)
    	h1 = self.up_conv3(l0 - x)
    	return h1 + h0

class UpBlockPix(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, scale=4, bias=True, activation='prelu', norm=None):
        super(UpBlockPix, self).__init__()
        self.up_conv1 = Upsampler(scale,num_filter)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = Upsampler(scale,num_filter)        

    def forward(self, x):
    	h0 = self.up_conv1(x)
    	l0 = self.up_conv2(h0)
    	h1 = self.up_conv3(l0 - x)
    	return h1 + h0
        
class D_UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu', norm=None):
        super(D_UpBlock, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)        

    def forward(self, x):
    	x = self.conv(x)
    	h0 = self.up_conv1(x)
    	l0 = self.up_conv2(h0)
    	h1 = self.up_conv3(l0 - x)
    	return h1 + h0

class D_UpBlockPix(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, scale=4, bias=True, activation='prelu', norm=None):
        super(D_UpBlockPix, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.up_conv1 = Upsampler(scale,num_filter)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = Upsampler(scale,num_filter)

    def forward(self, x):
    	x = self.conv(x)
    	h0 = self.up_conv1(x)
    	l0 = self.up_conv2(h0)
    	h1 = self.up_conv3(l0 - x)
    	return h1 + h0

class DownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(DownBlock, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
    	l0 = self.down_conv1(x)
    	h0 = self.down_conv2(l0)
    	l1 = self.down_conv3(h0 - x)
    	return l1 + l0

class DownBlockPix(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, scale=4,bias=True, activation='prelu', norm=None):
        super(DownBlockPix, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv2 = Upsampler(scale,num_filter)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
    	l0 = self.down_conv1(x)
    	h0 = self.down_conv2(l0)
    	l1 = self.down_conv3(h0 - x)
    	return l1 + l0

class D_DownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu', norm=None):
        super(D_DownBlock, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
    	x = self.conv(x)
    	l0 = self.down_conv1(x)
    	h0 = self.down_conv2(l0)
    	l1 = self.down_conv3(h0 - x)
    	return l1 + l0

class D_DownBlockPix(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, scale=4, bias=True, activation='prelu', norm=None):
        super(D_DownBlockPix, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv2 = Upsampler(scale,num_filter)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
    	x = self.conv(x)
    	l0 = self.down_conv1(x)
    	h0 = self.down_conv2(l0)
    	l1 = self.down_conv3(h0 - x)
    	return l1 + l0

class PSBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, scale_factor, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm='batch'):
        super(PSBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size * scale_factor**2, kernel_size, stride, padding, bias=bias)
        self.ps = torch.nn.PixelShuffle(scale_factor)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.ps(self.conv(x)))
        else:
            out = self.ps(self.conv(x))

        if self.activation is not None:
            out = self.act(out)
        return out


class Upsampler(torch.nn.Module):
    def __init__(self, scale, n_feat, bn=False, act='prelu', bias=True):
        super(Upsampler, self).__init__()
        modules = []
        for _ in range(int(math.log(scale, 2))):
            modules.append(ConvBlock(n_feat, 4 * n_feat, 3, 1, 1, bias, activation=None, norm=None))
            modules.append(torch.nn.PixelShuffle(2))
            if bn: modules.append(torch.nn.BatchNorm2d(n_feat))
            #modules.append(torch.nn.PReLU())
        self.up = torch.nn.Sequential(*modules)
        
        self.activation = act
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.up(x)
        if self.activation is not None:
            out = self.act(out)
        return out
             

class Upsample2xBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True, upsample='deconv', activation='relu', norm='batch'):
        super(Upsample2xBlock, self).__init__()
        scale_factor = 2
        # 1. Deconvolution (Transposed convolution)
        if upsample == 'deconv':
            self.upsample = DeconvBlock(input_size, output_size,
                                        kernel_size=4, stride=2, padding=1,
                                        bias=bias, activation=activation, norm=norm)

        # 2. Sub-pixel convolution (Pixel shuffler)
        elif upsample == 'ps':
            self.upsample = PSBlock(input_size, output_size, scale_factor=scale_factor,
                                    bias=bias, activation=activation, norm=norm)

        # 3. Resize and Convolution
        elif upsample == 'rnc':
            self.upsample = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=scale_factor, mode='nearest'),
                ConvBlock(input_size, output_size,
                          kernel_size=3, stride=1, padding=1,
                          bias=bias, activation=activation, norm=norm)
            )

    def forward(self, x):
        out = self.upsample(x)
        return out

class Net(nn.Module):
    def __init__(self, num_channels, base_filter, feat, num_stages, scale_factor):
        super(Net, self).__init__()
        
        if scale_factor == 2:
        	kernel = 6
        	stride = 2
        	padding = 2
        elif scale_factor == 4:
        	kernel = 8
        	stride = 4
        	padding = 2
        elif scale_factor == 8:
        	kernel = 12
        	stride = 8
        	padding = 2
        
        #Initial Feature Extraction
        self.feat0_rel = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu', norm=None)
        self.feat0_imag = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu', norm=None)
        self.feat1_rel = ConvBlock(feat, base_filter, 1, 1, 0, activation='prelu', norm=None)
        self.feat1_imag = ConvBlock(feat, base_filter, 1, 1, 0, activation='prelu', norm=None)
        #Back-projection stages
        self.up1_rel = UpBlock(base_filter, kernel, stride, padding)
        self.up1_imag = UpBlock(base_filter, kernel, stride, padding)
        self.down1_rel = DownBlock(base_filter, kernel, stride, padding)
        self.down1_imag = DownBlock(base_filter, kernel, stride, padding)
        self.up2_rel = UpBlock(base_filter, kernel, stride, padding)
        self.up2_imag = UpBlock(base_filter, kernel, stride, padding)
        #Reconstruction
        self.output_conv_rel = ConvBlock(num_stages*base_filter // 5, num_channels, 3, 1, 1, activation=None, norm=None)
        self.output_conv_imag = ConvBlock(num_stages*base_filter // 5, num_channels, 3, 1, 1, activation=None, norm=None)
        
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight).double()
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.data = m.bias.data.double()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight).double()
                if m.bias is not None:
                    m.bias.data = m.bias.data.double() 
            
    def forward(self, x_rel, x_imag):
        #real
        x_rel = self.feat0_rel(x_rel)
        x_rel = self.feat1_rel(x_rel)
        #imag
        x_imag = self.feat0_imag(x_imag)
        x_imag = self.feat1_imag(x_imag)
        #real
        h1_rel = self.up1_rel(x_rel)
        h2_rel = self.up2_rel(self.down1_rel(h1_rel))
        #imag
        h1_imag = self.up1_imag(x_imag)
        h2_imag = self.up2_imag(self.down1_imag(h1_imag))
        #real
        x_rel = self.output_conv_rel(torch.cat((h2_rel, h1_rel),1))
        #imag
        x_imag = self.output_conv_imag(torch.cat((h2_imag, h1_imag),1))
        
        return torch.complex(x_rel, x_imag)
