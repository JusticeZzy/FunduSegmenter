"""
Adapted from Mookiah 'M R K, Hogg S, MacGillivray T, et al. On the quantitative effects of compression of retinal fundus images on morphometric vascular measurements in VAMPIRE[J]. Computer methods and programs in biomedicine, 2021, 202: 105969.'

Implement by Zhenyi Zhao @Vampire,Computing,University of Dundee
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch=3, out_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=4, dilation=4)
        self.bn = nn.BatchNorm2d(out_ch)        
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.3)
        self.dropout = torch.nn.Dropout(p=0.2)        
        self.conv_block = nn.Sequential(
            self.conv1,
            self.bn,
            self.leakyrelu,
            self.dropout,
            self.conv2,
            self.leakyrelu,
            self.conv3,
            self.leakyrelu,            
        )

    def forward(self, x):
        return self.conv_block(x)


class BottleConvBlock(nn.Module):
    def __init__(self, in_ch=64, out_ch=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=4, dilation=4)
        self.bn1 = nn.BatchNorm2d(out_ch)  
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.3)
        self.dropout = torch.nn.Dropout(p=0.2)        
        self.conv_block = nn.Sequential(
            self.conv1,
            self.bn1,
            self.leakyrelu,
            self.dropout,
            self.conv2,
            self.bn2,
            self.leakyrelu,
            self.conv3,
            self.bn3,
            self.leakyrelu,            
        )

    def forward(self, x):
        return self.conv_block(x)


class UpConvBlock(nn.Module):
    def __init__(self, in_ch=192, out_ch=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, dilation=1)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.3)
        self.dropout = torch.nn.Dropout(p=0.2)        
        self.conv_block = nn.Sequential(
            self.conv1,
            self.leakyrelu,
            self.dropout,
            self.conv2,
            self.leakyrelu,
        )

    def forward(self, x):
        return self.conv_block(x)


class DUNet(nn.Module):
    def __init__(self, input_channel, output_channel, base_channel):
        super().__init__()
        self.conv1 = ConvBlock(input_channel, base_channel)
        self.conv2 = ConvBlock(base_channel, base_channel*2)
        self.conv3 = BottleConvBlock(base_channel*2, base_channel*4)
        self.conv4 = UpConvBlock(base_channel*6, base_channel*2)
        self.conv5 = UpConvBlock(base_channel*3, base_channel)
        self.outconv = nn.Conv2d(in_channels=base_channel, out_channels=output_channel, kernel_size=1, stride=1, padding=0, dilation=1)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.3)

    def forward(self, x):
        x1 = self.conv1(x)
        x = F.max_pool2d(x1, 2, 2)

        x2 = self.conv2(x)
        x = F.max_pool2d(x2, 2, 2)

        x = self.conv3(x)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat((x2, x), dim=1)
        x = self.conv4(x)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat((x1, x), dim=1)
        x = self.conv5(x)

        x = self.outconv(x)
        x = self.leakyrelu(x)

        return x      
