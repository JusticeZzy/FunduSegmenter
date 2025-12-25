"""
MIT License

Copyright (c) 2025 Zhenyi Zhao
Copyright (c) University of Dundee
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.utils.cbam import CBAM


class Conv(nn.Module):
    def __init__(self, input_channel=3, output_channel=64, norm='bn', bias=True):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, 3, 1, 1, bias=bias)
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(output_channel)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(num_groups=(output_channel//8), num_channels=output_channel)
        else:
            raise ValueError('Please select correct norm, bn or gn')
        self.relu = nn.ReLU()
        self.conv_block = nn.Sequential(
            self.conv,
            self.norm,
            self.relu,
        )

    def forward(self, x):
        return self.conv_block(x)


class PreAdapter(nn.Module):
    def __init__(self, input_channel=3, mid_channel=64, norm='bn'):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(1, 3, 1, 1)*0.1)

        self.conv1 = Conv(input_channel, mid_channel, norm)
        self.conv2 = Conv(mid_channel, mid_channel, norm)
               
        self.conv3 = nn.Conv2d(mid_channel, input_channel, 1, bias=True)    
        self.relu = nn.ReLU()

    def forward(self,x):
        x1 = F.interpolate(x, size=(224, 224), mode='bicubic')
        
        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)
        x = self.relu(x)
        
        x = F.interpolate(x, size=(224, 224), mode='bicubic')
        
        x = self.weight*x + x1        
        return x


class PostAdapter(nn.Module):
    def __init__(self, input_channel=3, mid_channel=64, output_channel=3, norm='bn'):
        super().__init__()

        self.proj1 = nn.Conv2d(1024, mid_channel, 1, bias=True)
        self.proj2 = nn.Conv2d(1024, mid_channel, 1, bias=True)
        self.proj3 = nn.Conv2d(1024, mid_channel, 1, bias=True)
        self.proj4 = nn.Conv2d(1024, mid_channel, 1, bias=True)
        
        self.convx6 = Conv(mid_channel, mid_channel, norm)        
        self.convx12 = Conv(mid_channel, mid_channel, norm)        
        self.convx18 = Conv(mid_channel, mid_channel, norm)       
        self.convx24 = Conv(mid_channel, mid_channel, norm)

        self.cbam1 = CBAM(mid_channel, 4)
        self.cbam2 = CBAM(mid_channel, 4)
        self.cbam3 = CBAM(mid_channel, 4)
        self.cbam4 = CBAM(mid_channel, 4)

        self.convcat1 = Conv(mid_channel*2, mid_channel, norm)
        self.convcat2 = Conv(mid_channel*2, mid_channel, norm)
        self.convcat3 = Conv(mid_channel*2, mid_channel, norm)
        self.convcat4 = Conv(mid_channel*2, mid_channel, norm)

        self.conv1 = Conv(input_channel, mid_channel, norm)
        self.conv2 = Conv(mid_channel, mid_channel, norm)
        self.conv3 = Conv(mid_channel, mid_channel, norm)
        self.conv4 = Conv(mid_channel, mid_channel, norm)
        self.conv5 = Conv(mid_channel, mid_channel, norm)

        self.outconv = nn.Conv2d(mid_channel, output_channel, 1, bias=True)

    def forward(self, x, mid_features, ori_size):     
        x6, x12, x18, x24 = mid_features[0], mid_features[1], mid_features[2], mid_features[3]

        b, hw, c = x6.shape
        h = w = int(np.sqrt(hw))

        x6 = x6.transpose(1, 2).reshape(b, c, h, w)
        x12 = x12.transpose(1, 2).reshape(b, c, h, w)
        x18 = x18.transpose(1, 2).reshape(b, c, h, w)
        x24 = x24.transpose(1, 2).reshape(b, c, h, w)

        x6 = self.proj1(x6)
        x12 = self.proj2(x12)
        x18 = self.proj3(x18)
        x24 = self.proj4(x24)

        x6 = F.interpolate(x6, scale_factor=16, mode='bicubic')
        x6 = self.convx6(x6)
        
        x12 = F.interpolate(x12, scale_factor=8, mode='bicubic')
        x12 = self.convx12(x12)
               
        x18 = F.interpolate(x18, scale_factor=4, mode='bicubic')
        x18 = self.convx18(x18)
                   
        x24 = F.interpolate(x24, scale_factor=2, mode='bicubic')
        x24 = self.convx24(x24)

        x6 = self.cbam1(x6)
        x12 = self.cbam2(x12)
        x18 = self.cbam3(x18)
        x24 = self.cbam4(x24)
        
        masks = self.conv1(x)
        masks = F.interpolate(masks, scale_factor=2, mode='bicubic')
                
        masks = self.convcat1(torch.cat([masks, x24], dim=1))
        masks = self.conv2(masks)
        masks = F.interpolate(masks, scale_factor=2, mode='bicubic')
        
        masks = self.convcat2(torch.cat([masks, x18], dim=1))
        masks = self.conv3(masks)
        masks = F.interpolate(masks, scale_factor=2, mode='bicubic')
        
        masks = self.convcat3(torch.cat([masks, x12], dim=1))
        masks = self.conv4(masks)
        masks = F.interpolate(masks, scale_factor=2, mode='bicubic')
        
        masks = self.convcat4(torch.cat([masks, x6], dim=1))
        masks = self.conv5(masks)  
       
        masks = self.outconv(masks)
        masks = F.interpolate(masks, size=ori_size, mode='bicubic')
        return masks


class VitAdapter(nn.Module):
    def __init__(self, in_ch, mid_ch):
        super().__init__()
        self.linear1 = nn.Linear(in_ch, mid_ch)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(mid_ch, in_ch)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x