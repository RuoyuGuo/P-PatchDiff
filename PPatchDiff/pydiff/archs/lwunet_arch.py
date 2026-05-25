import functools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim//2, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim*2, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)
    

@ARCH_REGISTRY.register()
class simpleUNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=3):
        super(simpleUNet, self).__init__()

        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, 32, kernel_size=3, stride=1, padding=1),
                                   nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.pool1 = Downsample(32)

        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                  nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.pool2 = Downsample(64)

        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.pool3 = Downsample(128)

        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                  nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.pool4 = nn.Conv2d(256, 256, 3, 2, 1)

        self.conv5 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                   nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True))
        
        self.upv6 = Upsample(256)
        self.conv6_1 = nn.Conv2d(128+256, 128, 1)
        self.conv6 = nn.Sequential( nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.upv7 = Upsample(128)
        self.conv7_1 = nn.Conv2d(64+128, 128, 1)
        self.conv7 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
                                     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.upv8 = Upsample(128)
        self.conv8_1 = nn.Conv2d(128, 64, 1)
        self.conv8 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.upv9 = Upsample(64)
        self.conv9_1 = nn.Conv2d(64, 32, 1)
        self.conv9 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                                     nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.conv10_1 = nn.Conv2d(32, out_channel, kernel_size=1, stride=1)
        
    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1) #+ pool1
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2) #+ pool2
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3) #+ pool3
        pool4 = self.pool4(conv4)

        conv5 = self.conv5(pool4)

        up6 = self.upv6(conv5)
        up6 = self.conv6_1(torch.cat([up6, conv4], 1))
        conv6 = self.conv6(up6)# + up6

        up7 = self.upv7(conv6)
        up7 = self.conv7_1(torch.cat([up7, conv3], 1))
        conv7 = self.conv7(up7) #+ up7

        up8 = self.upv8(conv7)
        up8 = self.conv8_1(torch.cat([up8, conv2], 1))
        conv8 = self.conv8(up8) #+ up8

        up9 = self.upv9(conv8)
        up9 = self.conv9_1(torch.cat([up9, conv1], 1))
        conv9 = self.conv9(up9) #+ up9

        conv10 = self.conv10_1(conv9)
        # out = nn.functional.pixel_shuffle(conv10, 2)
        out = conv10
        return out


model = simpleUNet()
