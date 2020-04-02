"""this Xception implemention is a modify version, we modify the original version as backbone of DeeplabV3+"""


import torch
from torch import nn


class SeparableConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        """make Separable Convlution module, we don't use non_linearity between depth convlution and point-wise convolution"""
        super(SeparableConv, self).__init__()
        self.depthconv = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, groups=in_channel, bias=False)
        self.pointwiseconv = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthconv(x)
        x = self.pointwiseconv(x)
        return x


class Block(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, stride):
        super(Block, self).__init__()
        self.shortcut_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)
        self.shortcut_bn = nn.BatchNorm2d(out_channel)
        self.separableconv1 = SeparableConv(in_channel, mid_channel)
        self.bn1 = nn.BatchNorm2d(mid_channel)
        self.separableconv2 = SeparableConv(mid_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=stride, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut_conv(x)
        shortcut = self.shortcut_bn(shortcut)
        shortcut = self.relu(shortcut)

        x = self.separableconv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.separableconv2(x)
        feature_to_save = x
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = x + shortcut

        return x, feature_to_save


class Entry_flow(nn.Module):
    def __init__(self):
        super(Entry_flow, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, 128, 2)
        self.block2 = Block(128, 256, 256, 2)
        self.block3 = Block(256, 728, 728, 2)

        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x, _ = self.block1(x)
        x, feature_4 = self.block2(x)   
        x, _ = self.block3(x)

        return x, feature_4


class Middle_flow(nn.Module):
    def __init__(self):
        super(Middle_flow, self).__init__()
        self.block = nn.ModuleList()
        for i in range(16):
            self.block.append(nn.Sequential(SeparableConv(728, 728),
                                            nn.BatchNorm2d(728),
                                            nn.ReLU6(inplace=True)))

    def forward(self, x):
        for i in range(16):
            shortcut = x
            x = self.block[i](x)
            x = x + shortcut
        return x


class Exit_flow(nn.Module):
    def __init__(self):
        super(Exit_flow, self).__init__()
        self.separableconv1 = SeparableConv(728, 728)
        self.bn1 = nn.BatchNorm2d(728)
        self.separableconv2 = SeparableConv(728, 1024)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):

        x = self.separableconv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.separableconv2(x)
        x = self.relu(x)

        return x


class Xception(nn.Module):

    def __init__(self):
        super(Xception, self).__init__()
        self.entry_flow = Entry_flow()
        self.middle_flow = Middle_flow()
        self.exit_flow = Exit_flow()

    def forward(self, img):
        x, feature_4 = self.entry_flow(img)
        x = self.middle_flow(x)
        feature_16 = self.exit_flow(x)
        return feature_4, feature_16

