import torch
from torch import nn


# class SeparableConv(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(SeparableConv, self).__init__()
#         self.depthconv = nn.Conv2d(in_channel, in_channel, 3, groups=in_channel, padding=1)
#         self.pointwiseconv = nn.Conv2d(in_channel, out_channel, 1)
#
#     def forward(self, x):
#         x = self.depthconv(x)
#         x = self.pointwiseconv(x)
#         return x
#
#
# class Block(nn.Module):
#     def __init__(self, in_channel, mid_channel, out_channel, stride):
#         super(Block, self).__init__()
#         self.shortcut_conv = nn.Conv2d(in_channel, out_channel, 1, stride=stride)
#         self.shortcut_bn = nn.BatchNorm2d(out_channel)
#         self.separableconv1 = SeparableConv(in_channel, mid_channel)
#         self.bn1 = nn.BatchNorm2d(mid_channel)
#         self.separableconv2 = SeparableConv(mid_channel, out_channel)
#         self.bn2 = nn.BatchNorm2d(out_channel)
#
#         self.relu = nn.ReLU()
#         self.max_pool = nn.MaxPool2d(3, stride, padding=1)
#
#     def forward(self, x):
#         shortcut = self.shortcut_conv(x)
#         shortcut = self.shortcut_bn(shortcut)
#
#         x = self.relu(x)
#         x = self.separableconv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.separableconv2(x)
#         feature_to_save = self.bn2(x)
#         x = self.max_pool(feature_to_save)
#         x = x + shortcut
#
#         return x, feature_to_save
#
#
# class Entry_flow(nn.Module):
#     def __init__(self):
#         super(Entry_flow, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
#         self.bn1 = nn.BatchNorm2d(num_features=32)
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#
#         self.shortcut1 = nn.Conv2d(64, 128, 1, 2)
#         self.short_bn1 = nn.BatchNorm2d(128)
#         self.separable_conv3 = SeparableConv(64, 128)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.separable_conv4 = SeparableConv(128, 128)
#         self.bn4 = nn.BatchNorm2d(128)
#
#         self.block1 = Block(128, 256, 256, 1)
#         self.block2 = Block(256, 256, 256, 2)
#         self.block3 = Block(256, 728, 728, 1)
#         self.block4 = Block(728, 728, 728, 2)
#
#         self.relu = nn.ReLU()
#         self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#
#         shortcut = self.shortcut1(x)
#         shortcut = self.short_bn1(shortcut)
#         x = self.separable_conv3(x)
#         x = self.bn3(x)
#         x = self.relu(x)
#         x = self.separable_conv4(x)
#         x = self.bn4(x)
#         x = self.max_pool(x)
#         x = x + shortcut
#
#         x, _ = self.block1(x)
#         x, feature1 = self.block2(x)
#         x, _ = self.block3(x)
#         x, feature2 = self.block4(x)
#
#         return x, feature1, feature2
#
#
# class Middle_flow(nn.Module):
#     def __init__(self):
#         super(Middle_flow, self).__init__()
#         self.relu = nn.ReLU()
#         self.separable_conv = SeparableConv(728, 728)
#         self.bn = nn.BatchNorm2d(728)
#
#     def forward(self, x):
#         for i in range(16):
#             shortcut = x
#             for j in range(3):
#                 x = self.relu(x)
#                 x = self.separable_conv(x)
#                 x = self.bn(x)
#             x = x + shortcut
#         return x
#
#
# class Exit_flow(nn.Module):
#     def __init__(self):
#         super(Exit_flow, self).__init__()
#         self.shortcut = nn.Conv2d(728, 1024, 1, 2)
#         self.short_bn = nn.BatchNorm2d(1024)
#
#         self.separableconv1 = SeparableConv(728, 728)
#         self.bn1 = nn.BatchNorm2d(728)
#         self.separableconv2 = SeparableConv(728,1024)
#         self.bn2 = nn.BatchNorm2d(1024)
#         self.separableconv3 = SeparableConv(1024, 1536)
#         self.bn3 = nn.BatchNorm2d(1536)
#         self.separableconv4 = SeparableConv(1536, 2048)
#         self.bn4 = nn.BatchNorm2d(2048)
#
#         self.relu = nn.ReLU()
#         self.max_pool = nn.MaxPool2d(3, 2, padding=1)
#
#     def forward(self, x):
#         shortcut = self.shortcut(x)
#         shortcut = self.short_bn(shortcut)
#
#         x = self.relu(x)
#         x = self.separableconv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.separableconv2(x)
#         x = self.bn2(x)
#         feature_to_save = x
#         x = self.max_pool(x)
#
#         x = x + shortcut
#         x = self.separableconv3(x)
#         x = self.bn3(x)
#         x = self.relu(x)
#         x = self.separableconv4(x)
#         x = self.bn4(x)
#         x = self.relu(x)
#
#         return x, feature_to_save
#
#
# class Xception(nn.Module):
#
#     def __init__(self):
#         super(Xception, self).__init__()
#         self.entry_flow = Entry_flow()
#         self.middle_flow = Middle_flow()
#         self.exit_flow = Exit_flow()
#
#     def forward(self, img):
#         x, feature_4, feature_8 = self.entry_flow(img)
#         x = self.middle_flow(x)
#         x, feature_16 = self.exit_flow(x)
#         return feature_4, feature_16

class SeparableConv(nn.Module):
    def __init__(self, in_channel, out_channel):
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
        self.separable_conv = SeparableConv(728, 728)
        self.bn = nn.BatchNorm2d(728)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        for i in range(2):
            shortcut = x
            for j in range(3):
                x = self.separable_conv(x)
                x = self.bn(x)
                x = self.relu(x)
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


if __name__ == '__main__':
    x = torch.randn([1, 3, 192, 512])
    model = Xception()
    feature_4, feature_16 = model(x)
    print(feature_16.size(), feature_4.size())


