"""create MobileNet V1 and V2"""


import torch.nn as nn
import torch
# from torchsummary import summary


class SeperableConv(nn.Module):

    def __init__(self, in_channel, out_channel, s):
        super(SeperableConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=s, padding=1, groups=in_channel)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.pointwise_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pointwise_conv(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, s, expansion_factor):
        super(Bottleneck, self).__init__()
        self.stride = s
        if self.stride == 1:
            self.short_conv = nn.Conv2d(in_channel, out_channel, 1)
        middle_channel = int(in_channel * expansion_factor)
        self.full_conv = nn.Conv2d(in_channel, middle_channel, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(middle_channel)
        self.relu1 = nn.ReLU6()
        self.depth_conv = nn.Conv2d(middle_channel, middle_channel, kernel_size=3,
                                    stride=s, padding=1, groups=middle_channel)
        self.bn2 = nn.BatchNorm2d(middle_channel)
        self.relu2 = nn.ReLU6()
        self.linear_conv = nn.Conv2d(middle_channel, out_channel, kernel_size=1)

    def forward(self, input):
        x = self.full_conv(input)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.depth_conv(x)
        x = self.bn2(x)
        x = self.relu2(x)
        out = self.linear_conv(x)
        if self.stride == 1:
            out = out + self.short_conv(input)
        return out


def make_layer(layer_list, name):
    """just for the same resolution"""
    layers = []
    if name == "V1":
        in_channel, out_channel, s = layer_list[1], layer_list[2], layer_list[3]
        for i in range(layer_list[0]):
            layers += [SeperableConv(in_channel, out_channel, s)]
            if i == 0:
                in_channel *= 2
                s = 1
    if name == "V2":
        in_channel, out_channel, s, expansion_factor = layer_list[1], layer_list[2], layer_list[3], layer_list[4]
        for i in range(layer_list[0]):
            layers += [Bottleneck(in_channel, out_channel, s, expansion_factor)]
            if i == 0:
                in_channel = out_channel
                s = 1
    return nn.Sequential(*layers)


class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.layer1 = make_layer([1, 32, 64, 1], "V1")
        self.layer2 = make_layer([2, 64, 128, 2], "V1")
        self.layer3 = make_layer([2, 128, 256, 2], "V1")
        self.layer4 = make_layer([6, 256, 512, 2], "V1")
        self.layer5 = make_layer([2, 512, 1024, 2], "V1")
        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(in_features=1024, out_features=1000)
        self.out = nn.Softmax(0)

    def forward(self, input):
        x = self.conv1(input)  # s = 2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)  # s = 4
        feature_at_8 = self.layer3(x)  # s = 8
        feature_at_16 = self.layer4(feature_at_8)  # s = 16
        feature_at_32 = self.layer5(feature_at_16)  # s = 32
        x = self.avg_pool(feature_at_32)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        out = self.out(x)
        return out


class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU6()

        self.layer1 = make_layer([1, 32, 16, 1, 1], "V2")
        self.layer2 = make_layer([2, 16, 24, 2, 6], "V2")
        self.layer3 = make_layer([3, 24, 32, 2, 6], "V2")
        self.layer4 = make_layer([4, 32, 64, 2, 6], "V2")
        self.layer5 = make_layer([3, 64, 96, 1, 6], "V2")
        # modify for deeplab_v3p
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=1024, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(1024)
        self.relu2 = nn.ReLU6()
        # self.layer6 = make_layer([3, 96, 160, 2, 6], "V2")
        # self.layer7 = make_layer([1, 160, 320, 1, 6], "V2")

        # self.conv2 = nn.Conv2d(in_channels=320, out_channels=1280, kernel_size=1)
        # self.bn2 = nn.BatchNorm2d(1280)
        # self.relu2 = nn.ReLU6()

    def forward(self, input):
        x = self.conv1(input)  # s = 2
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        feature_4 = self.layer2(x)  # s = 4
        feature_8 = self.layer3(feature_4)  # s = 8
        x = self.layer4(feature_8)  # s = 16
        x = self.layer5(x)
        x = self.conv2(x)
        x = self.bn2(x)
        feature_16 = self.relu2(x)
        # x = self.layer6(feature_at_16)
        # x = self.layer7(x)
        # x = self.conv2(x)
        # x = self.bn2(x)
        # feature_at_32 = self.relu2(x)
        return feature_4, feature_16


if __name__ == '__main__':
    x = torch.randn([1, 3, 224, 224])
    v2 = MobileNetV2()
    print(summary(v2, (3, 224, 224)))
    print(v2(x)[1].size())











