import torch
from torch import nn
import torch.nn.functional as F
from backbone.Xception import Xception
from backbone.MobileNet import MobileNetV2


class ASPP(nn.Module):
    def __init__(self, in_channel, out_channel, atrous_rate):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1),
                                     nn.BatchNorm2d(out_channel),
                                     nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, padding=atrous_rate[0],
                                               dilation=atrous_rate[0]),
                                     nn.BatchNorm2d(out_channel),
                                     nn.ReLU(inplace=True))
        self.branch3 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, padding=atrous_rate[1],
                                               dilation=atrous_rate[1]),
                                     nn.BatchNorm2d(out_channel),
                                     nn.ReLU(inplace=True))
        self.branch4 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, padding=atrous_rate[2],
                                               dilation=atrous_rate[2]),
                                     nn.BatchNorm2d(out_channel),
                                     nn.ReLU(inplace=True))
        self.branch5 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channel, out_channel, 1, ),
                                     nn.BatchNorm2d(out_channel),
                                     nn.ReLU(inplace=True))
        self.conv_cat = nn.Sequential(nn.Conv2d(out_channel * 5, out_channel, 1),
                                      nn.BatchNorm2d(out_channel),
                                      nn.ReLU(inplace=True))

    def forward(self, x):
        _, c, h, w = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = self.branch5(x)
        global_feature = F.interpolate(global_feature, (h, w), None, 'bilinear', True)
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class Encoder(nn.Module):
    def __init__(self, backbone='Xception'):
        super(Encoder, self).__init__()
        if backbone == 'Xception':
            self.backbone = Xception()
        if backbone == 'MobileNetV2':
            self.backbone = MobileNetV2()
        self.aspp = ASPP(1024, 256, [6, 12, 18])

    def forward(self, x):
        feature_4, feature_16 = self.backbone(x)
        feature_aspp = self.aspp(feature_16)
        return feature_4, feature_aspp


class Deeplab_v3p(nn.Module):
    def __init__(self, num_class, backbone='Xception'):
        super(Deeplab_v3p, self).__init__()
        self.num_class = num_class
        self.encoder = Encoder(backbone=backbone)
        self.upsample_assp = nn.Upsample(scale_factor=4, mode='bilinear')
        if backbone == 'Xception':
            self.short_conv = nn.Sequential(nn.Conv2d(256, 256, 1),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True))
        elif backbone == 'MobileNetV2':
            self.short_conv = nn.Sequential(nn.Conv2d(24, 256, 1),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True))
        self.cat_conv = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True))
        self.upsample_cat = nn.Upsample(scale_factor=4, mode='bilinear')
        self.conv_logits = nn.Conv2d(256, self.num_class, 1)

    def forward(self, x):
        low_feature, feature_aspp = self.encoder(x)
        up_assp = self.upsample_assp(feature_aspp)
        low_feature = self.short_conv(low_feature)
        concat_feature = torch.cat([low_feature, up_assp], dim=1)
        concat_feature = self.cat_conv(concat_feature)
        feature = self.upsample_cat(concat_feature)
        logits = self.conv_logits(feature)
        return logits











