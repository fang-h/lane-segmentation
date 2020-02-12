"""we use HRNetV2 for lane segmentation, this implementation is adopted in the follow links:
   https://github.com/HRNet/HRNet-Semantic-Segmentation"""


import os
import logging


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


BN_MOMENTUM = 0.1  # nn.BatchNorm2d's default
logger = logging.getLogger(__name__)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, input):
        residual = input

        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(input)

        x += residual
        out = self.relu(x)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(channel, int(channel * self.expansion), kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(int(channel * self.expansion), momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, input):
        residual = input

        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            residual = self.downsample(input)

        x += residual
        out = self.relu(x)

        return out


class HighResolutionModule(nn.Module):

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output):
        """
        :param num_branches: num_branches of stage_x, int
        :param blocks: block type,
        :param num_blocks: number of blocks of every branch to create, list
        :param num_inchannels:   inchannels of every branch, list
        :param num_channels: mid_channels of every branch's block, list
        :param fuse_method: sum or cat, string
        :param multi_scale_output:
        """
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, num_blocks, num_inchannels, num_channels)
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches,  num_blocks, num_inchannels, num_channels):

        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKES({})'.format(num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM)INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branches(self, branch_index, block, num_blocks,
                           num_channels, stride=1):
        """
        :param branch_index: index in all the branches, int
        :param block: block type
        :param num_blocks: number of block to create, int
        :param num_channels: block's mid_channel
        :param stride:
        :return:
        """
        # the first block's residual may need to change the resolution or channels
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM)
            )

        layers = []
        # create first blocks
        layers.append(
            block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample)
        )
        # change the num_inchannels for other blocks of every branch
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        # create other blocks
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(self.num_inchannels[branch_index], num_channels[branch_index])
            )
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        """
        create every branch of stage_x
        :param num_branches: list
        :param block: string
        :param num_blocks: list
        :param num_channels: list
        :return:
        """
        branches = []
        for i in range(num_branches):
            branches.append(
                self._make_one_branches(i, block, num_blocks, num_channels,)
            )
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        """
        fuse every branch's feature map
        :return:
        """
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        # if multi_scale_output is True, this stage is output multi_scale feature
        for i in range(num_branches if self.multi_scale_output else 1):
            # index i is the i-th output feature map
            fuse_layer = []  # save every input(0 to num_branch) to one output(index i) before fuse_method(sum, cat)
            for j in range(num_branches):
                # index j is the j-th input feature map to fuse
                if j > i:
                    # low_resolution feature(branches[j]) to high_resolution feature(branches[i])
                    # just by one Upsample operation to increase resolution
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(
                            num_inchannels[j], num_inchannels[i],
                            kernel_size=1, stride=1, padding=0, bias=False
                        ),
                        nn.BatchNorm2d(self.num_inchannels[i]),
                        nn.Upsample(scale_factor=2 ** (j - i), mode='bilinear')
                    ))
                elif j == i:
                    # same resolution, no operation
                    fuse_layer.append(None)
                else:
                    # high_resolution feature(branch[j]) to low_resolution feature(branch[i])
                    # need (i - j) times conv_stride2 to decrease the resolution
                    conv3x3s = []
                    for k in range(i - j):
                        # index k is the k-th conv_stride2
                        if k == i - j - 1:
                            # last operation of conv_stride2, the output channels is index by i
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3,
                                              kernel_size=3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            # if not last operation of conv_stride2, the output channels is set to index j,
                            # same with the branch[j]'s channel
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3,
                                              kernel_size=3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(inplace=True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):

        # get output of every branch before fuse
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        # create fuse
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            # index i is the i-th output of fuse layer
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]  # the same resolution just by sum
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class SemanticHighResolutionNet(nn.Module):

    def __init__(self, cfg):
        super(SemanticHighResolutionNet, self).__init__()

        # create stem block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)  # stride 2
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)  # stride 2
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)

        # stage1 contains 4 residual unit
        self.layer1 = self._make_layer(Bottleneck, in_channels=64, channels=64, blocks=4)
        stage1_out_channels = 64 * Bottleneck.expansion  #  256

        # stage2
        self.stage2_cfg = cfg['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']  # [32, 64]
        block = blocks_dict[self.stage2_cfg['BLOCK']]  # 'BASIC'
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]  # [32, 64]
        self.transition1 = self._make_transition_layer([stage1_out_channels], num_channels)  # ModuleList
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels
        )

        # stage3
        self.stage3_cfg = cfg['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']  # [32, 64, 128]
        block = blocks_dict[self.stage3_cfg['BLOCK']]  # 'BASIC'
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]  # [32, 64, 128]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        self.stage4_cfg = cfg['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']  # [32, 64, 128, 256]
        block = blocks_dict[self.stage4_cfg['BLOCK']]  # 'BASIC
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]  # # [32, 64, 128, 256]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=True)

        last_input_channels = np.int(np.sum(pre_stage_channels))

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_input_channels,
                out_channels=last_input_channels,
                kernel_size=1,
                stride=1
            ),
            nn.BatchNorm2d(last_input_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=last_input_channels, out_channels=cfg['NUM_CLASS'],
                      kernel_size=cfg['FINAL_CONV_KERNEL'],
                      stride=1,
                      padding=1 if cfg['FINAL_CONV_KERNEL'] == 3 else 0)
        )

    def _make_layer(self, block, in_channels, channels, blocks, stride=1):
        """this function is used to create stage1 which contains 4 residual unit
        block: a module type, such as a standard residual block
        in_channels: number of channels to the first block
        channels:  number of channels at the second layer of block,
        blocks: number of block"""
        downsample = None
        if stride != 1 or \
            in_channels != channels * block.expansion:  # change residual's resolution or channels
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion, momentum=BN_MOMENTUM)
            )
        layers = []
        layers.append(block(in_channels, channels, stride, downsample))
        in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(in_channels, channels))
        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        # stage2:[256], [32, 64]
        #
        num_branches_cur = len(num_channels_cur_layer)  # stage2: 2
        num_branches_pre = len(num_channels_pre_layer)  # stage2: 1
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:  # keep the resolution unchanged， change the channels
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    # if the pre_channels and cur_channels is not equal, change by conv
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i],
                                  kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    # # if the pre_channels and cur_channels is equal, no opeartion
                    transition_layers.append(None)
            else:  # change the resolution and channels
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(inchannels, outchannels,
                                      kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        """
        :param layer_config:
        :param num_inchannels:
        :param multi_scale_output: in semantic, mutil_scale_output is always True
        :return:
        """
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used in last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches, block, num_blocks, num_inchannels,
                    num_channels, fuse_method, reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):

        # stem block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # stage1
        x = self.layer1(x)

        # stage2
        x_list = []  # save the feature map output by transition layer1
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                # channel and resolution have no changed
                x_list.append(x)
        # HighResolutionModule
        y_list = self.stage2(x_list)

        # stage3
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # Upsample other low resolution feature to high resolution feature
        x0_h, x0_w = x[0].size()[2], x[0].size()[3]
        x1 = F.upsample(x[1], size=(x0_h, x0_w), mode='bilinear')
        x2 = F.upsample(x[2], size=(x0_h, x0_w), mode='bilinear')
        x3 = F.upsample(x[3], size=(x0_h, x0_w), mode='bilinear')

        x = torch.cat([x[0], x1, x2, x3], dim=1)

        x = self.last_layer(x)

        # Upsample to input_size
        x = nn.Upsample(scale_factor=4, mode='bilinear')(x)

        return x

