###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import upsample
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.scatter_gather import scatter

from Models import resnet

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class BaseNet(nn.Module):
    def __init__(self, n_classes, backbone='resnet50', dilated=True, norm_layer=None, multi_grid=False, multi_dilation=None):
        super(BaseNet, self).__init__()
        self.n_classes = n_classes
        # copying modules from pretrained models
        if backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True, dilated=dilated,
                                              norm_layer=norm_layer,
                                              multi_grid=multi_grid, multi_dilation=multi_dilation)
            self.backbone2 = resnet.resnet50(pretrained=True, dilated=dilated,
                                              norm_layer=norm_layer,
                                              multi_grid=multi_grid, multi_dilation=multi_dilation)
        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True, dilated=dilated,
                                               norm_layer=norm_layer,
                                               multi_grid=multi_grid,multi_dilation=multi_dilation)
        elif backbone == 'resnet152':
            self.backbone = resnet.resnet152(pretrained=True, dilated=dilated,
                                               norm_layer=norm_layer,
                                               multi_grid=multi_grid, multi_dilation=multi_dilation)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        # bilinear upsample options
        self._up_kwargs = up_kwargs

    def base_forward(self, rgb, ir):
        rgb = self.backbone.conv1(rgb)
        rgb = self.backbone.bn1(rgb)
        rgb = self.backbone.relu(rgb)
        rgb = self.backbone.maxpool(rgb)
        rgb1 = self.backbone.layer1(rgb)
        rgb2 = self.backbone.layer2(rgb1)
        rgb3 = self.backbone.layer3(rgb2)
        rgb4 = self.backbone.layer4(rgb3)
        ir = self.backbone2.conv1(ir)
        ir = self.backbone2.bn1(ir)
        ir = self.backbone2.relu(ir)
        ir = self.backbone2.maxpool(ir)
        ir1 = self.backbone2.layer1(ir)
        ir2 = self.backbone2.layer2(ir1)
        ir3 = self.backbone2.layer3(ir2)
        ir4 = self.backbone2.layer4(ir3)
        # rgb = self.backbone.conv1(rgb)
        # rgb = self.backbone.bn1(rgb)
        # rgb = self.backbone.relu1(rgb)
        # rgb = self.backbone.conv2(rgb)
        # rgb = self.backbone.bn2(rgb)
        # rgb = self.backbone.relu2(rgb)
        # rgb = self.backbone.conv3(rgb)
        # rgb = self.backbone.bn3(rgb)
        # rgb = self.backbone.relu3(rgb)
        # rgb = self.backbone.maxpool(rgb)
        # rgb1 = self.backbone.layer1(rgb)
        # rgb2 = self.backbone.layer2(rgb1)
        # rgb3 = self.backbone.layer3(rgb2)
        # rgb4 = self.backbone.layer4(rgb3)
        # ir = self.backbone.conv1(ir)
        # ir = self.backbone.bn1(ir)
        # ir = self.backbone.relu1(ir)
        # ir = self.backbone.conv2(ir)
        # ir = self.backbone.bn2(ir)
        # ir = self.backbone.relu2(ir)
        # ir = self.backbone.conv3(ir)
        # ir = self.backbone.bn3(ir)
        # ir = self.backbone.relu3(ir)
        # ir = self.backbone.maxpool(ir)
        # ir1 = self.backbone.layer1(ir)
        # ir2 = self.backbone.layer2(ir1)
        # ir3 = self.backbone.layer3(ir2)
        # ir4 = self.backbone.layer4(ir3)
        # return ir1, ir1, ir1, ir1, ir1, ir2, ir3, ir4
        # return rgb1, rgb2, rgb3, rgb4, rgb4, rgb4, rgb4, rgb4
        return rgb1, rgb2, rgb3, rgb4, ir1, ir2, ir3, ir4