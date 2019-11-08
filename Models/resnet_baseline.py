import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import vis
import seaborn as sns
from .base import BaseNet


class ASPPModule(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """

    def __init__(self, features, inner_features=256, out_features=512, dilations=[6, 12, 18]):
        super(ASPPModule, self).__init__()

        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                   nn.Conv2d(features, inner_features, kernel_size=1, padding=0,
                                             bias=False),
                                   nn.BatchNorm2d(inner_features),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(inner_features),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
            nn.BatchNorm2d(inner_features),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            nn.BatchNorm2d(inner_features),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            nn.BatchNorm2d(inner_features),
            nn.ReLU(inplace=True))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inner_features * 5, out_features, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        _, _, h, w = x.size()

        feat1 = F.interpolate(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)

        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)

        bottle = self.bottleneck(out)
        return bottle

class ResNetBaseLine(BaseNet):
    def __init__(self, n_classes, backbone='resnet50', norm_layer=None, **kwargs):
        super(ResNetBaseLine, self).__init__(n_classes, backbone=backbone, norm_layer=norm_layer, **kwargs)
        # self.conv11 = nn.Sequential(
        #     nn.Conv2d(2048, 512, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True)
        # )
        # self.conv21 = nn.Sequential(
        #     nn.Conv2d(2048, 256, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True)
        # )
        self.head = nn.Sequential(ASPPModule(2048, out_features=256),
                                  # nn.Conv2d(256, n_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.score1 = nn.Sequential(
            # nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            nn.Conv2d(256, n_classes, kernel_size=1, bias=False)
        )

    def forward(self, rgb, ir):
        _, _, h, w = rgb.size()
        rgb1, rgb2, rgb3, rgb4, ir1, ir2, ir3, ir4 = self.base_forward(rgb, ir)

        # out1 = self.conv11(rgb4)
        # out2 = self.conv21(ir4)

        out = rgb4 + ir4

        out = self.head(out)
        out = self.score1(out)
        out = F.interpolate(out, size=(h, w), **self._up_kwargs)

        output = [out]

        return output