import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.modules.utils import _pair, _quadruple
from torch.autograd import Variable
import numpy as np
import math
# from torch_geometric.nn import MessagePassing
# from torch_geometric.nn.pool import knn_graph
import matplotlib.pyplot as plt
from utils import vis
from torch.nn import Parameter
from collections import OrderedDict
import seaborn as sns


class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # HW C
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # C HW
        energy = torch.bmm(proj_query, proj_key)  # HW HW
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # C HW

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # C HW
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class GeometryPrior(torch.nn.Module):
    def __init__(self, k, channels, multiplier=0.5):
        super(GeometryPrior, self).__init__()
        self.channels = channels
        self.k = k
        self.position = 2 * torch.rand(1, 2, 8, 8, requires_grad=True) - 1
        self.l1 = torch.nn.Conv2d(2, int(multiplier * channels), 1)
        self.l2 = torch.nn.Conv2d(int(multiplier * channels), channels, 1)

    def forward(self, x):
        x = self.l1(self.position.cuda())
        x = self.l2(F.relu(x))
        return x.view(1, self.channels, 64)


class NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(NonLocalBlockND, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 8
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            # bn(self.in_channels, momentum=0.1)
        )
        # nn.init.constant_(self.W[1].weight, 0)
        # nn.init.constant_(self.W[1].bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.g = nn.Sequential(max_pool_layer, self.g)
        self.phi = nn.Sequential(max_pool_layer, self.phi)

        self.gamma = Parameter(torch.zeros(1))
        # self.gp = GeometryPrior(53, 128)

    def forward(self, x):
        '''
        :param x: (b, c, h, w)
        :return:
        '''

        batch_size = x.size(0)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)  # b hw c

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # b c hw/4
        theta_phi = torch.matmul(theta_x, phi_x)  # b hw hw/4
        # theta_phi = theta_phi * (1. / self.inter_channels ** 0.5)
        # theta_phi = theta_phi + self.gp(0)
        p = F.softmax(theta_phi, dim=-1)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # b c hw/4

        y = torch.matmul(g_x, p.permute(0, 2, 1))  # b c hw
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = self.gamma * W_y + x

        return z


class NonLocalRGBIR(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(NonLocalRGBIR, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.rgb_g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.rgb_W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels, momentum=0.9)
        )
        nn.init.constant_(self.rgb_W[1].weight, 0)
        nn.init.constant_(self.rgb_W[1].bias, 0)

        self.rgb_theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                 kernel_size=1, stride=1, padding=0)

        self.rgb_phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

        self.rgb_g = nn.Sequential(max_pool_layer, self.rgb_g)
        self.rgb_phi = nn.Sequential(max_pool_layer, self.rgb_phi)

        self.ir_g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                            kernel_size=1, stride=1, padding=0)

        self.ir_W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels, momentum=0.9)
        )
        nn.init.constant_(self.ir_W[1].weight, 0)
        nn.init.constant_(self.ir_W[1].bias, 0)

        self.ir_theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                kernel_size=1, stride=1, padding=0)

        self.ir_phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                              kernel_size=1, stride=1, padding=0)

        self.ir_g = nn.Sequential(max_pool_layer, self.ir_g)
        self.ir_phi = nn.Sequential(max_pool_layer, self.ir_phi)

        self.gp = GeometryPrior(53, 1600)

    def forward(self, rgb, ir):
        '''
        :param x: (b, c, h, w)
        :return:
        '''

        batch_size = rgb.size(0)

        rgb_theta_x = self.rgb_theta(rgb).view(batch_size, self.inter_channels, -1)
        rgb_theta_x = rgb_theta_x.permute(0, 2, 1)  # b hw c

        rgb_phi_x = self.rgb_phi(rgb).view(batch_size, self.inter_channels, -1)  # b c hw/4
        rgb_out = torch.matmul(rgb_theta_x, rgb_phi_x)  # b hw hw/4
        del rgb_theta_x
        del rgb_phi_x
        rgb_out = rgb_out + self.gp(0)
        rgb_out = F.softmax(rgb_out, dim=-1)

        rgb_g = self.rgb_g(rgb).view(batch_size, self.inter_channels, -1)  # b c hw/4

        rgb_out = torch.matmul(rgb_g, rgb_out.permute(0, 2, 1))  # b c hw
        del rgb_g
        rgb_out = rgb_out.view(batch_size, self.inter_channels, *rgb.size()[2:])
        rgb_out = self.rgb_W(rgb_out)
        rgb_out = rgb_out + rgb

        ir_theta_x = self.ir_theta(ir).view(batch_size, self.inter_channels, -1)
        ir_theta_x = ir_theta_x.permute(0, 2, 1)  # b hw c

        ir_phi_x = self.ir_phi(ir).view(batch_size, self.inter_channels, -1)  # b c hw/4
        ir_out = torch.matmul(ir_theta_x, ir_phi_x)  # b hw hw/4
        del ir_theta_x
        del ir_phi_x
        ir_out = ir_out + self.gp(0)
        ir_out = F.softmax(ir_out, dim=-1)

        ir_g = self.ir_g(ir).view(batch_size, self.inter_channels, -1)  # b c hw/4

        ir_out = torch.matmul(ir_g, ir_out.permute(0, 2, 1))  # b c hw
        del ir_g
        ir_out = ir_out.view(batch_size, self.inter_channels, *rgb.size()[2:])
        ir_out = self.ir_W(ir_out)
        ir_out = ir_out + ir

        return rgb_out, ir_out


def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        for p in m.parameters():
            p.requires_grad = False


class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=bias)

    def forward(self, x):
        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        # (n, num_state, num_node) -> (n, num_state, num_node)
        h = self.relu(h)
        h = self.conv2(h)
        return h


class GloRe_Unit(nn.Module):
    def __init__(self, num_in, num_mid, stride=(1, 1), kernel=1):
        super(GloRe_Unit, self).__init__()

        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)

        kernel_size = (kernel, kernel)
        padding = (1, 1) if kernel == 3 else (0, 0)

        # reduce dimension
        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=kernel_size, padding=padding)
        # generate graph transformation function
        self.conv_proj = nn.Conv2d(num_in, self.num_n, kernel_size=kernel_size, padding=padding)
        # ----------
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        # ----------
        # tail: extend dimension
        self.fc_2 = nn.Conv2d(self.num_s, num_in, kernel_size=kernel_size, padding=padding, stride=(1, 1),
                              groups=1, bias=False)

        self.blocker = nn.BatchNorm2d(num_in)

        # self.gp = GeometryPrior(53, 128)

    def forward(self, x):
        '''
        :param x: (n, c, h, w)
        '''
        batch_size = x.size(0)

        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        x_state_reshaped = self.conv_state(x).view(batch_size, self.num_s, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_proj_reshaped = self.conv_proj(x).view(batch_size, self.num_n, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_rproj_reshaped = x_proj_reshaped

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # projection: pixel space -> instance space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
        # x_n_state = x_n_state + self.gp(0)
        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        x_n_rel = self.gcn(x_n_state)
        # x_n_rel = F.softmax(x_n_rel, dim=-1)

        # reverse projection: instance space -> pixel space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # (n, num_state, h*w) --> (n, num_state, h, w)
        x_state = x_state_reshaped.view(batch_size, self.num_s, *x.size()[2:])

        # -----------------
        # final
        out = x + self.blocker(self.fc_2(x_state))

        return out


class GloRe_Fusion(nn.Module):
    def __init__(self, num_in, num_mid, stride=(1, 1), kernel=1):
        super(GloRe_Fusion, self).__init__()

        self.num_s = int(2 * num_mid)
        self.num_n = int(2 * num_mid)

        kernel_size = (kernel, kernel)
        padding = (1, 1) if kernel == 3 else (0, 0)

        # reduce dimension
        self.rgb_conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=kernel_size, padding=padding)
        # generate graph transformation function
        self.rgb_conv_proj = nn.Conv2d(num_in, self.num_n, kernel_size=kernel_size, padding=padding)
        # ----------
        self.rgb_gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        # ----------
        # tail: extend dimension
        self.rgb_fc_2 = nn.Conv2d(self.num_s, num_in, kernel_size=kernel_size, padding=padding, stride=(1, 1),
                                  groups=1, bias=False)

        self.rgb_blocker = nn.BatchNorm2d(num_in)

        self.rgb_gate = nn.Conv2d(num_in, num_in, kernel_size=3, padding=1)

        # reduce dimension
        self.ir_conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=kernel_size, padding=padding)
        # generate graph transformation function
        self.ir_conv_proj = nn.Conv2d(num_in, self.num_n, kernel_size=kernel_size, padding=padding)
        # ----------
        self.ir_gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        # ----------
        # tail: extend dimension
        self.ir_fc_2 = nn.Conv2d(self.num_s, num_in, kernel_size=kernel_size, padding=padding, stride=(1, 1),
                                 groups=1, bias=False)

        self.ir_blocker = nn.BatchNorm2d(num_in)

        self.ir_gate = nn.Conv2d(num_in, num_in, kernel_size=3, padding=1)

    def forward(self, rgb, ir):
        '''
        :param x: (n, c, h, w)
        '''
        batch_size, channel, _, _ = rgb.size()

        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        rgb_state_reshaped = self.rgb_conv_state(rgb).view(batch_size, self.num_s, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        rgb_proj_reshaped = self.rgb_conv_proj(rgb).view(batch_size, self.num_n, -1)

        rgb_rproj_reshaped = rgb_proj_reshaped
        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        ir_state_reshaped = self.ir_conv_state(ir).view(batch_size, self.num_s, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        ir_proj_reshaped = self.ir_conv_proj(ir).view(batch_size, self.num_n, -1)

        ir_rproj_reshaped = ir_proj_reshaped
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # projection: pixel space -> instance space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        rgbir_n_state = torch.matmul(rgb_state_reshaped, ir_proj_reshaped.permute(0, 2, 1))
        rgbir_n_state = rgbir_n_state * (1. / rgbir_n_state.size(2))

        # projection: pixel space -> instance space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        irrgb_n_state = torch.matmul(ir_state_reshaped, rgb_proj_reshaped.permute(0, 2, 1))
        irrgb_n_state = irrgb_n_state * (1. / irrgb_n_state.size(2))

        # # (n, num_state, num_node) -> (n, num_node, num_state)
        # #                          -> (n, num_state, num_node)
        rgbir_n_state = self.rgb_gcn(rgbir_n_state)
        irrgb_n_state = self.ir_gcn(irrgb_n_state)

        # reverse projection: instance space -> pixel space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        rgb_state_reshaped = torch.matmul(rgbir_n_state, rgb_rproj_reshaped)
        ir_state_reshaped = torch.matmul(irrgb_n_state, ir_rproj_reshaped)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # (n, num_state, h*w) --> (n, num_state, h, w)
        rgb_state = rgb_state_reshaped.view(batch_size, self.num_s, *rgb.size()[2:])
        ir_state = ir_state_reshaped.view(batch_size, self.num_s, *rgb.size()[2:])

        # -----------------
        # final
        rgb_gate = F.sigmoid(self.rgb_gate(rgb))
        ir_gate = F.sigmoid(self.ir_gate(ir))

        rgb = rgb + self.rgb_blocker(self.rgb_fc_2(rgb_state))
        ir = ir + self.ir_blocker(self.ir_fc_2(ir_state))

        rgb = (1 + rgb_gate) * rgb
        ir = (1 + ir_gate) * ir
        return rgb, ir


class GCNetHead(nn.Module):
    def __init__(self, in_channels, n, repeat):
        super(GCNetHead, self).__init__()

        self.gcn = nn.Sequential(OrderedDict([("GCN%02d" % i,
                                               GloRe_Unit(in_channels, n, kernel=1)
                                               ) for i in range(repeat)]))

    def forward(self, x):
        gc_feat = self.gcn(x)
        return gc_feat


class EnetGnn(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.rgb_g_layers = nn.Sequential(
            nn.Linear(channels, channels // 2),
            # nn.ReLU(inplace=True),
            # nn.Linear(channels // 4, channels),
            # nn.ReLU(inplace=True)
        )
        self.gamma = Parameter(torch.zeros(1))
        # self.ir_g_layers = nn.Sequential(
        #     nn.Linear(channels * 2, channels),
        #     nn.ReLU(inplace=True),
        # nn.Linear(channels // 4, channels),
        # nn.ReLU(inplace=True)
        # )
        # self.se_rgb = nn.Sequential(
        #     nn.Linear(channels * 2, channels // 16),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(channels // 16, channels),
        #     nn.Sigmoid()
        # nn.ReLU(inplace=True)
        # )
        # self.se_ir = nn.Sequential(
        #     nn.Linear(channels * 2, channels // 16),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(channels // 16, channels),
        #     nn.Sigmoid()
        # nn.ReLU(inplace=True)
        # )
        # self.gamma1 = Parameter(torch.ones(1))
        # self.gamma2 = Parameter(torch.ones(1))
        # self.out_conv = nn.Linear(channels, channels, bias=False)

    # adapted from https://discuss.pytorch.org/t/build-your-own-loss-function-in-pytorch/235/6
    # (x - y)^2 = x^2 - 2*x*y + y^2
    # def get_knn_indices(self, rgb, ir, k):
    #     r = torch.bmm(rgb, ir.permute(0, 2, 1))
    #     N = r.size()[0]
    #     HW = r.size()[1]
    #     batch_indices = torch.zeros((N, HW, k)).cuda()
    #
    #     for idx, val in enumerate(r):
    #         # get the diagonal elements
    #         diag = val.diag().unsqueeze(0)
    #         diag = diag.expand_as(val)
    #         # compute the distance matrix
    #         D = (diag + diag.t() - 2 * val).sqrt()
    #         topk, indices = torch.topk(D, k=k, largest=False)
    #         del D
    #         del diag
    #         del val
    #         batch_indices[idx] = indices.data
    #     return batch_indices

    def get_knn_indices(self, batch_mat, k):
        r = torch.bmm(batch_mat, batch_mat.permute(0, 2, 1))
        N = r.size()[0]
        HW = r.size()[1]
        batch_indices = torch.zeros((N, HW, k)).cuda()

        # for idx, val in enumerate(r):
        #     # get the diagonal elements
        #     diag = val.diag().unsqueeze(0)
        #     diag = diag.expand_as(val)
        #     # compute the distance matrix
        #     D = (diag + diag.t() - 2 * val).sqrt()
        #     topk, indices = torch.topk(D, k=k, largest=False)
        #     del D
        #     del diag
        #     del val
        #     batch_indices[idx] = indices.data
        # return batch_indices

        # r = -2 * r
        # square = torch.sum(batch_mat * batch_mat, dim=-1)
        # square = square.unsqueeze(dim=-1)
        # square_t = square.permute(0, 2, 1)
        # adj = square + r + square_t
        _, indices = torch.topk(r, k=k, largest=False)
        return indices

    def forward(self, cat, rgb_in, gnn_iterations, k):
        # rgb = F.normalize(rgb, dim=1)
        # ir = F.normalize(ir, dim=1)
        # h = rgb + ir

        N = cat.size()[0]
        C = cat.size()[1]
        H = cat.size()[2]
        W = cat.size()[3]
        K = k

        rgb = cat.view(N, C, H * W).permute(0, 2, 1).contiguous()  # N H*W C
        # ir = ir.view(N, C, H*W).permute(0, 2, 1).contiguous()  # N H*W C
        # rgb = rgb.permute(0, 2, 3, 1).view(N*H*W, C).contiguous()  # N*H*W C
        # ir = ir.permute(0, 2, 3, 1).view(N*H*W, C).contiguous()  # N*H*W C

        # get k nearest neighbors
        # a = F.normalize(a, dim=-1)
        # rgb_knn = self.get_knn_indices(F.normalize(rgb, dim=-1), k=k)  # N HW K
        # rgb_knn = rgb_knn.view(N*H*W*K).long()  # NHWK
        # # rgb_knn = rgb_knn.view(N, H*W, K).long()  # NHWK
        # ir_knn = self.get_knn_indices(F.normalize(ir, dim=-1), k=k)  # N HW K
        # ir_knn = ir_knn.view(N*H*W*K).long()  # NHWK
        rgb_knn = self.get_knn_indices(rgb, k=k)  # N HW K
        rgb_knn = rgb_knn.view(N * H * W * K).long()  # NHWK
        # ir_knn = ir_knn.view(N, H*W, K).long()  # NHWK
        # rgb_knn = torch.cat([rgb_knn, ir_knn], dim=1)

        # knn vis
        # a = torch.zeros(H * W)
        # ind = rgb_knn[0, 300, :]
        # for i in ind:
        #     a[i.long()] = 1
        # a = a.view(H, W)
        # plt.subplot(121)
        # # rgb_vis = rgb.view(H, W, 5)
        # # rgb_vis = 0.5 * 50 * a.cpu().numpy() + (1 - 0.5) * rgb_vis[:, :, :3].cpu().numpy()
        # plt.imshow(a)
        # a = torch.zeros(H * W)
        # ind = ir_knn[0, 1500, :]
        # for i in ind:
        #     a[i.long()] = 1
        # a = a.view(H, W)
        # plt.subplot(122)
        # # ir_vis = ir.view(H, W, 5)
        # # ir_vis = 0.5 * 50 * a.cpu().numpy() + (1 - 0.5) * ir_vis[:, :, :3].cpu().numpy()
        # plt.imshow(a)
        # plt.show()

        # prepare CNN encoded features for RNN

        # # loop over timestamps to unroll
        # h = h.permute(0, 2, 3, 1).view(N, H*W, C).contiguous()  # N H*W C
        h_rgb = rgb
        h_rgb = h_rgb.view(N * H * W, C)  # NHW C
        # h_ir = ir
        # h_ir = h_ir.view(N * H * W, C)  # NHW C

        for i in range(gnn_iterations):
            #     # do this for every  sample in batch, not nice, but I don't know how to use index_select batchwise
            #     # fetch features from nearest neighbors
            h_rgb = torch.index_select(h_rgb, 0, rgb_knn).view(N * H * W, K, C)  # NHW K C
            # ir_neighbor_features = torch.index_select(h_ir, 0, ir_knn).view(N*H*W, K, C)  # NHW K C

            # rgb_central = h_rgb.unsqueeze(dim=-2)
            # rgb_central = rgb_central.repeat_interleave(k, dim=-2)
            # ir_central = h_ir.unsqueeze(dim=-2)
            # ir_central = ir_central.repeat_interleave(k, dim=-2)
            # run neighbor features through MLP g and activation function
            # rgb_features = torch.cat([rgb_neighbor_features, rgb_neighbor_features - ir_neighbor_features], dim=-1)
            # ir_features = torch.cat([ir_neighbor_features, ir_neighbor_features - rgb_neighbor_features], dim=-1)

            h_rgb = self.rgb_g_layers(h_rgb)
            # ir_features = self.ir_g_layers(ir_features)

            #     # average over activated neighbors
            h_rgb = torch.mean(h_rgb, dim=1)  # NHW C
            # m_ir, _ = torch.mean(ir_features, dim=1)

            #     # concatenate current state with messages
            #     # concat = torch.cat((h, m_rgb, m_ir), 1)  # N HW 3C
            # concat = torch.cat((m_rgb, m_ir), 1)  # NHW 2C
            # concat = m_rgb + m_ir

            # attention = torch.bmm(m_rgb.permute(1, 0), m_ir)

            #     # se concat
            h_rgb = h_rgb.view(N, H * W, -1)  # N HW 2C
            # concat = concat.mean(dim=1, keepdim=True)           # N 1 2C
            h_rgb = torch.bmm(h_rgb.permute(0, 2, 1), h_rgb)
            h_rgb = F.softmax(h_rgb, dim=-1)
            # concat_rgb = self.se_rgb(concat)     # N 1 C
            # concat_ir = self.se_ir(concat)
            # h_rgb = concat_rgb * h_rgb.view(N, H*W, C)
            # h_ir = (1 - concat_rgb) * h_ir.view(N, H*W, C)
            # h = self.gamma1 * h_rgb + self.gamma2 * h_ir
            # h = self.out_conv(h)
            #     # h = F.relu(h, inplace=True)
            h = torch.bmm(rgb_in.view(N, C // 2, H * W).permute(0, 2, 1).contiguous(), h_rgb)

        # plot hist
        # h = h.view(-1).cpu().detach().numpy()
        # m_rgb = m_rgb.view(-1).cpu().detach().numpy()
        # m_ir = m_ir.view(-1).cpu().detach().numpy()
        # cnn_encoder_output = cnn_encoder_output.view(-1).cpu().detach().numpy()
        # plt.subplot(141)
        # plt.hist(h)
        # plt.subplot(142)
        # plt.hist(m_rgb)
        # plt.subplot(143)
        # plt.hist(m_ir)
        # plt.subplot(144)
        # plt.hist(cnn_encoder_output)
        # plt.show()

        # format RNN activations back to image, concatenate original CNN embedding, return
        h = h.view(N, H, W, C // 2).permute(0, 3, 1, 2).contiguous()  # N C H W
        # return F.relu(h, inplace=True)
        return self.gamma * h + rgb_in


class DeepLabMultiGnn(nn.Module):
    def __init__(self, n_classes):
        super(DeepLabMultiGnn, self).__init__()

        # conv1
        features1 = []
        features1.append(nn.Conv2d(3, 64, 3, padding=1))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.Conv2d(64, 64, 3, padding=1))
        features1.append(nn.ReLU(inplace=True))
        self.features1 = nn.Sequential(*features1)
        self.rgb_pool1 = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True)  # 1/2

        # conv2
        features2 = []
        features2.append(nn.Conv2d(64, 128, 3, padding=1))
        features2.append(nn.ReLU(inplace=True))
        features2.append(nn.Conv2d(128, 128, 3, padding=1))
        features2.append(nn.ReLU(inplace=True))
        self.features2 = nn.Sequential(*features2)
        self.rgb_pool2 = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True)  # 1/4

        # conv3
        features3 = []
        features3.append(nn.Conv2d(128, 256, 3, padding=1))
        features3.append(nn.ReLU(inplace=True))
        features3.append(nn.Conv2d(256, 256, 3, padding=1))
        features3.append(nn.ReLU(inplace=True))
        features3.append(nn.Conv2d(256, 256, 3, padding=1))
        features3.append(nn.ReLU(inplace=True))
        self.features3 = nn.Sequential(*features3)
        self.rgb_pool3 = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True)  # 1/8

        # conv4
        features4 = []
        features4.append(nn.Conv2d(256, 512, 3, padding=1))
        features4.append(nn.ReLU(inplace=True))
        features4.append(nn.Conv2d(512, 512, 3, padding=1))
        features4.append(nn.ReLU(inplace=True))
        features4.append(nn.Conv2d(512, 512, 3, padding=1))
        features4.append(nn.ReLU(inplace=True))
        self.features4 = nn.Sequential(*features4)
        self.rgb_pool4 = nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True)  # 1/16

        # conv5
        features5 = []
        features5.append(nn.Conv2d(512, 512, 3, dilation=2, padding=2))
        features5.append(nn.ReLU(inplace=True))
        features5.append(nn.Conv2d(512, 512, 3, dilation=2, padding=2))
        features5.append(nn.ReLU(inplace=True))
        features5.append(nn.Conv2d(512, 512, 3, dilation=2, padding=2))
        features5.append(nn.ReLU(inplace=True))
        self.features5 = nn.Sequential(*features5)
        self.rgb_pool5 = nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True)  # 1/32

        self.ir_features1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.ir_pool1 = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True)

        self.ir_features2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.ir_pool2 = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True)

        self.ir_features3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.ir_pool3 = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True)

        self.ir_features4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.ir_pool4 = nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True)

        self.ir_features5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
        )

        fc = []
        fc.append(nn.AvgPool2d(3, stride=1, padding=1))
        fc.append(nn.Conv2d(512, 1024, 3, padding=12, dilation=12))
        fc.append(nn.ReLU(inplace=True))
        fc.append(nn.Conv2d(1024, 1024, 1))
        fc.append(nn.ReLU(inplace=True))
        fc.append(nn.Dropout(p=0.5))
        self.fc = nn.Sequential(*fc)

        self.score = nn.Conv2d(1024, n_classes, 1)
        self.gnn = EnetGnn(1, 256)

        self._initialize_weights()

    def _initialize_weights(self):

        vgg16 = torchvision.models.vgg16(pretrained=True)
        vgg_features = [
            vgg16.features[:4],
            vgg16.features[5:9],
            vgg16.features[10:16],
            vgg16.features[17:23],
            vgg16.features[24:],
        ]
        features = [
            self.features1,
            self.features2,
            self.features3,
            self.features4,
            self.features5
        ]

        for l1, l2 in zip(vgg_features, features):
            for ll1, ll2 in zip(l1.children(), l2.children()):
                if isinstance(ll1, nn.Conv2d) and isinstance(ll2, nn.Conv2d):
                    assert ll1.weight.size() == ll2.weight.size()
                    assert ll1.bias.size() == ll2.bias.size()
                    ll2.weight.data.copy_(ll1.weight.data)
                    ll2.bias.data.copy_(ll1.bias.data)

        ir_features = [
            self.ir_features1,
            self.ir_features2,
            self.ir_features3,
            self.ir_features4,
            self.ir_features5
        ]

        for l1, l2 in zip(vgg_features, ir_features):
            for ll1, ll2 in zip(l1.children(), l2.children()):
                if isinstance(ll1, nn.Conv2d) and isinstance(ll2, nn.Conv2d):
                    assert ll1.weight.size() == ll2.weight.size()
                    assert ll1.bias.size() == ll2.bias.size()
                    ll2.weight.data.copy_(ll1.weight.data)
                    ll2.bias.data.copy_(ll1.bias.data)

        nn.init.normal_(self.score.weight, std=0.01)
        nn.init.constant_(self.score.bias, 0)

    def forward(self, rgb, ir):
        _, _, h, w = rgb.size()
        rgb_in, ir_in = rgb, ir
        rgb = self.features1(rgb)
        ir = self.ir_features1(ir)
        rgb = self.rgb_pool1(rgb)  # 1/2
        ir = self.ir_pool1(ir)

        rgb = self.features2(rgb)
        ir = self.ir_features2(ir)
        rgb = self.rgb_pool2(rgb)  # 1/4
        ir = self.ir_pool2(ir)

        rgb = self.features3(rgb)
        ir = self.ir_features3(ir)
        out = self.rgb_pool3(rgb + ir)  # 1/8
        ir = F.max_pool2d(ir, kernel_size=3, stride=2, padding=1)

        out = self.gnn(out, out, ir, gnn_iterations=1, k=10)

        out = self.features4(out)
        out = self.rgb_pool4(out)  # 1/16

        out = self.features5(out)
        out = self.rgb_pool5(out)  # 1/32

        out = self.fc(out)
        out = self.score(out)
        out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
        return out

    def get_parameters(self, score=False):
        if score:
            for m in [self.score, self.gnn]:
                for p in m.parameters():
                    yield p
        else:
            for module in [self.features1, self.features2, self.features3,
                           self.features4, self.features5, self.ir_features1,
                           self.ir_features2, self.ir_features3, self.fc]:
                for m in module.modules():
                    for p in m.parameters():
                        if p.requires_grad:
                            yield p


class DeepLabResNetGnnUnet(nn.Module):
    def __init__(self, n_classes):
        super(DeepLabResNetGnnUnet, self).__init__()
        self.resnet_rgb = ResNet(Bottleneck, [3, 4, 6, 3])
        self.resnet_ir = ResNet(Bottleneck, [3, 4, 6, 3])
        # self.atrous_rates = [6, 12, 18, 24]
        # self.aspp_rgb = ASPP(2048, 512, self.atrous_rates)
        # self.aspp_ir = ASPP(2048, 512, self.atrous_rates)
        self.conv11 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv21 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv22 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # self.down1 = nn.Conv2d(2048, 512, kernel_size=1)
        # self.down2 = nn.Conv2d(2048, 512, kernel_size=1)
        # self.up1 = nn.Conv2d(512, 2048, kernel_size=1)
        # self.up2 = nn.Conv2d(512, 2048, kernel_size=1)

        self.uplayer1 = UpLayer(1024, 512, upsample=False)
        self.uplayer2 = UpLayer(1024, 512)
        self.uplayer3 = UpLayer(512, 256)
        self.uplayer4 = UpLayer(256, 64)

        # self.fusion1 = Fusion(1024)
        # self.fusion2 = Fusion(512)
        # self.fusion3 = Fusion(256)

        # self.att2 = PAM_Module(512)
        # self.att3 = PAM_Module(1024)
        # self.att4 = PAM_Module(2048)

        # self.non1 = NonLocalBlockND(512)
        # self.non2 = NonLocalBlockND(1024)
        # self.non3 = NonLocalBlockND(2048)
        # self.non_rgb_ir = NonLocalRGBIR(512)

        self.glore_res1 = GCNetHead(64, 32, 1)
        self.glore_res2 = GCNetHead(256, 32, 1)
        self.glore_res3 = GCNetHead(512, 32, 1)
        # self.glore_res4 = GCNetHead(2048, 32, 1)

        # self.glore1 = GCNetHead(512, 32, 1)
        # self.glore2 = GCNetHead(512, 32, 1)

        # self.glore_fusion = GloRe_Fusion(512, 32, 1)

        # self.score1 = nn.Conv2d(512, n_classes, kernel_size=1)
        self.score1 = nn.Sequential(
            # nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            nn.Dropout2d(0.1, False),
            nn.Conv2d(64, n_classes, 1))
        self.score2 = nn.Sequential(nn.Dropout2d(0.1, False),
                                    nn.Conv2d(512, n_classes, 1))

        # self.gnn1 = EnetGnn(1024)
        # self.gnn2 = EnetGnn(1024)

    def get_adj(self, mat):
        N, C, H, W = mat.size()
        theta_x = self.theta(mat).view(N, 1024, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(mat).view(N, 1024, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        return f_div_C

    def forward(self, rgb, ir):
        _, _, h, w = rgb.size()
        # x2 = F.interpolate(x, size=(int(h * 0.75) + 1, int(h * 0.75) + 1), mode='bilinear', align_corners=True)
        # x3 = F.interpolate(x, size=(int(h * 0.5) + 1, int(h * 0.5) + 1), mode='bilinear', align_corners=True)
        rgb = self.resnet_rgb.conv1(rgb)
        rgb = self.resnet_rgb.bn1(rgb)
        rgb = self.resnet_rgb.relu(rgb)
        skip1 = rgb

        rgb = self.resnet_rgb.maxpool(rgb)
        rgb = self.resnet_rgb.layer1(rgb)
        skip2 = rgb
        # ir = self.resnet_ir.conv1(ir)
        # ir = self.resnet_ir.layer1(ir)

        rgb = self.resnet_rgb.layer2(rgb)
        skip3 = rgb
        # ir = self.resnet_ir.layer2(ir)
        # rgb = self.att2(rgb)
        # rgb = self.non1(rgb)
        # rgb = self.glore_res2(rgb)

        rgb = self.resnet_rgb.layer3(rgb)
        # skip4 = rgb
        # ir = self.resnet_ir.layer3(ir)
        # rgb = self.att3(rgb)
        # rgb = self.non2(rgb)
        # rgb = self.glore_res3(rgb)

        rgb = self.resnet_rgb.layer4(rgb)
        # skip4 = rgb
        # ir = self.resnet_ir.layer4(ir)
        # rgb = self.att4(rgb)
        # rgb = self.non3(rgb)
        # rgb = self.glore_res4(rgb)

        # rgb = self.resnet_rgb(rgb)
        # ir = self.resnet_ir(ir)

        rgb = self.conv11(rgb)
        # ir = self.conv21(ir)

        # rgb = self.uplayer1(rgb + skip4)
        rgb = self.uplayer2(rgb)
        rgb += skip3
        rgb = self.glore_res3(rgb)

        rgb = self.uplayer3(rgb)
        rgb += skip2
        rgb = self.glore_res2(rgb)

        rgb = self.uplayer4(rgb)
        rgb += skip1
        rgb = self.glore_res1(rgb)

        # rgb, ir = self.non_rgb_ir(rgb, ir)
        # cat = torch.cat([rgb, ir], dim=1)
        # rgb = self.gnn1(cat, rgb, 1, 16)
        # rgb, ir = self.glore_fusion(rgb, ir)
        # rgb = self.non1(rgb)
        # ir = self.glore2(ir)

        # rgb, ir = self.glore_fusion(rgb, ir)

        # rgb = self.conv12(rgb)
        # ir = self.conv22(ir)

        # ir = self.gnn2(cat, ir, 1, 16)

        # rgbir = rgb + ir
        # rgb_adj = self.get_adj(rgb)
        # ir_adj = self.get_adj(ir)
        # rgbir_adj = self.get_adj(rgbir)

        # rgb = self.gcn1(rgb, rgb_adj)
        # ir = self.gcn2(ir, ir_adj)
        # rgbir = self.gcn3(rgbir, rgbir_adj)

        # rgb = self.down1(rgb)
        # ir = self.down2(ir)

        # cat = torch.cat([rgb, ir], dim=1)
        # rgb = self.gnn1(cat, rgb, 1, 8)
        # ir = self.gnn2(cat, ir, 1, 8)

        # rgb = self.up1(rgb)
        # ir = self.up2(ir)

        # rgb = rgb + ir
        rgb = self.score1(rgb)
        # ir = self.score2(ir)

        # rgb = self.score1(rgb)
        # ir = self.score1(ir)

        # ir = self.conv12(ir)
        # rgb = rgb + ir
        # rgb = self.aspp_rgb(rgb)
        # ir = self.aspp_ir(ir)
        # out = rgb + ir
        # out = torch.cat((rgb, ir), dim=1)
        # out = self.fusion1(rgb_skip3, ir_skip3, out)
        # out = self.uplayer1(out)
        # out = self.fusion2(rgb_skip2, ir_skip2, out)
        # out = self.uplayer2(out)
        # out = self.fusion3(rgb_skip1, ir_skip1, out)
        # out = self.uplayer3(out)
        # out = self.uplayer4(out)
        rgb = F.interpolate(rgb, size=(h, w), mode='bilinear', align_corners=True)
        # ir = F.interpolate(ir, size=(h, w), mode='bilinear', align_corners=True)
        return rgb
        # return ir

    def get_parameters(self, score=False):
        # if score:
        #     for m in [self.conv11, self.score1, self.gnn1, self.gnn2]:
        #         for p in m.parameters():
        #             if p.requires_grad:
        #                 yield p
        # else:
        #     for m in [self.resnet_rgb, self.resnet_ir]:
        #         for p in m.parameters():
        #             if p.requires_grad:
        #                 yield p
        if score:
            for m in self.modules():
                for p in m.parameters():
                    if p.requires_grad:
                        yield p


class Fusion(nn.Module):
    def __init__(self, channels):
        super(Fusion, self).__init__()
        self.rgb_conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.ir_conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.upconv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.rgb_conv2 = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True)
        )
        self.ir_conv2 = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb, ir, up):
        rgb = self.rgb_conv1(rgb)
        ir = self.ir_conv1(ir)
        up = self.upconv1(up)
        # rgb = torch.cat((rgb, up), dim=1)
        # ir = torch.cat((ir, up), dim=1)
        rgb = rgb + up
        ir = ir + up
        rgb = self.rgb_conv2(rgb)
        ir = self.ir_conv2(ir)
        out = torch.cat((rgb, ir), dim=1)
        return out


class UpLayer(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, upsample=True):
        super(UpLayer, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()
        self.skip = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.upsample = upsample

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        identity = out

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.skip(identity)
        # 交换激活到上采样后

        # if self.upsample:
        #     out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)

        out = self.relu2(out)
        return out


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()

        # rate1, rate2, rate3, rate4 = atrous_rates
        # self.conv1 = nn.Conv2d(2048, out_channels, kernel_size=3, padding=rate1, dilation=rate1, bias=True)
        # self.conv2 = nn.Conv2d(2048, out_channels, kernel_size=3, padding=rate2, dilation=rate2, bias=True)
        # self.conv3 = nn.Conv2d(2048, out_channels, kernel_size=3, padding=rate3, dilation=rate3, bias=True)
        # self.conv4 = nn.Conv2d(2048, out_channels, kernel_size=3, padding=rate4, dilation=rate4, bias=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(2048, out_channels, kernel_size=1, padding=0, bias=True),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.conv1(x)
        # features2 = self.conv2(x)
        # features3 = self.conv3(x)
        # out = features1 + features2 + features3

        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    Adapted from https://github.com/speedinghzl/pytorch-segmentation-toolbox/blob/master/networks/deeplabv3.py
    """

    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # )
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2)

        self._initialize_weights()

    def _initialize_weights(self):
        resnet = torchvision.models.resnet50(pretrained=True)
        self.conv1.load_state_dict(resnet.conv1.state_dict())
        self.bn1.load_state_dict(resnet.bn1.state_dict())
        self.layer1.load_state_dict(resnet.layer1.state_dict())
        self.layer2.load_state_dict(resnet.layer2.state_dict())
        self.layer3.load_state_dict(resnet.layer3.state_dict())
        self.layer4.load_state_dict(resnet.layer4.state_dict())

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
