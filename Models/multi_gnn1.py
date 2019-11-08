import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.modules.utils import _pair, _quadruple
from torch.autograd import Variable
import numpy as np
import math
import matplotlib.pyplot as plt
from utils import vis
from torch.nn import Parameter
from collections import OrderedDict
import seaborn as sns
import networkx as nx
import scipy.sparse as sp
from .base import BaseNet

graph1 = {0:[1, 2, 4, 6, 8, 9, 11, 12],
		 1:[0, 2, 4, 6, 7, 8, 9],
		 2:[0, 1, 4, 6, 7, 8, 9, 11, 12],
		 3:[4, 5, 8, 10, 11],
		 4:[0, 2, 3, 5, 6, 7, 8, 10, 11, 12],
		 5:[3, 4, 8, 10, 11],
		 6:[0, 1, 2, 4, 7, 8, 9, 11, 12],
		 7:[0, 1, 2, 4, 6, 8, 9, 11],
		 8:[0, 2, 3, 4, 5, 6, 7, 9, 10, 11],
		 9:[0, 1, 2, 4, 6, 7, 8, 11],
		 10:[3, 4, 5, 8, 11],
		 11:[2, 3, 4, 6, 7, 8, 9, 10],
		 12:[0, 2, 3, 4, 6, 8, 9, 11]
		}

graph1 = {0:[2, 4, 6, 8, 9, 12],
		 1:[0, 2, 6, 7, 8],
		 2:[0, 1, 4, 6, 7, 8, 9, 11],
		 3:[4, 8, 11],
		 4:[2, 3, 5, 8, 10, 11],
		 5:[3, 4, 8, 10, 11],
		 6:[0, 1, 2, 4, 7, 9, 12],
		 7:[1, 2, 4, 6, 8, 11],
		 8:[2, 3, 4, 5, 11],
		 9:[0, 2, 4, 6, 8, 11],
		 10:[3, 4, 5, 8, 11],
		 11:[3, 4, 8],
		 12:[0, 2, 3, 4, 6, 8]
		}

graph = {0:[2, 6, 8, 12],
		 1:[2, 6, 7, 8],
		 2:[0, 1, 4, 6, 7, 8],
		 3:[4, 8, 11],
		 4:[3, 8, 10, 11],
		 5:[3, 4, 8, 10],
		 6:[0, 1, 2, 7, 8],
		 7:[1, 2, 4, 6, 8, 11],
		 8:[3, 4, 11],
		 9:[0, 2, 4, 6, 8, 11],
		 10:[3, 4, 8, 11],
		 11:[3, 4, 8],
		 12:[0, 2, 4, 6, 8]
		}

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(adj)) # return a adjacency matrix of adj ( type is numpy)
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0])) #
    # return sparse_to_tuple(adj_normalized)
    return adj_normalized.todense()

adj = Variable(torch.from_numpy(preprocess_adj(graph)).float()).cuda()
# adj = adj.unsqueeze(0).unsqueeze(0).expand(1, 1, 13, 13).cuda()
# adj = adj.unsqueeze(0).expand(1, 13, 13).cuda()

class Featuremaps_to_Graph(nn.Module):

    def __init__(self,input_channels,hidden_layers,nodes=13):
        super(Featuremaps_to_Graph, self).__init__()
        self.pre_fea = Parameter(torch.FloatTensor(input_channels,nodes))
        self.weight = Parameter(torch.FloatTensor(input_channels,hidden_layers))
        self.reset_parameters()

    def reset_parameters(self):
        for ww in self.parameters():
            torch.nn.init.xavier_uniform_(ww)

    def forward(self, input):
        n,c,h,w = input.size()
        # print('fea input',input.size())
        input1 = input.view(n,c,h*w)
        input1 = input1.transpose(1,2) # n x hw x c
        # print('fea input1', input1.size())
        ############## Feature maps to node ################
        fea_node = torch.matmul(input1,self.pre_fea) # n x hw x n_classes
        weight_node = torch.matmul(input1,self.weight) # n x hw x hidden_layer
        # softmax fea_node
        fea_node = F.softmax(fea_node,dim=1)
        # print(fea_node.size(),weight_node.size())
        graph_node = F.relu(torch.matmul(fea_node.transpose(1,2),weight_node))
        return graph_node # n x n_class x hidden_layer

class Graph_to_Featuremaps(nn.Module):
    # this is a special version
    def __init__(self,input_channels,output_channels,hidden_layers,nodes=16):
        super(Graph_to_Featuremaps, self).__init__()
        self.node_fea = Parameter(torch.FloatTensor(input_channels+hidden_layers,1))
        self.weight = Parameter(torch.FloatTensor(hidden_layers,output_channels))
        self.reset_parameters()

    def reset_parameters(self):
        for ww in self.parameters():
            torch.nn.init.xavier_uniform_(ww)

    def forward(self, input, res_feature):
        '''
        :param input: 1 x batch x nodes x hidden_layer
        :param res_feature: batch x channels x h x w
        :return:
        '''
        batchi,channeli,hi,wi = res_feature.size()
        # print(res_feature.size())
        # print(input.size())
        try:
            _,batch,nodes,hidden = input.size()
        except:
            # print(input.size())
            input = input.unsqueeze(0)
            _,batch, nodes, hidden = input.size()

        assert batch == batchi
        input1 = input.transpose(0,1).expand(batch,hi*wi,nodes,hidden)
        res_feature_after_view = res_feature.view(batch,channeli,hi*wi).transpose(1,2)
        res_feature_after_view1 = res_feature_after_view.unsqueeze(2).expand(batch,hi*wi,nodes,channeli)
        new_fea = torch.cat((res_feature_after_view1,input1),dim=3)

        # print(self.node_fea.size(),new_fea.size())
        new_node = torch.matmul(new_fea, self.node_fea) # batch x hw x nodes x 1
        new_weight = torch.matmul(input, self.weight)  # batch x node x channel
        new_node = new_node.view(batch, hi*wi, nodes)
        # 0721
        new_node = F.softmax(new_node, dim=-1)
        #
        feature_out = torch.matmul(new_node,new_weight)
        # print(feature_out.size())
        feature_out = feature_out.transpose(2,3).contiguous().view(res_feature.size())
        feature_out = F.relu(feature_out)
        return feature_out + res_feature

class GraphLearner(nn.Module):
    def __init__(self, in_feature_dim, combined_feature_dim):
        super(GraphLearner, self).__init__()

        '''
        ## Variables:
        - in_feature_dim: dimensionality of input features
        - combined_feature_dim: dimensionality of the joint hidden embedding
        - K: number of graph nodes/objects on the image
        '''

        # Parameters
        self.in_dim = in_feature_dim
        self.combined_dim = combined_feature_dim

        # Embedding layers
        self.edge_layer_1 = nn.Linear(in_feature_dim,
                                      combined_feature_dim)
        self.edge_layer_2 = nn.Linear(combined_feature_dim,
                                      combined_feature_dim)
        # self.edge_layer_1 = nn.utils.weight_norm(self.edge_layer_1)
        # self.edge_layer_2 = nn.utils.weight_norm(self.edge_layer_2)

    def forward(self, graph_nodes):
        '''
        ## Inputs:
        - graph_nodes (batch_size, K, in_feat_dim): input features
        ## Returns:
        - adjacency matrix (batch_size, K, K)
        '''

        B, N, C = graph_nodes.size()
        # graph_nodes = graph_nodes.permute(0, 2, 3, 1).contiguous().view(-1, C)  # BHW C

        # layer 1
        h = self.edge_layer_1(graph_nodes)
        h = F.relu(h)

        # layer 2
        h = self.edge_layer_2(h)
        h = F.relu(h)

        # outer product
        h = h.view(-1, N, self.combined_dim)
        adjacency_matrix = torch.matmul(h, h.transpose(1, 2))
        adjacency_matrix = F.softmax(adjacency_matrix, dim=-1)
        return adjacency_matrix

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
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
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1) # HW C
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height) # C HW
        energy = torch.bmm(proj_query, proj_key) # HW HW
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height) # C HW

        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # C HW
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 8
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(256, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)

        feat_sum = sa_conv + sc_conv

        sasc_output = self.conv8(feat_sum)

        output = sasc_output
        return output

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
            bn(self.in_channels, momentum=0.1)
            )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        # self.g = nn.Sequential(max_pool_layer, self.g)
        # self.phi = nn.Sequential(max_pool_layer, self.phi)

        # self.gamma = Parameter(torch.zeros(1))
        # self.gp = GeometryPrior(53, 128)

    def forward(self, x):
        '''
        :param x: (b, c, h, w)
        :return:
        '''

        batch_size = x.size(0)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)    # b hw c

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)    # b c hw/4
        theta_phi = torch.matmul(theta_x, phi_x)    # b hw hw/4
        theta_phi = theta_phi * self.inter_channels ** (-.5)
        # theta_phi = theta_phi + self.gp(0)
        p = F.softmax(theta_phi, dim=-1)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)    # b c hw/4

        y = torch.matmul(g_x, p.permute(0, 2, 1))    # b c hw
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        # z = self.gamma * W_y + x
        z = W_y + x

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
        rgb_theta_x = rgb_theta_x.permute(0, 2, 1)    # b hw c

        rgb_phi_x = self.rgb_phi(rgb).view(batch_size, self.inter_channels, -1)    # b c hw/4
        rgb_out = torch.matmul(rgb_theta_x, rgb_phi_x)    # b hw hw/4
        del rgb_theta_x
        del rgb_phi_x
        rgb_out = rgb_out + self.gp(0)
        rgb_out = F.softmax(rgb_out, dim=-1)

        rgb_g = self.rgb_g(rgb).view(batch_size, self.inter_channels, -1)    # b c hw/4

        rgb_out = torch.matmul(rgb_g, rgb_out.permute(0, 2, 1))    # b c hw
        del rgb_g
        rgb_out = rgb_out.view(batch_size, self.inter_channels, *rgb.size()[2:])
        rgb_out = self.rgb_W(rgb_out)
        rgb_out = rgb_out + rgb

        ir_theta_x = self.ir_theta(ir).view(batch_size, self.inter_channels, -1)
        ir_theta_x = ir_theta_x.permute(0, 2, 1)    # b hw c

        ir_phi_x = self.ir_phi(ir).view(batch_size, self.inter_channels, -1)    # b c hw/4
        ir_out = torch.matmul(ir_theta_x, ir_phi_x)    # b hw hw/4
        del ir_theta_x
        del ir_phi_x
        ir_out = ir_out + self.gp(0)
        ir_out = F.softmax(ir_out, dim=-1)

        ir_g = self.ir_g(ir).view(batch_size, self.inter_channels, -1)    # b c hw/4

        ir_out = torch.matmul(ir_g, ir_out.permute(0, 2, 1))    # b c hw
        del ir_g
        ir_out = ir_out.view(batch_size, self.inter_channels, *rgb.size()[2:])
        ir_out = self.ir_W(ir_out)
        ir_out = ir_out +ir

        return rgb_out, ir_out

def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        for p in m.parameters():
            p.requires_grad = False

def freeze_backbone_bn(module):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False

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

class GraphConvolution(nn.Module):

    def __init__(self,in_features,out_features,bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features,out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1./math.sqrt(self.weight(1))
        # self.weight.data.uniform_(-stdv,stdv)
        torch.nn.init.xavier_uniform_(self.weight)
        # if self.bias is not None:
        #     self.bias.data.uniform_(-stdv,stdv)

    def forward(self, input, adj=None, relu=False):
        support = torch.matmul(input, self.weight)
        # print(support.size(),adj.size())
        if adj is not None:
            output = torch.matmul(adj, support)
        else:
            output = support
        # print(output.size())
        if self.bias is not None:
            return output + self.bias
        else:
            if relu:
                return F.relu(output)
            else:
                return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GAT(nn.Module):
    """
    inputs: (batch, channels, nodes)
    """
    def __init__(self, in_channel):
        super(GAT, self).__init__()
        self.in_channel = in_channel

        self.conv = nn.Conv1d(in_channel, 1, kernel_size=1, bias=False)

    def forward(self, x):
        _, c, n = x.size()  # b c n
        f1 = self.conv(x)  # b 1 n
        f2 = self.conv(x)

        logits = f1.permute(0, 2, 1) + f2  # b n n
        coefs = F.softmax(F.leaky_relu(logits), dim=-1)
        vals = torch.matmul(coefs, x.permute(0, 2, 1)).permute(0, 2, 1)  # b, c, n
        return F.elu(vals)

class LatentGNN(nn.Module):
    def __init__(self, in_channels, nodes=16):
        super(LatentGNN, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = self.in_channels // 4
        self.nodes = nodes

        # reduce dimension
        self.reduction = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=(1, 1), padding=0,
                                   bias=False)

        self.vis2latent = nn.Conv2d(self.in_channels, self.nodes, kernel_size=(1, 1), padding=0)

        self.restore = nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=(1, 1), padding=0,
                                 bias=False)

        self.fc = nn.Linear(self.inter_channels, self.inter_channels)
        self.blocker = nn.BatchNorm2d(self.in_channels)

    def forward(self, x):
        '''
        :param x: (n, c, h, w)
        '''
        N, C, H, W = x.size()

        x_reduce = self.reduction(x).view(N, self.inter_channels, -1).contiguous()  # N C/4 HW

        vis2latent = self.vis2latent(x).view(N, self.nodes, -1).contiguous()  # N d HW

        # vis2latent = F.softmax(vis2latent, dim=-1)
        z = torch.matmul(x_reduce, vis2latent.permute(0, 2, 1))  # N C/4 d

        affinity = torch.matmul(z.permute(0, 2, 1), z)  # N d d
        # affinity = F.relu(affinity, inplace=True)
        affinity = F.softmax(affinity, dim=1)
        z_hat = torch.matmul(z, affinity)  # N C/4 d
        # z_hat = self.fc(z_hat.permute(0, 2, 1)).permute(0, 2, 1)

        x_new = torch.matmul(z_hat, vis2latent)  # N C/4 HW

        x_new = self.restore(x_new.view(N, self.inter_channels, H, W))

        out = x + F.relu(self.blocker(x_new))

        return out

class GloRe_Unit(nn.Module):
    def __init__(self, num_in, num_mid, stride=(1,1), kernel=1):
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
        # global reasoning
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)

        # graph attention
        # self.gat = GAT(self.num_s, self.num_s)

        # multi graph attention
        # self.attentions = [GAT(self.num_s) for _ in range(4)]
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)
        # self.reduce = nn.Conv1d(self.num_s * 4, self.num_s, kernel_size=1, bias=False)

        # graph learner
        # self.adj = GraphLearner(128, 128)
        # self.gcn1 = GraphConvolution(self.num_s, self.num_s)
        # self.gcn2 = GraphConvolution(self.num_s, self.num_s)
        # self.gcn3 = GraphConvolution(self.num_s, self.num_s)
        # ----------
        # tail: extend dimension
        self.fc_2 = nn.Conv2d(self.num_s, num_in, kernel_size=kernel_size, padding=padding, stride=(1,1),
                              groups=1, bias=False)

        self.blocker = nn.BatchNorm2d(num_in)

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

        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        # adj = torch.matmul(x_n_state.permute(0, 2, 1), x_n_state)
        # adj = F.softmax(adj, dim=-1)

        # adj = self.adj(x_n_state.permute(0, 2, 1))
        # x_n_rel = self.gcn1(x_n_state.permute(0, 2, 1), adj=adj, relu=True).permute(0, 2, 1)
        # x_n_rel = self.gcn2(x_n_rel.permute(0, 2, 1), adj=adj, relu=True).permute(0, 2, 1)
        # x_n_rel = self.gcn3(x_n_rel.permute(0, 2, 1), adj=adj, relu=True).permute(0, 2, 1)

        # x_n_rel = torch.cat([att(x_n_state) for att in self.attentions], dim=1)
        # x_n_rel = self.reduce(x_n_rel)

        # x_n_rel = self.gat(x_n_state)

        x_n_rel = self.gcn(x_n_state)

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

class GCNetHead(nn.Module):
    def __init__(self, in_channels, n, repeat):
        super(GCNetHead, self).__init__()

        inter_channels = 256
        # self.conv51 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
        #                             nn.BatchNorm2d(inter_channels),
        #                             nn.ReLU())

        self.gcn = nn.Sequential(OrderedDict([ ("GCN%02d"%i,
                          GloRe_Unit(inter_channels, n, kernel=1)
                          ) for i in range(repeat) ]))

        # self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
        #                             nn.BatchNorm2d(inter_channels),
        #                             nn.ReLU())

    def forward(self, x):
        # x = self.conv51(x)
        x = self.gcn(x)
        # x = self.conv52(x)
        return x

class GloRe_Fusion(nn.Module):
    def __init__(self, num_in, num_mid, stride=(1, 1), kernel=1):
        super(GloRe_Fusion, self).__init__()
        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)

        kernel_size = (kernel, kernel)
        padding = (1, 1) if kernel == 3 else (0, 0)

        # reduce dimension
        self.rgb_conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=kernel_size, padding=padding)
        # generate graph transformation function
        self.rgb_conv_proj = nn.Conv2d(num_in, self.num_n, kernel_size=kernel_size, padding=padding)
        # ----------
        # global reasoning
        self.rgb_gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        # tail: extend dimension
        self.rgb_fc_2 = nn.Conv2d(self.num_s, num_in, kernel_size=kernel_size, padding=padding, stride=(1, 1),
                              groups=1, bias=False)

        self.rgb_blocker = nn.BatchNorm2d(num_in)

        # reduce dimension
        self.ir_conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=kernel_size, padding=padding)
        # generate graph transformation function
        self.ir_conv_proj = nn.Conv2d(num_in, self.num_n, kernel_size=kernel_size, padding=padding)
        # ----------
        # global reasoning
        self.ir_gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        # tail: extend dimension
        self.ir_fc_2 = nn.Conv2d(self.num_s, num_in, kernel_size=kernel_size, padding=padding, stride=(1, 1),
                              groups=1, bias=False)

        self.ir_blocker = nn.BatchNorm2d(num_in)

    def forward(self, rgb, ir):
        '''
        :param x: (n, c, h, w)
        '''
        batch_size = rgb.size(0)
        # rgb
        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        rgb_state_reshaped = self.rgb_conv_state(rgb).view(batch_size, self.num_s, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        rgb_proj_reshaped = self.rgb_conv_proj(rgb).view(batch_size, self.num_n, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        rgb_rproj_reshaped = rgb_proj_reshaped

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # projection: pixel space -> instance space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        rgb_n_state = torch.matmul(rgb_state_reshaped, rgb_proj_reshaped.permute(0, 2, 1))
        rgb_n_state = rgb_n_state * (1. / rgb_state_reshaped.size(2))

        rgb_n_rel = self.rgb_gcn(rgb_n_state)

        # reverse projection: instance space -> pixel space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        rgb_state_reshaped = torch.matmul(rgb_n_rel, rgb_rproj_reshaped)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # (n, num_state, h*w) --> (n, num_state, h, w)
        rgb_state = rgb_state_reshaped.view(batch_size, self.num_s, *rgb.size()[2:])

        # -----------------
        # final
        rgb_out = rgb + self.rgb_blocker(self.rgb_fc_2(rgb_state))

        # ir
        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        ir_state_reshaped = self.ir_conv_state(ir).view(batch_size, self.num_s, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        ir_proj_reshaped = self.ir_conv_proj(ir).view(batch_size, self.num_n, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        ir_rproj_reshaped = ir_proj_reshaped

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # projection: pixel space -> instance space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        ir_n_state = torch.matmul(ir_state_reshaped, ir_proj_reshaped.permute(0, 2, 1))
        ir_n_state = ir_n_state * (1. / ir_state_reshaped.size(2))

        ir_n_rel = self.ir_gcn(ir_n_state)

        # reverse projection: instance space -> pixel space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        ir_state_reshaped = torch.matmul(ir_n_rel, ir_rproj_reshaped)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # (n, num_state, h*w) --> (n, num_state, h, w)
        ir_state = ir_state_reshaped.view(batch_size, self.num_s, *ir.size()[2:])

        # -----------------
        # final
        ir_out = ir + self.ir_blocker(self.ir_fc_2(ir_state))

        return rgb_out, ir_out

class GCNetFusionHead(nn.Module):
    def __init__(self, in_channels, n, repeat):
        super(GCNetFusionHead, self).__init__()

        inter_channels = 512
        self.rgb_conv51 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU(inplace=True))

        self.rgb_conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU(inplace=True))

        self.ir_conv51 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU(inplace=True))

        self.ir_conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU(inplace=True))

        self.fu_gcn = nn.Sequential(OrderedDict([ ("GCN%02d"%i,
                          GloRe_Fusion(inter_channels, n, kernel=1)
                          ) for i in range(repeat) ]))

    def forward(self, rgb, ir):
        rgb = self.rgb_conv51(rgb)
        ir = self.ir_conv51(ir)

        rgb, ir = self.fu_gcn(rgb, ir)

        rgb = self.rgb_conv52(rgb)
        ir = self.ir_conv52(ir)
        return rgb, ir

class DaulGraph(nn.Module):
    def __init__(self, num_in, num_mid, stride=(1,1), kernel=1):
        super(DaulGraph, self).__init__()
        self.num_s = int(2 * num_mid)
        # self.num_s = num_in // 4
        self.num_n = int(1 * num_mid)

        self.inter_channels = num_in // 4

        kernel_size = (kernel, kernel)
        padding = (1, 1) if kernel == 3 else (0, 0)

        # reduce dimension
        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=kernel_size, padding=padding)
        # generate graph transformation function
        self.conv_proj = nn.Conv2d(num_in, self.num_n, kernel_size=kernel_size, padding=padding)
        # ----------
        # global reasoning
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)

        # tail: extend dimension
        self.fc_2 = nn.Conv2d(self.num_s, num_in, kernel_size=kernel_size, padding=padding, stride=(1, 1),
                              groups=1, bias=False)

        self.blocker = nn.BatchNorm2d(num_in)

        # nonlocal
        self.g = nn.Conv2d(in_channels=num_in, out_channels=num_in // 4,
                         kernel_size=1, stride=1, padding=0)

        self.W = nn.Conv2d(in_channels=num_in // 4, out_channels=num_in,
                    kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=num_in, out_channels=num_in // 4,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=num_in, out_channels=num_in // 4,
                           kernel_size=1, stride=1, padding=0)

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

        # projection: pixel space -> instance space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        x_n_state = x_n_state * (1. / x_state_reshaped.size(2))

        x_n_rel = self.gcn(x_n_state)

        # reverse projection: instance space -> pixel space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)

        # (n, num_state, h*w) --> (n, num_state, h, w)
        x_state = x_state_reshaped.view(batch_size, self.num_s, *x.size()[2:])

        # nonlocal
        avg_x = F.avg_pool2d(x, kernel_size=2, stride=2)
        theta_x = self.theta(avg_x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)  # b hw c

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # b c hw/4
        theta_phi = torch.matmul(theta_x, phi_x)  # b hw hw/4
        theta_phi = theta_phi * self.inter_channels ** (-.5)
        p = F.softmax(theta_phi, dim=-1)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # b c hw/4

        y = torch.matmul(g_x, p.permute(0, 2, 1))  # b c hw
        y = y.view(batch_size, self.inter_channels, *avg_x.size()[2:])
        W_y = self.W(y)
        W_y = F.interpolate(W_y, size=(x.size(2), x.size(3)), mode='nearest')

        out = x + self.blocker(self.fc_2(x_state)) + W_y

        return out

class EnetGnn(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.rgb_g_layers = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
            nn.ReLU(inplace=True)
        )
        self.gamma = Parameter(torch.zeros(1))
        self.feat2graph = Featuremaps_to_Graph(1024, 512)
        self.graph2feat = Graph_to_Featuremaps(1024, 512, 512)
        self.graph_learner = GraphLearner(512, 256)
        self.gcn = GCN(num_state=512, num_node=16)
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
        self.gamma1 = Parameter(torch.zeros(1))
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

        _, indices = torch.topk(r, k=k, largest=False)
        return indices

    def forward(self, cat, rgb_in, gnn_iterations=1, k=16):

        # rgb = F.normalize(rgb, dim=1)
        # ir = F.normalize(ir, dim=1)
        # h = rgb + ir

        N = cat.size()[0]
        C = cat.size()[1]
        H = cat.size()[2]
        W = cat.size()[3]
        K = k
        identity = rgb_in
        # cat = cat.view(N, C, H*W).permute(0, 2, 1).contiguous()  # N H*W C
        # ir = ir.view(N, C, H*W).permute(0, 2, 1).contiguous()  # N H*W C
        rgb_in = rgb_in.permute(0, 2, 3, 1).view(N, H*W, C).contiguous()  # N*H*W C
        # ir = ir.permute(0, 2, 3, 1).view(N*H*W, C).contiguous()  # N*H*W C

        # get k nearest neighbors
        # a = F.normalize(a, dim=-1)
        rgb_knn = self.get_knn_indices(rgb_in, k=k)  # N HW K
        rgb_knn = rgb_knn.view(N*H*W*K).long()  # NHWK
        # rgb_knn = rgb_knn.view(N, H*W, K).long()  # NHWK
        # ir_knn = self.get_knn_indices(F.normalize(ir, dim=-1), k=k)  # N HW K
        # ir_knn = ir_knn.view(N*H*W*K).long()  # NHWK
        # cat = self.feat2graph(cat)  # B N C
        # _, nodes, channels = cat.size()
        # adj = self.graph_learner(cat)  # BHW
        # top_k, top_ind = torch.topk(
        #     adj, k=16, dim=-1, sorted=False)
        # top_k BHW 16
        # top_k = torch.stack([F.softmax(top_k[:, k], dim=-1) for k in range(top_k.size(1))]).transpose(0, 1)
        #
        # rgb_in = rgb_in.view(N, C // 2, H*W).permute(0, 2, 1).contiguous()
        # rgb_in = rgb_in.unsqueeze(1).expand(N, H*W, H*W, C // 2)
        # idx = top_ind.unsqueeze(-1).expand(N, H*W, top_k.size(-1), C // 2)
        # rgb_in = torch.gather(rgb_in, dim=2, index=idx)
        # rgb_in = top_k.unsqueeze(-1) * rgb_in

        # rgb_knn = self.get_knn_indices(cat, k=k)  # N HW K
        # rgb_knn = rgb_knn.view(N*H*W*K).long()  # NHWK
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
        h_rgb = rgb_in
        h_rgb = h_rgb.view(N * H * W , C)  # NHW C
        # h_ir = ir
        # h_ir = h_ir.view(N * H * W, C)  # NHW C

        for i in range(gnn_iterations):
        #     # do this for every  sample in batch, not nice, but I don't know how to use index_select batchwise
        #     # fetch features from nearest neighbors
            h_rgb = torch.index_select(h_rgb, 0, rgb_knn).view(N*H*W, K, C)  # NHW K C
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
        #     h_rgb = h_rgb.view(N, H*W, -1)    # N HW 2C
            # concat = concat.mean(dim=1, keepdim=True)           # N 1 2C
            # h_rgb = torch.bmm(h_rgb.permute(0, 2, 1), h_rgb)
            # h_rgb = h_rgb * h_rgb.size(1) ** (-.5)
            h_rgb = h_rgb.view(N, H*W, C)
            h_rgb = torch.bmm(h_rgb, h_rgb.permute(0, 2, 1))
            h_rgb = F.softmax(h_rgb, dim=1)  # HW C
            # concat_rgb = self.se_rgb(concat)     # N 1 C
            # concat_ir = self.se_ir(concat)
            # h_rgb = concat_rgb * h_rgb.view(N, H*W, C)
            # h_ir = (1 - concat_rgb) * h_ir.view(N, H*W, C)
            # h = self.gamma1 * h_rgb + self.gamma2 * h_ir
            # h = self.out_conv(h)
        #     # h = F.relu(h, inplace=True)
            h = torch.bmm(identity.view(N, C, H*W).contiguous(), h_rgb)

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
        h = h.view(N, C, H, W).contiguous()  # N C H W
        # return F.relu(h, inplace=True)
        return h + identity

class DynamicMessagePassing(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.rgb_g_layers = nn.Sequential(
            nn.Linear(channels, channels),
            # nn.ReLU(inplace=True),
            # nn.Linear(channels // 4, channels),
            # nn.ReLU(inplace=True)
        )
        self.gamma = Parameter(torch.zeros(1))

    def get_knn_indices(self, batch_mat, k):
        r = torch.bmm(batch_mat, batch_mat.permute(0, 2, 1))
        N = r.size()[0]
        HW = r.size()[1]
        # batch_indices = torch.zeros((N, HW, k)).cuda()

        r = -2 * r
        square = torch.sum(batch_mat * batch_mat, dim=-1)
        square = square.unsqueeze(dim=-1)
        square_t = square.permute(0, 2, 1)
        adj = square + r + square_t
        _, indices = torch.topk(adj, k=k, largest=False)
        return indices

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

    def forward(self, rgb_in, gnn_iterations=1, S=16):

        N = rgb_in.size()[0]
        C = rgb_in.size()[1]
        H = rgb_in.size()[2]
        W = rgb_in.size()[3]
        S = S
        identity = rgb_in

        rgb_in = rgb_in.permute(0, 2, 3, 1).view(N, H*W, C).contiguous()  # N H*W C
        # ir = ir.permute(0, 2, 3, 1).view(N*H*W, C).contiguous()  # N*H*W C

        indices = []
        for n in range(N):
            ind = torch.randperm(rgb_in.size(1))[:S]
            indices.append(ind)
        indices = torch.cat(indices, dim=0).view(N, -1)

        # ind_plus = torch.arange(N*H*W) * S
        # ind_plus = ind_plus.view(N*H*W, 1)
        # indices = indices + ind_plus
        # indices = self.get_knn_indices(rgb_in, k=S)

        h_rgb = rgb_in
        h_rgb = h_rgb.view(N * H * W , C)  # NHW C
        # h_ir = ir
        # h_ir = h_ir.view(N * H * W, C)  # NHW C

        for i in range(gnn_iterations):
        #     # do this for every  sample in batch, not nice, but I don't know how to use index_select batchwise
        #     # fetch features from nearest neighbors
            S_rgb = torch.index_select(h_rgb, 0, indices.view(-1).cuda()).view(N, S, C)  # N S C
            # ir_neighbor_features = torch.index_select(h_ir, 0, ir_knn).view(N*H*W, K, C)  # NHW K C

            adj = torch.matmul(h_rgb.view(N, H*W, C), S_rgb.permute(0, 2, 1))  # N HW S

            h_rgb = self.rgb_g_layers(S_rgb)  # N S C
            # ir_features = self.ir_g_layers(ir_features)

            h_rgb = torch.matmul(adj, h_rgb)  # N HW C

            # h_rgb = torch.matmul(rgb_in, h_rgb.permute(0, 2, 1))  # N HW S

            # h_rgb = F.softmax(h_rgb, dim=-1)
            # h_rgb = torch.matmul(h_rgb, S_rgb)  # N HW C

            h_rgb = h_rgb.view(N, H, W, C).permute(0, 3, 1, 2).contiguous()
            h_rgb = F.relu(identity + self.gamma * h_rgb)  # NHW C

        # h_rgb = h_rgb.permute(0, 2, 1).view(N, C, H, W).contiguous()  # N C H W
        return h_rgb

class DeepLabASPPResNetGnn(BaseNet):
    def __init__(self, n_classes, backbone='resnet50', norm_layer=None, **kwargs):
        super(DeepLabASPPResNetGnn, self).__init__(n_classes, backbone=backbone, norm_layer=norm_layer, **kwargs)
        # freeze_backbone_bn(self.resnet_rgb)
        # freeze_backbone_bn(self.resnet_ir)

        self.conv11 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv21 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv22 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # self.gcn1 = GraphConvolution(256, 256)
        # self.gcn2 = GraphConvolution(256, 256)
        # self.gcn3 = GraphConvolution(256, 256)
        # self.feat2graph = Featuremaps_to_Graph(256, 256, nodes=13)
        # self.graph2feat = Graph_to_Featuremaps(256, 256, 256)

        # self.dy = DynamicMessagePassing(512)
        # self.down1 = nn.Conv2d(2048, 512, kernel_size=1)
        # self.down2 = nn.Conv2d(2048, 512, kernel_size=1)
        # self.up1 = nn.Conv2d(512, 2048, kernel_size=1)
        # self.up2 = nn.Conv2d(512, 2048, kernel_size=1)

        # self.att = PAM_Module(256)
        # self.att3 = PAM_Module(1024)
        # self.att4 = PAM_Module(512)
        # self.att = CAM_Module(256)
        # self.dahead = DANetHead(2048, 13)

        # self.non1 = NonLocalBlockND(256)
        # self.non2 = NonLocalBlockND(1024)
        # self.non3 = NonLocalBlockND(2048)
        # self.non_rgb_ir = NonLocalRGBIR(512)

        self.glore1 = GCNetHead(256, 64, 1)
        # self.glore2 = GCNetHead(256, 64, 1)

        # self.dgraph1 = DaulGraph(256, 64)
        # self.dgraph2 = DaulGraph(256, 64)

        # self.glore_fusion = GloRe_Fusion(256, 64, 1)

        # self.latentgnn = LatentGNN(512, nodes=16)

        self.score1 = nn.Sequential(nn.Dropout2d(0.1, False),
                                    nn.Conv2d(256, n_classes, 1))

        # self.dsn1 = nn.Conv2d(256, n_classes, 1)
        # self.dsn2 = nn.Conv2d(512, n_classes, 1)
        # self.dsn3 = nn.Conv2d(1024, n_classes, 1)
        # self.dsn4 = nn.Conv2d(2048, 1, 1)
        # self.gnn1 = EnetGnn(512)
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
        rgb1, rgb2, rgb3, rgb4, ir1, ir2, ir3, ir4 = self.base_forward(rgb, ir)

        rgb = self.conv11(rgb4)
        # ir = self.conv21(ir4)

        # rgb_graph = self.feat2graph(rgb)
        # rgb_graph = self.gcn1(rgb_graph, adj, relu=True)
        # rgb_graph = self.gcn2(rgb_graph, adj, relu=True)
        # rgb_graph = self.gcn3(rgb_graph, adj, relu=True)
        # rgb = self.graph2feat(rgb_graph, rgb)

        # rgb = self.glore1(rgb)
        # rgb = self.non1(rgb)
        # ir = self.glore2(ir)
        # ir = self.non1(ir)
        # rgb = self.dgraph1(rgb)
        # ir = self.dgraph2(ir)
        # rgb, ir = self.glore_fusion(rgb, ir)
        # rgb = self.att(rgb)
        # rgb = self.glore_fusion(rgb, ir)
        # rgb = rgb + ir

        # rgb = self.conv12(rgb)
        # ir = self.conv22(ir)

        # out = rgb + ir

        out = self.score1(rgb)
        # ir = self.score1(ir)

        out = F.interpolate(out, size=(h, w), **self._up_kwargs)
        # ir = F.interpolate(ir, size=(h, w), mode='bilinear', align_corners=True)
        # dsn1 = F.interpolate(self.dsn1(rgb1), size=(h, w), **self._up_kwargs)
        # dsn2 = F.interpolate(self.dsn2(rgb2), size=(h, w), **self._up_kwargs)
        # dsn3 = F.interpolate(self.dsn3(rgb3), size=(h, w), **self._up_kwargs)
        output = [out]
        # output = [ir]

        return output

    def get_parameters(self, key):
        for m in self.named_modules():
            if key == '1x':
                if 'backbone' in m[0] and isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        yield p
            elif key == '10x':
                # if 'backbone' not in m[0] and isinstance(m[1], (nn.Conv2d, nn.BatchNorm2d)):
                if 'backbone' not in m[0] and isinstance(m[1], (nn.Conv2d, nn.BatchNorm2d, nn.Conv1d)):
                    for p in m[1].parameters():
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
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(UpLayer, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.skip = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.skip(identity)
           # 交换激活到上采样后

        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
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
