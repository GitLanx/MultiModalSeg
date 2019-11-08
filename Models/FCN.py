import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
import sklearn
import numpy as np
import scipy.sparse as sp
import scipy
import math
from torch.nn.parameter import Parameter


# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


def freeze(m):
    for p in m.parameters():
        p.requires_grad = False

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            # nn.init.constant_(self.W.weight, 0)
            # nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

def grid(h, w, dtype=np.float32):
    """Return the embedding of a grid graph."""
    M = h * w
    x = np.linspace(0, 1, w, dtype=dtype)
    y = np.linspace(0, 1, h, dtype=dtype)
    xx, yy = np.meshgrid(x, y)
    z = np.empty((M, 2), dtype)
    z[:, 0] = xx.reshape(M)
    z[:, 1] = yy.reshape(M)
    return z

def distance_sklearn_metrics(z, k=4, metric='euclidean'):
    """Compute exact pairwise distances."""
    # d = sklearn.metrics.pairwise.pairwise_distances(
    #         z, metric=metric, n_jobs=-2)
    # d = scipy.spatial.distance.pdist(z, 'euclidean')
    # d = scipy.spatial.distance.squareform(d)

    z = torch.from_numpy(z).cuda()
    r = torch.mm(z, z.permute(1, 0))
    N = r.size()[0]
    HW = r.size()[1]

    diag = r.diag().unsqueeze(0)
    diag = diag.expand_as(r)
    # compute the distance matrix
    D = (diag + diag.t() - 2 * r).sqrt()
    topk, indices = torch.topk(D, k=k + 1, largest=False)
    del D
    del diag
    del r

    return topk[:, 1:], indices[:, 1:]
    # k-NN graph.
    # idx = np.argsort(d)[:, 1:k+1]
    # d.sort()
    # d = d[:, 1:k+1]
    # return d, idx

def adjacency(dist, idx):
    """Return the adjacency matrix of a kNN graph."""
    M, k = dist.size()
    # assert M, k == idx.shape
    # assert dist.min() >= 0

    # Weights.
    sigma2 = torch.mean(dist[:, -1])**2
    dist = torch.exp(- dist**2 / sigma2)

    # Weight matrix.
    I = torch.arange(0, M).repeat_interleave(k).contiguous().view(1, -1).cuda()
    J = idx.contiguous().view(1, -1)
    V = dist.contiguous().view(-1)
    indices = torch.cat([I, J], dim=0)
    W = torch.sparse.FloatTensor(indices, V, torch.Size([M, M])).cuda()
    # W = scipy.sparse.coo_matrix((V.cpu().numpy(), (I.cpu().numpy(), J.cpu().numpy())), shape=(M, M))

    # No self-connections.
    # W.setdiag(1)

    # Non-directed graph.
    # bigger = W.T > W
    # W = W - W.multiply(bigger) + W.T.multiply(bigger)
    #
    # assert W.nnz % 2 == 0
    # assert np.abs(W - W.T).mean() < 1e-10
    # assert type(W) is scipy.sparse.csr.csr_matrix
    return W

def build_graph(x, k=4):
    N, C, H, W = x.size()
    # tempx = x.permute(0, 2, 3, 1).contiguous().view(-1, C).cpu().numpy()
    # graph = kneighbors_graph(tempx, 8, mode='connectivity', include_self=True)
    # h_idx = np.arange(H)
    # w_idx = np.arange(W)
    # a = np.reshape(np.meshgrid(h_idx, w_idx), (2, -1)).T
    # graph = kneighbors_graph(a, 4, mode='connectivity', include_self=True)
    # adj = graph.tocoo()
    # rowsum = np.array(graph.sum(1))
    # d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # adj = graph.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    # adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    # indices = torch.from_numpy(
    # np.vstack((adj.row, adj.col)).astype(np.int64))
    # values = torch.from_numpy(adj.data)
    # shape = torch.Size(adj.shape)
    # return torch.from_numpy(adj).cuda()
    graph = grid(H, W)
    # graph = x
    # graph = graph.permute(0, 2, 3, 1).contiguous().view(N*H*W, C)
    dist, idx = distance_sklearn_metrics(graph, k=4, metric='euclidean')
    A = adjacency(dist, idx)
    A = A.to_dense() + torch.eye(A.size(0)).cuda()
    # A = laplacian(A, normalized=True)
    # A = A.tocoo()
    # rowsum = np.array(A.sum(1))
    # d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # A = A.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    D = torch.sum(A, dim=1)
    D = torch.pow(D, -0.5)
    D = torch.diag(D)
    A = torch.mm(torch.mm(A, D).t(), D)
    return A
    # indices = torch.from_numpy(
    # np.vstack((A.row, A.col)).astype(np.int64))
    # values = torch.from_numpy(A.data)
    # shape = torch.Size(A.shape)
    # return torch.sparse.FloatTensor(indices, values, shape).cuda()

def laplacian(W, normalized=True):
    """Return the Laplacian of the weigth matrix."""

    # Degree matrix.
    d = W.sum(dim=0)

    # Laplacian matrix.
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        # d += np.spacing(np.array(0, W.dtype))
        d = 1 / torch.sqrt(d)
        D = torch.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is scipy.sparse.csr.csr_matrix
    return L

class GCN(nn.Module):
    def __init__(self, nfeat, nhid):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)

    def forward(self, x, adj):
        identity = x
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        x = x + identity
        x = F.relu(x, inplace=True)
        return x

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        N, C, H, W = input.size()
        input = input.permute(0, 2, 3, 1).contiguous().view(-1, C)
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj.float(), support)
        if self.bias is not None:
            output =  output + self.bias
        output = output.view(N, H, W, C).permute(0, 3, 1, 2).contiguous()
        return output

class FCN8sAtOnce(nn.Module):
    def __init__(self, n_classes):
        super(FCN8sAtOnce, self).__init__()

        features1 = []
        # conv1
        features1.append(nn.Conv2d(3, 64, 3, padding=100))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.Conv2d(64, 64, 3, padding=1))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))  # 1/2

        # conv2
        features1.append(nn.Conv2d(64, 128, 3, padding=1))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.Conv2d(128, 128, 3, padding=1))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))  # 1/4

        # conv3
        features1.append(nn.Conv2d(128, 256, 3, padding=1))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.Conv2d(256, 256, 3, padding=1))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.Conv2d(256, 256, 3, padding=1))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))  # 1/8
        self.features1 = nn.Sequential(*features1)

        features2 = []
        # conv4
        features2.append(nn.Conv2d(256, 512, 3, padding=1))
        features2.append(nn.ReLU(inplace=True))
        features2.append(nn.Conv2d(512, 512, 3, padding=1))
        features2.append(nn.ReLU(inplace=True))
        features2.append(nn.Conv2d(512, 512, 3, padding=1))
        features2.append(nn.ReLU(inplace=True))
        features2.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))  # 1/16
        self.features2 = nn.Sequential(*features2)

        features3 = []
        # conv5
        features3.append(nn.Conv2d(512, 512, 3, padding=1))
        features3.append(nn.ReLU(inplace=True))
        features3.append(nn.Conv2d(512, 512, 3, padding=1))
        features3.append(nn.ReLU(inplace=True))
        features3.append(nn.Conv2d(512, 512, 3, padding=1))
        features3.append(nn.ReLU(inplace=True))
        features3.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))  # 1/32
        self.features3 = nn.Sequential(*features3)

        fc = []
        # fc6
        fc.append(nn.Conv2d(512, 4096, 7))
        fc.append(nn.ReLU(inplace=True))
        fc.append(nn.Dropout2d())

        # fc7
        fc.append(nn.Conv2d(4096, 4096, 1))
        fc.append(nn.ReLU(inplace=True))
        fc.append(nn.Dropout2d())
        self.fc = nn.Sequential(*fc)

        self.score_fr = nn.Conv2d(4096, n_classes, 1)
        self.score_pool3 = nn.Conv2d(256, n_classes, 1)
        self.score_pool4 = nn.Conv2d(512, n_classes, 1)

        self.upscore2 = nn.ConvTranspose2d(
            n_classes, n_classes, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            n_classes, n_classes, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_classes, n_classes, 4, stride=2, bias=False)
        freeze(self.upscore2)
        freeze(self.upscore8)
        freeze(self.upscore_pool4)

        # self.non_local = _NonLocalBlockND(256)
        self.non_local2 = _NonLocalBlockND(512, bn_layer=False)
        # self.gnn1 = GCN(256, 256)
        # self.gnn2 = GCN(512, 512)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in [self.score_fr, self.score_pool3, self.score_pool4]:
            m.weight.data.zero_()
            m.bias.data.zero_()

        for m in [self.upscore2, self.upscore8, self.upscore_pool4]:
            assert m.kernel_size[0] == m.kernel_size[1]
            initial_weight = get_upsampling_weight(
                m.in_channels, m.out_channels, m.kernel_size[0])
            m.weight.data.copy_(initial_weight)
        
        vgg16 = torchvision.models.vgg16(pretrained=True)
        vgg_features = [
            vgg16.features[:17],
            vgg16.features[17:24],
            vgg16.features[24:],
        ]
        features = [
            self.features1,
            self.features2,
            self.features3,
        ]

        for l1, l2 in zip(vgg_features, features):
            for ll1, ll2 in zip(l1.children(), l2.children()):
                if isinstance(ll1, nn.Conv2d) and isinstance(ll2, nn.Conv2d):
                    assert ll1.weight.size() == ll2.weight.size()
                    assert ll1.bias.size() == ll2.bias.size()
                    ll2.weight.data.copy_(ll1.weight.data)
                    ll2.bias.data.copy_(ll1.bias.data)

                    # freeze(ll2)

        for l1, l2 in zip(vgg16.classifier.children(), self.fc):
            if isinstance(l1, nn.Linear) and isinstance(l2, nn.Conv2d):
                l2.weight.data.copy_(l1.weight.data.view(l2.weight.size()))
                l2.bias.data.copy_(l1.bias.data.view(l2.bias.size()))

                # freeze(l2)

    def forward(self, x):
        pool3 = self.features1(x)       # 1/8

        # graph = F.interpolate(x, size=(pool3.size()[2], pool3.size()[3]), mode='bilinear', align_corners=True)
        # adj = build_graph(graph, k=10)
        # pool3 = self.gnn1(pool3, adj)
        # pool3 = self.non_local(pool3)
        # pool3 = pool31 + pool32

        pool4 = self.features2(pool3)   # 1/16

        # graph = F.interpolate(x, size=(pool4.size()[2], pool4.size()[3]), mode='bilinear', align_corners=True)
        # adj = build_graph(graph, k=10)
        # pool4 = self.gnn2(pool4, adj)

        pool5 = self.features3(pool4)     # 1/32
        pool5 = self.non_local2(pool5)
        out = self.fc(pool5)
        out = self.score_fr(out)
        upscore2 = self.upscore2(out)   # 1/16

        # score_pool4 = self.score_pool4(pool4 * 0.01)  # XXX: scaling to train at once
        score_pool4 = self.score_pool4(pool4)
        score_pool4 = score_pool4[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        upscore_pool4 = self.upscore_pool4(upscore2 + score_pool4)  # 1/8
        del upscore2
        del score_pool4
        # score_pool3 = self.score_pool3(pool3 * 0.0001)  # XXX: scaling to train at once
        score_pool3 = self.score_pool3(pool3)
        score_pool3 = score_pool3[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        out = self.upscore8(upscore_pool4 + score_pool3)

        out = out[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return out

    def get_parameters(self):
        for m in self.modules():
            for p in m.parameters():
                if p.requires_grad:
                    yield p