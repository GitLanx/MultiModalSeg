import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
import sklearn
import numpy as np
import scipy.sparse as sp
import scipy
import math
from torch.nn.parameter import Parameter


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

    # z = torch.from_numpy(z).cuda()
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

def laplacian(W, normalized=True):
    """Return the Laplacian of the weigth matrix."""

    # Degree matrix.
    d = W.sum(axis=0)

    # Laplacian matrix.
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is scipy.sparse.csr.csr_matrix
    return L

class DeepLabLargeFOV(nn.Module):
    """Adapted from official implementation:

    http://www.cs.jhu.edu/~alanlab/ccvl/DeepLab-LargeFOV/train.prototxt

     input dimension equal to
     n = 32 * k - 31, e.g., 321 (for k = 11)
     Dimension after pooling w. subsampling:
     (16 * k - 15); (8 * k - 7); (4 * k - 3); (2 * k - 1); (k).
     For k = 11, these translate to  
               161;          81;          41;          21;  11
    """
    def __init__(self, n_classes):
        super(DeepLabLargeFOV, self).__init__()

        features1 = []
        features1.append(nn.Conv2d(3, 64, 3, padding=1))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.Conv2d(64, 64, 3, padding=1))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True))
        self.features1 = nn.Sequential(*features1)

        features2 = []
        features2.append(nn.Conv2d(64, 128, 3, padding=1))
        features2.append(nn.ReLU(inplace=True))
        features2.append(nn.Conv2d(128, 128, 3, padding=1))
        features2.append(nn.ReLU(inplace=True))
        features2.append(nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True))
        self.features2 = nn.Sequential(*features2)

        features3 = []
        features3.append(nn.Conv2d(128, 256, 3, padding=1))
        features3.append(nn.ReLU(inplace=True))
        features3.append(nn.Conv2d(256, 256, 3, padding=1))
        features3.append(nn.ReLU(inplace=True))
        features3.append(nn.Conv2d(256, 256, 3, padding=1))
        features3.append(nn.ReLU(inplace=True))
        features3.append(nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True))
        self.features3 = nn.Sequential(*features3)

        features4 = []
        features4.append(nn.Conv2d(256, 512, 3, padding=1))
        features4.append(nn.ReLU(inplace=True))
        features4.append(nn.Conv2d(512, 512, 3, padding=1))
        features4.append(nn.ReLU(inplace=True))
        features4.append(nn.Conv2d(512, 512, 3, padding=1))
        features4.append(nn.ReLU(inplace=True))
        features4.append(nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True))
        self.features4 = nn.Sequential(*features4)

        features5 = []
        features5.append(nn.Conv2d(512, 512, 3, padding=2, dilation=2))
        features5.append(nn.ReLU(inplace=True))
        features5.append(nn.Conv2d(512, 512, 3, padding=2, dilation=2))
        features5.append(nn.ReLU(inplace=True))
        features5.append(nn.Conv2d(512, 512, 3, padding=2, dilation=2))
        features5.append(nn.ReLU(inplace=True))
        features5.append(nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True))
        self.features5 = nn.Sequential(*features5)

        fc = []
        fc.append(nn.AvgPool2d(3, stride=1, padding=1))
        fc.append(nn.Conv2d(512, 1024, 3, padding=12, dilation=12))
        fc.append(nn.ReLU(inplace=True))
        fc.append(nn.Conv2d(1024, 1024, 1))
        fc.append(nn.ReLU(inplace=True))
        fc.append(nn.Dropout(p=0.5))
        self.fc = nn.Sequential(*fc)

        self.score = nn.Conv2d(1024, n_classes, 1)
        self.gnn = GCN(256, 256)
        self._initialize_weights()

    def _initialize_weights(self):

        # vgg = torchvision.models.vgg16(pretrained=True)
        # state_dict = vgg.features.state_dict()
        # self.features.load_state_dict(state_dict)

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

        # for m in self.fc.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)
        #         nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.score.weight, std=0.01)
        nn.init.constant_(self.score.bias, 0)

    def forward(self, x):
        _, _, h, w = x.size()
        out = self.features1(x)
        out = self.features2(out)
        out = self.features3(out)

        # x = F.interpolate(x, size=(out.size()[2], out.size()[3]), mode='bilinear', align_corners=True)
        # adj = self.build_graph(x, k=10)
        # out = self.gnn(out, adj)

        # out = self.non_local(out)

        out = self.features4(out)
        out = self.features5(out)
        out = self.fc(out)
        out = self.score(out)
        out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
        return out

    def get_parameters(self, bias=False, score=False):
        # if score:
        #     if bias:
        #         yield self.score.bias
        #     else:
        #         yield self.score.weight
        # else:
        #     for module in [self.features1, self.features2, self.features3, self.features4,
        #                    self.features5, self.fc]:
        #         for m in module.modules():
        #             if isinstance(m, nn.Conv2d):
        #                 if bias:
        #                     yield m.bias
        #                 else:
        #                     yield m.weight
        pass

    def build_graph(self, x, k=4):
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
        graph = graph.permute(0, 2, 3, 1).contiguous().view(N*H*W, C)
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

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

if __name__ == "__main__":
    import torch
    import time
    model = DeepLabLargeFOV(13)
    print(f'==> Testing {model.__class__.__name__} with PyTorch')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    x = torch.Tensor(1, 3, 320, 320)
    x = x.to(device)

    torch.cuda.synchronize()
    t_start = time.time()
    for i in range(10):
        model(x)
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start

    print(f'Speed: {(elapsed_time / 10) * 1000:.2f} ms')
