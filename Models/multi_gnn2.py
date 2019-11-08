import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.modules.utils import _pair, _quadruple
from torch.autograd import Variable
import numpy as np
# from torch_geometric.nn import MessagePassing
# from torch_geometric.nn.pool import knn_graph
import matplotlib.pyplot as plt
from utils import vis
from torch.nn import Parameter

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

def add_position(rgb, ir):
    '''N, C, H, W'''
    x = torch.arange(rgb.size()[2], dtype=torch.float32).view(1, 1, -1, 1).cuda()
    x = x.repeat_interleave(rgb.size()[3], dim=-1)
    # x = x.repeat_interleave(rgb.size()[0], dim=0) / max_dim
    x = x.repeat_interleave(rgb.size()[0], dim=0)
    y = torch.arange(rgb.size()[3], dtype=torch.float32).view(1, 1, 1, -1).cuda()
    y = y.repeat_interleave(rgb.size()[2], dim=-2)
    # y = y.repeat_interleave(rgb.size()[0], dim=0) / max_dim
    y = y.repeat_interleave(rgb.size()[0], dim=0)
    rgb = torch.cat((rgb, x, y), dim=1)   # N C H W
    ir = torch.cat((ir, x, y), dim=1)
    rgb_max_val, _ = torch.max(rgb, 2, keepdim=True)
    rgb_max_val, _ = torch.max(rgb_max_val, 3, keepdim=True)
    rgb_min_val, _ = torch.min(rgb, 2, keepdim=True)
    rgb_min_val, _ = torch.min(rgb_min_val, 3, keepdim=True)
    ir_max_val, _ = torch.max(ir, 2, keepdim=True)
    ir_max_val, _ = torch.max(ir_max_val, 3, keepdim=True)
    ir_min_val, _ = torch.min(ir, 2, keepdim=True)
    ir_min_val, _ = torch.min(ir_min_val, 3, keepdim=True)
    rgb = (rgb - rgb_min_val) / (rgb_max_val - rgb_min_val)
    ir = (ir - ir_min_val) / (ir_max_val - ir_min_val)
    return rgb, ir

def norm(feature_maps):
    '''N, HW C'''
    max_val, _ = torch.max(feature_maps, 1, keepdim=True)
    min_val, _ = torch.min(feature_maps, 1, keepdim=True)
    feature_maps = (feature_maps - min_val) / (max_val - min_val + 1e-6)
    return feature_maps

class FCN8sAtOnceMultiGnn2(nn.Module):
    def __init__(self, n_classes):
        super(FCN8sAtOnceMultiGnn2, self).__init__()

        # conv1
        features1 = []
        features1.append(nn.Conv2d(3, 64, 3, padding=100))
        features1.append(nn.LeakyReLU())
        features1.append(nn.Conv2d(64, 64, 3, padding=1))
        features1.append(nn.LeakyReLU())
        self.features1 = nn.Sequential(*features1)
        self.rgb_pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        features2 = []
        features2.append(nn.Conv2d(64, 128, 3, padding=1))
        features2.append(nn.LeakyReLU())
        features2.append(nn.Conv2d(128, 128, 3, padding=1))
        features2.append(nn.LeakyReLU())
        self.features2 = nn.Sequential(*features2)
        self.rgb_pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        features3 = []
        features3.append(nn.Conv2d(128, 256, 3, padding=1))
        features3.append(nn.LeakyReLU())
        features3.append(nn.Conv2d(256, 256, 3, padding=1))
        features3.append(nn.LeakyReLU())
        features3.append(nn.Conv2d(256, 256, 3, padding=1))
        features3.append(nn.LeakyReLU())
        self.features3 = nn.Sequential(*features3)
        self.rgb_pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        features4 = []
        features4.append(nn.Conv2d(256, 512, 3, padding=1))
        features4.append(nn.LeakyReLU())
        features4.append(nn.Conv2d(512, 512, 3, padding=1))
        features4.append(nn.LeakyReLU())
        features4.append(nn.Conv2d(512, 512, 3, padding=1))
        features4.append(nn.LeakyReLU())
        self.features4 = nn.Sequential(*features4)
        self.rgb_pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16
        
        # conv5
        features5 = []
        features5.append(nn.Conv2d(512, 512, 3, padding=1))
        features5.append(nn.LeakyReLU())
        features5.append(nn.Conv2d(512, 512, 3, padding=1))
        features5.append(nn.LeakyReLU())
        features5.append(nn.Conv2d(512, 512, 3, padding=1))
        features5.append(nn.LeakyReLU())
        self.features5 = nn.Sequential(*features5)
        self.rgb_pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        self.ir_features1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU()
        )
        self.ir_pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.ir_features2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.LeakyReLU()
        )
        self.ir_pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.ir_features3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LeakyReLU(),
        )
        self.ir_pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.ir_features4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(),
        )
        self.ir_pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.ir_features5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(),
        )
        self.ir_pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        fc = []
        # fc6
        fc.append(nn.Conv2d(512, 4096, 7))
        fc.append(nn.LeakyReLU())
        fc.append(nn.Dropout2d())

        # fc7
        fc.append(nn.Conv2d(4096, 4096, 1))
        fc.append(nn.LeakyReLU())
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

        self.upscore2.apply(freeze)
        self.upscore8.apply(freeze)
        self.upscore_pool4.apply(freeze)

        self._initialize_weights()

        self.gnn1 = EnetGnn(1, 256)
        self.gnn2 = EnetGnn(1, 512)
        self.gnn3 = EnetGnn(1, 512)

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

        for l1, l2 in zip(vgg16.classifier.children(), self.fc):
            if isinstance(l1, nn.Linear) and isinstance(l2, nn.Conv2d):
                l2.weight.data.copy_(l1.weight.data.view(l2.weight.size()))
                l2.bias.data.copy_(l1.bias.data.view(l2.bias.size()))

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

    def forward(self, rgb, ir):
        _, _, h, w = rgb.size()
        rgb_in, ir_in = rgb, ir
        rgb = self.features1(rgb)
        ir = self.ir_features1(ir)

        rgb = self.rgb_pool1(rgb)     # 1/2
        ir = self.ir_pool1(ir)

        rgb = self.features2(rgb)
        ir = self.ir_features2(ir)

        rgb = self.rgb_pool2(rgb)     # 1/4
        ir = self.ir_pool2(ir)

        rgb = self.features3(rgb)
        ir = self.ir_features3(ir)
        rgb = self.gnn1(rgb, ir, gnn_iterations=1, k=16)

        # rgb = self.rgb_pool3(rgb)     # 1/8
        # ir = self.ir_pool3(ir)
        pool3 = rgb

        rgb = self.features4(rgb)
        # ir = self.ir_features4(ir)
        # rgb = self.gnn2(rgb, ir, gnn_iterations=1, k=10)

        rgb = self.rgb_pool4(rgb)     # 1/16
        # ir = self.ir_pool4(ir)
        pool4 = rgb

        rgb = self.features5(rgb)
        # ir = self.ir_features5(ir)
        # rgb = self.gnn3(rgb, ir, gnn_iterations=1, k=10)
        rgb = self.rgb_pool5(rgb)     # 1/32
        # ir = self.ir_pool5(ir)
        
        out = self.fc(rgb)
        out = self.score_fr(out)
        out = self.upscore2(out)   # 1/16
        del rgb
        del ir

        # score_pool4 = self.score_pool4(pool4 * 0.01)  # XXX: scaling to train at once
        score_pool4 = self.score_pool4(pool4)
        score_pool4 = score_pool4[:, :, 5:5 + out.size()[2], 5:5 + out.size()[3]]
        out = self.upscore_pool4(out + score_pool4)  # 1/8
        del score_pool4

        # score_pool3 = self.score_pool3(pool3 * 0.0001)  # XXX: scaling to train at once
        score_pool3 = self.score_pool3(pool3)
        # score_pool3 = score_pool3 * gnn
        score_pool3 = score_pool3[:, :,
              9:9 + out.size()[2],
              9:9 + out.size()[3]]
        out = self.upscore8(out + score_pool3)
        del score_pool3

        out = out[:, :, 31:31 + h, 31:31 + w].contiguous()

        return out

    def get_parameters(self, double=False):
        if double:
            for module in [self.gnn1, self.gnn2, self.gnn3, self.ir_features1, self.ir_features2, 
                           self.ir_features3, self.ir_features4, self.ir_features5]:
                for m in module.modules():
                    for p in m.parameters():
                        yield p
        else:
            for module in [self.features1, self.features2, self.features3,
                           self.features4, self.features5, self.fc, self.score_fr, self.score_pool3,
                           self.score_pool4]:
                for m in module.modules():
                    for p in m.parameters():
                        if p.requires_grad:
                            yield p


class EnetGnn(nn.Module):
    def __init__(self, mlp_num_layers, channels):
        super().__init__()

        self.rgb_g_layers = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.LeakyReLU(),
            # nn.Linear(channels // 4, channels),
            # nn.LeakyReLU()
        )
        self.ir_g_layers = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.LeakyReLU(),
            # nn.Linear(channels // 4, channels),
            # nn.LeakyReLU()
        )
        self.se_rgb = nn.Sequential(
            nn.Linear(channels * 2, channels // 16),
            nn.LeakyReLU(),
            nn.Linear(channels // 16, channels),
            nn.Sigmoid()
            # nn.LeakyReLU()
        )
        self.se_ir = nn.Sequential(
            nn.Linear(channels * 2, channels // 16),
            nn.LeakyReLU(),
            nn.Linear(channels // 16, channels),
            nn.Sigmoid()
            # nn.LeakyReLU()
        )
        self.gamma1 = Parameter(torch.ones(1))
        self.gamma2 = Parameter(torch.ones(1))
        self.out_conv = nn.Linear(channels, channels, bias=False)
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

        for idx, val in enumerate(r):
            # get the diagonal elements
            diag = val.diag().unsqueeze(0)
            diag = diag.expand_as(val)
            # compute the distance matrix
            D = (diag + diag.t() - 2 * val).sqrt()
            topk, indices = torch.topk(D, k=k, largest=False)
            del D
            del diag
            del val
            batch_indices[idx] = indices.data
        return batch_indices

        # r = -2 * r
        # square = torch.sum(batch_mat * batch_mat, dim=-1)
        # square = square.unsqueeze(dim=-1)
        # square_t = square.permute(0, 2, 1)
        # adj = square + r + square_t
        # _, indices = torch.topk(adj, k=k, largest=False)
        # return indices

    def forward(self, rgb, ir, gnn_iterations, k):

        # extract for convenience
        rgb = F.max_pool2d(rgb, kernel_size=2, stride=2, ceil_mode=True)
        ir = F.max_pool2d(ir, kernel_size=2, stride=2, ceil_mode=True)
        # rgb = F.normalize(rgb, dim=1)
        # ir = F.normalize(ir, dim=1)
        # h = rgb + ir

        N = rgb.size()[0]
        C = rgb.size()[1]
        H = rgb.size()[2]
        W = rgb.size()[3]
        K = k

        rgb = rgb.view(N, C, H*W).permute(0, 2, 1).contiguous()  # N H*W C
        ir = ir.view(N, C, H*W).permute(0, 2, 1).contiguous()  # N H*W C
        # rgb = rgb.permute(0, 2, 3, 1).view(N*H*W, C).contiguous()  # N*H*W C
        # ir = ir.permute(0, 2, 3, 1).view(N*H*W, C).contiguous()  # N*H*W C

        # get k nearest neighbors
        # a = F.normalize(a, dim=-1)
        rgb_knn = self.get_knn_indices(F.normalize(rgb, dim=-1), k=k)  # N HW K
        rgb_knn = rgb_knn.view(N*H*W*K).long()  # NHWK
        # rgb_knn = rgb_knn.view(N, H*W, K).long()  # NHWK
        ir_knn = self.get_knn_indices(F.normalize(ir, dim=-1), k=k)  # N HW K
        ir_knn = ir_knn.view(N*H*W*K).long()  # NHWK
        # ir_knn = ir_knn.view(N, H*W, K).long()  # NHWK
        # rgb_knn = torch.cat([rgb_knn, ir_knn], dim=1)

        # knn vis
        # a = torch.zeros(H * W)
        # ind = rgb_knn[0, 1500, :]
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
        h_ir = ir
        h_ir = h_ir.view(N * H * W, C)  # NHW C

        for i in range(gnn_iterations):
        #     # do this for every  sample in batch, not nice, but I don't know how to use index_select batchwise
        #     # fetch features from nearest neighbors
            rgb_neighbor_features = torch.index_select(h_rgb, 0, rgb_knn).view(N*H*W, K, C)  # NHW K C
            ir_neighbor_features = torch.index_select(h_ir, 0, ir_knn).view(N*H*W, K, C)  # NHW K C

            # rgb_central = h_rgb.unsqueeze(dim=-2)
            # rgb_central = rgb_central.repeat_interleave(k, dim=-2)
            # ir_central = h_ir.unsqueeze(dim=-2)
            # ir_central = ir_central.repeat_interleave(k, dim=-2)
            # run neighbor features through MLP g and activation function
            rgb_features = torch.cat([rgb_neighbor_features, rgb_neighbor_features - ir_neighbor_features], dim=-1)
            ir_features = torch.cat([ir_neighbor_features, ir_neighbor_features - rgb_neighbor_features], dim=-1)

            rgb_features = self.rgb_g_layers(rgb_features)
            ir_features = self.ir_g_layers(ir_features)
            # ir_features = self.rgb_g_layers(ir_features)

        #     # average over activated neighbors
            # m_rgb, _ = torch.max(rgb_features, dim=1)  # NHW C
            # m_ir, _ = torch.max(ir_features, dim=1)
            m_rgb = torch.mean(rgb_features, dim=1)  # NHW C
            m_ir = torch.mean(ir_features, dim=1)

        #     # concatenate current state with messages
        #     # concat = torch.cat((h, m_rgb, m_ir), 1)  # N HW 3C
            concat = torch.cat((m_rgb, m_ir), 1)  # NHW 2C
            # concat = m_rgb + m_ir

            # attention = torch.bmm(m_rgb.permute(1, 0), m_ir)

        #     # se concat
            concat = concat.view(N, H*W, -1)    # N HW 2C
            concat = concat.mean(dim=1, keepdim=True)           # N 1 2C
            concat_rgb = self.se_rgb(concat)     # N 1 C
            # concat_ir = self.se_ir(concat)
            h_rgb = concat_rgb * h_rgb.view(N, H*W, C)
            h_ir = (1 - concat_rgb) * h_ir.view(N, H*W, C)
            h = self.gamma1 * h_rgb + self.gamma2 * h_ir
            # h = self.out_conv(h)
        #     # h = F.relu(h, inplace=True)

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
        h = h.view(N, H, W, C).permute(0, 3, 1, 2).contiguous()  # N C H W
        return F.relu(h, inplace=True)
        # return h


if __name__ == "__main__":
    import torch
    import time
    model = FCN8sAtOnceMultiGnn(13)
    print(f'==> Testing {model.__class__.__name__} with PyTorch')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.backends.cudnn.benchmark = True

    model = model.to(device)
    model.eval()

    x1 = torch.Tensor(1, 3, 320, 320)
    x2 = torch.Tensor(1, 3, 320, 320)
    x1 = x1.to(device)
    x2 = x2.to(device)

    torch.cuda.synchronize()
    t_start = time.time()
    for i in range(10):
        model(x1, x2)
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start

    print(f'Speed: {(elapsed_time / 10) * 1000:.2f} ms')