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


class FCN8sAtOnceMultiGnn(nn.Module):
    def __init__(self, n_classes):
        super(FCN8sAtOnceMultiGnn, self).__init__()

        # conv1
        features1 = []
        features1.append(nn.Conv2d(3, 64, 3, padding=100))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.Conv2d(64, 64, 3, padding=1))
        features1.append(nn.ReLU(inplace=True))
        self.features1 = nn.Sequential(*features1)
        self.rgb_pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        features2 = []
        features2.append(nn.Conv2d(64, 128, 3, padding=1))
        features2.append(nn.ReLU(inplace=True))
        features2.append(nn.Conv2d(128, 128, 3, padding=1))
        features2.append(nn.ReLU(inplace=True))
        self.features2 = nn.Sequential(*features2)
        self.rgb_pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        features3 = []
        features3.append(nn.Conv2d(128, 256, 3, padding=1))
        features3.append(nn.ReLU(inplace=True))
        features3.append(nn.Conv2d(256, 256, 3, padding=1))
        features3.append(nn.ReLU(inplace=True))
        features3.append(nn.Conv2d(256, 256, 3, padding=1))
        features3.append(nn.ReLU(inplace=True))
        self.features3 = nn.Sequential(*features3)
        self.rgb_pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        features4 = []
        features4.append(nn.Conv2d(256, 512, 3, padding=1))
        features4.append(nn.ReLU(inplace=True))
        features4.append(nn.Conv2d(512, 512, 3, padding=1))
        features4.append(nn.ReLU(inplace=True))
        features4.append(nn.Conv2d(512, 512, 3, padding=1))
        features4.append(nn.ReLU(inplace=True))
        self.features4 = nn.Sequential(*features4)
        self.rgb_pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16
        
        # conv5
        features5 = []
        features5.append(nn.Conv2d(512, 512, 3, padding=1))
        features5.append(nn.ReLU(inplace=True))
        features5.append(nn.Conv2d(512, 512, 3, padding=1))
        features5.append(nn.ReLU(inplace=True))
        features5.append(nn.Conv2d(512, 512, 3, padding=1))
        features5.append(nn.ReLU(inplace=True))
        self.features5 = nn.Sequential(*features5)
        self.rgb_pool5 = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True)  # 1/32

        self.ir_features1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.ir_pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.ir_features2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.ir_pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.ir_features3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.ir_pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.ir_features4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.ir_pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.ir_features5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.ir_pool5 = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True)

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
        # ir1 = ir
        rgb = self.rgb_pool1(rgb)     # 1/2
        ir = self.ir_pool1(ir)
        # del ir1

        rgb = self.features2(rgb)
        ir = self.ir_features2(ir)
        # ir2 = ir
        rgb = self.rgb_pool2(rgb)     # 1/4
        ir = self.ir_pool2(ir)
        # del ir2

        rgb = self.features3(rgb)
        ir = self.ir_features3(ir)
        # ir3 = ir
        rgb = self.rgb_pool3(rgb + ir)     # 1/8
        # ir = self.ir_pool3(ir)
        # del ir3

        rgb_in = F.interpolate(rgb_in, size=(rgb.size()[2], rgb.size()[3]), mode='bilinear', align_corners=True)
        ir_in = F.interpolate(ir_in, size=(rgb.size()[2], rgb.size()[3]), mode='bilinear', align_corners=True)
        rgb_in, ir_in = add_position(rgb_in, ir_in)
        rgb = self.gnn1(rgb, rgb_in, ir_in, gnn_iterations=1, k=10)
        pool3 = rgb

        rgb = self.features4(rgb)
        # ir = self.ir_features4(ir)
        # ir4 = ir
        rgb = self.rgb_pool4(rgb)     # 1/16
        # ir = self.ir_pool4(ir)
        # del ir4
        # rgb_in = F.interpolate(rgb_in, size=(rgb.size()[2], rgb.size()[3]), mode='bilinear', align_corners=True)
        # ir_in = F.interpolate(ir_in, size=(rgb.size()[2], rgb.size()[3]), mode='bilinear', align_corners=True)
        # rgb_in, ir_in = add_position(rgb_in, ir_in)
        # rgb = self.gnn2(rgb, rgb_in, ir_in, gnn_iterations=1, k=10)
        pool4 = rgb

        rgb = self.features5(rgb)
        # ir = self.ir_features5(ir)
        # ir5 = ir
        rgb = self.rgb_pool5(rgb)     # 1/32
        # ir = self.ir_pool5(ir)
        # del ir5
        # rgb_in = F.interpolate(rgb_in, size=(rgb.size()[2], rgb.size()[3]), mode='bilinear', align_corners=True)
        # ir_in = F.interpolate(ir_in, size=(rgb.size()[2], rgb.size()[3]), mode='bilinear', align_corners=True)
        # rgb_in, ir_in = add_position(rgb_in, ir_in)
        # rgb = self.gnn3(rgb, rgb_in, ir_in, gnn_iterations=1, k=10)
        
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
            for module in [self.gnn1, self.gnn2, self.gnn3, self.ir_features1, self.ir_features2, self.ir_features3]:
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

        # self.rgb_g_layers = nn.Sequential(
        #     nn.Linear(channels * 1, channels),
        #     nn.ReLU(inplace=True)
        # )
        # self.ir_g_layers = nn.Sequential(
        #     nn.Linear(channels * 1, channels),
        #     nn.ReLU(inplace=True)
        # )
        self.rgb_g_rnn_layers = nn.ModuleList([nn.Linear(channels * 2, channels) for l in range(mlp_num_layers)])
        self.rgb_g_rnn_actfs = nn.ModuleList([nn.ReLU(inplace=True) for l in range(mlp_num_layers)])
        self.rgb_q_rnn_layer = nn.Linear(channels * 3, channels)
        self.rgb_q_rnn_actf = nn.ReLU(inplace=True)
        self.ir_g_rnn_layers = nn.ModuleList([nn.Linear(channels * 2, channels) for l in range(mlp_num_layers)])
        self.ir_g_rnn_actfs = nn.ModuleList([nn.ReLU(inplace=True) for l in range(mlp_num_layers)])
        self.se = nn.Sequential(
            nn.Linear(channels * 2, channels // 16),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 16, channels),
            nn.Sigmoid()
            # nn.ReLU(inplace=True)
        )
        self.gamma = Parameter(torch.ones(1))
        # self.ir_q_rnn_layer = nn.Linear(channels * 2, channels)
        # self.ir_q_rnn_actf = nn.ReLU(inplace=True)
        # self.output_conv = nn.Conv2d(channels, 13, 1, stride=1, padding=0, bias=True)

    # adapted from https://discuss.pytorch.org/t/build-your-own-loss-function-in-pytorch/235/6
    # (x - y)^2 = x^2 - 2*x*y + y^2
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

    def forward(self, cnn_encoder_output, rgb, ir, gnn_iterations, k):

        # extract for convenience
        N = cnn_encoder_output.size()[0]
        C = cnn_encoder_output.size()[1]
        H = cnn_encoder_output.size()[2]
        W = cnn_encoder_output.size()[3]
        K = k

        rgb = rgb.view(N, rgb.size()[1], H*W).permute(0, 2, 1).contiguous()  # N H*W 5
        ir = ir.view(N, ir.size()[1], H*W).permute(0, 2, 1).contiguous()  # N H*W 5

        # cnn_knn = cnn_encoder_output.view(N, cnn_encoder_output.size()[1], H*W).permute(0, 2, 1).contiguous()

        # get k nearest neighbors
        # a = F.normalize(a, dim=-1)
        rgb_knn = self.get_knn_indices(rgb, k=k)  # N HW K
        rgb_knn = rgb_knn.view(N,H*W*K).long()  # NHWK
        # rgb_knn = self.get_knn_indices(rgb, k=k)  # N HW K
        # rgb_knn = rgb_knn.view(N, H*W, K).long()  # NHWK
        ir_knn = self.get_knn_indices(ir, k=k)  # N HW K
        ir_knn = ir_knn.view(N,H*W*K).long()  # NHWK
        # ir_knn = ir_knn.view(N, H*W, K).long()  # NHWK
        # rgb_knn = torch.cat([rgb_knn, ir_knn], dim=1)

        # knn vis
        # a = torch.zeros(1, H * W)
        # ind = rgb_knn[0, 1090, :]
        # for i in ind:
        #     a[torch.tensor(0, dtype=torch.long), i.long()] = 1
        # a = a.view(H, W, 1)
        # plt.subplot(121)
        # rgb_vis = rgb.view(H, W, 5)
        # rgb_vis = 0.5 * 50 * a.cpu().numpy() + (1 - 0.5) * rgb_vis[:, :, :3].cpu().numpy()
        # plt.imshow(rgb_vis)
        # a = torch.zeros(1, H * W)
        # ind = ir_knn[0, 1090, :]
        # for i in ind:
        #     a[torch.tensor(0, dtype=torch.long), i.long()] = 1
        # a = a.view(H, W, 1)
        # plt.subplot(122)
        # ir_vis = ir.view(H, W, 5)
        # ir_vis = 0.5 * 50 * a.cpu().numpy() + (1 - 0.5) * ir_vis[:, :, :3].cpu().numpy()
        # plt.imshow(ir_vis)
        # plt.show()

        # prepare CNN encoded features for RNN
        h = cnn_encoder_output  # N C H W
        h = h.permute(0, 2, 3, 1).contiguous()  # N H W C
        h = h.view(N, H * W, C)  # NHW C

        # # loop over timestamps to unroll
        # h = cnn_encoder_output  # N C H W
        # h = h.permute(0, 2, 3, 1).contiguous()  # N H W C
        # h = h.view(N * H * W, C)  # NHW C
        # for i in range(gnn_iterations):
        #     # do this for every  sample in batch, not nice, but I don't know how to use index_select batchwise
        #     # fetch features from nearest neighbors
        #     rgb_neighbor_features = torch.index_select(h, 0, Variable(rgb_knn)).view(N*H*W, K, C)  # NHW K C
        #     ir_neighbor_features = torch.index_select(h, 0, Variable(ir_knn)).view(N*H*W, K, C)  # NHW K C
                    
        #     x_central = h.unsqueeze(dim=-2)
        #     x_central = x_central.repeat_interleave(k, dim=-2)
        #     # run neighbor features through MLP g and activation function
        #     # rgb_features = torch.cat([x_central, rgb_neighbor_features - x_central], dim=-1)
        #     # ir_features = torch.cat([x_central, ir_neighbor_features - x_central], dim=-1)

        #     # rgb_neighbor_features = self.rgb_g_layers(rgb_neighbor_features)
        #     # ir_neighbor_features = self.ir_g_layers(ir_neighbor_features)
        #     rgb_features = self.rgb_g_layers(rgb_neighbor_features)
        #     ir_features = self.ir_g_layers(ir_neighbor_features)

        #     # for idx, g_layer in enumerate(self.rgb_g_rnn_layers):
        #     #     rgb_neighbor_features = self.rgb_g_rnn_actfs[idx](g_layer(rgb_neighbor_features))  # NHW K C
        #     # for idx, g_layer in enumerate(self.ir_g_rnn_layers):
        #     #     ir_neighbor_features = self.ir_g_rnn_actfs[idx](g_layer(ir_neighbor_features))  # NHW K C

        #     # average over activated neighbors
        #     # m_rgb[n] = torch.mean(rgb_neighbor_features, dim=1)  # HW C
        #     # m_ir[n] = torch.mean(ir_neighbor_features, dim=1)
        #     # m_rgb, _ = torch.max(rgb_neighbor_features, dim=1)  # NHW C
        #     # m_ir, _ = torch.max(ir_neighbor_features, dim=1)
        #     m_rgb, _ = torch.max(rgb_features, dim=1)  # NHW C
        #     m_ir, _ = torch.max(ir_features, dim=1)

        #     # concatenate current state with messages
        #     # concat = torch.cat((h, m_rgb, m_ir), 1)  # N HW 3C
        #     # concat = torch.cat((m_rgb, m_ir), 1)  # N HW 2C

        #     # se concat
        #     concat = torch.cat((m_rgb, m_ir), 1)  # NHW 2C
        #     # concat = m_rgb
        #     concat = concat.view(N, H*W, -1)    # N HW 2C
        #     concat = concat.mean(dim=1, keepdim=True)           # N 1 2C
        #     concat = self.se(concat)
        #     h = concat * h.view(N, H*W, C)
        #     # h = F.relu(h, inplace=True)

        #     # add messages
        #     # concat = h + m_rgb + m_ir

        #     # get new features by running MLP q and activation function
        #     # h = self.rgb_q_rnn_actf(self.rgb_q_rnn_layer(concat))  # N HW C

        m_rgb = h.clone()
        m_ir = h.clone()
        for i in range(gnn_iterations):
            # do this for every  sample in batch, not nice, but I don't know how to use index_select batchwise
            for n in range(N):
                # fetch features from nearest neighbors
                rgb_neighbor_features = torch.index_select(h[n], 0, Variable(rgb_knn[n])).view(H*W, K, C)  # H*W K C
                ir_neighbor_features = torch.index_select(h[n], 0, Variable(ir_knn[n])).view(H*W, K, C)  # H*W K C
                # run neighbor features through MLP g and activation function
                rgb_features = torch.cat([rgb_neighbor_features, rgb_neighbor_features - ir_neighbor_features], dim=-1)
                ir_features = torch.cat([ir_neighbor_features, ir_neighbor_features - rgb_neighbor_features], dim=-1)
                for idx, g_layer in enumerate(self.rgb_g_rnn_layers):
                    rgb_features = self.rgb_g_rnn_actfs[idx](g_layer(rgb_features))  # HW K C
                for idx, g_layer in enumerate(self.ir_g_rnn_layers):
                    ir_features = self.ir_g_rnn_actfs[idx](g_layer(ir_features))  # HW K C
                # average over activated neighbors
                # m_rgb[n] = torch.mean(rgb_neighbor_features, dim=1)  # HW C
                # m_ir[n] = torch.mean(ir_neighbor_features, dim=1)
                m_rgb[n], _ = torch.max(rgb_features, dim=1)  # HW C
                m_ir[n], _ = torch.max(ir_features, dim=1)

            # concatenate current state with messages
            # concat = torch.cat((h, m_rgb, m_ir), 2)  # N HW 3C

            # se concat
            # concat = m_rgb
            concat = torch.cat((m_rgb, m_ir), 2)  # N HW 2C
            concat = concat.mean(dim=1, keepdim=True)           # N 1 2C
            concat = self.se(concat)    # N HW C
            # concat = torch.bmm(concat.permute(0, 2, 1), concat)
            # concat = torch.max(concat, -1, keepdim=True)[0].expand_as(concat)-concat
            # concat = F.softmax(concat, dim=-1)
            # h = torch.bmm(h, concat)
            h = concat * h
            # h = F.relu(h, inplace=True)

            # add messages
            # concat = h + m_rgb + m_ir

            # get new features by running MLP q and activation function
            # h = self.rgb_q_rnn_actf(self.rgb_q_rnn_layer(concat))  # N HW C

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
        # output = self.output_conv(torch.cat((cnn_encoder_output, h), 1))  # N 2C H W
        # output = self.output_conv(cnn_encoder_output + h)
        # output = self.output_conv(h)
        h = self.gamma * h + cnn_encoder_output
        # output = F.softmax(output, dim=1)
        # return output
        return F.relu(h, inplace=True)


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, edge_index, size):
        # N HW C
        # N HW K
        N, C, H, W = size
        x_central = x
        batch_size, num_points, num_dims = x.size()
        k = edge_index.size()[-1]
        idx_ = torch.arange(batch_size) * num_points
        idx_ = idx_.view(batch_size, 1, 1).cuda()
        edge_index = edge_index + idx_

        # edge_index = edge_index.view(N, H * W * self.k)
        #
        edge_index = edge_index.view(-1)

        x_flat = x.view(-1, num_dims)
        neighbors = torch.index_select(x_flat, 0, edge_index)
        neighbors = neighbors.view(batch_size, num_points, k, num_dims)
        x_central = x_central.unsqueeze(dim=-2)
        x_central = x_central.repeat_interleave(k, dim=-2)
        edge_feature = torch.cat([x_central, neighbors - x_central], dim=-1)  # batch, HW, k, C
        edge_feature = edge_feature.permute([0, 3, 1, 2])
        edge_feature = self.conv(edge_feature)
        edge_feature, _ = torch.max(edge_feature, dim=-1)
        edge_feature = edge_feature.view(batch_size, C, H, W)
        return edge_feature


class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels, k=8):
        super(DynamicEdgeConv, self).__init__(in_channels, out_channels)
        self.k = k

    def forward(self, x, batch=None):
        size = x.size()
        N, C, H, W = size
        x = x.view(N, C, H*W).permute(0, 2, 1).contiguous()
        edge_index = self.get_knn_indices(x, self.k)    # N, H*W, k
        return super(DynamicEdgeConv, self).forward(x, edge_index, size)
    
    def get_knn_indices(self, batch_mat, k):
        # batch_mat: N HW C
        r = torch.bmm(batch_mat, batch_mat.permute(0, 2, 1)) 
        N = r.size()[0]
        HW = r.size()[1]

        r = -2 * r
        square = torch.sum(batch_mat * batch_mat, dim=-1)
        square = square.unsqueeze(dim=-1)
        square_t = square.permute(0, 2, 1)
        adj = square + r + square_t
        _, indices = torch.topk(adj, k=k, largest=False)
        return indices
        # r = torch.bmm(batch_mat, batch_mat.permute(0, 2, 1)) 
        # N = r.size()[0]
        # HW = r.size()[1]
        # batch_indices = torch.zeros((N, HW, k)).cuda()

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