import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.modules.utils import _pair, _quadruple
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from utils import vis

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


class FCN8sAtOnceMulti(nn.Module):
    def __init__(self, n_classes):
        super(FCN8sAtOnceMulti, self).__init__()

        # nn.ReLU(inplace=True)
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
        self.rgb_pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

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
        self.ir_pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

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
        rgb = self.features1(rgb)
        ir = self.ir_features1(ir)
        rgb = self.rgb_pool1(rgb)
        ir = self.ir_pool1(ir)

        rgb = self.features2(rgb)
        ir = self.ir_features2(ir)
        rgb = self.rgb_pool2(rgb)
        ir = self.ir_pool2(ir)

        rgb = self.features3(rgb)
        ir = self.ir_features3(ir)
        rgb = self.rgb_pool3(rgb + ir)
        # ir = self.ir_pool3(ir)
        pool3 = rgb             # 1/8

        rgb = self.features4(rgb)
        # ir = self.ir_features4(ir)
        rgb = self.rgb_pool4(rgb)
        # ir = self.ir_pool4(ir)
        pool4 = rgb             # 1/16

        rgb = self.features5(rgb)
        # ir = self.ir_features5(ir)
        rgb = self.rgb_pool5(rgb)   # 1/32
        
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
        score_pool3 = score_pool3[:, :,
              9:9 + out.size()[2],
              9:9 + out.size()[3]]
        out = self.upscore8(out + score_pool3)
        del score_pool3
        out = out[:, :, 31:31 + h, 31:31 + w].contiguous()

        return out

    def get_parameters(self, double=False):
        if double:
            for module in [self.ir_features1, self.ir_features2, self.ir_features3, self.ir_features4,
                           self.ir_features5]:
                for m in module.modules():
                    for p in m.parameters():
                        yield p
        else:
            for module in [self.features1, self.features2, self.features3, self.features4,
                        self.features5, self.fc, self.score_fr, self.score_pool3, self.score_pool4]:
                for m in module.modules():
                    for p in m.parameters():
                        yield p

if __name__ == "__main__":
    import torch
    import time
    model = FCN8sAtOnceMulti(13)
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