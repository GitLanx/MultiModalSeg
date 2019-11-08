import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class EncoderDecoder(nn.Module):
    def __init__(self, n_classes):
        super(EncoderDecoder, self).__init__()

        # conv1
        features1 = []
        features1.append(nn.Conv2d(3, 64, 3, padding=1))
        features1.append(nn.BatchNorm2d(64))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.Conv2d(64, 64, 3, padding=1))
        features1.append(nn.BatchNorm2d(64))
        features1.append(nn.ReLU(inplace=True))
        self.features1 = nn.Sequential(*features1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2
        
        # conv2
        features2 = []
        features2.append(nn.Conv2d(64, 128, 3, padding=1))
        features2.append(nn.BatchNorm2d(128))
        features2.append(nn.ReLU(inplace=True))
        features2.append(nn.Conv2d(128, 128, 3, padding=1))
        features2.append(nn.BatchNorm2d(128))
        features2.append(nn.ReLU(inplace=True))
        self.features2 = nn.Sequential(*features2)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        features3 = []
        features3.append(nn.Conv2d(128, 256, 3, padding=1))
        features3.append(nn.BatchNorm2d(256))
        features3.append(nn.ReLU(inplace=True))
        features3.append(nn.Conv2d(256, 256, 3, padding=1))
        features3.append(nn.BatchNorm2d(256))
        features3.append(nn.ReLU(inplace=True))
        features3.append(nn.Conv2d(256, 256, 3, padding=1))
        features3.append(nn.BatchNorm2d(256))
        features3.append(nn.ReLU(inplace=True))
        self.features3 = nn.Sequential(*features3)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        features4 = []
        features4.append(nn.Conv2d(256, 512, 3, padding=1))
        features4.append(nn.BatchNorm2d(512))
        features4.append(nn.ReLU(inplace=True))
        features4.append(nn.Conv2d(512, 512, 3, padding=1))
        features4.append(nn.BatchNorm2d(512))
        features4.append(nn.ReLU(inplace=True))
        features4.append(nn.Conv2d(512, 512, 3, padding=1))
        features4.append(nn.BatchNorm2d(512))
        features4.append(nn.ReLU(inplace=True))
        self.features4 = nn.Sequential(*features4)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        features5 = []
        features5.append(nn.Conv2d(512, 512, 3, padding=1))
        features5.append(nn.BatchNorm2d(512))
        features5.append(nn.ReLU(inplace=True))
        features5.append(nn.Conv2d(512, 512, 3, padding=1))
        features5.append(nn.BatchNorm2d(512))
        features5.append(nn.ReLU(inplace=True))
        features5.append(nn.Conv2d(512, 512, 3, padding=1))
        features5.append(nn.BatchNorm2d(512))
        features5.append(nn.ReLU(inplace=True))
        self.features5 = nn.Sequential(*features5)

        self.fc = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # convTranspose1
        self.unpool6 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        features6 = []
        features6.append(nn.Conv2d(1024, 512, 3, padding=1))
        features6.append(nn.BatchNorm2d(512))
        features6.append(nn.ReLU(inplace=True))
        features6.append(nn.Conv2d(512, 256, 3, padding=1))
        features6.append(nn.BatchNorm2d(256))
        features6.append(nn.ReLU(inplace=True))
        self.features6 = nn.Sequential(*features6)

        # convTranspose2
        self.unpool7 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        features7 = []
        features7.append(nn.Conv2d(512, 256, 3, padding=1))
        features7.append(nn.BatchNorm2d(256))
        features7.append(nn.ReLU(inplace=True))
        features7.append(nn.Conv2d(256, 128, 3, padding=1))
        features7.append(nn.BatchNorm2d(128))
        features7.append(nn.ReLU(inplace=True))
        self.features7 = nn.Sequential(*features7)

        # convTranspose3
        self.unpool8 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        features8 = []
        features8.append(nn.Conv2d(256, 128, 3, padding=1))
        features8.append(nn.BatchNorm2d(128))
        features8.append(nn.ReLU(inplace=True))
        features8.append(nn.Conv2d(128, 64, 3, padding=1))
        features8.append(nn.BatchNorm2d(64))
        features8.append(nn.ReLU(inplace=True))
        self.features8 = nn.Sequential(*features8)

        # convTranspose5
        self.unpool9 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.final = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=3, padding=1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)
            # if isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0.001)

        vgg16 = torchvision.models.vgg16_bn(pretrained=True)
        vgg_features = [
            vgg16.features[0:6],
            vgg16.features[7:13],
            vgg16.features[14:23],
            vgg16.features[24:33],
            vgg16.features[34:43]
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
                    ll2.weight.data = ll1.weight.data
                    ll2.bias.data = ll1.bias.data
                if isinstance(ll1, nn.BatchNorm2d) and isinstance(ll2, nn.BatchNorm2d):
                    assert ll1.weight.size() == ll2.weight.size()
                    assert ll1.bias.size() == ll2.bias.size()
                    ll2.weight.data = ll1.weight.data
                    ll2.bias.data = ll1.bias.data

    def forward(self, x):
        out = self.features1(x)
        i1 = out
        out = self.pool1(out)
        out = self.features2(out)
        i2 = out
        out = self.pool2(out)
        out = self.features3(out)
        i3 = out
        out = self.pool3(out)
        out = self.features4(out)
        i4 = out
        out = self.pool4(out)
        out = self.features5(out)
        out = self.fc(out)

        # out = self.unpool6(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.features6(torch.cat((out, i4), dim=1))
        del i4
        # out = self.unpool7(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.features7(torch.cat((out, i3), dim=1))
        del i3
        # out = self.unpool8(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.features8(torch.cat((out, i2), dim=1))
        del i2
        # out = self.unpool9(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.final(torch.cat((out, i1), dim=1))
        del i1

        return out


class EncoderDecoder1(nn.Module):
    def __init__(self, n_classes):
        super(EncoderDecoder, self).__init__()

        # conv1
        features1 = []
        features1.append(nn.Conv2d(3, 64, 3, padding=1))
        features1.append(nn.BatchNorm2d(64))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.Conv2d(64, 64, 3, padding=1))
        features1.append(nn.BatchNorm2d(64))
        features1.append(nn.ReLU(inplace=True))
        self.features1 = nn.Sequential(*features1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2
        
        # conv2
        features2 = []
        features2.append(nn.Conv2d(64, 128, 3, padding=1))
        features2.append(nn.BatchNorm2d(128))
        features2.append(nn.ReLU(inplace=True))
        features2.append(nn.Conv2d(128, 128, 3, padding=1))
        features2.append(nn.BatchNorm2d(128))
        features2.append(nn.ReLU(inplace=True))
        self.features2 = nn.Sequential(*features2)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        features3 = []
        features3.append(nn.Conv2d(128, 256, 3, padding=1))
        features3.append(nn.BatchNorm2d(256))
        features3.append(nn.ReLU(inplace=True))
        features3.append(nn.Conv2d(256, 256, 3, padding=1))
        features3.append(nn.BatchNorm2d(256))
        features3.append(nn.ReLU(inplace=True))
        features3.append(nn.Conv2d(256, 256, 3, padding=1))
        features3.append(nn.BatchNorm2d(256))
        features3.append(nn.ReLU(inplace=True))
        self.features3 = nn.Sequential(*features3)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        features4 = []
        features4.append(nn.Conv2d(256, 512, 3, padding=1))
        features4.append(nn.BatchNorm2d(512))
        features4.append(nn.ReLU(inplace=True))
        features4.append(nn.Conv2d(512, 512, 3, padding=1))
        features4.append(nn.BatchNorm2d(512))
        features4.append(nn.ReLU(inplace=True))
        features4.append(nn.Conv2d(512, 512, 3, padding=1))
        features4.append(nn.BatchNorm2d(512))
        features4.append(nn.ReLU(inplace=True))
        self.features4 = nn.Sequential(*features4)
        self.pool4 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)  # 1/16

        # conv5
        features5 = []
        features5.append(nn.Conv2d(512, 512, 3, padding=1))
        features5.append(nn.BatchNorm2d(512))
        features5.append(nn.ReLU(inplace=True))
        features5.append(nn.Conv2d(512, 512, 3, padding=1))
        features5.append(nn.BatchNorm2d(512))
        features5.append(nn.ReLU(inplace=True))
        features5.append(nn.Conv2d(512, 512, 3, padding=1))
        features5.append(nn.BatchNorm2d(512))
        features5.append(nn.ReLU(inplace=True))
        self.features5 = nn.Sequential(*features5)

        self.fc = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # convTranspose1
        features6 = []
        features6.append(nn.Conv2d(512, 512, 3, padding=1))
        features6.append(nn.BatchNorm2d(512))
        features6.append(nn.ReLU(inplace=True))
        features6.append(nn.Conv2d(512, 256, 3, padding=1))
        features6.append(nn.BatchNorm2d(256))
        features6.append(nn.ReLU(inplace=True))
        self.features6 = nn.Sequential(*features6)

        # convTranspose2
        features7 = []
        features7.append(nn.Conv2d(256, 256, 3, padding=1))
        features7.append(nn.BatchNorm2d(256))
        features7.append(nn.ReLU(inplace=True))
        features7.append(nn.Conv2d(256, 128, 3, padding=1))
        features7.append(nn.BatchNorm2d(128))
        features7.append(nn.ReLU(inplace=True))
        self.features7 = nn.Sequential(*features7)

        # convTranspose3
        features8 = []
        features8.append(nn.Conv2d(128, 128, 3, padding=1))
        features8.append(nn.BatchNorm2d(128))
        features8.append(nn.ReLU(inplace=True))
        features8.append(nn.Conv2d(128, 64, 3, padding=1))
        features8.append(nn.BatchNorm2d(64))
        features8.append(nn.ReLU(inplace=True))
        self.features8 = nn.Sequential(*features8)

        # convTranspose4
        self.final = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=3, padding=1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)
            # if isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0.001)

        vgg16 = torchvision.models.vgg16_bn(pretrained=True)
        vgg_features = [
            vgg16.features[0:6],
            vgg16.features[7:13],
            vgg16.features[14:23],
            vgg16.features[24:33],
            vgg16.features[34:43]
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
                    ll2.weight.data = ll1.weight.data
                    ll2.bias.data = ll1.bias.data
                if isinstance(ll1, nn.BatchNorm2d) and isinstance(ll2, nn.BatchNorm2d):
                    assert ll1.weight.size() == ll2.weight.size()
                    assert ll1.bias.size() == ll2.bias.size()
                    ll2.weight.data = ll1.weight.data
                    ll2.bias.data = ll1.bias.data

    def forward(self, x):
        out = self.features1(x)
        i1 = out
        out = self.pool1(out)
        out = self.features2(out)
        i2 = out
        out = self.pool2(out)
        out = self.features3(out)
        i3 = out
        out = self.pool3(out)
        out = self.features4(out)
        i4 = out
        out = self.pool4(out)
        out = self.features5(out)
        out = self.fc(out)

        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.features6(out + i4)
        del i4
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.features7(out + i3)
        del i3
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.features8(out + i2)
        del i2
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.final(out + i1)
        return out

if __name__ == "__main__":
    import torch
    import time
    model = EncoderDecoder(13)
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
