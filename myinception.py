# Same as the version from the official start_epoch
# ttps://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
# Only change being that it can take variable sized inputs
# See line 122

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

__all__ = ['Inception3', 'inception_v3']

model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}


def inception_v3(pretrained=False, **kwargs):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        model = Inception3(**kwargs)
        model.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google'])) #pretrained from imaegnet
        return model

    return Inception3(**kwargs)


class Logits(nn.Module):
    def __init__(self, channels, num_classes):
        super(Logits, self).__init__()
        self.Conv2d_1c_1x1 = Conv2d_1c_1x1(channels, num_classes)

    def forward(self, x):
        return self.Conv2d_1c_1x1(x)


class Inception3(nn.Module):

    def __init__(self, num_classes=200, aux_logits=True, transform_input=False):
        super(Inception3, self).__init__()

        self.m_pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64, bug=True)

        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = AuxLogits(768, num_classes)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280, bug=True)
        self.Mixed_7c = InceptionE(2048)
        # self.Logits = nn.Linear(2048, num_classes)
        self.Logits = Logits(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.data.numel()))
                values = values.view(m.weight.data.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        if self.transform_input:
            #This code is required if model is imagenet pre-trained model
            x = x.clone()
            # x  = x - 0.5
            # x  = x * 2.0
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5

        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        #x = x.mean(2).mean(2).unsqueeze(2).unsqueeze(3) #global pool

        #x = F.adaptive_avg_pool2d(x, 1)
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = F.dropout(x, p=0.5, training=self.training)
        # 1 x 1 x 2048
       # x = x.view(x.size(0), -1)
        # 2048
        x = self.Logits(x)
        x = x.view(x.size(0), -1)
        # 1000 (num_classes)
        if self.training and self.aux_logits:
            return x, aux

        return x


class InceptionABranch0(nn.Module):
    def __init__(self, in_channels):
        super(InceptionABranch0, self).__init__()
        self.Conv2d_0a_1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

    def forward(self, x):
        return self.Conv2d_0a_1x1(x)


class InceptionABranch1(nn.Module):
    def __init__(self, in_channels, bug):
        super(InceptionABranch1, self).__init__()
        self.bug = bug
        if bug:
            self.Conv2d_0b_1x1 = BasicConv2d(in_channels, 48, kernel_size=1)
            self.Conv_1_0c_5x5 = BasicConv2d(48, 64, kernel_size=5, padding=2)
        else:

            self.Conv2d_0a_1x1 = BasicConv2d(in_channels, 48, kernel_size=1)
            self.Conv2d_0b_5x5 = BasicConv2d(48, 64, kernel_size=5, padding=2)

    def forward(self, x):
        if self.bug:
            return self.Conv_1_0c_5x5(self.Conv2d_0b_1x1(x))
        else:
            return self.Conv2d_0b_5x5(self.Conv2d_0a_1x1(x))


class InceptionABranch2(nn.Module):
    def __init__(self, in_channels):
        super(InceptionABranch2, self).__init__()
        self.Conv2d_0a_1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.Conv2d_0b_3x3 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.Conv2d_0c_3x3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

    def forward(self, x):
        return self.Conv2d_0c_3x3(self.Conv2d_0b_3x3(self.Conv2d_0a_1x1(x)))


class InceptionABranch3(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionABranch3, self).__init__()
        self.Conv2d_0b_1x1 = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        AvgPool_0a_3x3 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        return self.Conv2d_0b_1x1(AvgPool_0a_3x3)


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features, bug=False):
        super(InceptionA, self).__init__()
        # self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        #
        # self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        # self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)
        #
        # self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        # self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        # self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)
        # self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

        self.Branch_0 = InceptionABranch0(in_channels)

        self.Branch_1 = InceptionABranch1(in_channels, bug)

        self.Branch_2 = InceptionABranch2(in_channels)
        self.Branch_3 = InceptionABranch3(in_channels, pool_features)

    def forward(self, x):
        # branch1x1 = self.branch1x1(x)
        #
        # branch5x5 = self.branch5x5_1(x)
        # branch5x5 = self.branch5x5_2(branch5x5)
        #
        # branch3x3dbl = self.branch3x3dbl_1(x)
        # branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        # branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        #
        # branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        # branch_pool = self.branch_pool(branch_pool)
        #
        # outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]

        branch1x1 = self.Branch_0(x)
        branch5x5 = self.Branch_1(x)
        branch3x3dbl = self.Branch_2(x)

        branch_pool = self.Branch_3(x)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]

        return torch.cat(outputs, 1)


class InceptionBBranch0(nn.Module):
    def __init__(self, in_channels):
        super(InceptionBBranch0, self).__init__()
        self.Conv2d_1a_1x1 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

    def forward(self, x):
        return self.Conv2d_1a_1x1(x)


class InceptionBBranch1(nn.Module):
    def __init__(self, in_channels):
        super(InceptionBBranch1, self).__init__()
        self.Conv2d_0a_1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.Conv2d_0b_3x3 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.Conv2d_1a_1x1 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        return self.Conv2d_1a_1x1(self.Conv2d_0b_3x3(self.Conv2d_0a_1x1(x)))


class InceptionBBranch2(nn.Module):
    def __init__(self):
        super(InceptionBBranch2, self).__init__()

    def forward(self, x):
        return F.max_pool2d(x, kernel_size=3, stride=2)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        # self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        # self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        # self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        # self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)
        #
        self.Branch_0 = InceptionBBranch0(in_channels)
        self.Branch_1 = InceptionBBranch1(in_channels)
        self.Branch_2 = InceptionBBranch2()

    def forward(self, x):
        # branch3x3 = self.branch3x3(x)
        #
        # branch3x3dbl = self.branch3x3dbl_1(x)
        # branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        # branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        #
        # branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        branch3x3 = self.Branch_0(x)
        branch3x3dbl = self.Branch_1(x)
        branch_pool = self.Branch_2(x)
        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionCBranch0(nn.Module):
    def __init__(self, in_channels):
        super(InceptionCBranch0, self).__init__()
        self.Conv2d_0a_1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        return self.Conv2d_0a_1x1(x)


class InceptionCBranch1(nn.Module):
    def __init__(self, in_channels, c7):
        super(InceptionCBranch1, self).__init__()
        self.Conv2d_0a_1x1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.Conv2d_0b_1x7 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.Conv2d_0c_7x1 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

    def forward(self, x):
        return self.Conv2d_0c_7x1(self.Conv2d_0b_1x7(self.Conv2d_0a_1x1(x)))


class InceptionCBranch2(nn.Module):
    def __init__(self, in_channels, c7):
        super(InceptionCBranch2, self).__init__()
        self.Conv2d_0a_1x1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.Conv2d_0b_7x1 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.Conv2d_0c_1x7 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.Conv2d_0d_7x1 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.Conv2d_0e_1x7 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

    def forward(self, x):
        return self.Conv2d_0e_1x7(self.Conv2d_0d_7x1(self.Conv2d_0c_1x7(self.Conv2d_0b_7x1(self.Conv2d_0a_1x1(x)))))


class InceptionCBranch3(nn.Module):
    def __init__(self, in_channels):
        super(InceptionCBranch3, self).__init__()
        self.Conv2d_0b_1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        return self.Conv2d_0b_1x1(F.avg_pool2d(x, kernel_size=3, stride=1, padding=1))


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.Branch_0 = InceptionCBranch0(in_channels)
        self.Branch_1 = InceptionCBranch1(in_channels, channels_7x7)
        self.Branch_2 = InceptionCBranch2(in_channels, channels_7x7)
        self.Branch_3 = InceptionCBranch3(in_channels)

        # self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)
        #
        # c7 = channels_7x7
        # self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        # self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        # self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))
        #
        # self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        # self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        # self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        # self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        # self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        # self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        # branch1x1 = self.branch1x1(x)
        #
        # branch7x7 = self.branch7x7_1(x)
        # branch7x7 = self.branch7x7_2(branch7x7)
        # branch7x7 = self.branch7x7_3(branch7x7)
        #
        # branch7x7dbl = self.branch7x7dbl_1(x)
        # branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        # branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        # branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        # branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        #
        # branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        # branch_pool = self.branch_pool(branch_pool)

        branch1x1 = self.Branch_0(x)
        branch7x7 = self.Branch_1(x)
        branch7x7dbl = self.Branch_2(x)
        branch_pool = self.Branch_3(x)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionDBranch0(nn.Module):
    def __init__(self, in_channels):
        super(InceptionDBranch0, self).__init__()
        self.Conv2d_0a_1x1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.Conv2d_1a_3x3 = BasicConv2d(192, 320, kernel_size=3, stride=2)

    def forward(self, x):
        return self.Conv2d_1a_3x3(self.Conv2d_0a_1x1(x))


class InceptionDBranch1(nn.Module):
    def __init__(self, in_channels):
        super(InceptionDBranch1, self).__init__()
        self.Conv2d_0a_1x1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.Conv2d_0b_1x7 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.Conv2d_0c_7x1 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.Conv2d_1a_3x3 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        return self.Conv2d_1a_3x3(self.Conv2d_0c_7x1(self.Conv2d_0b_1x7(self.Conv2d_0a_1x1(x))))


class InceptionDBranch2(nn.Module):
    def __init__(self):
        super(InceptionDBranch2, self).__init__()

    def forward(self, x):
        return F.max_pool2d(x, kernel_size=3, stride=2)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        # self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        # self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)
        #
        # self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        # self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        # self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        # self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.Branch_0 = InceptionDBranch0(in_channels)
        self.Branch_1 = InceptionDBranch1(in_channels)
        self.Branch_2 = InceptionDBranch2()

    def forward(self, x):
        # branch3x3 = self.branch3x3_1(x)
        # branch3x3 = self.branch3x3_2(branch3x3)
        #
        branch3x3 = self.Branch_0(x)

        # branch7x7x3 = self.branch7x7x3_1(x)
        # branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        # branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        # branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch7x7x3 = self.Branch_1(x)
        branch_pool = self.Branch_2(x)
        # branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionEBranch0(nn.Module):
    def __init__(self, in_channels):
        super(InceptionEBranch0, self).__init__()
        self.Conv2d_0a_1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

    def forward(self, x):
        return self.Conv2d_0a_1x1(x)


class InceptionEBranch1(nn.Module):
    def __init__(self, in_channels, bug):
        super(InceptionEBranch1, self).__init__()
        self.bug = bug

        self.Conv2d_0a_1x1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.Conv2d_0b_1x3 = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        if self.bug:
            self.Conv2d_0b_3x1 = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        else:
            self.Conv2d_0c_3x1 = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

    def forward(self, x):
        branch3x3 = self.Conv2d_0a_1x1(x)
        if self.bug:
            branch3x3 = [
                self.Conv2d_0b_1x3(branch3x3),
                self.Conv2d_0b_3x1(branch3x3),
            ]
        else:
            branch3x3 = [
                self.Conv2d_0b_1x3(branch3x3),
                self.Conv2d_0c_3x1(branch3x3),
            ]
        return torch.cat(branch3x3, 1)


class InceptionEBranch2(nn.Module):
    def __init__(self, in_channels):
        super(InceptionEBranch2, self).__init__()
        self.Conv2d_0a_1x1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.Conv2d_0b_3x3 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.Conv2d_0c_1x3 = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.Conv2d_0d_3x1 = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

    def forward(self, x):
        branch3x3dbl = self.Conv2d_0a_1x1(x)
        branch3x3dbl = self.Conv2d_0b_3x3(branch3x3dbl)
        branch3x3dbl = [
            self.Conv2d_0c_1x3(branch3x3dbl),
            self.Conv2d_0d_3x1(branch3x3dbl),
        ]
        return torch.cat(branch3x3dbl, 1)


class InceptionEBranch3(nn.Module):
    def __init__(self, in_channels):
        super(InceptionEBranch3, self).__init__()
        self.Conv2d_0b_1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        return self.Conv2d_0b_1x1(branch_pool)


class InceptionE(nn.Module):

    def __init__(self, in_channels, bug=False):
        super(InceptionE, self).__init__()
        # self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)
        self.Branch_0 = InceptionEBranch0(in_channels)

        # self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        # self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        # self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.Branch_1 = InceptionEBranch1(in_channels, bug)

        # self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        # self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        # self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        # self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.Branch_2 = InceptionEBranch2(in_channels)

        # self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)
        self.Branch_3 = InceptionEBranch3(in_channels)

    def forward(self, x):
        # branch1x1 = self.branch1x1(x)
        #
        #
        # branch3x3 = self.branch3x3_1(x)
        # branch3x3 = [
        #     self.branch3x3_2a(branch3x3),
        #     self.branch3x3_2b(branch3x3),
        # ]
        # branch3x3 = torch.cat(branch3x3, 1)
        #
        # branch3x3dbl = self.branch3x3dbl_1(x)
        # branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        # branch3x3dbl = [
        #     self.branch3x3dbl_3a(branch3x3dbl),
        #     self.branch3x3dbl_3b(branch3x3dbl),
        # ]
        # branch3x3dbl = torch.cat(branch3x3dbl, 1)
        #
        # branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        # branch_pool = self.branch_pool(branch_pool)

        branch1x1 = self.Branch_0(x)

        branch3x3 = self.Branch_1(x)

        branch3x3dbl = self.Branch_2(x)
        branch_pool = self.Branch_3(x)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

class Conv2d_2b_1x1(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(Conv2d_2b_1x1, self).__init__()
        self.conv =nn.Conv2d(in_channels, num_classes, bias=True, kernel_size=1)
        self.conv.stddev = 0.001

    def forward(self, x):
        return self.conv(x)

class AuxLogits(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(AuxLogits, self).__init__()
        self.Conv2d_1b_1x1 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.Conv2d_2a_5x5 = BasicConv2d(128, 768, kernel_size=5)
        self.Conv2d_2a_5x5.stddev = 0.01
       # self.fc = nn.Linear(768, num_classes)
        self.Conv2d_2b_1x1 = Conv2d_2b_1x1(768, num_classes)



    def forward(self, x):
        # 17 x 17 x 768
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x = self.Conv2d_1b_1x1(x)
        # 5 x 5 x 128
        x = self.Conv2d_2a_5x5(x)
        # 1 x 1 x 768
        #x = x.view(x.size(0), -1)
        # 768
        x = self.Conv2d_2b_1x1(x)
        x = x.view(x.size(0), -1)

        return x


class Conv2d_1c_1x1(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(Conv2d_1c_1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, bias=True, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.BatchNorm = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.BatchNorm(x)
        return F.relu(x, inplace=True)