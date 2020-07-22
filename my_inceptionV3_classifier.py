# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.myinception import inception_v3
import pdb
# from model.faster_rcnn.inception import Inception3, InceptionAux
from model.utils.config import cfg
from torchvision.models import Inception3
import matplotlib.pyplot as plt
import torchvision

from torch.nn import init


class inceptionV3(nn.Module):
    def __init__(self, classes, pretrained=False, class_agnostic=False, include_id_loss=False, use_cross_entropy=True,
                 total_ids=5532):
        super(inceptionV3, self).__init__()
        self.model_path = 'data/pretrained_model/cub_inceptionv3.pth'
        self.pretrained = pretrained
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        self.criterion = nn.CrossEntropyLoss()
        self.total_ids = total_ids

       # model = models.inception_v3(pretrained=False)
        model  = inception_v3(pretrained=False) #need imagenet pretrained model?
        model = model.cuda()

        checkpoint = torch.load(self.model_path) #load iNaturalist pretrained model
        state_dict = checkpoint #['state_dict']
        model.load_state_dict(state_dict)

        for name, param in model.named_parameters():
            param.requires_grad = False

        #model.AuxLogits.fc = nn.Linear(768, self.total_ids)
        #model.AuxLogits.fc.weight.requires_grad = False

        ## required gradient will be true
        model.Logits.Conv2d_1c_1x1.conv = nn.Conv2d(2048, 200, bias=True, kernel_size=1)

        self.classy = model

    def forward(self, x, y):

        if self.training:
            y_pred, aux = self.classy(Variable(x))
            #loss2 = self.criterion(aux, Variable(y))
            loss2=0
            loss1 = self.criterion(y_pred, Variable(y))

            loss = loss1 + 0.4 * loss2

        else:
            y_pred = self.classy(Variable(x))
            loss = 0


        return loss, y_pred


    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

            # normal_init(self.classy.AuxLogits.fc, 0, 0.001)

        normal_init(self.classy.Logits.Conv2d_1c_1x1.conv, 0, 0.001, truncated=True)

    # def _init_weights(self):
    #     def normal_init(m):
    #         """
    #         weight initalizer: truncated normal and random normal.
    #         """
    #         # x is a parameter
    #         m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    #         if isinstance(m, nn.Conv2d):
    #             import scipy.stats as stats
    #             stddev = m.stddev if hasattr(m, 'stddev') else 0.1
    #             X = stats.truncnorm(-2, 2, scale=stddev)
    #             values = torch.Tensor(X.rvs(m.weight.data.numel()))
    #             values = values.view(m.weight.data.size())
    #             m.weight.data.copy_(values)
    #
    #             m.bias.data.zero_() #assuming each convolution has bias term
    #
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #
    #
    #     normal_init(self.classy.Logits.Conv2d_1c_1x1.conv)





    def create_architecture(self):
        self._init_weights()


