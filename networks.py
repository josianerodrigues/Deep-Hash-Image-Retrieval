import torch.nn as nn
import torch
import torch.nn.functional as F
from resnet import resnet50
from torchvision import models
#from base_config import cfg


import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import numpy as np


class EmbeddingResnet(nn.Module):
    def __init__(self):
        super(EmbeddingResnet, self).__init__()

        resnet = resnet50(pretrained=True)
        self.features = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4, resnet.avgpool)
        self.Linear = nn.Linear(8192, 256)
        self.prelu1 =  nn.Sigmoid()

        for p in self.features[0].parameters(): p.requires_grad=False
        for p in self.features[1].parameters(): p.requires_grad=False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad=False


    def forward(self, x):
        features = self.features.forward(x)
        features = features.view(features.size(0), -1)
        features = self.Linear(features)
        #features = F.normalize(features, p=2, dim
        output = self.prelu1(features)

        return output

    def get_embedding(self, x):
        features = self.features.forward(x)
        features = features.view(features.size(0), -1)
        features = self.Linear(features)
        #features = F.normalize(features, p=2, dim
        output = self.prelu1(features)

        return output



class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        #self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(256, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        #output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.embedding_net.get_embedding(x)       