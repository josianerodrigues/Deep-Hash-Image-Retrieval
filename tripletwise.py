import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

from trainer import fit
import numpy as np
cuda = torch.cuda.is_available()

from torchvision import datasets
from torchvision.datasets import CIFAR100
from torchvision import transforms

mean, std = 0.1307, 0.3081

train_dataset = CIFAR100('data/', train=True, download=True,
                            transform=transforms.Compose(
                                                        [transforms.Resize(256),
                                                transforms.RandomCrop(227),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))

test_dataset = CIFAR100('data/', train=False, download=True,
                        transform=transforms.Compose(
                                        [transforms.Resize(227),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))

n_classes = 100


# Set up the network and training parameters
from networks import EmbeddingResnet
from losses import OnlineTripletLoss
from utils import AllTripletSelector,HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector # Strategies for selecting triplets within a minibatch
from metrics import AverageNonzeroTripletsMetric

margin = 1.
embedding_net = EmbeddingResnet()
model = embedding_net


if cuda:
    model.cuda()
loss_fn = OnlineTripletLoss(margin, SemihardNegativeTripletSelector(margin))
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
#optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 25
log_interval = 50


from datasets import BalancedBatchSampler

train_batch_sampler = BalancedBatchSampler(train_dataset.targets, n_classes=100, n_samples=5)
test_batch_sampler = BalancedBatchSampler(test_dataset.targets, n_classes=100, n_samples=5)


kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}
online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)


fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AverageNonzeroTripletsMetric()])
