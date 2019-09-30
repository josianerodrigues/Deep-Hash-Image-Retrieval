from torchvision.datasets import CIFAR100
from torchvision import transforms
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from trainer import fit
import numpy as np

cuda = torch.cuda.is_available()

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

# Set up data loaders
batch_size = 35
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# Set up the network and training parameters
from networks import EmbeddingNet, ClassificationNet, EmbeddingResnet
from metrics import AccumulatedAccuracyMetric

embedding_net = EmbeddingResnet()
model = ClassificationNet(embedding_net, n_classes=n_classes)
if cuda:
    model.cuda()
loss_fn = nn.NLLLoss()
lr = 1e-2
#optimizer = optim.Adam(model.parameters(), lr=lr)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 25
log_interval = 50

fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AccumulatedAccuracyMetric()])
