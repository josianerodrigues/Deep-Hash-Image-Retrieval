import os
import argparse
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import hamming, cdist
#from net import AlexNetPlusLatent

from torchvision.datasets import CIFAR100
from torchvision import transforms
from timeit import time

import torch
from networks import EmbeddingResnet, TripletNet, ClassificationNet, SiameseNet
from losses import TripletLoss
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler

parser = argparse.ArgumentParser(description='Deep Hashing evaluate mAP')
parser.add_argument('--pretrained', type=str, default=0, metavar='pretrained_model',
					help='loading pretrained model(default = None)')
parser.add_argument('--bits', type=int, default=48, metavar='bts',
					help='binary bits')
args = parser.parse_args()
cuda = torch.cuda.is_available()


def load_data():
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

	# Set up data loaders
	batch_size = 35
	kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)    

	return train_loader, test_loader

def binary_output(dataloader):
	embedding_net = EmbeddingResnet()
	net = TripletNet(embedding_net)
	#net = SiameseNet(embedding_net)
	#net = ClassificationNet(embedding_net, 100)
	net.load_state_dict(torch.load('./model/%s' %args.pretrained))

	use_cuda = torch.cuda.is_available()
	if use_cuda:
		net.cuda()
	full_batch_output = torch.cuda.FloatTensor()
	full_batch_label = torch.cuda.LongTensor()
	net.eval()
	for batch_idx, (inputs, targets) in enumerate(dataloader):
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()
		inputs, targets = Variable(inputs, volatile=True), Variable(targets)
		outputs = net.get_embedding(inputs).data
		full_batch_output = torch.cat((full_batch_output, outputs.data), 0)
		full_batch_label = torch.cat((full_batch_label, targets.data), 0)
	return torch.round(full_batch_output), full_batch_label

def precision(trn_binary, trn_label, tst_binary, tst_label):
	trn_binary = trn_binary.cpu().numpy()
	trn_binary = np.asarray(trn_binary, np.int32)
	trn_label = trn_label.cpu().numpy()
	tst_binary = tst_binary.cpu().numpy()
	tst_binary = np.asarray(tst_binary, np.int32)
	tst_label = tst_label.cpu().numpy()
	query_times = tst_binary.shape[0]
	trainset_len = train_binary.shape[0]
	AP = np.zeros(query_times)
	Ns = np.arange(1, trainset_len + 1)
	total_time_start = time.time()
	for i in range(query_times):
		print('Query ', i+1)
		query_label = tst_label[i]
		query_binary = tst_binary[i,:]
		print(query_binary)
		query_result = np.count_nonzero(query_binary != trn_binary, axis=1)    #don't need to divide binary length
		sort_indices = np.argsort(query_result)
		buffer_yes= np.equal(query_label, trn_label[sort_indices]).astype(int)
		P = np.cumsum(buffer_yes) / Ns
		AP[i] = np.sum(P * buffer_yes) /sum(buffer_yes)
	map = np.mean(AP)
	print(map)
	print('total query time = ', time.time() - total_time_start)



if os.path.exists('./result/train_binary') and os.path.exists('./result/train_label') and \
   os.path.exists('./result/test_binary') and os.path.exists('./result/test_label') and args.pretrained == 0:
	train_binary = torch.load('./result/train_binary')
	train_label = torch.load('./result/train_label')
	test_binary = torch.load('./result/test_binary')
	test_label = torch.load('./result/test_label')

else:
	trainloader, testloader = load_data()
	train_binary, train_label = binary_output(trainloader)
	test_binary, test_label = binary_output(testloader)
	if not os.path.isdir('result'):
		os.mkdir('result')
	torch.save(train_binary, './result/train_binary')
	torch.save(train_label, './result/train_label')
	torch.save(test_binary, './result/test_binary')
	torch.save(test_label, './result/test_label')


precision(train_binary, train_label, test_binary, test_label)
