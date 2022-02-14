import sys
import torch
import torch.utils.data
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import argparse
import math
from data_loader import pascal_loader
import pandas as pd
from dgc_entropy import dgc_entropy
from vade import vade
from vae import vae
from sklearn import svm



transform = transforms.Compose(
    [transforms.ToTensor()])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)

train_features = []
train_labels = []
test_features = []
test_labels = []

net = models.resnext50(pretrained=True)
newnet = torch.nn.Sequential(*(list(net.children())[:-1]))
newnet = newnet.cuda()
for i,(inputs,labels) in enumerate(trainloader):
	print(i)
	inputs = inputs.cuda()
	feature = newnet(inputs)
	feature = feature.squeeze(2).squeeze(2)
	train_features.append(feature.detach().cpu().numpy())
	train_labels.append(labels.numpy())
train_features = np.concatenate(train_features)
train_labels = np.concatenate(train_labels)

for i,(inputs,labels) in enumerate(testloader):
	inputs = inputs.cuda()
	feature = newnet(inputs)
	feature = feature.squeeze(2).squeeze(2)
	test_features.append(feature.detach().cpu().numpy())
	test_labels.append(labels.numpy())
test_features = np.concatenate(test_features)
test_labels = np.concatenate(test_labels)

np.save("train_features",train_features)
np.save("test_features",test_features)
np.save("train_labels",train_labels)
np.save("test_labels",test_labels)
