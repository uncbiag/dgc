import sys
import torch
import torch.utils.data
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
import argparse
import math
import pandas as pd
from dgc import dgc
from vade import vade
from vae import vae
from vade_linear import vade_linear
from sklearn import svm
import util
#from data_loader import stl_loader




parser = argparse.ArgumentParser(description='STL EXPERIMENT')
parser.add_argument('--lr', type=float, default=0.002, metavar='N',
                    help='learning rate for training (default: 0.001)')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--pretrain', type=str, default="", metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--save', type=str, default="", metavar='N',
                    help='number of epochs to train (default: 10)')
args = parser.parse_args()
#args.pretrain="/playpen/deep_clustering/pacman/pretrain_model.pt"
args.save = "/playpen-raid/yifengs/deep_clustering/mnist_test/vae.pt"

'''
total_f = torch.from_numpy(np.load("total_f.npy"))
total_l = torch.from_numpy(np.load("total_l.npy"))
dataloader = DataLoader(TensorDataset(total_f,total_l),batch_size=args.batch_size,shuffle=True)
'''

train_f = torch.tensor(np.load("train_f_resnet50.npy"))
test_f = torch.tensor(np.load("test_f_resnet50.npy"))
train_l = torch.tensor(np.load("train_l_resnet50.npy"))
test_l = torch.tensor(np.load("test_l_resnet50.npy"))
init_f = torch.cat([train_f,test_f])
init_l = torch.cat([train_l,test_l])
trainloader = DataLoader(TensorDataset(train_f,train_l),batch_size=args.batch_size,shuffle=True)
testloader = DataLoader(TensorDataset(test_f,test_l),batch_size=args.batch_size,shuffle=True)
initloader = DataLoader(TensorDataset(init_f,init_l),batch_size=args.batch_size,shuffle=True)


extractor = util.linear_eval(2048,100)
extractor.load_state_dict(torch.load('./pretrain_resnet50.pt'))
extractor.eval()
model = dgc(input_dim=2048, y_dim = 100, z_dim=10, n_centroids=20, binary=False,
        encodeLayer=[500,500,2000], decodeLayer=[2000,500,500])

#model = vade(input_dim=2048, z_dim=10, n_centroids=20, binary=False,
#        encodeLayer=[500,500,2000], decodeLayer=[2000,500,500])

#model = vae(input_dim=784, z_dim=10, binary=True,
#        encodeLayer=[500,500,2000], decodeLayer=[2000,500,500])
    #model.load_model(args.pretrain)
#model.pretrain_model(initloader,10,128)
print("Initializing through GMM..")
model.initialize_gmm(initloader,args.batch_size)


model.fit(trainloader, testloader, lr=args.lr, num_epochs=args.epochs,
        anneal=True, direct_predict_prob=False)
#if args.save != "":
#	model.save_model(args.save)


#scp run_model.py yifengs@biag-lambda1.cs.unc.edu:/playpen-raid/yifengs/deep_clustering/cifar_test

'''

model = VaDE(input_dim=784, z_dim=10, y_dim = 1, n_centroids=4, binary=True,
        encodeLayer=[500,500,2000], decodeLayer=[2000,500,500])
if args.pretrain != "":
    print("Loading model from %s..." % args.pretrain)
    model.load_model(args.pretrain)
print("Initializing through GMM..")
model.initialize_gmm(train_loader,args.batch_size)
model.fit(train_loader, test_loader, lr=args.lr, b_size=args.batch_size, num_epochs=args.epochs, anneal=True)
if args.save != "":
    model.save_model(args.save)

'''


