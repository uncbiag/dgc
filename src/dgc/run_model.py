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
from dgc_entropy import dgc_entropy
from vade import vade
from sklearn import svm
from util_class import split_train_test
import util
#from data_loader import stl_loader




parser = argparse.ArgumentParser(description='DGC EXPERIMENT')
parser.add_argument('--lr', type=float, default=0.01, metavar='N',
                    help='learning rate for training (default: 0.001)')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--pretrain', type=str, default="", metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--save', type=str, default="", metavar='N',
                    help='number of epochs to train (default: 10)')
args = parser.parse_args()
#args.pretrain="/playpen/deep_clustering/pacman/pretrain_model.pt"
args.save = "/playpen-raid/yifengs/dgc/pacman/dgc.pt"


dataset = 'pacman'
trainloader, testloader, initloader = util.load_sample_datasets(arg.batch_size,dataset)
if dataset == 'pacman':
    task_name = 'regression'
    model = dgc(input_dim=2,  y_dim = 1, z_dim=10, n_centroids=2, task = task_name, binary=True,
                encodeLayer=[128,256,256], decodeLayer=[256,256,128])
elif dataset == 'cifar100':
    task_name = 'classification'
    torch.manual_seed(157)
    np.random.seed(157)
    model = dgc(input_dim=2048, y_dim = 100, z_dim=10, n_centroids=20, task = task_name, binary=False,
            encodeLayer=[500,500,2000], decodeLayer=[2000,500,500])

model.fit(initloader, initloader, lr=args.lr, num_epochs=args.epochs,
        anneal=True,direct_predict_prob=False)
#if args.save != "":
model.save_model(args.save)


