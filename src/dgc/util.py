import os, argparse
import sys
import scipy

import timeit
import gzip

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import datasets, models, transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
import math
from sklearn.mixture import GaussianMixture

import pickle as pkl

from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn.cluster import KMeans
import math

from sklearn import metrics, mixture

from torch.optim.optimizer import Optimizer
from sklearn.manifold import TSNE
#from sklearn.utils.linear_assignment_ import linear_assignment

import matplotlib.pyplot as plt
import scipy.io as scio
import dgc.likelihoods as likelihoods





class res_cutlayers(nn.Module):
    def __init__(self,net,delete):
        super(res_cutlayers, self).__init__()
        self.conv = nn.Sequential(
            *list(net.children())[:-delete]
            )

    def forward(self, x): 
        x = self.conv(x) 
        return x             



def bigEncNet(input_channels):
    hidden1 = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2),
    nn.BatchNorm2d(num_features=48),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
    nn.Dropout(0.2)
    )
    hidden2 = nn.Sequential(
    nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2),
    nn.BatchNorm2d(num_features=64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
    nn.Dropout(0.2)
    )
    hidden3 = nn.Sequential(
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
    nn.BatchNorm2d(num_features=128),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
    nn.Dropout(0.2)
    )
    hidden4 = nn.Sequential(
    nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2),
    nn.BatchNorm2d(num_features=160),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
    nn.Dropout(0.2)
    )
    hidden5 = nn.Sequential(
    nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2),
    nn.BatchNorm2d(num_features=192),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
    nn.Dropout(0.2)
    )
    hidden6 = nn.Sequential(
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
    nn.BatchNorm2d(num_features=192),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
    nn.Dropout(0.2)
    )
    hidden7 = nn.Sequential(
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
    nn.BatchNorm2d(num_features=192),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
    nn.Dropout(0.2)
    )
    hidden8 = nn.Sequential(
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
    nn.BatchNorm2d(num_features=192),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
    nn.Dropout(0.2)
    )
    net = [hidden1,hidden2,hidden3,hidden4,hidden5,hidden6,hidden7,hidden8]
    return nn.Sequential(*net)





def buildEncoderNetwork(input_channels, kernel_num):
    net = []

    net.append(nn.Conv2d(input_channels, kernel_num//4, kernel_size=4, stride=2, padding=1))
    net.append(nn.BatchNorm2d(kernel_num//4))
    net.append(nn.ReLU(True))

    net.append(nn.Conv2d(kernel_num//4, kernel_num//2, kernel_size=4, stride=2, padding=1))
    net.append(nn.BatchNorm2d(kernel_num//2))
    net.append(nn.ReLU(True))

    net.append(nn.Conv2d(kernel_num//2, kernel_num, kernel_size=4,stride=2,padding=1))
    net.append(nn.BatchNorm2d(kernel_num))
    net.append(nn.ReLU(True))
    return nn.Sequential(*net)




def buildDecoderNetwork(kernel_num, output_channels):
    net = []
    net.append(nn.ConvTranspose2d(kernel_num,kernel_num//2, kernel_size=4,stride=2, padding=1))
    net.append(nn.BatchNorm2d(kernel_num//2))
    net.append(nn.ReLU(True))

    net.append(nn.ConvTranspose2d(kernel_num//2, kernel_num//4, kernel_size=4, stride=2, padding=1))
    net.append(nn.BatchNorm2d(kernel_num//4))
    net.append(nn.ReLU(True))

    net.append(nn.ConvTranspose2d(kernel_num//4, output_channels, kernel_size=4, stride=2, padding=1))
    net.append(nn.BatchNorm2d(output_channels))
    net.append(nn.ReLU(True))

    return nn.Sequential(*net)


def buildNetwork(layers, activation="relu", dropout=0):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        elif activation=="tanh":
            net.append(nn.Tanh())
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)


'''
def buildDecoderNetwork(hidden_size, nFilters, output_channels):
    net = []
    
    net.append(nn.ConvTranspose2d(hidden_size, 8*nFilters, kernel_size=4,stride=2, padding=1))
    net.append(nn.BatchNorm2d(8*nFilters))
    net.append(nn.ReLU(True))

    net.append(nn.ConvTranspose2d(8*nFilters, 4*nFilters, kernel_size=4, stride=2, padding=1))
    net.append(nn.BatchNorm2d(4*nFilters))
    net.append(nn.ReLU(True))


    net.append(nn.ConvTranspose2d(4*nFilters, 2*nFilters, kernel_size=4, stride=2, padding=1))
    net.append(nn.BatchNorm2d(2*nFilters))
    net.append(nn.ReLU(True))

    net.append(nn.ConvTranspose2d(2*nFilters, nFilters, kernel_size=4, stride=2, padding=1))
    net.append(nn.BatchNorm2d(nFilters))
    net.append(nn.ReLU(True))

    net.append(nn.ConvTranspose2d(nFilters, output_channels, kernel_size=4, stride=2, padding=1))
    return nn.Sequential(*net)
'''


def buildNetwork(layers, activation="relu", dropout=0):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        elif activation=="tanh":
            net.append(nn.Tanh())
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)



# Split into train/test sets
class split_train_test(object):
    def __init__(self, ratio,seed):
        self.ratio = ratio
        self.seed = seed

    def __call__(self,dataset):
        np.random.seed(self.seed)
        index = np.random.permutation(len(dataset[1]))
        train_index, test_index = index[:math.floor(len(index)*self.ratio)], index[math.floor(len(index)*self.ratio):]
        train_dataset = [dataset[0][train_index,:],dataset[1][train_index],dataset[2][train_index]]
        test_dataset = [dataset[0][test_index,:],dataset[1][test_index],dataset[2][test_index]]
        return train_dataset, test_dataset


def load_sample_datasets(batch_size, path, dataset = 'pacman'):
    if dataset == 'pacman':
        pacman_data = np.load(path+"/pacman_data.npy")
        classes = np.load(path+"/pacman_classes.npy")
        res = np.load(path+"/pacman_response_linear_exp.npy")


        # split the dataset
        split = split_train_test(0.8,123)
        pacman_dataset = [pacman_data,classes,res]
        train_d,test_d = split(pacman_dataset)

        train_f = torch.tensor(train_d[0]).float()
        test_f = torch.tensor(test_d[0]).float()
        train_l = torch.tensor(train_d[1].astype(int))
        test_l = torch.tensor(test_d[1].astype(int))
        train_y = torch.tensor(train_d[2]).float()
        test_y = torch.tensor(test_d[2]).float()
        init_f = torch.cat([train_f,test_f])
        init_l = torch.cat([train_l,test_l])
        init_y = torch.cat([train_y,test_y])


        trainloader = DataLoader(TensorDataset(train_f,train_y,train_l),batch_size=batch_size,shuffle=True)
        testloader = DataLoader(TensorDataset(test_f,test_y, test_l),batch_size=batch_size,shuffle=True)
        initloader = DataLoader(TensorDataset(init_f,init_y,init_l),batch_size=batch_size,shuffle=True)


    elif dataset == 'cifar100':
        train_f = torch.tensor(np.load(path+"/train_f_resnet50.npy"))
        test_f = torch.tensor(np.load(path+"/test_f_resnet50.npy"))
        train_l = torch.tensor(np.load(path+"/train_l_resnet50.npy"))
        test_l = torch.tensor(np.load(path+"/test_l_resnet50.npy"))
        init_f = torch.cat([train_f,test_f])
        init_l = torch.cat([train_l,test_l])
        trainloader = DataLoader(TensorDataset(train_f,train_l),batch_size=batch_size,shuffle=True)
        testloader = DataLoader(TensorDataset(test_f,test_l),batch_size=batch_size,shuffle=True)
        initloader = DataLoader(TensorDataset(init_f,init_l),batch_size=batch_size,shuffle=True)


    return trainloader,testloader,initloader



def form_dataloaders(batch_size,train_d,test_d):

    # first index: features
    # second index: side-information 
    # third index: cluster indices
    train_f = torch.tensor(train_d[0]).float()
    test_f = torch.tensor(test_d[0]).float()
    train_y = torch.tensor(train_d[1]).float()
    test_y = torch.tensor(test_d[1]).float()
    train_l = torch.tensor(train_d[2].astype(int))
    test_l = torch.tensor(test_d[2].astype(int))
    init_f = torch.cat([train_f,test_f])
    init_l = torch.cat([train_l,test_l])
    init_y = torch.cat([train_y,test_y])

    trainloader = DataLoader(TensorDataset(train_f,train_y,train_l),batch_size=batch_size,shuffle=True)
    testloader = DataLoader(TensorDataset(test_f,test_y, test_l),batch_size=batch_size,shuffle=True)
    initloader = DataLoader(TensorDataset(init_f,init_y,init_l),batch_size=batch_size,shuffle=True)


    return trainloader,testloader,initloader



def vade_get_lambda_k(z, u_p, lambda_p, theta_p, n_centroids):
    Z = z.unsqueeze(2).expand(z.size()[0], z.size()[1], n_centroids) # NxDxK
    u_tensor3 = u_p.unsqueeze(0).expand(z.size()[0], u_p.size()[0], u_p.size()[1]) # NxDxK
    lambda_tensor3 = lambda_p.unsqueeze(0).expand(z.size()[0], lambda_p.size()[0], lambda_p.size()[1])
    theta_tensor2 = theta_p.unsqueeze(0).expand(z.size()[0], n_centroids) # NxK
    '''
    p_c_z = torch.exp(torch.log(theta_tensor2) - torch.sum(0.5*torch.log(2*math.pi*lambda_tensor3)+\
        (Z-u_tensor3)**2/(2*lambda_tensor3), dim=1)) + 1e-8 # NxK
    p_c_z = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)
    #p_c_z = torch.log(theta_tensor2) - torch.sum(0.5*torch.log(2*math.pi*lambda_tensor3)+\
    #    (Z-u_tensor3)**2/(2*lambda_tensor3), dim=1)
    #p_c_z = torch.softmax(p_c_z,1)
    '''
    #p_c_z = torch.log(theta_tensor2) - torch.sum(0.5*torch.log(2*math.pi*lambda_tensor3)+\
    #    (Z-u_tensor3)**2/(lambda_tensor3), dim=1)# NxK
    p_c_z = torch.exp(torch.log(theta_tensor2) - torch.sum(0.5*torch.log(2*math.pi*lambda_tensor3)+\
        (Z-u_tensor3)**2/(2*lambda_tensor3), dim=1)) + 1e-8 # NxK

    p_c_z = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)
    if torch.sum(torch.isnan(p_c_z))!=0:
        import pdb; pdb.set_trace()

    #return torch.softmax(p_c_z,1)
    return p_c_z




def dgc_get_lambda_k(z, side_info, u_p, lambda_p, theta_p, n_centroids):
    #lambda_p = lambda_p.data.clamp_(1e-2)
    Z = z.unsqueeze(2).expand(z.size()[0], z.size()[1], n_centroids) # NxDxK
    u_tensor3 = u_p.unsqueeze(0).expand(z.size()[0], u_p.size()[0], u_p.size()[1]) # NxDxK
    lambda_tensor3 = lambda_p.unsqueeze(0).expand(z.size()[0], lambda_p.size()[0], lambda_p.size()[1])
    theta_tensor2 = theta_p.unsqueeze(0).expand(z.size()[0], n_centroids) # NxK
    '''
    p_c_z = torch.exp(torch.log(theta_tensor2) - torch.sum(0.5*torch.log(2*math.pi*lambda_tensor3)+\
        (Z-u_tensor3)**2/(2*lambda_tensor3), dim=1)) + 1e-8 # NxK
    p_c_z = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)
    #p_c_z = torch.log(theta_tensor2) - torch.sum(0.5*torch.log(2*math.pi*lambda_tensor3)+\
    #    (Z-u_tensor3)**2/(2*lambda_tensor3), dim=1)
    #p_c_z = torch.softmax(p_c_z,1)
    '''
    p_c_z = torch.log(theta_tensor2) - torch.sum(0.5*torch.log(2*math.pi*lambda_tensor3)+\
        (Z-u_tensor3)**2/(lambda_tensor3), dim=1)# NxK
    #p_c_z = torch.exp(torch.log(theta_tensor2) - torch.sum(0.5*torch.log(2*math.pi*lambda_tensor3)+\
    #    (Z-u_tensor3)**2/(2*lambda_tensor3), dim=1)) + 1e-8 # NxK

    #p_c_z = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

    #lambda_k = side_info*p_c_z / (torch.sum(side_info*p_c_z, dim=1, keepdim=True)+1e-8)
    #lambda_k = torch.softmax((side_info+neg_entropy)*p_c_z,1)
    lambda_k = torch.softmax(side_info+p_c_z,1)
    #if torch.sum(torch.isnan(lambda_k))!=0:
    #    import pdb; pdb.set_trace()
    if torch.sum(torch.isnan(lambda_k)):
        import pdb; pdb.set_trace()
    return lambda_k 
    #return torch.softmax(side_info+neg_entropy,1) + 1e-8
    #return torch.softmax(p_c_z,1)



def vade_loss_function(recon_x, x, z, z_mean, z_log_var, lambda_k, u_p, lambda_p, theta_p, n_centroids, print_loss = False):
    Z = z.unsqueeze(2).expand(z.size()[0], z.size()[1], n_centroids) # NxDxK
    z_mean_t = z_mean.unsqueeze(2).expand(z_mean.size()[0], z_mean.size()[1], n_centroids)
    z_log_var_t = z_log_var.unsqueeze(2).expand(z_log_var.size()[0], z_log_var.size()[1], n_centroids)
    u_tensor3 = u_p.unsqueeze(0).expand(z.size()[0], u_p.size()[0], u_p.size()[1]) # NxDxK
    lambda_tensor3 = lambda_p.unsqueeze(0).expand(z.size()[0], lambda_p.size()[0], lambda_p.size()[1])
    theta_tensor2 = theta_p.unsqueeze(0).expand(z.size()[0], n_centroids) # NxK
    
#------------------------------------------------------------------------------------------------------------------------
    # Calculate loss
    mse_loss = nn.MSELoss(reduction='sum')
    MSE= mse_loss(recon_x,x)
    logpzc = likelihoods.get_logpzc(z_mean_t,z_log_var_t,lambda_k,u_tensor3, lambda_tensor3)
    qentropy = likelihoods.get_qentropy(z_log_var)
    logpc = likelihoods.get_logpc(lambda_k,theta_tensor2)
    logqcx = likelihoods.get_logqcx(lambda_k)
    if print_loss:
        print("------------------------------------------------------------------------------------------------------")
        print(torch.mean(MSE))
        print(torch.mean(logpzc))
        print(torch.mean(qentropy))
        print(torch.mean(logpc))
        print(torch.mean(logqcx))
        print("------------------------------------------------------------------------------------------------------")

    loss = torch.mean(MSE + logpzc + qentropy + logpc + logqcx)
    #loss = torch.mean(y_entropy_regu)
    if torch.isnan(loss):
        import pdb; pdb.set_trace()
    return loss, MSE.detach().cpu().numpy(),logpzc.detach().cpu().numpy(),qentropy.detach().cpu().numpy(),logpc.detach().cpu().numpy(),logqcx.detach().cpu().numpy()



def dgc_loss_function(recon_x, x, z, z_mean, z_log_var, side_info, lambda_k, u_p, lambda_p, theta_p, n_centroids, print_loss = False):
    Z = z.unsqueeze(2).expand(z.size()[0], z.size()[1], n_centroids) # NxDxK
    z_mean_t = z_mean.unsqueeze(2).expand(z_mean.size()[0], z_mean.size()[1], n_centroids)
    z_log_var_t = z_log_var.unsqueeze(2).expand(z_log_var.size()[0], z_log_var.size()[1], n_centroids)
    u_tensor3 = u_p.unsqueeze(0).expand(z.size()[0], u_p.size()[0], u_p.size()[1]) # NxDxK
    lambda_tensor3 = lambda_p.unsqueeze(0).expand(z.size()[0], lambda_p.size()[0], lambda_p.size()[1])
    theta_tensor2 = theta_p.unsqueeze(0).expand(z.size()[0], n_centroids) # NxK
    
#------------------------------------------------------------------------------------------------------------------------
    # Calculate loss
    #side_info_loss = likelihoods.contrastive_side_info_loss(side_info,mask)/x.size(0)
    side_info_loss = -torch.sum(side_info*lambda_k,1)
    mse_loss = nn.MSELoss(reduction='sum')
    MSE= mse_loss(recon_x,x)
    #MSE = torch.sum((recon_x-x)**2,1)
    logpzc = likelihoods.get_logpzc(z_mean_t,z_log_var_t,lambda_k,u_tensor3, lambda_tensor3)
    qentropy = likelihoods.get_qentropy(z_log_var)
    logpc = likelihoods.get_logpc(lambda_k,theta_tensor2)
    logqcx = likelihoods.get_logqcx(lambda_k)
    if print_loss:
        print("------------------------------------------------------------------------------------------------------")
        print(torch.mean(side_info_loss))
        print(torch.mean(MSE))
        print(torch.mean(logpzc))
        print(torch.mean(qentropy))
        print(torch.mean(logpc))
        print(torch.mean(logqcx))
        print("------------------------------------------------------------------------------------------------------")

    loss = torch.mean(side_info_loss + MSE + (logpzc + qentropy + logpc + logqcx))
    #loss = torch.mean(y_entropy_regu)
    if torch.isnan(loss):
        import pdb; pdb.set_trace()
    return loss



def vae_loss_function(recon_x, x, z_mean, z_log_var, sigma=0, bce=True, print_loss = False):
    #------------------------------------------------------------------------------------------------------------------------
    # Calculate loss
    recon_x = recon_x.view(recon_x.size(0), -1)
    x = x.view(x.size(0), -1)
    #BCE = likelihoods.get_BCE(x,recon_x)
    bce_loss = nn.BCELoss(reduction='sum')
    mse_loss = nn.MSELoss(reduction='sum')
    if bce:
        recon_loss = bce_loss(recon_x,x)/x.size(0)
    else:
        recon_loss = mse_loss(recon_x,x)
    KLD = -0.5 * torch.mean(1 + z_log_var - z_mean**2 - z_log_var.exp(), 1)
    if print_loss:
        print("------------------------------------------------------------------------------------------------------")
        print(torch.mean(recon_loss))
        print(torch.mean(KLD))
        print("------------------------------------------------------------------------------------------------------")

    loss = torch.mean(recon_loss + KLD )
    #loss = torch.mean(y_entropy_regu)
    #if torch.isnan(loss):
    #    import pdb; pdb.set_trace()
    return loss, float(torch.mean(recon_loss).detach().cpu().numpy()),float(torch.mean(KLD).detach().cpu().numpy())



def reparameterize(mu, logvar, train):
    if train:
        std = torch.exp(logvar*0.5)
        return std * torch.randn_like(mu) + mu
    else:
      return mu




def adjust_learning_rate(init_lr, optimizer, epoch):
    lr = max(init_lr * (0.9 ** (epoch//10)), 0.0001)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr



def create_gmmparam(n_centroids, z_dim, fix_prior = False):
    theta_p = nn.Parameter(torch.ones(n_centroids)/n_centroids)
    #theta_p.requires_grad = False
    u_p = nn.Parameter(torch.zeros(z_dim, n_centroids))
    lambda_p = nn.Parameter(torch.ones(z_dim, n_centroids))
    if fix_prior:
        theta_p.requires_grad=False
        u_p.requires_grad = False
        lambda_p.requires_grad = False
    return u_p, theta_p, lambda_p




def shuffle(length,seed):
    np.random.seed(seed)
    new_order = list(np.arange(length))
    np.random.shuffle(new_order)
    return new_order



def cluster_acc(Y_pred, Y):
  #from sklearn.utils.linear_assignment_ import linear_assignment
  from scipy.optimize import linear_sum_assignment as linear_assignment
  assert Y_pred.size == Y.size
  D = max(Y_pred.max(), Y.max())+1
  w = np.zeros((D,D), dtype=np.int64)
  for i in range(Y_pred.size):
    w[Y_pred[i], Y[i]] += 1
  ind = linear_assignment(w.max() - w)
  ind = np.concatenate((ind[0].reshape(1,len(ind[0])),ind[1].reshape(1,len(ind[0])))).transpose()
  return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w



def pred_acc(pred,prob,label,problem='classification',mode = 'max'):
    if problem=="classification":
        if mode=='max':
            index = np.argmax(prob,1)
            class_pred = np.argmax(pred[np.arange(len(index)),index,:],1)
            pred_acc = np.sum(class_pred==label)/len(label)
        else:
            class_pred = np.argmax(np.sum(prob.reshape(prob.shape[0],prob.shape[1],1)*pred,1),1)
            pred_acc = np.sum(class_pred==label)/len(label)
    elif problem=="multi_binary_classification":
        if mode=='max':
            index = np.argmax(prob,1)
            class_pred = pred[np.arange(len(index)),index,:]
            class_pred[class_pred>=0.5]=1
            class_pred[class_pred<0.5]=0
            pred_acc = np.mean(np.mean(class_pred==label,1))
        else:
            class_pred = np.sum(prob.reshape(prob.shape[0],prob.shape[1],1)*pred,1)
            class_pred[class_pred>=0.5]=1
            class_pred[class_pred<0.5]=0
            pred_acc = np.mean(np.mean(class_pred==label,1))
    return pred_acc



def derive_feature(dataloader,pretrained_net,pretrained_choice,label=True):
    if label:
        feature = []
        label = []
        for i,(j,k) in enumerate(dataloader):
            print(i)
            label.append(k.numpy())
            if torch.cuda.is_available():
                j = j.cuda()
            temp = pretrained_net(j)
            if pretrained_choice=='resnet50':
                temp = torch.mean(temp,(2,3))
            temp = temp.detach().cpu().numpy()
            feature.append(temp)
        feature = np.concatenate(feature)
        label = np.concatenate(label)
        return feature,label
    else:
        feature = []
        for i,j in enumerate(dataloader):
            print(i)
            j = j[0]
            if torch.cuda.is_available():
                j = j.cuda()
            temp = pretrained_net(j)
            if pretrained_choice=='resnet50':
                temp = torch.mean(temp,(2,3))
            temp = temp.detach().cpu().numpy()
            feature.append(temp)

        feature = np.concatenate(feature)
        return feature

def gen_features(dataset,num_clusters,pretrained_choice = 'resnet50',combined=False,kmeans_test = False, save_feature = False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if pretrained_choice=='resnet50':
        resnet50 = models.resnet50(pretrained=True)
        pretrained_net = nn.Sequential(*list(resnet50.children())[:-2])
        pretrained_net.eval()
    elif pretrained_choice=='vit':
        pretrained_net = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
        pretrained_net.eval()

    if torch.cuda.is_available():
        pretrained_net = pretrained_net.cuda()

    if dataset=="stl":
        trainloader = torch.utils.data.DataLoader(
            torchvision.datasets.STL10('./data', split = 'train',
               transform=torchvision.transforms.Compose([
                                       torchvision.transforms.Resize(32),
                                       torchvision.transforms.ToTensor(),
                                       normalize
                                     ]), download=True),
          batch_size=256, shuffle=True)

        testloader = torch.utils.data.DataLoader(
            torchvision.datasets.STL10('./data', split = 'test',
               transform=torchvision.transforms.Compose([
                                       torchvision.transforms.Resize(32),
                                       torchvision.transforms.ToTensor(),
                                       normalize
                                     ]), download=True),
          batch_size=256, shuffle=True)
        train_feature,train_label = derive_feature(trainloader,pretrained_net)
        test_feature,test_label = derive_feature(testloader,pretrained_net)

    elif dataset=='cifar10':
        trainloader = torch.utils.data.DataLoader(
                        torchvision.datasets.CIFAR10('./data', train = True, transform=torchvision.transforms.Compose([
                                               torchvision.transforms.Resize(224),
                                               torchvision.transforms.ToTensor(),
                                               normalize
                                             ]),download = True),
                        batch_size = 16,shuffle=True)
        testloader = torch.utils.data.DataLoader(
                        torchvision.datasets.CIFAR10('./data', train = False, transform=torchvision.transforms.Compose([
                                               torchvision.transforms.Resize(224),
                                               torchvision.transforms.ToTensor(),
                                               normalize
                                             ]),download = True),
                        batch_size = 16,shuffle=True)

        train_feature,train_label = derive_feature(trainloader,pretrained_net)
        test_feature,test_label = derive_feature(testloader,pretrained_net)
    elif dataset=='cifar100':
        path = './'
        data=scio.loadmat(path+'train.mat')
        train_x = data['data']
        train_y = data['fine_labels']
        train_c = data['coarse_labels']


        data=scio.loadmat(path+'test.mat')
        test_x = data['data']
        test_y = data['fine_labels']
        test_c = data['coarse_labels']

        '''
        y = []
        c = []
        y.append(train_y)
        y.append(test_y)
        c.append(train_c)
        c.append(test_c)
        total_y = np.concatenate(y,axis=0)
        total_c = np.concatenate(c,axis=0)
        y = torch.from_numpy(y)
        c = torch.from_numpy(c)
        '''

        train_x = np.reshape(train_x,(-1,3,32,32))
        test_x = np.reshape(test_x,(-1,3,32,32))

        train_x = torch.from_numpy(train_x.astype('float32')/255)
        test_x = torch.from_numpy(test_x.astype('float32')/255)



        transform=torchvision.transforms.Compose([
                                               torchvision.transforms.Resize(224),
                                               normalize
                                             ])

        train_x = transform(train_x)
        test_x = transform(test_x)

        trainloader = DataLoader(TensorDataset(train_x),batch_size=128,shuffle=False)
        testloader = DataLoader(TensorDataset(test_x),batch_size=128,shuffle=False)
        with torch.no_grad():
            train_feature = derive_feature(trainloader,pretrained_net,pretrained_choice,False)
            test_feature = derive_feature(testloader,pretrained_net,pretrained_choice,False)
        train_label = np.concatenate([train_y,train_c],axis=1)
        test_label = np.concatenate([test_y,test_c],axis=1)
    elif dataset=='generic':
        # load data
        train_x = torch.from_numpy(np.load("train_im.npy")).float()
        train_y = np.load("train_res.npy")
        train_c = np.load("train_cluster_index.npy").astype(int)
        test_x = torch.from_numpy(np.load("test_im.npy")).float()
        test_y = np.load("test_res.npy")
        test_c = np.load("test_cluster_index.npy").astype(int)
        transform=torchvision.transforms.Compose([
                                               normalize
                                             ])
        train_x = transform(train_x)
        test_x = transform(test_x)
        trainloader = DataLoader(TensorDataset(train_x),batch_size=16,shuffle=False)
        testloader = DataLoader(TensorDataset(test_x),batch_size=16,shuffle=False)
        train_feature = derive_feature(trainloader,pretrained_net,pretrained_choice,False)
        test_feature = derive_feature(testloader,pretrained_net,pretrained_choice,False)
        train_label = train_c
        test_label = test_c

    if combined:
        total_feature,total_label = np.concatenate([train_feature,test_feature]),np.concatenate([train_label,test_label])
        if kmeans_test:
            kmeans = KMeans(n_clusters=num_clusters,random_state=0).fit(total_feature)
            cls_index = kmeans.labels_
            if dataset=='cifar100':
                acc,_ = cluster_acc(cls_index,total_label[:,1])
                NMI = metrics.normalized_mutual_info_score(total_label[:,1], cls_index,average_method='arithmetic') 
                print('| Kmeans ACC = {:6f} NMI = {:6f}'.format(acc,NMI))                
            else:
                acc,_ = cluster_acc(cls_index,total_label)
                NMI = metrics.normalized_mutual_info_score(total_label, cls_index,average_method='arithmetic') 
                print('| Kmeans ACC = {:6f} NMI = {:6f}'.format(acc,NMI))

        if save_feature:
            np.save("total_f_"+pretrained_choice,total_feature)
            np.save("total_l_"+pretrained_choice,total_label)
        return total_feature,total_label

    else:
        np.save("train_f_"+pretrained_choice,train_feature)
        np.save("train_l_"+pretrained_choice,train_label)
        np.save("test_f_"+pretrained_choice,test_feature)
        np.save("test_l_"+pretrained_choice,test_label)      
        return train_feature,train_label,test_feature,test_label


# calculate the entropy of a stochastic matrix along a given axis
def entropy_loss(prob,axis=0):
    return -torch.sum(prob*torch.log(prob+1e-8),dim=axis)



def ensemble_gaussian_like(y,mu,log_var,n_centroids):
    y = y.expand(y.shape[0],n_centroids)
    y_likelihoods = 1/torch.sqrt(2*math.pi*torch.exp(log_var))*torch.exp(-0.5*(y-mu)**2/torch.exp(log_var))
    return y_likelihoods


def ensemble_gaussian_loglike(y,mu,log_var,n_centroids):
    y = y.expand(y.shape[0],n_centroids)
    log_y_likelihoods = -0.5*(math.log(2*math.pi) + log_var) - 0.5*(y-mu)**2/torch.exp(log_var)
    return log_y_likelihoods


def check(a,b,index):
    target = a[index]
    for i in range(b.shape[0]):
        if np.sum(target==b[i]) == target.shape[1]*target.shape[2]*target.shape[3]:
            return 1
    return 0



# A linear evaluation layer
class linear_eval(nn.Module):
    def __init__(self,f_dim,num_class,problem='image'):
        super(linear_eval, self).__init__()
        self.f_dim = f_dim
        self.num_class = num_class
        #self.linear_eval = nn.Sequential(nn.Linear(f_dim,512),nn.ReLU(),nn.Linear(512,100),nn.Softmax(dim=1))
        if problem=='image':
            self.linear_eval = nn.Sequential(nn.Linear(f_dim,num_class),nn.Softmax(dim=1))
        else:
            self.linear_eval = nn.Sequential(buildNetwork([f_dim] + problem),nn.Linear(problem[-1],num_class))

    def forward(self,x):
        return self.linear_eval(x)



# A neural network training/evaluating function
def trainer(model,data, b_size, optimizer,loss_fn, num_epochs,scheduler=None):
    train_x,train_y,test_x,test_y = torch.tensor(data[0][0]).float(),torch.tensor(data[0][1]).float(),torch.tensor(data[1][0]).float(),torch.tensor(data[1][1]).float()
    trainloader = DataLoader(TensorDataset(train_x,train_y),batch_size=b_size,shuffle=True)
    testloader = DataLoader(TensorDataset(test_x,test_y),batch_size=b_size,shuffle=False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
    for epoch in range(num_epochs):
        train_loss, test_loss= 0.0, 0.0
        train_pred, train_label, test_pred, test_label = [], [], [], []
        # train the model
        model.train()
        for i,(inputs,labels) in enumerate(trainloader):
            if use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            preds = model(inputs)
            loss = loss_fn(preds,labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_pred.append(np.argmax(preds.detach().cpu().numpy(),1))
            train_label.append(labels.cpu().numpy())
        # evaluate the model
        model.eval()
        for i,(inputs,labels) in enumerate(testloader):
            if use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            preds = model(inputs)
            loss = loss_fn(preds,labels)
            test_loss += loss.item()
            test_pred.append(np.argmax(preds.detach().cpu().numpy(),1))
            test_label.append(labels.cpu().numpy())
        if scheduler:
            scheduler.step()
        train_acc = np.sum(np.concatenate(train_pred)==np.concatenate(train_label))/len(np.concatenate(train_label))
        test_acc = np.sum(np.concatenate(test_pred)==np.concatenate(test_label))/len(np.concatenate(test_label))
        print("#Epoch %3d:,Train Loss: %.5f, Valid Loss: %.5f, Train ACC:%.5f, Test ACC:%.5f" % (
            epoch, train_loss / len(trainloader), test_loss / len(testloader), train_acc, test_acc))
    return model


# A simple function for sampling from the learned model
def sample_model(model,num_simulation_points,task_name):
    reconstruction = []
    side_info = []
    latent_code = []

    mean = model.u_p.detach().numpy()
    var = model.lambda_p.detach().numpy()

    num_clusters = mean.shape[1]

    for i in range(num_clusters):
        latent_i = np.random.multivariate_normal(mean[:,i],np.diag(var[:,i]),num_simulation_points)
        latent_code.append(latent_i)
    latent_code = torch.tensor(np.concatenate(latent_code,axis=0)).float()
    if task_name == 'regression':
        gen_mean = torch.sigmoid(model.out_mu(latent_code))
        gen_var = model.out_log_sigma(latent_code)
        for i in range(num_clusters):
            side_info.append(gen_mean[i*num_simulation_points:(i+1)*num_simulation_points,i])
        side_info = torch.stack(side_info).unsqueeze(-1)
    else:
        gen_probs = torch.softmax(model.prob_ensemble(latent_code).reshape(latent_code.shape[0],model.n_centroids,model.y_dim),2)
        for i in range(num_clusters):
            side_info.append(gen_probs[i*num_simulation_points:(i+1)*num_simulation_points,i,:])
        side_info = torch.stack(side_info)
    x_samples = model._dec(model.decoder(latent_code))
    if model._dec_act is not None:
        x_samples = model._dec_act(x_samples)
    x_samples = x_samples.detach().numpy()
    latent_code = latent_code.detach().numpy()
    side_info = side_info.detach().numpy()
    return latent_code,x_samples,side_info


# Example
'''
# An example for image data
import numpy as np
import util
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler_choice
train_x = np.load("train_f_resnet50.npy")
train_y = np.load("train_l_resnet50.npy")[:,0]
test_x = np.load("test_f_resnet50.npy")
test_y = np.load("test_l_resnet50.npy")[:,0]
data = [[train_x,train_y],[test_x,test_y]]
loss_fn = nn.CrossEntropyLoss()
model = util.linear_eval(train_x.shape[1],100,problem='image')
optimizer = optim.Adam(model.parameters(), lr=0.0001) 
#scheduler = scheduler_choice.StepLR(optimizer, step_size=40, gamma=0.1)
model = util.trainer(model,data,128,optimizer,loss_fn,4000)


# An example for pacman data
import numpy as np
import util
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler_choice
x = np.load("pacman_data.npy")
y = np.load("pacman_response_linear_exp.npy")
y = y.reshape(len(y),1)
np.random.seed(123)
index = np.arange(x.shape[0])
np.random.shuffle(index)
train_point = int(len(index)*0.8)
data = [[x[index[:train_point],:],y[index[:train_point]]],[x[index[train_point:],:],y[index[train_point:]]]]
loss_fn = nn.MSELoss(reduction='mean')
model = util.linear_eval(2,1,problem=[20])
optimizer = optim.Adam(model.parameters(), lr=0.0001) 
#scheduler = scheduler_choice.StepLR(optimizer, step_size=40, gamma=0.1)
model = util.trainer(model,data,512,optimizer,loss_fn,200)
'''




#------------------------------------------------------------------------------------------------------------------
# The followings are some functions for constraint clustering



# This function turns labels into masked constraints. For instance, for a sample with label 2 some other samples
# with labels [2,3,0,5,7,2,2,0], the mask would be [1,0,0,0,0,1,1,0], i.e. we looks for samples with the same labels
# Input: l (a length n torch array)
def label_to_mask(l):
    mask = []
    for i in range(len(l)):
        sub_mask = np.zeros(len(l))
        curr_l = l[i]
        for j in range(len(l)):
            if l[j]==curr_l and j!=i:
                sub_mask[j] = 1
        mask.append(sub_mask)
    return torch.tensor(np.array(mask)).float()



# This function is to augment a batch of images
def data_augmentation(im_batch,l_batch, aug_strategy,aug_factor=1,colored=False):

    side_info_mask = torch.zeros(im_batch.shape[0]*aug_factor)
    final_side_info_mask = []
    im_size = (im_batch.shape[-2],im_batch.shape[-1])

    if aug_strategy == 'standard':
        # Standard augmentation strategy
        aug_f = transforms.Compose([
            transforms.RandomResizedCrop(size = im_size),
            transforms.RandomHorizontalFlip()
        ])
    elif aug_strategy == 'simclr':
        # Augmentation strategy from the SimCLR paper
        if colored:
            aug_f = transforms.Compose([
                transforms.RandomResizedCrop(size = im_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=.5, hue=.3)
                ], p=0.5),
                transforms.RandomGrayscale(p=0.1),
            ])
        else:
            aug_f = transforms.Compose([
                #transforms.RandomResizedCrop(size = im_size),
                #transforms.RandomHorizontalFlip()
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5)
            ])
    else:
        raise ValueError('Invalid augmentation strategy')

    final_im = [im_batch]
    for i in range(aug_factor):
        final_im.append(aug_f(im_batch))
    final_im = torch.cat(final_im)
    for i in range(final_im.shape[0]):
        side_info_mask = torch.zeros(im_batch.shape[0]*(aug_factor+1))
        side_info_mask[np.delete(np.arange(i%im_batch.shape[0],final_im.shape[0],im_batch.shape[0]),i//im_batch.shape[0])]=1
        final_side_info_mask.append(side_info_mask)
    final_side_info_mask = torch.stack(final_side_info_mask)
    return final_im, torch.cat([l_batch]*(aug_factor+1)), final_side_info_mask.float()











