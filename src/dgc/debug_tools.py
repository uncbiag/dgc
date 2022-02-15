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
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import dgc.likelihoods as dgc



##########################################################################################
##### This file contains a collection of functions for general dubugging purposes ########
##########################################################################################



# This function visualizes the latent space. It requires a function that maps the data points
# into the latent space and a dataloader that contains the data.
def visual_latent_space(enc,loader,b_size, num_comps, pre_num_comps = 0, pre_reduce = False,tsne = False):
    l_code = []
    order = np.arange(len(loader))
    for i in range(0,len(order),b_size):
        if i+b_size < len(order):
            batch = order[np.arange(i,i+b_size)]
        else:
            batch = order[np.arange(i,len(order))]
        inputs, cluster_index = loader[batch]
        if use_cuda:
            inputs = inputs.cuda()
        z, outputs, mu, logvar = enc(inputs)
        l_code.append(z.detach().cpu().numpy())
    l_code = np.concatenate(l_code)
    if tsne:
        if pre_reduce:
            pre_embed = PCA(n_components = pre_num_comps).fit_transform(l_code)
            embed = TSNE(n_components=num_comps).fit_transform(pre_embed)
        else:
            mbed = TSNE(n_components=num_comps).fit_transform(l_code)
    else:
        embed = PCA(n_components = num_comps).fit_transform(l_code)

    return embed




# This function visualizes a generic set of high-dimensional features
# input feature shape: N x F (pytorch tensor)

def visual_generic_f(input_f, num_comps, pre_num_comps = 0, pre_reduce = False,tsne = False):
    if str(type(input_f))!="<class 'numpy.ndarray'>":
        input_f = input_f.detach().cpu().numpy()
    if len(input_f.shape)==2:
        if tsne:
            if pre_reduce:
                pre_embed = PCA(n_components = pre_num_comps).fit_transform(input_f)
                embed = TSNE(n_components=num_comps).fit_transform(pre_embed)
            else:
                embed = TSNE(n_components=num_comps).fit_transform(input_f)
        else:
            embed = PCA(n_components = num_comps).fit_transform(input_f)
    else:
        embed = []
        for i in range(input_f.shape[0]):
            if tsne:
                if pre_reduce:
                    pre_embed = PCA(n_components = pre_num_comps).fit_transform(input_f[i])
                    embed.append(TSNE(n_components=num_comps).fit_transform(pre_embed))
                else:
                    embed.append(TSNE(n_components=num_comps).fit_transform(input_f[i]))
            else:
                embed.append(PCA(n_components = num_comps).fit_transform(input_f[i]))

    return np.array(embed)





# This function visualizes the learned log-variances (or variances) through plotting a histogram
# that summarizes their values
# f_vals: N x F (pytorch tensor)
# N: number of clusters; F: Feature Dimension
def hist_values(f_vals,num_bins):
    if str(type(f_vals))!="<class 'numpy.ndarray'>":
        f_vals = f_vals.detach().cpu().numpy()
    if len(f_vals)==2:
        f_vals = f_vals.flatten()
        plt.hist(f_vals, density=True, bins=num_bins)  # density=False would make counts
        plt.ylabel('Probability')
        plt.xlabel('Data');
        plt.savefig('hist.jpg')
    else:
        for i in range(f_vals.shape[0]):
            print(i)
            temp_f_vals = f_vals[i].flatten()
            plt.hist(temp_f_vals, density=True, bins=num_bins)  # density=False would make counts
            plt.ylabel('Probability')
            plt.xlabel('Logvar');
            plt.savefig('/playpen-raid/yifengs/dgc/mnist_test/'+str(i)+'.jpg')
    #plt.show()





# This is a specific function for debugging VaDE.
def check_prob_kl(z_mean, z_log_var, lambda_k, u_p, lambda_p, theta_p, n_centroids):
    z_mean = z_mean.unsqueeze(2).expand(z_mean.size()[0], z_mean.size()[1], n_centroids)
    z_log_var = z_log_var.unsqueeze(2).expand(z_log_var.size()[0], z_log_var.size()[1], n_centroids)
    u_p = u_p.unsqueeze(0).expand(z_mean.size()[0], u_p.size()[0], u_p.size()[1]) # NxDxK
    lambda_p = lambda_p.unsqueeze(0).expand(z_mean.size()[0], lambda_p.size()[0], lambda_p.size()[1])
    theta_p= theta_p.unsqueeze(0).expand(z_mean.size()[0], n_centroids) # NxK

    logpzc = 0.5 * torch.sum(
          torch.log(2*np.pi*lambda_p) +
          torch.exp(z_log_var) / lambda_p +
                   (z_mean - u_p) ** 2 / lambda_p,
          dim=1)

    qentropy = -0.5*torch.sum(1+z_log_var+math.log(2*math.pi), dim=1)

    kl = logpzc + qentropy

    return int(torch.sum(torch.max(lambda_k,1)[1]==torch.min(kl,1)[1]))




# This is to check pairwise KL distance between learned variational posteriors in VaDE
def pairwise_kl(z_mean, z_var,  u_p, var_p, need_transpose=True):

    b_size = z_mean.size()[0]

    if need_transpose:
        u_p = torch.transpose(u_p,1,0)
        var_p = torch.transpose(var_p,1,0)

    z_mean = z_mean.unsqueeze(2).expand(z_mean.size()[0], z_mean.size()[1], b_size)
    z_var = z_var.unsqueeze(2).expand(z_var.size()[0], z_var.size()[1], b_size)
    u_p = u_p.unsqueeze(0).expand(b_size, u_p.size()[0], u_p.size()[1]) # NxDxK
    var_p = var_p.unsqueeze(0).expand(b_size, var_p.size()[0], var_p.size()[1])

    logpzc = 0.5 * torch.sum(
          torch.log(2*np.pi*var_p) +
                z_var / var_p +
                   (z_mean - u_p) ** 2 / var_p,
          dim=1)

    qentropy = -0.5*torch.sum(1+torch.log(z_var)+math.log(2*math.pi), dim=1)

    kl = logpzc + qentropy

    return kl


# This is to check the entropy of a discrete distribution from cluster indices
# Input: cluster_ind (numpy array)
# Input shape: n (the number of samples)
def discrete_entropy(cluster_ind):
    unique,count = np.unique(cluster_ind,return_counts=True)
    dist = count/np.sum(count)
    return -np.sum(dist*np.log(dist))
    #num_classes = len(np.unique(cluster_ind))
    #dist = [0]*num_classes
    #for i in range(num_classes):
    #    dist[i] = len(np.where(cluster_ind==i)[0])/len(cluster_ind)
    #dist = np.array(dist)
    #return -np.sum(dist*np.log(dist))



# This is to check clustering consistency. I.e., this is to check samples that truly belong 
# to the same clsuter will be assigned to the same cluster
def check_clus_consistency(pred,truth,summary=False,approx_true_error=False):
    truth_uni,truth_count = np.unique(truth,return_counts=True)
    pred_entropy = []
    overall_count = []
    error = 0
    for i in range(len(truth_uni)):
        truth_index = np.where(truth==truth_uni[i])[0]
        pred_uni,pred_count = np.unique(pred[truth_index],return_counts=True)
        if approx_true_error:
            error += np.sum(np.sort(pred_count)[:len(pred_count)-1])
        pred_entropy.append(discrete_entropy(pred[truth_index]))
        overall_count.append(pred_count/np.sum(pred_count))
    if approx_true_error:
        error/=len(pred)
        print("Approximated error:",error)
    if summary:
        return np.array(pred_entropy)
    else:
        return np.array(overall_count)



# This is to check clustering confidence. I.e., this is to check samples that are clustered 
# in a confident manner and samples that are not. 
def check_confidence(prob,pred,truth):
    truth_uni,truth_count = np.unique(truth,return_counts=True)
    right_prob = []
    wrong_prob = []
    for i in range(len(truth_uni)):
        wrong_max = 0
        truth_index = np.where(truth==truth_uni[i])[0]
        pred_digit = pred[truth_index]
        prob_digit = prob[truth_index]
        pred_uni,pred_count = np.unique(pred_digit,return_counts=True)
        max_pos = np.argmax(pred_count)
        other_pos = [i for i in range(len(pred_count))]
        del other_pos[max_pos]
        right_prob.append(np.max(prob_digit[np.where(pred_digit==pred_uni[max_pos])[0]]))
        for j in other_pos:
            sub_prob = prob_digit[np.where(pred_digit==pred_uni[j])[0]]
            if np.max(sub_prob)>wrong_max:
                wrong_max = np.max(sub_prob)
        wrong_prob.append(wrong_max)
    return np.array([right_prob,wrong_prob])










