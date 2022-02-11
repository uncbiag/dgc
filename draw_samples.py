import os
import torch
import math
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import time
from sklearn import datasets
import matplotlib.pyplot as plt
from vae_clustering import VaDE
from dgc_w_o_balancing import dgc_w_o_balancing
from dgc_w_balancing import dgc_w_balancing
from dgc_w_balancing_new import dgc_w_balancing_new
import argparse
from util_class import split_train_test
from data_loader import pacman_loader
from pretrain_vae import pretrain
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from new_dgc_w_o_balancing import new_dgc_w_o_balancing
from dgc_entropy import dgc_entropy
from sklearn.manifold import TSNE


def get_pcz(z,u_p,lambda_p,theta_p):
    Z = z.unsqueeze(2).expand(z.size()[0], z.size()[1], 2) # NxDxK
    u_tensor3 = u_p.unsqueeze(0).expand(z.size()[0], u_p.size()[0], u_p.size()[1]) # NxDxK
    lambda_tensor3 = lambda_p.unsqueeze(0).expand(z.size()[0], lambda_p.size()[0], lambda_p.size()[1])
    theta_tensor2 = theta_p.unsqueeze(0).expand(z.size()[0], 2) # NxK
    #gamma_x = torch.clamp(gamma_x,min=0.5)
    p_c_z = torch.exp(torch.log(theta_tensor2) - torch.sum(0.5*torch.log(2*math.pi*lambda_tensor3)+\
        (Z-u_tensor3)**2/(2*lambda_tensor3), dim=1)) + 1e-10 # NxK
    p_c_z = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)


    return p_c_z


def cluster_acc(Y_pred, Y):
  from sklearn.utils.linear_assignment_ import linear_assignment
  assert Y_pred.size == Y.size
  D = max(Y_pred.max(), Y.max())+1
  w = np.zeros((D,D), dtype=np.int64)
  for i in range(Y_pred.size):
    w[Y_pred[i], Y[i]] += 1
  ind = linear_assignment(w.max() - w)
  return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w


def get_lambda_k( z, z_mean, z_log_var, u_p, lambda_p, theta_p, y_likelihood, gamma_x):
    Z = z.unsqueeze(2).expand(z.size()[0], z.size()[1], 2) # NxDxK
    z_mean_t = z_mean.unsqueeze(2).expand(z_mean.size()[0], z_mean.size()[1], 2)
    z_log_var_t = z_log_var.unsqueeze(2).expand(z_log_var.size()[0], z_log_var.size()[1], 2)
    u_tensor3 = u_p.unsqueeze(0).expand(z.size()[0], u_p.size()[0], u_p.size()[1]) # NxDxK
    lambda_tensor3 = lambda_p.unsqueeze(0).expand(z.size()[0], lambda_p.size()[0], lambda_p.size()[1])
    theta_tensor2 = theta_p.unsqueeze(0).expand(z.size()[0], 2) # NxK
    #gamma_x = torch.clamp(gamma_x,min=0.5)
    p_c_z = torch.exp(torch.log(theta_tensor2) - torch.sum(0.5*torch.log(2*math.pi*lambda_tensor3)+\
        (Z-u_tensor3)**2/(2*lambda_tensor3), dim=1)) + 1e-10 # NxK
    p_c_z = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)
    #p_c_z = torch.log(theta_tensor2) - torch.sum(0.5*torch.log(2*math.pi*lambda_tensor3)+\
    #    (Z-u_tensor3)**2/(2*lambda_tensor3), dim=1)
    #p_c_z = torch.softmax(p_c_z,1)
    lambda_k = (gamma_x/(1-gamma_x))*torch.log(y_likelihood) + torch.log(p_c_z)
    lambda_k = torch.softmax(lambda_k,1)
    #lambda_k = torch.softmax(((y_likelihood)**gamma_x)*p_c_z,1) + 1e-30
    #lambda_k = (((y_likelihood)**gamma_x)*p_c_z)/torch.sum(((y_likelihood)**gamma_x)*p_c_z,dim=1,keepdim=True)

    return lambda_k

def reparameterize(mu, logvar):
  std = logvar.mul(0.5).exp_()
  eps = Variable(std.data.new(std.size()).normal_())
  # num = np.array([[ 1.096506  ,  0.3686553 , -0.43172026,  1.27677995,  1.26733758,
  #       1.30626082,  0.14179629,  0.58619505, -0.76423112,  2.67965817]], dtype=np.float32)
  # num = np.repeat(num, mu.size()[0], axis=0)
  # eps = Variable(torch.from_numpy(num))
  return mu



def ensemble_gaussian_like(y,mu,log_var):
    y = y.expand(y.shape[0],2)
    y_likelihoods = 1/torch.sqrt(2*math.pi*torch.exp(log_var))*torch.exp(-0.5*(y-mu)**2/torch.exp(log_var))
    return y_likelihoods

pacman_data = np.load("pacman_data.npy")
pacman_classes = np.load("pacman_classes.npy")
pacman_res = np.load("pacman_response_linear_exp.npy")
miss = np.load("miss.npy")


# DGC ENTROPY
#model = dgc_entropy(input_dim=2, z_dim=80, y_dim = 1, n_centroids=2, binary=True,
#        encodeLayer=[64,128,256,256], decodeLayer=[256,128,64])
#model.load_model("entropy_exp.pt")

'''
# split the dataset
split = split_train_test(0.8,123)
pacman_dataset = [pacman_data,classes,res]
train_d,test_d = split(pacman_dataset)


data = train_d[0]
b= np.zeros(len(train_d[2]))
b[miss]=1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(data[:,0], data[:,1], train_d[2],c=b, marker='o')
ax.set_xlabel('Pacman X-axis')
ax.set_ylabel('Pacman Y-axis')
ax.set_zlabel('Generated Non-linear Response')
ax.set_yticklabels([])
ax.set_xticklabels([])
#ax.view_init(0, 180)
plt.show()
'''


# VAE with Gaussian mixture (VaDE)
#model = VaDE(input_dim=2, z_dim=80, y_dim = 1, n_centroids=2, binary=True,
#        encodeLayer=[64,128,256,256], decodeLayer=[256,128,64])
#model.load_model("vade.pt")


#wo_balance6
#model = dgc_w_o_balancing(input_dim=2, z_dim=80, y_dim = 1, n_centroids=2, binary=True,
#       encodeLayer=[64,128,256,256], decodeLayer=[256,256,128,64])
#model.load_model("wo_balance9.pt")

#wo_balance10 (linear exponential)
#model = new_dgc_w_o_balancing(input_dim=2, z_dim=80, y_dim = 1, n_centroids=2, binary=True,
#        encodeLayer=[64,128,256,256], decodeLayer=[256,128,64])
#model.load_model("wo_balance10.pt")


model = dgc_entropy(input_dim=2, z_dim=80, y_dim = 1, n_centroids=2, binary=True,
        encodeLayer=[64,128,256,256], decodeLayer=[256,128,64])
model.load_model("entropy.pt")


# PCA plot
a = np.load("pacman_data.npy")
b = np.load("pacman_classes.npy")
c = np.load("pacman_response_linear_exp.npy")
from util_class import split_train_test
split = split_train_test(0.8,123)
train_d,test_d = split([a,b,c])
a,b,c = test_d[0],test_d[1],test_d[2]
a = torch.tensor(a).float()
z, output1, output2, mu, logvar = model.forward(a)
z = z.detach().numpy()
z_embedded = TSNE(n_components=2).fit_transform(z)
#pca = PCA(n_components=2)
#pca.fit(z)
#comp = pca.components_
#plt.plot(comp[1,:],comp[0,:])
#plt.show()





# linear case
#model = new_dgc_w_o_balancing(input_dim=2, z_dim=80, y_dim = 1, n_centroids=2, binary=True,
#        encodeLayer=[64,128,256,256], decodeLayer=[256,128,64])
#model.load_model("wo_balance_linear.pt")

'''
data = np.load("pacman_data.npy")
classes = np.load("pacman_classes.npy")
res = np.load("pacman_response_linear_exp.npy")
from util_class import split_train_test
split = split_train_test(0.8,123)
train_d,test_d = split([data,classes,res])
data,classes,res = test_d[0],test_d[1],test_d[2]
data = torch.tensor(data).float()
res = torch.tensor(res).float()
res = res.reshape(len(res),1)


h = model.encoder(data)
mu = model._enc_mu(h)
logvar = model._enc_log_sigma(h)
z = model.reparameterize(mu, logvar)
y_preds_mu = torch.sigmoid(model.out_mu(z))
y_preds_log_var = model.out_log_sigma(z)
# Calculate response likelihood under different clusters
y_likelihood = torch.clamp(ensemble_gaussian_like(res,y_preds_mu,y_preds_log_var),1e-08)

norm_y = y_likelihood/torch.sum(y_likelihood,1,keepdim=True)+1e-40
gamma_x = 1-torch.sum(-norm_y*torch.log(norm_y),1)/math.log(2)
gamma_x = -0.4 + torch.sigmoid(gamma_x)*1.7782
gamma_x = gamma_x.reshape(len(gamma_x),1) # Nx1

lambda_k = get_lambda_k(z,mu,logvar,model.u_p,model.lambda_p,model.theta_p,y_likelihood,gamma_x)
pred = torch.max(lambda_k,1)[0].detach().numpy().astype(int)
acc = cluster_acc(pred,classes)
print(acc[0])

'''


'''
z = np.load("latent_rep.npy")
classes = np.load("ground_truth.npy")
pred = np.load("pred.npy")
pca = PCA(n_components=3)
pca.fit(z)
pca_coor = np.matmul(z,pca.components_.reshape(16,3))
plt.scatter(pca_coor[:,0],pca_coor[:,2],c=pred)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Ground Truth Clusters")
plt.show()

'''






#------------------------------------------------------------------------------------------------------------
# Draw Samples and plot 
#------------------------------------------------------------------------------------------------------------
'''
mean = model.u_p.detach().numpy()
var = model.lambda_p.detach().numpy()


sample1 = np.random.multivariate_normal(mean[:,0],np.diag(var[:,0]),4000)
sample2 = np.random.multivariate_normal(mean[:,1],np.diag(var[:,1]),4000)
samples = torch.tensor(np.concatenate((sample1,sample2),axis=0)).float()
sample1 = torch.tensor(sample1).float()
sample2 = torch.tensor(sample2).float()

#gen_mean = model.out_mu(samples)
gen_mean = torch.sigmoid(model.out_mu(samples)).detach().numpy()
gen_var = model.out_log_sigma(samples).detach().numpy()
res1 = []
res2 = []
for i in range(gen_mean.shape[0]):
	res1.append(np.random.normal(gen_mean[i,0],np.exp(gen_var[i,0])))
	res2.append(np.random.normal(gen_mean[i,1],np.exp(gen_var[i,1])))

res1 = np.array(res1)
res2 = np.array(res2)
res = np.concatenate((gen_mean[0:4000,0],gen_mean[4000:8000,1]))


ind = np.arange(4000)
plt.plot(ind,np.sort(res[4000:8000]))
plt.plot(ind,np.sort(res[0:4000])[::-1])
plt.show()
'''



#res = res.reshape(res.shape[1],res.shape[0])



#pcz = get_pcz(samples,model.u_p,model.lambda_p,model.theta_p)
#res = torch.sum(pcz*res,1).detach().numpy()


#res = np.zeros(samples.shape[0])
#res[0:4000] = np.min(gen_mean[0:4000,:],1)
#res[4000:len(res)] = np.max(gen_mean[4000:len(res),:],1)

'''
b = np.zeros(samples.shape[0])
b[0:4000] = 1
recon_x1 = torch.tanh(model._dec1(model.decoder1(sample1))).detach().numpy()
recon_x2 = torch.tanh(model._dec2(model.decoder2(sample2))).detach().numpy()
plt.scatter(recon_x1[:,0],recon_x1[:,1],c="yellow",alpha=0.1)
plt.scatter(recon_x2[:,0],recon_x2[:,1],c="purple",alpha=0.1)
plt.xlabel('Pacman X-axis')
plt.ylabel('Pacman Y-axis')
plt.show()



 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(recon_x1[:,0], recon_x1[:,1], res[0:4000],c="yellow"
  , marker='o')
ax.scatter(recon_x2[:,0], recon_x2[:,1], res[4000:8000],c="purple"
  , marker='o')
ax.set_xlabel('Pacman X-axis')
ax.set_ylabel('Pacman Y-axis')
ax.set_zlabel('Generated Non-linear Response')
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.show()
'''





# VaDE

'''
mean = model.u_p.detach().numpy()
var = model.lambda_p.detach().numpy()

sample1 = np.random.multivariate_normal(mean[:,0],np.diag(var[:,0]),4000)
sample2 = np.random.multivariate_normal(mean[:,1],np.diag(var[:,1]),4000)
samples = torch.tensor(np.concatenate((sample1,sample2),axis=0)).float()
recon_x = torch.tanh(model._dec(model.decoder(samples))).detach().numpy()
b = np.ones(samples.shape[0])
b[4000:8000] = 0
plt.scatter(recon_x[:,0],recon_x[:,1],c=b,alpha=0.05)
#plt.scatter(pacman_data[:,0],pacman_data[:,1])
plt.show()

'''


'''
# Debugging DGC
from dgc import dgc
from vade import vade
import numpy as np
import torch
import matplotlib.pyplot as plt
model = dgc(input_dim=2,  y_dim = 1, z_dim=10, n_centroids=2, binary=True,
        encodeLayer=[128,256,256], decodeLayer=[256,256,128])
#model = vade(input_dim=2,  z_dim=5, n_centroids=2, binary=True,
#        encodeLayer=[32,64], decodeLayer=[64,32])
model.load_model('dgc.pt')
mean = model.u_p.detach().numpy()
var = model.lambda_p.detach().numpy()


sample1 = np.random.multivariate_normal(mean[:,0],np.diag(var[:,0]),4000)
sample2 = np.random.multivariate_normal(mean[:,1],np.diag(var[:,1]),4000)
samples = torch.tensor(np.concatenate((sample1,sample2),axis=0)).float()
sample1 = torch.tensor(sample1).float()
sample2 = torch.tensor(sample2).float()

gen_mean = torch.sigmoid(model.out_mu(samples)).detach().numpy()
gen_var = model.out_log_sigma(samples).detach().numpy()
res1 = []
res2 = []
for i in range(gen_mean.shape[0]):
    res1.append(np.random.normal(gen_mean[i,0],np.exp(gen_var[i,0])))
    res2.append(np.random.normal(gen_mean[i,1],np.exp(gen_var[i,1])))

res1 = np.array(res1)
res2 = np.array(res2)
res = np.concatenate((gen_mean[0:4000,0],gen_mean[4000:8000,1]))

ind = np.arange(4000)
plt.plot(ind,np.sort(res[4000:8000]))
plt.plot(ind,np.sort(res[0:4000])[::-1])
plt.show()

# reconstruct pacman
recon_x = torch.tanh(model._dec(model.decoder(samples))).detach().numpy()
b = np.ones(samples.shape[0])
b[4000:8000] = 0
plt.scatter(recon_x[:,0],recon_x[:,1],c=b,alpha=0.05)
#plt.scatter(pacman_data[:,0],pacman_data[:,1])
plt.show()
'''











