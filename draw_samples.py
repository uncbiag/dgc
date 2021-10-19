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
from vae import vae
from dgc_entropy import dgc_entropy
from vade import vade

def vae_draw_sample(model,k,plot=False):
  import matplotlib.pyplot as plt
  mu = np.zeros(model.z_dim)
  cov = np.eye(model.z_dim)
  sample = torch.tensor(np.random.multivariate_normal(mu,cov,k)).float()
  h = model._dec(sample)
  h = h.view(-1, 128,4,4)
  recon = model.decoder(h)
  recon = torch.sigmoid(recon.view(-1, model.nChannels, model.height, model.width))
  recon = recon.detach().numpy()
  recon =  np.transpose(recon,(0,2,3,1))
  if plot==True:
    plt.imshow(recon[1])
    plt.show()
  return recon


train_transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),])
test_transform = transforms.Compose(
    [transforms.ToTensor(),])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)
for i,(j,k) in enumerate(testloader):
  if i==0:
    inputs=j
    break


def recon_sample(inputs,model,k,plot=False):
  import matplotlib.pyplot as plt
  h = model.encoder(inputs)
  h = h.view(h.size(0), h.size(1)*h.size(2)*h.size(3))
  mu = model._enc_mu(h)
  logvar = model._enc_log_sigma(h)
  sample = util.reparameterize(mu, logvar,False)
  h = model._dec(sample)
  h = h.view(-1, 128,4,4)
  recon = model.decoder(h)
  recon = torch.sigmoid(recon.view(-1, model.nChannels, model.height, model.width))
  recon = recon.detach().numpy()
  recon =  np.transpose(recon,(0,2,3,1))
  if plot==True:
    plt.imshow(recon[k])
    plt.show()
  return recon


def mix_gaus_draw_sample(model,cen_ind, k,plot=False):
  import matplotlib.pyplot as plt
  mean = model.u_p.detach().numpy()
  var = model.lambda_p.detach().numpy()
  sample = torch.tensor(np.random.multivariate_normal(mean[:,cen_ind],np.diag(var[:,cen_ind]),k)).float()
  h = model._dec(sample)
  h = h.view(-1, 256,4,4)
  recon = model.decoder(h)
  recon = torch.sigmoid(recon.view(-1, model.nChannels, model.height, model.width))
  recon = recon.detach().numpy()
  recon =  np.transpose(recon,(0,2,3,1))
  if plot==True:
    plt.imshow(recon[1])
    plt.show()
  return recon



'''
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




model = dgc_w_o_balancing(input_dim=784, z_dim=10, y_dim = 1, n_centroids=4, binary=True,
        encodeLayer=[500,500,2000], decodeLayer=[2000,500,500])

model.load_model("w_balancing2.pt")




#------------------------------------------------------------------------------------------------------------
# Draw Samples and plot 
#------------------------------------------------------------------------------------------------------------
mean = model.u_p.detach().numpy()
var = model.lambda_p.detach().numpy()


sample1 = np.random.multivariate_normal(mean[:,1],np.diag(var[:,1]),1)
sample2 = np.random.multivariate_normal(mean[:,2],np.diag(var[:,2]),1)
sample3 = np.random.multivariate_normal(mean[:,0],np.diag(var[:,0]),1)
sample4 = np.random.multivariate_normal(mean[:,3],np.diag(var[:,3]),1)
samples = torch.tensor(np.concatenate((sample1,sample2,sample3,sample4),axis=0)).float()



sample1 = torch.tensor( np.random.multivariate_normal(mean[:,0],np.diag(var[:,0]),10)).float()
sample2 = torch.tensor( np.random.multivariate_normal(mean[:,1],np.diag(var[:,1]),10)).float()
sample3 = torch.tensor( np.random.multivariate_normal(mean[:,2],np.diag(var[:,2]),10)).float()
sample4 = torch.tensor( np.random.multivariate_normal(mean[:,3],np.diag(var[:,3]),10)).float()


recon_x1 = torch.sigmoid(model._dec1(model.decoder1(sample1))).detach().numpy().reshape(10,28,28)
recon_x2 = torch.sigmoid(model._dec2(model.decoder2(sample2))).detach().numpy().reshape(10,28,28)
recon_x3 = torch.sigmoid(model._dec3(model.decoder3(sample3))).detach().numpy().reshape(10,28,28)
recon_x4 = torch.sigmoid(model._dec4(model.decoder4(sample4))).detach().numpy().reshape(10,28,28)


recon_x = torch.sigmoid(model._dec(model.decoder(samples))).detach().numpy().reshape(4,28,28)




num_row = 2
num_col = 2
num=4
# plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num):
    ax = axes[i//num_col, i%num_col]
    ax.axis('off')
    ax.imshow(recon_x[i], cmap='gray')
plt.tight_layout()
plt.show()



#------------------------------------------------------------------------------------------------------------
# Plot accuracies
#------------------------------------------------------------------------------------------------------------
fig, ax1 = plt.subplots()

#a = np.load("wo_balancing_acc_linear_exp.npy")
a = np.load("w_accuracy_new.npy")
b = np.load("acc_dgc_w_o.npy")
#e = np.load("wo_balancing_acc_linear_exp.npy")
c = np.load("acc_vae_mix.npy")
d = np.load("gammax_new.npy")
ind = np.arange(len(a))
color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
#ax1.plot(ind, a, color="g",label="DGC w Balancing Coeff")
ax1.plot(ind,a,color="r", label = "DGC w Balancing Coeff")
ax1.plot(ind,b,color="m", label = "DGC w/o Balancing Coeff")
ax1.plot(ind,c,color="k", label = "VAE with GMM")
ax1.tick_params(axis='y')
ax1.legend()

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Balancing Coeff', color=color)  # we already handled the x-label with ax1
ax2.plot(ind,d, color="b",label = "Balancing Coeff")
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0,1)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()





index = np.arange(100)
plt.plot(index,a[9:109],c='red',label="DGC")
plt.plot(index,b[9:109],c='blue',label="VaDE")
plt.xlabel("Epoch")
plt.ylabel("Clustering Accuracy")
plt.legend()
plt.show()  

'''


























