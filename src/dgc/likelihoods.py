import torch
import numpy as np
import math


def get_pcz(z, u_p, theta_p, lambda_p):
  logpcz = torch.log(theta_p) - 0.5 * torch.sum(
      torch.log(2 * np.pi * lambda_p) +
      (z - u_p) ** 2 / lambda_p, dim=1)

  return torch.softmax(logpcz, dim=1)  # NxK



def get_logqcx(lambda_k):
  return torch.sum(torch.log(lambda_k + 1e-10) * lambda_k, dim=1)



def get_qentropy(z_log_var):
  return -0.5*torch.sum(1+z_log_var+math.log(2*math.pi), dim=1)



def get_logpzc(z_mean, z_log_var, lambda_k, u_p, lambda_p):
  return torch.sum(
      0.5 * lambda_k * torch.sum(
          torch.log(2*np.pi*lambda_p) +
          torch.exp(z_log_var) / lambda_p +
                   (z_mean - u_p) ** 2 / lambda_p,
          dim=1),
      dim=1)



def get_logpc(lambda_k, theta_p):
  return -torch.sum(torch.log(theta_p) * lambda_k, dim=1)



def get_BCE(x,recon_x):
  return -torch.mean(x*torch.log(torch.clamp(recon_x, min=1e-10))+
    (1-x)*torch.log(torch.clamp(1-recon_x, min=1e-10)), 1)



def get_ensemble(lambda_k,log_y_likelihood):
  return -torch.sum(lambda_k*log_y_likelihood,1)



log2pi = math.log(2*math.pi)
def log_likelihood_samples_unit_gaussian(samples):
  return -0.5*log2pi*samples.size()[1] - torch.sum(0.5*(samples)**2, 1)



def log_likelihood_samplesImean_sigma(samples, mu, logvar):
  return -0.5*log2pi*samples.size()[1] - torch.sum(0.5*(samples-mu)**2/torch.exp(logvar) + 0.5*logvar, 1)


def contrastive_side_info_loss(side_info,mask,z_loss=False,both=False):
  # Batch size: n, feature dimension: d
  mask = mask.unsqueeze(-1) # nxnx1
  masked_side_info = mask*side_info.unsqueeze(0) # nxnxd, nx1xd
  contrastive_mat = torch.einsum('ijk,ihk->ijh',side_info.unsqueeze(1),masked_side_info)
  contrastive_mat = contrastive_mat.squeeze(1)
  return -torch.sum(torch.log(contrastive_mat+1))
  # contrastive_mat: batch_size x batch_size x batch_size
  #contrastive_mat = torch.matmul(masked_side_info,torch.transpose(masked_side_info,2,1))






