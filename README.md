# Deep Goal-Oriented Clustering

This is the depository for the paper, Deep Goal-Oriented Clustering (DGC). This depository contains code to replicate the CIFAR 100-20 experiment detailed in the paper. Here we give a brief description of DGC.

DGC is built upon VAE, and uses similar variational techniques to maximize a variation lower bound of the data log-likelihood. A (deep) variational method can be efficiently summarized in terms of its generative & infernece steps, which we describe next. 

## Generative process for DGC
Let x,y,z and c denote the input data, the side-information, the latent code, and the index for a given Gaussian mixture component, we then have

p(x,y,z,c) = p(y|z,c)p(x|z)p(z|c)p(c)

In words, we first sample a component index from p(c), sample the latent code z from p(z|c), and then we reconstruct the input x and predict for the side-information y (see the figure below for a figurative illustration). 

![](https://github.com/uncbiag/dgc/blob/main/bayesian_net.png?raw=true|width=20)

## Inference for DGC
For the variational lower bound of DGC, please refer to Eq. 2 in the main paper. In a nutshell, we want to maximize the log-likelihood by maximizing its variational lower bound. 


## Test the model on Pacman
To run the model on the Pacman dataset, simply do
```python
# Test model on the sythetic dataset Pacman
from util import load_sample_datasets
from dgc import dgc

dataset = 'pacman'
side_task_name = 'regression'
batch_size = 128
learning_rate = 0.01
epochs = 50

trainloader, testloader, _ = load_sample_datasets(batch_size,dataset)
model = dgc(input_dim=2,  y_dim = 1, z_dim=10, n_centroids=2, task = task_name, binary=True,
            encodeLayer=[128,256,256], decodeLayer=[256,256,128])
model.fit(trainloader, testloader, lr=learning_rate, num_epochs=epochs
        anneal=True, direct_predict_prob=False)
```
