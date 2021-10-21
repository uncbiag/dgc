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


## Running the model
To run the model, simply do
```python
python run_model.py
```
