# Deep Goal-Oriented Clustering

This is the depository for the paper, Deep Goal-Oriented Clustering (DGC). This depository contains code to replicate the CIFAR 100-20 experiment detailed in the paper. Here we give a brief description of DGC.

DGC is built upon VAE, and uses similar variational techniques to maximize a variation lower bound of the data log-likelihood. A (deep) variational method can be efficiently summarized in terms of its generative & infernece steps, which we describe next. 

## Generative process for DGC

