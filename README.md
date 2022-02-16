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
To run the model on the Pacman dataset, first install the package 
```shell
pip install dgc
```
After the installation, simply follow the following
```python
# Test model on the sythetic dataset Pacman
from dgc import load_sample_datasets
from dgc import dgc

dataset = 'pacman'
task_name = 'regression'
batch_size = 128
learning_rate = 0.01
epochs = 50

trainloader, testloader, _ = load_sample_datasets(batch_size,dataset)
model = dgc(input_dim=2,  y_dim = 1, z_dim=10, n_centroids=2, task = task_name, binary=True,
            encodeLayer=[128,256,256], decodeLayer=[256,256,128])
model.fit(trainloader, testloader, lr=learning_rate, num_epochs=epochs,
        anneal=True, direct_predict_prob=False)
```

## Test the model on your dataset
To run DGC on your own dataset, you will need to have the following files (all of which are assumed to be numpy arrays)
1. **train_features.npy**: this contains the training features
2. **train_side_info.npy**: this contains the training side-information (can be either discrete or continous)
3. **train_cluster_labels.npy**: this contains the training clustering labels
4. **test_features.npy**: this contains the test features
5. **test_side_info.npy**: this contains the test side-information (can be either discrete or continous)
6. **test_cluster_labels.npy**: this contains the test clustering labels

You can create your own dataloader or passing the files into the built-in loader function
```python
from dgc import form_dataloaders
from dgc import dgc

batch_size = 128  # whatever you want
task_name = 'regression' # or classification
batch_size = 128
learning_rate = 0.01
epochs = 50
input_dim = 2
y_dim = 1,
z_dim = 10,
n_centroids = 10
binary = True
encoder_size = [128,256,256]
decoder_size = [256,256,128]
learning_rate = 0.01
num_epochs = 10
anneal = True
direct_predict_prob = False
save_model_path = './sample_model.pt'


train_data = [train_features,train_side_info,train_cluster_labels] # order matters here
test_data = [test_features,test_side_info,test_cluster_labels]
trainloader,testloader,_ = form_dataloaders(batch_size,train_data,test_data)

model = dgc(input_dim=input_dim,  y_dim = y_dim, z_dim=z_dim, n_centroids=n_centroids, task = task_name, binary=binary,
            encodeLayer=encoder_size, decodeLayer=decoder_size)
model.fit(trainloader, testloader, lr=learning_rate, num_epochs=epochs
        anneal=anneal, direct_predict_prob=direct_predict_prob)
        
# If you want to save the model
model.save_model(save_model_path)
```
