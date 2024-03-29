# Deep Goal-Oriented Clustering

This is the depository for the paper, Deep Goal-Oriented Clustering (DGC). This page documents the python library ```dgc```, including instructions to re-create one of the experiments in the paper and instructions on how to apply DGC to a customized dataset. Here we give a brief description of DGC.

DGC is built upon VAE, and uses similar variational techniques to maximize the variation lower bound of the data log-likelihood. A (deep) variational method can be efficiently summarized in terms of its generative & infernece processes, which we describe next. 

## Generative process for DGC
Let x,y,z and c denote the input data, the side-information, the latent code, and the index for a given Gaussian mixture component, we then have

p(x,y,z,c) = p(y|z,c)p(x|z)p(z|c)p(c)

In words, we first sample a component index from p(c), sample the latent code z from p(z|c), and then we reconstruct the input x and predict for the side-information y (see the figure below for a figurative illustration). 

<p align="center">
<img align="middle" src="./bayesian_net.png" alt="DGC Generative Model" width="300" height="300" />
</p>


## Inference for DGC
For the variational lower bound of DGC, please refer to ```Eq. 2``` in the main paper. In a nutshell, we want to maximize the log-likelihood by maximizing its variational lower bound. 


## Install DGC
To install DGC, simply run
```shell
pip install dgc
```

## Test the model on Pacman
Simply follow the following to train a DGC model on the Pacman dataset. Please refer to Sec.5.2 in the main paper for more details. 
```python
# Load functions from dgc
from dgc import load_sample_datasets
from dgc import dgc

dataset = 'pacman'
task_name = 'regression'
batch_size = 128
learning_rate = 0.01
epochs = 50
path = './pacman'

trainloader, testloader, _ = load_sample_datasets(batch_size, path, dataset)
model = dgc(input_dim=2,  y_dim = 1, z_dim=10, n_centroids=2, task = task_name, binary=True,
            encodeLayer=[128,256,256], decodeLayer=[256,256,128])
model.fit(trainloader, testloader, lr=learning_rate, num_epochs=epochs,
        anneal=True, direct_predict_prob=False)
```
After training the model, one might want to visualize samples from the learned model, for both gaining insight and as a sanity check that our model is learning the correct data strcuture. If so, one can use the built-in function to sample from the model and plot the samples. As an example, assume we just trained a DGC model on the pacman (by running the code above), and now we want to visualize the samples from the learned model
```python
# Load the sampling function
from dgc import sample_model
import matplotlib.pyplot as plt

task_name = 'regression'
num_simu_points = 1000 # number of simulated/sampled points PER CLUSTER

# Use the trained model from the Pacman example
latent_sample, input_sample, side_info_sample = sample_model(model,num_simu_points,task_name)
input_sample1 = input_sample[:num_simu_points,:]
input_sample2 = input_sample[num_simu_points:,:]
 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(input_sample1[:,0], input_sample1[:,1], np.squeeze(side_info_sample[0],-1) ,c="yellow"
  , marker='o')
ax.scatter(input_sample2[:,0], input_sample2[:,1], np.squeeze(side_info_sample[1],-1) ,c="purple"
  , marker='o')
ax.set_xlabel('Pacman X-axis')
ax.set_ylabel('Pacman Y-axis')
ax.set_zlabel('Generated Non-linear Response')
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.show()
```
This will replicate Fig.3b in the main paper.


## Test the model on your dataset
To run DGC on your own dataset, you will need to have the following files (all of which are assumed to be numpy arrays)
1. **train_features.npy**: this contains the training features
2. **train_side_info.npy**: this contains the training side-information (can be either discrete or continous)
3. **train_cluster_labels.npy**: this contains the training clustering labels
4. **test_features.npy**: this contains the test features
5. **test_side_info.npy**: this contains the test side-information (can be either discrete or continous)
6. **test_cluster_labels.npy**: this contains the test clustering labels

As noted in the manuscript, when having access to the side-information at test time, ```test_side_info.npy``` is simply the ground truth side-information. However, when not having access to the side-information at test time, ```test_side_info.npy``` should contain something like
```python
# test_features are the input features (i.e. x) at test time
test_side_info = side_info_prediction_net(test_features)
```

You can create your own dataloader or passing the files into the built-in loader function. We use the built-in loader function for convenience here
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
