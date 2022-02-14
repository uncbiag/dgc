import os
import torch
import math
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import pickle
import torch.utils.data
import torch.cuda


class stl_loader(Dataset):
	def __init__(self,images,cluster_index):
		self.images = images
		self.cluster_index = cluster_index


	def __len__(self):
		return self.images.shape[0]


	def __getitem__(self,index):
		image_b = torch.tensor(self.images[index]).float()
		c_index = self.cluster_index[index]
		return image_b, c_index



