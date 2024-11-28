from xgcm import Grid
import pop_tools
import gcsfs
import fsspec as fs
import numpy as np
import xesmf as xe
import xarray as xr
import random
import matplotlib.pyplot as plt
import warnings
from xgcm import Grid
import importlib
import preprocessing
import os
import xrft
import gcm_filters
import random

warnings.filterwarnings("ignore")

importlib.reload(preprocessing)
from preprocessing import preprocess_data

from gcm_filtering import filter_inputs_dataset
from gcm_filtering import filter_inputs

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

class SimpleCNN(nn.Module):
    def __init__(self, in_channels, image_height, image_width, output_channels):
        super(SimpleCNN, self).__init__()
        
        self.in_channels = in_channels
        self.image_height = image_height
        self.image_width = image_width
        self.output_channels = output_channels

        # Define layers with the parameters passed to the constructor
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(64 * (image_height // 8) * (image_width // 8), 128)  # Adjust for image size after pooling
        self.fc2 = nn.Linear(128, output_channels * image_height * image_width)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = nn.ReLU()(self.conv3(x))
        x = nn.MaxPool2d(kernel_size=2)(x)

        x = x.reshape(x.size(0), -1)  # Flatten the tensor
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        x = x.reshape(-1, self.output_channels, self.image_height, self.image_width)  # Reshape to output image

        return x

class TensorDataset(Dataset):
    def __init__(self, tensor_data, input_channels=2):
        """
        Initialize the dataset with tensor data and the number of input channels.

        Parameters:
        tensor_data (Tensor): The 4D tensor dataset (time_steps, x, y, vars)
        input_channels (int): The number of input channels to use for the input images
        """
        self.tensor_data = tensor_data  # The 4D tensor dataset
        self.time_steps = tensor_data.shape[0]  # Total number of time steps
        self.input_channels = input_channels  # Number of input channels

    def __len__(self):
        return self.time_steps  # Total number of samples (time steps)

    def __getitem__(self, idx):
        """
        Retrieve the input images and target for a specific time step.
        
        Parameters:
        idx (int): The index of the time step.

        Returns:
        tuple: A tuple containing input images and the target for the specified time step.
        """
        data = self.tensor_data[idx]  # Get data for the specified time step
        
        # Extract the input images using the first 'input_channels' variables
        input_images = data[:, :, :self.input_channels]  # Take the first 'input_channels' variables

        # The target is always the last channel (last variable in the last dimension)
        target = data[:, :, -1]  # The last variable is the target

        return input_images, target  # Return input (images) and target

