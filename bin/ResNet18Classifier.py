import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from src.models.ResNet18.ResNet18 import ResNet18
from src.models.ResNet18.ResNet18Train import ModelTrain
from src.models.ResNet18.ResNet18Test import ModelTest
from src.Common.utils import mnist_loader, svhn_loader
import matplotlib.pyplot as plt
import numpy as np


# Define parameters
train_dataset = 'mnist'
test_dataset = 'mnist'
model = ResNet18()
batch_size = 100 # Batchs size of Dataloader for gradient descent
num_epochs = 1

if __name__ == "__main__":
    # Load Train and Test datasets
    if (train_dataset, test_dataset) == ('mnist', 'mnist'):
        train_loader, test_loader = mnist_loader(batch_size)
    elif (train_dataset, test_dataset) == ('svhn', 'svhn'):
        train_loader, test_loader = svhn_loader(batch_size)
    elif (train_dataset, test_dataset) == ('mnist', 'svhn'):
        train_loader, _ = mnist_loader(batch_size)
        _, test_loader = svhn_loader(batch_size)
    elif (train_dataset, test_dataset) == ('svhn', 'mnist'):
        train_loader, _ = svhn_loader(batch_size)
        _, test_loader = mnist_loader(batch_size)
   
    # Train
    model_trained = ModelTrain(model, train_loader, num_epochs=num_epochs)

    # Test
    ModelTest(model_trained, test_loader)