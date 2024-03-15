import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from src.ResNet18.ResNet18 import ResNet18
from src.utils import *
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    dataset = 'svhn'  # 'mnist' or 'svhn'


    # -----------------------------------------------------------------------------
    # Import and initialization
    # -----------------------------------------------------------------------------
    # Import data loaders
    batch_size = 100 # Batchs size for gradient descent
    if dataset == 'mnist':
        train_loader, test_loader = mnist_loader(batch_size)
    elif dataset == 'svhn':
        train_loader, test_loader = svhn_loader(batch_size)


    # -----------------------------------------------------------------------------
    # Display pictures
    # -----------------------------------------------------------------------------
    # Get the first batch of images and labels from the train loader
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # Show images and print labels
    images_5 = torchvision.utils.make_grid(images[:5])
    if dataset == 'mnist':
        npimg = unnormalize_mnist(images_5)
    elif dataset == 'svhn':
        npimg = unnormalize_svhn(images_5)

    plt.imshow(npimg)
    plt.show()
    print(' '.join('%5s' % labels[j] for j in range(5)))