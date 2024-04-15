import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.Common.DataAugment import *


# -----------------------------------------------------------------------------
# load_mnist
# -----------------------------------------------------------------------------
def load_mnist(batch_size=64, download=True):
    """
    Load and preprocess the MNIST dataset.
    """
    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3), # Convert to 3-channel grayscale to match SVHN's 3 channels
        transforms.ToTensor(),
        transforms.Lambda(
            lambda x: augment_image(x, affine_params={'degrees': 30, 'translate': (0.1, 0.1), 'scale': (0.8, 1.2)},
                                    colorize_params={'color_variability': 0.2})),
        transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1] range
    ])

    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3), # Convert to 3-channel grayscale to match SVHN's 3 channels
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1] range
    ])
    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=download, transform=transform_train)
    test_dataset = datasets.MNIST(root='./data', train=False, download=download, transform=transform_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# -----------------------------------------------------------------------------
# load_mnist
# -----------------------------------------------------------------------------
def load_svhn(batch_size=64, download=True):
    """
    Load and preprocess the SVHN dataset.
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1] range
    ])

    # Load the SVHN dataset
    train_dataset = datasets.SVHN(root='./data', split='train', download=download, transform=transform)
    test_dataset = datasets.SVHN(root='./data', split='test', download=download, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


import matplotlib.pyplot as plt
import numpy as np


def visualize_predictions(images, true_labels, predicted_labels, title=""):
    """
    Visualizes a batch of images and labels.

    Args:
    images (Tensor): The batch of images.
    true_labels (list): Actual labels for the images.
    predicted_labels (list): Predicted labels for the images.
    title (str): Title for the plot.
    """
    plt.figure(figsize=(10, 5))
    for i, (img, true, pred) in enumerate(zip(images, true_labels, predicted_labels)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img.permute(1, 2, 0))
        plt.title(f"True: {true}\nPred: {pred}")
        plt.axis('off')
    plt.suptitle(title)
    plt.show()


