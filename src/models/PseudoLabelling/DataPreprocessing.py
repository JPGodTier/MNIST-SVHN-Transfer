import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# -----------------------------------------------------------------------------
# load_mnist
# -----------------------------------------------------------------------------
def load_mnist(batch_size=64, download=True):
    """
    Load and preprocess the MNIST dataset.
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3), # Convert to 3-channel grayscale to match SVHN's 3 channels
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1] range
    ])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=download, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=download, transform=transform)

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
