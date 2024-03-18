import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


# -----------------------------------------------------------------------------
# mnist_loader
# -----------------------------------------------------------------------------
def mnist_loader(batch_size):
    # Resize the data and transform the data to torch.FloatTensor and normalize it
    transform = transforms.Compose([
        transforms.Pad(padding=2), # Resize the image to 32x32
        transforms.Grayscale(num_output_channels=3), # Convert grayscale to RGB by replicating channels
        transforms.ToTensor(), # Transform to tensor type
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)) # Mean and STD for MNIST
    ])

    # Load the training and test datasets
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Create data loaders
    mnist_train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return mnist_train_loader, mnist_test_loader


# -----------------------------------------------------------------------------
# svhn_loader
# -----------------------------------------------------------------------------
def svhn_loader(batch_size):
    # Resize the data and transform the data to torch.FloatTensor and normalize it
    transform = transforms.Compose([
        #transforms.Resize((224, 224)), # Resize the image to 224x224 for ResNet18
        transforms.ToTensor(), # Transorm to tensor type
        transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)) # Mean and STD for SVHN
    ])

    # Load the training and test datasets
    train_dataset = torchvision.datasets.SVHN(root='./data/SVHN/raw', split='train', download=True, transform=transform)
    test_dataset = torchvision.datasets.SVHN(root='./data/SVHN/raw', split='test', download=True, transform=transform)

    # Create data loaders
    svhn_train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    svhn_test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return svhn_train_loader, svhn_test_loader

# -----------------------------------------------------------------------------
# unnormalize_svhn
# -----------------------------------------------------------------------------
def unnormalize_svhn(img):
        mean = np.array([0.4376821, 0.4437697, 0.47280442])
        std = np.array([0.19803012, 0.20101562, 0.19703614])
        img = img.numpy().transpose((1, 2, 0))  # Convert to numpy image shape (H x W x C)
        img = std * img + mean
        img = np.clip(img, 0, 1)  # Clip to ensure image range stays between 0 and 1
        return img


# -----------------------------------------------------------------------------
# unnormalize_mnist
# -----------------------------------------------------------------------------
def unnormalize_mnist(img):
        mean = np.array([0.1307, 0.1307, 0.1307])
        std = np.array([0.3081, 0.3081, 0.3081])
        img = img.numpy().transpose((1, 2, 0))  # Convert to numpy image shape (H x W x C)
        img = std * img + mean
        img = np.clip(img, 0, 1)  # Clip to ensure image range stays between 0 and 1
        return img