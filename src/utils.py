import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Function
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# mnist_loader
# -----------------------------------------------------------------------------
def mnist_loader(batch_size):
    # Resize the data and transform the data to torch.FloatTensor and normalize it
    transform = transforms.Compose([
        transforms.Resize(32), # Resize the image to 32x32
        transforms.Grayscale(3), # Convert grayscale to RGB by replicating channels
        transforms.ToTensor(), # Transform to tensor type
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Mean and STD for MNIST
    ])

    # Load the training and test datasets
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Create data loaders
    mnist_train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    return mnist_train_loader, mnist_test_loader


# -----------------------------------------------------------------------------
# svhn_loader
# -----------------------------------------------------------------------------
def svhn_loader(batch_size):
    # Resize the data and transform the data to torch.FloatTensor and normalize it
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(), # Transorm to tensor type
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the training and test datasets
    train_dataset = torchvision.datasets.SVHN(root='./data/SVHN/color/raw/train', split='train', download=True, transform=transform)
    test_dataset = torchvision.datasets.SVHN(root='./data/SVHN/color/raw/test', split='test', download=True, transform=transform)

    # Create data loaders
    svhn_train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    svhn_test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    return svhn_train_loader, svhn_test_loader


# -----------------------------------------------------------------------------
# svhn_loader_greyscale
# -----------------------------------------------------------------------------
def svhn_loader_greyscale(batch_size):
    # Resize the data and transform the data to torch.FloatTensor and normalize it
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.Grayscale(3), # Convert grayscale to RGB by replicating channels
        transforms.ToTensor(), # Transorm to tensor type
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the training and test datasets
    train_dataset = torchvision.datasets.SVHN(root='./data/SVHN/greyscale/raw/train', split='train', download=True, transform=transform)
    test_dataset = torchvision.datasets.SVHN(root='./data/SVHN/greyscale/raw/test', split='test', download=True, transform=transform)

    # Create data loaders
    svhn_train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    svhn_test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    return svhn_train_loader, svhn_test_loader

# -----------------------------------------------------------------------------
# unnormalize_svhn
# -----------------------------------------------------------------------------
def unnormalize_svhn(img):
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        img = img.numpy().transpose((1, 2, 0))  # Convert to numpy image shape (H x W x C)
        img = std * img + mean
        img = np.clip(img, 0, 1)  # Clip to ensure image range stays between 0 and 1
        return img


# -----------------------------------------------------------------------------
# unnormalize_mnist
# -----------------------------------------------------------------------------
def unnormalize_mnist(img):
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        img = img.numpy().transpose((1, 2, 0))  # Convert to numpy image shape (H x W x C)
        img = std * img + mean
        img = np.clip(img, 0, 1)  # Clip to ensure image range stays between 0 and 1
        return img


# -----------------------------------------------------------------------------
# GradReverse
# -----------------------------------------------------------------------------
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * -ctx.lambd, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


# -----------------------------------------------------------------------------
# display_images
# -----------------------------------------------------------------------------
def display_images(loader, len=1):
    
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)

    # Get some random training images
    dataiter = iter(loader)
    images, _ = next(dataiter)

    # Show images
    images = torchvision.utils.make_grid(images[1])
    npimages = images.numpy()
    print(npimages)
    plt.imshow(np.transpose(npimages, (1, 2, 0)))
    plt.show()