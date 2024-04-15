import torch.nn as nn
import torch.nn.functional as F
from src.Common.DataAugment import *
from torch.utils.data import Dataset


# -----------------------------------------------------------------------------
# CNNModel
# -----------------------------------------------------------------------------
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Convolutional layer block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional layer block 2
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)  # 10 classes for MNIST & SVHN

    def forward(self, x):
        # Convolutional layer block 1
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))

        # Convolutional layer block 2
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))

        # Flattening layer
        x = x.view(-1, 128 * 8 * 8)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -----------------------------------------------------------------------------
# generate_pseudo_labels
# -----------------------------------------------------------------------------
def generate_pseudo_labels(model, device, data_loader, threshold=0.9):
    model.eval()
    pseudo_labels = []
    indices = []
    for batch_idx, (data, _) in enumerate(data_loader):
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
            probs = F.softmax(output, dim=1)
            max_probs, predictions = torch.max(probs, dim=1)
            valid_mask = max_probs > threshold
            valid_indices = torch.where(valid_mask)[0].cpu().numpy()
            valid_predictions = predictions[valid_mask].cpu().numpy()

            pseudo_labels.extend(valid_predictions)
            indices.extend(valid_indices + batch_idx * data_loader.batch_size)

    return indices, pseudo_labels


# -----------------------------------------------------------------------------
# CombinedDataset
# -----------------------------------------------------------------------------
class CombinedDataset(Dataset):
    def __init__(self, mnist_dataset, svhn_dataset, svhn_indices, svhn_pseudo_labels):
        self.mnist_dataset = mnist_dataset
        self.svhn_dataset = svhn_dataset
        self.svhn_indices = svhn_indices
        self.svhn_pseudo_labels = svhn_pseudo_labels
        self.svhn_map = dict(zip(svhn_indices, svhn_pseudo_labels))

    def __len__(self):
        return len(self.mnist_dataset) + len(self.svhn_indices)

    def __getitem__(self, idx):
        if idx < len(self.mnist_dataset):
            data, label = self.mnist_dataset[idx]
        else:
            svhn_idx = idx - len(self.mnist_dataset)
            actual_index = self.svhn_indices[svhn_idx]
            data, _ = self.svhn_dataset[actual_index]
            label = self.svhn_map[actual_index]

        return data, label


# -----------------------------------------------------------------------------
# combine_datasets
# -----------------------------------------------------------------------------
def combine_datasets(mnist_dataloader, svhn_dataloader, svhn_indices, svhn_pseudo_labels):
    """
    Combines MNIST and pseudo-labeled SVHN datasets into a single DataLoader.

    Args:
        mnist_dataloader: DataLoader for the MNIST dataset.
        svhn_dataloader: DataLoader for the SVHN dataset.
        svhn_indices: List of indices for SVHN images that have valid pseudo-labels.
        svhn_pseudo_labels: List of pseudo-labels for the SVHN dataset.

    Returns:
        A DataLoader that can iterate over the combined dataset.
    """
    mnist_dataset = mnist_dataloader.dataset
    svhn_dataset = svhn_dataloader.dataset

    # Create a combined dataset using the MNIST dataset and the subset of the SVHN dataset with valid pseudo-labels
    combined_dataset = CombinedDataset(mnist_dataset, svhn_dataset, svhn_indices, svhn_pseudo_labels)
    combined_loader = DataLoader(combined_dataset, batch_size=mnist_dataloader.batch_size, shuffle=True)

    return combined_loader
