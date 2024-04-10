import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.datasets import MNIST, SVHN
from torchvision.transforms import ToTensor
from torch.utils.data.dataset import Subset


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
    """
    Generate pseudo-labels for a dataset using a trained model.

    Args:
        model: The trained model used for generating predictions.
        device: The device on which the model and data are located (CPU or GPU).
        data_loader: DataLoader for the dataset to generate pseudo-labels for.
        threshold: The confidence threshold for accepting a prediction as a pseudo-label.

    Returns:
        A list of pseudo-labels for the SVHN dataset.
    """
    # Set the model to evaluation mode
    model.eval()

    # Get Pseudo labels
    pseudo_labels = []
    for data, _ in data_loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
            probs = F.softmax(output, dim=1)
            max_probs, predictions = torch.max(probs, dim=1)

            # Filter predictions based on the confidence threshold
            valid_predictions = predictions[max_probs > threshold].cpu().numpy()
            pseudo_labels.extend(valid_predictions)

    return pseudo_labels


# -----------------------------------------------------------------------------
# CombinedDataset
# -----------------------------------------------------------------------------
class CombinedDataset(Dataset):
    """
    A custom dataset that combines two datasets: MNIST and SVHN with pseudo-labels.
    """

    def __init__(self, mnist_dataset, svhn_dataset, svhn_pseudo_labels):
        """
        Initializes the combined dataset.

        Args:
            mnist_dataset: The MNIST dataset.
            svhn_dataset: The SVHN dataset.
            svhn_pseudo_labels: A list of pseudo-labels for the SVHN dataset.
        """
        self.mnist_dataset = mnist_dataset
        self.svhn_dataset = svhn_dataset
        self.svhn_pseudo_labels = svhn_pseudo_labels

        # Sanity check
        assert len(svhn_dataset) == len(svhn_pseudo_labels), "SVHN dataset and pseudo-labels must have the same size."

    def __len__(self):
        return len(self.mnist_dataset) + len(self.svhn_dataset)

    def __getitem__(self, idx):
        if idx < len(self.mnist_dataset):
            data, label = self.mnist_dataset[idx]
        else:
            # Adjust index for SVHN dataset
            svhn_idx = idx - len(self.mnist_dataset)
            data, _ = self.svhn_dataset[svhn_idx]
            label = self.svhn_pseudo_labels[svhn_idx]

        return data, label


# -----------------------------------------------------------------------------
# combine_datasets
# -----------------------------------------------------------------------------
def combine_datasets(mnist_loader, svhn_loader, svhn_pseudo_labels):
    """
    Combines MNIST and pseudo-labeled SVHN datasets into a single DataLoader.

    Args:
        mnist_loader: DataLoader for the MNIST dataset.
        svhn_loader: DataLoader for the SVHN dataset.
        svhn_pseudo_labels: List of pseudo-labels for the SVHN dataset.

    Returns:
        A DataLoader that can iterate over the combined dataset.
    """
    mnist_dataset = mnist_loader.dataset
    svhn_dataset = svhn_loader.dataset

    combined_dataset = CombinedDataset(mnist_dataset, svhn_dataset, svhn_pseudo_labels)
    combined_loader = DataLoader(combined_dataset, batch_size=mnist_loader.batch_size, shuffle=True)

    return combined_loader
