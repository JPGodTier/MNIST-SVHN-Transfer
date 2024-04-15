import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np

def ModelTest(model, test_loader):
    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)    

    # Set the model to evaluation mode
    model = model.to(device)
    model.eval()

    # Disable gradient computation
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the {total} test images: {100 * correct / total}%')