import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from src.ResNet18.ResNet18 import ResNet18
from src.utils import mnist_loader, svhn_loader
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # Import and initialization
    # -----------------------------------------------------------------------------
    # Import data loaders
    batch_size = 100 # Batchs size for gradient descent
    #train_loader, test_loader = mnist_loader(batch_size)
    #in_channels = 1
    train_loader, test_loader = svhn_loader(batch_size)
    in_channels = 3

    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    # Initialize the model, the criterion and the optimizer
    model = ResNet18(in_channels=in_channels)
    model = nn.DataParallel(model)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    # -----------------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------------
    # Set the model to training mode
    model.train()

    # Number of epochs
    num_epochs = 5

    # Loop over the dataset multiple times
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            # Move inputs and labels to the device
            inputs, labels = inputs.to(device), labels.to(device) 

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100}')
                running_loss = 0.0

    print('Finished Training')


    # -----------------------------------------------------------------------------
    # Evaluate
    # -----------------------------------------------------------------------------
    # Set the model to evaluation mode
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

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')