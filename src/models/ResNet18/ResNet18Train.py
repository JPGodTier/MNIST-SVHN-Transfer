import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from src.models.ResNet18.ResNet18 import ResNet18
from src.Common.utils import mnist_loader, svhn_loader
import matplotlib.pyplot as plt
import numpy as np


def ModelTrain(model, train_loader, num_epochs=5):
    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    # Initialize the model, the criterion and the optimizer
    model = nn.DataParallel(model)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Set the model to training mode
    model.train()

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
    
    return model