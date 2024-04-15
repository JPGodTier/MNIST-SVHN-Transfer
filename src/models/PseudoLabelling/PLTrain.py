import torch
from torch import nn, optim


# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
def train(model, device, train_loader, optimizer, criterion, epoch, log_interval=100):
    model.train()  # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # Move data and targets to the configured device

        optimizer.zero_grad()  # Clear the gradients of all optimized variables
        output = model(data)  # Forward pass: compute the output of the model
        loss = criterion(output, target)  # Calculate the loss
        loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters
        optimizer.step()  # Perform a single optimization step (parameter update)

        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


# -----------------------------------------------------------------------------
# validate
# -----------------------------------------------------------------------------
def validate(model, device, validation_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    validation_loss = 0
    correct = 0
    with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            validation_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    validation_loss /= len(validation_loader.dataset)
    print(f'\nValidation set: Average loss: {validation_loss:.4f}, '
          f'Accuracy: {correct}/{len(validation_loader.dataset)} '
          f'({100. * correct / len(validation_loader.dataset):.0f}%)\n')

    return validation_loss
