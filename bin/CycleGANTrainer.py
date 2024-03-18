import torch
from src.Models.CycleGAN.CycleGAN import CycleGAN
from src.utils import mnist_loader, svhn_loader


# Define parameters
batch_size = 100 # Batchs size of Dataloader for gradient descent
num_epochs = 20

if __name__ == "__main__":
    # Load SVHN and MNIST datasets
    dataloader_S_train, dataloader_S_test = svhn_loader(batch_size)
    dataloader_M_train, dataloader_M_test = mnist_loader(batch_size)

   # Initialize CycleGAN model
    cycleGAN = CycleGAN()
    cycleGAN = cycleGAN.to(cycleGAN.device)
    
    # Train
    print('Training starts')
    cycleGAN_trained = cycleGAN.train(dataloader_S_train, dataloader_M_train, num_epochs)

    # Save trained model
    torch.save(cycleGAN.G_SM.state_dict(), './models/G_SM_model.pth')
    torch.save(cycleGAN.G_MS.state_dict(), './models/G_MS_model.pth')




