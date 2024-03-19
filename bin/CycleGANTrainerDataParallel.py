import torch
import time
from torch.nn import DataParallel
from src.Models.CycleGAN.CycleGAN import CycleGAN
from src.utils import mnist_loader, svhn_loader


# Define parameters
batch_size = 50 # Batchs size of Dataloader for gradient descent
num_epochs = 20

if __name__ == "__main__":
    start_time = time.time()

    # Load SVHN and MNIST datasets
    dataloader_S_train, dataloader_S_test = svhn_loader(batch_size)
    dataloader_M_train, dataloader_M_test = mnist_loader(batch_size)

   # Initialize CycleGAN model
    cycleGAN = CycleGAN(loss_identity_weight=1, loss_cycle_weight=1)
    print(f"Using device {cycleGAN.device}")
    
    # Move model to the default device
    cycleGAN = cycleGAN.to(cycleGAN.device)

    # Check if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"{torch.cuda.device_count()} GPUs available. Using DataParallel.")
        cycleGAN = DataParallel(cycleGAN)
    
    # Adjusted training call
    print('Training starts')
    if isinstance(cycleGAN, DataParallel):
        cycleGAN.module.train_cycle_GAN(dataloader_S_train, dataloader_M_train, num_epochs)
    else:
        cycleGAN.train_cycle_GAN(dataloader_S_train, dataloader_M_train, num_epochs)

    # Save trained model (note: adjust saving for DataParallel if needed)
    if isinstance(cycleGAN, DataParallel):
        torch.save(cycleGAN.module.G_SM.state_dict(), './savedmodels/G_SM_model.pth')
        torch.save(cycleGAN.module.G_MS.state_dict(), './savedmodels/G_MS_model.pth')
    else:
        torch.save(cycleGAN.G_SM.state_dict(), './savedmodels/G_SM_model.pth')
        torch.save(cycleGAN.G_MS.state_dict(), './savedmodels/G_MS_model.pth')

    end_time = time.time()
    duration = end_time - start_time
    print(f"The process took {duration/60} minutes to complete.")
