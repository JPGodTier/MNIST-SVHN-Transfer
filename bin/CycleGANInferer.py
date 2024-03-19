import torch
import numpy as np
import matplotlib.pyplot as plt
from src.Models.UNet.UNet import UNet
from src.utils import mnist_loader, svhn_loader, unnormalize_mnist, unnormalize_svhn


# Define parameters
batch_size = 100 # Batchs size of Dataloader for gradient descent


if __name__ == "__main__":
    # Load SVHN and MNIST datasets
    _, dataloader_S_test = svhn_loader(batch_size)

    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # This check is specific to macOS with Apple Silicon
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")


    # Load G_SM model
    G_SM = UNet().to(device)  # Initialize the model structure (make sure it matches the saved model)
    G_SM.load_state_dict(torch.load('./savedmodels/G_SM_model.pth'))
    G_SM.eval()  # Set to evaluation mode
    
    # Process the first batch and get the first 10 SVHN images
    images, _ = next(iter(dataloader_S_test))
    images = images.to(device)[:10]

    # Transform SVHN into MNIST-like picture
    with torch.no_grad():  # No need to track gradients
        transformed = G_SM(images)  # Transform SVHN to MNIST style

    # Display images
    fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(10, 4))  # Set up the subplot grid
    for i in range(10):
        # Display original SVHN image
        img = images[i].cpu()
        svhn_img = unnormalize_svhn(img)  # Unnormalize SVHN image
        axes[0, i].imshow(svhn_img)
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')

        # Display transformed MNIST-like image
        transformed_img = transformed[i].cpu()
        mnist_img = unnormalize_mnist(transformed_img)  # Unnormalize transformed image
        axes[1, i].imshow(mnist_img, cmap='gray')
        axes[1, i].set_title("Transformed")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()




