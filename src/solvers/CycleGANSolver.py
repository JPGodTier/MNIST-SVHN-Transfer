import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from src.models.CycleGAN.UNet.UNet import UNet
from src.models.CycleGAN.PatchGAN.PatchGAN import PatchGAN
from src.Common.utils import unnormalize_mnist, unnormalize_svhn


class CycleGANSolver(nn.Module):
    # -----------------------------------------------------------------------------
    # __init__
    # -----------------------------------------------------------------------------
    def __init__(self, lr=0.0002, beta1=0.5, beta2=0.999, loss_cycle_weight=10):
        super(CycleGANSolver, self).__init__()
        # Init Generators and Discriminators
        self.G_SM = UNet()
        self.G_MS = UNet()
        self.D_S = PatchGAN()
        self.D_M = PatchGAN()

        # Select device
        if torch.cuda.is_available():
           self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
           self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # On device
        self.G_SM.to(self.device)
        self.G_MS.to(self.device)
        self.D_S.to(self.device)
        self.D_M.to(self.device)

        # Init Hyperparameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.loss_cycle_weight = loss_cycle_weight

        # Init optimizers for Generators and Discriminators
        self.optimizer_G = optim.Adam(
            list(self.G_SM.parameters()) + list(self.G_MS.parameters()),
            lr=lr, betas=(beta1, beta2))
        self.optimizer_D = optim.Adam(
            list(self.D_S.parameters()) + list(self.D_M.parameters()),
            lr=lr, betas=(beta1, beta2))
        
        # Init loss criteria
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()


    # -----------------------------------------------------------------------------
    # gan_discriminators_loss
    # -----------------------------------------------------------------------------
    def gan_discriminators_loss(self, real_S, real_M):
        # Loss for D_M
        pred_real_M = self.D_M(real_M)
        loss_D_M_real = self.criterion_GAN(pred_real_M, torch.ones_like(pred_real_M))
        pred_fake_M = self.D_M(self.G_SM(real_S))
        loss_D_M_fake = self.criterion_GAN(pred_fake_M, torch.zeros_like(pred_fake_M))

        # Loss for D_S
        pred_real_S = self.D_S(real_S)
        loss_D_S_real = self.criterion_GAN(pred_real_S, torch.ones_like(pred_real_S))
        pred_fake_S = self.D_S(self.G_MS(real_M))
        loss_D_S_fake = self.criterion_GAN(pred_fake_S, torch.zeros_like(pred_fake_S))


        return loss_D_M_real + loss_D_M_fake + loss_D_S_real + loss_D_S_fake
    

    # -----------------------------------------------------------------------------
    # gan_generators_loss
    # -----------------------------------------------------------------------------
    def gan_generators_loss(self, real_S, real_M):
        # Loss for G_SM
        pred_fake_M = self.D_M(self.G_SM(real_S))
        loss_G_SM = self.criterion_GAN(pred_fake_M, torch.ones_like(pred_fake_M))

        # Loss for G_MS
        pred_fake_S = self.D_S(self.G_MS(real_M))
        loss_G_MS = self.criterion_GAN(pred_fake_S, torch.ones_like(pred_fake_S))

        return loss_G_SM + loss_G_MS


    # -----------------------------------------------------------------------------
    # cycle_generators_loss
    # -----------------------------------------------------------------------------
    def cycle_generators_loss(self, real_S, real_M):
        reconstructed_S = self.G_MS(self.G_SM(real_S))
        loss_cycle_S = self.criterion_cycle(reconstructed_S, real_S)

        reconstructed_M = self.G_SM(self.G_MS(real_M))
        loss_cycle_M = self.criterion_cycle(reconstructed_M, real_M)

        return loss_cycle_S + loss_cycle_M
    

    # -----------------------------------------------------------------------------
    # train_generators
    # -----------------------------------------------------------------------------
    def train_generators(self, real_S, real_M):
        self.optimizer_G.zero_grad()

        # GAN loss
        loss_GAN = self.gan_generators_loss(real_S, real_M)

        # Cycle loss
        loss_cycle = self.cycle_generators_loss(real_S, real_M)

        # Total generator loss
        loss_G = loss_GAN + loss_cycle * self.loss_cycle_weight
        loss_G.backward()
        self.optimizer_G.step()

        return loss_G


    # -----------------------------------------------------------------------------
    # train_discriminators
    # -----------------------------------------------------------------------------
    def train_discriminators(self, real_S, real_M):
        self.optimizer_D.zero_grad()

        loss_D = self.gan_discriminators_loss(real_S, real_M)
        loss_D.backward()
        self.optimizer_D.step()

        return loss_D
    

    # -----------------------------------------------------------------------------
    # train_cycle_GAN
    # -----------------------------------------------------------------------------
    def train_cycle_GAN(self, dataloader_S_train, dataloader_M_train, dataloader_S_test, num_epochs, num_gen_training, path):
        for epoch in range(num_epochs):
            print(f"Epoch [{epoch+1}/{num_epochs}] starts")

            # Start Training mode
            self.G_MS.train()
            self.G_SM.train()
            self.D_S.train()
            self.D_M.train()

            running_loss_G = 0.0
            running_loss_D = 0.0

            loss_G_cumulated = 0.0
            loss_D_cumulated = 0.0

            iteration = 0
            for real_S, real_M in zip(dataloader_S_train, dataloader_M_train):
                # First element of real_S and real_M are the images of the batch
                real_S = real_S[0].to(self.device)
                real_M = real_M[0].to(self.device)

                # Discriminators' forward and backward pass
                loss_D = self.train_discriminators(real_S, real_M)
                loss_D_cumulated += loss_D
                running_loss_D += loss_D

                # Generators' forward and backward pass
                for i in range(num_gen_training):
                    loss_G = self.train_generators(real_S, real_M)
                loss_G_cumulated += loss_G
                running_loss_G += loss_G

                # Print losses for each 50 iteration
                iteration += 1
                if iteration % 50 == 0:
                    print(f"Iteration {iteration}: Loss_G: {loss_G_cumulated/50:.4f}, Loss_D: {loss_D_cumulated/50:.4f}")
                    loss_G_cumulated = 0
                    loss_D_cumulated = 0

            # Average losses over the dataset
            avg_loss_G = running_loss_G / len(dataloader_S_train)
            avg_loss_D = running_loss_D / len(dataloader_M_train)

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss_G: {avg_loss_G:.4f}, Loss_D: {avg_loss_D:.4f}")

            ### Print transformed pictures of epoch i
            # Start Evaluation mode
            self.G_MS.eval()
            self.G_SM.eval()
            self.D_S.eval()
            self.D_M.eval()

            # Load SVHN and MNIST datasets
            images, _ = next(iter(dataloader_S_test))
            images = images.to(self.device)[40:50]

            # Transform SVHN into MNIST-like picture
            with torch.no_grad():  # No need to track gradients
                transformed = self.G_SM(images)  # Transform SVHN to MNIST style

            # Display images
            fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(15, 4))  # Set up the subplot grid
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
            plt.savefig(os.path.join(path, f'transformed_images_figure_epoch{epoch+1}.png'))
