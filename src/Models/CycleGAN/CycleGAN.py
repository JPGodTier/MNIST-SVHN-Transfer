import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from src.Models.UNet import UNet
from src.Models.PatchGAN import PatchGAN


class CycleGAN(nn.Module):
    # -----------------------------------------------------------------------------
    # __init__
    # -----------------------------------------------------------------------------
    def __init__(self, lr=0.0002, beta1=0.5, beta2=0.999, loss_identity_weight=5, loss_cycle_weight=10):
        super(CycleGAN, self).__init__()
        # Init Generators and Discriminators
        self.G_SM = UNet()
        self.G_MS = UNet()
        self.D_S = PatchGAN()
        self.D_M = PatchGAN()

        # On device
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.G_SM.to(self.device)
        self.G_MS.to(self.device)
        self.D_S.to(self.device)
        self.D_M.to(self.device)

        # Init Hyperparameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.loss_identity_weight = loss_identity_weight
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
        self.criterion_identity = nn.L1Loss()


    # -----------------------------------------------------------------------------
    # identity_loss
    # -----------------------------------------------------------------------------
    def identity_loss(self, real_S, real_M):
        same_M = self.G_SM(real_M)
        loss_identity_M = self.criterion_identity(same_M, real_M)
        same_S = self.G_MS(real_S)
        loss_identity_S = self.criterion_identity(same_S, real_S)
        return (loss_identity_S + loss_identity_M) * self.loss_identity_weight


    # -----------------------------------------------------------------------------
    # gan_loss
    # -----------------------------------------------------------------------------
    def gan_loss(self, real_S, real_M):
        fake_M = self.G_SM(real_S)
        pred_fake_M = self.D_M(fake_M)
        loss_GAN_SM = self.criterion_GAN(pred_fake_M, torch.ones_like(pred_fake_M))

        fake_S = self.G_MS(real_M)
        pred_fake_S = self.D_S(fake_S)
        loss_GAN_MS = self.criterion_GAN(pred_fake_S, torch.ones_like(pred_fake_S))

        return loss_GAN_SM + loss_GAN_MS


    # -----------------------------------------------------------------------------
    # cycle_loss
    # -----------------------------------------------------------------------------
    def cycle_loss(self, real_S, real_M):
        fake_M = self.G_SM(real_S)
        reconstructed_S = self.G_MS(fake_M)
        loss_cycle_S = self.criterion_cycle(reconstructed_S, real_S)

        fake_S = self.G_MS(real_M)
        reconstructed_M = self.G_SM(fake_S)
        loss_cycle_M = self.criterion_cycle(reconstructed_M, real_M)

        return (loss_cycle_S + loss_cycle_M) * self.loss_cycle_weight


    # -----------------------------------------------------------------------------
    # discriminator_loss
    # -----------------------------------------------------------------------------
    def discriminator_loss(self, discriminator, real_data, fake_data):
        pred_real = discriminator(real_data)
        loss_D_real = self.criterion_GAN(pred_real, torch.ones_like(pred_real))
        pred_fake = discriminator(fake_data)
        loss_D_fake = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
        return (loss_D_real + loss_D_fake) / 2
    

    # -----------------------------------------------------------------------------
    # train_generators
    # -----------------------------------------------------------------------------
    def train_generators(self, real_S, real_M):
        self.optimizer_G.zero_grad()

        # Identity loss
        loss_identity = self.identity_loss(real_S, real_M)

        # GAN loss
        loss_GAN = self.gan_loss(real_S, real_M)

        # Cycle loss
        loss_cycle = self.cycle_loss(real_S, real_M)

        # Total generator loss
        loss_G = loss_identity + loss_GAN + loss_cycle
        loss_G.backward()
        self.optimizer_G.step()


    # -----------------------------------------------------------------------------
    # train_discriminators
    # -----------------------------------------------------------------------------
    def train_discriminators(self, real_S, real_M):
        self.optimizer_D.zero_grad()

        loss_D_S = self.discriminator_loss(self.D_S, real_S, self.G_MS(real_M).detach())
        loss_D_M = self.discriminator_loss(self.D_M, real_M, self.G_SM(real_S).detach())

        loss_D = (loss_D_S + loss_D_M) / 2
        loss_D.backward()
        self.optimizer_D.step()
    

    # -----------------------------------------------------------------------------
    # train
    # -----------------------------------------------------------------------------
    def train(self, dataloader_S, dataloader_M, num_epochs):
        for epoch in range(num_epochs):
            for real_S, real_M in zip(dataloader_S, dataloader_M):
                real_S = real_S.to(self.device)
                real_M = real_M.to(self.device)

                # Generators' forward and backward pass
                self.train_generators(real_S, real_M)

                # Discriminators' forward and backward pass
                self.train_discriminators(real_S, real_M)

            print(f"Epoch [{epoch}/{num_epochs}]")