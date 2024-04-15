import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from src.Common.utils import unnormalize_svhn, unnormalize_mnist
from src.Common.DataAugment import augment_images
from src.models.CyCADA.Generator import Generator
from src.models.CyCADA.ImageDiscriminator import ImageDiscriminator
from src.models.CyCADA.LabelDiscriminator import LabelDiscriminator
from src.models.CyCADA.TaskClassifier import TaskClassifier


class CyCADASolver(nn.Module):
    def __init__(self, lr_stepA=0.0001, lr_stepB= 0.00001, lr_stepC=0.0002, path='savedmodels/CyCADA', data_augment_source=True):
        super(CyCADASolver, self).__init__()
        # Init Generator and both Predictors
        self.F_S = TaskClassifier() # To be pre-trained
        self.G_ST = Generator()
        self.G_TS = Generator()
        self.D_S = ImageDiscriminator()
        self.D_T = ImageDiscriminator()
        self.F_T = TaskClassifier()
        self.D_feat = LabelDiscriminator()

        # Select device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # On device
        self.F_S.to(self.device)
        self.G_ST.to(self.device)
        self.G_TS.to(self.device)
        self.D_S.to(self.device)
        self.D_T.to(self.device)
        self.F_T.to(self.device)
        self.D_feat.to(self.device)

        # Init Hyperparameters
        self.lr_stepA = lr_stepA
        self.lr_stepB = lr_stepB
        self.lr_stepC = lr_stepC
        self.path = path
        self.data_augment_source = data_augment_source

        # Init optimizers
        self.optimizer_F_S = optim.Adam(self.F_S.parameters(), lr=self.lr_stepA)
        self.optimizer_G = optim.Adam(list(self.G_ST.parameters()) + list(self.G_TS.parameters()), lr=self.lr_stepB)
        self.optimizer_D = optim.Adam(list(self.D_S.parameters()) + list(self.D_T.parameters()), lr=self.lr_stepB)
        self.optimizer_F_T = optim.Adam(self.F_T.parameters(), lr=self.lr_stepB)
        self.optimizer_D_feat = optim.Adam(self.D_feat.parameters(), lr=self.lr_stepB)

        # Init classification loss criteria
        self.criterion_GAN = nn.BCELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_task = nn.CrossEntropyLoss()

        # Init D_feat_accuracy
        self.D_feat_acc = 0.0
    

    # -----------------------------------------------------------------------------
    # train_stepA
    # -----------------------------------------------------------------------------
    def train_stepA(self, epoch, dataloader_S_train):
        """Train F_S"""
        # Start Training mode
        self.F_S.train()

        # Initialization of cumulated losses over epoch
        loss_stepA_epoch = 0.0
        loss_stepA_iteration = 0.0

        # Iterate over mini-batches
        for iteration, x_S_train in enumerate(dataloader_S_train):
            self.optimizer_F_S.zero_grad()

            img_S, label_S = x_S_train
            img_S, label_S = img_S.to(self.device), label_S.long().to(self.device)
            if self.data_augment_source:
                img_S = augment_images(img_S)

            # Calculating classification losses
            output_S = self.F_S(img_S)
            loss_stepA = self.criterion_task(output_S, label_S)

            # Backpropagation and gradient descent
            loss_stepA.backward()
            self.optimizer_F_S.step()

            loss_stepA_iteration += loss_stepA

            # Every 50 iterations, print average of classification losses
            if iteration % 50 == 0:
                print(f"Train Step A - Iteration {iteration}: \tLoss: {loss_stepA_iteration/50:.4f}")
                
                # Reset iteration losses
                loss_stepA_iteration = 0.0
            
            loss_stepA_epoch += loss_stepA

        # Every epoch, print average of classification losses
        avg_loss_stepA = loss_stepA_epoch / len(dataloader_S_train)

        print(f"Train Step A - Epoch [{epoch+1}]: \tLoss: {avg_loss_stepA:.4f}")


    # -----------------------------------------------------------------------------
    # test_stepA
    # -----------------------------------------------------------------------------
    def test_stepA(self, epoch, dataloader_S_test):
        # Start inference mode
        self.F_S.eval()

        correct_predictions_stepA = 0
        size = 0

        # Iterate over mini-batches
        for x_S_test in dataloader_S_test:
            img_S, label_S = x_S_test
            img_S, label_S = img_S.to(self.device), label_S.long().to(self.device)

            # Calculate predictions
            output_S = self.F_S(img_S)
            pred_S = output_S.data.max(1)[1]

            correct_predictions_stepA += pred_S.eq(label_S.data).cpu().sum()
            size += label_S.data.size()[0]

        print(f"Test Step A - Epoch [{epoch+1}]: \tAccuracy: {correct_predictions_stepA/size*100:.2f}%")
        print(f"-----------------------------------------------------------------------------------------------\n")
    

    # -----------------------------------------------------------------------------
    # train_discriminators - Step B.1
    # -----------------------------------------------------------------------------
    def train_discriminators(self, img_S_real, img_T_real):
        # Start Training mode
        self.D_S.train()
        self.D_T.train()
        self.optimizer_D.zero_grad()

        # Forward
        img_T_fake = self.G_ST(img_S_real)
        img_S_fake = self.G_TS(img_T_real)

        # GAN Losses for D_T and D_S
        pred_D_T_real = self.D_T(img_T_real)
        loss_D_T_real = self.criterion_GAN(pred_D_T_real, torch.ones_like(pred_D_T_real))
        pred_D_T_fake = self.D_T(img_T_fake)
        loss_D_T_fake = self.criterion_GAN(pred_D_T_fake, torch.zeros_like(pred_D_T_fake))
        loss_D_T = loss_D_T_real + loss_D_T_fake

        pred_D_S_real = self.D_S(img_S_real)
        loss_D_S_real = self.criterion_GAN(pred_D_S_real, torch.ones_like(pred_D_S_real))
        pred_D_S_fake = self.D_T(img_S_fake)
        loss_D_S_fake = self.criterion_GAN(pred_D_S_fake, torch.zeros_like(pred_D_S_fake))
        loss_D_S = loss_D_S_real + loss_D_S_fake

        loss_stepB_D_GAN = (loss_D_T + loss_D_S) * 0.5

        # Backward
        loss_stepB_D_GAN.backward()

        # Optimization
        self.optimizer_D.step()

        return loss_stepB_D_GAN


    # -----------------------------------------------------------------------------
    # train_generators - Step B.2
    # -----------------------------------------------------------------------------
    def train_generators(self, img_S_real, img_T_real):
        # Start Training mode
        self.G_ST.train()
        self.G_TS.train()
        self.optimizer_G.zero_grad()

        # Forward
        img_T_fake = self.G_ST(img_S_real)
        img_S_reco = self.G_TS(img_T_fake) # Reconstructed source image
        pred_S_real = self.F_S(img_S_real)
        label_S_real = pred_S_real.data.max(1)[1]
        pred_T_fake = self.F_S(img_T_fake)

        img_S_fake = self.G_TS(img_T_real)
        img_T_reco = self.G_ST(img_S_fake) # Reconstructed target image
        pred_T_real = self.F_S(img_T_real)
        label_T_real = pred_T_real.data.max(1)[1]
        pred_S_fake = self.F_S(img_S_fake)

        # GAN Losses for G_ST and G_TS
        pred_D_T_fake = self.D_T(img_T_fake)
        loss_step_G_GAN_1 = self.criterion_GAN(pred_D_T_fake, torch.ones_like(pred_D_T_fake))

        pred_D_S_fake = self.D_S(img_S_fake)
        loss_step_G_GAN_2 = self.criterion_GAN(pred_D_S_fake, torch.ones_like(pred_D_S_fake))

        loss_stepB_G_GAN = loss_step_G_GAN_1 + loss_step_G_GAN_2

        # Cycle Losses for G_ST and G_TS
        loss_stepB_G_cycle = self.criterion_cycle(img_S_reco, img_S_real) + self.criterion_cycle(img_T_reco, img_T_real)

        # Semantic Losses for G_ST and G_TS
        loss_stepB_G_semantic = self.criterion_task(pred_T_fake, label_S_real) + self.criterion_task(pred_S_fake, label_T_real)

        # Backward
        loss_stepB_G = loss_stepB_G_GAN + loss_stepB_G_cycle + loss_stepB_G_semantic
        loss_stepB_G.backward()

        # Optimization
        self.optimizer_G.step()

        return loss_stepB_G_GAN, loss_stepB_G_cycle, loss_stepB_G_semantic

    # -----------------------------------------------------------------------------
    # train_D_feat - Step B.3
    # -----------------------------------------------------------------------------
    def train_D_feat(self, img_S_real, img_T_real):
        # Start Training mode
        self.D_feat.train()
        self.optimizer_D_feat.zero_grad()

        # Forward
        img_T_fake = self.G_ST(img_S_real)

        # GAN Losses for D_feat
        pred_D_feat_real = self.D_feat(self.F_T(img_T_real))
        loss_D_feat_real = self.criterion_GAN(pred_D_feat_real, torch.ones_like(pred_D_feat_real))
        pred_D_feat_fake = self.D_feat(self.F_T(img_T_fake))
        loss_D_feat_fake = self.criterion_GAN(pred_D_feat_fake, torch.zeros_like(pred_D_feat_fake))
        loss_stepB_D_feat_GAN = (loss_D_feat_real + loss_D_feat_fake) * 0.5

        # Calculate accuracy
        pred_real_correct = (pred_D_feat_real > 0.5).float()
        pred_fake_correct = (pred_D_feat_fake < 0.5).float()
        total_correct = pred_real_correct.sum().item() + pred_fake_correct.sum().item()
        total_predictions = pred_D_feat_real.size(0) + pred_D_feat_fake.size(0)
        self.D_feat_accuracy = total_correct / total_predictions

        # Backward
        loss_stepB_D_feat_GAN.backward()

        # Optimization
        self.optimizer_D_feat.step()

        return loss_stepB_D_feat_GAN
    

    # -----------------------------------------------------------------------------
    # train_F_T - Step B.4
    # -----------------------------------------------------------------------------
    def train_F_T(self, img_S_real, label_S):
        # Start Training mode
        self.F_T.train()
        self.optimizer_F_T.zero_grad()

        # Forward
        img_T_fake = self.G_ST(img_S_real)
        pred_label_T = self.F_T(img_T_fake)

        # Task Loss
        loss_stepB_F_T_task = self.criterion_task(pred_label_T, label_S)

        # GAN Loss for F_T (only if D_feat_acc > 0.6)
        pred_D_feat_fake = self.D_feat(self.F_T(img_T_fake))
        if self.D_feat_acc > 0.6:
            loss_stepB_F_T_GAN = self.criterion_GAN(pred_D_feat_fake, torch.ones_like(pred_D_feat_fake))
        else:
            loss_stepB_F_T_GAN = 0

        # Backward
        loss_stepB_F_T = loss_stepB_F_T_task + loss_stepB_F_T_GAN
        loss_stepB_F_T.backward()

        # Optimization
        self.optimizer_F_T.step()

        return loss_stepB_F_T_task, loss_stepB_F_T_GAN


    # -----------------------------------------------------------------------------
    # train_stepB
    # -----------------------------------------------------------------------------
    def train_stepB(self, epoch, dataloader_S_train, dataloader_T_train):
        """Pixel space adaptation - Train G_ST, G_TS, D_S, D_T, F_T"""
        self.F_S.eval() # F_S is already pre-trained and is fixed

        # Initialization of cumulated losses over epoch
        loss_stepB_D_GAN_epoch = 0.0
        loss_stepB_G_GAN_epoch = 0.0
        loss_stepB_G_cycle_epoch = 0.0
        loss_stepB_G_semantic_epoch = 0.0
        loss_stepB_D_feat_GAN_epoch = 0.0
        loss_stepB_F_T_GAN_epoch = 0.0
        loss_stepB_F_T_task_epoch = 0.0
        

        # Iterate over mini-batches
        for x_S_train, x_T_train in zip(dataloader_S_train, dataloader_T_train):
            
            img_S, label_S = x_S_train
            img_S, label_S = img_S.to(self.device), label_S.long().to(self.device)
            if self.data_augment_source:
                img_S = augment_images(img_S)
            img_T, _ = x_T_train
            img_T = img_T.to(self.device)

            # Step B.1: Training the Discriminators
            loss_stepB_D_GAN = self.train_discriminators(img_S, img_T)
            loss_stepB_D_GAN_epoch += loss_stepB_D_GAN

            # Step B.2: Training the Generators
            for j in range(4):
                loss_stepB_G_GAN, loss_stepB_G_cycle, loss_stepB_G_semantic = self.train_generators(img_S, img_T)
            loss_stepB_G_GAN_epoch += loss_stepB_G_GAN
            loss_stepB_G_cycle_epoch += loss_stepB_G_cycle
            loss_stepB_G_semantic_epoch += loss_stepB_G_semantic

            # Step B.3: Training D_feat
            loss_stepB_D_feat_GAN = self.train_D_feat(img_S, img_T)
            loss_stepB_D_feat_GAN_epoch += loss_stepB_D_feat_GAN

            # Step B.4: Training F_T
            loss_stepB_F_T_task, loss_stepB_F_T_GAN = self.train_F_T(img_S, label_S)
            loss_stepB_F_T_task_epoch += loss_stepB_F_T_task
            loss_stepB_F_T_GAN_epoch += loss_stepB_F_T_GAN
        
        # Every epoch, print average of the losses
        size = min(len(dataloader_S_train), len(dataloader_T_train))
        loss_stepB_D_GAN_avg = loss_stepB_D_GAN_epoch / size
        loss_stepB_G_GAN_avg = loss_stepB_G_GAN_epoch / size
        loss_stepB_G_cycle_avg = loss_stepB_G_cycle_epoch / size
        loss_stepB_G_semantic_avg = loss_stepB_G_semantic_epoch / size
        loss_stepB_D_feat_GAN_avg = loss_stepB_D_feat_GAN_epoch / size
        loss_stepB_F_T_GAN_avg = loss_stepB_F_T_GAN_epoch / size
        loss_stepB_F_T_task_avg = loss_stepB_F_T_task_epoch / size

        print(f"Train Step B - Epoch [{epoch+1}]: \
              Discriminator GAN loss: {loss_stepB_D_GAN_avg:.4f}, \
              Generator GAN Loss: {loss_stepB_G_GAN_avg:.4f}, \
              Cycle loss: {loss_stepB_G_cycle_avg:.4f}, \
              Semantic Loss: {loss_stepB_G_semantic_avg:.4f}, \
              D_feat GAN Loss: {loss_stepB_D_feat_GAN_avg:.4f}, \
              F_T GAN Loss: {loss_stepB_F_T_GAN_avg:.4f}, \
              Task loss: {loss_stepB_F_T_task_avg:.4f}")
        
        self.display(epoch, img_S)
    
    
    # -----------------------------------------------------------------------------
    # test_stepB
    # -----------------------------------------------------------------------------
    @torch.no_grad()
    def test_stepB(self, epoch, dataloader_T_test):
        # Start inference mode
        self.F_T.eval()

        correct_predictions_stepB = 0
        size = 0

        # Iterate over mini-batches
        for x_T_test in dataloader_T_test:
            img_T, label_T = x_T_test
            img_T, label_T = img_T.to(self.device), label_T.long().to(self.device)

            # Calculate predictions
            output_T = self.F_T(img_T)
            pred_T = output_T.data.max(1)[1]

            correct_predictions_stepB += pred_T.eq(label_T.data).cpu().sum()
            size += label_T.data.size()[0]

        print(f"Test Step B - Epoch [{epoch+1}]: \tAccuracy: {correct_predictions_stepB/size*100:.2f}%")
        print(f"-----------------------------------------------------------------------------------------------\n")
    

    # -----------------------------------------------------------------------------
    # display
    # -----------------------------------------------------------------------------
    def display(self, epoch, img_S):
        """Print transformed pictures of current epoch"""
        # Start Evaluation mode
        self.G_ST.eval()

        # Select 10 pictures
        img_S = img_S[:10]

        # Transform source into target-like picture
        with torch.no_grad():  # No need to track gradients
            img_T_transf = self.G_ST(img_S)  # Transform source to target style

        # Display images
        fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(15, 4))  # Set up the subplot grid
        for i in range(10):
            # Display original image
            img = img_S[i].cpu()
            mnist_img = unnormalize_mnist(img)  # Unnormalize original image
            axes[0, i].imshow(mnist_img)
            axes[0, i].set_title("Original")
            axes[0, i].axis('off')

            # Display transformed target-like image
            transformed_img = img_T_transf[i].cpu()
            svhn_img = unnormalize_svhn(transformed_img)  # Unnormalize transformed image
            axes[1, i].imshow(svhn_img, cmap='gray')
            axes[1, i].set_title("Transformed")
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(self.path, f'transformed_images_figure_epoch{epoch+1}.png'))
