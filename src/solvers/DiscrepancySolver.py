import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.models.DiscrepancyClassifier.FeatureGenerator.FeatureGenerator import FeatureGenerator
from src.models.DiscrepancyClassifier.ClassPredictor.ClassPredictor import ClassPredictor
from src.Common.DataAugment import augment_images


class DiscrepancySolver(nn.Module):
    def __init__(self, lr=0.0002, n_step_C=4, n_classes=10, weight_decay=0.0005, data_augment_source=True):
        super(DiscrepancySolver, self).__init__()
        # Init Generator and both Predictors
        self.G = FeatureGenerator()
        self.F1 = ClassPredictor()
        self.F2 = ClassPredictor()

        # Select device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # On device
        self.G.to(self.device)
        self.F1.to(self.device)
        self.F2.to(self.device)

        # Init Hyperparameters
        self.lr = lr
        self.n_step_C = n_step_C
        self.n_classes = n_classes
        self.weight_decay = weight_decay
        self.data_augment_source = data_augment_source

        # Init optimizers for Generators and Discriminators
        self.optimizer_G = optim.Adam(self.G.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.optimizer_F1 = optim.Adam(self.F1.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.optimizer_F2 = optim.Adam(self.F2.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Init classification loss criteria
        self.criterion_classif = nn.CrossEntropyLoss()


    # -----------------------------------------------------------------------------
    # discrepancy
    # -----------------------------------------------------------------------------
    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1, dim=1) - F.softmax(out2, dim=1)))
    

    # -----------------------------------------------------------------------------
    # train_step_A
    # -----------------------------------------------------------------------------
    def train_step_A(self, x_S_train):
        """Train G, F1 and F2 (minimize both classification losses on S_train)"""
        self.optimizer_G.zero_grad()
        self.optimizer_F1.zero_grad()
        self.optimizer_F2.zero_grad()

        img_S, label_S = x_S_train
        img_S = img_S.to(self.device)
        if self.data_augment_source:
            img_S = augment_images(img_S)
        label_S = label_S.long().to(self.device)

        # Calculating classification losses
        feat_S = self.G(img_S)
        output_S1 = self.F1(feat_S)
        loss_step_A1 = self.criterion_classif(output_S1, label_S)
        output_S2 = self.F2(feat_S)
        loss_step_A2 = self.criterion_classif(output_S2, label_S)
        loss_step_A = loss_step_A1 + loss_step_A2

        # Backpropagation and gradient descent
        loss_step_A.backward()
        self.optimizer_G.step()
        self.optimizer_F1.step()
        self.optimizer_F2.step()
    

    # -----------------------------------------------------------------------------
    # train_step_B
    # -----------------------------------------------------------------------------
    def train_step_B(self, x_S_train, x_T_train):
        """Fix G, Train F1 and F2 (minimize both classification losses on S_train and maximize discrepancy on T_train)"""
        self.optimizer_F1.zero_grad()
        self.optimizer_F2.zero_grad()

        img_S, label_S = x_S_train
        img_S = img_S.to(self.device)
        if self.data_augment_source:
            img_S = augment_images(img_S)
        label_S = label_S.long().to(self.device)
        img_T, _ = x_T_train
        img_T = img_T.to(self.device)

        # Calculating classification losses
        feat_S = self.G(img_S)
        output_S1 = self.F1(feat_S)
        loss_step_B1 = self.criterion_classif(output_S1, label_S)
        output_S2 = self.F2(feat_S)
        loss_step_B2 = self.criterion_classif(output_S2, label_S)

        # Calculating discrepancy loss
        feat_T = self.G(img_T)
        output_T1 = self.F1(feat_T)
        output_T2 = self.F2(feat_T)
        loss_step_B3 = self.discrepancy(output_T1, output_T2)

        loss_step_B = loss_step_B1 + loss_step_B2 - loss_step_B3

        # Backpropagation and gradient descent
        loss_step_B.backward()
        self.optimizer_F1.step()
        self.optimizer_F2.step()

        # Returning the classification losses for printing in train function
        return loss_step_B1, loss_step_B2
    

    # -----------------------------------------------------------------------------
    # train_step_C
    # -----------------------------------------------------------------------------
    def train_step_C(self, x_T_train):
        """Fix F1 and F2, (Multiple) Train G (minimize discrepancy on T_train)"""
        self.optimizer_G.zero_grad()

        img_T, _ = x_T_train
        img_T = img_T.to(self.device)

        # Calculating discrepancy loss
        feat_T = self.G(img_T)
        output_T1 = self.F1(feat_T)
        output_T2 = self.F2(feat_T)
        loss_step_C = self.discrepancy(output_T1, output_T2)

        # Backpropagation and gradient descent
        loss_step_C.backward()
        self.optimizer_G.step()

        # Returning the discrepancy loss for printing in train function
        return loss_step_C


    # -----------------------------------------------------------------------------
    # train
    # -----------------------------------------------------------------------------
    def train(self, epoch, dataloader_S_train, dataloader_T_train):
        # Start Training mode
        self.G.train()
        self.F1.train()
        self.F2.train()

        # Initialization of cumulated losses over epoch
        loss_clf1_epoch = 0.0           # Classification loss of Generator + Predictor 1
        loss_clf2_epoch = 0.0           # Classification loss of Generator + Predictor 2
        loss_discrep_epoch = 0.0        # Discrepancy loss between predictions out of Predictors 1 and 2

        # Initialization of cumulated losses over iterations - reset after 50 iterations
        loss_clf1_iteration = 0.0
        loss_clf2_iteration = 0.0
        loss_discrep_iteration = 0.0
        
        # Iterate over mini-batches
        for iteration, (x_S_train, x_T_train) in enumerate(zip(dataloader_S_train, dataloader_T_train)):
            # Train step A - Train G, F1 and F2 (minimize both classification losses on S_train)
            self.train_step_A(x_S_train)

            # Train step B - Fix G, Train F1 and F2 (minimize both classiication losses on S_train and maximize discrepancy on T_train)
            loss_clf1, loss_clf2 = self.train_step_B(x_S_train, x_T_train)

            # Train step C - Fix F1 and F2, (Multiple) Train G (minimize discrepancy on T_train)
            for i in range(self.n_step_C):
                loss_discrep = self.train_step_C(x_T_train)
            
            loss_clf1_iteration += loss_clf1
            loss_clf2_iteration += loss_clf2
            loss_discrep_iteration += loss_discrep

            # Every 50 iterations, print both averages of classification losses and average of discrepancy losses
            if iteration % 50 == 0:
                print(f"Train - Iteration {iteration}: \tLoss_clf1: {loss_clf1_iteration/50:.4f}, Loss_clf2: {loss_clf2_iteration/50:.4f}, Loss_discrepancy: {loss_discrep_iteration/50:.4f}")
                
                # Reset iteration losses
                loss_clf1_iteration = 0.0
                loss_clf2_iteration = 0.0
                loss_discrep_iteration = 0.0
            
            loss_clf1_epoch += loss_clf1
            loss_clf2_epoch += loss_clf2
            loss_discrep_epoch += loss_discrep

        # Every epoch, print both averages of classification losses and average of discrepancy losses
        size = min(len(dataloader_S_train), len(dataloader_T_train))
        avg_loss_clf1 = loss_clf1_epoch / size
        avg_loss_clf2 = loss_clf2_epoch / size
        avg_loss_discrep = loss_discrep_epoch / size

        print(f"Train - Epoch [{epoch+1}]: \t\tLoss_clf1: {avg_loss_clf1:.4f}, Loss_clf2: {avg_loss_clf2:.4f}, Loss_discrepancy: {avg_loss_discrep:.4f}")
    

    # -----------------------------------------------------------------------------
    # test
    # -----------------------------------------------------------------------------
    def test(self, epoch, dataloader_T_test):
        # Start inference mode
        self.G.eval()
        self.F1.eval()
        self.F2.eval()

        correct_predictions_clf1 = 0
        correct_predictions_clf2 = 0
        correct_predictions_ensemble = 0
        size = 0

        # Iterate over mini-batches
        for x_T_test in dataloader_T_test:
            img_T, label_T = x_T_test
            img_T, label_T = img_T.to(self.device), label_T.long().to(self.device)

            # Calculate logit outputs of Predictors 1 and 2 and ensemble (sum)
            feat_T = self.G(img_T)
            output_T1 = self.F1(feat_T)
            output_T2 = self.F2(feat_T)
            output_ensemble = output_T1 + output_T2

            # Calculate predictions
            pred1 = output_T1.data.max(1)[1]
            pred2 = output_T2.data.max(1)[1]
            pred_ensemble = output_ensemble.data.max(1)[1]

            correct_predictions_clf1 += pred1.eq(label_T.data).cpu().sum()
            correct_predictions_clf2 += pred2.eq(label_T.data).cpu().sum()
            correct_predictions_ensemble += pred_ensemble.eq(label_T.data).cpu().sum()
            size += label_T.data.size()[0]

        print(f"Test - Epoch [{epoch+1}]: Accuracy_clf1: {correct_predictions_clf1/size*100:.2f}%, Accuracy_clf2: {correct_predictions_clf2/size*100:.2f}%, Accuracy_ensemble: {correct_predictions_ensemble/size*100:.2f}%")
        print(f"-----------------------------------------------------------------------------------------------\n")