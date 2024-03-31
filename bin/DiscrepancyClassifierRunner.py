import time
from src.solvers.DiscrepancySolver import DiscrepancySolver
from src.utils import svhn_loader, mnist_loader

# Hyperparameters
batch_size = 128
lr = 0.0002
n_step_C = 4
num_epochs = 50
weight_decay = 0.0005


if __name__ == "__main__":
    start_time = time.time()

    # Load SVHN and MNIST datasets
    dataloader_S_train, dataloader_S_test = svhn_loader(batch_size)
    dataloader_M_train, dataloader_M_test = mnist_loader(batch_size)

    DClassifier = DiscrepancySolver(lr=lr, n_step_C=n_step_C, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        DClassifier.train(epoch, dataloader_S_train, dataloader_M_train)
        DClassifier.test(epoch, dataloader_M_test)