import time
from src.solvers.DiscrepancySolver import DiscrepancySolver
from src.Common.utils import svhn_loader, mnist_loader

# Hyperparameters
batch_size = 128
lr = 0.0002
n_step_C = 4
num_epochs = 50
weight_decay = 0.0005
data_augment_source = False


if __name__ == "__main__":
    start_time = time.time()

    # Load SVHN and MNIST datasets
    dataloader_S_train, dataloader_S_test = mnist_loader(batch_size)
    dataloader_T_train, dataloader_T_test = svhn_loader(batch_size)

    DClassifier = DiscrepancySolver(lr=lr, n_step_C=n_step_C, weight_decay=weight_decay, data_augment_source=data_augment_source)
    for epoch in range(num_epochs):
        DClassifier.train(epoch, dataloader_S_train, dataloader_T_train)
        DClassifier.test(epoch, dataloader_T_test)

    end_time = time.time()
    duration = end_time - start_time
    print(f"The process took {duration/60} minutes to complete.")