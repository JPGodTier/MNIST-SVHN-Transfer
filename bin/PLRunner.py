from torch.optim.lr_scheduler import StepLR
from src.models.PseudoLabelling.PseudoLabelling import CNNModel, generate_pseudo_labels, combine_datasets
from src.solvers.PLSolver import train, validate
from src.Common.DataAugment import *


def main():
    # Device config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyper-parameters
    learning_rate = 0.001
    step_size = 1
    gamma = 0.3
    mnist_training_epoch = 5
    mnist_svhn_training_epoch = 5
    pseudo_labelling_iter = 5
    pseudo_label_threshold = 0.90

    # Load data
    mnist_train_loader, mnist_test_loader = mnist_loader(batch_size=64)
    svhn_train_loader, svhn_test_loader = svhn_loader(batch_size=64)

    # Model Init
    model = CNNModel().to(device)

    # Loss & optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # learning rate scheduler
    scheduler = StepLR(optimizer,
                       step_size=step_size,
                       gamma=gamma)

    # MNIST Training
    print("Starting initial training on MNIST...")
    for epoch in range(mnist_training_epoch):
        train(model, device, mnist_train_loader, optimizer, criterion, epoch)
        validate(model, device, mnist_test_loader, criterion)

        # Update the learning rate
        scheduler.step()

    for iteration in range(pseudo_labelling_iter):
        # Generate pseudo-labels for SVHN
        print(f"Generating pseudo-labels for SVHN, iteration {iteration+1}...")
        svhn_indices, svhn_pseudo_labels = generate_pseudo_labels(model,
                                                                  device,
                                                                  svhn_train_loader,
                                                                  threshold=pseudo_label_threshold)
        if not svhn_pseudo_labels:
            print(f"Could not predict pseudo-labels for SVHN, threshold used {pseudo_label_threshold}")
            return
        else:
            print(f"Successfully generated {len(svhn_pseudo_labels)} pseudo labels")

        # Combine MNIST & pseudo-labeled SVHN
        combined_train_loader = combine_datasets(mnist_train_loader, svhn_train_loader, svhn_indices, svhn_pseudo_labels)

        # Train combination
        if combined_train_loader is not None:
            print("Starting training with combined MNIST and pseudo-labeled SVHN...")
            for epoch in range(mnist_svhn_training_epoch):
                train(model, device, combined_train_loader, optimizer, criterion, epoch)
                validate(model, device, mnist_test_loader, criterion)
                validate(model, device, svhn_test_loader, criterion)

                # Update the learning rate
                scheduler.step()

    # Save the model
    save_model(model, "final_model.pth")


def save_model(model, save_path="model.pth"):
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == '__main__':
    main()
