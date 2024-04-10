import torch
from src.models.PseudoLabelling.PseudoLabelling import CNNModel, generate_pseudo_labels, combine_datasets
from src.models.PseudoLabelling.DataPreprocessing import load_mnist, load_svhn
from src.models.PseudoLabelling.PLTrain import train, validate


def main():
    # Device config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    mnist_train_loader, mnist_test_loader = load_mnist()
    svhn_train_loader, svhn_test_loader = load_svhn()

    # Model Init
    model = CNNModel().to(device)

    # Loss & optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # MNIST Training
    print("Starting initial training on MNIST...")
    max_epoch = 11
    for epoch in range(max_epoch):
        train(model, device, mnist_train_loader, optimizer, criterion, epoch)
        validate(model, device, mnist_test_loader, criterion)

    # Generate pseudo-labels for SVHN
    print("Generating pseudo-labels for SVHN...")
    threshold = 0.8
    svhn_pseudo_labels = generate_pseudo_labels(model, device, svhn_train_loader, threshold=threshold)
    if not svhn_pseudo_labels:
        print(f"Could not predict pseudo-labels for SVHN, threshold used {threshold}")
        return

    # Combine MNIST & pseudo-labeled SVHN
    combined_train_loader = combine_datasets(mnist_train_loader, svhn_train_loader, svhn_pseudo_labels)

    # Train combination
    if combined_train_loader is not None:
        print("Starting training with combined MNIST and pseudo-labeled SVHN...")
        for epoch in range(1, 11):
            train(model, device, combined_train_loader, optimizer, criterion, epoch)
            validate(model, device, mnist_test_loader, criterion)
            validate(model, device, svhn_test_loader, criterion)

    # Save the model
    save_model(model, "final_model.pth")


def save_model(model, save_path="model.pth"):
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == '__main__':
    main()
