import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from datetime import datetime
import csv
import sys

# Add the parent directory to the system path to find the 'model' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import *

# Update the dataset to handle MNIST data
from torchvision import datasets, transforms

# --- Configuration ---
DATASET_NAME = 'CIFAR10'  # Options: 'MNIST', 'CIFAR10', 'ImageNet'
# ---------------------

if DATASET_NAME == 'MNIST':
    # Define transformations for MNIST data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize for 1 channel
    ])
    # Load MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    model_class = MnistCNN
    out_dim = 10
elif DATASET_NAME == 'CIFAR10':
    # Define transformations for CIFAR-10 data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for 3 channels
    ])
    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    model_class = CifarCNN
    out_dim = 10
elif DATASET_NAME == 'ImageNet':
    # NOTE: ImageNet is a very large dataset that must be downloaded manually.
    # You must provide the path to the dataset root folder.
    # The folder should contain 'train' and 'val' subdirectories.
    imagenet_data_path = './data/imagenet' # <--- CHANGE THIS PATH

    # Define transformations for ImageNet data
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Load ImageNet dataset
    train_dataset = datasets.ImageNet(root=imagenet_data_path, split='train', transform=transform)
    test_dataset = datasets.ImageNet(root=imagenet_data_path, split='val', transform=transform)
    model_class = ImageNetCNN
    out_dim = 1000
else:
    raise ValueError("Unsupported dataset specified. Choose 'MNIST', 'CIFAR10', or 'ImageNet'.")


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)


# Use CrossEntropyLoss for classification
criterion = nn.CrossEntropyLoss()


num_epochs = 100
train_losses = []
train_accs = []
test_accs = []

def compute_acc(loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    model.train()
    return correct / total

# Create a timestamped directory for results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = os.path.join('results', timestamp)
os.makedirs(results_dir, exist_ok=True)

lr = 0.0001
model_configs = [
    {"hidden_dim": [64, 128, 64], "lr": lr},  # Example configuration for MNISTModel
    {"hidden_dim": [64, 128, 256, 128, 64], "lr": lr},
    {"hidden_dim": [64, 128, 256, 512, 256, 128, 64], "lr": lr},
]

# Store results for comparison
model_results = []

for i, config in enumerate(model_configs):
    hidden_dim = config["hidden_dim"]
    lr = config["lr"]

    # Initialize the correct model based on the dataset
    model = model_class(hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model (reuse your training loop)
    train_losses = []
    train_accs = []
    test_accs = []
    grad_norms = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Calculate and store gradient norm
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            grad_norms.append(total_norm)

            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

            # Print batch-level information
            if batch_idx % 100 == 0:  # Print every 100 batches
                print(f"Model {i+1}, Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Batch Loss: {loss.item():.4f}")

        # Calculate epoch-level metrics
        epoch_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_loss)
        train_acc = compute_acc(train_loader, model)
        test_acc = compute_acc(test_loader, model)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        # Print epoch-level information
        print(f"Model {i+1}, Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    # Save results for this model
    model_results.append({
        "hidden_dim": hidden_dim,
        "lr": lr,
        "train_losses": train_losses,
        "train_accs": train_accs,
        "test_accs": test_accs,
        "grad_norms": grad_norms,
        "num_params": model.count_trainable_params(model)
    })

# Save the trained model and metrics for each configuration
for i, result in enumerate(model_results):
    # Create a subdirectory for each model configuration
    config_dir = os.path.join(results_dir, f"model_{i+1}")
    os.makedirs(config_dir, exist_ok=True)

    # Save the model
    model_path = os.path.join(config_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)

    # Save loss and accuracy data
    metrics_path = os.path.join(config_dir, 'metrics.npz')
    np.savez(metrics_path, train_losses=result["train_losses"], train_accs=result["train_accs"], test_accs=result["test_accs"])

    # Save metrics to CSV
    csv_path = os.path.join(config_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Iteration', 'Epoch', 'Training Loss', 'Training Accuracy', 'Test Accuracy', 'Gradient Norm'])
        
        num_iterations_per_epoch = len(train_loader)
        for it, grad_norm in enumerate(result["grad_norms"]):
            epoch = it // num_iterations_per_epoch
            writer.writerow([
                it + 1,
                epoch + 1,
                result["train_losses"][epoch],
                result["train_accs"][epoch],
                result["test_accs"][epoch],
                grad_norm
            ])

    # Save configuration details
    config_info_path = os.path.join(config_dir, 'info.md')
    with open(config_info_path, 'w') as f:
        f.write(f"# Model {i+1} Information\n\n")
        f.write(f"**Dataset:** {DATASET_NAME}\n")
        f.write(f"**Hidden Layers:** {result['hidden_dim']}\n")
        f.write(f"**Learning Rate:** {result['lr']}\n")
        f.write(f"**Number of Trainable Parameters:** {result['num_params']}\n")

    # Save plots for this model
    # Training loss
    plt.figure(figsize=(8, 5))
    plt.plot(result["train_losses"], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Model {i+1} - Training Loss')
    plt.legend()
    plt.savefig(os.path.join(config_dir, 'training_loss.png'))
    plt.close()

    # Training accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(result["train_accs"], label='Training Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Model {i+1} - Training Accuracy')
    plt.legend()
    plt.savefig(os.path.join(config_dir, 'training_accuracy.png'))
    plt.close()

    # Test accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(result["test_accs"], label='Test Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Model {i+1} - Test Accuracy')
    plt.legend()
    plt.savefig(os.path.join(config_dir, 'test_accuracy.png'))
    plt.close()

    # Gradient norm
    plt.figure(figsize=(8, 5))
    plt.plot(result["grad_norms"], label='Gradient Norm', color='red')
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm')
    plt.title(f'Model {i+1} - Gradient Norm During Training')
    plt.legend()
    plt.savefig(os.path.join(config_dir, 'gradient_norm.png'))
    plt.close()

    # Combined plot
    plt.figure(figsize=(10, 6))
    plt.plot(result["train_losses"], label='Training Loss', color='blue')
    plt.plot(result["train_accs"], label='Training Accuracy', color='green')
    plt.plot(result["test_accs"], label='Test Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title(f'Model {i+1} - Loss and Accuracy Comparison')
    plt.legend()
    plt.savefig(os.path.join(config_dir, 'loss_and_accuracy_comparison.png'))
    plt.close()

# Combined plots for all models
# Plot training loss for all models
plt.figure(figsize=(10, 6))
for i, result in enumerate(model_results):
    plt.plot(result["train_losses"], label=f"Model {i+1} (Params: {result['num_params']})")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison Across Models')
plt.legend()
plt.savefig(os.path.join(results_dir, 'combined_training_loss.png'))
plt.close()

# Plot training accuracy for all models
plt.figure(figsize=(10, 6))
for i, result in enumerate(model_results):
    plt.plot(result["train_accs"], label=f"Model {i+1} (Params: {result['num_params']})")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Comparison Across Models')
plt.legend()
plt.savefig(os.path.join(results_dir, 'combined_training_accuracy.png'))
plt.close()

# Plot test accuracy for all models
plt.figure(figsize=(10, 6))
for i, result in enumerate(model_results):
    plt.plot(result["test_accs"], label=f"Model {i+1} (Params: {result['num_params']})")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy Comparison Across Models')
plt.legend()
plt.savefig(os.path.join(results_dir, 'combined_test_accuracy.png'))
plt.close()