import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import copy

# Add the parent directory to the system path to find the 'model' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import MnistCNN

# --- Configuration ---
LEARNING_RATE = 0.001
NUM_EPOCHS = 200
BATCH_SIZE = 512
# Use a simple CNN architecture for this task
# The hidden_dim parameter is required by the MnistCNN constructor
MODEL_CONFIG = {"hidden_dim": [64, 128, 64]} 

# --- Setup Results Directory ---
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
results_dir = os.path.join(project_root, 'results', f'hw1_3_{timestamp}')
os.makedirs(results_dir, exist_ok=True)

# --- Data Loading and Preparation ---
print("Loading and preparing data...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load original training and test sets
train_dataset_normal = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Create the dataset with random labels
train_images = train_dataset_normal.data.unsqueeze(1).float() / 255.0 # Add channel dim and normalize
train_images = transforms.Normalize((0.5,), (0.5,))(train_images)

original_labels = train_dataset_normal.targets
# Correctly shuffle the original labels to create the random label set
random_labels = original_labels[torch.randperm(len(original_labels))]
train_dataset_random = TensorDataset(train_images, random_labels)

# Create DataLoaders
train_loader_normal = DataLoader(train_dataset_normal, batch_size=BATCH_SIZE, shuffle=True)
train_loader_random = DataLoader(train_dataset_random, batch_size=BATCH_SIZE, shuffle=True)
# test_loader_normal = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
# test_loader_random = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print("Data preparation complete.")

# --- Helper Functions ---
def compute_accuracy(loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return 100 * correct / total

def train_model(model, train_loader, num_epochs):
    """Trains a model and returns loss and accuracy history."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_losses = []
    train_accs = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (batch_x, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = compute_accuracy(train_loader, model)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%")
        
    return train_losses, train_accs

# --- Main Execution ---

# 2. Train on Random Labels
print("\n--- Training on RANDOM labels ---")
model_random = MnistCNN(**MODEL_CONFIG)
losses_random, accs_random = train_model(model_random, train_loader_random, NUM_EPOCHS)
# test_acc_random = compute_accuracy(test_loader, model_random)
# print(f"\nFinal Test Accuracy (Random Labels): {test_acc_random:.2f}%")


# 1. Train on Normal Labels
print("\n--- Training on NORMAL labels ---")
model_normal = MnistCNN(**MODEL_CONFIG)
losses_normal, accs_normal = train_model(model_normal, train_loader_normal, NUM_EPOCHS)
# test_acc_normal = compute_accuracy(test_loader, model_normal)
# print(f"\nFinal Test Accuracy (Normal Labels): {test_acc_normal:.2f}%")



# --- Visualization ---
print("\nGenerating plots...")
# Plot Training Loss
plt.figure(figsize=(10, 5))
plt.plot(losses_normal, label='Normal Labels')
plt.plot(losses_random, label='Random Labels')
plt.title('Training Loss vs. Epochs')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'training_loss_comparison.png'))
plt.close()

# Plot Training Accuracy
plt.figure(figsize=(10, 5))
plt.plot(accs_normal, label='Normal Labels')
plt.plot(accs_random, label='Random Labels')
plt.title('Training Accuracy vs. Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'training_accuracy_comparison.png'))
plt.close()

print(f"\nExperiment complete. Results are saved in: {results_dir}")